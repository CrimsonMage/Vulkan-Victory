//! Filler pages: a page of zeros the title repeats (the unused parts of
//! its tiled resources) is shown by stand-ins, zeroed read-only memory of
//! ours with no view, rather than a view per place.

use super::*;

/// A page a request repeats (the same page at consecutive addresses, as
/// titles fill the unused parts of their tiled resources) is a filler
/// candidate: one that holds only zeros becomes a filler page, shown by
/// stand-ins ([`fill`]) rather than a view per place. Nothing else is:
/// a page the title merely shows at several places (its pool, the GPU
/// views of it, a scatter that names it twice) is zeros because it is
/// fresh, and data comes into it later through one of those places, which
/// a stand-in elsewhere would never show (runs vk17 and vk35 made 1,348
/// such pages fillers at startup, then read zeros through their stand-ins
/// where DirectStorage had placed resources). Nor is a page a
/// DirectStorage request is writing now ([`Target`]).
pub(super) fn note_fillers(physical: &mut Physical, runs: &[Run]) {
    for r in runs.iter().filter(|r| r.repeat) {
        let key = r.slice.key(0);
        if physical.fillers.contains_key(&key) || physical.not_fillers.contains(&key) || physical.targeted.contains_key(&key) {
            continue;
        }
        if zero(&r.slice) {
            tracing::info!(
                section = r.slice.section.id,
                page = r.slice.index,
                places = r.bytes / PAGE,
                "xmem: a page of zeros the title repeats: shown there by stand-ins, not views"
            );
            physical.fillers.insert(key, r.slice.clone());
        } else {
            physical.not_fillers.insert(key);
        }
    }
}

/// What xmem noted about the pages a change took away from somewhere goes
/// for those that nothing shows any more (no view, no stand-in): if the
/// title shows one again, it is looked at afresh. At the end of a change.
pub(super) fn forget_unshown(physical: &mut Physical) {
    for key in std::mem::take(&mut physical.unshown) {
        if physical.shown.contains_key(&key) {
            continue;
        }
        physical.not_fillers.remove(&key);
        if physical.fillers.contains_key(&key) && !physical.views.values().any(|m| m.fill && m.slice.key(0) == key) {
            physical.fillers.remove(&key);
        }
    }
}

/// Whether the page holds only zeros.
fn zero(slice: &Slice) -> bool {
    View::map(slice, 1).is_some_and(|mut v| v.bytes().iter().all(|&b| b == 0))
}

/// Shows the filler page `slice` at every page of `[base, base + len)`,
/// which must be free placeholders, by a stand-in: zeroed memory of ours,
/// read-only, with no view (the title reads its filler pages and never
/// writes them; a write is let through and counted, `on_fault`). False if
/// the host refuses.
pub(super) fn fill(physical: &mut Physical, base: usize, len: usize, slice: Slice, site: Site) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    // SAFETY: private memory in place of the placeholder just made.
    let ok = placeholder(base, len)
        && unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                base as *const c_void,
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READONLY,
                std::ptr::null_mut(),
                0,
            )
        } as usize
            == base;
    if !ok {
        tracing::warn!(address = format_args!("{base:#x}"), len, error = %std::io::Error::last_os_error(), "a stand-in for a filler page failed; views instead");
        return false;
    }
    victory_gpu::mapstat::filled(site, len);
    fill_ranges().insert(base, len);
    physical.views.insert(base, Mapped { len, slice, fill: true });
    true
}

/// Filler pages found written (`check_fillers`), to be shown by views from
/// the next mapping on.
static DIRTY: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Every 10 s (the views line): a filler page that no longer holds only
/// zeros was written after all, and its stand-ins show what it held
/// before. Said aloud, and its stand-ins become views at the next mapping
/// change. Never waits for the page table (the line comes from wherever).
pub(super) fn check_fillers() {
    let fillers: Vec<Slice> = match PHYSICAL.try_lock() {
        Ok(physical) => physical.fillers.values().cloned().collect(),
        Err(_) => return,
    };
    for slice in fillers {
        if !zero(&slice) {
            tracing::warn!(section = slice.section.id, page = slice.index, "xmem: a filler page was written; its stand-ins become views");
            DIRTY.lock().unwrap_or_else(|e| e.into_inner()).push(slice.key(0));
        }
    }
}

/// Stand-ins of filler pages found written become views. With the change
/// lock held.
pub(super) fn unfill_dirty(physical: &mut Physical) {
    let dirty = std::mem::take(&mut *DIRTY.lock().unwrap_or_else(|e| e.into_inner()));
    for key in dirty {
        unfill(physical, key);
    }
}

/// The filler page `key` is one no longer: its stand-ins become views.
/// With the change lock held.
pub(super) fn unfill(physical: &mut Physical, key: (usize, usize)) {
    if physical.fillers.remove(&key).is_none() {
        return;
    }
    physical.not_fillers.insert(key);
    let stand_ins: Vec<usize> = physical.views.iter().filter(|(_, m)| m.fill && m.slice.key(0) == key).map(|(&at, _)| at).collect();
    for at in stand_ins {
        let Some(m) = physical.views.get(&at).cloned() else {
            continue;
        };
        unmap_view(physical, at, at, at + m.len, Site::Remnant);
        for p in (at..at + m.len).step_by(PAGE) {
            map(physical, p, PAGE, m.slice.clone(), Site::Remnant);
        }
    }
    forget_unshown(physical);
}

/// Before victory writes into pages (DirectStorage): a filler page among
/// them is one no longer, and its stand-ins become views first.
pub(super) fn unfill_written(runs: &[(Slice, usize)]) {
    let mut physical = physical();
    let keys: Vec<(usize, usize)> = runs
        .iter()
        .flat_map(|(slice, n)| (0..*n).map(|k| slice.key(k)))
        .filter(|key| physical.fillers.contains_key(key))
        .collect();
    if keys.is_empty() {
        return;
    }
    unfill_for_dstorage(&mut physical, keys);
}

/// The filler pages `keys`, which DirectStorage is about to write, are
/// fillers no longer: their stand-ins become views.
pub(super) fn unfill_for_dstorage(physical: &mut Physical, keys: Vec<(usize, usize)>) {
    let _changing = victory_gpu::title_memory_remapping();
    for key in keys {
        tracing::warn!(section = key.0, page = key.1, "xmem: DirectStorage writes a filler page; its stand-ins become views");
        unfill(physical, key);
    }
}

/// Stand-ins by address, and their bytes, for the fault handler (which
/// must not wait for the page table).
pub(super) fn fill_ranges() -> std::sync::RwLockWriteGuard<'static, BTreeMap<usize, usize>> {
    FILL_RANGES.write().unwrap_or_else(|e| e.into_inner())
}

pub(super) static FILL_RANGES: std::sync::RwLock<BTreeMap<usize, usize>> = std::sync::RwLock::new(BTreeMap::new());
