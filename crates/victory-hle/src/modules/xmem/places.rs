//! Where title memory shows physical pages, for DirectStorage requests
//! (which pages they write, and in diagnostics builds whether every other
//! place reads what they wrote) and for crash reports.

use super::*;

/// The pages a DirectStorage request writes, while it runs: none becomes
/// a filler page meanwhile ([`note_fillers`]), and one that is a filler
/// page already stops being one before the bytes come (its stand-ins
/// become views, which show them).
pub(crate) struct Target {
    keys: Vec<(usize, usize)>,
}

impl Drop for Target {
    fn drop(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        let mut physical = physical();
        for key in &self.keys {
            if let Some(n) = physical.targeted.get_mut(key) {
                *n -= 1;
                if *n == 0 {
                    physical.targeted.remove(key);
                }
            }
        }
    }
}

fn target(physical: &mut Physical, mut keys: Vec<(usize, usize)>) -> Target {
    keys.sort_unstable();
    keys.dedup();
    for &key in &keys {
        *physical.targeted.entry(key).or_default() += 1;
    }
    let fillers: Vec<(usize, usize)> = keys.iter().copied().filter(|key| physical.fillers.contains_key(key)).collect();
    if !fillers.is_empty() {
        unfill_for_dstorage(physical, fillers);
    }
    Target { keys }
}

/// The physical pages under bytes `[offset, offset + len)` of a request's
/// page destination.
pub(crate) fn target_pages(pages: &[usize], offset: usize, len: usize) -> Target {
    let end = offset.saturating_add(len);
    let Some(pages) = pages.get(offset / PAGE..end.div_ceil(PAGE)) else {
        return Target { keys: Vec::new() };
    };
    let mut physical = physical();
    let keys = pages.iter().filter_map(|page| physical.pages.get(page).map(|slice| slice.key(0))).collect();
    target(&mut physical, keys)
}

/// The physical pages shown now at a request's memory destination
/// `[at, at + len)` (none where xmem shows nothing: the title's regions
/// and heaps).
pub(crate) fn target_memory(at: usize, len: usize) -> Target {
    let mut physical = physical();
    let keys = pieces(&physical, at, len).into_iter().map(|piece| piece.key).collect();
    target(&mut physical, keys)
}

/// What xmem maps at `addr`, and where that starts: a view or a stand-in.
fn mapped_at(physical: &Physical, addr: usize) -> Option<(usize, &Mapped)> {
    physical.views.range(..=addr).next_back().filter(|(base, m)| addr < *base + m.len).map(|(&base, m)| (base, m))
}

/// A part of title memory inside one page that xmem maps.
struct Piece {
    /// Where it starts (its bytes run to the end of its page at most).
    at: usize,
    /// Where it starts inside its page.
    within: usize,
    /// The page shown there ([`Slice::key`]), and its section.
    key: (usize, usize),
    section: Arc<Section>,
    /// Shown by a stand-in for a filler page, not a view.
    fill: bool,
}

/// The parts of `[at, at + len)` that xmem maps, a page at a time.
fn pieces(physical: &Physical, at: usize, len: usize) -> Vec<Piece> {
    let end = at.saturating_add(len);
    let mut out = Vec::new();
    let mut p = at;
    while p < end {
        let next = ((p / PAGE + 1) * PAGE).min(end);
        if let Some((base, m)) = mapped_at(physical, p) {
            out.push(Piece {
                at: p,
                within: (p - base) % PAGE,
                key: (m.slice.section.id, m.index_at(base, p)),
                section: m.slice.section.clone(),
                fill: m.fill,
            });
        }
        p = next;
    }
    out
}

/// Every other place the page of `piece` shows now, at the piece's bytes:
/// the section itself (victory's view of it, once mapped), the title's
/// views, and the stand-ins of a filler page; each with what it is. Not
/// the GPU views of the pool (no host view there).
fn other_places(physical: &Physical, piece: &Piece) -> Vec<(usize, &'static str)> {
    let start = piece.at - piece.within;
    let mut places = Vec::new();
    let whole = piece.section.whole.load(Ordering::Acquire);
    if whole != 0 {
        places.push((whole + piece.key.1 * PAGE + piece.within, "the section"));
    }
    for &a in physical.shown.get(&piece.key).into_iter().flatten() {
        if a != start && mapped_at(physical, a).is_some_and(|(_, m)| !m.fill) {
            places.push((a + piece.within, "view"));
        }
    }
    if physical.fillers.contains_key(&piece.key) {
        for (&base, m) in physical.views.iter().filter(|(_, m)| m.fill && m.slice.key(0) == piece.key) {
            for q in (base..base + m.len).step_by(PAGE).filter(|&q| q != start) {
                places.push((q + piece.within, "stand-in"));
            }
        }
    }
    places
}

/// Diagnostics: after a DirectStorage request wrote `[at, at + len)`
/// through the title's address, the same bytes are read through every
/// other place that shows those pages now ([`other_places`]); a place that
/// differs is said once (the title reading there would see other bytes
/// than the request placed).
#[cfg(feature = "diagnostics")]
pub(crate) fn check_aliases(at: usize, len: usize) {
    static SAID: LazyLock<Mutex<std::collections::HashSet<(usize, usize)>>> = LazyLock::new(Default::default);
    let physical = physical();
    for piece in pieces(&physical, at, len) {
        // Mapped first, so that the section is a place to compare with.
        let _ = piece.section.whole();
        let n = (PAGE - piece.within).min(PAGE - piece.at % PAGE).min(at + len - piece.at);
        // SAFETY: xmem maps the piece (a view or a stand-in), and nothing
        // changes that while the page table is held.
        let here = unsafe { std::slice::from_raw_parts(piece.at as *const u8, n) };
        for (alias, kind) in other_places(&physical, &piece) {
            // SAFETY: as above, for the other place.
            let there = unsafe { std::slice::from_raw_parts(alias as *const u8, n) };
            let Some(k) = here.iter().zip(there).position(|(a, b)| a != b) else {
                continue;
            };
            let first = SAID
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert((piece.at - piece.within, alias - piece.within));
            if first {
                tracing::warn!(
                    destination = format_args!("{at:#x}"),
                    address = format_args!("{:#x}", piece.at),
                    alias = format_args!("{alias:#x}"),
                    kind,
                    offset = piece.at - at + k,
                    section = piece.key.0,
                    page = piece.key.1,
                    destination_is_stand_in = piece.fill,
                    filler = physical.fillers.contains_key(&piece.key),
                    "dstorage: a memory destination reads differently through another place that shows its page"
                );
            }
        }
    }
}

/// Whether `[at, at + n)` is committed and readable (inside one region).
fn readable(at: usize, n: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_GUARD, VirtualQuery};
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a query into our structure.
    let ok = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } != 0;
    ok && info.State == MEM_COMMIT
        && info.Protect & PAGE_GUARD == 0
        && info.Protect & 0xff != PAGE_NOACCESS
        && at + n <= info.BaseAddress as usize + info.RegionSize
}

/// Whether up to 64 bytes at `addr` (inside its host page) read the same
/// at `alias`: " (same)", " (differs from +k)", or nothing where either is
/// not readable.
fn compare(addr: usize, alias: usize) -> String {
    let n = 64.min(0x1000 - (addr & 0xfff)).min(0x1000 - (alias & 0xfff));
    if !readable(addr, n) || !readable(alias, n) {
        return String::new();
    }
    // SAFETY: both checked readable just now.
    let (a, b) = unsafe { (std::slice::from_raw_parts(addr as *const u8, n), std::slice::from_raw_parts(alias as *const u8, n)) };
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(k) => format!(" (differs from +{k:#x})"),
        None => " (same)".into(),
    }
}

/// What xmem shows at `addr`, for a crash report: a view or a stand-in,
/// its section and page, the other places that show that page and whether
/// they read the same there, whether it is a filler page, and whether a
/// DirectStorage request writes it now. Never waits for the page table
/// (the faulting thread may hold it).
pub(super) fn describe(addr: usize) -> Option<String> {
    let text = describe_table(addr)?;
    Some(text + &super::ring::touching(addr))
}

/// What xmem's table says of `addr` (and, for memory in no view, what the
/// address space holds there).
fn describe_table(addr: usize) -> Option<String> {
    use std::fmt::Write;
    let Ok(physical) = PHYSICAL.try_lock() else {
        return Some("xmem: the page table is held (by this thread?)".into());
    };
    let Some(piece) = pieces(&physical, addr, 1).pop() else {
        if let Some(&(section, index)) = physical.aliased.get(&(addr & !(PAGE - 1))) {
            return Some(format!("xmem: a page of a GPU view of the pool (no host view), section {section} page {index}"));
        }
        return reserved_by_xmem(addr).then(|| unmapped_text(addr));
    };
    let (base, m) = mapped_at(&physical, addr)?;
    let mut text = format!(
        "xmem: {} at {base:#x} (+{:#x}), section {} page {} +{:#x}",
        if piece.fill { "stand-in for a filler page" } else { "view" },
        m.len,
        piece.key.0,
        piece.key.1,
        piece.within
    );
    let places = other_places(&physical, &piece);
    let _ = write!(text, "; {} other places", places.len());
    for &(alias, kind) in places.iter().take(8) {
        let _ = write!(text, ", {kind} {alias:#x}{}", compare(addr, alias));
    }
    let gpu = physical.shown.get(&piece.key).map_or(0, |a| a.iter().filter(|a| physical.aliased.contains_key(a)).count());
    let filler = if physical.fillers.contains_key(&piece.key) {
        "a filler page"
    } else if physical.not_fillers.contains(&piece.key) {
        "not a filler page (looked at)"
    } else {
        "never looked at as a filler"
    };
    let _ = write!(
        text,
        "; GPU view places {gpu}; {filler}; DirectStorage requests writing it now: {}",
        physical.targeted.get(&piece.key).copied().unwrap_or(0)
    );
    Some(text)
}

/// Memory xmem reserved with no view in its table: an `XMemVirtualAlloc`
/// region (private memory of the title's, committed) or a free placeholder
/// of a reservation for physical pages (anything else is a disagreement
/// between the table and the address space).
fn unmapped_text(addr: usize) -> String {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a query into our structure.
    let n = unsafe { VirtualQuery(addr as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
    let what = match (n != 0, info.State, info.Type) {
        (false, ..) => "no answer from the address space",
        (true, MEM_COMMIT, MEM_PRIVATE) => "a region of private memory (XMemVirtualAlloc), no physical pages",
        (true, MEM_RESERVE, MEM_PRIVATE) => "a free placeholder (nothing mapped)",
        (true, MEM_COMMIT, MEM_MAPPED) => "a mapped view the table does not know (table and address space disagree)",
        _ => "neither a view nor a region",
    };
    format!("xmem: reserved by xmem, no view in its table: {what} (state {:#x}, type {:#x})", info.State, info.Type)
}

/// The host's flag that joins adjacent placeholders (not in windows-sys).
const MEM_COALESCE_PLACEHOLDERS: u32 = 1;

/// Makes `[base, base + len)` exactly one placeholder: the placeholders it
/// lies in are joined if it spans more than one (views unmapped one by one
/// leave one each), and what lies outside it is split off. Placeholders
/// change in the process alone (no wineserver request under Proton).
pub(super) fn placeholder(base: usize, len: usize) -> bool {
    use std::ffi::c_void;
    use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualFree};
    use windows_sys::Win32::System::Memory::{MEM_PRIVATE, VirtualQuery};
    let query = |at: usize| {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        let n = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        // A free placeholder: reserved and private (a view is mapped memory).
        (n != 0 && info.State == MEM_RESERVE && info.Type == MEM_PRIVATE).then_some(info)
    };
    let end = base + len;
    let Some(info) = query(base) else {
        return false;
    };
    let start = info.AllocationBase as usize;
    let mut stop = info.BaseAddress as usize + info.RegionSize;
    let mut pieces = 1;
    while stop < end {
        match query(stop) {
            Some(next) if next.AllocationBase as usize == stop => {
                stop = next.BaseAddress as usize + next.RegionSize;
                pieces += 1;
            }
            _ => return false,
        }
    }
    if pieces > 1 {
        // SAFETY: adjacent free placeholders, joined into one.
        if unsafe { VirtualFree(start as *mut c_void, stop - start, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) } == 0 {
            return false;
        }
        victory_gpu::mapstat::merged();
    }
    if start != base || stop != end {
        // SAFETY: splits the placeholder around the range.
        if unsafe { VirtualFree(base as *mut c_void, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) } == 0 {
            return false;
        }
        victory_gpu::mapstat::split();
    }
    true
}
