//! The title's views of its physical pages, as the runtime maps them. A
//! write through one view of a page is a write of the bytes every other
//! view shows: `crate::watch` guards them all and notes them all written.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// The console's page.
const K64: usize = 64 << 10;

struct Views {
    /// By address: bytes, the section's handle, the offset in it.
    views: BTreeMap<usize, (usize, usize, u64)>,
    /// By section and byte offset of a 64 KiB page: the address of that
    /// page in every view that shows it. A write through one view of a
    /// page is a write of the bytes every other view shows
    /// (`crate::watch` guards them all).
    pages: BTreeMap<(usize, u64), Vec<usize>>,
}

static VIEWS: Mutex<Views> =
    Mutex::new(Views { views: BTreeMap::new(), pages: BTreeMap::new() });

fn views() -> std::sync::MutexGuard<'static, Views> {
    VIEWS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The runtime mapped `[at, at + len)` as a view of `section` from
/// `offset` on. Call after [`crate::title_memory_remapped`] of the range.
pub fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    if crate::vulkan::on() {
        return crate::vulkan::title_memory_view(at, len, section, offset);
    }
    {
        let mut v = views();
        v.views.insert(at, (len, section, offset));
        for k in 0..len.div_ceil(K64) {
            v.pages.entry((section, offset + (k * K64) as u64)).or_default().push(at + k * K64);
        }
    }
    // This view is writable from now on: a guard over the same bytes at
    // another view must look at them again, and the pages are noted written
    // there (`crate::watch`).
    crate::watch::views_mapped(section, offset, len);
    // Windows over these addresses look their pages up again, and the
    // directory's entries there follow (`crate::memory`).
    crate::gather::remapped(at as u64, len as u64);
    crate::memory::views_changed(at as u64, len as u64);
}

/// What `[at, at + len)` showed is gone ([`crate::title_memory_remapped`]):
/// the views that touch it are forgotten (the runtime tells again those it
/// maps anew).
pub(crate) fn views_gone(at: usize, len: usize) {
    // The directory's entries there follow (`crate::memory`).
    for (base, bytes) in gone(at, len) {
        crate::memory::views_changed(base as u64, bytes as u64);
    }
}

/// Forgets the views over `[at, at + len)`; returns them (address, bytes).
fn gone(at: usize, len: usize) -> Vec<(usize, usize)> {
    let mut v = views();
    let mut out = Vec::new();
    if v.views.is_empty() {
        return out;
    }
    // A length of 0 (a free of a whole allocation) names the view at `at`
    // alone; an empty range would also make the range below panic.
    let end = at.saturating_add(len.max(1));
    let first = v.views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at).map(|(base, _)| *base);
    let gone: Vec<usize> = first.into_iter().chain(v.views.range(at.saturating_add(1)..end).map(|(base, _)| *base)).collect();
    for base in gone {
        let Some((bytes, section, offset)) = v.views.remove(&base) else {
            continue;
        };
        out.push((base, bytes));
        for k in 0..bytes.div_ceil(K64) {
            let key = (section, offset + (k * K64) as u64);
            if let Some(pages) = v.pages.get_mut(&key) {
                let page = base + k * K64;
                pages.retain(|&shown| shown != page);
                if pages.is_empty() {
                    v.pages.remove(&key);
                }
            }
        }
    }
    out
}

/// The title views that overlap `[start, end)`, as `(address, bytes)`.
pub(crate) fn views_between(start: usize, end: usize) -> Vec<(usize, usize)> {
    if start >= end {
        return Vec::new();
    }
    let v = views();
    let mut out = Vec::new();
    if let Some((&base, &(bytes, _, _))) = v.views.range(..=start).next_back()
        && base + bytes > start
    {
        out.push((base, bytes));
    }
    for (&base, &(bytes, _, _)) in v.views.range(start.saturating_add(1)..end) {
        out.push((base, bytes));
    }
    out
}

/// If `[start, start + len)` shows one run of a section (views that continue
/// each other in address and in the section's bytes): the section and the
/// byte offset of `start` in it (`crate::unified`, which opens sections as
/// heaps). `None` for memory that is no view, or pages scattered in it.
pub(crate) fn section_run(start: usize, len: usize) -> Option<(usize, u64)> {
    let end = start.checked_add(len.max(1))?;
    let v = views();
    let (&base, &(bytes, section, offset)) = v.views.range(..=start).next_back()?;
    if base + bytes <= start {
        return None;
    }
    let first = offset + (start - base) as u64;
    let (mut reach, mut next) = (base + bytes, offset + bytes as u64);
    while reach < end {
        let &(b, s, o) = v.views.get(&reach)?;
        if s != section || o != next {
            return None;
        }
        reach += b;
        next = o + b as u64;
    }
    Some((section, first))
}

/// The section and byte offset of the 64 KiB page at `page`, if one view
/// shows all of it (`crate::unified::source`).
pub(crate) fn section_page(page: usize) -> Option<(usize, u64)> {
    let v = views();
    let (&base, &(bytes, section, offset)) = v.views.range(..=page).next_back()?;
    (page.checked_add(K64)? <= base + bytes).then(|| (section, offset + (page - base) as u64))
}

/// The views that overlap `[start, end)`, as (address, bytes, section,
/// offset of the view's first byte in it).
pub(crate) fn section_spans(start: usize, end: usize) -> Vec<(usize, usize, usize, u64)> {
    if start >= end {
        return Vec::new();
    }
    let v = views();
    let mut out = Vec::new();
    if let Some((&base, &(bytes, section, offset))) = v.views.range(..=start).next_back()
        && base + bytes > start
    {
        out.push((base, bytes, section, offset));
    }
    for (&base, &(bytes, section, offset)) in v.views.range(start.saturating_add(1)..end) {
        out.push((base, bytes, section, offset));
    }
    out
}

/// How far from `start` (at most `most` bytes) one run of a section reaches
/// ([`section_run`]): 0 for memory that is no view.
pub(crate) fn run_reach(start: usize, most: usize) -> usize {
    let v = views();
    let Some((&base, &(bytes, section, offset))) = v.views.range(..=start).next_back() else {
        return 0;
    };
    if base + bytes <= start {
        return 0;
    }
    let end = start.saturating_add(most);
    let (mut reach, mut next) = (base + bytes, offset + bytes as u64);
    while reach < end {
        match v.views.get(&reach) {
            Some(&(b, s, o)) if s == section && o == next => {
                reach += b;
                next = o + b as u64;
            }
            _ => break,
        }
    }
    reach.min(end) - start
}

/// The address of the 64 KiB page shown at `page` in every title view that
/// shows it, `page`'s own address included; empty if `page` is not a view.
pub(crate) fn pages_of(page: usize) -> Vec<usize> {
    let v = views();
    let Some((&base, &(bytes, section, offset))) = v.views.range(..=page).next_back() else {
        return Vec::new();
    };
    let within = page.wrapping_sub(base);
    if within >= bytes || !within.is_multiple_of(K64) {
        return Vec::new();
    }
    v.pages.get(&(section, offset + within as u64)).cloned().unwrap_or_default()
}

/// The title views that show any of the physical bytes `[offset, offset +
/// len)` of `section`, as `(address, bytes)` ranges in address order; not
/// those of a page shown in very many views (`MANY_VIEWS`).
pub(crate) fn views_of_section(section: usize, offset: u64, len: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let v = views();
    let end = offset.saturating_add(len as u64);
    let mut pages: Vec<usize> = Vec::new();
    let mut at = offset - offset % K64 as u64;
    while at < end {
        if let Some(shown) = v.pages.get(&(section, at))
            && shown.len() <= crate::watch::MANY_VIEWS
        {
            pages.extend(shown.iter().copied());
        }
        match at.checked_add(K64 as u64) {
            Some(next) => at = next,
            None => break,
        }
    }
    pages.sort_unstable();
    pages.dedup();
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(pages.len());
    for page in pages {
        match runs.last_mut() {
            Some((first, bytes)) if *first + *bytes == page => *bytes += K64,
            _ => runs.push((page, K64)),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the runtime forgets when memory is mapped anew.
    #[test]
    fn remapped_memory_forgets_its_views() {
        let at = 0x7_4000_0000usize;
        title_memory_view(at, 4 * K64, 77, 0);
        title_memory_view(at + 4 * K64, K64, 77, 4 * K64 as u64);
        assert_eq!(views_between(at, at + 5 * K64), [(at, 4 * K64), (at + 4 * K64, K64)]);
        assert_eq!(pages_of(at + K64), [at + K64]);
        // One page inside the first view goes: the whole view is forgotten
        // (the runtime maps and tells again what stays of it).
        views_gone(at + K64, K64);
        assert!(views_between(at, at + K64).is_empty());
        assert!(pages_of(at + K64).is_empty());
        assert_eq!(views_between(at + 4 * K64, at + 5 * K64), [(at + 4 * K64, K64)]);
        views_gone(at + 4 * K64, K64);
        assert!(views_between(at + 4 * K64, at + 5 * K64).is_empty());
    }
}
