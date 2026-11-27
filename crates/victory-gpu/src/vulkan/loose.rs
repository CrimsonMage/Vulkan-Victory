//! Loose title memory the card reads (memory in no `XMemVirtualAlloc`
//! region: the title's own allocator's blocks): the committed run of pages
//! around an address, joined with loose neighbours over committed gaps.

use super::Memory;

impl Memory {
    /// `[base, base + len)` widened over the loose regions next to it
    /// whose gap is committed read-write memory (at most [`JOIN_GAP`]),
    /// so that many small allocations the card reads (the title's 64 KiB
    /// blocks, vk21's 151 imports past the map's 64 ranges) are one import
    /// and one range of the map.
    pub(super) fn join_neighbours(&self, mut base: usize, mut len: usize) -> (usize, usize) {
        loop {
            let below = self.regions.range(..base).next_back().filter(|(a, r)| r.loose && **a + r.len <= base);
            let above = self.regions.range(base + len..).next().filter(|(_, r)| r.loose);
            let mut grew = false;
            if let Some((&a, r)) = below
                && base - (a + r.len) <= JOIN_GAP
                && all_usable(a + r.len, base)
            {
                len += base - a;
                base = a;
                grew = true;
            }
            if let Some((&a, r)) = above
                && a - (base + len) <= JOIN_GAP
                && all_usable(base + len, a)
            {
                len = a + r.len - base;
                grew = true;
            }
            if !grew {
                return (base, len);
            }
        }
    }

    /// Takes the regions overlapping `[base, base + len)` out (their
    /// imports retired); the union.
    pub(super) fn merge_loose(&mut self, base: usize, len: usize) -> (usize, usize) {
        let (mut lo, mut hi) = (base, base + len);
        let over: Vec<usize> = self
            .regions
            .range(..hi)
            .filter(|&(&a, r)| a + r.len > lo)
            .map(|(&a, _)| a)
            .collect();
        for a in over {
            if let Some(r) = self.regions.remove(&a) {
                lo = lo.min(a);
                hi = hi.max(a + r.len);
                self.retired.extend(r.import);
            }
        }
        (lo, hi - lo)
    }
}

/// The widest gap between loose regions that is imported to join them.
const JOIN_GAP: usize = 16 << 20;

/// Whether every page of `[lo, hi)` is committed read-write memory.
pub(super) fn all_usable(lo: usize, hi: usize) -> bool {
    let mut p = lo;
    while p < hi {
        let Some(i) = page_info(p).filter(usable) else { return false };
        p = i.BaseAddress as usize + i.RegionSize;
    }
    true
}

/// Where the committed read-write pages from `lo` on end, at most `hi`
/// (`lo` itself when its page is not such memory).
pub(super) fn usable_end(lo: usize, hi: usize) -> usize {
    let mut p = lo;
    while p < hi {
        let Some(i) = page_info(p).filter(usable) else { return p };
        p = i.BaseAddress as usize + i.RegionSize;
    }
    hi
}

/// The runs of committed read-write pages in `[lo, hi)`, as (first
/// address, bytes).
pub(super) fn usable_runs(lo: usize, hi: usize) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut p = lo;
    while p < hi {
        let end = usable_end(p, hi);
        if end > p {
            runs.push((p, end - p));
            p = end;
        } else {
            // to the end of what is there instead (free, reserved or protected pages)
            p = page_info(p).map_or(hi, |i| (i.BaseAddress as usize + i.RegionSize).max(p + 4096));
        }
    }
    runs
}

/// The committed read-write run of pages holding `at` (one allocation),
/// from the allocation's base, so that an address below one touched
/// earlier in the same allocation is in it too.
pub(super) fn committed_run(at: u64) -> Option<(usize, usize)> {
    let at = at as usize;
    let here = page_info(at).filter(usable)?;
    let base = here.AllocationBase as usize;
    let (mut p, mut start) = (base, None);
    while let Some(i) = page_info(p).filter(|i| i.AllocationBase as usize == base) {
        let (lo, hi) = (i.BaseAddress as usize, i.BaseAddress as usize + i.RegionSize);
        if usable(&i) {
            start.get_or_insert(lo);
        } else if hi <= at {
            start = None;
        } else {
            break;
        }
        p = hi;
    }
    start.map(|s| (s, p - s))
}

type PageInfo = windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION;

fn page_info(p: usize) -> Option<PageInfo> {
    // SAFETY: plain data, filled by the query.
    let mut info: PageInfo = unsafe { std::mem::zeroed() };
    // SAFETY: a query into our own structure.
    let ok = unsafe { windows_sys::Win32::System::Memory::VirtualQuery(p as *const _, &mut info, size_of::<PageInfo>()) } != 0;
    ok.then_some(info)
}

/// Committed read-write memory; a page write tracking keeps read-only is
/// that too (it is opened for the import, `crate::track::writing`).
fn usable(i: &PageInfo) -> bool {
    let writable = matches!(i.Protect & 0x1ff, 0x04 | 0x40) || (i.Protect & 0x1ff == 0x02 && crate::track::protected(i.BaseAddress as usize));
    i.State == windows_sys::Win32::System::Memory::MEM_COMMIT && writable
}
