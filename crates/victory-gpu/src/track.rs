//! Write tracking: the card never reads a stale copy of title memory.
//!
//! The card holds copies of title memory in two places: the pool's GPU
//! view (one card page per physical page, `victory_vk::pool`) and the
//! images the title fills from its memory (`vulkan/images`). The console
//! has one memory: a plain CPU store is what the GPU reads next. Here a
//! 64 KiB page the card has a copy of is write-protected in every CPU view
//! that shows it (`xmem` knows them: [`views_with`]). The first store into
//! it faults ([`write_fault`], from `xmem`'s handler), which notes the page
//! written and opens it again. [`flush`], before every submission and every
//! present, protects the written pages again and then has their copies
//! made again: protect first, copy second, so a store between the two is
//! seen at the next flush.
//!
//! - **Hot pages.** A page found written by [`STREAK`] + 1 flushes in a
//!   row (the title writes it all the time) stays open for [`HOT_FRAMES`]
//!   frames: no fault storm. Its pool page is copied at every flush. An
//!   image over it is not filled at every flush (a fill detiles the whole
//!   texture): while its page is hot it is checked by fingerprint, as
//!   untracked memory is (`images/fill.rs`).
//! - **Host calls** that write title memory from the kernel do not fault
//!   into the handler, they fail: such a call holds [`writing`] around it,
//!   which opens the pages first and keeps them open until it is dropped.
//! - **Victory's own views** of physical pages (each section's whole view)
//!   are never protected: every write through them is announced
//!   (`title_memory_written`).
//!
//! The state is one byte per 64 KiB page of title address space, kept only
//! for the 256 MiB stretches that hold a tracked page. No page is mirrored.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};

/// A page of title memory, as `xmem` maps physical pages.
pub const PAGE: usize = 64 << 10;
/// Title memory lies below this (`xmem` places it under 2^40 unless the
/// title asks for more bits; memory above is not tracked).
const TOP: usize = 1 << 40;

/// Read-only by us: a store into it faults.
const PROTECTED: u8 = 0x80;
/// Stored into since its copies were made: open, and listed for the next
/// flush (or about to be).
const WRITTEN: u8 = 0x40;
/// Written frame after frame: open, copied at every flush.
const HOT: u8 = 0x20;
/// An image may lie over it.
const IMAGE: u8 = 0x10;
/// Found written by the flush under way.
const SEEN: u8 = 0x08;
/// The flushes in a row that found it written, at most this.
const STREAK: u8 = 0x07;
/// Frames a hot page stays open before it is protected again, to learn
/// whether the title still writes it.
const HOT_FRAMES: u16 = 32;

type Leaf = [AtomicU8; 4096];
/// The marks of each 256 MiB of title address space, made when a page
/// there is first tracked; never freed.
static LEAVES: [AtomicPtr<Leaf>; TOP >> 28] = [const { AtomicPtr::new(std::ptr::null_mut()) }; TOP >> 28];

fn mark(at: usize) -> Option<&'static AtomicU8> {
    let leaf = LEAVES.get(at >> 28)?.load(Ordering::Acquire);
    // SAFETY: a leaf, once stored, lives for the run.
    unsafe { leaf.as_ref() }.map(|l| &l[at >> 16 & 0xfff])
}

fn mark_made(at: usize) -> Option<&'static AtomicU8> {
    let slot = LEAVES.get(at >> 28)?;
    if slot.load(Ordering::Acquire).is_null() {
        let leaf: *mut Leaf = Box::into_raw(Box::new([const { AtomicU8::new(0) }; 4096]));
        if slot.compare_exchange(std::ptr::null_mut(), leaf, Ordering::AcqRel, Ordering::Acquire).is_err() {
            // SAFETY: ours alone, never stored.
            drop(unsafe { Box::from_raw(leaf) });
        }
    }
    mark(at)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The pages written since the last flush, and the ones a flush is to
/// look at for another reason ([`written`], [`copied`]). The fault handler
/// pushes here, so nothing that holds it writes title memory.
static LIST: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// In a list entry: the page was not written, only copied unprotected.
const COPIED_ONLY: usize = 1;

/// The ranges host calls are writing now ([`writing`]). Held around every
/// change of a page's protection but the handler's, so that no page is
/// protected under such a call.
static HOST: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Faults taken, protection changes made, pages copied again by flushes,
/// flushes, their time in ns, written pages with images over them, faults
/// on pages under images.
pub(crate) static COUNT: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
pub(crate) const IMAGE_FAULTS: usize = 6;
pub(crate) const FAULTS: usize = 0;
const CHANGES: usize = 1;
pub(crate) const COPIED: usize = 2;
pub(crate) const FLUSHES: usize = 3;
pub(crate) const FLUSH_NS: usize = 4;
pub(crate) const IMAGES: usize = 5;
/// Pages open as hot now.
pub(crate) static HOT_NOW: AtomicU64 = AtomicU64::new(0);

fn set(page: usize, protection: u32) -> bool {
    let mut old = 0;
    COUNT[CHANGES].fetch_add(1, Ordering::Relaxed);
    // SAFETY: a page of title memory; a page not there fails the call.
    let done = unsafe { VirtualProtect(page as *const _, PAGE, protection, &mut old) != 0 };
    // The 4 KiB pages a card home keeps read-only stay so.
    if protection == PAGE_READWRITE {
        crate::vulkan::homes::opened(page, PAGE);
    }
    done
}

/// A store faulted at `at`: if the page is one we protected, it is noted
/// written and opened, and the store goes on (true). Takes no lock a
/// writer of title memory may hold: victory's own writers fault here too.
pub fn write_fault(at: usize) -> bool {
    // A page of a card home's cluster first (its own 4 KiB tracking); the
    // 64 KiB page around it may be tracked here as well.
    let home = crate::vulkan::homes::write_fault(at);
    let Some(m) = mark(at) else { return home };
    let page = at & !(PAGE - 1);
    let taken = m.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| (v & PROTECTED != 0).then_some(v & !PROTECTED | WRITTEN));
    match taken {
        Ok(v) => {
            let open = set(page, PAGE_READWRITE);
            // Listed only once open: a flush that protects the page again
            // has it in its list, or will at the next.
            lock(&LIST).push(page);
            COUNT[FAULTS].fetch_add(1, Ordering::Relaxed);
            COUNT[IMAGE_FAULTS].fetch_add(u64::from(v & IMAGE != 0), Ordering::Relaxed);
            open || home
        }
        // Open by its mark (another thread's fault is opening it, or a
        // flush protected it between that thread's two steps): opened,
        // and listed so that the next flush sees it.
        Err(v) if v & (WRITTEN | HOT) != 0 => {
            let open = set(page, PAGE_READWRITE);
            lock(&LIST).push(page);
            open || home
        }
        Err(_) => home,
    }
}

/// The pages of `[at, at + len)`.
fn pages(at: usize, len: usize) -> impl Iterator<Item = usize> {
    let end = at.saturating_add(len.max(1)).min(TOP);
    ((at & !(PAGE - 1))..end).step_by(PAGE)
}

/// The marked pages of `[at, at + len)` (stretches with no marks skipped).
fn marked(at: usize, len: usize) -> impl Iterator<Item = (usize, &'static AtomicU8)> {
    let end = at.saturating_add(len.max(1)).min(TOP);
    let mut page = at & !(PAGE - 1);
    std::iter::from_fn(move || {
        while page < end {
            let here = page;
            if LEAVES[here >> 28].load(Ordering::Acquire).is_null() {
                page = (here | ((1 << 28) - 1)) + 1;
                continue;
            }
            page += PAGE;
            if let Some(m) = mark(here).filter(|m| m.load(Ordering::Acquire) != 0) {
                return Some((here, m));
            }
        }
        None
    })
}

fn in_host(host: &[(usize, usize)], page: usize) -> bool {
    host.iter().any(|&(at, len)| page < at.saturating_add(len) && at < page + PAGE)
}

/// What a host call that writes title memory holds ([`writing`]).
pub struct Writing(Option<(usize, usize)>);

/// A host call is about to write `[at, at + len)` from the kernel (a file
/// read into title memory, the driver pinning pages for an import): a
/// protected page there would fail the call, not fault. The tracked pages
/// are opened and listed as written, and none is protected until what this
/// returns is dropped.
pub fn writing(at: usize, len: usize) -> Writing {
    if at >= TOP || len == 0 {
        return Writing(None);
    }
    let mut host = lock(&HOST);
    host.push((at, len));
    crate::vulkan::homes::host_writes(at, len);
    for (page, m) in marked(at, len) {
        let was = m.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| (v & PROTECTED != 0).then_some(v & !PROTECTED | WRITTEN));
        match was {
            Ok(_) => {
                set(page, PAGE_READWRITE);
                lock(&LIST).push(page);
            }
            // A fault of another thread may be about to open it.
            Err(v) if v & WRITTEN != 0 => {
                set(page, PAGE_READWRITE);
            }
            Err(_) => {}
        }
    }
    Writing(Some((at, len)))
}

impl Drop for Writing {
    fn drop(&mut self) {
        if let Some(range) = self.0 {
            let mut host = lock(&HOST);
            if let Some(i) = host.iter().position(|&r| r == range) {
                host.swap_remove(i);
            }
        }
    }
}

/// The memory shown at `page` may have been written without a fault (a
/// new view of it is open, or the title set its protection itself): the
/// next flush protects its views if the card holds a copy, and has the
/// copies made again.
pub fn written(page: usize) {
    if page < TOP {
        lock(&LIST).push(page & !(PAGE - 1));
    }
}

/// The pool page shown at `page` was copied to the card with its CPU
/// views open: the next flush protects them and has that copy made again
/// (images over it were filled under protection: they stay).
pub(crate) fn copied(page: usize) {
    if page < TOP {
        lock(&LIST).push(page & !(PAGE - 1) | COPIED_ONLY);
    }
}

/// The view at `page` is going away: its mark goes. True if stores into
/// it are not yet flushed (the caller lists another place that shows the
/// same memory, [`written`]).
pub fn forget(page: usize) -> bool {
    mark(page).is_some_and(|m| m.swap(0, Ordering::AcqRel) & (WRITTEN | HOT) != 0)
}

/// `[at, at + len)` is about to be freed or decommitted: tracking ends
/// there (pages open again); the next flush looks at what was tracked, in
/// case the memory stays.
pub fn released(at: usize, len: usize) {
    if at >= TOP {
        return;
    }
    let _host = lock(&HOST);
    crate::vulkan::homes::host_writes(at, len);
    for (page, m) in marked(at, len) {
        if m.swap(0, Ordering::AcqRel) & PROTECTED != 0 {
            set(page, PAGE_READWRITE);
        }
        lock(&LIST).push(page);
    }
}

/// Whether we keep `page` read-only now (a page found read-only is
/// title memory all the same: `vulkan/loose.rs`).
pub(crate) fn protected(page: usize) -> bool {
    mark(page).is_some_and(|m| m.load(Ordering::Acquire) & PROTECTED != 0) || crate::vulkan::homes::tracks(page)
}

/// Whether a host call is writing a byte of `[at, at + len)` now
/// ([`writing`]): its pages stay open.
pub(crate) fn host_holds(at: usize, len: usize) -> bool {
    lock(&HOST).iter().any(|&(h, n)| at < h.saturating_add(n) && h < at + len)
}

/// The memory a page address shows: every address that shows it now, and
/// whether a CPU view is there (the pool's GPU view has none). No places:
/// not memory `xmem` maps.
pub struct Shown {
    pub page: usize,
    pub places: Vec<(usize, bool)>,
}

impl Shown {
    fn cpu(&self) -> impl Iterator<Item = usize> + '_ {
        self.places.iter().filter(|p| p.1).map(|p| p.0)
    }
}

/// `xmem`'s answer for page addresses: what each shows, handed to the
/// closure while the page table stays as it is (its lock held).
pub type Views = fn(&[usize], &mut dyn FnMut(&[Shown]));

static VIEWS: OnceLock<Views> = OnceLock::new();

/// `xmem` gives its page table's answers; tracking starts.
pub fn views_with(views: Views) {
    let _ = VIEWS.set(views);
}

/// The pages of `[at, at + len)` whose memory the CPU sees at another
/// address too (a store there is no store here: `vulkan/homes.rs`).
pub(crate) fn aliased(at: u64, len: u64) -> usize {
    let Some(views) = VIEWS.get() else { return 0 };
    if at as usize >= TOP {
        return 0;
    }
    let all: Vec<usize> = pages(at as usize, len as usize).collect();
    let mut n = 0;
    views(&all, &mut |shown| n += shown.iter().filter(|s| s.cpu().count() > 1).count());
    n
}

/// The card is about to copy `[at, at + len)` (a fill of the pool's GPU
/// view, of an image: `image`): the pages are protected in every CPU view
/// first. False if some of it is memory `xmem` does not map or the host
/// refuses to protect: that memory is not tracked.
pub(crate) fn protect(at: u64, len: u64, image: bool) -> bool {
    let Some(views) = VIEWS.get() else { return false };
    if at as usize >= TOP {
        return false;
    }
    let all: Vec<usize> = pages(at as usize, len as usize).collect();
    let mut tracked = true;
    views(&all, &mut |shown| {
        let host = lock(&HOST);
        for s in shown {
            tracked &= !s.places.is_empty();
            for c in s.cpu() {
                tracked &= protect_one(c, if image { IMAGE } else { 0 }, &host);
            }
        }
    });
    tracked
}

/// Protects the CPU view page `c` unless it is protected, or open and
/// listed (the next flush protects it and has the copies made again).
fn protect_one(c: usize, bits: u8, host: &[(usize, usize)]) -> bool {
    let Some(m) = mark_made(c) else { return false };
    let v = m.load(Ordering::Acquire);
    if v & (PROTECTED | WRITTEN | HOT) != 0 {
        m.fetch_or(bits, Ordering::AcqRel);
        // A hot page stays open: an image over it is not tracked.
        return v & HOT == 0 || bits & IMAGE == 0;
    }
    if in_host(host, c) {
        m.store(v | bits | WRITTEN, Ordering::Release);
        lock(&LIST).push(c);
        return true;
    }
    m.store(v | bits | PROTECTED, Ordering::Release);
    if !set(c, PAGE_READONLY) {
        m.store(0, Ordering::Release);
        return false;
    }
    true
}

/// Ends the tracking of `c`: open, no mark.
fn release(c: usize) {
    if let Some(m) = mark(c)
        && m.swap(0, Ordering::AcqRel) & PROTECTED != 0
    {
        set(c, PAGE_READWRITE);
    }
}

/// The mark of `c` goes unless it is protected (then it is open already).
fn clear_open(c: usize) {
    if let Some(m) = mark(c) {
        let _ = m.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| (v & PROTECTED == 0).then_some(0));
    }
}

/// Protects every CPU view of `s` (the copy follows); `own` goes into the
/// mark of `rep`, which counts the page's frames.
fn protect_views(s: &Shown, rep: usize, own: u8) {
    for c in s.cpu() {
        let Some(m) = mark_made(c) else { continue };
        let own = if c == rep { own } else { 0 };
        // A view still protected is left as it is (its bits apart): a
        // store into it is faulting into the handler, or will.
        let was = m.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some(PROTECTED | v & IMAGE | own));
        if was.is_ok_and(|v| v & PROTECTED == 0) && !set(c, PAGE_READONLY) {
            m.store(0, Ordering::Release);
        }
    }
}

/// What the card holds of a page's memory.
#[derive(Clone, Copy, Default)]
pub(crate) struct Held {
    /// A page of the pool's GPU view, on the card now.
    pub pool: bool,
    /// An image filled from it (queued to be filled again by the asking).
    pub image: bool,
}

#[derive(Default)]
struct Flushing {
    /// Hot pages (a CPU view each) and the frames they stay open.
    hot: HashMap<usize, u16>,
    /// Pages found written by this flush, and by the one before.
    seen: Vec<usize>,
    before: Vec<usize>,
}

static FLUSHING: Mutex<Option<Flushing>> = Mutex::new(None);

/// Before work is given to the card: every page listed since the last
/// flush (and every hot page) is protected again in its CPU views, if the
/// card holds a copy of it (`held`: asked with what each page shows and
/// whether images over it are to be filled again; it queues their fills).
/// Returns the addresses whose pool pages are to be copied again, which
/// the caller does next. `frame_end`: the flush of a present.
pub(crate) fn flush(frame_end: bool, held: &mut dyn FnMut(&[Shown], &[bool]) -> Vec<Held>) -> Vec<usize> {
    let Some(views) = VIEWS.get() else { return Vec::new() };
    let mut guard = lock(&FLUSHING);
    let st = guard.get_or_insert_with(Flushing::default);
    // Written before only copied, and each page once.
    let mut listed = std::mem::take(&mut *lock(&LIST));
    listed.extend(st.hot.keys().map(|p| p | COPIED_ONLY));
    listed.sort_unstable();
    listed.dedup_by_key(|e| *e & !COPIED_ONLY);
    let pages: Vec<usize> = listed.iter().map(|e| e & !COPIED_ONLY).collect();
    let mut copy = Vec::new();
    if !pages.is_empty() {
        views(&pages, &mut |shown| {
            // Images are filled again over pages that were written, not
            // over hot ones (their images are checked by fingerprint).
            let written: Vec<bool> = shown.iter().zip(&listed).map(|(s, e)| e & COPIED_ONLY == 0 && !s.cpu().any(|c| st.hot.contains_key(&c))).collect();
            let fill: Vec<bool> = shown.iter().zip(&written).map(|(s, w)| *w && may_hold_images(s)).collect();
            let mut held = held(shown, &fill);
            let host = lock(&HOST);
            let mut done = HashSet::new();
            for (i, s) in shown.iter().enumerate() {
                if s.cpu().chain([s.page]).any(|c| done.contains(&c)) {
                    continue;
                }
                done.extend(s.cpu().chain([s.page]));
                if !fill[i] {
                    // Not asked: an image lies over it if one did.
                    held[i].image = s.cpu().filter_map(mark).any(|m| m.load(Ordering::Acquire) & IMAGE != 0);
                }
                st.settle(s, held[i], (written[i], frame_end), &host, &mut copy);
            }
        });
    }
    st.end_flush();
    HOT_NOW.store(st.hot.len() as u64, Ordering::Relaxed);
    copy
}

/// Whether images are to be looked for over a written page: one was filled
/// from it, or nothing is known of it yet (a new view).
fn may_hold_images(s: &Shown) -> bool {
    let mut any = false;
    for v in s.cpu().filter_map(mark).map(|m| m.load(Ordering::Acquire)) {
        if v & IMAGE != 0 || v & (PROTECTED | WRITTEN | HOT) == 0 {
            return true;
        }
        any = true;
    }
    !any
}

impl Flushing {
    /// One listed page (`written`, or only copied): tracking ends if the
    /// card holds nothing of it; else it is hot (left open) or protected
    /// again in all its CPU views.
    fn settle(&mut self, s: &Shown, h: Held, (written, frame_end): (bool, bool), host: &[(usize, usize)], copy: &mut Vec<usize>) {
        if h.pool {
            copy.extend(s.places.iter().map(|p| p.0));
        }
        let Some(rep) = s.cpu().next() else {
            // No CPU view (or memory that is `xmem`'s no more): only
            // announced writes reach it.
            self.hot.remove(&s.page);
            release(s.page);
            return;
        };
        if !h.pool && !h.image {
            // Evicted, dropped or unmapped: nothing to keep fresh. A view
            // still protected stays so until its first store.
            for c in s.cpu() {
                self.hot.remove(&c);
                clear_open(c);
            }
            return;
        }
        for m in s.cpu().filter_map(mark_made) {
            if h.image {
                m.fetch_or(IMAGE, Ordering::AcqRel);
            } else {
                m.fetch_and(!IMAGE, Ordering::AcqRel);
            }
        }
        if s.cpu().any(|c| in_host(host, c)) {
            // A host call is writing it: open and listed until that ends.
            lock(&LIST).push(rep);
            return;
        }
        if let Some(own) = self.counted(s, rep, (written, frame_end)) {
            protect_views(s, rep, own);
        }
    }

    /// Counts this flush for the page whose first CPU view is `rep`, and
    /// decides: `None` if it is hot (left open); else the bits its mark
    /// takes with the protection.
    fn counted(&mut self, s: &Shown, rep: usize, (written, frame_end): (bool, bool)) -> Option<u8> {
        let v = mark_made(rep)?.load(Ordering::Acquire);
        let (streak, seen) = (v & STREAK, v & SEEN != 0);
        if let Some(key) = s.cpu().find(|c| self.hot.contains_key(c)) {
            let left = self.hot.get(&key).map_or(0, |l| l.saturating_sub(u16::from(frame_end)));
            if left > 0 {
                self.hot.insert(key, left);
                return None;
            }
            // Protected again: one more store makes it hot at once.
            self.hot.remove(&key);
        } else if written && !seen && streak == STREAK {
            self.hot.insert(rep, HOT_FRAMES);
            for m in s.cpu().filter_map(mark) {
                let _ = m.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| (v & PROTECTED == 0).then_some(HOT | STREAK | v & IMAGE));
            }
            return None;
        }
        if seen || !written {
            return Some(v & (SEEN | STREAK));
        }
        self.seen.push(rep);
        Some(SEEN | (streak + 1).min(STREAK))
    }

    /// The flush ends: a page written in the flush before and not in this
    /// one starts its streak anew.
    fn end_flush(&mut self) {
        for m in self.before.drain(..).filter_map(mark) {
            let _ = m.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| (v & SEEN == 0).then_some(v & !STREAK));
        }
        for m in self.seen.iter().copied().filter_map(mark) {
            m.fetch_and(!SEEN, Ordering::AcqRel);
        }
        self.before = std::mem::take(&mut self.seen);
    }
}

/// Diagnostics: what one fault costs on this host, in ns (the store into
/// a protected page of our own, through the system's exception dispatch
/// and the handler), and one protection change. `None` where no page
/// below [`TOP`] is to be had.
pub(crate) fn fault_cost() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::Memory::{
        MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RELEASE, MEM_RESERVE, MemExtendedParameterAddressRequirements, VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const ROUNDS: u32 = 64;
    // No flush meanwhile: it would end the tracking of a page `xmem` does
    // not know.
    let _flushing = lock(&FLUSHING);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 32) as *mut _,
        HighestEndingAddress: (TOP - 1) as *mut _,
        Alignment: 0,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: a page of our own; the parameter outlives the call.
    let page = unsafe { VirtualAlloc2(GetCurrentProcess(), std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, &mut parameter, 1) } as usize;
    let m = (page != 0).then(|| mark_made(page)).flatten()?;
    let before = COUNT[FAULTS].load(Ordering::Relaxed);
    let (mut fault, mut change) = (0u64, 0u64);
    for _ in 0..ROUNDS {
        m.store(PROTECTED, Ordering::Release);
        let t = std::time::Instant::now();
        set(page, PAGE_READONLY);
        change += t.elapsed().as_nanos() as u64;
        let t = std::time::Instant::now();
        // SAFETY: our page; the store faults into the handler, which opens it.
        unsafe { (page as *mut u8).write_volatile(1) };
        fault += t.elapsed().as_nanos() as u64;
    }
    m.store(0, Ordering::Release);
    lock(&LIST).retain(|&p| p != page);
    let taken = COUNT[FAULTS].load(Ordering::Relaxed) - before;
    COUNT[FAULTS].fetch_sub(taken, Ordering::Relaxed);
    COUNT[CHANGES].fetch_sub(u64::from(ROUNDS) * 2, Ordering::Relaxed);
    // SAFETY: our page.
    unsafe { VirtualFree(page as *mut _, 0, MEM_RELEASE) };
    (taken > 0).then_some((fault / u64::from(ROUNDS), change / u64::from(ROUNDS)))
}
