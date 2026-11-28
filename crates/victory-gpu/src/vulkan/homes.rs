//! Card homes (`docs/vulkan-host.md`, "Homes"): the buffers programs
//! store into live on the card.
//!
//! Title memory is system RAM the card reads and writes in place, and a
//! program's scattered stores into it are what compute costs (a store
//! into cached system RAM about 250 times one into card memory). Class 3
//! of `docs/kiss.md`'s table (the GPU writes, the GPU reads) has its home
//! in video memory:
//!
//! - **What gets one.** Buffer memory the title makes an unordered-access
//!   view of [`LEAST`] bytes or more in, where it lies in a region the
//!   title asked for write-combined: the title does not read such memory
//!   back (run hm0: no CPU read of 45,000 probed pages there); or in
//!   loose memory (its own allocator's blocks) every page of which its
//!   CPU has at another address too: the card's side of physical pages
//!   mapped twice ([`view`]). Buffers that touch are one cluster with
//!   one home ([`Homes::want`]). A
//!   cluster gets its home once it is seen to change ([`Asking`]: a few
//!   of its pages are looked at every [`ROUND`] presents; the card stores
//!   into some clusters every frame and into others never) or store
//!   staging stages a view in it ([`staged`]), the one that changed most
//!   first, while the memory map and the card have room ([`each_frame`]).
//! - **The home is the cluster's bytes, exactly.** The memory map shows
//!   them at the home's device address in every generation
//!   ([`Homes::cut`]), so every access by address goes there, and the
//!   host's own resolution of title addresses does ([`Homes::of`]).
//!   Copied in once when the home is made; never back while it lives.
//! - **CPU stores reach the home.** The cluster's 4 KiB pages are
//!   read-only. The first store into one faults ([`write_fault`]): the
//!   page is copied as it was and opened. Before the next submission
//!   ([`flush`]) the page is protected again, and the bytes of the
//!   cluster that differ from the copy go to the home; bytes the CPU did
//!   not change stay the card's. A page is read-only at every address
//!   the CPU sees it at ([`Span::others`]).
//! - **A home goes** when the title frees its memory (nothing copied), and
//!   under pressure (the pool's GPU view cannot keep the pages programs
//!   read on the card: it asks again; the map has no ranges left for
//!   it): its bytes go back to title memory in one copy, with CPU stores
//!   into the cluster held until that copy has run ([`Homes::leave`]).
//!
//! shortcut: a command buffer begun before a home is made or goes and
//! submitted after it runs with its older map (the title's lists are
//! recorded as they are written, and some stay open over a present): what
//! it stores into the cluster then is lost, and it reads the cluster as
//! title memory holds it.
//!
//! shortcut: the CPU reads a cluster's bytes as they were when its home
//! was made (and what it stored since). A store of the value title memory
//! already holds is not seen: where the card changed that byte meanwhile,
//! the card's stays.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use victory_vk::account;
use victory_vk::ash::vk;
use victory_vk::map::{MAX_RANGES, Range};
use victory_vk::memory::{Buffer, Place, TITLE_USAGE};
use victory_vk::pool::PAGE;
use victory_vk::timeline::Mark;
use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};

use super::{Backend, Memory, budget, lock, memory};

const SMALL: usize = 4096;
/// The least unordered-access view that asks for a home (the other
/// layers' floor: under it lie the small results the title reads back).
const LEAST: u64 = 1 << 20;
/// A cluster's pages that are looked at, how often (in presents), and
/// the changes seen that make it ripe for a home.
const SAMPLES: usize = 8;
const ROUND: u64 = 16;
const RIPE: u32 = 8;
/// Ranges of the memory map left for the cuts of store staging (a
/// dispatch has had six views staged: two ranges each).
const SPARE: usize = 12;
/// Card memory left free when a home is made.
const MARGIN: u64 = 32 << 20;
/// What a home's going waits for its copy at most (ns).
const PATIENCE: u64 = 2_000_000_000;

/// Homes made, homes gone (pressure or no range; the memory freed),
/// clusters turned down for the map, for the card; write faults, those
/// at a byte of the cluster, pages gone up, runs, bytes; write faults
/// through another address of a page.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
const MADE: usize = 0;
const LEFT: usize = 1;
const FREED: usize = 2;
const NO_RANGE: usize = 3;
const NO_ROOM: usize = 4;
const FAULTS: usize = 5;
const INSIDE: usize = 6;
const PAGES: usize = 7;
const RUNS: usize = 8;
const BYTES: usize = 9;
const ELSEWHERE: usize = 10;

/// Counts; the count before.
fn count(what: usize, n: u64) -> u64 {
    STATS[what].fetch_add(n, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Write tracking of the clusters' pages

/// A page's state: read-only (a store faults), being copied by a fault,
/// open (copied, listed in [`WRITTEN`]), held (read-only while the home
/// goes: a store waits).
const ARMED: u8 = 1;
const BUSY: u8 = 2;
const OPEN: u8 = 3;
const HELD: u8 = 4;

/// The pages of a home's cluster.
struct Span {
    /// The first page, and the cluster's bytes.
    first: usize,
    at: usize,
    end: usize,
    state: Box<[AtomicU8]>,
    /// The other addresses the CPU sees the cluster's 64 KiB pages at
    /// (loose memory is views of physical pages, and the title maps them
    /// twice): (the other address, the page here), by address. `xmem`
    /// tells of every view that comes or goes ([`mapped`], [`unmapped`]).
    others: Mutex<Vec<(usize, usize)>>,
}

impl Span {
    fn pages(&self) -> usize {
        self.state.len() * SMALL
    }

    /// The index of the 4 KiB page at `page`: the cluster's own address
    /// or another address of it.
    fn index(&self, page: usize) -> Option<usize> {
        let here = if page >= self.first && page < self.first + self.pages() {
            page
        } else {
            let others = lock(&self.others);
            let &(other, here) = others.get(others.partition_point(|o| o.0 <= page).checked_sub(1)?)?;
            if page >= other + crate::track::PAGE {
                return None;
            }
            here + (page - other)
        };
        (here >= self.first && here < self.first + self.pages()).then(|| (here - self.first) / SMALL)
    }

    /// Every address the CPU sees page `i` at, its own first.
    fn places(&self, i: usize) -> Vec<usize> {
        let here = self.first + i * SMALL;
        let page = here & !(crate::track::PAGE - 1);
        std::iter::once(here).chain(lock(&self.others).iter().filter(|o| o.1 == page).map(|o| o.0 + (here - page))).collect()
    }

    /// Protects page `i` at every address the CPU sees it at.
    fn protect(&self, i: usize, protection: u32) {
        for place in self.places(i) {
            protect(place, SMALL, protection);
        }
    }

    /// Protects the cluster's pages at the other address `other` of the
    /// 64 KiB page `here`.
    fn protect_other(&self, (other, here): (usize, usize), protection: u32) {
        let lo = here.max(self.first);
        let hi = (here + crate::track::PAGE).min(self.first + self.pages());
        if lo < hi {
            protect(other + (lo - here), hi - lo, protection);
        }
    }

    /// The indices of the cluster's pages `[at, at + len)` shows, with
    /// the address each is shown at there (its own or another).
    fn shown_in(&self, at: usize, len: usize) -> Vec<(usize, usize)> {
        let end = at.saturating_add(len);
        let mut runs = vec![(self.first, self.first)];
        runs.extend(lock(&self.others).iter().filter(|o| o.0 < end && at < o.0 + crate::track::PAGE).copied());
        let mut out = Vec::new();
        for (shown, here) in runs {
            let whole = if shown == self.first { self.pages() } else { crate::track::PAGE };
            for page in (at.max(shown) & !(SMALL - 1)..end.min(shown + whole)).step_by(SMALL) {
                let i = (here + (page - shown)).wrapping_sub(self.first) / SMALL;
                if here + (page - shown) >= self.first && i < self.state.len() {
                    out.push((i, page));
                }
            }
        }
        out
    }
}

static SPANS: RwLock<Vec<Arc<Span>>> = RwLock::new(Vec::new());
/// Pages stored into since the last flush, each as it was before.
static WRITTEN: Mutex<Vec<(usize, Box<[u8; SMALL]>)>> = Mutex::new(Vec::new());

/// Protects the pages of `[at, at + len)`; false if some page was not
/// (it is not there). Page by page where the range is not one allocation
/// (loose memory is views of physical pages side by side).
fn protect(at: usize, len: usize, protection: u32) -> bool {
    let mut old = 0;
    // SAFETY: pages of title memory; pages not there fail the call.
    let one = |at: usize, len: usize, old: &mut u32| unsafe { VirtualProtect(at as *const _, len, protection, old) != 0 };
    one(at, len, &mut old) || (len > SMALL && (at..at + len).step_by(SMALL).fold(true, |all, page| one(page, SMALL, &mut old) && all))
}

/// The page at `page` as it is now; `None` if it is not there.
fn read(page: usize) -> Option<[u8; SMALL]> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut now = [0u8; SMALL];
    let mut done = 0;
    // SAFETY: the kernel checks the source; the destination is ours.
    (unsafe { ReadProcessMemory(GetCurrentProcess(), page as *const _, now.as_mut_ptr().cast(), SMALL, &mut done) } != 0).then_some(now)
}

/// The span holding the page at `page` (its own address or another)
/// and the page's index in it.
fn span_of(page: usize) -> Option<(Arc<Span>, usize)> {
    let spans = SPANS.read().unwrap_or_else(|e| e.into_inner());
    spans.iter().find_map(|s| s.index(page).map(|i| (s.clone(), i)))
}

/// Page `i` of `span`, armed until now (the caller's exchange made it
/// busy): copied as it is, listed, and opened at `page`, the address the
/// store goes through (another address stays read-only until a store
/// comes through it).
fn open(span: &Span, i: usize, page: usize) {
    let mut before = Box::new([0u8; SMALL]);
    // SAFETY: a page of title memory we keep read-only.
    unsafe { std::ptr::copy_nonoverlapping(page as *const u8, before.as_mut_ptr(), SMALL) };
    protect(page, SMALL, PAGE_READWRITE);
    span.state[i].store(OPEN, Ordering::Release);
    // Listed once open: a flush that takes the list has the page, or the
    // next one has.
    lock(&WRITTEN).push((span.first + i * SMALL, before));
}

/// A store faulted at `at`: if the page is a cluster's, it is copied as
/// it was (the first store since the last flush) and opened, and the
/// store goes on (true). Takes no lock a writer of title memory may hold.
pub(crate) fn write_fault(at: usize) -> bool {
    let page = at & !(SMALL - 1);
    let Some((span, i)) = span_of(page) else { return false };
    let here = span.first + i * SMALL + (at - page);
    count(FAULTS, 1);
    count(INSIDE, u64::from(here >= span.at && here < span.end));
    count(ELSEWHERE, u64::from(page != span.first + i * SMALL));
    loop {
        match span.state[i].load(Ordering::Acquire) {
            ARMED => {
                if span.state[i].compare_exchange(ARMED, BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    open(&span, i, page);
                    return true;
                }
            }
            // Another thread's fault is copying it: the store waits for that.
            BUSY => std::hint::spin_loop(),
            // Its home is going: the store waits until title memory holds
            // the home's bytes.
            HELD => std::thread::yield_now(),
            // Open by its state: read-only at this address still, or by
            // someone else's doing.
            _ => {
                protect(page, SMALL, PAGE_READWRITE);
                return true;
            }
        }
    }
}

/// A host call is about to write `[at, at + len)` from the kernel
/// (`crate::track::writing`), or the title sets its protection itself
/// (`crate::track::released`): the armed pages there are opened as a
/// fault opens them.
pub(crate) fn host_writes(at: usize, len: usize) {
    let spans: Vec<Arc<Span>> = SPANS.read().unwrap_or_else(|e| e.into_inner()).to_vec();
    for s in spans {
        for (i, page) in s.shown_in(at, len) {
            if s.state[i].compare_exchange(ARMED, BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                open(&s, i, page);
            } else if s.state[i].load(Ordering::Acquire) == OPEN {
                protect(page, SMALL, PAGE_READWRITE);
            }
        }
    }
}

/// Write tracking opened `[at, at + len)` (a 64 KiB page of its own): the
/// armed pages there are read-only again.
pub(crate) fn opened(at: usize, len: usize) {
    let spans = SPANS.read().unwrap_or_else(|e| e.into_inner());
    for s in spans.iter() {
        for (i, page) in s.shown_in(at, len) {
            if s.state[i].load(Ordering::Acquire) == ARMED {
                protect(page, SMALL, PAGE_READONLY);
            }
        }
    }
}

/// `xmem` mapped a view page at `at` over memory `others` show too: if a
/// cluster's page is among them, `at` is another address of it from here
/// on, read-only where the page is armed (told before the title has the
/// address, so no store through it is missed).
pub(crate) fn mapped(at: usize, others: &[usize]) {
    let spans = SPANS.read().unwrap_or_else(|e| e.into_inner());
    for s in spans.iter() {
        let Some(here) = others.iter().find_map(|&o| s.index(o).map(|i| (s.first + i * SMALL) & !(crate::track::PAGE - 1))) else { continue };
        let mut all = lock(&s.others);
        all.retain(|o| o.0 != at);
        let k = all.partition_point(|o| o.0 < at);
        all.insert(k, (at, here));
        drop(all);
        for (i, page) in s.shown_in(at, crate::track::PAGE) {
            if page >= at && s.state[i].load(Ordering::Acquire) != OPEN {
                protect(page, SMALL, PAGE_READONLY);
            }
        }
    }
}

/// `xmem` takes the view page at `at` away: no cluster's other address
/// any more.
pub(crate) fn unmapped(at: usize) {
    for s in SPANS.read().unwrap_or_else(|e| e.into_inner()).iter() {
        lock(&s.others).retain(|o| o.0 != at);
    }
}

/// Whether `page` is a cluster's (read-only by us, or open): title memory
/// all the same (`vulkan/loose.rs`).
pub(crate) fn tracks(page: usize) -> bool {
    span_of(page & !(SMALL - 1)).is_some()
}

/// The pages of `[at, at + len)` are tracked from here: read-only, at
/// `others` (the other address, the page here) too.
fn track(at: u64, len: u64, mut others: Vec<(usize, usize)>) -> Arc<Span> {
    let first = at as usize & !(SMALL - 1);
    let pages = ((at + len) as usize - first).div_ceil(SMALL);
    others.sort_unstable();
    let span = Arc::new(Span { first, at: at as usize, end: (at + len) as usize, state: (0..pages).map(|_| AtomicU8::new(ARMED)).collect(), others: Mutex::new(others.clone()) });
    SPANS.write().unwrap_or_else(|e| e.into_inner()).push(span.clone());
    if !protect(first, pages * SMALL, PAGE_READONLY) {
        tracing::warn!(at = format_args!("{at:#x}"), mb = len >> 20, "Vulkan: pages of a home's cluster took no protection; CPU stores into them do not reach the home");
    }
    for o in others {
        span.protect_other(o, PAGE_READONLY);
    }
    span
}

/// Every page of `span` is read-only and stores into it wait; the pages
/// that were open, each as it was before the CPU stored into it.
fn hold(span: &Span) -> Vec<(usize, Box<[u8; SMALL]>)> {
    for (i, state) in span.state.iter().enumerate() {
        loop {
            match state.load(Ordering::Acquire) {
                BUSY => std::hint::spin_loop(),
                OPEN => {
                    if state.compare_exchange(OPEN, HELD, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                        span.protect(i, PAGE_READONLY);
                        break;
                    }
                }
                was => {
                    if state.compare_exchange(was, HELD, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                        break;
                    }
                }
            }
        }
    }
    let mut written = lock(&WRITTEN);
    let (ours, others) = std::mem::take(&mut *written).into_iter().partition(|(page, _)| *page >= span.first && *page < span.first + span.pages());
    *written = others;
    ours
}

/// Tracking of `span` ends: its pages are open (if they are still there).
fn untrack(span: &Arc<Span>) {
    SPANS.write().unwrap_or_else(|e| e.into_inner()).retain(|s| !Arc::ptr_eq(s, span));
    lock(&WRITTEN).retain(|(page, _)| *page < span.first || *page >= span.first + span.pages());
    protect(span.first, span.pages(), PAGE_READWRITE);
    for o in std::mem::take(&mut *lock(&span.others)) {
        span.protect_other(o, PAGE_READWRITE);
    }
    for state in span.state.iter() {
        state.store(OPEN, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// The homes

pub(super) struct Home {
    at: u64,
    len: u64,
    buffer: Buffer,
    span: Arc<Span>,
    /// The memory map shows it (a map without ranges left does not: it
    /// goes at the next flush).
    shown: bool,
    /// Its turn among the homes made.
    turn: u64,
    /// The changes that made its cluster ripe ([`Asking`]).
    needs: u32,
}

/// A cluster that asks for a home: its bytes, what [`SAMPLES`] of its
/// pages held when last looked at, how often one had changed, and the
/// changes that make it ripe ([`RIPE`]; twice as many each time its home
/// went under the pool's pressure, so that a home the card has no room
/// for is tried ever more rarely).
#[derive(Default)]
struct Asking {
    len: u64,
    held: [u64; SAMPLES],
    changes: u32,
    needs: u32,
}

impl Asking {
    /// Looks at the pages again.
    fn look(&mut self, at: u64) {
        for (k, held) in self.held.iter_mut().enumerate() {
            let page = (at + self.len * (2 * k as u64 + 1) / (2 * SAMPLES as u64)) as usize & !(SMALL - 1);
            let now = read(page).map_or(0, |p| super::census::hash(&p));
            self.changes += u32::from(*held != 0 && now != 0 && now != *held);
            *held = now;
        }
    }
}

#[derive(Default)]
pub(super) struct Homes {
    live: BTreeMap<u64, Home>,
    /// Clusters that ask for a home, by first address.
    wanted: BTreeMap<u64, Asking>,
    /// Homes gone, freed once nothing that may name them can still run.
    dying: Vec<(Mark, Buffer)>,
    turns: u64,
    /// The rounds in a row that found the pool's GPU view short.
    short: u32,
}

impl Homes {
    /// The home's buffer and offset of title address `title`.
    pub(super) fn of(&self, title: u64) -> Option<(vk::Buffer, u64)> {
        let (_, h) = self.live.range(..=title).next_back()?;
        (h.shown && title < h.at + h.len).then(|| (h.buffer.buffer, title - h.at))
    }

    /// Whether a home holds a byte of `[at, at + len)`, and whether one
    /// holds them all.
    pub(super) fn over(&self, at: u64, len: u64) -> (bool, bool) {
        let end = at.saturating_add(len.max(1));
        let mut over = self.live.range(..end).rev().take_while(|(_, h)| h.at + h.len > at).filter(|(_, h)| h.shown);
        match over.next() {
            Some((_, h)) => (true, h.at <= at && end <= h.at + h.len),
            None => (false, false),
        }
    }

    /// `ranges` (sorted, disjoint) with each home's bytes at its buffer:
    /// what lay under them is cut away. A home the map has no ranges left
    /// for is not shown.
    pub(super) fn cut(&mut self, mut ranges: Vec<Range>, most: usize) -> Vec<Range> {
        for h in self.live.values_mut() {
            let (at, end) = (h.at, h.at + h.len);
            let mut next = Vec::with_capacity(ranges.len() + 2);
            for r in &ranges {
                let (lo, hi) = (r.title_start, r.title_start + r.title_size);
                if hi <= at || lo >= end {
                    next.push(*r);
                    continue;
                }
                if lo < at {
                    next.push(Range { title_size: at - lo, ..*r });
                }
                if hi > end {
                    next.push(Range { title_start: end, title_size: hi - end, device_base: r.device_base + (end - lo) });
                }
            }
            next.push(Range { title_start: at, title_size: h.len, device_base: h.buffer.address });
            h.shown = next.len() <= most;
            if h.shown {
                ranges = next;
            }
        }
        ranges
    }

    /// Whether `[at, at + len)` has a home or lies in a cluster that asks
    /// for one.
    fn known(&self, at: u64, len: u64) -> bool {
        self.over(at, len).1 || self.wanted.range(..=at).next_back().is_some_and(|(&a, w)| at + len <= a + w.len)
    }

    /// Buffer memory `[at, at + len)` asks for a home: joined with the
    /// asking clusters it touches.
    fn want(&mut self, at: u64, len: u64) {
        if self.known(at, len) {
            return;
        }
        let (mut lo, mut hi) = (at, at + len);
        let joined: Vec<u64> = self.wanted.range(..=hi).filter(|&(&a, w)| a + w.len >= lo).map(|(&a, _)| a).collect();
        for a in joined {
            if let Some(w) = self.wanted.remove(&a) {
                lo = lo.min(a);
                hi = hi.max(a + w.len);
            }
        }
        self.wanted.insert(lo, Asking { len: hi - lo, needs: RIPE, ..Asking::default() });
    }

    /// The home at `at` leaves the map; `back`: its bytes go back to
    /// title memory in one copy first. CPU stores into the cluster wait
    /// meanwhile ([`hold`]): what the open pages hold goes up, then the
    /// copy runs, then the pages are opened for good. (A store between
    /// the two would be overwritten by the card's older byte.) The caller
    /// makes the map again.
    fn leave(m: &mut Memory, b: &Backend, at: u64, back: bool) {
        let Some(h) = m.homes.live.remove(&at) else { return };
        let ram = back.then(|| m.imported_whole(h.at, h.len)).flatten();
        if let Some((ram, offset)) = ram {
            let mut copies = Vec::new();
            for (page, before) in hold(&h.span) {
                changed(&h, page, &before, |from, n| copies.push(vk::BufferCopy { src_offset: offset + (from - h.at), dst_offset: from - h.at, size: n }));
            }
            let whole = [vk::BufferCopy { src_offset: 0, dst_offset: offset, size: h.len }];
            let given = b.timeline.record(&b.host, |d, cmd| {
                // SAFETY: a command buffer of ours; the home and the import
                // hold the regions.
                unsafe {
                    if !copies.is_empty() {
                        d.cmd_copy_buffer(cmd, ram, h.buffer.buffer, &copies);
                        super::images::full_barrier(d, cmd);
                    }
                    d.cmd_copy_buffer(cmd, h.buffer.buffer, ram, &whole);
                }
            });
            match given {
                Ok(ticket) => {
                    if !b.timeline.wait(&b.host, ticket, PATIENCE) {
                        tracing::warn!(at = format_args!("{:#x}", h.at), "Vulkan: a home's bytes were not back in title memory in time");
                    }
                }
                Err(e) => tracing::error!(error = %format!("{e:#}"), at = format_args!("{:#x}", h.at), "Vulkan: a home's bytes did not go back to title memory"),
            }
        } else if back {
            tracing::warn!(at = format_args!("{:#x}", h.at), mb = h.len >> 20, "Vulkan: a home went with no one import under it; title memory there is as it was when the home was made");
        }
        untrack(&h.span);
        count(if back { LEFT } else { FREED }, 1);
        tracing::info!(at = format_args!("{:#x}", h.at), kb = h.len >> 10, back, homes = m.homes.live.len(), "Vulkan: a home is gone");
        m.homes.dying.push((b.timeline.mark(), h.buffer));
    }

    /// Makes the home of the cluster `[at, at + len)`: card memory, the
    /// pages tracked, the bytes copied in behind everything given so far.
    fn make(m: &mut Memory, b: &Backend, (at, len, needs): (u64, u64, u32), others: Vec<(usize, usize)>) -> bool {
        if m.want_range(at, len) && m.import_regions(&b.host) {
            m.rebuild_map();
        }
        let Some((ram, offset)) = m.imported_whole(at, len) else { return false };
        let buffer = match Buffer::new(&b.host, len, TITLE_USAGE, Place::Card) {
            Ok(buffer) => buffer,
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), mb = len >> 20, "Vulkan: no card memory for a home");
                return false;
            }
        };
        // Tracked before the copy is given: a store from here on goes up
        // behind it.
        let span = track(at, len, others);
        let whole = [vk::BufferCopy { src_offset: offset, dst_offset: 0, size: len }];
        // SAFETY: a command buffer of ours; the import and the home hold
        // the region.
        if let Err(e) = b.timeline.record(&b.host, |d, cmd| unsafe { d.cmd_copy_buffer(cmd, ram, buffer.buffer, &whole) }) {
            tracing::error!(error = %format!("{e:#}"), "Vulkan: a home was not filled; it is not made");
            untrack(&span);
            m.homes.dying.push((b.timeline.mark(), buffer));
            return false;
        }
        m.homes.turns += 1;
        count(MADE, 1);
        tracing::info!(
            at = format_args!("{at:#x}"),
            kb = len >> 10,
            device = format_args!("{:#x}", buffer.address),
            homes = m.homes.live.len() + 1,
            recordings_open = b.timeline.open(),
            "Vulkan: a home on the card for buffers programs store into"
        );
        let turn = m.homes.turns;
        m.homes.live.insert(at, Home { at, len, buffer, span, shown: true, turn, needs });
        true
    }
}

/// The runs of bytes of `home`'s cluster in the page at `page` that
/// differ from `before` (the page as it was before the CPU stored into
/// it), as (title address, bytes).
fn changed(home: &Home, page: usize, before: &[u8; SMALL], mut run: impl FnMut(u64, u64)) {
    let Some(now) = read(page) else { return };
    let lo = (home.at as usize).max(page) - page;
    let hi = ((home.at + home.len) as usize).min(page + SMALL) - page;
    let mut i = lo;
    while i < hi {
        if now[i] == before[i] {
            i += 1;
            continue;
        }
        let from = i;
        while i < hi && now[i] != before[i] {
            i += 1;
        }
        count(RUNS, 1);
        count(BYTES, (i - from) as u64);
        run((page + from) as u64, (i - from) as u64);
    }
}

impl Memory {
    /// The import and offset of title memory `[at, at + len)` where one
    /// imported buffer holds it all.
    fn imported_whole(&self, at: u64, len: u64) -> Option<(vk::Buffer, u64)> {
        let (buffer, offset) = self.imported_of(at)?;
        (self.imported_of(at + len - 1) == Some((buffer, offset + len - 1))).then_some((buffer, offset))
    }

    /// Whether `[at, at + len)` lies in memory whose buffers get homes: a
    /// region the title asked for write-combined (true), or loose memory
    /// (false: only where the CPU sees every page of it at another
    /// address too, [`view`]).
    fn takes_homes(&self, at: u64, len: u64) -> Option<bool> {
        let (&a, r) = self.regions.range(..=at as usize).next_back()?;
        (at + len <= (a + r.len) as u64 && (r.combined || r.loose)).then_some(r.combined)
    }

    /// Diagnostics: the loose regions, as (first address, end).
    pub(super) fn loose_spans(&self) -> Vec<(u64, u64)> {
        self.regions.iter().filter(|(_, r)| r.loose).map(|(&a, r)| (a as u64, (a + r.len) as u64)).collect()
    }

    /// The homes that hold a byte of `[at, at + len)` go: the title frees
    /// the memory (nothing copied), or `xmem` is about to take a view of
    /// physical pages away (`back`: the pages stay, and what the card
    /// stored goes back into them first). Whether any went.
    pub(super) fn homes_freed(&mut self, b: &Backend, at: u64, len: u64, back: bool) -> bool {
        self.homes.wanted.retain(|&a, w| a + w.len <= at || a >= at + len);
        let gone: Vec<u64> = self.homes.live.values().filter(|h| h.at < at + len && at < h.at + h.len).map(|h| h.at).collect();
        for &a in &gone {
            Homes::leave(self, b, a, back);
        }
        !gone.is_empty()
    }

    /// An announced write of `[at, at + len)` (DirectStorage, a file read:
    /// through views of our own or from the kernel, no fault to see):
    /// what of it lies in a home goes up.
    pub(super) fn homes_written(&self, b: &Backend, at: u64, len: u64) {
        let over = self.homes.live.values().filter(|h| h.at < at + len && at < h.at + h.len);
        let copies: Vec<_> = over
            .filter_map(|h| {
                let (lo, hi) = (at.max(h.at), (at + len).min(h.at + h.len));
                let (ram, offset) = self.imported_whole(lo, hi - lo)?;
                Some((ram, h.buffer.buffer, vk::BufferCopy { src_offset: offset, dst_offset: lo - h.at, size: hi - lo }))
            })
            .collect();
        upload(b, &copies);
    }
}

/// The title made an unordered-access view of `view` bytes in the buffer
/// memory `[base, base + len)`.
pub(super) fn view(b: &Backend, base: u64, len: u64, view: u64) {
    if view < LEAST || len == 0 {
        return;
    }
    let combined = {
        let m = memory(b);
        match m.takes_homes(base, len) {
            Some(combined) if !m.homes.known(base, len) => combined,
            _ => return,
        }
    };
    // Loose memory: only the card's side of physical pages the title's
    // CPU sees at an address of its own (its 32-bit arena is loose
    // memory too, shown once, and the CPU reads and writes it all the
    // time). Asked before the memory lock: `xmem` answers under its own.
    if !combined && crate::track::elsewhere(base, len).iter().any(|p| p.1.is_empty()) {
        return;
    }
    memory(b).homes.want(base, len);
}

/// Store staging staged the view `[at, at + len)`: the card stores into
/// it, whatever its pages look like (a program that stores the bytes
/// already there changes none), so the cluster it lies in is ripe.
pub(super) fn staged(b: &Backend, at: u64, len: u64) {
    let mut m = memory(b);
    if let Some((_, w)) = m.homes.wanted.range_mut(..=at).next_back().filter(|(a, w)| at + len <= **a + w.len) {
        w.changes = w.changes.max(w.needs);
    }
}

/// Before work is given to the card: a home the map does not show goes
/// (its bytes back first), and the bytes of the clusters the CPU stored
/// into since the last flush go to their homes.
pub(super) fn flush(b: &Backend) {
    if SPANS.read().unwrap_or_else(|e| e.into_inner()).is_empty() {
        return;
    }
    let mut m = memory(b);
    let hidden: Vec<u64> = m.homes.live.values().filter(|h| !h.shown).map(|h| h.at).collect();
    for &at in &hidden {
        Homes::leave(&mut m, b, at, true);
    }
    if !hidden.is_empty() {
        m.rebuild_map();
    }
    upload(b, &m.stored());
}

/// Copies what the CPU stored to the homes (the import, the home, the
/// bytes), on the host line: behind everything given before.
fn upload(b: &Backend, copies: &[(vk::Buffer, vk::Buffer, vk::BufferCopy)]) {
    if copies.is_empty() {
        return;
    }
    // SAFETY: a command buffer of ours; imports and homes that hold the
    // regions.
    let given = b.timeline.record(&b.host, |d, cmd| unsafe {
        for (ram, home, region) in copies {
            d.cmd_copy_buffer(cmd, *ram, *home, std::slice::from_ref(region));
        }
    });
    if let Err(e) = given {
        tracing::error!(error = %format!("{e:#}"), "Vulkan: CPU stores into a home's cluster did not reach the home");
    }
}

impl Memory {
    /// The pages stored into since the last flush are protected again,
    /// and what the CPU changed of their clusters is to be copied: (the
    /// import, the home, the bytes). A page a host call is writing stays
    /// open, and listed; one with no import under it now stays listed.
    fn stored(&self) -> Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)> {
        let mut copies = Vec::new();
        let mut kept = Vec::new();
        for (page, before) in std::mem::take(&mut *lock(&WRITTEN)) {
            let Some((span, i)) = span_of(page) else { continue };
            let Some(home) = self.homes.live.get(&(span.at as u64)) else { continue };
            // Protected again first: a store from here on is the next
            // flush's.
            let held = span.places(i).iter().any(|&place| crate::track::host_holds(place, SMALL));
            if !held {
                span.state[i].store(ARMED, Ordering::Release);
                span.protect(i, PAGE_READONLY);
            }
            let mut missed = false;
            changed(home, page, &before, |from, n| match self.imported_of(from) {
                Some((ram, offset)) => copies.push((ram, home.buffer.buffer, vk::BufferCopy { src_offset: offset, dst_offset: from - home.at, size: n })),
                None => missed = true,
            });
            count(PAGES, 1);
            if held || missed {
                kept.push((page, before));
            }
        }
        lock(&WRITTEN).extend(kept);
        copies
    }

    /// The pool's GPU view comes first: while pages programs read by
    /// address have no card page (they read zeros; the world's load does
    /// that for a while with no home at all) no home is made, and after
    /// two rounds of it the home made last goes and asks again. Whether
    /// the view is short.
    fn homes_yield(&mut self, b: &Backend) -> bool {
        let zeros: u64 = self.pools.iter().map(|p| p.cold_pages().2).sum();
        self.homes.short = if zeros == 0 { 0 } else { self.homes.short + 1 };
        let last = self.homes.live.values().max_by_key(|h| h.turn).map(|h| (h.at, h.len, h.needs));
        if let Some((at, len, needs)) = last.filter(|_| self.homes.short >= 2) {
            tracing::info!(pages_off_the_card = zeros, "Vulkan: the pool's GPU view has no room for the pages programs read; a home goes");
            Homes::leave(self, b, at, true);
            self.rebuild_map();
            self.homes.want(at, len);
            if let Some(w) = self.homes.wanted.get_mut(&at) {
                w.needs = needs.saturating_mul(2);
            }
        }
        zeros != 0
    }

    /// Why the cluster `[at, at + len)` gets no home now: the memory map
    /// has no ranges for it, or the card no room. Of what the pool's view
    /// holds beyond the pages programs read, half is room (it gives blocks
    /// back as ours passes the limit).
    fn no_home(&self, b: &Backend, at: u64, len: u64) -> Option<usize> {
        if self.map.ranges().len() + 2 + SPARE > MAX_RANGES {
            return Some(NO_RANGE);
        }
        let spare: u64 = self.pools.iter().map(|p| p.card_bytes().saturating_sub(p.stats.pages_in_card.saturating_sub(p.cold_pages().1) * PAGE)).sum();
        let (ours, limit) = (account::card_total(), budget::limit(&b.host));
        // Half of it: the view gives whole blocks back, and the pages
        // programs read lie in all of them (run hs6: a 79 MiB home made on
        // all of it left pages off the card within seconds, twice, and the
        // frames after showed dark spots on the character's suit).
        if ours + len + MARGIN <= limit + spare / 2 {
            return None;
        }
        if STATS[NO_ROOM].load(Ordering::Relaxed) < 16 {
            tracing::info!(at = format_args!("{at:#x}"), mb = len >> 20, ours_mb = ours >> 20, limit_mb = limit >> 20, pool_spare_mb = spare >> 20, "Vulkan: no room on the card for a home");
        }
        Some(NO_ROOM)
    }

    /// The asking clusters are looked at; the ripe one that changed most,
    /// if the map and the card have room for its home (else it asks again
    /// from the start).
    fn homes_ripe(&mut self, b: &Backend) -> Option<(u64, u64)> {
        for (&at, w) in self.homes.wanted.iter_mut() {
            w.look(at);
        }
        let (at, len) = self.homes.wanted.iter().filter(|(_, w)| w.changes >= w.needs).max_by_key(|(_, w)| w.changes).map(|(&at, w)| (at, w.len))?;
        // Its bytes must not straddle a home made before (a buffer that
        // reaches into one): that one goes first.
        let over: Vec<u64> = self.homes.live.values().filter(|h| h.at < at + len && at < h.at + h.len).map(|h| h.at).collect();
        for &a in &over {
            Homes::leave(self, b, a, true);
        }
        if !over.is_empty() {
            self.rebuild_map();
        }
        let Some(why) = self.no_home(b, at, len) else { return Some((at, len)) };
        count(why, 1);
        if let Some(w) = self.homes.wanted.get_mut(&at) {
            w.changes = 0;
        }
        None
    }
}

/// Every present: homes gone are freed once nothing can name them. Every
/// [`ROUND`] presents: under pressure the home made last goes; else the
/// asking clusters are looked at and the ripest gets its home.
pub(super) fn each_frame(b: &Backend) {
    let Some((at, len)) = ripe(b) else { return };
    // The other addresses the CPU sees its pages at: asked before the
    // memory lock (`xmem` answers under its own). A view mapped between
    // this and the tracking below is not seen as one.
    let others = crate::track::elsewhere(at, len).into_iter().flat_map(|(page, others)| others.into_iter().map(move |o| (o, page))).collect();
    let mut m = memory(b);
    if let Some(w) = m.homes.wanted.remove(&at).filter(|w| w.len == len)
        && Homes::make(&mut m, b, (at, len, w.needs), others)
    {
        m.rebuild_map();
    }
}

/// [`each_frame`] under the memory lock: the cluster to make a home for.
fn ripe(b: &Backend) -> Option<(u64, u64)> {
    let mut m = memory(b);
    if m.homes.live.is_empty() && m.homes.wanted.is_empty() && m.homes.dying.is_empty() {
        return None;
    }
    for (mut mark, buffer) in std::mem::take(&mut m.homes.dying) {
        if b.timeline.passed(&b.host, &mut mark) {
            // SAFETY: out of every map, every command buffer begun while
            // it was in one is submitted, and the card has run them.
            unsafe { buffer.destroy(&b.host) };
        } else {
            m.homes.dying.push((mark, buffer));
        }
    }
    if m.tick.is_multiple_of(ROUND) && !m.homes_yield(b) { m.homes_ripe(b) } else { None }
}

/// The 10 s line.
pub(super) fn report(b: &Backend) {
    let n = |i: usize| STATS[i].load(Ordering::Relaxed);
    let (homes, bytes, asking) = {
        let m = memory(b);
        (m.homes.live.len(), m.homes.live.values().map(|h| h.len).sum::<u64>(), m.homes.wanted.values().map(|w| w.len).sum::<u64>())
    };
    if n(MADE) + n(NO_RANGE) + n(NO_ROOM) + asking == 0 {
        return;
    }
    tracing::info!(
        homes,
        mb = bytes >> 20,
        asking_mb = asking >> 20,
        made = n(MADE),
        gone_back = n(LEFT),
        gone_freed = n(FREED),
        no_range = n(NO_RANGE),
        no_room = n(NO_ROOM),
        cpu_write_faults = n(FAULTS),
        at_a_byte_of_the_cluster = n(INSIDE),
        through_another_address = n(ELSEWHERE),
        pages_gone_up = n(PAGES),
        runs = n(RUNS),
        bytes = n(BYTES),
        "Vulkan: homes (card memory for the buffers programs store into; CPU stores into them, and what of those went to the card)"
    );
}
