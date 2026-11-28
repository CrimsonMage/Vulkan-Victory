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
//!   back (run hm0: no CPU read of 45,000 probed pages there). Buffers
//!   that touch are one cluster with one home ([`Homes::want`]). A
//!   cluster gets its home once it is seen to change ([`Asking`]: a few
//!   of its pages are looked at every [`ROUND`] presents; the card stores
//!   into some clusters every frame and into others never), the one that
//!   changed most first, while the memory map and the card have room
//!   ([`each_frame`]).
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
//!   not change stay the card's.
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
/// at a byte of the cluster, pages gone up, runs, bytes.
static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
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
}

impl Span {
    fn pages(&self) -> usize {
        self.state.len() * SMALL
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

/// The span holding page `page` and the page's index in it.
fn span_of(page: usize) -> Option<(Arc<Span>, usize)> {
    let spans = SPANS.read().unwrap_or_else(|e| e.into_inner());
    let s = spans.iter().find(|s| page >= s.first && page < s.first + s.pages())?;
    Some((s.clone(), (page - s.first) / SMALL))
}

/// Page `i` of `span`, armed until now (the caller's exchange made it
/// busy): copied as it is, listed and opened.
fn open(span: &Span, i: usize) {
    let page = span.first + i * SMALL;
    let mut before = Box::new([0u8; SMALL]);
    // SAFETY: a page of title memory we keep read-only.
    unsafe { std::ptr::copy_nonoverlapping(page as *const u8, before.as_mut_ptr(), SMALL) };
    protect(page, SMALL, PAGE_READWRITE);
    span.state[i].store(OPEN, Ordering::Release);
    // Listed once open: a flush that takes the list has the page, or the
    // next one has.
    lock(&WRITTEN).push((page, before));
}

/// A store faulted at `at`: if the page is a cluster's, it is copied as
/// it was (the first store since the last flush) and opened, and the
/// store goes on (true). Takes no lock a writer of title memory may hold.
pub(crate) fn write_fault(at: usize) -> bool {
    let Some((span, i)) = span_of(at & !(SMALL - 1)) else { return false };
    count(FAULTS, 1);
    count(INSIDE, u64::from(at >= span.at && at < span.end));
    loop {
        match span.state[i].load(Ordering::Acquire) {
            ARMED => {
                if span.state[i].compare_exchange(ARMED, BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    open(&span, i);
                    return true;
                }
            }
            // Another thread's fault is copying it: the store waits for that.
            BUSY => std::hint::spin_loop(),
            // Its home is going: the store waits until title memory holds
            // the home's bytes.
            HELD => std::thread::yield_now(),
            // Open by its state, read-only by someone else's doing.
            _ => {
                protect(at & !(SMALL - 1), SMALL, PAGE_READWRITE);
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
    let spans: Vec<Arc<Span>> = SPANS.read().unwrap_or_else(|e| e.into_inner()).iter().filter(|s| at < s.first + s.pages() && s.first < at.saturating_add(len)).cloned().collect();
    for s in spans {
        let lo = (at.max(s.first) - s.first) / SMALL;
        let hi = (at.saturating_add(len).min(s.first + s.pages()) - s.first).div_ceil(SMALL);
        for i in lo..hi {
            if s.state[i].compare_exchange(ARMED, BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                open(&s, i);
            }
        }
    }
}

/// Write tracking opened `[at, at + len)` (a 64 KiB page of its own): the
/// armed pages there are read-only again.
pub(crate) fn opened(at: usize, len: usize) {
    let spans = SPANS.read().unwrap_or_else(|e| e.into_inner());
    for s in spans.iter().filter(|s| at < s.first + s.pages() && s.first < at + len) {
        let lo = (at.max(s.first) - s.first) / SMALL;
        let hi = ((at + len).min(s.first + s.pages()) - s.first).div_ceil(SMALL);
        for i in lo..hi {
            if s.state[i].load(Ordering::Acquire) == ARMED {
                protect(s.first + i * SMALL, SMALL, PAGE_READONLY);
            }
        }
    }
}

/// Whether `page` is a cluster's (read-only by us, or open): title memory
/// all the same (`vulkan/loose.rs`).
pub(crate) fn tracks(page: usize) -> bool {
    span_of(page & !(SMALL - 1)).is_some()
}

/// The pages of `[at, at + len)` are tracked from here: read-only.
fn track(at: u64, len: u64) -> Arc<Span> {
    let first = at as usize & !(SMALL - 1);
    let pages = ((at + len) as usize - first).div_ceil(SMALL);
    let span = Arc::new(Span { first, at: at as usize, end: (at + len) as usize, state: (0..pages).map(|_| AtomicU8::new(ARMED)).collect() });
    SPANS.write().unwrap_or_else(|e| e.into_inner()).push(span.clone());
    if !protect(first, pages * SMALL, PAGE_READONLY) {
        tracing::warn!(at = format_args!("{at:#x}"), mb = len >> 20, "Vulkan: pages of a home's cluster took no protection; CPU stores into them do not reach the home");
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
                        protect(span.first + i * SMALL, SMALL, PAGE_READONLY);
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
}

/// A cluster that asks for a home: its bytes, what [`SAMPLES`] of its
/// pages held when last looked at, and how often one had changed.
#[derive(Default)]
struct Asking {
    len: u64,
    held: [u64; SAMPLES],
    changes: u32,
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
        self.wanted.insert(lo, Asking { len: hi - lo, ..Asking::default() });
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
    fn make(m: &mut Memory, b: &Backend, at: u64, len: u64) -> bool {
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
        let span = track(at, len);
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
        m.homes.live.insert(at, Home { at, len, buffer, span, shown: true, turn });
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
    /// region the title asked for write-combined.
    fn takes_homes(&self, at: u64, len: u64) -> bool {
        self.regions.range(..=at as usize).next_back().is_some_and(|(&a, r)| r.combined && at + len <= (a + r.len) as u64)
    }

    /// The homes that hold a byte of `[at, at + len)` go, nothing copied
    /// (the title freed or unmapped the memory); whether any did.
    pub(super) fn homes_freed(&mut self, b: &Backend, at: u64, len: u64) -> bool {
        self.homes.wanted.retain(|&a, w| a + w.len <= at || a >= at + len);
        let gone: Vec<u64> = self.homes.live.values().filter(|h| h.at < at + len && at < h.at + h.len).map(|h| h.at).collect();
        for &a in &gone {
            Homes::leave(self, b, a, false);
        }
        !gone.is_empty()
    }
}

/// The title made an unordered-access view of `view` bytes in the buffer
/// memory `[base, base + len)`.
pub(super) fn view(b: &Backend, base: u64, len: u64, view: u64) {
    if view < LEAST || len == 0 {
        return;
    }
    {
        let m = memory(b);
        if !m.takes_homes(base, len) || m.homes.known(base, len) {
            return;
        }
    }
    // Memory the CPU sees at another address too (physical pages mapped
    // twice) takes stores we would not see. Asked before the memory
    // lock: `xmem` answers under its own.
    let aliased = crate::track::aliased(base, len);
    if aliased != 0 {
        static SAID: AtomicU64 = AtomicU64::new(0);
        if SAID.fetch_add(1, Ordering::Relaxed) < 16 {
            tracing::info!(at = format_args!("{base:#x}"), kb = len >> 10, pages = aliased, "Vulkan: no home for buffer memory the CPU sees at another address too");
        }
        return;
    }
    memory(b).homes.want(base, len);
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
    let copies = m.stored();
    if copies.is_empty() {
        return;
    }
    // SAFETY: a command buffer of ours; imports and homes that hold the
    // regions.
    let given = b.timeline.record(&b.host, |d, cmd| unsafe {
        for (ram, home, region) in &copies {
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
            let held = crate::track::host_holds(page, SMALL);
            if !held {
                span.state[i].store(ARMED, Ordering::Release);
                protect(page, SMALL, PAGE_READONLY);
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
        let last = self.homes.live.values().max_by_key(|h| h.turn).map(|h| (h.at, h.len));
        if let Some((at, len)) = last.filter(|_| self.homes.short >= 2) {
            tracing::info!(pages_off_the_card = zeros, "Vulkan: the pool's GPU view has no room for the pages programs read; a home goes");
            Homes::leave(self, b, at, true);
            self.rebuild_map();
            self.homes.want(at, len);
        }
        zeros != 0
    }

    /// Why the cluster `[at, at + len)` gets no home now: the memory map
    /// has no ranges for it, or the card no room. What the pool's view
    /// holds beyond the pages programs read is room (it gives blocks back
    /// as ours passes the limit).
    fn no_home(&self, b: &Backend, at: u64, len: u64) -> Option<usize> {
        if self.map.ranges().len() + 2 + SPARE > MAX_RANGES {
            return Some(NO_RANGE);
        }
        let spare: u64 = self.pools.iter().map(|p| p.card_bytes().saturating_sub(p.stats.pages_in_card.saturating_sub(p.cold_pages().1) * PAGE)).sum();
        let (ours, limit) = (account::card_total(), budget::limit(&b.host));
        if ours + len + MARGIN <= limit + spare {
            return None;
        }
        if STATS[NO_ROOM].load(Ordering::Relaxed) < 16 {
            tracing::info!(at = format_args!("{at:#x}"), mb = len >> 20, ours_mb = ours >> 20, limit_mb = limit >> 20, pool_spare_mb = spare >> 20, "Vulkan: no room on the card for a home");
        }
        Some(NO_ROOM)
    }

    /// The asking clusters are looked at, and the ripe one that changed
    /// most gets its home, if the map and the card have room (else it
    /// asks again from the start).
    fn homes_ripen(&mut self, b: &Backend) {
        for (&at, w) in self.homes.wanted.iter_mut() {
            w.look(at);
        }
        let ripe = self.homes.wanted.iter().filter(|(_, w)| w.changes >= RIPE).max_by_key(|(_, w)| w.changes).map(|(&at, w)| (at, w.len));
        let Some((at, len)) = ripe else { return };
        // Its bytes must not straddle a home made before (a buffer that
        // reaches into one): that one goes first.
        let over: Vec<u64> = self.homes.live.values().filter(|h| h.at < at + len && at < h.at + h.len).map(|h| h.at).collect();
        for &a in &over {
            Homes::leave(self, b, a, true);
        }
        if !over.is_empty() {
            self.rebuild_map();
        }
        if let Some(why) = self.no_home(b, at, len) {
            count(why, 1);
            if let Some(w) = self.homes.wanted.get_mut(&at) {
                w.changes = 0;
            }
            return;
        }
        self.homes.wanted.remove(&at);
        if Homes::make(self, b, at, len) {
            self.rebuild_map();
        }
    }
}

/// Every present: homes gone are freed once nothing can name them. Every
/// [`ROUND`] presents: under pressure the home made last goes; else the
/// asking clusters are looked at and the ripest gets its home.
pub(super) fn each_frame(b: &Backend) {
    let mut m = memory(b);
    if m.homes.live.is_empty() && m.homes.wanted.is_empty() && m.homes.dying.is_empty() {
        return;
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
    if m.tick.is_multiple_of(ROUND) && !m.homes_yield(b) {
        m.homes_ripen(b);
    }
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
        pages_gone_up = n(PAGES),
        runs = n(RUNS),
        bytes = n(BYTES),
        "Vulkan: homes (card memory for the buffers programs store into; CPU stores into them, and what of those went to the card)"
    );
}
