//! What a frame's milliseconds are made of: one line every five seconds,
//! `frame time: ...`, beside the gate line.
//!
//! - **The card**: two timestamps around each submission of ours (before
//!   its lists, after them). Everything runs on one queue, so the stamps of consecutive
//!   submissions follow each other: busy is the sum of their spans, idle
//!   what lies between one's end and the next one's start.
//! - **The CPU**: time inside `Host::submit`, every wait for the card by
//!   why ([`Wait`]), the landing of flights on the relay thread, and how
//!   long the title's fence events took to be set.
//! - **Pipelines**: in every 30th frame, a timestamp at each pipeline change
//!   in the title's lists; the five with the most GPU time.
//!
//! Waits shorter than 20 µs are not counted (nothing was in the air).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Why a thread of ours stood still for the card.
#[derive(Clone, Copy)]
pub(crate) enum Wait {
    /// The submission lock (another thread was submitting or settling).
    Lock,
    /// A submission that is waited for settles everything first.
    Settle,
    /// Preparing reads title memory a flight's shaders may write.
    Reads,
    /// A step of ours (`CopyMemoryToMemoryX` on the CPU) waits for all.
    Step,
    /// A title copy on the CPU into title memory flights may read.
    CopyIn,
    /// 256 host lists in the air.
    Lists,
    /// The title blocks on a fence without an event.
    Fence,
    /// A submission that is waited for: the card running it.
    Own,
}

const WAITS_NAMED: [&str; 8] = ["lock", "settle", "reads", "step", "copy-in", "lists", "fence", "own"];

/// Per [`Wait`]: nanoseconds and how many.
static WAITS: [[AtomicU64; 2]; 8] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; 8];

/// Plain sums ([`Sum`]).
static SUMS: [AtomicU64; SUMS_N] = [const { AtomicU64::new(0) }; SUMS_N];
const SUMS_N: usize = 14;

#[derive(Clone, Copy)]
pub(crate) enum Sum {
    /// Nanoseconds inside `Host::submit`, waits included.
    SubmitNs,
    Submits,
    /// Nanoseconds the relay thread spent landing flights.
    LandNs,
    /// The title's fence events: set, and microseconds from the ask to the set.
    FenceSet,
    FenceUs,
    /// Bytes of in-place buffers (title memory the card reads over the bus)
    /// named by lists, once per list.
    InPlaceBytes,
    /// Bytes of buffers of their own (title memory that is not in place,
    /// filled at each submission that names it) named by lists, once per
    /// list.
    CopyBytes,
    /// Of `SubmitNs`: bringing textures up to date (detiling, hashing).
    TextureNs,
    /// Bytes of windows over scattered pages (`crate::gather`, the
    /// fallback: copies on the card of the title's pages) named by lists,
    /// once per list.
    WindowBytes,
    /// Copies into such windows from their pages (runs of pages the title
    /// changed), and their bytes.
    Gathers,
    GatherBytes,
    /// Bytes of pool views (title memory read and written by directory,
    /// `crate::memory`) named by lists, once per list.
    PoolBytes,
    /// Blocks of the directory (4 KiB) copied to the card.
    DirectoryBlocks,
    /// Nanoseconds the copy-queue workers spent detiling first uploads
    /// (`crate::texture::ahead`), outside any submission.
    AheadNs,
}

pub(crate) fn add(sum: Sum, n: u64) {
    SUMS[sum as usize].fetch_add(n, Relaxed);
}

/// The parts of a submission's preparing (`Host::submit`), for the split
/// of "in submissions".
#[derive(Clone, Copy)]
pub(crate) enum Phase {
    /// The submission lock and the flights in the air (the waits named
    /// beside, as waits).
    Enter,
    /// Descriptors the title copied into heaps as bytes (`sync_heaps`).
    Descriptors,
    /// The sweep of idle copies, slots that waited for a texture, the
    /// announcement of finished shader writes.
    Upkeep,
    /// The lists' steps on the CPU: fills, copies, textures; and the
    /// timestamps.
    Steps,
    /// `ExecuteCommandLists` and the signal.
    Execute,
}

const PHASES: usize = 5;
const PHASES_NAMED: [&str; PHASES] = ["enter", "descriptors", "upkeep", "steps", "execute"];

/// Per [`Phase`]: ticks.
static PHASE_SUMS: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];

/// The part `phase` of a submission ran from `since` (`crate::fast::ticks`)
/// until now; returns now, the start of the next part.
pub(crate) fn phase(phase: Phase, since: u64) -> u64 {
    let now = crate::fast::ticks();
    PHASE_SUMS[phase as usize].fetch_add(now.wrapping_sub(since), Relaxed);
    now
}

/// Parts of a dispatch's binding (`draw.rs` `State::bind` for compute),
/// for the split of "dispatch" in the line: ticks and how many.
#[derive(Clone, Copy)]
pub(crate) enum Part {
    /// Bindings done whole, and those left out as bound already (no ticks).
    Whole,
    Skipped,
    /// The descriptors the stages read through tables (views made, buffers
    /// named, textures readable).
    Tables,
    /// Title memory by address (`crate::memory::ready`).
    Memory,
    /// Root views (constant, shader resource and unordered access).
    Roots,
}

const PARTS: usize = 5;
const PARTS_NAMED: [&str; PARTS] = ["whole", "skipped", "tables", "by-address memory", "root views"];
static PART_SUMS: [[AtomicU64; 2]; PARTS] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; PARTS];

/// The part `part` of a dispatch's binding ran from `since`
/// (`crate::fast::ticks`) until now; returns now.
pub(crate) fn part(part: Part, since: u64) -> u64 {
    let now = crate::fast::ticks();
    PART_SUMS[part as usize][0].fetch_add(now.wrapping_sub(since), Relaxed);
    PART_SUMS[part as usize][1].fetch_add(1, Relaxed);
    now
}

/// What a recording call of the title was, for the split of the
/// recording time ([`Timed`]).
#[derive(Clone, Copy)]
pub(crate) enum Call {
    /// Root arguments, heaps, views, targets, viewports, pipeline: kept
    /// for the next draw or set at once.
    State,
    Draw,
    /// Mesh dispatches (the world's geometry).
    Mesh,
    Dispatch,
    /// Indirect executions.
    Indirect,
    /// List resets.
    Reset,
    /// Work of ours recorded into the title's list (`record_own`).
    Own,
    Barrier,
    /// Buffer, texture and memory copies, resolves, query resolves.
    Copy,
    Clear,
    /// Waits for and writes of memory values.
    Value,
}

const CALLS: usize = 11;
const CALLS_NAMED: [&str; CALLS] =
    ["state", "draw", "mesh", "dispatch", "indirect", "reset", "own", "barrier", "copy", "clear", "value"];
/// The kinds up to here go into "recording into our lists" as before
/// (the calls that pass through the list's draw state); the others are
/// shown beside it.
const RECORDING: usize = Call::Own as usize + 1;

/// Per [`Call`]: ticks and calls.
static CALL_SUMS: [[AtomicU64; 2]; CALLS] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; CALLS];

/// Times one recording call of the title, by kind, from its making to its
/// drop. Only the outermost on a thread counts (a copy that records a
/// draw of ours is a copy). Summed on the thread and added to the shared
/// sums every 64 calls: one shared counter bumped by six threads at every
/// call was a cache line passed between cores a dozen times a draw. What
/// a thread holds back is at most 63 calls, well under a millisecond in a
/// 5 s line.
pub(crate) struct Timed {
    kind: Call,
    since: u64,
    outer: bool,
}

#[derive(Clone, Copy)]
struct Held {
    depth: u32,
    calls: u32,
    sums: [(u64, u64); CALLS],
}

thread_local! {
    static HELD: std::cell::Cell<Held> =
        const { std::cell::Cell::new(Held { depth: 0, calls: 0, sums: [(0, 0); CALLS] }) };
}

impl Timed {
    pub(crate) fn new(kind: Call) -> Timed {
        let outer = HELD.with(|held| {
            let mut h = held.get();
            h.depth += 1;
            held.set(h);
            h.depth == 1
        });
        Timed { kind, since: if outer { crate::fast::ticks() } else { 0 }, outer }
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        let ticks = if self.outer { crate::fast::ticks().wrapping_sub(self.since) } else { 0 };
        HELD.with(|held| {
            let mut h = held.get();
            h.depth -= 1;
            if !self.outer {
                held.set(h);
                return;
            }
            let k = self.kind as usize;
            h.sums[k].0 += ticks;
            h.sums[k].1 += 1;
            h.calls += 1;
            if h.calls >= 64 {
                for (sum, &(t, n)) in CALL_SUMS.iter().zip(&h.sums) {
                    if n > 0 {
                        sum[0].fetch_add(t, Relaxed);
                        sum[1].fetch_add(n, Relaxed);
                    }
                }
                h.sums = [(0, 0); CALLS];
                h.calls = 0;
            }
            held.set(h);
        });
    }
}

/// The title's side of the list calls (`crate::list` `thin`): ticks and
/// calls of writing them down; the workers' side: ticks and calls of
/// translating them (the by-call sums above are taken there too, inside the
/// list methods the workers run); submissions' ticks waiting until their
/// lists were translated, and those submissions.
static WRITTEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static TRANSLATED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static JOINED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    /// [`WRITTEN`] held on the thread, added every 64 calls (as [`Timed`]).
    static WRITING: std::cell::Cell<(u64, u32)> = const { std::cell::Cell::new((0, 0)) };
}

/// A title thread wrote a list call down in `ticks`.
#[inline]
pub(crate) fn written_down(ticks: u64) {
    WRITING.with(|w| {
        let (t, n) = w.get();
        let (t, n) = (t + ticks, n + 1);
        if n >= 64 {
            WRITTEN[0].fetch_add(t, Relaxed);
            WRITTEN[1].fetch_add(u64::from(n), Relaxed);
            w.set((0, 0));
        } else {
            w.set((t, n));
        }
    });
}

/// A worker translated `calls` list calls in `ticks`.
pub(crate) fn translated(ticks: u64, calls: u64) {
    TRANSLATED[0].fetch_add(ticks, Relaxed);
    TRANSLATED[1].fetch_add(calls, Relaxed);
}

/// A submission waited `ticks` until its lists were translated.
pub(crate) fn joined(ticks: u64) {
    JOINED[0].fetch_add(ticks, Relaxed);
    JOINED[1].fetch_add(1, Relaxed);
}

/// `since` until now, as nanoseconds of `sum`.
pub(crate) fn add_since(sum: Sum, since: Instant) {
    add(sum, since.elapsed().as_nanos() as u64);
}

/// A thread waited for the card from `since` until now, because of `why`.
pub(crate) fn waited(why: Wait, since: Instant) {
    let ns = since.elapsed().as_nanos() as u64;
    if ns >= 20_000 {
        WAITS[why as usize][0].fetch_add(ns, Relaxed);
        WAITS[why as usize][1].fetch_add(1, Relaxed);
    }
}

// The card's timestamps.

/// Submissions timed at once at most (one slot each until collected).
const SLOTS: usize = 1024;
/// Timestamps for the pipelines of a sampled frame.
const SAMPLE_STAMPS: u32 = 32768;
/// Taken by a list 64 at a time.
const CHUNK: u32 = 64;
/// A frame in this many is sampled by pipeline.
const SAMPLE_EVERY: u64 = 30;

struct Slot {
    allocator: ID3D12CommandAllocator,
    lists: [ID3D12GraphicsCommandList; 2],
    /// The submission it times.
    owner: u64,
}

struct Gpu {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    slots: Vec<Slot>,
    free: Vec<usize>,
    used: Vec<usize>,
    /// The last stamp collected (the end of the latest submission timed).
    last_end: u64,
    /// Ticks since the last line: busy, idle.
    busy: u64,
    idle: u64,
    timed: u64,
    untimed: u64,
    /// For pipelines.
    sample_heap: ID3D12QueryHeap,
    sample_back: ID3D12Resource,
    next: u32,
    /// Lists that took stamps and were not collected (or dropped) yet.
    outstanding: u32,
    /// Closed, submitted lists' stamps: the submission, the stamps' slots,
    /// the pipeline of each span between two stamps.
    pending: Vec<(u64, Vec<u32>, Vec<String>)>,
    /// Ticks per pipeline since the last line, and frames sampled.
    pipelines: std::collections::HashMap<String, u64>,
    sampled: u64,
    frequency: u64,
}

// SAFETY: free-threaded D3D12 objects, used under the mutex.
unsafe impl Send for Gpu {}

static GPU: std::sync::LazyLock<Mutex<Option<Gpu>>> = std::sync::LazyLock::new(|| Mutex::new(make()));

fn gpu() -> std::sync::MutexGuard<'static, Option<Gpu>> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn make() -> Option<Gpu> {
    let host = crate::host::host()?;
    let device = &host.device;
    let heap = |count: u32| -> Option<ID3D12QueryHeap> {
        let mut heap: Option<ID3D12QueryHeap> = None;
        let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: count, NodeMask: 0 };
        // SAFETY: plain creation.
        unsafe { device.CreateQueryHeap(&desc, &mut heap) }.ok()?;
        heap
    };
    // SAFETY: a plain query of our queue.
    let frequency = unsafe { host.queue().GetTimestampFrequency() }.ok().filter(|&f| f > 0);
    let made = (|| {
        Some(Gpu {
            heap: heap(2 * SLOTS as u32)?,
            back: host.read_back_buffer(16 * SLOTS as u64)?,
            slots: Vec::new(),
            free: Vec::new(),
            used: Vec::new(),
            last_end: 0,
            busy: 0,
            idle: 0,
            timed: 0,
            untimed: 0,
            sample_heap: heap(SAMPLE_STAMPS)?,
            sample_back: host.read_back_buffer(8 * u64::from(SAMPLE_STAMPS))?,
            next: 0,
            outstanding: 0,
            pending: Vec::new(),
            pipelines: Default::default(),
            sampled: 0,
            frequency: frequency?,
        })
    })();
    if made.is_none() {
        tracing::warn!("no GPU timestamps for the frame-time line");
    }
    made
}

/// A slot for a submission: a free one, or a new one while there are fewer
/// than [`SLOTS`].
fn slot(g: &mut Gpu) -> Option<usize> {
    if let Some(i) = g.free.pop() {
        return Some(i);
    }
    if g.slots.len() >= SLOTS {
        return None;
    }
    let host = crate::host::host()?;
    // SAFETY: plain creation; the lists are closed at once.
    let made = unsafe {
        (|| -> windows::core::Result<Slot> {
            let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list = || -> windows::core::Result<ID3D12GraphicsCommandList> {
                let l: ID3D12GraphicsCommandList =
                    host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                l.Close()?;
                Ok(l)
            };
            Ok(Slot { lists: [list()?, list()?], allocator, owner: 0 })
        })()
    };
    g.slots.push(made.ok()?);
    Some(g.slots.len() - 1)
}

/// The two stamp lists for submission `number`: before its lists, after
/// them. `None` if it is not timed.
pub(crate) fn begin(number: u64) -> Option<[ID3D12CommandList; 2]> {
    let mut g = gpu();
    let g = g.as_mut()?;
    let Some(i) = slot(g) else {
        g.untimed += 1;
        return None;
    };
    let base = 2 * i as u32;
    let (heap, back) = (g.heap.clone(), g.back.clone());
    let s = &mut g.slots[i];
    // SAFETY: the slot's last submission finished (it was collected); its
    // own allocator and lists, recorded one after the other.
    let recorded = unsafe {
        (|| -> windows::core::Result<[ID3D12CommandList; 2]> {
            s.allocator.Reset()?;
            for (k, list) in s.lists.iter().enumerate() {
                list.Reset(&s.allocator, None)?;
                list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, base + k as u32);
                if k == 1 {
                    list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, base, 2, &back, 8 * u64::from(base));
                }
                list.Close()?;
            }
            Ok([s.lists[0].cast()?, s.lists[1].cast()?])
        })()
    };
    match recorded {
        Ok(lists) => {
            s.owner = number;
            g.used.push(i);
            g.timed += 1;
            Some(lists)
        }
        Err(_) => {
            g.free.push(i);
            g.untimed += 1;
            None
        }
    }
}

/// Reads `count` stamps from `first` on out of a read-back buffer.
fn read(back: &ID3D12Resource, first: u32, count: u32) -> Vec<u64> {
    let mut data = std::ptr::null_mut();
    let range = D3D12_RANGE { Begin: 8 * first as usize, End: 8 * (first + count) as usize };
    // SAFETY: our read-back buffer, read inside its size; the resolves into
    // this range have run.
    unsafe {
        if back.Map(0, Some(&range), Some(&mut data)).is_err() || data.is_null() {
            return vec![0; count as usize];
        }
        let words = data.cast::<u64>().add(first as usize);
        let out = (0..count as usize).map(|k| words.add(k).read_volatile()).collect();
        back.Unmap(0, Some(&D3D12_RANGE::default()));
        out
    }
}

/// Collects the stamps of submissions that have finished.
fn collect(g: &mut Gpu) {
    let finished = crate::host::submissions_finished();
    let mut spans = Vec::new();
    // One mapping for all of them.
    let all = if g.used.iter().any(|&slot| g.slots[slot].owner <= finished) {
        read(&g.back, 0, 2 * g.slots.len() as u32)
    } else {
        Vec::new()
    };
    let mut i = 0;
    while i < g.used.len() {
        let slot = g.used[i];
        if g.slots[slot].owner <= finished {
            spans.push((all[2 * slot], all[2 * slot + 1]));
            g.used.swap_remove(i);
            g.free.push(slot);
        } else {
            i += 1;
        }
    }
    spans.sort_unstable();
    for (a, c) in spans {
        if a == 0 || c < a {
            continue;
        }
        // Counted once where spans overlap (a stamp taken before the work
        // ahead of it has finished).
        let a = a.max(g.last_end);
        if g.last_end != 0 && a > g.last_end {
            g.idle += a - g.last_end;
        }
        g.busy += c.saturating_sub(a);
        g.last_end = g.last_end.max(c);
    }
    let mut k = 0;
    while k < g.pending.len() {
        if g.pending[k].0 > finished {
            k += 1;
            continue;
        }
        let (_, slots, names) = g.pending.swap_remove(k);
        g.outstanding = g.outstanding.saturating_sub(1);
        if let (Some(&lo), Some(&hi)) = (slots.iter().min(), slots.iter().max()) {
            let t = read(&g.sample_back, lo, hi - lo + 1);
            let at = |s: u32| t[(s - lo) as usize];
            for (n, name) in names.iter().enumerate() {
                let (a, b) = (at(slots[n]), at(slots[n + 1]));
                if a != 0 && b >= a {
                    *g.pipelines.entry(name.clone()).or_default() += b - a;
                }
            }
        }
    }
}

// Pipelines, per list.

/// A list's stamps in a sampled frame.
#[derive(Default)]
pub(crate) struct ListStamps {
    /// The slots taken, in order: [start, start + used) each.
    chunks: Vec<(u32, u32)>,
    /// The pipeline of the span each stamp but the last begins.
    names: Vec<String>,
    /// No more stamps (the last was written).
    done: bool,
    /// Closed and resolved: ready to go with a submission.
    closed: bool,
}

impl ListStamps {
    fn slots(&self) -> Vec<u32> {
        self.chunks.iter().flat_map(|&(s, n)| s..s + n).collect()
    }
}

/// Whether lists recorded now start taking stamps.
static SAMPLING: AtomicBool = AtomicBool::new(false);

/// One more stamp in `stamps`, into `list`; `last`: the list closes (no
/// pipeline begins). A chunk always keeps one slot spare for the last.
fn stamp(g: &mut Gpu, list: &ID3D12GraphicsCommandList, stamps: &mut ListStamps, name: Option<&str>) {
    if stamps.done {
        return;
    }
    let room = stamps.chunks.last().map_or(0, |c| CHUNK - c.1);
    let mut last = name.is_none();
    if room <= 1 && !last {
        if g.next + CHUNK <= SAMPLE_STAMPS {
            stamps.chunks.push((g.next, 0));
            g.next += CHUNK;
        } else if room == 1 {
            // No more slots: this one ends the list's last span.
            last = true;
        } else {
            stamps.done = true;
            return;
        }
    }
    let Some(chunk) = stamps.chunks.last_mut() else {
        stamps.done = true;
        return;
    };
    let slot = chunk.0 + chunk.1;
    chunk.1 += 1;
    // SAFETY: an open list and our query heap.
    unsafe { list.EndQuery(&g.sample_heap, D3D12_QUERY_TYPE_TIMESTAMP, slot) };
    match name {
        Some(name) if !last => stamps.names.push(name.to_owned()),
        _ => stamps.done = true,
    }
}

/// A list binds pipeline `name`: in a sampled frame, a stamp where its span
/// begins.
pub(crate) fn pipeline(list: &ID3D12GraphicsCommandList, stamps: &Mutex<Option<ListStamps>>, name: &str) {
    let mut s = stamps.lock().unwrap_or_else(|e| e.into_inner());
    if s.is_none() && !SAMPLING.load(Relaxed) {
        return;
    }
    let mut g = gpu();
    let Some(g) = g.as_mut() else { return };
    if s.is_none() {
        g.outstanding += 1;
    }
    let stamps = s.get_or_insert_with(ListStamps::default);
    stamp(g, list, stamps, Some(name));
}

/// The list closes: its last span ends, and its stamps are resolved.
pub(crate) fn close(list: &ID3D12GraphicsCommandList, stamps: &Mutex<Option<ListStamps>>) {
    let mut s = stamps.lock().unwrap_or_else(|e| e.into_inner());
    let Some(stamps) = s.as_mut() else { return };
    let mut g = gpu();
    let Some(g) = g.as_mut() else { return };
    stamp(g, list, stamps, None);
    for &(start, used) in &stamps.chunks {
        if used > 0 {
            // SAFETY: an open list; stamps written in it, our buffer.
            unsafe {
                list.ResolveQueryData(&g.sample_heap, D3D12_QUERY_TYPE_TIMESTAMP, start, used, &g.sample_back, 8 * u64::from(start))
            };
        }
    }
    stamps.closed = true;
}

/// The list is submitted as `number`: its stamps are read once that has run.
pub(crate) fn submitted(stamps: &Mutex<Option<ListStamps>>, number: u64) {
    let mut s = stamps.lock().unwrap_or_else(|e| e.into_inner());
    if !s.as_ref().is_some_and(|s| s.closed) {
        return;
    }
    let Some(taken) = s.take() else { return };
    let slots = taken.slots();
    let mut g = gpu();
    let Some(g) = g.as_mut() else { return };
    if slots.len() >= 2 && taken.names.len() + 1 == slots.len() {
        g.pending.push((number, slots, taken.names));
    } else {
        g.outstanding = g.outstanding.saturating_sub(1);
    }
}

/// The list is reset: stamps never submitted are dropped.
pub(crate) fn reset(stamps: &Mutex<Option<ListStamps>>) {
    if stamps.lock().unwrap_or_else(|e| e.into_inner()).take().is_some()
        && let Some(g) = gpu().as_mut()
    {
        g.outstanding = g.outstanding.saturating_sub(1);
    }
}

// The line.

struct Window {
    since: Instant,
    frames: u64,
    /// Frames presented since the start.
    count: u64,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// At every present: stamps collected, the sampled frame begun or ended,
/// and the line every five seconds.
pub(crate) fn frame() {
    let mut w = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    let w = w.get_or_insert_with(|| {
        // The rate of the recording ticks is measured from here on.
        crate::fast::start_clock();
        Window { since: Instant::now(), frames: 0, count: 0 }
    });
    w.frames += 1;
    w.count += 1;
    {
        let mut g = gpu();
        if let Some(g) = g.as_mut() {
            collect(g);
            if SAMPLING.swap(false, Relaxed) {
                // The sampled frame ended.
            } else if w.count.is_multiple_of(SAMPLE_EVERY) {
                if g.outstanding == 0 && g.pending.is_empty() {
                    g.next = 0;
                    g.sampled += 1;
                    SAMPLING.store(true, Relaxed);
                } else if w.count.is_multiple_of(10 * SAMPLE_EVERY) {
                    // Lists that took stamps and were never submitted nor
                    // reset: start over rather than stop sampling.
                    g.outstanding = 0;
                    g.pending.clear();
                }
            }
        }
    }
    let elapsed = w.since.elapsed();
    if elapsed < std::time::Duration::from_secs(5) {
        return;
    }
    let frames = w.frames.max(1) as f64;
    let ms_frame = elapsed.as_secs_f64() * 1000.0 / frames;
    (w.since, w.frames) = (Instant::now(), 0);
    let per = |ns: u64| ns as f64 / 1e6 / frames;
    let take = |a: &AtomicU64| a.swap(0, Relaxed);
    let mut waits = String::new();
    let mut waited = 0;
    for (k, name) in WAITS_NAMED.iter().enumerate() {
        let (ns, n) = (take(&WAITS[k][0]), take(&WAITS[k][1]));
        if n > 0 {
            waited += ns;
            waits.push_str(&format!("{}{name} {:.1}/{:.1}", if waits.is_empty() { "" } else { ", " }, per(ns), n as f64 / frames));
        }
    }
    let sums: Vec<u64> = SUMS.iter().map(take).collect();
    let mut phases = String::from("by part:");
    for (k, name) in PHASES_NAMED.iter().enumerate() {
        let ns = crate::fast::ticks_to_ns(take(&PHASE_SUMS[k]));
        phases.push_str(&format!(" {name} {:.1}", per(ns)));
    }
    // The recording by kind of call: ms and calls per frame; the kinds
    // past `RECORDING` are not in the recording sum.
    let mut calls = String::from("by call ms/n:");
    let (mut recording, mut besides) = (0, false);
    for (k, name) in CALLS_NAMED.iter().enumerate() {
        let (ticks, n) = (take(&CALL_SUMS[k][0]), take(&CALL_SUMS[k][1]));
        let ns = crate::fast::ticks_to_ns(ticks);
        if k < RECORDING {
            recording += ns;
        }
        if n > 0 {
            if k >= RECORDING && !besides {
                besides = true;
                calls.push_str(" | besides:");
            }
            calls.push_str(&format!(" {name} {:.1}/{:.0}", per(ns), n as f64 / frames));
        }
    }
    let mut dispatch = String::from("dispatch binding ms/n:");
    for (k, name) in PARTS_NAMED.iter().enumerate() {
        let (ticks, n) = (take(&PART_SUMS[k][0]), take(&PART_SUMS[k][1]));
        dispatch.push_str(&format!(" {name} {:.1}/{:.0}", per(crate::fast::ticks_to_ns(ticks)), n as f64 / frames));
    }
    #[cfg(windows)]
    let indirect = crate::indirect::frame_text(frames);
    #[cfg(not(windows))]
    let indirect = String::new();
    let (mut card, mut pipes) = (String::from("no GPU timestamps"), String::new());
    if let Some(g) = gpu().as_mut() {
        let ms = |ticks: u64| ticks as f64 * 1000.0 / g.frequency as f64 / frames;
        card = format!(
            "GPU busy {:.1} ms, idle {:.1}; {:.1} submissions/frame ({} untimed)",
            ms(g.busy),
            ms(g.idle),
            g.timed as f64 / frames,
            g.untimed
        );
        (g.busy, g.idle, g.timed, g.untimed) = (0, 0, 0, 0);
        let sampled = std::mem::take(&mut g.sampled).max(1) as f64;
        let mut top: Vec<(String, u64)> = g.pipelines.drain().collect();
        let all: u64 = top.iter().map(|t| t.1).sum();
        top.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        let in_frame = |ticks: u64| ticks as f64 * 1000.0 / g.frequency as f64 / sampled;
        pipes = format!("top pipelines (ms of a sampled frame, of {:.1} in all):", in_frame(all));
        for (name, ticks) in top.iter().take(5) {
            pipes.push_str(&format!(" {name} {:.1};", in_frame(*ticks)));
        }
    }
    let fence_set = sums[Sum::FenceSet as usize];
    let pair = |a: &[AtomicU64; 2]| (per(crate::fast::ticks_to_ns(take(&a[0]))), take(&a[1]) as f64 / frames);
    let ((written_ms, written_n), (translated_ms, translated_n), (joined_ms, joined_n)) =
        (pair(&WRITTEN), pair(&TRANSLATED), pair(&JOINED));
    #[cfg(windows)]
    let workers = crate::list::workers();
    #[cfg(not(windows))]
    let workers = 0;
    let lists = format!(
        "list calls written down by title threads {written_ms:.1} ms ({written_n:.0} calls, summed); translated by {workers} workers {translated_ms:.1} ms ({translated_n:.0} calls, summed); \
         submissions waited for translation {joined_ms:.1} ms ({joined_n:.1} of them)"
    );
    #[cfg(windows)]
    let ahead = crate::texture::ahead::line();
    #[cfg(not(windows))]
    let ahead = String::new();
    let cache = crate::memory::cache_text(frames);
    tracing::info!(
        "frame time: {ms_frame:.1} ms/frame over {frames:.0} frames: {card}; {lists}; translation into our lists {:.1} ms (summed; {calls}); \
         in submissions {:.1} ms ({:.1} of them; {phases}; textures brought up to date {:.1} ms; first uploads detiled by the copy-queue workers {:.1} ms, {ahead}), waiting for the card {:.1} ms [{waits}]; flights landed in {:.1} ms; title fence events set {:.1}/frame, {:.1} ms after the ask \
         on average; buffers named by lists: in place (read over the bus) {:.0} MB, pool by directory {:.0} MB, windows {:.0} MB (the fallback, \
         copies on the card; gathers {:.1}, {:.1} MB), not in place (filled at submission) {:.0} MB; directory blocks copied {:.1}; {cache}; {dispatch}; {indirect}; {pipes}",
        per(recording),
        per(sums[Sum::SubmitNs as usize]),
        sums[Sum::Submits as usize] as f64 / frames,
        per(sums[Sum::TextureNs as usize]),
        per(sums[Sum::AheadNs as usize]),
        per(waited),
        per(sums[Sum::LandNs as usize]),
        fence_set as f64 / frames,
        if fence_set > 0 { sums[Sum::FenceUs as usize] as f64 / 1000.0 / fence_set as f64 } else { 0.0 },
        sums[Sum::InPlaceBytes as usize] as f64 / frames / 1e6,
        sums[Sum::PoolBytes as usize] as f64 / frames / 1e6,
        sums[Sum::WindowBytes as usize] as f64 / frames / 1e6,
        sums[Sum::Gathers as usize] as f64 / frames,
        sums[Sum::GatherBytes as usize] as f64 / frames / 1e6,
        sums[Sum::CopyBytes as usize] as f64 / frames / 1e6,
        sums[Sum::DirectoryBlocks as usize] as f64 / frames,
    );
}
