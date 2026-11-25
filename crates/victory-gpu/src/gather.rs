//! Windows of title memory over scattered pages (`docs/unified-memory.md`,
//! "Windows over scattered pages"; `docs/kiss.md`). The title maps its
//! physical-page pool page by page in scattered order, so a 4 MiB window
//! of that memory is no run of one section and no buffer can be placed
//! over it (`crate::shadow`). Its home stays the pages, whose sections are
//! opened as heaps (`crate::unified`); the window is a buffer on the card
//! that GPU copies fill from them, one copy per run of pages:
//!
//! - **in**, before a list that names the window: the pages the title
//!   changed since the last copy in (all of them the first time). Changed
//!   means mapped anew (`XMemMapPhysicalPages`, page-mapping copies:
//!   [`remapped`]), announced written (DirectStorage completion, file
//!   reads, title copies on the CPU: [`written`]), or written by shaders
//!   through another view of the same pages (an in-place buffer, another
//!   window), noted at the submission that runs them ([`around`]);
//! - **out**, after a list whose shaders write the window: the ranges they
//!   write, copied into the pages, so that the title and every other view
//!   of them see it.
//!
//! The copies go to the one host queue in order around the list, between
//! barriers that order everything before them: queue order is the order.
//! A page copied in after a list wrote it through another view has what
//! that list wrote; a page copied out reaches the views copied in after.
//! Nothing is filled by the CPU, compared, or copied at every submission,
//! and the CPU never reads the window.
//!
//! Pages no unit holds that the title has not mapped, or shows by
//! read-only stand-ins (`xmem`'s filler page), read as zeros. A window
//! with a page of memory the title may write that lies in no unit stays a
//! buffer of its own, filled at each submission (`crate::shadow`).
//!
//! shortcut: writes the title's CPU makes into its pages without saying
//! so are not seen by the window until something announced changes them.
//!
//! Lock order: the index, then a window's pages, then the views' and the
//! units' locks (looking pages up); a window is never dropped holding the
//! index.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::shadow::Shadow;
use crate::unified::{CHUNK, Home};

/// The console's page, and a window's page.
const K64: u64 = 64 << 10;

/// Where a window's page comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Src {
    /// Not looked up since the title mapped there.
    Unknown,
    /// No page of a unit: reads as zeros.
    Zero,
    /// A unit's page at that offset of its heap.
    At(Home, u64),
}

struct Pages {
    src: Vec<Src>,
    /// Pages whose copy in the window is behind its source.
    behind: Vec<bool>,
    /// Nothing copied in yet: the buffer is zeros where pages are.
    fresh: bool,
}

/// A window's pages: where each comes from and whether the window's copy of
/// it is behind (`crate::shadow::Shadow::gather`).
pub(crate) struct Gather {
    pages: Mutex<Pages>,
    /// Some page is behind: the next list that names the window copies.
    behind: AtomicBool,
}

impl Gather {
    /// The pages of the window `[start, start + len)`, all behind and
    /// looked up again by the first copy in (the title may map there
    /// before the window is registered); `None` if no page lies in a unit,
    /// or one is memory the title may write that lies in none.
    pub(crate) fn new(start: u64, len: u64) -> Option<Gather> {
        let n = len.div_ceil(K64) as usize;
        let mut any = false;
        for i in 0..n as u64 {
            let page = start + i * K64;
            match crate::unified::source(page) {
                Some(_) => any = true,
                None if crate::unified::private_page(page) => return None,
                None => {}
            }
        }
        any.then(|| Gather {
            pages: Mutex::new(Pages { src: vec![Src::Unknown; n], behind: vec![true; n], fresh: true }),
            behind: AtomicBool::new(true),
        })
    }

    fn pages(&self) -> MutexGuard<'_, Pages> {
        self.pages.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The windows by their pages' sources, for the writes and remaps that
/// make their copies behind.
#[derive(Default)]
struct Index {
    /// By (unit, page offset): the windows showing that page (window
    /// number, page of the window).
    by_page: BTreeMap<(Home, u64), Vec<(u64, u32)>>,
    /// By window number.
    windows: BTreeMap<u64, Weak<Shadow>>,
    /// By (start, window number): end.
    by_address: BTreeMap<(u64, u64), u64>,
    /// The longest window's bytes.
    longest: u64,
}

static INDEX: Mutex<Index> =
    Mutex::new(Index { by_page: BTreeMap::new(), windows: BTreeMap::new(), by_address: BTreeMap::new(), longest: 0 });

fn index() -> MutexGuard<'static, Index> {
    INDEX.lock().unwrap_or_else(|e| e.into_inner())
}

/// Windows alive and their bytes; windows made since the start.
static LIVE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether any window over scattered pages is alive.
pub(crate) fn any() -> bool {
    LIVE[0].load(Ordering::Relaxed) > 0
}

/// For the unified line: windows alive, their bytes, windows made.
pub(crate) fn live() -> [u64; 3] {
    LIVE.each_ref().map(|n| n.load(Ordering::Relaxed))
}

fn unindex(by_page: &mut BTreeMap<(Home, u64), Vec<(u64, u32)>>, key: (Home, u64), id: u64, page: u32) {
    if let Some(users) = by_page.get_mut(&key) {
        users.retain(|&u| u != (id, page));
        if users.is_empty() {
            by_page.remove(&key);
        }
    }
}

/// A new window ([`Gather::new`]) is made; its pages are indexed when the
/// first copy in looks them up.
pub(crate) fn register(shadow: &Arc<Shadow>) {
    if shadow.gather().is_none() {
        return;
    }
    let (start, len) = shadow.span();
    let id = shadow.id();
    let mut ix = index();
    ix.windows.insert(id, Arc::downgrade(shadow));
    ix.by_address.insert((start, id), start + len);
    ix.longest = ix.longest.max(len);
    LIVE[0].fetch_add(1, Ordering::Relaxed);
    LIVE[1].fetch_add(len, Ordering::Relaxed);
    LIVE[2].fetch_add(1, Ordering::Relaxed);
}

/// Window `id` at `start` goes (its last holder let it go).
pub(crate) fn forget(id: u64, start: u64, gather: &Gather) {
    let mut ix = index();
    let ix = &mut *ix;
    ix.windows.remove(&id);
    let end = ix.by_address.remove(&(start, id));
    for (i, src) in gather.pages().src.iter().enumerate() {
        if let Src::At(home, offset) = *src {
            unindex(&mut ix.by_page, (home, offset), id, i as u32);
        }
    }
    LIVE[0].fetch_sub(1, Ordering::Relaxed);
    LIVE[1].fetch_sub(end.map_or(0, |e| e - start), Ordering::Relaxed);
}

/// The pages under `spans` (unit, offset, bytes) changed: the windows
/// showing them, but window `except`, copy them in again before the next
/// list that names them.
fn mark(spans: &[(Home, u64, u64)], except: Option<u64>) {
    if spans.is_empty() {
        return;
    }
    let mut hit: Vec<(Arc<Shadow>, u32)> = Vec::new();
    {
        let ix = index();
        for &(home, offset, len) in spans {
            let first = offset & !(K64 - 1);
            for (_, users) in ix.by_page.range((home, first)..(home, offset + len)) {
                for &(id, page) in users {
                    if Some(id) != except
                        && let Some(shadow) = ix.windows.get(&id).and_then(Weak::upgrade)
                    {
                        hit.push((shadow, page));
                    }
                }
            }
        }
    }
    for (shadow, page) in &hit {
        if let Some(gather) = shadow.gather() {
            if let Some(behind) = gather.pages().behind.get_mut(*page as usize) {
                *behind = true;
            }
            gather.behind.store(true, Ordering::Release);
        }
    }
    // (Dropped here, without the index: a last holder's drop takes it.)
    drop(hit);
}

/// Title memory `[at, at + len)` was written by a writer that says so
/// (`crate::unified::written`): windows over those pages, wherever they
/// show them, copy them in again.
pub(crate) fn written(at: u64, len: u64) {
    if !any() || len == 0 {
        return;
    }
    let mut spans = Vec::new();
    crate::unified::spans(at, len, &mut spans);
    mark(&spans, None);
}

/// The title mapped `[at, at + len)` anew (or the runtime told the view it
/// now shows there): windows over those addresses look their pages up
/// again and copy them in.
pub(crate) fn remapped(at: u64, len: u64) {
    if !any() || len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let mut held = Vec::new();
    {
        let mut ix = index();
        let ix = &mut *ix;
        let from = at.saturating_sub(ix.longest);
        let hits: Vec<(u64, u64, u64)> =
            ix.by_address.range((from, 0)..(end, 0)).filter(|(_, e)| **e > at).map(|(&(s, id), &e)| (s, id, e)).collect();
        for (start, id, stop) in hits {
            let Some(shadow) = ix.windows.get(&id).and_then(Weak::upgrade) else {
                continue;
            };
            if let Some(gather) = shadow.gather() {
                let mut pages = gather.pages();
                let first = ((at.max(start) - start) / K64) as usize;
                let last = ((end.min(stop) - start).div_ceil(K64) as usize).min(pages.src.len());
                for i in first..last {
                    if let Src::At(home, offset) = pages.src[i] {
                        unindex(&mut ix.by_page, (home, offset), id, i as u32);
                    }
                    pages.src[i] = Src::Unknown;
                    pages.behind[i] = true;
                }
                gather.behind.store(true, Ordering::Release);
            }
            held.push(shadow);
        }
    }
    drop(held);
}

/// A copy of window bytes, to record.
enum Op {
    /// From a unit's page(s) into the window: unit, its offset, the
    /// window's offset, bytes.
    In(ID3D12Resource, Home, u64, u64, u64),
    /// Zeros into the window: its offset, bytes.
    Zero(ID3D12Resource, u64, u64),
    /// From the window into a unit's page(s): unit, its offset, the
    /// window's offset, bytes.
    Out(ID3D12Resource, Home, u64, u64, u64),
}

/// Zeros the card reads for pages no unit holds ([`Src::Zero`]), and its
/// bytes.
const ZEROS: u64 = 1 << 20;

/// Appends `op` to `ops`, joined to the last one where it continues it in
/// the same buffers (not across a chunk of the unit's heap).
fn push(ops: &mut Vec<Op>, op: Op) {
    let same = |a: &ID3D12Resource, b: &ID3D12Resource| a.as_raw() == b.as_raw();
    match (ops.last_mut(), &op) {
        (Some(Op::In(w, h, from, to, n)), Op::In(w2, h2, from2, to2, n2))
        | (Some(Op::Out(w, h, from, to, n)), Op::Out(w2, h2, from2, to2, n2))
            if same(w, w2) && h == h2 && *from + *n == *from2 && *to + *n == *to2 && *from / CHUNK == (*from2 + *n2 - 1) / CHUNK =>
        {
            *n += *n2;
        }
        (Some(Op::Zero(w, to, n)), Op::Zero(w2, to2, n2)) if same(w, w2) && *to + *n == *to2 && *n + *n2 <= ZEROS => {
            *n += *n2;
        }
        _ => ops.push(op),
    }
}

/// The copies in of `shadow`'s pages that are behind, looked up first if
/// the title mapped them anew.
fn plan_in(shadow: &Arc<Shadow>, ops: &mut Vec<Op>) {
    let Some(gather) = shadow.gather() else {
        return;
    };
    if !gather.behind.swap(false, Ordering::AcqRel) {
        return;
    }
    let (start, _) = shadow.span();
    let id = shadow.id();
    let mut ix = index();
    let mut pages = gather.pages();
    let fresh = std::mem::replace(&mut pages.fresh, false);
    for i in 0..pages.src.len() {
        if !pages.behind[i] {
            continue;
        }
        pages.behind[i] = false;
        let page = start + i as u64 * K64;
        if pages.src[i] == Src::Unknown {
            pages.src[i] = match crate::unified::source(page) {
                Some((home, offset)) => {
                    ix.by_page.entry((home, offset)).or_default().push((id, i as u32));
                    Src::At(home, offset)
                }
                None => {
                    if crate::unified::private_page(page) {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            tracing::warn!(
                                page = format_args!("{page:#x}"),
                                "a window over scattered pages now shows memory no unit holds: it reads as zeros there"
                            )
                        });
                    }
                    Src::Zero
                }
            };
        }
        let to = i as u64 * K64;
        match pages.src[i] {
            Src::At(home, offset) => push(ops, Op::In(shadow.buffer.clone(), home, offset, to, K64)),
            Src::Zero if !fresh => push(ops, Op::Zero(shadow.buffer.clone(), to, K64)),
            _ => {}
        }
    }
}

/// The copies out of what shaders wrote at `[address, address + len)` of
/// `shadow` into its pages; `marks` gets the units' bytes written.
fn plan_out(shadow: &Arc<Shadow>, address: u64, len: u64, ops: &mut Vec<Op>, marks: &mut Vec<(Home, u64, u64)>) {
    let Some(gather) = shadow.gather() else {
        return;
    };
    let (start, bytes) = shadow.span();
    let (lo, hi) = (address.max(start), address.saturating_add(len).min(start + bytes));
    let pages = gather.pages();
    let mut at = lo;
    while at < hi {
        let i = ((at - start) / K64) as usize;
        let page = start + i as u64 * K64;
        let stop = (page + K64).min(hi);
        if let Some(&Src::At(home, offset)) = pages.src.get(i) {
            let (to, n) = (offset + (at - page), stop - at);
            push(ops, Op::Out(shadow.buffer.clone(), home, to, at - start, n));
            marks.push((home, to, n));
        }
        at = stop;
    }
}

/// At submission, for one list: the copies to run before it (pages of the
/// windows it names that are behind) and after it (what its shaders wrote
/// into windows, `writes`: window, address, bytes). The writes, and those
/// it makes in place (`in_place`: address, bytes), make the other windows
/// over the same pages behind, so that a later list's copy in has them.
pub(crate) fn around(
    named: &[Arc<Shadow>],
    writes: &[(Arc<Shadow>, u64, u64)],
    in_place: &[(u64, u64)],
) -> [Option<ID3D12CommandList>; 2] {
    let mut before = Vec::new();
    for shadow in named {
        plan_in(shadow, &mut before);
    }
    let mut after = Vec::new();
    for (shadow, address, len) in writes {
        let mut marks = Vec::new();
        plan_out(shadow, *address, *len, &mut after, &mut marks);
        mark(&marks, Some(shadow.id()));
    }
    if any() && !in_place.is_empty() {
        let mut spans = Vec::new();
        for &(address, len) in in_place {
            crate::unified::spans(address, len, &mut spans);
        }
        mark(&spans, None);
    }
    [record(&before), record(&after)]
}

/// A list of ours for the copies, with what it names held until it has run.
struct CopyList {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    list7: Option<ID3D12GraphicsCommandList7>,
    kept: Vec<ID3D12Resource>,
}

// SAFETY: free-threaded D3D12 objects; a list is recorded by one thread and
// handed back only once its submission has run.
unsafe impl Send for CopyList {}

static SPARE: Mutex<Vec<CopyList>> = Mutex::new(Vec::new());

/// A copy list in a submission: back to [`SPARE`] once that has run
/// (`crate::host::retire`).
struct Running(Option<CopyList>);

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(mut list) = self.0.take() {
            list.kept.clear();
            SPARE.lock().unwrap_or_else(|e| e.into_inner()).push(list);
        }
    }
}

fn copy_list() -> Option<CopyList> {
    if let Some(list) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return Some(list);
    }
    let host = crate::host::host()?;
    // SAFETY: plain object creation; the list is closed at once.
    let made = unsafe {
        (|| -> windows::core::Result<CopyList> {
            let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let list7 = crate::host::enhanced_barriers().then(|| list.cast().ok()).flatten();
            Ok(CopyList { allocator, list, list7, kept: Vec::new() })
        })()
    };
    match made {
        Ok(list) => Some(list),
        Err(e) => {
            tracing::warn!(error = %e, "no host list for the copies of windows over scattered pages");
            None
        }
    }
}

/// The buffer of zeros ([`ZEROS`] bytes).
fn zeros() -> Option<&'static ID3D12Resource> {
    struct Zeros(ID3D12Resource);
    // SAFETY: a free-threaded D3D12 resource, only read.
    unsafe impl Send for Zeros {}
    // SAFETY: as above.
    unsafe impl Sync for Zeros {}
    static ZERO: OnceLock<Option<Zeros>> = OnceLock::new();
    ZERO.get_or_init(|| {
        let (buffer, data) = crate::host::host()?.shared_buffer(ZEROS)?;
        // SAFETY: the buffer's mapping holds that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, ZEROS as usize) };
        Some(Zeros(buffer))
    })
    .as_ref()
    .map(|z| &z.0)
}

/// Records `ops` into a copy list of ours, between barriers that order
/// everything before and after them; `None` if there is nothing to copy.
fn record(ops: &[Op]) -> Option<ID3D12CommandList> {
    use crate::frametime::{Sum, add};
    if ops.is_empty() {
        return None;
    }
    let mut c = copy_list()?;
    // Buffers over units' chunks, asked once per list.
    let mut spans: HashMap<(Home, u64), Option<ID3D12Resource>> = HashMap::new();
    let mut span = |home: Home, offset: u64| -> Option<(ID3D12Resource, u64)> {
        let chunk = offset / CHUNK;
        let buffer = spans
            .entry((home, chunk))
            .or_insert_with(|| crate::unified::span_buffer(home, chunk * CHUNK).map(|(b, _)| b))
            .clone()?;
        Some((buffer, offset - chunk * CHUNK))
    };
    let mut kept = Vec::new();
    let mut lost = 0u64;
    // SAFETY: our list, not executing (it came back after its submission
    // ran); live buffers, each copy inside both.
    let recorded = unsafe {
        c.allocator.Reset().and_then(|()| c.list.Reset(&c.allocator, None)).map(|()| {
            crate::host::order_writes(&c.list, c.list7.as_ref());
            for op in ops {
                match op {
                    Op::In(window, home, from, to, n) => match span(*home, *from) {
                        Some((source, at)) => {
                            c.list.CopyBufferRegion(window, *to, &source, at, *n);
                            add(Sum::Gathers, 1);
                            add(Sum::GatherBytes, *n);
                            kept.push(source);
                        }
                        None => lost += 1,
                    },
                    Op::Zero(window, to, n) => match zeros() {
                        Some(z) => {
                            c.list.CopyBufferRegion(window, *to, z, 0, *n);
                            add(Sum::Gathers, 1);
                            add(Sum::GatherBytes, *n);
                        }
                        None => lost += 1,
                    },
                    Op::Out(window, home, to, from, n) => match span(*home, *to) {
                        Some((dest, at)) => {
                            c.list.CopyBufferRegion(&dest, at, window, *from, *n);
                            add(Sum::Scatters, 1);
                            add(Sum::ScatterBytes, *n);
                            kept.push(dest);
                        }
                        None => lost += 1,
                    },
                }
                let (Op::In(window, ..) | Op::Zero(window, ..) | Op::Out(window, ..)) = op;
                kept.push(window.clone());
            }
            crate::host::order_writes(&c.list, c.list7.as_ref());
        })
    }
    // SAFETY: our list, recorded above.
    .and_then(|()| unsafe { c.list.Close() });
    if let Err(e) = recorded {
        crate::host::fail(format_args!("the copies of windows over scattered pages did not record ({e})"));
    }
    if lost > 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(lost, "copies of windows over scattered pages left out: a unit could not be opened"));
    }
    c.kept = kept;
    let list = c.list.cast().ok();
    // Back for reuse once this submission has run.
    crate::host::retire(Running(Some(c)));
    list
}

/// The buffer of a window of `len` bytes: on the card while there is room
/// (under 85% of the budget), else in the PC's memory; whether it is a
/// pooled buffer of the PC's memory (given back when it goes).
pub(crate) fn window_buffer(len: u64) -> Option<(ID3D12Resource, bool)> {
    let host = crate::host::host()?;
    let room = crate::memcap::card_used(host).is_none_or(|(used, budget)| budget == 0 || used + len <= budget / 100 * 85);
    if room && let Some(buffer) = host.uav_buffer(len) {
        return Some((buffer, false));
    }
    host.shared_buffer(len).map(|(buffer, _)| (buffer, true))
}
