//! Unified memory (`docs/unified-memory.md`): the title's memory is the
//! GPU's memory. Title memory stays in the PC's memory where the title has
//! it, and the parts the GPU uses are opened as host heaps
//! (`OpenExistingHeapFromAddress1` for the regions `xmem` reserves,
//! `OpenExistingHeapFromFileMapping` for its physical-page sections). A
//! title buffer or window of title memory is then a buffer placed over the
//! title's own bytes (`crate::shadow`, "in place"): shaders read what the
//! title wrote and the title reads what shaders wrote, with no copy either
//! way and no write watching.
//!
//! Host textures still come from title memory by detiling, since the
//! layouts differ. They are detiled again only after a write the title
//! announces: DirectStorage completion, a file read, a page mapping, a
//! title copy, a shader write through an in-place buffer ([`written`],
//! [`gpu_wrote`]); a texture no announced write ever touched is one the
//! title fills with the CPU, and is checked by its content hash
//! (`texture.rs`).
//!
//! `VICTORY_UNIFIED`: unset or 1, on (the default; 0 is taken as 1: the
//! copies of title buffers it went back to are gone); 2, on and the probe
//! once at the first sweep (`unified_probe.rs`); 3, also the probe's
//! tile-mapping test.
//!
//! A region or a section is opened whole at its first GPU use, and let go
//! when the title frees it (`crate::release`). Every heap is checked once
//! when it is opened: vkd3d-proton gives memory of its own, and says
//! nothing, when an import fails. `xmem` leaves a page of its own after
//! each region and each section (the canary page), where the CPU writes a
//! pattern the GPU must read back through the heap.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;
use windows_sys::Win32::System::Memory as mem;

use crate::unified_map::Epochs;
use crate::unified_probe::{Rig, buffer_desc, hr, place as place_resource};

/// The placement unit and the console's page.
const K64: u64 = 64 << 10;
/// Title addresses lie below this; host memory lies above.
const TIB: u64 = 1 << 40;

/// `VICTORY_UNIFIED`, see the module comment.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_UNIFIED").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1) {
        0 => {
            tracing::warn!("VICTORY_UNIFIED=0 is gone (title buffers have no copies to go back to): unified memory stays on");
            1
        }
        level => level,
    })
}

/// Whether unified memory is on (`VICTORY_UNIFIED` is not 0).
pub fn on() -> bool {
    level() != 0
}

/// Whether a unit is opened.
enum State {
    Untried,
    Open(ID3D12Heap, u64),
    Refused,
}

/// A region or a section: the title's bytes (the canary page follows them)
/// and whether it is opened. A unit is opened whole, at the first GPU use of
/// an address in it, so a title buffer never lies across two heaps (opening
/// 256 MiB pieces instead left the title's 174 MB GPU heap across a piece
/// edge, a card copy, and pinned no less: run p7 opened 6,565 of 6,713 MB).
struct Unit {
    len: u64,
    state: State,
}

#[derive(Default)]
struct Units {
    /// `xmem` regions by first address.
    regions: BTreeMap<usize, Unit>,
    /// `xmem` sections by handle.
    sections: HashMap<usize, Unit>,
    /// Big buffers outside the units, opened by themselves ([`loose`]), by
    /// first address; and the first addresses that did not open.
    loose: BTreeMap<usize, Loose>,
    loose_refused: HashMap<u64, u64>,
    /// Buffers over [`CHUNK`]s of opened units' heaps, by unit and chunk
    /// number: what the GPU copies of windows over scattered pages read
    /// and write (`crate::gather`).
    chunks: HashMap<(Home, u64), ID3D12Resource>,
    /// The probe's queue, for the canaries.
    rig: Option<Rig>,
}

// SAFETY: the heaps and the queue are free-threaded D3D12 objects; all use
// is under the mutex.
unsafe impl Send for Units {}

static UNITS: std::sync::LazyLock<Mutex<Units>> = std::sync::LazyLock::new(Mutex::default);

fn units() -> std::sync::MutexGuard<'static, Units> {
    UNITS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Since the start: units opened and the bytes open now, refused, buffers
/// placed and their bytes, title copies done on the GPU, CPU copies that
/// waited for flights, textures checked by an announced write, by their
/// hash.
static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// For the opening line ([`tick`]): bytes open now in regions and in
/// sections; bytes committed for the import (pages the title had only
/// reserved); bytes let go because the title freed the memory; bytes open
/// now for big buffers outside the units ([`loose`]).
static OPENING: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

pub(crate) enum Stat {
    GpuCopy = 5,
    CpuCopyWaited = 6,
    TextureByEvent = 7,
    TextureByHash = 8,
}

/// Bytes of title memory opened as host heaps now: the system counts them
/// as PC memory mapped for the card, but they are the title's own memory,
/// not allocations of ours (`crate::memcap` leaves them out).
pub(crate) fn opened_bytes() -> u64 {
    STATS[1].load(Ordering::Relaxed)
}

/// Bytes of opened title memory let go because the title freed it.
pub(crate) fn released_bytes() -> u64 {
    OPENING[3].load(Ordering::Relaxed)
}

pub(crate) fn count(stat: Stat) {
    STATS[stat as usize].fetch_add(1, Ordering::Relaxed);
}

/// `xmem` reserved a region for the title at `at`, `len` bytes (a multiple
/// of 64 KiB), with a committed canary page of its own right after it.
pub fn title_region(at: usize, len: usize) {
    let old = units().regions.insert(at, Unit { len: len as u64, state: State::Untried });
    if let Some(old) = old {
        let_go(&old, 0);
    }
}

/// `xmem` made a section for physical pages, `len` bytes of the title's
/// and a canary page after them.
pub fn title_section(handle: usize, len: u64) {
    units().sections.insert(handle, Unit { len, state: State::Untried });
}

/// The section's handle is closed (its pages are all freed): it is let go;
/// its heap lives on in the buffers placed in it until those go, as for a
/// region the title frees ([`released`]).
pub fn title_section_gone(handle: usize) {
    let gone = {
        let mut u = units();
        u.chunks.retain(|(home, _), _| *home != Home::Section(handle));
        u.sections.remove(&handle)
    };
    if let Some(unit) = gone {
        let_go(&unit, 1);
    }
}

/// Counts an opened unit as let go (`kind`: 0 region, 1 section); dropping
/// it drops our reference to its heap.
fn let_go(unit: &Unit, kind: usize) {
    if let State::Open(..) = unit.state {
        STATS[1].fetch_sub(unit.len, Ordering::Relaxed);
        OPENING[kind].fetch_sub(unit.len, Ordering::Relaxed);
        OPENING[3].fetch_add(unit.len, Ordering::Relaxed);
    }
}

/// The title freed or decommitted `[at, at + len)` (`crate::release`): a
/// region that lies in it whole goes. Placed buffers hold their heap, so
/// the heap goes when the last of them does.
pub(crate) fn released(at: usize, len: usize) {
    if !on() || at as u64 >= TIB || len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let mut u = units();
    let gone: Vec<usize> = u
        .regions
        .range(at..end)
        .filter(|(base, unit)| **base + unit.len as usize <= end)
        .map(|(base, _)| *base)
        .collect();
    for base in gone {
        u.chunks.retain(|(home, _), _| *home != Home::Region(base));
        if let Some(unit) = u.regions.remove(&base) {
            let_go(&unit, 0);
        }
    }
    let gone: Vec<usize> = u.loose.range((at & !(K64 as usize - 1))..end).filter(|(lo, l)| **lo + l.len as usize <= end.next_multiple_of(K64 as usize)).map(|(lo, _)| *lo).collect();
    for lo in gone {
        if let Some(l) = u.loose.remove(&lo) {
            STATS[1].fetch_sub(l.len, Ordering::Relaxed);
            OPENING[4].fetch_sub(l.len, Ordering::Relaxed);
            OPENING[3].fetch_add(l.len, Ordering::Relaxed);
        }
    }
    u.loose_refused.retain(|&lo, _| lo < at as u64 || lo >= end as u64);
}

/// A buffer placed over title memory.
pub(crate) struct Placed {
    pub(crate) buffer: ID3D12Resource,
    /// The host GPU address of the title memory's first byte.
    pub(crate) gpu: u64,
    /// Bytes from the buffer's first byte to the title memory's first.
    pub(crate) bias: u64,
    /// In a section's view: what the title maps there may change.
    pub(crate) view: bool,
}

/// A buffer placed over title memory `[start, start + len)`, if the memory
/// is a region or one run of a section that opens (opened on first use).
pub(crate) fn place(start: u64, len: u64) -> Option<Placed> {
    if !on() || start >= TIB || len == 0 {
        return None;
    }
    let end = start.checked_add(len)?;
    let host = crate::host::host()?;
    let device = &host.device;
    let run = crate::views::section_run(start as usize, len as usize);
    let mut u = units();
    let u = &mut *u;
    let (heap, size, offset, view) = match run {
        Some((section, offset)) => {
            let unit = u.sections.get_mut(&section)?;
            if offset.checked_add(len)? > unit.len {
                return None;
            }
            let (heap, size) = open(device, &mut u.rig, unit, Home::Section(section))?;
            (heap, size, offset, true)
        }
        None => match u.regions.range_mut(..=start as usize).next_back() {
            Some((&base, unit)) if start < base as u64 + unit.len => {
                if end > base as u64 + unit.len {
                    return None;
                }
                let (heap, size) = open(device, &mut u.rig, unit, Home::Region(base))?;
                (heap, size, start - base as u64, false)
            }
            _ => loose(device, u, start, end)?,
        },
    };
    let aligned = offset & !(K64 - 1);
    let bias = offset - aligned;
    let width = (bias + len).next_multiple_of(K64).min(size - aligned);
    let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    let buffer = place_resource(device, &heap, aligned, &buffer_desc(width, flags), D3D12_RESOURCE_STATE_COMMON).ok()?;
    STATS[3].fetch_add(1, Ordering::Relaxed);
    STATS[4].fetch_add(len, Ordering::Relaxed);
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + bias;
    Some(Placed { buffer, gpu, bias, view })
}

/// A unit: a region by its first address, a section by its handle.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum Home {
    Region(usize),
    Section(usize),
}

/// The pieces the buffers over units' heaps for window copies cover
/// ([`span_buffer`]): a copy never crosses a chunk's end.
pub(crate) const CHUNK: u64 = 256 << 20;

/// Where the 64 KiB page of title memory at `page` lives, if in a unit: a
/// section a view shows (the whole page in one view), else a region.
pub(crate) fn source(page: u64) -> Option<(Home, u64)> {
    if !on() || page >= TIB {
        return None;
    }
    // A page of the GPU view of the pool: no host view shows it.
    let viewed = crate::memory::gpu_page(page)
        .map(|p| (p.section, p.offset))
        .or_else(|| crate::views::section_page(page as usize));
    if let Some((section, offset)) = viewed {
        let u = units();
        let unit = u.sections.get(&section)?;
        return (offset + K64 <= unit.len && !matches!(unit.state, State::Refused)).then_some((Home::Section(section), offset));
    }
    let u = units();
    let (&base, unit) = u.regions.range(..=page as usize).next_back()?;
    let offset = page - base as u64;
    (offset + K64 <= unit.len && !matches!(unit.state, State::Refused)).then_some((Home::Region(base), offset))
}

/// The units' bytes that title memory `[at, at + len)` is, as (unit,
/// offset, bytes): the parts views of sections show, and the parts in
/// regions. Memory in neither is left out.
pub(crate) fn spans(at: u64, len: u64, out: &mut Vec<(Home, u64, u64)>) {
    if !on() || len == 0 || at >= TIB {
        return;
    }
    let end = at.saturating_add(len);
    if crate::memory::gpu_view(at, len) {
        // The GPU view of the pool, page by page.
        let mut a = at;
        while a < end {
            let stop = ((a & !(K64 - 1)) + K64).min(end);
            if let Some(p) = crate::memory::gpu_page(a) {
                out.push((Home::Section(p.section), p.offset + (a & (K64 - 1)), stop - a));
            }
            a = stop;
        }
        return;
    }
    for (base, bytes, section, offset) in crate::views::section_spans(at as usize, end as usize) {
        let (lo, hi) = ((base as u64).max(at), (base as u64 + bytes as u64).min(end));
        if lo < hi {
            out.push((Home::Section(section), offset + (lo - base as u64), hi - lo));
        }
    }
    let u = units();
    let first = u.regions.range(..=at as usize).next_back().map(|(&b, _)| b).unwrap_or(at as usize);
    for (&base, unit) in u.regions.range(first..end as usize) {
        let (lo, hi) = ((base as u64).max(at), (base as u64 + unit.len).min(end));
        if lo < hi {
            out.push((Home::Region(base), lo - base as u64, hi - lo));
        }
    }
}

/// A buffer over the chunk of `home`'s heap that holds `offset` (the unit
/// opened now if it was not yet), and the offset of `offset` in it: for
/// the GPU copies between windows and their pages (`crate::gather`). Kept
/// until the unit goes.
pub(crate) fn span_buffer(home: Home, offset: u64) -> Option<(ID3D12Resource, u64)> {
    let chunk = offset / CHUNK;
    let host = crate::host::host()?;
    let mut u = units();
    if let Some(buffer) = u.chunks.get(&(home, chunk)) {
        return Some((buffer.clone(), offset - chunk * CHUNK));
    }
    let u = &mut *u;
    let unit = match home {
        Home::Region(base) => u.regions.get_mut(&base)?,
        Home::Section(handle) => u.sections.get_mut(&handle)?,
    };
    let (heap, size) = open(&host.device, &mut u.rig, unit, home)?;
    let first = chunk * CHUNK;
    if first >= size {
        return None;
    }
    let width = CHUNK.min(size - first);
    let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    let buffer = place_resource(&host.device, &heap, first, &buffer_desc(width, flags), D3D12_RESOURCE_STATE_COMMON).ok()?;
    u.chunks.insert((home, chunk), buffer.clone());
    Some((buffer, offset - first))
}

/// A buffer over the whole heap of section `handle` (opened now if it was
/// not yet), for the directory's copy of it (`crate::memory`). Kept until
/// the section goes.
pub(crate) fn section_buffer(handle: usize) -> Option<ID3D12Resource> {
    let host = crate::host::host()?;
    let mut u = units();
    let home = Home::Section(handle);
    if let Some(buffer) = u.chunks.get(&(home, u64::MAX)) {
        return Some(buffer.clone());
    }
    let u = &mut *u;
    let unit = u.sections.get_mut(&handle)?;
    let (heap, size) = open(&host.device, &mut u.rig, unit, home)?;
    let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    let buffer = place_resource(&host.device, &heap, 0, &buffer_desc(size, flags), D3D12_RESOURCE_STATE_COMMON).ok()?;
    // (Under a chunk number no window copy uses.)
    u.chunks.insert((home, u64::MAX), buffer.clone());
    Some(buffer)
}

/// Whether the 64 KiB page at `page` is committed memory the title may
/// write (a page with no unit that a window cannot take from one): not
/// for pages that are not there or read-only (the filler page's
/// stand-ins, `xmem`), which read as zeros in a window.
pub(crate) fn private_page(page: u64) -> bool {
    usable(page as usize, K64 as usize).is_ok()
}

/// The unit's heap and its bytes, opening it the first time.
fn open(device: &ID3D12Device, rig: &mut Option<Rig>, unit: &mut Unit, home: Home) -> Option<(ID3D12Heap, u64)> {
    if let State::Untried = unit.state {
        let started = std::time::Instant::now();
        // SAFETY: the unit's memory, as `xmem` made it, and its canary page.
        let opened = unsafe { open_unit(device, rig, unit.len, home) };
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let (what, at, kind) = match home {
            Home::Region(at) => ("region", at, 0),
            Home::Section(handle) => ("section", handle, 1),
        };
        unit.state = match opened {
            Ok(heap) => {
                STATS[0].fetch_add(1, Ordering::Relaxed);
                STATS[1].fetch_add(unit.len, Ordering::Relaxed);
                OPENING[kind].fetch_add(unit.len, Ordering::Relaxed);
                crate::memcap::count(crate::memcap::Use::Opened, unit.len);
                tracing::info!(what, at = format_args!("{at:#x}"), mb = unit.len >> 20, ms = format_args!("{ms:.0}"), "unified: title memory opened as a heap");
                State::Open(heap, unit.len + K64)
            }
            Err(why) => {
                STATS[2].fetch_add(1, Ordering::Relaxed);
                tracing::warn!(what, at = format_args!("{at:#x}"), mb = unit.len >> 20, why, "unified: title memory NOT opened; its buffers are buffers of their own, filled at each submission");
                State::Refused
            }
        };
    }
    match &unit.state {
        State::Open(heap, size) => Some((heap.clone(), *size)),
        _ => None,
    }
}

/// Title buffers that lie in no `xmem` unit (memory a committed resource
/// got from the title's allocator, like its 174 MB GPU heap at 0xb9c0000 of
/// run p7, and the buffers over views at 0xc34530000-0xc482e0000 of run
/// p10) are opened as a heap of their own, over the pages that hold them,
/// whatever their size: a committed buffer's memory is 64 KiB pages of its
/// own (`resource.rs` `footprint`), so no two of them share a page. Else
/// they would be a buffer of their own filled at each submission
/// (`crate::shadow`); until 2026-11-25 they were copies compared and merged
/// back after every submission that wrote them (run p10, at 16 MiB and up:
/// 18 buffers, 209 MB named a frame, ~200 ms of waits a frame).
/// A heap opened over the pages of one buffer outside the units.
struct Loose {
    len: u64,
    heap: ID3D12Heap,
    /// Over views of sections (the title's pages, mapped by `xmem`): let go
    /// when the title maps anything else there ([`remapped`]).
    view: bool,
}

/// The heap of a buffer outside the units that holds `[start, end)`: the
/// heap, its bytes, the offset of `start` in it, and whether it lies over
/// views; opened now if its pages are committed read-write ([`usable`])
/// and no other heap holds any of them. Over the 64 KiB pages that hold
/// the buffer, else over its 4 KiB pages (the next 64 KiB may not be
/// committed). A view page victory's guards keep read-only is opened first
/// (memory read in place needs no write tracking). Refused memory is asked
/// again two seconds later (its tail may be committed by then).
fn loose(device: &ID3D12Device, u: &mut Units, start: u64, end: u64) -> Option<(ID3D12Heap, u64, u64, bool)> {
    if let Some((&at, l)) = u.loose.range(..=start as usize).next_back()
        && end <= at as u64 + l.len
    {
        return Some((l.heap.clone(), l.len, start - at as u64, l.view));
    }
    let now = crate::host::now_ms();
    if u.loose_refused.get(&start).is_some_and(|&until| now < until) {
        return None;
    }
    let wide = (start & !(K64 - 1), end.next_multiple_of(K64));
    let narrow = (start & !0xfff, end.next_multiple_of(0x1000));
    // Not over another one, nor over a region (a page is opened once).
    let (lo, hi) = narrow;
    if u.loose.range(..hi as usize).next_back().is_some_and(|(&at, l)| at as u64 + l.len > lo)
        || u.regions.range(..hi as usize).next_back().is_some_and(|(&at, r)| at as u64 + r.len + K64 > lo)
    {
        let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 32 || n.is_power_of_two() {
            tracing::warn!(
                what = "buffer",
                at = format_args!("{start:#x}"),
                kb = (end - start) >> 10,
                n,
                "unified: title memory NOT opened: its pages lie partly in another heap opened over a buffer or region; a buffer of its own"
            );
        }
        return None;
    }
    let usable_wide = usable(wide.0 as usize, (wide.1 - wide.0) as usize)
        .ok()
        .filter(|_| {
            !u.loose.range(..wide.1 as usize).next_back().is_some_and(|(&at, l)| at as u64 + l.len > wide.0)
                && !u.regions.range(..wide.1 as usize).next_back().is_some_and(|(&at, r)| at as u64 + r.len + K64 > wide.0)
        });
    let found = match usable_wide {
        Some(view) => Ok((wide, view)),
        None => match usable(lo as usize, (hi - lo) as usize) {
            Ok(view) => Ok((narrow, view)),
            Err(Refusal::ReadOnly(_)) => {
                // Guarded by victory (a view a copy watches): opened, then
                // asked again.
                crate::watch::writing(lo as usize, (hi - lo) as usize);
                usable(lo as usize, (hi - lo) as usize).map(|view| (narrow, view))
            }
            Err(why) => Err(why),
        },
    };
    let started = std::time::Instant::now();
    let opened = found.map_err(|why| why.to_string()).and_then(|((lo, hi), view)| {
        // SAFETY: the pages are committed and readable and writable.
        unsafe {
            touch(lo as *mut u8, (hi - lo) as usize);
            device
                .cast::<ID3D12Device13>()
                .and_then(|d| d.OpenExistingHeapFromAddress1::<ID3D12Heap>(lo as *const _, (hi - lo) as usize))
                .map_err(|e| format!("OpenExistingHeapFromAddress1: {e}"))
                .and_then(|heap| match check(device, &mut u.rig, &heap, lo as *const u8, hi - lo, start - lo) {
                    Ok(true) => Ok((heap, lo, hi, view)),
                    Ok(false) => Err("nothing in it to check it by yet".to_owned()),
                    Err(why) => Err(why),
                })
        }
    });
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    match opened {
        Ok((heap, lo, hi, view)) => {
            let len = hi - lo;
            u.loose_refused.remove(&start);
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(len, Ordering::Relaxed);
            OPENING[4].fetch_add(len, Ordering::Relaxed);
            crate::memcap::count(crate::memcap::Use::Opened, len);
            tracing::info!(what = "buffer", at = format_args!("{lo:#x}"), mb = len >> 20, view, ms = format_args!("{ms:.0}"), "unified: title memory opened as a heap");
            u.loose.insert(lo as usize, Loose { len, heap: heap.clone(), view });
            Some((heap, len, start - lo, view))
        }
        Err(why) => {
            let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 32 || n.is_power_of_two() {
                tracing::warn!(what = "buffer", at = format_args!("{start:#x}"), kb = (end - start) >> 10, why, n, "unified: title memory NOT opened; a buffer of its own for now (asked again in 2 s)");
            }
            u.loose_refused.insert(start, now + 2_000);
            None
        }
    }
}

/// Why memory cannot be opened ([`usable`]): the first page that says so.
enum Refusal {
    /// Committed, but read-only (victory's guards make view pages so).
    ReadOnly(String),
    Other(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::ReadOnly(s) | Refusal::Other(s) => f.write_str(s),
        }
    }
}

/// Whether every page of `[at, at + len)` is committed and both readable and
/// writable, whatever the caching modifiers (write-combined, uncached: the
/// title's graphics memory may ask for them); `Ok(true)` if some of it is a
/// view of a section. Else the first pages that are not, with their state,
/// protection and type.
fn usable(at: usize, len: usize) -> Result<bool, Refusal> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    const READ_WRITE: [u32; 2] = [0x04, 0x40]; // PAGE_READWRITE, PAGE_EXECUTE_READWRITE
    const READ_ONLY: [u32; 2] = [0x02, 0x20]; // PAGE_READONLY, PAGE_EXECUTE_READ
    const GUARD: u32 = 0x100;
    let end = at + len;
    let mut p = at;
    let mut view = false;
    while p < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        if unsafe { VirtualQuery(p as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return Err(Refusal::Other(format!("no answer for {p:#x}")));
        }
        let stop = (info.BaseAddress as usize + info.RegionSize).max(p + 4096);
        let base = info.Protect & 0xff;
        let say = || {
            format!(
                "pages {p:#x}..{:#x} ({} KiB into the buffer's pages): state {:#x}, protection {:#x}, type {:#x}",
                stop.min(end),
                (p - at) >> 10,
                info.State,
                info.Protect,
                info.Type
            )
        };
        if info.State != MEM_COMMIT || info.Protect & GUARD != 0 {
            return Err(Refusal::Other(say()));
        }
        if READ_ONLY.contains(&base) {
            return Err(Refusal::ReadOnly(say()));
        }
        if !READ_WRITE.contains(&base) {
            return Err(Refusal::Other(say()));
        }
        view |= info.Type == MEM_MAPPED;
        p = stop;
    }
    Ok(view)
}

/// The title maps something else at `[at, at + len)` (`crate::watch`): a
/// big buffer's heap over views there shows the old pages, so it goes (its
/// in-place buffers are made stale by `crate::shadow`, and the next ask
/// opens the memory anew).
pub(crate) fn remapped(at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let mut u = units();
    if u.loose.is_empty() {
        return;
    }
    let gone: Vec<usize> = u
        .loose
        .range(..end)
        .rev()
        .take_while(|(lo, l)| **lo + l.len as usize > at)
        .filter(|(_, l)| l.view)
        .map(|(lo, _)| *lo)
        .collect();
    for lo in gone {
        if let Some(l) = u.loose.remove(&lo) {
            STATS[1].fetch_sub(l.len, Ordering::Relaxed);
            OPENING[4].fetch_sub(l.len, Ordering::Relaxed);
        }
    }
}

/// Writes every page of `[at, at + len)` once in place (an atomic add of 0,
/// which changes no byte whatever another thread does), so that pages a
/// write watch keeps read-only become writable and can be pinned.
unsafe fn touch(at: *mut u8, len: usize) {
    for page in (at as usize..at as usize + len).step_by(4096) {
        // SAFETY: committed read-write memory of the title's.
        unsafe { (*(page as *const std::sync::atomic::AtomicU8)).fetch_add(0, Ordering::Relaxed) };
    }
}

/// Whether `heap` is the memory at `cpu` (`len` bytes): vkd3d-proton gives
/// memory of its own, and says nothing, when an import fails. The GPU reads
/// a word the CPU sees as not zero (in the ask's first 64 KiB, then at 64
/// places over the piece) and must find the same; a word the title changes
/// meanwhile is passed over. If every word looked at is zero, a pattern is
/// written into a zero word and taken out again at once (a canary in the
/// title's memory for a millisecond), and the GPU must read it.
/// `Ok(false)`: nothing could be told.
unsafe fn check(device: &ID3D12Device, rig: &mut Option<Rig>, heap: &ID3D12Heap, cpu: *const u8, len: u64, hint: u64) -> Result<bool, String> {
    use std::sync::atomic::AtomicU64 as Word;
    if rig.is_none() {
        *rig = Some(Rig::new(device).map_err(|e| format!("no queue for the canary: {e}"))?);
    }
    let rig = rig.as_mut().ok_or("no queue")?;
    let gpu_word = |rig: &mut Rig, at: u64| -> Result<u64, String> {
        let page = at & !(K64 - 1);
        let desc = buffer_desc(K64.min(len - page), D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
        let probe = place_resource(device, heap, page, &desc, D3D12_RESOURCE_STATE_COMMON);
        let probe = probe.map_err(|e| format!("no buffer to check with: {}", hr::<()>(&Err(e))))?;
        let got = rig.fetch(&probe, at - page, 8).map_err(|e| format!("canary not read: {e}"))?;
        Ok(u64::from_le_bytes(got.get(..8).and_then(|b| b.try_into().ok()).unwrap_or([0; 8])))
    };
    // SAFETY: an aligned word of the piece's committed memory.
    let word = |at: u64| unsafe { &*(cpu.add(at as usize) as *const Word) };
    let first = hint & !7;
    let places = (first..(first + K64).min(len & !7)).step_by(8).chain((0..64u64).map(|k| (len / 64 * k) & !7));
    let mut tries = 0;
    for at in places {
        let v = word(at).load(Ordering::SeqCst);
        if v == 0 {
            continue;
        }
        let got = gpu_word(rig, at)?;
        if got == v {
            return Ok(true);
        }
        if word(at).load(Ordering::SeqCst) == v {
            return Err("the heap is not the title's memory (the driver's silent fallback): a word differs".to_owned());
        }
        tries += 1;
        if tries >= 8 {
            return Ok(false);
        }
    }
    // All zero where looked: the last word of the piece, or the first if
    // the ask lies there.
    let at = if hint + K64 >= len { 0 } else { (len - 8) & !7 };
    let canary = u64::from_le_bytes(pattern()[..8].try_into().unwrap_or([0xa5; 8])) | 1;
    if word(at).compare_exchange(0, canary, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        return Ok(false);
    }
    let got = gpu_word(rig, at);
    let _ = word(at).compare_exchange(canary, 0, Ordering::SeqCst, Ordering::SeqCst);
    if got? == canary {
        Ok(true)
    } else {
        Err("the heap is not the title's memory (the driver's silent fallback): canary differs".to_owned())
    }
}


/// A canary pattern no memory holds by chance.
fn pattern() -> [u8; 16] {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut out = [0u8; 16];
    for (i, b) in out.iter_mut().enumerate() {
        *b = (t.rotate_left(i as u32 * 5) as u8) ^ (0xa5u8.wrapping_add(i as u8 * 37));
    }
    out
}

/// Opens a region or a section as a heap and checks it with its canary
/// page.
unsafe fn open_unit(device: &ID3D12Device, rig: &mut Option<Rig>, len: u64, home: Home) -> Result<ID3D12Heap, String> {
    let total = len + K64;
    let canary = pattern();
    // SAFETY (the whole function): the unit's memory as `xmem` made it.
    unsafe {
        let (heap, view) = match home {
            Home::Region(base) => {
                ready_region(base, total as usize)?;
                let heap = device
                    .cast::<ID3D12Device13>()
                    .and_then(|d| d.OpenExistingHeapFromAddress1::<ID3D12Heap>(base as *const _, total as usize))
                    .map_err(|e| format!("OpenExistingHeapFromAddress1: {e}"))?;
                std::ptr::copy_nonoverlapping(canary.as_ptr(), (base + len as usize) as *mut u8, 16);
                (heap, None)
            }
            Home::Section(handle) => {
                let heap = device
                    .cast::<ID3D12Device3>()
                    .and_then(|d| d.OpenExistingHeapFromFileMapping::<ID3D12Heap>(windows::Win32::Foundation::HANDLE(handle as *mut _)))
                    .map_err(|e| format!("OpenExistingHeapFromFileMapping: {e}"))?;
                let view = mem::MapViewOfFile(handle as _, mem::FILE_MAP_WRITE, (len >> 32) as u32, len as u32, K64 as usize);
                if view.Value.is_null() {
                    return Err("no view of the canary page".to_owned());
                }
                crate::mapstat::mapped(crate::mapstat::Site::Canary, K64 as usize);
                std::ptr::copy_nonoverlapping(canary.as_ptr(), view.Value.cast::<u8>(), 16);
                (heap, Some(view))
            }
        };
        let check = (|| {
            if rig.is_none() {
                *rig = Some(Rig::new(device).map_err(|e| format!("no queue for the canary: {e}"))?);
            }
            let rig = rig.as_mut().ok_or("no queue")?;
            let flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
            let probe = place_resource(device, &heap, len, &buffer_desc(K64, flags), D3D12_RESOURCE_STATE_COMMON);
            let probe = probe.map_err(|e| format!("no buffer on the canary page: {}", hr::<()>(&Err(e))))?;
            let got = rig.fetch(&probe, 0, 16).map_err(|e| format!("canary not read: {e}"))?;
            if got[..] == canary[..] {
                Ok(())
            } else {
                Err("the heap is not the title's memory (the driver's silent fallback): canary differs".to_owned())
            }
        })();
        if let Some(view) = view {
            mem::UnmapViewOfFile(view);
            crate::mapstat::unmapped(crate::mapstat::Site::Canary, K64 as usize);
        }
        check.map(|()| heap)
    }
}

/// Makes every page of a region committed and writable for the import:
/// reserved pages are committed (counted: run p7 found none), and every
/// page is written once in place (an atomic add of 0, which changes no byte
/// whatever another thread does) so that pages a write watch keeps
/// read-only become writable and can be pinned. Refuses pages the title
/// protected otherwise.
unsafe fn ready_region(base: usize, total: usize) -> Result<(), String> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc, VirtualQuery};
    let end = base + total;
    let mut at = base;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return Err(format!("no answer for {at:#x}"));
        }
        let stop = (info.BaseAddress as usize + info.RegionSize).min(end);
        match info.State {
            // Read-write, whatever the caching modifiers.
            MEM_COMMIT if matches!(info.Protect & 0x1ff, 0x04 | 0x40) => {}
            // SAFETY: reserved pages of the region.
            MEM_RESERVE if !unsafe { VirtualAlloc(at as *const _, stop - at, MEM_COMMIT, PAGE_READWRITE) }.is_null() => {
                OPENING[2].fetch_add((stop - at) as u64, Ordering::Relaxed);
            }
            state => return Err(format!("pages at {at:#x} are state {state:#x}, protection {:#x}", info.Protect)),
        }
        for page in (at..stop).step_by(4096) {
            // SAFETY: committed read-write memory of the title's.
            unsafe { (*(page as *const std::sync::atomic::AtomicU8)).fetch_add(0, Ordering::Relaxed) };
        }
        at = stop;
    }
    Ok(())
}

// Announced writes, for the host textures (`texture.rs`).

static EPOCHS: Mutex<Epochs> = Mutex::new(Epochs::new());

fn epochs() -> std::sync::MutexGuard<'static, Epochs> {
    EPOCHS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Title memory `[at, at + len)` was written by a writer that says so
/// (DirectStorage, a file read, a page mapping, a title copy done on the
/// CPU, a shader write once its submission finished).
pub(crate) fn written(at: usize, len: usize) {
    if on() && len > 0 {
        epochs().write(at as u64, len as u64);
        // Windows over those pages take them again (`crate::gather`).
        crate::gather::written(at as u64, len as u64);
    }
}

/// The newest announced write over `[at, at + len)`: 0 if none ever was
/// (memory only the title's CPU writes).
pub(crate) fn newest(at: usize, len: usize) -> u64 {
    settle();
    epochs().newest(at as u64, len as u64)
}

/// Shader writes through in-place buffers, per submission (in order and
/// joined), until it has finished.
#[allow(clippy::type_complexity)]
static GPU_WRITES: Mutex<Vec<(Vec<(usize, usize)>, u64)>> = Mutex::new(Vec::new());

/// At submission: submission `number`'s shaders write `ranges` (in order,
/// joined) in place. Announced once it has finished ([`settle`]).
pub(crate) fn gpu_wrote(ranges: &[(usize, usize)], number: u64) {
    if !ranges.is_empty() {
        GPU_WRITES.lock().unwrap_or_else(|e| e.into_inner()).push((ranges.to_vec(), number));
    }
}

/// Announces the shader writes of the submissions that have finished: the
/// textures over them are stale.
pub(crate) fn settle() {
    let finished = crate::host::submissions_finished();
    let done: Vec<Vec<(usize, usize)>> = {
        let mut pending = GPU_WRITES.lock().unwrap_or_else(|e| e.into_inner());
        let landed = pending.iter().take_while(|(_, number)| *number <= finished).count();
        if landed == 0 {
            return;
        }
        pending.drain(..landed).map(|(ranges, _)| ranges).collect()
    };
    let mut joined = Vec::new();
    for ranges in &done {
        joined.extend_from_slice(ranges);
    }
    joined.sort_unstable();
    let mut epochs = epochs();
    let mut last: Option<(usize, usize)> = None;
    for (from, to) in joined {
        match &mut last {
            Some(l) if from <= l.1 => l.1 = l.1.max(to),
            _ => {
                if let Some((a, b)) = last.replace((from, to)) {
                    epochs.write(a as u64, (b - a) as u64);
                }
            }
        }
    }
    if let Some((a, b)) = last {
        epochs.write(a as u64, (b - a) as u64);
    }
}

/// Whether `[at, at + len)` lies in title memory that may be opened (a
/// region or a section's view): the GPU reads it in place, so a CPU write
/// there must not overtake submissions still running.
pub(crate) fn in_unit(at: usize, len: usize) -> bool {
    if !on() || at as u64 >= TIB {
        return false;
    }
    if crate::views::section_run(at, len).is_some() {
        return true;
    }
    let u = units();
    u.regions.range(..=at).next_back().is_some_and(|(&base, unit)| (at + len) as u64 <= base as u64 + unit.len)
}

/// At every sweep: the probe if asked for, and the unified line every ten
/// seconds.
pub(crate) fn tick() {
    crate::unified_probe::tick();
    crate::tiles::tick();
    crate::resource::targets_tick();
    crate::release::tick();
    if !on() {
        return;
    }
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 10_000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let (live, live_bytes) = crate::shadow::in_place_live();
    let own = crate::shadow::own_made();
    let ranges = epochs().len();
    let windows = crate::gather::live();
    let d = crate::memory::stats();
    tracing::info!(
        "unified: {} heaps opened ({} MB of title memory), {} refused; buffers in place: {} alive ({} MB), {} placed since the start ({} MB); \
         by directory: pages of the pool's GPU view mapped {} ({} MB), entries written {}, directory blocks copied to the card {}, \
         sections opened as copies {}, refused {}; \
         windows over scattered pages (fallback: copies on the card gathered from their pages) {} alive ({} MB), {} made; \
         not in place (buffers of their own, filled at each submission that names them) {} made ({} MB), lists whose shaders wrote one {}; \
         windows let go {}; title copies on the GPU {}, on the CPU after waiting for the card {}; textures checked for an announced write {}, \
         by their hash {}; announced-write ranges kept {}",
        s[0],
        s[1] >> 20,
        s[2],
        live,
        live_bytes >> 20,
        s[3],
        s[4] >> 20,
        d[0],
        d[0] >> 4,
        d[1],
        d[2],
        d[3],
        d[4],
        windows[0],
        windows[1] >> 20,
        windows[2],
        own[0],
        own[1] >> 20,
        own[2],
        own[3],
        s[5],
        s[6],
        s[7],
        s[8],
        ranges
    );
    let o: Vec<u64> = OPENING.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    tracing::info!(
        "unified opening: title memory open now {} MB (regions {} MB, sections {} MB, buffers outside them {} MB); pages \
         committed for the import {} MB; let go because the title freed the memory {} MB",
        s[1] >> 20,
        o[0] >> 20,
        o[1] >> 20,
        o[4] >> 20,
        o[2] >> 20,
        o[3] >> 20,
    );
}
