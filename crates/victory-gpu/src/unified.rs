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
//! `VICTORY_UNIFIED`: unset or 1, on (the default); 0, off: the copies and
//! write watching of before, the way back for comparing; 2, on and the
//! probe once at the first sweep (`unified_probe.rs`); 3, also the probe's
//! tile-mapping test.
//!
//! Title memory is opened piece by piece at its first GPU use (`SLICE`),
//! never committing a page for it, and let go when the title frees it
//! (`crate::release`). Every piece is checked once when it is opened:
//! vkd3d-proton gives memory of its own, and says nothing, when an import
//! fails; the GPU must read through the heap what the CPU sees there
//! (`check`).

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
    *LEVEL.get_or_init(|| std::env::var("VICTORY_UNIFIED").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1))
}

/// Whether unified memory is on (`VICTORY_UNIFIED` is not 0).
pub fn on() -> bool {
    level() != 0
}

/// Title memory is opened piece by piece, at the first GPU use in each
/// piece, never a whole region or section at once: a piece is at most the
/// slice of this grid around that use (absolute addresses for a region,
/// offsets for a section), stretched only to hold the use itself. Opening
/// pins the piece (resident for good), so memory the GPU never reads in
/// place (render targets and textures the title keeps there, regions
/// reserved for later) is never pinned.
const SLICE: u64 = 256 << 20;

/// An opened piece of a unit.
struct Piece {
    len: u64,
    heap: ID3D12Heap,
}

/// A region or a section: the title's bytes (a canary page of ours follows
/// them) and the pieces of it opened so far.
struct Unit {
    len: u64,
    /// Opened pieces by offset in the unit; they never overlap.
    pieces: BTreeMap<u64, Piece>,
    /// Ranges that did not open, by offset: end, and the moment (ms) from
    /// which to try again (`u64::MAX`: never).
    refused: BTreeMap<u64, (u64, u64)>,
    /// A region looked over once for the opening line ([`survey`]).
    surveyed: bool,
}

impl Unit {
    fn new(len: u64) -> Unit {
        Unit { len, pieces: BTreeMap::new(), refused: BTreeMap::new(), surveyed: false }
    }

    fn opened(&self) -> u64 {
        self.pieces.values().map(|p| p.len).sum()
    }
}

#[derive(Default)]
struct Units {
    /// `xmem` regions by first address.
    regions: BTreeMap<usize, Unit>,
    /// `xmem` sections by handle.
    sections: HashMap<usize, Unit>,
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

/// Since the start: pieces opened and the bytes open now, refused, buffers
/// placed and their bytes, title copies done on the GPU, CPU copies that
/// waited for flights, textures checked by an announced write, by their
/// hash.
static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// For the opening line ([`tick`]): bytes open now in regions and in
/// sections; bytes of the units touched (a piece opened in them); of the
/// regions touched, bytes committed and bytes only reserved when first
/// looked over; asks that ran over a piece's edge (kept as copies); asks
/// over pages that are not committed (copies); pieces put off because no
/// word of them could tell the title's memory from other memory; bytes let
/// go because the title freed the memory.
static OPENING: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

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
    OPENING[8].load(Ordering::Relaxed)
}

pub(crate) fn count(stat: Stat) {
    STATS[stat as usize].fetch_add(1, Ordering::Relaxed);
}

/// `xmem` reserved a region for the title at `at`, `len` bytes (a multiple
/// of 64 KiB), with a committed canary page of its own right after it.
pub fn title_region(at: usize, len: usize) {
    let old = units().regions.insert(at, Unit::new(len as u64));
    if let Some(old) = old {
        let_go(&old, true);
    }
}

/// `xmem` made a section for physical pages, `len` bytes of the title's
/// and a canary page after them.
pub fn title_section(handle: usize, len: u64) {
    units().sections.insert(handle, Unit::new(len));
}

/// The section's handle is closed (its pages are all freed): its pieces
/// are let go; the heaps live on in the buffers placed in them until those
/// go, as for a region the title frees ([`released`]).
pub fn title_section_gone(handle: usize) {
    let gone = units().sections.remove(&handle);
    if let Some(unit) = gone {
        let_go(&unit, false);
    }
}

/// Counts a unit's pieces as let go (dropping it drops our heaps).
fn let_go(unit: &Unit, region: bool) {
    let bytes = unit.opened();
    if bytes > 0 {
        STATS[1].fetch_sub(bytes, Ordering::Relaxed);
        OPENING[if region { 0 } else { 1 }].fetch_sub(bytes, Ordering::Relaxed);
        OPENING[8].fetch_add(bytes, Ordering::Relaxed);
    }
}

/// The title freed or decommitted `[at, at + len)` (`crate::release`): the
/// region it lies in goes if all of it is freed, else the pieces that lie
/// inside it whole. Placed buffers hold their heap, so the heap goes when
/// the last of them does.
pub(crate) fn released(at: usize, len: usize) {
    if !on() || at as u64 >= TIB || len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let mut u = units();
    let bases: Vec<usize> = u
        .regions
        .range(..end)
        .rev()
        .take_while(|(base, unit)| **base + unit.len as usize > at)
        .map(|(base, _)| *base)
        .collect();
    for base in bases {
        let Some(unit) = u.regions.get_mut(&base) else { continue };
        if at <= base && base + unit.len as usize <= end {
            if let Some(unit) = u.regions.remove(&base) {
                let_go(&unit, true);
            }
            continue;
        }
        let (lo, hi) = ((at.max(base) - base) as u64, (end.min(base + unit.len as usize) - base) as u64);
        let inside: Vec<u64> = unit.pieces.range(lo..hi).filter(|(o, p)| **o + p.len <= hi).map(|(o, _)| *o).collect();
        let mut bytes = 0;
        for o in inside {
            if let Some(p) = unit.pieces.remove(&o) {
                bytes += p.len;
            }
        }
        unit.refused.retain(|&o, &mut (e, _)| e <= lo || o >= hi);
        if bytes > 0 {
            STATS[1].fetch_sub(bytes, Ordering::Relaxed);
            OPENING[0].fetch_sub(bytes, Ordering::Relaxed);
            OPENING[8].fetch_add(bytes, Ordering::Relaxed);
        }
    }
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
/// is a region or one run of a section and its piece opens (opened on
/// first use).
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
    let (heap, lo, size, offset) = match run {
        Some((section, offset)) => {
            let unit = u.sections.get_mut(&section)?;
            if offset.checked_add(len)? > unit.len {
                return None;
            }
            let (heap, lo, size) = piece(device, &mut u.rig, unit, Home::Section(section), offset, len)?;
            (heap, lo, size, offset)
        }
        None => {
            let (&base, unit) = u.regions.range_mut(..=start as usize).next_back()?;
            if end > base as u64 + unit.len {
                return None;
            }
            let offset = start - base as u64;
            let (heap, lo, size) = piece(device, &mut u.rig, unit, Home::Region(base), offset, len)?;
            (heap, lo, size, offset)
        }
    };
    let offset = offset - lo;
    let aligned = offset & !(K64 - 1);
    let bias = offset - aligned;
    let width = (bias + len).next_multiple_of(K64).min(size - aligned);
    let flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    let buffer = place_resource(device, &heap, aligned, &buffer_desc(width, flags), D3D12_RESOURCE_STATE_COMMON).ok()?;
    STATS[3].fetch_add(1, Ordering::Relaxed);
    STATS[4].fetch_add(len, Ordering::Relaxed);
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + bias;
    Some(Placed { buffer, gpu, bias, view: run.is_some() })
}

#[derive(Clone, Copy)]
enum Home {
    Region(usize),
    Section(usize),
}

/// The opened piece of `unit` that holds `[offset, offset + len)`: its
/// heap, its first byte's offset in the unit and its bytes; opened now if
/// there is none. `None`: the ask runs over the edge of a piece, lies on
/// pages that are not committed, or its piece does not open (the caller
/// copies).
fn piece(device: &ID3D12Device, rig: &mut Option<Rig>, unit: &mut Unit, home: Home, offset: u64, len: u64) -> Option<(ID3D12Heap, u64, u64)> {
    let end = offset + len;
    if let Some((&lo, p)) = unit.pieces.range(..=offset).next_back()
        && offset < lo + p.len
    {
        if end <= lo + p.len {
            return Some((p.heap.clone(), lo, p.len));
        }
        OPENING[5].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let now = crate::host::now_ms();
    if let Some((_, &(stop, until))) = unit.refused.range(..end).next_back()
        && stop > offset
        && now < until
    {
        return None;
    }
    // The slice of the grid around the ask, stretched to hold the ask, and
    // clipped to the pieces beside it.
    let origin = match home {
        Home::Region(base) => base as u64,
        Home::Section(_) => 0,
    };
    let grid = (origin + offset) & !(SLICE - 1);
    let mut lo = (grid.max(origin) - origin).min(offset & !(K64 - 1));
    let mut hi = (grid + SLICE - origin).max(end.next_multiple_of(K64)).min(unit.len);
    if let Some((&before, p)) = unit.pieces.range(..offset).next_back() {
        lo = lo.max(before + p.len);
    }
    if let Some((&after, _)) = unit.pieces.range(offset..).next() {
        if after < end {
            OPENING[5].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        hi = hi.min(after);
    }
    if let Home::Region(base) = home {
        if !unit.surveyed {
            unit.surveyed = true;
            survey(base, unit.len);
        }
        // Only pages the title committed: none is committed for the import.
        match committed_run(base, lo, hi, offset, end) {
            Some((a, b)) => (lo, hi) = (a, b),
            None => {
                OPENING[6].fetch_add(1, Ordering::Relaxed);
                unit.refused.insert(offset, (end, now + 2_000));
                return None;
            }
        }
    }
    if lo > offset || hi < end || hi <= lo {
        OPENING[5].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let started = std::time::Instant::now();
    // SAFETY: the unit's memory as `xmem` made it; `[lo, hi)` lies in it.
    let opened = unsafe { open_piece(device, rig, home, lo, hi - lo, offset - lo) };
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    let (what, at) = match home {
        Home::Region(at) => ("region", at),
        Home::Section(handle) => ("section", handle),
    };
    match opened {
        Ok(Some(heap)) => {
            let bytes = hi - lo;
            if unit.pieces.is_empty() {
                OPENING[2].fetch_add(unit.len, Ordering::Relaxed);
            }
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(bytes, Ordering::Relaxed);
            OPENING[if matches!(home, Home::Region(_)) { 0 } else { 1 }].fetch_add(bytes, Ordering::Relaxed);
            crate::memcap::count(crate::memcap::Use::Opened, bytes);
            tracing::info!(
                what,
                at = format_args!("{at:#x}"),
                offset_mb = lo >> 20,
                mb = bytes >> 20,
                unit_mb = unit.len >> 20,
                ms = format_args!("{ms:.0}"),
                "unified: title memory opened as a heap"
            );
            unit.pieces.insert(lo, Piece { len: bytes, heap: heap.clone() });
            Some((heap, lo, bytes))
        }
        Ok(None) => {
            // Nothing in it tells the title's bytes from other memory yet:
            // asked again in a while.
            OPENING[7].fetch_add(1, Ordering::Relaxed);
            unit.refused.insert(lo, (hi, now + 2_000));
            None
        }
        Err(why) => {
            STATS[2].fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                what,
                at = format_args!("{at:#x}"),
                offset_mb = lo >> 20,
                mb = (hi - lo) >> 20,
                why,
                "unified: title memory NOT opened; its buffers keep copies"
            );
            unit.refused.insert(lo, (hi, u64::MAX));
            None
        }
    }
}

/// Once per region, at its first piece: its committed and reserved bytes
/// (the opening of before committed and pinned all of it).
fn survey(base: usize, len: u64) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let end = base + len as usize;
    let mut at = base;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return;
        }
        let stop = (info.BaseAddress as usize + info.RegionSize).min(end).max(at + 4096);
        match info.State {
            MEM_COMMIT => OPENING[3].fetch_add((stop - at) as u64, Ordering::Relaxed),
            MEM_RESERVE => OPENING[4].fetch_add((stop - at) as u64, Ordering::Relaxed),
            _ => 0,
        };
        at = stop;
    }
}

/// Of the region at `base`, the run of committed read-write pages within
/// `[lo, hi)` that holds `[offset, end)`, cut inward to 64 KiB; `None` if
/// a page of the ask is not committed.
fn committed_run(base: usize, lo: u64, hi: u64, offset: u64, end: u64) -> Option<(u64, u64)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery};
    let mut at = lo;
    let mut run: Option<u64> = None;
    while at < hi {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        if unsafe { VirtualQuery((base as u64 + at) as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return None;
        }
        let stop = (info.BaseAddress as u64 + info.RegionSize as u64).saturating_sub(base as u64).min(hi).max(at + 4096);
        if info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE {
            run.get_or_insert(at);
        } else {
            if at >= end {
                // The run that holds the ask ends here.
                break;
            }
            run = None;
            if stop > offset {
                // A page of the ask is not committed.
                return None;
            }
        }
        at = stop;
    }
    let from = run?;
    let to = at.min(hi);
    (from <= offset && to >= end).then(|| (from.next_multiple_of(K64), to & !(K64 - 1)))
}

/// Opens `[lo, lo + len)` of a region or a section as a heap, and checks it
/// ([`check`]); `hint`: where in it the GPU is about to read. `Ok(None)`:
/// opened, but nothing in it could tell the title's memory from other
/// memory yet.
unsafe fn open_piece(device: &ID3D12Device, rig: &mut Option<Rig>, home: Home, lo: u64, len: u64, hint: u64) -> Result<Option<ID3D12Heap>, String> {
    let open = |at: *const u8| {
        device
            .cast::<ID3D12Device13>()
            // SAFETY: committed memory of `len` bytes at `at`.
            .and_then(|d| unsafe { d.OpenExistingHeapFromAddress1::<ID3D12Heap>(at.cast(), len as usize) })
            .map_err(|e| format!("OpenExistingHeapFromAddress1: {e}"))
    };
    // SAFETY (the whole function): the unit's memory as `xmem` made it.
    unsafe {
        match home {
            Home::Region(base) => {
                let at = (base as u64 + lo) as *mut u8;
                touch(at, len as usize);
                let heap = open(at)?;
                Ok(check(device, rig, &heap, at, len, hint)?.then_some(heap))
            }
            Home::Section(handle) => {
                // A view of the piece alone, opened and let go again, as
                // vkd3d-proton does with a whole section
                // (`OpenExistingHeapFromFileMapping`): the driver keeps the
                // pages pinned, and they are the pages every view shows.
                let view = mem::MapViewOfFile(handle as _, mem::FILE_MAP_WRITE, (lo >> 32) as u32, lo as u32, len as usize);
                if view.Value.is_null() {
                    return Err("no view of the piece".to_owned());
                }
                crate::mapstat::mapped(crate::mapstat::Site::Canary, len as usize);
                let at = view.Value.cast::<u8>();
                let checked = open(at).and_then(|heap| Ok(check(device, rig, &heap, at, len, hint)?.then_some(heap)));
                mem::UnmapViewOfFile(view);
                crate::mapstat::unmapped(crate::mapstat::Site::Canary, len as usize);
                checked
            }
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
    }
}

/// The newest announced write over `[at, at + len)`: 0 if none ever was
/// (memory only the title's CPU writes).
pub(crate) fn newest(at: usize, len: usize) -> u64 {
    settle();
    epochs().newest(at as u64, len as u64)
}

/// Shader writes through in-place buffers, by the submission that makes
/// them, until it has finished.
static GPU_WRITES: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());

/// At submission: the submission now being made writes `[at, at + len)`
/// through an in-place buffer. Announced once it has finished ([`settle`]).
pub(crate) fn gpu_wrote(at: usize, len: usize) {
    let number = crate::host::submissions_started();
    GPU_WRITES.lock().unwrap_or_else(|e| e.into_inner()).push((at, len, number));
}

/// Announces the shader writes of the submissions that have finished: the
/// textures over them are stale, and the copies that remain (`shadow.rs`)
/// take those pages again.
pub(crate) fn settle() {
    let finished = crate::host::submissions_finished();
    let done: Vec<(usize, usize)> = {
        let mut pending = GPU_WRITES.lock().unwrap_or_else(|e| e.into_inner());
        if pending.is_empty() {
            return;
        }
        let mut done = Vec::new();
        pending.retain(|&(at, len, number)| {
            let keep = number > finished;
            if !keep {
                done.push((at, len));
            }
            keep
        });
        done
    };
    for (at, len) in done {
        written(at, len);
        crate::watch::gpu_wrote(at, len);
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
    let ranges = epochs().len();
    tracing::info!(
        "unified: {} heaps opened ({} MB of title memory), {} refused; buffers in place: {} alive ({} MB), {} placed since the start ({} MB); \
         title copies on the GPU {}, on the CPU after waiting for the card {}; textures checked for an announced write {}, by their hash {}; \
         announced-write ranges kept {}",
        s[0],
        s[1] >> 20,
        s[2],
        live,
        live_bytes >> 20,
        s[3],
        s[4] >> 20,
        s[5],
        s[6],
        s[7],
        s[8],
        ranges
    );
    let o: Vec<u64> = OPENING.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let pieces: usize = {
        let u = units();
        u.regions.values().chain(u.sections.values()).map(|u| u.pieces.len()).sum()
    };
    tracing::info!(
        "unified opening: title memory open now {} MB in {} pieces (regions {} MB, sections {} MB) of {} MB in the units touched; \
         the regions touched had {} MB committed and {} MB only reserved when first looked over (opened whole before, all of it \
         committed and pinned); asks kept as copies: over a piece's edge {}, over pages not committed {}; pieces put off (nothing to \
         check them by yet) {}; let go because the title freed the memory {} MB",
        s[1] >> 20,
        pieces,
        o[0] >> 20,
        o[1] >> 20,
        o[2] >> 20,
        o[3] >> 20,
        o[4] >> 20,
        o[5],
        o[6],
        o[7],
        o[8] >> 20,
    );
}
