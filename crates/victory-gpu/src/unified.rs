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
//! Every heap is checked once when it is opened: vkd3d-proton gives memory
//! of its own, and says nothing, when an import fails. `xmem` leaves a
//! page of its own after each region and each section (the canary page),
//! where the CPU writes a pattern the GPU must read back through the heap.

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
/// Title buffers of this size and up keep their copy in the card's memory
/// (`shadow.rs` `card_for`): the title's big GPU heap (174 MB) is read by
/// most of its shaders, and the card reads its own memory at 199 GB/s
/// against 12.6 GB/s from the PC's (probe, RTX 5060 Ti on PCIe 4 x8).
pub(crate) const BIG: u64 = 16 << 20;

/// `VICTORY_UNIFIED`, see the module comment.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_UNIFIED").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1))
}

/// Whether unified memory is on (`VICTORY_UNIFIED` is not 0).
pub fn on() -> bool {
    level() != 0
}

enum State {
    Untried,
    Open(ID3D12Heap, u64),
    Refused,
}

/// A region or a section: the title's bytes (the canary page follows them)
/// and whether it is opened.
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

/// Since the start: units opened and their bytes, refused, buffers placed
/// and their bytes, title copies done on the GPU, CPU copies that waited
/// for flights, textures checked by an announced write, by their hash.
static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

pub(crate) enum Stat {
    GpuCopy = 5,
    CpuCopyWaited = 6,
    TextureByEvent = 7,
    TextureByHash = 8,
}

/// Bytes of title memory opened as host heaps so far: the system counts
/// them as PC memory mapped for the card, but they are the title's own
/// memory, not allocations of ours (`crate::memcap` leaves them out).
pub(crate) fn opened_bytes() -> u64 {
    STATS[1].load(Ordering::Relaxed)
}

pub(crate) fn count(stat: Stat) {
    STATS[stat as usize].fetch_add(1, Ordering::Relaxed);
}

/// `xmem` reserved a region for the title at `at`, `len` bytes (a multiple
/// of 64 KiB), with a committed canary page of its own right after it.
pub fn title_region(at: usize, len: usize) {
    units().regions.insert(at, Unit { len: len as u64, state: State::Untried });
}

/// `xmem` made a section for physical pages, `len` bytes of the title's
/// and a canary page after them.
pub fn title_section(handle: usize, len: u64) {
    units().sections.insert(handle, Unit { len, state: State::Untried });
}

/// The section's handle is closed: a new section may get its number. The
/// heap lives on in the buffers placed in it.
pub fn title_section_gone(handle: usize) {
    units().sections.remove(&handle);
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
    let (heap, size, offset) = match run {
        Some((section, offset)) => {
            let unit = u.sections.get_mut(&section)?;
            if offset.checked_add(len)? > unit.len {
                return None;
            }
            let (heap, size) = open(device, &mut u.rig, unit, Home::Section(section))?;
            (heap, size, offset)
        }
        None => {
            let (&base, unit) = u.regions.range_mut(..=start as usize).next_back()?;
            if end > base as u64 + unit.len {
                return None;
            }
            let (heap, size) = open(device, &mut u.rig, unit, Home::Region(base))?;
            (heap, size, start - base as u64)
        }
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
    Some(Placed { buffer, gpu, bias, view: run.is_some() })
}

#[derive(Clone, Copy)]
enum Home {
    Region(usize),
    Section(usize),
}

/// The unit's heap and its bytes, opening it the first time.
fn open(device: &ID3D12Device, rig: &mut Option<Rig>, unit: &mut Unit, home: Home) -> Option<(ID3D12Heap, u64)> {
    if let State::Untried = unit.state {
        let started = std::time::Instant::now();
        // SAFETY: the unit's memory, as `xmem` made it, and its canary page.
        let opened = unsafe { open_unit(device, rig, unit.len, home) };
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let (what, at) = match home {
            Home::Region(at) => ("region", at),
            Home::Section(handle) => ("section", handle),
        };
        unit.state = match opened {
            Ok(heap) => {
                STATS[0].fetch_add(1, Ordering::Relaxed);
                STATS[1].fetch_add(unit.len, Ordering::Relaxed);
                crate::memcap::count(crate::memcap::Use::Opened, unit.len);
                tracing::info!(what, at = format_args!("{at:#x}"), mb = unit.len >> 20, ms = format_args!("{ms:.0}"), "unified: title memory opened as a heap");
                State::Open(heap, unit.len + K64)
            }
            Err(why) => {
                STATS[2].fetch_add(1, Ordering::Relaxed);
                tracing::warn!(what, at = format_args!("{at:#x}"), mb = unit.len >> 20, why, "unified: title memory NOT opened; its buffers keep copies");
                State::Refused
            }
        };
    }
    match &unit.state {
        State::Open(heap, size) => Some((heap.clone(), *size)),
        _ => None,
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
        }
        check.map(|()| heap)
    }
}

/// Makes every page of a region committed and writable for the import:
/// reserved pages are committed (the title would commit them before use),
/// and every page is written once in place (an atomic add of 0, which
/// changes no byte whatever another thread does) so that pages a write
/// watch keeps read-only become writable and can be pinned. Refuses pages
/// the title protected otherwise.
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
            MEM_COMMIT if info.Protect == PAGE_READWRITE => {}
            // SAFETY: reserved pages of the region.
            MEM_RESERVE if !unsafe { VirtualAlloc(at as *const _, stop - at, MEM_COMMIT, PAGE_READWRITE) }.is_null() => {}
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
}
