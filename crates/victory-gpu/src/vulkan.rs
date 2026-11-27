//! The Vulkan backend (`docs/vulkan-host.md`), the one GPU back end:
//!
//! - **Queues and fences** (`engine.rs`). Each title queue keeps a line
//!   of items (fence waits, parts of submitted lists, signals) and has a
//!   timeline semaphore of its own ([`victory_vk::timeline`]). Fences
//!   between queues become semaphore waits; bottom-of-pipe writes are made
//!   by the card (`writes.rs`); a landing thread sets fences as the
//!   lines' values are reached.
//! - **Title memory.** Regions, sections, views and page reservations are
//!   noted as `xmem` announces them. The pool's GPU view (the reservation
//!   `CopyPageMappingsBatchX` maps into) is a sparse buffer of card pages
//!   ([`victory_vk::pool`]): bound at each mapping call, filled from the
//!   section's import when a page is first shown, at every announced
//!   write and after every CPU store into it (`crate::track`). Sections
//!   are imported by host pointer (their whole views) when
//!   a page of theirs is first shown. The memory map (ABI v1) is built from
//!   what is imported.
//! - **Presentation.** A swapchain on the title's window (or one of ours);
//!   each present acquires, clears and presents, behind everything the
//!   queues gave. Planes are not composed yet: their images are drawn by
//!   the title's draws, which run from M2b on.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use victory_vk::ash::vk;
use victory_vk::host::Host;
use victory_vk::map::{MemoryMap, Range};
use victory_vk::memory::Imported;
use victory_vk::pool::{PAGE, PageKey, PoolView, Source};
use victory_vk::swapchain::Swapchain;
use victory_vk::timeline::Timeline;

mod bound;
mod baking;
mod slow;
mod passes;
mod targets;
mod budget;
mod compose;
mod descriptors;
mod engine;
mod gpu_stats;
mod gputime;
mod guard;
mod loose;
mod paging;
mod probe;
mod remap;
mod present;
mod rect;
mod images;
pub(crate) mod record;
mod stage_cache;
mod stages;
mod tables;
mod writes;

pub use paging::{GpuPage, title_pages_aliased};
pub(crate) use paging::title_memory_written;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) struct Backend {
    host: Host,
    timeline: Timeline,
    memory: Mutex<Memory>,
    engine: Mutex<engine::Engine>,
    /// Woken when a flight lands or work is given.
    changed: Condvar,
    present: Mutex<Option<Swapchain>>,
}

fn backend() -> Option<&'static Backend> {
    static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            let made = Host::new().and_then(|host| {
                let timeline = Timeline::new(&host)?;
                Ok(Backend {
                    host,
                    timeline,
                    memory: Mutex::new(Memory::default()),
                    engine: Mutex::new(engine::Engine::default()),
                    changed: Condvar::new(),
                    present: Mutex::new(None),
                })
            });
            match made {
                Ok(b) => {
                    // Texture quality from the card's own memory, as the
                    // D3D12 host decides it.
                    let dedicated = b.host.heaps().iter().filter(|(h, _)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|(h, _)| h.size).max();
                    crate::quality::set(dedicated.unwrap_or(0));
                    let c = &b.host.caps;
                    tracing::info!(
                        device = %c.device_name,
                        driver = %c.driver,
                        descriptor_buffer = c.descriptor_buffer,
                        pipeline_library = c.pipeline_library,
                        queues = b.host.queues.lanes.len(),
                        "Vulkan device made"
                    );
                    for (need, name, purpose) in c.missing() {
                        tracing::info!(?need, %name, purpose, "Vulkan device lacks");
                    }
                    Some(b)
                }
                Err(e) => {
                    tracing::error!(error = %format!("{e:#}"), "no Vulkan device: nothing runs on the card");
                    None
                }
            }
        })
        .as_ref()
        .inspect(|_| {
            start_landing();
            start_reports();
        })
}

// ---------------------------------------------------------------------------
// Title memory

struct Region {
    len: usize,
    import: Option<Imported>,
    refused: bool,
    wanted: bool,
    /// Loose memory (no `XMemVirtualAlloc` region): joined with loose
    /// neighbours when the pages between are committed.
    loose: bool,
}

struct Section {
    len: u64,
    /// The whole view `xmem` keeps of it (known from the first page shown).
    whole: usize,
    import: Option<Imported>,
    refused: bool,
}

#[derive(Default)]
struct Memory {
    /// `XMemVirtualAlloc` regions: first address -> bytes, and the import
    /// once made: when an API call first names an address in it (a
    /// descriptor heap, a view, a root argument, an index buffer, a copy),
    /// at the next recording ([`touch`]), as the D3D12 path opens regions
    /// that hold a buffer it uses. Regions nothing names (CPU memory) are
    /// never pinned.
    regions: BTreeMap<usize, Region>,
    sections: HashMap<usize, Section>,
    /// Large views of sections (one run of one section): title address ->
    /// (bytes, section, offset). Ranges of the memory map.
    views: BTreeMap<usize, (usize, usize, u64)>,
    /// Page reservations: first address -> bytes.
    reservations: BTreeMap<usize, usize>,
    pools: Vec<PoolView>,
    map: MemoryMap,
    /// Imports replaced by larger ones: kept for the run (recorded command
    /// buffers not yet submitted may bind maps that name them).
    retired: Vec<Imported>,
    /// Imports out of use (`remap.rs`), freed once no command buffer that
    /// may name them is unsubmitted or running; and the ranges of loose
    /// imports over pages mapped anew, to import again once every page
    /// there is usable (base, bytes, frame).
    dying: Vec<(victory_vk::timeline::Mark, Imported)>,
    rejoin: Vec<(usize, usize, u64)>,
    /// The frame, for the pool view's least-recently-touched eviction.
    tick: u64,
    warned: u32,
}

/// Views at least this long are memory-map ranges (the pool view, mapped
/// whole once); smaller ones are the CPU view's scattered pages.
const BIG_VIEW: usize = 16 << 20;

impl Memory {
    /// The import of a section (made the first time), as a fill source.
    fn source(&mut self, host: &Host, key: PageKey) -> Option<Source> {
        let s = self.sections.get_mut(&key.0)?;
        if s.import.is_none() && !s.refused && s.whole != 0 {
            // SAFETY: `xmem` keeps the whole view mapped until the section
            // goes (`title_section_gone` drops the import first).
            match unsafe {
                Imported::new(
                    host,
                    s.whole as *mut _,
                    s.len,
                    vk::BufferUsageFlags::empty(),
                )
            } {
                Ok(i) => {
                    tracing::info!(
                        section = format_args!("{:#x}", key.0),
                        mb = s.len >> 20,
                        view = format_args!("{:#x}", s.whole),
                        device = format_args!("{:#x}", i.device_base),
                        "Vulkan: a section imported in place"
                    );
                    s.import = Some(i);
                }
                Err(e) => {
                    s.refused = true;
                    tracing::warn!(section = format_args!("{:#x}", key.0), error = %format!("{e:#}"), "Vulkan: a section was not imported; its pages show zeros");
                }
            }
        }
        let i = s.import.as_ref()?;
        (key.1.is_multiple_of(PAGE) && key.1 + PAGE <= i.len).then_some(Source {
            buffer: i.buffer,
            offset: key.1,
        })
    }

    /// Imports the regions not imported yet (their pages committed first,
    /// as the D3D12 path opens them); whether any was.
    fn import_regions(&mut self, host: &Host) -> bool {
        let mut any = false;
        for (&at, r) in self.regions.iter_mut() {
            if r.import.is_some() || r.refused || !r.wanted {
                continue;
            }
            // Pages write tracking protects are opened for the driver's
            // pinning, and stay open until the import is made.
            let _writing = crate::track::writing(at, r.len);
            // SAFETY: the region as `xmem` reserved it; committed whole here.
            let ready = unsafe { ready_region(at, r.len) };
            let made = ready.map_err(|e| anyhow::anyhow!(e)).and_then(|()| {
                // SAFETY: committed read-write pages that stay mapped until
                // the region is released (`title_memory_released`).
                unsafe { Imported::new(host, at as *mut _, r.len as u64, vk::BufferUsageFlags::empty()) }
            });
            match made {
                Ok(i) => {
                    tracing::info!(
                        at = format_args!("{at:#x}"),
                        mb = r.len >> 20,
                        device = format_args!("{:#x}", i.device_base),
                        "Vulkan: a region imported in place"
                    );
                    r.import = Some(i);
                    any = true;
                }
                Err(e) => {
                    r.refused = true;
                    tracing::warn!(at = format_args!("{at:#x}"), mb = r.len >> 20, error = %format!("{e:#}"), "Vulkan: a region was not imported; the card reads zeros there");
                }
            }
        }
        any
    }

    /// The buffer and offset of a title address, for copies, index and
    /// vertex buffers.
    fn buffer_of(&self, title: u64) -> Option<(vk::Buffer, u64)> {
        for p in &self.pools {
            if p.contains(title) {
                return Some((p.buffer, title - p.title_start));
            }
        }
        if let Some((&at, r)) = self.regions.range(..=title as usize).next_back()
            && title < (at + r.len) as u64
            && let Some(i) = &r.import
        {
            return Some((i.buffer, title - at as u64));
        }
        if let Some((&at, &(len, section, offset))) = self.views.range(..=title as usize).next_back()
            && title < (at + len) as u64
            && let Some(i) = self.sections.get(&section).and_then(|s| s.import.as_ref())
        {
            return Some((i.buffer, offset + (title - at as u64)));
        }
        None
    }

    /// The memory map from what is imported and the pool views.
    fn rebuild_map(&mut self) {
        let mut ranges = Vec::new();
        let mut add = |r: Range| ranges.push(r);
        for p in &self.pools {
            add(Range {
                title_start: p.title_start,
                title_size: p.len,
                device_base: p.device_base,
            });
        }
        for (&at, r) in &self.regions {
            if let Some(i) = &r.import {
                add(Range {
                    title_start: at as u64,
                    title_size: r.len as u64,
                    device_base: i.device_base,
                });
            }
        }
        for (&at, &(len, section, offset)) in &self.views {
            if let Some(i) = self.sections.get(&section).and_then(|s| s.import.as_ref()) {
                add(Range {
                    title_start: at as u64,
                    title_size: len as u64,
                    device_base: i.device_base + offset,
                });
            }
        }
        let mut map = MemoryMap::default();
        for r in MemoryMap::coalesce(ranges) {
            if let Err(e) = map.insert(r) {
                // Shaders reading there read zeros (and writes are lost).
                if MAP_REFUSED.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::warn!(at = format_args!("{:#x}", r.title_start), mb = r.title_size >> 20, error = %e, "Vulkan: title memory left out of the memory map");
                }
            }
        }
        map.generation = self.map.generation + 1;
        self.map = map;
    }

    /// The CPU address of a title address: itself, or for the pool's GPU
    /// view the page's place in its section's whole view.
    fn cpu_of(&self, title: u64) -> Option<(u64, Option<usize>)> {
        for (i, p) in self.pools.iter().enumerate() {
            if p.contains(title) {
                let (key, off) = p.page_at(title)?;
                let whole = self.sections.get(&key.0)?.whole;
                return (whole != 0).then_some((whole as u64 + key.1 + off, Some(i)));
            }
        }
        Some((title, None))
    }
}

/// Makes every page of a region committed and writable for the import:
/// reserved pages are committed, and every page is written once in place
/// (an atomic add of 0, which changes no byte whatever another thread
/// does) so that it can be pinned. Refuses pages the title protected
/// otherwise.
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

fn memory(b: &Backend) -> MutexGuard<'_, Memory> {
    lock(&b.memory)
}

/// `xmem` reserved a region for the title at `at`, `len` bytes (a multiple
/// of 64 KiB): imported when the card first needs it.
pub fn title_region(at: usize, len: usize) {
    if let Some(b) = backend() {
        memory(b).regions.insert(
            at,
            Region {
                len,
                import: None,
                refused: false,
                wanted: false,
                loose: false,
            },
        );
    }
}

/// `xmem` made a section for physical pages, `len` bytes of the title's.
pub fn title_section(handle: usize, len: u64) {
    if let Some(b) = backend() {
        memory(b).sections.insert(
            handle,
            Section {
                len,
                whole: 0,
                import: None,
                refused: false,
            },
        );
    }
}

/// The section's handle is closed (its pages are all freed): its import
/// goes once the card has finished with it.
pub fn title_section_gone(handle: usize) {
    let Some(b) = backend() else { return };
    let gone = memory(b).sections.remove(&handle);
    if let Some(Section {
        import: Some(i), ..
    }) = gone
    {
        // Fills from it and recorded copies may still name it.
        let mut m = memory(b);
        m.bury(b, i);
        m.rebuild_map();
    }
}

/// The runtime mapped `[at, at + len)` as a view of `section` from
/// `offset` on.
pub fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    if let Some(b) = backend() {
        let mut m = memory(b);
        m.rejoin_over(at, len);
        if len < BIG_VIEW {
            return;
        }
        m.views
            .retain(|&a, &mut (l, ..)| a + l <= at || a >= at + len);
        m.views.insert(at, (len, section, offset));
    }
}

/// `xmem` unmapped `[at, at + len)` (a view of physical pages, or a filler
/// stand-in): the views and imports there leave the map, card images over it
/// go. An import is dropped only after the card has finished everything
/// given so far (its pages are gone from the process).
pub fn title_memory_unmapped(at: usize, len: usize) {
    let Some(b) = backend() else { return };
    images::released(b, at as u64, len as u64, images::Dropped::Unmapped);
    let mut m = memory(b);
    let end = at + len;
    let before = m.views.len();
    m.views.retain(|&a, &mut (l, ..)| a + l <= at || a >= end);
    let remapped = m.drop_stale_loose(b, at, end);
    let gone: Vec<usize> = m.regions.range(..end).filter(|&(&a, r)| a + r.len > at && a >= at && a + r.len <= end).map(|(&a, _)| a).collect();
    let mut imports = Vec::new();
    for a in gone {
        if let Some(Region { import: Some(i), .. }) = m.regions.remove(&a) {
            imports.push(i);
        }
    }
    let (stale, kept): (Vec<Imported>, Vec<Imported>) =
        std::mem::take(&mut m.retired).into_iter().partition(|i| i.title_base >= at as u64 && i.title_base + i.len <= end as u64);
    m.retired = kept;
    imports.extend(stale);
    if m.views.len() == before && imports.is_empty() && !remapped {
        return;
    }
    m.rebuild_map();
    for i in imports {
        m.bury(b, i);
    }
}

/// `xmem` reserved `[at, at + len)` for physical pages (a placeholder).
pub fn title_page_reservation(at: usize, len: usize) {
    if let Some(b) = backend() {
        memory(b).reservations.insert(at, len);
    }
}

/// The title released a texture resource (its last reference).
pub(crate) fn resource_released(address: u64, desc: &crate::resource::ResourceDesc) {
    if let Some(b) = backend() {
        images::resource_gone(b, address, desc);
    }
}

/// The title freed, decommitted or unmapped `[at, at + len)`: card images
/// and imports over it go, and pool pages there are unmapped.
pub fn title_memory_released(at: usize, len: usize) {
    if len == 0 || (at as u64) >= 1 << 40 {
        return;
    }
    let Some(b) = backend() else { return };
    images::released(b, at as u64, len as u64, images::Dropped::Freed);
    let mut m = memory(b);
    let gone: Vec<usize> = m
        .regions
        .iter()
        .filter(|&(&a, r)| a >= at && a + r.len <= at + len)
        .map(|(&a, _)| a)
        .collect();
    let mut dropped = m.drop_stale_loose(b, at, at + len);
    for a in gone {
        if let Some(Region { import: Some(i), .. }) = m.regions.remove(&a) {
            // The card, or a command buffer not yet submitted, may still
            // read it.
            m.bury(b, i);
            dropped = true;
        }
    }
    if dropped {
        m.rebuild_map();
    }
    let tick = m.tick;
    for view in &mut m.pools {
        let lo = (at as u64).max(view.title_start);
        let hi = (at as u64 + len as u64).min(view.title_start + view.len);
        if lo < hi {
            let none = vec![None; ((hi - lo) / PAGE) as usize];
            let _ = view.map(&b.host, &b.timeline, lo, &none, tick);
        }
    }
}

/// Reads `dst.len()` bytes at title address `src` for the CPU (waits for
/// memory values): `None` if `src` is not one of ours to answer.
pub(crate) fn read(src: u64, dst: &mut [u8]) -> Option<bool> {
    let b = backend()?;
    let m = memory(b);
    m.cpu_of(src)?.1?;
    // Page by page: the pool's pages lie apart in their sections.
    let mut done = 0usize;
    while done < dst.len() {
        let here = src + done as u64;
        let n = ((PAGE - here % PAGE) as usize).min(dst.len() - done);
        let Some((cpu, Some(_))) = m.cpu_of(here) else { return Some(false) };
        // SAFETY: a page of a section's whole view, mapped while it lives.
        unsafe { std::ptr::copy_nonoverlapping(cpu as *const u8, dst.as_mut_ptr().add(done), n) };
        done += n;
    }
    Some(true)
}

/// Whether an earlier run kept a translation of the container `full`
/// (`stage_cache.rs`).
pub(crate) fn stage_known(full: &[u8; 32]) -> bool {
    stage_cache::known(full)
}

/// The title created a pipeline: its stages are translated and compiled
/// once (`stages`).
pub(crate) fn pipeline_created(shaders: &[std::sync::Arc<crate::pipeline::Shader>]) {
    if let Some(b) = backend() {
        stages::layout(b);
        stages::created(b, shaders);
    }
}

// ---------------------------------------------------------------------------
// Queues and fences (`engine.rs`)

pub(crate) use engine::{execute, fence_signalled, list_released, list_reset, list_wait, list_write, part_commands, signal, wait, wait_for_fence};
use budget::{card_memory, follow_budget};
use engine::{engine, lost_once, start_landing};

/// Ranges the memory map could not take (full, or overlapping another
/// way), counted for the 10 s line.
static MAP_REFUSED: AtomicU64 = AtomicU64::new(0);

/// Where the title's submitting and presenting threads, the list workers
/// and the view-creating threads spend time (the 10 s line, per frame).
#[derive(Clone, Copy)]
pub(crate) enum Split {
    /// `ExecuteCommandLists` waiting for its lists to be recorded.
    Join,
    /// New images set up and textures filled (detiling, staging).
    Flush,
    /// Parts submitted.
    Submit,
    /// The present: planes composed, acquire, submit and present.
    Present,
    /// Recording on the list workers (summed over workers).
    Record,
    /// Descriptor writes at view creation (summed over threads).
    Views,
}

static SPLIT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub(crate) fn split(what: Split, since: Instant) {
    SPLIT[what as usize].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Diagnostics: recording time and calls by the stream's op (the 10 s
/// line tells the costliest).
static OPS: [[AtomicU64; 2]; 64] = [const { [const { AtomicU64::new(0) }; 2] }; 64];

/// A recorded call of op `op` took the time since `since`.
pub(crate) fn op_time(op: u64, since: Instant) {
    let o = &OPS[(op as usize).min(63)];
    o[0].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    o[1].fetch_add(1, Ordering::Relaxed);
}

/// Diagnostics: the steps of one recording path, timed into [`OPS`] from
/// a first index on (48.. a dispatch's preparing: parts, stage, pass and
/// command buffer, dump hooks, windows and immediates, map, bind, push;
/// 57.. the dispatch: watch, the call, the note).
pub(crate) struct Lap(u64, Instant);

impl Lap {
    pub(crate) fn start(first: u64) -> Lap {
        Lap(first, Instant::now())
    }

    /// The step ends; the next begins.
    pub(crate) fn next(&mut self) {
        if cfg!(feature = "diagnostics") {
            op_time(self.0, self.1);
            self.0 += 1;
            self.1 = Instant::now();
        }
    }
}

impl Drop for Lap {
    fn drop(&mut self) {
        self.next();
    }
}

fn ops_text(frames: u64) -> String {
    let mut all: Vec<(usize, u64, u64)> = OPS.iter().enumerate().map(|(k, o)| (k, o[0].swap(0, Ordering::Relaxed), o[1].swap(0, Ordering::Relaxed))).filter(|o| o.2 != 0).collect();
    all.sort_by_key(|o| std::cmp::Reverse(o.1));
    let f = frames as f64;
    all.iter().take(16).map(|&(k, ns, n)| format!("op{k}: {:.1} ms {:.0} calls {:.1} us each", ns as f64 / 1e6 / f, n as f64 / f, ns as f64 / 1e3 / n as f64)).collect::<Vec<_>>().join(" | ")
}

// ---------------------------------------------------------------------------
// For the recording (M2b)

/// The backend, for the list workers' recording.
pub(crate) fn backend_for_recording() -> Option<&'static Backend> {
    backend().filter(|b| !b.timeline.lost())
}

/// The memory map as it is now (regions imported first), for a command
/// buffer about to bind it: its generation and block.
fn map_block(b: &Backend, (zero, sink): (u64, u64)) -> (u64, Vec<u8>) {
    let mut m = memory(b);
    if m.import_regions(&b.host) {
        m.rebuild_map();
    }
    (m.map.generation, m.map.block(zero, sink, tables::address(b)))
}

/// The map's generation now (regions wanted imported first).
fn map_generation(b: &Backend) -> u64 {
    let mut m = memory(b);
    if m.import_regions(&b.host) {
        m.rebuild_map();
    }
    m.map.generation
}

/// Diagnostics: root views by how much of their memory-map range is left
/// from their address (a root view's V# has 4 GiB of records): [in a range
/// with 4 GiB left, in a range with less (it reads zeros), in no range].
pub(crate) static ROOT_VIEWS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Diagnostics: a root view at `at` is pushed for a draw or dispatch.
/// True when its range has less than 4 GiB left or there is none.
pub(crate) fn root_view_short(b: &Backend, at: u64) -> bool {
    let left = memory(b).map.range_of(at).map(|r| r.title_start + r.title_size - at);
    let k = match left {
        Some(l) if l >= u64::from(u32::MAX) => 0,
        Some(_) => 1,
        None => 2,
    };
    ROOT_VIEWS[k].fetch_add(1, Ordering::Relaxed);
    k != 0
}

/// Diagnostics: whether the current memory map has a range holding `at`.
fn map_reaches(b: &Backend, at: u64) -> bool {
    memory(b).map.range_of(at).is_some()
}

/// Whether `[at, at + len)` is title memory the CPU reads: a title
/// allocation at its own address, or the pool's GPU view through the
/// pool's CPU view.
fn cpu_readable(b: &Backend, at: u64, len: u64) -> bool {
    let m = memory(b);
    // Pool memory: `read_title` reads each page through the pool's CPU view
    // (and refuses pages with no home).
    m.pools.iter().any(|p| p.contains(at) && p.contains(at + len.max(1) - 1))
        || m.regions.range(..=at as usize).next_back().is_some_and(|(&a, r)| at + len <= (a + r.len) as u64)
}

/// Diagnostics: the bytes of title memory the CPU reads from `at` to the
/// end of its allocation or pool view (0: none there).
fn title_left(b: &Backend, at: u64) -> u64 {
    let m = memory(b);
    if let Some(p) = m.pools.iter().find(|p| p.contains(at)) {
        return p.title_start + p.len - at;
    }
    m.regions.range(..=at as usize).next_back().map_or(0, |(&a, r)| ((a + r.len) as u64).saturating_sub(at))
}

/// The buffer and offset of title address `title` (one page of it named:
/// [`buffer_over`]).
fn buffer_of(b: &Backend, title: u64) -> Option<(vk::Buffer, u64)> {
    buffer_over(b, title, 1)
}

/// The buffer and offset of title memory `[title, title + len)` a
/// recording names; evicted pages of the pool's GPU view in it come back
/// first.
fn buffer_over(b: &Backend, title: u64, len: u64) -> Option<(vk::Buffer, u64)> {
    let mut m = memory(b);
    paging::ensure(b, &mut m, title, len);
    if let Some(found) = m.buffer_of(title) {
        return Some(found);
    }
    // In a region not imported yet: imported now (the map follows at the
    // next command buffer).
    if m.want(title) && m.import_regions(&b.host) {
        m.rebuild_map();
    }
    m.buffer_of(title)
}

/// The device address of title memory `[title, title + len)`, if one
/// range of the memory map holds it all (imported now if it must be).
fn device_of(b: &Backend, title: u64, len: u64) -> Option<u64> {
    buffer_over(b, title, len)?;
    let r = memory(b).map.range_of(title)?;
    (title + len <= r.title_start + r.title_size).then(|| r.device_base + (title - r.title_start))
}

/// An API call named title address `at`: the region holding it is imported
/// before the next command buffer binds the map; in the pool's GPU view,
/// its page comes back if it was evicted (a root view: its first page).
pub(crate) fn touch(at: u64) {
    if at == 0 {
        return;
    }
    if let Some(b) = backend() {
        let mut m = memory(b);
        m.want(at);
        paging::ensure(b, &mut m, at, 1);
    }
}

/// A descriptor heap was made: the card reads shader-visible ones (ABI
/// section 2: tables are the low 32 bits of their address).
pub(crate) fn heap_created(start: u64, size: u64, kind: u32, visible: bool, placed: bool) {
    if !visible {
        return;
    }
    let below_4g = start + size <= 1 << 32;
    if below_4g {
        tracing::info!(start = format_args!("{start:#x}"), kb = size >> 10, kind, placed, "Vulkan: a shader-visible descriptor heap");
    } else {
        tracing::warn!(start = format_args!("{start:#x}"), kb = size >> 10, kind, placed, "Vulkan: a shader-visible descriptor heap above 4 GiB: tables (32-bit addresses) cannot reach it");
    }
    // The whole heap, whatever is committed or mapped where now: every
    // piece of it is imported (it may lie across several views).
    let mut m = match backend() {
        Some(b) => memory(b),
        None => return,
    };
    for at in (start..start + size).step_by(1 << 20).chain([start + size - 1]) {
        m.want(at);
    }
}

impl Memory {
    /// Marks the memory holding `at` wanted: its region, or for memory in
    /// no region, view or pool (a committed block of the title's own
    /// allocator, the Rust heap) the committed run of pages holding it,
    /// as a region of its own (the D3D12 path's "loose" heaps). Whether
    /// something is still to import.
    fn want(&mut self, at: u64) -> bool {
        match self.regions.range_mut(..=at as usize).next_back() {
            Some((&base, r)) if at < (base + r.len) as u64 => {
                if r.import.is_none() && !r.refused {
                    r.wanted = true;
                    return true;
                }
                return false;
            }
            _ => {}
        }
        if self.pools.iter().any(|p| p.contains(at)) {
            return false;
        }
        // A big view of a section is in the map once its section is
        // imported (pool pages bring that); otherwise the view itself is
        // imported below, as loose memory (the title's heaps sit in such
        // views: vk15's tables, 18 MB into a placed heap, never reached).
        if let Some((&v, &(len, section, _))) = self.views.range(..=at as usize).next_back()
            && at < (v + len) as u64
            && self.sections.get(&section).is_some_and(|s| s.import.is_some())
        {
            return false;
        }
        // Loose memory: the run of committed read-write pages around it.
        let Some((base, len)) = loose::committed_run(at) else {
            static UNKNOWN: AtomicU64 = AtomicU64::new(0);
            if UNKNOWN.fetch_add(1, Ordering::Relaxed) < 24 {
                tracing::warn!(at = format_args!("{at:#x}"), "Vulkan: an address the card reads is in no title memory the host knows");
            }
            return false;
        };
        // Over loose memory imported before (the allocation has grown, or
        // was touched first at a higher address): one region of the union;
        // the old imports go once the card is done with them.
        let (base, len) = self.join_neighbours(base, len);
        let (base, len) = self.merge_loose(base, len);
        tracing::info!(at = format_args!("{at:#x}"), base = format_args!("{base:#x}"), mb = len >> 20, "Vulkan: loose title memory the card reads, imported as a region");
        self.regions.insert(
            base,
            Region {
                len,
                import: None,
                refused: false,
                wanted: true,
                loose: true,
            },
        );
        true
    }
}

/// Reads `len` bytes of title memory at `at` for the CPU (pages of the
/// pool's GPU view from their homes).
fn read_title(b: &Backend, at: u64, len: u64) -> anyhow::Result<Vec<u8>> {
    // Diagnostics read whatever a draw pushed (not always an address): only
    // memory known to be title memory the CPU reads (run vk26 crashed in a
    // copy from address 0 here).
    anyhow::ensure!(at != 0 && cpu_readable(b, at, len), "{at:#x} is not title memory the CPU reads");
    let m = memory(b);
    let mut out = vec![0u8; len as usize];
    let mut done = 0u64;
    while done < len {
        let here = at + done;
        let page_left = PAGE - (here % PAGE);
        let n = page_left.min(len - done);
        let (cpu, _) = m.cpu_of(here).filter(|&(c, _)| c != 0).ok_or_else(|| anyhow::anyhow!("title memory at {here:#x} has no CPU view"))?;
        // The kernel makes the copy: a page the title has decommitted or
        // unmapped since (a V# of a capture's walk names any range) is an
        // error here, not a fault (run gi6 ended in one).
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut copied = 0;
        // SAFETY: the destination is `n` bytes of `out`; the kernel checks the source.
        let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), cpu as *const _, out.as_mut_ptr().add(done as usize).cast(), n as usize, &mut copied) };
        anyhow::ensure!(ok != 0, "title memory at {here:#x} is not readable");
        done += n;
    }
    Ok(out)
}

fn eds3(b: &Backend) -> Option<&'static victory_vk::ash::ext::extended_dynamic_state3::Device> {
    static E: OnceLock<Option<victory_vk::ash::ext::extended_dynamic_state3::Device>> = OnceLock::new();
    E.get_or_init(|| {
        (b.host.caps.dynamic_blend || b.host.caps.dynamic_raster)
            .then(|| victory_vk::ash::ext::extended_dynamic_state3::Device::new(&b.host.instance, &b.host.device))
    })
    .as_ref()
}

fn mesh(b: &Backend) -> Option<&'static victory_vk::ash::ext::mesh_shader::Device> {
    static M: OnceLock<Option<victory_vk::ash::ext::mesh_shader::Device>> = OnceLock::new();
    M.get_or_init(|| {
        b.host
            .caps
            .mesh_shader
            .then(|| victory_vk::ash::ext::mesh_shader::Device::new(&b.host.instance, &b.host.device))
    })
    .as_ref()
}

/// `Create*View` / `CreateSampler`: the descriptor into the title's slot.
pub(crate) fn write_descriptor(handle: u64, view: &crate::descriptor::View) {
    descriptors::write(handle, view);
}


// ---------------------------------------------------------------------------
// Presentation

pub(crate) use compose::Shown;
pub(crate) use present::present;
use present::{FRAME, PLANES_SHOWN};



/// The 10 s lines are written by a thread of ours: the title's presenting
/// thread may be deep in its own stack (vk12 overflowed in `report`).
fn start_reports() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-vk-report".into()).spawn(|| {
            let Some(b) = backend() else { return };
            budget::watch();
            let mut last = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(250));
                budget::steps(b);
                engine::report_stalls(b);
                if last.elapsed() >= Duration::from_secs(10) {
                    last = Instant::now();
                    report(b, FRAME.load(Ordering::Relaxed));
                }
            }
        });
    });
}

/// One line every 10 s: the queues and their lines (GPU busy time per
/// frame and share of the wall clock), the timeline, title memory.
fn report(b: &Backend, frame: u64) {
    // Per frame since the last line, in ms.
    static LAST_FRAME: AtomicU64 = AtomicU64::new(0);
    let frames = frame.saturating_sub(LAST_FRAME.swap(frame, Ordering::Relaxed)).max(1);
    let per = |i: usize| SPLIT[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames as f64;
    let ms = [per(0), per(1), per(2), per(3), per(4), per(5)];
    tracing::info!(
        frames,
        join_ms = format_args!("{:.1}", ms[0]),
        flush_ms = format_args!("{:.1}", ms[1]),
        submit_ms = format_args!("{:.1}", ms[2]),
        present_ms = format_args!("{:.1}", ms[3]),
        record_ms = format_args!("{:.1}", ms[4]),
        views_ms = format_args!("{:.1}", ms[5]),
        "Vulkan: frame time split (ms a frame; record and views summed over threads)"
    );
    if cfg!(feature = "diagnostics") {
        tracing::info!(ops = %ops_text(frames), "Vulkan: recording by call (a frame; the stream's op numbers)");
        let r = |k: usize| ROOT_VIEWS[k].load(Ordering::Relaxed);
        tracing::info!(whole = r(0), short = r(1), unmapped = r(2), programs = %record::short_root_programs(), "Vulkan: root views pushed, by whether their map range has their V#'s 4 GiB left (short and unmapped read zeros)");
    }
    let (flights, held, lines) = engine::report_text(b, frames);
    let m = memory(b);
    let imported: u64 = m
        .sections
        .values()
        .filter_map(|s| s.import.as_ref())
        .map(|i| i.len)
        .sum();
    let pool = m
        .pools
        .iter()
        .fold(victory_vk::pool::Stats::default(), |mut a, p| {
            a.blocks += p.stats.blocks;
            a.pages_in_card += p.stats.pages_in_card;
            a.filled += p.stats.filled;
            a.evicted += p.stats.evicted;
            a.refilled += p.stats.refilled;
            a.binds += p.stats.binds;
            a
        });
    tracing::info!(
        frame,
        timeline = b.timeline.last(),
        reached = b.timeline.completed(&b.host),
        flights,
        held,
        forced_waits = engine::FORCED.load(Ordering::Relaxed),
        lines = %lines,
        presents_with_planes = PLANES_SHOWN.load(Ordering::Relaxed),
        regions = m.regions.len(),
        sections = m.sections.len(),
        sections_imported_mb = imported >> 20,
        map_ranges = m.map.ranges().len(),
        map_refused = MAP_REFUSED.load(Ordering::Relaxed),
        pool_views = m.pools.len(),
        pool_card_mb = (pool.pages_in_card * PAGE) >> 20,
        pool_shown_mb = (m.pools.iter().map(|p| p.shown_pages()).sum::<u64>() * PAGE) >> 20,
        pool_blocks = pool.blocks,
        pool_filled = pool.filled,
        pool_evicted = pool.evicted,
        pool_refilled = pool.refilled,
        pool_binds = pool.binds,
        "Vulkan: queues, timeline and title memory"
    );
    drop(m);
    stages::report();
    gpu_stats::collect(b);
    record::report();
    images::report();
    paging::report_written(frames);
    card_memory(b);
}

/// `victory bake`: compiles every translation the stage cache holds, as
/// the game would, into the pipeline caches; writes the pipeline index
/// (`baking.rs`).
pub fn bake(jobs: Option<usize>, index: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    baking::run(jobs, index)
}

/// `victory bake`, before compiling: the containers (BLAKE3, the first 16
/// bytes in hex) the title used with no translation under this
/// translator's key.
pub fn bake_wanted() -> std::collections::BTreeSet<String> {
    baking::wanted()
}

/// `victory bake`: translates a container of the title's packs as the game
/// would and keeps it (with its program name).
pub fn bake_translate(container: &[u8]) -> Result<(), String> {
    baking::translate(container)
}

/// `victory bake`, at the end: older keys' translations with a current
/// twin are deleted; how many.
pub fn bake_prune() -> usize {
    baking::prune()
}
