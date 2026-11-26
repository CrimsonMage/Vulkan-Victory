//! The Vulkan backend behind `VICTORY_GPU=vulkan` (`docs/vulkan-host.md`,
//! milestone M2a). The D3D12 host is not made at all ([`crate::host::host`]
//! answers `None`); in its place:
//!
//! - **Queues and fences.** Each title queue keeps a line of items: waits
//!   for fences, submitted lists (cut at their waits for memory values into
//!   parts), signals. A part whose waits memory meets, or a write in the
//!   air will meet, goes to the one host queue behind everything before it
//!   and signals the next value of the timeline semaphore
//!   ([`victory_vk::timeline`]). A landing thread follows the timeline: as
//!   each part's value is reached, its bottom-of-pipe writes are done and
//!   the fence signals behind it are set, then every queue's line runs on.
//!   M2a parts carry no draws or dispatches yet (M2b): only their order,
//!   writes and signals.
//! - **Title memory.** Regions, sections, views and page reservations are
//!   noted as `xmem` announces them. The pool's GPU view (the reservation
//!   `CopyPageMappingsBatchX` maps into) is a sparse buffer of card pages
//!   ([`victory_vk::pool`]): bound at each mapping call, filled from the
//!   section's import when a page is first shown and at every announced
//!   write. Sections are imported by host pointer (their whole views) when
//!   a page of theirs is first shown. The memory map (ABI v1) is built from
//!   what is imported.
//! - **Presentation.** A swapchain on the title's window (or one of ours);
//!   each present acquires, clears and presents, behind everything the
//!   queues gave. Planes are not composed yet: their images are drawn by
//!   the title's draws, which run from M2b on.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use victory_vk::ash::vk;
use victory_vk::host::Host;
use victory_vk::map::{MemoryMap, Range};
use victory_vk::memory::Imported;
use victory_vk::pool::{PAGE, PageKey, PoolView, Source};
use victory_vk::swapchain::Swapchain;
use victory_vk::timeline::{Recording, Timeline};

mod compose;
mod descriptors;
mod gpu_stats;
mod loose;
mod present;
mod rect;
mod images;
pub(crate) mod record;
mod stage_cache;
mod stages;

use crate::list::ValueWait;
use crate::memory::GpuPage;
use crate::object::This;
use crate::queue::Written;

/// Whether the Vulkan backend runs (`VICTORY_GPU=vulkan`; `d3d12` or unset:
/// the D3D12 path).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var("VICTORY_GPU").unwrap_or_default();
        let on = asked.trim().eq_ignore_ascii_case("vulkan");
        if on {
            tracing::info!("VICTORY_GPU=vulkan: the Vulkan backend runs (milestone M2b: draws, dispatches, card images, descriptors, planes)");
        }
        on
    })
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) struct Backend {
    host: Host,
    timeline: Timeline,
    memory: Mutex<Memory>,
    engine: Mutex<Engine>,
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
                    engine: Mutex::new(Engine::default()),
                    changed: Condvar::new(),
                    present: Mutex::new(None),
                })
            });
            match made {
                Ok(b) => {
                    let c = &b.host.caps;
                    tracing::info!(
                        device = %c.device_name,
                        driver = %c.driver,
                        descriptor_buffer = c.descriptor_buffer,
                        pipeline_library = c.pipeline_library,
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
            // SAFETY: the region as `xmem` reserved it; committed whole here.
            let ready = unsafe { crate::unified::ready_region(at, r.len) };
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

fn memory(b: &Backend) -> MutexGuard<'_, Memory> {
    lock(&b.memory)
}

pub(crate) fn title_region(at: usize, len: usize) {
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

pub(crate) fn title_section(handle: usize, len: u64) {
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

pub(crate) fn title_section_gone(handle: usize) {
    let Some(b) = backend() else { return };
    let gone = memory(b).sections.remove(&handle);
    if let Some(Section {
        import: Some(i), ..
    }) = gone
    {
        // Fills from it may be in the air.
        b.timeline.wait(&b.host, b.timeline.last(), 5_000_000_000);
        // SAFETY: the card has finished everything given so far.
        unsafe { i.destroy(&b.host) };
    }
}

pub(crate) fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    if len < BIG_VIEW {
        return;
    }
    if let Some(b) = backend() {
        let mut m = memory(b);
        m.views
            .retain(|&a, &mut (l, ..)| a + l <= at || a >= at + len);
        m.views.insert(at, (len, section, offset));
    }
}

/// `xmem` unmapped `[at, at + len)` (a view of physical pages, or a filler
/// stand-in): the views and imports there leave the map, card images over it
/// go. An import is dropped only after the card has finished everything
/// given so far (its pages are gone from the process).
pub(crate) fn title_memory_unmapped(at: usize, len: usize) {
    let Some(b) = backend() else { return };
    images::released(b, at as u64, len as u64, images::Dropped::Unmapped);
    let mut m = memory(b);
    let end = at + len;
    let before = m.views.len();
    m.views.retain(|&a, &mut (l, ..)| a + l <= at || a >= end);
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
    if m.views.len() == before && imports.is_empty() {
        return;
    }
    m.rebuild_map();
    drop(m);
    if !imports.is_empty() {
        b.timeline.wait(&b.host, b.timeline.last(), 5_000_000_000);
        for i in imports {
            // SAFETY: the card has finished everything given so far.
            unsafe { i.destroy(&b.host) };
        }
    }
}

pub(crate) fn title_page_reservation(at: usize, len: usize) {
    if let Some(b) = backend() {
        memory(b).reservations.insert(at, len);
    }
}

/// Every 10 s: all the card memory we allocated, by kind, against the
/// driver's own count of the card heaps' use (`VK_EXT_memory_budget`);
/// the difference is the driver's (shader local memory for spills,
/// internal heaps). Host-heap allocations and imports are listed apart:
/// they are system memory.
fn card_memory(b: &Backend) {
    let (card, host) = victory_vk::account::totals_mb();
    let (heap_used, heap_budget) = b
        .host
        .heaps()
        .iter()
        .filter(|(h, _)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .filter_map(|(_, bu)| *bu)
        .fold((0, 0), |(u, t), (budget, usage)| (u + (usage >> 20), t + (budget >> 20)));
    let ours: u64 = card.iter().sum();
    tracing::info!(
        images_mb = card[0],
        images_bound_mb = images::used_mb(),
        pool_mb = card[1],
        buffers_mb = card[3],
        other_mb = card[4],
        ours_mb = ours,
        heap_used_mb = heap_used,
        heap_budget_mb = heap_budget,
        driver_mb = heap_used.saturating_sub(ours),
        host_buffers_mb = host[3],
        imports_mb = host[2],
        "Vulkan: card memory"
    );
}

/// The live budget: every 60 frames each pool view may take what the
/// card's budget leaves beside everything else in use now (images,
/// imports, the swapchain, other programs), less 512 MiB of room; at least
/// 512 MiB. Without `VK_EXT_memory_budget` the start's figure stays.
fn follow_budget(b: &Backend, frame: u64) {
    if frame % 60 != 0 || !b.host.caps.memory_budget {
        return;
    }
    let Some((budget, usage)) = b
        .host
        .heaps()
        .iter()
        .filter(|(h, _)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .filter_map(|(_, bu)| *bu)
        .max_by_key(|(budget, _)| *budget)
    else {
        return;
    };
    let mut m = memory(b);
    for p in &mut m.pools {
        let others = usage.saturating_sub(p.card_bytes());
        p.budget = budget.saturating_sub(others).saturating_sub(512 << 20).max(512 << 20);
    }
}

/// Bytes of card memory the pool's GPU view may take: the card's budget
/// less 2 GiB for render targets, textures and the swapchain, at least
/// 512 MiB.
fn pool_budget(host: &Host) -> u64 {
    let card = host
        .heaps()
        .iter()
        .filter(|(h, _)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|(h, budget)| budget.map_or(h.size, |(b, _)| b))
        .max()
        .unwrap_or(0);
    card.saturating_sub(2 << 30).max(512 << 20)
}

pub(crate) fn title_pages_aliased(to: usize, pages: &[Option<GpuPage>]) {
    let Some(b) = backend().filter(|b| !b.timeline.lost()) else {
        return;
    };
    let to64 = to as u64;
    // Images over pages that now show other memory (or none) go (before
    // the memory lock): the title streams by mapping fresh pages under new
    // textures and taking old ones away, so images follow its live ones.
    images::released(b, to64, pages.len() as u64 * PAGE, images::Dropped::Remapped);
    let mut m = memory(b);
    let m = &mut *m;
    if !m.pools.iter().any(|p| p.contains(to64)) {
        let Some((&at, &len)) = m
            .reservations
            .range(..=to)
            .next_back()
            .filter(|(a, l)| to < *a + *l)
        else {
            if m.warned < 4 {
                m.warned += 1;
                tracing::warn!(
                    to = format_args!("{to:#x}"),
                    "Vulkan: pages mapped outside any page reservation are not shown"
                );
            }
            return;
        };
        let budget = pool_budget(&b.host);
        match PoolView::new(&b.host, &b.timeline, at as u64, len as u64, budget) {
            Ok(view) => {
                tracing::info!(
                    at = format_args!("{at:#x}"),
                    mb = len >> 20,
                    device = format_args!("{:#x}", view.device_base),
                    budget_mb = budget >> 20,
                    unbound_pages_read_zeros = b.host.caps.non_resident_strict,
                    "Vulkan: the pool's GPU view is a sparse buffer of card pages"
                );
                m.pools.push(view);
                m.rebuild_map();
            }
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "Vulkan: no sparse buffer for the pool's GPU view");
                return;
            }
        }
    }
    let mut entries = Vec::with_capacity(pages.len());
    let mut imported = false;
    for page in pages {
        entries.push(page.and_then(|g| {
            let s = m.sections.get_mut(&g.section)?;
            if s.whole == 0 {
                s.whole = g.cpu - g.offset as usize;
            }
            let had = s.import.is_some();
            let key = (g.section, g.offset);
            let src = m.source(&b.host, key);
            imported |= !had && src.is_some();
            src.map(|s| (key, s))
        }));
    }
    if imported {
        m.rebuild_map();
    }
    let lost = pages
        .iter()
        .zip(&entries)
        .filter(|(p, e)| p.is_some() && e.is_none())
        .count();
    if lost > 0 && m.warned < 8 {
        m.warned += 1;
        tracing::warn!(
            lost,
            to = format_args!("{to:#x}"),
            "Vulkan: pages of an unknown or unimported section show zeros in the pool's GPU view"
        );
    }
    let tick = m.tick;
    if let Some(view) = m.pools.iter_mut().find(|p| p.contains(to64))
        && let Err(e) = view.map(&b.host, &b.timeline, to64, &entries, tick)
    {
        tracing::error!(error = %format!("{e:#}"), "Vulkan: pages of the pool's GPU view were not mapped");
    }
}

pub(crate) fn title_memory_written(at: usize, len: usize) {
    let Some(b) = backend().filter(|b| !b.timeline.lost()) else {
        return;
    };
    // Card images over the memory are filled again.
    images::written(at as u64, len as u64);
    let mut guard = memory(b);
    let m = &mut *guard;
    let (at, len) = (at as u64, len as u64);
    if !m
        .pools
        .iter()
        .any(|p| at < p.title_start + p.len && at + len > p.title_start)
    {
        return;
    }
    let tick = m.tick;
    // The sections are imported already (their pages were shown).
    let sources: HashMap<usize, (vk::Buffer, u64)> = m
        .sections
        .iter()
        .filter_map(|(&h, s)| s.import.as_ref().map(|i| (h, (i.buffer, i.len))))
        .collect();
    for view in &mut m.pools {
        let r = view.written(&b.host, &b.timeline, at, len, tick, |key: PageKey| {
            let &(buffer, len) = sources.get(&key.0)?;
            (key.1.is_multiple_of(PAGE) && key.1 + PAGE <= len).then_some(Source {
                buffer,
                offset: key.1,
            })
        });
        if let Err(e) = r {
            tracing::error!(error = %format!("{e:#}"), "Vulkan: an announced write was not taken into the pool's GPU view");
        }
    }
}

/// The title released a texture resource (its last reference).
pub(crate) fn resource_released(address: u64, desc: &crate::resource::ResourceDesc) {
    if on()
        && let Some(b) = backend()
    {
        images::resource_gone(b, address, desc);
    }
}

pub(crate) fn title_memory_released(at: usize, len: usize) {
    let Some(b) = backend() else { return };
    images::released(b, at as u64, len as u64, images::Dropped::Freed);
    let mut m = memory(b);
    let gone: Vec<usize> = m
        .regions
        .iter()
        .filter(|&(&a, r)| a >= at && a + r.len <= at + len)
        .map(|(&a, _)| a)
        .collect();
    let mut dropped = false;
    for a in gone {
        if let Some(Region { import: Some(i), .. }) = m.regions.remove(&a) {
            // The card may still read it.
            b.timeline.wait(&b.host, b.timeline.last(), 5_000_000_000);
            // SAFETY: the card has finished everything given so far.
            unsafe { i.destroy(&b.host) };
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

/// The title created a pipeline an earlier run used: its stages are
/// translated and compiled once (`stages`).
pub(crate) fn pipeline_created(shaders: &[std::sync::Arc<crate::pipeline::Shader>]) {
    if let Some(b) = backend() {
        stages::layout(b);
        stages::created(b, shaders);
    }
}

// ---------------------------------------------------------------------------
// Queues and fences

#[derive(Default)]
struct Part {
    waits: Vec<ValueWait>,
    writes: Vec<(u64, Written, usize)>,
    /// The part's recorded command buffers (M2b).
    cmds: Vec<Recording>,
}

enum Item {
    Wait(usize, u64),
    Run(VecDeque<Part>),
    Signal(usize, u64),
}

struct Flight {
    value: u64,
    writes: Vec<(u64, Written, usize)>,
    signals: Vec<(usize, u64)>,
}

#[derive(Default)]
struct Engine {
    queues: HashMap<usize, VecDeque<Item>>,
    flights: VecDeque<Flight>,
    /// What each title list recorded since its reset.
    lists: HashMap<usize, Vec<Part>>,
}

static FORCED: AtomicU64 = AtomicU64::new(0);
/// Ranges the memory map could not take (full, or overlapping another
/// way), counted for the 10 s line.
static MAP_REFUSED: AtomicU64 = AtomicU64::new(0);

impl Engine {
    fn signalled(&self, fence: usize, value: u64) -> bool {
        crate::fence::completed(fence as This) >= value
            || self
                .flights
                .iter()
                .any(|f| f.signals.iter().any(|&(x, v)| x == fence && v >= value))
    }

    fn met(&self, wait: &ValueWait) -> bool {
        if wait.met() {
            return true;
        }
        let (at, _) = wait.place();
        // The last write in the air at that address decides.
        self.flights
            .iter()
            .flat_map(|f| f.writes.iter())
            .rev()
            .find(|(a, ..)| *a as usize == at)
            .is_some_and(|(_, what, _)| matches!(what, Written::Value(v) if wait.holds_for(*v)))
    }

    /// Runs every queue's line as far as it can; `force` passes waits that
    /// are not met (at a present, as the D3D12 path does).
    fn pump(&mut self, b: &Backend, force: bool) {
        loop {
            let keys: Vec<usize> = self.queues.keys().copied().collect();
            let mut moved = false;
            for q in keys {
                moved |= self.run_line(b, q, force);
            }
            if !moved {
                break;
            }
        }
        self.queues.retain(|_, l| !l.is_empty());
    }

    /// Runs queue `q`'s line as far as it can; whether anything moved.
    fn run_line(&mut self, b: &Backend, q: usize, force: bool) -> bool {
        let mut moved = false;
        while let Some(front) = self.queues.get_mut(&q).and_then(|l| l.pop_front()) {
            if let Some(held) = self.run_item(b, front, force) {
                self.queues.get_mut(&q).expect("queue").push_front(held);
                break;
            }
            moved = true;
        }
        moved
    }

    /// Runs one item of a line; what is left of it if it must wait.
    fn run_item(&mut self, b: &Backend, item: Item, force: bool) -> Option<Item> {
        match item {
            Item::Wait(fence, value) => {
                let signalled = self.signalled(fence, value);
                if !signalled && !force {
                    return Some(Item::Wait(fence, value));
                }
                if !signalled {
                    FORCED.fetch_add(1, Ordering::Relaxed);
                }
            }
            Item::Signal(fence, value) => match self.flights.back_mut() {
                Some(f) => f.signals.push((fence, value)),
                None => crate::fence::set(fence as This, value),
            },
            Item::Run(mut parts) => {
                while let Some(part) = parts.pop_front() {
                    let ready = part.waits.iter().all(|w| self.met(w));
                    if !ready && !force {
                        parts.push_front(part);
                        return Some(Item::Run(parts));
                    }
                    if !ready {
                        FORCED.fetch_add(1, Ordering::Relaxed);
                    }
                    self.submit_part(b, part);
                }
            }
        }
        None
    }

    /// Submits a part's commands (or a marker for its writes) behind
    /// everything before it.
    fn submit_part(&mut self, b: &Backend, part: Part) {
        // A part with neither commands nor writes has nothing to run.
        if part.writes.is_empty() && part.cmds.is_empty() {
            return;
        }
        let submitted = if part.cmds.is_empty() {
            b.timeline.submit(&b.host, &[])
        } else {
            images::flush(b);
            b.timeline.submit_recordings(&b.host, part.cmds)
        };
        match submitted {
            Ok(value) => self.flights.push_back(Flight {
                value,
                writes: part.writes,
                signals: Vec::new(),
            }),
            Err(e) => {
                if b.timeline.lost() {
                    lost_once();
                } else {
                    tracing::error!(error = %format!("{e:#}"), "Vulkan: a submission failed; its writes are done now");
                }
                for (at, what, bytes) in part.writes {
                    write_now(b, at, what, bytes);
                }
            }
        }
    }
}

/// A bottom-of-pipe write, done by the CPU when its part has run: into the
/// title's memory, or for the pool's GPU view into the page's home and
/// then the page's card copy.
fn write_now(b: &Backend, at: u64, what: Written, bytes: usize) {
    let target = memory(b).cpu_of(at);
    match target {
        Some((cpu, pool)) => {
            what.write(cpu as *mut u8, bytes);
            if pool.is_some() {
                title_memory_written(at as usize, bytes);
            }
        }
        None => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    at = format_args!("{at:#x}"),
                    "Vulkan: a write into a page of the pool's GPU view nothing shows was dropped"
                )
            });
        }
    }
}

fn engine(b: &Backend) -> MutexGuard<'_, Engine> {
    lock(&b.engine)
}

/// A list's wait for a memory value: what it records next is behind it.
pub(crate) fn list_wait(list: This, wait: ValueWait) {
    let Some(b) = backend() else { return };
    let mut e = engine(b);
    let parts = e.lists.entry(list as usize).or_default();
    match parts.last_mut() {
        Some(p) if p.writes.is_empty() && p.cmds.is_empty() => p.waits.push(wait),
        _ => parts.push(Part {
            waits: vec![wait],
            ..Part::default()
        }),
    }
}

/// A list goes: what it recorded and did not submit goes with it.
pub(crate) fn list_released(list: This) {
    list_reset(list);
    record::forget(list);
}

/// A list's recorded command buffer, into its current part (at a cut or
/// its close).
pub(crate) fn part_commands(list: This, rec: Recording) {
    let Some(b) = backend() else { return };
    let mut e = engine(b);
    let parts = e.lists.entry(list as usize).or_default();
    if parts.is_empty() {
        parts.push(Part::default());
    }
    parts.last_mut().expect("part").cmds.push(rec);
}

/// A list's bottom-of-pipe write.
pub(crate) fn list_write(list: This, at: u64, what: Written, bytes: usize) {
    let Some(b) = backend() else { return };
    let mut e = engine(b);
    let parts = e.lists.entry(list as usize).or_default();
    if parts.is_empty() {
        parts.push(Part::default());
    }
    parts
        .last_mut()
        .expect("part")
        .writes
        .push((at, what, bytes));
}

/// A list's reset (or release): what it recorded and did not submit goes.
pub(crate) fn list_reset(list: This) {
    if let Some(b) = backend() {
        let parts = engine(b).lists.remove(&(list as usize));
        for part in parts.into_iter().flatten() {
            for rec in part.cmds {
                b.timeline.abandon(&b.host, rec);
            }
        }
    }
}

pub(crate) fn execute(queue: This, lists: &[This]) {
    let Some(b) = backend() else { return };
    // What the title's threads wrote down into them is recorded first.
    let t = Instant::now();
    crate::list::join(lists);
    split(Split::Join, t);
    let mut e = engine(b);
    let mut parts = VecDeque::new();
    for &l in lists {
        match e.lists.remove(&(l as usize)) {
            Some(p) => parts.extend(p),
            // A list with nothing to wait for or write still runs, in order.
            None => parts.push_back(Part::default()),
        }
    }
    e.queues
        .entry(queue as usize)
        .or_default()
        .push_back(Item::Run(parts));
    let t = Instant::now();
    images::flush(b);
    split(Split::Flush, t);
    let t = Instant::now();
    e.pump(b, false);
    split(Split::Submit, t);
    b.changed.notify_all();
}

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
    (m.map.generation, m.map.block(zero, sink))
}

/// The map's generation now (regions wanted imported first).
fn map_generation(b: &Backend) -> u64 {
    let mut m = memory(b);
    if m.import_regions(&b.host) {
        m.rebuild_map();
    }
    m.map.generation
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

/// The buffer and offset of title address `title`.
fn buffer_of(b: &Backend, title: u64) -> Option<(vk::Buffer, u64)> {
    let mut m = memory(b);
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

/// An API call named title address `at`: the region holding it is imported
/// before the next command buffer binds the map.
pub(crate) fn touch(at: u64) {
    if at == 0 {
        return;
    }
    if let Some(b) = backend() {
        memory(b).want(at);
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
    let m = memory(b);
    let mut out = vec![0u8; len as usize];
    let mut done = 0u64;
    while done < len {
        let here = at + done;
        let page_left = PAGE - (here % PAGE);
        let n = page_left.min(len - done);
        let (cpu, _) = m.cpu_of(here).ok_or_else(|| anyhow::anyhow!("title memory at {here:#x} has no CPU view"))?;
        // SAFETY: committed title memory (or a section page's home).
        unsafe { std::ptr::copy_nonoverlapping(cpu as *const u8, out.as_mut_ptr().add(done as usize), n as usize) };
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

/// Saves the pipeline cache (at the end of a run).
pub(crate) fn save_cache() {
    if let Some(b) = backend() {
        stages::save_cache(b);
    }
}

pub(crate) fn signal(queue: This, fence: This, value: u64) {
    let Some(b) = backend() else {
        crate::fence::set(fence, value);
        return;
    };
    let mut e = engine(b);
    e.queues
        .entry(queue as usize)
        .or_default()
        .push_back(Item::Signal(fence as usize, value));
    e.pump(b, false);
    b.changed.notify_all();
}

pub(crate) fn wait(queue: This, fence: This, value: u64) {
    let Some(b) = backend() else { return };
    let mut e = engine(b);
    e.queues
        .entry(queue as usize)
        .or_default()
        .push_back(Item::Wait(fence as usize, value));
    e.pump(b, false);
}

/// A fence was signalled by the CPU: lines waiting for it run on.
pub(crate) fn fence_signalled() {
    let Some(b) = backend() else { return };
    engine(b).pump(b, false);
    b.changed.notify_all();
}

/// A CPU wait for a fence with no event: blocks until it is reached, or
/// until nothing given can reach it (then `false`).
pub(crate) fn wait_for_fence(fence: This, value: u64) -> bool {
    let Some(b) = backend() else { return false };
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut e = engine(b);
    loop {
        if crate::fence::completed(fence) >= value {
            return true;
        }
        if e.flights.is_empty() && e.queues.is_empty() {
            return false;
        }
        if Instant::now() > deadline {
            tracing::warn!(value, "Vulkan: a fence wait gave up after 10 s");
            return false;
        }
        e = b
            .changed
            .wait_timeout(e, Duration::from_millis(50))
            .unwrap_or_else(|p| p.into_inner())
            .0;
    }
}

fn start_landing() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-vk-landing".into())
            .spawn(landing);
    });
}

/// Says once, loudly, that the device was lost.
fn lost_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::error!(
            "Vulkan: the device was lost; the run goes on without the card (bottom-of-pipe writes and fence signals on the CPU, no presents)"
        )
    });
}

/// Follows the timeline: lands flights in order (their writes, then their
/// signals) and runs the queues' lines on.
fn landing() {
    let Some(b) = backend() else { return };
    loop {
        let next = {
            let e = engine(b);
            let next = e.flights.front().map(|f| f.value);
            if next.is_none() {
                let _ = b.changed.wait_timeout(e, Duration::from_millis(100));
                continue;
            }
            next
        };
        let Some(value) = next else { continue };
        // A lost device reaches nothing: what is in the air lands now, so
        // that the title's fences still move (its writes and signals on
        // the CPU, as without a card).
        let lost = b.timeline.lost();
        if lost {
            lost_once();
        } else if !b.timeline.wait(&b.host, value, 100_000_000) {
            continue;
        }
        let value = if lost { u64::MAX } else { value };
        let mut e = engine(b);
        while e.flights.front().is_some_and(|f| f.value <= value) {
            let f = e.flights.pop_front().expect("flight");
            for (at, what, bytes) in f.writes {
                write_now(b, at, what, bytes);
            }
            for (fence, v) in f.signals {
                crate::fence::set(fence as This, v);
            }
        }
        e.pump(b, false);
        drop(e);
        b.changed.notify_all();
    }
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
            loop {
                std::thread::sleep(Duration::from_secs(10));
                report(b, FRAME.load(Ordering::Relaxed));
            }
        });
    });
}

/// One line every 10 s: the queues, the timeline, title memory.
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
    let (flights, held) = {
        let e = engine(b);
        (
            e.flights.len(),
            e.queues.values().map(VecDeque::len).sum::<usize>(),
        )
    };
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
        forced_waits = FORCED.load(Ordering::Relaxed),
        presents_with_planes = PLANES_SHOWN.load(Ordering::Relaxed),
        regions = m.regions.len(),
        sections = m.sections.len(),
        sections_imported_mb = imported >> 20,
        map_ranges = m.map.ranges().len(),
        map_refused = MAP_REFUSED.load(Ordering::Relaxed),
        pool_views = m.pools.len(),
        pool_card_mb = (pool.pages_in_card * PAGE) >> 20,
        pool_blocks = pool.blocks,
        pool_filled = pool.filled,
        pool_evicted = pool.evicted,
        pool_refilled = pool.refilled,
        pool_binds = pool.binds,
        "Vulkan: queues, timeline and title memory"
    );
    stages::report();
    gpu_stats::collect(b);
    record::report();
    images::report();
    card_memory(b);
}
