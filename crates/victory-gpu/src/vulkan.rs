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
use victory_vk::timeline::Timeline;

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
            tracing::info!("VICTORY_GPU=vulkan: the Vulkan backend runs (milestone M2a: queues, fences, memory, present; no draws yet)");
        }
        on
    })
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct Backend {
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
        .inspect(|_| start_landing())
}

// ---------------------------------------------------------------------------
// Title memory

struct Section {
    len: u64,
    /// The whole view `xmem` keeps of it (known from the first page shown).
    whole: usize,
    import: Option<Imported>,
    refused: bool,
}

#[derive(Default)]
struct Memory {
    /// `XMemVirtualAlloc` regions: first address -> bytes (imported in M2b,
    /// when programs read them).
    regions: BTreeMap<usize, usize>,
    sections: HashMap<usize, Section>,
    /// Large views of sections (one run of one section): title address ->
    /// (bytes, section, offset). Ranges of the memory map.
    views: BTreeMap<usize, (usize, usize, u64)>,
    /// Page reservations: first address -> bytes.
    reservations: BTreeMap<usize, usize>,
    pools: Vec<PoolView>,
    map: MemoryMap,
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

    /// The memory map from what is imported and the pool views.
    fn rebuild_map(&mut self) {
        let mut map = MemoryMap::default();
        let mut add = |r: Range| {
            let _ = map.insert(r);
        };
        for p in &self.pools {
            add(Range {
                title_start: p.title_start,
                title_size: p.len,
                device_base: p.device_base,
            });
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
        memory(b).regions.insert(at, len);
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

pub(crate) fn title_page_reservation(at: usize, len: usize) {
    if let Some(b) = backend() {
        memory(b).reservations.insert(at, len);
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
    let mut m = memory(b);
    let m = &mut *m;
    let to64 = to as u64;
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

pub(crate) fn title_memory_released(at: usize, len: usize) {
    let Some(b) = backend() else { return };
    let mut m = memory(b);
    m.regions
        .retain(|&a, &mut l| !(a >= at && a + l <= at + len));
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
    let (cpu, pool) = m.cpu_of(src)?;
    pool?;
    // SAFETY: a page of a section's whole view, mapped while it lives.
    unsafe { std::ptr::copy_nonoverlapping(cpu as *const u8, dst.as_mut_ptr(), dst.len()) };
    Some(true)
}

// ---------------------------------------------------------------------------
// Queues and fences

#[derive(Default)]
struct Part {
    waits: Vec<ValueWait>,
    writes: Vec<(u64, Written, usize)>,
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
            let mut moved = false;
            let keys: Vec<usize> = self.queues.keys().copied().collect();
            for q in keys {
                while let Some(front) = self.queues.get_mut(&q).and_then(|l| l.pop_front()) {
                    match front {
                        Item::Wait(fence, value) => {
                            if !force && !self.signalled(fence, value) {
                                self.queues
                                    .get_mut(&q)
                                    .expect("queue")
                                    .push_front(Item::Wait(fence, value));
                                break;
                            }
                            if force && !self.signalled(fence, value) {
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
                                    break;
                                }
                                if !ready {
                                    FORCED.fetch_add(1, Ordering::Relaxed);
                                }
                                // M2a: a part carries no commands yet, so one
                                // without writes has nothing to run (from M2b
                                // on its command buffers go here).
                                if part.writes.is_empty() {
                                    continue;
                                }
                                match b.timeline.submit(&b.host, &[]) {
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
                            if !parts.is_empty() {
                                self.queues
                                    .get_mut(&q)
                                    .expect("queue")
                                    .push_front(Item::Run(parts));
                                break;
                            }
                        }
                    }
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        self.queues.retain(|_, l| !l.is_empty());
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
        Some(p) if p.writes.is_empty() => p.waits.push(wait),
        _ => parts.push(Part {
            waits: vec![wait],
            writes: Vec::new(),
        }),
    }
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
        engine(b).lists.remove(&(list as usize));
    }
}

pub(crate) fn execute(queue: This, lists: &[This]) {
    let Some(b) = backend() else { return };
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
    e.pump(b, false);
    b.changed.notify_all();
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

pub(crate) fn present(frame: u64, size: Option<(u32, u32)>) {
    let Some(b) = backend() else { return };
    {
        let mut e = engine(b);
        e.pump(b, true);
    }
    memory(b).tick = frame;
    let mut sc = lock(&b.present);
    if sc.is_none() {
        let (w, h) = size.unwrap_or((1920, 1080));
        let hwnd = crate::host::title_window()
            .map(Ok)
            .unwrap_or_else(|| crate::host::window(w, h));
        // SAFETY: a plain query of this module's instance handle.
        let hinstance = unsafe { windows::Win32::System::LibraryLoader::GetModuleHandleW(None) };
        match (hwnd, hinstance) {
            (Ok(hwnd), Ok(inst)) => {
                match unsafe { Swapchain::new(&b.host, inst.0 as isize, hwnd.0 as isize) } {
                    Ok(s) => {
                        tracing::info!(
                            width = s.extent.width,
                            height = s.extent.height,
                            format = ?s.format,
                            mode = ?s.mode,
                            "Vulkan: presenting to a window"
                        );
                        *sc = Some(s);
                    }
                    Err(e) => {
                        tracing::error!(error = %format!("{e:#}"), "Vulkan: no swapchain");
                        return;
                    }
                }
            }
            _ => {
                tracing::error!("Vulkan: no window to present to");
                return;
            }
        }
    }
    let Some(s) = sc.as_mut() else { return };
    if b.timeline.lost() {
        lost_once();
        return;
    }
    let r = s.frame(&b.host, &b.timeline, |d, cmd, t| {
        let color = vk::ClearColorValue {
            float32: [0.02, 0.03, 0.06, 1.0],
        };
        // SAFETY: the image is in TRANSFER_DST_OPTIMAL while recording.
        unsafe {
            d.cmd_clear_color_image(
                cmd,
                t.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &color,
                &[victory_vk::swapchain::color_range()],
            )
        };
    });
    if let Err(e) = r {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 4 {
            tracing::error!(error = %format!("{e:#}"), "Vulkan: a present failed");
        }
    }
    drop(sc);
    report(b, frame);
}

/// One line every 10 s: the queues, the timeline, title memory.
fn report(b: &Backend, frame: u64) {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    {
        let mut last = lock(&LAST);
        if last.is_some_and(|t| t.elapsed() < Duration::from_secs(10)) {
            return;
        }
        *last = Some(Instant::now());
    }
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
        regions = m.regions.len(),
        sections = m.sections.len(),
        sections_imported_mb = imported >> 20,
        map_ranges = m.map.ranges().len(),
        pool_views = m.pools.len(),
        pool_card_mb = (pool.pages_in_card * PAGE) >> 20,
        pool_blocks = pool.blocks,
        pool_filled = pool.filled,
        pool_evicted = pool.evicted,
        pool_refilled = pool.refilled,
        pool_binds = pool.binds,
        "Vulkan: queues, timeline and title memory"
    );
}
