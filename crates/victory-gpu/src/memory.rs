//! Title memory by directory (`docs/unified-memory.md`, "Title memory by
//! directory"; `docs/kiss.md`): the one way translated shaders reach
//! title memory outside the buffers the host places over it.
//!
//! The title maps its page pool page by page in scattered order (~62k
//! single-page views in run p13, and its GPU view of the pool only by
//! page-mapping copies), so no host buffer spans a window of it. Its home
//! stays the pages: each section of physical pages is opened as a heap
//! (`crate::unified`), and a shader finds a page by its title address in
//! the **directory**, one entry per 64 KiB of title address space: the
//! number of the copy holding the page and its byte offset there
//! (`victory_shader::emit::MEMORY_RUN_POOL`). A copy is a raw view over a
//! whole opened section (at most 512 MiB, the largest raw view there is),
//! or over a buffer the host placed over title memory (what raw-address
//! code names in regions and buffers, [`ready`]).
//!
//! - **Pool views.** A raw, structured or constant buffer view whose memory
//!   lies in a page reservation of the title's ([`page_reservation`]) is a
//!   pool view: its host descriptor is null and its heap slot's pool table
//!   entry holds its title address and bytes (`descriptor/host.rs`
//!   `make_slot`). Root views over pool memory likewise, through a
//!   dynamic slot's entry (`draw.rs`). Shaders read and write the view's
//!   bytes through the directory, so what they write lands in title memory
//!   itself; nothing is copied, gathered or copied back.
//! - **Entries.** A remap is an entry written: the page-mapping copies
//!   into the GPU view of the pool ([`title_pages_aliased`], no host view at
//!   all there) and, for pages a pool view or raw-address code names, the
//!   views the runtime maps ([`views_changed`]). Entries live in a mirror
//!   here; the blocks written since go to the directory, which lies in the
//!   card's memory, by GPU copies in a list of ours at the head of the next
//!   submission ([`flush`]): remaps land in queue order, as on the console,
//!   and the card's scattered lookups stay in its own memory.
//! - **Copies** are numbered when an entry first names them and opened
//!   (sections pinned) at the next [`ready`] or [`flush`], whose views go
//!   into every heap's memory run. A section no entry names is never
//!   opened (the title's CPU-only page pools).
//! - **Raw-address code** (the ray-tracing structure builder and the
//!   programs that walk its structures) reads the same directory; pages it
//!   finds empty are noted (the faults), read back once a frame and given
//!   copies from the next frame on ([`ready`]).
//!
//! - **The card's page cache** (`cache.rs`): pool pages lists name get a
//!   copy in the card's memory, and the entries of every title page that
//!   shows such a page name the copy instead of the section.
//!
//! Lock order: the directory, then `crate::views`, then the units
//! (`crate::unified`); the GPU-view pages ([`GPU_PAGES`]) on their own.

mod cache;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES, MEMORY_RUN_POOL, POOL_ENTRY_BYTES, POOL_SLOTS};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer).
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// Entries per block of the directory copied to the card (4 KiB).
const BLOCK: usize = 512;
/// The most a root view over pool memory names from its address on: a
/// root view has no size.
const ROOT_REACH: u64 = 16 << 20;

/// Bytes from the 16-byte line a buffer copy's raw view starts on to the
/// copy's first byte: the in-place buffer's offset below 16 bytes, 0 for a
/// buffer of its own (the directory's offsets count from the view's start).
fn skew(shadow: &Shadow) -> u64 {
    shadow.offset(shadow.span().0) % 16
}

/// What copy `n` is.
enum Copy {
    /// A section of physical pages, by handle, with the buffer over its
    /// whole heap once opened (`None` until then; refused sections stay
    /// `None` and read zeros).
    Section(usize, Option<ID3D12Resource>, bool),
    /// A chunk ([`crate::unified::CHUNK`]) of a region's heap, by region
    /// and chunk number, with the buffer over it once opened, as sections
    /// are: the title's root views and buffer views over its regions (its
    /// constants, its streams) by address, every byte at home.
    Chunk(crate::unified::Home, u64, Option<ID3D12Resource>, bool),
    /// A buffer of title memory (raw-address code's regions and buffers).
    Shadow(Arc<Shadow>),
    /// A block of the card's page cache (`cache.rs`); `None` once given
    /// back (its views read zeros once no submission can use it).
    Card(Option<ID3D12Resource>),
}

/// A heap's memory run: its base (copy 0's slot) and how many copies have
/// their view there.
#[derive(Clone, Copy)]
struct Run {
    base: u32,
    made: u32,
}

struct Memory {
    /// The directory in the card's memory and the faults buffer (with its
    /// mapping), made once there is a host device ([`Memory::gpu`]).
    gpu: Option<(ID3D12Resource, ID3D12Resource, *mut u32)>,
    /// The entries: low dword the copy, high dword the byte offset.
    mirror: Vec<u64>,
    /// Pages whose entry follows what the title maps (named by a pool view
    /// or raw-address code, or in the GPU view of the pool).
    demanded: Vec<u64>,
    /// Blocks written since the last [`flush`].
    dirty: Vec<u64>,
    any_dirty: bool,
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Copy>,
    /// Copies the heaps' runs are short of: opened and made at the next
    /// [`ready`] or [`flush`].
    unmade: bool,
    /// Sections by handle: their copy number.
    sections: HashMap<usize, u32>,
    /// Region chunks: their copy number.
    chunks: HashMap<(crate::unified::Home, u64), u32>,
    /// Per host view heap (by address): its run. A heap that goes says so
    /// ([`heap_gone`]) before it is freed.
    heaps: HashMap<usize, Run>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for lately that are not the title's memory, and when
    /// they were last forgotten (ms): a constant that only looks like an
    /// address is asked about by every dispatch.
    refused: HashSet<u64>,
    refused_at: u64,
    /// The card's page cache of pool pages (`cache.rs`).
    cache: cache::Cache,
}

// SAFETY: the mapping is of a buffer the struct owns; the heaps named in
// `heaps` say when they go (under the mutex); all access is under it.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// Since the start: entries written, blocks copied to the card, copies
/// opened, copies refused.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn memory() -> std::sync::MutexGuard<'static, Option<Memory>> {
    MEMORY.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a draw binds for a shader that reads title memory by directory:
/// the heap slot of copy 0 (`victory_shader::emit::USER_MEMORY_BASE`).
pub(crate) struct Ready {
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Memory {
        let _owner = victory_core::heap::enter(victory_core::heap::Owner::Directory);
        let pages = MEMORY_PAGES as usize;
        let blocks = pages / BLOCK;
        Memory {
            gpu: None,
            mirror: vec![0; pages],
            demanded: vec![0; pages / 64],
            // All of it the first time: the card's memory is not known to
            // start as zeros.
            dirty: vec![u64::MAX; blocks.div_ceil(64)],
            any_dirty: true,
            copies: Vec::new(),
            unmade: false,
            sections: HashMap::new(),
            chunks: HashMap::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: HashSet::new(),
            refused_at: 0,
            cache: cache::Cache::default(),
        }
    }

    /// The directory, the faults buffer and its mapping, made the first
    /// time (there must be a host device).
    fn gpu(&mut self) -> Option<(ID3D12Resource, ID3D12Resource, *mut u32)> {
        if self.gpu.is_none() {
            let host = host()?;
            let directory = host.uav_buffer(u64::from(MEMORY_PAGES) * 8)?;
            let (faults, data) = host.shared_buffer(u64::from(MEMORY_FAULTS) * 4)?;
            // SAFETY: a fresh mapping of that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, MEMORY_FAULTS as usize * 4) };
            tracing::info!(
                mb = MEMORY_PAGES * 8 >> 20,
                "title memory by directory: the directory lies in the card's memory, written by copies at each submission"
            );
            self.gpu = Some((directory, faults, data.cast()));
        }
        self.gpu.clone()
    }

    /// Writes the entry of `page` (a page number).
    fn set(&mut self, page: usize, entry: u64) {
        if page >= self.mirror.len() - 1 || self.mirror[page] == entry {
            return;
        }
        let old = std::mem::replace(&mut self.mirror[page], entry);
        self.cache.moved(page as u32, old, entry);
        self.block_dirty(page);
        STATS[0].fetch_add(1, Ordering::Relaxed);
    }

    /// The directory block of `page` goes to the card at the next [`flush`].
    fn block_dirty(&mut self, page: usize) {
        let block = page / BLOCK;
        self.dirty[block / 64] |= 1 << (block % 64);
        self.any_dirty = true;
    }

    fn demanded(&self, page: usize) -> bool {
        self.demanded.get(page / 64).is_some_and(|w| w >> (page % 64) & 1 != 0)
    }

    fn demand(&mut self, page: usize) {
        if let Some(w) = self.demanded.get_mut(page / 64) {
            *w |= 1 << (page % 64);
        }
    }

    /// The copy number of section `handle`, numbered now if it has none.
    fn section(&mut self, handle: usize) -> Option<u32> {
        if let Some(&n) = self.sections.get(&handle) {
            return Some(n);
        }
        if self.copies.len() >= COPIES as usize {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(COPIES, "title memory by directory: no number left for another copy; its pages read zeros"));
            return None;
        }
        self.copies.push(Copy::Section(handle, None, false));
        self.unmade = true;
        let n = self.copies.len() as u32;
        self.sections.insert(handle, n);
        Some(n)
    }

    /// The copy number of chunk `chunk` of region `home`, numbered now if
    /// it has none.
    fn chunk(&mut self, home: crate::unified::Home, chunk: u64) -> Option<u32> {
        if let Some(&n) = self.chunks.get(&(home, chunk)) {
            return Some(n);
        }
        if self.copies.len() >= COPIES as usize {
            return None;
        }
        self.copies.push(Copy::Chunk(home, chunk, None, false));
        self.unmade = true;
        let n = self.copies.len() as u32;
        self.chunks.insert((home, chunk), n);
        // Every page of the chunk at once (a region is not remapped): a
        // root view or buffer read past the pages asked for still finds
        // its bytes.
        if let crate::unified::Home::Region(base) = home {
            let first = base as u64 + chunk * crate::unified::CHUNK;
            let last = u64::from(MEMORY_PAGES) - 1;
            let len = crate::unified::region_len(base).map_or(0, |len| len.saturating_sub(chunk * crate::unified::CHUNK).min(crate::unified::CHUNK));
            for page in first / PAGE..(first + len).div_ceil(PAGE).min(last) {
                let at = page * PAGE - first;
                self.demand(page as usize);
                self.set(page as usize, u64::from(n) | at << 32);
            }
        }
        Some(n)
    }

    /// The entry of the title page at `address` as the runtime maps it now:
    /// a page of the GPU view of the pool, a view of a section, or a page
    /// of a region (its chunk); 0 for memory that is none of these.
    fn entry_now(&mut self, address: u64) -> u64 {
        let found = gpu_page(address)
            .map(|p| (p.section, p.offset))
            .or_else(|| crate::views::section_page(address as usize));
        if found.is_none()
            && let Some((home @ crate::unified::Home::Region(_), offset)) = crate::unified::source(address & !(PAGE - 1))
        {
            let chunk = offset / crate::unified::CHUNK;
            return match self.chunk(home, chunk) {
                Some(n) => u64::from(n) | (offset % crate::unified::CHUNK) << 32,
                None => 0,
            };
        }
        match found.and_then(|(section, offset)| Some((self.section(section)?, offset))) {
            Some((n, offset)) => u64::from(n) | offset << 32,
            None => 0,
        }
    }

    /// The pages of `[address, address + len)` follow what the title maps
    /// there from now on.
    fn demand_range(&mut self, address: u64, len: u64) {
        let (first, end) = (address / PAGE, address.saturating_add(len.max(1)).div_ceil(PAGE));
        for page in first..end.min(u64::from(MEMORY_PAGES) - 1) {
            let page = page as usize;
            if self.demanded(page) {
                continue;
            }
            self.demand(page);
            let entry = self.entry_now(page as u64 * PAGE);
            self.set(page, entry);
        }
    }

    /// Opens the copies numbered since and makes their views in every
    /// heap's run.
    fn make_copies(&mut self) {
        if !self.unmade {
            return;
        }
        self.unmade = false;
        for copy in &mut self.copies {
            match copy {
                Copy::Section(handle, buffer @ None, tried @ false) => {
                    *tried = true;
                    *buffer = crate::unified::section_buffer(*handle);
                    STATS[if buffer.is_some() { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
                }
                Copy::Chunk(home, chunk, buffer @ None, tried @ false) => {
                    *tried = true;
                    *buffer = crate::unified::span_buffer(*home, *chunk * crate::unified::CHUNK).map(|(b, _)| b);
                    STATS[if buffer.is_some() { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
                    if buffer.is_none() {
                        tracing::warn!(?home, chunk, "title memory by directory: a region's chunk did not open; its pages read zeros");
                    }
                }
                _ => {}
            }
        }
        let heaps: Vec<(usize, Run)> = self.heaps.iter().map(|(&k, &r)| (k, r)).collect();
        for (key, run) in heaps {
            // SAFETY: a heap named here is alive: it leaves the map under
            // this lock before it is freed (`heap_gone`).
            let heap = unsafe { &*(key as *const HostHeap) };
            let made = self.make_views(heap, run);
            self.heaps.insert(key, Run { made, ..run });
        }
    }

    /// Makes the views of the copies `heap`'s run is short of; returns how
    /// many it has now.
    fn make_views(&self, heap: &HostHeap, run: Run) -> u32 {
        let mut made = run.made;
        for (n, copy) in self.copies.iter().enumerate().skip(run.made as usize) {
            let slot = run.base + 1 + n as u32;
            match copy {
                Copy::Section(_, Some(buffer), _) => {
                    // SAFETY: a live buffer.
                    let width = unsafe { buffer.GetDesc() }.Width;
                    // The largest raw view: 2^27 dwords, a whole section.
                    heap.raw_uav_at(slot, buffer, 0, width.min(512 << 20));
                }
                Copy::Chunk(_, _, Some(buffer), _) => {
                    // SAFETY: a live buffer.
                    let width = unsafe { buffer.GetDesc() }.Width;
                    heap.raw_uav_at(slot, buffer, 0, width.min(512 << 20));
                }
                // Not opened (yet): reads zeros until it is.
                Copy::Section(_, None, false) | Copy::Chunk(_, _, None, false) => break,
                Copy::Section(_, None, true) | Copy::Chunk(_, _, None, true) | Copy::Card(None) => heap.null_raw_uav_at(slot),
                Copy::Card(Some(buffer)) => heap.raw_uav_at(slot, buffer, 0, cache::BLOCK_BYTES),
                Copy::Shadow(shadow) => {
                    let (first, len) = shadow.span();
                    // From the 16 bytes at or below the copy's first byte in
                    // its buffer (an in-place buffer starts on the 64 KiB
                    // line below it).
                    let at = shadow.offset(first);
                    heap.raw_uav_at(slot, &shadow.buffer, at - at % 16, len + at % 16);
                }
            }
            made = n as u32 + 1;
        }
        made
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's (raw-address code).
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 || self.mirror[page as usize] != 0 {
            return;
        }
        // Pool memory: its entry follows the runtime's mappings.
        if covers(address) {
            self.demand_range(page * PAGE, PAGE);
            return;
        }
        let start = page * PAGE;
        // A region's page: its chunk.
        let entry = self.entry_now(start);
        if entry != 0 {
            self.demand(page as usize);
            self.set(page as usize, entry);
            return;
        }
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let known = self.copies.iter().position(|c| matches!(c, Copy::Shadow(s) if Arc::ptr_eq(s, &shadow)));
        let n = match known {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(Copy::Shadow(shadow.clone()));
                self.unmade = true;
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    if self.mirror[(at / PAGE) as usize] == 0 {
                        let offset = at - first + skew(&shadow);
                        self.set((at / PAGE) as usize, (n as u64 + 1) | offset << 32);
                    }
                    at += PAGE;
                }
                n
            }
            None => return,
        };
        let offset = shadow.offset_in(start).unwrap_or(0) + skew(&shadow);
        self.set(page as usize, (n as u64 + 1) | offset << 32);
    }

    /// The pages raw-address code found empty since the last frame.
    fn take_faults(&mut self) {
        let Some((_, _, faults)) = self.gpu.clone() else {
            return;
        };
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { faults.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: AtomicU64 = AtomicU64::new(0);
            if SAID.fetch_add(1, Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }
}

/// Whether a shader that reads and writes title memory by address reaches
/// any of `[from, to)`: a page of it has a directory entry (a page without
/// one reads zeros and drops stores). Pages of the pool's GPU view have one
/// whenever the title maps them, other pool pages once something named
/// them (`crate::indirect`).
pub(crate) fn reachable(from: u64, to: u64) -> bool {
    let guard = memory();
    let Some(m) = guard.as_ref() else {
        return false;
    };
    let last = u64::from(MEMORY_PAGES) - 1;
    (from / PAGE..to.div_ceil(PAGE).min(last)).any(|page| m.mirror[page as usize] != 0)
}

/// Runs `f` on the directory, made the first time.
fn with<T>(f: impl FnOnce(&mut Memory) -> T) -> T {
    let mut guard = memory();
    f(guard.get_or_insert_with(Memory::new))
}

/// Before a draw whose shader reads title memory by directory: `heap`'s
/// memory run, made the first time, with a view of every copy opened so
/// far; the list holds what it names. For raw-address code (`by_address`)
/// also: the faults read once a frame, and what its root arguments name
/// given copies (`addresses`: its root views; `constants`: its root
/// constants, a title address in two of them or in one in 256-byte units:
/// the structure builder passes structures all three ways), and every
/// buffer copy kept up to date when `list` is submitted (and, with
/// `writes`, declared written). None if the host cannot hold it; the draw
/// is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    by_address: Option<(&[u64], &[u32], bool)>,
) -> Option<Ready> {
    let mut guard = memory();
    let m = guard.get_or_insert_with(Memory::new);
    let (directory, faults, _) = m.gpu()?;
    let numbered = m.copies.len();
    if let Some((addresses, constants, _)) = by_address {
        let frame = crate::frame::current();
        if m.frame != frame {
            m.frame = frame;
            // Memory the title commits later is found within two seconds.
            let now = crate::host::now_ms();
            if now >= m.refused_at + 2_000 {
                m.refused.clear();
                m.refused_at = now;
            }
            m.take_faults();
        }
        for &address in addresses {
            m.name(address);
        }
        for (i, &c) in constants.iter().enumerate() {
            if let Some(&high) = constants.get(i + 1)
                && (1..=0xFF).contains(&high)
                && c >= 0x1_0000
            {
                m.name(u64::from(c) | u64::from(high) << 32);
            }
            if c >= 0x0100_0000 {
                m.name(u64::from(c) << 8);
            }
        }
    }
    m.make_copies();
    let key = std::ptr::from_ref(heap) as usize;
    let run = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let first = heap.dynamic(MEMORY_RUN_POOL + 1 + COPIES)?;
            let base = first + MEMORY_RUN_POOL;
            match heap.pool_table() {
                Some((table, _)) => heap.raw_at(base - MEMORY_RUN_POOL, table, 0, u64::from(POOL_SLOTS) * u64::from(POOL_ENTRY_BYTES)),
                None => heap.null_raw_at(base - MEMORY_RUN_POOL),
            }
            heap.raw_uav_at(base - 2, &faults, 0, u64::from(MEMORY_FAULTS) * 4);
            heap.raw_at(base - 1, &directory, 0, u64::from(MEMORY_PAGES) * 8);
            for slot in base..=base + COPIES {
                heap.null_raw_uav_at(slot);
            }
            Run { base, made: 0 }
        }
    };
    let made = m.make_views(heap, run);
    m.heaps.insert(key, Run { made, ..run });
    // Once per list: everything below lives as long as the process or the
    // heap, which the list holds.
    const RUN: usize = usize::MAX - 0x6c;
    if list.first_use(RUN) {
        list.keep(&directory);
        list.keep(&faults);
    }
    if let Some((_, _, writes)) = by_address {
        // Once per list, and for copies numbered since: the builder alone
        // dispatches hundreds of times a frame.
        // shortcut: a copy another list numbers meanwhile is brought up to
        // date by that list's submission, which may run after this one's.
        const READ: usize = usize::MAX - 0x6d;
        const WRITE: usize = usize::MAX - 0x6e;
        let first_read = list.first_use(READ);
        let first_write = writes && list.first_use(WRITE);
        for (n, copy) in m.copies.iter().enumerate() {
            let Copy::Shadow(shadow) = copy else {
                continue;
            };
            let (start, len) = shadow.span();
            if first_read || n >= numbered {
                shadow.refresh(list, start, len);
            }
            if writes && (first_write || n >= numbered) {
                shadow.gpu_writes(list, start, len);
            }
        }
    }
    Some(Ready { base: run.base })
}

/// The host view heap at `key` goes: out of the runs before it is freed.
pub(crate) fn heap_gone(key: usize) {
    if let Some(m) = memory().as_mut() {
        m.heaps.remove(&key);
    }
}

/// At each submission, ahead of its lists: copies numbered since are
/// opened and given views, the card's page cache takes in the pool pages
/// `runs` name (`cache.rs`), and the directory blocks written since go to
/// the card, in a list of ours that runs first (queue order: lists
/// submitted before still see the entries they ran with). Also, per list
/// of `runs` (empty if none has any): a list of ours to run right after
/// it, that copies what its shaders wrote into cached pages back to the
/// pages (`submission`: this one's number).
pub(crate) fn flush(runs: &[&HostList], submission: u64) -> (Option<ID3D12CommandList>, Vec<Option<ID3D12CommandList>>) {
    // Read before the directory's lock (the lists' own locks are taken
    // after it only while recording).
    for list in runs {
        list.count_pool_bytes();
    }
    // Nothing named is read while the cache has no block: it tries to make
    // one first (names follow from the next lists recorded on).
    let named: Vec<Named> = if cache::live() || with(|m| m.cache_wake()) {
        runs.iter().map(|l| l.pool_names()).collect()
    } else {
        Vec::new()
    };
    let mut guard = memory();
    let Some(m) = guard.as_mut() else {
        return (None, Vec::new());
    };
    m.make_copies();
    let plan = if named.is_empty() { cache::Plan::default() } else { m.cache_plan(&named, submission) };
    // Blocks the cache made are copies too.
    m.make_copies();
    let needed = m.any_dirty || !plan.is_empty();
    let head = m.record_head(&plan);
    if head.is_none() && needed && m.cache.active() {
        // Entries would name slots nothing copied in, or slots given to
        // other pages.
        crate::host::fail(format_args!("the card page cache's copies in or the directory could not be recorded (no list or staging buffer)"));
    }
    let after = plan.after.iter().map(cache::record_after).collect();
    (head, after)
}

impl Memory {
    /// The list of ours ahead of a submission: the cache's copies in
    /// (`plan`), then the directory blocks written since. `None` if there
    /// is nothing to do.
    fn record_head(&mut self, plan: &cache::Plan) -> Option<ID3D12CommandList> {
        if !self.any_dirty && plan.is_empty() {
            return None;
        }
        let (directory, _, _) = self.gpu()?;
        let blocks: Vec<usize> = if self.any_dirty {
            (0..self.dirty.len() * 64).filter(|&b| self.dirty[b / 64] >> (b % 64) & 1 != 0).collect()
        } else {
            Vec::new()
        };
        let staging = if blocks.is_empty() {
            None
        } else {
            let bytes = (blocks.len() * BLOCK * 8) as u64;
            let (staging, data) = host()?.shared_buffer(bytes)?;
            for (i, &block) in blocks.iter().enumerate() {
                // Entries of pages the cache holds name its copy.
                for k in 0..BLOCK {
                    let entry = self.cache.entry(self.mirror[block * BLOCK + k]);
                    // SAFETY: the staging mapping holds `bytes` bytes; entry
                    // `k` of block `i` of it.
                    unsafe { data.cast::<u64>().add(i * BLOCK + k).write_unaligned(entry) };
                }
            }
            Some(staging)
        };
        let mut c = own_list()?;
        // SAFETY: our list, not executing (it came back after its submission
        // ran); live buffers, each copy inside both.
        let recorded = unsafe {
            c.allocator.Reset().and_then(|()| c.list.Reset(&c.allocator, None)).map(|()| {
                crate::host::order_writes(&c.list, c.list7.as_ref());
                if !plan.is_empty() {
                    plan.record_in(&c.list, c.list7.as_ref());
                }
                if let Some(staging) = &staging {
                    let mut i = 0;
                    while i < blocks.len() {
                        // Blocks that follow on, as one copy.
                        let mut j = i + 1;
                        while j < blocks.len() && blocks[j] == blocks[j - 1] + 1 {
                            j += 1;
                        }
                        let block_bytes = (BLOCK * 8) as u64;
                        c.list.CopyBufferRegion(
                            &directory,
                            blocks[i] as u64 * block_bytes,
                            staging,
                            i as u64 * block_bytes,
                            (j - i) as u64 * block_bytes,
                        );
                        i = j;
                    }
                }
                crate::host::order_writes(&c.list, c.list7.as_ref());
            })
        }
        // SAFETY: our list, recorded above.
        .and_then(|()| unsafe { c.list.Close() });
        if let Err(e) = recorded {
            crate::host::fail(format_args!("the directory of title memory could not be written ({e})"));
        }
        self.dirty.fill(0);
        self.any_dirty = false;
        STATS[1].fetch_add(blocks.len() as u64, Ordering::Relaxed);
        crate::frametime::add(crate::frametime::Sum::DirectoryBlocks, blocks.len() as u64);
        c.kept.extend(plan.kept.iter().cloned());
        if let Some(staging) = staging {
            c.kept.push(staging.clone());
            c.pooled.push(staging);
        }
        c.kept.push(directory);
        let list = c.list.cast().ok();
        crate::host::retire(Running(Some(c)));
        list
    }
}

/// What a list names of the pool by directory, for the card's page cache
/// (`HostList::pool_names`): the ranges whose pages it should hold
/// (address, bytes), the ranges its shaders write by directory, and the
/// other title memory its shaders or card copies write.
pub(crate) struct Named {
    pub(crate) reads: Vec<(u64, u64)>,
    pub(crate) writes: Vec<(u64, u64)>,
    pub(crate) other: Vec<(u64, u64)>,
    /// What shaders that pick buffers from the heap by index may write
    /// (from, to): not declared view by view.
    pub(crate) heap: Vec<(u64, u64)>,
    /// Pool views of the heap such shaders may read (address, bytes).
    pub(crate) maybe: Vec<(u64, u64)>,
}

// Page reservations and the GPU view of the pool.

/// The title's reservations of 64 KiB pages that take physical pages
/// (`xmem`'s placeholders), by start: end, and whether the title maps
/// pages there only by page-mapping copies (the GPU view of its pool,
/// which no host view shows).
static RESERVATIONS: RwLock<BTreeMap<u64, (u64, bool)>> = RwLock::new(BTreeMap::new());

/// `xmem` reserved `[at, at + len)` for physical pages (a placeholder).
pub fn page_reservation(at: usize, len: usize) {
    if crate::vulkan::on() {
        return crate::vulkan::title_page_reservation(at, len);
    }
    if len > 0 {
        RESERVATIONS.write().unwrap_or_else(|e| e.into_inner()).insert(at as u64, (at as u64 + len as u64, false));
        RESERVED.fetch_add(1, Ordering::Release);
    }
}

/// Changes of [`RESERVATIONS`] so far: what [`covers`] remembers per thread
/// holds while it stays.
static RESERVED: AtomicU64 = AtomicU64::new(1);

/// The reservation holding `address`: its start, end and whether it is a
/// GPU view.
fn reservation(address: u64) -> Option<(u64, u64, bool)> {
    let r = RESERVATIONS.read().unwrap_or_else(|e| e.into_inner());
    let (&start, &(end, gpu)) = r.range(..=address).next_back()?;
    (address < end).then_some((start, end, gpu))
}

/// Whether title memory at `address` is pool memory: in a page reservation,
/// where the title maps physical pages (one at a time, scattered), so the
/// GPU reaches it by directory.
/// Asked by every draw of each root view it reads: the thread's last
/// reservation found, and its last address found in none, are remembered
/// until the reservations change (no lock then).
pub(crate) fn covers(address: u64) -> bool {
    thread_local! {
        // (start, end, changes) of the last found; (page, changes) of the
        // last not found.
        static LAST: std::cell::Cell<((u64, u64, u64), (u64, u64))> = const { std::cell::Cell::new(((0, 0, 0), (u64::MAX, 0))) };
    }
    let changes = RESERVED.load(Ordering::Acquire);
    let ((start, end, seen), (page, missed)) = LAST.get();
    if seen == changes && (start..end).contains(&address) {
        return true;
    }
    if missed == changes && page == address / PAGE {
        return false;
    }
    match reservation(address) {
        Some((start, end, _)) => {
            LAST.set(((start, end, changes), (page, missed)));
            true
        }
        None => {
            LAST.set(((start, end, seen), (address / PAGE, changes)));
            false
        }
    }
}

/// A page of the GPU view of the pool: the section, the byte offset in it,
/// and where the CPU sees it (the section's whole view, `xmem`).
#[derive(Clone, Copy, Debug)]
pub struct GpuPage {
    pub section: usize,
    pub offset: u64,
    pub cpu: usize,
}

/// Whether any reservation is a GPU view (the CPU's reads ask [`gpu_view`]
/// at every copy).
static ANY_GPU_VIEW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The pages of GPU views, by title address.
static GPU_PAGES: RwLock<BTreeMap<u64, GpuPage>> = RwLock::new(BTreeMap::new());

/// The page of a GPU view at `address`, if mapped.
pub(crate) fn gpu_page(address: u64) -> Option<GpuPage> {
    GPU_PAGES.read().unwrap_or_else(|e| e.into_inner()).get(&(address & !(PAGE - 1))).copied()
}

/// Whether `[at, at + len)` touches a GPU view of the pool: memory no
/// host view shows, which the CPU reads through the pages' sections
/// ([`read`], [`write`]).
pub(crate) fn gpu_view(at: u64, len: u64) -> bool {
    if !ANY_GPU_VIEW.load(Ordering::Relaxed) {
        return false;
    }
    let r = RESERVATIONS.read().unwrap_or_else(|e| e.into_inner());
    let end = at.saturating_add(len.max(1));
    r.range(..end).rev().take_while(|(_, (stop, _))| *stop > at).any(|(_, &(_, gpu))| gpu)
}

/// The title had its GPU map pool pages at `[to, to + 64 KiB * pages.len())`
/// (`CopyPageMappingsBatchX`; `None`: unmapped there): no host view is
/// made there. The pages' entries follow; textures and windows over the
/// range see it remapped, and unmapped memory is let go.
pub fn title_pages_aliased(to: usize, pages: &[Option<GpuPage>]) {
    if crate::vulkan::on() {
        return crate::vulkan::title_pages_aliased(to, pages);
    }
    let (to, len) = (to as u64, pages.len() as u64 * PAGE);
    if len == 0 {
        return;
    }
    {
        let mut r = RESERVATIONS.write().unwrap_or_else(|e| e.into_inner());
        let hit = r.range_mut(..=to).next_back().filter(|(_, (end, _))| *end > to);
        match hit {
            Some((_, (_, gpu))) => *gpu = true,
            // Not a reservation `xmem` told: the range itself.
            None => {
                r.insert(to, (to + len, true));
                RESERVED.fetch_add(1, Ordering::Release);
            }
        }
        ANY_GPU_VIEW.store(true, Ordering::Relaxed);
    }
    {
        let mut g = GPU_PAGES.write().unwrap_or_else(|e| e.into_inner());
        for (k, page) in pages.iter().enumerate() {
            let at = to + k as u64 * PAGE;
            match page {
                Some(p) => g.insert(at, *p),
                None => g.remove(&at),
            };
        }
    }
    with(|m| {
        for (k, page) in pages.iter().enumerate() {
            let at = to + k as u64 * PAGE;
            let index = (at / PAGE) as usize;
            if index >= m.mirror.len() - 1 {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!(at = format_args!("{at:#x}"), "a GPU view of the pool past the directory's reach: shaders read zeros there"));
                break;
            }
            m.demand(index);
            let entry = match page.and_then(|p| Some((m.section(p.section)?, p.offset))) {
                Some((n, offset)) => u64::from(n) | offset << 32,
                None => 0,
            };
            m.set(index, entry);
        }
    });
    // What lay there is gone: placed textures arrive or go, windows look
    // their pages up again, textures detile again.
    crate::watch::title_memory_remapped(to as usize, len as usize);
    if pages.iter().all(Option::is_none) {
        crate::release::title_memory_released(to as usize, len as usize);
    }
}

/// The runtime mapped or unmapped views over `[at, at + len)`
/// (`crate::views`): entries of pages there that something named follow.
pub(crate) fn views_changed(at: u64, len: u64) {
    let mut guard = memory();
    let Some(m) = guard.as_mut() else {
        return;
    };
    let (first, end) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    for page in first..end.min(u64::from(MEMORY_PAGES) - 1) {
        let page = page as usize;
        if m.demanded(page) && gpu_page(page as u64 * PAGE).is_none() {
            let entry = m.entry_now(page as u64 * PAGE);
            m.set(page, entry);
        }
    }
}

/// A pool view of `[address, address + len)` is made: its pages' entries
/// follow what the title maps there from now on.
pub(crate) fn demand(address: u64, len: u64) {
    with(|m| m.demand_range(address, len));
}

/// Pages [`reach`] has seen to (one bit each), read without the lock.
static REACHED: [AtomicU64; MEMORY_PAGES as usize / 64] = [const { AtomicU64::new(0) }; MEMORY_PAGES as usize / 64];

/// Whether every page of `[first, end)` was seen to by [`reach`]: a word
/// of the bits at a time (a root view's reach is 256 pages).
fn all_reached(first: u64, end: u64) -> bool {
    let mut page = first;
    while page < end {
        let word = page / 64;
        let (lo, hi) = (page % 64, (end - word * 64).min(64));
        let mask = (if hi == 64 { u64::MAX } else { (1 << hi) - 1 }) & !((1u64 << lo) - 1);
        if REACHED[word as usize].load(Ordering::Relaxed) & mask != mask {
            return false;
        }
        page = (word + 1) * 64;
    }
    true
}

/// Shaders read and write `[address, address + len)` by directory (a root
/// view, a buffer view read by address): its pages get entries where the
/// memory is the title's (pool pages follow the runtime's mappings, region
/// pages name their chunk), and the page at `address` a copy of the buffer
/// there if it is in neither. Without the lock where every page was seen
/// to before: a draw asks this of every root view it reads. A page found
/// empty later is noted by the compute shaders that read it (the faults);
/// graphics shaders read zeros there.
pub(crate) fn reach(address: u64, len: u64) {
    let last = u64::from(MEMORY_PAGES) - 1;
    let (first, end) = (address / PAGE, address.saturating_add(len.max(1)).div_ceil(PAGE).min(last));
    if all_reached(first, end) {
        return;
    }
    with(|m| {
        m.demand_range(first * PAGE, (end - first) * PAGE);
        if first < end && m.mirror[first as usize] == 0 {
            m.map(address);
        }
    });
    for page in first..end {
        REACHED[(page / 64) as usize].fetch_or(1 << (page % 64), Ordering::Relaxed);
    }
}

/// The title freed region `[at, at + len)` (`crate::unified::released`):
/// the entries naming its chunks read zeros from the next submission on,
/// and [`reach`] sees to its pages again.
pub(crate) fn region_released(at: u64, len: u64) {
    let mut guard = memory();
    let Some(m) = guard.as_mut() else {
        return;
    };
    let home = crate::unified::Home::Region(at as usize);
    let gone: Vec<u32> = m.chunks.iter().filter(|((h, _), _)| *h == home).map(|(_, &n)| n).collect();
    if gone.is_empty() {
        return;
    }
    m.chunks.retain(|(h, _), _| *h != home);
    for &n in &gone {
        if let Some(Copy::Chunk(_, _, buffer, tried)) = m.copies.get_mut(n as usize - 1) {
            // Its view stays until a heap's run is made again; the buffer
            // holds the heap until then.
            *tried = true;
            let _ = buffer;
        }
    }
    let last = u64::from(MEMORY_PAGES) - 1;
    for page in at / PAGE..(at + len).div_ceil(PAGE).min(last) {
        let entry = m.mirror[page as usize];
        if gone.contains(&(entry as u32)) {
            m.set(page as usize, 0);
        }
        REACHED[(page / 64) as usize].fetch_and(!(1 << (page % 64)), Ordering::Relaxed);
    }
}

/// A root view at `address` over pool memory: the bytes it names (to the
/// end of its reservation, at most [`ROOT_REACH`]), demanded.
pub(crate) fn root_reach(address: u64) -> u64 {
    let len = reservation(address).map_or(PAGE, |(_, end, _)| (end - address).min(ROOT_REACH));
    demand(address, len);
    len
}

/// A list names a pool view of `[address, address + len)`: counted for the
/// frame line once per list, its pages taken into the card's page cache
/// at submission, and with `writes` declared written (CPU steps of later
/// submissions that read it wait for it, textures over it detile again
/// once it has run, windows over its pages copy them again, the cache
/// copies what it wrote back to the pages after the list).
pub(crate) fn named(list: &HostList, address: u64, len: u64, writes: bool) {
    named_as(list, address, len, len, writes);
}

/// The bytes from a root view's address on that the card's page cache
/// takes in: a root view has no size ([`root_reach`] is only how far it
/// may go), and a root constant buffer reads at most 64 KiB.
const ROOT_CACHED: u64 = 128 << 10;

/// [`named`] for a root view over pool memory, `reach` bytes from
/// `address` on ([`root_reach`]): the cache takes in only its first
/// [`ROOT_CACHED`] bytes; what lies further is read from the pages unless
/// something else named it.
pub(crate) fn named_root(list: &HostList, address: u64, reach: u64, writes: bool) {
    named_as(list, address, reach, reach.min(ROOT_CACHED), writes);
}

/// A pool view of the heap, `[address, address + len)`, that shaders which
/// pick buffers by index may read (`draw.rs` `heap_buffers`, once a frame
/// for every view of the heap): counted as [`named`] is, and taken into the
/// card's page cache only where it has room (nothing tells which views
/// such a shader reads).
pub(crate) fn named_heap(list: &HostList, address: u64, len: u64) {
    if list.first_pool_name(address | NAMED_HEAP) {
        list.pool_bytes(len);
        if cache::live() {
            list.name_pool_maybe(address, len);
        }
    }
}

/// Kinds of pool naming, above the 40 bits of a title address
/// (`HostList::first_pool_name`).
const NAMED_READ: u64 = 1 << 41;
const NAMED_HEAP: u64 = 1 << 42;

fn named_as(list: &HostList, address: u64, len: u64, cached: u64, writes: bool) {
    // Declared always (the cache may hold the pages by the time the list
    // runs; what it writes must reach them): as before the cache.
    if writes {
        list.defer_pool_write(address as usize, len as usize);
    }
    // Once per view and list, with no lock; for the cache only while it has
    // a block.
    if list.first_pool_name(address | NAMED_READ) {
        list.pool_bytes(len);
        if cache::live() {
            list.name_pool(address, cached);
        }
    }
}

/// Title memory `[at, at + len)` was written by a writer that says so
/// (`crate::unified::written`: DirectStorage, file reads, page mappings,
/// title copies on the CPU): the card's copies of those pages are taken
/// again at the next submission (`cache.rs`).
pub(crate) fn written(at: u64, len: u64) {
    cache::written(at, len);
}

/// A shader writes buffers it picks from the heap by index, and the heap's
/// unordered access views are not known yet: what it writes into pool pages
/// the card's page cache holds may be lost (counted in the frame line).
pub(crate) fn heap_written_unknown() {
    cache::heap_written_unknown();
}

/// Bytes of idle host copies of textures the card's page cache wants let go
/// of for a block (`crate::resource::let_go_idle`); taken by the asking.
pub(crate) fn room_wanted() -> u64 {
    cache::room_wanted()
}

/// For the frame line: the card's page cache (`cache.rs`).
pub(crate) fn cache_text(frames: f64) -> String {
    cache::frame_text(frames)
}

/// CPU access to GPU views of the pool (`crate::host::copy_mapped`,
/// `read_into`, `crate::texture`): byte for byte through the pages'
/// sections. Reads `len` bytes at `at` into `dst`; pages not mapped read
/// as zeros with `zeros`, else fail the read (false).
pub(crate) fn read(at: u64, dst: *mut u8, len: usize, zeros: bool) -> bool {
    let mut done = 0usize;
    let mut ok = true;
    while done < len {
        let a = at + done as u64;
        let n = ((PAGE - a % PAGE) as usize).min(len - done);
        match gpu_page(a) {
            // SAFETY: the section's whole view holds the page; `dst` has
            // room for `len` bytes.
            Some(p) => unsafe { std::ptr::copy_nonoverlapping((p.cpu + (a % PAGE) as usize) as *const u8, dst.add(done), n) },
            None if zeros => {
                // SAFETY: as above.
                unsafe { std::ptr::write_bytes(dst.add(done), 0, n) }
            }
            None => ok = false,
        }
        done += n;
    }
    ok
}

/// Writes `len` bytes from `src` at `at` of a GPU view of the pool (a
/// title copy run on the CPU): through the pages' sections; unmapped pages
/// are left out.
pub(crate) fn write(at: u64, src: *const u8, len: usize) {
    let mut done = 0usize;
    while done < len {
        let a = at + done as u64;
        let n = ((PAGE - a % PAGE) as usize).min(len - done);
        if let Some(p) = gpu_page(a) {
            // SAFETY: the section's whole view holds the page; `src` holds
            // `len` bytes.
            unsafe { std::ptr::copy((src).add(done), (p.cpu + (a % PAGE) as usize) as *mut u8, n) };
        }
        done += n;
    }
}

/// Of the 4 KiB pages of `[at, at + len)` in a GPU view: how many are
/// mapped, and how many there are (`crate::texture::readable`); with `all`,
/// it stops at the first that is not.
pub(crate) fn mapped(at: u64, len: u64, all: bool) -> (usize, usize) {
    const KERNEL_PAGE: u64 = 4096;
    let (first, last) = (at / KERNEL_PAGE, (at + len.max(1) - 1) / KERNEL_PAGE);
    let g = GPU_PAGES.read().unwrap_or_else(|e| e.into_inner());
    let mut there = 0;
    for page in first..=last {
        if g.contains_key(&((page * KERNEL_PAGE) & !(PAGE - 1))) {
            there += 1;
        } else if all {
            break;
        }
    }
    (there, (last - first + 1) as usize)
}

/// For the unified line: pool pages of GPU views mapped now, entries
/// written, directory blocks copied, sections opened and refused as
/// copies.
pub(crate) fn stats() -> [u64; 5] {
    let pages = GPU_PAGES.read().unwrap_or_else(|e| e.into_inner()).len() as u64;
    let s = STATS.each_ref().map(|n| n.load(Ordering::Relaxed));
    [pages, s[0], s[1], s[2], s[3]]
}

// Lists of ours for GPU copies (the directory's, `crate::gather`'s).

/// A list of ours, with what it names held until it has run, and buffers
/// of the PC's memory to give back to the pool then.
pub(crate) struct OwnList {
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) list7: Option<ID3D12GraphicsCommandList7>,
    pub(crate) kept: Vec<ID3D12Resource>,
    pub(crate) pooled: Vec<ID3D12Resource>,
}

// SAFETY: free-threaded D3D12 objects; a list is recorded by one thread and
// handed back only once its submission has run.
unsafe impl Send for OwnList {}

static SPARE: Mutex<Vec<OwnList>> = Mutex::new(Vec::new());

/// For the process memory line: lists of ours for copies kept spare.
pub(crate) fn spare_lists() -> usize {
    SPARE.lock().unwrap_or_else(|e| e.into_inner()).len()
}

/// A list of ours in a submission: back to [`SPARE`] once that has run
/// (`crate::host::retire`).
pub(crate) struct Running(pub(crate) Option<OwnList>);

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(mut list) = self.0.take() {
            list.kept.clear();
            for buffer in list.pooled.drain(..) {
                crate::memcap::give(crate::memcap::Kind::Shared, buffer);
            }
            SPARE.lock().unwrap_or_else(|e| e.into_inner()).push(list);
        }
    }
}

/// A closed list of ours to record copies into (`Reset` it first).
pub(crate) fn own_list() -> Option<OwnList> {
    if let Some(list) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return Some(list);
    }
    let host = crate::host::host()?;
    // SAFETY: plain object creation; the list is closed at once.
    let made = unsafe {
        (|| -> windows::core::Result<OwnList> {
            let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let list7 = crate::host::enhanced_barriers().then(|| list.cast().ok()).flatten();
            Ok(OwnList { allocator, list, list7, kept: Vec::new(), pooled: Vec::new() })
        })()
    };
    match made {
        Ok(list) => Some(list),
        Err(e) => {
            tracing::warn!(error = %e, "no host list for copies of ours");
            None
        }
    }
}
