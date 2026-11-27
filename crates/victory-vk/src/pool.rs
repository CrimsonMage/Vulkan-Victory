//! The pool's GPU view in card memory (`docs/vulkan-host.md`, "Resources
//! and title memory"; the coordinator's design of 2026-11-26).
//!
//! On the console the title maps pool pages one 64 KiB page at a time into
//! a GPU view its CPU never touches (`CopyPageMappingsBatchX`). Here that
//! view is one sparse buffer whose pages are bound to pages of card memory,
//! one card page per physical page however many title pages show it,
//! bound in queue order at each mapping call and filled from the physical
//! page's home (the section's import) by a card copy when the page is
//! first shown, whenever the title announces it written (DirectStorage
//! completion, title copies) and after CPU stores into it, which the
//! caller's write tracking finds ([`PoolView::stored`]). Only pages
//! actually shown get card memory.
//!
//! Room: card pages come in 64 MiB blocks while everything of ours on the
//! card stays under the view's [`PoolView::ceiling`] (the one card-memory
//! ceiling the images share). Past it, the page touched least recently
//! (mapped, written, or named by a recording: [`PoolView::ensure`]) is
//! *evicted*: every title page showing it is unbound and its card page is
//! taken. Pages touched in the current tick go last, but they go: a map
//! never fails. An evicted page comes back, filled again from its home, at
//! its next announced write, mapping or naming. Until then it reads what an
//! unbound sparse page reads (zeros on a driver with
//! `residencyNonResidentStrict`; shortcut: a page read only through
//! descriptors the host never sees named stays so until the title touches
//! it). Pages nothing shows are unbound, never bound to a page of zeros:
//! binding a whole 16 GiB reservation page by page at creation stalled the
//! queue for seconds in the first M2a run.
//!
//! Every bind is checked against the buffer's size and the block's before
//! it goes to the queue, binds of one call are made once per page (the last
//! wins) and merged into runs, and they go in this order: unbinds, then the
//! fills of new pages, then the binds to card pages.
//!
//! Imported host memory cannot back a sparse buffer on NVIDIA (vkprobe,
//! 2026-11-26), which is why the pages live in card memory.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Context, Result, ensure};
use ash::vk;

use crate::host::Host;
use crate::memory::TITLE_USAGE;
use crate::timeline::{BufferBinds, Timeline};

pub const PAGE: u64 = 64 << 10;
/// Pages per block of card memory (64 MiB).
const BLOCK_PAGES: u32 = 1024;
/// Card memory the view may always take, past the ceiling if it must.
pub const LEAST: u64 = 256 << 20;

/// A physical page: its section (an opaque key) and its offset there.
pub type PageKey = (usize, u64);

/// Where a physical page's bytes are on the card's side: a buffer over
/// the section's import and the page's offset in it.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub buffer: vk::Buffer,
    pub offset: u64,
}

struct Block {
    memory: vk::DeviceMemory,
    /// A plain buffer over the whole block: the destination of fills.
    buffer: vk::Buffer,
    free: Vec<u32>,
    /// Given back ([`PoolView::shrink`]): no page is taken from it, and its
    /// memory is freed (null) once the card is past its unbinds.
    dead: bool,
}

#[derive(Clone, Copy)]
struct Slot {
    block: u32,
    page: u32,
}

struct Page {
    /// Its card page; `None` while evicted.
    slot: Option<Slot>,
    /// Title pages (indices in the view) showing it.
    shown: Vec<u64>,
    touched: u64,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    pub blocks: u64,
    pub pages_in_card: u64,
    pub filled: u64,
    pub evicted: u64,
    pub refilled: u64,
    pub binds: u64,
}

pub struct PoolView {
    pub buffer: vk::Buffer,
    pub title_start: u64,
    pub len: u64,
    pub device_base: u64,
    memory_type: u32,
    blocks: Vec<Block>,
    pages: HashMap<PageKey, Page>,
    /// Title page index -> the physical page it shows.
    bound: HashMap<u64, PageKey>,
    /// The pages that have a card page, least recently touched first.
    lru: BTreeSet<(u64, PageKey)>,
    /// The view takes another block only while everything of ours on the
    /// card ([`crate::account::card_total`]) stays under this.
    pub ceiling: u64,
    pub stats: Stats,
    /// Blocks given back, freed once the timeline passes the ticket.
    retired: Vec<(u64, usize)>,
    /// Title addresses of evicted pages [`PoolView::written`] and
    /// [`PoolView::ensure`] brought back: copies made with their CPU views
    /// not yet write-protected, for the caller to take and have tracked.
    pub back: Vec<u64>,
}

/// What a refresh of pages is for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Why {
    /// A recording names them: touched; evicted ones come back.
    Named,
    /// An announced write: as named, and those on the card filled again.
    Written,
    /// The CPU stored into them (write tracking): those on the card are
    /// filled again; evicted ones stay so (filled when they come back).
    Stored,
}

impl PoolView {
    /// A view of `len` bytes from title address `title_start`, every page
    /// unbound.
    pub fn new(
        host: &Host,
        timeline: &Timeline,
        title_start: u64,
        len: u64,
        ceiling: u64,
    ) -> Result<PoolView> {
        let d = &host.device;
        let len = len.next_multiple_of(PAGE);
        // SAFETY: valid create infos; everything is destroyed by `destroy`.
        let buffer = crate::account::traced(host, "sparse buffer", || unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .flags(
                        vk::BufferCreateFlags::SPARSE_BINDING
                            | vk::BufferCreateFlags::SPARSE_RESIDENCY,
                    )
                    .size(len)
                    .usage(TITLE_USAGE),
                None,
            )
        })
        .context("the pool view's sparse buffer")?;
        let needs = unsafe { d.get_buffer_memory_requirements(buffer) };
        ensure!(
            PAGE.is_multiple_of(needs.alignment),
            "sparse page {:#x}",
            needs.alignment
        );
        let memory_type = host
            .memory_type(
                needs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )
            .context("no card memory type for the pool view")?;
        let device_base = unsafe {
            d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        let mut view = PoolView {
            buffer,
            title_start,
            len,
            device_base,
            memory_type,
            blocks: Vec::new(),
            pages: HashMap::new(),
            bound: HashMap::new(),
            lru: BTreeSet::new(),
            ceiling,
            stats: Stats::default(),
            retired: Vec::new(),
            back: Vec::new(),
        };
        let _ = timeline;
        view.grow(host)?;
        Ok(view)
    }

    fn grow(&mut self, host: &Host) -> Result<()> {
        let d = &host.device;
        let bytes = u64::from(BLOCK_PAGES) * PAGE;
        // SAFETY: valid infos; destroyed by `destroy`.
        unsafe {
            let memory = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(bytes)
                        .memory_type_index(self.memory_type),
                    None,
                )
                .context("a block of card memory for the pool view")?;
            let buffer = match d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            ) {
                Ok(b) => b,
                Err(e) => {
                    d.free_memory(memory, None);
                    return Err(e.into());
                }
            };
            d.bind_buffer_memory(buffer, memory, 0)?;
            crate::account::allocated(
                host,
                memory,
                self.memory_type,
                bytes,
                crate::account::Kind::Pool,
            );
            let block = Block {
                memory,
                buffer,
                free: (0..BLOCK_PAGES).rev().collect(),
                dead: false,
            };
            // A freed block's place is taken again (slots name blocks by place).
            match self
                .blocks
                .iter()
                .position(|b| b.dead && b.memory == vk::DeviceMemory::null())
            {
                Some(i) => self.blocks[i] = block,
                None => self.blocks.push(block),
            }
        }
        self.stats.blocks += 1;
        Ok(())
    }

    fn take(&mut self) -> Option<Slot> {
        for (i, b) in self.blocks.iter_mut().enumerate() {
            if let Some(page) = b.free.pop() {
                return Some(Slot {
                    block: i as u32,
                    page,
                });
            }
        }
        None
    }

    /// The sparse binds for `binds` (title page, card page or none): once
    /// per page, the last wins; checked; merged into runs. Unbinds first.
    fn sparse(
        &self,
        binds: Vec<(u64, Option<Slot>)>,
    ) -> Result<(Vec<vk::SparseMemoryBind>, Vec<vk::SparseMemoryBind>)> {
        let pages: std::collections::BTreeMap<u64, Option<Slot>> = binds.into_iter().collect();
        let block_bytes = u64::from(BLOCK_PAGES) * PAGE;
        let (mut unbind, mut bind): (Vec<vk::SparseMemoryBind>, Vec<vk::SparseMemoryBind>) =
            (Vec::new(), Vec::new());
        // Runs: (first title page, pages, first card page).
        let mut run: Option<(u64, u64, Option<Slot>)> = None;
        let close =
            |run: (u64, u64, Option<Slot>), unbind: &mut Vec<_>, bind: &mut Vec<_>| -> Result<()> {
                let (t, n, slot) = run;
                ensure!(
                    (t + n) * PAGE <= self.len,
                    "a bind past the pool view: pages {t}+{n} of {}",
                    self.len / PAGE
                );
                let b = vk::SparseMemoryBind::default()
                    .resource_offset(t * PAGE)
                    .size(n * PAGE);
                match slot {
                    None => unbind.push(b),
                    Some(s) => {
                        let block = self
                            .blocks
                            .get(s.block as usize)
                            .context("a bind to a block that does not exist")?;
                        let offset = u64::from(s.page) * PAGE;
                        ensure!(
                            offset + n * PAGE <= block_bytes,
                            "a bind past its block: page {}+{n}",
                            s.page
                        );
                        bind.push(b.memory(block.memory).memory_offset(offset));
                    }
                }
                Ok(())
            };
        for (t, slot) in pages {
            let joins = run.is_some_and(|(rt, rn, rs)| {
                rt + rn == t
                    && match (rs, slot) {
                        (None, None) => true,
                        (Some(a), Some(b)) => {
                            a.block == b.block && u64::from(a.page) + rn == u64::from(b.page)
                        }
                        _ => false,
                    }
            });
            if joins {
                if let Some(r) = run.as_mut() {
                    r.1 += 1;
                }
                continue;
            }
            if let Some(r) = run.take() {
                close(r, &mut unbind, &mut bind)?;
            }
            run = Some((t, 1, slot));
        }
        if let Some(r) = run {
            close(r, &mut unbind, &mut bind)?;
        }
        Ok((unbind, bind))
    }

    /// Physical pages the title shows in the view (its working set, on the
    /// card or evicted).
    pub fn shown_pages(&self) -> u64 {
        self.pages.len() as u64
    }

    /// Bytes of card memory the view's live blocks take.
    pub fn card_bytes(&self) -> u64 {
        self.blocks.iter().filter(|b| !b.dead).count() as u64 * u64::from(BLOCK_PAGES) * PAGE
    }

    /// Gives card memory back until the view's blocks take at most
    /// `target` bytes: the blocks with the fewest pages in use are emptied
    /// (their pages evicted and unbound, as an eviction does) and freed once
    /// the card is past the unbinds.
    pub fn shrink(&mut self, host: &Host, timeline: &Timeline, target: u64) -> Result<()> {
        self.free_retired(host, timeline);
        let mut binds = Vec::new();
        let mut emptied = Vec::new();
        while self.card_bytes() > target {
            let Some(i) = self.coldest_block() else {
                break;
            };
            let lru = &mut self.lru;
            for (key, p) in &mut self.pages {
                if p.slot.is_some_and(|s| s.block as usize == i) {
                    p.slot = None;
                    lru.remove(&(p.touched, *key));
                    binds.extend(p.shown.iter().map(|&t| (t, None)));
                    self.stats.evicted += 1;
                    self.stats.pages_in_card -= 1;
                }
            }
            self.blocks[i].free.clear();
            self.blocks[i].dead = true;
            emptied.push(i);
        }
        if emptied.is_empty() {
            return Ok(());
        }
        self.flush(host, timeline, Vec::new(), binds)?;
        let ticket = timeline.last();
        self.retired
            .extend(emptied.into_iter().map(|i| (ticket, i)));
        Ok(())
    }

    /// The live block whose most recently touched page is the oldest (an
    /// empty block is the coldest).
    fn coldest_block(&self) -> Option<usize> {
        let mut newest: Vec<Option<u64>> = vec![None; self.blocks.len()];
        for p in self.pages.values() {
            if let Some(s) = p.slot {
                let n = &mut newest[s.block as usize];
                *n = Some(n.map_or(p.touched, |t| t.max(p.touched)));
            }
        }
        (0..self.blocks.len())
            .filter(|&i| !self.blocks[i].dead)
            .min_by_key(|&i| newest[i].map_or((0, 0), |t| (1, t)))
    }

    /// Frees the blocks given back whose unbinds the card has passed.
    fn free_retired(&mut self, host: &Host, timeline: &Timeline) {
        if self.retired.is_empty() {
            return;
        }
        let done = timeline.completed(host);
        let d = &host.device;
        for &(_, i) in self.retired.iter().filter(|(t, _)| *t <= done) {
            let b = &mut self.blocks[i];
            // SAFETY: nothing is bound to the block and no submission that
            // used it is still running.
            unsafe {
                d.destroy_buffer(b.buffer, None);
                crate::account::freed(b.memory);
                d.free_memory(b.memory, None);
            }
            b.memory = vk::DeviceMemory::null();
            b.buffer = vk::Buffer::null();
        }
        self.retired.retain(|(t, _)| *t > done);
    }

    /// A card page for a new physical page: a free one, a new block while
    /// ours stays under the ceiling, or the least recently touched page's
    /// (evicted; pages of the current tick last). The view may always take
    /// [`LEAST`]. `None` only when it has no card page at all and the driver
    /// gives no block: the page then stays evicted (it reads zeros) and
    /// nothing is half done.
    fn slot(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        binds: &mut Vec<(u64, Option<Slot>)>,
    ) -> Option<Slot> {
        if let Some(s) = self.take() {
            return Some(s);
        }
        self.free_retired(host, timeline);
        let block = u64::from(BLOCK_PAGES) * PAGE;
        let room = crate::account::card_total() + block <= self.ceiling
            || self.card_bytes() + block <= LEAST;
        if room && self.grow(host).is_ok() {
            return self.take();
        }
        let (_, victim) = self.lru.pop_first()?;
        let page = self.pages.get_mut(&victim).expect("a page in the LRU");
        let slot = page.slot.take().expect("a page in the LRU has a card page");
        binds.extend(page.shown.iter().map(|&t| (t, None)));
        self.stats.evicted += 1;
        self.stats.pages_in_card -= 1;
        Some(slot)
    }

    /// Marks `key` touched at `tick` (its place in the LRU follows).
    fn touch(&mut self, key: PageKey, tick: u64) {
        let Some(p) = self.pages.get_mut(&key) else { return };
        if p.slot.is_some() {
            self.lru.remove(&(p.touched, key));
            self.lru.insert((tick, key));
        }
        p.touched = tick;
    }

    /// Gives `key` (evicted, or new) card page `s`: every title page
    /// showing it is bound to it.
    fn place(&mut self, key: PageKey, s: Slot, binds: &mut Vec<(u64, Option<Slot>)>) {
        let p = self.pages.get_mut(&key).expect("a page to place");
        p.slot = Some(s);
        binds.extend(p.shown.iter().map(|&t| (t, Some(s))));
        self.lru.insert((p.touched, key));
        self.stats.pages_in_card += 1;
    }

    fn free(&mut self, slot: Slot) {
        self.blocks[slot.block as usize].free.push(slot.page);
    }

    /// The title maps `pages` at title address `at` (one entry per 64 KiB
    /// page: the physical page and where its bytes are, or `None` to show
    /// nothing). Fills and binds go to the queue in this order, behind
    /// everything given before.
    pub fn map(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        at: u64,
        pages: &[Option<(PageKey, Source)>],
        tick: u64,
    ) -> Result<()> {
        let first = (at - self.title_start) / PAGE;
        let mut binds = Vec::new();
        let mut fills: Vec<(Source, Slot)> = Vec::new();
        for (k, page) in pages.iter().enumerate() {
            let t = first + k as u64;
            if t * PAGE >= self.len {
                break;
            }
            // What it showed lets go of it.
            self.let_go(t);
            let Some((key, source)) = page else {
                binds.push((t, None));
                continue;
            };
            let p = self.pages.entry(*key).or_insert(Page {
                slot: None,
                shown: Vec::new(),
                touched: tick,
            });
            p.shown.push(t);
            self.bound.insert(t, *key);
            self.touch(*key, tick);
            if let Some(s) = self.pages[key].slot {
                binds.push((t, Some(s)));
                continue;
            }
            match self.slot(host, timeline, &mut binds) {
                Some(s) => {
                    if self.pages[key].shown.len() > 1 {
                        // Evicted, now back: its other title pages too.
                        self.stats.refilled += 1;
                    }
                    fills.push((*source, s));
                    self.place(*key, s, &mut binds);
                }
                None => binds.push((t, None)),
            }
        }
        self.flush(host, timeline, fills, binds)
    }

    /// Title page `t` shows nothing any more; a physical page nothing shows
    /// gives its card page back.
    fn let_go(&mut self, t: u64) {
        let Some(old) = self.bound.remove(&t) else { return };
        let Some(p) = self.pages.get_mut(&old) else { return };
        p.shown.retain(|&x| x != t);
        if p.shown.is_empty() {
            let gone = self.pages.remove(&old).expect("page");
            if let Some(s) = gone.slot {
                self.lru.remove(&(gone.touched, old));
                self.free(s);
                self.stats.pages_in_card -= 1;
            }
        }
    }

    fn flush(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        fills: Vec<(Source, Slot)>,
        binds: Vec<(u64, Option<Slot>)>,
    ) -> Result<()> {
        let (unbind, bind) = self.sparse(binds)?;
        if !unbind.is_empty() {
            self.stats.binds += unbind.len() as u64;
            timeline.bind_sparse(host, &[(self.buffer, unbind)])?;
        }
        if !fills.is_empty() {
            self.stats.filled += fills.len() as u64;
            let blocks: Vec<vk::Buffer> = self.blocks.iter().map(|b| b.buffer).collect();
            timeline.record(host, |d, cmd| unsafe {
                for (src, slot) in &fills {
                    d.cmd_copy_buffer(
                        cmd,
                        src.buffer,
                        blocks[slot.block as usize],
                        &[vk::BufferCopy::default()
                            .src_offset(src.offset)
                            .dst_offset(u64::from(slot.page) * PAGE)
                            .size(PAGE)],
                    );
                }
            })?;
        }
        if !bind.is_empty() {
            self.stats.binds += bind.len() as u64;
            let all: Vec<BufferBinds> = vec![(self.buffer, bind)];
            timeline.bind_sparse(host, &all)?;
        }
        Ok(())
    }

    /// The title announced `[at, at + len)` written: the physical pages
    /// shown there are filled again from `source` (evicted ones come back).
    pub fn written(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        range: (u64, u64),
        tick: u64,
        source: impl Fn(PageKey) -> Option<Source>,
    ) -> Result<()> {
        let keys = self.keys_in(range);
        self.refresh(host, timeline, keys, (tick, Why::Written), source).map(drop)
    }

    /// A recording names `[at, at + len)` (a root view, an index or
    /// indirect buffer, a copy): its pages are touched, and evicted ones
    /// come back, filled from `source`, before the recording runs (the
    /// binds go to the host line, which every later submission waits for).
    pub fn ensure(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        range: (u64, u64),
        tick: u64,
        source: impl Fn(PageKey) -> Option<Source>,
    ) -> Result<()> {
        let keys = self.keys_in(range);
        self.refresh(host, timeline, keys, (tick, Why::Named), source).map(drop)
    }

    /// The CPU stored into the pages shown at the title addresses `at`
    /// (those of this view): the ones on the card are filled again from
    /// `source`, in one submission. How many were.
    pub fn stored(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        at: &[u64],
        source: impl Fn(PageKey) -> Option<Source>,
    ) -> Result<u64> {
        let mut keys: Vec<PageKey> = at.iter().filter_map(|&t| self.page_at(t)).map(|(key, _)| key).collect();
        keys.sort_unstable();
        keys.dedup();
        self.refresh(host, timeline, keys, (0, Why::Stored), source)
    }

    /// Whether the page shown at title address `title` is on the card now.
    pub fn resident(&self, title: u64) -> bool {
        self.page_at(title).is_some_and(|(key, _)| self.pages[&key].slot.is_some())
    }

    /// The physical pages shown in `[at, at + len)`, once each.
    fn keys_in(&self, (at, len): (u64, u64)) -> Vec<PageKey> {
        let lo = at.max(self.title_start);
        let hi = at.saturating_add(len).min(self.title_start + self.len);
        if lo >= hi {
            return Vec::new();
        }
        let mut keys: Vec<PageKey> = ((lo - self.title_start) / PAGE
            ..(hi - self.title_start).div_ceil(PAGE))
            .filter_map(|t| self.bound.get(&t).copied())
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// The physical pages `keys`, by [`Why`]: touched at `tick`; evicted
    /// ones filled and bound again; those on the card filled again. How
    /// many were filled.
    fn refresh(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        keys: Vec<PageKey>,
        (tick, why): (u64, Why),
        source: impl Fn(PageKey) -> Option<Source>,
    ) -> Result<u64> {
        let mut binds = Vec::new();
        let mut fills = Vec::new();
        for key in keys {
            if why != Why::Stored {
                self.touch(key, tick);
            }
            let resident = self.pages[&key].slot;
            let as_it_is = match why {
                Why::Named => resident.is_some(),
                Why::Written => false,
                Why::Stored => resident.is_none(),
            };
            if as_it_is {
                continue;
            }
            let Some(src) = source(key) else { continue };
            let slot = match resident {
                Some(s) => s,
                None => {
                    let Some(s) = self.slot(host, timeline, &mut binds) else { continue };
                    self.place(key, s, &mut binds);
                    self.stats.refilled += 1;
                    self.back.extend(self.pages[&key].shown.first().map(|t| self.title_start + t * PAGE));
                    s
                }
            };
            fills.push((src, slot));
        }
        let filled = fills.len() as u64;
        if fills.is_empty() && binds.is_empty() {
            return Ok(0);
        }
        self.flush(host, timeline, fills, binds)?;
        Ok(filled)
    }

    /// Diagnostics: the pages shown, and the runs they make (title pages
    /// in a row showing physical pages in a row of one section): how
    /// scattered the title's mapping is.
    pub fn runs(&self) -> (u64, u64) {
        let mut shown: Vec<(u64, PageKey)> = self.bound.iter().map(|(&i, &k)| (i, k)).collect();
        shown.sort_unstable();
        let runs = shown
            .windows(2)
            .filter(|w| {
                let ((i, a), (j, b)) = (w[0], w[1]);
                !(j == i + 1 && b.0 == a.0 && b.1 == a.1 + PAGE)
            })
            .count() as u64;
        (shown.len() as u64, runs + u64::from(!shown.is_empty()))
    }

    /// The physical page a title address of the view shows.
    pub fn page_at(&self, title: u64) -> Option<(PageKey, u64)> {
        let off = title.checked_sub(self.title_start)?;
        let key = *self.bound.get(&(off / PAGE))?;
        Some((key, off % PAGE))
    }

    pub fn contains(&self, title: u64) -> bool {
        title >= self.title_start && title - self.title_start < self.len
    }

    /// # Safety
    /// The device is idle.
    pub unsafe fn destroy(&self, host: &Host) {
        let d = &host.device;
        unsafe {
            d.destroy_buffer(self.buffer, None);
            for b in &self.blocks {
                d.destroy_buffer(b.buffer, None);
                crate::account::freed(b.memory);
                d.free_memory(b.memory, None);
            }
        }
    }
}
