//! The pool's GPU view in card memory (`docs/vulkan-host.md`, "Resources
//! and title memory"; the coordinator's design of 2026-11-26).
//!
//! On the console the title maps pool pages one 64 KiB page at a time into
//! a GPU view its CPU never touches (`CopyPageMappingsBatchX`). Here that
//! view is one sparse buffer whose pages are bound to pages of card memory,
//! one card page per physical page however many title pages show it,
//! bound in queue order at each mapping call and filled from the physical
//! page's home (the section's import) by a card copy when the page is
//! first shown and whenever the title announces it written (DirectStorage
//! completion, title copies). Only pages actually shown get card memory.
//!
//! Room: card pages come in 64 MiB blocks while the view's use stays under
//! its budget. Past it, the page touched least recently (mapped or written,
//! never one touched in the current tick) is *evicted*: every title page
//! showing it is unbound and its card page is freed. An evicted page comes
//! back, filled again, at its next announced write or mapping. Until then
//! it reads what an unbound sparse page reads (zeros on a driver with
//! `residencyNonResidentStrict`; shortcut: a page the title never announces
//! again stays so). Pages nothing shows are unbound, never bound to a page
//! of zeros: binding a whole 16 GiB reservation page by page at creation
//! stalled the queue for seconds in the first M2a run.
//!
//! Every bind is checked against the buffer's size and the block's before
//! it goes to the queue, binds of one call are made once per page (the last
//! wins) and merged into runs, and they go in this order: unbinds, then the
//! fills of new pages, then the binds to card pages.
//!
//! Imported host memory cannot back a sparse buffer on NVIDIA (vkprobe,
//! 2026-11-26), which is why the pages live in card memory.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use ash::vk;

use crate::host::Host;
use crate::memory::TITLE_USAGE;
use crate::timeline::{BufferBinds, Timeline};

pub const PAGE: u64 = 64 << 10;
/// Pages per block of card memory (64 MiB).
const BLOCK_PAGES: u32 = 1024;

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
    /// Bytes of card memory the view may take.
    pub budget: u64,
    pub stats: Stats,
}

impl PoolView {
    /// A view of `len` bytes from title address `title_start`, every page
    /// unbound.
    pub fn new(
        host: &Host,
        timeline: &Timeline,
        title_start: u64,
        len: u64,
        budget: u64,
    ) -> Result<PoolView> {
        let d = &host.device;
        let len = len.next_multiple_of(PAGE);
        // SAFETY: valid create infos; everything is destroyed by `destroy`.
        let buffer = unsafe {
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
        }
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
            budget,
            stats: Stats::default(),
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
            self.blocks.push(Block {
                memory,
                buffer,
                free: (0..BLOCK_PAGES).rev().collect(),
            });
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

    /// Bytes of card memory the view's blocks take.
    pub fn card_bytes(&self) -> u64 {
        self.blocks.len() as u64 * u64::from(BLOCK_PAGES) * PAGE
    }

    /// A card page for a new physical page: a free one, a new block within
    /// the budget, or the least recently touched page's (evicted).
    fn slot(
        &mut self,
        host: &Host,
        tick: u64,
        binds: &mut Vec<(u64, Option<Slot>)>,
    ) -> Result<Slot> {
        if let Some(s) = self.take() {
            return Ok(s);
        }
        let used = self.blocks.len() as u64 * u64::from(BLOCK_PAGES) * PAGE;
        if used + u64::from(BLOCK_PAGES) * PAGE <= self.budget.max(used + 1)
            && self.grow(host).is_ok()
        {
            return self.take().context("a new block has no page");
        }
        // Evict the least recently touched page not touched this tick.
        let victim = self
            .pages
            .iter()
            .filter(|(_, p)| p.slot.is_some() && p.touched < tick)
            .min_by_key(|(_, p)| p.touched)
            .map(|(k, _)| *k)
            .context("the pool view is out of card memory and every page is in use this tick")?;
        let page = self.pages.get_mut(&victim).expect("victim");
        let slot = page.slot.take().expect("victim has a page");
        binds.extend(page.shown.iter().map(|&t| (t, None)));
        self.stats.evicted += 1;
        self.stats.pages_in_card -= 1;
        Ok(slot)
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
            if let Some(old) = self.bound.remove(&t)
                && let Some(p) = self.pages.get_mut(&old)
            {
                p.shown.retain(|&x| x != t);
                if p.shown.is_empty() {
                    let gone = self.pages.remove(&old).expect("page");
                    if let Some(s) = gone.slot {
                        self.free(s);
                        self.stats.pages_in_card -= 1;
                    }
                }
            }
            let Some((key, source)) = page else {
                binds.push((t, None));
                continue;
            };
            let slot = match self.pages.get(key).and_then(|p| p.slot) {
                Some(s) => s,
                None => {
                    let s = self.slot(host, tick, &mut binds)?;
                    fills.push((*source, s));
                    self.stats.pages_in_card += 1;
                    let entry = self.pages.entry(*key).or_insert(Page {
                        slot: None,
                        shown: Vec::new(),
                        touched: tick,
                    });
                    if !entry.shown.is_empty() {
                        // Evicted, now back: its other title pages too.
                        self.stats.refilled += 1;
                        let others = entry.shown.clone();
                        entry.slot = Some(s);
                        binds.extend(others.iter().map(|&o| (o, Some(s))));
                    }
                    self.pages.get_mut(key).expect("page").slot = Some(s);
                    s
                }
            };
            let p = self.pages.get_mut(key).expect("page");
            p.touched = tick;
            p.shown.push(t);
            self.bound.insert(t, *key);
            binds.push((t, Some(slot)));
        }
        self.flush(host, timeline, fills, binds)
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
        at: u64,
        len: u64,
        tick: u64,
        source: impl Fn(PageKey) -> Option<Source>,
    ) -> Result<()> {
        let lo = at.max(self.title_start);
        let hi = (at + len).min(self.title_start + self.len);
        if lo >= hi {
            return Ok(());
        }
        let mut keys: Vec<PageKey> = ((lo - self.title_start) / PAGE
            ..(hi - self.title_start).div_ceil(PAGE))
            .filter_map(|t| self.bound.get(&t).copied())
            .collect();
        keys.sort_unstable();
        keys.dedup();
        let mut binds = Vec::new();
        let mut fills = Vec::new();
        for key in keys {
            let Some(src) = source(key) else { continue };
            let slot = match self.pages.get(&key).and_then(|p| p.slot) {
                Some(s) => s,
                None => {
                    let s = self.slot(host, tick, &mut binds)?;
                    let p = self.pages.get_mut(&key).expect("bound page");
                    p.slot = Some(s);
                    let shown = p.shown.clone();
                    binds.extend(shown.iter().map(|&t| (t, Some(s))));
                    self.stats.refilled += 1;
                    self.stats.pages_in_card += 1;
                    s
                }
            };
            self.pages.get_mut(&key).expect("bound page").touched = tick;
            fills.push((src, slot));
        }
        self.flush(host, timeline, fills, binds)
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
                d.free_memory(b.memory, None);
            }
        }
    }
}
