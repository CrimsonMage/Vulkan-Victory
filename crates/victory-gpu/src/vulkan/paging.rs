//! The pool's GPU view as the title pages it (`victory_vk::pool`): page
//! mappings, announced writes, and recordings that name pool memory.
//!
//! The view takes card pages while everything of ours stays under the one
//! card-memory ceiling (`budget.rs`), and evicts its least recently touched
//! pages past it; a map never fails. A recording that names pool memory (a
//! root view, an index or indirect buffer, a copy) brings its evicted pages
//! back before it runs: slow is acceptable, zeros are wrong.
//!
//! A page on the card is a copy of its home, so its CPU views are
//! write-protected before the copy is made (`crate::track`): at a mapping
//! here, and for a page brought back under the memory lock at the next
//! flush ([`flush_written`]), which copies it once more.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_vk::pool::{PAGE, PageKey, PoolView, Source};

use super::{Memory, Section, backend, budget, images, memory};

/// A page of the GPU view of the pool: the section, the byte offset in it,
/// and where the CPU sees it (the section's whole view, `xmem`).
#[derive(Clone, Copy, Debug)]
pub struct GpuPage {
    pub section: usize,
    pub offset: u64,
    pub cpu: usize,
}

/// Where a physical page's bytes are: its section's import.
fn source_in(sections: &HashMap<usize, Section>, key: PageKey) -> Option<Source> {
    let import = sections.get(&key.0)?.import.as_ref()?;
    (key.1.is_multiple_of(PAGE) && key.1 + PAGE <= import.len).then_some(Source {
        buffer: import.buffer,
        offset: key.1,
    })
}

/// The title had its GPU map pool pages at `[to, to + 64 KiB * pages.len())`
/// (`CopyPageMappingsBatchX`; `None`: unmapped there).
pub fn title_pages_aliased(to: usize, pages: &[Option<GpuPage>]) {
    let Some(b) = backend().filter(|b| !b.timeline.lost()) else {
        return;
    };
    let to64 = to as u64;
    // Images over pages that now show other memory (or none) go (before
    // the memory lock): the title streams by mapping fresh pages under new
    // textures and taking old ones away, so images follow its live ones.
    images::released(b, to64, pages.len() as u64 * PAGE, images::Dropped::Remapped);
    // Protect, then copy: a store after this is seen at the next flush.
    crate::track::protect(to64, pages.len() as u64 * PAGE, false);
    let mut m_guard = memory(b);
    let m = &mut *m_guard;
    if !m.pools.iter().any(|p| p.contains(to64)) && !new_view(b, m, to) {
        return;
    }
    let entries = m.page_sources(b, to, pages);
    let tick = m.tick;
    if let Some(view) = m.pools.iter_mut().find(|p| p.contains(to64))
        && let Err(e) = view.map(&b.host, &b.timeline, to64, &entries, tick)
    {
        tracing::error!(error = %format!("{e:#}"), "Vulkan: pages of the pool's GPU view were not mapped");
    }
}

/// The pool's GPU view over the page reservation holding `to`, made now.
fn new_view(b: &super::Backend, m: &mut Memory, to: usize) -> bool {
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
        return false;
    };
    let ceiling = budget::limit(&b.host);
    match PoolView::new(&b.host, &b.timeline, at as u64, len as u64, ceiling) {
        Ok(view) => {
            tracing::info!(
                at = format_args!("{at:#x}"),
                mb = len >> 20,
                device = format_args!("{:#x}", view.device_base),
                ceiling_mb = ceiling >> 20,
                unbound_pages_read_zeros = b.host.caps.non_resident_strict,
                "Vulkan: the pool's GPU view is a sparse buffer of card pages"
            );
            m.pools.push(view);
            m.rebuild_map();
            true
        }
        Err(e) => {
            tracing::error!(error = %format!("{e:#}"), "Vulkan: no sparse buffer for the pool's GPU view");
            false
        }
    }
}

impl Memory {
    /// Each mapped page's physical page and home (sections imported the
    /// first time; the memory map follows).
    fn page_sources(
        &mut self,
        b: &super::Backend,
        to: usize,
        pages: &[Option<GpuPage>],
    ) -> Vec<Option<(PageKey, Source)>> {
        let mut entries = Vec::with_capacity(pages.len());
        let mut imported = false;
        for page in pages {
            entries.push(page.and_then(|g| {
                let s = self.sections.get_mut(&g.section)?;
                if s.whole == 0 {
                    s.whole = g.cpu - g.offset as usize;
                }
                let had = s.import.is_some();
                let key = (g.section, g.offset);
                let src = self.source(&b.host, key);
                imported |= !had && src.is_some();
                src.map(|s| (key, s))
            }));
        }
        if imported {
            self.rebuild_map();
        }
        let lost = pages
            .iter()
            .zip(&entries)
            .filter(|(p, e)| p.is_some() && e.is_none())
            .count();
        if lost > 0 && self.warned < 8 {
            self.warned += 1;
            tracing::warn!(
                lost,
                to = format_args!("{to:#x}"),
                "Vulkan: pages of an unknown or unimported section show zeros in the pool's GPU view"
            );
        }
        entries
    }
}

pub(crate) fn title_memory_written(at: usize, len: usize) {
    let Some(b) = backend().filter(|b| !b.timeline.lost()) else {
        return;
    };
    // Card images over the memory are filled again.
    images::written(at as u64, len as u64);
    let mut guard = memory(b);
    let Memory {
        pools,
        sections,
        tick,
        ..
    } = &mut *guard;
    let (at, len) = (at as u64, len as u64);
    for view in pools
        .iter_mut()
        .filter(|p| at < p.title_start + p.len && at + len > p.title_start)
    {
        // The sections are imported already (their pages were shown).
        let r = view.written(&b.host, &b.timeline, (at, len), *tick, |key| {
            source_in(sections, key)
        });
        if let Err(e) = r {
            tracing::error!(error = %format!("{e:#}"), "Vulkan: an announced write was not taken into the pool's GPU view");
        }
        came_back(view);
    }
}

/// Pages brought back under the memory lock were copied with their CPU
/// views open: the next flush protects them and copies them again.
fn came_back(view: &mut PoolView) {
    for at in view.back.drain(..) {
        crate::track::copied(at as usize);
    }
}

/// Before work is given to the card (a submission, a present:
/// `frame_end`): the pages the CPU stored into since the last flush are
/// protected again (`crate::track`), then their card pages are copied
/// again and the images over them queued to be filled again.
pub(super) fn flush_written(b: &super::Backend, frame_end: bool) {
    use crate::track::{COPIED, COUNT, FLUSH_NS, FLUSHES, Held, IMAGES};
    if b.timeline.lost() {
        return;
    }
    let start = std::time::Instant::now();
    let copy = crate::track::flush(frame_end, &mut |shown, fill| {
        let mut held: Vec<Held> = {
            let m = memory(b);
            let on_card = |at: usize| m.pools.iter().any(|p| p.resident(at as u64));
            shown.iter().map(|s| Held { pool: s.places.iter().any(|p| on_card(p.0)), image: false }).collect()
        };
        for ((s, h), _) in shown.iter().zip(&mut held).zip(fill).filter(|(_, fill)| **fill) {
            for &(at, _) in &s.places {
                h.image |= images::written(at as u64, PAGE);
            }
            COUNT[IMAGES].fetch_add(u64::from(h.image), Ordering::Relaxed);
        }
        held
    });
    if !copy.is_empty() {
        let copy: Vec<u64> = copy.into_iter().map(|at| at as u64).collect();
        let mut guard = memory(b);
        let Memory { pools, sections, .. } = &mut *guard;
        for view in pools.iter_mut() {
            match view.stored(&b.host, &b.timeline, &copy, |key| source_in(sections, key)) {
                Ok(n) => {
                    COUNT[COPIED].fetch_add(n, Ordering::Relaxed);
                }
                Err(e) => tracing::error!(error = %format!("{e:#}"), "Vulkan: pages the CPU wrote were not copied into the pool's GPU view again"),
            }
        }
    }
    COUNT[FLUSHES].fetch_add(1, Ordering::Relaxed);
    COUNT[FLUSH_NS].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// The 10 s line of write tracking, per frame since the last one: faults
/// taken, pages copied again, pages left open as hot, and what it cost.
pub(super) fn report_written(frames: u64) {
    use crate::track::{COPIED, COUNT, FAULTS, FLUSH_NS, FLUSHES, HOT_NOW, IMAGE_FAULTS, IMAGES};
    static LAST: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
    let since = |i: usize| {
        let now = COUNT[i].load(Ordering::Relaxed);
        now - LAST[i].swap(now, Ordering::Relaxed)
    };
    let per = |n: u64| n as f64 / frames as f64;
    let (faults, flush_ns) = (since(FAULTS), since(FLUSH_NS));
    // What a fault costs now (the store into a protected page of our own,
    // through the system's exception dispatch), measured with each line.
    let cost = cfg!(feature = "diagnostics").then(crate::track::fault_cost).flatten();
    let (fault_ns, change_ns) = cost.unwrap_or((0, 0));
    tracing::info!(
        faults = COUNT[FAULTS].load(Ordering::Relaxed),
        faults_a_frame = format_args!("{:.1}", per(faults)),
        under_images_a_frame = format_args!("{:.1}", per(since(IMAGE_FAULTS))),
        copied_a_frame = format_args!("{:.1}", per(since(COPIED))),
        image_pages_a_frame = format_args!("{:.2}", per(since(IMAGES))),
        hot = HOT_NOW.load(Ordering::Relaxed),
        flushes_a_frame = format_args!("{:.1}", per(since(FLUSHES))),
        flush_ms_a_frame = format_args!("{:.2}", per(flush_ns) / 1e6),
        fault_ms_a_frame = format_args!("{:.2}", per(faults * fault_ns) / 1e6),
        fault_us = format_args!("{:.1}", fault_ns as f64 / 1e3),
        protect_us = format_args!("{:.1}", change_ns as f64 / 1e3),
        "Vulkan: write tracking (CPU stores into pages the card has copies of)"
    );
}

/// A recording names `[at, at + len)`: if it is pool memory, its evicted
/// pages come back before the recording runs.
pub(super) fn ensure(b: &super::Backend, m: &mut Memory, at: u64, len: u64) {
    let Memory {
        pools,
        sections,
        tick,
        ..
    } = m;
    let Some(view) = pools.iter_mut().find(|p| p.contains(at)) else {
        return;
    };
    let r = view.ensure(&b.host, &b.timeline, (at, len.max(1)), *tick, |key| {
        source_in(sections, key)
    });
    if let Err(e) = r {
        tracing::error!(error = %format!("{e:#}"), "Vulkan: evicted pages of the pool's GPU view were not brought back");
    }
    came_back(view);
}
