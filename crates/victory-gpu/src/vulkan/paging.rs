//! The pool's GPU view as the title pages it (`victory_vk::pool`): page
//! mappings, announced writes, and recordings that name pool memory.
//!
//! The view takes card pages while everything of ours stays under the one
//! card-memory ceiling (`budget.rs`), and evicts its least recently touched
//! pages past it; a map never fails. A recording that names pool memory (a
//! root view, an index or indirect buffer, a copy) brings its evicted pages
//! back before it runs: slow is acceptable, zeros are wrong.

use std::collections::HashMap;

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
    drop(m_guard);
    super::stale::copied(b, to64, pages.len() as u64 * PAGE);
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
    }
    drop(guard);
    super::stale::copied(b, at, len);
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
}
