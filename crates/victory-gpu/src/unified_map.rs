//! The unified-memory map (`docs/unified-memory.md`, stage 1): where each
//! title address lives on the host GPU side, and which textures a write
//! the title announces (DirectStorage, a page mapping, a GPU copy) makes
//! stale. Pure bookkeeping: no host calls, so it is tested on any host.
//!
//! A title address resolves to a heap opened on the title's memory (one
//! per `XMemVirtualAlloc` region, one per physical-page section) and an
//! offset in it. Physical pages shown at several addresses (aliases) are
//! several spans naming the same heap offset; nothing is copied between
//! them because they are the same memory. A range resolves to one home
//! only if it is contiguous in one heap; a range over pages scattered in
//! a section (the title's tiled heap) is [`Resolved::Scattered`].
//!
//! Not wired yet (stage 1 wires it); kept compiled so it stays honest.
#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};

/// A heap the runtime opened on title memory, by the runtime's number.
pub(crate) type HeapId = u32;

/// Where title bytes live: a heap and the byte offset in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Home {
    pub(crate) heap: HeapId,
    pub(crate) offset: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Resolved {
    /// The whole range is contiguous in one heap from this home on.
    Home(Home),
    /// Every byte has a home, but not one contiguous run of one heap.
    Scattered,
    /// Some byte is not title memory the map knows.
    Unknown,
}

/// Title address spans and their homes.
#[derive(Default)]
pub(crate) struct Map {
    /// By first title address: length, heap, offset of the first byte.
    spans: BTreeMap<u64, (u64, HeapId, u64)>,
}

impl Map {
    /// `[at, at + len)` now shows heap `heap` from `offset` on (an
    /// allocation opened, or a view of a section mapped there).
    pub(crate) fn insert(&mut self, at: u64, len: u64, heap: HeapId, offset: u64) {
        if len == 0 {
            return;
        }
        self.remove(at, len);
        self.spans.insert(at, (len, heap, offset));
    }

    /// `[at, at + len)` shows nothing any more (unmapped, freed); spans
    /// reaching past it keep their outside parts.
    pub(crate) fn remove(&mut self, at: u64, len: u64) {
        let end = at.saturating_add(len);
        if len == 0 {
            return;
        }
        let first = self.spans.range(..at).next_back().filter(|(s, (l, ..))| *s + l > at).map(|(s, _)| *s);
        let hit: Vec<u64> = first.into_iter().chain(self.spans.range(at..end).map(|(s, _)| *s)).collect();
        for s in hit {
            let Some((l, heap, offset)) = self.spans.remove(&s) else { continue };
            if s < at {
                self.spans.insert(s, (at - s, heap, offset));
            }
            if s + l > end {
                self.spans.insert(end, (s + l - end, heap, offset + (end - s)));
            }
        }
    }

    /// Where `[at, at + len)` lives (`len` 0 is taken as 1).
    pub(crate) fn resolve(&self, at: u64, len: u64) -> Resolved {
        let end = at.saturating_add(len.max(1));
        let Some((&s, &(l, heap, offset))) = self.spans.range(..=at).next_back() else {
            return Resolved::Unknown;
        };
        if s + l <= at {
            return Resolved::Unknown;
        }
        let home = Home { heap, offset: offset + (at - s) };
        // Walk the spans that follow while they continue both the address
        // and the heap offset.
        let (mut reach, mut next_offset, mut contiguous) = (s + l, offset + l, true);
        while reach < end {
            match self.spans.get(&reach) {
                Some(&(l2, h2, o2)) => {
                    contiguous &= h2 == heap && o2 == next_offset;
                    reach += l2;
                    next_offset = o2 + l2;
                }
                None => return Resolved::Unknown,
            }
        }
        if contiguous { Resolved::Home(home) } else { Resolved::Scattered }
    }

    /// Every title address that shows heap `heap`'s byte `offset` (the
    /// aliases of a physical page), in address order.
    pub(crate) fn shown_at(&self, heap: HeapId, offset: u64) -> Vec<u64> {
        self.spans
            .iter()
            .filter(|(_, (l, h, o))| *h == heap && (*o..*o + *l).contains(&offset))
            .map(|(s, (_, _, o))| s + (offset - o))
            .collect()
    }
}

/// Host textures made from title memory, and which of them a write made
/// stale: the only "detection" left, fed by the title's explicit events.
#[derive(Default)]
pub(crate) struct Stale {
    /// By first title address and texture: the end address.
    textures: BTreeMap<(u64, u64), u64>,
    /// The longest texture span, to bound the overlap search.
    longest: u64,
    stale: BTreeSet<u64>,
}

impl Stale {
    /// Texture `id` (a host copy) is made from `[at, at + len)`; it starts
    /// stale (it has never been detiled).
    pub(crate) fn register(&mut self, id: u64, at: u64, len: u64) {
        self.textures.insert((at, id), at.saturating_add(len));
        self.longest = self.longest.max(len);
        self.stale.insert(id);
    }

    pub(crate) fn unregister(&mut self, id: u64, at: u64) {
        self.textures.remove(&(at, id));
        self.stale.remove(&id);
    }

    /// `[at, at + len)` was written by an announced writer: every texture
    /// over it is stale. Returns how many were marked.
    pub(crate) fn written(&mut self, at: u64, len: u64) -> usize {
        let end = at.saturating_add(len);
        let from = at.saturating_sub(self.longest);
        let hit: Vec<u64> = self
            .textures
            .range((from, 0)..(end, 0))
            .filter(|&(&(start, _), &stop)| start < end && stop > at)
            .map(|(&(_, id), _)| id)
            .collect();
        let n = hit.len();
        self.stale.extend(hit);
        n
    }

    /// Whether texture `id` must be detiled again before it is sampled;
    /// the answer clears it.
    pub(crate) fn take(&mut self, id: u64) -> bool {
        self.stale.remove(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 64 << 10;

    #[test]
    fn private_memory_resolves_at_its_offset() {
        let mut m = Map::default();
        m.insert(0x10_0000_0000, 64 * P, 1, 0);
        assert_eq!(m.resolve(0x10_0000_0000 + 3 * P + 16, 256), Resolved::Home(Home { heap: 1, offset: 3 * P + 16 }));
        assert_eq!(m.resolve(0x10_0000_0000 + 63 * P, 2 * P), Resolved::Unknown);
        assert_eq!(m.resolve(0x0f_ffff_0000, 16), Resolved::Unknown);
    }

    #[test]
    fn aliases_of_a_page_share_one_home() {
        let mut m = Map::default();
        // Section 7's pages 4..8 at a pool view and again in the tiled heap.
        m.insert(0x20_0000_0000, 4 * P, 7, 4 * P);
        m.insert(0x30_0000_0000, 4 * P, 7, 4 * P);
        let a = m.resolve(0x20_0000_0000 + P, 100);
        assert_eq!(a, m.resolve(0x30_0000_0000 + P, 100));
        assert_eq!(m.shown_at(7, 5 * P + 8), vec![0x20_0000_0000 + P + 8, 0x30_0000_0000 + P + 8]);
    }

    #[test]
    fn views_that_continue_each_other_are_one_run_and_scattered_pages_are_not() {
        let mut m = Map::default();
        let t = 0x40_0000_0000;
        // Page by page, as CopyPageMappingsBatchX maps them.
        m.insert(t, P, 3, 10 * P);
        m.insert(t + P, P, 3, 11 * P);
        m.insert(t + 2 * P, P, 3, 40 * P);
        assert_eq!(m.resolve(t + 100, P), Resolved::Home(Home { heap: 3, offset: 10 * P + 100 }));
        assert_eq!(m.resolve(t + P, 2 * P), Resolved::Scattered);
    }

    #[test]
    fn remapping_cuts_spans_and_keeps_the_rest() {
        let mut m = Map::default();
        let t = 0x50_0000_0000;
        m.insert(t, 8 * P, 2, 0);
        m.insert(t + 2 * P, P, 9, 5 * P);
        assert_eq!(m.resolve(t + P, 10), Resolved::Home(Home { heap: 2, offset: P }));
        assert_eq!(m.resolve(t + 2 * P, 10), Resolved::Home(Home { heap: 9, offset: 5 * P }));
        assert_eq!(m.resolve(t + 3 * P, 10), Resolved::Home(Home { heap: 2, offset: 3 * P }));
        m.remove(t + 2 * P, P);
        assert_eq!(m.resolve(t + 2 * P, 10), Resolved::Unknown);
        assert_eq!(m.resolve(t, 2 * P), Resolved::Home(Home { heap: 2, offset: 0 }));
    }

    #[test]
    fn announced_writes_make_the_textures_over_them_stale() {
        let mut s = Stale::default();
        s.register(1, 0x1000_0000, 4 * P);
        s.register(2, 0x1000_0000 + 4 * P, P);
        s.register(3, 0x2000_0000, P);
        assert!(s.take(1) && s.take(2) && s.take(3));
        assert!(!s.take(1));
        assert_eq!(s.written(0x1000_0000 + 3 * P, 2 * P), 2);
        assert!(s.take(1) && s.take(2) && !s.take(3));
        assert_eq!(s.written(0x1000_0000 + 5 * P, P), 0);
        s.unregister(3, 0x2000_0000);
        assert_eq!(s.written(0x2000_0000, P), 0);
    }
}
