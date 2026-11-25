//! Announced writes of title memory (`docs/unified-memory.md`, section 6):
//! which ranges DirectStorage, a page mapping, a title copy or a GPU write
//! changed, and when. A host texture asks for the newest write over its
//! memory and detiles again only if it is newer than its last detiling.
//! Pure bookkeeping, tested on any host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::BTreeMap;

/// Disjoint written ranges, each with the number of its newest write.
/// Touching ranges are joined under the newer number: a texture over the
/// older part may detile once more than needed (its content hash then
/// finds nothing new), never once less.
pub(crate) struct Epochs {
    /// By first address: end address, newest write.
    ranges: BTreeMap<u64, (u64, u64)>,
    now: u64,
}

impl Epochs {
    pub(crate) const fn new() -> Epochs {
        Epochs { ranges: BTreeMap::new(), now: 0 }
    }

    /// `[at, at + len)` was written; returns the write's number.
    pub(crate) fn write(&mut self, at: u64, len: u64) -> u64 {
        self.now += 1;
        if len == 0 {
            return self.now;
        }
        let (mut start, mut end) = (at, at.saturating_add(len));
        // Every range that overlaps or touches goes into the new one.
        let first = self.ranges.range(..start).next_back().filter(|(_, (e, _))| *e >= start).map(|(s, _)| *s);
        let hit: Vec<u64> = first.into_iter().chain(self.ranges.range(start..=end).map(|(s, _)| *s)).collect();
        for s in hit {
            if let Some((e, _)) = self.ranges.remove(&s) {
                start = start.min(s);
                end = end.max(e);
            }
        }
        self.ranges.insert(start, (end, self.now));
        self.now
    }

    /// The newest write over `[at, at + len)`, 0 if none ever was.
    pub(crate) fn newest(&self, at: u64, len: u64) -> u64 {
        let end = at.saturating_add(len.max(1));
        let first = self.ranges.range(..at).next_back().filter(|(_, (e, _))| *e > at).map(|(_, (_, n))| *n);
        first.into_iter().chain(self.ranges.range(at..end).map(|(_, (_, n))| *n)).max().unwrap_or(0)
    }

    /// The number of the newest write anywhere.
    #[cfg(test)]
    pub(crate) fn now(&self) -> u64 {
        self.now
    }

    /// Ranges kept (for the log line).
    pub(crate) fn len(&self) -> usize {
        self.ranges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 64 << 10;

    #[test]
    fn a_write_is_seen_only_over_its_range() {
        let mut e = Epochs::new();
        let w = e.write(10 * P, 2 * P);
        assert_eq!(e.newest(11 * P, 16), w);
        assert_eq!(e.newest(9 * P, P), 0);
        assert_eq!(e.newest(12 * P, P), 0);
        assert_eq!(e.newest(9 * P, 2 * P), w);
    }

    #[test]
    fn touching_writes_join_under_the_newer_number() {
        let mut e = Epochs::new();
        e.write(0, P);
        let b = e.write(P, P);
        assert_eq!(e.len(), 1);
        assert_eq!(e.newest(0, 16), b);
        let c = e.write(10 * P, P);
        assert_eq!(e.len(), 2);
        assert_eq!(e.newest(0, 11 * P), c);
        assert_eq!(e.newest(5 * P, P), 0);
    }

    #[test]
    fn a_write_over_several_ranges_swallows_them() {
        let mut e = Epochs::new();
        e.write(0, P);
        e.write(4 * P, P);
        e.write(8 * P, P);
        let d = e.write(P / 2, 8 * P);
        assert_eq!(e.len(), 1);
        assert_eq!(e.newest(8 * P + 100, 4), d);
        assert_eq!(e.now(), d);
    }
}
