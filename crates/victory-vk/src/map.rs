//! The memory map of `docs/vulkan-abi.md` section 3 (ABI v1): title
//! address ranges and the device addresses they start at, sorted and
//! disjoint, at most 256, written as the block translated programs read.

use anyhow::{Result, bail, ensure};

/// Ranges a map holds at most.
pub const MAX_RANGES: usize = victory_shader::vk::MAP_RANGES as usize;
/// Bytes of a map block: the header (zero, sink, count, padding, the
/// tables buffer) and 256 ranges of 32 bytes.
pub const BLOCK_BYTES: usize = 32 + MAX_RANGES * 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub title_start: u64,
    pub title_size: u64,
    pub device_base: u64,
}

#[derive(Clone, Default, Debug)]
pub struct MemoryMap {
    ranges: Vec<Range>,
    /// Bumped at every change: a block written for an older generation is
    /// stale.
    pub generation: u64,
}

impl MemoryMap {
    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    /// Adds a range; it must not overlap one there.
    pub fn insert(&mut self, range: Range) -> Result<()> {
        ensure!(range.title_size > 0, "an empty range");
        let at = self
            .ranges
            .partition_point(|r| r.title_start < range.title_start);
        let end = range.title_start + range.title_size;
        if let Some(next) = self.ranges.get(at)
            && next.title_start < end
        {
            bail!(
                "range {:#x}+{:#x} overlaps {:#x}",
                range.title_start,
                range.title_size,
                next.title_start
            );
        }
        if at > 0 {
            let prev = self.ranges[at - 1];
            if prev.title_start + prev.title_size > range.title_start {
                bail!(
                    "range {:#x} overlaps {:#x}",
                    range.title_start,
                    prev.title_start
                );
            }
        }
        ensure!(
            self.ranges.len() < MAX_RANGES,
            "the memory map is full ({MAX_RANGES} ranges)"
        );
        self.ranges.insert(at, range);
        self.generation += 1;
        Ok(())
    }

    /// `ranges` sorted, with neighbours and overlaps that map title
    /// addresses to the device the same way (one device offset) joined:
    /// views of one section side by side, an import and the larger one
    /// that replaced it.
    pub fn coalesce(mut ranges: Vec<Range>) -> Vec<Range> {
        ranges.sort_by_key(|r| (r.title_start, std::cmp::Reverse(r.title_size)));
        let mut out: Vec<Range> = Vec::with_capacity(ranges.len());
        for r in ranges {
            if let Some(last) = out.last_mut() {
                let last_end = last.title_start + last.title_size;
                let same_offset = r.device_base.wrapping_sub(r.title_start) == last.device_base.wrapping_sub(last.title_start);
                if r.title_start <= last_end && same_offset {
                    last.title_size = last.title_size.max(r.title_start + r.title_size - last.title_start);
                    continue;
                }
            }
            out.push(r);
        }
        out
    }

    /// Shows `size` bytes from `title_start` on at `device_base` instead:
    /// the range that holds them is cut around them (into up to three),
    /// so they must lie in one range, and the map must have room for the
    /// pieces.
    pub fn cut(&mut self, title_start: u64, size: u64, device_base: u64) -> Result<()> {
        ensure!(size > 0, "an empty cut");
        let end = title_start + size;
        let whole = self.range_of(title_start).filter(|r| end <= r.title_start + r.title_size);
        let Some(whole) = whole else { bail!("no range holds {title_start:#x}+{size:#x}") };
        let whole_end = whole.title_start + whole.title_size;
        let pieces = 1 + usize::from(whole.title_start < title_start) + usize::from(end < whole_end);
        ensure!(self.ranges.len() - 1 + pieces <= MAX_RANGES, "the memory map is full ({MAX_RANGES} ranges)");
        self.remove(whole.title_start);
        if whole.title_start < title_start {
            self.insert(Range { title_size: title_start - whole.title_start, ..whole })?;
        }
        self.insert(Range { title_start, title_size: size, device_base })?;
        if end < whole_end {
            self.insert(Range { title_start: end, title_size: whole_end - end, device_base: whole.device_base + (end - whole.title_start) })?;
        }
        Ok(())
    }

    /// Removes the range that starts at `title_start`.
    pub fn remove(&mut self, title_start: u64) -> Option<Range> {
        let at = self
            .ranges
            .iter()
            .position(|r| r.title_start == title_start)?;
        self.generation += 1;
        Some(self.ranges.remove(at))
    }

    /// The range holding `title`.
    pub fn range_of(&self, title: u64) -> Option<Range> {
        let at = self.ranges.partition_point(|r| r.title_start <= title);
        let r = *self.ranges.get(at.checked_sub(1)?)?;
        (title - r.title_start < r.title_size).then_some(r)
    }

    /// The device address of `title`, as a translated program computes it.
    pub fn device(&self, title: u64) -> Option<u64> {
        self.range_of(title)
            .map(|r| r.device_base + (title - r.title_start))
    }

    /// The block programs read (ABI v1), little-endian. `tables`: the
    /// device address of the buffer that holds the programs' code tables
    /// (section 3.1; read only by programs that have tables).
    pub fn block(&self, zero: u64, sink: u64, tables: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(BLOCK_BYTES);
        out.extend_from_slice(&zero.to_le_bytes());
        out.extend_from_slice(&sink.to_le_bytes());
        out.extend_from_slice(&(self.ranges.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&tables.to_le_bytes());
        for r in &self.ranges {
            out.extend_from_slice(&r.title_start.to_le_bytes());
            out.extend_from_slice(&r.title_size.to_le_bytes());
            out.extend_from_slice(&r.device_base.to_le_bytes());
            out.extend_from_slice(&[0; 8]);
        }
        out.resize(BLOCK_BYTES, 0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(start: u64, size: u64, base: u64) -> Range {
        Range {
            title_start: start,
            title_size: size,
            device_base: base,
        }
    }

    #[test]
    fn sorted_lookup_and_overlap() {
        let mut m = MemoryMap::default();
        m.insert(r(0x3000, 0x1000, 0x10_0000)).unwrap();
        m.insert(r(0x1000, 0x1000, 0x20_0000)).unwrap();
        assert!(m.insert(r(0x1800, 0x100, 0)).is_err());
        assert!(m.insert(r(0x0, 0x1001, 0)).is_err());
        assert_eq!(m.ranges()[0].title_start, 0x1000);
        assert_eq!(m.device(0x1004), Some(0x20_0004));
        assert_eq!(m.device(0x3fff), Some(0x10_0fff));
        assert_eq!(m.device(0x2000), None);
        assert_eq!(m.device(0x4000), None);
        assert_eq!(m.device(0x10), None);
        m.remove(0x1000).unwrap();
        assert_eq!(m.device(0x1004), None);
    }

    #[test]
    fn cut_inside_at_an_edge_and_refused() {
        let mut m = MemoryMap::default();
        m.insert(r(0x1000, 0x3000, 0x10_0000)).unwrap();
        m.cut(0x2000, 0x100, 0x50_0000).unwrap();
        assert_eq!(m.ranges(), [r(0x1000, 0x1000, 0x10_0000), r(0x2000, 0x100, 0x50_0000), r(0x2100, 0x1f00, 0x10_1100)]);
        assert_eq!(m.device(0x1fff), Some(0x10_0fff));
        assert_eq!(m.device(0x20ff), Some(0x50_00ff));
        assert_eq!(m.device(0x2100), Some(0x10_1100));
        // at a range's start: two pieces
        m.cut(0x1000, 0x10, 0x60_0000).unwrap();
        assert_eq!(m.ranges().len(), 4);
        assert_eq!(m.device(0x1010), Some(0x10_0010));
        // over two ranges, and in none
        assert!(m.cut(0x1ff0, 0x20, 0).is_err());
        assert!(m.cut(0x8000, 0x10, 0).is_err());
        assert_eq!(m.ranges().len(), 4);
        // no room for the pieces: nothing changes
        let mut full = MemoryMap::default();
        for k in 0..MAX_RANGES as u64 {
            full.insert(r(k << 16, 0x1000, k << 32)).unwrap();
        }
        assert!(full.cut(0x10, 0x10, 0).is_err());
        assert_eq!(full.device(0x10), Some(0x10));
        full.cut(0, 0x1000, 0x70_0000).unwrap();
        assert_eq!(full.device(0x10), Some(0x70_0010));
    }

    /// More ranges than the six search steps of the first maps reached
    /// (64): the first, the middle and the last resolve, and what lies
    /// between ranges does not.
    #[test]
    fn a_full_map_resolves_first_middle_and_last() {
        let mut m = MemoryMap::default();
        for k in 0..MAX_RANGES as u64 {
            m.insert(r(0x1_0000 + (k << 16), 0x8000, 0x9_0000_0000 + (k << 20))).unwrap();
        }
        assert!(m.insert(r(0, 0x100, 0)).is_err());
        for k in [0, 63, 64, 65, MAX_RANGES as u64 / 2, MAX_RANGES as u64 - 1] {
            let start = 0x1_0000 + (k << 16);
            assert_eq!(m.device(start), Some(0x9_0000_0000 + (k << 20)));
            assert_eq!(m.device(start + 0x7fff), Some(0x9_0000_0000 + (k << 20) + 0x7fff));
            assert_eq!(m.device(start + 0x8000), None);
        }
        let b = m.block(0, 0, 0);
        assert_eq!(b.len(), BLOCK_BYTES);
        assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), MAX_RANGES as u32);
        let last = 32 + 32 * (MAX_RANGES - 1);
        assert_eq!(u64::from_le_bytes(b[last..last + 8].try_into().unwrap()), 0x1_0000 + ((MAX_RANGES as u64 - 1) << 16));
    }

    #[test]
    fn block_layout() {
        let mut m = MemoryMap::default();
        m.insert(r(0x1000, 0x2000, 0xabc0_0000)).unwrap();
        let b = m.block(0x11, 0x22, 0x33);
        assert_eq!(b.len(), BLOCK_BYTES);
        assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), 0x11);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 0x22);
        assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(b[24..32].try_into().unwrap()), 0x33);
        assert_eq!(u64::from_le_bytes(b[32..40].try_into().unwrap()), 0x1000);
        assert_eq!(u64::from_le_bytes(b[40..48].try_into().unwrap()), 0x2000);
        assert_eq!(
            u64::from_le_bytes(b[48..56].try_into().unwrap()),
            0xabc0_0000
        );
    }
}
