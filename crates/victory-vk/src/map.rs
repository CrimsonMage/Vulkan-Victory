//! The memory map of `docs/vulkan-abi.md` section 3 (ABI v1): title
//! address ranges and the device addresses they start at, sorted and
//! disjoint, at most 64, written as the block translated programs read.

use anyhow::{Result, bail, ensure};

/// Ranges a map holds at most.
pub const MAX_RANGES: usize = 64;
/// Bytes of a map block: the header (zero, sink, count, padding) and 64
/// ranges of 32 bytes.
pub const BLOCK_BYTES: usize = 32 + MAX_RANGES * 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub title_start: u64,
    pub title_size: u64,
    pub device_base: u64,
}

#[derive(Default, Debug)]
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

    /// The block programs read (ABI v1), little-endian.
    pub fn block(&self, zero: u64, sink: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(BLOCK_BYTES);
        out.extend_from_slice(&zero.to_le_bytes());
        out.extend_from_slice(&sink.to_le_bytes());
        out.extend_from_slice(&(self.ranges.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0; 12]);
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
    fn block_layout() {
        let mut m = MemoryMap::default();
        m.insert(r(0x1000, 0x2000, 0xabc0_0000)).unwrap();
        let b = m.block(0x11, 0x22);
        assert_eq!(b.len(), BLOCK_BYTES);
        assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), 0x11);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 0x22);
        assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(b[32..40].try_into().unwrap()), 0x1000);
        assert_eq!(u64::from_le_bytes(b[40..48].try_into().unwrap()), 0x2000);
        assert_eq!(
            u64::from_le_bytes(b[48..56].try_into().unwrap()),
            0xabc0_0000
        );
    }
}
