//! Where each program's code tables lie in the tables buffer
//! (`docs/vulkan-abi.md` section 3.1).
//!
//! A translated program reads the constant tables of its console code from
//! one buffer of the host's, at the buffer's address (the memory map's
//! `tables`) plus the program's place in it, a specialization constant. A
//! specialization constant is part of what the driver's pipeline cache
//! goes by, so a place is given once and kept: a line a place (`key bytes
//! offset`), written beside the translations and read back by the next
//! run. (The buffer's address is another in every process; as the constant
//! it would cost a compile of every such program in every run.)

/// A place starts at a multiple of this (the programs load dwords).
const ALIGN: u32 = 16;

/// The places given so far, in a buffer of `bytes` bytes.
#[derive(Debug, Default)]
pub struct Places {
    bytes: u32,
    /// (key, bytes, offset), in the order given.
    list: Vec<(String, u32, u32)>,
    end: u32,
}

impl Places {
    /// The places earlier runs gave (`lines`, as [`Places::place`] returned
    /// them). A line that does not read, lies outside the buffer or
    /// overlaps one before it is dropped: its tables get a new place.
    pub fn load<'a>(bytes: u32, lines: impl IntoIterator<Item = &'a str>) -> Places {
        let mut p = Places { bytes, ..Places::default() };
        for line in lines {
            let mut f = line.split_whitespace();
            let (Some(key), Some(len), Some(at), None) = (f.next(), f.next(), f.next(), f.next()) else { continue };
            let (Ok(len), Ok(at)) = (len.parse::<u32>(), at.parse::<u32>()) else { continue };
            let Some(end) = at.checked_add(len).filter(|&e| e <= bytes && at % ALIGN == 0) else { continue };
            if p.list.iter().all(|&(_, l, a)| at >= a + l || end <= a) {
                p.list.push((key.to_string(), len, at));
                p.end = p.end.max(end.next_multiple_of(ALIGN));
            }
        }
        p
    }

    /// The place (bytes from the buffer's start) of the `len` bytes of
    /// tables named `key` (a hash of them), and the line to keep when the
    /// place is new. `None`: the buffer is full.
    pub fn place(&mut self, key: &str, len: u32) -> Option<(u32, Option<String>)> {
        if let Some(&(_, _, at)) = self.list.iter().find(|(k, l, _)| k == key && *l == len) {
            return Some((at, None));
        }
        let at = self.end;
        let end = at.checked_add(len).filter(|&e| e <= self.bytes)?;
        self.list.push((key.to_string(), len, at));
        self.end = end.next_multiple_of(ALIGN);
        Some((at, Some(format!("{key} {len} {at}"))))
    }

    /// Bytes of the buffer in use.
    pub fn used(&self) -> u32 {
        self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_place_is_given_once_and_found_again_by_the_next_run() {
        let mut p = Places::load(4096, []);
        let (a, line_a) = p.place("aa", 100).unwrap();
        let (b, line_b) = p.place("bb", 20).unwrap();
        assert_eq!((a, b), (0, 112));
        assert_eq!(p.place("aa", 100), Some((0, None)));
        // the same hash with other bytes is other tables
        let (c, line_c) = p.place("aa", 104).unwrap();
        assert_eq!(c, 144);
        assert_eq!(p.used(), 256);
        // the next run, whatever it asks for first
        let lines = [line_a.unwrap(), line_b.unwrap(), line_c.unwrap()];
        let mut q = Places::load(4096, lines.iter().map(String::as_str));
        assert_eq!(q.place("bb", 20), Some((112, None)));
        assert_eq!(q.place("aa", 104), Some((144, None)));
        assert_eq!(q.place("aa", 100), Some((0, None)));
        assert_eq!(q.place("cc", 8).unwrap().0, 256);
    }

    #[test]
    fn lines_that_cannot_hold_are_dropped_and_a_full_buffer_gives_no_place() {
        // two runs at once gave one place twice; a line past the buffer;
        // one that is no line; one off the alignment
        let lines = ["aa 100 0", "bb 20 96", "cc 64 4090", "dd x 0", "ee 8 200", "ff 16 208 9", "gg 16 4080"];
        let mut p = Places::load(4096, lines);
        assert_eq!(p.place("aa", 100), Some((0, None)));
        assert_eq!(p.place("gg", 16), Some((4080, None)));
        // the dropped ones are placed anew; nothing fits after the last
        assert_eq!(p.used(), 4096);
        assert_eq!(p.place("bb", 20), None);
        let mut q = Places::load(256, ["aa 100 0"]);
        assert_eq!(q.place("bb", 144).unwrap().0, 112);
        assert_eq!(q.place("cc", 1), None);
        assert_eq!(q.place("bb", 144), Some((112, None)));
    }
}
