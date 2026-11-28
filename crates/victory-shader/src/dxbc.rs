//! The DXBC container: a header, then a table of FourCC-tagged parts.
//!
//! ```text
//! 0x00  "DXBC"
//! 0x04  digest[16]
//! 0x14  version major u16, minor u16
//! 0x18  total size u32
//! 0x1C  part count u32
//! 0x20  part offsets u32[count]
//! part: fourcc[4], size u32, data[size]
//! ```

use crate::ShaderError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const DXIL: FourCC = FourCC(*b"DXIL");
    /// Console code of another compiler, in place of `DXIL` (`native/fspx.rs`).
    pub const ACCC: FourCC = FourCC(*b"ACCC");
    pub const STAT: FourCC = FourCC(*b"STAT");
    pub const RTS0: FourCC = FourCC(*b"RTS0");
    pub const PSV0: FourCC = FourCC(*b"PSV0");
    pub const ISG1: FourCC = FourCC(*b"ISG1");
    pub const OSG1: FourCC = FourCC(*b"OSG1");
    /// Per-primitive outputs of a mesh shader.
    pub const PSG1: FourCC = FourCC(*b"PSG1");
    pub const SFI0: FourCC = FourCC(*b"SFI0");
    pub const HASH: FourCC = FourCC(*b"HASH");
    pub const ILDB: FourCC = FourCC(*b"ILDB");
    pub const ILDN: FourCC = FourCC(*b"ILDN");
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &b in &self.0 {
            let c = if b.is_ascii_graphic() { b as char } else { '.' };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub fourcc: FourCC,
    /// Offset of the part header within the container.
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Container<'a> {
    pub digest: [u8; 16],
    pub version: (u16, u16),
    pub parts: Vec<Part<'a>>,
}

const HEADER_SIZE: usize = 0x20;
const MAX_PARTS: usize = 256;

impl<'a> Container<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < HEADER_SIZE || &data[0..4] != b"DXBC" {
            return Err(ShaderError::NotDxbc);
        }
        let digest = data[4..20].try_into().expect("16 bytes");
        let version = (u16_at(data, 0x14)?, u16_at(data, 0x16)?);
        let total = u32_at(data, 0x18)? as usize;
        if total > data.len() || total < HEADER_SIZE {
            return Err(ShaderError::Malformed("container size disagrees with data"));
        }
        let data = &data[..total];
        let count = u32_at(data, 0x1C)? as usize;
        if count > MAX_PARTS {
            return Err(ShaderError::Malformed("implausible part count"));
        }
        let mut parts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = u32_at(data, HEADER_SIZE + 4 * i)? as usize;
            let fourcc = FourCC(
                data.get(offset..offset + 4)
                    .ok_or(ShaderError::Malformed("part header out of bounds"))?
                    .try_into()
                    .expect("4 bytes"),
            );
            let size = u32_at(data, offset + 4)? as usize;
            let body = data
                .get(offset + 8..offset + 8 + size)
                .ok_or(ShaderError::Malformed("part data out of bounds"))?;
            parts.push(Part {
                fourcc,
                offset,
                data: body,
            });
        }
        Ok(Container {
            digest,
            version,
            parts,
        })
    }

    pub fn part(&self, fourcc: FourCC) -> Option<&Part<'a>> {
        self.parts.iter().find(|p| p.fourcc == fourcc)
    }

    /// The entry function's name, where the `PSV0` part gives it (runtime
    /// information of version 3: the name's place in the part's string
    /// table is its last dword; the table follows the resource bindings).
    pub fn entry_name(&self) -> Option<&'a str> {
        const INFO3: usize = 52;
        let d = self.part(FourCC::PSV0)?.data;
        let info = u32_at(d, 0).ok()? as usize;
        if info < INFO3 {
            return None;
        }
        let name_at = u32_at(d, 4 + INFO3 - 4).ok()? as usize;
        let mut at = 4 + info;
        let resources = u32_at(d, at).ok()? as usize;
        at += 4;
        if resources > 0 {
            at += 4 + resources.checked_mul(u32_at(d, at).ok()? as usize)?;
        }
        let table = d.get(at + 4..)?.get(..u32_at(d, at).ok()? as usize)?;
        let name = table.get(name_at..)?.split(|&b| b == 0).next()?;
        std::str::from_utf8(name).ok().filter(|n| !n.is_empty())
    }

    /// The part that holds the console programs: `DXIL`, or `ACCC` in a
    /// container without one.
    pub fn code_part(&self) -> Option<&Part<'a>> {
        self.part(FourCC::DXIL).or(self.part(FourCC::ACCC))
    }
}

/// The containers stored as they are in `data` (a file of any kind, such
/// as a DLL that carries its shaders), found by the magic: each one that
/// parses, cut to the size it states.
pub fn containers(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = data[from..].windows(4).position(|w| w == b"DXBC").map(|i| from + i) {
        from = at + 4;
        let total = u32_at(data, at + 0x18).map_or(0, |n| n as usize);
        if let Some(c) = data.get(at..at + total).filter(|c| Container::parse(c).is_ok()) {
            out.push(c);
            from = at + total;
        }
    }
    out
}

pub(crate) fn u16_at(data: &[u8], at: usize) -> Result<u16, ShaderError> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("2 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u64_at(data: &[u8], at: usize) -> Result<u64, ShaderError> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

/// Builds a container from parts (test helper).
#[cfg(test)]
pub fn build(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let table = HEADER_SIZE + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_parts() {
        let bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4]), (FourCC::HASH, &[9; 20])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        assert_eq!(c.part(FourCC::HASH).unwrap().data.len(), 20);
        assert!(c.part(FourCC::DXIL).is_none());
    }

    /// A `PSV0` part with runtime information of `info` bytes, `resources`
    /// bindings of 24 bytes and the string table `table`, the entry's name
    /// at `name_at` of it.
    fn psv0(info: usize, resources: u32, table: &[u8], name_at: u32) -> Vec<u8> {
        let mut d = (info as u32).to_le_bytes().to_vec();
        d.resize(4 + info, 0);
        if info >= 52 {
            d[4 + 48..4 + 52].copy_from_slice(&name_at.to_le_bytes());
        }
        d.extend_from_slice(&resources.to_le_bytes());
        if resources > 0 {
            d.extend_from_slice(&24u32.to_le_bytes());
            d.resize(d.len() + 24 * resources as usize, 0xEE);
        }
        d.extend_from_slice(&(table.len() as u32).to_le_bytes());
        d.extend_from_slice(table);
        d
    }

    #[test]
    fn reads_the_entry_name_of_the_pipeline_state_part() {
        let name = |psv: &[u8]| Container::parse(&build(&[(FourCC::PSV0, psv)])).unwrap().entry_name().map(str::to_owned);
        assert_eq!(name(&psv0(52, 2, b"\0some_pass\0\0\0\0", 1)).as_deref(), Some("some_pass"));
        assert_eq!(name(&psv0(52, 0, b"\0main\0\0\0", 1)).as_deref(), Some("main"));
        // older runtime information has no name; an empty or missing one is none
        assert_eq!(name(&psv0(48, 2, b"\0some_pass\0", 1)), None);
        assert_eq!(name(&psv0(52, 2, b"\0some_pass\0", 0)), None);
        assert_eq!(name(&psv0(52, 2, b"\0abc", 9)), None);
        assert_eq!(name(&psv0(52, 1, b"", 0)[..60]), None);
        assert_eq!(Container::parse(&build(&[(FourCC::SFI0, &[0; 8])])).unwrap().entry_name(), None);
    }

    #[test]
    fn finds_the_containers_in_a_file() {
        let (a, b) = (build(&[(FourCC::SFI0, &[1, 2, 3, 4])]), build(&[(FourCC::ACCC, b"FSPX"), (FourCC::HASH, &[9; 20])]));
        // a magic with nothing behind it, two containers, one cut short
        let mut file = b"MZ..DXBC....".to_vec();
        file.extend_from_slice(&a);
        file.extend_from_slice(&[0; 7]);
        file.extend_from_slice(&b);
        file.extend_from_slice(&a[..a.len() - 1]);
        assert_eq!(containers(&file), [&a[..], &b[..]]);
        assert!(containers(b"DXB").is_empty());
        let c = Container::parse(&b).unwrap();
        assert_eq!(c.code_part().unwrap().fourcc, FourCC::ACCC);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Container::parse(b"nope").unwrap_err(), ShaderError::NotDxbc);
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let last = bytes.len() - 1;
        bytes[0x20] = 0xFF; // corrupt the first part offset
        assert!(Container::parse(&bytes).is_err());
        bytes.truncate(last);
        assert!(Container::parse(&bytes).is_err());
    }
}
