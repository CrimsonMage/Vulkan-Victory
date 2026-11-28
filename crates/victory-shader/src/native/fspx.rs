//! The compute program of an `ACCC` part: what a container carries in place
//! of `DXIL` when another compiler made the console code (AMD's, for the
//! FidelityFX upscaler's passes). The part is an `FSPX` payload, a small
//! directory around the program. Layout as observed (offsets from "FSPX"):
//!
//! ```text
//! +0x00 "FSPX" u32 version (1), u32 entry count, u32 size; entries of 16 B
//!       from +0x10: u32 kind, u32 offset, u32 size, u32 0
//! kind 0  dispatch state, 7 dwords: 3 NumThreadsX (bits 0-15) and Y
//!         (16-31); 4 NumThreadsZ (0-15), bit 20 wave32 (else wave64);
//!         5 COMPUTE_PGM_RSRC1; 6 COMPUTE_PGM_RSRC2
//! kind 1  a hash (not read)
//! kind 6  the GFX10 code; it starts with words that do nothing and carry
//!         the shader hash (`v_nop` in its long encoding, with literals)
//! kind 7  records: u32 0x403, u32 size - 0x14, u32 code dwords, u64 0,
//!         u8 0x40, u8 first user-data dword, u8 count of 0x41 records,
//!         u8 0; then records of 32 B from +0x18, each a u32 tag:
//!   0x41  a descriptor range the code reads: u8 resource type (as FSSM's
//!         used entries), u8 root type, u16 0, u32 table offset (dwords),
//!         u32 register, u32 view kind (1 buffer, 4 texture, 0 a root
//!         view), u8 descriptor dwords, u8 user-data dword, u8 root slot,
//!         u8 0, u32 -1, u32 0
//!   0x42  a view declaration (not read)
//!   0x43  the shader hash; the last record
//! ```
//!
//! The directory is read as the upscaler's own reader reads it: an entry
//! whose bytes start inside the directory or end past the payload is
//! skipped, the last entry of a kind counts, and a payload without kind 0
//! or kind 6 has no program. The records (kind 7) say where the root
//! parameters arrive, which the translation needs; without them the
//! program is refused.

use super::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource, bytes, le32,
};
use crate::ShaderError::{self, Malformed};
use crate::dxbc::u32_at;

pub(super) const MAGIC: &[u8; 4] = b"FSPX";
const VERSION: u32 = 1;
const HEADER: usize = 0x10;
const ENTRY: usize = 16;
const MAX_ENTRIES: usize = 64;
const KIND_STATE: u32 = 0;
const KIND_CODE: u32 = 6;
const KIND_RECORDS: u32 = 7;
const STATE_THREADS_XY: usize = 0x0c;
const STATE_THREADS_Z: usize = 0x10;
const STATE_WAVE32: u32 = 1 << 20;
const STATE_RSRC1: usize = 0x14;
const STATE_RSRC2: usize = 0x18;
const RECORDS_CODE_DWORDS: usize = 8;
const RECORDS: usize = 0x18;
const RECORD: usize = 32;
const TAG_USED: u32 = 0x41;
const TAG_VIEW: u32 = 0x42;
const TAG_HASH: u32 = 0x43;

/// The last entry of `kind` in the directory of `part` that lies in it.
fn entry(part: &[u8], kind: u32) -> Result<Option<&[u8]>, ShaderError> {
    let count = u32_at(part, 8)? as usize;
    if count > MAX_ENTRIES {
        return Err(Malformed("implausible FSPX entry count"));
    }
    let total = (u32_at(part, 12)? as usize).min(part.len());
    let directory = HEADER + ENTRY * count;
    if directory > total {
        return Err(Malformed("FSPX directory out of bounds"));
    }
    let mut found = None;
    for at in (HEADER..directory).step_by(ENTRY) {
        let (offset, size) = (u32_at(part, at + 4)?, u32_at(part, at + 8)?);
        let inside = offset as usize >= directory && u64::from(offset) + u64::from(size) <= total as u64;
        if u32_at(part, at)? == kind && inside && size > 0 {
            found = bytes(part, offset.into(), size.into());
        }
    }
    Ok(found)
}

/// The descriptor range a `0x41` record names.
fn used(e: &[u8; RECORD]) -> UsedResource {
    UsedResource {
        kind: match e[4] {
            0 => ResourceType::Srv,
            1 => ResourceType::Sampler,
            2 => ResourceType::Cbv,
            3 => ResourceType::Uav,
            n => ResourceType::Other(n),
        },
        root: RootType::from_raw(e[5].into()),
        table_offset_dwords: le32(e, 8),
        lower_bound: le32(e, 12),
        stride: 0,
        dwords: e[20],
        user_data: e[21],
        slot: e[22],
    }
}

/// The ranges the records name, and the root parameters they arrive by:
/// one per root slot (a table's ranges share its user-data dword).
fn resources(records: &[u8]) -> Result<(Vec<RootParam>, Vec<UsedResource>), ShaderError> {
    let mut layout: Vec<RootParam> = Vec::new();
    let mut ranges = Vec::new();
    for e in records.get(RECORDS..).unwrap_or(&[]).as_chunks::<RECORD>().0 {
        match le32(e, 0) {
            TAG_USED => {}
            TAG_VIEW => continue,
            TAG_HASH => break,
            _ => return Err(Malformed("unknown FSPX record")),
        }
        let u = used(e);
        let sgprs = match u.root {
            RootType::Table => 1,
            RootType::Descriptor => 2,
            // the code holds its words: no user data
            RootType::ImmediateSampler => 0,
            _ => return Err(Malformed("an FSPX resource of an unknown root type")),
        };
        if sgprs > 0 && !layout.iter().any(|p| p.slot == u.slot) {
            layout.push(RootParam {
                root: u.root,
                table_offset_dwords: 0,
                sgprs,
                base_register: u.lower_bound,
                user_data: u.user_data,
                slot: u.slot,
            });
        }
        ranges.push(u);
    }
    Ok((layout, ranges))
}

/// The compute program of an `FSPX` payload.
pub(super) fn program(part: &[u8]) -> Result<NativeProgram, ShaderError> {
    if u32_at(part, 4)? != VERSION {
        return Err(Malformed("unknown FSPX version"));
    }
    let state = entry(part, KIND_STATE)?.ok_or(Malformed("FSPX payload without dispatch state"))?;
    let code = entry(part, KIND_CODE)?.ok_or(Malformed("FSPX payload without code"))?;
    let records = entry(part, KIND_RECORDS)?.ok_or(Malformed("FSPX payload without resource records"))?;
    if u32_at(records, RECORDS_CODE_DWORDS)? as usize * 4 != code.len() {
        return Err(Malformed("FSPX code size disagrees"));
    }
    let (xy, z) = (u32_at(state, STATE_THREADS_XY)?, u32_at(state, STATE_THREADS_Z)?);
    let wave64 = z & STATE_WAVE32 == 0;
    // VGPRs in granules of 4 (wave64) or 8 (wave32).
    let granules = (u32_at(state, STATE_RSRC1)? & 0x3F) + 1;
    let vgprs = (granules * if wave64 { 4 } else { 8 }).min(256) as u16;
    let (layout, used) = resources(records)?;
    Ok(NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: u32_at(state, STATE_RSRC2)?,
            threads: [xy & 0xFFFF, xy >> 16, z & 0xFFFF],
        }),
        code: code.to_vec(),
    })
}

/// A synthetic payload (test code builds it; no shipped program's bytes).
#[cfg(test)]
pub(crate) mod build {
    use super::*;

    /// One `0x41` record: (resource type, root type, table offset in
    /// dwords, register, descriptor dwords, user-data dword, root slot).
    pub(crate) type Used = (u8, u8, u32, u32, u8, u8, u8);

    /// The dispatch state of a `threads` group with `rsrc1` and `rsrc2`.
    pub(crate) fn state(threads: [u16; 3], wave32: bool, rsrc1: u32, rsrc2: u32) -> Vec<u8> {
        let xy = u32::from(threads[0]) | u32::from(threads[1]) << 16;
        let z = u32::from(threads[2]) | if wave32 { STATE_WAVE32 } else { 0 };
        [0x1111_1111, 0x2222_2222, 0, xy, z, rsrc1, rsrc2].iter().flat_map(|w: &u32| w.to_le_bytes()).collect()
    }

    /// Kind 7 for a program of `code_bytes`: the `used` ranges, one view
    /// declaration, the hash record.
    pub(crate) fn records(code_bytes: usize, used: &[Used]) -> Vec<u8> {
        let mut out = Vec::new();
        let size = RECORDS + RECORD * (used.len() + 2);
        for w in [0x403, size as u32 - 0x14, code_bytes as u32 / 4, 0, 0] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&[0x40, 4, used.len() as u8, 0]);
        for &(kind, root, offset, register, dwords, user_data, slot) in used {
            out.extend_from_slice(&TAG_USED.to_le_bytes());
            out.extend_from_slice(&[kind, root, 0, 0]);
            // the view kind: a texture's descriptor is 8 dwords, a buffer's 4
            let view = match dwords {
                8 => 4,
                4 => 1,
                _ => 0,
            };
            for w in [offset, register, view] {
                out.extend_from_slice(&w.to_le_bytes());
            }
            out.extend_from_slice(&[dwords, user_data, slot, 0]);
            out.extend_from_slice(&u32::MAX.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        out.extend_from_slice(&TAG_VIEW.to_le_bytes());
        out.extend_from_slice(&[0; RECORD - 4]);
        out.extend_from_slice(&TAG_HASH.to_le_bytes());
        out.extend_from_slice(&[0xAB; 16]);
        out.extend_from_slice(&[0; RECORD - 20]);
        out
    }

    /// The payload of `entries` (kind, bytes), in that order.
    pub(crate) fn payload(entries: &[(u32, &[u8])]) -> Vec<u8> {
        let mut at = HEADER + ENTRY * entries.len();
        let total = at + entries.iter().map(|(_, b)| b.len()).sum::<usize>();
        let mut out = MAGIC.to_vec();
        for w in [VERSION, entries.len() as u32, total as u32] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for (kind, bytes) in entries {
            for w in [*kind, at as u32, bytes.len() as u32, 0] {
                out.extend_from_slice(&w.to_le_bytes());
            }
            at += bytes.len();
        }
        for (_, bytes) in entries {
            out.extend_from_slice(bytes);
        }
        out
    }

    /// A whole payload as the upscaler's are laid out: kinds 0, 1, 6, 7.
    pub(crate) fn program(threads: [u16; 3], wave32: bool, rsrc1: u32, rsrc2: u32, code: &[u8], used: &[Used]) -> Vec<u8> {
        payload(&[
            (KIND_STATE, &state(threads, wave32, rsrc1, rsrc2)),
            (1, &[0x5A; 20]),
            (KIND_CODE, code),
            (KIND_RECORDS, &records(code.len(), used)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_programs;

    const CODE: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    /// A UAV buffer and an SRV texture in two tables, a root constant view,
    /// a sampler whose words the code holds.
    const USED: [build::Used; 4] = [(3, 3, 0x58, 11, 4, 4, 0), (0, 3, 0x10, 2, 8, 5, 1), (2, 1, 0, 0, 2, 6, 2), (1, 4, 0, 0, 4, 0, 0)];

    fn set(payload: &mut [u8], at: usize, value: u32) {
        payload[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn reads_the_program_and_its_dispatch_state() {
        // 8 user SGPRs, group id x and y, 1 KiB of LDS; 3 VGPR granules.
        let payload = build::program([64, 1, 1], false, 0x402C_0342, 0x1_0190, &CODE, &USED);
        let found = native_programs(&payload).unwrap();
        assert_eq!(found.len(), 1);
        let p = &found[0];
        assert_eq!((p.stage, p.wave64, p.vgprs), (HwStage::Cs, true, 12));
        assert_eq!(p.code, CODE);
        let StageIo::Cs(cs) = &p.io else { panic!("a compute program") };
        assert_eq!(cs.threads, [64, 1, 1]);
        assert_eq!((cs.user_sgprs(), cs.lds_bytes()), (8, 1024));
        assert_eq!(cs.system_sgprs(), ([Some(8), Some(9), None], None));
        // the same state as 32 lanes: VGPRs in granules of 8
        let payload = build::program([8, 4, 2], true, 0x402C_0342, 0x9C, &CODE, &USED);
        let p = &native_programs(&payload).unwrap()[0];
        assert_eq!((p.wave64, p.vgprs), (false, 24));
        let StageIo::Cs(cs) = &p.io else { panic!("a compute program") };
        assert_eq!(cs.threads, [8, 4, 2]);
    }

    #[test]
    fn reads_where_the_root_parameters_arrive() {
        let payload = build::program([32, 1, 1], true, 0, 0x9C, &CODE, &USED);
        let p = &native_programs(&payload).unwrap()[0];
        let param = |root, sgprs, base_register, user_data, slot| RootParam { root, table_offset_dwords: 0, sgprs, base_register, user_data, slot };
        // a table's address is one dword, a root view's two; the sampler
        // of the root signature has none
        assert_eq!(p.layout, [param(RootType::Table, 1, 11, 4, 0), param(RootType::Table, 1, 2, 5, 1), param(RootType::Descriptor, 2, 0, 6, 2)]);
        assert_eq!(p.used.len(), 4);
        assert_eq!(
            p.used[0],
            UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0x58, lower_bound: 11, stride: 0, dwords: 4, user_data: 4, slot: 0 }
        );
        assert_eq!((p.used[3].kind, p.used[3].root), (ResourceType::Sampler, RootType::ImmediateSampler));
    }

    #[test]
    fn two_ranges_of_a_table_are_one_parameter() {
        let used = [(3, 3, 0, 0, 8, 4, 0), (3, 3, 8, 1, 8, 4, 0), (0, 3, 0, 0, 8, 5, 1)];
        let payload = build::program([32, 1, 1], true, 0, 0x9C, &CODE, &used);
        let p = &native_programs(&payload).unwrap()[0];
        assert_eq!(p.layout.iter().map(|r| (r.slot, r.user_data, r.sgprs)).collect::<Vec<_>>(), [(0, 4, 1), (1, 5, 1)]);
        assert_eq!(p.used.len(), 3);
    }

    /// The directory as the upscaler's reader takes it: the last entry of
    /// a kind, entries outside the payload skipped.
    #[test]
    fn the_directory_is_read_as_the_upscaler_reads_it() {
        let state = build::state([32, 1, 1], true, 0, 0x9C);
        let other = build::state([16, 1, 1], true, 0, 0x9C);
        let records = build::records(CODE.len(), &USED);
        let entries: [(u32, &[u8]); 5] = [(KIND_STATE, &other), (KIND_CODE, &CODE), (9, &[7; 12]), (KIND_RECORDS, &records), (KIND_STATE, &state)];
        let mut payload = build::payload(&entries);
        let threads = |payload: &[u8]| match native_programs(payload).map(|p| p[0].io.clone()) {
            Ok(StageIo::Cs(cs)) => Ok(cs.threads[0]),
            Ok(_) => panic!("a compute program"),
            Err(e) => Err(e),
        };
        assert_eq!(threads(&payload), Ok(32));
        // the last state entry reaching past the payload: the one before counts
        let last = HEADER + ENTRY * 4;
        let size = payload.len() as u32;
        set(&mut payload, last + 8, size);
        assert_eq!(threads(&payload), Ok(16));
        // and with the first one inside the directory there is none
        set(&mut payload, HEADER + 4, HEADER as u32);
        assert_eq!(threads(&payload), Err(Malformed("FSPX payload without dispatch state")));
    }

    #[test]
    fn rejects_payloads_without_a_program() {
        let good = build::program([32, 1, 1], true, 0, 0x9C, &CODE, &USED);
        assert!(native_programs(&good).is_ok());
        let bad = |change: &dyn Fn(&mut Vec<u8>)| {
            let mut p = good.clone();
            change(&mut p);
            native_programs(&p).unwrap_err()
        };
        assert_eq!(bad(&|p| set(p, 4, 2)), Malformed("unknown FSPX version"));
        assert_eq!(bad(&|p| set(p, 8, 65)), Malformed("implausible FSPX entry count"));
        assert_eq!(bad(&|p| set(p, 12, 0x20)), Malformed("FSPX directory out of bounds"));
        // kinds 0, 6 and 7 renamed in turn
        assert_eq!(bad(&|p| set(p, HEADER, 99)), Malformed("FSPX payload without dispatch state"));
        assert_eq!(bad(&|p| set(p, HEADER + ENTRY * 2, 99)), Malformed("FSPX payload without code"));
        assert_eq!(bad(&|p| set(p, HEADER + ENTRY * 3, 99)), Malformed("FSPX payload without resource records"));
        let records = u32_at(&good, HEADER + ENTRY * 3 + 4).unwrap() as usize;
        assert_eq!(bad(&|p| set(p, records + RECORDS_CODE_DWORDS, 3)), Malformed("FSPX code size disagrees"));
        assert_eq!(bad(&|p| set(p, records + RECORDS, 0x44)), Malformed("unknown FSPX record"));
        // a root type no record of the upscaler has (constants)
        assert_eq!(bad(&|p| p[records + RECORDS + 5] = 0), Malformed("an FSPX resource of an unknown root type"));
        assert!(native_programs(&good[..good.len() - 1]).is_err());
        assert!(native_programs(b"FSPX").is_err());
    }

    /// A program as the upscaler's begin and end: three words that do
    /// nothing and carry a hash (`v_nop` in its long encoding with a
    /// literal each), then the end.
    fn program_code() -> Vec<u8> {
        let nop = |hash: u32, literal: u32| [0xD580_0000 | hash & 0x7FFF, 0x03FE_0105, literal];
        let words = [nop(0x1234, 0xDEAD_BEEF), nop(0x0567, 0x0BAD_F00D), nop(0x7ABC, 0x1357_9BDF)].concat();
        words.iter().chain(&[0xBF81_0000]).flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The container goes through the translator like the title's own:
    /// the group's size, and the user data by the records.
    #[test]
    fn a_container_with_an_accc_part_translates() {
        use crate::dxbc::{FourCC, build as container};
        let payload = build::program([32, 1, 1], true, 0x402C_0341, 0x9C, &program_code(), &USED);
        let good = container(&[(FourCC::HASH, &[7; 20]), (FourCC::ACCC, &payload)]);
        let vs = crate::vk::translate(&good).expect("translation");
        assert_eq!((vs.stage, vs.local_size), (HwStage::Cs, [32, 1, 1]));
        let user: Vec<(u32, u8, u32)> = vs.user_data.iter().map(|u| (u.index, u.slot, u.dword)).collect();
        assert_eq!(user, [(8, 0, 0), (9, 1, 0), (10, 2, 0), (11, 2, 1)]);
        // a file with the container in it; with one more whose code is no code
        let mut file = b"MZ file".to_vec();
        file.extend_from_slice(&good);
        assert_eq!(crate::vk::report::accc_passes(&file), Ok(1));
        let bad = build::program([32, 1, 1], true, 0x402C_0341, 0x9C, &[0xFF; 8], &USED);
        file.extend_from_slice(&container(&[(FourCC::ACCC, &bad)]));
        let why = crate::vk::report::accc_passes(&file).unwrap_err();
        assert!(why.starts_with("pass 1 of 2: "), "{why}");
        // the title's own kind of container is no pass
        assert!(crate::vk::report::accc_passes(&container(&[(FourCC::SFI0, &[0; 8])])).is_err());
    }
}
