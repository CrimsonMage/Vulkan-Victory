//! Waterfall loops over per-lane descriptors: the cases of origin/main
//! (fxceless/victory-fork 6742906, GPL-3.0), `crates/victory-shader/src/
//! emit/waterfall_tests.rs`, with its synthetic GFX10 programs from
//! `resource.rs` (`resource::tests::WATERFALL_*`, `cmpx_waterfall`), kept
//! as written. Each lane must read the resource its own descriptor names:
//! the loop reads the first active lane's descriptor, serves the lanes that
//! hold the same one and drops them from EXEC until none is left.
//!
//! They lower the loop to one access with each lane's own heap index and
//! run it on a D3D12 device and on WARP; here the loop runs as the console
//! runs it (our translation keeps it), in the interpreter, with images that
//! answer every sample with the slot of the T# sampled (its WORD7, as red):
//! slot 1 is their texture of red 1.0, slot 2 the one of 2.0.
//!
//! Not ported: their checks of the lowering's shape (`nonuniform_handles`,
//! `wave_ops`), which have no meaning here.

#![allow(clippy::all)]

use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::vk::interp::{Images, Machine, Memory, Stop};
use crate::vk::interp_group::{Group, Mode};
use crate::vk::tir::{ImageClass, ImageKind, Sampler};

/// `resource::tests::WATERFALL_JOINED_WAVE32`: one of two image descriptors loaded from the
/// table at s0, chosen by control flow as the compiler lowers an if/else on a per-lane
/// condition: lanes below 16 move the first into v[4:11], the others the second, each arm
/// skipped when no lane takes it. The sample goes to the buffer at dword 16 of the table, at
/// lane * 4.
const WATERFALL_JOINED_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0602,
    0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
    0xF40C_0802,
    0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x7E04_0D00, // v_cvt_f32_u32 v2, v0
    0x7E06_0302, // v_mov_b32 v3, v2
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
    0x8914_147E, // s_xor_b32 s20, exec_lo, s20
    0xBF88_0008, // s_cbranch_execz -> 0x70
    0x7E08_0218, // v_mov_b32 v4, s24
    0x7E0A_0219, // v_mov_b32 v5, s25
    0x7E0C_021A, // v_mov_b32 v6, s26
    0x7E0E_021B, // v_mov_b32 v7, s27
    0x7E10_021C, // v_mov_b32 v8, s28
    0x7E12_021D, // v_mov_b32 v9, s29
    0x7E14_021E, // v_mov_b32 v10, s30
    0x7E16_021F, // v_mov_b32 v11, s31
    0xBE94_3D14, // 0x70: s_or_saveexec_b32 s20, s20
    0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
    0xBF88_0008, // s_cbranch_execz -> 0x9c
    0x7E08_0220, // v_mov_b32 v4, s32
    0x7E0A_0221, // v_mov_b32 v5, s33
    0x7E0C_0222, // v_mov_b32 v6, s34
    0x7E0E_0223, // v_mov_b32 v7, s35
    0x7E10_0224, // v_mov_b32 v8, s36
    0x7E12_0225, // v_mov_b32 v9, s37
    0x7E14_0226, // v_mov_b32 v10, s38
    0x7E16_0227, // v_mov_b32 v11, s39
    0x887E_147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
    0xBE97_037E, // s_mov_b32 s23, exec_lo
    0x7E18_0504, // 0xa4: v_readfirstlane_b32 s12, v4
    0x7E1A_0505, // v_readfirstlane_b32 s13, v5
    0x7E1C_0506, // v_readfirstlane_b32 s14, v6
    0x7E1E_0507, // v_readfirstlane_b32 s15, v7
    0x7E20_0508, // v_readfirstlane_b32 s16, v8
    0x7E22_0509, // v_readfirstlane_b32 s17, v9
    0x7E24_050A, // v_readfirstlane_b32 s18, v10
    0x7E26_050B, // v_readfirstlane_b32 s19, v11
    0xD4E2_0015,
    0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2_0016,
    0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715_1615, // s_and_b32 s21, s21, s22
    0xD4E2_0016,
    0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
    0x8715_1615, // s_and_b32 s21, s21, s22
    0xD4E2_0016,
    0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
    0x8715_1615, // s_and_b32 s21, s21, s22
    0xBE95_3C15, // s_and_saveexec_b32 s21, s21
    0xBF80_0000, // s_nop 0
    0xF09C_0F08,
    0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89_FFE6, // s_cbranch_execnz -> 0xa4
    0xBEFE_0317, // s_mov_b32 exec_lo, s23
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// `resource::tests::WATERFALL_INDEXED_WAVE32`: the console compiler's waterfall (the title's
/// ray-tracing resolve and others): each lane holds its entry's byte offset in the table at
/// s0 (v7 = (lane & 1) * 32); the loop reads the first active lane's offset, narrows EXEC to
/// the lanes with the same one (`v_cmpx_eq_u32`), loads that entry with `s_load_dwordx8` and
/// samples, then drops those lanes (`s_andn2_wrexec_b32`).
const WATERFALL_INDEXED_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x7E04_0D00, // v_cvt_f32_u32 v2, v0
    0x7E06_0302, // v_mov_b32 v3, v2
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0x360E_0081, // v_and_b32 v7, 1, v0
    0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBE97_037E, // s_mov_b32 s23, exec_lo
    0xBE95_037E, // s_mov_b32 s21, exec_lo
    0x7E28_0507, // 0x40: v_readfirstlane_b32 s20, v7
    0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
    0xF40C_0302,
    0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0F08,
    0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
    0xBF89_FFF6, // s_cbranch_execnz -> 0x40
    0xBEFE_0317, // s_mov_b32 exec_lo, s23
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// `resource::tests::WATERFALL_JOINED_MIXED_WAVE32`: [`WATERFALL_JOINED_WAVE32`] with other
/// sources on each path: lanes below 16 load table entry (lane & 1) with two
/// `global_load_dwordx4`, the others pick per lane with `v_cndmask_b32` (lanes below 24 the
/// first entry, the rest the second).
const WATERFALL_JOINED_MIXED_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0602,
    0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
    0xF40C_0802,
    0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x7E04_0D00, // v_cvt_f32_u32 v2, v0
    0x7E06_0302, // v_mov_b32 v3, v2
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0x3602_0081, // v_and_b32 v1, 1, v0
    0x3402_0285, // v_lshlrev_b32 v1, 5, v1
    0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
    0x8914_147E, // s_xor_b32 s20, exec_lo, s20
    0xBF88_0004, // s_cbranch_execz -> 0x68
    0xDC38_8000,
    0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
    0xDC38_8010,
    0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
    0xBE94_3D14, // 0x68: s_or_saveexec_b32 s20, s20
    0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
    0xBF88_0011, // s_cbranch_execz -> 0xb8
    0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
    0x7E08_0218, // v_mov_b32 v4, s24
    0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
    0x7E0A_0219, // v_mov_b32 v5, s25
    0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
    0x7E0C_021A, // v_mov_b32 v6, s26
    0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
    0x7E0E_021B, // v_mov_b32 v7, s27
    0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
    0x7E10_021C, // v_mov_b32 v8, s28
    0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
    0x7E12_021D, // v_mov_b32 v9, s29
    0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
    0x7E14_021E, // v_mov_b32 v10, s30
    0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
    0x7E16_021F, // v_mov_b32 v11, s31
    0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
    0x887E_147E, // 0xb8: s_or_b32 exec_lo, exec_lo, s20
    0xBE97_037E, // s_mov_b32 s23, exec_lo
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E18_0504, // 0xc4: v_readfirstlane_b32 s12, v4
    0x7E1A_0505, // v_readfirstlane_b32 s13, v5
    0x7E1C_0506, // v_readfirstlane_b32 s14, v6
    0x7E1E_0507, // v_readfirstlane_b32 s15, v7
    0x7E20_0508, // v_readfirstlane_b32 s16, v8
    0x7E22_0509, // v_readfirstlane_b32 s17, v9
    0x7E24_050A, // v_readfirstlane_b32 s18, v10
    0x7E26_050B, // v_readfirstlane_b32 s19, v11
    0xD4E2_0015,
    0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2_0016,
    0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715_1615, // s_and_b32 s21, s21, s22
    0xD4E2_0016,
    0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
    0x8715_1615, // s_and_b32 s21, s21, s22
    0xD4E2_0016,
    0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
    0x8715_1615, // s_and_b32 s21, s21, s22
    0xBE95_3C15, // s_and_saveexec_b32 s21, s21
    0xBF80_0000, // s_nop 0
    0xF09C_0F08,
    0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89_FFE6, // s_cbranch_execnz -> 0xc4
    0xBEFE_0317, // s_mov_b32 exec_lo, s23
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// `resource::tests::WATERFALL_JOINED_BUFFER_WAVE32`: a waterfall loop over one of two buffer
/// descriptors (four dwords) loaded from the table at s0, chosen by nested if/else: lanes
/// below 16 move the first into v[4:7], lanes from 16 to 23 the second, and the others keep
/// what v[4:7] held. Only lanes below 24 (the same mask) run the loop, which loads dword
/// `lane` of the chosen buffer, and store it to the buffer at dword 16 of the table.
const WATERFALL_JOINED_BUFFER_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0602,
    0xFA00_0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
    0xF408_0802,
    0xFA00_0020, // s_load_dwordx4 s[32:35], s[4:5], 0x20
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0xD4C4_001E,
    0x0002_0098, // v_cmp_gt_u32_e64 s30, 24, v0
    0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
    0x8914_147E, // s_xor_b32 s20, exec_lo, s20
    0xBF88_0004, // s_cbranch_execz -> 0x50
    0x7E08_0218, // 0x40: v_mov_b32 v4, s24
    0x7E0A_0219, // v_mov_b32 v5, s25
    0x7E0C_021A, // v_mov_b32 v6, s26
    0x7E0E_021B, // v_mov_b32 v7, s27
    0xBE94_3D14, // 0x50: s_or_saveexec_b32 s20, s20
    0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
    0xBF88_0007, // s_cbranch_execz -> 0x78
    0xBE9F_3C1E, // s_and_saveexec_b32 s31, s30
    0xBF88_0004, // s_cbranch_execz -> 0x74
    0x7E08_0220, // 0x64: v_mov_b32 v4, s32
    0x7E0A_0221, // v_mov_b32 v5, s33
    0x7E0C_0222, // v_mov_b32 v6, s34
    0x7E0E_0223, // v_mov_b32 v7, s35
    0x887E_1F7E, // 0x74: s_or_b32 exec_lo, exec_lo, s31
    0x887E_147E, // 0x78: s_or_b32 exec_lo, exec_lo, s20
    0xBE97_3C1E, // s_and_saveexec_b32 s23, s30
    0xBF88_0013, // s_cbranch_execz -> 0xd0
    0xBE9D_037E, // s_mov_b32 s29, exec_lo
    0x7E18_0504, // 0x88: v_readfirstlane_b32 s12, v4
    0x7E1A_0505, // v_readfirstlane_b32 s13, v5
    0x7E1C_0506, // v_readfirstlane_b32 s14, v6
    0x7E1E_0507, // v_readfirstlane_b32 s15, v7
    0xD4E2_0015,
    0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2_0016,
    0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715_1615, // s_and_b32 s21, s21, s22
    0xBE95_3C15, // s_and_saveexec_b32 s21, s21
    0xE030_1000,
    0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89_FFF1, // s_cbranch_execnz -> 0x88
    0xBEFE_031D, // s_mov_b32 exec_lo, s29
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0x887E_177E, // 0xd0: s_or_b32 exec_lo, exec_lo, s23
    0xBF81_0000, // s_endpgm
];

fn word_at(words: &[u32], word: u32) -> usize {
    words.iter().position(|&w| w == word).unwrap()
}

/// `resource::tests::cmpx_waterfall` (wave32): the waterfall with `v_cmpx_eq` narrowing EXEC
/// itself, and the retry as `todo & !EXEC` (`folded`: in one `s_andn2_wrexec`), as LLVM's
/// AMDGPU backend writes it for GFX10 and later.
fn cmpx_waterfall(folded: bool) -> Vec<u32> {
    let base = WATERFALL_JOINED_WAVE32;
    let reads = word_at(base, 0x7E18_0504);
    let sample = word_at(base, 0xF09C_0F08);
    let mut w = base[..=word_at(base, 0xBE97_037E)].to_vec();
    w.push(0xBE95_037E); // s_mov_b32 s21, exec_lo
    w.extend(&base[reads..reads + 8]); // v_readfirstlane_b32 s12..s19
    w.extend([
        0x7DE4_080C, // v_cmpx_eq_u64 s[12:13], v[4:5]
        0x7DE4_0C0E, // v_cmpx_eq_u64 s[14:15], v[6:7]
        0x7DE4_1010, // v_cmpx_eq_u64 s[16:17], v[8:9]
        0x7DE4_1412, // v_cmpx_eq_u64 s[18:19], v[10:11]
    ]);
    w.extend(&base[sample..sample + 2]); // image_sample_lz
    if folded {
        w.extend([0xBE95_4715, 0xBF89_FFF0]); // s_andn2_wrexec_b32 s21, s21; s_cbranch_execnz
    } else {
        w.extend([0x8A15_7E15, 0xBEFE_0315, 0xBF89_FFEF]); // s_andn2_b32 s21, s21, exec_lo; s_mov_b32 exec_lo, s21
    }
    w.extend(&base[word_at(base, 0xBEFE_0317)..]);
    w
}

/// Ours, after `resource::tests::WATERFALL_FETCHED_WAVE64`: a console wave of 64 whose lane
/// i loads entry i of the table at s0 with two `global_load_dwordx4`; the loop reads the
/// first active lane's descriptor into s[12:19] (64-bit lane masks), samples with the lanes
/// whose descriptor it is, and repeats for the others. Theirs writes the sample to LDS; this
/// one to the buffer whose V# is at dword 4 of the table at s1, at lane * 4.
const WATERFALL_FETCHED_WAVE64: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0x3402_0085, // v_lshlrev_b32 v1, 5, v0
    0xDC38_8000,
    0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
    0xDC38_8010,
    0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A03,
    0xFA00_0010, // s_load_dwordx4 s[40:43], s[6:7], 0x10 (ours)
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0 (ours)
    0x7E04_0D00, // v_cvt_f32_u32 v2, v0
    0x7E06_0302, // v_mov_b32 v3, v2
    0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
    0xBE94_047E, // s_mov_b64 s[20:21], exec
    0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
    0x7E1A_0505, // v_readfirstlane_b32 s13, v5
    0x7E1C_0506, // v_readfirstlane_b32 s14, v6
    0x7E1E_0507, // v_readfirstlane_b32 s15, v7
    0x7E20_0508, // v_readfirstlane_b32 s16, v8
    0x7E22_0509, // v_readfirstlane_b32 s17, v9
    0x7E24_050A, // v_readfirstlane_b32 s18, v10
    0x7E26_050B, // v_readfirstlane_b32 s19, v11
    0xD4E2_0016,
    0x0002_080C, // v_cmp_eq_u64_e64 s[22:23], s[12:13], v[4:5]
    0xD4E2_0018,
    0x0002_0C0E, // v_cmp_eq_u64_e64 s[24:25], s[14:15], v[6:7]
    0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
    0xD4E2_0018,
    0x0002_1010, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[8:9]
    0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
    0xD4E2_0018,
    0x0002_1412, // v_cmp_eq_u64_e64 s[24:25], s[18:19], v[10:11]
    0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
    0xBE96_2416, // s_and_saveexec_b64 s[22:23], s[22:23]
    0xBF80_0000, // s_nop 0
    0xF09C_0F08,
    0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x89FE_167E, // s_xor_b64 exec, exec, s[22:23]
    0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
    0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen (ours, for their ds_write_b32)
    0xBF81_0000, // s_endpgm
];

/// A compute program of one console wave with the table of views at s0 and the table of
/// samplers at s1 (their `build`).
fn build(words: &[u32], wave64: bool) -> NativeProgram {
    let table = |user_data: u8| RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data, slot: user_data };
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [if wave64 { 64 } else { 32 }, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Every sample reads (the T#'s slot, 0, 0, 1) as floats.
struct SlotImages;

impl Images for SlotImages {
    fn image(&mut self, _: ImageKind, _: ImageClass, _: Sampler, _: u16, args: &[[u64; 4]], _: usize) -> [u64; 4] {
        let slot = (args[0][0] & 0xFF_FFFF) as f32;
        [u64::from(slot.to_bits()), 0, 0, u64::from(1.0f32.to_bits())]
    }
}

/// The T# of the image in `slot` (`docs/vulkan-abi.md` 5.2: WORD7 the slot, WORD6 its check).
fn tsharp(slot: u32) -> [u32; 8] {
    [0, 0, 0, 0, 0, 0, slot ^ 0x7A3C_5E91, slot]
}

/// A raw V# over 64 KiB at `base`.
fn vsharp(base: u32) -> [u32; 4] {
    [base, 0, 0x1_0000, 0x3000_0FAC]
}

const VIEWS: u64 = 0x1000;
const SAMPLERS: u64 = 0x1800;
const OUT: u32 = 0x2000;

/// The output buffer's first `lanes` dwords after the program ran over `mem` (the views'
/// table at s0, the samplers' at s1), in every way of [`super::wave64`]'s: as one wave, and
/// for a console wave of 64 as two halves in both orders, which must agree.
fn run(native: &NativeProgram, mem: &Memory, lanes: u32) -> Result<Vec<u32>, String> {
    let (f, _) = crate::vk::thread_ir_native(native).map_err(|e| format!("translation: {e}"))?;
    let setup = |m: &mut Machine| {
        m.push[4] = VIEWS as u32;
        m.push[5] = SAMPLERS as u32;
        m.images = Some(Box::new(SlotImages));
    };
    let mut all = Vec::new();
    for (mode, upper_first) in [(Mode::Waves, false), (Mode::Halves, false), (Mode::Halves, true)] {
        let mut g = Group::new(&f, native.wave64, crate::vk::lift::Lift::host_threads(native), mode, mem.clone(), setup);
        g.upper_first = upper_first;
        all.push(match g.run() {
            Stop::Returned => Ok((0..u64::from(lanes)).map(|i| g.mem.dwords.get(&(u64::from(OUT) + 4 * i)).copied().unwrap_or(0)).collect::<Vec<u32>>()),
            s => Err(format!("{mode:?} (upper first: {upper_first}): {s:?}")),
        });
    }
    if all[1] != all[0] || all[2] != all[0] {
        return Err(format!("the ways differ: {all:x?}"));
    }
    all.swap_remove(0)
}

/// The red channel each lane sampled from two textures, red 1.0 at table entry 0 and 2.0 at
/// entry 1 (their `sampled`).
fn sampled(words: &[u32]) -> Result<Vec<f32>, String> {
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    mem.put(VIEWS, &tsharp(1));
    mem.put(VIEWS + 0x20, &tsharp(2));
    mem.put(VIEWS + 0x40, &vsharp(OUT));
    mem.put(SAMPLERS, &[0; 4]);
    run(&build(words, false), &mem, 32).map(|w| w.into_iter().map(f32::from_bits).collect())
}

/// Theirs: `lanes_sample_the_texture_their_if_else_arm_chose`.
#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    // Lanes below 16 took the first texture, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    assert_eq!(sampled(WATERFALL_JOINED_WAVE32), Ok(want));
}

/// Theirs: `lanes_sample_the_texture_their_arm_loaded_or_picked`.
#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    // Lanes below 16 loaded entry (lane & 1); of the others, lanes below 24 picked the first,
    // the rest the second.
    let want: Vec<f32> = (0..32)
        .map(|l| match l {
            0..16 => [1.0, 2.0][l & 1],
            16..24 => 1.0,
            _ => 2.0,
        })
        .collect();
    assert_eq!(sampled(WATERFALL_JOINED_MIXED_WAVE32), Ok(want));
}

/// Theirs: `lanes_load_from_the_buffer_their_nested_arms_chose`.
#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    mem.put(VIEWS, &vsharp(0x3000));
    mem.put(VIEWS + 0x20, &vsharp(0x4000));
    mem.put(VIEWS + 0x40, &vsharp(OUT));
    mem.put(0x3000, &(0..32).map(|i| 100 + i).collect::<Vec<u32>>());
    mem.put(0x4000, &(0..32).map(|i| 200 + i).collect::<Vec<u32>>());
    // Dword `lane` of the first buffer below 16, of the second from 16 to 23; lanes from 24
    // on load and store nothing.
    let want: Vec<u32> = (0..32)
        .map(|l| match l {
            0..16 => 100 + l,
            16..24 => 200 + l,
            _ => 0,
        })
        .collect();
    assert_eq!(run(&build(WATERFALL_JOINED_BUFFER_WAVE32, false), &mem, 32), Ok(want));
}

/// Theirs: `lanes_sample_the_entry_their_offset_names_in_the_console_waterfall`.
#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    // Even lanes hold entry 0's offset, odd lanes entry 1's.
    let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
    assert_eq!(sampled(WATERFALL_INDEXED_WAVE32), Ok(want));
}

/// Theirs: `lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx`.
#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        assert_eq!(sampled(&cmpx_waterfall(folded)), Ok(want.clone()), "folded {folded}");
    }
}

/// Ours: the waterfall of a console wave of 64 ([`WATERFALL_FETCHED_WAVE64`]) with `kinds`
/// descriptors spread over the lanes of both halves (lane i holds slot 1 + (i * step) mod
/// kinds): every lane samples its own, whichever half's first lane the loop serves; as one
/// wave and as the halves the card runs.
#[test]
fn a_wave_of_64_serves_every_lanes_descriptor_across_its_halves() {
    for (kinds, step) in [(1u32, 1u32), (2, 1), (5, 1), (5, 3), (64, 1), (2, 32)] {
        let slot = |lane: u32| if step == 32 { 1 + lane / 32 } else { 1 + lane * step % kinds };
        let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
        for lane in 0..64 {
            mem.put(VIEWS + 0x20 * u64::from(lane), &tsharp(slot(lane)));
        }
        mem.put(SAMPLERS, &[0; 4]);
        mem.put(SAMPLERS + 0x10, &vsharp(OUT));
        let want: Vec<u32> = (0..64).map(|l| (slot(l) as f32).to_bits()).collect();
        assert_eq!(run(&build(WATERFALL_FETCHED_WAVE64, true), &mem, 64), Ok(want), "{kinds} descriptors, step {step}");
    }
}
