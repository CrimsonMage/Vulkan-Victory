//! Instruction-level cases from the belobug fork's translator (GPL-3.0,
//! a fork of the same upstream): `crates/victory-shader/src/emit/valu_tests.rs`,
//! its RDNA2 hardware behaviour, run here through `vk::thread_ir_native`
//! and `vk::interp` instead of DXIL on a D3D12 device. The cases and their
//! encoding helpers are theirs (kept as written); the harness (`build_cs`,
//! `run_cases`) is ours. A case that fails names every input it fails on.

#![allow(clippy::all, dead_code, unused_variables)]

use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::vk::interp::{Machine, Memory, Stop};

mod valu1;

mod valu2;

mod origin1;

mod intersect;

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SOPK `op sdst, simm16`; the compares read `sdst` as their operand.
const fn sopk(op: u8, sdst: u8, simm16: u16) -> u32 {
    0xB000_0000 | (op as u32 & 0x1F) << 23 | (sdst as u32 & 0x7F) << 16 | simm16 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false)
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in

/// A synthetic compute program (their `HostShader`): 4 dwords of root
/// constants in s0..s3, the table address in s4.
pub(super) struct HostShader {
    native: NativeProgram,
    threads: u32,
}

fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 1, 0)
}

/// As theirs: s[4:5] = the table, s[8:11] = its first V#, v1..v4 = s0..s3,
/// `body`, then `v0` stored to dword 0 of that buffer.
fn build_cs(body: &[u32], threads: u32, lds_bytes: u32) -> HostShader {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i)));
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let param = |root, sgprs, user_data, slot| RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![param(RootType::Constants, 4, 0, 0), param(RootType::Table, 1, 4, 1)],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: (5 << 1) | ((lds_bytes / 512) << 15), threads: [threads, 1, 1] }),
        code,
    };
    HostShader { native, threads }
}

/// The table at 0x1000 holds a raw V# over 0x2000.., the program's output
/// dword 0 there.
fn run_one(shader: &HostShader, inputs: [u32; 4]) -> Result<u32, String> {
    let (f, _) = crate::vk::thread_ir_native(&shader.native).map_err(|e| format!("translation: {e}"))?;
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    mem.put(0x1000, &[0x2000, 0, 0x1_0000, 0x3000_0FAC]);
    let lanes = if shader.threads > 32 { 64 } else { 32 };
    let mut m = Machine::new(&f, lanes, mem);
    m.push[4..8].copy_from_slice(&inputs);
    m.push[8] = 0x1000;
    match m.run() {
        Stop::Returned => Ok(m.mem.dwords.get(&0x2000).copied().unwrap_or(0)),
        s => Err(format!("{s:?}")),
    }
}

fn run_cases(shader: &HostShader, cases: &[[u32; 4]], check: impl FnMut(&str, usize, [u32; 4], u32)) {
    run_cases_on(shader, cases, &[("interp", false)], check);
}

/// Every case through the interpreter; the failures of all of them at the end.
fn run_cases_on(shader: &HostShader, cases: &[[u32; 4]], _devices: &[(&str, bool)], mut check: impl FnMut(&str, usize, [u32; 4], u32)) {
    let mut failed = Vec::new();
    for (i, &inputs) in cases.iter().enumerate() {
        let got = match run_one(shader, inputs) {
            Ok(g) => g,
            Err(e) => {
                failed.push(format!("case {i} {inputs:08x?}: {e}"));
                continue;
            }
        };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check("interp", i, inputs, got)));
        if let Err(p) = r {
            let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            failed.push(msg.lines().next().unwrap_or("").to_string());
        }
    }
    assert!(failed.is_empty(), "{} of {} cases failed:\n{}", failed.len(), cases.len(), failed.join("\n"));
}

/// Ours (not theirs): a constant table after `s_endpgm`, read through
/// `s_getpc_b64` + its distance + an index from s0, as
/// CS_ApplyCombinedDeferred reads its tap offsets: the table's own words
/// (the reference: the code's bytes at their code offset in memory).
#[test]
fn constant_tables_in_the_code_read_through_getpc() {
    const TABLE: [u32; 6] = [0x3F80_0000, 0xBF00_0000, 7, 0xDEAD_BEEF, 0, 0x4120_0000];
    // the prologue is 7 dwords: s_getpc_b64 at byte 28 gives 32
    let body = |dist: u32| {
        vec![
            0xBE86_1F00,             // s_getpc_b64 s[6:7]
            sop2(0x00, 6, 6, 255),   // s_add_u32 s6, s6, dist
            dist,
            sop2(0x04, 7, 7, 0x80),  // s_addc_u32 s7, s7, 0
            sop2(0x1E, 12, 0, 0x82), // s_lshl_b32 s12, s0, 2 (s[8:11]: the V#)
            sop2(0x00, 6, 6, 12),    // s_add_u32 s6, s6, s12
            sop2(0x04, 7, 7, 0x80),  // s_addc_u32 s7, s7, 0
            0x3D << 26 | 13 << 6 | 3, // s_load_dword s13, s[6:7], 0
            125 << 25,
            vop1(0x001, 0, sgpr(13)), // v_mov_b32 v0, s13
        ]
    };
    let start = build_cs(&body(0), 1, 0).native.code.len() as u32;
    let mut shader = build_cs(&body(start - 32), 1, 0);
    shader.native.code.extend(TABLE.iter().flat_map(|w| w.to_le_bytes()));
    // the reference: the code's bytes at their code offsets, as memory
    let words: Vec<u32> = shader.native.code.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    mem.put(0, &words);
    let cases: Vec<[u32; 4]> = (0..TABLE.len() as u32).map(|i| [i, 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, i, _, got| {
        let reference = mem.dwords.get(&u64::from(start + 4 * i as u32)).copied();
        assert_eq!(Some(got), reference, "{backend}: table dword {i}");
        assert_eq!(got, TABLE[i], "{backend}: table dword {i}");
    });
}

/// Ours (not theirs): `v_cvt_pkrtz_f16_f32 v0, v1, v2` rounds toward zero,
/// past the largest half too (±65504, not infinity), whatever the host's
/// own conversion rounds by.
#[test]
fn pkrtz_rounds_toward_zero() {
    let shader = build_valu_cs(&[0x2F << 25 | 2 << 9 | 257]);
    let cases = [
        ([1.0f32.to_bits(), 2.0f32.to_bits(), 0, 0], 0x4000_3C00),
        ([70000.0f32.to_bits(), (-70000.0f32).to_bits(), 0, 0], 0xFBFF_7BFF),
        // 1 + 2^-10 - 2^-23: nearer 0x3C01, toward zero 0x3C00
        ([0x3F80_1FFF, 0xBF80_1FFF, 0, 0], 0xBC00_3C00),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        assert_eq!(got, cases[i].1, "{backend}: v_cvt_pkrtz_f16_f32 {inp:x?} = {got:#010x}");
    });
}

fn scalar_quad_masks_pack_nibbles_and_set_scc() {
    let pack_words = |opcode: u32| {
        [
            0xBE90_0000 | opcode << 8,   // s_quadmask s16, s0 (pairs for b64)
            sop2(0x0a, 18, 129, 128),  // s_cselect_b32 s18, 1, 0
            sop2(0x1e, 18, 18, 144),   // s_lshl_b32 s18, s18, 16
            sop2(0x10, 16, 16, 18),    // s_or_b32 s16, s16, s18
            vop1(1, 0, sgpr(16)),      // v_mov_b32 v0, s16
        ]
    };
    let cases = [
        ([0, 0, 0, 0], [0, 0]),
        ([0x8000_0420, 0x0100_0001, 0, 0], [0x1_0086, 0x1_4186]),
        ([0, 0x8000_0000, 0, 0], [0, 0x1_8000]),
        ([u32::MAX, u32::MAX, 0, 0], [0x1_00ff, 0x1_ffff]),
    ];
    let inputs = cases.map(|(input, _)| input);
    for (wide, opcode) in [0x2c, 0x2d].into_iter().enumerate() {
        let shader = build_valu_cs(&pack_words(opcode));
        run_cases(&shader, &inputs, |backend, i, _, got| {
            assert_eq!(got, cases[i].1[wide], "{backend}, opcode {opcode:#x}");
        });
    }
    let high = build_valu_cs(&[
        0xBE90_2D00, // s_quadmask_b64 s[16:17], s[0:1]
        vop1(1, 0, sgpr(17)),
    ]);
    run_cases(&high, &inputs, |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: high dword");
    });
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

/// MTBUF `op vdata, off, s[srsrc:srsrc+3], 0 format:format` (at byte 0).
const fn mtbuf(op: u8, vdata: u8, srsrc: u8, format: u8) -> [u32; 2] {
    [
        0x3A << 26 | (format as u32 & 0x7F) << 19 | (op as u32 & 7) << 16,
        0x80 << 24
            | (op as u32 >> 3 & 1) << 21
            | (srsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8,
    ]
}

/// Stores `v1` to the buffer, then reads it back into `v0` with the tbuffer
/// load `op` in buffer format `format`.
fn tbuffer_load_of_v1(op: u8, format: u8) -> HostShader {
    let mut body = buffer_store_dword(1, 8, 0).to_vec();
    body.extend(mtbuf(op, 0, 8, format));
    build_valu_cs(&body)
}

/// Codes of the normalized formats sampled below: every 8-bit code; for
/// 16 bits a stride, the RX 6700 XT's division misses (257, 261, ...) and
/// the ends.
fn normalized_codes(bits: u32) -> Vec<u32> {
    if bits <= 10 {
        return (0..1 << bits).collect();
    }
    let edges = [1, 2, 3, 255, 256, 257, 261, 265, 269, 511, 512, 1000, 0x7FFF, 0x8000, 0x8001, 0xFFFE];
    (0..=0xFFFF).step_by(257).chain(edges).collect()
}

/// The float `v_cvt_f32_f64` gives for `x`: Rust's rounding (to nearest
/// even), with results below the smallest normal float flushed to signed
/// zero as the console's default float mode does.
fn cvt_f32_f64_reference(x: f64) -> u32 {
    let r = x as f32;
    if r.is_subnormal() || (r == 0.0 && x != 0.0) {
        r.to_bits() & 0x8000_0000
    } else {
        r.to_bits()
    }
}

/// The lane `v_permlane16_b32` (`cross` false) or `v_permlanex16_b32` reads
/// for `lane`, with selector dwords `s1` (lanes 0-7 of a row) and `s2`.
fn permlane_source(lane: u32, s1: u32, s2: u32, cross: bool) -> u32 {
    let sel = u64::from(s1) | u64::from(s2) << 32;
    let pick = (sel >> (4 * (lane & 15)) & 15) as u32;
    let row = if cross {
        (lane & !31) | ((lane & 16) ^ 16)
    } else {
        lane & !15
    };
    row | pick
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

const S_BARRIER: u32 = 0xBF8A_0000;

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_MIN_F32: u8 = 0x12;
const DS_MAX_F32: u8 = 0x13;
const DS_MIN_RTN_F32: u8 = 0x32;
const DS_MAX_RTN_F32: u8 = 0x33;
const DS_READ_B32: u8 = 0x36;

/// What RDNA2 writes for `ds_max_f32` / `ds_min_f32` over a dword holding
/// `old` with `data` (bits), as measured on gfx1036 with f32 denormals kept:
/// an sNaN, data first, then memory, is written quieted; otherwise a NaN
/// loses to a number (two quiet NaNs give the data); numbers compare as
/// floats with -0 below +0.
fn lds_min_max_ref(old: u32, data: u32, max: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let snan = |x: u32| nan(x) && x & 0x40_0000 == 0;
    let key = |x: u32| {
        let b = x as i32;
        if b < 0 { b ^ 0x7FFF_FFFF } else { b }
    };
    if snan(data) {
        return data | 0x40_0000;
    }
    if snan(old) {
        return old | 0x40_0000;
    }
    match (nan(data), nan(old)) {
        (true, true) | (false, true) => return data,
        (true, false) => return old,
        (false, false) => {}
    }
    let take = if max {
        key(data) > key(old)
    } else {
        key(data) < key(old)
    };
    if take { data } else { old }
}
