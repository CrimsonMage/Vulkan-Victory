//! Whole-wave sections on host subgroups that lack lanes, after origin/main
//! (fxceless/victory-fork 6742906, GPL-3.0): `emit/wave.rs` (the section's
//! simulation, in our tree as `crate::wave`) and `emit/lower.rs` `shape`
//! (each value a section leaves as a wave operation over the host's active
//! lanes), with the compiler's own instruction sequence from
//! `wave::WAVE_OR_V7_TO_S7` and from the tiled-lighting pixel programs
//! (minimum and maximum mask word, OR of a word's bits).
//!
//! The console's wave has all its lanes: between `s_or_saveexec -1` and the
//! restore of EXEC every lane runs, the inactive ones with the operation's
//! identity. A host subgroup may be partly filled (pixels at a triangle's
//! edge, a small triangle): the interpreter's `live` says which
//! invocations are there; one that is not there computes nothing, is in no
//! ballot or reduction, and a shuffle from it reads a poison. The result
//! must be the operation over the lanes that are there and in EXEC.

#![allow(clippy::all, unused_variables)]

use super::*;

/// The section for a wave of 32: `op` over the lanes of `EXEC ? v7 :
/// identity` into s7 (identity: an inline constant code).
fn section32(op: u32, identity: u32) -> Vec<u32> {
    let o = op << 25;
    vec![
        0xBE84_3DC1, // s_or_saveexec_b32 s4, -1
        0xD501_0001,
        0x0012_0E00 | identity, // v_cndmask_b32 v1, identity, v7, s4
        o | 0x0002_02E9,
        0xDE54_C101, // v_<op> v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
        o | 0x0002_02E9,
        0x9360_1201, // v_<op> v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
        o | 0x0002_02E9,
        0x0009_2401, // v_<op> v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
        o | 0x0002_02FA,
        0xFF09_4001, // v_<op> v1, v1, v1 row_mirror bound_ctrl:1
        0xD778_0002,
        0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
        o | 0x0002_0302, // v_<op> v1, v2, v1
        0xD760_0007,
        0x0001_0101, // v_readlane_b32 s7, v1, 0
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
    ]
}

/// The section for a wave of 64 (`wave::WAVE_OR_V7_TO_S7` with `op` for
/// its OR).
fn section64(op: u32, identity: u32) -> Vec<u32> {
    let mut w: Vec<u32> = crate::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | op << 25 } else { w }).collect();
    w[2] = 0x0012_0E00 | identity;
    w
}

/// Lane l's value v7 from its number (v0) and s1; the lanes at or past s0
/// leave EXEC; the section; s7 stored by the lanes in EXEC.
fn program(section: &[u32], bits: bool) -> Vec<u32> {
    let mut body = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        if bits {
            vop2(0x1A, 7, vgpr(0), 5) // v_lshlrev_b32 v7, v0, v5: a bit a lane
        } else {
            vop2(0x1D, 7, sgpr(1), 0) // v_xor_b32 v7, s1, v0
        },
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0
    ];
    body.extend_from_slice(section);
    body.push(vop1(0x001, 0, sgpr(7))); // v_mov_b32 v0, s7
    body
}

/// Dword 0 of the output after a run with the invocations of `live`
/// there; none if nothing was stored.
fn run_live(shader: &HostShader, inputs: [u32; 4], live: u32) -> Result<Option<u32>, String> {
    let (f, _) = crate::vk::thread_ir_native(&shader.native).map_err(|e| format!("translation: {e}"))?;
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    mem.put(0x1000, &[0x2000, 0, 0x1_0000, 0x3000_0FAC]);
    let mut m = Machine::new(&f, 32, mem);
    m.live = (0..32).map(|l| live >> l & 1 != 0).collect();
    m.push[4..8].copy_from_slice(&inputs);
    m.push[8] = 0x1000;
    match m.run() {
        Stop::Returned => Ok(m.mem.stored.get(&0x2000).copied()),
        s => Err(format!("{s:?}")),
    }
}

/// (name, VOP2 opcode, identity's inline constant, identity, fold).
type Case = (&'static str, u32, u32, u32, fn(u32, u32) -> u32);

const CASES: [Case; 6] = [
    ("v_or_b32", 0x1C, 0x80, 0, |a, b| a | b),
    ("v_xor_b32", 0x1D, 0x80, 0, |a, b| a ^ b),
    ("v_and_b32", 0x1B, 0xC1, u32::MAX, |a, b| a & b),
    ("v_min_u32", 0x13, 0xC1, u32::MAX, |a, b| a.min(b)),
    ("v_max_u32", 0x14, 0x80, 0, |a, b| a.max(b)),
    ("v_add_nc_u32", 0x25, 0x80, 0, |a, b| a.wrapping_add(b)),
];

/// Host subgroups: whole, a prefix of 12 lanes (rows 8-11 reach lane 0 only
/// through lanes 12-15), one quad away from lane 0, two quads apart, every
/// other quad, one lane, all but lane 0.
const LIVE: [u32; 7] = [u32::MAX, 0x0000_0FFF, 0x0000_00F0, 0x0000_0F0F, 0x0F0F_0F0F, 1 << 17, !1];

/// A pixel program of a 64-lane console wave on one host subgroup: the
/// lane's number from its pixel (x + 8 y, as the interpreter places the
/// lanes), then [`program`], s7 exported as mrt0 by the lanes in EXEC.
fn pixel64(section: &[u32], bits: bool) -> NativeProgram {
    let mut words = vec![
        vop1(7, 2, vgpr(0)),       // v_cvt_u32_f32 v2, v0 (x)
        vop1(7, 3, vgpr(1)),       // v_cvt_u32_f32 v3, v1 (y)
        vop2(0x1A, 3, 128 + 3, 3), // v_lshlrev_b32 v3, 3, v3
        vop2(0x25, 0, vgpr(3), 2), // v_add_nc_u32 v0, v3, v2
    ];
    words.extend(program(section, bits));
    words.extend([0xF800_180F, 0, S_ENDPGM]); // exp mrt0, v0, v0, v0, v0 done vm
    NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 2, base_register: 0, user_data: 0, slot: 0 }],
        used: Vec::new(),
        // SPI_PS_INPUT_ADDR: POS_X_FLOAT_ENA, POS_Y_FLOAT_ENA
        io: StageIo::Ps { inputs: Vec::new(), input_addr: 3 << 8, db_shader_control: 0, patches: Vec::new() },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// What every lane that exported wrote as mrt0.x (none if none did; an
/// error if they differ).
fn run_pixel(native: &NativeProgram, inputs: [u32; 2], live: u32) -> Result<Option<u32>, String> {
    let (f, _) = crate::vk::thread_ir_native(native).map_err(|e| format!("translation: {e}"))?;
    let mut m = Machine::new(&f, 32, Memory::default());
    m.live = (0..32).map(|l| live >> l & 1 != 0).collect();
    let at = crate::vk::lift::push_window(HwStage::Ps) as usize;
    m.push[at..at + 2].copy_from_slice(&inputs);
    match m.run() {
        Stop::Returned => {}
        s => return Err(format!("{s:?}")),
    }
    let mut vals: Vec<u32> = m.outputs.iter().filter(|(k, _)| k.0 == 0 && k.1 == 0).map(|(_, &x)| x as u32).collect();
    vals.dedup();
    match vals.len() {
        0 => Ok(None),
        1 => Ok(Some(vals[0])),
        _ => Err(format!("the lanes disagree: {vals:x?}")),
    }
}

/// One operation and value pattern over every set of lanes: the failures.
fn check_one(wave64: bool, case: Case, bits: bool) -> Vec<String> {
    let (name, op, code, identity, fold) = case;
    let shader = build_cs(&program(&section32(op, code), bits), 32, 0);
    let pixel = pixel64(&section64(op, code), bits);
    let mut failed = Vec::new();
    for (live, (below, s1)) in LIVE.into_iter().flat_map(|l| [(32u32, 0x15u32), (20, 0x2A), (7, 0)].map(|x| (l, x))) {
        let value = |l: u32| if bits { 1u32 << l } else { l ^ s1 };
        let mut lanes = (0..32).filter(|l| live >> l & 1 != 0 && *l < below).peekable();
        let want = lanes.peek().is_some().then(|| lanes.fold(identity, |a, l| fold(a, value(l))));
        let got = if wave64 { run_pixel(&pixel, [below, s1], live) } else { run_live(&shader, [below, s1, 0, 0], live) };
        if got != Ok(want) {
            failed.push(format!("{name} (bits {bits}), lanes {live:#010x} below {below}: {got:x?}, want {want:x?}"));
        }
    }
    failed
}

fn check(wave64: bool) {
    let failed: Vec<String> = CASES.into_iter().flat_map(|c| [true, false].map(|bits| check_one(wave64, c, bits))).flatten().collect();
    assert!(failed.is_empty(), "{} failed:\n{}", failed.len(), failed.join("\n"));
}

/// A wave of 32 (the tiled-lighting pixel programs' light picking).
#[test]
fn a_wave_reduction_is_over_the_lanes_that_are_there() {
    check(false);
}

/// A console wave of 64 that one host subgroup stands for (a pixel
/// program): the section reads lane 32 too; the reduction is over the
/// subgroup, each lane once (a sum is not doubled).
#[test]
fn a_wave64_reduction_on_one_subgroup_counts_each_lane_once() {
    check(true);
}
