//! Ours: a console wave of 64 that one host subgroup stands for (a pixel
//! program: a fragment subgroup has at most 32 invocations and meets no
//! other). The subgroup is the wave's lower half and the upper half holds
//! no work, as in any partly filled console wave; what the program reads
//! of the wave's 64 lanes must be what that wave gives (lift_hi.rs): the
//! upper dword of a mask made of EXEC and compares is 0, of an inverted
//! mask or a constant -1 all ones, a count or search of such a mask is
//! exact, and none of it is listed as an approximation. A compare made
//! where every lane runs (a whole-wave section) has an upper dword nothing
//! here knows: that stays listed.
//!
//! The model is origin/main's for the same programs (`emit/lower.rs`: one
//! host wave per console wave on a pixel shader, "the lanes that are
//! here"); the cases are ours.

#![allow(clippy::all)]

use super::*;

const EXEC: u8 = 126;

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

const S_MOV_B64: u8 = 0x04;
const S_BCNT1_I32_B64: u8 = 0x10;
const S_FF1_I32_B64: u8 = 0x14;
const S_XOR_B64: u8 = 0x13;
const V_CMP_GT_U32_VCC_S0_V0: u32 = 0x7D88_0000;
const V_CMPX_GT_U32_S0_V0: u32 = 0x7DA8_0000;

/// A wave64 pixel program: the lane's number (x + 8 y, as the interpreter
/// places the lanes) in v0, v7 = v0 ^ s1, `body`, v0 exported as mrt0.
fn pixel(body: &[u32]) -> NativeProgram {
    let mut words = vec![
        vop1(7, 2, vgpr(0)),       // v_cvt_u32_f32 v2, v0 (x)
        vop1(7, 3, vgpr(1)),       // v_cvt_u32_f32 v3, v1 (y)
        vop2(0x1A, 3, 128 + 3, 3), // v_lshlrev_b32 v3, 3, v3
        vop2(0x25, 0, vgpr(3), 2), // v_add_nc_u32 v0, v3, v2
        vop2(0x1D, 7, sgpr(1), 0), // v_xor_b32 v7, s1, v0
    ];
    words.extend_from_slice(body);
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

/// What the lanes that exported wrote as mrt0.x (one value: they must
/// agree; none if none did), and the approximations the translation lists.
fn run(native: &NativeProgram, names: &[&str], inputs: [u32; 2], live: u32) -> Result<(Option<u32>, Vec<&'static str>), String> {
    // (the words are the instructions meant)
    let p = crate::ir::build(&native.code, true, native.vgprs).map_err(|e| format!("{e}"))?;
    for name in names {
        if !p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == *name) {
            return Err(format!("no {name} in the program"));
        }
    }
    let (f, iface) = crate::vk::thread_ir_native(native).map_err(|e| format!("translation: {e}"))?;
    let mut m = Machine::new(&f, 32, Memory::default());
    m.live = (0..32).map(|l| live >> l & 1 != 0).collect();
    let at = crate::vk::lift::push_window(HwStage::Ps) as usize;
    m.push[at..at + 2].copy_from_slice(&inputs);
    match m.run() {
        Stop::Returned => {}
        s => return Err(format!("{s:?}")),
    }
    let mut vals: Vec<u32> = m.outputs.iter().filter(|(k, _)| k.0 == 0 && k.1 == 0).map(|(_, &x)| x as u32).collect();
    vals.sort_unstable();
    vals.dedup();
    if vals.len() > 1 {
        return Err(format!("the lanes disagree: {vals:x?}"));
    }
    Ok((vals.first().copied(), iface.approximate.iter().map(|a| a.note).collect()))
}

/// Subgroups: whole, 12 lanes, two quads apart, one lane.
const LIVE: [u32; 4] = [u32::MAX, 0x0000_0FFF, 0x0000_0F0F, 1 << 17];

/// EXEC's upper dword, and that of EXEC narrowed by a compare, read as
/// numbers: 0, exactly (no lane of the upper half holds work).
#[test]
fn the_upper_dword_of_exec_and_of_a_compare_is_zero() {
    let exec = pixel(&[sop1(S_MOV_B64, 2, EXEC), vop1(0x001, 0, sgpr(3))]); // s_mov_b64 s[2:3], exec; v_mov_b32 v0, s3
    let compare = pixel(&[V_CMP_GT_U32_VCC_S0_V0, vop1(0x001, 0, sgpr(107))]); // v_mov_b32 v0, vcc_hi
    for live in LIVE {
        assert_eq!(run(&exec, &["s_mov_b64"], [0, 0], live), Ok((Some(0), vec![])), "exec, lanes {live:#x}");
        assert_eq!(run(&compare, &["v_cmp_gt_u32"], [20, 0], live), Ok((Some(0), vec![])), "compare, lanes {live:#x}");
    }
}

/// `s_xor_b64 s[2:3], vcc, -1` (the compiler's NOT of a mask) and
/// `s_mov_b64 s[2:3], -1`: the console's upper dword is all ones, whatever
/// lanes there are. It read 0 before.
#[test]
fn the_upper_dword_of_an_inverted_mask_is_all_ones() {
    let inverted = pixel(&[V_CMP_GT_U32_VCC_S0_V0, sop2(S_XOR_B64, 2, 106, 193), vop1(0x001, 0, sgpr(3))]);
    let ones = pixel(&[sop1(S_MOV_B64, 2, 193), vop1(0x001, 0, sgpr(3))]);
    for live in LIVE {
        assert_eq!(run(&inverted, &["s_xor_b64"], [20, 0], live), Ok((Some(u32::MAX), vec![])), "inverted, lanes {live:#x}");
        assert_eq!(run(&ones, &["s_mov_b64"], [0, 0], live), Ok((Some(u32::MAX), vec![])), "-1, lanes {live:#x}");
    }
}

/// `s_bcnt1_i32_b64` of EXEC narrowed to the lanes below s0 counts the
/// lanes that are there and below s0; of the mask inverted, the other 64.
#[test]
fn a_count_of_a_mask_is_the_waves() {
    let count = pixel(&[V_CMPX_GT_U32_S0_V0, sop1(S_BCNT1_I32_B64, 2, EXEC), vop1(0x001, 0, sgpr(2))]);
    let rest = pixel(&[
        V_CMP_GT_U32_VCC_S0_V0,
        sop2(S_XOR_B64, 4, 106, 193), // s_xor_b64 s[4:5], vcc, -1
        sop1(S_BCNT1_I32_B64, 2, 4),  // s_bcnt1_i32_b64 s2, s[4:5]
        vop1(0x001, 0, sgpr(2)),
    ]);
    for (live, below) in LIVE.into_iter().flat_map(|l| [32u32, 20, 7].map(|b| (l, b))) {
        let n = (0..32).filter(|l| live >> l & 1 != 0 && *l < below).count() as u32;
        let want = (n > 0).then_some(n);
        assert_eq!(run(&count, &["s_bcnt1_i32_b64"], [below, 0], live), Ok((want, vec![])), "lanes {live:#x} below {below}");
        assert_eq!(run(&rest, &["s_bcnt1_i32_b64", "s_xor_b64"], [below, 0], live), Ok((Some(64 - n), vec![])), "the rest, lanes {live:#x} below {below}");
    }
}

/// `s_ff1_i32_b64` of EXEC and `v_readlane_b32` of the lane it found (the
/// compiler's "first active lane's value", PS_MapFires, PS_ResolveDeferred):
/// the first lane that is there, and its value.
#[test]
fn the_first_lane_of_a_mask_and_its_value() {
    let first = pixel(&[
        V_CMPX_GT_U32_S0_V0,
        sop1(S_FF1_I32_B64, 2, EXEC), // s_ff1_i32_b64 s2, exec
        0xD760_0003,
        2 << 9 | 263, // v_readlane_b32 s3, v7, s2
        vop1(0x001, 0, sgpr(3)),
    ]);
    for (live, below, key) in LIVE.into_iter().flat_map(|l| [(32u32, 0x15u32), (20, 0x2A), (7, 0)].map(|x| (l, x.0, x.1))) {
        let want = (0..32).find(|l| live >> l & 1 != 0 && *l < below).map(|l| l ^ key);
        assert_eq!(run(&first, &["s_ff1_i32_b64", "v_readlane_b32"], [below, key], live), Ok((want, vec![])), "lanes {live:#x} below {below}");
    }
}

/// A compare made inside a whole-wave section (every lane of the wave
/// runs, the upper half's on what they happen to hold): its upper dword as
/// a number is not known, and stays listed.
#[test]
fn a_compare_where_every_lane_runs_stays_listed() {
    let section = pixel(&[
        0xBE84_25C1, // s_or_saveexec_b64 s[4:5], -1
        V_CMP_GT_U32_VCC_S0_V0,
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        vop1(0x001, 0, sgpr(107)),
    ]);
    let (_, notes) = run(&section, &["s_or_saveexec_b64", "v_cmp_gt_u32"], [20, 0], u32::MAX).expect("runs");
    assert_eq!(notes, ["a wave64 mask's high half read as a number"]);
}

/// The lower dword of an inverted mask has the bits of the lanes the
/// subgroup lacks (they hold no work: a compare gave them 0).
#[test]
fn the_lower_dword_of_an_inverted_mask_has_the_lanes_that_are_not_there() {
    let inverted = pixel(&[V_CMP_GT_U32_VCC_S0_V0, sop2(S_XOR_B64, 2, 106, 193), vop1(0x001, 0, sgpr(2))]);
    for (live, below) in LIVE.into_iter().flat_map(|l| [32u32, 20, 7].map(|b| (l, b))) {
        let set = (0..32).filter(|l| live >> l & 1 != 0 && *l < below).fold(0u32, |m, l| m | 1 << l);
        assert_eq!(run(&inverted, &["s_xor_b64"], [below, 0], live), Ok((Some(!set), vec![])), "lanes {live:#x} below {below}");
    }
}
