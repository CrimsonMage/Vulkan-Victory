//! The packed operations of AMD's compute compiler (`lift_pk.rs`), as
//! ProjectLeonida's translator tests state them (GPL-3.0;
//! `tests/semantic/test_alu_dot4_i8.py`, `test_alu_dot_pk_fmac.py`,
//! `test_alu_pk_int16.py`): the words measured on RDNA2 hardware (gfx1036,
//! their `tools/hw_oracle`) are theirs, the programs and the harness ours.

use super::*;

/// VOP3P `op vdst, src0, src1, src2` with its half selectors (plain: 0, 7).
const fn vop3p(op: u8, vdst: u8, src: [u16; 3], opsel: u8, opsel_hi: u8) -> [u32; 2] {
    [
        0x33 << 26 | (op as u32) << 16 | (opsel_hi as u32 & 4) << 12 | (opsel as u32 & 7) << 11 | vdst as u32,
        (opsel_hi as u32 & 3) << 27 | (src[0] as u32 & 0x1FF) | (src[1] as u32 & 0x1FF) << 9 | (src[2] as u32 & 0x1FF) << 18,
    ]
}

const DOT4: u8 = 0x16;
const DOT4C: u8 = 0x0D;
const PK_ADD_U16: u8 = 0x0A;
const PK_SUB_U16: u8 = 0x0B;
const PK_FMAC_F16: u8 = 0x3C;
/// A literal follows the instruction.
const LITERAL: u16 = 255;
/// Inline integer `n` (0 to 64).
const fn int(n: u16) -> u16 {
    128 + n
}

/// (S0, S1, addend, result): `v_dot4_i32_i8 v8, v1, v2, v3`, and
/// `v_dot4c_i32_i8 v10, v1, v2` with v10 holding the addend, either source
/// order, measured on gfx1036.
const DOT4_MEASURED: [[u32; 4]; 10] = [
    [0x0102_0304, 0x0506_0708, 0x0000_0000, 0x0000_0046],
    [0x0102_0304, 0x0506_0708, 0x0000_03E8, 0x0000_042E],
    [0xFF80_017F, 0x7F7F_80FF, 0x0000_0000, 0xFFFF_BF02],
    [0xFF80_017F, 0x7F7F_80FF, 0x0001_0000, 0x0000_BF02],
    [0x8080_8080, 0x8080_8080, 0x0000_0000, 0x0001_0000],
    [0x8080_8080, 0x7F7F_7F7F, 0x0000_0005, 0xFFFF_0205],
    // no saturation without clamp: wraps upward, downward
    [0x7F7F_7F7F, 0x7F7F_7F7F, 0x7FFF_FFFF, 0x8000_FC03],
    [0x8080_8080, 0x7F7F_7F7F, 0x8000_0000, 0x7FFF_0200],
    [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0x0000_0003],
    [0x0000_0000, 0x1234_5678, 0xDEAD_BEEF, 0xDEAD_BEEF],
];

/// (S0, S1, addend, result) with op_sel 2 and op_sel_hi 5 (the raw words
/// 0xCC165008 0x0C0E0501), measured on gfx1036: the two halves of the
/// second source exchanged.
const DOT4_OP_SEL_MEASURED: [[u32; 4]; 5] = [
    [0x0403_0201, 0x0807_0605, 0x0000_0000, 62],
    [0x0403_0201, 0x0807_0605, 0x0000_03E8, 1062],
    [0x7F01_FE80, 0x02FD_7F81, 0x0000_0000, 16382],
    [0x1122_3344, 0x5566_7788, 0x0001_0000, 74750],
    [0xFFFF_FFFF, 0x0102_0304, 0x0000_0007, 0xFFFF_FFFD],
];

fn dot4_ref(a: u32, b: u32, addend: u32) -> u32 {
    (0..4).fold(addend, |sum, i| {
        let byte = |x: u32| i32::from((x >> (8 * i)) as u8 as i8);
        sum.wrapping_add((byte(a) * byte(b)) as u32)
    })
}

fn cases(rows: &[[u32; 4]]) -> Vec<[u32; 4]> {
    rows.iter().map(|r| [r[0], r[1], r[2], 0]).collect()
}

#[test]
fn dot4_of_signed_bytes_matches_rdna2() {
    let plain = vop3p(DOT4, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 7);
    // v_mov_b32 v0, v3 first: the accumulating form adds to its destination
    let acc = |s0: u8, s1: u8| [vop1(0x001, 0, vgpr(3)), vop2(DOT4C, 0, vgpr(s0), s1)];
    for (what, body) in [("v_dot4_i32_i8", plain.to_vec()), ("v_dot4c_i32_i8", acc(1, 2).to_vec()), ("v_dot4c_i32_i8, sources swapped", acc(2, 1).to_vec())] {
        run_cases(&build_valu_cs(&body), &cases(&DOT4_MEASURED), |backend, i, inputs, got| {
            assert_eq!(dot4_ref(inputs[0], inputs[1], inputs[2]), DOT4_MEASURED[i][3], "the reference, case {i}");
            assert_eq!(got, DOT4_MEASURED[i][3], "{backend}: {what} {inputs:08x?}");
        });
    }
}

#[test]
fn dot4_reads_its_factors_through_the_half_selectors() {
    let body = vop3p(DOT4, 0, [vgpr(1), vgpr(2), vgpr(3)], 2, 5);
    assert_eq!(body, [0xCC16_5000, 0x0C0E_0501], "their raw words but for the destination");
    run_cases(&build_valu_cs(&body), &cases(&DOT4_OP_SEL_MEASURED), |backend, i, inputs, got| {
        assert_eq!(got, DOT4_OP_SEL_MEASURED[i][3], "{backend}: {inputs:08x?}");
    });
}

/// The forms the upscaler's programs have with a constant factor. A
/// constant is its 32-bit value under the selectors (measured by them for
/// the packed 16-bit integer operations; for the dot product it is the
/// rule, not a measurement): the inline 1 with both selectors turned is
/// 0x0001_0000 (byte 2 alone), the inline 10 with the high half reading
/// the low one 0x000A_000A.
#[test]
fn dot4_constant_factors_are_dwords_under_the_selectors() {
    let forms: [(&str, Vec<u32>, u32); 4] = [
        ("1, both selectors turned", vop3p(DOT4, 0, [vgpr(1), int(1), vgpr(3)], 2, 5).to_vec(), 0x0001_0000),
        ("10, the low half twice", vop3p(DOT4, 0, [vgpr(1), int(10), vgpr(3)], 0, 5).to_vec(), 0x000A_000A),
        // 2.0f (inline 244) is 0x4000_0000: its high half as the low one
        ("2.0, both selectors turned", vop3p(DOT4, 0, [vgpr(1), 244, vgpr(3)], 2, 5).to_vec(), 0x0000_4000),
        ("a literal", [vop3p(DOT4, 0, [vgpr(1), LITERAL, vgpr(3)], 0, 7).to_vec(), vec![0x7F80_FF05]].concat(), 0x7F80_FF05),
    ];
    let inputs = [[0x0403_0201, 0, 100, 0], [0x80FF_7F81, 0, 0xFFFF_FFF0, 0], [0x1122_3344, 0, 0, 0]];
    for (what, body, factor) in forms {
        run_cases(&build_valu_cs(&body), &inputs, |backend, _, inputs, got| {
            assert_eq!(got, dot4_ref(inputs[0], factor, inputs[2]), "{backend}: {what} {inputs:08x?}");
        });
    }
}

/// Measured by them: `clamp` saturates and the negate bits flip sign bits
/// of the operands. No program seen has either; refused, not guessed.
#[test]
fn dot4_with_clamp_or_negation_is_refused() {
    let plain = vop3p(DOT4, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 7);
    for (what, words) in [("clamp", [plain[0] | 1 << 15, plain[1]]), ("neg_lo", [plain[0], plain[1] | 1 << 29]), ("neg_hi", [plain[0] | 1 << 8, plain[1]])] {
        let e = crate::vk::thread_ir_native(&build_valu_cs(&words).native).err();
        assert!(matches!(e, Some(crate::ShaderError::Unsupported { what: "a saturating or negated dot product", .. })), "{what}: {e:?}");
    }
    // the addend read by halves
    let halves = vop3p(DOT4, 0, [vgpr(1), vgpr(2), vgpr(3)], 4, 7);
    assert!(crate::vk::thread_ir_native(&build_valu_cs(&halves).native).is_err());
}

/// (S0, S1, D before, D): `v_pk_fmac_f16` measured on gfx1036 in the
/// default mode (round to nearest even, f16 denormals kept).
const PK_FMAC_MEASURED: [[u32; 4]; 22] = [
    [0x3C01_4000, 0x3C03_4200, 0xBC00_3C00, 0x1C01_4700],
    [0x3C01_3C01, 0x3C01_3C03, 0xBC02_BC00, 0x0010_1C01],
    [0x3C01_3C01, 0x3E00_3C01, 0x8001_BC02, 0x3E01_0010],
    [0x3E01_3C01, 0x3E01_3E00, 0x0000_8001, 0x4082_3E01],
    [0xBE01_3E01, 0x3E01_3E01, 0x8000_0000, 0xC082_4082],
    [0x0000_BE01, 0x3C00_3E01, 0x8000_8000, 0x0000_C082],
    [0x8000_0000, 0x3C00_3C00, 0x8000_8000, 0x8000_0000],
    [0x3C00_8000, 0xBC00_3C00, 0x3C00_8000, 0x0000_8000],
    [0x8000_3C00, 0x3C00_BC00, 0x0000_3C00, 0x0000_0000],
    [0x7C00_8000, 0x4000_3C00, 0x3C00_0000, 0x7C00_0000],
    [0x7C00_7C00, 0xC000_4000, 0x7C00_3C00, 0xFE00_7C00],
    [0x7C00_7C00, 0x0000_C000, 0x3C00_7C00, 0xFE00_FE00],
    [0x7BFF_7C00, 0x4000_0000, 0x0000_3C00, 0x7C00_FE00],
    [0x7E55_7BFF, 0x3C00_4000, 0x3C00_0000, 0x7E55_7C00],
    [0x7C01_7E55, 0x3C00_3C00, 0x3C00_3C00, 0x7E01_7E55],
    [0x3C00_7C01, 0x3C00_3C00, 0x7D23_3C00, 0x7F23_7E01],
    [0x0400_3C00, 0x3800_3C00, 0x0000_7D23, 0x0200_7F23],
    [0x0001_0400, 0x4400_3800, 0x0000_0000, 0x0004_0200],
    [0x0003_0001, 0x3800_4400, 0x0000_0000, 0x0002_0004],
    [0x3C00_0003, 0x3C00_3800, 0x6400_0000, 0x6401_0002],
    [0x4400_3C00, 0xC200_3C00, 0x4900_6400, 0xC000_6401],
    [0x4000_4400, 0x4200_C200, 0x3C00_4900, 0x4700_C000],
];

/// Each half of D is S0 × S1 + D of that half, rounded once. A NaN result
/// is a NaN here (which one is the card's to say).
#[test]
fn pk_fmac_f16_accumulates_each_half_fused() {
    let body = [vop1(0x001, 0, vgpr(3)), vop2(PK_FMAC_F16, 0, vgpr(1), 2)];
    run_cases(&build_valu_cs(&body), &cases(&PK_FMAC_MEASURED), |backend, i, inputs, got| {
        for half in [0, 16] {
            let (g, want) = ((got >> half) as u16, (PK_FMAC_MEASURED[i][3] >> half) as u16);
            if is_f16_nan(want) {
                assert!(is_f16_nan(g), "{backend}: case {i} {inputs:08x?}: {g:#06x} is no NaN");
            } else {
                assert_eq!(g, want, "{backend}: case {i} {inputs:08x?} bit {half}");
            }
        }
    });
}

/// Their rule for the packed 16-bit integer pairs: result half 0 takes
/// each source's half by op_sel, half 1 by op_sel_hi; a constant is its
/// 32-bit value under them.
fn pk_ref(sub: bool, a: u32, b: u32, opsel: u8, opsel_hi: u8) -> u32 {
    let half = |x: u32, high: bool| (x >> if high { 16 } else { 0 }) & 0xFFFF;
    let one = |sel: u8| {
        let (x, y) = (half(a, sel & 1 != 0), half(b, sel & 2 != 0));
        (if sub { x.wrapping_sub(y) } else { x.wrapping_add(y) }) & 0xFFFF
    };
    one(opsel) | one(opsel_hi) << 16
}

#[test]
fn pk_add_and_sub_u16_wrap_each_half() {
    let inputs = [[0x0001_FFFF, 0x0001_0001, 0, 0], [0x8000_7FFF, 0x8000_8001, 0, 0], [0x1234_5678, 0xFFFF_0008, 0, 0], [0, 0x0001_0000, 0, 0]];
    for (sub, op) in [(false, PK_ADD_U16), (true, PK_SUB_U16)] {
        for (opsel, opsel_hi) in [(0, 3), (2, 1), (0, 2), (3, 0)] {
            let body = vop3p(op, 0, [vgpr(1), vgpr(2), 0x80], opsel, opsel_hi | 4);
            run_cases(&build_valu_cs(&body), &inputs, |backend, _, inputs, got| {
                assert_eq!(got, pk_ref(sub, inputs[0], inputs[1], opsel, opsel_hi), "{backend}: sub {sub} op_sel {opsel} op_sel_hi {opsel_hi} {inputs:08x?}");
            });
        }
    }
    // constants as the upscaler's programs have them: the inline 1 added to
    // the high half alone, a literal with a half each, the inline 7 first
    let forms: [(Vec<u32>, u32, u8, u8, bool); 3] = [
        (vop3p(PK_ADD_U16, 0, [vgpr(1), int(1), 0x80], 2, 1 | 4).to_vec(), 1, 2, 1, false),
        ([vop3p(PK_ADD_U16, 0, [vgpr(1), LITERAL, 0x80], 0, 3 | 4).to_vec(), vec![0xFFFF_0008]].concat(), 0xFFFF_0008, 0, 3, false),
        (vop3p(PK_ADD_U16, 0, [int(7), vgpr(1), 0x80], 0, 3 | 4).to_vec(), 7, 0, 3, true),
    ];
    for (body, constant, opsel, opsel_hi, first) in forms {
        run_cases(&build_valu_cs(&body), &inputs, |backend, _, inputs, got| {
            let (a, b) = if first { (constant, inputs[0]) } else { (inputs[0], constant) };
            assert_eq!(got, pk_ref(false, a, b, opsel, opsel_hi), "{backend}: {body:08x?} {inputs:08x?}");
        });
    }
}

/// AMD's compute compiler keeps numbers in vcc_lo and vcc_hi (two words
/// of a V#, weights) and moves a pair in with `s_mov_b64`, in waves of 32
/// as well (the upscaler's network passes do: 8 of its programs). They
/// stay the two numbers; as a lane mask's move the upper one had no value
/// ("a value read before it is defined"). For waves of 64:
/// `oracle/wave64.rs`.
#[test]
fn a_pair_of_numbers_moved_into_vcc_stays_numbers() {
    let body = [
        0xBEEA_0402,                  // s_mov_b64 vcc, s[2:3]
        vop1(0x001, 0, sgpr(106)),    // v_mov_b32 v0, vcc_lo
        vop2(0x25, 0, sgpr(107), 0),  // v_add_nc_u32 v0, vcc_hi, v0
    ];
    let inputs = [[0, 0, 5, 7], [0, 0, 0xFFFF_FFFF, 2], [0, 0, 0x8000_0001, 0x7FFF_FFFF], [0, 0, 0, 0]];
    run_cases(&build_valu_cs(&body), &inputs, |backend, _, inputs, got| {
        assert_eq!(got, inputs[2].wrapping_add(inputs[3]), "{backend}: {inputs:08x?}");
    });
}

/// `image_store v0, v2, s[24:31] dmask:0x3 dim:2D a16 d16` (the upscaler's
/// post pass stores one 16-bit component so): the coordinates are the two
/// halves of v2, the data the two halves of v0. A float view takes them as
/// f16, an integer view (T# word 7 bit 24) as the 16 bits. The T# is the
/// table's second descriptor pair, its words 6 and 7 the host's.
#[test]
fn a_d16_image_store_takes_its_data_by_halves() {
    // the check word of the host's T#s (`docs/vulkan-abi.md` section 5.2)
    const T_CHECK: u32 = 0x7A3C_5E91;
    let body = [
        0xF40C_0602, // s_load_dwordx8 s[24:31], s[4:5], 0x20
        0xFA00_0020,
        vop1(0x001, 0, sgpr(0)), // v_mov_b32 v0, s0
        vop1(0x001, 2, sgpr(1)), // v_mov_b32 v2, s1
        0xF020_0308,             // image_store, dmask 3, 2D
        0xC006_0002,             // a16 d16, s[24:31], v0, v2
    ];
    let (f, _) = crate::vk::thread_ir_native(&build_valu_cs(&body).native).expect("translation");
    for (word7, want) in [(5u32, [1.5f32.to_bits(), (-2.0f32).to_bits()]), (5 | 1 << 24, [0x3E00, 0xC000])] {
        let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
        mem.put(0x1000, &[0x2000, 0, 0x1_0000, 0x3000_0FAC]);
        mem.put(0x1020, &[0, 0, 0, 0, 0, 0, word7 ^ T_CHECK, word7]);
        let mut m = Machine::new(&f, 32, mem);
        // data: 1.5 and -2.0 as f16; coordinates (7, 9)
        m.push[4..6].copy_from_slice(&[0xC000_3E00, 0x0009_0007]);
        m.push[8] = 0x1000;
        assert_eq!(m.run(), Stop::Returned);
        let (_, args) = m.image_stores.first().expect("a store");
        assert_eq!(args[0][0] as u32, word7, "the T#'s word 7");
        assert_eq!([args[1][0] as u32, args[1][1] as u32], [7, 9], "the coordinates");
        assert_eq!([args[2][0] as u32, args[2][1] as u32], want, "view word 7 {word7:#x}");
    }
}
