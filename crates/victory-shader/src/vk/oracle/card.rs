//! Ours (not theirs): what a translation computes with the card's floats
//! (`interp::card_floats`: a denormal operand of a float comparison is
//! zero on NVIDIA) against what the console computes (IEEE 754).

use super::*;

/// Bits a comparison must tell apart: the zeros, denormals (the stencil
/// plane's integers among them: CS_GenerateMaterialMasks tests them with
/// `v_cmp_eq_f32 0`), the smallest normals, ordinary numbers, the
/// infinities and NaNs.
const VALUES: [u32; 18] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x0000_0008,
    0x0000_0045,
    0x007F_FFFF,
    0x0080_0000,
    0x8000_0001,
    0x8000_0008,
    0x807F_FFFF,
    0x8080_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0x7F80_0001,
    0xFFC0_0000,
];

/// Runs `check` with IEEE floats, then with the card's.
fn on_both(mut check: impl FnMut(&str)) {
    check("IEEE floats");
    crate::vk::interp::card_floats(true);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check("the card's floats")));
    crate::vk::interp::card_floats(false);
    if let Err(p) = r {
        std::panic::resume_unwind(p);
    }
}

/// Every `v_cmp_*_f32` relation over every pair: the IEEE answer, with the
/// card's floats too (the comparison is made on the bits).
#[test]
fn f32_compares_take_denormals_as_numbers_on_the_card() {
    type Rel = fn(f32, f32) -> bool;
    // VOPC opcodes 0x00..=0x0f
    let rels: [(&str, Rel); 16] = [
        ("f", |_, _| false),
        ("lt", |a, b| a < b),
        ("eq", |a, b| a == b),
        ("le", |a, b| a <= b),
        ("gt", |a, b| a > b),
        ("lg", |a, b| a < b || a > b),
        ("ge", |a, b| a >= b),
        ("o", |a, b| !a.is_nan() && !b.is_nan()),
        ("u", |a, b| a.is_nan() || b.is_nan()),
        ("nge", |a, b| !(a >= b)),
        ("nlg", |a, b| !(a < b || a > b)),
        ("ngt", |a, b| !(a > b)),
        ("nle", |a, b| !(a <= b)),
        ("neq", |a, b| a != b),
        ("nlt", |a, b| !(a < b)),
        ("tru", |_, _| true),
    ];
    let inputs: Vec<[u32; 4]> = VALUES.iter().flat_map(|&a| VALUES.iter().map(move |&b| [a, b, 0, 0])).collect();
    for (opcode, (name, rel)) in rels.iter().enumerate() {
        // v_cmp_<rel>_f32_e64 s20, v1, v2; v_cndmask_b32_e64 v0, 0, 1, s20
        let mut body = vop3(opcode as u16, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        on_both(|floats| {
            run_cases(&shader, &inputs, |_, _, inp, got| {
                let want = u32::from(rel(f32::from_bits(inp[0]), f32::from_bits(inp[1])));
                assert_eq!(got, want, "{floats}: v_cmp_{name}_f32({:#010x}, {:#010x}) = {got}, want {want}", inp[0], inp[1]);
            });
        });
    }
}

/// The same with the modifiers of the VOP3 form: `abs` on the first
/// source, `neg` on the second (they move the sign bit of the bits).
#[test]
fn f32_compares_apply_abs_and_neg_to_the_bits() {
    // v_cmp_lt_f32_e64 s20, |v1|, -v2; v_cndmask_b32_e64 v0, 0, 1, s20
    let mut body = vop3_mod(0x001, 20, vgpr(1), vgpr(2), 0x80, 1, 2, 0, false).to_vec();
    body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = VALUES.iter().flat_map(|&a| VALUES.iter().map(move |&b| [a, b, 0, 0])).collect();
    on_both(|floats| {
        run_cases(&shader, &inputs, |_, _, inp, got| {
            let (a, b) = (f32::from_bits(inp[0] & 0x7FFF_FFFF), f32::from_bits(inp[1] ^ 0x8000_0000));
            let want = u32::from(a < b);
            assert_eq!(got, want, "{floats}: |{:#010x}| < -{:#010x} = {got}, want {want}", inp[0], inp[1]);
        });
    });
}

/// `v_cmp_class_f32`: each value is in its class alone, a denormal in a
/// denormal's (not a zero's) with the card's floats too.
#[test]
fn v_cmp_class_f32_reads_the_classes_off_the_bits() {
    // v_cmp_class_f32_e64 s6, v1, v2 (VOP3 opcode 0x088); v_cndmask_b32_e64 v0, 0, 1, s6
    let mut body = vop3(0x088, 6, vgpr(1), vgpr(2), 0).to_vec();
    body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(6)));
    let shader = build_valu_cs(&body);
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // a signalling NaN
        (0xFFBF_FFFF, 0),
        (0x7FC0_0000, 1), // a quiet NaN
        (0xFFC0_0001, 1),
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal
        (0x8080_0000, 3),
        (0x8000_0001, 4), // -denormal
        (0x807F_FFFF, 4),
        (0x8000_0000, 5), // -0
        (0x0000_0000, 6), // +0
        (0x0000_0008, 7), // +denormal
        (0x007F_FFFF, 7),
        (0x0080_0000, 8), // +normal
        (0x3F80_0000, 8),
        (0x7F80_0000, 9), // +inf
    ];
    let inputs: Vec<[u32; 4]> = classes.iter().flat_map(|&(x, _)| (0..10).map(move |bit| [x, 1 << bit, 0, 0])).collect();
    on_both(|floats| {
        run_cases(&shader, &inputs, |_, i, inp, got| {
            let want = u32::from(inp[1] == 1 << classes[i / 10].1);
            assert_eq!(got, want, "{floats}: v_cmp_class_f32({:#010x}, {:#05x}) = {got}, want {want}", inp[0], inp[1]);
        });
    });
}
