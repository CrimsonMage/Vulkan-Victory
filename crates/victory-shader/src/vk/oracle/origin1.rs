//! Behaviour origin/main (fxceless/victory-fork 6742906, GPL-3.0: our own
//! upstream's D3D12 path, lit and playable on an NVIDIA card) states for
//! the console, as cases through `vk::thread_ir_native` and the
//! interpreter. Each test names the file and function of theirs that says
//! it.

#![allow(clippy::all, unused_variables)]

use super::*;

/// The table's second descriptor (at 0x1010): a raw V# of `limit` bytes
/// over 0x4000.
const SECOND: u64 = 0x4000;

/// `s_load_dwordx4 s[12:15], s[4:5], 16`: the second V# of the table.
const LOAD_SECOND: [u32; 2] = [0x3D << 26 | 0x02 << 18 | 12 << 6 | 2, 125 << 25 | 16];

/// Runs `shader` over the usual table plus a second V# of `limit` bytes
/// whose memory holds `data`; the memory afterwards.
fn run_second(shader: &HostShader, inputs: [u32; 4], limit: u32, data: &[u32]) -> Memory {
    let (f, _) = crate::vk::thread_ir_native(&shader.native).expect("translation");
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    mem.put(0x1000, &[0x2000, 0, 0x1_0000, 0x3000_0FAC]);
    mem.put(0x1010, &[SECOND as u32, 0, limit, 0x3000_0FAC]);
    mem.put(SECOND, data);
    let mut m = Machine::new(&f, 32, mem);
    m.push[4..8].copy_from_slice(&inputs);
    m.push[8] = 0x1000;
    assert_eq!(m.run(), Stop::Returned);
    m.mem
}

fn out(mem: &Memory, dword: u64) -> u32 {
    mem.dwords.get(&(0x2000 + 4 * dword)).copied().unwrap_or(0)
}

const DATA: [u32; 6] = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444, 0x5555_5555, 0x6666_6666];

/// What a load of `n` dwords at byte `o` of a view of `limit` bytes reads:
/// every dword bounded on its own.
fn dwords_ref(o: u32, limit: u32, n: u32) -> Vec<u32> {
    (0..n).map(|k| if u64::from(o) + 4 * u64::from(k) + 4 <= u64::from(limit) { DATA[(o / 4 + k) as usize] } else { 0 }).collect()
}

/// origin/main `emit/lower.rs` `raw_dwords`: "The console checks each
/// dword on its own; a host may check a call of several dwords as a whole
/// and answer ZEROS for all of them when only the last lies past the
/// view's end (the lead for run 277's black water)." Its README: four
/// dwords at byte 328 of a 340-byte constant view, in 90 lighting
/// programs. Here: `s_buffer_load_dwordx4 s[16:19], s[12:15], s0` on views
/// of 0 to 24 bytes.
#[test]
fn a_scalar_load_past_a_views_end_reads_the_dwords_inside() {
    let mut body = LOAD_SECOND.to_vec();
    // s_buffer_load_dwordx4 s[16:19], s[12:15], s0
    body.extend([0x3D << 26 | 10 << 18 | 16 << 6 | 6, 0]);
    for k in 0..4u8 {
        body.push(vop1(0x001, 0, sgpr(16 + k)));
        body.extend(buffer_store_dword(0, 8, 4 + 4 * u16::from(k)));
    }
    let shader = build_valu_cs(&body);
    let mut failed = Vec::new();
    for limit in [0u32, 4, 8, 12, 16, 20, 24] {
        for o in [0u32, 4, 8] {
            let mem = run_second(&shader, [o, 0, 0, 0], limit, &DATA);
            let got: Vec<u32> = (1..5).map(|d| out(&mem, d)).collect();
            let want = dwords_ref(o, limit, 4);
            if got != want {
                failed.push(format!("offset {o} of {limit} bytes: {got:08x?}, want {want:08x?}"));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The same for the eight- and sixteen-dword forms (read in groups).
#[test]
fn a_wide_scalar_load_past_a_views_end_reads_the_dwords_inside() {
    let data: Vec<u32> = (1..=20).map(|k| 0x0101_0101 * k).collect();
    for (opc, n) in [(11u32, 8u8), (12, 16)] {
        let mut body = LOAD_SECOND.to_vec();
        body.extend([0x3D << 26 | opc << 18 | 16 << 6 | 6, 0]);
        for k in 0..n {
            body.push(vop1(0x001, 0, sgpr(16 + k)));
            body.extend(buffer_store_dword(0, 8, 4 + 4 * u16::from(k)));
        }
        let shader = build_valu_cs(&body);
        for limit in [0u32, 12, 28, 36, 60, 64, 80] {
            let mem = run_second(&shader, [0, 0, 0, 0], limit, &data);
            let got: Vec<u32> = (1..=u64::from(n)).map(|d| out(&mem, d)).collect();
            let want: Vec<u32> = (0..u32::from(n)).map(|k| if 4 * k + 4 <= limit { data[k as usize] } else { 0 }).collect();
            assert_eq!(got, want, "{n} dwords of a view of {limit} bytes");
        }
    }
}

/// The vector form: `buffer_load_dwordx4 v[4:7], v1, s[12:15], 0 offen`
/// (and x2, x3), each dword bounded on its own (as above).
#[test]
fn a_vector_load_past_a_views_end_reads_the_dwords_inside() {
    let mut failed = Vec::new();
    for (opc, n) in [(13u32, 2u32), (15, 3), (14, 4)] {
        let mut body = LOAD_SECOND.to_vec();
        body.extend([0x38 << 26 | opc << 18 | 1 << 12, 0x80 << 24 | 3 << 16 | 4 << 8 | 1]);
        for k in 0..n as u8 {
            body.extend(buffer_store_dword(4 + k, 8, 4 + 4 * u16::from(k)));
        }
        let shader = build_valu_cs(&body);
        for limit in [0u32, 4, 8, 12, 16, 20] {
            for o in [0u32, 4] {
                let mem = run_second(&shader, [o, 0, 0, 0], limit, &DATA);
                let got: Vec<u32> = (1..=u64::from(n)).map(|d| out(&mem, d)).collect();
                let want = dwords_ref(o, limit, n);
                if got != want {
                    failed.push(format!("x{n} at {o} of {limit} bytes: {got:08x?}, want {want:08x?}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A store of several dwords writes the ones inside the view and drops
/// the rest: `buffer_store_dwordx4 v[1:4], off, s[12:15], 0 offset:4`.
#[test]
fn a_vector_store_past_a_views_end_writes_the_dwords_inside() {
    let mut body = LOAD_SECOND.to_vec();
    body.extend([0x38 << 26 | 30 << 18 | 4, 0x80 << 24 | 3 << 16 | 1 << 8]);
    let shader = build_valu_cs(&body);
    for limit in [0u32, 4, 8, 12, 16, 20, 24] {
        let mem = run_second(&shader, [0xA1, 0xA2, 0xA3, 0xA4], limit, &DATA);
        let got: Vec<u32> = (0..6).map(|d| mem.dwords.get(&(SECOND + 4 * d)).copied().unwrap_or(0)).collect();
        let want: Vec<u32> = (0..6u32).map(|d| if (1..5).contains(&d) && 4 * d + 4 <= limit { 0xA0 + d } else { DATA[d as usize] }).collect();
        assert_eq!(got, want, "a view of {limit} bytes");
    }
}

/// `s_mov_b32 s<n>, literal`.
fn s_mov_lit(n: u8, x: u32) -> [u32; 2] {
    [s_mov_b32(n, 255), x]
}

/// The "compare" sampler of `sampler.rs`'s console-compiled set: less or
/// equal, linear, opaque black border.
const COMPARE_WORDS: [u32; 4] = [0x3006, 0x00FF_F000, 0x1450_0000, 0x4000_0000];

/// A program that samples one 2D image (a T# of zeros in s[24:31]) with the
/// immediate sampler `COMPARE_WORDS` in s[16:19], by MIMG opcode `opcode`
/// (address v2.. as the opcode takes it), result to v0.
fn sampling(opcode: u32) -> HostShader {
    let mut body = Vec::new();
    for (k, w) in COMPARE_WORDS.into_iter().enumerate() {
        body.extend(s_mov_lit(16 + k as u8, w));
    }
    for k in 0..8u8 {
        body.push(s_mov_b32(24 + k, 128));
    }
    // v2 = s0 (the reference or s), v3 = s1, v4 = s2
    body.extend([vop1(0x001, 2, sgpr(0)), vop1(0x001, 3, sgpr(1)), vop1(0x001, 4, sgpr(2))]);
    // image_<op> v0, v[2:..], s[24:31], s[16:19] dmask:0x1 dim:2D
    body.extend([0xF000_0108 | opcode << 18, 0x0086_0002]);
    build_valu_cs(&body)
}

/// On the console the comparison is the instruction's: `image_sample_c_lz`
/// compares with the S#'s function whatever else uses that S#
/// (origin/main `emit/lower.rs` `cmp_samplers`: an immediate sampler a `_c`
/// sample uses becomes a comparison static sampler, `emit/mod.rs`
/// `filter |= 0x80`). The host makes immediate samplers from the words the
/// translator lists: listed as the console wrote them, the sampler does
/// not compare and the sample returns the stored depth, not the test's
/// result. The words of a comparison's sampler carry `sampler::COMPARES`.
#[test]
fn an_immediate_sampler_of_a_comparison_sample_compares() {
    use crate::sampler::{COMPARES, StaticSampler};
    use crate::vk::tir::{ImageKind, Op, Sampler};
    // image_sample_c_lz (0x2f): the reference, then s, t
    let (f, iface) = crate::vk::thread_ir_native(&sampling(0x2f).native).expect("translation");
    let marked = [COMPARE_WORDS[0] | COMPARES, COMPARE_WORDS[1], COMPARE_WORDS[2], COMPARE_WORDS[3]];
    assert_eq!(iface.samplers, vec![marked]);
    let s = StaticSampler::from_words(marked);
    // D3D12_FILTER: comparison reduction, min and mag linear; LESS_EQUAL
    assert_eq!((s.filter, s.comparison, s.address, s.border), (0x94, 4, [4, 1, 1], 1));
    let sample = f.nodes.iter().find(|n| matches!(n.op, Op::Image { .. })).expect("a sample");
    assert!(matches!(sample.op, Op::Image { kind: ImageKind::Sample { dref: true, lod: true, .. }, sampler: Sampler::Static(0), .. }));
    // image_sample_lz (0x27) with the same words: no comparison
    let (_, iface) = crate::vk::thread_ir_native(&sampling(0x27).native).expect("translation");
    assert_eq!(iface.samplers, vec![COMPARE_WORDS]);
    assert_eq!(StaticSampler::from_words(COMPARE_WORDS).filter, 0x14);
}

/// A pixel program of 32 lanes whose only input is the pixel's x (v0: the
/// interpreter gives lane l the centre 0.5 + l % 8).
fn pixel(body: &[u32]) -> NativeProgram {
    let mut words = body.to_vec();
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        // SPI_PS_INPUT_ADDR: POS_X_FLOAT_ENA
        io: StageIo::Ps { inputs: Vec::new(), input_addr: 1 << 8, db_shader_control: 0, patches: Vec::new() },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A pixel's last export (`done`) hands EXEC to the pipeline as the valid
/// mask: the lanes outside it are killed, which is how the console's
/// `discard` and `clip` end (origin/main `emit/lower.rs` `export`:
/// `discard(!guard)` at the `done` export; ACO ends a discard the same
/// way). Here the lanes at x < 4 leave EXEC before `exp mrt0 ... done vm`:
/// they must be demoted, the others must write the colour.
#[test]
fn a_pixel_outside_exec_at_the_last_export_is_killed() {
    use crate::vk::tir::{Item, Op};
    let native = pixel(&[
        vop1(0x001, 1, 242),  // v_mov_b32 v1, 1.0
        0x7C22_00F6,          // v_cmpx_lt_f32 exec, 4.0, v0
        0xF800_180F,
        0x0101_0101,          // exp mrt0, v1, v1, v1, v1 done vm
    ]);
    let (f, iface) = crate::vk::thread_ir_native(&native).expect("translation");
    assert!(iface.demotes);
    // the demotion's guard: a condition per lane
    let kills: Vec<u32> = f
        .blocks
        .iter()
        .flat_map(|b| &b.body)
        .filter_map(|it| match it {
            Item::Guard { cond, body, .. } if body.iter().any(|x| matches!(x, Item::N(v) if f.node(*v).op == Op::Demote)) => Some(*cond),
            _ => None,
        })
        .collect();
    assert_eq!(kills.len(), 1, "one demotion, at the last export");
    let mut m = Machine::new(&f, 32, Memory::default());
    assert_eq!(m.run(), Stop::Returned);
    for l in 0..32usize {
        let inside = l % 8 >= 4;
        assert_eq!(m.value(kills[0], l)[0] != 0, !inside, "lane {l} killed");
        let wrote = m.outputs.keys().any(|k| k.0 == 0 && k.3 == l);
        assert_eq!(wrote, inside, "lane {l} writes mrt0");
    }
}

/// The console's pixel position w is w (D3D's `SV_Position.w`; the
/// hardware's POS_W register, of which Mesa takes the reciprocal to make
/// GL's `gl_FragCoord.w`); origin/main `emit/lower.rs` `input` loads
/// `SV_Position` column 3 as it is. Vulkan's `FragCoord.w` is 1/w: the
/// program must be given its reciprocal. (The interpreter's FragCoord.w is
/// 1.0, so the form is checked.)
#[test]
fn a_pixels_position_w_is_w_not_its_reciprocal() {
    use crate::vk::tir::{Builtin, Op};
    let mut native = pixel(&[
        0xF800_180F,
        0x0000_0000, // exp mrt0, v0, v0, v0, v0 done vm
    ]);
    // SPI_PS_INPUT_ADDR: POS_W_FLOAT_ENA alone: w arrives in v0
    native.io = StageIo::Ps { inputs: Vec::new(), input_addr: 1 << 11, db_shader_control: 0, patches: Vec::new() };
    let (f, _) = crate::vk::thread_ir_native(&native).expect("translation");
    let out = f.nodes.iter().find(|n| matches!(n.op, Op::Output { slot: 0, comp: 0 })).expect("mrt0.x");
    let v = f.node(out.args[0]);
    assert_eq!(v.op, Op::Spv(crate::vk::spirv::op::F_DIV), "1 / FragCoord.w: {v:?}");
    assert_eq!(f.as_const(v.args[0]), Some(u64::from(1.0f32.to_bits())));
    assert_eq!(f.node(v.args[1]).op, Op::Builtin(Builtin::FragCoord(3)));
}

/// Float to integer as the console converts: toward zero, saturated, 0
/// for a NaN (origin/main `emit/lower.rs`, `v_cvt_u32_f32`: "D3D's
/// conversions saturate and give 0 for NaN, as the hardware does (checked
/// on the RX 6700 XT and WARP)"). SPIR-V's conversion is undefined there:
/// the interpreter answers `CONVERT_POISON` for an unclamped one.
#[test]
fn float_to_integer_conversions_saturate_and_take_nan_to_zero() {
    let f = |x: f32| x.to_bits();
    // (input bits, u32, i32, floor to i32, round-half-up to i32)
    let cases: &[(u32, u32, u32, u32, u32)] = &[
        (f(0.0), 0, 0, 0, 0),
        (0x8000_0000, 0, 0, 0, 0),
        (f(1.5), 1, 1, 1, 2),
        (f(-1.5), 0, -1i32 as u32, -2i32 as u32, -1i32 as u32),
        (f(-0.25), 0, 0, -1i32 as u32, 0),
        (f(255.999), 255, 255, 255, 256),
        (f(4294967040.0), 0xFFFF_FF00, 0x7FFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF),
        (f(4294967296.0), 0xFFFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF),
        (f(1e30), 0xFFFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF),
        (f(2147483520.0), 0x7FFF_FF80, 0x7FFF_FF80, 0x7FFF_FF80, 0x7FFF_FF80),
        (f(2147483648.0), 0x8000_0000, 0x7FFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF),
        (f(-2147483648.0), 0, 0x8000_0000, 0x8000_0000, 0x8000_0000),
        (f(-1e30), 0, 0x8000_0000, 0x8000_0000, 0x8000_0000),
        (f(f32::INFINITY), 0xFFFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF),
        (f(f32::NEG_INFINITY), 0, 0x8000_0000, 0x8000_0000, 0x8000_0000),
        (0x7FC0_0000, 0, 0, 0, 0),
        (0xFFC0_0001, 0, 0, 0, 0),
        (0x7F80_0001, 0, 0, 0, 0),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [c.0, 0, 0, 0]).collect();
    // v_cvt_u32_f32 7, v_cvt_i32_f32 8, v_cvt_rpi_i32_f32 12, v_cvt_flr_i32_f32 13
    for (opc, pick) in [(7u16, 1usize), (8, 2), (13, 3), (12, 4)] {
        let shader = build_valu_cs(&[vop1(opc, 0, vgpr(1))]);
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = [cases[i].1, cases[i].2, cases[i].3, cases[i].4][pick - 1];
            assert_eq!(got, want, "{backend}: opcode {opc} of {:#010x} = {got:#010x}, want {want:#010x}", inp[0]);
        });
    }
}

/// A pixel lane that reaches the program's end outside EXEC on a path
/// with no last export left without exporting: killed (origin/main
/// `emit/lower.rs`, `LaneExit::End`: "A lane leaving without its final
/// export was killed"). Here the wave skips the export when s0 is 0 (a
/// scalar branch), after the lanes at x < 4 left EXEC: on that path they
/// are killed at the end; on the other at the export.
#[test]
fn a_pixel_that_ends_outside_exec_without_an_export_is_killed() {
    use crate::vk::tir::{Item, Op};
    let mut native = pixel(&[
        vop1(0x001, 1, 242), // v_mov_b32 v1, 1.0
        0x7C22_00F6,         // v_cmpx_lt_f32 exec, 4.0, v0
        0xBF06_8000,         // s_cmp_eq_u32 s0, 0
        0xBF85_0002,         // s_cbranch_scc1 -> the end
        0xF800_180F,
        0x0101_0101,         // exp mrt0, v1, v1, v1, v1 done vm
    ]);
    native.layout = vec![RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 0, slot: 0 }];
    let (f, _) = crate::vk::thread_ir_native(&native).expect("translation");
    let demotes = |f: &crate::vk::tir::Func, b: usize| -> Vec<u32> {
        f.blocks[b]
            .body
            .iter()
            .filter_map(|it| match it {
                Item::Guard { cond, body, .. } if body.iter().any(|x| matches!(x, Item::N(v) if f.node(*v).op == Op::Demote)) => Some(*cond),
                _ => None,
            })
            .collect()
    };
    let per_block: Vec<Vec<u32>> = (0..f.blocks.len()).map(|b| demotes(&f, b)).collect();
    let n: usize = per_block.iter().map(Vec::len).sum();
    assert_eq!(n, 2, "a kill at the export and one at the end: {per_block:?}");
    for s0 in [0u32, 1] {
        let mut m = Machine::new(&f, 32, Memory::default());
        m.push[push_window_ps()] = s0;
        assert_eq!(m.run(), Stop::Returned);
        // the kills of the blocks this run entered
        let run: Vec<u32> = (0..f.blocks.len()).filter(|&b| m.visits[b] > 0).flat_map(|b| per_block[b].clone()).collect();
        for l in 0..32usize {
            let inside = l % 8 >= 4;
            let killed = run.iter().any(|&c| m.value(c, l)[0] != 0);
            assert_eq!(killed, !inside, "s0 {s0}, lane {l}");
        }
    }
}

/// The push dword of a pixel program's user-data entry 0.
fn push_window_ps() -> usize {
    crate::vk::lift::push_window(HwStage::Ps) as usize
}
