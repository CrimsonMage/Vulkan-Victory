//! Behaviour the NVIDIA fork states for the console
//! (trustno1-coder/VIctory-NVIDIA 0f0ada1, GPL-3.0: a Direct3D 12 fork of
//! the same upstream), as cases through `vk::thread_ir_native`, the
//! interpreter and the module's own words. The programs are theirs (kept as
//! written); they ran them on a D3D12 device. Each test names the test of
//! theirs it is.

#![allow(clippy::all)]

use super::*;
use crate::native::NativeProgram;
use crate::vk::spirv::en;

const TABLE: u64 = 0x1000;

/// A pixel program of 32 lanes with `db_shader_control`.
fn pixel(words: &[u32], db_shader_control: u32) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 0, slot: 0 }],
        used: Vec::new(),
        io: StageIo::Ps { inputs: Vec::new(), input_addr: 0, db_shader_control, patches: Vec::new() },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The module of a pixel program, as the translation makes it.
fn pixel_module(native: &NativeProgram) -> Vec<u32> {
    let (f, iface) = crate::vk::thread_ir_native(native).expect("translation");
    let g = crate::vk::graph::Graph::new(&f).unwrap();
    let st = crate::vk::emit::Stage {
        stage: HwStage::Ps,
        local_size: [1, 1, 1],
        out_types: Default::default(),
        locs: Default::default(),
        mesh: (0, 0),
        mesh_split: None,
        loop_limit: 1 << 14,
        guard_id: 0,
        no_sel: Default::default(),
    };
    crate::vk::emit::Emit::new(&f, &g, &iface, &st).run().unwrap().words
}

/// The execution modes a module declares for its entry point.
fn modes(words: &[u32]) -> Vec<u32> {
    let insts = instructions(words);
    let entry = insts.iter().find(|i| i.0 == 15).expect("an entry point").1[1];
    insts.iter().filter(|i| i.0 == 16 && i.1[0] == entry).map(|i| i.1[1]).collect()
}

/// Theirs: `early_depth_is_given_where_the_console_tests_before_the_program`
/// and `the_consoles_early_depth_test_is_declared_with_the_switch` (41b4a05;
/// no switch here). A pixel program whose console code tests depth and
/// stencil before it runs (`DB_SHADER_CONTROL` bit 12) declares
/// `EarlyFragmentTests`; one that gives the tests their depth or stencil,
/// or does not ask, does not.
#[test]
fn the_consoles_early_depth_test_is_declared() {
    use crate::vk::{EarlyDepth, SigRow, early_depth};
    let colour = [
        0x7E00_02F2u32, // v_mov_b32 v0, 1.0
        0xF800_180F,
        0x0000_0000, // exp mrt0, v0, v0, v0, v0 done vm
        S_ENDPGM,
    ];
    let early = |db: u32| modes(&pixel_module(&pixel(&colour, db))).contains(&en::EM_EARLY_FRAGMENT_TESTS);
    // The title's transparent pass: early Z then late Z, kill, run for
    // no-op colour writes, depth before the program.
    let glass = 0x1450;
    assert!(early(glass), "asked for: declared");
    // Early Z then late Z alone is the hardware's choice, not the title's.
    assert!(!early(0x50));
    assert!(!early(0));
    // A depth or stencil export keeps the late test.
    for export in [1, 2, 4] {
        assert!(!early(glass | export), "export bit {export}");
    }
    // ... also one the control word does not name: the module that writes
    // depth says so and takes no early test.
    let depth = [
        0x7E00_02F2u32, // v_mov_b32 v0, 1.0
        0xF800_0081,
        0x0000_0000, // exp mrtz, v0, off, off, off
        0xF800_180F,
        0x0000_0000, // exp mrt0, v0, v0, v0, v0 done vm
        S_ENDPGM,
    ];
    let m = modes(&pixel_module(&pixel(&depth, glass)));
    assert!(m.contains(&en::EM_DEPTH_REPLACING) && !m.contains(&en::EM_EARLY_FRAGMENT_TESTS), "{m:?}");
    // The output signature: SV_Target (64) alone; SV_Depth (65),
    // SV_DepthGreaterEqual (67), SV_DepthLessEqual (68), SV_StencilRef (69).
    let row = |sv| SigRow { name: "X".into(), index: 0, sv, comp: 3, reg: 0, mask: 1, half: false };
    assert_eq!(early_depth(glass, &[row(64)]), EarlyDepth::Given);
    assert_eq!(early_depth(0x50, &[row(64)]), EarlyDepth::NotAsked);
    assert_eq!(early_depth(glass | 1, &[row(64)]), EarlyDepth::Exports);
    for sv in [65, 67, 68, 69] {
        assert_eq!(early_depth(glass, &[row(64), row(sv)]), EarlyDepth::Exports, "sv {sv}");
    }
}

/// The T# of the image in `slot` (`docs/vulkan-abi.md` 5.2).
fn tsharp(slot: u32) -> [u32; 8] {
    [0, 0, 0, 0, 0, 0, slot ^ 0x7A3C_5E91, slot]
}

/// Theirs: `an_ordered_section_uses_rasterizer_ordered_views`. A pixel
/// program with an ordered section, as the console compiler makes one (the
/// title's hair cards keep their nearest layers that way): the wait for
/// overlapped waves is skipped and the program runs inside the fragment
/// shader interlock: one begin and one end, in the entry point, outside
/// control flow, with the ordered execution mode. The same accesses
/// outside a section take no interlock; a wait without the section's start
/// stays the approximation it was.
#[test]
fn an_ordered_section_runs_inside_the_interlock() {
    use crate::vk::spirv::op;
    let section = [
        0xBE82_0300u32, // s_mov_b32 s2, s0
        0xBE83_0380,    // s_mov_b32 s3, 0
        0xF40C_0201,    // s_load_dwordx8 s[8:15], s[2:3], 0
        0xFA00_0000,
        0x9390_FF01, // s_bfe_u32 s16, s1, 0x2001c
        0x0002_001C,
        0x9391_FF01, // s_bfe_u32 s17, s1, 0xa0000
        0x000A_0000,
        0x8F10_8110, // s_lshl_b32 s16, s16, 1
        0x9392_FF01, // s_bfe_u32 s18, s1, 0xa0010
        0x000A_0010,
        0xBE91_0711, // s_not_b32 s17, s17
        0x9013_9F01, // s_lshr_b32 s19, s1, 31
        0x8810_8110, // s_or_b32 s16, s16, 1
        0x8112_1112, // s_add_i32 s18, s18, s17
        0xBF06_8013, // s_cmp_eq_u32 s19, 0
        0xB990_1019, // s_setreg_b32 hwreg(HW_REG_POPS_PACKER, 0, 3), s16
        0x8514_807E, // s_cselect_b32 s20, exec_lo, 0
        0x8115_11EF, // s_add_i32 s21, src_pops_exiting_wave_id, s17
        0xBF08_1215, // s_cmp_gt_u32 s21, s18
        0x8516_807E, // s_cselect_b32 s22, exec_lo, 0
        0x8814_1614, // s_or_b32 s20, s20, s22
        0x876A_147E, // s_and_b32 vcc_lo, exec_lo, s20
        0xBF87_0004, // s_cbranch_vccnz past the wait
        0xBF8E_FFFF, // s_sleep 0xffff
        0x8115_11EF, // s_add_i32 s21, src_pops_exiting_wave_id, s17
        0xBF0A_1512, // s_cmp_lt_u32 s18, s21
        0xBF84_FFFC, // s_cbranch_scc0 back to the sleep
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xF000_3F88, // image_load v[2:5], v[0:1], s[8:15] dmask:0xf 2D glc dlc
        0x0002_0200,
        0x0604_04F2, // v_add_f32 v2, 1.0, v2
        0xF020_3F08, // image_store v[2:5], v[0:1], s[8:15] dmask:0xf 2D glc
        0x0002_0200,
        0xBEFC_0380, // s_mov_b32 m0, 0
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
    ];
    let end = [
        0xF800_180Fu32, // exp mrt0 v2, v2, v2, v2 done vm
        0x0202_0202,
        S_ENDPGM,
    ];
    let pops = |native: &NativeProgram| {
        let (_, iface) = crate::vk::thread_ir_native(native).expect("translation");
        (iface.ordered, iface.approximate.iter().any(|a| a.note.contains("POPS")))
    };
    let ordered = pixel(&[&section[..], &end].concat(), 0);
    assert_eq!(pops(&ordered), (true, false), "ordered, and nothing approximated");
    let words = pixel_module(&ordered);
    let insts = instructions(&words);
    let entry = insts.iter().find(|i| i.0 == 15).expect("an entry point").1[1];
    assert!(insts.iter().any(|i| i.0 == 17 && i.1 == [en::CAP_FRAGMENT_SHADER_PIXEL_INTERLOCK_EXT]));
    assert!(insts.iter().any(|i| i.0 == 10 && i.1[0] == u32::from_le_bytes(*b"SPV_") && i.1[2] == u32::from_le_bytes(*b"frag")));
    assert!(modes(&words).contains(&en::EM_PIXEL_INTERLOCK_ORDERED_EXT));
    // the entry point is: begin, the program (a call), end, return
    let at = insts.iter().position(|i| i.0 == u32::from(op::FUNCTION) && i.1[1] == entry).expect("the entry point's function");
    let body: Vec<u32> = insts[at + 1..].iter().map(|i| i.0).take_while(|&o| o != u32::from(op::FUNCTION_END)).collect();
    let (begin, finish) = (u32::from(op::BEGIN_INVOCATION_INTERLOCK_EXT), u32::from(op::END_INVOCATION_INTERLOCK_EXT));
    assert_eq!(body, [u32::from(op::LABEL), begin, u32::from(op::FUNCTION_CALL), finish, u32::from(op::RETURN)]);
    assert_eq!(insts.iter().filter(|i| i.0 == begin || i.0 == finish).count(), 2, "one begin, one end");
    // the wait ends at once: the program runs to its end and stores
    let (f, _) = crate::vk::thread_ir_native(&ordered).unwrap();
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    mem.put(TABLE, &tsharp(1));
    let mut m = Machine::new(&f, 32, mem);
    m.push[crate::vk::lift::push_window(HwStage::Ps) as usize] = TABLE as u32;
    m.limit = 64;
    assert_eq!(m.run(), Stop::Returned);
    assert_eq!(m.image_stores.len(), 32, "every pixel stores");
    // The same accesses outside a section: no interlock.
    let plain = pixel(&[&section[..4], &section[28..35], &end].concat(), 0);
    assert_eq!(pops(&plain), (false, false));
    let words = pixel_module(&plain);
    assert!(instructions(&words).iter().all(|i| i.0 != begin && i.0 != finish && !(i.0 == 17 && i.1[0] == en::CAP_FRAGMENT_SHADER_PIXEL_INTERLOCK_EXT)));
    assert!(!modes(&words).contains(&en::EM_PIXEL_INTERLOCK_ORDERED_EXT));
    // Without the section's start the wait is no section we know (theirs
    // refuses the program): the order is not kept, and that is said.
    let mut unmarked = [&section[..], &end].concat();
    unmarked.remove(16);
    assert_eq!(pops(&pixel(&unmarked, 0)), (false, true));
}
