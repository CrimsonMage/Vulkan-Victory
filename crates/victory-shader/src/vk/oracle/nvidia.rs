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
