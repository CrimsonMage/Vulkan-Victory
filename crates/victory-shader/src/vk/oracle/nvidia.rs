//! Behaviour the NVIDIA fork states for the console
//! (trustno1-coder/VIctory-NVIDIA 0f0ada1, GPL-3.0: a Direct3D 12 fork of
//! the same upstream), as cases through `vk::thread_ir_native`, the
//! interpreter and the module's own words. The programs are theirs (kept as
//! written); they ran them on a D3D12 device. Each test names the test of
//! theirs it is.
//!
//! Not ported, of the tests asked for: the shape checks of
//! `typed_d16_table_uav_roundtrip_uint_sint_and_float` and
//! `typed_non_d16_table_load_preserves_integer_bits` (which root views
//! their lowering asks for: `SLOT_TABLE_STRIDES`; ours reads the format
//! from the V#), and the refusals of
//! `a_counter_texture_is_set_by_a_store_of_one_channel` (a store whose view
//! their tracing cannot type; ours reads the type from the T#).

#![allow(clippy::all)]

use super::wave64::{dword0, native as wave64_native, native_of, run_group, stores};
use super::*;
use crate::native::NativeProgram;
use crate::vk::interp::Images;
use crate::vk::meetings;
use crate::vk::spirv::en;
use crate::vk::tir::{ImageClass, ImageKind, Sampler};

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

/// A wave64 compute program of `threads` threads with one 512-byte LDS unit.
fn lds_program(body: &[u32], threads: u32) -> NativeProgram {
    let mut native = native_of(body, true, threads);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15;
    }
    native
}

/// The kinds of the meetings before LDS accesses the translation keeps,
/// in block order.
fn meetings_of(native: &NativeProgram) -> Vec<u32> {
    use crate::vk::tir::{Item, Op};
    let (f, _) = crate::vk::thread_ir_native(native).expect("translation");
    let kinds: Vec<u32> = f
        .blocks
        .iter()
        .flat_map(|b| &b.body)
        .filter_map(|it| match it {
            Item::N(v) if f.node(*v).op == Op::Xchg && f.node(*v).imm.first() == Some(&crate::vk::uniform::MEET) => f.node(*v).imm.get(1).copied(),
            _ => None,
        })
        .collect();
    assert_eq!(kinds.len(), meetings::count(&f));
    kinds
}

const DS_WRITE_V1_V0: [u32; 2] = [0xD834_0000, 0x0000_0001];
const DS_ADD_V1_V0: [u32; 2] = [0xD800_0000, 0x0000_0001];
const WAITCNT: u32 = 0xBF8C_C07F;
const S_BARRIER: u32 = 0xBF8A_0000;

/// `ds_read_b32 v<dst>, v<addr>`.
const fn ds_read(dst: u32, addr: u32) -> [u32; 2] {
    [0xD8D8_0000, dst << 24 | addr]
}

/// Theirs: `lds_accesses_after_the_last_plain_write_need_no_meeting`
/// (89cfae2 `LdsOrder`), their program: before a barrier a lane writes its
/// dword and reads it back; after it the lanes read and add. Which
/// accesses the halves of a wave must meet before: none before the write
/// (nothing precedes it) nor before the read after the barrier (the
/// barrier is a meeting); the read after the write and the add after the
/// read keep theirs. (Theirs lets the add go too, and calls the read
/// beside it unstable: a half may then read before the other half's add.)
/// And the program computes the same as one wave and as two halves in
/// either order.
#[test]
fn lds_accesses_that_commute_need_no_meeting() {
    let body = |tail: &[u32]| {
        let mut w = vec![0x3402_0082]; // v_lshlrev_b32 v1, 2, v0
        w.extend(DS_WRITE_V1_V0);
        w.extend(ds_read(4, 1));
        w.extend([WAITCNT, S_BARRIER]);
        w.extend(ds_read(5, 1));
        w.extend(DS_ADD_V1_V0);
        w.extend_from_slice(tail);
        w
    };
    let theirs = lds_program(&body(&[]), 128);
    assert_eq!(meetings_of(&theirs), [meetings::READ, meetings::ADD]);
    // ... then each lane reads its dword twice more: one meeting for both
    // reads (they follow the add), and every thread finds t, t, 2t, 2t.
    let mut tail = ds_read(6, 1).to_vec();
    tail.extend(ds_read(7, 1));
    tail.push(WAITCNT);
    tail.extend(stores([4, 5, 6, 7]));
    let native = lds_program(&body(&tail), 128);
    assert_eq!(meetings_of(&native), [meetings::READ, meetings::ADD, meetings::READ]);
    let want: Vec<u32> = (0..128).flat_map(|t| [t, t, 2 * t, 2 * t]).collect();
    assert_eq!(run_group(&native, [0; 4], 4 * 128, 0), Ok(want));
}

/// Ours: why a read beside an atomic keeps its meeting. Every lane of a
/// wave adds 1 to one counter, then reads it: the console's wave has made
/// all 64 adds before any lane reads, so every lane reads 64. Two halves
/// that did not meet between would read 32 in the half that ran first
/// (`run_group` runs both orders and counts LDS dwords the halves accessed
/// with no meeting between).
#[test]
fn a_read_after_the_waves_adds_sees_them_all() {
    let mut body = vec![
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7E04_0281, // v_mov_b32 v2, 1
        0xD800_0000,
        0x0000_0201, // ds_add_u32 v1, v2
        0xD800_0000,
        0x0000_0201, // ds_add_u32 v1, v2
    ];
    body.extend(ds_read(4, 1));
    body.extend(ds_read(5, 1));
    body.push(WAITCNT);
    body.extend(stores([4, 5, 4, 5]));
    let native = lds_program(&body, 64);
    // the two adds commute, the two reads commute; the read after the adds
    // does not
    assert_eq!(meetings_of(&native), [meetings::READ]);
    assert_eq!(run_group(&native, [0; 4], 4 * 64, 0), Ok(vec![128; 4 * 64]));
}

/// Theirs: `some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on`.
/// "Some lane is left" of a mask operation as a number: an upscaler's
/// programs wait for other groups in a loop that ends when every lane saw
/// its flag, or after enough rounds: both tests become -1 or 0 and SCC of
/// `timeout | ~waiting` ends it.
#[test]
fn some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on() {
    let native = wave64_native(&[
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7D84_0200, // v_cmp_eq_u32 vcc, s0, v1
        0x8792_7E6A, // s_and_b64 s[18:19], vcc, exec
        0x8592_80C1, // s_cselect_b64 s[18:19], -1, 0
        0xBF09_8501, // s_cmp_ge_u32 s1, 5
        0x8594_80C1, // s_cselect_b64 s[20:21], -1, 0
        0x8B92_1214, // s_orn2_b64 s[18:19], s[20:21], s[18:19]
        0x8506_8081, // s_cselect_b32 s6, 1, 0
        0x7E04_0206, // v_mov_b32 v2, s6
        0xE070_0000,
        0x8002_0200, // buffer_store_dword v2, off, s[8:11], 0
    ]);
    // (lane s0 waits, if there is one; s1 rounds)
    for (wait, rounds, done) in [(0, 0, 0), (37, 4, 0), (63, 4, 0), (64, 0, 1), (37, 5, 1), (64, 9, 1)] {
        assert_eq!(dword0(&native, wait, rounds), Ok(done), "lane {wait} after {rounds} rounds");
    }
}

/// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 idxen` (theirs).
const fn mubuf_idxen(op: u8, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [0x38 << 26 | (op as u32) << 18 | 1 << 13, 0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32]
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], byte_offset` (theirs).
const fn s_load_dwordx4_off(dst: u8, base: u8, byte_offset: u32) -> [u32; 2] {
    [0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1), (125 << 25) | (byte_offset & 0x1F_FFFF)]
}

/// Their `build_table_typed_cs`: a compute program of 4 threads whose
/// descriptors live in a table at s4.
fn table_typed_cs(body: &[u32]) -> NativeProgram {
    let mut words = vec![s_mov_b32(5, 0x80)]; // s5 = 0
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 0 }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [4, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A V# of `records` elements of `stride` bytes in GFX10 buffer `format`
/// at `base` (`docs/vulkan-abi.md` section 4; format 0, stride 0: raw).
fn vsharp(base: u32, stride: u32, records: u32, format: u32) -> [u32; 4] {
    [base, stride << 16, records, 0x3000_0FAC | format << 12]
}

/// Runs a table program over `views` (the table's entries, 32 bytes each,
/// from entry `first` on) and `data` at the first view's base; the memory
/// afterwards.
fn run_table(native: &NativeProgram, first: u32, views: &[[u32; 4]], data: &[u32]) -> Memory {
    let (f, _) = crate::vk::thread_ir_native(native).expect("translation");
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    for (k, v) in views.iter().enumerate() {
        mem.put(TABLE + 32 * (u64::from(first) + k as u64), v);
    }
    mem.put(u64::from(views[0][0]), data);
    let mut m = Machine::new(&f, 32, mem);
    m.push[8] = TABLE as u32 + 32 * first;
    assert_eq!(m.run(), Stop::Returned);
    m.mem
}

fn dwords(mem: &Memory, base: u32, n: u32) -> Vec<u32> {
    (0..u64::from(n)).map(|i| mem.dwords.get(&(u64::from(base) + 4 * i)).copied().unwrap_or(0)).collect()
}

/// Theirs: `typed_d16_table_uav_roundtrip_uint_sint_and_float`.
/// `compute_graph_ops` (facial lip-sync and the wrapped submesh solver)
/// reads and writes FP16 bit patterns of `R16G16B16A16_UINT` views with
/// `buffer_load_format_d16_xyzw` and `buffer_store_format_d16_xyzw`: the
/// halves come through as they are for UINT, SINT and FLOAT views, at any
/// place of the table.
#[test]
fn typed_d16_table_uav_roundtrip_uint_sint_and_float() {
    let mut body = Vec::new();
    // s[8:11] = table[0] (src typed UAV), s[12:15] = table[1] (dst typed UAV),
    // s[16:19] = table[2] (raw UAV dump of loaded v[2:3]).
    body.extend_from_slice(&s_load_dwordx4_off(8, 4, 0));
    body.extend_from_slice(&s_load_dwordx4_off(12, 4, 32));
    body.extend_from_slice(&s_load_dwordx4_off(16, 4, 64));
    // buffer_load_format_d16_xyzw v[2:3], v0, s[8:11], 0 idxen (op 0x83)
    body.extend_from_slice(&mubuf_idxen(0x83, 2, 0, 8));
    // buffer_store_format_d16_xyzw v[2:3], v0, s[12:15], 0 idxen (op 0x87)
    body.extend_from_slice(&mubuf_idxen(0x87, 2, 0, 12));
    // v4 = v0 << 3 (byte offset = thread * 8)
    body.push(0x3408_0083); // v_lshlrev_b32 v4, 3, v0
    // buffer_store_dwordx2 v[2:3], v4, s[16:19], 0 offen (op 0x1D, offen bit 12)
    body.extend_from_slice(&[0x38 << 26 | 0x1D << 18 | 1 << 12, 0x80 << 24 | (16u32 >> 2) << 16 | (2u32 << 8) | 4u32]);
    let native = table_typed_cs(&body);
    // 4 elements x 4 u16 channels = 8 u32 dwords.
    // Includes FP16 subnormal (0x0001), -0.0 (0x8000), 1.0 (0x3C00), -1.0 (0xBC00),
    // NaN (0x7E00), max u16 (0xFFFF), and arbitrary patterns.
    let uint_bits: [u32; 8] = [0x3C00_0001, 0x7E00_BC00, 0xFFFF_8000, 0xABCD_1234, 0x0002_0000, 0x7BFF_FBFF, 0xDEAD_BEEF, 0xCAFE_BABE];
    let sint_bits: [u32; 8] = [0xFFFF_8000, 0x7FFF_0001, 0xC000_0000, 0x8001_7FFE, 0xFFFF_FFFF, 0x0000_0000, 0xFEDC_BA98, 0x7654_3210];
    // Normal finite FP16 values that round-trip cleanly through f32.
    let float_bits: [u32; 8] = [0x4000_3C00, 0x3800_BC00, 0xC200_3400, 0x4500_0000, 0x3E00_BE00, 0x4400_C400, 0x3C00_0000, 0x4200_C000];
    // (GFX10 buffer formats 69, 70, 71: 16_16_16_16 UINT, SINT, FLOAT)
    let cases = [
        ("R16G16B16A16_UINT @0", 69, &uint_bits, 0u32),
        ("R16G16B16A16_UINT @3 (dynamic offset)", 69, &uint_bits, 3),
        ("R16G16B16A16_SINT @0", 70, &sint_bits, 0),
        ("R16G16B16A16_FLOAT @0", 71, &float_bits, 0),
    ];
    let (src, dst, raw) = (0x2000, 0x3000, 0x4000);
    for (label, format, input, first) in cases {
        let views = [vsharp(src, 8, 4, format), vsharp(dst, 8, 4, format), vsharp(raw, 0, 64, 0)];
        let mem = run_table(&native, first, &views, input);
        assert_eq!(dwords(&mem, raw, 8), input, "{label}: buffer_load_format_d16_xyzw mismatch");
        assert_eq!(dwords(&mem, dst, 8), input, "{label}: buffer_store_format_d16_xyzw mismatch");
    }
}

/// Theirs: `typed_non_d16_table_load_preserves_integer_bits`. Non-`d16`
/// typed loads (`buffer_load_format_x`) are used by
/// `CS_BlendshapeTranslationsHSSGraphOutputExtract` (reading `HSSGraphOut`
/// `R16_UINT`) and `CS_PedStandardSkinning_*` (reading bone indices): exact
/// integer bits for `R16_UINT`, `R16_SINT` and `R32_UINT`.
#[test]
fn typed_non_d16_table_load_preserves_integer_bits() {
    let mut body = Vec::new();
    // s[8:11] = table[0] (src typed SRV), s[12:15] = table[1] (raw UAV out)
    body.extend_from_slice(&s_load_dwordx4_off(8, 4, 0));
    body.extend_from_slice(&s_load_dwordx4_off(12, 4, 32));
    // buffer_load_format_x v2, v0, s[8:11], 0 idxen (op 0x00)
    body.extend_from_slice(&mubuf_idxen(0x00, 2, 0, 8));
    // v4 = v0 << 2 (byte offset = thread * 4)
    body.push(0x3408_0082); // v_lshlrev_b32 v4, 2, v0
    // buffer_store_dword v2, v4, s[12:15], 0 offen (op 0x1C, offen bit 12)
    body.extend_from_slice(&[0x38 << 26 | 0x1C << 18 | 1 << 12, 0x80 << 24 | (12u32 >> 2) << 16 | (2u32 << 8) | 4u32]);
    let native = table_typed_cs(&body);
    let r16_data = [0x3C00_0001u32, 0xFFFF_BC00u32];
    let r32_data = [0xDEAD_BEEFu32, 0x0000_0001, 0x8000_0000, 0x7FFF_FFFF];
    // (GFX10 buffer formats 11, 12, 20: 16 UINT, 16 SINT, 32 UINT)
    let cases: [(&str, u32, u32, &[u32], [u32; 4]); 3] = [
        ("R16_UINT", 11, 2, &r16_data, [0x0001, 0x3C00, 0xBC00, 0xFFFF]),
        ("R16_SINT", 12, 2, &r16_data, [0x0001, 0x3C00, 0xFFFF_BC00, 0xFFFF_FFFF]),
        ("R32_UINT", 20, 4, &r32_data, r32_data),
    ];
    let (src, raw) = (0x2000, 0x4000);
    for (label, format, elem_bytes, input, want) in cases {
        let views = [vsharp(src, elem_bytes, 4, format), vsharp(raw, 0, 64, 0)];
        let mem = run_table(&native, 0, &views, input);
        assert_eq!(dwords(&mem, raw, 4), want, "{label}");
    }
}

/// Their size tests' program frame: the table at s[2:3], its raw UAV's
/// descriptor (entry 2) in s[20:23]; a wave32 compute program of one thread
/// with the table of views at s0.
fn one_thread_cs(words: &[u32]) -> NativeProgram {
    let param = |root, user_data| RootParam { root, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data, slot: user_data };
    let mut code = vec![
        0xBE82_0300u32, // s_mov_b32 s2, s0
        0xBE83_0380,    // s_mov_b32 s3, 0
        0xF408_0501,    // s_load_dwordx4 s[20:23], s[2:3], 64
        0xFA00_0040,
    ];
    code.extend_from_slice(words);
    code.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![param(RootType::Table, 0), param(RootType::Constants, 1)],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [1, 1, 1] }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// One texel of an unsigned counter texture: atomics add to it and give
/// the old value, a store sets it (the data is the last operand).
struct Counter<'a>(&'a std::cell::Cell<u64>);

impl Images for Counter<'_> {
    fn image(&mut self, kind: ImageKind, _: ImageClass, _: Sampler, _: u16, args: &[[u64; 4]], _: usize) -> [u64; 4] {
        let (old, data) = (self.0.get(), args[args.len() - 1][0]);
        match kind {
            ImageKind::Atomic(_) => self.0.set(old + data),
            ImageKind::Store { .. } => self.0.set(data),
            _ => {}
        }
        [old, 0, 0, 0]
    }
}

/// Theirs: `a_counter_texture_is_set_by_a_store_of_one_channel`. An
/// upscaler counts its groups in a texture with atomics and sets the
/// counter back with a store of one channel, 16 bits wide (`d16`) or not.
#[test]
fn a_counter_texture_is_set_by_a_store_of_one_channel() {
    // image_store v3, v[0:1], s[8:15] dmask:0x1 2D, with and without d16
    for store in [[0xF020_1108u32, 0x0002_0300], [0xF020_1108, 0x8002_0300]] {
        let native = one_thread_cs(&[
            0xF40C_0201, // s_load_dwordx8 s[8:15], s[2:3], 0
            0xFA00_0000,
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E04_0283, // v_mov_b32 v2, 3
            0xF044_3108, // image_atomic_add v2, v[0:1], s[8:15] dmask:0x1 2D glc
            0x0002_0200,
            0x7E06_0289, // v_mov_b32 v3, 9
            store[0],
            store[1],
            0x7E08_0281, // v_mov_b32 v4, 1
            0xF044_3108, // image_atomic_add v4, v[0:1], s[8:15] dmask:0x1 2D glc
            0x0002_0400,
        ]);
        let (f, _) = crate::vk::thread_ir_native(&native).unwrap_or_else(|e| panic!("store {:#x}: {e}", store[1]));
        let texel = std::cell::Cell::new(0);
        let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
        // an unsigned storage view in slot 1: INTEGER and STORAGE (word 7)
        mem.put(TABLE, &tsharp(1 | 1 << 24 | 1 << 25));
        mem.put(TABLE + 64, &vsharp(0x2000, 0, 64, 0));
        let mut m = Machine::new(&f, 32, mem);
        m.push[4] = TABLE as u32;
        m.images = Some(Box::new(Counter(&texel)));
        assert_eq!(m.run(), Stop::Returned);
        drop(m);
        assert_eq!(texel.get(), 10, "store {:#x}", store[1]);
    }
}

/// Every sample answers with word 2 of the immediate sampler it was made
/// with (0xFFFF_FFFF for a sampler from the heap).
struct WhichSampler(Vec<[u32; 4]>);

impl Images for WhichSampler {
    fn image(&mut self, _: ImageKind, _: ImageClass, sampler: Sampler, _: u16, _: &[[u64; 4]], _: usize) -> [u64; 4] {
        let word = match sampler {
            Sampler::Static(n) => self.0[n as usize][2],
            _ => u32::MAX,
        };
        [u64::from(word), 0, 0, 0]
    }
}

/// Theirs: `immediate_sampler_chosen_by_control_flow`: two immediate
/// sampler descriptors moved into s[16:19] by the arms of a scalar branch;
/// the sample after the join uses the one its arm chose. Ours takes an S#
/// that is not four constants for a heap sampler and samples with slot 0
/// of the sampler heap (`docs/opcode-audit.md`, open).
#[test]
#[ignore = "open: an S# chosen between immediate samplers by a branch is read as a heap sampler"]
fn immediate_sampler_chosen_by_control_flow() {
    let s_mov_lit = |dst: u8, lit: u32| [0xBE80_03FF | (dst as u32) << 16, lit];
    let s_load_x8 = |dst: u8, base: u8| [0x3D << 26 | 0x03 << 18 | (dst as u32) << 6 | (base as u32 >> 1), 125 << 25];
    // image_sample_lz v4, v[2:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
    let image_sample_lz: [u32; 2] = [
        0x3C << 26 | 0x21 << 18 | 1 << 8 | 1 << 3, // dmask=1, dim=2D(1)
        (16u32 >> 2) << 21 | (8u32 >> 2) << 16 | 4 << 8 | 2,
    ];
    // buffer_store_dword v4, off, s[20:23], 0
    let buf_store: [u32; 2] = [0x38 << 26 | 0x1C << 18, 0x80 << 24 | (20u32 >> 2) << 16 | 4 << 8];
    // Two distinct valid immediate sampler descriptors (different wrap/filter bits).
    let samp_a: [u32; 4] = [0x0000_0092, 0x00C0_0000, 0x1850_0000, 0];
    let samp_b: [u32; 4] = [0x0000_0092, 0x00C0_0000, 0x1850_0001, 0];
    for (case_name, alt) in [("same_sampler", samp_a), ("different_samplers", samp_b)] {
        let mut words = Vec::new();
        words.extend_from_slice(&s_load_x8(8, 2)); // s[8:15] = texture SRV (offset 0)
        // v2 = 0.5, v3 = 0.5 (sample coordinates).
        words.push(0x7E00_02F0 | (2 << 17)); // v_mov_b32 v2, 0.5
        words.push(0x7E00_02F0 | (3 << 17)); // v_mov_b32 v3, 0.5
        words.push(0xBF00_8001); // s_cmp_eq_u32 s1, 0 (const 128)
        let br_else = words.len();
        words.push(0); // placeholder s_cbranch_scc0
        // then arm: load samp_a into s[16:19]
        for (i, &w) in samp_a.iter().enumerate() {
            words.extend_from_slice(&s_mov_lit(16 + i as u8, w));
        }
        let br_join = words.len();
        words.push(0); // placeholder s_branch
        let else_pc = words.len();
        // else arm: load alt into s[16:19]
        for (i, &w) in alt.iter().enumerate() {
            words.extend_from_slice(&s_mov_lit(16 + i as u8, w));
        }
        let join_pc = words.len();
        words[br_else] = 0xBF84_0000 | ((else_pc - br_else - 1) as u32 & 0xFFFF);
        words[br_join] = 0xBF82_0000 | ((join_pc - br_join - 1) as u32 & 0xFFFF);
        words.extend_from_slice(&image_sample_lz);
        words.extend_from_slice(&buf_store);
        let native = one_thread_cs(&words);
        let (f, iface) = crate::vk::thread_ir_native(&native).unwrap_or_else(|e| panic!("{case_name}: {e}"));
        for (branch, chosen) in [(0u32, samp_a), (1, alt)] {
            let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
            mem.put(TABLE, &tsharp(1));
            mem.put(TABLE + 64, &vsharp(0x2000, 0, 64, 0));
            let mut m = Machine::new(&f, 32, mem);
            m.push[4] = TABLE as u32;
            m.push[5] = branch;
            m.images = Some(Box::new(WhichSampler(iface.samplers.clone())));
            assert_eq!(m.run(), Stop::Returned);
            assert_eq!(m.mem.dwords.get(&0x2000).copied(), Some(chosen[2]), "{case_name} branch={branch}");
        }
    }
}
