//! The ray/node test (`image_bvh64_intersect_ray`, `image_bvh_intersect_ray`),
//! after origin/main's `crates/victory-shader/src/emit/intersect_tests.rs`
//! (GPL-3.0): its synthetic program (one test per thread, a record each)
//! as written, run through `vk::thread_ir_native` and the interpreter.
//!
//! Ported: `a_node_test_that_answers_no_hit_reads_nothing` (as behaviour:
//! what every thread's result registers hold, and that no node is read).
//! Not ported: the four `..._matches_the_reference_on_the_card` tests and
//! `a_descriptor_of_numbers_is_title_memory_read_through_the_directory`
//! (they compare their DXIL on a D3D12 device with `crate::bvh`; our node
//! test is a SPIR-V helper, `emit_bvh.rs`, which the interpreter does not
//! run), and `a_structure_that_returns_triangle_ids_is_refused_by_name`
//! (their refusal; we always return the barycentric form, docs/vulkan-abi.md
//! section 9).

#![allow(clippy::all, unused_variables)]

use super::*;
use crate::vk::lift::TEST_BVH_BUILDER;
use crate::vk::tir::{Helper, Op};

/// Threads of one group, each with its own record.
const THREADS: u32 = 32;
/// Bytes of a thread's record: the node pointer (two dwords), the ray's
/// extent, origin, direction and inverse direction, and whether the
/// thread runs the instruction.
const RECORD: u32 = 64;
/// What the result registers hold before the instruction.
const BEFORE: [u32; 4] = [1, 2, 3, 4];
/// Dword 1 of the title's descriptor: boxes sorted.
const BOX_SORT: u32 = 0x8000_0000;

/// Their program: per thread, the record's twelve operand dwords in
/// v1-v12, the instruction's result in v13-v16 (under a lane mask of the
/// threads whose record says so), stored to the UAV at 16 bytes a thread.
/// The descriptor is the title's: numbers over the whole address space.
fn program(wide: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x3428_0086, // v_lshlrev_b32 v20, 6, v0
        0xE038_1000,
        0x8002_0114, // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0514, // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0914, // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030_1030,
        0x8002_1114, // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
    ];
    for k in 0..4u32 {
        // v_mov_b32 v13+k, 1+k
        w.push(0x7E00_0200 | (13 + k) << 17 | (0x81 + k));
    }
    w.extend([
        0xBE98_03FF,
        0, // s_mov_b32 s24, 0
        0xBE99_0B81, // s_brev_b32 s25, 1 (BOX_SORT, as the title makes it)
        0xBE9A_03C1, // s_mov_b32 s26, -1
        0xBE9B_03FF,
        0x8900_03FF, // s_mov_b32 s27, 0x890003ff
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
    ]);
    let first = 0x0000_0D01 | (24 >> 2) << 16;
    if wide {
        // image_bvh64_intersect_ray v[13:16], [v1 .. v12], s[24:27]
        w.extend([0xF19C_9F07, first, 0x0504_0302, 0x0908_0706, 0x000C_0B0A]);
    } else {
        // image_bvh_intersect_ray v[13:16], [v1, v3 .. v12], s[24:27]
        w.extend([0xF198_9F07, first, 0x0605_0403, 0x0A09_0807, 0x0000_0C0B]);
    }
    w.extend([
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0x342A_0084, // v_lshlrev_b32 v21, 4, v0
        0xE078_1000,
        0x8003_0D15, // buffer_store_dwordx4 v[13:16], v21, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn native(words: &[u32]) -> NativeProgram {
    let param = |root, sgprs, user_data, slot| RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 22,
        layout: vec![param(RootType::Constants, 4, 0, 0), param(RootType::Table, 1, 4, 1)],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [THREADS, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

const RECORDS: u64 = 0x2000;
const OUT: u64 = 0x3000;

/// With the structure builder off (the default) a node test answers "no
/// hit" and reads nothing: a box node (pointer kinds 4 and 5) enters no
/// child, any other node is missed, whatever lies where the node would be
/// (origin/main `emit/intersect.rs` `no_hit`; their test
/// `a_node_test_that_answers_no_hit_reads_nothing`). Threads whose record
/// says not to run keep what their registers held.
#[test]
fn a_node_test_that_answers_no_hit_reads_nothing() {
    for wide in [true, false] {
        let native = native(&program(wide));
        TEST_BVH_BUILDER.set(Some(false));
        let built = crate::vk::thread_ir_native(&native);
        TEST_BVH_BUILDER.set(None);
        let (f, _) = built.expect("translation");
        assert!(!f.nodes.iter().any(|n| n.op == Op::Call(Helper::BvhRay)), "no node test is written");
        // every memory read that is not put below reads noise: a node
        // that was read would not give these answers
        let mut mem = Memory { range: (0, 1 << 32), noise: Some(7), ..Default::default() };
        mem.put(0x1000, &[RECORDS as u32, 0, THREADS * RECORD, 0x3000_0FAC]);
        mem.put(0x1020, &[OUT as u32, 0, THREADS * 16, 0x3000_0FAC]);
        for t in 0..THREADS {
            // pointer kinds 0-7 in turn, at nodes 1..; a ray along +x
            let pointer = (1 + t) << 3 | (t & 7);
            let mut rec = vec![pointer, 0, 100.0f32.to_bits()];
            rec.extend([0.5f32, 0.25, 0.125, 1.0, 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY].map(f32::to_bits));
            rec.push(u32::from(t % 5 != 4)); // every fifth thread stays out
            rec.extend([0; 3]);
            mem.put(RECORDS + u64::from(t * RECORD), &rec);
        }
        let mut m = Machine::new(&f, 32, mem);
        m.push[8] = 0x1000;
        assert_eq!(m.run(), Stop::Returned);
        assert!(m.mem.missing.is_empty(), "nothing but the records is read: {:x?}", m.mem.missing.keys().take(4).collect::<Vec<_>>());
        for t in 0..THREADS {
            let got: Vec<u32> = (0..4).map(|k| m.mem.dwords.get(&(OUT + u64::from(16 * t + 4 * k))).copied().unwrap_or(0)).collect();
            let want = if t % 5 == 4 {
                BEFORE
            } else if t & 6 == 4 {
                [0xffff_ffff; 4]
            } else {
                [f32::INFINITY.to_bits(), 1.0f32.to_bits(), 0, 0]
            };
            assert_eq!(got, want, "wide {wide}, thread {t} (pointer kind {})", t & 7);
        }
    }
}

/// With the builder on (`VICTORY_BVH_BUILDER=1`) the node is tested: one
/// call of the node test per instruction.
#[test]
fn with_the_builder_on_the_node_is_tested() {
    for wide in [true, false] {
        let native = native(&program(wide));
        TEST_BVH_BUILDER.set(Some(true));
        let built = crate::vk::thread_ir_native(&native);
        TEST_BVH_BUILDER.set(None);
        let (f, _) = built.expect("translation");
        assert_eq!(f.nodes.iter().filter(|n| n.op == Op::Call(Helper::BvhRay)).count(), 1);
    }
}
