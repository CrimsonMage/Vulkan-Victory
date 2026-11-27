//! Console waves of 64 on host subgroups of 32: the behavioural cases of
//! origin/main (fxceless/victory-fork 6742906, GPL-3.0),
//! `crates/victory-shader/src/emit/wave64_tests.rs` and
//! `emit/wave64_after_loop.rs`: the programs and what the console's wave
//! computes are theirs (kept as written, with their CPU references); they
//! ran them on a D3D12 device, here each runs in the interpreter three
//! times ([`run_group`]): each console wave stepped as one (the reference),
//! and as the two host subgroups the card runs, exchanging where the lift
//! says so, the lower half first and the upper half first. All three must
//! give the same memory, and that is compared with the console's.
//!
//! Not ported (they test origin's own lowering, not what a wave computes):
//! `lane_loop_exit_includes_repeated_postdominators`,
//! `a_skip_whose_parked_scalars_…`, `a_skip_whose_parked_scalar_…`,
//! `a_pair_parked_in_lanes_…`, `a_parked_pair_another_save_or_restore_…`
//! (structure passes), `listed_wave64_uses_paired_backend`,
//! `paired_fp32_refuses_an_unconfirmed_guest_mode`,
//! `paired_permute_rejects_gds`, `paired_lds_respects_host_shared_memory_limit`
//! (their paired backend's choices and limits),
//! `nvidia_profile_many_wave64_groups` (a timing run).

#![allow(clippy::all)]

use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::vk::interp::{Machine, Memory, Stop};
use crate::vk::interp_group::{Group, Mode};
use crate::wave::WAVE_OR_V7_TO_S7;

const S_ENDPGM: u32 = 0xBF81_0000;
const EXEC: u32 = 126;
const LIT: u32 = 255;

const fn v(n: u32) -> u32 {
    256 + n
}

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

/// A branch by `imm` dwords from the next instruction (or `s_barrier`, `imm` 0).
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}

/// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`
const fn store(data: u32, addr: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0000 | data << 8 | addr]
}

/// `v<regs>` stored as each thread's dwords of the UAV, the thread's index (v0) times 16 on.
fn stores(regs: [u32; 4]) -> Vec<u32> {
    let mut w = vec![vop2(0x1A, 15, k(4), 0)]; // v_lshlrev_b32 v15, 4, v0
    for (n, r) in regs.into_iter().enumerate() {
        w.extend(store(r, 15, 4 * n as u32));
    }
    w
}

/// `v0` stored by every thread to dword 0 of the UAV.
const STORE_V0: [u32; 2] = [0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16];

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let param = |root, sgprs, user_data, slot| RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![param(RootType::Constants, 4, 0, 0), param(RootType::Table, 1, 4, 1)],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [threads, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn with_lds(mut native: NativeProgram, threads: u32) -> NativeProgram {
    if let StageIo::Cs(cs) = &mut native.io {
        cs.threads = [threads, 1, 1];
        cs.rsrc2 |= 1 << 15; // one 512-byte LDS unit
    }
    native
}

/// One way of running the group: (mode, the upper half of each wave first).
const WAYS: [(Mode, bool); 3] = [(Mode::Waves, false), (Mode::Halves, false), (Mode::Halves, true)];

/// The first `dwords` dwords of the UAV after the group `group` ran with
/// the root constants `consts`, one of [`WAYS`].
fn run_way(native: &NativeProgram, consts: [u32; 4], dwords: u32, group: u32, way: (Mode, bool)) -> Result<Vec<u32>, String> {
    let (f, _) = crate::vk::thread_ir_native(native).map_err(|e| format!("translation: {e}"))?;
    let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
    // the table at 0x1000: a raw V# over 0x2000..
    mem.put(0x1000, &[0x2000, 0, 0x1_0000, 0x3000_0FAC]);
    let threads = crate::vk::lift::Lift::host_threads(native);
    let setup = |m: &mut Machine| {
        m.push[4..8].copy_from_slice(&consts);
        m.push[8] = 0x1000;
        m.group = [group, 0, 0];
    };
    let mut g = Group::new(&f, native.wave64, threads, way.0, mem, setup);
    g.upper_first = way.1;
    match g.run() {
        // (the halves' LDS accesses are ordered by their meetings)
        Stop::Returned if g.lds_races > 0 => Err(format!("{} LDS dwords the halves accessed with no meeting between", g.lds_races)),
        Stop::Returned => Ok((0..u64::from(dwords)).map(|i| g.mem.dwords.get(&(0x2000 + 4 * i)).copied().unwrap_or(0)).collect()),
        s => Err(format!("{s:?}")),
    }
}

/// [`run_way`] in every way; they must agree (the halves the card runs
/// with the wave stepped as one).
fn run_group(native: &NativeProgram, consts: [u32; 4], dwords: u32, group: u32) -> Result<Vec<u32>, String> {
    let wave = run_way(native, consts, dwords, group, WAYS[0]);
    for way in &WAYS[1..] {
        let halves = run_way(native, consts, dwords, group, *way);
        if halves != wave {
            let at = match (&wave, &halves) {
                (Ok(a), Ok(b)) => a.iter().zip(b).position(|(x, y)| x != y).map(|i| format!("dword {i}: {:#x} as one wave, {:#x} as halves", a[i], b[i])),
                _ => None,
            };
            return Err(format!("as halves (upper first: {}) the program differs from the wave: {}", way.1, at.unwrap_or_else(|| format!("{wave:x?} / {halves:x?}"))));
        }
    }
    wave
}

/// Every case `(s0, s1)` against what the console stores; the failures of
/// all of them at the end (their `execute_wave32`).
fn execute(native: &NativeProgram, cases: &[(u32, u32)], expected: impl Fn(u32, u32) -> Vec<u32>) {
    let mut failed = Vec::new();
    for &(s0, s1) in cases {
        let want = expected(s0, s1);
        match run_group(native, [s0, s1, 0, 0], want.len() as u32, 0) {
            Ok(got) if got == want => {}
            Ok(got) => {
                let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                failed.push(format!("s0={s0:#x} s1={s1:#x}: word {i}: got {:#x}, expected {:#x}", got[i], want[i]));
            }
            Err(e) => failed.push(format!("s0={s0:#x} s1={s1:#x}: {e}")),
        }
    }
    assert!(failed.is_empty(), "{} of {} cases failed:\n{}", failed.len(), cases.len(), failed.join("\n"));
}

/// Dword 0 of the UAV.
fn dword0(native: &NativeProgram, s0: u32, s1: u32) -> Result<u32, String> {
    run_group(native, [s0, s1, 0, 0], 1, 0).map(|o| o[0])
}

// ---- emit/wave64_tests.rs ----

/// The console's 16-bit float addition on normal numbers and zero (theirs).
fn add16(a: u16, b: u16) -> u16 {
    let wide = |h: u16| if h == 0 { 0.0 } else { f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13) };
    let f = wide(a) + wide(b);
    if f == 0.0 {
        return 0;
    }
    let b = f.to_bits();
    let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
    let rest = b & 0x1FFF;
    (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
}

/// Thread t's term 0x3C00 + 37 t in v7, then the wave sum of 16-bit floats into s7 (the wave
/// OR's words with v_add_f16 for v_or_b32: opcode 50 for 28), stored by every thread.
fn sum16_words() -> Vec<u32> {
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.push(0x7E00_0207); // v_mov_b32 v0, s7
    words.extend(STORE_V0);
    words
}

fn tree16(terms: Vec<u16>) -> u16 {
    let mut tree = terms;
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add16(c[0], c[1])).collect();
    }
    tree[0]
}

/// Theirs: `a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit`.
#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    let native = native(&sum16_words());
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let in_a_row = terms.iter().copied().reduce(add16).unwrap();
    let tree = tree16(terms);
    assert_ne!(tree, in_a_row, "these terms tell the tree from another order");
    assert_eq!(dword0(&native, 0, 0), Ok(u32::from(tree)), "tree {tree:#x}, in a row {in_a_row:#x}");
}

/// Theirs: `a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit`: the
/// lanes at or past s0 have left; on the console they add +0.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0,           // s_cbranch_execz to the end (filled in below)
    ];
    words.extend(sum16_words());
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let mut failed = Vec::new();
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let tree = tree16((0..64u32).map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 }).collect());
        let got = dword0(&native, stay, 0);
        if got != Ok(u32::from(tree)) {
            failed.push(format!("{stay} lanes stay: got {got:x?}, the console's tree {tree:#x}"));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Theirs: `a_16_bit_sum_rounds_to_nearest_when_asked_to`: 0x3C03 + 0x4000 is 3 + 1.5 steps,
/// which the console rounds up to 0x4202.
#[test]
fn a_16_bit_sum_rounds_to_nearest() {
    let mut words = vec![
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
    ];
    words.extend(STORE_V0);
    assert_eq!(dword0(&native(&words), 0, 0).map(|x| x & 0xFFFF), Ok(0x4202));
}

/// Theirs: `program(skip)`. 64 threads in one console wave. Each thread t holds the bit
/// `1 << (t >> 1)`; the threads below s0 OR theirs over the wave inside `if (t < s0) { ... }`
/// and store the result to dword 0.
fn wave_or_inside(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// Theirs: `a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide`: the OR is of both
/// halves, whichever half's lanes take the branch.
#[test]
fn a_wave_or_inside_a_lane_branch_is_of_both_halves() {
    let mut failed = Vec::new();
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        let native = wave_or_inside(skip);
        for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
            let got = dword0(&native, below, 0);
            if got != Ok(expected) {
                failed.push(format!("{name}: {below} threads in the branch: {got:x?}, expected {expected:#x}"));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Theirs: `a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is` (a 32-lane wave).
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let mut words = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
    ];
    words.extend(STORE_V0);
    let native = native_of(&words, false, 32);
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        assert_eq!(dword0(&native, below, other), Ok(expected), "{below} lanes in the mask, s1 = {other}");
    }
}

/// Theirs: `scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide` (a
/// 32-lane wave): with some lane inside, every lane sees the inner test. Lane 31 stores.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_is_the_waves() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        assert_eq!(dword0(&native, below, s1), Ok(expected), "{below} lanes inside, s1 = {s1}");
    }
}

/// Theirs: `threads_past_a_dispatch_x_count_do_nothing`, without a count (all 64 threads
/// run): every lane stores (bit count of EXEC on entry, wave OR of 1 << (lane >> 1)) at
/// `uav[lane * 2..]`. Their program reads its table through a spilled user-data pointer;
/// here it has the plain layout.
#[test]
fn every_thread_counts_exec_and_ors_over_both_halves() {
    let mut words = vec![
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
    ];
    words.extend_from_slice(&WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x3402_0083, // v_lshlrev_b32 v1, 3, v0 (8 bytes a lane)
        0x7E04_0206, // v_mov_b32 v2, s6
        0x7E06_0207, // v_mov_b32 v3, s7
        0x38 << 26 | 0x1D << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 2 << 8 | 1, // buffer_store_dwordx2 v[2:3], v1, s[8:11], 0 offen
    ]);
    let want: Vec<u32> = (0..64).flat_map(|_| [64, u32::MAX]).collect();
    assert_eq!(run_group(&native(&words), [0; 4], 128, 0), Ok(want));
}

/// Theirs, the cases with a thread count (45, 1): `DispatchX` with flags bit 5 counts
/// threads, and the threads at or past the count are not in EXEC.
#[test]
#[ignore = "needs a dispatch that counts threads (their USER_DISPATCH_THREADS; origin-findings.md row 19: our host passes x as groups)"]
fn threads_past_a_dispatch_x_count_do_nothing() {}

/// Theirs: `far_early_out`. 128 threads (two console waves) that leave early when no lane of
/// their wave is below s0: `s_cbranch_execnz` over a far jump to the end. Then the gate:
/// every thread stores 1, or 2 when some lane of its console wave that is still running has
/// the thread number s1.
fn far_early_out(extra: &[u32], work_at_end: bool) -> NativeProgram {
    let gate = [
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE90_106A, // s_bcnt1_i32_b64 s16, vcc
        0xBF06_8010, // s_cmp_eq_u32 s16, 0
        0x8511_8281, // s_cselect_b32 s17, 1, 2
        0x7E02_0211, // v_mov_b32 v1, s17
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 1 << 8 | 2, // buffer_store_dword v1, v2, s[8:11], 0 offen
    ];
    let jump = [
        0xBE8E_1F00, // s_getpc_b64 s[14:15]
        0x800E_FF0E,
        4 * (4 + gate.len() as u32), // s_add_u32 s14, s14, the way to the end
        0x820F_800F,                 // s_addc_u32 s15, s15, 0
        0xBE80_200E,                 // s_setpc_b64 s[14:15]
    ];
    let mut words = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF89_0000 | (extra.len() + jump.len()) as u32, // s_cbranch_execnz over the jump
    ];
    words.extend_from_slice(extra);
    words.extend_from_slice(&jump);
    words.extend_from_slice(&gate);
    if work_at_end {
        words.push(0x7E06_0281); // v_mov_b32 v3, 1
    }
    native_of(&words, true, 128)
}

/// What the gate of [`far_early_out`] stores in each of 128 threads.
fn gate_expected(below: u32, failing: u32) -> Vec<u32> {
    (0..128u32)
        .map(|t| {
            let some_fail = failing < below && failing / 64 == t / 64;
            match (t < below, some_fail) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => 2,
            }
        })
        .collect()
}

/// Theirs: `a_far_early_out_is_left_out_and_the_gate_counts_the_lanes_of_both_halves`, and
/// the programs of `a_far_jump_that_does_work_is_not_an_early_out` (the same gate).
#[test]
fn a_far_early_out_and_the_gate_count_the_lanes_of_both_halves() {
    let mut failed = Vec::new();
    for (name, native) in [
        ("plain", far_early_out(&[], false)),
        ("work in the jump", far_early_out(&[0xBE94_0385], false)), // s_mov_b32 s20, 5
        ("work at the target", far_early_out(&[], true)),
    ] {
        for (below, failing) in [(128u32, 40u32), (128, 100), (128, 63), (40, 50), (40, 10), (33, 32), (1, 0), (0, 5)] {
            let got = run_group(&native, [below, failing, 0, 0], 128, 0);
            if got != Ok(gate_expected(below, failing)) {
                let got = got.map(|g| g.iter().map(|x| x.to_string()).collect::<String>());
                failed.push(format!("{name}: {below} threads in bounds, thread {failing} fails: {got:?}"));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Theirs: `a_backward_permute_reads_within_the_lanes_half`: lane t names lane t ^ 33, in the
/// other half, and reads lane t ^ 1.
#[test]
fn a_backward_permute_reads_within_the_lanes_half() {
    let body = [
        0x3A08_00A1, // v_xor_b32 v4, 33, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xDACC_0000,
        0x0200_0004, // ds_bpermute_b32 v2, v4, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0201, // buffer_store_dword v2, v1, s[8:11], 0 offen
    ];
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads).map(|t| t ^ 1).collect();
        assert_eq!(run_group(&native_of(&body, true, threads), [0; 4], threads, 0), Ok(want), "{threads} threads");
    }
}

// ---- emit/wave64_after_loop.rs ----

/// Theirs: `loop_program`. Lane t goes round a loop `(t & s0) + 1` times, adding `t & s1` to
/// its sum each round. After it, over the lanes whose `t & s1` is not 0: their count
/// (`s_bcnt1_i32_b64`), the first of them (`v_readfirstlane_b32` of the thread index) and
/// each one's number among them (`v_mbcnt_lo/hi`). Every thread stores its sum, its number
/// (0x55 where it has none), the count and the first. `spread`: 128 threads, placed by the
/// group id (s17).
fn loop_program(spread: bool) -> NativeProgram {
    let mut w = vec![
        vop2(0x1B, 4, 0, 0), // v_and_b32 v4, s0, v0: the lane's last round
        vop2(0x1B, 5, 1, 0), // v_and_b32 v5, s1, v0: what the lane adds a round
        vop1(0x01, 7, k(0)), // v_mov_b32 v7, 0
        vop1(0x01, 8, LIT),  // v_mov_b32 v8, 0x55
        0x55,
        sop1(0x03, 6, k(0)),      // s_mov_b32 s6, 0: the round
        sop1(0x04, 2, EXEC),      // s_mov_b64 s[2:3], exec
        sop1(0x04, 12, EXEC),     // s_mov_b64 s[12:13], exec: the lanes still going round
        sop1(0x04, 14, EXEC),     // s_mov_b64 s[14:15], exec
        sop2(0x0F, EXEC, 14, 12), // loop: s_and_b64 exec, s[14:15], s[12:13]
        vop2(0x25, 7, v(5), 7),   // v_add_nc_u32 v7, v5, v7
        sop1(0x04, EXEC, 14),     // s_mov_b64 exec, s[14:15]
        sop2(0x02, 6, 6, k(1)),   // s_add_i32 s6, s6, 1
    ];
    w.extend(vop3(0xC4, 14, 6, v(4))); // v_cmp_gt_u32 s[14:15], s6, v4: the lanes that are done
    w.extend([
        sop1(0x24, 14, 14),       // s_and_saveexec_b64 s[14:15], s[14:15]
        sop2(0x15, 12, 12, EXEC), // s_andn2_b64 s[12:13], s[12:13], exec
        sopp(0x05, -9),           // s_cbranch_scc1 loop
        sop1(0x04, EXEC, 2),      // s_mov_b64 exec, s[2:3]
        vopc(0xD5, k(0), 5),      // v_cmpx_ne_u32 exec, 0, v5
        sop1(0x10, 7, EXEC),      // s_bcnt1_i32_b64 s7, exec
        vop1(0x02, 16, v(0)),     // v_readfirstlane_b32 s16, v0
    ]);
    w.extend(vop3(0x365, 8, EXEC, k(0))); // v_mbcnt_lo_u32_b32 v8, exec_lo, 0
    w.extend(vop3(0x366, 8, EXEC + 1, v(8))); // v_mbcnt_hi_u32_b32 v8, exec_hi, v8
    w.extend([
        sop1(0x04, EXEC, 2), // s_mov_b64 exec, s[2:3]
        vop1(0x01, 9, 7),    // v_mov_b32 v9, s7
        vop1(0x01, 10, 16),  // v_mov_b32 v10, s16
    ]);
    if !spread {
        w.extend(stores([7, 8, 9, 10]));
        return native(&w);
    }
    w.extend([
        vop1(0x01, 11, 17), // group ID x
        vop2(0x1a, 11, k(7), 11),
        vop2(0x25, 15, v(11), 0),
        vop2(0x1a, 15, k(4), 15),
    ]);
    for (n, reg) in [7, 8, 9, 10].into_iter().enumerate() {
        w.extend(store(reg, 15, 4 * n as u32));
    }
    let mut native = native(&w);
    let StageIo::Cs(cs) = &mut native.io else { unreachable!() };
    cs.rsrc2 = (17 << 1) | (1 << 7);
    cs.threads = [128, 1, 1];
    native
}

/// What the console's lanes store with the root constants `s0` and `s1` (theirs).
fn console(threads: u32, s0: u32, s1: u32) -> Vec<u32> {
    (0..threads)
        .flat_map(|t| {
            let first = t / 64 * 64;
            let counted: Vec<u32> = (first..first + 64).filter(|t| t & s1 != 0).collect();
            let number = if t & s1 != 0 { counted.iter().filter(|&&u| u < t).count() as u32 } else { 0x55 };
            [(t & s1) * ((t & s0) + 1), number, counted.len() as u32, counted[0]]
        })
        .collect()
}

/// Theirs: `nvidia_after_divergent_loop_matches_cpu` (and the cases of
/// `listed_wave64_uses_paired_backend`).
#[test]
fn after_a_divergent_loop_counts_and_first_lanes_match_the_console() {
    let cases = [(0, 1), (3, 1), (7, 5), (24, 63), (7, 32), (7, 16), (63, 9)];
    execute(&loop_program(false), &cases, |s0, s1| console(64, s0, s1));
}

/// Theirs: `nvidia_two_waves_progress_independently`: bit 6 tells the two waves' trip counts
/// and masks apart.
#[test]
fn two_waves_progress_independently() {
    let mut native = loop_program(false);
    let StageIo::Cs(cs) = &mut native.io else { unreachable!() };
    cs.threads = [128, 1, 1];
    execute(&native, &[(127, 65), (63, 65), (7, 32), (7, 16)], |s0, s1| console(128, s0, s1));
}

/// Theirs: `nvidia_spread_dispatch_runs_asked_wave64_groups`, one group at a time: the
/// group's id places its output (their host spreads 257 asked groups over x and z; ours
/// passes the counts on).
#[test]
fn a_groups_id_places_its_output() {
    let native = loop_program(true);
    for group in [0u32, 1, 3] {
        let mut want = vec![0; 128 * 4 * group as usize];
        want.extend(console(128, 127, 65));
        assert_eq!(run_group(&native, [127, 65, 0, 0], want.len() as u32, group), Ok(want), "group {group}");
    }
}

/// Theirs: `execute_lds`: every lane writes its slot and reads its partner's in the other
/// half, with no explicit `s_barrier`.
fn lds_between_halves(threads: u32) {
    let mut body = vec![
        vop2(0x1a, 1, k(2), 0), // v1 = byte address of this lane
        vop2(0x25, 2, k(1), 0), // v2 = thread index + 1
        0xd834_0000,
        0x0000_0201,             // ds_write_b32 v1, v2
        vop2(0x1d, 3, k(32), 0), // v3 = partner lane
        vop2(0x1a, 3, k(2), 3),
        0xd8d8_0000,
        0x0400_0003, // ds_read_b32 v4, v3, no explicit s_barrier
    ];
    body.extend(vop3(0x365, 5, 193, k(0))); // v_mbcnt_lo
    body.extend(vop3(0x366, 5, 193, v(5))); // v_mbcnt_hi
    body.extend(stores([4, 5, 2, 0]));
    execute(&with_lds(native(&body), threads), &[(0, 0)], |_, _| {
        (0..threads).flat_map(|lane| [(lane ^ 32) + 1, lane % 64, lane + 1, lane]).collect()
    });
}

/// Theirs: `nvidia_lds_between_wave_halves_matches_cpu`.
#[test]
fn lds_between_wave_halves_matches_the_console() {
    lds_between_halves(64);
}

/// Theirs: `nvidia_lds_between_two_waves_matches_cpu`.
#[test]
fn lds_between_two_waves_matches_the_console() {
    lds_between_halves(128);
}

/// Theirs: `nvidia_lds_across_console_waves_matches_cpu`: a write, `s_barrier`, a read of the
/// other console wave's slot.
#[test]
fn lds_across_console_waves_matches_the_console() {
    let mut body = vec![
        vop2(0x1a, 1, k(2), 0),
        vop2(0x25, 2, k(1), 0),
        0xd834_0000,
        0x0000_0201,
        sopp(0x0a, 0),
        vop2(0x1d, 3, k(64), 0),
        vop2(0x1a, 3, k(2), 3),
        0xd8d8_0000,
        0x0400_0003,
    ];
    body.extend(stores([4, 2, 0, 0]));
    execute(&with_lds(native(&body), 128), &[(0, 0)], |_, _| (0..128).flat_map(|t| [(t ^ 64) + 1, t + 1, t, t]).collect());
}

/// Theirs: `nvidia_water_lds_pair_exchange`: `ds_write2_b32` and `ds_read2_b32` of the
/// partner lane's pair.
#[test]
fn water_lds_pair_exchange() {
    let mut body = vec![
        vop2(0x1a, 1, k(3), 0),
        vop2(0x25, 2, k(1), 0),
        vop2(0x25, 3, k(64), 0),
        0xd838_0100,
        0x0003_0201,
        vop2(0x1d, 4, k(32), 0),
        vop2(0x1a, 4, k(3), 4),
        0xd8dc_0001,
        0x0700_0004,
    ];
    body.extend(stores([7, 8, 2, 3]));
    execute(&with_lds(native(&body), 64), &[(0, 0)], |_, _| {
        (0..64).flat_map(|lane| [(lane ^ 32) + 64, (lane ^ 32) + 1, lane + 1, lane + 64]).collect()
    });
}

/// Theirs: `nvidia_water_lds_masked_min_max`: the lanes whose `t ^ s1` is below s0 take
/// part in a `ds_min_u32` and a `ds_max_u32`; all read both back.
#[test]
fn water_lds_masked_min_max() {
    let mut body = vec![
        vop2(0x1a, 1, k(3), 0),
        vop1(0x01, 2, k(0)),
        vop1(0x01, 3, 193),
        0xd838_0100,
        0x0003_0201,
        vop1(0x01, 4, k(0)),
        vop2(0x25, 5, k(1), 0),
        vop2(0x1d, 6, 1, 0),
        sop1(0x04, 2, EXEC),
        vopc(0xd4, 0, 6),
        sopp(0x08, 4),
        0xd81c_0004,
        0x0000_0504,
        0xd820_0000,
        0x0000_0504,
        sop1(0x04, EXEC, 2),
        0xd8dc_0100,
        0x0700_0004,
    ];
    body.extend(stores([7, 8, 0, 5]));
    execute(&with_lds(native(&body), 64), &[(0, 0), (1, 0), (31, 0), (32, 32), (33, 32), (64, 0)], |limit, xor| {
        let active = (0..64).filter(|lane| (lane ^ xor) < limit).map(|lane| lane + 1);
        let max = active.clone().max().unwrap_or(0);
        let min = active.min().unwrap_or(u32::MAX);
        (0..64).flat_map(|lane| [max, min, lane, lane + 1]).collect()
    });
}

/// Theirs: `nvidia_wave_reduction_and_lane_reads_match_cpu`: the low half contributes bit 0
/// and the high half bit 1 to a wave OR; `v_readlane_b32` of lane 37.
#[test]
fn a_wave_reduction_and_lane_reads_match_the_console() {
    let mut body = vec![vop2(0x16, 7, k(5), 0), vop2(0x25, 7, k(1), 7)];
    body.extend(WAVE_OR_V7_TO_S7);
    body.extend(vop3(0x360, 16, v(0), k(37))); // v_readlane_b32 s16, v0, 37
    body.extend([vop1(0x01, 9, 7), vop1(0x01, 10, 16)]);
    body.extend(stores([9, 10, 0, 0]));
    execute(&native(&body), &[(0, 0)], |_, _| (0..64).flat_map(|lane| [3, 37, lane, lane]).collect());
}

/// Theirs: `permute_half_fixture` (`nvidia_paired_permute_obeys_exec`,
/// `exact_permute_obeys_exec`): `ds_bpermute_b32` under an EXEC of the lanes below s0, with
/// an offset: a lane outside EXEC keeps its 17, a source outside EXEC reads 0.
#[test]
fn a_permute_obeys_exec() {
    for offset in [0u32, 4, 128] {
        let mut body = vec![
            sop1(0x04, 2, EXEC),
            vop2(0x25, 2, k(1), 0),
            vop2(0x1d, 3, k(33), 0),
            vop2(0x1a, 3, k(2), 3),
            vop1(0x01, 4, k(17)),
            vopc(0xd4, 0, 0),
            0xdacc_0000 | offset,
            0x0400_0203,
            sop1(0x04, EXEC, 2),
        ];
        body.extend(stores([4, 0, 2, 0]));
        execute(&native(&body), &[(0, 0), (1, 0), (31, 0), (32, 0), (33, 0), (63, 0), (64, 0)], |limit, _| {
            (0..64)
                .flat_map(|lane| {
                    let partner = (lane & 32) | (((lane ^ 33) + offset / 4) & 31);
                    let value = if lane >= limit {
                        17
                    } else if partner >= limit {
                        0
                    } else {
                        partner + 1
                    };
                    [value, lane, lane + 1, lane]
                })
                .collect()
        });
    }
}

/// Theirs: `fp32_reduce`: the wave OR's words with `op` for its OR and `identity` for its 0.
fn fp32_reduce(op: u32, identity: u32) -> Vec<u32> {
    let mut words = WAVE_OR_V7_TO_S7.to_vec();
    if identity != 0 {
        words[2] = (words[2] & !0x1ff) | LIT;
        words.insert(3, identity);
    }
    for w in &mut words {
        if *w >> 25 == 28 {
            *w = (*w & 0x01ff_ffff) | op << 25;
        }
    }
    words
}

/// Theirs: `fp32_program`: lane t holds s1 where `t & 3` is 0 and `other[t & 3 - 1]`
/// elsewhere (`lds`: through LDS from the partner lane of the other half); the lanes whose
/// `t ^ 45` is below s0 reduce theirs over the wave; every thread stores the result.
fn fp32_program(op: u32, identity: u32, other: [u32; 3], lds: bool) -> NativeProgram {
    let mut body = vec![vop1(1, 7, 1), vop2(0x1b, 4, k(3), 0)];
    for (i, bits) in other.into_iter().enumerate() {
        body.extend([
            vopc(0xc5, k(i as u32 + 1), 4), // v_cmp_ne_u32 vcc, i+1, v4
            vop2(1, 7, LIT, 7),
            bits, // v_cndmask_b32 v7, bits, v7, vcc
        ]);
    }
    if lds {
        body.extend([
            vop2(0x1a, 2, k(2), 0),
            0xd834_0000,
            0x0000_0702, // ds_write_b32 v2, v7
            vop2(0x1d, 3, k(32), 0),
            vop2(0x1a, 3, k(2), 3),
        ]);
    }
    body.extend([
        sop1(4, 12, EXEC),
        vop2(0x1d, 5, k(45), 0), // sparse EXEC across either half
        vopc(0xd4, 0, 5),
    ]);
    if lds {
        body.extend([0xd8d8_0000, 0x0700_0003]); // ds_read_b32 v7, v3, under EXEC
    }
    body.extend(fp32_reduce(op, identity));
    body.extend([sop1(4, EXEC, 12), vop1(1, 9, 7)]);
    body.extend(stores([9, 0, 0, 0]));
    let program = native(&body);
    if lds { with_lds(program, 64) } else { program }
}

/// The console's 32-bit float sum (3), minimum (15) and maximum (16) of two lanes, as their
/// paired backend's reference has it: denormals in and out are zeros with the sign kept
/// (`ftz`: their `VICTORY_WAVE_PAIRS_FP32=non-ieee-rne-ftz`), else IEEE as it is.
fn fp32_reference(op: u32, a: u32, b: u32, ftz: bool) -> u32 {
    let flush = |bits: u32| if ftz && bits & 0x7fff_ffff < 0x0080_0000 { bits & 0x8000_0000 } else { bits };
    let (a, b) = (flush(a), flush(b));
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    if op == 3 {
        if x.is_nan() {
            return a;
        }
        if y.is_nan() {
            return b;
        }
        if x.is_infinite() && y == -x {
            return 0xffc0_0000;
        }
        return flush((x + y).to_bits());
    }
    if y.is_nan() {
        return a;
    }
    if x.is_nan() {
        return b;
    }
    if x == 0.0 && y == 0.0 {
        return if op == 15 { a | b } else { a & b };
    }
    if (op == 15 && x < y) || (op == 16 && x > y) { a } else { b }
}

/// Theirs: `fp32_expected`: the console's tree over the 64 lanes.
fn fp32_expected(op: u32, ident: u32, values: [u32; 4], active: u32, lds: bool, ftz: bool) -> Vec<u32> {
    let r = |a, b| fp32_reference(op, a, b, ftz);
    let mut lanes: [u32; 64] =
        std::array::from_fn(|lane| if (lane as u32 ^ 45) < active { values[(if lds { lane ^ 32 } else { lane }) & 3] } else { ident });
    for (width, indices) in
        [(8, &[1, 0, 3, 2, 5, 4, 7, 6][..]), (8, &[2, 2, 0, 0, 6, 6, 4, 4][..]), (8, &[4, 4, 4, 4, 0, 0, 0, 0][..])]
    {
        let old = lanes;
        lanes = std::array::from_fn(|i| r(old[i / width * width + indices[i % width]], old[i]));
    }
    let old = lanes;
    lanes = std::array::from_fn(|i| r(old[i ^ 15], old[i]));
    let old = lanes;
    lanes = std::array::from_fn(|i| r(old[(i / 16 ^ 1) * 16], old[i]));
    let result = r(lanes[32], lanes[0]);
    (0..64).flat_map(|lane| [result, lane, lane, lane]).collect()
}

/// A NaN's payload and a zero's sign are not compared (which operand's NaN an operation
/// returns, and the sign of a zero result of a minimum or maximum, are the host's: opcode
/// audit, `v_max_f32`).
fn same_float(a: u32, b: u32) -> bool {
    let nan = |x: u32| x & 0x7fff_ffff > 0x7f80_0000;
    a == b || (nan(a) && nan(b)) || (a | b) & 0x7fff_ffff == 0
}

/// The cases of an fp32 reduction against the console's tree: the failures.
fn fp32_cases(op: u32, ident: u32, other: [u32; 3], lds: bool, cases: &[(u32, u32)], ftz: bool) -> Vec<String> {
    let native = fp32_program(op, ident, other, lds);
    let mut failed = Vec::new();
    for &(n, src) in cases {
        let want = fp32_expected(op, ident, [src, other[0], other[1], other[2]], n, lds, ftz);
        match run_group(&native, [n, src, 0, 0], 256, 0) {
            Ok(got) if got.iter().zip(&want).all(|(&a, &b)| same_float(a, b)) => {}
            Ok(got) => failed.push(format!("op {op} other {other:x?} lds {lds}: {n} lanes, s1 {src:#x}: got {:#x}, the console's tree {:#x}", got[0], want[0])),
            Err(e) => failed.push(format!("op {op} other {other:x?} lds {lds}: {n} lanes, s1 {src:#x}: {e}")),
        }
    }
    failed
}

/// Theirs: `nvidia_paired_fp32_ordered_sum`, the cases whose terms and partial sums are
/// normal numbers, zeros, infinities or NaNs: the sum in the console's order (a tree over
/// the lanes, not a row).
#[test]
fn a_32_bit_float_sum_is_the_consoles_tree() {
    let other = [1.0f32.to_bits(), (-16_777_216.0f32).to_bits(), 1.0f32.to_bits()];
    let values = [16_777_216.0f32.to_bits(), other[0], other[1], other[2]];
    let linear = values.into_iter().cycle().take(64).fold(0, |sum, term| fp32_reference(3, sum, term, false));
    assert_ne!(fp32_expected(3, 0, values, 64, false, false)[0], linear);
    let big = 16_777_216.0f32.to_bits();
    let mut failed = fp32_cases(3, 0, other, false, &[(0, 0), (32, big), (33, big), (64, big), (64, 0x7f80_0021), (19, 0x7fc0_0011)], false);
    failed.extend(fp32_cases(3, 0x8000_0000, [0x8000_0000; 3], false, &[(64, 0x8000_0000)], false));
    failed.extend(fp32_cases(3, 0, [0xff80_0000, 0x0080_0001, 0x8080_0000], true, &[(7, 0x7f80_0000), (64, 0x7f80_0000), (64, 0x0080_0000)], false));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Theirs: the cases of `nvidia_paired_fp32_ordered_sum` with a denormal term or partial
/// sum, by their reference: flushed to zero. Ours keeps them (IEEE, as the interpreter and
/// `oracle/card.rs` have the console; the card flushes).
#[test]
#[ignore = "their reference flushes denormals to zero (a guest mode they asked to be confirmed, VICTORY_WAVE_PAIRS_FP32); ours computes IEEE: differs only in sums with denormal terms"]
fn a_32_bit_float_sum_flushes_denormals() {
    let mut failed = fp32_cases(3, 0x8000_0000, [0x8000_0000; 3], false, &[(1, 0x8000_0001)], true);
    failed.extend(fp32_cases(3, 0, [0, 0x8080_0000, 0x8000_0000], false, &[(64, 0x0080_0001), (64, 0x0080_0002)], true));
    failed.extend(fp32_cases(3, 0, [0x8080_0001, 0x0080_0000, 0x8080_0001], false, &[(64, 0x0080_0000)], true));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Theirs: `nvidia_paired_fp32_min_max` (a NaN loses to a number; which NaN wins and a
/// zero's sign are not compared: [`same_float`]).
#[test]
fn a_32_bit_float_minimum_and_maximum_over_the_wave() {
    let mut failed = Vec::new();
    for (op, ident) in [(15, 0x7f80_0000), (16, 0xff80_0000)] {
        for other in [[0, 0x8000_0000, 0x8000_0001], [0x7fc0_0001, 0x7f80_0002, 0xffc0_0003]] {
            let cases = [(0, 0), (17, 0x7fc0_0011), (32, 0x7f80_0000), (64, 0x7f80_0011), (64, 1)];
            failed.extend(fp32_cases(op, ident, other, false, &cases, false));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Theirs: `water_sums` (`paired_fp32_takes_the_water_sums`): two 32-bit float sums in a
/// row, then the 64-bit float count of the lanes (a sum of `term`). They test which forms
/// their paired backend takes; here the sums themselves: the two sums are the tree of s1
/// over the lanes in EXEC, the count is the term times those lanes.
#[test]
fn the_waters_sums_are_the_consoles_trees() {
    let water_sums = |term_hi: u32| {
        let mut body = vec![
            vop1(1, 7, 1), // v_mov_b32 v7, s1
            sop1(4, 12, EXEC),
            vop2(0x1d, 5, k(45), 0), // sparse EXEC across either half
            vopc(0xd4, 0, 5),
        ];
        body.extend(fp32_reduce(3, 0));
        body.push(vop1(1, 8, 7)); // v_mov_b32 v8, s7: the first sum
        body.extend(fp32_reduce(3, 0));
        body.push(vop1(1, 10, k(0))); // v_mov_b32 v10, 0
        body.extend([vop1(1, 11, LIT), term_hi]);
        body.extend([
            0xBE92_25C1, // s_or_saveexec_b64 s[18:19], -1
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_0280, // v_mov_b32 v3, 0
            0xD501_0002,
            0x004A_1502, // v_cndmask_b32 v2, v2, v10, s[18:19]
            0xD501_0003,
            0x004A_1703, // v_cndmask_b32 v3, v3, v11, s[18:19]
        ]);
        for dpp in [0xDE54_C100, 0x9360_1200, 0x0009_2400] {
            body.extend([
                0x7E08_02E9,
                dpp | 2, // v_mov_b32 v4, v2 dpp8:[...]
                0x7E0A_02E9,
                dpp | 3, // v_mov_b32 v5, v3 dpp8:[...]
                0xD564_0002,
                0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
            ]);
        }
        body.extend([
            0x7E08_02FA,
            0xFF09_4002, // v_mov_b32 v4, v2 row_mirror bound_ctrl:1
            0x7E0A_02FA,
            0xFF09_4003, // v_mov_b32 v5, v3 row_mirror bound_ctrl:1
            0xD564_0002,
            0x0002_0504,
            0xD778_0004,
            0x0201_0102, // v_permlanex16_b32 v4, v2, 0, 0
            0xD778_0005,
            0x0201_0103, // v_permlanex16_b32 v5, v3, 0, 0
            0xD564_0002,
            0x0002_0504,
            0xD760_0010,
            0x0001_4102, // v_readlane_b32 s16, v2, 32
            0xD760_0011,
            0x0001_4103, // v_readlane_b32 s17, v3, 32
            0xD564_0002,
            0x0002_0410, // v_add_f64 v[2:3], s[16:17], v[2:3]
            0xD760_0010,
            0x0001_0102, // v_readlane_b32 s16, v2, 0
            0xD760_0011,
            0x0001_0103, // v_readlane_b32 s17, v3, 0
            sop1(3, 14, 16),
            sop1(3, 15, 17),
            sop1(4, EXEC, 18), // s_mov_b64 exec, s[18:19]
            sop1(4, EXEC, 12),
            vop1(1, 9, 7),   // the second sum
            vop1(1, 10, 14), // the count's low dword
            vop1(1, 11, 15), // the count's high dword
        ]);
        body.extend(stores([8, 9, 10, 11]));
        native(&body)
    };
    let mut failed = Vec::new();
    // (the count's term, 1.0 in the water; -1.0 counts too)
    for term_hi in [0x3FF0_0000u32, 0xBFF0_0000, 0] {
        let native = water_sums(term_hi);
        for (n, src) in [(64u32, 1.5f32.to_bits()), (40, 16_777_217.0f32.to_bits()), (1, 3.0f32.to_bits()), (0, 5.0f32.to_bits())] {
            let sum = fp32_expected(3, 0, [src; 4], n, false, false)[0];
            let term = f64::from_bits(u64::from(term_hi) << 32);
            let count = (0..n).fold(0.0f64, |a, _| a + term).to_bits();
            // (the first sum is copied under the sparse EXEC: a lane outside it keeps its 0)
            let want: Vec<u32> = (0..64u32).flat_map(|t| [if t ^ 45 < n { sum } else { 0 }, sum, count as u32, (count >> 32) as u32]).collect();
            match run_group(&native, [n, src, 0, 0], 256, 0) {
                Ok(got) if got == want => {}
                Ok(got) => failed.push(format!("term {term_hi:#x}, {n} lanes: got {:x?}, the console's {:x?}", &got[..4], &want[..4])),
                Err(e) => failed.push(format!("term {term_hi:#x}, {n} lanes: {e}")),
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Theirs: `tile_min_max` (`accumulator_inits_are_hoisted_to_the_start`, the water tile
/// programs' LDS use): thread 0 alone stores the tile's starting maximum (dword 0) and
/// minimum (-1, dword 1), then every thread `ds_max_u32`s and `ds_min_u32`s its index in and
/// reads both back. They test that the starts are hoisted (their threads return early);
/// here what the console's wave reads back: the maximum and minimum over all 64 threads.
#[test]
fn a_tiles_minimum_and_maximum_start_before_every_threads_update() {
    for max_start in [0u32, 5, 100] {
        let mut body = vec![
            sop1(4, 12, EXEC),   // s_mov_b64 s[12:13], exec
            vopc(0xd2, k(0), 0), // v_cmpx_eq_u32 exec, 0, v0
            vop1(1, 3, LIT),
            max_start,         // v_mov_b32 v3, max_start
            vop1(1, 4, k(65)), // v_mov_b32 v4, -1
            vop1(1, 5, k(0)),  // v_mov_b32 v5, 0
            0xD838_0100,
            0x0004_0305,       // ds_write2_b32 v5, v3, v4 offset1:1
            sop1(4, EXEC, 12), // s_mov_b64 exec, s[12:13]
            vop1(1, 5, k(0)),  // v_mov_b32 v5, 0
            0xD81C_0004,
            0x0000_0005, // ds_min_u32 v5, v0 offset:4
            0xD820_0000,
            0x0000_0005, // ds_max_u32 v5, v0
            0xD8DC_0100,
            0x0600_0005, // ds_read2_b32 v[6:7], v5 offset1:1
        ];
        body.extend(stores([6, 7, 0, 0]));
        execute(&with_lds(native(&body), 64), &[(0, 0)], |_, _| (0..64).flat_map(|t| [max_start.max(63), 0, t, t]).collect());
    }
}

/// Ours: as halves the programs above do exchange (the comparison of the
/// ways would say nothing if they did not): the divergent loop's masks and
/// first lane, the lane branch's wave OR, the LDS meetings.
#[test]
fn the_halves_of_these_programs_meet() {
    let exchanges = |native: &NativeProgram, consts: [u32; 4]| {
        let (f, _) = crate::vk::thread_ir_native(native).expect("translation");
        let mut mem = Memory { range: (0, 1 << 32), ..Default::default() };
        mem.put(0x1000, &[0x2000, 0, 0x1_0000, 0x3000_0FAC]);
        let setup = |m: &mut Machine| {
            m.push[4..8].copy_from_slice(&consts);
            m.push[8] = 0x1000;
        };
        let mut g = Group::new(&f, native.wave64, crate::vk::lift::Lift::host_threads(native), Mode::Halves, mem, setup);
        assert_eq!(g.run(), Stop::Returned);
        g.exchanges
    };
    let counts = [
        exchanges(&loop_program(false), [7, 5, 0, 0]),
        exchanges(&wave_or_inside(0xBF88_0000), [40, 0, 0, 0]),
        exchanges(&fp32_program(3, 0, [0; 3], true), [40, 1.0f32.to_bits(), 0, 0]),
    ];
    eprintln!("exchanges: {counts:?}");
    assert!(counts.iter().all(|&c| c > 0), "{counts:?}");
}
