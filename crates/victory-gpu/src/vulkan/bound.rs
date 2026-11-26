//! The program that bounds indirect records (`record/indirect.rs`), written
//! with the translator's SPIR-V builder. One thread per record word:
//!
//! ```text
//! push { u64 src; u64 dst; u64 count; u32 words; u32 stride; u32 records; u32 kind; }
//! t = global id; if t >= records * words: return
//! r = t / words; w = t % words
//! n = min(*count, records)                  // count: the title's, or a word of u32::MAX
//! if t == 0: dst[0] = n
//! h0, h1, h2 = src[r * stride + 0, 4, 8]; v = src[r * stride + 4 w]
//! draws (kind 0, 1): bad if h0 >= 2^23 or h1 > 2^26 / max(h0, 1) with h0 != 0
//! groups (kind 2, 10; mesh cuts y and z to 1):
//!     bad if any count > 65535 or h0 * y > 2^21 / max(z, 1) with z != 0
//! dst[16 + 4 t] = (r < n && !bad) ? v : 0   // records packed, words * 4 apart
//! ```
//!
//! The bounds are those of the D3D12 path (`crate::indirect`): records the
//! title's own passes write are garbage when such a pass is dropped or
//! mistranslated, and a garbage draw hung the card (Xid 109).

use victory_shader::vk::spirv::{Id, Module, en, glsl, op};

/// Threads per group.
pub(super) const GROUP: u32 = 64;
/// The push constants' bytes.
pub(super) const PUSH_BYTES: usize = 40;

/// Kinds as the signature numbers them (PC numbering).
pub(super) const DRAW: u32 = 0;
pub(super) const DRAW_INDEXED: u32 = 1;
pub(super) const DISPATCH: u32 = 2;
pub(super) const MESH: u32 = 10;

const MOST_VERTICES: u32 = 1 << 23;
const MOST_DRAWN: u32 = 1 << 26;
const MOST_GROUPS: u32 = 1 << 21;
const MOST_COUNT: u32 = 65_535;

/// The types and values the body uses.
struct T {
    u32: Id,
    u64: Id,
    bool: Id,
    psb: Id,
}

fn load(m: &mut Module, t: &T, address: Id) -> Id {
    let p = m.op(op::CONVERT_U_TO_PTR, t.psb, &[address]);
    m.op(op::LOAD, t.u32, &[p, en::MEM_ALIGNED, 4])
}

fn store(m: &mut Module, t: &T, address: Id, value: Id) {
    let p = m.op(op::CONVERT_U_TO_PTR, t.psb, &[address]);
    m.op_void(op::STORE, &[p, value, en::MEM_ALIGNED, 4]);
}

/// `base + 4 * index` as a 64-bit address.
fn word_at(m: &mut Module, t: &T, base: Id, index: Id) -> Id {
    let wide = m.op(op::U_CONVERT, t.u64, &[index]);
    let four = m.c_u64(4);
    let bytes = m.op(op::I_MUL, t.u64, &[wide, four]);
    m.op(op::I_ADD, t.u64, &[base, bytes])
}

/// The push constant block and its loads: (src, dst, count, words, stride,
/// records, kind).
fn push(m: &mut Module, t: &T) -> (Id, [Id; 7]) {
    let block = m.t_struct(&[t.u64, t.u64, t.u64, t.u32, t.u32, t.u32, t.u32], 0);
    m.decorate(block, en::DEC_BLOCK, &[]);
    for (k, offset) in [0, 8, 16, 24, 28, 32, 36].into_iter().enumerate() {
        m.member_decorate(block, k as u32, en::DEC_OFFSET, &[offset]);
    }
    let ptr = m.t_ptr(en::SC_PUSH_CONSTANT, block);
    let var = m.variable(ptr, en::SC_PUSH_CONSTANT);
    let p64 = m.t_ptr(en::SC_PUSH_CONSTANT, t.u64);
    let p32 = m.t_ptr(en::SC_PUSH_CONSTANT, t.u32);
    let mut out = [0; 7];
    for (k, slot) in out.iter_mut().enumerate() {
        let (pt, ty) = if k < 3 { (p64, t.u64) } else { (p32, t.u32) };
        let index = m.c_u32(k as u32);
        let at = m.op(op::ACCESS_CHAIN, pt, &[var, index]);
        *slot = m.op(op::LOAD, ty, &[at]);
    }
    (var, out)
}

/// Whether a draw record (h0 vertices or indices, h1 instances) is out of
/// bounds.
fn draw_bad(m: &mut Module, t: &T, h0: Id, h1: Id) -> Id {
    let most_v = m.c_u32(MOST_VERTICES);
    let big = m.op(op::U_GREATER_THAN_EQUAL, t.bool, &[h0, most_v]);
    let one = m.c_u32(1);
    let div = m.ext(glsl::U_MAX, t.u32, &[h0, one]);
    let most_d = m.c_u32(MOST_DRAWN);
    let limit = m.op(op::U_DIV, t.u32, &[most_d, div]);
    let many = m.op(op::U_GREATER_THAN, t.bool, &[h1, limit]);
    let zero = m.c_u32(0);
    let some = m.op(op::I_NOT_EQUAL, t.bool, &[h0, zero]);
    let too_many = m.op(op::LOGICAL_AND, t.bool, &[some, many]);
    m.op(op::LOGICAL_OR, t.bool, &[big, too_many])
}

/// Whether a group record (x, y, z, the last two already cut for mesh
/// programs) is out of bounds.
fn groups_bad(m: &mut Module, t: &T, x: Id, y: Id, z: Id) -> Id {
    let most = m.c_u32(MOST_COUNT);
    let mut bad = m.op(op::U_GREATER_THAN, t.bool, &[x, most]);
    for g in [y, z] {
        let over = m.op(op::U_GREATER_THAN, t.bool, &[g, most]);
        bad = m.op(op::LOGICAL_OR, t.bool, &[bad, over]);
    }
    // Each count is at most 65,535 when this matters, so x * y fits.
    let xy = m.op(op::I_MUL, t.u32, &[x, y]);
    let one = m.c_u32(1);
    let div = m.ext(glsl::U_MAX, t.u32, &[z, one]);
    let most_g = m.c_u32(MOST_GROUPS);
    let limit = m.op(op::U_DIV, t.u32, &[most_g, div]);
    let many = m.op(op::U_GREATER_THAN, t.bool, &[xy, limit]);
    let zero = m.c_u32(0);
    let some = m.op(op::I_NOT_EQUAL, t.bool, &[z, zero]);
    let too_many = m.op(op::LOGICAL_AND, t.bool, &[some, many]);
    m.op(op::LOGICAL_OR, t.bool, &[bad, too_many])
}

/// The bounded value of word `w` of record `r` (`n` records live).
fn bounded_word(m: &mut Module, t: &T, base: Id, w: Id, r: Id, n: Id, kind: Id) -> Id {
    let (one, two, zero) = (m.c_u32(1), m.c_u32(2), m.c_u32(0));
    let h0 = load(m, t, base);
    let a1 = word_at(m, t, base, one);
    let h1 = load(m, t, a1);
    let a2 = word_at(m, t, base, two);
    let h2 = load(m, t, a2);
    let aw = word_at(m, t, base, w);
    let v = load(m, t, aw);
    let mesh_k = m.c_u32(MESH);
    let mesh = m.op(op::I_EQUAL, t.bool, &[kind, mesh_k]);
    let cut1 = m.ext(glsl::U_MIN, t.u32, &[h1, one]);
    let cut2 = m.ext(glsl::U_MIN, t.u32, &[h2, one]);
    let g1 = m.op(op::SELECT, t.u32, &[mesh, cut1, h1]);
    let g2 = m.op(op::SELECT, t.u32, &[mesh, cut2, h2]);
    let d_bad = draw_bad(m, t, h0, h1);
    let g_bad = groups_bad(m, t, h0, g1, g2);
    let is_draw = m.op(op::U_LESS_THAN_EQUAL, t.bool, &[kind, one]);
    let bad = m.op(op::SELECT, t.bool, &[is_draw, d_bad, g_bad]);
    let good = m.op(op::LOGICAL_NOT, t.bool, &[bad]);
    let live = m.op(op::U_LESS_THAN, t.bool, &[r, n]);
    let keep = m.op(op::LOGICAL_AND, t.bool, &[live, good]);
    let later = m.op(op::U_GREATER_THAN_EQUAL, t.bool, &[w, one]);
    let cut_w = m.op(op::LOGICAL_AND, t.bool, &[mesh, later]);
    let cut_v = m.ext(glsl::U_MIN, t.u32, &[v, one]);
    let value = m.op(op::SELECT, t.u32, &[cut_w, cut_v, v]);
    m.op(op::SELECT, t.u32, &[keep, value, zero])
}

/// The module's SPIR-V words.
pub(super) fn module() -> Vec<u32> {
    let mut m = Module::new();
    let t = T {
        u32: m.t_int(32),
        u64: m.t_int(64),
        bool: m.t_bool(),
        psb: 0,
    };
    let t = T {
        psb: m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, t.u32),
        ..t
    };
    let void = m.t_void();
    let fty = m.t_func(void);
    let v3 = m.t_vec(t.u32, 3);
    let pin = m.t_ptr(en::SC_INPUT, v3);
    let gid_var = m.variable(pin, en::SC_INPUT);
    m.decorate(gid_var, en::DEC_BUILTIN, &[28]); // GlobalInvocationId
    let main = m.id();
    m.op_to(op::FUNCTION, void, main, &[0, fty]);
    let entry = m.id();
    m.label(entry);
    let (pc, [src, dst, count, words, stride, records, kind]) = push(&mut m, &t);
    let gid = m.op(op::LOAD, v3, &[gid_var]);
    let tid = m.op(op::COMPOSITE_EXTRACT, t.u32, &[gid, 0]);
    let total = m.op(op::I_MUL, t.u32, &[records, words]);
    let inside = m.op(op::U_LESS_THAN, t.bool, &[tid, total]);
    let (body, end) = (m.id(), m.id());
    m.op_void(op::SELECTION_MERGE, &[end, 0]);
    m.op_void(op::BRANCH_CONDITIONAL, &[inside, body, end]);
    m.label(body);
    let r = m.op(op::U_DIV, t.u32, &[tid, words]);
    let rw = m.op(op::I_MUL, t.u32, &[r, words]);
    let w = m.op(op::I_SUB, t.u32, &[tid, rw]);
    let c = load(&mut m, &t, count);
    let n = m.ext(glsl::U_MIN, t.u32, &[c, records]);
    let zero = m.c_u32(0);
    let first = m.op(op::I_EQUAL, t.bool, &[tid, zero]);
    let (write_n, after) = (m.id(), m.id());
    m.op_void(op::SELECTION_MERGE, &[after, 0]);
    m.op_void(op::BRANCH_CONDITIONAL, &[first, write_n, after]);
    m.label(write_n);
    store(&mut m, &t, dst, n);
    m.op_void(op::BRANCH, &[after]);
    m.label(after);
    let r64 = m.op(op::U_CONVERT, t.u64, &[r]);
    let s64 = m.op(op::U_CONVERT, t.u64, &[stride]);
    let off = m.op(op::I_MUL, t.u64, &[r64, s64]);
    let base = m.op(op::I_ADD, t.u64, &[src, off]);
    let value = bounded_word(&mut m, &t, base, w, r, n, kind);
    let four = m.c_u32(4);
    let slot = m.op(op::I_ADD, t.u32, &[tid, four]);
    let out = word_at(&mut m, &t, dst, slot);
    store(&mut m, &t, out, value);
    m.op_void(op::BRANCH, &[end]);
    m.label(end);
    m.op_void(op::RETURN, &[]);
    m.op_void(op::FUNCTION_END, &[]);
    m.entry_point(en::EXEC_GL_COMPUTE, main, "main", &[gid_var, pc]);
    m.execution_mode(main, en::EM_LOCAL_SIZE, &[GROUP, 1, 1]);
    m.finish().0
}
