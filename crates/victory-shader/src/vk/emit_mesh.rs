//! Mesh outputs: per-vertex arrays at the semantics' locations, the
//! triangle indices, and the split of a console group whose outputs do not
//! fit the device's mesh output memory.
//!
//! A culled primitive is the degenerate triangle (0, 0, 0), which makes no
//! fragments: no `CullPrimitiveEXT` output (it would count one scalar per
//! primitive against the output memory).
//!
//! **Split.** The console group's vertices fall in blocks of `block`
//! (vertex v in block v / block); host group part p (WorkgroupId.y modulo
//! the parts) holds the vertices of the p-th set of three blocks (in
//! lexicographic order), block k of its set at slots k × block onward. A
//! triangle goes to exactly one part: the set of its own blocks, made up
//! to three with the lowest other blocks. Each part runs the whole console
//! program; only its window of vertices and its triangles are written.

use super::*;
use crate::vk::tir::Node;

/// A console group run as several host groups (`VkShader::mesh_parts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSplit {
    /// Vertices per block.
    pub block: u32,
    /// Blocks over the console group's vertices (at least 4).
    pub blocks: u32,
}

impl MeshSplit {
    /// The split whose window is at most `window` vertices, for a console
    /// group of `vertices`; `None` when it would not be smaller.
    pub fn for_window(vertices: u32, window: u32) -> Option<MeshSplit> {
        let block = window / 3;
        let blocks = vertices.div_ceil(block.max(1));
        (block > 0 && blocks > 3).then_some(MeshSplit { block, blocks })
    }

    /// Vertex slots of one part.
    pub fn window(self) -> u32 {
        3 * self.block
    }

    /// Host groups per console group.
    pub fn parts(self) -> u32 {
        let n = self.blocks;
        n * (n - 1) * (n - 2) / 6
    }

    /// Each part's blocks as a bit mask, in lexicographic order.
    fn sets(self) -> Vec<u32> {
        let n = self.blocks;
        let mut out = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    out.push(1 << a | 1 << b | 1 << c);
                }
            }
        }
        out
    }
}

impl Emit<'_> {
    /// Declared vertices: the console's maximum, or a part's window.
    pub(super) fn mesh_vertices(&self) -> u32 {
        self.st.mesh_split.map_or(self.st.mesh.0, MeshSplit::window).max(1)
    }

    /// A mesh output array (per vertex, or per primitive for the indices).
    pub(super) fn mesh_output(&mut self, slot: u16) -> (Id, Id) {
        if let Some(&o) = self.outputs.get(&slot) {
            return o;
        }
        let (u, f) = (self.t.u32_, self.t.f32_);
        let (elem, var) = if slot == OUT_PRIMITIVE {
            let v3 = self.m.t_vec(u, 3);
            let arr = self.m.t_array(v3, self.st.mesh.1.max(1));
            let p = self.m.t_ptr(en::SC_OUTPUT, arr);
            let v = self.m.variable(p, en::SC_OUTPUT);
            self.m.decorate(v, en::DEC_BUILTIN, &[en::BI_PRIMITIVE_TRIANGLE_INDICES_EXT]);
            (u, v)
        } else {
            let (clip, cull) = self.iface.clip_cull;
            let elem = match slot {
                OUT_CLIP => self.m.t_array(f, u32::from(clip).max(1)),
                OUT_CULL => self.m.t_array(f, u32::from(cull).max(1)),
                OUT_POINT_SIZE => f,
                _ => self.m.t_vec(f, 4),
            };
            let arr = self.m.t_array(elem, self.mesh_vertices());
            let p = self.m.t_ptr(en::SC_OUTPUT, arr);
            let v = self.m.variable(p, en::SC_OUTPUT);
            if slot >= OUT_POSITION {
                self.builtin_output(v, slot);
            } else {
                self.m.decorate(v, en::DEC_LOCATION, &[u32::from(slot)]);
            }
            (f, v)
        };
        self.interface.push(var);
        self.outputs.insert(slot, (var, elem));
        (var, elem)
    }

    /// Mesh: the counts at the maxima, every primitive degenerate until its
    /// export; split: this part's blocks.
    pub(super) fn mesh_prologue(&mut self) {
        let (mv, mp) = (self.mesh_vertices(), self.st.mesh.1.max(1));
        let u = self.t.u32_;
        let (kv, kp) = (self.c32(mv), self.c32(mp));
        self.m.op_void(op::SET_MESH_OUTPUTS, &[kv, kp]);
        if let Some(s) = self.st.mesh_split {
            self.mesh_part = self.part_set(s);
        }
        let (var, _) = self.mesh_output(OUT_PRIMITIVE);
        let me = self.builtin(Builtin::LocalInvocationIndex);
        let threads = self.st.local_size[0].max(1);
        let v3 = self.m.t_vec(u, 3);
        let none = self.m.c_null(v3);
        let p3 = self.m.t_ptr(en::SC_OUTPUT, v3);
        let mut k = 0;
        while k < mp {
            let kk = self.c32(k);
            let i = self.m.op(op::I_ADD, u, &[me, kk]);
            let inside = self.m.op(op::U_LESS_THAN, self.t.bool_, &[i, kp]);
            self.when(inside, |e| {
                let ptr = e.m.op(op::ACCESS_CHAIN, p3, &[var, i]);
                e.m.op_void(op::STORE, &[ptr, none]);
            });
            k += threads;
        }
    }

    /// `body` in a selection on `cond`, in the current block.
    fn when(&mut self, cond: Id, body: impl FnOnce(&mut Self)) {
        let (lt, lm) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[lm, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[cond, lt, lm]);
        self.m.label(lt);
        body(self);
        self.m.op_void(op::BRANCH, &[lm]);
        self.m.label(lm);
        self.block = lm;
    }

    /// This part's set of blocks (a mask), from WorkgroupId.y.
    fn part_set(&mut self, s: MeshSplit) -> Id {
        let u = self.t.u32_;
        let sets: Vec<Id> = s.sets().into_iter().map(|m| self.c32(m)).collect();
        let arr = self.m.t_array_plain(u, s.parts());
        let table = self.m.c_composite(arr, &sets);
        let pa = self.m.t_ptr(en::SC_PRIVATE, arr);
        let var = self.m.variable_init(pa, en::SC_PRIVATE, table);
        self.interface.push(var);
        let y = self.builtin(Builtin::WorkgroupId(1));
        let parts = self.c32(s.parts());
        let part = self.m.op(op::U_MOD, u, &[y, parts]);
        let pu = self.m.t_ptr(en::SC_PRIVATE, u);
        let ptr = self.m.op(op::ACCESS_CHAIN, pu, &[var, part]);
        self.m.op(op::LOAD, u, &[ptr])
    }

    /// Mesh outputs: [index] of the arrays (clamped), then the component;
    /// the primitive export as three indices.
    pub(super) fn mesh_output_op(&mut self, n: &Node, slot: u16, comp: u8, args: &[Id], var: Id, elem: Id) {
        let u = self.t.u32_;
        let max = if slot == OUT_PRIMITIVE { self.st.mesh.1 } else { self.st.mesh.0 };
        let km = self.c32(max.max(1) - 1);
        let i = self.m.ext(glsl::U_MIN, u, &[args[1], km]);
        if slot == OUT_PRIMITIVE {
            let x = if self.f.ty(n.args[0]) == Ty::F32 { self.m.op(op::BITCAST, u, &[args[0]]) } else { args[0] };
            return self.mesh_primitive(x, var, i);
        }
        let x = if self.f.ty(n.args[0]) == Ty::F32 { args[0] } else { self.m.op(op::BITCAST, elem, &[args[0]]) };
        let p = self.m.t_ptr(en::SC_OUTPUT, elem);
        let k = self.c32(u32::from(comp));
        // a point size is one scalar per vertex
        let chain = |at: Id| if slot == OUT_POINT_SIZE { vec![var, at] } else { vec![var, at, k] };
        let Some(s) = self.st.mesh_split else {
            let ptr = self.m.op(op::ACCESS_CHAIN, p, &chain(i));
            self.m.op_void(op::STORE, &[ptr, x]);
            return;
        };
        let (inside, at) = self.window_slot(s, i);
        let ops = chain(at);
        self.when(inside, |e| {
            let ptr = e.m.op(op::ACCESS_CHAIN, p, &ops);
            e.m.op_void(op::STORE, &[ptr, x]);
        });
    }

    /// Split: whether console vertex `v` is in this part's window, and its
    /// slot there.
    fn window_slot(&mut self, s: MeshSplit, v: Id) -> (Id, Id) {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let (one, z) = (self.c32(1), self.c32(0));
        let size = self.c32(s.block);
        let blk = self.m.op(op::U_DIV, u, &[v, size]);
        let bit = self.m.op(op::SHIFT_LEFT_LOGICAL, u, &[one, blk]);
        let has = self.m.op(op::BITWISE_AND, u, &[self.mesh_part, bit]);
        let inside = self.m.op(op::I_NOT_EQUAL, b, &[has, z]);
        let below = self.m.op(op::I_SUB, u, &[bit, one]);
        let lower = self.m.op(op::BITWISE_AND, u, &[self.mesh_part, below]);
        let rank = self.m.op(op::BIT_COUNT, u, &[lower]);
        let start = self.m.op(op::I_MUL, u, &[blk, size]);
        let off = self.m.op(op::I_SUB, u, &[v, start]);
        let base = self.m.op(op::I_MUL, u, &[rank, size]);
        (inside, self.m.op(op::I_ADD, u, &[base, off]))
    }

    /// Three 10-bit indices; bit 31 culls (the degenerate triangle).
    fn mesh_primitive(&mut self, x: Id, var: Id, i: Id) {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let m = self.c32(0x3FF);
        let mut idx = [0; 3];
        for (k, out) in idx.iter_mut().enumerate() {
            let sh = self.c32(10 * k as u32);
            let s = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[x, sh]);
            *out = self.m.op(op::BITWISE_AND, u, &[s, m]);
        }
        let k31 = self.c32(31);
        let c = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[x, k31]);
        let z = self.c32(0);
        let mut keep = self.m.op(op::I_EQUAL, b, &[c, z]);
        if let Some(s) = self.st.mesh_split {
            let (mine, slots) = self.triangle_part(s, idx);
            keep = self.m.op(op::LOGICAL_AND, b, &[keep, mine]);
            idx = slots;
        }
        let v3 = self.m.t_vec(u, 3);
        let tri = self.m.op(op::COMPOSITE_CONSTRUCT, v3, &idx);
        let none = self.m.c_null(v3);
        let tri = self.m.op(op::SELECT, v3, &[keep, tri, none]);
        let p3 = self.m.t_ptr(en::SC_OUTPUT, v3);
        let ptr = self.m.op(op::ACCESS_CHAIN, p3, &[var, i]);
        self.m.op_void(op::STORE, &[ptr, tri]);
    }

    /// Split: whether the triangle goes to this part (its blocks are in
    /// the part's set, and the rest of the set are the lowest blocks it
    /// does not use), and its indices in the window.
    fn triangle_part(&mut self, s: MeshSplit, idx: [Id; 3]) -> (Id, [Id; 3]) {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let (one, z) = (self.c32(1), self.c32(0));
        let last = self.c32(s.blocks * s.block - 1);
        let size = self.c32(s.block);
        let mut used = z;
        let mut slots = [0; 3];
        for (k, &v) in idx.iter().enumerate() {
            let v = self.m.ext(glsl::U_MIN, u, &[v, last]);
            let blk = self.m.op(op::U_DIV, u, &[v, size]);
            let bit = self.m.op(op::SHIFT_LEFT_LOGICAL, u, &[one, blk]);
            used = self.m.op(op::BITWISE_OR, u, &[used, bit]);
            slots[k] = self.window_slot(s, v).1;
        }
        let set = self.mesh_part;
        let not_set = self.m.op(op::NOT, u, &[set]);
        let outside = self.m.op(op::BITWISE_AND, u, &[used, not_set]);
        let within = self.m.op(op::I_EQUAL, b, &[outside, z]);
        // the set's other blocks: none, or the lowest one or two unused
        let unused = self.m.op(op::NOT, u, &[used]);
        let pad = self.m.op(op::BITWISE_AND, u, &[set, unused]);
        let low1 = self.lowest_bit(unused);
        let minus1 = self.m.op(op::I_SUB, u, &[unused, one]);
        let rest = self.m.op(op::BITWISE_AND, u, &[unused, minus1]);
        let low2 = self.lowest_bit(rest);
        let low12 = self.m.op(op::BITWISE_OR, u, &[low1, low2]);
        let mut canon = self.m.op(op::I_EQUAL, b, &[pad, z]);
        for want in [low1, low12] {
            let eq = self.m.op(op::I_EQUAL, b, &[pad, want]);
            canon = self.m.op(op::LOGICAL_OR, b, &[canon, eq]);
        }
        (self.m.op(op::LOGICAL_AND, b, &[within, canon]), slots)
    }

    fn lowest_bit(&mut self, x: Id) -> Id {
        let u = self.t.u32_;
        let neg = self.m.op(op::S_NEGATE, u, &[x]);
        self.m.op(op::BITWISE_AND, u, &[x, neg])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_into_sets_of_three_blocks() {
        let s = MeshSplit::for_window(128, 96).unwrap();
        assert_eq!((s.block, s.blocks, s.parts(), s.window()), (32, 4, 4, 96));
        assert_eq!(s.sets(), vec![0b0111, 0b1011, 0b1101, 0b1110]);
        let s = MeshSplit::for_window(128, 64).unwrap();
        assert_eq!((s.block, s.blocks, s.parts(), s.window()), (21, 7, 35, 63));
        assert_eq!(MeshSplit::for_window(64, 96), None);
    }

    /// The part a triangle goes to, as the module computes it.
    fn part_of(s: MeshSplit, tri: [u32; 3]) -> Vec<usize> {
        let used = tri.iter().fold(0u32, |m, &v| m | 1 << (v / s.block));
        let unused = !used;
        let low1 = unused & unused.wrapping_neg();
        let rest = unused & unused.wrapping_sub(1);
        let low2 = rest & rest.wrapping_neg();
        let sets = s.sets();
        (0..sets.len())
            .filter(|&p| {
                let pad = sets[p] & unused;
                used & !sets[p] == 0 && (pad == 0 || pad == low1 || pad == low1 | low2)
            })
            .collect()
    }

    #[test]
    fn every_triangle_goes_to_one_part() {
        for s in [MeshSplit::for_window(128, 96).unwrap(), MeshSplit::for_window(128, 64).unwrap()] {
            for a in (0..128).step_by(7) {
                for b in (0..128).step_by(5) {
                    for c in (0..128).step_by(11) {
                        assert_eq!(part_of(s, [a, b, c]).len(), 1, "{s:?} {a} {b} {c}");
                    }
                }
            }
        }
    }
}
