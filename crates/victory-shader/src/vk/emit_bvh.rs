//! `Helper::BvhRay`: the console's ray/node test (`image_bvh_intersect_ray`,
//! `image_bvh64_intersect_ray`) in software, written once per module. The
//! node is read from title memory (lift_image.rs translates its address,
//! 128 bytes in one range). The arithmetic follows the reference the D3D12
//! path tested against the card (the deleted `bvh.rs`, after the public
//! software emulation in Mesa's radv): four slab tests and a sorting network
//! for box nodes (16- or 32-bit bounds), the watertight ray/triangle test
//! (Woop, Benthin, Wald) for triangle nodes, a triangle miss for anything
//! else. Branch-free: every lane computes both tests and keeps one.
//!
//! Node format: a pointer's low 3 bits are the kind (0-3 triangle k of a
//! triangle node, 4 box16, 5 box32, 6 instance, 7 procedural). Box nodes:
//! child pointers at dwords 0-3 (`0xffffffff` none), boxes from dword 4
//! (min xyz, max xyz each; halves for box16). Triangle nodes: five vertices
//! of three floats at dwords 0-14, dword 15 one byte per triangle (bits 1:0
//! the vertex barycentric I weighs, 3:2 J's).

use super::*;

/// A child pointer that is not there, or a box the ray misses.
const NONE: u32 = 0xffff_ffff;

/// The node's vertices that make triangle 0-3, four bits each (vertex 0
/// lowest), twelve bits per triangle.
const TRIANGLE_VERTICES: u64 = 0x042_432_231_210;

impl Emit<'_> {
    /// Parameters: node device address (u64), kind (u32), sort (u32, not 0:
    /// boxes nearest first), extent, origin xyz, direction xyz, inverse
    /// direction xyz (floats). Returns the four result dwords.
    pub(super) fn write_bvh(&mut self, fid: Id) {
        let (u, q, f) = (self.t.u32_, self.t.u64_, self.t.f32_);
        let v4 = self.m.t_vec(u, 4);
        let mut params = vec![q, u, u];
        params.extend([f; 10]);
        let fty = self.m.global_fn_type(v4, &params);
        self.m.op_to(op::FUNCTION, v4, fid, &[0, fty]);
        let ps: Vec<Id> = params
            .iter()
            .map(|&t| {
                let id = self.m.id();
                self.m.op_to(55, t, id, &[]); // OpFunctionParameter
                id
            })
            .collect();
        let entry = self.m.id();
        self.m.label(entry);
        let (dev, kind, sort, extent) = (ps[0], ps[1], ps[2], ps[3]);
        let origin = [ps[4], ps[5], ps[6]];
        let dir = [ps[7], ps[8], ps[9]];
        let inv = [ps[10], ps[11], ps[12]];
        // dwords 0-27 of the node
        let w: Vec<Id> = (0..28u64)
            .map(|k| {
                let o = self.m.c_u64(4 * k);
                let a = self.m.op(op::I_ADD, q, &[dev, o]);
                self.load_u32_at(a)
            })
            .collect();
        let boxes = self.bvh_boxes(&w, kind, sort, extent, origin, inv);
        let tri = self.bvh_triangle(dev, &w, kind, origin, dir, inv);
        let b = self.t.bool_;
        let (k4, k5, k3) = (self.c32(4), self.c32(5), self.c32(3));
        let is4 = self.m.op(op::I_EQUAL, b, &[kind, k4]);
        let is5 = self.m.op(op::I_EQUAL, b, &[kind, k5]);
        let is_box = self.m.op(op::LOGICAL_OR, b, &[is4, is5]);
        let is_tri = self.m.op(op::U_LESS_THAN_EQUAL, b, &[kind, k3]);
        let miss = self.bvh_miss_words();
        let out: Vec<Id> = (0..4)
            .map(|c| {
                let t = self.m.op(op::SELECT, u, &[is_tri, tri[c], miss[c]]);
                self.m.op(op::SELECT, u, &[is_box, boxes[c], t])
            })
            .collect();
        let r = self.m.op(op::COMPOSITE_CONSTRUCT, v4, &out);
        self.m.op_void(254, &[r]); // OpReturnValue
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    fn load_u32_at(&mut self, a: Id) -> Id {
        let u = self.t.u32_;
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
        let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[a]);
        self.m.op(op::LOAD, u, &[ptr, en::MEM_ALIGNED, 4])
    }

    /// What a triangle miss returns: +infinity, 1.0, 0, 0.
    fn bvh_miss_words(&mut self) -> [Id; 4] {
        [self.c32(f32::INFINITY.to_bits()), self.c32(1.0f32.to_bits()), self.c32(0), self.c32(0)]
    }

    /// A float operation the driver may not fuse with another (the
    /// triangle test is watertight only if the edge functions round as
    /// written).
    fn fop(&mut self, o: u16, a: Id, b: Id) -> Id {
        let f = self.t.f32_;
        let x = self.m.op(o, f, &[a, b]);
        self.m.decorate(x, 42, &[]); // NoContraction
        x
    }

    fn fcmp(&mut self, o: u16, a: Id, b: Id) -> Id {
        let b_ = self.t.bool_;
        self.m.op(o, b_, &[a, b])
    }

    fn fext(&mut self, g: u32, a: Id, b: Id) -> Id {
        let f = self.t.f32_;
        self.m.ext(g, f, &[a, b])
    }

    /// The box test: child pointers of the boxes entered before the extent,
    /// nearest first when `sort`.
    fn bvh_boxes(&mut self, w: &[Id], kind: Id, sort: Id, extent: Id, origin: [Id; 3], inv: [Id; 3]) -> [Id; 4] {
        let (u, f, b) = (self.t.u32_, self.t.f32_, self.t.bool_);
        let k5 = self.c32(5);
        let wide = self.m.op(op::I_EQUAL, b, &[kind, k5]);
        let v2 = self.m.t_vec(f, 2);
        // the 24 bounds: floats (box32) or halves widened (box16)
        let bounds: Vec<Id> = (0..24usize)
            .map(|i| {
                let full = self.m.op(op::BITCAST, f, &[w[4 + i]]);
                let pair = self.m.ext(glsl::UNPACK_HALF_2X16, v2, &[w[4 + i / 2]]);
                let half = self.m.op(op::COMPOSITE_EXTRACT, f, &[pair, (i % 2) as u32]);
                self.m.op(op::SELECT, f, &[wide, full, half])
            })
            .collect();
        // an infinite inverse would make a NaN on a box face: clamped
        // (maximum then minimum: a NaN becomes the lower bound)
        let (lo, hi) = (self.m.c_f32((-f32::MAX).to_bits()), self.m.c_f32(f32::MAX.to_bits()));
        let inv = inv.map(|x| {
            let m = self.fext(glsl::N_MAX, x, lo);
            self.fext(glsl::N_MIN, m, hi)
        });
        let zero = self.m.c_f32(0);
        let inf = self.m.c_f32(f32::INFINITY.to_bits());
        let none = self.c32(NONE);
        let mut ids = Vec::new();
        let mut dists = Vec::new();
        for n in 0..4 {
            let bx = &bounds[6 * n..6 * n + 6];
            let mut near = Vec::new();
            let mut far = Vec::new();
            for a in 0..3 {
                let l = self.fop(op::F_SUB, bx[a], origin[a]);
                let l = self.fop(op::F_MUL, l, inv[a]);
                let h = self.fop(op::F_SUB, bx[3 + a], origin[a]);
                let h = self.fop(op::F_MUL, h, inv[a]);
                near.push(self.fext(glsl::N_MIN, l, h));
                far.push(self.fext(glsl::N_MAX, l, h));
            }
            let enter = self.fext(glsl::N_MAX, near[0], near[1]);
            let enter = self.fext(glsl::N_MAX, enter, near[2]);
            let leave = self.fext(glsl::N_MIN, far[0], far[1]);
            let leave = self.fext(glsl::N_MIN, leave, far[2]);
            // a box whose minimum x is NaN is switched off
            let nan = self.m.op(op::IS_NAN, b, &[bx[0]]);
            let on = self.m.op(op::LOGICAL_NOT, b, &[nan]);
            let e0 = self.fext(glsl::N_MAX, enter, zero);
            let inside = self.fcmp(op::F_ORD_GREATER_THAN_EQUAL, leave, e0);
            let before = self.fcmp(op::F_ORD_LESS_THAN, enter, extent);
            let hit = self.m.op(op::LOGICAL_AND, b, &[on, inside]);
            let hit = self.m.op(op::LOGICAL_AND, b, &[hit, before]);
            ids.push(self.m.op(op::SELECT, u, &[hit, w[n], none]));
            dists.push(self.m.op(op::SELECT, f, &[hit, enter, inf]));
        }
        let z = self.c32(0);
        let sorting = self.m.op(op::I_NOT_EQUAL, b, &[sort, z]);
        for (a, c) in [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)] {
            let less = self.fcmp(op::F_ORD_LESS_THAN, dists[c], dists[a]);
            let swap = self.m.op(op::LOGICAL_AND, b, &[sorting, less]);
            let (da, dc) = (dists[a], dists[c]);
            dists[a] = self.m.op(op::SELECT, f, &[swap, dc, da]);
            dists[c] = self.m.op(op::SELECT, f, &[swap, da, dc]);
            let (ia, ic) = (ids[a], ids[c]);
            ids[a] = self.m.op(op::SELECT, u, &[swap, ic, ia]);
            ids[c] = self.m.op(op::SELECT, u, &[swap, ia, ic]);
        }
        [ids[0], ids[1], ids[2], ids[3]]
    }

    /// The watertight triangle test of triangle `kind & 3`: the hit
    /// distance's numerator, the denominator and the numerators of I and J
    /// (as float bits), or a miss.
    fn bvh_triangle(&mut self, dev: Id, w: &[Id], kind: Id, origin: [Id; 3], dir: [Id; 3], inv: [Id; 3]) -> [Id; 4] {
        let (u, q, f, b) = (self.t.u32_, self.t.u64_, self.t.f32_, self.t.bool_);
        let (k3, k8, k12, k15, k4, k2) = (self.c32(3), self.c32(8), self.c32(12), self.c32(15), self.c32(4), self.c32(2));
        let k = self.m.op(op::BITWISE_AND, u, &[kind, k3]);
        // the vertex barycentrics I and J weigh, the first the remaining one
        let sh = self.m.op(op::I_MUL, u, &[k, k8]);
        let id = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[w[15], sh]);
        let i = self.m.op(op::BITWISE_AND, u, &[id, k3]);
        let j = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[id, k2]);
        let j = self.m.op(op::BITWISE_AND, u, &[j, k3]);
        let first = self.m.op(op::I_SUB, u, &[k3, i]);
        let first = self.m.op(op::I_SUB, u, &[first, j]);
        let first = self.m.op(op::BITWISE_AND, u, &[first, k3]);
        let table = self.m.c_u64(TRIANGLE_VERTICES);
        let tsh = self.m.op(op::I_MUL, u, &[k, k12]);
        let map = self.m.op(op::SHIFT_RIGHT_LOGICAL, q, &[table, tsh]);
        let map = self.m.op(op::U_CONVERT, u, &[map]);
        let vertex = |s: &mut Self, n: Id| -> [Id; 3] {
            let nsh = s.m.op(op::I_MUL, u, &[n, k4]);
            let v = s.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[map, nsh]);
            let v = s.m.op(op::BITWISE_AND, u, &[v, k15]);
            let k12b = s.c32(12);
            let off = s.m.op(op::I_MUL, u, &[v, k12b]);
            std::array::from_fn(|c| {
                let kc = s.c32(4 * c as u32);
                let o = s.m.op(op::I_ADD, u, &[off, kc]);
                let o = s.m.op(op::U_CONVERT, q, &[o]);
                let a = s.m.op(op::I_ADD, q, &[dev, o]);
                let x = s.load_u32_at(a);
                s.m.op(op::BITCAST, f, &[x])
            })
        };
        let p = [vertex(self, first), vertex(self, i), vertex(self, j)];
        // the largest axis kz; kx, ky the others keeping the winding
        let v3 = self.m.t_vec(f, 3);
        let dv = self.m.op(op::COMPOSITE_CONSTRUCT, v3, &dir);
        let size: Vec<Id> = dir.iter().map(|&d| self.m.ext(glsl::F_ABS, f, &[d])).collect();
        let (c0, c1, c2) = (self.c32(0), self.c32(1), self.c32(2));
        let ge01 = self.fcmp(op::F_ORD_GREATER_THAN_EQUAL, size[0], size[1]);
        let ge02 = self.fcmp(op::F_ORD_GREATER_THAN_EQUAL, size[0], size[2]);
        let ge12 = self.fcmp(op::F_ORD_GREATER_THAN_EQUAL, size[1], size[2]);
        let a = self.m.op(op::SELECT, u, &[ge02, c0, c2]);
        let bb = self.m.op(op::SELECT, u, &[ge12, c1, c2]);
        let kz = self.m.op(op::SELECT, u, &[ge01, a, bb]);
        let kx = self.m.op(op::I_ADD, u, &[kz, c1]);
        let kx = self.m.op(op::U_MOD, u, &[kx, k3]);
        let ky = self.m.op(op::I_ADD, u, &[kz, c2]);
        let ky = self.m.op(op::U_MOD, u, &[ky, k3]);
        let dz = self.m.op(77, f, &[dv, kz]); // OpVectorExtractDynamic
        let zf = self.m.c_f32(0);
        let neg = self.fcmp(op::F_ORD_LESS_THAN, dz, zf);
        let (kx, ky) = (self.m.op(op::SELECT, u, &[neg, ky, kx]), self.m.op(op::SELECT, u, &[neg, kx, ky]));
        let iv = self.m.op(op::COMPOSITE_CONSTRUCT, v3, &inv);
        let sz = self.m.op(77, f, &[iv, kz]);
        let dx = self.m.op(77, f, &[dv, kx]);
        let dy = self.m.op(77, f, &[dv, ky]);
        let sx = self.fop(op::F_MUL, dx, sz);
        let sy = self.fop(op::F_MUL, dy, sz);
        let sheared = |s: &mut Self, v: &[Id; 3]| -> (Id, Id, Id) {
            let r: Vec<Id> = (0..3).map(|c| s.fop(op::F_SUB, v[c], origin[c])).collect();
            let rv = s.m.op(op::COMPOSITE_CONSTRUCT, v3, &r);
            let (rx, ry, rz) = (s.m.op(77, f, &[rv, kx]), s.m.op(77, f, &[rv, ky]), s.m.op(77, f, &[rv, kz]));
            let tx = s.fop(op::F_MUL, sx, rz);
            let ty = s.fop(op::F_MUL, sy, rz);
            (s.fop(op::F_SUB, rx, tx), s.fop(op::F_SUB, ry, ty), s.fop(op::F_MUL, sz, rz))
        };
        let (ax, ay, az) = sheared(self, &p[0]);
        let (bx, by, bz) = sheared(self, &p[1]);
        let (cx, cy, cz) = sheared(self, &p[2]);
        let cross = |s: &mut Self, a: Id, b_: Id, c: Id, d: Id| {
            let x = s.fop(op::F_MUL, a, b_);
            let y = s.fop(op::F_MUL, c, d);
            s.fop(op::F_SUB, x, y)
        };
        let uu = cross(self, cx, by, cy, bx);
        let vv = cross(self, ax, cy, ay, cx);
        let ww = cross(self, bx, ay, by, ax);
        let any = |s: &mut Self, o: u16| {
            let x = s.fcmp(o, uu, zf);
            let y = s.fcmp(o, vv, zf);
            let z = s.fcmp(o, ww, zf);
            let xy = s.m.op(op::LOGICAL_OR, b, &[x, y]);
            s.m.op(op::LOGICAL_OR, b, &[xy, z])
        };
        let behind = any(self, op::F_ORD_LESS_THAN);
        let before = any(self, op::F_ORD_GREATER_THAN);
        let vw = self.fop(op::F_ADD, vv, ww);
        let det = self.fop(op::F_ADD, uu, vw);
        let t0 = self.fop(op::F_MUL, uu, az);
        let t1 = self.fop(op::F_MUL, vv, bz);
        let t01 = self.fop(op::F_ADD, t0, t1);
        let t2 = self.fop(op::F_MUL, ww, cz);
        let t = self.fop(op::F_ADD, t01, t2);
        let dpos = self.fcmp(op::F_ORD_GREATER_THAN, det, zf);
        let tpos = self.fcmp(op::F_ORD_GREATER_THAN_EQUAL, t, zf);
        let dneg = self.fcmp(op::F_ORD_LESS_THAN, det, zf);
        let tneg = self.fcmp(op::F_ORD_LESS_THAN_EQUAL, t, zf);
        let ahead_p = self.m.op(op::LOGICAL_AND, b, &[dpos, tpos]);
        let ahead_n = self.m.op(op::LOGICAL_AND, b, &[dneg, tneg]);
        let ahead = self.m.op(op::LOGICAL_OR, b, &[ahead_p, ahead_n]);
        let both = self.m.op(op::LOGICAL_AND, b, &[behind, before]);
        let straddle = self.m.op(op::LOGICAL_NOT, b, &[both]);
        let hit = self.m.op(op::LOGICAL_AND, b, &[straddle, ahead]);
        let miss = self.bvh_miss_words();
        let vals = [t, det, vv, ww];
        std::array::from_fn(|c| {
            let x = self.m.op(op::BITCAST, u, &[vals[c]]);
            self.m.op(op::SELECT, u, &[hit, x, miss[c]])
        })
    }
}
