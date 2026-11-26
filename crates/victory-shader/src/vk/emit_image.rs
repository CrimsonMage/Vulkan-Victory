//! Image instructions.

use super::*;

/// An image instruction's operands by the flags that say which are there.
#[derive(Clone, Copy)]
pub(super) struct ImgArgs {
    /// T# word 7.
    tw: Id,
    /// S# word 3 (dynamic samplers).
    sw: Option<Id>,
    coord: Id,
    dref: Option<Id>,
    bias: Option<Id>,
    lod: Option<Id>,
    grad: Option<(Id, Id)>,
    offset: Option<Id>,
    min_lod: Option<Id>,
    sample: Option<Id>,
    data: Option<Id>,
    cmp: Option<Id>,
}

impl ImgArgs {
    fn of(sampler: Sampler, flags: u16, args: &[Id]) -> Self {
        let mut i = 1;
        let mut next = |n: usize| {
            i += n;
            &args[i - n..i]
        };
        let sw = (sampler == Sampler::Dynamic).then(|| next(1)[0]);
        let coord = next(1)[0];
        let mut opt = |bit: u16| (flags & bit != 0).then(|| next(1)[0]);
        let (dref, bias, lod) = (opt(img::DREF), opt(img::BIAS), opt(img::LOD));
        let grad = (flags & img::GRAD != 0).then(|| {
            let g = next(2);
            (g[0], g[1])
        });
        let mut opt = |bit: u16| (flags & bit != 0).then(|| next(1)[0]);
        let (offset, min_lod, sample) = (opt(img::OFFSET), opt(img::MIN_LOD), opt(img::SAMPLE));
        let (data, cmp) = (opt(img::DATA), opt(img::CMP));
        ImgArgs { tw: args[0], sw, coord, dref, bias, lod, grad, offset, min_lod, sample, data, cmp }
    }
}

impl Emit<'_> {
    pub(super) fn image(&mut self, kind: ImageKind, class: ImageClass, sampler: Sampler, flags: u16, args: &[Id], tid: Id) -> Id {
        let a = ImgArgs::of(sampler, flags, args);
        match kind {
            ImageKind::Sample { .. } | ImageKind::Gather { .. } | ImageKind::QueryLod => self.img_sample(kind, class, sampler, a, tid),
            ImageKind::Load { .. } | ImageKind::Store { .. } | ImageKind::Size => self.img_typed(kind, class, a, tid),
            ImageKind::Levels => {
                self.m.capability(en::CAP_IMAGE_QUERY);
                if class.ms {
                    return self.c32(1);
                }
                let c = ImageClass { dynamic: false, ..class };
                let (im, _) = self.load_image(c, false, false, false, a.tw);
                self.m.op(op::IMAGE_QUERY_LEVELS, self.t.u32_, &[im])
            }
            ImageKind::Atomic(o) => {
                let v2 = self.m.t_vec(self.t.u32_, 2);
                if tid == v2 { self.img_atomic64(o, class, a) } else { self.img_atomic(o, class, a) }
            }
        }
    }

    fn img_sample(&mut self, kind: ImageKind, class: ImageClass, sampler: Sampler, a: ImgArgs, tid: Id) -> Id {
        let f = self.t.f32_;
        let v4f = self.m.t_vec(f, 4);
        let depth = a.dref.is_some();
        let (im, _) = self.load_image(class, false, false, depth, a.tw);
        let s = self.load_sampler(sampler, a.sw);
        let ity = self.m.t_image_d(f, u32::from(class.dim), depth, class.arrayed, class.ms, 1);
        let sity = self.m.t_sampled_image(ity);
        let si = self.m.op(op::SAMPLED_IMAGE, sity, &[im, s]);
        if kind == ImageKind::QueryLod {
            self.m.capability(en::CAP_IMAGE_QUERY);
            if !self.frag {
                return self.m.c_null(tid);
            }
            let v2 = self.m.t_vec(f, 2);
            return self.m.op(op::IMAGE_QUERY_LOD, v2, &[si, a.coord]);
        }
        let (mask, ops) = self.sample_operands(kind, a);
        let mut o = vec![si, a.coord];
        if let ImageKind::Gather { comp, .. } = kind {
            let gm = mask & !(en::IMG_LOD | en::IMG_GRAD | en::IMG_BIAS | en::IMG_MIN_LOD);
            let opc = match a.dref {
                Some(d) => {
                    o.push(d);
                    op::IMAGE_DREF_GATHER
                }
                None => {
                    o.push(self.c32(u32::from(comp)));
                    op::IMAGE_GATHER
                }
            };
            if gm != 0 {
                o.push(gm);
                o.extend(a.offset);
            }
            return self.m.op(opc, v4f, &o);
        }
        o.extend(a.dref);
        if mask != 0 {
            o.push(mask);
            o.extend(ops);
        }
        let lodded = mask & (en::IMG_LOD | en::IMG_GRAD) != 0;
        if depth {
            let opc = if lodded { op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD } else { op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD };
            let r = self.m.op(opc, f, &o);
            // the TIR reads a vec4 and takes component 0
            let z = self.m.c_f32(0);
            return self.m.op(op::COMPOSITE_CONSTRUCT, v4f, &[r, z, z, z]);
        }
        let opc = if lodded { op::IMAGE_SAMPLE_EXPLICIT_LOD } else { op::IMAGE_SAMPLE_IMPLICIT_LOD };
        self.m.op(opc, v4f, &o)
    }

    /// The image operands of a sample: no derivatives outside pixel
    /// shaders (level 0, no bias), a clamp only where SPIR-V takes one.
    fn sample_operands(&mut self, kind: ImageKind, a: ImgArgs) -> (u32, Vec<Id>) {
        let mut mask = 0u32;
        let mut ops = Vec::new();
        let implicit = a.lod.is_none() && a.grad.is_none();
        let explicit_lod = if implicit && !self.frag { Some(self.m.c_f32(0)) } else { a.lod };
        if let Some(b) = a.bias.filter(|_| self.frag && implicit) {
            mask |= en::IMG_BIAS;
            ops.push(b);
        }
        if let Some(l) = explicit_lod {
            mask |= en::IMG_LOD;
            ops.push(l);
        }
        if let Some((dx, dy)) = a.grad {
            mask |= en::IMG_GRAD;
            ops.extend([dx, dy]);
        }
        if let Some(o) = a.offset {
            // a constant offset is valid on every sample; a variable one
            // only on gathers (the lift refuses it elsewhere)
            if self.m.is_constant(o) {
                mask |= en::IMG_CONST_OFFSET;
            } else {
                mask |= en::IMG_OFFSET;
                self.m.capability(en::CAP_IMAGE_GATHER_EXTENDED);
            }
            ops.push(o);
        }
        let clamp = matches!(kind, ImageKind::Sample { .. }) && (explicit_lod.is_none() || a.grad.is_some());
        if let Some(m) = a.min_lod.filter(|_| clamp) {
            mask |= en::IMG_MIN_LOD;
            self.m.capability(en::CAP_MIN_LOD);
            ops.push(m);
        }
        (mask, ops)
    }

    /// Loads, stores and size queries of an image whose kind the T# says
    /// at run time (storage bit 25, integer bit 24): a selection per
    /// variant.
    fn img_typed(&mut self, kind: ImageKind, class: ImageClass, a: ImgArgs, tid: Id) -> Id {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let v4u = self.m.t_vec(u, 4);
        let k25 = self.c32(1 << 25);
        let k24 = self.c32(1 << 24);
        let z = self.c32(0);
        let sbit = self.m.op(op::BITWISE_AND, u, &[a.tw, k25]);
        let is_storage = self.m.op(op::I_NOT_EQUAL, b, &[sbit, z]);
        let ibit = self.m.op(op::BITWISE_AND, u, &[a.tw, k24]);
        let is_int = self.m.op(op::I_NOT_EQUAL, b, &[ibit, z]);
        let store = matches!(kind, ImageKind::Store { .. });
        let size = kind == ImageKind::Size;
        let merge = self.m.id();
        let (lt, le) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
        let mut incoming = Vec::new();
        if store || size {
            // stores: storage images, integer or float; sizes: storage or sampled
            let (cond, arms) = if store { (is_int, [(true, true), (true, false)]) } else { (is_storage, [(true, false), (false, false)]) };
            self.m.op_void(op::BRANCH_CONDITIONAL, &[cond, lt, le]);
            for (lab, (s, i)) in [(lt, arms[0]), (le, arms[1])] {
                self.m.label(lab);
                self.block = lab;
                let r = self.img_variant(kind, class, a, tid, s, i);
                incoming.push((r, self.block));
                self.m.op_void(op::BRANCH, &[merge]);
            }
        } else {
            self.m.op_void(op::BRANCH_CONDITIONAL, &[is_storage, lt, le]);
            for (lab, storage) in [(lt, true), (le, false)] {
                self.m.label(lab);
                let inner = self.m.id();
                let (li, lf) = (self.m.id(), self.m.id());
                self.m.op_void(op::SELECTION_MERGE, &[inner, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[is_int, li, lf]);
                let mut inc = Vec::new();
                for (l2, int) in [(li, true), (lf, false)] {
                    self.m.label(l2);
                    self.block = l2;
                    inc.extend([self.img_variant(kind, class, a, tid, storage, int), l2]);
                    self.m.op_void(op::BRANCH, &[inner]);
                }
                self.m.label(inner);
                let r = self.m.op(op::PHI, v4u, &inc);
                incoming.push((r, inner));
                self.m.op_void(op::BRANCH, &[merge]);
            }
        }
        self.m.label(merge);
        self.block = merge;
        if store {
            return 0;
        }
        let result_ty = if size { tid } else { v4u };
        self.m.op(op::PHI, result_ty, &[incoming[0].0, incoming[0].1, incoming[1].0, incoming[1].1])
    }

    /// One variant of a load, store or size query.
    fn img_variant(&mut self, kind: ImageKind, class: ImageClass, a: ImgArgs, tid: Id, storage: bool, int: bool) -> Id {
        let size = kind == ImageKind::Size;
        // a size query keeps a cube a cube (loads read it as an array)
        let c = if size && !storage { ImageClass { dynamic: false, ..class } } else { class };
        let (im, _) = self.load_image(c, int, storage, false, a.tw);
        let sample = |o: &mut Vec<Id>| {
            if let Some(s) = a.sample {
                o.extend([en::IMG_SAMPLE, s]);
            }
        };
        let mut o = vec![im, a.coord];
        if size {
            self.m.capability(en::CAP_IMAGE_QUERY);
            if storage || class.ms {
                return self.m.op(op::IMAGE_QUERY_SIZE, tid, &[im]);
            }
            return self.m.op(op::IMAGE_QUERY_SIZE_LOD, tid, &[im, a.lod.unwrap()]);
        }
        let (u, f) = (self.t.u32_, self.t.f32_);
        let (v4u, v4f) = (self.m.t_vec(u, 4), self.m.t_vec(f, 4));
        if matches!(kind, ImageKind::Store { .. }) {
            let texel = if int { a.data.unwrap() } else { self.m.op(op::BITCAST, v4f, &[a.data.unwrap()]) };
            o.push(texel);
            sample(&mut o);
            self.m.op_void(op::IMAGE_WRITE, &o);
            return 0;
        }
        let rt = if int { v4u } else { v4f };
        let r = if storage {
            sample(&mut o);
            self.m.op(op::IMAGE_READ, rt, &o)
        } else {
            match a.sample {
                Some(_) => sample(&mut o),
                None => {
                    let l = a.lod.unwrap_or_else(|| self.c32(0));
                    o.extend([en::IMG_LOD, l]);
                }
            }
            self.m.op(op::IMAGE_FETCH, rt, &o)
        };
        if int { r } else { self.m.op(op::BITCAST, v4u, &[r]) }
    }

    /// An image atomic through a texel pointer of an R32ui storage image.
    fn img_atomic(&mut self, o: u16, class: ImageClass, a: ImgArgs) -> Id {
        let u = self.t.u32_;
        let (var, ity) = self.image_var_f(class, true, true, false, 33);
        let k = self.c32(0xFF_FFFF);
        let slot = self.m.op(op::BITWISE_AND, u, &[a.tw, k]);
        let pi = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, ity);
        let ptr = self.m.op(op::ACCESS_CHAIN, pi, &[var, slot]);
        let pt = self.m.t_ptr(11, u); // StorageClass Image
        let s = a.sample.unwrap_or_else(|| self.c32(0));
        let tp = self.m.op(60, pt, &[ptr, a.coord, s]); // OpImageTexelPointer
        let (scope, sem) = (self.c32(en::SCOPE_DEVICE), self.c32(0));
        if o == op::ATOMIC_COMPARE_EXCHANGE {
            self.m.op(o, u, &[tp, scope, sem, sem, a.data.unwrap(), a.cmp.unwrap()])
        } else {
            self.m.op(o, u, &[tp, scope, sem, a.data.unwrap()])
        }
    }

    /// A 64-bit image atomic (data and result: uvec2, low dword first)
    /// through a texel pointer of the image's R64_UINT storage view, set 0
    /// binding 3 at the T#'s slot (ABI v5).
    fn img_atomic64(&mut self, o: u16, class: ImageClass, a: ImgArgs) -> Id {
        let (u, q) = (self.t.u32_, self.t.u64_);
        let (var, ity) = self.image64_var(class);
        let k = self.c32(0xFF_FFFF);
        let slot = self.m.op(op::BITWISE_AND, u, &[a.tw, k]);
        let pi = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, ity);
        let ptr = self.m.op(op::ACCESS_CHAIN, pi, &[var, slot]);
        let pt = self.m.t_ptr(11, q); // StorageClass Image
        let s = a.sample.unwrap_or_else(|| self.c32(0));
        let tp = self.m.op(60, pt, &[ptr, a.coord, s]); // OpImageTexelPointer
        let (scope, sem) = (self.c32(en::SCOPE_DEVICE), self.c32(0));
        let x = self.m.op(op::BITCAST, q, &[a.data.unwrap()]);
        let old = if o == op::ATOMIC_COMPARE_EXCHANGE {
            let c = self.m.op(op::BITCAST, q, &[a.cmp.unwrap()]);
            self.m.op(o, q, &[tp, scope, sem, sem, x, c])
        } else {
            self.m.op(o, q, &[tp, scope, sem, x])
        };
        let v2 = self.m.t_vec(u, 2);
        self.m.op(op::BITCAST, v2, &[old])
    }

    /// The `R64ui` storage images of set 0 binding 3, one variable per
    /// shape.
    fn image64_var(&mut self, c: ImageClass) -> (Id, Id) {
        let q = self.t.u64_;
        // cubes are stored as 2D arrays (as on binding 1)
        let (dim, arrayed) = if c.dim == 3 { (1, true) } else { (c.dim, c.arrayed) };
        let ity = self.m.t_image_df(q, u32::from(dim), false, arrayed, c.ms, 2, IMAGE_FORMAT_R64UI);
        if let Some(&v) = self.images64.get(&(dim, arrayed, c.ms)) {
            return (v, ity);
        }
        self.m.extension("SPV_EXT_shader_image_int64");
        self.m.capability(CAP_INT64_IMAGE_EXT);
        self.m.capability(en::CAP_INT64_ATOMICS);
        if dim == 0 {
            self.m.capability(en::CAP_IMAGE_1D);
        }
        if c.ms {
            self.m.capability(en::CAP_STORAGE_IMAGE_MULTISAMPLE);
        }
        let arr = self.m.t_array(ity, 1 << 20);
        let p = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, arr);
        let v = self.m.variable(p, en::SC_UNIFORM_CONSTANT);
        self.m.decorate(v, en::DEC_DESCRIPTOR_SET, &[0]);
        self.m.decorate(v, en::DEC_BINDING, &[3]);
        self.interface.push(v);
        self.images64.insert((dim, arrayed, c.ms), v);
        (v, ity)
    }
}

/// SPIR-V image format `R64ui` and capability `Int64ImageEXT`
/// (SPV_EXT_shader_image_int64).
const IMAGE_FORMAT_R64UI: u32 = 40;
const CAP_INT64_IMAGE_EXT: u32 = 5016;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::lift::Interface;
    use crate::vk::tir::{Block, Func, Node};

    /// Decorations `Binding` of the module's variables.
    fn bindings(words: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            if words[at] & 0xffff == 71 && words.get(at + 2) == Some(&33) {
                out.push(words[at + 3]);
            }
            at += n.max(1);
        }
        out
    }

    #[test]
    fn a_two_dword_atomic_is_a_u64_atomic_on_binding_3() {
        let mut f = Func::default();
        let u2 = Ty::vec(S::U32, 2);
        let tw = f.konst(Ty::U32, 1);
        let (a, b) = (f.konst(Ty::U32, 2), f.konst(Ty::U32, 3));
        let coord = f.push(Node { op: Op::Spv(op::COMPOSITE_CONSTRUCT), ty: u2, args: vec![a, b], imm: vec![] });
        let data = f.push(Node { op: Op::Spv(op::COMPOSITE_CONSTRUCT), ty: u2, args: vec![b, a], imm: vec![] });
        let class = ImageClass { dim: 1, arrayed: false, ms: false, uint: true, storage: true, dynamic: false, cube_layers: false };
        let kind = ImageKind::Atomic(op::ATOMIC_U_MAX);
        let at = f.push(Node { op: Op::Image { kind, class, sampler: Sampler::None, flags: img::DATA }, ty: u2, args: vec![tw, coord, data], imm: vec![] });
        f.blocks.push(Block { phis: vec![], preds: vec![], body: vec![Item::N(coord), Item::N(data), Item::N(at)], term: Term::Return });
        let g = Graph::new(&f).unwrap();
        let iface = Interface::default();
        let st = Stage {
            stage: HwStage::Ps,
            local_size: [1, 1, 1],
            out_types: HashMap::new(),
            locs: HashMap::new(),
            mesh: (0, 0),
            mesh_split: None,
            loop_limit: 1 << 14,
            guard_id: 0,
            no_sel: HashSet::new(),
        };
        let e = Emit::new(&f, &g, &iface, &st).run().unwrap();
        let u = crate::vk::usage::usage(&e.words);
        assert!(u.capabilities.contains(&CAP_INT64_IMAGE_EXT));
        assert!(u.capabilities.contains(&en::CAP_INT64_ATOMICS));
        assert_eq!(bindings(&e.words), vec![3]);
    }
}
