//! Image instructions.

use super::*;

impl Emit<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn image(&mut self, _v: V, kind: ImageKind, class: ImageClass, sampler: Sampler, flags: u16, args: &[Id], tid: Id) -> Id {
        let (u, f) = (self.t.u32_, self.t.f32_);
        let v4u = self.m.t_vec(u, 4);
        let v4f = self.m.t_vec(f, 4);
        let tw = args[0];
        let mut i = 1;
        let sw = if sampler == Sampler::Dynamic {
            i += 1;
            Some(args[1])
        } else {
            None
        };
        let coord = args[i];
        i += 1;
        let mut opt = |bit: u16| -> Option<Id> {
            if flags & bit != 0 {
                let x = args[i];
                i += 1;
                Some(x)
            } else {
                None
            }
        };
        let dref = opt(img::DREF);
        let bias = opt(img::BIAS);
        let lod = opt(img::LOD);
        let grad = if flags & img::GRAD != 0 {
            let (a, b) = (args[i], args[i + 1]);
            i += 2;
            Some((a, b))
        } else {
            None
        };
        let mut opt = |bit: u16| -> Option<Id> {
            if flags & bit != 0 {
                let x = args[i];
                i += 1;
                Some(x)
            } else {
                None
            }
        };
        let offset = opt(img::OFFSET);
        let min_lod = opt(img::MIN_LOD);
        let sample = opt(img::SAMPLE);
        let data = opt(img::DATA);
        let cmp = opt(img::CMP);
        match kind {
            ImageKind::Sample { .. } | ImageKind::Gather { .. } | ImageKind::QueryLod => {
                let depth = dref.is_some();
                let (im, _) = self.load_image(class, false, false, depth, tw);
                let s = self.load_sampler(sampler, sw);
                let ity = self.m.t_image_d(f, u32::from(class.dim), depth, class.arrayed, class.ms, 1);
                let sity = self.m.t_sampled_image(ity);
                let si = self.m.op(op::SAMPLED_IMAGE, sity, &[im, s]);
                if kind == ImageKind::QueryLod {
                    self.m.capability(en::CAP_IMAGE_QUERY);
                    if !self.frag {
                        return self.m.c_null(tid);
                    }
                    let v2 = self.m.t_vec(f, 2);
                    return self.m.op(op::IMAGE_QUERY_LOD, v2, &[si, coord]);
                }
                let mut mask = 0u32;
                let mut ops = Vec::new();
                let implicit = lod.is_none() && grad.is_none();
                let explicit_lod = if implicit && !self.frag {
                    // no derivatives outside pixel shaders: level 0
                    Some(self.m.c_f32(0))
                } else {
                    lod
                };
                if let Some(b) = bias {
                    if self.frag && implicit {
                        mask |= en::IMG_BIAS;
                        ops.push(b);
                    }
                }
                if let Some(l) = explicit_lod {
                    mask |= en::IMG_LOD;
                    ops.push(l);
                }
                if let Some((dx, dy)) = grad {
                    mask |= en::IMG_GRAD;
                    ops.push(dx);
                    ops.push(dy);
                }
                if let Some(o) = offset {
                    mask |= en::IMG_OFFSET;
                    self.m.capability(en::CAP_IMAGE_GATHER_EXTENDED);
                    ops.push(o);
                }
                if let Some(m) = min_lod {
                    if matches!(kind, ImageKind::Sample { .. }) && (explicit_lod.is_none() || grad.is_some()) {
                        mask |= en::IMG_MIN_LOD;
                        self.m.capability(en::CAP_MIN_LOD);
                        ops.push(m);
                    }
                }
                let lodded = mask & (en::IMG_LOD | en::IMG_GRAD) != 0;
                if let ImageKind::Gather { comp, .. } = kind {
                    let mut o = vec![si, coord];
                    let gm = mask & !(en::IMG_LOD | en::IMG_GRAD | en::IMG_BIAS | en::IMG_MIN_LOD);
                    let gops: Vec<Id> = if gm & en::IMG_OFFSET != 0 { vec![offset.unwrap()] } else { vec![] };
                    let r = if let Some(d) = dref {
                        o.push(d);
                        if gm != 0 {
                            o.push(gm);
                            o.extend(gops);
                        }
                        self.m.op(op::IMAGE_DREF_GATHER, v4f, &o)
                    } else {
                        let c = self.c32(u32::from(comp));
                        o.push(c);
                        if gm != 0 {
                            o.push(gm);
                            o.extend(gops);
                        }
                        self.m.op(op::IMAGE_GATHER, v4f, &o)
                    };
                    return r;
                }
                let mut o = vec![si, coord];
                if let Some(d) = dref {
                    o.push(d);
                }
                if mask != 0 {
                    o.push(mask);
                    o.extend(ops);
                }
                if let Some(_d) = dref {
                    let opc = if lodded { op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD } else { op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD };
                    let r = self.m.op(opc, f, &o);
                    // the TIR reads a vec4 and takes component 0
                    let z = self.m.c_f32(0);
                    return self.m.op(op::COMPOSITE_CONSTRUCT, v4f, &[r, z, z, z]);
                }
                let opc = if lodded { op::IMAGE_SAMPLE_EXPLICIT_LOD } else { op::IMAGE_SAMPLE_IMPLICIT_LOD };
                self.m.op(opc, v4f, &o)
            }
            ImageKind::Load { .. } | ImageKind::Store { .. } | ImageKind::Size => {
                // runtime choice: storage bit 25, integer bit 24
                let b = self.t.bool_;
                let k25 = self.c32(1 << 25);
                let k24 = self.c32(1 << 24);
                let z = self.c32(0);
                let sbit = self.m.op(op::BITWISE_AND, u, &[tw, k25]);
                let is_storage = self.m.op(op::I_NOT_EQUAL, b, &[sbit, z]);
                let ibit = self.m.op(op::BITWISE_AND, u, &[tw, k24]);
                let is_int = self.m.op(op::I_NOT_EQUAL, b, &[ibit, z]);
                let store = matches!(kind, ImageKind::Store { .. });
                let size = kind == ImageKind::Size;
                let variants: Vec<(bool, bool)> = if store {
                    vec![(true, true), (true, false)]
                } else if size {
                    vec![(true, false), (false, false)]
                } else {
                    vec![(true, true), (true, false), (false, true), (false, false)]
                };
                let result_ty = if store { 0 } else if size { tid } else { v4u };
                let merge = self.m.id();
                let mut incoming = Vec::new();
                // a chain of selections: if (storage) { if (int) A else B } else { ... }
                let emit_variant = |e: &mut Self, storage: bool, int: bool| -> Id {
                    // a size query keeps a cube a cube (loads read it as an array)
                    let c = if size && !storage { ImageClass { dynamic: false, ..class } } else { class };
                    let (im, _) = e.load_image(c, int, storage, false, tw);
                    if size {
                        e.m.capability(en::CAP_IMAGE_QUERY);
                        if storage || class.ms {
                            return e.m.op(op::IMAGE_QUERY_SIZE, tid, &[im]);
                        }
                        return e.m.op(op::IMAGE_QUERY_SIZE_LOD, tid, &[im, lod.unwrap()]);
                    }
                    if store {
                        let texel = if int { data.unwrap() } else { e.m.op(op::BITCAST, v4f, &[data.unwrap()]) };
                        let mut o = vec![im, coord, texel];
                        if let Some(s) = sample {
                            o.push(en::IMG_SAMPLE);
                            o.push(s);
                        }
                        e.m.op_void(op::IMAGE_WRITE, &o);
                        return 0;
                    }
                    let rt = if int { v4u } else { v4f };
                    let r = if storage {
                        let mut o = vec![im, coord];
                        if let Some(s) = sample {
                            o.push(en::IMG_SAMPLE);
                            o.push(s);
                        }
                        e.m.op(op::IMAGE_READ, rt, &o)
                    } else {
                        let mut o = vec![im, coord];
                        if let Some(s) = sample {
                            o.push(en::IMG_SAMPLE);
                            o.push(s);
                        } else {
                            let l = lod.unwrap_or(z);
                            o.push(en::IMG_LOD);
                            o.push(l);
                        }
                        e.m.op(op::IMAGE_FETCH, rt, &o)
                    };
                    if int { r } else { e.m.op(op::BITCAST, v4u, &[r]) }
                };
                if variants.len() == 2 {
                    let cond = if store { is_int } else { is_storage };
                    let (lt, le) = (self.m.id(), self.m.id());
                    self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[cond, lt, le]);
                    for (lab, (s, i)) in [(lt, variants[0]), (le, variants[1])] {
                        self.m.label(lab);
                        self.block = lab;
                        let r = emit_variant(self, s, i);
                        incoming.push((r, self.block));
                        self.m.op_void(op::BRANCH, &[merge]);
                    }
                } else {
                    let (ls, ln) = (self.m.id(), self.m.id());
                    self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[is_storage, ls, ln]);
                    for (lab, storage) in [(ls, true), (ln, false)] {
                        self.m.label(lab);
                        let inner = self.m.id();
                        let (li, lf) = (self.m.id(), self.m.id());
                        self.m.op_void(op::SELECTION_MERGE, &[inner, 0]);
                        self.m.op_void(op::BRANCH_CONDITIONAL, &[is_int, li, lf]);
                        let mut inc = Vec::new();
                        for (l2, int) in [(li, true), (lf, false)] {
                            self.m.label(l2);
                            self.block = l2;
                            let r = emit_variant(self, storage, int);
                            inc.push((r, l2));
                            self.m.op_void(op::BRANCH, &[inner]);
                        }
                        self.m.label(inner);
                        let r = self.m.op(op::PHI, v4u, &[inc[0].0, inc[0].1, inc[1].0, inc[1].1]);
                        incoming.push((r, inner));
                        self.m.op_void(op::BRANCH, &[merge]);
                    }
                }
                self.m.label(merge);
                self.block = merge;
                if store {
                    return 0;
                }
                self.m.op(op::PHI, result_ty, &[incoming[0].0, incoming[0].1, incoming[1].0, incoming[1].1])
            }
            ImageKind::Levels => {
                self.m.capability(en::CAP_IMAGE_QUERY);
                if class.ms {
                    return self.c32(1);
                }
                let c = ImageClass { dynamic: false, ..class };
                let (im, _) = self.load_image(c, false, false, false, tw);
                self.m.op(op::IMAGE_QUERY_LEVELS, u, &[im])
            }
            ImageKind::Atomic(o) => {
                let (var, ity) = self.image_var_f(class, true, true, false, 33);
                let k = self.c32(0xFF_FFFF);
                let slot = self.m.op(op::BITWISE_AND, u, &[tw, k]);
                let pi = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, ity);
                let ptr = self.m.op(op::ACCESS_CHAIN, pi, &[var, slot]);
                let pt = self.m.t_ptr(11, u); // StorageClass Image
                let s = sample.unwrap_or_else(|| self.c32(0));
                let tp = self.m.op(60, pt, &[ptr, coord, s]); // OpImageTexelPointer
                let (scope, sem) = (self.c32(en::SCOPE_DEVICE), self.c32(0));
                if o == op::ATOMIC_COMPARE_EXCHANGE {
                    self.m.op(o, u, &[tp, scope, sem, sem, data.unwrap(), cmp.unwrap()])
                } else {
                    self.m.op(o, u, &[tp, scope, sem, data.unwrap()])
                }
            }
        }
    }
}
