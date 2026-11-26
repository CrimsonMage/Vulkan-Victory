//! Image instructions (MIMG).

use super::*;
use crate::vk::tir::{Sampler, img};

impl Lift<'_> {
    fn addr_dw(&mut self, inst: &Inst, addr: &[Operand], k: usize) -> Res<V> {
        let x = if addr.len() == 1 {
            let o = addr[0];
            self.opnd_dw(inst, &o, k as u8, false)?
        } else {
            let o = *addr.get(k).ok_or(unsupported(inst.machine.offset, "an image address"))?;
            self.opnd(inst, &o)?
        };
        Ok(self.u32v(x))
    }

    pub(super) fn image(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Image { data, addr, rsrc, sampler, dmask, dim, a16, d16, tfe, lwe, cache, .. } = inst.fields.clone()
        else {
            unreachable!()
        };
        if a16 || tfe || lwe {
            return Err(unsupported(off, "an image access with a16, tfe or lwe"));
        }
        if name.starts_with("image_bvh") {
            return self.bvh_miss(inst, &data);
        }
        // T# word 7 (slot and flags)
        let tw = self.opnd_dw(inst, &rsrc, 7, false)?;
        let tw = self.u32v(tw);
        let (sdim, arrayed, ms, coords) = match dim {
            0 => (0u8, false, false, 1usize),
            1 => (1, false, false, 2),
            2 => (2, false, false, 3),
            3 => (3, false, false, 3),
            4 => (0, true, false, 2),
            5 => (1, true, false, 3),
            6 => (1, false, true, 2),
            _ => (1, true, true, 3),
        };
        let class = ImageClass { dim: sdim, arrayed, ms, uint: false, storage: false, dynamic: false };
        let rest = name.trim_start_matches("image_");
        // destination
        let dreg = match data {
            Operand::Vector { reg, .. } => reg,
            _ => return Err(unsupported(off, name)),
        };
        let mut k = 0usize; // address dword cursor
        let mut args = vec![tw];
        let mut flags = 0u16;
        let samp_ref = |l: &mut Self, args: &mut Vec<V>| -> Res<Sampler> {
            let s = sampler.ok_or(unsupported(off, "a sample without a sampler"))?;
            let w: Vec<V> = (0..4)
                .map(|i| {
                    let x = l.opnd_dw(inst, &s, i, false)?;
                    Ok(l.u32v(x))
                })
                .collect::<Res<_>>()?;
            let c: Vec<Option<u64>> = w.iter().map(|&x| l.f.as_const(x)).collect();
            if c.iter().all(Option::is_some) {
                let words = [c[0].unwrap() as u32, c[1].unwrap() as u32, c[2].unwrap() as u32, c[3].unwrap() as u32];
                let n = match l.iface.samplers.iter().position(|s| *s == words) {
                    Some(n) => n,
                    None => {
                        l.iface.samplers.push(words);
                        l.iface.samplers.len() - 1
                    }
                };
                return Ok(Sampler::Static(n as u32));
            }
            args.push(w[3]);
            Ok(Sampler::Dynamic)
        };
        let fcoord = |l: &mut Self, k: &mut usize, n: usize| -> Res<V> {
            let mut c = Vec::new();
            for _ in 0..n {
                let x = l.addr_dw(inst, &addr, *k)?;
                c.push(l.f32v(x));
                *k += 1;
            }
            Ok(if n == 1 { c[0] } else { l.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F32, n as u8), &c) })
        };
        if rest.starts_with("sample") || rest.starts_with("gather4") {
            let gather = rest.starts_with("gather4");
            let mods = rest.trim_start_matches("sample").trim_start_matches("gather4");
            let has = |m: &str| mods.split('_').any(|t| t == m);
            let offset = has("o");
            let bias = has("b");
            let dref = has("c");
            let grad = has("d") || has("cd");
            let lz = has("lz");
            let lod = has("l") || lz;
            let min_lod = has("cl");
            if has("cd") {
                return Err(unsupported(off, "coarse derivatives"));
            }
            let s = samp_ref(self, &mut args)?;
            // the address order: offset, bias, z-compare, derivatives, coords, lod/clamp
            let off_v = if offset {
                let o = self.addr_dw(inst, &addr, k)?;
                k += 1;
                Some(o)
            } else {
                None
            };
            let bias_v = if bias {
                let b = self.addr_dw(inst, &addr, k)?;
                k += 1;
                Some(self.f32v(b))
            } else {
                None
            };
            let dref_v = if dref {
                let d = self.addr_dw(inst, &addr, k)?;
                k += 1;
                Some(self.f32v(d))
            } else {
                None
            };
            let dims = match sdim {
                0 => 1,
                2 | 3 => 3,
                _ => 2,
            };
            let grad_v = if grad {
                if sdim == 3 {
                    return Err(unsupported(off, "cube derivatives"));
                }
                let mut dx = Vec::new();
                let mut dy = Vec::new();
                for _ in 0..dims {
                    let x = self.addr_dw(inst, &addr, k)?;
                    dx.push(self.f32v(x));
                    k += 1;
                }
                for _ in 0..dims {
                    let x = self.addr_dw(inst, &addr, k)?;
                    dy.push(self.f32v(x));
                    k += 1;
                }
                let mk = |l: &mut Self, v: &[V]| {
                    if v.len() == 1 { v[0] } else { l.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F32, v.len() as u8), v) }
                };
                Some((mk(self, &dx), mk(self, &dy)))
            } else {
                None
            };
            let coord = if sdim == 3 {
                let c = fcoord(self, &mut k, 3)?;
                self.cube_dir(c)
            } else {
                fcoord(self, &mut k, coords)?
            };
            args.push(coord);
            let lod_v = if lod && !lz {
                let x = self.addr_dw(inst, &addr, k)?;
                k += 1;
                Some(self.f32v(x))
            } else if lz {
                Some(self.kf(0.0))
            } else {
                None
            };
            let min_v = if min_lod {
                let x = self.addr_dw(inst, &addr, k)?;
                Some(self.f32v(x))
            } else {
                None
            };
            if let Some(d) = dref_v {
                flags |= img::DREF;
                args.push(d);
            }
            if let Some(b) = bias_v {
                flags |= img::BIAS;
                args.push(b);
            }
            if let Some(l) = lod_v {
                flags |= img::LOD;
                args.push(l);
            }
            if let Some((dx, dy)) = grad_v {
                flags |= img::GRAD;
                args.push(dx);
                args.push(dy);
            }
            if let Some(o) = off_v {
                // packed 6-bit offsets per coordinate
                flags |= img::OFFSET;
                let mut comps = Vec::new();
                for c in 0..dims.min(if sdim == 3 { 0 } else { 3 }) {
                    let (sh, w) = (self.k32(8 * c as u32), self.k32(6));
                    comps.push(self.spv(op::BIT_FIELD_S_EXTRACT, Ty::U32, &[o, sh, w]));
                }
                if comps.is_empty() {
                    return Err(unsupported(off, "a cube offset"));
                }
                let v = if comps.len() == 1 {
                    comps[0]
                } else {
                    self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, comps.len() as u8), &comps)
                };
                args.push(v);
            }
            if let Some(m) = min_v {
                flags |= img::MIN_LOD;
                args.push(m);
            }
            let kind = if gather {
                ImageKind::Gather { comp: dmask.trailing_zeros() as u8, lod, dref, offset }
            } else {
                ImageKind::Sample { lod, bias, grad, dref, offset, min_lod }
            };
            let r = self.emit(Op::Image { kind, class, sampler: s, flags }, Ty::vec(S::F32, 4), args);
            let mask = if gather { 0xF } else { dmask };
            if dref && !gather {
                // a comparison returns one value
                let x = if self.f.ty(r) == Ty::F32 { r } else { self.extract(r, 0) };
                return self.write_result(inst, dreg, &[x], d16);
            }
            let vals: Vec<V> = (0..4).filter(|c| mask >> c & 1 != 0).map(|c| self.extract(r, c)).collect();
            return self.write_result(inst, dreg, &vals, d16);
        }
        match rest {
            "load" | "load_mip" | "store" | "store_mip" => {
                let mut c = Vec::new();
                for _ in 0..coords {
                    c.push(self.addr_dw(inst, &addr, k)?);
                    k += 1;
                }
                let coord = if c.len() == 1 { c[0] } else { self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, c.len() as u8), &c) };
                args.push(coord);
                let mip = rest.ends_with("_mip");
                if mip {
                    let m = self.addr_dw(inst, &addr, k)?;
                    k += 1;
                    flags |= img::LOD;
                    args.push(m);
                }
                if ms {
                    let s = self.addr_dw(inst, &addr, k)?;
                    flags |= img::SAMPLE;
                    args.push(s);
                }
                let class = ImageClass { dynamic: true, ..class };
                if rest.starts_with("load") {
                    let r = self.emit(
                        Op::Image { kind: ImageKind::Load { mip }, class, sampler: Sampler::None, flags },
                        Ty::vec(S::U32, 4),
                        args,
                    );
                    let vals: Vec<V> = (0..4).filter(|c| dmask >> c & 1 != 0).map(|c| self.extract(r, c)).collect();
                    self.write_result(inst, dreg, &vals, d16)
                } else {
                    if d16 {
                        return Err(unsupported(off, "a d16 image store"));
                    }
                    let mut vals = Vec::new();
                    let mut j = 0u8;
                    for c in 0..4 {
                        if dmask >> c & 1 != 0 {
                            let x = self.opnd_dw(inst, &data, j, false)?;
                            vals.push(self.u32v(x));
                            j += 1;
                        } else {
                            vals.push(self.k32(0));
                        }
                    }
                    let v = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, 4), &vals);
                    flags |= img::DATA;
                    args.push(v);
                    self.emit(
                        Op::Image { kind: ImageKind::Store { mip }, class, sampler: Sampler::None, flags },
                        Ty::VOID,
                        args,
                    );
                    Ok(())
                }
            }
            _ if rest.starts_with("atomic_") => {
                let opc = atomic_op(name).ok_or(unsupported(off, name))?;
                let mut c = Vec::new();
                for _ in 0..coords {
                    c.push(self.addr_dw(inst, &addr, k)?);
                    k += 1;
                }
                let coord = if c.len() == 1 { c[0] } else { self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, c.len() as u8), &c) };
                args.push(coord);
                let x = self.opnd_dw(inst, &data, 0, false)?;
                let x = self.u32v(x);
                flags |= img::DATA;
                args.push(x);
                if rest.contains("cmpswap") {
                    let y = self.opnd_dw(inst, &data, 1, false)?;
                    let y = self.u32v(y);
                    flags |= img::CMP;
                    args.push(y);
                }
                let class = ImageClass { uint: true, storage: true, ..class };
                let r = self.emit(Op::Image { kind: ImageKind::Atomic(opc), class, sampler: Sampler::None, flags }, Ty::U32, args);
                if cache.glc {
                    self.def(inst, Reg::V(dreg), R::D(r));
                }
                Ok(())
            }
            "get_resinfo" => {
                let m = self.addr_dw(inst, &addr, 0)?;
                flags |= img::LOD;
                args.push(self.k32(0));
                let _ = args.pop();
                args.push(m);
                let class = ImageClass { dynamic: true, ..class };
                let n = match (sdim, arrayed) {
                    (0, false) => 1,
                    (0, true) | (1, false) | (3, false) => 2,
                    _ => 3,
                };
                let sz = self.emit(
                    Op::Image { kind: ImageKind::Size, class, sampler: Sampler::None, flags },
                    Ty::vec(S::U32, n),
                    args.clone(),
                );
                let lv = self.emit(
                    Op::Image { kind: ImageKind::Levels, class, sampler: Sampler::None, flags: 0 },
                    Ty::U32,
                    vec![tw],
                );
                let mut all = Vec::new();
                for c in 0..3u32 {
                    all.push(if c < u32::from(n) { if n == 1 { sz } else { self.extract(sz, c) } } else { self.k32(1) });
                }
                all.push(lv);
                let vals: Vec<V> = (0..4).filter(|c| dmask >> c & 1 != 0).map(|c| all[c as usize]).collect();
                self.write_result(inst, dreg, &vals, false)
            }
            "get_lod" => {
                let s = samp_ref(self, &mut args)?;
                let coord = fcoord(self, &mut k, coords.min(3))?;
                args.push(coord);
                let r = self.emit(
                    Op::Image { kind: ImageKind::QueryLod, class, sampler: s, flags: 0 },
                    Ty::vec(S::F32, 2),
                    args,
                );
                let vals: Vec<V> = (0..2).filter(|c| dmask >> c & 1 != 0).map(|c| self.extract(r, c)).collect();
                self.write_result(inst, dreg, &vals, false)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    /// Writes image results to consecutive VGPRs (d16: two per dword).
    fn write_result(&mut self, inst: &Inst, dreg: u8, vals: &[V], d16: bool) -> Res<()> {
        if d16 {
            for (i, pair) in vals.chunks(2).enumerate() {
                let mut halves = Vec::new();
                for &x in pair {
                    let h = match self.f.ty(x) {
                        Ty::F32 => {
                            let h = self.spv(op::F_CONVERT, Ty::F16, &[x]);
                            self.spv(op::BITCAST, Ty::U16, &[h])
                        }
                        _ => self.spv(op::U_CONVERT, Ty::U16, &[x]),
                    };
                    halves.push(h);
                }
                if halves.len() == 1 {
                    halves.push(self.f.konst(Ty::U16, 0));
                }
                let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U16, 2), &halves);
                let w = self.spv(op::BITCAST, Ty::U32, &[v2]);
                self.def(inst, Reg::V(dreg + i as u8), R::D(w));
            }
            return Ok(());
        }
        for (i, &x) in vals.iter().enumerate() {
            self.def(inst, Reg::V(dreg + i as u8), R::D(x));
        }
        Ok(())
    }

    /// (s, t, face) as the console's cube instructions give them, to a
    /// direction.
    fn cube_dir(&mut self, c: V) -> V {
        let s = self.extract(c, 0);
        let t = self.extract(c, 1);
        let face = self.extract(c, 2);
        let k15 = self.kf(1.5);
        let two = self.kf(2.0);
        let u = self.spv(op::F_SUB, Ty::F32, &[s, k15]);
        let u = self.spv(op::F_MUL, Ty::F32, &[u, two]);
        let v = self.spv(op::F_SUB, Ty::F32, &[t, k15]);
        let v = self.spv(op::F_MUL, Ty::F32, &[v, two]);
        let fi = self.spv(op::CONVERT_F_TO_U, Ty::U32, &[face]);
        let k7 = self.k32(7);
        let fi = self.spv(op::BITWISE_AND, Ty::U32, &[fi, k7]);
        let nu = self.spv(op::F_NEGATE, Ty::F32, &[u]);
        let nv = self.spv(op::F_NEGATE, Ty::F32, &[v]);
        let (one, mone) = (self.kf(1.0), self.kf(-1.0));
        let table: [[V; 3]; 6] = [[one, nv, nu], [mone, nv, u], [u, one, v], [u, mone, nv], [u, nv, one], [nu, nv, mone]];
        let mut out = Vec::new();
        for comp in 0..3 {
            let mut x = table[5][comp];
            for f in (0..5).rev() {
                let kf = self.k32(f as u32);
                let is = self.spv(op::I_EQUAL, Ty::BOOL, &[fi, kf]);
                x = self.spv(op::SELECT, Ty::F32, &[is, table[f][comp], x]);
            }
            out.push(x);
        }
        self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F32, 3), &out)
    }

    /// The ray/node test misses every node (as `VICTORY_BVH_MISS`): a box
    /// node returns no child (all lanes 0xFFFFFFFF), a triangle a miss.
    fn bvh_miss(&mut self, inst: &Inst, data: &Operand) -> Res<()> {
        self.approx("ray/node tests miss every node (no BVH yet)");
        let Operand::Vector { reg, dwords } = *data else {
            return Err(unsupported(inst.machine.offset, "a BVH test"));
        };
        let all = self.k32(u32::MAX);
        for k in 0..dwords.max(4) {
            self.def(inst, Reg::V(reg + k), R::D(all));
        }
        Ok(())
    }
}
