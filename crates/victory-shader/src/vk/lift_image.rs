//! Image instructions (MIMG).

use super::*;
use crate::vk::tir::{Sampler, img};

impl Lift<'_> {
    pub(super) fn addr_dw(&mut self, inst: &Inst, addr: &[Operand], k: usize) -> Res<V> {
        if self.a16 {
            return self.a16_vals.get(k).copied().ok_or(unsupported(inst.machine.offset, "an image address"));
        }
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
        if tfe || lwe {
            return Err(unsupported(off, "an image access with tfe or lwe"));
        }
        if name.starts_with("image_bvh") {
            return self.bvh(inst, name);
        }
        let Operand::Vector { reg: dreg, .. } = data else { return Err(unsupported(off, name)) };
        let (sdim, arrayed, ms, coords) = dim_class(dim);
        let rest = name.trim_start_matches("image_");
        let tw = self.t_word7(inst, &rsrc)?;
        self.a16 = false;
        self.a16_vals.clear();
        let addr = if a16 { self.a16_addr(inst, &addr, rest, sdim, coords, ms)? } else { addr };
        let mut m = Mimg {
            inst,
            name,
            rest,
            addr,
            k: 0,
            args: vec![tw],
            flags: 0,
            class: ImageClass { dim: sdim, arrayed, ms, uint: false, storage: false, dynamic: false, cube_layers: false },
            coords,
            dreg,
            dmask,
            d16,
            tw,
            sampler,
            data,
            glc: cache.glc,
        };
        if rest.starts_with("sample") || rest.starts_with("gather4") {
            return self.image_sample(&mut m);
        }
        match rest {
            "load" | "load_mip" => self.image_load(&mut m),
            "store" | "store_mip" => self.image_store(&mut m),
            _ if rest.starts_with("atomic_") => self.image_atomic(&mut m),
            "get_resinfo" => self.image_resinfo(&mut m),
            "get_lod" => self.image_lod(&mut m),
            _ => Err(unsupported(off, name)),
        }
    }

    /// The address as components in order. With a16 the components after
    /// the whole dwords (offset, bias, z-compare) are 16-bit, two per dword,
    /// low half first: the derivatives (x then y, each set from a fresh
    /// dword), then coordinates, level and clamp. The components go to
    /// `a16_vals`; the operands returned are placeholders.
    fn a16_addr(&mut self, inst: &Inst, addr: &[Operand], rest: &str, sdim: u8, coords: usize, ms: bool) -> Res<Vec<Operand>> {
        let sample = rest.starts_with("sample") || rest.starts_with("gather4");
        let mods = SampleMods::of(rest);
        let (pre, g, more) = if sample {
            let pre = usize::from(mods.offset) + usize::from(mods.bias) + usize::from(mods.dref);
            let g = if mods.grad { grad_dims(sdim) } else { 0 };
            (pre, g, coords + usize::from(mods.lod && !mods.lz) + usize::from(mods.min_lod))
        } else {
            (0, 0, coords + usize::from(rest.ends_with("_mip")) + usize::from(ms))
        };
        let mut vals = Vec::new();
        for i in 0..pre {
            let x = self.addr_dw(inst, addr, i)?;
            // the bias is an f16 in the low half of its dword
            let bias = sample && mods.bias && i == usize::from(mods.offset);
            vals.push(if bias { self.half_f32(x, false) } else { x });
        }
        let mut dw = pre;
        let halves = if g > 0 { [g, g, more] } else { [0, 0, more] };
        for n in halves {
            for j in 0..n {
                let x = self.addr_dw(inst, addr, dw + j / 2)?;
                vals.push(if sample {
                    let f = self.half_f32(x, j % 2 == 1);
                    self.u32v(f)
                } else {
                    let (o, w) = (self.k32(16 * (j % 2) as u32), self.k32(16));
                    self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[x, o, w])
                });
            }
            dw += n.div_ceil(2);
        }
        let out = vec![Operand::Literal(0); vals.len()];
        self.a16_vals = vals;
        self.a16 = true;
        Ok(out)
    }

    /// The next address component (as bits).
    fn next_u(&mut self, m: &mut Mimg) -> Res<V> {
        let x = self.addr_dw(m.inst, &m.addr, m.k)?;
        m.k += 1;
        Ok(x)
    }

    fn next_f(&mut self, m: &mut Mimg) -> Res<V> {
        let x = self.next_u(m)?;
        Ok(self.f32v(x))
    }

    /// `n` components as a vector (a scalar for one).
    fn next_vec(&mut self, m: &mut Mimg, n: usize, float: bool) -> Res<V> {
        let mut c = Vec::new();
        for _ in 0..n {
            c.push(if float { self.next_f(m)? } else { self.next_u(m)? });
        }
        Ok(self.vec_of(&c, if float { S::F32 } else { S::U32 }))
    }

    fn vec_of(&mut self, c: &[V], s: S) -> V {
        if c.len() == 1 { c[0] } else { self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(s, c.len() as u8), c) }
    }

    /// An optional operand: its flag and value.
    fn push_opt(m: &mut Mimg, flag: u16, v: Option<V>) {
        if let Some(v) = v {
            m.flags |= flag;
            m.args.push(v);
        }
    }

    /// The sampler: an immediate S# (constant words) is a static sampler
    /// of the interface; else the S#'s word 3 (its slot) is an argument.
    ///
    /// `compare`: the instruction is a comparison (`_c`). On the console the
    /// comparison is the instruction's: every S# holds a function, used or
    /// not. The host's sampler must compare for these and must not for the
    /// others, so an immediate sampler a comparison uses is listed with
    /// [`crate::sampler::COMPARES`] set in word 0 (origin/main
    /// `emit/lower.rs` `cmp_samplers`, `emit/mod.rs`: its static sampler
    /// gets the comparison filter). Heap samplers compare by the title's
    /// own description.
    fn image_sampler(&mut self, m: &mut Mimg, compare: bool) -> Res<Sampler> {
        let s = m.sampler.ok_or(unsupported(m.inst.machine.offset, "a sample without a sampler"))?;
        let mut w = Vec::new();
        for i in 0..4 {
            let x = self.opnd_dw(m.inst, &s, i, false)?;
            w.push(self.u32v(x));
        }
        let c: Option<Vec<u32>> = w.iter().map(|&x| self.f.as_const(x).map(|k| k as u32)).collect();
        if let Some(words) = c {
            let mark = if compare { crate::sampler::COMPARES } else { 0 };
            let words = [words[0] | mark, words[1], words[2], words[3]];
            let n = self.iface.samplers.iter().position(|s| *s == words).unwrap_or_else(|| {
                self.iface.samplers.push(words);
                self.iface.samplers.len() - 1
            });
            return Ok(Sampler::Static(n as u32));
        }
        m.args.push(w[3]);
        Ok(Sampler::Dynamic)
    }

    /// Sample and gather. The address order: offset, bias, z-compare,
    /// derivatives, coordinates, level, clamp.
    fn image_sample(&mut self, m: &mut Mimg) -> Res<()> {
        let off = m.inst.machine.offset;
        let gather = m.rest.starts_with("gather4");
        let md = SampleMods::of(m.rest);
        if md.coarse {
            return Err(unsupported(off, "coarse derivatives"));
        }
        // a cube with derivatives or offsets: its faces as a 2D array
        let layers = m.class.dim == 3 && (md.grad || md.offset);
        let s = self.image_sampler(m, md.dref)?;
        let off_v = if md.offset { Some(self.next_u(m)?) } else { None };
        let bias_v = if md.bias { Some(self.next_f(m)?) } else { None };
        let dref_v = if md.dref { Some(self.next_f(m)?) } else { None };
        let dims = grad_dims(m.class.dim);
        let grad_v = if md.grad { Some((self.next_vec(m, dims, true)?, self.next_vec(m, dims, true)?)) } else { None };
        let coord = if m.class.dim == 3 {
            let c = self.next_vec(m, 3, true)?;
            if layers { self.cube_layer(c) } else { self.cube_dir(c) }
        } else {
            self.next_vec(m, m.coords, true)?
        };
        m.args.push(coord);
        let lod_v = match (md.lz, md.lod) {
            (true, _) => Some(self.kf(0.0)),
            (false, true) => Some(self.next_f(m)?),
            _ => None,
        };
        let min_v = if md.min_lod { Some(self.next_f(m)?) } else { None };
        Self::push_opt(m, img::DREF, dref_v);
        Self::push_opt(m, img::BIAS, bias_v);
        Self::push_opt(m, img::LOD, lod_v);
        if let Some((dx, dy)) = grad_v {
            m.flags |= img::GRAD;
            m.args.extend([dx, dy]);
        }
        if layers {
            m.class = ImageClass { dim: 1, arrayed: true, cube_layers: true, ..m.class };
        }
        let offs = off_v.map(|o| self.sample_offset(o, dims, m.class.dim, off)).transpose()?;
        Self::push_opt(m, img::OFFSET, offs);
        Self::push_opt(m, img::MIN_LOD, min_v);
        let (lod, dref, offset) = (md.lod, md.dref, md.offset);
        let kind = if gather {
            ImageKind::Gather { comp: m.dmask.trailing_zeros() as u8, lod, dref, offset }
        } else {
            ImageKind::Sample { lod, bias: md.bias, grad: md.grad, dref, offset, min_lod: md.min_lod }
        };
        let args = std::mem::take(&mut m.args);
        let r = self.emit(Op::Image { kind, class: m.class, sampler: s, flags: m.flags }, Ty::vec(S::F32, 4), args);
        self.sample_result(m, r, gather, dref)
    }

    /// A sample's channels by dmask (a gather's four; a comparison's one).
    fn sample_result(&mut self, m: &Mimg, r: V, gather: bool, dref: bool) -> Res<()> {
        if dref && !gather {
            let x = if self.f.ty(r) == Ty::F32 { r } else { self.extract(r, 0) };
            return self.write_result(m.inst, m.dreg, &[x], m.d16);
        }
        let mask = if gather { 0xF } else { m.dmask };
        let vals: Vec<V> = (0..4).filter(|c| mask >> c & 1 != 0).map(|c| self.extract(r, c)).collect();
        self.write_result(m.inst, m.dreg, &vals, m.d16)
    }

    /// Packed 6-bit texel offsets, one per coordinate (none for cubes).
    fn sample_offset(&mut self, o: V, dims: usize, sdim: u8, off: usize) -> Res<V> {
        if sdim == 3 {
            return Err(unsupported(off, "a cube offset"));
        }
        let mut comps = Vec::new();
        for c in 0..dims.min(3) {
            comps.push(match self.f.as_const(o) {
                // a literal offset stays a constant: SPIR-V takes only
                // constant offsets outside gathers
                Some(k) => {
                    let six = (k >> (8 * c)) as u32 & 0x3f;
                    self.k32(if six & 0x20 != 0 { six | !0x3f } else { six })
                }
                None => {
                    let (sh, w) = (self.k32(8 * c as u32), self.k32(6));
                    self.spv(op::BIT_FIELD_S_EXTRACT, Ty::U32, &[o, sh, w])
                }
            });
        }
        Ok(self.vec_of(&comps, S::U32))
    }

    /// Integer coordinates, then the level (`_mip`) and the sample (MSAA).
    fn image_texel(&mut self, m: &mut Mimg) -> Res<bool> {
        let coord = self.next_vec(m, m.coords, false)?;
        m.args.push(coord);
        let mip = m.rest.ends_with("_mip");
        if mip {
            let l = self.next_u(m)?;
            Self::push_opt(m, img::LOD, Some(l));
        }
        if m.class.ms {
            let s = self.next_u(m)?;
            Self::push_opt(m, img::SAMPLE, Some(s));
        }
        m.class.dynamic = true;
        Ok(mip)
    }

    fn image_load(&mut self, m: &mut Mimg) -> Res<()> {
        let mip = self.image_texel(m)?;
        let args = std::mem::take(&mut m.args);
        let op = Op::Image { kind: ImageKind::Load { mip }, class: m.class, sampler: Sampler::None, flags: m.flags };
        let r = self.emit(op, Ty::vec(S::U32, 4), args);
        let vals: Vec<V> = (0..4).filter(|c| m.dmask >> c & 1 != 0).map(|c| self.extract(r, c)).collect();
        self.write_result(m.inst, m.dreg, &vals, m.d16)
    }

    fn image_store(&mut self, m: &mut Mimg) -> Res<()> {
        if m.d16 {
            return Err(unsupported(m.inst.machine.offset, "a d16 image store"));
        }
        let mip = self.image_texel(m)?;
        let mut vals = Vec::new();
        let mut j = 0u8;
        for c in 0..4 {
            if m.dmask >> c & 1 == 0 {
                vals.push(self.k32(0));
                continue;
            }
            let x = self.opnd_dw(m.inst, &m.data, j, false)?;
            vals.push(self.u32v(x));
            j += 1;
        }
        let v = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, 4), &vals);
        Self::push_opt(m, img::DATA, Some(v));
        let args = std::mem::take(&mut m.args);
        let op = Op::Image { kind: ImageKind::Store { mip }, class: m.class, sampler: Sampler::None, flags: m.flags };
        self.emit(op, Ty::VOID, args);
        Ok(())
    }

    /// An image atomic. Two data dwords (four for cmpswap) are one 64-bit
    /// texel, low dword first (ACO's 64-bit image atomics: the alpha
    /// prepass programs' depth-and-payload maximum): a u64 atomic on the
    /// image's R64_UINT view (emit_image.rs); the result is the old texel's
    /// two dwords.
    fn image_atomic(&mut self, m: &mut Mimg) -> Res<()> {
        let opc = atomic_op(m.name).ok_or(unsupported(m.inst.machine.offset, m.name))?;
        let cmpswap = m.rest.contains("cmpswap");
        let words: u8 = if m.dmask.count_ones() > if cmpswap { 2 } else { 1 } { 2 } else { 1 };
        if words == 2 && matches!(opc, op::ATOMIC_F_MIN_EXT | op::ATOMIC_F_MAX_EXT) {
            return Err(unsupported(m.inst.machine.offset, "a 64-bit float image atomic"));
        }
        let coord = self.next_vec(m, m.coords, false)?;
        m.args.push(coord);
        let x = self.atomic_word(m, 0, words)?;
        Self::push_opt(m, img::DATA, Some(x));
        if cmpswap {
            let y = self.atomic_word(m, words, words)?;
            Self::push_opt(m, img::CMP, Some(y));
        }
        let class = ImageClass { uint: true, storage: true, ..m.class };
        let args = std::mem::take(&mut m.args);
        let ty = Ty::vec(S::U32, words);
        let r = self.emit(Op::Image { kind: ImageKind::Atomic(opc), class, sampler: Sampler::None, flags: m.flags }, ty, args);
        if m.glc {
            for k in 0..words {
                let d = if words == 1 { r } else { self.extract(r, u32::from(k)) };
                self.def(m.inst, Reg::V(m.dreg + k), R::D(d));
            }
        }
        Ok(())
    }

    /// Data dwords `first..first + words` of an atomic: a u32, or a uvec2
    /// (low dword first).
    fn atomic_word(&mut self, m: &Mimg, first: u8, words: u8) -> Res<V> {
        let mut dws = Vec::new();
        for k in first..first + words {
            let x = self.opnd_dw(m.inst, &m.data, k, false)?;
            dws.push(self.u32v(x));
        }
        Ok(if words == 1 { dws[0] } else { self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, 2), &dws) })
    }

    /// Width, height, depth or layers (1 past the image's), and levels.
    fn image_resinfo(&mut self, m: &mut Mimg) -> Res<()> {
        let lvl = self.addr_dw(m.inst, &m.addr, 0)?;
        // no coordinate (a placeholder), then the level
        let z = self.k32(0);
        m.args.extend([z, lvl]);
        let class = ImageClass { dynamic: true, ..m.class };
        let n = match (class.dim, class.arrayed) {
            (0, false) => 1,
            (0, true) | (1, false) | (3, false) => 2,
            _ => 3,
        };
        let args = std::mem::take(&mut m.args);
        let sz = self.emit(Op::Image { kind: ImageKind::Size, class, sampler: Sampler::None, flags: img::LOD }, Ty::vec(S::U32, n), args);
        let lv = self.emit(Op::Image { kind: ImageKind::Levels, class, sampler: Sampler::None, flags: 0 }, Ty::U32, vec![m.tw, z]);
        let mut all = Vec::new();
        for c in 0..3u32 {
            all.push(match c < u32::from(n) {
                true if n == 1 => sz,
                true => self.extract(sz, c),
                false => self.k32(1),
            });
        }
        all.push(lv);
        let vals: Vec<V> = (0..4).filter(|c| m.dmask >> c & 1 != 0).map(|c| all[c as usize]).collect();
        self.write_result(m.inst, m.dreg, &vals, false)
    }

    fn image_lod(&mut self, m: &mut Mimg) -> Res<()> {
        let s = self.image_sampler(m, false)?;
        let coord = self.next_vec(m, m.coords.min(3), true)?;
        m.args.push(coord);
        let args = std::mem::take(&mut m.args);
        let r = self.emit(Op::Image { kind: ImageKind::QueryLod, class: m.class, sampler: s, flags: 0 }, Ty::vec(S::F32, 2), args);
        let vals: Vec<V> = (0..2).filter(|c| m.dmask >> c & 1 != 0).map(|c| self.extract(r, c)).collect();
        self.write_result(m.inst, m.dreg, &vals, false)
    }

    /// Writes image results to consecutive VGPRs (d16: two per dword).
    pub(super) fn write_result(&mut self, inst: &Inst, dreg: u8, vals: &[V], d16: bool) -> Res<()> {
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
    /// T# word 7 (slot and flags) if word 6 vouches for it (WORD6 = WORD7 ^
    /// T_CHECK, as the host writes its views); else 0: slot 0, the null
    /// image, float and sampled.
    fn t_word7(&mut self, inst: &Inst, rsrc: &Operand) -> Res<V> {
        let w7 = self.opnd_dw(inst, rsrc, 7, false)?;
        let w7 = self.u32v(w7);
        let w6 = self.opnd_dw(inst, rsrc, 6, false)?;
        let w6 = self.u32v(w6);
        let x = self.spv(op::BITWISE_XOR, Ty::U32, &[w6, w7]);
        let k = self.k32(T_CHECK);
        let ok = self.spv(op::I_EQUAL, Ty::BOOL, &[x, k]);
        let z = self.k32(0);
        Ok(self.spv(op::SELECT, Ty::U32, &[ok, w7, z]))
    }

    /// (s, t, face) as the console's cube instructions give them (s and t
    /// in [1, 2], face = slice * 8 + face) to a point of the six-layer 2D
    /// array: (s - 1, t - 1, slice * 6 + face).
    pub(super) fn cube_layer(&mut self, c: V) -> V {
        let s = self.extract(c, 0);
        let t = self.extract(c, 1);
        let face = self.extract(c, 2);
        let one = self.kf(1.0);
        let x = self.spv(op::F_SUB, Ty::F32, &[s, one]);
        let y = self.spv(op::F_SUB, Ty::F32, &[t, one]);
        let fi = self.spv(op::CONVERT_F_TO_U, Ty::U32, &[face]);
        let (k3, k7, k6) = (self.k32(3), self.k32(7), self.k32(6));
        let slice = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[fi, k3]);
        let f = self.spv(op::BITWISE_AND, Ty::U32, &[fi, k7]);
        let base = self.spv(op::I_MUL, Ty::U32, &[slice, k6]);
        let layer = self.spv(op::I_ADD, Ty::U32, &[base, f]);
        let layer = self.spv(op::CONVERT_U_TO_F, Ty::F32, &[layer]);
        self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F32, 3), &[x, y, layer])
    }

    pub(super) fn cube_dir(&mut self, c: V) -> V {
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

    /// `image_bvh_intersect_ray` / `image_bvh64_intersect_ray`: the node
    /// the pointer names (`((T# base 40 bits) << 8) + (pointer >> 3) * 64`,
    /// 128 bytes of title memory in one range, else the zero block) tested
    /// by `Helper::BvhRay`. Operands: the pointer (one or two dwords), the
    /// ray's extent, origin, direction and inverse direction.
    pub(super) fn bvh(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Image { data, addr, rsrc, a16, d16, .. } = inst.fields.clone() else { unreachable!() };
        let Operand::Vector { reg, .. } = data else { return Err(unsupported(off, name)) };
        let wide = name.starts_with("image_bvh64");
        let mut x = Vec::new();
        for o in &addr {
            let Operand::Vector { reg: r, dwords } = *o else { return Err(unsupported(off, name)) };
            for k in 0..dwords {
                let v = self.val(inst, Reg::V(r + k))?;
                let v = self.data(v);
                x.push(self.u32v(v));
            }
        }
        if a16 || d16 || x.len() != if wide { 12 } else { 11 } {
            return Err(unsupported(off, "a ray/node test with 16-bit operands"));
        }
        if !bvh_builder() {
            // No structures: a box enters no child, anything else is missed.
            let k6 = self.k32(6);
            let pair = self.spv(op::BITWISE_AND, Ty::U32, &[x[0], k6]);
            let k4 = self.k32(4);
            let is_box = self.spv(op::I_EQUAL, Ty::BOOL, &[pair, k4]);
            let none = self.k32(BVH_NONE);
            for (k, miss) in BVH_MISS.into_iter().enumerate() {
                let miss = self.k32(miss.to_bits());
                let c = self.spv(op::SELECT, Ty::U32, &[is_box, none, miss]);
                self.def(inst, Reg::V(reg + k as u8), R::D(c));
            }
            return Ok(());
        }
        let ptr = if wide { self.u64_of(x[0], x[1]) } else { self.spv(op::U_CONVERT, Ty::U64, &[x[0]]) };
        let ray: Vec<V> = x[if wide { 2 } else { 1 }..].iter().map(|&v| self.f32v(v)).collect();
        let w: Vec<V> = (0..2).map(|k| self.opnd_dw(inst, &rsrc, k, false).map(|v| self.u32v(v))).collect::<Res<_>>()?;
        let kff = self.k32(0xFF);
        let hi = self.spv(op::BITWISE_AND, Ty::U32, &[w[1], kff]);
        let base = self.u64_of(w[0], hi);
        let k8 = self.k32(8);
        let base = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U64, &[base, k8]);
        let k3 = self.k32(3);
        let node = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U64, &[ptr, k3]);
        let k6 = self.k32(6);
        let node = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U64, &[node, k6]);
        let node = self.spv(op::I_ADD, Ty::U64, &[base, node]);
        let k128 = self.k32(128);
        let dev = self.emit(Op::Call(Helper::TranslateRange), Ty::U64, vec![node, k128]);
        let lo = self.spv(op::U_CONVERT, Ty::U32, &[ptr]);
        let k7 = self.k32(7);
        let kind = self.spv(op::BITWISE_AND, Ty::U32, &[lo, k7]);
        let k31 = self.k32(31);
        let sort = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[w[1], k31]);
        let mut args = vec![dev, kind, sort];
        args.extend(ray);
        let r = self.emit(Op::Call(Helper::BvhRay), Ty::vec(S::U32, 4), args);
        for k in 0..4u8 {
            let c = self.extract(r, u32::from(k));
            self.def(inst, Reg::V(reg + k), R::D(c));
        }
        Ok(())
    }
}

/// A child pointer that is not there (a box the ray does not enter).
pub(super) const BVH_NONE: u32 = 0xffff_ffff;

/// What a node test that hits no triangle returns: the distance's
/// numerator +infinity over the denominator 1.0, no weights.
pub(super) const BVH_MISS: [f32; 4] = [f32::INFINITY, 1.0, 0.0, 0.0];

#[cfg(test)]
thread_local! {
    /// The tests' way to translate with the structure builder on or off.
    pub(crate) static TEST_BVH_BUILDER: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether the title's ray-tracing structure builder runs:
/// `VICTORY_BVH_BUILDER=1`, the switch the host reads to run the builder's
/// dispatches (`victory-gpu` `vulkan/record.rs` `bvh_builder`; off by
/// default). While it is off nothing builds the structures the title's
/// ray-traced passes walk, and a node test would read whatever bytes lie
/// where a structure would be: not the scene, different from launch to
/// launch, in a walk the title bounds at 65,535 steps a wave. So a node
/// test then answers "no hit" and reads nothing: a box node enters no
/// child and any other node is missed, as in a scene with nothing in it;
/// every ray leaves at its first test and shadow rays find no occluder
/// (origin/main `emit/intersect.rs` `no_hit`). The switch's value is part
/// of the stage cache key (`translation_fingerprint`), so both forms are
/// kept.
fn bvh_builder() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_BVH_BUILDER.get() {
        return on;
    }
    std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v != "0")
}

/// One image instruction being lowered: its fields, the address cursor and
/// the arguments gathered so far.
pub(super) struct Mimg<'i> {
    inst: &'i Inst,
    name: &'static str,
    /// The name after `image_`.
    rest: &'static str,
    addr: Vec<Operand>,
    /// Address component cursor.
    k: usize,
    args: Vec<V>,
    flags: u16,
    class: ImageClass,
    coords: usize,
    dreg: u8,
    dmask: u8,
    d16: bool,
    /// T# word 7.
    tw: V,
    sampler: Option<Operand>,
    data: Operand,
    glc: bool,
}

/// The modifiers of a sample or gather name (`_o`, `_b`, `_c`, `_d`, ...).
struct SampleMods {
    offset: bool,
    bias: bool,
    dref: bool,
    grad: bool,
    coarse: bool,
    lz: bool,
    lod: bool,
    min_lod: bool,
}

impl SampleMods {
    fn of(rest: &str) -> Self {
        let mods = rest.trim_start_matches("sample").trim_start_matches("gather4");
        let has = |m: &str| mods.split('_').any(|t| t == m);
        SampleMods {
            offset: has("o"),
            bias: has("b"),
            dref: has("c"),
            grad: has("d") || has("cd"),
            coarse: has("cd"),
            lz: has("lz"),
            lod: has("l") || has("lz"),
            min_lod: has("cl"),
        }
    }
}

/// (SPIR-V dim, arrayed, multisampled, coordinates) of a MIMG dim.
fn dim_class(dim: u8) -> (u8, bool, bool, usize) {
    match dim {
        0 => (0, false, false, 1),
        1 => (1, false, false, 2),
        2 => (2, false, false, 3),
        3 => (3, false, false, 3),
        4 => (0, true, false, 2),
        5 => (1, true, false, 3),
        6 => (1, false, true, 2),
        _ => (1, true, true, 3),
    }
}

/// Derivative components per direction.
/// Derivative components per direction: a cube's are its faces' (s, t).
fn grad_dims(sdim: u8) -> usize {
    match sdim {
        0 => 1,
        2 => 3,
        _ => 2,
    }
}

/// WORD6 of a host T# is WORD7 ^ this (docs/vulkan-abi.md 5.2).
pub const T_CHECK: u32 = 0x7A3C_5E91;
