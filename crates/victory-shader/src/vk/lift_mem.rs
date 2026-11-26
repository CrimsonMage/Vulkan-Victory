//! Memory: scalar loads, buffers (V#), LDS, flat and global.

use super::*;

impl Lift<'_> {
    pub(super) fn translate(&mut self, addr: V) -> V {
        self.emit(Op::Call(Helper::Translate), Ty::U64, vec![addr])
    }

    pub(super) fn smem(&mut self, inst: &Inst, name: &'static str, data: Option<Operand>, base: Operand, offset: i32, soffset: Operand) -> Res<()> {
        let off = inst.machine.offset;
        if matches!(name, "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb") {
            return Ok(());
        }
        if name == "s_memtime" || name == "s_memrealtime" {
            let z = self.k32(0);
            self.sdst(inst, &data, 0, R::D(z));
            self.sdst(inst, &data, 1, R::D(z));
            return Ok(());
        }
        if !name.contains("_load_") {
            return Err(unsupported(off, name));
        }
        let Some(Operand::Scalar { code: d, dwords: n }) = data else {
            return Err(unsupported(off, name));
        };
        let so = self.opnd(inst, &soffset)?;
        let so = self.u32v(so);
        let k = self.k32(offset as u32);
        let o = self.spv(op::I_ADD, Ty::U32, &[so, k]);
        let words = if name.starts_with("s_buffer_") {
            let vs = self.vsharp(inst, &base)?;
            self.smem_words(inst, d, n, |l, at, m, fl| {
                let ka = l.k32(at * 4);
                let oo = l.spv(op::I_ADD, Ty::U32, &[o, ka]);
                l.buffer_load_raw(&vs, oo, m, fl)
            })?
        } else {
            let a = self.opnd_u64(inst, &base)?;
            let o64 = self.spv(op::U_CONVERT, Ty::U64, &[o]);
            let addr = self.spv(op::I_ADD, Ty::U64, &[a, o64]);
            self.smem_words(inst, d, n, |l, at, m, fl| {
                let ka = l.f.konst(Ty::U64, u64::from(at * 4));
                let aa = l.spv(op::I_ADD, Ty::U64, &[addr, ka]);
                let dev = l.translate(aa);
                let ty = Ty::vec(if fl { S::F32 } else { S::U32 }, m as u8);
                Ok(l.emit(Op::Load { align: 4 }, ty, vec![dev]))
            })?
        };
        for (i, w) in words.into_iter().enumerate() {
            self.def(inst, Reg::S(d + i as u8), R::D(w));
        }
        Ok(())
    }

    /// `n` dwords for s[d..] in loads of up to four (`load(at, count,
    /// floats)`), floats where the registers are read as such.
    fn smem_words(&mut self, inst: &Inst, d: u8, n: u8, mut load: impl FnMut(&mut Self, u32, u32, bool) -> Res<V>) -> Res<Vec<V>> {
        let mut out = Vec::new();
        let mut at = 0u32;
        while at < u32::from(n) {
            let m = (u32::from(n) - at).min(4);
            let fl = self.floats(inst, (at..at + m).map(|k| Reg::S(d + k as u8)));
            let v = load(self, at, m, fl)?;
            for i in 0..m {
                out.push(if m == 1 { v } else { self.extract(v, i) });
            }
            at += m;
        }
        Ok(out)
    }

    /// A V#'s parts: device base (u64), byte limit, stride.
    pub(super) fn vsharp(&mut self, inst: &Inst, rsrc: &Operand) -> Res<VSharp> {
        let w: Vec<V> = (0..4)
            .map(|k| {
                let x = self.opnd_dw(inst, rsrc, k, false)?;
                Ok(self.u32v(x))
            })
            .collect::<Res<_>>()?;
        if let Some(&vs) = self.vsharps.get(&w) {
            return Ok(vs);
        }
        // the deepest block that defines a word dominates every access
        // (prelude values and constants: the entry block; a word made inside
        // a guard, or of unknown birth: no sharing)
        let homes: Vec<usize> = w
            .iter()
            .filter(|&&v| self.f.as_const(v).is_none() && !self.f.prelude.contains(&v))
            .map(|v| self.born.get(v).copied().unwrap_or(usize::MAX))
            .collect();
        if homes.contains(&usize::MAX) {
            return Ok(self.vsharp_of(&w));
        }
        let home = homes.into_iter().max_by_key(|&b| self.depth[b]).unwrap_or(0);
        let home = if self.done[home].is_some() { home } else { self.cur_block };
        let vs = if home == self.cur_block {
            // before an open guard, with the block's other scalar values
            let g = std::mem::replace(&mut self.guarded, false);
            let vs = self.vsharp_of(&w);
            self.guarded = g;
            vs
        } else {
            let saved = std::mem::take(&mut self.cur);
            let (g, nc, cb) = (self.guarded, self.no_cse, self.cur_block);
            self.cur = std::mem::take(&mut self.done[home].as_mut().unwrap().body);
            (self.guarded, self.no_cse, self.cur_block) = (false, true, home);
            let vs = self.vsharp_of(&w);
            self.done[home].as_mut().unwrap().body = std::mem::replace(&mut self.cur, saved);
            (self.guarded, self.no_cse, self.cur_block) = (g, nc, cb);
            vs
        };
        self.vsharps.insert(w, vs);
        Ok(vs)
    }

    pub(super) fn vsharp_of(&mut self, w: &[V]) -> VSharp {
        // CSE across instructions in a block: the nodes repeat.
        let m16 = self.k32(0xFFFF);
        let hi = self.spv(op::BITWISE_AND, Ty::U32, &[w[1], m16]);
        let base = self.u64_of(w[0], hi);
        let k16 = self.k32(16);
        let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[w[1], k16]);
        let k = self.k32(0x3FFF);
        let stride = self.spv(op::BITWISE_AND, Ty::U32, &[s, k]);
        let one = self.k32(1);
        let st1 = self.gl(glsl::U_MAX, Ty::U32, &[stride, one]);
        let nr64 = self.spv(op::U_CONVERT, Ty::U64, &[w[2]]);
        let st64 = self.spv(op::U_CONVERT, Ty::U64, &[st1]);
        let bytes64 = self.spv(op::I_MUL, Ty::U64, &[nr64, st64]);
        let max = self.f.konst(Ty::U64, 0xFFFF_FFFF);
        let bytes64 = self.gl(glsl::U_MIN, Ty::U64, &[bytes64, max]);
        let bytes = self.spv(op::U_CONVERT, Ty::U32, &[bytes64]);
        // the device base, or the zero block when the bytes are in no range
        let dev = self.emit(Op::Call(Helper::TranslateRange), Ty::U64, vec![base, bytes]);
        let zero = self.fallback(false);
        let mapped = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[dev, zero]);
        let z = self.k32(0);
        let limit = self.spv(op::SELECT, Ty::U32, &[mapped, bytes, z]);
        let shape = self.typed_shape(w[3]);
        VSharp { shape, dev, limit, stride }
    }

    /// The shape (emit_typed.rs) of a V#'s FORMAT, once per V#.
    fn typed_shape(&mut self, w3: V) -> V {
        let k12 = self.k32(12);
        let f = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[w3, k12]);
        self.emit(Op::Call(Helper::TypedShape), Ty::U32, vec![f])
    }

    /// Whether a V# holds `kb` bytes at byte offset `o`.
    fn in_bounds(&mut self, vs: &VSharp, o: V, kb: V) -> V {
        let room = self.spv(op::U_GREATER_THAN_EQUAL, Ty::BOOL, &[vs.limit, kb]);
        let lim = self.spv(op::I_SUB, Ty::U32, &[vs.limit, kb]);
        let le = self.spv(op::U_LESS_THAN_EQUAL, Ty::BOOL, &[o, lim]);
        self.spv(op::LOGICAL_AND, Ty::BOOL, &[room, le])
    }

    /// The device address for `kb` bytes at byte offset `o` of a V#; out
    /// of bounds the zero (loads) or sink (stores) block.
    pub(super) fn access_addr(&mut self, vs: &VSharp, o: V, kb: V, store: bool) -> V {
        let ok = self.in_bounds(vs, o, kb);
        let o64 = self.spv(op::U_CONVERT, Ty::U64, &[o]);
        let a = self.spv(op::I_ADD, Ty::U64, &[vs.dev, o64]);
        let fallback = self.fallback(store);
        self.spv(op::SELECT, Ty::U64, &[ok, a, fallback])
    }

    /// A store's (or atomic's) address for `bytes` at byte offset `o`.
    pub(super) fn buffer_addr(&mut self, vs: &VSharp, o: V, bytes: u32) -> V {
        let kb = self.k32(bytes);
        self.access_addr(vs, o, kb, true)
    }

    pub(super) fn fallback(&mut self, store: bool) -> V {
        // the memory map's zero and sink blocks
        self.f.global(Op::MapField(u32::from(store)), Ty::U64)
    }

    /// d16 results: values (f32 bits) as f16, two per dword.
    pub(super) fn pack_d16(&mut self, vals: &[V]) -> Vec<V> {
        vals.chunks(2)
            .map(|pair| {
                let mut h = Vec::new();
                for &x in pair {
                    let f = self.f32v(x);
                    let hf = self.spv(op::F_CONVERT, Ty::F16, &[f]);
                    h.push(hf);
                }
                if h.len() == 1 {
                    h.push(self.f.konst(Ty::F16, 0));
                }
                let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F16, 2), &h);
                self.spv(op::BITCAST, Ty::U32, &[v2])
            })
            .collect()
    }

    pub(super) fn buffer_load_raw(&mut self, vs: &VSharp, o: V, n: u32, float: bool) -> Res<V> {
        let kb = self.k32(4 * n);
        let a = self.access_addr(vs, o, kb, false);
        Ok(self.emit(Op::Load { align: 4 }, Ty::vec(if float { S::F32 } else { S::U32 }, n as u8), vec![a]))
    }

    /// Whether the values `inst` writes to `regs` are read as floats.
    pub(super) fn floats(&self, inst: &Inst, regs: impl Iterator<Item = Reg>) -> bool {
        let mut any = false;
        for r in regs {
            match inst.write(r).and_then(|v| self.floaty.get(&v)) {
                Some(&x) if x > 0 => any = true,
                _ => return false,
            }
        }
        any
    }

    pub(super) fn buffer(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Buffer { data, rsrc, format, lds, tfe, .. } = inst.fields.clone() else { unreachable!() };
        if lds || tfe {
            return Err(unsupported(off, "a buffer access to LDS or with tfe"));
        }
        let vs = self.vsharp(inst, &rsrc)?;
        let o = self.buffer_offset(inst, &vs)?;
        let Some(src @ Operand::Vector { reg, dwords }) = data else {
            return Err(unsupported(off, name));
        };
        if name.contains("_atomic_") {
            return self.buffer_atomic(inst, name, &vs, o, &src, reg);
        }
        let load = name.contains("_load_");
        let (size, kind) = self.buffer_kind(name, format, load, u32::from(dwords), off)?;
        if let Kind::VFormat(ch, d16) = kind {
            return self.buffer_vformat(inst, &vs, o, reg, ch, d16);
        }
        if load {
            let kind = match kind {
                Kind::Dwords(n) if self.floats(inst, (0..n).map(|k| Reg::V(reg + k as u8))) => Kind::Floats(n),
                k => k,
            };
            let kb = self.k32(size);
            let a = self.access_addr(&vs, o, kb, false);
            let vals = self.load_kind(a, kind)?;
            for (i, v) in vals.into_iter().enumerate() {
                self.def(inst, Reg::V(reg + i as u8), R::D(v));
            }
        } else {
            let a = self.buffer_addr(&vs, o, size);
            let vals: Vec<V> = (0..dwords).map(|i| self.opnd_dw(inst, &src, i, false)).collect::<Res<_>>()?;
            self.store_kind(a, kind, &vals)?;
        }
        Ok(())
    }

    /// A buffer access's byte offset: soffset + offset, plus index x stride
    /// (idxen) and the vector offset (offen).
    fn buffer_offset(&mut self, inst: &Inst, vs: &VSharp) -> Res<V> {
        let Fields::Buffer { addr, soffset, offset, offen, idxen, .. } = inst.fields.clone() else { unreachable!() };
        let so = self.opnd(inst, &soffset)?;
        let so = self.u32v(so);
        let ko = self.k32(u32::from(offset));
        let mut o = self.spv(op::I_ADD, Ty::U32, &[so, ko]);
        let mut k = 0u8;
        if idxen {
            let idx = self.opnd_dw(inst, &addr, 0, false)?;
            let idx = self.u32v(idx);
            let b = self.spv(op::I_MUL, Ty::U32, &[idx, vs.stride]);
            o = self.spv(op::I_ADD, Ty::U32, &[o, b]);
            k = 1;
        }
        if offen {
            let vo = self.opnd_dw(inst, &addr, k, false)?;
            let vo = self.u32v(vo);
            o = self.spv(op::I_ADD, Ty::U32, &[o, vo]);
        }
        Ok(o)
    }

    /// A buffer atomic; the old value returns with glc.
    fn buffer_atomic(&mut self, inst: &Inst, name: &'static str, vs: &VSharp, o: V, src: &Operand, reg: u8) -> Res<()> {
        let off = inst.machine.offset;
        let glc = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
        let opc = atomic_op(name).ok_or(unsupported(off, name))?;
        let a = self.buffer_addr(vs, o, 4);
        let x = self.opnd_dw(inst, src, 0, false)?;
        let x = self.u32v(x);
        let mut args = vec![a, x];
        if name.contains("cmpswap") {
            let c = self.opnd_dw(inst, src, 1, false)?;
            let c = self.u32v(c);
            args.push(c);
        }
        let r = self.emit(Op::Atomic(opc), Ty::U32, args);
        if glc {
            self.def(inst, Reg::V(reg), R::D(r));
        }
        Ok(())
    }

    /// Element size and conversion of a buffer access.
    fn buffer_kind(&mut self, name: &str, format: Option<u8>, load: bool, n: u32, off: usize) -> Res<(u32, Kind)> {
        if name.contains("_format_") {
            return self.buffer_format_kind(name, format, load, off);
        }
        let ends = |a: &str, b: &str, c: &str| name.ends_with(a) || name.ends_with(b) || name.ends_with(c);
        Ok(if ends("_ubyte", "_sbyte", "_byte") {
            (1, Kind::Byte(name.ends_with("_sbyte")))
        } else if ends("_ushort", "_sshort", "_short") {
            (2, Kind::Short(name.ends_with("_sshort")))
        } else if name.contains("_d16") {
            return Err(unsupported(off, "a d16 buffer access"));
        } else {
            (4 * n, Kind::Dwords(n))
        })
    }

    /// A formatted access: the instruction's format (tbuffer), or the V#'s.
    fn buffer_format_kind(&mut self, name: &str, format: Option<u8>, load: bool, off: usize) -> Res<(u32, Kind)> {
        let ch = name.rsplit('_').next().unwrap().len() as u32; // x, xy, xyz, xyzw
        let d16 = name.contains("_d16_");
        Ok(match format {
            Some(f) => {
                let (bytes, kind) = tformat(f).ok_or(unsupported(off, "a typed buffer format"))?;
                (bytes, Kind::T(kind, ch, d16))
            }
            None if load => (0, Kind::VFormat(ch, d16)),
            None => {
                // stores through a V# format: 32-bit channels
                self.approx("a typed buffer store written as 32-bit channels");
                (4 * ch, if d16 { Kind::Halves(ch) } else { Kind::Dwords(ch) })
            }
        })
    }

    /// A load in the V#'s format, decoded by a helper; bounds by the stride.
    fn buffer_vformat(&mut self, inst: &Inst, vs: &VSharp, o: V, reg: u8, ch: u32, d16: bool) -> Res<()> {
        let four = self.k32(4);
        let bytes = self.gl(glsl::U_MAX, Ty::U32, &[vs.stride, four]);
        let a = self.access_addr(vs, o, bytes, false);
        let r = self.emit(Op::Call(Helper::Typed), Ty::vec(S::U32, 4), vec![a, vs.shape]);
        let vals: Vec<V> = (0..ch).map(|i| self.extract(r, i)).collect();
        let vals = if d16 { self.pack_d16(&vals) } else { vals };
        for (i, v) in vals.into_iter().enumerate() {
            self.def(inst, Reg::V(reg + i as u8), R::D(v));
        }
        Ok(())
    }

    pub(super) fn load_kind(&mut self, a: V, kind: Kind) -> Res<Vec<V>> {
        Ok(match kind {
            Kind::Dwords(n) | Kind::Floats(n) => {
                let s = if matches!(kind, Kind::Floats(_)) { S::F32 } else { S::U32 };
                let v = self.emit(Op::Load { align: 4 }, Ty::vec(s, n as u8), vec![a]);
                if n == 1 { vec![v] } else { (0..n).map(|i| self.extract(v, i)).collect() }
            }
            Kind::Byte(s) => {
                let v = self.emit(Op::Load { align: 1 }, Ty::U8, vec![a]);
                let o = if s { op::S_CONVERT } else { op::U_CONVERT };
                vec![self.spv(o, Ty::U32, &[v])]
            }
            Kind::Short(s) => {
                let v = self.emit(Op::Load { align: 2 }, Ty::U16, vec![a]);
                let o = if s { op::S_CONVERT } else { op::U_CONVERT };
                vec![self.spv(o, Ty::U32, &[v])]
            }
            Kind::T(t, ch, d16) => {
                let vals = self.typed_load(a, t, ch);
                if d16 { self.pack_d16(&vals) } else { vals }
            }
            Kind::VFormat(..) | Kind::Halves(_) => return Err(unsupported(0, "a typed buffer access")),
        })
    }

    pub(super) fn store_kind(&mut self, a: V, kind: Kind, vals: &[V]) -> Res<()> {
        match kind {
            Kind::Dwords(n) => {
                // floats stay floats when every dword is one
                let float = vals.iter().all(|&x| self.f.ty(x) == Ty::F32);
                let vals: Vec<V> = vals.iter().map(|&x| if float { x } else { self.u32v(x) }).collect();
                let v = if n == 1 {
                    vals[0]
                } else {
                    self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(if float { S::F32 } else { S::U32 }, n as u8), &vals)
                };
                self.emit(Op::Store { align: 4 }, Ty::VOID, vec![a, v]);
            }
            Kind::Byte(_) => {
                let x = self.u32v(vals[0]);
                let v = self.spv(op::U_CONVERT, Ty::U8, &[x]);
                self.emit(Op::Store { align: 1 }, Ty::VOID, vec![a, v]);
            }
            Kind::Short(_) => {
                let x = self.u32v(vals[0]);
                let v = self.spv(op::U_CONVERT, Ty::U16, &[x]);
                self.emit(Op::Store { align: 2 }, Ty::VOID, vec![a, v]);
            }
            Kind::Halves(n) => {
                // two f16 per data dword, written as 32-bit floats
                let mut out = Vec::new();
                for k in 0..n {
                    let w = self.u32v(vals[(k / 2) as usize]);
                    let f = self.half_f32(w, k % 2 == 1);
                    out.push(self.u32v(f));
                }
                let v = if n == 1 { out[0] } else { self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, n as u8), &out) };
                self.emit(Op::Store { align: 4 }, Ty::VOID, vec![a, v]);
            }
            Kind::T(..) | Kind::Floats(_) | Kind::VFormat(..) => return Err(unsupported(0, "a typed buffer store")),
        }
        Ok(())
    }

    /// A typed element read from its bytes (the format in the instruction).
    pub(super) fn typed_load(&mut self, a: V, t: TFormat, ch: u32) -> Vec<V> {
        if t.bits == 0 {
            return self.packed_load(a, t, ch);
        }
        let comps = u32::from(t.comps);
        let s = match t.bits {
            32 => S::U32,
            16 => S::U16,
            _ => S::U8,
        };
        let v = self.emit(Op::Load { align: u32::from(t.bits / 8).max(1) }, Ty::vec(s, t.comps), vec![a]);
        let signed = matches!(t.num, Num::Sint | Num::Snorm);
        let mut out = Vec::new();
        for c in 0..ch {
            let x = if c < comps {
                let e = if comps == 1 { v } else { self.extract(v, c) };
                let r = match t.bits {
                    32 => e,
                    _ => self.spv(if signed { op::S_CONVERT } else { op::U_CONVERT }, Ty::U32, &[e]),
                };
                self.typed_value(r, t.num, u32::from(t.bits))
            } else if c == 3 {
                // missing channels: 0, alpha 1
                self.k32(if matches!(t.num, Num::Uint | Num::Sint) { 1 } else { 1.0f32.to_bits() })
            } else {
                self.k32(0)
            };
            out.push(x);
        }
        out
    }

    /// A channel's raw bits (widened to 32) as its value's bits.
    fn typed_value(&mut self, r: V, num: Num, bits: u32) -> V {
        match num {
            Num::Uint | Num::Sint => r,
            Num::Float if bits == 32 => r,
            Num::Float => {
                // f16 bits to f32
                let h = self.spv(op::U_CONVERT, Ty::U16, &[r]);
                let hf = self.spv(op::BITCAST, Ty::F16, &[h]);
                let f = self.spv(op::F_CONVERT, Ty::F32, &[hf]);
                self.u32v(f)
            }
            Num::Unorm => {
                let f = self.spv(op::CONVERT_U_TO_F, Ty::F32, &[r]);
                let k = self.kf(1.0 / ((1u64 << bits) - 1) as f32);
                let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                self.u32v(m)
            }
            Num::Snorm => {
                let f = self.spv(op::CONVERT_S_TO_F, Ty::F32, &[r]);
                let k = self.kf(1.0 / ((1u64 << (bits - 1)) - 1) as f32);
                let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                let lo = self.kf(-1.0);
                let c = self.gl(glsl::N_MAX, Ty::F32, &[m, lo]);
                self.u32v(c)
            }
        }
    }

    /// A 2_10_10_10 element (x in the low bits): one dword.
    pub(super) fn packed_load(&mut self, a: V, t: TFormat, ch: u32) -> Vec<V> {
        let w = self.emit(Op::Load { align: 4 }, Ty::U32, vec![a]);
        let widths = [10u32, 10, 10, 2];
        let mut out = Vec::new();
        let mut at = 0u32;
        for (c, &bits) in widths.iter().enumerate() {
            let (o, wv) = (self.k32(at), self.k32(bits));
            at += bits;
            if c as u32 >= ch {
                continue;
            }
            let signed = matches!(t.num, Num::Snorm | Num::Sint);
            let opc = if signed { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
            let r = self.spv(opc, Ty::U32, &[w, o, wv]);
            let x = match t.num {
                Num::Uint | Num::Sint => r,
                Num::Unorm => {
                    let f = self.spv(op::CONVERT_U_TO_F, Ty::F32, &[r]);
                    let k = self.kf(1.0 / ((1u64 << bits) - 1) as f32);
                    let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                    self.u32v(m)
                }
                _ => {
                    let f = self.spv(op::CONVERT_S_TO_F, Ty::F32, &[r]);
                    let k = self.kf(1.0 / ((1u64 << (bits - 1)) - 1) as f32);
                    let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                    let lo = self.kf(-1.0);
                    let c = self.gl(glsl::N_MAX, Ty::F32, &[m, lo]);
                    self.u32v(c)
                }
            };
            out.push(x);
        }
        out
    }

    pub(super) fn lds_index(&mut self, inst: &Inst, addr: &Option<Operand>, off: u32) -> Res<V> {
        let a = match addr {
            Some(o) => {
                let x = self.opnd(inst, o)?;
                self.u32v(x)
            }
            None => self.k32(0),
        };
        let k = self.k32(off);
        let b = self.spv(op::I_ADD, Ty::U32, &[a, k]);
        let two = self.k32(2);
        Ok(self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, two]))
    }

    pub(super) fn ds(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Ds { dst, addr, data, offset0, offset1, gds } = inst.fields.clone() else { unreachable!() };
        if gds {
            return Err(unsupported(off, "GDS"));
        }
        let o16 = u32::from(offset0) | u32::from(offset1) << 8;
        let d = match dst {
            Some(Operand::Vector { reg, .. }) => Some(reg),
            _ => None,
        };
        let need_d = || d.ok_or(unsupported(off, name));
        // the two-address forms: offsets in elements (of 64 with st64)
        let pair = |w: u32| {
            let st = if name.contains("st64") { 64 } else { 1 };
            [offset0, offset1].map(|o| u32::from(o) * 4 * w * st)
        };
        match name {
            "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" => {
                let n = ds_dwords(name);
                let i = self.lds_index(inst, &addr, o16)?;
                self.lds_read(inst, i, n, need_d()?)?;
            }
            "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
                let w = if name.ends_with("b64") { 2 } else { 1 };
                let dd = need_d()?;
                for (j, o) in pair(w).into_iter().enumerate() {
                    let i = self.lds_index(inst, &addr, o)?;
                    self.lds_read(inst, i, w, dd + (j as u32 * w) as u8)?;
                }
            }
            "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" => {
                let i = self.lds_index(inst, &addr, o16)?;
                let Some(Operand::Vector { dwords, .. }) = data.first().copied() else {
                    return Err(unsupported(off, name));
                };
                self.lds_write(inst, i, &data[0], u32::from(dwords))?;
            }
            "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
                let w = if name.ends_with("b64") { 2 } else { 1 };
                for (j, o) in pair(w).into_iter().enumerate() {
                    let i = self.lds_index(inst, &addr, o)?;
                    self.lds_write(inst, i, &data[j], w)?;
                }
            }
            "ds_read_u16" | "ds_read_u8" | "ds_read_i16" | "ds_read_i8" => {
                let v = self.lds_read_small(inst, &addr, o16, name)?;
                self.def(inst, Reg::V(need_d()?), R::D(v));
            }
            "ds_write_b16" | "ds_write_b8" | "ds_write_b16_d16_hi" | "ds_write_b8_d16_hi" => {
                self.lds_write_small(inst, &addr, o16, &data[0], name)?
            }
            _ if name.starts_with("ds_") && atomic_op(name).is_some() => {
                let i = self.lds_index(inst, &addr, o16)?;
                let r = self.lds_atomic(inst, i, &data, name)?;
                if let Some(d) = d {
                    self.def(inst, Reg::V(d), R::D(r));
                }
            }
            _ => return Err(unsupported(off, name)),
        }
        Ok(())
    }

    /// `n` dwords from LDS dword `i` into v[d..].
    fn lds_read(&mut self, inst: &Inst, i: V, n: u32, d: u8) -> Res<()> {
        for k in 0..n {
            let kk = self.k32(k);
            let ii = self.spv(op::I_ADD, Ty::U32, &[i, kk]);
            let v = self.emit(Op::LdsLoad, Ty::U32, vec![ii]);
            self.def(inst, Reg::V(d + k as u8), R::D(v));
        }
        Ok(())
    }

    /// `n` dwords of `src` to LDS dword `i` on.
    fn lds_write(&mut self, inst: &Inst, i: V, src: &Operand, n: u32) -> Res<()> {
        for k in 0..n {
            let x = self.opnd_dw(inst, src, k as u8, false)?;
            let x = self.u32v(x);
            let kk = self.k32(k);
            let ii = self.spv(op::I_ADD, Ty::U32, &[i, kk]);
            self.emit(Op::LdsStore, Ty::VOID, vec![ii, x]);
        }
        Ok(())
    }

    /// A byte address's dword index and bit shift.
    fn lds_byte(&mut self, inst: &Inst, addr: &Option<Operand>, o16: u32) -> Res<(V, V)> {
        let a = match addr {
            Some(o) => {
                let x = self.opnd(inst, o)?;
                self.u32v(x)
            }
            None => self.k32(0),
        };
        let k = self.k32(o16);
        let b = self.spv(op::I_ADD, Ty::U32, &[a, k]);
        let (two, three) = (self.k32(2), self.k32(3));
        let i = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, two]);
        let bo = self.spv(op::BITWISE_AND, Ty::U32, &[b, three]);
        let sh = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[bo, three]);
        Ok((i, sh))
    }

    /// A byte or half read, zero or sign extended.
    fn lds_read_small(&mut self, inst: &Inst, addr: &Option<Operand>, o16: u32, name: &str) -> Res<V> {
        let (i, sh) = self.lds_byte(inst, addr, o16)?;
        let w = self.emit(Op::LdsLoad, Ty::U32, vec![i]);
        let width = self.k32(if name.ends_with("16") { 16 } else { 8 });
        let opc = if name.contains("_i") { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
        Ok(self.spv(opc, Ty::U32, &[w, sh, width]))
    }

    /// A half or byte of a dword: clear it, then set it (two atomics, so
    /// lanes writing the other bytes are kept).
    fn lds_write_small(&mut self, inst: &Inst, addr: &Option<Operand>, o16: u32, src: &Operand, name: &str) -> Res<()> {
        let (i, sh) = self.lds_byte(inst, addr, o16)?;
        let m = self.k32(if name.starts_with("ds_write_b16") { 0xFFFF } else { 0xFF });
        let x = self.opnd_dw(inst, src, 0, false)?;
        let mut x = self.u32v(x);
        if name.ends_with("_d16_hi") {
            // the data's high 16 bits
            let k16 = self.k32(16);
            x = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[x, k16]);
        }
        let x = self.spv(op::BITWISE_AND, Ty::U32, &[x, m]);
        let xs = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[x, sh]);
        let ms = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[m, sh]);
        let keep = self.spv(op::NOT, Ty::U32, &[ms]);
        self.emit(Op::LdsAtomic(op::ATOMIC_AND), Ty::U32, vec![i, keep]);
        self.emit(Op::LdsAtomic(op::ATOMIC_OR), Ty::U32, vec![i, xs]);
        Ok(())
    }

    /// An LDS atomic; ds_cmpst takes data0 = compare, data1 = new.
    fn lds_atomic(&mut self, inst: &Inst, i: V, data: &[Operand], name: &'static str) -> Res<V> {
        let opc = atomic_op(name).unwrap();
        let x = self.opnd_dw(inst, &data[0], 0, false)?;
        let x = self.u32v(x);
        let args = if name.contains("cmpst") {
            let y = self.opnd_dw(inst, &data[1], 0, false)?;
            let y = self.u32v(y);
            vec![i, y, x]
        } else {
            vec![i, x]
        };
        Ok(self.emit(Op::LdsAtomic(opc), Ty::U32, args))
    }

    pub(super) fn ds_cross(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Ds { dst, addr, data, offset0, offset1, .. } = inst.fields.clone() else { unreachable!() };
        let Some(Operand::Vector { reg: d, .. }) = dst else { return Err(unsupported(off, name)) };
        let a = self.active(inst)?;
        let lane = self.lane();
        let (src, from) = match name {
            "ds_swizzle_b32" => {
                let x = self.opnd(inst, addr.as_ref().or(data.first()).ok_or(unsupported(off, name))?)?;
                let x = self.u32v(x);
                let pat = u32::from(offset0) | u32::from(offset1) << 8;
                let from = if pat & 0x8000 != 0 {
                    // quad permute
                    let k3 = self.k32(3);
                    let base = self.k32(!3);
                    let qb = self.spv(op::BITWISE_AND, Ty::U32, &[lane, base]);
                    let q = self.spv(op::BITWISE_AND, Ty::U32, &[lane, k3]);
                    // selector by q: (pat >> (2q)) & 3
                    let one = self.k32(1);
                    let q2 = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[q, one]);
                    let kp = self.k32(pat & 0xFF);
                    let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[kp, q2]);
                    let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k3]);
                    self.spv(op::BITWISE_OR, Ty::U32, &[qb, s])
                } else {
                    // bitmask mode within groups of 32
                    let and = pat & 0x1F;
                    let or = pat >> 5 & 0x1F;
                    let xor = pat >> 10 & 0x1F;
                    let ka = self.k32(and | !0x1F);
                    let ko = self.k32(or);
                    let kx = self.k32(xor);
                    let t = self.spv(op::BITWISE_AND, Ty::U32, &[lane, ka]);
                    let t = self.spv(op::BITWISE_OR, Ty::U32, &[t, ko]);
                    self.spv(op::BITWISE_XOR, Ty::U32, &[t, kx])
                };
                (x, from)
            }
            _ => {
                // ds_bpermute_b32: dst = src[addr / 4]
                let x = self.opnd(inst, &data[0])?;
                let x = self.u32v(x);
                let ad = self.opnd(inst, addr.as_ref().unwrap())?;
                let ad = self.u32v(ad);
                let o = self.k32(u32::from(offset0) | u32::from(offset1) << 8);
                let ad = self.spv(op::I_ADD, Ty::U32, &[ad, o]);
                let two = self.k32(2);
                let l = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[ad, two]);
                let k31 = self.k32(31);
                if name == "ds_permute_b32" {
                    return Err(unsupported(off, name));
                }
                (x, self.spv(op::BITWISE_AND, Ty::U32, &[l, k31]))
            }
        };
        let k31 = self.k32(31);
        let from = self.spv(op::BITWISE_AND, Ty::U32, &[from, k31]);
        let v = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![src, from]);
        // an inactive source lane reads 0
        let src_active = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::BOOL, vec![a, from]);
        let z = self.k32(0);
        let v = self.spv(op::SELECT, Ty::U32, &[src_active, v, z]);
        let old = self.val(inst, Reg::V(d))?;
        let old = self.data(old);
        let old = self.u32v(old);
        let x = self.spv(op::SELECT, Ty::U32, &[a, v, old]);
        self.def(inst, Reg::V(d), R::D(x));
        Ok(())
    }

    pub(super) fn flat(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Flat { dst, addr, data, saddr, offset, lds, .. } = inst.fields.clone() else { unreachable!() };
        if lds || name.starts_with("scratch_") {
            return Err(unsupported(off, name));
        }
        let mut a = match (saddr, addr) {
            (Some(s @ Operand::Scalar { code, .. }), Some(v)) if code != crate::gfx10::NULL => {
                let base = self.opnd_u64(inst, &s)?;
                let vo = self.opnd(inst, &v)?;
                let vo = self.u32v(vo);
                let v64 = self.spv(op::U_CONVERT, Ty::U64, &[vo]);
                self.spv(op::I_ADD, Ty::U64, &[base, v64])
            }
            (_, Some(v)) => self.opnd_u64(inst, &v)?,
            _ => return Err(unsupported(off, name)),
        };
        let ko = self.f.konst(Ty::U64, i64::from(offset) as u64);
        a = self.spv(op::I_ADD, Ty::U64, &[a, ko]);
        let dev = self.translate(a);
        let kind = if name.ends_with("_ubyte") {
            Kind::Byte(false)
        } else if name.ends_with("_sbyte") {
            Kind::Byte(true)
        } else if name.ends_with("_ushort") {
            Kind::Short(false)
        } else if name.ends_with("_sshort") {
            Kind::Short(true)
        } else if name.contains("_atomic_") {
            let opc = atomic_op(name).ok_or(unsupported(off, name))?;
            let x = self.opnd_dw(inst, &data.unwrap(), 0, false)?;
            let x = self.u32v(x);
            let r = self.emit(Op::Atomic(opc), Ty::U32, vec![dev, x]);
            if let Some(Operand::Vector { reg, .. }) = dst {
                self.def(inst, Reg::V(reg), R::D(r));
            }
            return Ok(());
        } else {
            let n = match name.rsplit('_').next().unwrap() {
                "dword" => 1,
                "dwordx2" => 2,
                "dwordx3" => 3,
                "dwordx4" => 4,
                _ => return Err(unsupported(off, name)),
            };
            Kind::Dwords(n)
        };
        if name.contains("_load_") {
            let Some(Operand::Vector { reg, .. }) = dst else { return Err(unsupported(off, name)) };
            let vals = self.load_kind(dev, kind)?;
            for (i, v) in vals.into_iter().enumerate() {
                self.def(inst, Reg::V(reg + i as u8), R::D(v));
            }
        } else {
            let Some(d @ Operand::Vector { dwords, .. }) = data else { return Err(unsupported(off, name)) };
            let vals: Vec<V> = (0..dwords).map(|i| self.opnd_dw(inst, &d, i, false)).collect::<Res<_>>()?;
            self.store_kind(dev, kind, &vals)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct VSharp {
    /// The FORMAT's shape for typed loads (emit_typed.rs).
    pub(super) shape: V,
    pub(super) dev: V,
    pub(super) limit: V,
    pub(super) stride: V,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Num {
    Unorm,
    Snorm,
    Uint,
    Sint,
    Float,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TFormat {
    pub(super) comps: u8,
    /// Bits per component; 0: a packed 2_10_10_10 dword.
    pub(super) bits: u8,
    pub(super) num: Num,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Kind {
    /// A typed load through the V#'s format (channels, d16).
    VFormat(u32, bool),
    /// d16 typed store data written as 32-bit floats.
    Halves(u32),
    Dwords(u32),
    Floats(u32),
    Byte(bool),
    Short(bool),
    T(TFormat, u32, bool),
}

/// GFX10 buffer format codes (the unified FORMAT field of MTBUF / the V#):
/// element bytes and decoding.
pub(super) fn tformat(f: u8) -> Option<(u32, TFormat)> {
    use Num::*;
    let t = |comps, bits, num| Some((u32::from(comps) * u32::from(bits) / 8, TFormat { comps, bits, num }));
    let packed = |num| Some((4, TFormat { comps: 4, bits: 0, num }));
    match f {
        50 => packed(Unorm),
        51 => packed(Snorm),
        54 => packed(Uint),
        55 => packed(Sint),
        1 => t(1, 8, Unorm),
        2 => t(1, 8, Snorm),
        5 => t(1, 8, Uint),
        6 => t(1, 8, Sint),
        7 => t(1, 16, Unorm),
        8 => t(1, 16, Snorm),
        11 => t(1, 16, Uint),
        12 => t(1, 16, Sint),
        13 => t(1, 16, Float),
        14 => t(2, 8, Unorm),
        15 => t(2, 8, Snorm),
        18 => t(2, 8, Uint),
        19 => t(2, 8, Sint),
        20 => t(1, 32, Uint),
        21 => t(1, 32, Sint),
        22 => t(1, 32, Float),
        23 => t(2, 16, Unorm),
        24 => t(2, 16, Snorm),
        27 => t(2, 16, Uint),
        28 => t(2, 16, Sint),
        29 => t(2, 16, Float),
        56 => t(4, 8, Unorm),
        57 => t(4, 8, Snorm),
        60 => t(4, 8, Uint),
        61 => t(4, 8, Sint),
        62 => t(2, 32, Uint),
        63 => t(2, 32, Sint),
        64 => t(2, 32, Float),
        65 => t(4, 16, Unorm),
        66 => t(4, 16, Snorm),
        69 => t(4, 16, Uint),
        70 => t(4, 16, Sint),
        71 => t(4, 16, Float),
        72 => t(3, 32, Uint),
        73 => t(3, 32, Sint),
        74 => t(3, 32, Float),
        75 => t(4, 32, Uint),
        76 => t(4, 32, Sint),
        77 => t(4, 32, Float),
        _ => None,
    }
}

/// Dwords of a ds_read_b*/ds_write_b*.
fn ds_dwords(name: &str) -> u32 {
    match &name[name.len() - 3..] {
        "b32" => 1,
        "b64" => 2,
        "b96" => 3,
        _ => 4,
    }
}
