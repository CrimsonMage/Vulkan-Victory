//! The longer VALU instructions, one function each.

use super::*;

impl Lift<'_> {
    #[allow(unused_variables)]
    pub(super) fn op_cndmask_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let m = self.opnd_mask(inst, &v.src[2])?;
        // floats stay floats (constants follow the other side)
        let raw: Vec<V> = (0..2).map(|i| self.opnd(inst, &v.src[i])).collect::<Res<_>>()?;
        let is_f = |l: &Self, x: V| l.f.ty(x) == Ty::F32;
        let is_c = |l: &Self, x: V| l.f.as_const(x).is_some();
        let float = v.abs != 0
            || v.neg != 0
            || (v.sdwa.is_none()
                && raw.iter().any(|&x| is_f(self, x))
                && raw.iter().all(|&x| is_f(self, x) || is_c(self, x)));
        let (a, b) = if float {
            (self.fsrc(inst, v, 0)?, self.fsrc(inst, v, 1)?)
        } else {
            (self.usrc(inst, v, 0)?, self.usrc(inst, v, 1)?)
        };
        let t = self.f.ty(a);
        let x = self.spv(op::SELECT, t, &[m, b, a]);
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_fmac_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.fsrc(inst, v, 0)?;
        let b = self.fsrc(inst, v, 1)?;
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        let c = self.val(inst, Reg::V(reg))?;
        let c = self.data(c);
        let c = self.f32v(c);
        let x = if name.contains("legacy") {
            let p = self.mul_legacy(a, b);
            self.spv(op::F_ADD, Ty::F32, &[p, c])
        } else {
            self.gl(glsl::FMA, Ty::F32, &[a, b, c])
        };
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_med3_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.fsrc(inst, v, 0)?;
        let b = self.fsrc(inst, v, 1)?;
        let c = self.fsrc(inst, v, 2)?;
        // clamp idiom: med3(x, 0, 1)
        let x = if self.f.as_const(b).is_some() && self.f.as_const(c).is_some() {
            self.gl(glsl::N_CLAMP, Ty::F32, &[a, b, c])
        } else {
            let lo = self.gl(glsl::N_MIN, Ty::F32, &[a, b]);
            let hi = self.gl(glsl::N_MAX, Ty::F32, &[a, b]);
            let m = self.gl(glsl::N_MIN, Ty::F32, &[hi, c]);
            self.gl(glsl::N_MAX, Ty::F32, &[lo, m])
        };
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_fma_mix_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        // op_sel_hi[i]: src i is f16 (op_sel[i] its high half), else f32
        let mut s = [0; 3];
        for (i, slot) in s.iter_mut().enumerate() {
            let o = v.src[i];
            let x = self.opnd_dw(inst, &o, 0, v.opsel_hi >> i & 1 != 0)?;
            let x = self.u32v(x);
            let mut y = if v.opsel_hi >> i & 1 != 0 {
                self.half_f32(x, v.opsel >> i & 1 != 0)
            } else {
                self.f32v(x)
            };
            // v_fma_mix: neg_hi is abs
            if v.neg_hi >> i & 1 != 0 {
                y = self.gl(glsl::F_ABS, Ty::F32, &[y]);
            }
            if v.neg >> i & 1 != 0 {
                y = self.spv(op::F_NEGATE, Ty::F32, &[y]);
            }
            *slot = y;
        }
        let mut x = self.gl(glsl::FMA, Ty::F32, &[s[0], s[1], s[2]]);
        if name == "v_fma_mix_f32" {
            return f(self, x);
        }
        if v.clamp {
            let (z, o) = (self.kf(0.0), self.kf(1.0));
            x = self.gl(glsl::N_CLAMP, Ty::F32, &[x, z, o]);
        }
        let h = self.spv(op::F_CONVERT, Ty::F16, &[x]);
        let hb = self.spv(op::BITCAST, Ty::U16, &[h]);
        let h32 = self.spv(op::U_CONVERT, Ty::U32, &[hb]);
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        let old = self.val(inst, Reg::V(reg))?;
        let old = self.data(old);
        let old = self.u32v(old);
        let (o, w) = (self.k32(if name == "v_fma_mixhi_f16" { 16 } else { 0 }), self.k32(16));
        let r = self.spv(op::BIT_FIELD_INSERT, Ty::U32, &[old, h32, o, w]);
        self.def(inst, Reg::V(reg), R::D(r));
        Ok(())
    }

    #[allow(unused_variables)]
    pub(super) fn op_fma_f16(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.f16src(inst, v, 0)?;
        let b = self.f16src(inst, v, 1)?;
        let c = if name == "v_fmac_f16" {
            let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
            let c = self.val(inst, Reg::V(reg))?;
            let c = self.data(c);
            let c = self.u32v(c);
            let h = self.spv(op::U_CONVERT, Ty::U16, &[c]);
            self.spv(op::BITCAST, Ty::F16, &[h])
        } else {
            self.f16src(inst, v, 2)?
        };
        let x = if name == "v_med3_f16" {
            let lo = self.gl(glsl::N_MIN, Ty::F16, &[a, b]);
            let hi = self.gl(glsl::N_MAX, Ty::F16, &[a, b]);
            let m = self.gl(glsl::N_MIN, Ty::F16, &[hi, c]);
            self.gl(glsl::N_MAX, Ty::F16, &[lo, m])
        } else {
            self.gl(glsl::FMA, Ty::F16, &[a, b, c])
        };
        self.write_half(inst, v, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_pk_fma_f16(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let n = if name == "v_pk_fma_f16" { 3 } else { 2 };
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for i in 0..n {
            let x = self.opnd_dw(inst, &v.src[i], 0, true)?;
            let x = self.u32v(x);
            let v2 = self.spv(op::BITCAST, Ty::vec(S::F16, 2), &[x]);
            let e0 = self.extract(v2, 0);
            let e1 = self.extract(v2, 1);
            let mut l = if v.opsel >> i & 1 != 0 { e1 } else { e0 };
            let mut h = if v.opsel_hi >> i & 1 != 0 { e1 } else { e0 };
            if v.neg >> i & 1 != 0 {
                l = self.spv(op::F_NEGATE, Ty::F16, &[l]);
            }
            if v.neg_hi >> i & 1 != 0 {
                h = self.spv(op::F_NEGATE, Ty::F16, &[h]);
            }
            lo.push(l);
            hi.push(h);
        }
        let mut res = Vec::new();
        for s in [lo, hi] {
            let r = match name {
                "v_pk_fma_f16" => self.gl(glsl::FMA, Ty::F16, &[s[0], s[1], s[2]]),
                "v_pk_mul_f16" => self.spv(op::F_MUL, Ty::F16, &[s[0], s[1]]),
                "v_pk_add_f16" => self.spv(op::F_ADD, Ty::F16, &[s[0], s[1]]),
                "v_pk_max_f16" => self.gl(glsl::N_MAX, Ty::F16, &[s[0], s[1]]),
                _ => self.gl(glsl::N_MIN, Ty::F16, &[s[0], s[1]]),
            };
            let r = if v.clamp {
                let (z, o) = (self.f.konst(Ty::F16, 0), self.f.konst(Ty::F16, 0x3C00));
                self.gl(glsl::N_CLAMP, Ty::F16, &[r, z, o])
            } else {
                r
            };
            res.push(r);
        }
        let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F16, 2), &res);
        let x = self.spv(op::BITCAST, Ty::U32, &[v2]);
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        self.def(inst, Reg::V(reg), R::D(x));
        Ok(())
    }

    #[allow(unused_variables)]
    pub(super) fn op_lshrrev_b16(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.usrc(inst, v, 0)?;
        let b = self.usrc(inst, v, 1)?;
        let (z, s16) = (self.k32(0), self.k32(16));
        let x = match name {
            "v_lshrrev_b16" | "v_lshlrev_b16" | "v_ashrrev_i16" => {
                let k15 = self.k32(15);
                let s = self.spv(op::BITWISE_AND, Ty::U32, &[a, k15]);
                let ext = if name == "v_ashrrev_i16" { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
                let bb = self.spv(ext, Ty::U32, &[b, z, s16]);
                let o = match name {
                    "v_lshrrev_b16" => op::SHIFT_RIGHT_LOGICAL,
                    "v_lshlrev_b16" => op::SHIFT_LEFT_LOGICAL,
                    _ => op::SHIFT_RIGHT_ARITHMETIC,
                };
                self.spv(o, Ty::U32, &[bb, s])
            }
            "v_add_nc_u16" => self.spv(op::I_ADD, Ty::U32, &[a, b]),
            "v_sub_nc_u16" => self.spv(op::I_SUB, Ty::U32, &[a, b]),
            _ => self.spv(op::I_MUL, Ty::U32, &[a, b]),
        };
        let h = self.spv(op::U_CONVERT, Ty::U16, &[x]);
        self.write_half(inst, v, h)
    }

    #[allow(unused_variables)]
    pub(super) fn op_cvt_pk_u8_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        // byte (src1 & 3) of src2 replaced by u8(src0)
        let a = self.fsrc(inst, v, 0)?;
        let s = self.usrc(inst, v, 1)?;
        let old = self.usrc(inst, v, 2)?;
        let r = self.gl(glsl::ROUND_EVEN, Ty::F32, &[a]);
        let (z, k) = (self.kf(0.0), self.kf(255.0));
        let r = self.gl(glsl::N_CLAMP, Ty::F32, &[r, z, k]);
        let u = self.spv(op::CONVERT_F_TO_U, Ty::U32, &[r]);
        let k3 = self.k32(3);
        let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k3]);
        let s = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[s, k3]);
        let w = self.k32(8);
        let x = self.spv(op::BIT_FIELD_INSERT, Ty::U32, &[old, u, s, w]);
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_mul_u32_u24(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.usrc(inst, v, 0)?;
        let b = self.usrc(inst, v, 1)?;
        let signed = name.contains("i32_i24");
        let (z, w) = (self.k32(0), self.k32(24));
        let e = if signed { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
        let a = self.spv(e, Ty::U32, &[a, z, w]);
        let b = self.spv(e, Ty::U32, &[b, z, w]);
        let x = if name == "v_mul_hi_u32_u24" {
            let s = self.spv(op::U_MUL_EXTENDED, Ty::vec(S::U32, 2), &[a, b]);
            self.extract(s, 1)
        } else {
            self.spv(op::I_MUL, Ty::U32, &[a, b])
        };
        let x = if name.starts_with("v_mad") {
            let c = self.usrc(inst, v, 2)?;
            self.spv(op::I_ADD, Ty::U32, &[x, c])
        } else {
            x
        };
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_lshlrev_b64(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let s = self.usrc(inst, v, 0)?;
        let a = self.opnd_u64(inst, &v.src[1])?;
        let k63 = self.k32(63);
        let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k63]);
        let o = match name {
            "v_lshlrev_b64" => op::SHIFT_LEFT_LOGICAL,
            "v_lshrrev_b64" => op::SHIFT_RIGHT_LOGICAL,
            _ => op::SHIFT_RIGHT_ARITHMETIC,
        };
        let x = self.spv(o, Ty::U64, &[a, s]);
        let (lo, hi) = self.split64(x);
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        self.def(inst, Reg::V(reg), R::D(lo));
        self.def(inst, Reg::V(reg + 1), R::D(hi));
        Ok(())
    }

    #[allow(unused_variables)]
    pub(super) fn op_alignbit_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.usrc(inst, v, 0)?;
        let b = self.usrc(inst, v, 1)?;
        let s = self.usrc(inst, v, 2)?;
        let s = if name == "v_alignbyte_b32" {
            let k3 = self.k32(3);
            let t = self.spv(op::BITWISE_AND, Ty::U32, &[s, k3]);
            self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[t, k3])
        } else {
            let k31 = self.k32(31);
            self.spv(op::BITWISE_AND, Ty::U32, &[s, k31])
        };
        let ab = self.u64_of(b, a);
        let sh = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U64, &[ab, s]);
        let x = self.spv(op::U_CONVERT, Ty::U32, &[sh]);
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_perm_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.usrc(inst, v, 0)?;
        let b = self.usrc(inst, v, 1)?;
        let s = self.usrc(inst, v, 2)?;
        let Some(sel) = self.f.as_const(s) else {
            return Err(unsupported(off, "v_perm_b32 with a computed selector"));
        };
        // bytes 0-3 from b, 4-7 from a
        let mut x = self.k32(0);
        for k in 0..4u32 {
            let sb = (sel >> (8 * k)) as u32 & 0xFF;
            let byte = match sb {
                0..=3 => {
                    let (o, w) = (self.k32(8 * sb), self.k32(8));
                    self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[b, o, w])
                }
                4..=7 => {
                    let (o, w) = (self.k32(8 * (sb - 4)), self.k32(8));
                    self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[a, o, w])
                }
                12 => self.k32(0),
                13..=255 => self.k32(0xFF),
                _ => return Err(unsupported(off, "v_perm_b32 sign selects")),
            };
            let (o, w) = (self.k32(8 * k), self.k32(8));
            x = self.spv(op::BIT_FIELD_INSERT, Ty::U32, &[x, byte, o, w]);
        }
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_min3_u32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let s = name.ends_with("i32");
        let (mn, mx) = if s { (glsl::S_MIN, glsl::S_MAX) } else { (glsl::U_MIN, glsl::U_MAX) };
        let a = self.usrc(inst, v, 0)?;
        let b = self.usrc(inst, v, 1)?;
        let c = self.usrc(inst, v, 2)?;
        let x = if name.starts_with("v_med3") {
            let lo = self.gl(mn, Ty::U32, &[a, b]);
            let hi = self.gl(mx, Ty::U32, &[a, b]);
            let m = self.gl(mn, Ty::U32, &[hi, c]);
            self.gl(mx, Ty::U32, &[lo, m])
        } else {
            let g = if name.starts_with("v_min3") { mn } else { mx };
            let t = self.gl(g, Ty::U32, &[a, b]);
            self.gl(g, Ty::U32, &[t, c])
        };
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_mad_u64_u32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let a = self.usrc(inst, v, 0)?;
        let b = self.usrc(inst, v, 1)?;
        let c = self.opnd_u64(inst, &v.src[2])?;
        let s = name == "v_mad_i64_i32";
        let e = if s { op::S_CONVERT } else { op::U_CONVERT };
        let a64 = self.spv(e, Ty::U64, &[a]);
        let b64 = self.spv(e, Ty::U64, &[b]);
        let p = self.spv(op::I_MUL, Ty::U64, &[a64, b64]);
        let x = self.spv(op::I_ADD, Ty::U64, &[p, c]);
        let carry = self.spv(op::U_LESS_THAN, Ty::BOOL, &[x, p]);
        let cm = self.spv(op::LOGICAL_AND, Ty::BOOL, &[a_exec, carry]);
        self.mask_def(inst, &v.carry, cm);
        let (lo, hi) = self.split64(x);
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        self.def(inst, Reg::V(reg), R::D(lo));
        self.def(inst, Reg::V(reg + 1), R::D(hi));
        Ok(())
    }

    #[allow(unused_variables)]
    pub(super) fn op_mbcnt_lo_u32_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        // bits of the mask below this lane's console lane
        let m = self.usrc(inst, v, 0)?;
        let acc = self.usrc(inst, v, 1)?;
        let lane = self.console_lane();
        let k32 = self.k32(32);
        let n = if name == "v_mbcnt_lo_u32_b32" {
            self.gl(glsl::U_MIN, Ty::U32, &[lane, k32])
        } else {
            let l = self.gl(glsl::U_MAX, Ty::U32, &[lane, k32]);
            self.spv(op::I_SUB, Ty::U32, &[l, k32])
        };
        let below = self.mask_bits(n);
        let t = self.spv(op::BITWISE_AND, Ty::U32, &[m, below]);
        let c = self.spv(op::BIT_COUNT, Ty::U32, &[t]);
        let x = self.spv(op::I_ADD, Ty::U32, &[c, acc]);
        f(self, x)
    }

    #[allow(unused_variables)]
    pub(super) fn op_writelane_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let s = self.usrc(inst, v, 0)?;
        let l = self.usrc(inst, v, 1)?;
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        let old = self.val(inst, Reg::V(reg))?;
        let old = self.data(old);
        let old = self.u32v(old);
        let lane = self.lane();
        let k31 = self.k32(31);
        if self.wave64 && self.f.as_const(l).is_none_or(|c| c >= 32) {
            self.approx("v_writelane_b32 of a lane in the other half of a wave64");
        }
        let lm = self.spv(op::BITWISE_AND, Ty::U32, &[l, k31]);
        let me = self.spv(op::I_EQUAL, Ty::BOOL, &[lane, lm]);
        let x = self.spv(op::SELECT, Ty::U32, &[me, s, old]);
        self.def(inst, Reg::V(reg), R::D(x));
        Ok(())
    }

    #[allow(unused_variables)]
    pub(super) fn op_permlanex16_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let x = self.usrc(inst, v, 0)?;
        let s1 = self.usrc(inst, v, 1)?;
        let s2 = self.usrc(inst, v, 2)?;
        let lane = self.lane();
        let k15 = self.k32(15);
        let k7 = self.k32(7);
        let k3 = self.k32(3);
        let k8 = self.k32(8);
        let i = self.spv(op::BITWISE_AND, Ty::U32, &[lane, k15]);
        let lo = self.spv(op::U_LESS_THAN, Ty::BOOL, &[i, k8]);
        let sel = self.spv(op::SELECT, Ty::U32, &[lo, s1, s2]);
        let i7 = self.spv(op::BITWISE_AND, Ty::U32, &[i, k7]);
        let k2 = self.k32(2);
        let sh = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[i7, k2]);
        let nib = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[sel, sh]);
        let nib = self.spv(op::BITWISE_AND, Ty::U32, &[nib, k15]);
        let k16 = self.k32(16);
        let row = self.spv(op::BITWISE_AND, Ty::U32, &[lane, k16]);
        let row = if name == "v_permlanex16_b32" {
            self.spv(op::BITWISE_XOR, Ty::U32, &[row, k16])
        } else {
            row
        };
        let from = self.spv(op::BITWISE_OR, Ty::U32, &[row, nib]);
        let _ = k3;
        let r = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![x, from]);
        f(self, r)
    }
}
