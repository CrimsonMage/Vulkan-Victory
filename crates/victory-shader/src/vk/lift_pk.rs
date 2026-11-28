//! Packed operations the title's own programs lack and AMD's compute
//! compiler writes (the FidelityFX upscaler's network passes): dot
//! products of signed bytes, 16-bit integer pairs, the accumulating f16
//! pair.

use super::*;

impl Lift<'_> {
    /// What a VOP3P instruction reads of a source dword by its selectors:
    /// the low half is the dword's high half with op_sel set, the high
    /// half the dword's low half with op_sel_hi clear. A constant is its
    /// whole 32-bit value first (the compiler makes 0x0001_0000 as the
    /// inline 1 with both selectors turned).
    fn pk_halves(&mut self, x: V, lo_high: bool, hi_high: bool) -> V {
        if (lo_high, hi_high) == (false, true) {
            return x;
        }
        if let Some(c) = self.f.as_const(x) {
            let half = |high: bool| (c >> if high { 16 } else { 0 }) as u32 & 0xFFFF;
            return self.k32(half(lo_high) | half(hi_high) << 16);
        }
        let (z, k16) = (self.k32(0), self.k32(16));
        let lo = self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[x, if lo_high { k16 } else { z }, k16]);
        let hi = self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[x, if hi_high { k16 } else { z }, k16]);
        self.spv(op::BIT_FIELD_INSERT, Ty::U32, &[lo, hi, k16, k16])
    }

    /// Source `i` of a VOP3P instruction as the dword its selectors read.
    fn pk_src(&mut self, inst: &Inst, v: &Valu, i: usize) -> Res<V> {
        let x = self.usrc(inst, v, i)?;
        Ok(self.pk_halves(x, v.opsel >> i & 1 != 0, v.opsel_hi >> i & 1 != 0))
    }

    /// Signed byte `i` of `x` as a dword.
    fn byte_i32(&mut self, x: V, i: u32) -> V {
        if let Some(c) = self.f.as_const(x) {
            return self.k32((c >> (8 * i)) as u8 as i8 as u32);
        }
        let (o, w) = (self.k32(8 * i), self.k32(8));
        self.spv(op::BIT_FIELD_S_EXTRACT, Ty::U32, &[x, o, w])
    }

    /// `v_dot4c_i32_i8 D, S0, S1` (D += S0 · S1) and `v_dot4_i32_i8 D, S0,
    /// S1, S2` (D = S0 · S1 + S2): the products of the four signed bytes
    /// of two dwords summed with the addend, modulo 2^32 (measured on
    /// RDNA2 by ProjectLeonida, `oracle/packed.rs`). The second form reads
    /// its two factors through op_sel and its addend whole. With `clamp`
    /// the hardware saturates and its negate bits flip sign bits of the
    /// operands: no program seen has either, so both are refused.
    pub(super) fn op_dot4_i8(&mut self, inst: &Inst, name: &'static str, v: &Valu) -> Res<()> {
        let off = inst.machine.offset;
        if v.clamp || v.neg != 0 || v.neg_hi != 0 || v.abs != 0 || v.sdwa.is_some() {
            return Err(unsupported(off, "a saturating or negated dot product"));
        }
        let (a, b, addend) = if name == "v_dot4_i32_i8" {
            if (v.opsel >> 2 & 1, v.opsel_hi >> 2 & 1) != (0, 1) {
                return Err(unsupported(off, "a dot product whose addend is read by halves"));
            }
            (self.pk_src(inst, v, 0)?, self.pk_src(inst, v, 1)?, self.usrc(inst, v, 2)?)
        } else {
            let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
            let old = self.val(inst, Reg::V(reg))?;
            let old = self.data(old);
            (self.usrc(inst, v, 0)?, self.usrc(inst, v, 1)?, self.u32v(old))
        };
        // two registers: the function (thousands of these make a network
        // pass, and written out each is sixteen operations)
        if self.f.as_const(a).is_none() && self.f.as_const(b).is_none() {
            let sum = self.emit(Op::Call(Helper::Dot4), Ty::U32, vec![addend, a, b]);
            return self.vdst(inst, v, sum);
        }
        let mut sum = addend;
        for i in 0..4 {
            let (x, y) = (self.byte_i32(a, i), self.byte_i32(b, i));
            // a constant factor's zero bytes add nothing
            if self.f.as_const(x) == Some(0) || self.f.as_const(y) == Some(0) {
                continue;
            }
            let p = self.spv(op::I_MUL, Ty::U32, &[x, y]);
            sum = self.spv(op::I_ADD, Ty::U32, &[sum, p]);
        }
        self.vdst(inst, v, sum)
    }

    /// `v_pk_add_u16` / `v_pk_sub_u16`: each 16-bit half of the result is
    /// the sum (difference) of the halves the selectors read, modulo 2^16.
    /// (`clamp` saturates: refused.)
    pub(super) fn op_pk_add_u16(&mut self, inst: &Inst, name: &'static str, v: &Valu) -> Res<()> {
        if v.clamp || v.neg != 0 || v.neg_hi != 0 {
            return Err(unsupported(inst.machine.offset, "a saturating or negated 16-bit pair"));
        }
        let (a, b) = (self.pk_src(inst, v, 0)?, self.pk_src(inst, v, 1)?);
        let o = if name == "v_pk_add_u16" { op::I_ADD } else { op::I_SUB };
        let k16 = self.k32(16);
        // the low halves' carry goes into bits the high half replaces
        let lo = self.spv(o, Ty::U32, &[a, b]);
        let (ah, bh) = (self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, k16]), self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, k16]));
        let hi = self.spv(o, Ty::U32, &[ah, bh]);
        let x = self.spv(op::BIT_FIELD_INSERT, Ty::U32, &[lo, hi, k16, k16]);
        self.vdst(inst, v, x)
    }

    /// `v_pk_fmac_f16 D, S0, S1` (VOP2, no selectors): each f16 half of D
    /// becomes S0 × S1 + D of that half, fused.
    pub(super) fn op_pk_fmac_f16(&mut self, inst: &Inst, name: &'static str, v: &Valu) -> Res<()> {
        let off = inst.machine.offset;
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        if v.src.iter().take(2).any(|s| !matches!(s, Operand::Vector { .. })) || v.sdwa.is_some() {
            // what a constant's high half holds is not known
            return Err(unsupported(off, "a constant operand of v_pk_fmac_f16"));
        }
        let old = self.val(inst, Reg::V(reg))?;
        let old = self.data(old);
        let mut pairs = Vec::new();
        for x in [self.usrc(inst, v, 0)?, self.usrc(inst, v, 1)?, self.u32v(old)] {
            pairs.push(self.spv(op::BITCAST, Ty::vec(S::F16, 2), &[x]));
        }
        let mut res = Vec::new();
        for half in 0..2 {
            let s: Vec<V> = pairs.iter().map(|&p| self.extract(p, half)).collect();
            res.push(self.gl(glsl::FMA, Ty::F16, &s));
        }
        let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F16, 2), &res);
        let x = self.spv(op::BITCAST, Ty::U32, &[v2]);
        self.def(inst, Reg::V(reg), R::D(x));
        Ok(())
    }
}
