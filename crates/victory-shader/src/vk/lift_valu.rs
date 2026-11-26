//! Vector ALU instructions.

use super::*;

impl Lift<'_> {
    pub(super) fn valu(&mut self, inst: &Inst, name: &'static str, v: &Valu) -> Res<()> {
        // EXEC writes and scalar results run unguarded; compares and carries
        // into lane masks run in the guard (0 for the lanes outside it).
        let sgpr_out = name.starts_with("v_cmpx")
            || (matches!(v.dst, Some(Operand::Scalar { .. })) && !name.starts_with("v_cmp"));
        let cross = v.dpp.is_some()
            || matches!(
                name,
                "v_readlane_b32" | "v_readfirstlane_b32" | "v_writelane_b32" | "v_permlanex16_b32" | "v_permlane16_b32"
            );
        if sgpr_out || cross {
            self.need(inst);
            let a = self.active(inst)?;
            let all = self.f.as_const(a) == Some(1);
            self.sel = if all || name == "v_writelane_b32" { None } else { Some(a) };
            let r = self.valu_op(inst, name, v, a);
            self.sel = None;
            r
        } else {
            let g = self.guard(inst)?;
            self.guarded = g;
            let a = if g { self.kb(true) } else { self.active(inst)? };
            let r = self.valu_op(inst, name, v, a);
            self.guarded = false;
            r
        }
    }

    pub(super) fn fbin(&mut self, inst: &Inst, v: &Valu, o: u16, swap: bool) -> Res<V> {
        let a = self.fsrc(inst, v, 0)?;
        let b = self.fsrc(inst, v, 1)?;
        let (a, b) = if swap { (b, a) } else { (a, b) };
        Ok(self.spv(o, Ty::F32, &[a, b]))
    }

    pub(super) fn ubin(&mut self, inst: &Inst, v: &Valu, o: u16, swap: bool) -> Res<V> {
        let a = self.usrc(inst, v, 0)?;
        let b = self.usrc(inst, v, 1)?;
        let (a, b) = if swap { (b, a) } else { (a, b) };
        Ok(self.spv(o, Ty::U32, &[a, b]))
    }

    /// Legacy multiply: 0 times anything (infinity, NaN) is 0.
    pub(super) fn mul_legacy(&mut self, a: V, b: V) -> V {
        let p = self.spv(op::F_MUL, Ty::F32, &[a, b]);
        // A constant that is not zero (and finite) leaves the product exact.
        let finite_nz = |l: &Self, x: V| {
            l.f.as_const(x).is_some_and(|c| {
                let f = f32::from_bits(c as u32);
                f != 0.0 && f.is_finite()
            })
        };
        if finite_nz(self, a) || finite_nz(self, b) {
            return p;
        }
        if self.f.as_const(a).is_some_and(|c| f32::from_bits(c as u32) == 0.0)
            || self.f.as_const(b).is_some_and(|c| f32::from_bits(c as u32) == 0.0)
        {
            return self.kf(0.0);
        }
        let z = self.kf(0.0);
        let az = self.spv(op::F_ORD_EQUAL, Ty::BOOL, &[a, z]);
        let bz = self.spv(op::F_ORD_EQUAL, Ty::BOOL, &[b, z]);
        let any = self.spv(op::LOGICAL_OR, Ty::BOOL, &[az, bz]);
        self.spv(op::SELECT, Ty::F32, &[any, z, p])
    }

    pub(super) fn f16src(&mut self, inst: &Inst, v: &Valu, i: usize) -> Res<V> {
        let o = v.src.get(i).ok_or(unsupported(inst.machine.offset, "a missing operand"))?;
        let x = self.opnd_dw(inst, o, 0, true)?;
        let x = self.u32v(x);
        let high = v.opsel >> i & 1 != 0 || v.sdwa.is_some_and(|s| s.src_sel.get(i) == Some(&5));
        let mut f = self.half_f16(x, high);
        if v.abs >> i & 1 != 0 {
            f = self.gl(glsl::F_ABS, Ty::F16, &[f]);
        }
        if v.neg >> i & 1 != 0 {
            f = self.spv(op::F_NEGATE, Ty::F16, &[f]);
        }
        Ok(f)
    }

    /// Half `high` of a dword as f16 (the dword's split is shared).
    pub(super) fn half_f16(&mut self, x: V, high: bool) -> V {
        if let Some(c) = self.f.as_const(x) {
            return self.f.konst(Ty::F16, (c >> if high { 16 } else { 0 }) & 0xFFFF);
        }
        let v2 = self.spv(op::BITCAST, Ty::vec(S::F16, 2), &[x]);
        self.extract(v2, u32::from(high))
    }

    /// Half `high` of a dword, an f16, as f32.
    pub(super) fn half_f32(&mut self, x: V, high: bool) -> V {
        if let Some(c) = self.f.as_const(x) {
            let h = ((c >> if high { 16 } else { 0 }) & 0xFFFF) as u16;
            return self.kf(f16_to_f32(h));
        }
        let v2 = self.gl(glsl::UNPACK_HALF_2X16, Ty::vec(S::F32, 2), &[x]);
        self.extract(v2, u32::from(high))
    }

    /// Writes an f16 or u16 result into the destination's low half (high
    /// with op_sel bit 3 or an SDWA WORD_1 select), keeping the other half.
    pub(super) fn write_half(&mut self, inst: &Inst, v: &Valu, x: V) -> Res<()> {
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err(unsupported(inst.machine.offset, "a 16-bit result outside a VGPR"));
        };
        let mut x = x;
        if self.f.ty(x) == Ty::F16 {
            if v.clamp {
                let (z, o) = (self.f.konst(Ty::F16, 0), self.f.konst(Ty::F16, 0x3C00));
                x = self.gl(glsl::N_CLAMP, Ty::F16, &[x, z, o]);
            }
            x = self.spv(op::BITCAST, Ty::U16, &[x]);
        }
        let x32 = self.spv(op::U_CONVERT, Ty::U32, &[x]);
        let mut high = v.opsel >> 3 & 1 != 0;
        let mut keep = true;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4 | 6, _) => keep = false,
                (5, _) => (high, keep) = (true, false),
                _ => return Err(unsupported(inst.machine.offset, "an SDWA destination of a 16-bit operation")),
            }
        }
        let old = if keep {
            match inst.read(Reg::V(reg)) {
                Some(o) => {
                    let r = self.value(o, inst.machine.offset)?;
                    let d = self.data(r);
                    self.u32v(d)
                }
                None => self.k32(0),
            }
        } else {
            self.k32(0)
        };
        let (o, w) = (self.k32(if high { 16 } else { 0 }), self.k32(16));
        let r = self.spv(op::BIT_FIELD_INSERT, Ty::U32, &[old, x32, o, w]);
        self.def(inst, Reg::V(reg), R::D(r));
        Ok(())
    }

    pub(super) fn valu_op(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        if name.starts_with("v_cmp") {
            return self.compare(inst, name, v, a_exec);
        }
        if v.dpp.is_some() {
            return self.dpp(inst, name, v, a_exec);
        }
        let Some(part) = valu_part(name) else {
            return Err(unsupported(inst.machine.offset, name));
        };
        match part {
            0 => self.valu_misc0(inst, name, v, a_exec),
            1 => self.valu_misc1(inst, name, v, a_exec),
            2 => self.valu_float0(inst, name, v, a_exec),
            3 => self.valu_float1(inst, name, v, a_exec),
            4 => self.valu_float2(inst, name, v, a_exec),
            5 => self.valu_half0(inst, name, v, a_exec),
            6 => self.valu_half1(inst, name, v, a_exec),
            7 => self.valu_cvt0(inst, name, v, a_exec),
            8 => self.valu_cvt1(inst, name, v, a_exec),
            9 => self.valu_int0(inst, name, v, a_exec),
            10 => self.valu_int1(inst, name, v, a_exec),
            11 => self.valu_int2(inst, name, v, a_exec),
            12 => self.valu_int3(inst, name, v, a_exec),
            13 => self.valu_carry0(inst, name, v, a_exec),
            14 => self.valu_lane0(inst, name, v, a_exec),
            _ => Err(unsupported(inst.machine.offset, name)),
        }
    }

    pub(super) fn valu_misc0(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_nop" => Ok(()),
            "v_mov_b32" => {
                let x = self.opnd(inst, &v.src[0])?;
                let x = if v.sdwa.is_some() { self.usrc(inst, v, 0)? } else { x };
                f(self, x)
            }
            "v_cndmask_b32" => self.op_cndmask_b32(inst, name, v, a_exec),
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                let x = self.fsrc(inst, v, 0)?;
                let y = self.fsrc(inst, v, 1)?;
                let z = self.fsrc(inst, v, 2)?;
                let r = self.cube(name, x, y, z);
                f(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" => self.movrel(inst, name, v),
            "v_swap_b32" => {
                let Some(Operand::Vector { reg: d, .. }) = v.dst else { return Err(unsupported(off, name)) };
                let Operand::Vector { reg: s, .. } = v.src[0] else { return Err(unsupported(off, name)) };
                let a = self.val(inst, Reg::V(d))?;
                let b = self.val(inst, Reg::V(s))?;
                self.def(inst, Reg::V(d), b);
                self.def(inst, Reg::V(s), a);
                Ok(())
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_misc1(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_interp_p1_f32" | "v_interp_p2_f32" | "v_interp_p1ll_f16" | "v_interp_p1lv_f16" | "v_interp_p2_f16"
            | "v_interp_mov_f32" => {
                let dst = v.dst.ok_or(unsupported(off, name))?;
                self.interp(inst, dst)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_float0(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_add_f32" => {
                let x = self.fbin(inst, v, op::F_ADD, false)?;
                f(self, x)
            }
            "v_sub_f32" => {
                let x = self.fbin(inst, v, op::F_SUB, false)?;
                f(self, x)
            }
            "v_subrev_f32" => {
                let x = self.fbin(inst, v, op::F_SUB, true)?;
                f(self, x)
            }
            "v_mul_f32" => {
                let x = self.fbin(inst, v, op::F_MUL, false)?;
                f(self, x)
            }
            "v_mul_legacy_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                let x = self.mul_legacy(a, b);
                f(self, x)
            }
            "v_fma_f32" | "v_mad_f32" | "v_fma_legacy_f32" | "v_mad_legacy_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                let c = self.fsrc(inst, v, 2)?;
                let x = if name.contains("legacy") {
                    let p = self.mul_legacy(a, b);
                    self.spv(op::F_ADD, Ty::F32, &[p, c])
                } else {
                    self.gl(glsl::FMA, Ty::F32, &[a, b, c])
                };
                f(self, x)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_float1(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_fmac_f32" | "v_mac_f32" | "v_fmac_legacy_f32" | "v_mac_legacy_f32" => self.op_fmac_f32(inst, name, v, a_exec),
            "v_fmaak_f32" | "v_madak_f32" | "v_fmamk_f32" | "v_madmk_f32" => {
                // ak: a * b + K (K src[2]); mk: a * K + b (K src[1] in our order: src0, K, src1?)
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                let c = self.fsrc(inst, v, 2)?;
                let x = self.gl(glsl::FMA, Ty::F32, &[a, b, c]);
                f(self, x)
            }
            "v_min_f32" | "v_max_f32" => {
                let g = if name == "v_min_f32" { glsl::N_MIN } else { glsl::N_MAX };
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                let x = self.gl(g, Ty::F32, &[a, b]);
                f(self, x)
            }
            "v_min3_f32" | "v_max3_f32" => {
                let g = if name == "v_min3_f32" { glsl::N_MIN } else { glsl::N_MAX };
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                let c = self.fsrc(inst, v, 2)?;
                let t = self.gl(g, Ty::F32, &[a, b]);
                let x = self.gl(g, Ty::F32, &[t, c]);
                f(self, x)
            }
            "v_med3_f32" => self.op_med3_f32(inst, name, v, a_exec),
            "v_rcp_f32" | "v_rcp_iflag_f32" | "v_rcp_legacy_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                let one = self.kf(1.0);
                let x = self.spv(op::F_DIV, Ty::F32, &[one, a]);
                f(self, x)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_float2(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_rsq_f32" | "v_sqrt_f32" | "v_exp_f32" | "v_log_f32" | "v_fract_f32" | "v_floor_f32" | "v_ceil_f32"
            | "v_trunc_f32" | "v_rndne_f32" => {
                let g = match name {
                    "v_rsq_f32" => glsl::INVERSE_SQRT,
                    "v_sqrt_f32" => glsl::SQRT,
                    "v_exp_f32" => glsl::EXP2,
                    "v_log_f32" => glsl::LOG2,
                    "v_fract_f32" => glsl::FRACT,
                    "v_floor_f32" => glsl::FLOOR,
                    "v_ceil_f32" => glsl::CEIL,
                    "v_trunc_f32" => glsl::TRUNC,
                    _ => glsl::ROUND_EVEN,
                };
                let a = self.fsrc(inst, v, 0)?;
                let x = self.gl(g, Ty::F32, &[a]);
                f(self, x)
            }
            "v_sin_f32" | "v_cos_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                let k = self.kf(std::f32::consts::TAU);
                let r = self.spv(op::F_MUL, Ty::F32, &[a, k]);
                let x = self.gl(if name == "v_sin_f32" { glsl::SIN } else { glsl::COS }, Ty::F32, &[r]);
                f(self, x)
            }
            "v_ldexp_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                let x = self.gl(glsl::LDEXP, Ty::F32, &[a, b]);
                f(self, x)
            }
            "v_frexp_mant_f32" | "v_frexp_exp_i32_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                // struct { f32 mantissa; u32 exponent } (type marker Void x 2)
                let s = self.emit(Op::Glsl(glsl::FREXP_STRUCT), Ty { s: S::Void, n: 2 }, vec![a]);
                let x = if name == "v_frexp_mant_f32" {
                    self.emit_imm(Op::Spv(op::COMPOSITE_EXTRACT), Ty::F32, vec![s], vec![0])
                } else {
                    self.emit_imm(Op::Spv(op::COMPOSITE_EXTRACT), Ty::U32, vec![s], vec![1])
                };
                f(self, x)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_half0(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_mov_b16" => {
                let x = self.usrc(inst, v, 0)?;
                let h = self.spv(op::U_CONVERT, Ty::U16, &[x]);
                self.write_half(inst, v, h)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => self.op_fma_mix_f32(inst, name, v, a_exec),
            "v_add_f16" | "v_sub_f16" | "v_subrev_f16" | "v_mul_f16" | "v_max_f16" | "v_min_f16" => {
                let a = self.f16src(inst, v, 0)?;
                let b = self.f16src(inst, v, 1)?;
                let x = match name {
                    "v_add_f16" => self.spv(op::F_ADD, Ty::F16, &[a, b]),
                    "v_sub_f16" => self.spv(op::F_SUB, Ty::F16, &[a, b]),
                    "v_subrev_f16" => self.spv(op::F_SUB, Ty::F16, &[b, a]),
                    "v_mul_f16" => self.spv(op::F_MUL, Ty::F16, &[a, b]),
                    "v_max_f16" => self.gl(glsl::N_MAX, Ty::F16, &[a, b]),
                    _ => self.gl(glsl::N_MIN, Ty::F16, &[a, b]),
                };
                self.write_half(inst, v, x)
            }
            "v_fma_f16" | "v_mad_f16" | "v_fmac_f16" | "v_med3_f16" => self.op_fma_f16(inst, name, v, a_exec),
            "v_rcp_f16" | "v_rsq_f16" | "v_sqrt_f16" | "v_exp_f16" | "v_log_f16" | "v_sin_f16" | "v_cos_f16"
            | "v_trunc_f16" | "v_floor_f16" | "v_ceil_f16" | "v_fract_f16" | "v_rndne_f16" => {
                let a = self.f16src(inst, v, 0)?;
                let x = match name {
                    "v_rcp_f16" => {
                        let one = self.f.konst(Ty::F16, 0x3C00);
                        self.spv(op::F_DIV, Ty::F16, &[one, a])
                    }
                    "v_sin_f16" | "v_cos_f16" => {
                        let a32 = self.spv(op::F_CONVERT, Ty::F32, &[a]);
                        let k = self.kf(std::f32::consts::TAU);
                        let r = self.spv(op::F_MUL, Ty::F32, &[a32, k]);
                        let s = self.gl(if name == "v_sin_f16" { glsl::SIN } else { glsl::COS }, Ty::F32, &[r]);
                        self.spv(op::F_CONVERT, Ty::F16, &[s])
                    }
                    _ => {
                        let g = match name {
                            "v_rsq_f16" => glsl::INVERSE_SQRT,
                            "v_sqrt_f16" => glsl::SQRT,
                            "v_exp_f16" => glsl::EXP2,
                            "v_log_f16" => glsl::LOG2,
                            "v_trunc_f16" => glsl::TRUNC,
                            "v_floor_f16" => glsl::FLOOR,
                            "v_ceil_f16" => glsl::CEIL,
                            "v_fract_f16" => glsl::FRACT,
                            _ => glsl::ROUND_EVEN,
                        };
                        self.gl(g, Ty::F16, &[a])
                    }
                };
                self.write_half(inst, v, x)
            }
            "v_pk_fma_f16" | "v_pk_mul_f16" | "v_pk_add_f16" | "v_pk_max_f16" | "v_pk_min_f16" => self.op_pk_fma_f16(inst, name, v, a_exec),
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_half1(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_lshrrev_b16" | "v_lshlrev_b16" | "v_ashrrev_i16" | "v_add_nc_u16" | "v_sub_nc_u16" | "v_mul_lo_u16" => self.op_lshrrev_b16(inst, name, v, a_exec),
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_cvt0(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_cvt_f32_u32" | "v_cvt_f32_i32" => {
                let a = self.usrc(inst, v, 0)?;
                let o = if name == "v_cvt_f32_u32" { op::CONVERT_U_TO_F } else { op::CONVERT_S_TO_F };
                let x = self.spv(o, Ty::F32, &[a]);
                f(self, x)
            }
            "v_cvt_u32_f32" | "v_cvt_i32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = self.fsrc(inst, v, 0)?;
                if name == "v_cvt_flr_i32_f32" {
                    a = self.gl(glsl::FLOOR, Ty::F32, &[a]);
                } else if name == "v_cvt_rpi_i32_f32" {
                    let h = self.kf(0.5);
                    let s = self.spv(op::F_ADD, Ty::F32, &[a, h]);
                    a = self.gl(glsl::FLOOR, Ty::F32, &[s]);
                }
                let o = if name == "v_cvt_u32_f32" { op::CONVERT_F_TO_U } else { op::CONVERT_F_TO_S };
                let x = self.spv(o, Ty::U32, &[a]);
                f(self, x)
            }
            "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1" | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let a = self.usrc(inst, v, 0)?;
                let n = u32::from(name.as_bytes()[name.len() - 1] - b'0');
                let (o, w) = (self.k32(8 * n), self.k32(8));
                let b = self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[a, o, w]);
                let x = self.spv(op::CONVERT_U_TO_F, Ty::F32, &[b]);
                f(self, x)
            }
            "v_cvt_f32_f16" => {
                let h = self.f16src(inst, v, 0)?;
                let x = self.spv(op::F_CONVERT, Ty::F32, &[h]);
                f(self, x)
            }
            "v_cvt_f16_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                let h = self.spv(op::F_CONVERT, Ty::F16, &[a]);
                self.write_half(inst, v, h)
            }
            "v_cvt_pkrtz_f16_f32" | "v_cvt_pknorm_u16_f32" | "v_cvt_pknorm_i16_f32" => {
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                if name != "v_cvt_pkrtz_f16_f32" {
                    return Err(unsupported(off, name));
                }
                let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F32, 2), &[a, b]);
                let x = self.gl(glsl::PACK_HALF_2X16, Ty::U32, &[v2]);
                f(self, x)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_cvt1(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_pack_b32_f16" => {
                let a = self.f16src(inst, v, 0)?;
                let b = self.f16src(inst, v, 1)?;
                let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F16, 2), &[a, b]);
                let x = self.spv(op::BITCAST, Ty::U32, &[v2]);
                f(self, x)
            }
            "v_cvt_pk_u8_f32" => self.op_cvt_pk_u8_f32(inst, name, v, a_exec),
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_int0(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_add_nc_u32" | "v_add_u32" => {
                let x = self.ubin(inst, v, op::I_ADD, false)?;
                f(self, x)
            }
            "v_sub_nc_u32" | "v_sub_u32" => {
                let x = self.ubin(inst, v, op::I_SUB, false)?;
                f(self, x)
            }
            "v_subrev_nc_u32" | "v_subrev_u32" => {
                let x = self.ubin(inst, v, op::I_SUB, true)?;
                f(self, x)
            }
            "v_mul_lo_u32" | "v_mul_lo_i32" => {
                let x = self.ubin(inst, v, op::I_MUL, false)?;
                f(self, x)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let a = self.usrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                let o = if name == "v_mul_hi_u32" { op::U_MUL_EXTENDED } else { op::S_MUL_EXTENDED };
                let s = self.spv(o, Ty::vec(S::U32, 2), &[a, b]);
                let x = self.extract(s, 1);
                f(self, x)
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" | "v_mad_u32_u24" | "v_mad_i32_i24" | "v_mul_hi_u32_u24" => self.op_mul_u32_u24(inst, name, v, a_exec),
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_int1(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_xnor_b32" => {
                let a = self.usrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                let base = &name[2..name.len() - 4];
                let x = self.int_bin(base, a, b).ok_or(unsupported(off, name))?;
                f(self, x)
            }
            "v_not_b32" => {
                let a = self.usrc(inst, v, 0)?;
                let x = self.spv(op::NOT, Ty::U32, &[a]);
                f(self, x)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let s = self.usrc(inst, v, 0)?;
                let a = self.usrc(inst, v, 1)?;
                let k31 = self.k32(31);
                let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k31]);
                let o = match name {
                    "v_lshlrev_b32" => op::SHIFT_LEFT_LOGICAL,
                    "v_lshrrev_b32" => op::SHIFT_RIGHT_LOGICAL,
                    _ => op::SHIFT_RIGHT_ARITHMETIC,
                };
                let x = self.spv(o, Ty::U32, &[a, s]);
                f(self, x)
            }
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => self.op_lshlrev_b64(inst, name, v, a_exec),
            "v_lshl_add_u32" | "v_add_lshl_u32" | "v_lshl_or_b32" | "v_and_or_b32" | "v_or3_b32" | "v_add3_u32"
            | "v_xad_u32" | "v_xor3_b32" => {
                let a = self.usrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                let c = self.usrc(inst, v, 2)?;
                let k31 = self.k32(31);
                let x = match name {
                    "v_lshl_add_u32" => {
                        let s = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                        let t = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[a, s]);
                        self.spv(op::I_ADD, Ty::U32, &[t, c])
                    }
                    "v_add_lshl_u32" => {
                        let t = self.spv(op::I_ADD, Ty::U32, &[a, b]);
                        let s = self.spv(op::BITWISE_AND, Ty::U32, &[c, k31]);
                        self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[t, s])
                    }
                    "v_lshl_or_b32" => {
                        let s = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                        let t = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[a, s]);
                        self.spv(op::BITWISE_OR, Ty::U32, &[t, c])
                    }
                    "v_and_or_b32" => {
                        let t = self.spv(op::BITWISE_AND, Ty::U32, &[a, b]);
                        self.spv(op::BITWISE_OR, Ty::U32, &[t, c])
                    }
                    "v_or3_b32" => {
                        let t = self.spv(op::BITWISE_OR, Ty::U32, &[a, b]);
                        self.spv(op::BITWISE_OR, Ty::U32, &[t, c])
                    }
                    "v_xor3_b32" => {
                        let t = self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]);
                        self.spv(op::BITWISE_XOR, Ty::U32, &[t, c])
                    }
                    "v_add3_u32" => {
                        let t = self.spv(op::I_ADD, Ty::U32, &[a, b]);
                        self.spv(op::I_ADD, Ty::U32, &[t, c])
                    }
                    _ => {
                        let t = self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]);
                        self.spv(op::I_ADD, Ty::U32, &[t, c])
                    }
                };
                f(self, x)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let a = self.usrc(inst, v, 0)?;
                let o = self.usrc(inst, v, 1)?;
                let w = self.usrc(inst, v, 2)?;
                let k31 = self.k32(31);
                let o = self.spv(op::BITWISE_AND, Ty::U32, &[o, k31]);
                let w = self.spv(op::BITWISE_AND, Ty::U32, &[w, k31]);
                let x = self.bfe(a, o, w, name == "v_bfe_i32");
                f(self, x)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_int2(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_bfi_b32" => {
                let m = self.usrc(inst, v, 0)?;
                let a = self.usrc(inst, v, 1)?;
                let b = self.usrc(inst, v, 2)?;
                let t = self.spv(op::BITWISE_AND, Ty::U32, &[m, a]);
                let nm = self.spv(op::NOT, Ty::U32, &[m]);
                let u = self.spv(op::BITWISE_AND, Ty::U32, &[nm, b]);
                let x = self.spv(op::BITWISE_OR, Ty::U32, &[t, u]);
                f(self, x)
            }
            "v_bfm_b32" => {
                let w = self.usrc(inst, v, 0)?;
                let o = self.usrc(inst, v, 1)?;
                let x = self.bfm(w, o);
                f(self, x)
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => self.op_alignbit_b32(inst, name, v, a_exec),
            "v_perm_b32" => self.op_perm_b32(inst, name, v, a_exec),
            "v_bcnt_u32_b32" => {
                let a = self.usrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                let c = self.spv(op::BIT_COUNT, Ty::U32, &[a]);
                let x = self.spv(op::I_ADD, Ty::U32, &[c, b]);
                f(self, x)
            }
            "v_ffbl_b32" => {
                let a = self.usrc(inst, v, 0)?;
                let x = self.gl(glsl::FIND_I_LSB, Ty::U32, &[a]);
                f(self, x)
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_int3(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_ffbh_u32" => {
                let a = self.usrc(inst, v, 0)?;
                let x = self.clz(a);
                f(self, x)
            }
            "v_ffbh_i32" => {
                let a = self.usrc(inst, v, 0)?;
                let m = self.gl(glsl::FIND_S_MSB, Ty::U32, &[a]);
                let k = self.k32(31);
                let r = self.spv(op::I_SUB, Ty::U32, &[k, m]);
                let neg = self.k32(u32::MAX);
                let is = self.spv(op::I_EQUAL, Ty::BOOL, &[m, neg]);
                let x = self.spv(op::SELECT, Ty::U32, &[is, neg, r]);
                f(self, x)
            }
            "v_bfrev_b32" => {
                let a = self.usrc(inst, v, 0)?;
                let x = self.spv(op::BIT_REVERSE, Ty::U32, &[a]);
                f(self, x)
            }
            "v_min_u32" | "v_max_u32" | "v_min_i32" | "v_max_i32" => {
                let g = match name {
                    "v_min_u32" => glsl::U_MIN,
                    "v_max_u32" => glsl::U_MAX,
                    "v_min_i32" => glsl::S_MIN,
                    _ => glsl::S_MAX,
                };
                let a = self.usrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                let x = self.gl(g, Ty::U32, &[a, b]);
                f(self, x)
            }
            "v_min3_u32" | "v_max3_u32" | "v_min3_i32" | "v_max3_i32" | "v_med3_u32" | "v_med3_i32" => self.op_min3_u32(inst, name, v, a_exec),
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_carry0(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32"
            | "v_subrev_co_ci_u32" => {
                let mut a = self.usrc(inst, v, 0)?;
                let mut b = self.usrc(inst, v, 1)?;
                if name.starts_with("v_subrev") {
                    std::mem::swap(&mut a, &mut b);
                }
                let cin = if name.contains("_ci_") {
                    let m = self.opnd_mask(inst, &v.src[2])?;
                    Some(m)
                } else {
                    None
                };
                let (x, carry) = if name.contains("add") {
                    let s = self.spv(op::I_ADD, Ty::U32, &[a, b]);
                    let c1 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[s, a]);
                    match cin {
                        Some(ci) => {
                            let cu = self.u32v(ci);
                            let s2 = self.spv(op::I_ADD, Ty::U32, &[s, cu]);
                            let c2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[s2, s]);
                            (s2, self.spv(op::LOGICAL_OR, Ty::BOOL, &[c1, c2]))
                        }
                        None => (s, c1),
                    }
                } else {
                    let d = self.spv(op::I_SUB, Ty::U32, &[a, b]);
                    let b1 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[a, b]);
                    match cin {
                        Some(ci) => {
                            let cu = self.u32v(ci);
                            let b2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[d, cu]);
                            let d2 = self.spv(op::I_SUB, Ty::U32, &[d, cu]);
                            (d2, self.spv(op::LOGICAL_OR, Ty::BOOL, &[b1, b2]))
                        }
                        None => (d, b1),
                    }
                };
                let c = self.spv(op::LOGICAL_AND, Ty::BOOL, &[a_exec, carry]);
                self.mask_def(inst, &v.carry, c);
                f(self, x)
            }
            "v_mad_u64_u32" | "v_mad_i64_i32" => self.op_mad_u64_u32(inst, name, v, a_exec),
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn valu_lane0(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let _ = (off, a_exec);
        let f = |l: &mut Self, x: V| -> Res<()> { l.vdst(inst, v, x) };
        let _ = &f;
        match name {
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => self.op_mbcnt_lo_u32_b32(inst, name, v, a_exec),
            "v_readfirstlane_b32" => {
                let x = self.usrc(inst, v, 0)?;
                let r = if self.f.as_const(a_exec) == Some(1) {
                    self.emit(Op::Wave(op::GROUP_NON_UNIFORM_BROADCAST_FIRST, u32::MAX), Ty::U32, vec![x])
                } else {
                    let b = self.ballot(a_exec);
                    let l = self.gl(glsl::FIND_I_LSB, Ty::U32, &[b]);
                    let k31 = self.k32(31);
                    let l = self.spv(op::BITWISE_AND, Ty::U32, &[l, k31]);
                    self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![x, l])
                };
                self.sdst(inst, &v.dst, 0, R::D(r));
                Ok(())
            }
            "v_readlane_b32" => {
                let x = self.usrc(inst, v, 0)?;
                let l = self.usrc(inst, v, 1)?;
                if self.wave64 && self.f.as_const(l).is_none_or(|c| c >= 32) {
                    self.approx("v_readlane_b32 of a lane in the other half of a wave64");
                }
                let k31 = self.k32(31);
                let l = self.spv(op::BITWISE_AND, Ty::U32, &[l, k31]);
                let r = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![x, l]);
                self.sdst(inst, &v.dst, 0, R::D(r));
                Ok(())
            }
            "v_writelane_b32" => self.op_writelane_b32(inst, name, v, a_exec),
            "v_permlanex16_b32" | "v_permlane16_b32" => self.op_permlanex16_b32(inst, name, v, a_exec),
            _ => Err(unsupported(off, name)),
        }
    }

}

fn f16_to_f32(h: u16) -> f32 {
    let s = u32::from(h >> 15) << 31;
    let e = u32::from(h >> 10 & 0x1F);
    let m = u32::from(h & 0x3FF);
    let bits = match (e, m) {
        (0, 0) => s,
        (0, _) => {
            // subnormal
            let f = m as f32 / 1024.0 / 16384.0;
            return if s != 0 { -f } else { f };
        }
        (31, 0) => s | 0x7F80_0000,
        (31, _) => s | 0x7FC0_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Which family function lowers a VALU opcode.
fn valu_part(name: &str) -> Option<u8> {
    Some(match name {
        "v_nop" => 0,
        "v_mov_b32" => 0,
        "v_cndmask_b32" => 0,
        "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => 0,
        "v_movrels_b32" | "v_movreld_b32" => 0,
        "v_swap_b32" => 0,
        "v_interp_p1_f32" | "v_interp_p2_f32" | "v_interp_p1ll_f16" | "v_interp_p1lv_f16" | "v_interp_p2_f16" | "v_interp_mov_f32" => 1,
        "v_add_f32" => 2,
        "v_sub_f32" => 2,
        "v_subrev_f32" => 2,
        "v_mul_f32" => 2,
        "v_mul_legacy_f32" => 2,
        "v_fma_f32" | "v_mad_f32" | "v_fma_legacy_f32" | "v_mad_legacy_f32" => 2,
        "v_fmac_f32" | "v_mac_f32" | "v_fmac_legacy_f32" | "v_mac_legacy_f32" => 3,
        "v_fmaak_f32" | "v_madak_f32" | "v_fmamk_f32" | "v_madmk_f32" => 3,
        "v_min_f32" | "v_max_f32" => 3,
        "v_min3_f32" | "v_max3_f32" => 3,
        "v_med3_f32" => 3,
        "v_rcp_f32" | "v_rcp_iflag_f32" | "v_rcp_legacy_f32" => 3,
        "v_rsq_f32" | "v_sqrt_f32" | "v_exp_f32" | "v_log_f32" | "v_fract_f32" | "v_floor_f32" | "v_ceil_f32" | "v_trunc_f32" | "v_rndne_f32" => 4,
        "v_sin_f32" | "v_cos_f32" => 4,
        "v_ldexp_f32" => 4,
        "v_frexp_mant_f32" | "v_frexp_exp_i32_f32" => 4,
        "v_mov_b16" => 5,
        "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => 5,
        "v_add_f16" | "v_sub_f16" | "v_subrev_f16" | "v_mul_f16" | "v_max_f16" | "v_min_f16" => 5,
        "v_fma_f16" | "v_mad_f16" | "v_fmac_f16" | "v_med3_f16" => 5,
        "v_rcp_f16" | "v_rsq_f16" | "v_sqrt_f16" | "v_exp_f16" | "v_log_f16" | "v_sin_f16" | "v_cos_f16" | "v_trunc_f16" | "v_floor_f16" | "v_ceil_f16" | "v_fract_f16" | "v_rndne_f16" => 5,
        "v_pk_fma_f16" | "v_pk_mul_f16" | "v_pk_add_f16" | "v_pk_max_f16" | "v_pk_min_f16" => 5,
        "v_lshrrev_b16" | "v_lshlrev_b16" | "v_ashrrev_i16" | "v_add_nc_u16" | "v_sub_nc_u16" | "v_mul_lo_u16" => 6,
        "v_cvt_f32_u32" | "v_cvt_f32_i32" => 7,
        "v_cvt_u32_f32" | "v_cvt_i32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => 7,
        "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1" | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => 7,
        "v_cvt_f32_f16" => 7,
        "v_cvt_f16_f32" => 7,
        "v_cvt_pkrtz_f16_f32" | "v_cvt_pknorm_u16_f32" | "v_cvt_pknorm_i16_f32" => 7,
        "v_pack_b32_f16" => 8,
        "v_cvt_pk_u8_f32" => 8,
        "v_add_nc_u32" | "v_add_u32" => 9,
        "v_sub_nc_u32" | "v_sub_u32" => 9,
        "v_subrev_nc_u32" | "v_subrev_u32" => 9,
        "v_mul_lo_u32" | "v_mul_lo_i32" => 9,
        "v_mul_hi_u32" | "v_mul_hi_i32" => 9,
        "v_mul_u32_u24" | "v_mul_i32_i24" | "v_mad_u32_u24" | "v_mad_i32_i24" | "v_mul_hi_u32_u24" => 9,
        "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_xnor_b32" => 10,
        "v_not_b32" => 10,
        "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => 10,
        "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => 10,
        "v_lshl_add_u32" | "v_add_lshl_u32" | "v_lshl_or_b32" | "v_and_or_b32" | "v_or3_b32" | "v_add3_u32" | "v_xad_u32" | "v_xor3_b32" => 10,
        "v_bfe_u32" | "v_bfe_i32" => 10,
        "v_bfi_b32" => 11,
        "v_bfm_b32" => 11,
        "v_alignbit_b32" | "v_alignbyte_b32" => 11,
        "v_perm_b32" => 11,
        "v_bcnt_u32_b32" => 11,
        "v_ffbl_b32" => 11,
        "v_ffbh_u32" => 12,
        "v_ffbh_i32" => 12,
        "v_bfrev_b32" => 12,
        "v_min_u32" | "v_max_u32" | "v_min_i32" | "v_max_i32" => 12,
        "v_min3_u32" | "v_max3_u32" | "v_min3_i32" | "v_max3_i32" | "v_med3_u32" | "v_med3_i32" => 12,
        "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => 13,
        "v_mad_u64_u32" | "v_mad_i64_i32" => 13,
        "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => 14,
        "v_readfirstlane_b32" => 14,
        "v_readlane_b32" => 14,
        "v_writelane_b32" => 14,
        "v_permlanex16_b32" | "v_permlane16_b32" => 14,
        _ => return None,
    })
}

impl Lift<'_> {
    #[allow(unused_variables)]
    fn op_cndmask_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_fmac_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_med3_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_fma_mix_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_fma_f16(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_pk_fma_f16(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_lshrrev_b16(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_cvt_pk_u8_f32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_mul_u32_u24(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_lshlrev_b64(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_alignbit_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_perm_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_min3_u32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_mad_u64_u32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_mbcnt_lo_u32_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_writelane_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
    fn op_permlanex16_b32(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
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
