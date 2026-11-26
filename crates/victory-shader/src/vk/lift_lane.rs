//! Cross-lane and lane-indexed VALU instructions; compares; cube maps.

use super::*;

impl Lift<'_> {
    pub(super) fn fcmp(&mut self, rel: &str) -> Option<u16> {
        Some(match rel {
            "lt" => op::F_ORD_LESS_THAN,
            "eq" => op::F_ORD_EQUAL,
            "le" => op::F_ORD_LESS_THAN_EQUAL,
            "gt" => op::F_ORD_GREATER_THAN,
            "lg" => op::F_ORD_NOT_EQUAL,
            "ge" => op::F_ORD_GREATER_THAN_EQUAL,
            "nge" => op::F_UNORD_LESS_THAN,
            "nlg" => op::F_UNORD_EQUAL,
            "ngt" => op::F_UNORD_LESS_THAN_EQUAL,
            "nle" => op::F_UNORD_GREATER_THAN,
            "neq" => op::F_UNORD_NOT_EQUAL,
            "nlt" => op::F_UNORD_GREATER_THAN_EQUAL,
            _ => return None,
        })
    }

    /// `v_cmp_class`: whether `x` is in one of the classes of `mask`.
    pub(super) fn class(&mut self, x: V, mask: V, t: Ty) -> V {
        let Some(m) = self.f.as_const(mask) else {
            // dynamic class masks: rare; treat as "not in class"
            self.approx("v_cmp_class with a computed class mask");
            return self.kb(false);
        };
        let z = if t == Ty::F16 { self.f.konst(t, 0) } else { self.kf(0.0) };
        let nan = self.spv(op::IS_NAN, Ty::BOOL, &[x]);
        let inf = self.spv(op::IS_INF, Ty::BOOL, &[x]);
        let neg = self.spv(op::F_ORD_LESS_THAN, Ty::BOOL, &[x, z]);
        let is0 = self.spv(op::F_ORD_EQUAL, Ty::BOOL, &[x, z]);
        // sign bit (for zeros)
        let bits = if t == Ty::F16 {
            let b = self.spv(op::BITCAST, Ty::U16, &[x]);
            self.spv(op::U_CONVERT, Ty::U32, &[b])
        } else {
            self.u32v(x)
        };
        let sb = self.k32(if t == Ty::F16 { 0x8000 } else { 0x8000_0000 });
        let sbit = self.spv(op::BITWISE_AND, Ty::U32, &[bits, sb]);
        let zz = self.k32(0);
        let signed = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[sbit, zz]);
        let absx = self.gl(glsl::F_ABS, t, &[x]);
        let minn = if t == Ty::F16 { self.f.konst(t, 0x0400) } else { self.kf(f32::MIN_POSITIVE) };
        let denorm_mag = self.spv(op::F_ORD_LESS_THAN, Ty::BOOL, &[absx, minn]);
        let nz = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[is0]);
        let denorm = self.spv(op::LOGICAL_AND, Ty::BOOL, &[denorm_mag, nz]);
        let finite = {
            let a = self.spv(op::LOGICAL_OR, Ty::BOOL, &[nan, inf]);
            self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a])
        };
        let normal = {
            let nd = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[denorm_mag]);
            self.spv(op::LOGICAL_AND, Ty::BOOL, &[finite, nd])
        };
        let pos = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[signed]);
        let mut acc = self.kb(false);
        for bit in 0..10 {
            if m >> bit & 1 == 0 {
                continue;
            }
            let t = match bit {
                0 | 1 => nan,
                2 => self.spv(op::LOGICAL_AND, Ty::BOOL, &[inf, neg]),
                3 => self.spv(op::LOGICAL_AND, Ty::BOOL, &[normal, neg]),
                4 => self.spv(op::LOGICAL_AND, Ty::BOOL, &[denorm, signed]),
                5 => self.spv(op::LOGICAL_AND, Ty::BOOL, &[is0, signed]),
                6 => self.spv(op::LOGICAL_AND, Ty::BOOL, &[is0, pos]),
                7 => self.spv(op::LOGICAL_AND, Ty::BOOL, &[denorm, pos]),
                8 => {
                    let p = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[neg]);
                    let n = self.spv(op::LOGICAL_AND, Ty::BOOL, &[normal, p]);
                    self.spv(op::LOGICAL_AND, Ty::BOOL, &[n, nz])
                }
                _ => {
                    let p = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[neg]);
                    self.spv(op::LOGICAL_AND, Ty::BOOL, &[inf, p])
                }
            };
            acc = self.spv(op::LOGICAL_OR, Ty::BOOL, &[acc, t]);
        }
        acc
    }

    pub(super) fn compare(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let x = name.starts_with("v_cmpx_");
        let rest = name.trim_start_matches("v_cmpx_").trim_start_matches("v_cmp_");
        let (rel, ty) = rest.rsplit_once('_').ok_or(unsupported(off, name))?;
        let c = match (rel, ty) {
            ("class", "f32") => {
                let a = self.fsrc(inst, v, 0)?;
                let m = self.usrc(inst, v, 1)?;
                self.class(a, m, Ty::F32)
            }
            ("class", "f16") => {
                let a = self.f16src(inst, v, 0)?;
                let m = self.usrc(inst, v, 1)?;
                self.class(a, m, Ty::F16)
            }
            ("f", _) => self.kb(false),
            ("t" | "tru", _) => self.kb(true),
            ("o" | "u", "f32") => {
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                let na = self.spv(op::IS_NAN, Ty::BOOL, &[a]);
                let nb = self.spv(op::IS_NAN, Ty::BOOL, &[b]);
                let u = self.spv(op::LOGICAL_OR, Ty::BOOL, &[na, nb]);
                if rel == "u" { u } else { self.spv(op::LOGICAL_NOT, Ty::BOOL, &[u]) }
            }
            (_, "f32") => {
                let o = self.fcmp(rel).ok_or(unsupported(off, name))?;
                let a = self.fsrc(inst, v, 0)?;
                let b = self.fsrc(inst, v, 1)?;
                self.spv(o, Ty::BOOL, &[a, b])
            }
            (_, "f16") => {
                let o = self.fcmp(rel).ok_or(unsupported(off, name))?;
                let a = self.f16src(inst, v, 0)?;
                let b = self.f16src(inst, v, 1)?;
                self.spv(o, Ty::BOOL, &[a, b])
            }
            (_, "u32" | "i32") => {
                let o = int_cmp(rel, ty == "i32").ok_or(unsupported(off, name))?;
                let a = self.usrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                self.spv(o, Ty::BOOL, &[a, b])
            }
            (_, "u16" | "i16") => {
                let o = int_cmp(rel, ty == "i16").ok_or(unsupported(off, name))?;
                let a = self.usrc(inst, v, 0)?;
                let b = self.usrc(inst, v, 1)?;
                let opc = if ty == "i16" { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
                let (z, s) = (self.k32(0), self.k32(16));
                let a = self.spv(opc, Ty::U32, &[a, z, s]);
                let b = self.spv(opc, Ty::U32, &[b, z, s]);
                self.spv(o, Ty::BOOL, &[a, b])
            }
            (_, "u64" | "i64") => {
                let o = int_cmp(rel, ty == "i64").ok_or(unsupported(off, name))?;
                let a = self.opnd_u64(inst, &v.src[0])?;
                let b = self.opnd_u64(inst, &v.src[1])?;
                self.spv(o, Ty::BOOL, &[a, b])
            }
            _ => return Err(unsupported(off, name)),
        };
        let r = self.spv(op::LOGICAL_AND, Ty::BOOL, &[a_exec, c]);
        if x {
            self.def(inst, Reg::S(ir::EXEC), R::M(r, Half::Lo));
            if self.wave64 {
                self.def(inst, Reg::S(ir::EXEC + 1), R::M(r, Half::Hi));
            }
        } else {
            self.mask_def(inst, &v.dst, r);
        }
        Ok(())
    }

    /// The cube-map face and coordinates (`v_cube*`), as the hardware
    /// computes them.
    pub(super) fn cube(&mut self, name: &str, x: V, y: V, z: V) -> V {
        let ax = self.gl(glsl::F_ABS, Ty::F32, &[x]);
        let ay = self.gl(glsl::F_ABS, Ty::F32, &[y]);
        let az = self.gl(glsl::F_ABS, Ty::F32, &[z]);
        let zmaj = {
            let a = self.spv(op::F_ORD_GREATER_THAN_EQUAL, Ty::BOOL, &[az, ax]);
            let b = self.spv(op::F_ORD_GREATER_THAN_EQUAL, Ty::BOOL, &[az, ay]);
            self.spv(op::LOGICAL_AND, Ty::BOOL, &[a, b])
        };
        let ymaj = {
            let nz = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[zmaj]);
            let a = self.spv(op::F_ORD_GREATER_THAN_EQUAL, Ty::BOOL, &[ay, ax]);
            self.spv(op::LOGICAL_AND, Ty::BOOL, &[nz, a])
        };
        let zero = self.kf(0.0);
        let neg = |l: &mut Self, v: V| l.spv(op::F_ORD_LESS_THAN, Ty::BOOL, &[v, zero]);
        match name {
            "v_cubeid_f32" => {
                let (k0, k1, k2, k3, k4, k5) = (self.kf(0.0), self.kf(1.0), self.kf(2.0), self.kf(3.0), self.kf(4.0), self.kf(5.0));
                let zn = neg(self, z);
                let yn = neg(self, y);
                let xn = neg(self, x);
                let fz = self.spv(op::SELECT, Ty::F32, &[zn, k5, k4]);
                let fy = self.spv(op::SELECT, Ty::F32, &[yn, k3, k2]);
                let fx = self.spv(op::SELECT, Ty::F32, &[xn, k1, k0]);
                let t = self.spv(op::SELECT, Ty::F32, &[ymaj, fy, fx]);
                self.spv(op::SELECT, Ty::F32, &[zmaj, fz, t])
            }
            "v_cubema_f32" => {
                let two = self.kf(2.0);
                let t = self.spv(op::SELECT, Ty::F32, &[ymaj, y, x]);
                let m = self.spv(op::SELECT, Ty::F32, &[zmaj, z, t]);
                self.spv(op::F_MUL, Ty::F32, &[two, m])
            }
            "v_cubesc_f32" => {
                // z major: z<0 ? -x : x; y major: x; x major: x<0 ? z : -z
                let nx = self.spv(op::F_NEGATE, Ty::F32, &[x]);
                let nzv = self.spv(op::F_NEGATE, Ty::F32, &[z]);
                let zn = neg(self, z);
                let xn = neg(self, x);
                let sz = self.spv(op::SELECT, Ty::F32, &[zn, nx, x]);
                let sx = self.spv(op::SELECT, Ty::F32, &[xn, z, nzv]);
                let t = self.spv(op::SELECT, Ty::F32, &[ymaj, x, sx]);
                self.spv(op::SELECT, Ty::F32, &[zmaj, sz, t])
            }
            _ => {
                // tc: z major: -y; y major: y<0 ? -z : z; x major: -y
                let ny = self.spv(op::F_NEGATE, Ty::F32, &[y]);
                let nzv = self.spv(op::F_NEGATE, Ty::F32, &[z]);
                let yn = neg(self, y);
                let sy = self.spv(op::SELECT, Ty::F32, &[yn, nzv, z]);
                self.spv(op::SELECT, Ty::F32, &[ymaj, sy, ny])
            }
        }
    }

    pub(super) fn movrel(&mut self, inst: &Inst, name: &'static str, v: &Valu) -> Res<()> {
        let off = inst.machine.offset;
        let m0 = self.val(inst, Reg::S(ir::M0))?;
        let m0 = self.data(m0);
        let m0 = self.u32v(m0);
        let vgprs = self.p.vgprs as u32;
        if name == "v_movrels_b32" {
            let Operand::Vector { reg: base, .. } = v.src[0] else { return Err(unsupported(off, name)) };
            let n = vgprs - u32::from(base);
            let mut args = vec![m0];
            for k in 0..n {
                let r = self.val(inst, Reg::V(base + k as u8))?;
                let d = self.data(r);
                args.push(self.u32v(d));
            }
            let x = self.emit(Op::Pick, Ty::U32, args);
            self.vdst(inst, v, x)
        } else {
            let Some(Operand::Vector { reg: base, .. }) = v.dst else { return Err(unsupported(off, name)) };
            let x = self.usrc(inst, v, 0)?;
            let n = vgprs - u32::from(base);
            for k in 0..n {
                let reg = Reg::V(base + k as u8);
                if inst.write(reg).is_none() {
                    continue;
                }
                let old = self.val(inst, reg)?;
                let old = self.data(old);
                let old = self.u32v(old);
                let kk = self.k32(k);
                let me = self.spv(op::I_EQUAL, Ty::BOOL, &[m0, kk]);
                let y = self.spv(op::SELECT, Ty::U32, &[me, x, old]);
                self.def(inst, reg, R::D(y));
            }
            Ok(())
        }
    }

    /// A DPP source: the source lane of each lane, and whether it reads.
    pub(super) fn dpp(&mut self, inst: &Inst, name: &'static str, v: &Valu, a_exec: V) -> Res<()> {
        let off = inst.machine.offset;
        let lane = self.lane();
        let (from, valid, row_mask, bank_mask, bound_ctrl, fi) = match v.dpp.unwrap() {
            Dpp::Lanes { sel, fi } => {
                let k7 = self.k32(7);
                let i = self.spv(op::BITWISE_AND, Ty::U32, &[lane, k7]);
                let three = self.k32(3);
                let sh = self.spv(op::I_MUL, Ty::U32, &[i, three]);
                let ks = self.k32(sel);
                let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[ks, sh]);
                let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k7]);
                let nb = self.k32(!7);
                let b = self.spv(op::BITWISE_AND, Ty::U32, &[lane, nb]);
                let from = self.spv(op::BITWISE_OR, Ty::U32, &[b, s]);
                (from, None, 0xF, 0xF, false, fi)
            }
            Dpp::Row { ctrl, row_mask, bank_mask, bound_ctrl, fi } => {
                let c = u32::from(ctrl);
                let k15 = self.k32(15);
                let i = self.spv(op::BITWISE_AND, Ty::U32, &[lane, k15]);
                let nrow = self.k32(!15);
                let rowb = self.spv(op::BITWISE_AND, Ty::U32, &[lane, nrow]);
                let (from, valid) = match c {
                    0x000..=0x0FF => {
                        // quad_perm
                        let k3 = self.k32(3);
                        let q = self.spv(op::BITWISE_AND, Ty::U32, &[lane, k3]);
                        let one = self.k32(1);
                        let q2 = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[q, one]);
                        let kc = self.k32(c);
                        let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[kc, q2]);
                        let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k3]);
                        let nq = self.k32(!3);
                        let b = self.spv(op::BITWISE_AND, Ty::U32, &[lane, nq]);
                        (self.spv(op::BITWISE_OR, Ty::U32, &[b, s]), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        // row_shl / row_shr
                        let n = c & 0xF;
                        let kn = self.k32(n);
                        let (j, ok) = if c < 0x110 {
                            let j = self.spv(op::I_ADD, Ty::U32, &[i, kn]);
                            let ok = self.spv(op::U_LESS_THAN_EQUAL, Ty::BOOL, &[j, k15]);
                            (j, ok)
                        } else {
                            let ok = self.spv(op::U_GREATER_THAN_EQUAL, Ty::BOOL, &[i, kn]);
                            (self.spv(op::I_SUB, Ty::U32, &[i, kn]), ok)
                        };
                        let j = self.spv(op::BITWISE_AND, Ty::U32, &[j, k15]);
                        (self.spv(op::BITWISE_OR, Ty::U32, &[rowb, j]), Some(ok))
                    }
                    0x121..=0x12F => {
                        // row_ror
                        let kn = self.k32(c & 0xF);
                        let j = self.spv(op::I_ADD, Ty::U32, &[i, kn]);
                        let j = self.spv(op::BITWISE_AND, Ty::U32, &[j, k15]);
                        (self.spv(op::BITWISE_OR, Ty::U32, &[rowb, j]), None)
                    }
                    0x140 => {
                        let j = self.spv(op::BITWISE_XOR, Ty::U32, &[i, k15]);
                        (self.spv(op::BITWISE_OR, Ty::U32, &[rowb, j]), None)
                    }
                    0x141 => {
                        let k7 = self.k32(7);
                        let j = self.spv(op::BITWISE_XOR, Ty::U32, &[i, k7]);
                        (self.spv(op::BITWISE_OR, Ty::U32, &[rowb, j]), None)
                    }
                    0x150..=0x15F => {
                        // row_share: lane n of the row
                        let kn = self.k32(c & 0xF);
                        (self.spv(op::BITWISE_OR, Ty::U32, &[rowb, kn]), None)
                    }
                    0x160..=0x16F => {
                        // row_xmask
                        let kn = self.k32(c & 0xF);
                        let j = self.spv(op::BITWISE_XOR, Ty::U32, &[i, kn]);
                        (self.spv(op::BITWISE_OR, Ty::U32, &[rowb, j]), None)
                    }
                    _ => return Err(unsupported(off, "a DPP control")),
                };
                (from, valid, row_mask, bank_mask, bound_ctrl, fi)
            }
        };
        let k31 = self.k32(31);
        let from = self.spv(op::BITWISE_AND, Ty::U32, &[from, k31]);
        let x = self.usrc(inst, v, 0)?;
        let shuffled = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![x, from]);
        // an inactive source lane (fi clear) or an invalid one reads 0 (bound_ctrl) or skips
        let mut ok = valid.unwrap_or_else(|| self.kb(true));
        if !fi && self.f.as_const(a_exec) != Some(1) {
            let sa = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::BOOL, vec![a_exec, from]);
            ok = self.spv(op::LOGICAL_AND, Ty::BOOL, &[ok, sa]);
        }
        let z = self.k32(0);
        let src0 = self.spv(op::SELECT, Ty::U32, &[ok, shuffled, z]);
        // write enable: row/bank masks; a lane whose source is invalid without bound_ctrl keeps dst
        let mut we = self.kb(true);
        if row_mask != 0xF || bank_mask != 0xF {
            let k4 = self.k32(4);
            let row = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[lane, k4]);
            let k3 = self.k32(3);
            let row = self.spv(op::BITWISE_AND, Ty::U32, &[row, k3]);
            let rm = self.k32(u32::from(row_mask));
            let rb = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[rm, row]);
            let k2 = self.k32(2);
            let bank = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[lane, k2]);
            let bank = self.spv(op::BITWISE_AND, Ty::U32, &[bank, k3]);
            let bm = self.k32(u32::from(bank_mask));
            let bb = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[bm, bank]);
            let both = self.spv(op::BITWISE_AND, Ty::U32, &[rb, bb]);
            let one = self.k32(1);
            let bit = self.spv(op::BITWISE_AND, Ty::U32, &[both, one]);
            we = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[bit, z]);
        }
        if !bound_ctrl {
            we = self.spv(op::LOGICAL_AND, Ty::BOOL, &[we, ok]);
        }
        // compute the operation with src0 replaced
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err(unsupported(off, name)) };
        let old = self.val(inst, Reg::V(reg))?;
        let old = self.data(old);
        let old = self.u32v(old);
        let res = match name {
            "v_mov_b32" => src0,
            _ => {
                let b = self.usrc(inst, v, 1)?;
                let a = src0;
                match name {
                    "v_or_b32" => self.spv(op::BITWISE_OR, Ty::U32, &[a, b]),
                    "v_and_b32" => self.spv(op::BITWISE_AND, Ty::U32, &[a, b]),
                    "v_xor_b32" => self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]),
                    "v_add_nc_u32" => self.spv(op::I_ADD, Ty::U32, &[a, b]),
                    "v_max_u32" => self.gl(glsl::U_MAX, Ty::U32, &[a, b]),
                    "v_min_u32" => self.gl(glsl::U_MIN, Ty::U32, &[a, b]),
                    "v_max_i32" => self.gl(glsl::S_MAX, Ty::U32, &[a, b]),
                    "v_min_i32" => self.gl(glsl::S_MIN, Ty::U32, &[a, b]),
                    "v_add_f32" | "v_max_f32" | "v_min_f32" | "v_mul_f32" => {
                        let mut af = self.f32v(a);
                        if v.abs & 1 != 0 {
                            af = self.gl(glsl::F_ABS, Ty::F32, &[af]);
                        }
                        if v.neg & 1 != 0 {
                            af = self.spv(op::F_NEGATE, Ty::F32, &[af]);
                        }
                        let bf = self.fsrc(inst, v, 1)?;
                        let r = match name {
                            "v_add_f32" => self.spv(op::F_ADD, Ty::F32, &[af, bf]),
                            "v_mul_f32" => self.spv(op::F_MUL, Ty::F32, &[af, bf]),
                            "v_max_f32" => self.gl(glsl::N_MAX, Ty::F32, &[af, bf]),
                            _ => self.gl(glsl::N_MIN, Ty::F32, &[af, bf]),
                        };
                        self.u32v(r)
                    }
                    _ => return Err(unsupported(off, "a DPP operation")),
                }
            }
        };
        let x = self.spv(op::SELECT, Ty::U32, &[we, res, old]);
        self.def(inst, Reg::V(reg), R::D(x));
        Ok(())
    }
}
