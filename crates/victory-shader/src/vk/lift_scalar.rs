//! Scalar ALU instructions.

use super::*;

impl Lift<'_> {
    pub(super) fn sop(&mut self, inst: &Inst, name: &'static str, dst: &Option<Operand>, src: &[Operand], imm: Option<u16>) -> Res<()> {
        let off = inst.machine.offset;
        let b64 = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        let mask_like = |l: &Self, o: &Operand| -> bool {
            if let Operand::Scalar { code, .. } = *o
                && code != crate::gfx10::NULL
                && let Some(v) = inst.read(Reg::S(code))
            {
                return matches!(l.vals.get(&v), Some(R::M(..)))
                    || (code == ir::EXEC && !l.vals.contains_key(&v));
            }
            false
        };
        let dst_exec = matches!(dst, Some(Operand::Scalar { code, .. }) if *code == ir::EXEC || *code == ir::VCC);
        let stem = name.trim_start_matches("s_");
        let mut base = stem;
        while let Some((a, t)) = base.rsplit_once('_') {
            if matches!(t, "b32" | "b64" | "i32" | "u32" | "i64" | "u64" | "b16" | "i16" | "u16") {
                base = a;
            } else {
                break;
            }
        }
        let scc_def = |l: &mut Self, b: V| {
            if inst.write(Reg::Scc).is_some() {
                l.def(inst, Reg::Scc, R::B(b));
            }
        };
        // saveexec / wrexec family
        if name.contains("saveexec") || name.contains("wrexec") {
            let a = self.active(inst)?;
            let s = self.opnd_mask(inst, &src[0])?;
            let opname = base.trim_end_matches("_saveexec").trim_end_matches("_wrexec");
            let new = self.mask_bin(opname, s, a).ok_or(unsupported(off, name))?;
            let wr = name.contains("wrexec");
            self.mask_def(inst, dst, if wr { new } else { a });
            self.def(inst, Reg::S(ir::EXEC), R::M(new, Half::Lo));
            if self.wave64 {
                self.def(inst, Reg::S(ir::EXEC + 1), R::M(new, Half::Hi));
            }
            let any = self.any(new);
            scc_def(self, any);
            return Ok(());
        }
        match base {
            "mov" | "cmov" | "movk" | "cmovk" => {
                let cond = if base.starts_with("cmov") {
                    let r = self.val(inst, Reg::Scc)?;
                    Some(self.scc_of(r))
                } else {
                    None
                };
                if let Some(k) = imm {
                    let x = self.k32(i32::from(k as i16) as u32);
                    let x = if let Some(c) = cond {
                        let old = self.opnd(inst, &dst.unwrap())?;
                        let old = self.u32v(old);
                        self.spv(op::SELECT, Ty::U32, &[c, x, old])
                    } else {
                        x
                    };
                    self.sdst(inst, dst, 0, R::D(x));
                    return Ok(());
                }
                let n = if b64 { 2 } else { 1 };
                if b64 && (mask_like(self, &src[0]) || dst_exec) && cond.is_none() {
                    let m = self.opnd_mask(inst, &src[0])?;
                    self.mask_def(inst, dst, m);
                    return Ok(());
                }
                for k in 0..n {
                    let r = match src[0] {
                        Operand::Scalar { code, .. } if code != crate::gfx10::NULL => self.val(inst, Reg::S(code + k))?,
                        _ => R::D(self.opnd_dw(inst, &src[0], k, false)?),
                    };
                    let r = if let Some(c) = cond {
                        let x = self.data(r);
                        let x = self.u32v(x);
                        let old = self.opnd_dw(inst, &dst.unwrap(), k, false)?;
                        let old = self.u32v(old);
                        R::D(self.spv(op::SELECT, Ty::U32, &[c, x, old]))
                    } else {
                        r
                    };
                    self.sdst(inst, dst, k, r);
                }
                Ok(())
            }
            "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not" | "wqm" => {
                let unary = matches!(base, "not" | "wqm");
                let maskish = src.iter().any(|o| mask_like(self, o)) || dst_exec;
                if maskish || base == "wqm" {
                    let a = self.opnd_mask(inst, &src[0])?;
                    let r = if base == "not" {
                        self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a])
                    } else if base == "wqm" {
                        self.wqm(a)
                    } else {
                        let b = self.opnd_mask(inst, &src[1])?;
                        self.mask_bin(base, a, b).ok_or(unsupported(off, name))?
                    };
                    self.mask_def(inst, dst, r);
                    let any = self.any(r);
                    scc_def(self, any);
                    return Ok(());
                }
                let n = if b64 { 2 } else { 1 };
                let mut nz = None;
                for k in 0..n {
                    let a = self.opnd_dw(inst, &src[0], k, false)?;
                    let a = self.u32v(a);
                    let r = if unary {
                        self.spv(op::NOT, Ty::U32, &[a])
                    } else {
                        let b = self.opnd_dw(inst, &src[1], k, false)?;
                        let b = self.u32v(b);
                        self.int_bin(base, a, b).ok_or(unsupported(off, name))?
                    };
                    self.sdst(inst, dst, k, R::D(r));
                    nz = Some(match nz {
                        None => r,
                        Some(p) => self.spv(op::BITWISE_OR, Ty::U32, &[p, r]),
                    });
                }
                let z = self.k32(0);
                let b = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[nz.unwrap(), z]);
                scc_def(self, b);
                Ok(())
            }
            "cselect" => {
                let r = self.val(inst, Reg::Scc)?;
                let c = self.scc_of(r);
                if b64 && (src.iter().any(|o| mask_like(self, o)) || dst_exec) {
                    let a = self.opnd_mask(inst, &src[0])?;
                    let b = self.opnd_mask(inst, &src[1])?;
                    let m = self.spv(op::SELECT, Ty::BOOL, &[c, a, b]);
                    self.mask_def(inst, dst, m);
                    return Ok(());
                }
                for k in 0..if b64 { 2 } else { 1 } {
                    let a = self.opnd_dw(inst, &src[0], k, false)?;
                    let b = self.opnd_dw(inst, &src[1], k, false)?;
                    let (a, b) = (self.u32v(a), self.u32v(b));
                    let x = self.spv(op::SELECT, Ty::U32, &[c, a, b]);
                    self.sdst(inst, dst, k, R::D(x));
                }
                Ok(())
            }
            "bcnt1" | "bcnt0" => {
                let mut sum = self.k32(0);
                for k in 0..if b64 { 2 } else { 1 } {
                    let a = self.opnd_dw(inst, &src[0], k, false)?;
                    let a = self.u32v(a);
                    let a = if base == "bcnt0" { self.spv(op::NOT, Ty::U32, &[a]) } else { a };
                    let c = self.spv(op::BIT_COUNT, Ty::U32, &[a]);
                    sum = self.spv(op::I_ADD, Ty::U32, &[sum, c]);
                }
                if b64 && self.wave64 {
                    self.approx("a wave64 mask counted (s_bcnt1_i32_b64)");
                }
                self.sdst(inst, dst, 0, R::D(sum));
                let z = self.k32(0);
                let b = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[sum, z]);
                scc_def(self, b);
                Ok(())
            }
            "ff1" | "ff0" => {
                let a = self.opnd_dw(inst, &src[0], 0, false)?;
                let mut a = self.u32v(a);
                if base == "ff0" {
                    a = self.spv(op::NOT, Ty::U32, &[a]);
                }
                if b64 && self.wave64 {
                    self.approx("a wave64 mask searched (s_ff1_i32_b64)");
                }
                let x = self.gl(glsl::FIND_I_LSB, Ty::U32, &[a]);
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "flbit" => {
                let a = self.opnd(inst, &src[0])?;
                let a = self.u32v(a);
                let x = if name.ends_with("i32_i32") {
                    // count of leading sign bits
                    let m = self.gl(glsl::FIND_S_MSB, Ty::U32, &[a]);
                    let k = self.k32(31);
                    let r = self.spv(op::I_SUB, Ty::U32, &[k, m]);
                    let neg = self.k32(u32::MAX);
                    let is = self.spv(op::I_EQUAL, Ty::BOOL, &[m, neg]);
                    self.spv(op::SELECT, Ty::U32, &[is, neg, r])
                } else {
                    self.clz(a)
                };
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "brev" => {
                let a = self.opnd(inst, &src[0])?;
                let a = self.u32v(a);
                let x = self.spv(op::BIT_REVERSE, Ty::U32, &[a]);
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "abs" => {
                let a = self.opnd(inst, &src[0])?;
                let a = self.u32v(a);
                let x = self.gl(glsl::S_ABS, Ty::U32, &[a]);
                self.sdst(inst, dst, 0, R::D(x));
                let z = self.k32(0);
                let b = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                scc_def(self, b);
                Ok(())
            }
            "add" | "sub" | "addc" | "subb" | "addk" | "mul" | "mulk" | "mul_hi" | "min" | "max" | "lshl" | "lshr"
            | "ashr" | "bfe" | "bfm" | "lshl1_add" | "lshl2_add" | "lshl3_add" | "lshl4_add" | "absdiff" | "pack_ll"
            | "pack_lh" | "pack_hh" | "pack_hl" => self.salu_arith(inst, name, base, dst, src, imm, b64),
            "cmp_eq" | "cmp_lg" | "cmp_gt" | "cmp_ge" | "cmp_lt" | "cmp_le" | "cmpk_eq" | "cmpk_lg" | "cmpk_gt"
            | "cmpk_ge" | "cmpk_lt" | "cmpk_le" | "bitcmp0" | "bitcmp1" => {
                let signed = name.ends_with("i32");
                let (a, b) = if let Some(k) = imm {
                    let a = self.opnd(inst, &src[0])?;
                    let k = if signed { i32::from(k as i16) as u32 } else { u32::from(k) };
                    (self.u32v(a), self.k32(k))
                } else if b64 {
                    (self.opnd_u64(inst, &src[0])?, self.opnd_u64(inst, &src[1])?)
                } else {
                    let a = self.opnd(inst, &src[0])?;
                    let b = self.opnd(inst, &src[1])?;
                    (self.u32v(a), self.u32v(b))
                };
                let rel = base.trim_start_matches("cmpk_").trim_start_matches("cmp_");
                let c = if rel.starts_with("bitcmp") {
                    let one = self.k32(1);
                    let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, b]);
                    let bit = self.spv(op::BITWISE_AND, Ty::U32, &[s, one]);
                    let z = self.k32(if rel == "bitcmp0" { 0 } else { 1 });
                    self.spv(op::I_EQUAL, Ty::BOOL, &[bit, z])
                } else {
                    let o = int_cmp(rel, signed).ok_or(unsupported(off, name))?;
                    self.spv(o, Ty::BOOL, &[a, b])
                };
                self.def(inst, Reg::Scc, R::B(c));
                Ok(())
            }
            "bitset0" | "bitset1" => {
                let bit = self.opnd(inst, &src[0])?;
                let bit = self.u32v(bit);
                let old = self.opnd(inst, &dst.unwrap())?;
                let old = self.u32v(old);
                let one = self.k32(1);
                let k31 = self.k32(31);
                let bit = self.spv(op::BITWISE_AND, Ty::U32, &[bit, k31]);
                let m = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[one, bit]);
                let x = if base == "bitset1" {
                    self.spv(op::BITWISE_OR, Ty::U32, &[old, m])
                } else {
                    let nm = self.spv(op::NOT, Ty::U32, &[m]);
                    self.spv(op::BITWISE_AND, Ty::U32, &[old, nm])
                };
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "getpc" => {
                let x = self.k32((off + 4) as u32);
                self.sdst(inst, dst, 0, R::D(x));
                let z = self.k32(0);
                self.sdst(inst, dst, 1, R::D(z));
                Ok(())
            }
            // the call of the vertex fetch shader: its results are inputs;
            // a far jump (the block's exit already goes there)
            "swappc" | "setpc" => Ok(()),
            "setreg" | "setreg_imm32" | "setprio" | "version" | "waitcnt_vscnt" | "waitcnt_vmcnt" | "waitcnt_expcnt"
            | "waitcnt_lgkmcnt" => Ok(()),
            "getreg" => {
                let z = self.k32(0);
                self.sdst(inst, dst, 0, R::D(z));
                Ok(())
            }
            _ => Err(unsupported(off, name)),
        }
    }

    pub(super) fn wqm(&mut self, a: V) -> V {
        if self.f.as_const(a).is_some() {
            return a;
        }
        // any lane of my quad
        let b = self.ballot(a);
        let lane = self.lane();
        let m3 = self.k32(!3);
        let q = self.spv(op::BITWISE_AND, Ty::U32, &[lane, m3]);
        let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, q]);
        let f = self.k32(0xF);
        let n = self.spv(op::BITWISE_AND, Ty::U32, &[s, f]);
        let z = self.k32(0);
        self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[n, z])
    }

    pub(super) fn mask_bin(&mut self, base: &str, a: V, b: V) -> Option<V> {
        Some(match base {
            "and" => self.spv(op::LOGICAL_AND, Ty::BOOL, &[a, b]),
            "or" => self.spv(op::LOGICAL_OR, Ty::BOOL, &[a, b]),
            "xor" => self.spv(op::LOGICAL_NOT_EQUAL, Ty::BOOL, &[a, b]),
            "xnor" => self.spv(op::LOGICAL_EQUAL, Ty::BOOL, &[a, b]),
            "andn2" => {
                let nb = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[b]);
                self.spv(op::LOGICAL_AND, Ty::BOOL, &[a, nb])
            }
            "orn2" => {
                let nb = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[b]);
                self.spv(op::LOGICAL_OR, Ty::BOOL, &[a, nb])
            }
            "nand" => {
                let x = self.spv(op::LOGICAL_AND, Ty::BOOL, &[a, b]);
                self.spv(op::LOGICAL_NOT, Ty::BOOL, &[x])
            }
            "nor" => {
                let x = self.spv(op::LOGICAL_OR, Ty::BOOL, &[a, b]);
                self.spv(op::LOGICAL_NOT, Ty::BOOL, &[x])
            }
            _ => return None,
        })
    }

    pub(super) fn int_bin(&mut self, base: &str, a: V, b: V) -> Option<V> {
        Some(match base {
            "and" => self.spv(op::BITWISE_AND, Ty::U32, &[a, b]),
            "or" => self.spv(op::BITWISE_OR, Ty::U32, &[a, b]),
            "xor" => self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]),
            "xnor" => {
                let x = self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]);
                self.spv(op::NOT, Ty::U32, &[x])
            }
            "andn2" => {
                let nb = self.spv(op::NOT, Ty::U32, &[b]);
                self.spv(op::BITWISE_AND, Ty::U32, &[a, nb])
            }
            "orn2" => {
                let nb = self.spv(op::NOT, Ty::U32, &[b]);
                self.spv(op::BITWISE_OR, Ty::U32, &[a, nb])
            }
            "nand" => {
                let x = self.spv(op::BITWISE_AND, Ty::U32, &[a, b]);
                self.spv(op::NOT, Ty::U32, &[x])
            }
            "nor" => {
                let x = self.spv(op::BITWISE_OR, Ty::U32, &[a, b]);
                self.spv(op::NOT, Ty::U32, &[x])
            }
            _ => return None,
        })
    }

    pub(super) fn clz(&mut self, a: V) -> V {
        let m = self.gl(glsl::FIND_U_MSB, Ty::U32, &[a]);
        let k = self.k32(31);
        let r = self.spv(op::I_SUB, Ty::U32, &[k, m]);
        let neg = self.k32(u32::MAX);
        let z = self.k32(0);
        let is = self.spv(op::I_EQUAL, Ty::BOOL, &[a, z]);
        self.spv(op::SELECT, Ty::U32, &[is, neg, r])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn salu_arith(
        &mut self,
        inst: &Inst,
        name: &'static str,
        base: &str,
        dst: &Option<Operand>,
        src: &[Operand],
        imm: Option<u16>,
        b64: bool,
    ) -> Res<()> {
        let off = inst.machine.offset;
        let signed = name.ends_with("i32") || name.ends_with("i64");
        let a0 = |l: &mut Self| -> Res<V> {
            if imm.is_some() {
                let d = l.opnd(inst, &dst.unwrap())?;
                Ok(l.u32v(d))
            } else {
                let x = l.opnd(inst, &src[0])?;
                Ok(l.u32v(x))
            }
        };
        let b0 = |l: &mut Self| -> Res<V> {
            if let Some(k) = imm {
                Ok(l.k32(i32::from(k as i16) as u32))
            } else {
                let x = l.opnd(inst, &src[1])?;
                Ok(l.u32v(x))
            }
        };
        let z = self.k32(0);
        if b64 && matches!(base, "lshl" | "lshr" | "ashr" | "bfe") {
            let a = self.opnd_u64(inst, &src[0])?;
            let s = self.opnd(inst, &src[1])?;
            let s = self.u32v(s);
            let k63 = self.k32(63);
            let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k63]);
            let r = match base {
                "lshl" => self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U64, &[a, s]),
                "lshr" => self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U64, &[a, s]),
                "ashr" => self.spv(op::SHIFT_RIGHT_ARITHMETIC, Ty::U64, &[a, s]),
                _ => return Err(unsupported(off, name)),
            };
            let (lo, hi) = self.split64(r);
            self.sdst(inst, dst, 0, R::D(lo));
            self.sdst(inst, dst, 1, R::D(hi));
            let zz = self.f.konst(Ty::U64, 0);
            let nz = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[r, zz]);
            self.def(inst, Reg::Scc, R::B(nz));
            return Ok(());
        }
        if b64 && base == "bfm" {
            return Err(unsupported(off, name));
        }
        let a = a0(self)?;
        let b = b0(self)?;
        let (r, scc) = match base {
            "add" | "addk" | "addc" => {
                let mut x = self.spv(op::I_ADD, Ty::U32, &[a, b]);
                let mut carry = self.spv(op::U_LESS_THAN, Ty::BOOL, &[x, a]);
                if base == "addc" {
                    let r = self.val(inst, Reg::Scc)?;
                    let c = self.scc_of(r);
                    let cu = self.u32v(c);
                    let y = self.spv(op::I_ADD, Ty::U32, &[x, cu]);
                    let c2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[y, x]);
                    carry = self.spv(op::LOGICAL_OR, Ty::BOOL, &[carry, c2]);
                    x = y;
                }
                if signed {
                    // overflow: sign of a == sign of b != sign of result
                    let ax = self.spv(op::BITWISE_XOR, Ty::U32, &[a, x]);
                    let bx = self.spv(op::BITWISE_XOR, Ty::U32, &[b, x]);
                    let both = self.spv(op::BITWISE_AND, Ty::U32, &[ax, bx]);
                    carry = self.spv(op::S_LESS_THAN, Ty::BOOL, &[both, z]);
                }
                (x, Some(carry))
            }
            "sub" | "subb" => {
                let mut x = self.spv(op::I_SUB, Ty::U32, &[a, b]);
                let mut borrow = self.spv(op::U_LESS_THAN, Ty::BOOL, &[a, b]);
                if base == "subb" {
                    let r = self.val(inst, Reg::Scc)?;
                    let c = self.scc_of(r);
                    let cu = self.u32v(c);
                    let b2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[x, cu]);
                    x = self.spv(op::I_SUB, Ty::U32, &[x, cu]);
                    borrow = self.spv(op::LOGICAL_OR, Ty::BOOL, &[borrow, b2]);
                }
                if signed {
                    let ab = self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]);
                    let ax = self.spv(op::BITWISE_XOR, Ty::U32, &[a, x]);
                    let both = self.spv(op::BITWISE_AND, Ty::U32, &[ab, ax]);
                    borrow = self.spv(op::S_LESS_THAN, Ty::BOOL, &[both, z]);
                }
                (x, Some(borrow))
            }
            "mul" | "mulk" => (self.spv(op::I_MUL, Ty::U32, &[a, b]), None),
            "mul_hi" => {
                let o = if signed { op::S_MUL_EXTENDED } else { op::U_MUL_EXTENDED };
                let s = self.spv(o, Ty::vec(S::U32, 2), &[a, b]);
                (self.extract(s, 1), None)
            }
            "min" | "max" => {
                let g = match (base, signed) {
                    ("min", false) => glsl::U_MIN,
                    ("min", true) => glsl::S_MIN,
                    ("max", false) => glsl::U_MAX,
                    _ => glsl::S_MAX,
                };
                let x = self.gl(g, Ty::U32, &[a, b]);
                // SCC: a was picked
                let c = self.spv(op::I_EQUAL, Ty::BOOL, &[x, a]);
                (x, Some(c))
            }
            "lshl" | "lshr" | "ashr" => {
                let k31 = self.k32(31);
                let s = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                let o = match base {
                    "lshl" => op::SHIFT_LEFT_LOGICAL,
                    "lshr" => op::SHIFT_RIGHT_LOGICAL,
                    _ => op::SHIFT_RIGHT_ARITHMETIC,
                };
                let x = self.spv(o, Ty::U32, &[a, s]);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "lshl1_add" | "lshl2_add" | "lshl3_add" | "lshl4_add" => {
                let n = u32::from(base.as_bytes()[4] - b'0');
                let k = self.k32(n);
                let s = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[a, k]);
                let x = self.spv(op::I_ADD, Ty::U32, &[s, b]);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "bfe" => {
                let k31 = self.k32(31);
                let o = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                let k16 = self.k32(16);
                let w = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, k16]);
                let k7f = self.k32(0x7F);
                let w = self.spv(op::BITWISE_AND, Ty::U32, &[w, k7f]);
                let x = self.bfe(a, o, w, signed);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "bfm" => {
                let x = self.bfm(a, b);
                (x, None)
            }
            "absdiff" => {
                let d = self.spv(op::I_SUB, Ty::U32, &[a, b]);
                let x = self.gl(glsl::S_ABS, Ty::U32, &[d]);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "pack_ll" | "pack_lh" | "pack_hh" | "pack_hl" => {
                let k16 = self.k32(16);
                let m = self.k32(0xFFFF);
                let lo = if base.as_bytes()[5] == b'l' {
                    self.spv(op::BITWISE_AND, Ty::U32, &[a, m])
                } else {
                    self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, k16])
                };
                let hi = if base.as_bytes()[6] == b'l' {
                    self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[b, k16])
                } else {
                    let mh = self.k32(0xFFFF_0000);
                    self.spv(op::BITWISE_AND, Ty::U32, &[b, mh])
                };
                (self.spv(op::BITWISE_OR, Ty::U32, &[lo, hi]), None)
            }
            _ => return Err(unsupported(off, name)),
        };
        self.sdst(inst, dst, 0, R::D(r));
        if imm.is_some() && dst.is_none() {
            return Ok(());
        }
        if let Some(c) = scc {
            self.def(inst, Reg::Scc, R::B(c));
        }
        Ok(())
    }

    /// Bit-field extract with the hardware's semantics for any offset and
    /// width (masked to 5 bits / clamped).
    pub(super) fn bfe(&mut self, a: V, o: V, w: V, signed: bool) -> V {
        if let (Some(oc), Some(wc)) = (self.f.as_const(o), self.f.as_const(w))
            && oc + wc <= 32
            && wc > 0
        {
            let opc = if signed { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
            return self.spv(opc, Ty::U32, &[a, o, w]);
        }
        if let Some(0) = self.f.as_const(w) {
            return self.k32(0);
        }
        // (a >> o) & mask(w), or the arithmetic form.
        let k32 = self.k32(32);
        let wc = self.gl(glsl::U_MIN, Ty::U32, &[w, k32]);
        if signed {
            // (a << (32 - o - w)) >> (32 - w)
            let ow = self.spv(op::I_ADD, Ty::U32, &[o, wc]);
            let ow = self.gl(glsl::U_MIN, Ty::U32, &[ow, k32]);
            let l = self.spv(op::I_SUB, Ty::U32, &[k32, ow]);
            let k31 = self.k32(31);
            let l = self.gl(glsl::U_MIN, Ty::U32, &[l, k31]);
            let sh = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[a, l]);
            let r = self.spv(op::I_SUB, Ty::U32, &[k32, wc]);
            let r = self.gl(glsl::U_MIN, Ty::U32, &[r, k31]);
            let r2 = self.spv(op::I_ADD, Ty::U32, &[r, k32]);
            let _ = r2;
            self.spv(op::SHIFT_RIGHT_ARITHMETIC, Ty::U32, &[sh, r])
        } else {
            let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, o]);
            let m = self.mask_bits(wc);
            self.spv(op::BITWISE_AND, Ty::U32, &[s, m])
        }
    }

    /// (1 << w) - 1 for w in 0..=32.
    pub(super) fn mask_bits(&mut self, w: V) -> V {
        if let Some(c) = self.f.as_const(w) {
            return self.k32(if c >= 32 { u32::MAX } else { (1u32 << c) - 1 });
        }
        let one = self.k32(1);
        let k31 = self.k32(31);
        let wm = self.spv(op::BITWISE_AND, Ty::U32, &[w, k31]);
        let s = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[one, wm]);
        let m = self.spv(op::I_SUB, Ty::U32, &[s, one]);
        let k32 = self.k32(32);
        let big = self.spv(op::U_GREATER_THAN_EQUAL, Ty::BOOL, &[w, k32]);
        let all = self.k32(u32::MAX);
        self.spv(op::SELECT, Ty::U32, &[big, all, m])
    }

    pub(super) fn bfm(&mut self, w: V, o: V) -> V {
        let k31 = self.k32(31);
        let w = self.spv(op::BITWISE_AND, Ty::U32, &[w, k31]);
        let o = self.spv(op::BITWISE_AND, Ty::U32, &[o, k31]);
        let m = self.mask_bits(w);
        self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[m, o])
    }
}
