//! Scalar ALU instructions.

use super::*;

impl Lift<'_> {
    pub(super) fn sop(&mut self, inst: &Inst, name: &'static str, dst: &Option<Operand>, src: &[Operand], imm: Option<u16>) -> Res<()> {
        let off = inst.machine.offset;
        let b64 = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        let dst_exec = matches!(dst, Some(Operand::Scalar { code, .. }) if *code == ir::EXEC || *code == ir::VCC);
        let s = Sop { inst, name, base: sop_base(name), dst, src, imm, b64, dst_exec };
        if name.contains("saveexec") || name.contains("wrexec") {
            return self.sop_saveexec(&s);
        }
        match s.base {
            "mov" | "cmov" | "movk" | "cmovk" => self.sop_mov(&s),
            "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not" | "wqm" => self.sop_logic(&s),
            "cselect" => self.sop_cselect(&s),
            "bcnt1" | "bcnt0" => self.sop_bcnt(&s),
            "ff1" | "ff0" | "flbit" | "brev" | "abs" => self.sop_bits(&s),
            "add" | "sub" | "addc" | "subb" | "addk" | "mul" | "mulk" | "mul_hi" | "min" | "max" | "lshl" | "lshr"
            | "ashr" | "bfe" | "bfm" | "lshl1_add" | "lshl2_add" | "lshl3_add" | "lshl4_add" | "absdiff" | "pack_ll"
            | "pack_lh" | "pack_hh" | "pack_hl" => self.salu_arith(&s),
            "cmp_eq" | "cmp_lg" | "cmp_gt" | "cmp_ge" | "cmp_lt" | "cmp_le" | "cmpk_eq" | "cmpk_lg" | "cmpk_gt"
            | "cmpk_ge" | "cmpk_lt" | "cmpk_le" | "bitcmp0" | "bitcmp1" => self.sop_cmp(&s),
            "bitset0" | "bitset1" => self.sop_bitset(&s),
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

    /// Whether a scalar operand holds a lane mask (EXEC before any write).
    fn mask_like(&self, inst: &Inst, o: &Operand) -> bool {
        let Operand::Scalar { code, .. } = *o else { return false };
        if code == crate::gfx10::NULL {
            return false;
        }
        let Some(v) = inst.read(Reg::S(code)) else { return false };
        matches!(self.vals.get(&v), Some(R::M(..))) || (code == ir::EXEC && !self.vals.contains_key(&v))
    }

    /// SCC, where the instruction writes it.
    fn scc_def(&mut self, inst: &Inst, b: V) {
        if inst.write(Reg::Scc).is_some() {
            self.def(inst, Reg::Scc, R::B(b));
        }
    }

    /// SCC = the value is not zero.
    fn scc_nonzero(&mut self, inst: &Inst, x: V) {
        let z = self.k32(0);
        let b = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
        self.scc_def(inst, b);
    }

    /// The saveexec / wrexec family: EXEC = op(src, EXEC); dst = the old
    /// EXEC (saveexec) or the new (wrexec).
    fn sop_saveexec(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        let a = self.active(inst)?;
        let m = self.opnd_mask(inst, &s.src[0])?;
        let opname = s.base.trim_end_matches("_saveexec").trim_end_matches("_wrexec");
        let new = self.mask_bin(opname, m, a).ok_or(unsupported(inst.machine.offset, s.name))?;
        let wr = s.name.contains("wrexec");
        self.mask_def(inst, s.dst, if wr { new } else { a });
        self.def(inst, Reg::S(ir::EXEC), R::M(new, Half::Lo));
        if self.wave64 {
            self.def(inst, Reg::S(ir::EXEC + 1), R::M(new, Half::Hi));
        }
        let any = self.any(new);
        self.scc_def(inst, any);
        Ok(())
    }

    fn sop_mov(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        let cond = if s.base.starts_with("cmov") {
            let r = self.val(inst, Reg::Scc)?;
            Some(self.scc_of(r))
        } else {
            None
        };
        if let Some(k) = s.imm {
            let x = self.k32(i32::from(k as i16) as u32);
            let x = self.cond_keep(inst, s.dst, 0, cond, x)?;
            self.sdst(inst, s.dst, 0, R::D(x));
            return Ok(());
        }
        if s.b64 && (self.mask_like(inst, &s.src[0]) || s.dst_exec) && cond.is_none() {
            let m = self.opnd_mask(inst, &s.src[0])?;
            self.mask_def(inst, s.dst, m);
            return Ok(());
        }
        for k in 0..s.dwords() {
            let r = match s.src[0] {
                Operand::Scalar { code, .. } if code != crate::gfx10::NULL => self.val(inst, Reg::S(code + k))?,
                _ => R::D(self.opnd_dw(inst, &s.src[0], k, false)?),
            };
            let r = match cond {
                Some(_) => {
                    let x = self.data(r);
                    let x = self.u32v(x);
                    R::D(self.cond_keep(inst, s.dst, k, cond, x)?)
                }
                None => r,
            };
            self.sdst(inst, s.dst, k, r);
        }
        Ok(())
    }

    /// `x`, or under a false condition the destination's old dword `k`.
    fn cond_keep(&mut self, inst: &Inst, dst: &Option<Operand>, k: u8, cond: Option<V>, x: V) -> Res<V> {
        let Some(c) = cond else { return Ok(x) };
        let old = self.opnd_dw(inst, &dst.unwrap(), k, false)?;
        let old = self.u32v(old);
        Ok(self.spv(op::SELECT, Ty::U32, &[c, x, old]))
    }

    /// Bitwise logic: on lane masks as bools, else on dwords; SCC = any bit.
    fn sop_logic(&mut self, s: &Sop) -> Res<()> {
        let (inst, base) = (s.inst, s.base);
        let off = inst.machine.offset;
        let maskish = s.src.iter().any(|o| self.mask_like(inst, o)) || s.dst_exec;
        if maskish || base == "wqm" {
            let a = self.opnd_mask(inst, &s.src[0])?;
            let r = match base {
                "not" => self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]),
                "wqm" => self.wqm(a),
                _ => {
                    let b = self.opnd_mask(inst, &s.src[1])?;
                    self.mask_bin(base, a, b).ok_or(unsupported(off, s.name))?
                }
            };
            self.mask_def(inst, s.dst, r);
            let any = self.any(r);
            self.scc_def(inst, any);
            return Ok(());
        }
        let mut nz = None;
        for k in 0..s.dwords() {
            let a = self.opnd_dw(inst, &s.src[0], k, false)?;
            let a = self.u32v(a);
            let r = if base == "not" || base == "wqm" {
                self.spv(op::NOT, Ty::U32, &[a])
            } else {
                let b = self.opnd_dw(inst, &s.src[1], k, false)?;
                let b = self.u32v(b);
                self.int_bin(base, a, b).ok_or(unsupported(off, s.name))?
            };
            self.sdst(inst, s.dst, k, R::D(r));
            nz = Some(match nz {
                None => r,
                Some(p) => self.spv(op::BITWISE_OR, Ty::U32, &[p, r]),
            });
        }
        self.scc_nonzero(inst, nz.unwrap());
        Ok(())
    }

    fn sop_cselect(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        let r = self.val(inst, Reg::Scc)?;
        let c = self.scc_of(r);
        if s.b64 && (s.src.iter().any(|o| self.mask_like(inst, o)) || s.dst_exec) {
            let a = self.opnd_mask(inst, &s.src[0])?;
            let b = self.opnd_mask(inst, &s.src[1])?;
            let m = self.spv(op::SELECT, Ty::BOOL, &[c, a, b]);
            self.mask_def(inst, s.dst, m);
            return Ok(());
        }
        for k in 0..s.dwords() {
            let a = self.opnd_dw(inst, &s.src[0], k, false)?;
            let b = self.opnd_dw(inst, &s.src[1], k, false)?;
            let (a, b) = (self.u32v(a), self.u32v(b));
            let x = self.spv(op::SELECT, Ty::U32, &[c, a, b]);
            self.sdst(inst, s.dst, k, R::D(x));
        }
        Ok(())
    }

    fn sop_bcnt(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        let mut sum = self.k32(0);
        for k in 0..s.dwords() {
            let a = self.opnd_dw(inst, &s.src[0], k, false)?;
            let a = self.u32v(a);
            let a = if s.base == "bcnt0" { self.spv(op::NOT, Ty::U32, &[a]) } else { a };
            let c = self.spv(op::BIT_COUNT, Ty::U32, &[a]);
            sum = self.spv(op::I_ADD, Ty::U32, &[sum, c]);
        }
        if s.b64 && self.wave64 {
            self.approx("a wave64 mask counted (s_bcnt1_i32_b64)");
        }
        self.sdst(inst, s.dst, 0, R::D(sum));
        self.scc_nonzero(inst, sum);
        Ok(())
    }

    /// Bit searches, reversal and the absolute value (one dword).
    fn sop_bits(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        if s.base == "ff1" || s.base == "ff0" {
            let a = self.opnd_dw(inst, &s.src[0], 0, false)?;
            let mut a = self.u32v(a);
            if s.base == "ff0" {
                a = self.spv(op::NOT, Ty::U32, &[a]);
            }
            if s.b64 && self.wave64 {
                self.approx("a wave64 mask searched (s_ff1_i32_b64)");
            }
            let x = self.gl(glsl::FIND_I_LSB, Ty::U32, &[a]);
            self.sdst(inst, s.dst, 0, R::D(x));
            return Ok(());
        }
        let a = self.opnd(inst, &s.src[0])?;
        let a = self.u32v(a);
        let x = match s.base {
            "flbit" if s.name.ends_with("i32_i32") => self.sign_bits(a),
            "flbit" => self.clz(a),
            "brev" => self.spv(op::BIT_REVERSE, Ty::U32, &[a]),
            _ => self.gl(glsl::S_ABS, Ty::U32, &[a]),
        };
        self.sdst(inst, s.dst, 0, R::D(x));
        if s.base == "abs" {
            self.scc_nonzero(inst, x);
        }
        Ok(())
    }

    /// The count of leading sign bits (-1 for 0 and -1).
    fn sign_bits(&mut self, a: V) -> V {
        let m = self.gl(glsl::FIND_S_MSB, Ty::U32, &[a]);
        let k = self.k32(31);
        let r = self.spv(op::I_SUB, Ty::U32, &[k, m]);
        let neg = self.k32(u32::MAX);
        let is = self.spv(op::I_EQUAL, Ty::BOOL, &[m, neg]);
        self.spv(op::SELECT, Ty::U32, &[is, neg, r])
    }

    /// Compares into SCC (cmpk: a 16-bit immediate, signed for i32).
    fn sop_cmp(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        let signed = s.name.ends_with("i32");
        let (a, b) = if let Some(k) = s.imm {
            let a = self.opnd(inst, &s.src[0])?;
            let k = if signed { i32::from(k as i16) as u32 } else { u32::from(k) };
            (self.u32v(a), self.k32(k))
        } else if s.b64 {
            (self.opnd_u64(inst, &s.src[0])?, self.opnd_u64(inst, &s.src[1])?)
        } else {
            let a = self.opnd(inst, &s.src[0])?;
            let b = self.opnd(inst, &s.src[1])?;
            (self.u32v(a), self.u32v(b))
        };
        let rel = s.base.trim_start_matches("cmpk_").trim_start_matches("cmp_");
        let c = if rel.starts_with("bitcmp") {
            let one = self.k32(1);
            let sh = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, b]);
            let bit = self.spv(op::BITWISE_AND, Ty::U32, &[sh, one]);
            let z = self.k32(if rel == "bitcmp0" { 0 } else { 1 });
            self.spv(op::I_EQUAL, Ty::BOOL, &[bit, z])
        } else {
            let o = int_cmp(rel, signed).ok_or(unsupported(inst.machine.offset, s.name))?;
            self.spv(o, Ty::BOOL, &[a, b])
        };
        self.def(inst, Reg::Scc, R::B(c));
        Ok(())
    }

    fn sop_bitset(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        let bit = self.opnd(inst, &s.src[0])?;
        let bit = self.u32v(bit);
        let old = self.opnd(inst, &s.dst.unwrap())?;
        let old = self.u32v(old);
        let one = self.k32(1);
        let k31 = self.k32(31);
        let bit = self.spv(op::BITWISE_AND, Ty::U32, &[bit, k31]);
        let m = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[one, bit]);
        let x = if s.base == "bitset1" {
            self.spv(op::BITWISE_OR, Ty::U32, &[old, m])
        } else {
            let nm = self.spv(op::NOT, Ty::U32, &[m]);
            self.spv(op::BITWISE_AND, Ty::U32, &[old, nm])
        };
        self.sdst(inst, s.dst, 0, R::D(x));
        Ok(())
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

    /// Scalar arithmetic: dst = a op b (a: the destination for the k forms,
    /// b: their 16-bit immediate); SCC as each instruction defines it.
    fn salu_arith(&mut self, s: &Sop) -> Res<()> {
        let (inst, base) = (s.inst, s.base);
        let off = inst.machine.offset;
        if s.b64 && matches!(base, "lshl" | "lshr" | "ashr" | "bfe") {
            return self.shift64(s);
        }
        if s.b64 && base == "bfm" {
            return Err(unsupported(off, s.name));
        }
        let a = match s.imm {
            Some(_) => self.opnd(inst, &s.dst.unwrap())?,
            None => self.opnd(inst, &s.src[0])?,
        };
        let a = self.u32v(a);
        let b = match s.imm {
            Some(k) => self.k32(i32::from(k as i16) as u32),
            None => {
                let x = self.opnd(inst, &s.src[1])?;
                self.u32v(x)
            }
        };
        let (r, scc) = match base {
            "add" | "addk" | "addc" => self.salu_add(s, a, b)?,
            "sub" | "subb" => self.salu_sub(s, a, b)?,
            _ => self.salu_plain(s, a, b).ok_or(unsupported(off, s.name))?,
        };
        self.sdst(inst, s.dst, 0, R::D(r));
        if s.imm.is_some() && s.dst.is_none() {
            return Ok(());
        }
        if let Some(c) = scc {
            self.def(inst, Reg::Scc, R::B(c));
        }
        Ok(())
    }

    /// 64-bit shifts; SCC = the result is not zero.
    fn shift64(&mut self, s: &Sop) -> Res<()> {
        let inst = s.inst;
        let a = self.opnd_u64(inst, &s.src[0])?;
        let n = self.opnd(inst, &s.src[1])?;
        let n = self.u32v(n);
        let k63 = self.k32(63);
        let n = self.spv(op::BITWISE_AND, Ty::U32, &[n, k63]);
        let o = match s.base {
            "lshl" => op::SHIFT_LEFT_LOGICAL,
            "lshr" => op::SHIFT_RIGHT_LOGICAL,
            "ashr" => op::SHIFT_RIGHT_ARITHMETIC,
            _ => return Err(unsupported(inst.machine.offset, s.name)),
        };
        let r = self.spv(o, Ty::U64, &[a, n]);
        let (lo, hi) = self.split64(r);
        self.sdst(inst, s.dst, 0, R::D(lo));
        self.sdst(inst, s.dst, 1, R::D(hi));
        let zz = self.f.konst(Ty::U64, 0);
        let nz = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[r, zz]);
        self.def(inst, Reg::Scc, R::B(nz));
        Ok(())
    }

    /// The carry-in SCC as 0 or 1.
    fn carry_in(&mut self, inst: &Inst) -> Res<V> {
        let r = self.val(inst, Reg::Scc)?;
        let c = self.scc_of(r);
        Ok(self.u32v(c))
    }

    /// Sign overflow: the result's sign differs from both of `p` and `q`'s
    /// (p ^ x) & (q ^ x) < 0.
    fn overflow(&mut self, p: V, q: V, x: V) -> V {
        let px = self.spv(op::BITWISE_XOR, Ty::U32, &[p, x]);
        let qx = self.spv(op::BITWISE_XOR, Ty::U32, &[q, x]);
        let both = self.spv(op::BITWISE_AND, Ty::U32, &[px, qx]);
        let z = self.k32(0);
        self.spv(op::S_LESS_THAN, Ty::BOOL, &[both, z])
    }

    /// Add (with carry in for addc); SCC: unsigned carry or signed overflow.
    fn salu_add(&mut self, s: &Sop, a: V, b: V) -> Res<(V, Option<V>)> {
        let mut x = self.spv(op::I_ADD, Ty::U32, &[a, b]);
        let mut carry = self.spv(op::U_LESS_THAN, Ty::BOOL, &[x, a]);
        if s.base == "addc" {
            let cu = self.carry_in(s.inst)?;
            let y = self.spv(op::I_ADD, Ty::U32, &[x, cu]);
            let c2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[y, x]);
            carry = self.spv(op::LOGICAL_OR, Ty::BOOL, &[carry, c2]);
            x = y;
        }
        if s.signed() {
            // overflow: sign of a == sign of b != sign of result
            carry = self.overflow(a, b, x);
        }
        Ok((x, Some(carry)))
    }

    /// Subtract (with borrow in for subb); SCC: borrow or signed overflow.
    fn salu_sub(&mut self, s: &Sop, a: V, b: V) -> Res<(V, Option<V>)> {
        let mut x = self.spv(op::I_SUB, Ty::U32, &[a, b]);
        let mut borrow = self.spv(op::U_LESS_THAN, Ty::BOOL, &[a, b]);
        if s.base == "subb" {
            let cu = self.carry_in(s.inst)?;
            let b2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[x, cu]);
            x = self.spv(op::I_SUB, Ty::U32, &[x, cu]);
            borrow = self.spv(op::LOGICAL_OR, Ty::BOOL, &[borrow, b2]);
        }
        if s.signed() {
            let ab = self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]);
            let ax = self.spv(op::BITWISE_XOR, Ty::U32, &[a, x]);
            let both = self.spv(op::BITWISE_AND, Ty::U32, &[ab, ax]);
            let z = self.k32(0);
            borrow = self.spv(op::S_LESS_THAN, Ty::BOOL, &[both, z]);
        }
        Ok((x, Some(borrow)))
    }

    /// The rest of the arithmetic, and its SCC (none: SCC is kept).
    fn salu_plain(&mut self, s: &Sop, a: V, b: V) -> Option<(V, Option<V>)> {
        let base = s.base;
        let (x, scc) = match base {
            "mul" | "mulk" => (self.spv(op::I_MUL, Ty::U32, &[a, b]), None),
            "mul_hi" => {
                let o = if s.signed() { op::S_MUL_EXTENDED } else { op::U_MUL_EXTENDED };
                let m = self.spv(o, Ty::vec(S::U32, 2), &[a, b]);
                (self.extract(m, 1), None)
            }
            "min" | "max" => {
                let g = match (base, s.signed()) {
                    ("min", false) => glsl::U_MIN,
                    ("min", true) => glsl::S_MIN,
                    ("max", false) => glsl::U_MAX,
                    _ => glsl::S_MAX,
                };
                let x = self.gl(g, Ty::U32, &[a, b]);
                // SCC: a was picked
                let c = self.spv(op::I_EQUAL, Ty::BOOL, &[x, a]);
                return Some((x, Some(c)));
            }
            "lshl" | "lshr" | "ashr" => {
                let k31 = self.k32(31);
                let n = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                let o = match base {
                    "lshl" => op::SHIFT_LEFT_LOGICAL,
                    "lshr" => op::SHIFT_RIGHT_LOGICAL,
                    _ => op::SHIFT_RIGHT_ARITHMETIC,
                };
                (self.spv(o, Ty::U32, &[a, n]), Some(()))
            }
            "lshl1_add" | "lshl2_add" | "lshl3_add" | "lshl4_add" => {
                let k = self.k32(u32::from(base.as_bytes()[4] - b'0'));
                let sh = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[a, k]);
                (self.spv(op::I_ADD, Ty::U32, &[sh, b]), Some(()))
            }
            "bfe" => {
                let k31 = self.k32(31);
                let o = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                let k16 = self.k32(16);
                let w = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, k16]);
                let k7f = self.k32(0x7F);
                let w = self.spv(op::BITWISE_AND, Ty::U32, &[w, k7f]);
                (self.bfe(a, o, w, s.signed()), Some(()))
            }
            "bfm" => (self.bfm(a, b), None),
            "absdiff" => {
                let d = self.spv(op::I_SUB, Ty::U32, &[a, b]);
                (self.gl(glsl::S_ABS, Ty::U32, &[d]), Some(()))
            }
            "pack_ll" | "pack_lh" | "pack_hh" | "pack_hl" => (self.pack16(base, a, b), None),
            _ => return None,
        };
        // SCC: the result is not zero
        let scc = scc.map(|()| {
            let z = self.k32(0);
            self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z])
        });
        Some((x, scc))
    }

    /// s_pack_xy: the x half of a low, the y half of b high.
    fn pack16(&mut self, base: &str, a: V, b: V) -> V {
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
        self.spv(op::BITWISE_OR, Ty::U32, &[lo, hi])
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

/// A scalar instruction being lowered.
pub(super) struct Sop<'i> {
    inst: &'i Inst,
    name: &'static str,
    /// The name without `s_` and its type suffixes.
    base: &'static str,
    dst: &'i Option<Operand>,
    src: &'i [Operand],
    imm: Option<u16>,
    b64: bool,
    /// The destination is EXEC or VCC (a lane mask).
    dst_exec: bool,
}

impl Sop<'_> {
    fn dwords(&self) -> u8 {
        if self.b64 { 2 } else { 1 }
    }

    fn signed(&self) -> bool {
        self.name.ends_with("i32") || self.name.ends_with("i64")
    }
}

/// The name without `s_` and its type suffixes (`s_and_saveexec_b64` ->
/// `and_saveexec`).
fn sop_base(name: &'static str) -> &'static str {
    let mut base = name.trim_start_matches("s_");
    while let Some((a, t)) = base.rsplit_once('_') {
        if !matches!(t, "b32" | "b64" | "i32" | "u32" | "i64" | "u64" | "b16" | "i16" | "u16") {
            break;
        }
        base = a;
    }
    base
}
