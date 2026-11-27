//! The lanes of a console wave that hold no work, where one host subgroup
//! stands for the wave: what a lane mask read as a number holds for them.
//!
//! A fragment subgroup has at most 32 invocations, may be partly filled,
//! and has no way to meet another one. So the wave it stands for is a
//! console wave in which only the subgroup's invocations hold work: the
//! other lanes (all of lanes 32-63 of a wave64, and the lanes below 32 the
//! subgroup lacks) are in EXEC at no time the program chose, as in any
//! partly filled console wave. A ballot has no bits for them. The console's
//! mask does, and for most masks they are known: this pass finds them
//! before lifting, over the wave SSA, as "the bits of this dword at idle
//! lanes are all 0 / all 1". EXEC's are 0 at the start; a compare writes a
//! 0 for every lane outside EXEC, so its are 0 while EXEC's are; the
//! bitwise scalar operations, moves, selects, `s_wqm` and the `saveexec`
//! forms compute on them (`s_xor_b64 x, vcc, -1`, the compiler's NOT, gives
//! 1; `s_or_saveexec -1` sets EXEC's: the idle lanes run a whole-wave
//! section, and what compares give there is not known); phis join. Found
//! optimistically, so a mask a loop narrows keeps its 0.
//!
//! The lift then reads a mask's dword as a number exactly (`Lift::settled`,
//! `mask_word`): the upper dword of a wave64 on one subgroup is the
//! constant 0 (what it read before, now known exact) or all ones (where it
//! read 0); the lower dword is the ballot, with the bits of the lanes the
//! subgroup lacks set where the idle bit is 1. What is left unknown stays a
//! listed approximation.

use std::collections::HashMap;

use crate::gfx10::{Fields, Operand};
use crate::ir::{self, Inst, Program, Reg, Value};

/// What is known of a dword: nothing yet, all zeros, all ones, or unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum L {
    Unset,
    Zero,
    Ones,
    Any,
}

impl L {
    fn join(self, o: L) -> L {
        match (self, o) {
            (L::Unset, x) | (x, L::Unset) => x,
            (a, b) if a == b => a,
            _ => L::Any,
        }
    }

    fn not(self) -> L {
        match self {
            L::Zero => L::Ones,
            L::Ones => L::Zero,
            x => x,
        }
    }

    fn and(self, o: L) -> L {
        match (self, o) {
            (L::Zero, _) | (_, L::Zero) => L::Zero,
            (L::Ones, x) | (x, L::Ones) => x,
            (L::Unset, _) | (_, L::Unset) => L::Unset,
            _ => L::Any,
        }
    }

    fn or(self, o: L) -> L {
        self.not().and(o.not()).not()
    }

    fn xor(self, o: L) -> L {
        match (self, o) {
            (L::Zero, x) | (x, L::Zero) => x,
            (L::Ones, x) | (x, L::Ones) => x.not(),
            (L::Unset, _) | (_, L::Unset) => L::Unset,
            _ => L::Any,
        }
    }
}

/// `op(a, b)` of the scalar mask operations (`s_<op>_b64`; for the
/// `saveexec` forms `a` is the source and `b` EXEC).
fn logic(op: &str, a: L, b: L) -> L {
    match op {
        "and" => a.and(b),
        "or" => a.or(b),
        "xor" => a.xor(b),
        "andn1" => a.not().and(b),
        "andn2" => a.and(b.not()),
        "orn1" => a.not().or(b),
        "orn2" => a.or(b.not()),
        "nand" => a.and(b).not(),
        "nor" => a.or(b).not(),
        "xnor" => a.xor(b).not(),
        _ => L::Any,
    }
}

struct Pass {
    known: HashMap<Value, L>,
    changed: bool,
}

impl Pass {
    fn get(&self, v: Value) -> L {
        self.known.get(&v).copied().unwrap_or(L::Unset)
    }

    fn set(&mut self, v: Value, x: L) {
        let new = self.get(v).join(x);
        if new != self.get(v) {
            self.known.insert(v, new);
            self.changed = true;
        }
    }

    /// Dword `k` of a scalar operand.
    fn dword(&self, inst: &Inst, o: &Operand, k: u8) -> L {
        match *o {
            Operand::Scalar { code, .. } if code == crate::gfx10::NULL => L::Zero,
            Operand::Scalar { code, .. } => inst.read(Reg::S(code + k)).map_or(L::Any, |v| self.get(v)),
            // inline integers: 0, 1-64 (their upper dword 0), then -1 to
            // -16 (sign-extended)
            Operand::Const(128) => L::Zero,
            Operand::Const(193) => L::Ones,
            Operand::Const(129..=192) if k == 1 => L::Zero,
            Operand::Const(194..=208) if k == 1 => L::Ones,
            _ => L::Any,
        }
    }

    /// Dword `k` of a bitwise scalar operation's results: (register, what
    /// it holds).
    fn sop(&self, inst: &Inst, base: &str, code: u8, src: &[Operand], k: u8) -> Vec<(u8, L)> {
        let exec = inst.read(Reg::S(ir::EXEC + k)).map_or(L::Any, |v| self.get(v));
        let a = src.first().map_or(L::Any, |o| self.dword(inst, o, k));
        let b = src.get(1).map_or(L::Any, |o| self.dword(inst, o, k));
        if let Some(op) = base.strip_suffix("_saveexec") {
            return vec![(code + k, exec), (ir::EXEC + k, logic(op, a, exec))];
        }
        if let Some(op) = base.strip_suffix("_wrexec") {
            let new = logic(op, a, exec);
            return vec![(code + k, new), (ir::EXEC + k, new)];
        }
        let x = match base {
            "mov" | "wqm" => a,
            "not" => a.not(),
            "cselect" if a == b => a,
            "cselect" if a == L::Unset || b == L::Unset => a.join(b),
            "cselect" => L::Any,
            _ => logic(base, a, b),
        };
        vec![(code + k, x)]
    }

    /// What `inst` writes: the dwords it settles, everything else unknown.
    fn inst(&mut self, inst: &Inst, wave64: bool) {
        let mut settled: Vec<(u8, L)> = Vec::new();
        let name = inst.name();
        let wide = name.ends_with("_b64");
        match &inst.fields {
            Fields::Sop { dst: Some(Operand::Scalar { code, .. }), src, .. } if wide || name.ends_with("_b32") => {
                let base = name.trim_start_matches("s_").trim_end_matches("_b64").trim_end_matches("_b32");
                for k in 0..=u8::from(wide) {
                    settled.extend(self.sop(inst, base, *code, src, k));
                }
            }
            // a compare gives 0 to every lane outside EXEC
            Fields::Valu(v) if name.starts_with("v_cmp") => {
                for k in 0..=u8::from(wave64) {
                    let exec = inst.read(Reg::S(ir::EXEC + k)).map_or(L::Any, |v| self.get(v));
                    let x = if exec == L::Ones { L::Any } else { exec };
                    settled.push((ir::EXEC + k, x));
                    if let Some(Operand::Scalar { code, .. }) = v.dst {
                        settled.push((code + k, x));
                    }
                }
            }
            _ => {}
        }
        for &(reg, v) in &inst.writes {
            let x = match reg {
                Reg::S(c) => settled.iter().find(|s| s.0 == c).map_or(L::Any, |s| s.1),
                _ => L::Any,
            };
            self.set(v, x);
        }
    }
}

/// The scalar values whose bits at idle lanes are all 0 (false) or all 1
/// (true) (module comment).
pub(super) fn idle_bits(p: &Program) -> HashMap<Value, bool> {
    let mut pass = Pass { known: HashMap::new(), changed: true };
    for &(reg, v) in &p.inputs {
        pass.set(v, if matches!(reg, Reg::S(c) if c == ir::EXEC || c == ir::EXEC + 1) { L::Zero } else { L::Any });
    }
    while std::mem::take(&mut pass.changed) {
        for b in &p.blocks {
            for phi in &b.phis {
                let x = phi.args.iter().fold(L::Unset, |x, a| x.join(pass.get(*a)));
                pass.set(phi.value, x);
            }
            for inst in &b.insts {
                pass.inst(inst, p.wave64);
            }
        }
    }
    pass.known.into_iter().filter_map(|(v, x)| Some((v, match x { L::Zero => false, L::Ones => true, _ => return None }))).collect()
}
