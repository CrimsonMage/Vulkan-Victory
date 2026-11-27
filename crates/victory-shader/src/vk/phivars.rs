//! Phis of many operands as Function variables, on a finished module.
//!
//! The emitter's phi ladders (emit_cfg.rs) carry a jump's values out of
//! nested constructs as phis at every merge they pass, with undef on every
//! other predecessor: CS_ScreenProbes_ActivateRays had 900 phis of 282
//! operands at one merge, two thirds of its 3.4M words. A phi of more than
//! [`MIN_OPERANDS`] (value, predecessor) pairs becomes a variable: a store
//! of the value at the end of each predecessor that gives one (an undef
//! gives none), a load at the start of the phi's block.
//!
//! Why it means the same: an operand `(v, p)` of a phi in block `b` is
//! `v` as control leaves `p` for `b`; `v` dominates the end of `p` (a
//! phi's rule), so a store of `v` placed last in `p` (before its merge
//! instruction and branch) is valid and runs on every edge `p -> b`. The
//! load opens `b`; between that store and the load there is only the edge,
//! so it reads `v`. When `p` branches elsewhere the store is dead unless
//! control reaches `b` through another predecessor, whose own store
//! (later, on that path) wins, as the phi would take that predecessor's
//! operand. A predecessor with undef stores nothing: the load reads an
//! earlier value, which an undef allows. Each phi has its own variable,
//! and all stores of a block's predecessors precede all loads of the
//! block, so phis that read each other (loop-carried swaps) keep their
//! parallel meaning. Other phis are left as they are.

use std::collections::{HashMap, HashSet};

/// Phis of more (value, predecessor) pairs become variables.
pub const MIN_OPERANDS: usize = 16;

const OP_UNDEF: u32 = 1;
const OP_TYPE_POINTER: u32 = 32;
const OP_FUNCTION: u32 = 54;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_PHI: u32 = 245;
const OP_LABEL: u32 = 248;
const MERGES: [u32; 2] = [246, 247];
const TERMINATORS: [u32; 7] = [249, 250, 251, 252, 253, 254, 255];
const FUNCTION_STORAGE: u32 = 7;

/// A phi that becomes a variable.
struct Big {
    func: usize,
    block: u32,
    ty: u32,
    result: u32,
    pairs: Vec<(u32, u32)>,
}

/// The module's instructions as (start, length) word ranges.
fn instructions(words: &[u32]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = ((words[i] >> 16) as usize).max(1);
        out.push((i, n.min(words.len() - i)));
        i += n;
    }
    out
}

/// Instruction `k`'s opcode and word `i`.
struct Module<'a> {
    words: &'a [u32],
    ins: Vec<(usize, usize)>,
}

impl Module<'_> {
    fn op(&self, k: usize) -> u32 {
        self.words[self.ins[k].0] & 0xFFFF
    }
    fn at(&self, k: usize, i: usize) -> u32 {
        self.words[self.ins[k].0 + i]
    }
    fn len(&self, k: usize) -> usize {
        self.ins[k].1
    }

    /// The phis of more than [`MIN_OPERANDS`] pairs, by function and block.
    fn big_phis(&self) -> Vec<Big> {
        let (mut big, mut func, mut block) = (Vec::new(), 0usize, 0u32);
        for k in 0..self.ins.len() {
            match self.op(k) {
                OP_FUNCTION => func += 1,
                OP_LABEL => block = self.at(k, 1),
                OP_PHI if (self.len(k) - 3) / 2 > MIN_OPERANDS => {
                    let pairs = (0..(self.len(k) - 3) / 2).map(|p| (self.at(k, 3 + 2 * p), self.at(k, 4 + 2 * p))).collect();
                    big.push(Big { func, block, ty: self.at(k, 1), result: self.at(k, 2), pairs });
                }
                _ => {}
            }
        }
        big
    }
}

/// What the rewrite adds: pointer types, per function its variables, per
/// block the stores at its end and the loads at its start.
#[derive(Default)]
struct Plan {
    bound: u32,
    new_pointers: Vec<(u32, u32)>,
    entry_vars: HashMap<usize, Vec<(u32, u32)>>,
    stores: HashMap<u32, Vec<(u32, u32)>>,
    loads: HashMap<u32, Vec<(u32, u32, u32)>>,
    gone: HashSet<u32>,
}

fn plan(m: &Module<'_>, big: &[Big]) -> Plan {
    let undef: HashSet<u32> = (0..m.ins.len()).filter(|&k| m.op(k) == OP_UNDEF).map(|k| m.at(k, 2)).collect();
    let mut pointers: HashMap<u32, u32> = (0..m.ins.len())
        .filter(|&k| m.op(k) == OP_TYPE_POINTER && m.at(k, 2) == FUNCTION_STORAGE)
        .map(|k| (m.at(k, 3), m.at(k, 1)))
        .collect();
    let mut p = Plan { bound: m.words[3], ..Plan::default() };
    for b in big {
        if let std::collections::hash_map::Entry::Vacant(e) = pointers.entry(b.ty) {
            e.insert(p.bound);
            p.new_pointers.push((p.bound, b.ty));
            p.bound += 1;
        }
        let var = p.bound;
        p.bound += 1;
        for &(v, pred) in &b.pairs {
            if !undef.contains(&v) {
                p.stores.entry(pred).or_default().push((var, v));
            }
        }
        p.loads.entry(b.block).or_default().push((b.ty, b.result, var));
        p.entry_vars.entry(b.func).or_default().push((pointers[&b.ty], var));
        p.gone.insert(b.result);
    }
    p
}

/// `words` with its phis of more than [`MIN_OPERANDS`] pairs as variables;
/// nothing before word `keep` moves (positions the caller holds). `None`:
/// no such phi (or none that may change).
pub fn phis_to_vars(words: &[u32], keep: usize) -> Option<Vec<u32>> {
    let m = Module { words, ins: instructions(words) };
    let first_fn = (0..m.ins.len()).find(|&k| m.op(k) == OP_FUNCTION)?;
    if m.ins[first_fn].0 < keep {
        return None;
    }
    let big = m.big_phis();
    if big.is_empty() {
        return None;
    }
    let p = plan(&m, &big);
    let mut out = words[..5].to_vec();
    out[3] = p.bound;
    let mut r = Rewrite { p, out, func: 0, first_label: false, block: 0, pending: Vec::new() };
    for k in 0..m.ins.len() {
        if k == first_fn {
            for &(ptr, t) in &r.p.new_pointers {
                r.out.extend_from_slice(&[4 << 16 | OP_TYPE_POINTER, ptr, FUNCTION_STORAGE, t]);
            }
        }
        r.instruction(&m, k);
    }
    Some(r.out)
}

/// The rewrite's state as it walks the module.
struct Rewrite {
    p: Plan,
    out: Vec<u32>,
    func: usize,
    first_label: bool,
    block: u32,
    pending: Vec<(u32, u32, u32)>,
}

impl Rewrite {
    fn instruction(&mut self, m: &Module<'_>, k: usize) {
        let o = m.op(k);
        match o {
            OP_FUNCTION => (self.func, self.first_label) = (self.func + 1, true),
            OP_PHI if self.p.gone.contains(&m.at(k, 2)) => return,
            _ => {}
        }
        // the loads, once the block's phis (and an entry's variables) are past
        if !self.pending.is_empty() && o != OP_PHI && o != OP_VARIABLE {
            for (t, r, v) in self.pending.drain(..) {
                self.out.extend_from_slice(&[4 << 16 | OP_LOAD, t, r, v]);
            }
        }
        if MERGES.contains(&o) || TERMINATORS.contains(&o) {
            for (v, val) in self.p.stores.remove(&self.block).unwrap_or_default() {
                self.out.extend_from_slice(&[3 << 16 | OP_STORE, v, val]);
            }
        }
        self.out.extend_from_slice(&m.words[m.ins[k].0..m.ins[k].0 + m.len(k)]);
        if o == OP_LABEL {
            self.block = m.at(k, 1);
            if std::mem::take(&mut self.first_label) {
                for &(ptr, v) in self.p.entry_vars.get(&self.func).map(Vec::as_slice).unwrap_or(&[]) {
                    self.out.extend_from_slice(&[4 << 16 | OP_VARIABLE, ptr, v, FUNCTION_STORAGE]);
                }
            }
            self.pending = self.p.loads.remove(&self.block).unwrap_or_default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A module whose block 30 has a phi of 17 pairs, one of them undef:
    /// the phi goes, 16 stores and one load come, the variable opens the
    /// function.
    #[test]
    fn a_phi_of_many_operands_becomes_a_variable() {
        let mut m = vec![0x0723_0203, 0x0001_0600, 0, 100, 0];
        m.extend([4 << 16 | 21, 2, 32, 0]); // %2 = OpTypeInt 32 0
        m.extend([3 << 16 | OP_UNDEF, 2, 3]); // %3 = OpUndef %2
        m.extend([5 << 16 | OP_FUNCTION, 2, 4, 0, 5]);
        m.extend([2 << 16 | OP_LABEL, 6]);
        m.extend([2 << 16 | 249, 40]); // branch %40
        for p in 0..17u32 {
            m.extend([2 << 16 | OP_LABEL, 40 + p]);
            m.extend([2 << 16 | 249, 30]);
        }
        let mut phi = vec![(3 + 2 * 17) << 16 | OP_PHI, 2, 31];
        for p in 0..17u32 {
            phi.extend([if p == 5 { 3 } else { 60 + p }, 40 + p]);
        }
        m.extend([2 << 16 | OP_LABEL, 30]);
        m.extend(phi);
        m.extend([1 << 16 | 253, 1 << 16 | 56]); // OpReturn, OpFunctionEnd
        let out = phis_to_vars(&m, 0).expect("a big phi");
        let ins = instructions(&out);
        let count = |o: u32| ins.iter().filter(|&&(s, _)| out[s] & 0xFFFF == o).count();
        assert_eq!(count(OP_PHI), 0);
        assert_eq!(count(OP_STORE), 16);
        assert_eq!(count(OP_LOAD), 1);
        assert_eq!(count(OP_VARIABLE), 1);
        assert_eq!(count(OP_TYPE_POINTER), 1);
        // the variable right after the function's first label
        let label = ins.iter().position(|&(s, _)| out[s] & 0xFFFF == OP_LABEL).unwrap();
        assert_eq!(out[ins[label + 1].0] & 0xFFFF, OP_VARIABLE);
        assert!(phis_to_vars(&out, 0).is_none());
    }
}
