//! Dispatch loops: a block with many merge children places them as the
//! cases of one loop instead of a nest of block constructs, so a jump to
//! one is a continue (with the case's number and its phis' values) rather
//! than a ladder out of every construct of the nest.

use super::*;
use super::cfg::Ladder;

impl Emit<'_> {
    /// The loop over `x`'s terminator (case 0) and its merge children.
    pub(super) fn dispatch_loop(&mut self, x: usize, merges: &[usize]) {
        let mut slot_tys: Vec<Id> = Vec::new();
        let mut cases = Vec::new();
        for &z in merges {
            let mut used: Vec<usize> = Vec::new();
            let mut map = Vec::new();
            for t in self.phi_tys(z) {
                let s = (0..slot_tys.len()).find(|&s| slot_tys[s] == t && !used.contains(&s)).unwrap_or_else(|| {
                    slot_tys.push(t);
                    slot_tys.len() - 1
                });
                used.push(s);
                map.push(s);
            }
            cases.push((z, map));
        }
        let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
        let pre = self.block;
        self.branch_to(h);
        self.m.label(h);
        self.block = h;
        let u = self.t.u32_;
        let (cstate, zero) = (self.m.id(), self.c32(0));
        let state = self.m.op(op::PHI, u, &[zero, pre, cstate, cont]);
        let cslots: Vec<Id> = slot_tys.iter().map(|_| self.m.id()).collect();
        let mut slots = Vec::new();
        for (s, &t) in slot_tys.iter().enumerate() {
            let un = self.m.undef(t);
            slots.push(self.m.op(op::PHI, t, &[un, pre, cslots[s], cont]));
        }
        self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
        let body = self.m.id();
        self.branch_to(body);
        self.m.label(body);
        self.block = body;
        let n = cases.len();
        let disp = Disp { cases, slot_tys, state, slots, cstate, cslots, conts: Vec::new() };
        self.ctx.push(Ctx::Disp(x));
        self.cons.push(Construct { node: x, head: usize::MAX, header: h, merge, cont, arrivals: Vec::new(), disp: Some(Box::new(disp)) });
        self.base.push(self.ctx.len());
        self.disp_cases(x, 0, n + 1);
        self.base.pop();
        self.close();
        // left only by jumps to outer constructs, dispatched at the merge
        self.m.op_void(op::UNREACHABLE, &[]);
    }

    /// Cases `lo..hi` by a binary search on the state.
    fn disp_cases(&mut self, x: usize, lo: usize, hi: usize) {
        if hi - lo > 1 {
            let mid = (lo + hi) / 2;
            let k = self.c32(mid as u32);
            let state = self.cons.last().unwrap().disp.as_ref().unwrap().state;
            let c = self.m.op(op::U_LESS_THAN, self.t.bool_, &[state, k]);
            let (lt, le, sm) = (self.m.id(), self.m.id(), self.m.id());
            self.m.op_void(op::SELECTION_MERGE, &[sm, 0]);
            self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
            self.m.label(lt);
            self.block = lt;
            self.disp_cases(x, lo, mid);
            self.m.label(le);
            self.block = le;
            self.disp_cases(x, mid, hi);
            self.m.label(sm);
            self.block = sm;
            self.m.op_void(op::UNREACHABLE, &[]);
            return;
        }
        if lo == 0 {
            // the block's own branch
            self.cur = x;
            self.within(x, &[]);
            return;
        }
        let d = self.cons.last().unwrap().disp.as_ref().unwrap();
        let (z, map) = &d.cases[lo - 1];
        let vals: Vec<Id> = map.iter().map(|&s| d.slots[s]).collect();
        let z = *z;
        self.pending.insert(z, vals);
        self.tree(z);
    }

    /// A jump into a case of the dispatch loop at stack position `i`.
    pub(super) fn disp_continue(&mut self, i: usize, to: usize, vals: &[Id]) {
        let d = self.cons[i].disp.as_ref().unwrap();
        let j = d.cases.iter().position(|c| c.0 == to).unwrap();
        let map = d.cases[j].1.clone();
        let tys = d.slot_tys.clone();
        let mut full: Vec<Id> = tys.iter().map(|&t| self.m.undef(t)).collect();
        for (k, &s) in map.iter().enumerate() {
            full[s] = vals[k];
        }
        let state = self.c32(j as u32 + 1);
        let blk = self.block;
        let cont = self.cons[i].cont;
        self.cons[i].disp.as_mut().unwrap().conts.push((state, full, blk));
        self.branch_to(cont);
    }

    /// The continue target's phis: the next state and slot values.
    pub(super) fn disp_phis(&mut self, d: &Disp, h: Id) {
        let u = self.t.u32_;
        let edges = |pick: &dyn Fn(&(Id, Vec<Id>, Id)) -> Id| -> Vec<Id> { d.conts.iter().flat_map(|e| [pick(e), e.2]).collect() };
        let mut ops = edges(&|e| e.0);
        if ops.is_empty() {
            ops = vec![self.m.undef(u), h];
        }
        self.m.op_to(op::PHI, u, d.cstate, &ops);
        for (s, &t) in d.slot_tys.iter().enumerate() {
            let mut ops = edges(&|e| e.1[s]);
            if ops.is_empty() {
                ops = vec![self.m.undef(t), h];
            }
            self.m.op_to(op::PHI, t, d.cslots[s], &ops);
        }
    }

    /// At a merge inside a dispatch loop: jumps into its cases continue it,
    /// the rest pass on through its merge.
    pub(super) fn dispatch_cases(&mut self, l: Ladder) {
        let outer = self.ctx.len() - 1;
        let cases: Vec<usize> = self.cons[outer].disp.as_ref().unwrap().cases.iter().map(|c| c.0).collect();
        let mut rest = l.possible.clone();
        for (j, &z) in cases.iter().enumerate() {
            let k = code_case(outer, j);
            if !l.possible.contains(&k) {
                continue;
            }
            rest.retain(|&x| x != k);
            let kk = self.c32(k);
            let is = self.m.op(op::I_EQUAL, self.t.bool_, &[l.code, kk]);
            self.open_if(is);
            self.disp_continue(outer, z, &l.of(z));
            self.close_if();
        }
        if !rest.is_empty() {
            let z = self.c32(0);
            let nz = self.m.op(op::I_NOT_EQUAL, self.t.bool_, &[l.code, z]);
            let lt = self.open_if(nz);
            self.cons[outer].arrivals.push(Arrival { from: lt, code: l.code, possible: rest, vals: l.vals });
            self.to_merge(outer);
            self.close_if();
        }
    }
}
