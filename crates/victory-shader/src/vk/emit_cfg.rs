//! Structured control flow: the block graph placed by its dominator tree; selections at joins, loops, block constructs where jumps leave several constructs (phi ladders).

use super::*;

impl Emit<'_> {
    /// The values the edge `from -> to` gives `to`'s phis.
    pub(super) fn edge_vals(&self, from: usize, to: usize) -> Vec<Id> {
        let b = &self.f.blocks[to];
        let k = b.preds.iter().position(|&q| q == from).expect("an edge of the graph");
        b.phis.iter().map(|&ph| self.id(self.f.node(ph).args[k])).collect()
    }

    pub(super) fn phi_tys(&mut self, x: usize) -> Vec<Id> {
        let tys: Vec<Ty> = self.f.blocks[x].phis.iter().map(|&p| self.f.ty(p)).collect();
        tys.into_iter().map(|t| self.ty(t)).collect()
    }

    pub(super) fn tree(&mut self, x: usize) {
        let g = self.g;
        let exit = if g.header[x] { g.exit[x] } else { None };
        let after: Vec<usize> = if exit.is_some() { g.after_exit[x].clone() } else { Vec::new() };
        let mut merges: Vec<usize> = g.children[x]
            .iter()
            .copied()
            .filter(|&c| g.merge[c] && Some(c) != exit && !after.contains(&c))
            .collect();
        merges.sort_by_key(|&c| std::cmp::Reverse(g.rpo[c]));
        let header = g.header[x];
        let entry = self.pending.remove(&x).unwrap_or_default();
        if header {
            self.loop_tree(x, &after, &merges, exit, entry);
        } else {
            let phis = self.f.blocks[x].phis.clone();
            self.cur = x;
            for (k, &ph) in phis.iter().enumerate() {
                self.set_id(ph, entry[k]);
            }
            let b = self.f.blocks[x].body.clone();
            self.items(&b);
            if merges.len() >= DISPATCH_MIN {
                self.dispatch_loop(x, &merges);
            } else {
                self.within(x, &merges);
            }
        }
    }

    /// A loop headed by `x`: the blocks of the merges that follow its exit
    /// wrap it; the exit comes at the loop construct's merge.
    pub(super) fn loop_tree(&mut self, x: usize, after: &[usize], merges: &[usize], exit: Option<usize>, entry: Vec<Id>) {
        if let Some((&z, rest)) = after.split_first() {
            let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
            self.branch_to(h);
            self.m.label(h);
            self.block = h;
            self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
            let body = self.m.id();
            self.branch_to(body);
            self.m.label(body);
            self.block = body;
            self.ctx.push(Ctx::Block(z));
            self.cons.push(Construct { node: z, head: usize::MAX, header: h, merge, cont, arrivals: Vec::new(), disp: None });
            self.loop_tree(x, rest, merges, exit, entry);
            self.close();
            self.tree(z);
            return;
        }
        let phis = self.f.blocks[x].phis.clone();
        let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
        let pre = self.block;
        self.branch_to(h);
        self.m.label(h);
        self.block = h;
        let tys = self.phi_tys(x);
        let cids: Vec<Id> = phis.iter().map(|_| self.m.id()).collect();
        self.cur = x;
        for (k, &ph) in phis.iter().enumerate() {
            let id = self.m.op(op::PHI, tys[k], &[entry[k], pre, cids[k], cont]);
            self.set_id(ph, id);
        }
        self.cphi.insert(x, cids);
        self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
        let body = self.m.id();
        self.branch_to(body);
        self.m.label(body);
        self.block = body;
        self.ctx.push(Ctx::Loop(x));
        self.cons.push(Construct { node: x, head: usize::MAX, header: h, merge, cont, arrivals: Vec::new(), disp: None });
        let b = self.f.blocks[x].body.clone();
        self.items(&b);
        self.within(x, merges);
        self.close();
        match exit {
            // the loop's exit follows it
            Some(y) => self.tree(y),
            // a loop is left only by jumps to outer constructs
            None => self.m.op_void(op::UNREACHABLE, &[]),
        }
    }

    pub(super) fn within(&mut self, x: usize, merges: &[usize]) {
        match merges.split_first() {
            None => match self.f.blocks[x].term.clone() {
                Term::Jump(t) => self.jump(x, t),
                Term::Branch(c, t, e) => {
                    let c = self.id(c);
                    // both arms leave for different blocks: a plain
                    // conditional branch (no selection, no unreachable merge)
                    if self.plain_branch(x, c, t, e) {
                        return;
                    }
                    // else a selection with an unreachable merge; jumps out
                    // of it to a selection's merge are not allowed
                    let (lt, le, sm) = (self.m.id(), self.m.id(), self.m.id());
                    self.m.op_void(op::SELECTION_MERGE, &[sm, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
                    self.base.push(self.ctx.len());
                    self.m.label(lt);
                    self.block = lt;
                    self.jump(x, t);
                    self.m.label(le);
                    self.block = le;
                    self.jump(x, e);
                    self.base.pop();
                    self.m.label(sm);
                    self.block = sm;
                    self.m.op_void(op::UNREACHABLE, &[]);
                }
                Term::Return => self.m.op_void(op::RETURN, &[]),
            },
            Some((&y, [])) if matches!(self.f.blocks[x].term, Term::Branch(..)) && !self.st.no_sel.contains(&x) => {
                // the branch's join: a selection construct merging at it
                let Term::Branch(c, t, e) = self.f.blocks[x].term.clone() else { unreachable!() };
                let c = self.id(c);
                let merge = self.m.id();
                self.ctx.push(Ctx::Sel(y));
                self.cons.push(Construct { node: y, head: x, header: 0, merge, cont: 0, arrivals: Vec::new(), disp: None });
                let header = self.block;
                let arm = |e: &mut Self, to: usize| -> Id { if to == y { merge } else { e.m.id() } };
                let (lt, le) = (arm(self, t), arm(self, e));
                self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
                for (to, l) in [(t, lt), (e, le)] {
                    if to == y {
                        // straight to the join
                        let vals = self.edge_vals(x, y);
                        self.fwd.entry(y).or_default().push((vals, header));
                        self.merge_preds.entry(merge).or_default().push(header);
                    } else {
                        self.m.label(l);
                        self.block = l;
                        self.jump(x, to);
                    }
                }
                self.close();
                self.tree(y);
            }
            Some((&y, rest)) => {
                let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
                self.branch_to(h);
                self.m.label(h);
                self.block = h;
                self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
                let body = self.m.id();
                self.branch_to(body);
                self.m.label(body);
                self.block = body;
                self.ctx.push(Ctx::Block(y));
                self.cons.push(Construct { node: y, head: usize::MAX, header: h, merge, cont, arrivals: Vec::new(), disp: None });
                self.within(x, rest);
                self.close();
                self.tree(y);
            }
        }
    }

    /// Branches to the merge of construct `k`, noting the edge.
    pub(super) fn to_merge(&mut self, k: usize) {
        let m = self.note_merge(k);
        self.branch_to(m);
    }

    /// Notes the current block as a predecessor of construct `k`'s merge;
    /// the merge.
    fn note_merge(&mut self, k: usize) -> Id {
        let m = self.cons[k].merge;
        let blk = self.block;
        self.merge_preds.entry(m).or_default().push(blk);
        m
    }

    /// Phis at a merge: per predecessor, the value it brings or undef.
    pub(super) fn phis_at(&mut self, tys: &[Id], preds: &[Id], given: &dyn Fn(Id, usize) -> Option<Id>) -> Vec<Id> {
        let mut out = Vec::new();
        for (k, &t) in tys.iter().enumerate() {
            let mut ops = Vec::new();
            for &p in preds {
                let v = match given(p, k) {
                    Some(v) => v,
                    None => self.m.undef(t),
                };
                ops.push(v);
                ops.push(p);
            }
            out.push(self.m.op(op::PHI, t, &ops));
        }
        out
    }

    /// Ends the innermost construct: writes its continue target and opens
    /// its merge block, where a jump to an outer construct passes on.
    pub(super) fn close(&mut self) {
        let c = self.cons.pop().unwrap();
        let kind = self.ctx.pop().unwrap();
        if !matches!(kind, Ctx::Sel(_)) {
            self.m.label(c.cont);
            self.block = c.cont;
            if matches!(kind, Ctx::Loop(_)) {
                self.back_phis(c.node, c.header);
            }
            if let Some(d) = &c.disp {
                self.disp_phis(d, c.header);
            }
            self.branch_to(c.header);
        }
        self.m.label(c.merge);
        self.block = c.merge;
        let preds = self.merge_preds.remove(&c.merge).unwrap_or_default();
        // the block placed at this merge: a Block's node, or a loop's exit
        if let Some(y) = self.at_merge(kind)
            && !self.f.blocks[y].phis.is_empty()
        {
            let inc = self.fwd.remove(&y).unwrap_or_default();
            let tys = self.phi_tys(y);
            let ids = self.phis_at(&tys, &preds, &|p, k| inc.iter().find(|e| e.1 == p).map(|e| e.0[k]));
            self.pending.insert(y, ids);
        }
        if c.arrivals.is_empty() || self.ctx.is_empty() {
            return;
        }
        // the block whose code reaches the merge on every path
        let dom = match kind {
            Ctx::Block(y) | Ctx::Sel(y) => self.g.idom[y],
            Ctx::Loop(x) | Ctx::Disp(x) => x,
        };
        let ladder = self.ladder(&c.arrivals, &preds, dom);
        self.dispatch(ladder);
    }

    /// The block placed at a construct's merge.
    fn at_merge(&self, kind: Ctx) -> Option<usize> {
        match kind {
            Ctx::Block(y) | Ctx::Sel(y) => Some(y),
            Ctx::Loop(x) => self.g.exit[x],
            Ctx::Disp(_) => None,
        }
    }

    /// A loop's back edges meet at its continue target: the phis that feed
    /// the header's.
    fn back_phis(&mut self, x: usize, h: Id) {
        let cids = self.cphi.remove(&x).unwrap_or_default();
        let inc = self.back.remove(&x).unwrap_or_default();
        let tys = self.phi_tys(x);
        for (k, &t) in tys.iter().enumerate() {
            let mut ops: Vec<Id> = inc.iter().flat_map(|(vals, blk)| [vals[k], *blk]).collect();
            if ops.is_empty() {
                ops = vec![self.m.undef(t), h];
            }
            self.m.op_to(op::PHI, t, cids[k], &ops);
        }
    }

    /// The ladder at a merge: which jump passes (its code, 0 for none) and
    /// the values for each target's phis.
    fn ladder(&mut self, arr: &[Arrival], preds: &[Id], dom: usize) -> Ladder {
        let mut possible: Vec<u32> = arr.iter().flat_map(|a| a.possible.iter().copied()).collect();
        possible.sort_unstable();
        possible.dedup();
        let zero = self.c32(0);
        let u = self.t.u32_;
        let code = self.phis_at(&[u], preds, &|p, _| Some(arr.iter().find(|a| a.from == p).map_or(zero, |a| a.code)))[0];
        let mut targets: Vec<usize> = arr.iter().flat_map(|a| a.vals.iter().map(|v| v.0)).collect();
        targets.sort_unstable();
        targets.dedup();
        let mut vals = Vec::new();
        for t in targets {
            let tys = self.phi_tys(t);
            let mut ids = Vec::new();
            for (k, &ty) in tys.iter().enumerate() {
                let given = |p: Id| arr.iter().find(|a| a.from == p).and_then(|a| a.vals.iter().find(|v| v.0 == t)).map(|v| v.1[k]);
                ids.push(match self.available(preds, &given, dom) {
                    Some(v) => v,
                    None => self.phis_at(&[ty], preds, &|p, _| given(p))[0],
                });
            }
            vals.push((t, ids));
        }
        Ladder { code, possible, vals }
    }

    /// The one value every jump arriving at a merge brings, if its code
    /// reaches the merge on every path (then it needs no phi).
    fn available(&self, preds: &[Id], given: &dyn Fn(Id) -> Option<Id>, dom: usize) -> Option<Id> {
        let mut vals = preds.iter().filter_map(|&p| given(p));
        let v = vals.next()?;
        if vals.any(|w| w != v) {
            return None;
        }
        let d = *self.def_at.get(&v)?;
        self.g.dominates(d, dom).then_some(v)
    }

    /// After a merge, the jumps the ladder carries: those that end at the
    /// next construct out (its continue or its merge) go there; the rest
    /// pass on through its merge.
    fn dispatch(&mut self, l: Ladder) {
        let outer = self.ctx.len() - 1;
        if self.cons[outer].disp.is_some() {
            return self.dispatch_cases(l);
        }
        let codes: &[bool] = if matches!(self.ctx[outer], Ctx::Loop(_)) { &[true, false] } else { &[false] };
        let mut rest = l.possible.clone();
        for &cont in codes {
            let k = code(outer, cont);
            if !l.possible.contains(&k) {
                continue;
            }
            rest.retain(|&x| x != k);
            let kk = self.c32(k);
            let is = self.m.op(op::I_EQUAL, self.t.bool_, &[l.code, kk]);
            let lt = self.open_if(is);
            let node = self.cons[outer].node;
            if cont {
                self.back.entry(node).or_default().push((l.of(node), lt));
                let t = self.cons[outer].cont;
                self.branch_to(t);
            } else {
                if let Some(y) = self.at_merge(self.ctx[outer]) {
                    self.fwd.entry(y).or_default().push((l.of(y), lt));
                }
                self.to_merge(outer);
            }
            self.close_if();
        }
        if !rest.is_empty() {
            // further out: pass the code and the values on
            let z = self.c32(0);
            let nz = self.m.op(op::I_NOT_EQUAL, self.t.bool_, &[l.code, z]);
            let lt = self.open_if(nz);
            self.cons[outer].arrivals.push(Arrival { from: lt, code: l.code, possible: rest, vals: l.vals });
            self.to_merge(outer);
            self.close_if();
        }
    }

    /// Opens the true arm of a selection that ends at the next `close_if`;
    /// the arm's label.
    pub(super) fn open_if(&mut self, cond: Id) -> Id {
        let (lt, lf) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[lf, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[cond, lt, lf]);
        self.m.label(lt);
        self.block = lt;
        self.if_merge.push(lf);
        lt
    }

    pub(super) fn close_if(&mut self) {
        let lf = self.if_merge.pop().unwrap();
        self.m.label(lf);
        self.block = lf;
    }

    pub(super) fn jump(&mut self, from: usize, to: usize) {
        let g = self.g;
        if g.rpo[to] > g.rpo[from] && !g.merge[to] {
            let vals = self.edge_vals(from, to);
            self.pending.insert(to, vals);
            self.tree(to);
            return;
        }
        let l = self.leave(from, to);
        self.branch_to(l);
    }

    /// A two-way branch whose arms leave without a selection: both for
    /// different blocks outside, or one outside and the other placed here
    /// (a conditional branch may have one target that is not a break,
    /// continue or merge). Whether it was.
    fn plain_branch(&mut self, x: usize, c: Id, t: usize, e: usize) -> bool {
        match (self.exit_label(x, t), self.exit_label(x, e)) {
            (Some(a), Some(b)) if a != b => {
                let (lt, le) = (self.leave(x, t), self.leave(x, e));
                self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
            }
            (Some(_), None) | (None, Some(_)) => {
                let out_t = self.exit_label(x, t).is_some();
                let (out, here) = if out_t { (t, e) } else { (e, t) };
                let lo = self.leave(x, out);
                let lh = self.m.id();
                let (lt, le) = if out_t { (lo, lh) } else { (lh, lo) };
                self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
                self.m.label(lh);
                self.block = lh;
                self.jump(x, here);
            }
            (Some(a), Some(b)) => return self.merged_branch(x, c, (t, e), a == b && a == self.cons[self.ctx.len() - 1].merge),
            _ => return false,
        }
        true
    }

    /// Both arms leave for the same block: one edge whose code (or the
    /// dispatch state) and values are picked by the condition. `inner`: the
    /// block is the innermost construct's merge (else a dispatch loop's
    /// continue target, or neither: no). Whether it was.
    fn merged_branch(&mut self, x: usize, c: Id, (t, e): (usize, usize), inner: bool) -> bool {
        let label = self.exit_label(x, t).unwrap();
        let disp = (0..self.cons.len()).rev().find(|&k| self.cons[k].disp.is_some() && self.cons[k].cont == label);
        if !inner && disp.is_none() {
            return false;
        }
        let k = self.ctx.len() - 1;
        let before = self.cons[k].arrivals.len();
        self.leave(x, t);
        let mid = self.cons[k].arrivals.len();
        self.leave(x, e);
        let after = self.cons[k].arrivals.len();
        if inner {
            if let Some(p) = self.merge_preds.get_mut(&label) {
                p.pop();
            }
            self.merge_arrivals(k, c, before, mid, after);
        } else if let Some(i) = disp {
            self.merge_conts(i, c);
        }
        self.branch_to(label);
        true
    }

    /// The arrivals the two arms noted at construct `k`'s merge
    /// (`before..mid` the first arm's, `mid..after` the second's) as one: a
    /// direct arm brings code 0.
    fn merge_arrivals(&mut self, k: usize, c: Id, before: usize, mid: usize, after: usize) {
        let u = self.t.u32_;
        let z = self.c32(0);
        let mut arr: Vec<Arrival> = self.cons[k].arrivals.drain(before..after).collect();
        let (ct, ce) = match (mid - before, after - mid) {
            (1, 1) => (arr[0].code, arr[1].code),
            (1, 0) => (arr[0].code, z),
            (0, 1) => (z, arr[0].code),
            _ => unreachable!("an arm noted no arrival"),
        };
        let code = self.m.op(op::SELECT, u, &[c, ct, ce]);
        let mut a = arr.remove(0);
        for b in arr {
            a.possible.extend(b.possible);
            a.vals.extend(b.vals);
        }
        a.code = code;
        self.cons[k].arrivals.push(a);
    }

    /// The two continue edges the arms noted at dispatch loop `i` as one.
    fn merge_conts(&mut self, i: usize, c: Id) {
        let d = self.cons[i].disp.as_mut().unwrap();
        let (b, a) = (d.conts.pop().unwrap(), d.conts.pop().unwrap());
        let tys = d.slot_tys.clone();
        let u = self.t.u32_;
        let state = self.m.op(op::SELECT, u, &[c, a.0, b.0]);
        let slots: Vec<Id> = (0..tys.len()).map(|s| self.m.op(op::SELECT, tys[s], &[c, a.1[s], b.1[s]])).collect();
        self.cons[i].disp.as_mut().unwrap().conts.push((state, slots, a.2));
    }

    /// Whether construct `i` takes a jump directly: the innermost
    /// construct, or a loop or block construct with only selections inside
    /// it (break and continue of the innermost loop are allowed from
    /// selections).
    fn direct(&self, i: usize) -> bool {
        i == self.ctx.len() - 1 || (!matches!(self.ctx[i], Ctx::Sel(_)) && self.ctx[i + 1..].iter().all(|c| matches!(c, Ctx::Sel(_))))
    }

    /// The label a jump to `to` branches to (none: `to` is placed here).
    fn exit_label(&self, from: usize, to: usize) -> Option<Id> {
        let back = self.g.rpo[to] <= self.g.rpo[from];
        if !back && !self.g.merge[to] {
            return None;
        }
        let i = self.target_construct(to, back);
        let c = &self.cons[i];
        Some(match self.direct(i) {
            true if back || c.disp.is_some() => c.cont,
            true => c.merge,
            false => self.cons[self.ctx.len() - 1].merge,
        })
    }

    /// A jump out to `to` (a back edge or a merge), noted where its target
    /// takes it; the label to branch to.
    fn leave(&mut self, from: usize, to: usize) -> Id {
        let back = self.g.rpo[to] <= self.g.rpo[from];
        let vals = self.edge_vals(from, to);
        let i = self.target_construct(to, back);
        let direct = self.direct(i);
        // leaving a selection with no merge of its own (the arms of a
        // two-way branch) for a selection's merge is not allowed either
        let into = if direct { i } else { self.ctx.len() - 1 };
        if matches!(self.ctx[into], Ctx::Sel(_)) && self.base.iter().any(|&m| m > into) {
            self.bad_sel.push(self.cons[into].head);
        }
        if direct { self.leave_direct(i, to, back, vals) } else { self.leave_passing(i, to, back, vals) }
    }

    /// The construct a jump to `to` ends in: the loop it continues, or the
    /// innermost block, selection or loop (by its exit) merging at it.
    fn target_construct(&self, to: usize, back: bool) -> usize {
        let g = self.g;
        if back {
            return self.ctx.iter().rposition(|c| *c == Ctx::Loop(to)).expect("a back edge inside its loop");
        }
        let exit_of = self.ctx.iter().rposition(|c| matches!(*c, Ctx::Loop(h) if g.exit[h] == Some(to)));
        let block_at = self.ctx.iter().rposition(|c| *c == Ctx::Block(to) || *c == Ctx::Sel(to));
        let case_at = (0..self.ctx.len()).rev().find(|&k| self.cons[k].disp.as_ref().is_some_and(|d| d.cases.iter().any(|c| c.0 == to)));
        exit_of.max(block_at).max(case_at).expect("a merge after its block")
    }

    /// A break or continue of construct `i`.
    fn leave_direct(&mut self, i: usize, to: usize, back: bool, vals: Vec<Id>) -> Id {
        if self.cons[i].disp.is_some() {
            return self.disp_edge(i, to, &vals);
        }
        let blk = self.block;
        let list = if back { self.back.entry(to).or_default() } else { self.fwd.entry(to).or_default() };
        list.push((vals, blk));
        if back { self.cons[i].cont } else { self.note_merge(i) }
    }

    /// Out of several constructs: the code and the values ride a ladder of
    /// phis out to the target. A selection cannot take a dispatched branch
    /// out of a nested one: such a selection's header becomes a block
    /// construct in the next attempt.
    fn leave_passing(&mut self, i: usize, to: usize, back: bool, vals: Vec<Id>) -> Id {
        let inner = self.ctx.len() - 1;
        for k in i..inner {
            if matches!(self.ctx[k], Ctx::Sel(_)) {
                self.bad_sel.push(self.cons[k].head);
            }
        }
        let k = match &self.cons[i].disp {
            Some(d) => code_case(i, d.cases.iter().position(|c| c.0 == to).unwrap()),
            None => code(i, back),
        };
        let code = self.c32(k);
        let blk = self.block;
        self.cons[inner].arrivals.push(Arrival { from: blk, code, possible: vec![k], vals: vec![(to, vals)] });
        self.note_merge(inner)
    }
}

/// The jumps a merge passes on: the code of the one taken (0: none), the
/// codes possible, and each target's phi values.
pub(super) struct Ladder {
    pub(super) code: Id,
    pub(super) possible: Vec<u32>,
    pub(super) vals: Vec<(usize, Vec<Id>)>,
}

impl Ladder {
    pub(super) fn of(&self, t: usize) -> Vec<Id> {
        self.vals.iter().find(|v| v.0 == t).map(|v| v.1.clone()).unwrap_or_default()
    }
}

