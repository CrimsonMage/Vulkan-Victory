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
        let g = self.g.as_ref().unwrap();
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
            for (k, &ph) in phis.iter().enumerate() {
                self.ids[ph as usize] = entry[k];
            }
            let b = self.f.blocks[x].body.clone();
            self.items(&b);
            self.within(x, &merges);
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
            self.cons.push(Construct { node: z, head: usize::MAX, header: h, merge, cont, passes: false, wanted: 0, counter: None, arrivals: Vec::new() });
            self.loop_tree(x, rest, merges, exit, entry);
            self.close();
            self.tree(z);
            return;
        }
        let phis = self.f.blocks[x].phis.clone();
        let z = self.c32(0);
        let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
        let pre = self.block;
        self.branch_to(h);
        self.m.label(h);
        self.block = h;
        // the iteration count
        let u = self.t.u32_;
        let next = self.m.id();
        let count = self.m.op(op::PHI, u, &[z, pre, next, cont]);
        let counter = (count, next);
        let tys = self.phi_tys(x);
        let cids: Vec<Id> = phis.iter().map(|_| self.m.id()).collect();
        for (k, &ph) in phis.iter().enumerate() {
            let id = self.m.op(op::PHI, tys[k], &[entry[k], pre, cids[k], cont]);
            self.ids[ph as usize] = id;
        }
        self.cphi.insert(x, cids);
        self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
        let body = self.m.id();
        self.branch_to(body);
        self.m.label(body);
        self.block = body;
        self.ctx.push(Ctx::Loop(x));
        self.cons.push(Construct { node: x, head: usize::MAX, header: h, merge, cont, passes: false, wanted: 0, counter: Some(counter), arrivals: Vec::new() });
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
                    // both arms leave (an unreachable merge): jumps out of it
                    // to a selection's merge are not allowed
                    let c = self.id(c);
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
                self.cons.push(Construct { node: y, head: x, header: 0, merge, cont: 0, passes: false, wanted: 0, counter: None, arrivals: Vec::new() });
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
                self.cons.push(Construct { node: y, head: usize::MAX, header: h, merge, cont, passes: false, wanted: 0, counter: None, arrivals: Vec::new() });
                self.within(x, rest);
                self.close();
                self.tree(y);
            }
        }
    }

    /// Branches to the merge of construct `k`, noting the edge.
    pub(super) fn to_merge(&mut self, k: usize) {
        let m = self.cons[k].merge;
        let blk = self.block;
        self.merge_preds.entry(m).or_default().push(blk);
        self.branch_to(m);
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
        let (u, b) = (self.t.u32_, self.t.bool_);
        let h = c.header;
        if matches!(kind, Ctx::Sel(_)) {
            // no continue target
        } else {
        // continue target
        self.m.label(c.cont);
        self.block = c.cont;
        match c.counter {
            Some((count, next)) => {
                // the loop's back edges meet here
                let x = c.node;
                let cids = self.cphi.remove(&x).unwrap_or_default();
                let inc = self.back.remove(&x).unwrap_or_default();
                let tys = self.phi_tys(x);
                for (k, &t) in tys.iter().enumerate() {
                    let mut ops = Vec::new();
                    for (vals, blk) in &inc {
                        ops.push(vals[k]);
                        ops.push(*blk);
                    }
                    if ops.is_empty() {
                        let u = self.m.undef(t);
                        ops.push(u);
                        ops.push(h);
                    }
                    self.m.op_to(op::PHI, t, cids[k], &ops);
                }
                let one = self.c32(1);
                self.m.op_to(op::I_ADD, u, next, &[count, one]);
                let lim = self.c32(self.st.loop_limit);
                let over = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[next, lim]);
                // past the limit: leave through the merge, then return
                self.merge_preds.entry(c.merge).or_default().push(c.cont);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[over, c.merge, h]);
            }
            None => self.branch_to(h),
        }
        }
        self.m.label(c.merge);
        self.block = c.merge;
        let preds = self.merge_preds.remove(&c.merge).unwrap_or_default();
        // the block placed at this merge: a Block's node, or a loop's exit
        let at_merge = match kind {
            Ctx::Block(y) | Ctx::Sel(y) => Some(y),
            Ctx::Loop(x) => self.g.as_ref().unwrap().exit[x],
        };
        if let Some(y) = at_merge
            && !self.f.blocks[y].phis.is_empty()
        {
            let inc = self.fwd.remove(&y).unwrap_or_default();
            let tys = self.phi_tys(y);
            let ids = self.phis_at(&tys, &preds, &|p, k| inc.iter().find(|e| e.1 == p).map(|e| e.0[k]));
            self.pending.insert(y, ids);
        }
        // the ladder: which jump passes, with what values
        let mut code_id = 0;
        let mut ladder: Vec<(usize, Vec<Id>)> = Vec::new();
        let mut possible: Vec<u32> = c.arrivals.iter().flat_map(|a| a.possible.iter().copied()).collect();
        possible.sort_unstable();
        possible.dedup();
        if !c.arrivals.is_empty() {
            let zero = self.c32(0);
            let arr = &c.arrivals;
            let codes = self.phis_at(&[u], &preds, &|p, _| Some(arr.iter().find(|a| a.from == p).map_or(zero, |a| a.code)));
            code_id = codes[0];
            let mut targets: Vec<usize> = arr.iter().flat_map(|a| a.vals.iter().map(|v| v.0)).collect();
            targets.sort_unstable();
            targets.dedup();
            for t in targets {
                let tys = self.phi_tys(t);
                if tys.is_empty() {
                    ladder.push((t, Vec::new()));
                    continue;
                }
                let ids = self.phis_at(&tys, &preds, &|p, k| {
                    arr.iter().find(|a| a.from == p).and_then(|a| a.vals.iter().find(|v| v.0 == t)).map(|v| v.1[k])
                });
                ladder.push((t, ids));
            }
        }
        if c.counter.is_some() {
            // the iteration limit: arrived from the continue target
            let (t, f) = (self.m.c_bool(true), self.m.c_bool(false));
            let cont = c.cont;
            let over = self.phis_at(&[b], &preds, &|p, _| Some(if p == cont { t } else { f }))[0];
            let (lr, lc) = (self.m.id(), self.m.id());
            self.m.op_void(op::SELECTION_MERGE, &[lc, 0]);
            self.m.op_void(op::BRANCH_CONDITIONAL, &[over, lr, lc]);
            self.m.label(lr);
            self.m.op_void(op::RETURN, &[]);
            self.m.label(lc);
            self.block = lc;
        }
        let i = self.ctx.len();
        if i > 0 && code_id != 0 {
            let outer = i - 1;
            let z = self.c32(0);
            let outer_kind = self.ctx[outer];
            let codes: Vec<(u32, bool)> = match outer_kind {
                Ctx::Loop(_) => vec![(code(outer, true), true), (code(outer, false), false)],
                Ctx::Block(_) | Ctx::Sel(_) => vec![(code(outer, false), false)],
            };
            let of = |l: &Vec<(usize, Vec<Id>)>, t: usize| l.iter().find(|v| v.0 == t).map(|v| v.1.clone()).unwrap_or_default();
            let mut rest = possible.clone();
            for (k, cont) in codes {
                if !possible.contains(&k) {
                    continue;
                }
                rest.retain(|&x| x != k);
                let kk = self.c32(k);
                let is = self.m.op(op::I_EQUAL, b, &[code_id, kk]);
                let (lt, lf) = (self.m.id(), self.m.id());
                self.m.op_void(op::SELECTION_MERGE, &[lf, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[is, lt, lf]);
                self.m.label(lt);
                self.block = lt;
                let node = self.cons[outer].node;
                if cont {
                    let vals = of(&ladder, node);
                    self.back.entry(node).or_default().push((vals, lt));
                    let t = self.cons[outer].cont;
                    self.branch_to(t);
                } else {
                    let exit = match outer_kind {
                        Ctx::Block(y) | Ctx::Sel(y) => Some(y),
                        Ctx::Loop(x) => self.g.as_ref().unwrap().exit[x],
                    };
                    if let Some(y) = exit {
                        let vals = of(&ladder, y);
                        self.fwd.entry(y).or_default().push((vals, lt));
                    }
                    self.to_merge(outer);
                }
                self.m.label(lf);
                self.block = lf;
            }
            if !rest.is_empty() {
                // further out: pass the code and the values on
                let nz = self.m.op(op::I_NOT_EQUAL, b, &[code_id, z]);
                let (lt, lf) = (self.m.id(), self.m.id());
                self.m.op_void(op::SELECTION_MERGE, &[lf, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[nz, lt, lf]);
                self.m.label(lt);
                self.block = lt;
                self.cons[outer].arrivals.push(Arrival { from: lt, code: code_id, possible: rest, vals: ladder.clone() });
                self.to_merge(outer);
                self.m.label(lf);
                self.block = lf;
            }
        }
    }

    pub(super) fn jump(&mut self, from: usize, to: usize) {
        let g = self.g.as_ref().unwrap();
        let back = g.rpo[to] <= g.rpo[from];
        let merge = g.merge[to];
        let vals = self.edge_vals(from, to);
        let exit_of = (!back && merge)
            .then(|| self.ctx.iter().rposition(|c| matches!(*c, Ctx::Loop(h) if g.exit[h] == Some(to))))
            .flatten();
        let block_at = (!back && merge)
            .then(|| self.ctx.iter().rposition(|c| *c == Ctx::Block(to) || *c == Ctx::Sel(to)))
            .flatten();
        if back || merge {
            // the innermost of the construct ending at `to`: a Block, or a Loop it exits
            let i = if back {
                self.ctx.iter().rposition(|c| *c == Ctx::Loop(to)).expect("a back edge inside its loop")
            } else {
                match (exit_of, block_at) {
                    (Some(a), Some(b)) => a.max(b),
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => panic!("a merge after its block"),
                }
            };
            let inner = self.ctx.len() - 1;
            // direct: the innermost construct, or a loop or block construct
            // with only selections inside it (break and continue of the
            // innermost loop are allowed from selections)
            let direct = i == inner
                || (!matches!(self.ctx[i], Ctx::Sel(_)) && self.ctx[i + 1..].iter().all(|c| matches!(c, Ctx::Sel(_))));
            // leaving a selection with no merge of its own (the arms of a
            // two-way branch) for a selection's merge is not allowed either
            let into = if direct { i } else { inner };
            if matches!(self.ctx[into], Ctx::Sel(_)) && self.base.iter().any(|&m| m > into) {
                self.bad_sel.push(self.cons[into].head);
            }
            if direct {
                let blk = self.block;
                let list = if back { self.back.entry(to).or_default() } else { self.fwd.entry(to).or_default() };
                list.push((vals, blk));
                if back {
                    let t = self.cons[i].cont;
                    self.branch_to(t);
                } else {
                    self.to_merge(i);
                }
                return;
            }
            // out of several constructs: the code and the values ride a
            // ladder of phis out to the target. A selection cannot take a
            // dispatched branch out of a nested one: such a selection's
            // header becomes a block construct in the next attempt.
            for k in i..inner {
                if matches!(self.ctx[k], Ctx::Sel(_)) {
                    self.bad_sel.push(self.cons[k].head);
                }
            }
                for k in i + 1..=inner {
                self.cons[k].passes = true;
            }
            self.cons[i].wanted |= 1 << u8::from(back);
            let k = code(i, back);
            let code = self.c32(k);
            let blk = self.block;
            self.cons[inner].arrivals.push(Arrival { from: blk, code, possible: vec![k], vals: vec![(to, vals)] });
            self.to_merge(inner);
        } else {
            self.pending.insert(to, vals);
            self.tree(to);
        }
    }
}
