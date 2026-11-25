//! The control-flow graph in the shape compiled HLSL has, for converters
//! that only expect that shape.
//!
//! The lowering keeps the console program's graph (`structure.rs`): any
//! reducible graph is valid DXIL. The console's compiler branches past work
//! no lane needs to the program's end from anywhere, loops included, so a
//! loop is often left for several blocks, one of which other paths outside
//! the loop also reach. DXC's output never has that: a loop is left through
//! one block, which only the loop branches to, and a function returns once.
//!
//! dxil-spirv's structurizer before its rewrite (dxil-spirv 71967a3,
//! merged after vkd3d-proton 3.1 was tagged; Proton's vkd3d-proton ships
//! the old one) turns such loops into invalid SPIR-V ("exits the loop ...
//! not via a structured exit", a block without a terminator, a branch to a
//! missing block), or crashes; the NVIDIA driver's compiler then faults on
//! the module. So before the module is written:
//!
//! - every return goes through one block, and a short block that only
//!   goes on to it is copied for each block that jumps there;
//! - every loop, innermost first, has one block that branches back to its
//!   header, and is left through one block of its own: the edges leaving
//!   it go there instead, and that block goes on to where each edge went,
//!   by the number of the edge it came from (a chain of compares, as DXC
//!   lowers a flag set before a `break`). A number made in the loop and
//!   read after it is passed through that block's phis; an address or a
//!   resource handle (which no phi may carry) is made again there.
//!
//! Each path through the graph runs the same instructions as before, with
//! the compares in between, so the module computes the same.

use std::collections::HashMap;

use super::{Module, Op, Term, Ty, V, pred, successors};

/// The reached blocks, their order, predecessors and immediate dominators.
struct Graph {
    /// Reverse postorder of the reached blocks.
    order: Vec<usize>,
    /// Per block: its index in `order`, `usize::MAX` if not reached.
    number: Vec<usize>,
    /// Per block: the reached blocks that branch to it, once per block.
    preds: Vec<Vec<usize>>,
    idom: Vec<usize>,
}

impl Graph {
    fn new(m: &Module) -> Graph {
        let n = m.blocks.len();
        let succs = |b: usize| successors(m.blocks[b].term);
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            top.1 += 1;
            match succs(b).get(k) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let order = post;
        let mut number = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            number[b] = k;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for &s in succs(b).iter() {
                if preds[s].last() != Some(&b) {
                    preds[s].push(b);
                }
            }
        }
        // Cooper, Harvey and Kennedy.
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &order[1..] {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    if new == usize::MAX {
                        new = p;
                        continue;
                    }
                    let (mut x, mut y) = (p, new);
                    while x != y {
                        while number[x] > number[y] {
                            x = idom[x];
                        }
                        while number[y] > number[x] {
                            y = idom[y];
                        }
                    }
                    new = x;
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        Graph {
            order,
            number,
            preds,
            idom,
        }
    }

    fn reached(&self, b: usize) -> bool {
        self.number[b] != usize::MAX
    }

    fn dominates(&self, a: usize, mut b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            if b == 0 || self.idom[b] == usize::MAX {
                return false;
            }
            b = self.idom[b];
        }
    }

    /// Natural loops: (header, member flags), innermost (smallest) first.
    fn loops(&self, m: &Module) -> Vec<(usize, Vec<bool>)> {
        let n = m.blocks.len();
        let mut by_header: HashMap<usize, Vec<bool>> = HashMap::new();
        for &b in &self.order {
            for &h in successors(m.blocks[b].term).iter() {
                if !self.dominates(h, b) {
                    continue;
                }
                let body = by_header.entry(h).or_insert_with(|| {
                    let mut v = vec![false; n];
                    v[h] = true;
                    v
                });
                let mut work = vec![b];
                while let Some(x) = work.pop() {
                    if body[x] {
                        continue;
                    }
                    body[x] = true;
                    work.extend(self.preds[x].iter().copied());
                }
            }
        }
        let mut loops: Vec<(usize, Vec<bool>)> = by_header.into_iter().collect();
        loops.sort_by_key(|(h, body)| (body.iter().filter(|&&x| x).count(), self.number[*h]));
        loops
    }
}

/// Calls `f` with a mutable reference to every value `op` reads.
fn for_each_operand_mut(op: &mut Op, mut f: impl FnMut(&mut V)) {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
            f(a);
            f(b);
        }
        Op::Cast(_, v) | Op::Extract(v, _) | Op::Load(v) => f(v),
        Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
            f(a);
            f(b);
            f(c);
        }
        Op::Call(_, args) => args.iter_mut().for_each(f),
        Op::Phi(inc) => inc.iter_mut().for_each(|x| f(&mut x.0)),
        Op::Gep(_, p, idx) => {
            f(p);
            idx.iter_mut().for_each(f);
        }
    }
}

/// Most loops a program is given exits of their own for; past it the graph
/// is left as it is (a guard against a bug looping here, not a real limit).
const MAX_ROUNDS: usize = 4096;

/// Most instructions a block going straight to the return has for
/// [`Module::copy_tails`] to copy it.
const MAX_TAIL: usize = 16;

impl Module {
    /// One return, and one exit block of its own for every loop (see the
    /// module documentation).
    pub(crate) fn single_exits(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        if let Some(ret) = self.single_return() {
            self.copy_tails(ret);
        }
        // The exit blocks made for each loop (by header): they count as the
        // loop's for the loops around it, so those leave from there and the
        // inner loop keeps its exit to itself.
        let mut exit_of: Vec<Option<usize>> = vec![None; self.blocks.len()];
        for _ in 0..MAX_ROUNDS {
            let g = Graph::new(self);
            let mut changed = false;
            for (h, body) in g.loops(self) {
                let mut region = body.clone();
                region.resize(self.blocks.len(), false);
                for (b, owner) in exit_of.iter().enumerate() {
                    if let Some(o) = *owner
                        && o != h
                        && body[o]
                        && g.reached(b)
                    {
                        region[b] = true;
                    }
                }
                if self.give_loop_one_latch(&g, &body, h)
                    || self.give_loop_one_exit(&g, &region, &mut exit_of, h)
                {
                    changed = true;
                    break;
                }
            }
            if !changed {
                return;
            }
        }
    }

    /// The block that returns, made one if there were several.
    fn single_return(&mut self) -> Option<usize> {
        let g = Graph::new(self);
        let rets: Vec<usize> = g
            .order
            .iter()
            .copied()
            .filter(|&b| matches!(self.blocks[b].term, Some(Term::Ret)))
            .collect();
        if rets.len() < 2 {
            return rets.first().copied();
        }
        let ret = self.new_block();
        self.terminate(ret, Term::Ret);
        for b in rets {
            self.terminate(b, Term::Br(ret));
        }
        Some(ret)
    }

    /// A short block that goes straight to the return and is branched to
    /// from several places (the console code's jump to its last exports,
    /// from inside loops and ifs alike) is copied for each of them, so the
    /// return is the only block those jumps share, as an early `return` in
    /// HLSL compiles. Each path runs the same instructions as before.
    fn copy_tails(&mut self, ret: usize) {
        if !self.blocks[ret].phis.is_empty() {
            return;
        }
        let g = Graph::new(self);
        // Instructions read outside their own block.
        let mut home: HashMap<u32, usize> = HashMap::new();
        for &b in &g.order {
            let blk = &self.blocks[b];
            for &i in blk.phis.iter().chain(&blk.body) {
                home.insert(i, b);
            }
        }
        let mut read_elsewhere = vec![false; self.blocks.len()];
        for &b in &g.order {
            let blk = &self.blocks[b];
            let mut see = |v: V| {
                if let V::Inst(i) = v
                    && let Some(&h) = home.get(&i)
                    && h != b
                {
                    read_elsewhere[h] = true;
                }
            };
            for &i in blk.phis.iter().chain(&blk.body) {
                super::for_each_operand(&self.insts[i as usize].op, &mut see);
            }
            if let Some(Term::CondBr(c, ..)) = blk.term {
                see(c);
            }
        }
        for &b in &g.order {
            let blk = &self.blocks[b];
            if b == 0
                || !matches!(blk.term, Some(Term::Br(r)) if r == ret)
                || g.preds[b].len() < 2
                || blk.body.len() > MAX_TAIL
                || read_elsewhere[b]
                || !blk.body.iter().all(|&i| self.copyable(i))
            {
                continue;
            }
            let (phis, body) = (blk.phis.clone(), blk.body.clone());
            for &p in &g.preds[b] {
                let copy = self.new_block();
                let mut map: HashMap<u32, V> = HashMap::new();
                for &i in &phis {
                    if let Op::Phi(inc) = &self.insts[i as usize].op {
                        let v = inc.iter().find(|&&(_, q)| q == p).map(|x| x.0);
                        let v = v.unwrap_or_else(|| {
                            let ty = self.insts[i as usize].ty;
                            self.undef(ty)
                        });
                        map.insert(i, v);
                    }
                }
                for &i in &body {
                    let mut inst = self.insts[i as usize].clone();
                    for_each_operand_mut(&mut inst.op, |v| {
                        if let V::Inst(j) = *v
                            && let Some(&n) = map.get(&j)
                        {
                            *v = n;
                        }
                    });
                    let n = self.push(copy, inst.ty, inst.op);
                    if let Some(&tag) = self.tags.get(&i) {
                        if let V::Inst(k) = n {
                            self.tags.insert(k, tag);
                        }
                    }
                    map.insert(i, n);
                }
                self.terminate(copy, Term::Br(ret));
                let term = match self.blocks[p].term {
                    Some(Term::Br(_)) => Term::Br(copy),
                    Some(Term::CondBr(c, t, f)) => {
                        Term::CondBr(c, if t == b { copy } else { t }, if f == b { copy } else { f })
                    }
                    _ => unreachable!(),
                };
                self.blocks[p].term = Some(term);
            }
        }
    }

    /// Whether instruction `i` does the same in a copy of its block on
    /// each way in: not an operation across lanes (wave, quad, implicit
    /// derivatives: each copy would see only its own path's lanes), nor a
    /// barrier.
    fn copyable(&self, i: u32) -> bool {
        let Op::Call(f, _) = &self.insts[i as usize].op else {
            return true;
        };
        let f = &self.funcs[*f as usize];
        let name = f.name.strip_prefix("dx.op.").unwrap_or(&f.name);
        f.attr != Some(super::Attr::NoDuplicate)
            && super::opt::effect(f) != super::opt::Effect::Lanes
            && !["deriv", "sample", "calculateLOD", "barrier", "Barrier"]
                .iter()
                .any(|x| name.contains(x))
    }

    /// Whether a phi of `ty` is something the converters take: a number.
    /// A pointer is not (DXIL has none of it; dxil-spirv cannot tell its
    /// storage), nor a resource handle.
    fn joinable(&self, ty: Ty) -> bool {
        matches!(
            self.types[ty.0 as usize],
            super::TypeDef::Int(_) | super::TypeDef::Float | super::TypeDef::Double
        )
    }

    /// What value `i` is made of when it cannot go through a phi and is
    /// made again after the loop instead: an address, or a call without
    /// effect (a resource handle); `None` for anything else.
    fn remade_from(&self, i: u32) -> Option<Vec<V>> {
        match &self.insts[i as usize].op {
            Op::Gep(_, p, idx) => Some(std::iter::once(*p).chain(idx.iter().copied()).collect()),
            Op::Call(f, args) if self.funcs[*f as usize].attr == Some(super::Attr::ReadNone) => {
                Some(args.clone())
            }
            _ => None,
        }
    }

    /// Whether the loop's value `i` can be read after it through its exit
    /// block: a number, or a value made again from such values.
    fn passes_exit(&self, i: u32, block_of: &HashMap<u32, usize>) -> bool {
        if self.joinable(self.insts[i as usize].ty) {
            return true;
        }
        let Some(from) = self.remade_from(i) else {
            return false;
        };
        from.iter().all(|v| match v {
            V::Inst(j) if block_of.contains_key(j) => self.passes_exit(*j, block_of),
            _ => true,
        })
    }

    /// The loop's value `i` as read after it, in the exit block: a phi of
    /// it on the edges whose blocks it dominates, or the instruction made
    /// again there (see [`Module::passes_exit`]).
    fn exit_value(
        &mut self,
        i: u32,
        exit: usize,
        leaving: &[usize],
        g: &Graph,
        block_of: &HashMap<u32, usize>,
        renamed: &mut HashMap<u32, V>,
    ) -> V {
        if let Some(&v) = renamed.get(&i) {
            return v;
        }
        let ty = self.insts[i as usize].ty;
        let v = if self.joinable(ty) {
            let def = block_of[&i];
            let undef = self.undef(ty);
            let inc: Vec<(V, usize)> = leaving
                .iter()
                .map(|&b| (if g.dominates(def, b) { V::Inst(i) } else { undef }, b))
                .collect();
            let phi = self.phi(exit, ty);
            self.set_phi(phi, inc);
            phi
        } else {
            let mut op = self.insts[i as usize].op.clone();
            let mut from: Vec<V> = Vec::new();
            super::for_each_operand(&op, |v| from.push(v));
            let mapped: Vec<V> = from
                .into_iter()
                .map(|v| match v {
                    V::Inst(j) if block_of.contains_key(&j) => {
                        self.exit_value(j, exit, leaving, g, block_of, renamed)
                    }
                    _ => v,
                })
                .collect();
            let mut k = 0;
            for_each_operand_mut(&mut op, |v| {
                *v = mapped[k];
                k += 1;
            });
            self.push(exit, ty, op)
        };
        renamed.insert(i, v);
        v
    }

    /// Gives the loop headed by `header` one block that branches back to
    /// it (DXC's continue block), unless it has one; true if the graph
    /// changed. The header's phis take what the back edges gave from a phi
    /// of that block.
    fn give_loop_one_latch(&mut self, g: &Graph, body: &[bool], header: usize) -> bool {
        let back: Vec<usize> = g.preds[header].iter().copied().filter(|&p| body[p]).collect();
        if back.len() < 2 {
            return false;
        }
        let latch = self.new_block();
        for p in self.blocks[header].phis.clone() {
            let Op::Phi(inc) = self.insts[p as usize].op.clone() else {
                continue;
            };
            let ty = self.insts[p as usize].ty;
            let undef = self.undef(ty);
            let ins: Vec<(V, usize)> = back
                .iter()
                .map(|&b| (inc.iter().find(|&&(_, q)| q == b).map_or(undef, |x| x.0), b))
                .collect();
            let l = self.phi(latch, ty);
            self.set_phi(l, ins);
            let mut kept: Vec<(V, usize)> = inc.into_iter().filter(|&(_, b)| !back.contains(&b)).collect();
            kept.push((l, latch));
            self.set_phi(V::Inst(p), kept);
        }
        for &b in &back {
            let term = match self.blocks[b].term {
                Some(Term::Br(_)) => Term::Br(latch),
                Some(Term::CondBr(c, t, f)) => match (t == header, f == header) {
                    (true, true) => Term::Br(latch),
                    (true, false) => Term::CondBr(c, latch, f),
                    (false, true) => Term::CondBr(c, t, latch),
                    (false, false) => unreachable!(),
                },
                _ => unreachable!(),
            };
            self.blocks[b].term = Some(term);
        }
        self.terminate(latch, Term::Br(header));
        true
    }

    /// Gives the loop whose blocks are `region` one exit block of its own,
    /// unless it has that already or none; true if the graph changed.
    fn give_loop_one_exit(
        &mut self,
        g: &Graph,
        region: &[bool],
        exit_of: &mut Vec<Option<usize>>,
        header: usize,
    ) -> bool {
        // The blocks in the loop that leave it, in layout order, and where to.
        let mut leaving: Vec<usize> = Vec::new();
        let mut targets: Vec<usize> = Vec::new();
        for &b in &g.order {
            if !region[b] {
                continue;
            }
            let mut leaves = false;
            for &s in successors(self.blocks[b].term).iter() {
                if !region[s] {
                    leaves = true;
                    if !targets.contains(&s) {
                        targets.push(s);
                    }
                }
            }
            if leaves {
                leaving.push(b);
            }
        }
        match &targets[..] {
            [] => return false,
            [t] if g.preds[*t].iter().all(|&p| region[p]) => return false,
            _ => {}
        }
        // Values made in the loop and read outside it (other than by the
        // targets' phis on edges from the loop) are read from a phi of the
        // exit block instead.
        let mut block_of: HashMap<u32, usize> = HashMap::new();
        for &b in &g.order {
            if region[b] {
                let blk = &self.blocks[b];
                for &i in blk.phis.iter().chain(&blk.body) {
                    block_of.insert(i, b);
                }
            }
        }
        let mut outside_uses: Vec<u32> = Vec::new();
        let mut noted: std::collections::HashSet<u32> = std::collections::HashSet::new();
        {
            let mut note = |v: V, outside_uses: &mut Vec<u32>| {
                if let V::Inst(i) = v
                    && block_of.contains_key(&i)
                    && noted.insert(i)
                {
                    outside_uses.push(i);
                }
            };
            for &b in &g.order {
                if region[b] {
                    continue;
                }
                let blk = &self.blocks[b];
                for &i in &blk.phis {
                    if let Op::Phi(inc) = &self.insts[i as usize].op {
                        for &(v, p) in inc {
                            if !region[p] {
                                note(v, &mut outside_uses);
                            }
                        }
                    }
                }
                for &i in &blk.body {
                    super::for_each_operand(&self.insts[i as usize].op, |v| {
                        note(v, &mut outside_uses)
                    });
                }
                if let Some(Term::CondBr(c, ..)) = blk.term {
                    note(c, &mut outside_uses);
                }
            }
        }
        // A pointer cannot go through a phi (DXIL has none; dxil-spirv cannot
        // tell its storage): an address made in the loop is made again in
        // the exit block from what it was made of. A loop whose other
        // pointers are read after it is left as it is.
        if !outside_uses.iter().all(|&i| self.passes_exit(i, &block_of)) {
            return false;
        }
        let i1 = self.int(1);
        let i32t = self.int(32);
        let k = targets.len();
        let exit = self.new_block();
        // The chain: `chain[j]` branches to `targets[j]` (the last block
        // branches to the last two).
        let mut chain = vec![exit];
        for _ in 1..k.saturating_sub(1) {
            chain.push(self.new_block());
        }
        let from = |j: usize| chain[j.min(k.saturating_sub(2))];
        exit_of.resize(self.blocks.len(), None);
        for &c in &chain {
            exit_of[c] = Some(header);
        }

        // Per leaving block: the value its edges give a phi of the exit
        // block, from what each target's edge gives (a select when its two
        // edges leave to different targets).
        let incoming = |m: &mut Module, per_target: &dyn Fn(&mut Module, usize, usize) -> V| {
            let mut out = Vec::with_capacity(leaving.len());
            for &b in &leaving {
                let v = match m.blocks[b].term {
                    Some(Term::CondBr(c, t, f)) if !region[t] && !region[f] && t != f => {
                        let (vt, vf) = (per_target(m, b, t), per_target(m, b, f));
                        if vt == vf { vt } else { m.select(b, c, vt, vf) }
                    }
                    Some(Term::CondBr(_, t, f)) => {
                        let s = if region[t] { f } else { t };
                        per_target(m, b, s)
                    }
                    Some(Term::Br(s)) => per_target(m, b, s),
                    _ => unreachable!("a leaving block without a branch"),
                };
                out.push((v, b));
            }
            out
        };

        let mut renamed: HashMap<u32, V> = HashMap::new();
        for &i in &outside_uses {
            self.exit_value(i, exit, &leaving, g, &block_of, &mut renamed);
        }
        if !renamed.is_empty() {
            let rename = |v: &mut V| {
                if let V::Inst(i) = *v
                    && let Some(&n) = renamed.get(&i)
                {
                    *v = n;
                }
            };
            for &b in &g.order {
                if region[b] {
                    continue;
                }
                let blk = self.blocks[b].clone();
                for &i in &blk.phis {
                    if let Op::Phi(inc) = &mut self.insts[i as usize].op {
                        for (v, p) in inc.iter_mut() {
                            if !region[*p] {
                                rename(v);
                            }
                        }
                    }
                }
                for &i in &blk.body {
                    for_each_operand_mut(&mut self.insts[i as usize].op, rename);
                }
                if let Some(Term::CondBr(c, t, f)) = blk.term {
                    let mut c = c;
                    rename(&mut c);
                    self.blocks[b].term = Some(Term::CondBr(c, t, f));
                }
            }
        }

        // The targets' phis: what came from the loop comes from the exit
        // block's phi of it, through the chain.
        for (j, &t) in targets.iter().enumerate() {
            let phis = self.blocks[t].phis.clone();
            for p in phis {
                let Op::Phi(inc) = self.insts[p as usize].op.clone() else {
                    continue;
                };
                if !inc.iter().any(|&(_, b)| region[b]) {
                    continue;
                }
                let ty: Ty = self.insts[p as usize].ty;
                let undef = self.undef(ty);
                let ins = incoming(self, &|_, b, s| {
                    if s == t {
                        inc.iter().find(|&&(_, q)| q == b).map_or(undef, |x| x.0)
                    } else {
                        undef
                    }
                });
                let e = self.phi(exit, ty);
                self.set_phi(e, ins);
                let mut kept: Vec<(V, usize)> =
                    inc.into_iter().filter(|&(_, b)| !region[b]).collect();
                kept.push((e, from(j)));
                self.set_phi(V::Inst(p), kept);
            }
        }

        // Which target, and the chain.
        if k > 1 {
            let consts: Vec<V> = if k == 2 {
                vec![self.bool(true), self.bool(false)]
            } else {
                (0..k as u32).map(|j| self.i32(j)).collect()
            };
            let ins = incoming(self, &|_, _, s| {
                consts[targets.iter().position(|&x| x == s).expect("a target")]
            });
            let sel = self.phi(exit, if k == 2 { i1 } else { i32t });
            self.set_phi(sel, ins);
            if k == 2 {
                self.terminate(exit, Term::CondBr(sel, targets[0], targets[1]));
            } else {
                for j in 0..k - 1 {
                    let c = self.cmp(chain[j], pred::EQ, sel, consts[j]);
                    let other = if j + 2 == k { targets[k - 1] } else { chain[j + 1] };
                    self.terminate(chain[j], Term::CondBr(c, targets[j], other));
                }
            }
        } else {
            self.terminate(exit, Term::Br(targets[0]));
        }

        // The loop's edges out go to the exit block.
        for &b in &leaving {
            let term = match self.blocks[b].term {
                Some(Term::Br(_)) => Term::Br(exit),
                Some(Term::CondBr(c, t, f)) => match (region[t], region[f]) {
                    (false, false) => Term::Br(exit),
                    (true, false) => Term::CondBr(c, t, exit),
                    (false, true) => Term::CondBr(c, exit, f),
                    (true, true) => unreachable!(),
                },
                _ => unreachable!(),
            };
            self.blocks[b].term = Some(term);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::opt::tests::{kit, run};
    use super::super::{Attr, Bin, Module, Op, Term, V, pred, successors};
    use super::Graph;
    use std::collections::HashMap;

    /// How a generated block ends, before its values exist.
    #[derive(Clone, Copy)]
    enum End {
        Open,
        Br(usize),
        /// A branch on a condition from the registers.
        If(usize, usize, u32),
        /// A loop's back edge (to its header) while its counter allows, else on.
        Back(usize, usize),
        /// A `continue` (to the header) where a condition and the counter allow.
        Continue(usize, usize, u32),
        Ret,
    }

    impl End {
        fn succs(self) -> Vec<usize> {
            match self {
                End::Br(t) => vec![t],
                End::If(t, f, _) | End::Continue(t, f, _) | End::Back(t, f) => vec![t, f],
                _ => vec![],
            }
        }
    }

    /// A random reducible graph with the console code's jumps: breaks out
    /// of any number of loops, continues, and jumps to shared blocks that
    /// return, from anywhere. Every loop runs at most three times.
    struct Shape {
        ends: Vec<End>,
        /// Statements per block.
        work: Vec<u32>,
        rng: u64,
        tails: Vec<usize>,
    }

    impl Shape {
        fn next(&mut self, n: u64) -> u64 {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            self.rng % n
        }

        fn block(&mut self) -> usize {
            self.ends.push(End::Open);
            self.work.push(0);
            self.ends.len() - 1
        }

        /// Code from `cur` on; the block it ends in. `loops`: (header,
        /// exit) of the loops around, innermost last.
        fn region(&mut self, mut cur: usize, depth: u32, loops: &mut Vec<(usize, usize)>) -> usize {
            for _ in 0..1 + self.next(3) {
                match self.next(8) {
                    0 | 1 => self.work[cur] += 1 + self.next(3) as u32,
                    2 | 3 if depth > 0 => {
                        let (t, e, j) = (self.block(), self.block(), self.block());
                        let salt = self.next(1 << 16) as u32;
                        self.ends[cur] = End::If(t, e, salt);
                        let t_end = self.region(t, depth - 1, loops);
                        self.ends[t_end] = End::Br(j);
                        let e_end = self.region(e, depth - 1, loops);
                        self.ends[e_end] = End::Br(j);
                        cur = j;
                    }
                    4 if depth > 0 => {
                        let (h, exit) = (self.block(), self.block());
                        self.ends[cur] = End::Br(h);
                        loops.push((h, exit));
                        let end = self.region(h, depth - 1, loops);
                        loops.pop();
                        self.ends[end] = End::Back(h, exit);
                        cur = exit;
                    }
                    _ => {
                        // A jump out: a break of some loop around, a
                        // continue of the innermost, or to a shared tail.
                        let next = self.block();
                        let salt = self.next(1 << 16) as u32;
                        let pick = self.next(loops.len() as u64 + 2) as usize;
                        self.ends[cur] = if pick < loops.len() {
                            End::If(loops[pick].1, next, salt)
                        } else if pick == loops.len() && !loops.is_empty() {
                            End::Continue(loops[loops.len() - 1].0, next, salt)
                        } else {
                            let which = self.next(self.tails.len() as u64) as usize;
                            let t = self.tails[which];
                            End::If(t, next, salt)
                        };
                        cur = next;
                    }
                }
            }
            cur
        }
    }

    const REGS: usize = 3;

    /// A program of that shape over a few registers: a phi of each where
    /// blocks join, stores along the way, every register stored at a return.
    fn random_program(seed: u64) -> Module {
        let mut s = Shape {
            ends: Vec::new(),
            work: Vec::new(),
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            tails: Vec::new(),
        };
        let entry = s.block();
        let tails: Vec<usize> = (0..2).map(|_| s.block()).collect();
        for &t in &tails {
            s.work[t] = 1;
            s.ends[t] = End::Ret;
        }
        s.tails = tails;
        let end = s.region(entry, 4, &mut Vec::new());
        s.ends[end] = End::Ret;

        let mut k = kit();
        let writes = k.writes;
        let m = &mut k.m;
        let i32t = m.int(32);
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        let input = m.function("dx.op.loadInput.i32", fty, Some(Attr::ReadNone), false);
        assert_eq!(k.b, 0);
        let n = s.ends.len();
        for _ in 1..n {
            m.new_block();
        }
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, i) = *top;
            top.1 += 1;
            match s.ends[b].succs().get(i) {
                Some(&x) if !seen[x] => {
                    seen[x] = true;
                    stack.push((x, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let mut preds = vec![Vec::new(); n];
        for &b in &post {
            for x in s.ends[b].succs() {
                if !preds[x].contains(&b) {
                    preds[x].push(b);
                }
            }
        }
        let back_to = |e: End| match e {
            End::Back(h, _) | End::Continue(h, ..) => Some(h),
            _ => None,
        };
        let headers: Vec<usize> = (0..n).filter(|&b| s.ends.iter().any(|&e| back_to(e) == Some(b))).collect();
        let mut out: Vec<Vec<V>> = vec![Vec::new(); n];
        let mut phis: HashMap<usize, Vec<V>> = HashMap::new();
        let mut counter: HashMap<usize, V> = HashMap::new();
        let mut counted: HashMap<usize, V> = HashMap::new();
        let (zero, one, three, seven) = (m.i32(0), m.i32(1), m.i32(3), m.i32(7));
        let mut salt_n = 0u32;
        for &b in &post {
            let mut regs: Vec<V> = if b == 0 {
                (0..REGS as u32)
                    .map(|r| {
                        let (x, y) = (m.i32(r), m.i32(r * 5 + 1));
                        m.call(0, input, vec![x, y])
                    })
                    .collect()
            } else if preds[b].len() > 1 || headers.contains(&b) {
                let p: Vec<V> = (0..REGS).map(|_| m.phi(b, i32t)).collect();
                phis.insert(b, p.clone());
                p
            } else {
                out[preds[b][0]].clone()
            };
            if headers.contains(&b) {
                counter.insert(b, m.phi(b, i32t));
            }
            for _ in 0..s.work[b] {
                salt_n += 1;
                let r = salt_n as usize % REGS;
                let other = regs[(r + 1) % REGS];
                let c = m.i32(salt_n.wrapping_mul(0x2545_F491));
                let x = m.bin(b, if salt_n % 3 == 0 { Bin::Xor } else { Bin::Add }, regs[r], c);
                regs[r] = m.bin(b, Bin::Mul, x, other);
                if salt_n % 2 == 0 {
                    let at = m.i32(salt_n % 16);
                    m.call(b, writes, vec![at, regs[r]]);
                }
            }
            let test = |m: &mut Module, regs: &[V], salt: u32| {
                let x = m.bin(b, Bin::And, regs[salt as usize % REGS], seven);
                let k = m.i32(salt % 8);
                m.cmp(b, pred::IULT, x, k)
            };
            let term = match s.ends[b] {
                End::Ret => {
                    for (r, &v) in regs.iter().enumerate() {
                        let at = m.i32(r as u32);
                        m.call(b, writes, vec![at, v]);
                    }
                    Term::Ret
                }
                End::Br(t) => Term::Br(t),
                End::If(t, f, salt) => {
                    let c = test(m, &regs, salt);
                    Term::CondBr(c, t, f)
                }
                End::Back(h, exit) => {
                    let inc = m.bin(b, Bin::Add, counter[&h], one);
                    counted.insert(b, inc);
                    let c = m.cmp(b, pred::IULT, inc, three);
                    Term::CondBr(c, h, exit)
                }
                End::Continue(h, next, salt) => {
                    let inc = m.bin(b, Bin::Add, counter[&h], one);
                    counted.insert(b, inc);
                    let room = m.cmp(b, pred::IULT, inc, three);
                    let c = test(m, &regs, salt);
                    let c = m.bin(b, Bin::And, c, room);
                    Term::CondBr(c, h, next)
                }
                End::Open => unreachable!(),
            };
            m.terminate(b, term);
            out[b] = regs;
        }
        for (&b, p) in &phis {
            for (r, &phi) in p.iter().enumerate() {
                let inc = preds[b].iter().map(|&q| (out[q][r], q)).collect();
                m.set_phi(phi, inc);
            }
        }
        for (&h, &c) in &counter {
            let inc = preds[h]
                .iter()
                .map(|&q| {
                    let v = if back_to(s.ends[q]) == Some(h) { counted[&q] } else { zero };
                    (v, q)
                })
                .collect();
            m.set_phi(c, inc);
        }
        // Blocks no path reaches still need an end.
        for b in 0..n {
            if m.blocks[b].term.is_none() {
                m.terminate(b, Term::Ret);
            }
        }
        k.m
    }

    /// Every value is read where its definition dominates the reader, and
    /// every phi has a value for each way in.
    fn check_dominance(m: &Module) {
        let g = Graph::new(m);
        let mut home = HashMap::new();
        for &b in &g.order {
            for &i in m.blocks[b].phis.iter().chain(&m.blocks[b].body) {
                home.insert(i, b);
            }
        }
        for &b in &g.order {
            for &i in &m.blocks[b].phis {
                let Op::Phi(inc) = &m.insts[i as usize].op else {
                    unreachable!()
                };
                for &(v, p) in inc {
                    if let V::Inst(j) = v
                        && g.reached(p)
                    {
                        assert!(g.dominates(home[&j], p), "phi {i} in {b} reads {j} via {p}");
                    }
                }
                for &p in &g.preds[b] {
                    assert!(inc.iter().any(|x| x.1 == p), "phi {i} in {b} lacks {p}");
                }
            }
            let check = |v: V| {
                if let V::Inst(j) = v {
                    assert!(g.dominates(home[&j], b), "{j} read in {b}");
                }
            };
            for &i in &m.blocks[b].body {
                super::super::for_each_operand(&m.insts[i as usize].op, check);
            }
            if let Some(Term::CondBr(c, ..)) = m.blocks[b].term {
                check(c);
            }
        }
    }

    #[test]
    fn programs_with_jumps_out_of_loops_do_the_same_with_single_exits() {
        let mut changed = 0;
        for seed in 1..=500u64 {
            let plain = random_program(seed);
            check_dominance(&plain);
            let inputs: Vec<Vec<u32>> = (0..8u32)
                .map(|n| {
                    (0..7u32)
                        .map(|i| (seed as u32 ^ n).wrapping_mul(0x9E37_79B9).rotate_left(i * 7) ^ i)
                        .collect()
                })
                .collect();
            let expected: Vec<_> = inputs.iter().map(|i| run(&plain, i)).collect();
            for want in &expected {
                assert_eq!(want.last().map(|x| x.0.as_str()), Some("ret"), "seed {seed}");
            }
            let mut m = random_program(seed);
            m.single_exits();
            changed += usize::from(m.blocks.len() != plain.blocks.len());
            check_dominance(&m);
            for (i, want) in inputs.iter().zip(&expected) {
                assert_eq!(&run(&m, i), want, "seed {seed}");
            }
            // One return, and one back edge to each loop's header.
            let g = Graph::new(&m);
            let rets = g.order.iter().filter(|&&b| matches!(m.blocks[b].term, Some(Term::Ret))).count();
            assert_eq!(rets, 1, "seed {seed}");
            for (h, body) in g.loops(&m) {
                let back = g.preds[h].iter().filter(|&&p| body[p]).count();
                assert_eq!(back, 1, "seed {seed}: loop at {h}");
            }
            m.write();
        }
        assert!(changed > 250, "{changed} programs changed");
    }

    #[test]
    fn a_loop_left_for_a_block_reached_from_before_it_gets_its_own_exit() {
        // The entry jumps out early to `tail`, or into the loop; the loop
        // breaks to `tail` too, and otherwise ends in `discard`.
        let mut k = kit();
        let writes = k.writes;
        let m = &mut k.m;
        let i32t = m.int(32);
        let entry = 0;
        let (head, latch, tail, discard) = (m.new_block(), m.new_block(), m.new_block(), m.new_block());
        let (zero, one, three, five) = (m.i32(0), m.i32(1), m.i32(3), m.i32(5));
        let early = m.cmp(entry, pred::EQ, zero, one);
        m.terminate(entry, Term::CondBr(early, tail, head));
        let n = m.phi(head, i32t);
        let next = m.bin(head, Bin::Add, n, one);
        let brk = m.cmp(head, pred::EQ, next, five);
        m.terminate(head, Term::CondBr(brk, tail, latch));
        let more = m.cmp(latch, pred::IULT, next, three);
        m.terminate(latch, Term::CondBr(more, head, discard));
        m.set_phi(n, vec![(zero, entry), (next, latch)]);
        let t = m.phi(tail, i32t);
        m.set_phi(t, vec![(zero, entry), (next, head)]);
        m.call(tail, writes, vec![zero, t]);
        m.terminate(tail, Term::Ret);
        m.call(discard, writes, vec![one, next]);
        m.terminate(discard, Term::Ret);
        let want = run(&k.m, &[]);
        k.m.single_exits();
        check_dominance(&k.m);
        assert_eq!(run(&k.m, &[]), want);
        let g = Graph::new(&k.m);
        let loops = g.loops(&k.m);
        assert_eq!(loops.len(), 1);
        let body = &loops[0].1;
        let mut out: Vec<usize> = g
            .order
            .iter()
            .filter(|&&b| body[b])
            .flat_map(|&b| successors(k.m.blocks[b].term).to_vec())
            .filter(|&s| !body[s])
            .collect();
        out.dedup();
        assert_eq!(out.len(), 1, "one block leaves the loop");
        assert!(g.preds[out[0]].iter().all(|&p| body[p]), "and only the loop reaches it");
        k.m.write();
    }
}
