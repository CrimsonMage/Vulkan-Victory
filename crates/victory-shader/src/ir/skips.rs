//! Skips of a region no lane is left for (`s_cbranch_execz`) that every
//! lane may walk instead: [`without_skips`].

use std::collections::{HashMap, HashSet};

use super::{Exit, Inst, Program, Reg, Value, link, rename, reverse_postorder, split_irreducible};

/// The program without the branches that only skip a region when no lane
/// is left for it: an `s_cbranch_execz` whose target is
/// where the skipped code ends up anyway (`if (c) { ... }` with nothing
/// else leaving the region). Every lane then walks the region with EXEC
/// clear for those that would have skipped it, and LDS accesses and
/// exchanges inside it are in code every lane runs.
/// The console takes such a branch only when no lane of the wave is left,
/// and then its scalar code does not run: a branch is kept if the code
/// after it could see that (a scalar register or SCC that the region
/// changes and the join merges with the value from before; EXEC aside,
/// which is empty either way), and when the region holds a loop (and no
/// group barrier): a loop's way out may need a lane (an `s_cbranch_vccz`
/// back edge, a count taken by `v_readfirstlane`), so walked with EXEC
/// clear it may never end. `None` if nothing changes.
///
/// `deep` looks closer at what the code after the branch can see. A scalar the region moves
/// away and back (the compiler parks scalars in VGPR lanes around a branch and takes them
/// back before the join, `unspill`) is the value it copies; a join nothing reads shows no
/// one the difference (a restore of EXEC that the next one overwrites reads nothing); and
/// the join may have other ways in (an inner branch that ends together with an outer one).
/// That leaves out most of the skips of the title's large lighting programs, whose bodies
/// are then code every lane runs.
pub(crate) fn without_skips(p: &Program, deep: bool) -> Option<Program> {
    let facts = Facts::new(p, deep);
    let mut blocks = p.blocks.clone();
    let mut any = false;
    for b in 0..p.blocks.len() {
        let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
            continue;
        };
        if !p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz") {
            continue;
        }
        if (ends(p, taken) || facts.skipped(b, taken, not_taken)) && !facts.has_loop(not_taken, taken) {
            blocks[b].insts.pop();
            blocks[b].exit = Exit::Jump(not_taken);
            any = true;
        }
    }
    if !any {
        return None;
    }
    for block in &mut blocks {
        block.phis.clear();
    }
    link(&mut blocks);
    split_irreducible(&mut blocks).ok()?;
    Some(rename(blocks, p.wave64, p.vgprs))
}

/// A block that only ends the program.
fn ends(p: &Program, b: usize) -> bool {
    p.blocks[b].exit == Exit::End && p.blocks[b].insts.iter().all(|i| matches!(i.fields, crate::gfx10::Fields::Sopp { .. }))
}

/// What the program shows of its values, for deciding a skip.
struct Facts<'a> {
    p: &'a Program,
    deep: bool,
    /// Scalars moved from register to register (a scalar the compiler parked in a VGPR lane
    /// comes back by such a move, `unspill`) are the value they copy.
    copy: HashMap<Value, Value>,
    /// Values something reads, directly or through joins.
    live: HashSet<Value>,
    /// Each block's place in reverse postorder (edges to an earlier one are back edges).
    rank: Vec<usize>,
}

impl<'a> Facts<'a> {
    fn new(p: &'a Program, deep: bool) -> Self {
        let mut copy = HashMap::new();
        for inst in p.blocks.iter().flat_map(|b| &b.insts) {
            if matches!(inst.name(), "s_mov_b32" | "s_mov_b64")
                && inst.reads.len() == inst.writes.len()
                && inst.reads.iter().chain(&inst.writes).all(|r| matches!(r.0, Reg::S(_)))
            {
                copy.extend(inst.writes.iter().zip(&inst.reads).map(|(w, r)| (w.1, r.1)));
            }
        }
        let mut rank = vec![usize::MAX; p.blocks.len()];
        for (i, &b) in reverse_postorder(&p.blocks).iter().enumerate() {
            rank[b] = i;
        }
        Facts { p, deep, copy, live: live_values(p), rank }
    }

    fn root(&self, mut v: Value) -> Value {
        while self.deep
            && let Some(&c) = self.copy.get(&v)
        {
            v = c;
        }
        v
    }

    /// Whether the branch of `b` to `target` (else to `first`) is a pure skip.
    fn skipped(&self, b: usize, target: usize, first: usize) -> bool {
        let Some(inside) = self.region(b, target, first) else { return false };
        let join = &self.p.blocks[target];
        // Other ways into the join than the branch and the region (an inner branch that ends
        // together with an outer one) change nothing in that: the lanes that would have
        // skipped arrive from the region instead, with the same scalars.
        if !self.deep && join.preds.iter().any(|&q| q != b && !inside[q]) {
            return false;
        }
        self.join_sees_nothing(b, target, &inside)
    }

    /// The region from `first` to `target`, if it is entered only through the branch of
    /// `b` and does not end or come back to it.
    fn region(&self, b: usize, target: usize, first: usize) -> Option<Vec<bool>> {
        let blocks = &self.p.blocks;
        let mut inside = vec![false; blocks.len()];
        let mut stack = vec![first];
        while let Some(x) = stack.pop() {
            if x == target || inside[x] {
                continue;
            }
            if x == b {
                return None; // a loop back to the branch
            }
            inside[x] = true;
            match blocks[x].exit {
                Exit::End | Exit::Indirect => return None,
                exit => stack.extend(exit.successors()),
            }
        }
        let entered_elsewhere = |x: usize| blocks[x].preds.iter().any(|&q| q != b && !inside[q]);
        if (0..blocks.len()).any(|x| inside[x] && entered_elsewhere(x)) {
            return None;
        }
        Some(inside)
    }

    /// Whether the join sees no scalar the region changed (a join nothing reads shows no
    /// one the difference).
    fn join_sees_nothing(&self, b: usize, target: usize, inside: &[bool]) -> bool {
        let join = &self.p.blocks[target];
        join.phis.iter().all(|phi| {
            let scalar = match phi.reg {
                Reg::S(c) => c != 126 && c != 127,
                Reg::Scc => true,
                Reg::V(_) => false,
            };
            if !scalar || self.deep && !self.live.contains(&phi.value) {
                return true;
            }
            let from = |want: &dyn Fn(usize) -> bool| -> Vec<Value> {
                join.preds.iter().zip(&phi.args).filter(|(q, _)| want(**q)).map(|(_, a)| self.root(*a)).collect()
            };
            let before = from(&|q| q == b);
            from(&|q| inside[q]).iter().all(|a| before.contains(a))
        })
    }

    /// Whether the blocks from `first` up to `target` hold a loop and no
    /// group barrier (a wave that leaves out a barrier the others of its
    /// group meet is worse than a loop the guard ends).
    fn has_loop(&self, first: usize, target: usize) -> bool {
        let mut seen = vec![false; self.p.blocks.len()];
        let mut stack = vec![first];
        let mut looped = false;
        while let Some(x) = stack.pop() {
            if x == target || std::mem::replace(&mut seen[x], true) {
                continue;
            }
            if self.p.blocks[x].insts.iter().any(|i| i.name() == "s_barrier") {
                return false;
            }
            for y in self.p.blocks[x].exit.successors() {
                looped |= y != target && self.rank[y] <= self.rank[x];
                stack.push(y);
            }
        }
        looped
    }
}

/// A scalar operation none of whose results is read reads nothing either (the compiler
/// restores EXEC from a saved mask and overwrites it at once where two branches end
/// together).
fn idle(inst: &Inst, live: &HashSet<Value>) -> bool {
    let name = inst.name();
    let flow = ["s_setpc", "s_swappc", "s_call", "s_rfe", "s_movrel", "s_subvector"];
    matches!(inst.fields, crate::gfx10::Fields::Sop { .. })
        && !flow.iter().any(|f| name.starts_with(f))
        && inst.writes.iter().all(|w| !live.contains(&w.1))
}

/// Values something reads, directly or through joins.
fn live_values(p: &Program) -> HashSet<Value> {
    let mut live = HashSet::new();
    while p.blocks.iter().fold(false, |changed, block| mark_live(block, &mut live) | changed) {}
    live
}

/// One pass of [`live_values`] over a block; whether it found more.
fn mark_live(block: &super::Block, live: &mut HashSet<Value>) -> bool {
    let mut changed = false;
    for inst in &block.insts {
        if !idle(inst, live) {
            for r in &inst.reads {
                changed |= live.insert(r.1);
            }
        }
    }
    for phi in &block.phis {
        if live.contains(&phi.value) {
            for &a in &phi.args {
                changed |= live.insert(a);
            }
        }
    }
    changed
}
