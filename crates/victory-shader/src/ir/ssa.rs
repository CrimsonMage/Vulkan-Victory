//! SSA construction helpers of [`super::rename`]: reachability, liveness
//! and the cleanup of trivial phis.

use super::*;

/// Blocks the entry never reaches (code after an unconditional jump, a
/// jump-table tail) define nothing: they are emptied and dropped from their
/// successors' predecessors, so no read or phi argument is left without a
/// value (it was `NONE`, which panicked in the phi cleanup).
pub(super) fn drop_unreachable(blocks: &mut [Block], order: &[usize]) {
    let mut reachable = vec![false; blocks.len()];
    for &b in order {
        reachable[b] = true;
    }
    for (b, block) in blocks.iter_mut().enumerate() {
        block.preds.retain(|&p| reachable[p]);
        if !reachable[b] {
            block.insts.clear();
            block.phis.clear();
        }
    }
}

/// Registers live into and out of each block.
pub(super) fn liveness(blocks: &[Block], order: &[usize]) -> (Vec<RegSet>, Vec<RegSet>) {
    let empty = [0; REGS.div_ceil(64)];
    let mut uses = vec![empty; blocks.len()];
    let mut defs = vec![empty; blocks.len()];
    for (b, block) in blocks.iter().enumerate() {
        for inst in &block.insts {
            for &(reg, _) in &inst.reads {
                if !contains(&defs[b], reg) {
                    insert(&mut uses[b], reg);
                }
            }
            for &(reg, _) in &inst.writes {
                insert(&mut defs[b], reg);
            }
        }
    }
    let mut live_in = vec![empty; blocks.len()];
    let mut live_out = vec![empty; blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let mut out = empty;
            for s in blocks[b].exit.successors() {
                for (o, i) in out.iter_mut().zip(&live_in[s]) {
                    *o |= i;
                }
            }
            let mut inn = empty;
            for w in 0..inn.len() {
                inn[w] = uses[b][w] | out[w] & !defs[b][w];
            }
            if inn != live_in[b] || out != live_out[b] {
                (live_in[b], live_out[b]) = (inn, out);
                changed = true;
            }
        }
    }
    (live_in, live_out)
}

/// Phis are placed for every live register at every join; those whose
/// arguments are all one value (or the phi itself) are replaced by it.
pub(super) fn remove_trivial_phis(blocks: &mut [Block], values: u32) {
    let mut replaced: Vec<Value> = (0..values).map(Value).collect();
    let find = |replaced: &[Value], mut v: Value| {
        while replaced[v.0 as usize] != v {
            v = replaced[v.0 as usize];
        }
        v
    };
    let mut changed = true;
    while changed {
        changed = false;
        for block in blocks.iter_mut() {
            block.phis.retain(|phi| {
                let mut only = None;
                for &a in &phi.args {
                    let a = find(&replaced, a);
                    if a == phi.value || Some(a) == only {
                        continue;
                    }
                    if only.is_some() {
                        return true;
                    }
                    only = Some(a);
                }
                let Some(v) = only else { return true };
                replaced[phi.value.0 as usize] = v;
                changed = true;
                false
            });
        }
    }
    for block in blocks {
        for phi in &mut block.phis {
            for a in &mut phi.args {
                *a = find(&replaced, *a);
            }
        }
        for inst in &mut block.insts {
            for (_, v) in &mut inst.reads {
                *v = find(&replaced, *v);
            }
        }
    }
}
