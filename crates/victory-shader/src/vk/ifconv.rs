//! If-conversion: a branch whose arms are small blocks of pure arithmetic
//! that meet again becomes straight code: both arms run and the join's
//! phis pick their values by the condition (`OpSelect`). The console's
//! scalar code branches around a few instructions often; each branch costs
//! the host compiler a selection construct.

use super::spirv::op;
use super::tir::{Func, Helper, Item, Node, Op, Term, V, term_rpo, term_succ};

/// The most nodes an arm may have.
const ARM_MAX: usize = 16;

pub fn if_convert(f: &mut Func) {
    loop {
        let preds = term_preds(f);
        let mut changed = false;
        for x in term_rpo(f) {
            changed |= convert(f, &preds, x);
        }
        if !changed {
            break;
        }
    }
}

/// Predecessors by the terminators.
fn term_preds(f: &Func) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); f.blocks.len()];
    for (b, blk) in f.blocks.iter().enumerate() {
        for s in term_succ(blk) {
            preds[s].push(b);
        }
    }
    preds
}

/// An arm `a` of `x`: entered only from `x`, no phis, small and pure, and
/// going on to one block: that block.
fn arm(f: &Func, preds: &[Vec<usize>], x: usize, a: usize) -> Option<usize> {
    let b = &f.blocks[a];
    let Term::Jump(y) = b.term else { return None };
    let fits = preds[a] == [x] && b.phis.is_empty();
    (fits && size(f, &b.body).is_some_and(|n| n <= ARM_MAX)).then_some(y)
}

/// The nodes of a list of items, if all may run where the arm would not
/// (guards of pure code included: they still run only in their lanes).
fn size(f: &Func, items: &[Item]) -> Option<usize> {
    let mut n = 0;
    for it in items {
        n += match it {
            Item::N(v) if speculable(&f.node(*v).op) => 1,
            Item::N(_) => return None,
            Item::Guard { body, phis, .. } => size(f, body)? + phis.len(),
        };
    }
    Some(n)
}

/// Safe to run where the arm would not: arithmetic and title-address
/// translation (no memory, no lanes, no images).
fn speculable(op: &Op) -> bool {
    match op {
        Op::Call(h) => matches!(h, Helper::Translate | Helper::TranslateRange | Helper::RangeLeft),
        Op::Wave(..) | Op::Pick => false,
        o => o.pure(),
    }
}

/// Converts the branch of `x` if its arms (or one arm and the join) meet;
/// whether it did.
fn convert(f: &mut Func, preds: &[Vec<usize>], x: usize) -> bool {
    let Term::Branch(c, t, e) = f.blocks[x].term else { return false };
    let (yt, ye) = (arm(f, preds, x, t), arm(f, preds, x, e));
    // the join and the block each side comes from
    let (y, from_t, from_e) = match (yt, ye) {
        (Some(a), Some(b)) if a == b => (a, t, e),
        (Some(a), _) if a == e => (e, t, x),
        (_, Some(b)) if b == t => (t, x, e),
        _ => return false,
    };
    if y == x {
        return false;
    }
    let mut moved = Vec::new();
    for a in [t, e].into_iter().filter(|&a| a != y) {
        moved.append(&mut f.blocks[a].body);
        f.blocks[a].term = Term::Return;
    }
    f.blocks[x].body.extend(moved);
    // the join's phis: one edge from x, the value picked by the condition
    let yp = f.blocks[y].preds.clone();
    let (Some(kt), Some(ke)) = (yp.iter().position(|&p| p == from_t), yp.iter().position(|&p| p == from_e)) else {
        unreachable!("the join's preds name both sides")
    };
    for ph in f.blocks[y].phis.clone() {
        let n = f.node(ph).clone();
        let (vt, ve) = (n.args[kt], n.args[ke]);
        let v = if vt == ve { vt } else { select(f, x, c, vt, ve, n.ty) };
        let args = &mut f.nodes[ph as usize].args;
        args[kt] = v;
        args.remove(ke);
    }
    let preds_y = &mut f.blocks[y].preds;
    preds_y[kt] = x;
    preds_y.remove(ke);
    f.blocks[x].term = Term::Jump(y);
    true
}

/// `c ? a : b`, appended to block `x`.
fn select(f: &mut Func, x: usize, c: V, a: V, b: V, ty: super::tir::Ty) -> V {
    let v = f.push(Node { op: Op::Spv(op::SELECT), ty, args: vec![c, a, b], imm: Vec::new() });
    f.blocks[x].body.push(Item::N(v));
    v
}
