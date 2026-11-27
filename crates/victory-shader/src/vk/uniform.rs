//! Wave-uniform values of the thread IR, and what they make unnecessary.
//!
//! A value is **wave-uniform** when every invocation of a console wave
//! (both host subgroups of a wave64 run as halves) computes the same bits
//! for it: constants, user data, the memory map, the group id (and the
//! mesh group input made from it); a pure operation, helper or typed decode
//! of uniform operands; a load from a uniform address in a program that
//! writes no memory (a half could otherwise load before the other half's
//! store lands); a block phi of uniform values (block branches are the
//! console's scalar branches, taken by the whole wave); a guard's phi whose
//! condition and both values are uniform (an EXEC of all lanes or none);
//! an exchange's pair (both halves get the same two words); a ballot, any
//! or all of a uniform boolean, and a broadcast or shuffle of a uniform
//! value. Lane and thread ids, inputs, LDS loads, atomics, samples and
//! scans are not. Found optimistically (every value uniform, then the ones
//! that break a rule dropped until none does), so loop-carried values the
//! wave advances together stay uniform.
//!
//! Then, exactly:
//! - a broadcast, shuffle, any or all of a uniform value is that value
//!   (`v_readfirstlane`, `v_readlane` of a uniform VGPR: every lane's copy
//!   is the same, the lanes outside EXEC's too, since every invocation
//!   computes it);
//! - an exchange of a uniform word is that word twice, and an exchange whose
//!   pair nothing reads goes;
//! - in a program with no shared state (no LDS access, no memory or image
//!   write) the halves' meetings and the console's barriers order nothing
//!   and go.

use std::collections::HashMap;

use super::spirv::{glsl, op};
use super::tir::{Builtin, Func, ImageKind, Item, Node, Op, S, Term, Ty, V};

/// `Node::imm` of an exchange that is only a meeting of the halves (its
/// pair unused: before an LDS access, at a memory barrier).
pub const MEET: u32 = 1;

/// What the pass changed (statistics).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Simplified {
    pub uniform_reads: usize,
    pub exchanges: usize,
    pub meetings: usize,
    pub barriers: usize,
    /// Exchanges whose partner word each half computes itself.
    pub computed: usize,
}

/// Rewrites what uniform values make unnecessary; returns what it did and
/// which values (by node, before the rewrites) are uniform.
///
/// `paired`: every console wave is two full host subgroups (a wave64
/// compute or mesh group of whole waves), so an exchange of a word made
/// only of uniform values and the invocation's place can be computed by
/// each half for the other.
pub fn simplify(f: &mut Func, paired: bool) -> (Simplified, Vec<bool>) {
    let mut s = Simplified::default();
    let shared = has_shared_state(f);
    let uni = uniform_values(f, shared);
    let guards = guard_conds(f);
    let mut subst: HashMap<V, V> = HashMap::new();
    for v in 0..f.nodes.len() {
        let n = f.nodes[v].clone();
        match n.op {
            Op::Wave(o, _) if reads_one_lane(o) && uni[n.args[0] as usize] && f.ty(n.args[0]) == n.ty => {
                subst.insert(v as V, n.args[0]);
                s.uniform_reads += 1;
            }
            Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, _) => {
                if let Some(a) = active_pick(f, &n, &guards).filter(|&a| uni[a as usize] && f.ty(a) == n.ty) {
                    subst.insert(v as V, a);
                    s.uniform_reads += 1;
                }
            }
            Op::Xchg if n.imm.first() == Some(&MEET) => {
                if !shared {
                    let a = n.args[0];
                    f.nodes[v] = pair(a);
                    s.meetings += 1;
                }
            }
            Op::Xchg if uni[n.args[0] as usize] => {
                let a = n.args[0];
                f.nodes[v] = pair(a);
                s.exchanges += 1;
            }
            _ => {}
        }
    }
    substitute(f, &subst);
    s.exchanges += drop_unread_exchanges(f);
    if paired {
        s.computed = compute_partner_words(f, &uni);
    }
    if !shared {
        s.barriers = drop_barriers(f);
    }
    (s, uni)
}

/// Subgroup operations that give one lane's value (of argument 0).
fn reads_one_lane(o: u16) -> bool {
    matches!(
        o,
        op::GROUP_NON_UNIFORM_BROADCAST
            | op::GROUP_NON_UNIFORM_BROADCAST_FIRST
            | op::GROUP_NON_UNIFORM_SHUFFLE
            | op::GROUP_NON_UNIFORM_SHUFFLE_XOR
            | op::GROUP_NON_UNIFORM_QUAD_BROADCAST
            | op::GROUP_NON_UNIFORM_QUAD_SWAP
            | op::GROUP_NON_UNIFORM_ALL
            | op::GROUP_NON_UNIFORM_ANY
    )
}

/// The pair (a, a): what an exchange of a uniform word gives.
fn pair(a: V) -> Node {
    Node { op: Op::Spv(op::COMPOSITE_CONSTRUCT), ty: Ty::vec(S::U32, 2), args: vec![a, a], imm: Vec::new() }
}

/// LDS accesses, memory or image writes: what the halves' meetings and the
/// console's barriers order.
pub fn has_shared_state(f: &Func) -> bool {
    f.nodes.iter().any(|n| match n.op {
        Op::LdsLoad | Op::LdsStore | Op::LdsAtomic(_) | Op::Store { .. } | Op::Atomic(_) => true,
        Op::Image { kind, .. } => matches!(kind, ImageKind::Store { .. } | ImageKind::Atomic(_)),
        _ => false,
    })
}

/// Each guard phi's condition.
fn guard_conds(f: &Func) -> HashMap<V, V> {
    fn walk(list: &[Item], out: &mut HashMap<V, V>) {
        for it in list {
            if let Item::Guard { cond, body, phis } = it {
                out.extend(phis.iter().map(|&p| (p, *cond)));
                walk(body, out);
            }
        }
    }
    let mut out = HashMap::new();
    for b in &f.blocks {
        walk(&b.body, &mut out);
    }
    out
}

/// Which values are wave-uniform (module comment), by node. A ballot of a
/// uniform boolean is the same in both halves: a wave64's lower half is a
/// full subgroup whenever its upper half exists.
pub fn uniform_values(f: &Func, writes: bool) -> Vec<bool> {
    let guards = guard_conds(f);
    let mut uni: Vec<bool> = f.nodes.iter().map(|n| source(n, writes) != Some(false)).collect();
    loop {
        let mut changed = false;
        for v in 0..f.nodes.len() {
            if uni[v] && !holds(f, v as V, &uni, &guards, writes) {
                uni[v] = false;
                changed = true;
            }
        }
        if !changed {
            return uni;
        }
    }
}

/// A value uniform or not whatever its arguments are (`None`: by them).
fn source(n: &Node, writes: bool) -> Option<bool> {
    Some(match n.op {
        Op::Const(_) | Op::Undef | Op::Push(_) | Op::MapField(_) | Op::Xchg => true,
        Op::Builtin(b) => matches!(b, Builtin::WorkgroupId(_)),
        Op::Spv(_) | Op::Glsl(_) | Op::Call(_) | Op::Pick | Op::Phi | Op::Wave(..) => return None,
        Op::Load { .. } if !writes => return None,
        Op::Image { kind: ImageKind::Load { .. } | ImageKind::Size | ImageKind::Levels, .. } if !writes => return None,
        _ => false,
    })
}

fn holds(f: &Func, v: V, uni: &[bool], guards: &HashMap<V, V>, writes: bool) -> bool {
    let n = f.node(v);
    let u = |x: &V| uni[*x as usize];
    if let Some(s) = source(n, writes) {
        return s;
    }
    match n.op {
        Op::Phi => match guards.get(&v) {
            Some(&c) => {
                let (inner, outer) = (n.args[0], n.args[1]);
                match f.as_const(c) {
                    Some(0) => u(&outer),
                    Some(_) => u(&inner),
                    None => u(&c) && u(&inner) && u(&outer),
                }
            }
            None => n.args.iter().all(u),
        },
        Op::Wave(o, _) if reads_one_lane(o) || o == op::GROUP_NON_UNIFORM_BALLOT => {
            u(&n.args[0]) || (o == op::GROUP_NON_UNIFORM_SHUFFLE && active_pick(f, n, guards).is_some_and(|a| u(&a)))
        }
        Op::Wave(..) => false,
        _ => n.args.iter().all(u),
    }
}

/// A shuffle that reads `x` at the first lane where `e` holds
/// (`v_readfirstlane` under EXEC `e`, or of a lane `s_ff1` found in `e`),
/// where `x` is `e ? a : old` (a VALU write under `e`): `a`, the value
/// every lane in `e` holds.
fn active_pick(f: &Func, n: &Node, guards: &HashMap<V, V>) -> Option<V> {
    let e = first_lane_of(f, *n.args.get(1)?)?;
    let x = f.node(n.args[0]);
    match x.op {
        Op::Spv(op::SELECT) if x.args[0] == e => Some(x.args[1]),
        Op::Phi if guards.get(&n.args[0]) == Some(&e) => Some(x.args[0]),
        _ => None,
    }
}

/// The lane mask `e` whose first set lane (0-31 in its subgroup) `idx` is:
/// `findLSB(ballot(e)) & 31`, or the 64-bit search of a wave64's two
/// exchanged ballot words (lift_wave.rs `ff1_64`).
fn first_lane_of(f: &Func, idx: V) -> Option<V> {
    let n = f.node(idx);
    if n.op != Op::Spv(op::BITWISE_AND) || n.args.len() != 2 {
        return None;
    }
    let l = match (f.as_const(n.args[0]), f.as_const(n.args[1])) {
        (_, Some(31)) => n.args[0],
        (Some(31), _) => n.args[1],
        _ => return None,
    };
    let lsb = |v: V| {
        let n = f.node(v);
        (n.op == Op::Glsl(glsl::FIND_I_LSB)).then(|| n.args[0])
    };
    if let Some(w) = lsb(l) {
        return ballot_of(f, w);
    }
    // select(lo != 0, lsb(lo), select(hi != 0, lsb(hi) + 32, -1))
    let s = f.node(l);
    if s.op != Op::Spv(op::SELECT) {
        return None;
    }
    let lo = lsb(s.args[1])?;
    let inner = f.node(s.args[2]);
    if inner.op != Op::Spv(op::SELECT) || f.node(inner.args[1]).op != Op::Spv(op::I_ADD) {
        return None;
    }
    let hi = lsb(f.node(inner.args[1]).args[0])?;
    let (wl, wh) = (word_of(f, lo, 0)?, word_of(f, hi, 1)?);
    (wl == wh).then_some(())?;
    ballot_of(f, wl)
}

/// `w` when `v` is word `k` of the halves' exchange of `w` (or of the pair
/// it became).
fn word_of(f: &Func, v: V, k: u32) -> Option<V> {
    let n = f.node(v);
    (n.op == Op::Spv(op::COMPOSITE_EXTRACT) && n.imm == [k]).then_some(())?;
    let x = f.node(n.args[0]);
    match x.op {
        Op::Xchg => Some(x.args[0]),
        Op::Spv(op::COMPOSITE_CONSTRUCT) if x.args.len() == 2 && x.args[0] == x.args[1] => Some(x.args[0]),
        _ => None,
    }
}

/// `e` when `w` is the first word of `ballot(e)`.
fn ballot_of(f: &Func, w: V) -> Option<V> {
    let n = f.node(w);
    (n.op == Op::Spv(op::COMPOSITE_EXTRACT) && n.imm == [0]).then_some(())?;
    let b = f.node(n.args[0]);
    (b.op == Op::Wave(op::GROUP_NON_UNIFORM_BALLOT, u32::MAX)).then(|| b.args[0])
}

/// Values made only of uniform values, the subgroup id and (`lane`) the
/// lane, by pure operations: the other half's copy is the same expression
/// with the subgroup id's lowest bit flipped.
fn placed(f: &Func, v: V, uni: &[bool], lane: bool, memo: &mut HashMap<V, bool>) -> bool {
    if uni.get(v as usize).copied().unwrap_or(false) {
        return true;
    }
    if let Some(&m) = memo.get(&v) {
        return m;
    }
    let n = f.node(v);
    let r = match n.op {
        Op::Builtin(Builtin::SubgroupId) => true,
        Op::Builtin(Builtin::SubgroupLocalId) => lane,
        Op::Spv(o) if o != op::COMPOSITE_EXTRACT || f.node(n.args[0]).op != Op::Xchg => {
            n.args.iter().all(|&a| placed(f, a, uni, lane, memo))
        }
        Op::Glsl(_) => n.args.iter().all(|&a| placed(f, a, uni, lane, memo)),
        _ => false,
    };
    memo.insert(v, r);
    r
}

/// The other half's copy of placed value `v` (new nodes into `out`).
fn other_half(f: &mut Func, v: V, uni: &[bool], copies: &mut HashMap<V, V>, out: &mut Vec<V>) -> V {
    if uni.get(v as usize).copied().unwrap_or(false) || f.node(v).op == Op::Builtin(Builtin::SubgroupLocalId) {
        return v;
    }
    if let Some(&c) = copies.get(&v) {
        return c;
    }
    let n = f.node(v).clone();
    let c = if n.op == Op::Builtin(Builtin::SubgroupId) {
        let one = f.konst(Ty::U32, 1);
        f.push(Node { op: Op::Spv(op::BITWISE_XOR), ty: Ty::U32, args: vec![v, one], imm: Vec::new() })
    } else {
        let args = n.args.iter().map(|&a| other_half(f, a, uni, copies, out)).collect();
        f.push(Node { args, ..n })
    };
    out.push(c);
    copies.insert(v, c);
    c
}

/// An exchanged word each half can compute for the other: a placed value
/// that does not depend on the lane (an exchanged word is the same in its
/// whole half), or the first word of the ballot of a placed boolean (at
/// the top level of a block, where every invocation takes part). Returns
/// the other half's word (new nodes into `out`).
fn partner_word(f: &mut Func, own: V, uni: &[bool], out: &mut Vec<V>) -> Option<V> {
    let mut copies = HashMap::new();
    if placed(f, own, uni, false, &mut HashMap::new()) {
        return Some(other_half(f, own, uni, &mut copies, out));
    }
    let b = ballot_of(f, own)?;
    if !placed(f, b, uni, true, &mut HashMap::new()) {
        return None;
    }
    let ob = other_half(f, b, uni, &mut copies, out);
    let v4 = f.push(Node { op: Op::Wave(op::GROUP_NON_UNIFORM_BALLOT, u32::MAX), ty: Ty::vec(S::U32, 4), args: vec![ob], imm: Vec::new() });
    let w = f.push(Node { op: Op::Spv(op::COMPOSITE_EXTRACT), ty: Ty::U32, args: vec![v4], imm: vec![0] });
    out.extend([v4, w]);
    Some(w)
}

/// Exchanges whose partner word each half computes itself become that
/// computation (no meeting). Returns how many.
fn compute_partner_words(f: &mut Func, uni: &[bool]) -> usize {
    let mut count = 0;
    let mut blocks = std::mem::take(&mut f.blocks);
    for b in &mut blocks {
        let mut body = Vec::with_capacity(b.body.len());
        for it in std::mem::take(&mut b.body) {
            let Item::N(v) = it else {
                body.push(it);
                continue;
            };
            let n = f.node(v).clone();
            let mut made = Vec::new();
            if n.op == Op::Xchg && n.imm.first() != Some(&MEET)
                && let Some(other) = partner_word(f, n.args[0], uni, &mut made)
            {
                let own = n.args[0];
                let sg = f.global(Op::Builtin(Builtin::SubgroupId), Ty::U32);
                let (one, zero) = (f.konst(Ty::U32, 1), f.konst(Ty::U32, 0));
                let mut push = |f: &mut Func, op: u16, ty: Ty, args: Vec<V>| {
                    let x = f.push(Node { op: Op::Spv(op), ty, args, imm: Vec::new() });
                    made.push(x);
                    x
                };
                let half = push(f, op::BITWISE_AND, Ty::U32, vec![sg, one]);
                let lower = push(f, op::I_EQUAL, Ty::BOOL, vec![half, zero]);
                let lo = push(f, op::SELECT, Ty::U32, vec![lower, own, other]);
                let hi = push(f, op::SELECT, Ty::U32, vec![lower, other, own]);
                f.nodes[v as usize] = Node { op: Op::Spv(op::COMPOSITE_CONSTRUCT), ty: Ty::vec(S::U32, 2), args: vec![lo, hi], imm: Vec::new() };
                body.extend(made.into_iter().map(Item::N));
                count += 1;
            }
            body.push(Item::N(v));
        }
        b.body = body;
    }
    f.blocks = blocks;
    count
}

/// Replaces every use of a key by its value.
fn substitute(f: &mut Func, subst: &HashMap<V, V>) {
    if subst.is_empty() {
        return;
    }
    let find = |mut v: V| {
        while let Some(&w) = subst.get(&v) {
            v = w;
        }
        v
    };
    for n in &mut f.nodes {
        for a in &mut n.args {
            *a = find(*a);
        }
    }
    fn items(list: &mut [Item], find: &dyn Fn(V) -> V) {
        for it in list {
            if let Item::Guard { cond, body, .. } = it {
                *cond = find(*cond);
                items(body, find);
            }
        }
    }
    for b in &mut f.blocks {
        items(&mut b.body, &find);
        if let Term::Branch(c, _, _) = &mut b.term {
            *c = find(*c);
        }
    }
}

/// Exchanges (not meetings) whose pair nothing live reads become pairs of
/// their own word (removable). Returns how many.
fn drop_unread_exchanges(f: &mut Func) -> usize {
    let n = f.nodes.len();
    let exchange = |f: &Func, v: V| f.node(v).op == Op::Xchg && f.node(v).imm.first() != Some(&MEET);
    let mut live = vec![false; n];
    let mut work: Vec<V> = Vec::new();
    fn roots(f: &Func, list: &[Item], work: &mut Vec<V>, exchange: &dyn Fn(&Func, V) -> bool) {
        for it in list {
            match it {
                Item::N(v) => {
                    if !f.node(*v).op.removable() && !exchange(f, *v) {
                        work.push(*v);
                    }
                }
                Item::Guard { cond, body, .. } => {
                    work.push(*cond);
                    roots(f, body, work, exchange);
                }
            }
        }
    }
    for b in &f.blocks {
        roots(f, &b.body, &mut work, &exchange);
        if let Term::Branch(c, _, _) = b.term {
            work.push(c);
        }
    }
    while let Some(v) = work.pop() {
        if !std::mem::replace(&mut live[v as usize], true) {
            work.extend(f.node(v).args.iter().copied());
        }
    }
    let dead: Vec<usize> = (0..n).filter(|&v| !live[v] && exchange(f, v as V)).collect();
    for &v in &dead {
        let a = f.nodes[v].args[0];
        f.nodes[v] = pair(a);
    }
    dead.len()
}

/// Removes the console's barriers. Returns how many.
fn drop_barriers(f: &mut Func) -> usize {
    fn sweep(f: &Func, list: &mut Vec<Item>, count: &mut usize) {
        list.retain_mut(|it| match it {
            Item::N(v) => {
                let barrier = f.node(*v).op == Op::Barrier;
                *count += usize::from(barrier);
                !barrier
            }
            Item::Guard { body, .. } => {
                sweep(f, body, count);
                true
            }
        });
    }
    let mut count = 0;
    let mut blocks = std::mem::take(&mut f.blocks);
    for b in &mut blocks {
        sweep(f, &mut b.body, &mut count);
    }
    f.blocks = blocks;
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::tir::Block;

    fn wave(f: &mut Func, o: u16, ty: Ty, args: Vec<V>) -> V {
        f.push(Node { op: Op::Wave(o, u32::MAX), ty, args, imm: Vec::new() })
    }

    #[test]
    fn a_uniform_read_and_exchange_go_and_a_lane_value_stays() {
        let mut f = Func::default();
        let user = f.global(Op::Push(4), Ty::U32);
        let lane = f.global(Op::Builtin(Builtin::SubgroupLocalId), Ty::U32);
        let one = f.konst(Ty::U32, 1);
        let sum = f.push(Node { op: Op::Spv(op::I_ADD), ty: Ty::U32, args: vec![user, one], imm: vec![] });
        let first = wave(&mut f, op::GROUP_NON_UNIFORM_BROADCAST_FIRST, Ty::U32, vec![sum]);
        let other = wave(&mut f, op::GROUP_NON_UNIFORM_BROADCAST_FIRST, Ty::U32, vec![lane]);
        let x = f.push(Node { op: Op::Xchg, ty: Ty::vec(S::U32, 2), args: vec![first], imm: vec![] });
        let y = f.push(Node { op: Op::Xchg, ty: Ty::vec(S::U32, 2), args: vec![other], imm: vec![] });
        let out = |f: &mut Func, a: V| f.push(Node { op: Op::Output { slot: 0, comp: 0 }, ty: Ty::VOID, args: vec![a, one], imm: vec![] });
        let (o1, o2) = (out(&mut f, x), out(&mut f, y));
        let bar = f.push(Node { op: Op::Barrier, ty: Ty::VOID, args: vec![], imm: vec![] });
        let body = [sum, first, other, x, y, o1, bar, o2].map(Item::N).to_vec();
        f.blocks.push(Block { phis: vec![], preds: vec![], body, term: Term::Return });
        let (s, _) = simplify(&mut f, true);
        assert_eq!(s, Simplified { uniform_reads: 1, exchanges: 1, meetings: 0, barriers: 1, computed: 0 });
        assert_eq!(f.node(x).op, Op::Spv(op::COMPOSITE_CONSTRUCT));
        assert_eq!(f.node(x).args, vec![sum, sum]);
        assert_eq!(f.node(y).op, Op::Xchg);
    }

    #[test]
    fn a_ballot_of_the_lane_place_is_computed_for_the_other_half() {
        let mut f = Func::default();
        let lane = f.global(Op::Builtin(Builtin::SubgroupLocalId), Ty::U32);
        let sg = f.global(Op::Builtin(Builtin::SubgroupId), Ty::U32);
        let (five, k32) = (f.konst(Ty::U32, 5), f.konst(Ty::U32, 32));
        let spv = |f: &mut Func, o: u16, ty: Ty, args: Vec<V>| f.push(Node { op: Op::Spv(o), ty, args, imm: vec![] });
        let base = spv(&mut f, op::I_MUL, Ty::U32, vec![sg, k32]);
        let t = spv(&mut f, op::I_ADD, Ty::U32, vec![base, lane]);
        let lt = spv(&mut f, op::U_LESS_THAN, Ty::BOOL, vec![t, five]);
        let b4 = wave(&mut f, op::GROUP_NON_UNIFORM_BALLOT, Ty::vec(S::U32, 4), vec![lt]);
        let w = f.push(Node { op: Op::Spv(op::COMPOSITE_EXTRACT), ty: Ty::U32, args: vec![b4], imm: vec![0] });
        let x = f.push(Node { op: Op::Xchg, ty: Ty::vec(S::U32, 2), args: vec![w], imm: vec![] });
        let o = f.push(Node { op: Op::Output { slot: 0, comp: 0 }, ty: Ty::VOID, args: vec![x, five], imm: vec![] });
        let body = [base, t, lt, b4, w, x, o].map(Item::N).to_vec();
        f.blocks.push(Block { phis: vec![], preds: vec![], body, term: Term::Return });
        let (s, _) = simplify(&mut f, true);
        assert_eq!(s.computed, 1);
        assert_eq!(f.node(x).op, Op::Spv(op::COMPOSITE_CONSTRUCT));
        // the copy of the compare reads the subgroup id with its low bit flipped
        assert!(f.nodes.iter().any(|n| n.op == Op::Spv(op::BITWISE_XOR) && n.args[0] == sg));
    }
}
