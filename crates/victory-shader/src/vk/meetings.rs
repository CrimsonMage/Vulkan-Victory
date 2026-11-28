//! The meetings of a wave64's halves before shared-memory (LDS) accesses,
//! kept only where they order something.
//!
//! The console's wave runs an LDS instruction in all 64 lanes before the
//! next one; its two host subgroups are ordered only where they meet. The
//! lift puts a meeting before every LDS access and says in it what the
//! access does (lift_wave.rs `lds_meet`, [`access`]). Between two meetings
//! each half makes its accesses in its own time, so a meeting is needed
//! exactly where what the halves may do on its two sides does not commute.
//! What commutes: reads with reads (nothing changes), and atomics of one
//! [family](ADD) whose results nothing takes (every order leaves the same
//! memory, and nothing sees it between). A plain write, an atomic that
//! returns the old value, an exchange or a compare-exchange commutes with
//! nothing, itself included.
//!
//! So, forward over the block graph: the kinds of access the halves may
//! have made since they last surely met (a kept meeting, an exchange, the
//! console's barrier). A meeting whose access commutes with all of them
//! goes. Run on the finished program, after [`super::uniform`] took the
//! exchanges of uniform words away: only an exchange that is still there
//! is a meeting.
//!
//! The rule is trustno1-coder/VIctory-NVIDIA's `LdsOrder` (89cfae2: "an
//! access needs the host waves to meet only when a plain `ds_write` stands
//! in a piece it can reach or be reached from", pieces cut at `s_barrier`),
//! kept exact: theirs also lets reads and atomics pass each other (a half
//! may then read before the other half's add, which the console's wave
//! never does), ours does not; ours counts what precedes an access on a
//! path, not what shares its piece, so reads after the meeting that follows
//! a write need none either.

use std::collections::HashSet;

use super::tir::{Func, Item, Node, Op, V, term_succ};
use super::uniform::MEET;

/// What an LDS access does, in a meeting's `imm[1]`: a read.
pub const READ: u32 = 1;
/// A plain write (`ds_write*`).
pub const WRITE: u32 = 1 << 1;
/// An atomic whose old value the program takes (`*_rtn_*`), an exchange,
/// a compare-exchange, a float minimum or maximum.
pub const ORDERED: u32 = 1 << 2;
/// The atomics whose results nothing takes, by what commutes: adds and
/// subtractions; then each of the minima, maxima and bit operations.
pub const ADD: u32 = 1 << 3;
const FAMILIES: [(&str, u32); 9] = [
    ("ds_add_u32", ADD),
    ("ds_sub_u32", ADD),
    ("ds_min_i32", 1 << 4),
    ("ds_min_u32", 1 << 5),
    ("ds_max_i32", 1 << 6),
    ("ds_max_u32", 1 << 7),
    ("ds_and_b32", 1 << 8),
    ("ds_or_b32", 1 << 9),
    ("ds_xor_b32", 1 << 10),
];

/// The kind of the LDS access instruction `name` makes.
pub fn access(name: &str) -> u32 {
    if name.starts_with("ds_read") {
        READ
    } else if name.starts_with("ds_write") {
        WRITE
    } else {
        FAMILIES.iter().find(|f| f.0 == name).map_or(ORDERED, |f| f.1)
    }
}

/// Whether an access of `kind` must wait for the other half's accesses of
/// the kinds in `before`.
fn orders(before: u32, kind: u32) -> bool {
    if kind & (WRITE | ORDERED) != 0 { before != 0 } else { before & !kind != 0 }
}

/// The kind of the access the meeting `n` stands before; none for other
/// nodes and for a meeting at a memory barrier.
fn meeting(n: &Node) -> Option<u32> {
    (n.op == Op::Xchg && n.imm.first() == Some(&MEET)).then(|| n.imm.get(1).copied()).flatten()
}

/// The kinds of access made since the halves last met, after `items` run
/// from `before`; the meetings that order nothing into `gone`.
fn walk(f: &Func, items: &[Item], mut before: u32, top: bool, gone: &mut HashSet<V>) -> u32 {
    for it in items {
        match it {
            Item::N(v) => {
                let n = f.node(*v);
                match (meeting(n), &n.op) {
                    // (inside a guard not every invocation meets)
                    (Some(kind), _) if top && orders(before, kind) => before = kind,
                    (Some(kind), _) => {
                        before |= kind;
                        if top {
                            gone.insert(*v);
                        }
                    }
                    (None, Op::Xchg | Op::Barrier) if top => before = 0,
                    _ => {}
                }
            }
            Item::Guard { body, .. } => before = walk(f, body, before, false, gone),
        }
    }
    before
}

/// Removes the meetings before LDS accesses that order nothing (module
/// comment). Returns how many went.
pub fn drop_unneeded(f: &mut Func) -> usize {
    let n = f.blocks.len();
    // per block: the kinds of access since the last meeting, at its start
    let mut start = vec![0u32; n];
    let mut gone = HashSet::new();
    loop {
        let mut changed = false;
        gone.clear();
        for b in 0..n {
            let end = walk(f, &f.blocks[b].body, start[b], true, &mut gone);
            for s in term_succ(&f.blocks[b]) {
                changed |= start[s] | end != start[s];
                start[s] |= end;
            }
        }
        if !changed {
            break;
        }
    }
    for b in &mut f.blocks {
        b.body.retain(|it| !matches!(it, Item::N(v) if gone.contains(v)));
    }
    gone.len()
}

/// The meetings before LDS accesses in the program (statistics).
pub fn count(f: &Func) -> usize {
    f.blocks.iter().flat_map(|b| &b.body).filter(|it| matches!(it, Item::N(v) if meeting(f.node(*v)).is_some())).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::tir::{Block, S, Term, Ty};

    fn meet(f: &mut Func, kind: u32) -> Item {
        let z = f.konst(Ty::U32, 0);
        Item::N(f.push(Node { op: Op::Xchg, ty: Ty::vec(S::U32, 2), args: vec![z], imm: vec![MEET, kind] }))
    }

    fn kept(f: &Func) -> Vec<u32> {
        f.blocks.iter().flat_map(|b| &b.body).filter_map(|it| match it {
            Item::N(v) => meeting(f.node(*v)),
            Item::Guard { .. } => None,
        })
        .collect()
    }

    #[test]
    fn kinds_of_access() {
        assert_eq!(access("ds_read2st64_b32"), READ);
        assert_eq!(access("ds_write_b16_d16_hi"), WRITE);
        assert_eq!(access("ds_add_u32"), ADD);
        assert_eq!(access("ds_sub_u32"), ADD);
        assert_ne!(access("ds_max_u32"), access("ds_min_u32"));
        for ordered in ["ds_add_rtn_u32", "ds_wrxchg_rtn_b32", "ds_cmpst_rtn_b32", "ds_cmpst_b32", "ds_max_f32"] {
            assert_eq!(access(ordered), ORDERED, "{ordered}");
        }
    }

    /// One block: write, read, read, add, add, max, add with its result,
    /// read. The first access follows nothing; a read after a read and an
    /// add after an add order nothing; every other meeting stays.
    #[test]
    fn meetings_between_accesses_that_commute_go() {
        let mut f = Func::default();
        let max = access("ds_max_u32");
        let kinds = [WRITE, READ, READ, ADD, ADD, max, ORDERED, ORDERED, READ];
        let body = kinds.iter().map(|&k| meet(&mut f, k)).collect();
        f.blocks.push(Block { phis: vec![], preds: vec![], body, term: Term::Return });
        assert_eq!(drop_unneeded(&mut f), 3);
        assert_eq!(kept(&f), [READ, ADD, max, ORDERED, ORDERED, READ]);
    }

    /// A loop of reads after a barrier: nothing a read must wait for
    /// comes round the back edge. With a write in the loop every
    /// iteration's first read waits for the write before it.
    #[test]
    fn what_comes_round_a_loop_counts() {
        let build = |tail: Option<u32>| {
            let mut f = Func::default();
            let bar = Item::N(f.push(Node { op: Op::Barrier, ty: Ty::VOID, args: vec![], imm: vec![] }));
            let head = vec![meet(&mut f, WRITE), bar];
            let mut body = vec![meet(&mut f, READ), meet(&mut f, READ)];
            body.extend(tail.map(|k| meet(&mut f, k)));
            let c = f.konst(Ty::BOOL, 1);
            f.blocks.push(Block { phis: vec![], preds: vec![], body: head, term: Term::Jump(1) });
            f.blocks.push(Block { phis: vec![], preds: vec![0, 1], body, term: Term::Branch(c, 1, 2) });
            f.blocks.push(Block { phis: vec![], preds: vec![1], body: vec![], term: Term::Return });
            f
        };
        let mut reads = build(None);
        assert_eq!(drop_unneeded(&mut reads), 3);
        assert_eq!(kept(&reads), [] as [u32; 0]);
        let mut written = build(Some(WRITE));
        assert_eq!(drop_unneeded(&mut written), 2);
        assert_eq!(kept(&written), [READ, WRITE]);
    }

    /// An exchange of the halves is a meeting; one the uniform pass took
    /// away (a pair made by each half itself) is not.
    #[test]
    fn only_an_exchange_that_is_there_is_a_meeting() {
        let build = |exchange: Op| {
            let mut f = Func::default();
            let z = f.konst(Ty::U32, 0);
            let x = Item::N(f.push(Node { op: exchange, ty: Ty::vec(S::U32, 2), args: vec![z, z], imm: vec![] }));
            let body = vec![meet(&mut f, WRITE), x, meet(&mut f, READ)];
            f.blocks.push(Block { phis: vec![], preds: vec![], body, term: Term::Return });
            f
        };
        let mut met = build(Op::Xchg);
        assert_eq!(drop_unneeded(&mut met), 2);
        let mut alone = build(Op::Spv(crate::vk::spirv::op::COMPOSITE_CONSTRUCT));
        assert_eq!(drop_unneeded(&mut alone), 1);
        assert_eq!(kept(&alone), [READ]);
    }
}
