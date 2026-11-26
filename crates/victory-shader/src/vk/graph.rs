//! The block graph of a thread function as placement needs it: reverse
//! postorder, the dominator tree, loop headers, merges (blocks two forward
//! edges enter) and each loop's exit. Built once per function; every
//! placement attempt reads it.

use super::tir::{self, Func};

pub struct Graph {
    pub rpo: Vec<usize>,
    pub header: Vec<bool>,
    pub merge: Vec<bool>,
    pub children: Vec<Vec<usize>>,
    pub idom: Vec<usize>,
    /// Dominator-tree preorder number and the last number in the subtree.
    pre: Vec<(usize, usize)>,
    /// Per loop header: the exit placed at the loop construct's merge (a
    /// merge child of the header outside the loop, the most entered), so
    /// leaving the loop for it is a plain break.
    pub exit: Vec<Option<usize>>,
    /// Per loop header: its merge children outside the loop placed after
    /// the exit (their blocks wrap the loop).
    pub after_exit: Vec<Vec<usize>>,
    /// Blocks inside a loop (some natural loop holds them).
    pub in_loop: Vec<bool>,
}

impl Graph {
    pub fn new(f: &Func) -> Result<Self, &'static str> {
        let n = f.blocks.len();
        let succ: Vec<Vec<usize>> = f.blocks.iter().map(tir::term_succ).collect();
        let order = tir::term_rpo(f);
        let mut rpo = vec![usize::MAX; n];
        for (i, &b) in order.iter().enumerate() {
            rpo[b] = i;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for &s in &succ[b] {
                preds[s].push(b);
            }
        }
        let idom = idoms(&order, &preds, &rpo);
        let (header, merge) = headers_and_merges(&order, &succ, &rpo, &idom)?;
        let mut children = vec![Vec::new(); n];
        for &b in order.iter().skip(1) {
            children[idom[b]].push(b);
        }
        let mut exit = vec![None; n];
        let mut after_exit = vec![Vec::new(); n];
        let mut in_loop = vec![false; n];
        for &h in order.iter().filter(|&&h| header[h]) {
            let inside = natural_loop(h, &preds, &rpo);
            for (b, &i) in inside.iter().enumerate() {
                in_loop[b] |= i;
            }
            let (y, after) = loop_exit(h, &inside, &children, &merge, &preds, &rpo);
            exit[h] = y;
            after_exit[h] = after;
        }
        let pre = preorder(&children);
        Ok(Graph { rpo, header, merge, children, idom, pre, exit, after_exit, in_loop })
    }

    /// Whether block `a` dominates block `b`.
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        let (pa, ea) = self.pre[a];
        let pb = self.pre[b].0;
        pa <= pb && pb <= ea
    }
}

/// Dominator-tree preorder intervals from the entry.
fn preorder(children: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut pre = vec![(usize::MAX, 0); children.len()];
    let mut n = 0;
    let mut st = vec![(0usize, 0usize)];
    pre[0].0 = 0;
    while let Some(&mut (b, ref mut k)) = st.last_mut() {
        if let Some(&c) = children[b].get(*k) {
            *k += 1;
            n += 1;
            pre[c].0 = n;
            st.push((c, 0));
        } else {
            pre[b].1 = n;
            st.pop();
        }
    }
    pre
}

/// Immediate dominators (Cooper, Harvey and Kennedy); the entry is its own.
fn idoms(order: &[usize], preds: &[Vec<usize>], rpo: &[usize]) -> Vec<usize> {
    let mut idom = vec![usize::MAX; preds.len()];
    idom[0] = 0;
    let meet = |idom: &[usize], mut x: usize, mut y: usize| {
        while x != y {
            while rpo[x] > rpo[y] {
                x = idom[x];
            }
            while rpo[y] > rpo[x] {
                y = idom[y];
            }
        }
        x
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            let mut known = preds[b].iter().copied().filter(|&p| idom[p] != usize::MAX);
            let Some(first) = known.next() else { continue };
            let new = known.fold(first, |acc, p| meet(&idom, p, acc));
            if new != idom[b] {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Loop headers (targets of back edges, which must be dominated by them)
/// and merges.
fn headers_and_merges(
    order: &[usize],
    succ: &[Vec<usize>],
    rpo: &[usize],
    idom: &[usize],
) -> Result<(Vec<bool>, Vec<bool>), &'static str> {
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if b == 0 {
            return false;
        }
        b = idom[b];
    };
    let mut header = vec![false; succ.len()];
    let mut forward = vec![0usize; succ.len()];
    for &b in order {
        for &s in &succ[b] {
            if rpo[s] > rpo[b] {
                forward[s] += 1;
            } else if dominates(s, b) {
                header[s] = true;
            } else {
                return Err("irreducible control flow");
            }
        }
    }
    Ok((header, forward.iter().map(|&k| k >= 2).collect()))
}

/// The natural loop of header `h`: what reaches a back edge's source
/// without passing the header.
fn natural_loop(h: usize, preds: &[Vec<usize>], rpo: &[usize]) -> Vec<bool> {
    let mut inside = vec![false; preds.len()];
    inside[h] = true;
    let mut work: Vec<usize> = preds[h].iter().copied().filter(|&p| rpo[p] >= rpo[h]).collect();
    while let Some(b) = work.pop() {
        if !std::mem::replace(&mut inside[b], true) {
            work.extend(preds[b].iter().copied());
        }
    }
    inside
}

/// A loop's exit placed at its merge (the merge child outside the loop the
/// most edges leave for), and the merge children after it (they wrap the
/// loop).
fn loop_exit(
    h: usize,
    inside: &[bool],
    children: &[Vec<usize>],
    merge: &[bool],
    preds: &[Vec<usize>],
    rpo: &[usize],
) -> (Option<usize>, Vec<usize>) {
    let outside = |c: &usize| merge[*c] && !inside[*c];
    let edges = |c: usize| preds[c].iter().filter(|&&p| inside[p]).count();
    let best = children[h].iter().copied().filter(outside).filter(|&c| edges(c) > 0).max_by_key(|&c| (edges(c), std::cmp::Reverse(rpo[c])));
    let Some(y) = best else { return (None, Vec::new()) };
    let mut after: Vec<usize> = children[h].iter().copied().filter(outside).filter(|&c| rpo[c] > rpo[y]).collect();
    after.sort_by_key(|&c| std::cmp::Reverse(rpo[c]));
    (Some(y), after)
}

