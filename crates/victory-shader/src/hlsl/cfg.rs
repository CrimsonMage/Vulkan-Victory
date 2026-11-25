//! Structured HLSL control flow from a reducible graph of blocks.
//!
//! HLSL has `if`, loops, `break` and `continue`, but no `goto` and no
//! labelled jumps. The graph of a translated program is reducible
//! (`ir::build` splits irreducible loops), so it is structured as in
//! Ramsey's "Beyond Relooper" (JFP 2022): each node's code is placed by the
//! dominator tree; a loop header opens a loop, a node with several forward
//! predecessors (a merge) is placed after a block that its predecessors
//! leave. A loop is `for (;;) { ... }`, a block `do { ... } while (false);`;
//! every path ends in an explicit jump, so neither falls through. A jump past
//! more than the innermost construct sets `jmp` to its target and breaks; each
//! construct passes it outwards until the target's construct takes it.

use std::fmt::Write;

/// How a block ends, with HLSL text supplied by the lowering.
#[derive(Debug, Clone)]
pub enum End {
    Jump(usize),
    /// `if (cond) taken else not_taken`.
    Branch(String, usize, usize),
    /// Leaves the program (the text, e.g. `return;`, ends the path).
    Leave(String),
}

/// What the structurizer needs per block.
pub trait Blocks {
    fn count(&self) -> usize;
    /// The block's statements (without its exit).
    fn body(&self, b: usize) -> &str;
    fn end(&self, b: usize) -> &End;
    /// Statements before taking the edge `from -> to` (phi copies).
    fn edge(&self, from: usize, to: usize) -> String;
    /// Statements on a back edge to loop header `h`, before `continue`
    /// (the iteration count and its limit).
    fn back_edge(&self, h: usize) -> String;
    /// Statements where a loop headed by `h` is entered (before `for`).
    fn loop_entry(&self, h: usize) -> String;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctx {
    Loop(usize),
    Block(usize),
}

struct Graph {
    succ: Vec<Vec<usize>>,
    rpo: Vec<usize>,
    order: Vec<usize>,
    idom: Vec<usize>,
    header: Vec<bool>,
    merge: Vec<bool>,
    children: Vec<Vec<usize>>,
}

fn successors(end: &End) -> Vec<usize> {
    match *end {
        End::Jump(t) => vec![t],
        End::Branch(_, t, f) => vec![t, f],
        End::Leave(_) => vec![],
    }
}

impl Graph {
    fn new(bl: &dyn Blocks) -> Result<Self, &'static str> {
        let n = bl.count();
        let succ: Vec<Vec<usize>> = (0..n).map(|b| successors(bl.end(b))).collect();
        // Reverse postorder from block 0.
        let mut post = Vec::new();
        let mut seen = vec![false; n];
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            if *k < succ[b].len() {
                let s = succ[b][*k];
                *k += 1;
                if !seen[s] {
                    seen[s] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        let order: Vec<usize> = post.into_iter().rev().collect();
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
        // Cooper, Harvey and Kennedy.
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX {
                        p
                    } else {
                        let (mut x, mut y) = (p, new);
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
                }
                if new != idom[b] {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let dominates = |a: usize, mut b: usize| loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = idom[b];
        };
        let mut header = vec![false; n];
        let mut forward = vec![0usize; n];
        for &b in &order {
            for &s in &succ[b] {
                if rpo[s] <= rpo[b] {
                    if !dominates(s, b) {
                        return Err("irreducible control flow");
                    }
                    header[s] = true;
                } else {
                    forward[s] += 1;
                }
            }
        }
        let merge: Vec<bool> = forward.iter().map(|&k| k >= 2).collect();
        let mut children = vec![Vec::new(); n];
        for &b in order.iter().skip(1) {
            children[idom[b]].push(b);
        }
        Ok(Graph {
            succ,
            rpo,
            order,
            idom,
            header,
            merge,
            children,
        })
    }
}

struct Out<'a> {
    g: Graph,
    bl: &'a dyn Blocks,
    text: String,
    depth: usize,
    /// Per construct (by stack position), whether a jump to an outer one
    /// passes through it (then its end passes `jmp` on).
    passes: Vec<bool>,
    uses_jmp: bool,
}

/// Jump codes: construct at stack position `i`; continue (loop) or exit.
fn code(i: usize, cont: bool) -> usize {
    2 * i + 1 + usize::from(cont)
}

impl Out<'_> {
    fn line(&mut self, s: &str) {
        for l in s.lines() {
            if l.is_empty() {
                continue;
            }
            for _ in 0..self.depth {
                self.text.push_str("  ");
            }
            self.text.push_str(l);
            self.text.push('\n');
        }
    }

    fn tree(&mut self, x: usize, ctx: &mut Vec<Ctx>) {
        let merges: Vec<usize> = {
            let mut m: Vec<usize> = self.g.children[x]
                .iter()
                .copied()
                .filter(|&c| self.g.merge[c])
                .collect();
            // Highest reverse-postorder number first: it is placed last.
            m.sort_by_key(|&c| std::cmp::Reverse(self.g.rpo[c]));
            m
        };
        if self.g.header[x] {
            let entry = self.bl.loop_entry(x);
            self.line(&entry);
            self.line("[loop] for (;;) {");
            self.depth += 1;
            ctx.push(Ctx::Loop(x));
            self.passes.push(false);
            self.within(x, &merges, ctx);
            self.close(ctx, "}");
        } else {
            self.within(x, &merges, ctx);
        }
    }

    fn within(&mut self, x: usize, merges: &[usize], ctx: &mut Vec<Ctx>) {
        match merges.split_first() {
            None => {
                let body = self.bl.body(x).to_owned();
                self.line(&body);
                match self.bl.end(x).clone() {
                    End::Jump(t) => self.jump(x, t, ctx),
                    End::Branch(cond, t, f) => {
                        self.line(&format!("if ({cond}) {{"));
                        self.depth += 1;
                        self.jump(x, t, ctx);
                        self.depth -= 1;
                        self.line("} else {");
                        self.depth += 1;
                        self.jump(x, f, ctx);
                        self.depth -= 1;
                        self.line("}");
                    }
                    End::Leave(text) => self.line(&text),
                }
            }
            Some((&y, rest)) => {
                self.line("do {");
                self.depth += 1;
                ctx.push(Ctx::Block(y));
                self.passes.push(false);
                self.within(x, rest, ctx);
                self.close(ctx, "} while (false);");
                self.tree(y, ctx);
            }
        }
    }

    /// Ends the innermost construct; passes on a jump to an outer one.
    fn close(&mut self, ctx: &mut Vec<Ctx>, text: &str) {
        self.depth -= 1;
        self.line(text);
        let i = ctx.len() - 1;
        ctx.pop();
        let passes = self.passes.pop().unwrap_or(false);
        if !passes || i == 0 {
            return;
        }
        // We are now directly inside construct i - 1.
        let outer = i - 1;
        let mut s = String::from("if (jmp != 0u) {\n");
        match ctx[outer] {
            Ctx::Loop(_) => {
                let _ = writeln!(s, "  if (jmp == {}u) {{ jmp = 0u; continue; }}", code(outer, true));
                let _ = writeln!(s, "  if (jmp == {}u) {{ jmp = 0u; break; }}", code(outer, false));
            }
            Ctx::Block(_) => {
                let _ = writeln!(s, "  if (jmp == {}u) {{ jmp = 0u; break; }}", code(outer, false));
            }
        }
        if self.passes[outer] {
            s.push_str("  break;\n");
        }
        s.push('}');
        self.line(&s);
    }

    fn jump(&mut self, from: usize, to: usize, ctx: &mut Vec<Ctx>) {
        let copies = self.bl.edge(from, to);
        self.line(&copies);
        let back = self.g.rpo[to] <= self.g.rpo[from];
        if back {
            let count = self.bl.back_edge(to);
            self.line(&count);
            let i = ctx
                .iter()
                .rposition(|c| *c == Ctx::Loop(to))
                .expect("a back edge inside its loop");
            self.leave(i, true, ctx);
        } else if self.g.merge[to] {
            let i = ctx
                .iter()
                .rposition(|c| *c == Ctx::Block(to))
                .expect("a merge after its block");
            self.leave(i, false, ctx);
        } else {
            self.tree(to, ctx);
        }
    }

    /// Continues loop `i` or exits block/loop `i` of the stack.
    fn leave(&mut self, i: usize, cont: bool, ctx: &[Ctx]) {
        let inner = ctx.len() - 1;
        if i == inner {
            self.line(if cont { "continue;" } else { "break;" });
            return;
        }
        // Constructs i+1..=inner pass the jump on.
        for k in i + 1..=inner {
            self.passes[k] = true;
        }
        self.uses_jmp = true;
        self.line(&format!("jmp = {}u; break;", code(i, cont)));
    }
}

/// The function body of `bl`'s graph; whether it declares `jmp` is up to the
/// caller (always: `uint jmp = 0u;`).
pub fn structure(bl: &dyn Blocks) -> Result<String, &'static str> {
    let g = Graph::new(bl)?;
    let _ = (&g.succ, &g.order, &g.idom);
    let mut out = Out {
        g,
        bl,
        text: String::new(),
        depth: 1,
        passes: Vec::new(),
        uses_jmp: false,
    };
    let mut ctx = Vec::new();
    out.tree(0, &mut ctx);
    let mut text = String::new();
    if out.uses_jmp {
        text.push_str("  uint jmp = 0u;\n");
    }
    text.push_str(&out.text);
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct G(Vec<End>);
    impl Blocks for G {
        fn count(&self) -> usize {
            self.0.len()
        }
        fn body(&self, _b: usize) -> &str {
            ""
        }
        fn end(&self, b: usize) -> &End {
            &self.0[b]
        }
        fn edge(&self, from: usize, to: usize) -> String {
            format!("// {from} -> {to}")
        }
        fn back_edge(&self, h: usize) -> String {
            format!("// back {h}")
        }
        fn loop_entry(&self, _h: usize) -> String {
            String::new()
        }
    }

    #[test]
    fn diamond_and_loop_with_break() {
        // 0 -> 1 | 2 -> 3 ; 3 loops to 3 or exits to 4; 4 -> end.
        let g = G(vec![
            End::Branch("a".into(), 1, 2),
            End::Jump(3),
            End::Jump(3),
            End::Branch("b".into(), 3, 4),
            End::Leave("return;".into()),
        ]);
        let s = structure(&g).unwrap();
        assert!(s.contains("for (;;)"), "{s}");
        assert!(s.contains("continue;"), "{s}");
        assert!(s.contains("do {"), "{s}");
    }

    #[test]
    fn a_break_out_of_two_loops_uses_the_jump_variable() {
        // 0 -> 1 (outer header) -> 2 (inner header) -> 2 | 3 | 5 ; 3 -> 1 | 4 ; 5 -> 4.
        let g = G(vec![
            End::Jump(1),
            End::Jump(2),
            End::Branch("x".into(), 2, 6),
            End::Branch("y".into(), 1, 4),
            End::Leave("return;".into()),
            End::Jump(4),
            End::Branch("z".into(), 3, 5),
        ]);
        let s = structure(&g).unwrap();
        assert!(s.contains("jmp"), "{s}");
    }
}
