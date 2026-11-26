//! The thread IR: one console lane's program as a typed SSA function.
//!
//! Built from the wave SSA (`crate::ir`) by [`super::lift`]. The wave's
//! block graph is kept as it is (scalar branches are uniform branches); EXEC
//! is a per-invocation boolean, and the vector instructions that run under
//! one EXEC value are a [`Item::Guard`]: `if (active) { ... }`, with a phi
//! per register they write at its end (lanes outside EXEC keep the old
//! value, exactly as the hardware does). Lane masks are per-invocation
//! booleans; every invocation of a subgroup runs every block, so a ballot of
//! such a boolean is the console's mask.
//!
//! Pure operations reuse the SPIR-V core and GLSL.std.450 vocabularies
//! (opcode numbers, [`Op::Spv`] / [`Op::Glsl`]): that is a complete,
//! documented set of typed scalar operations, and it keeps this IR small.
//! Memory, images, interface and wave operations are this IR's own.

use std::collections::HashMap;

/// A node (value) index.
pub type V = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum S {
    Void,
    Bool,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ty {
    pub s: S,
    pub n: u8,
}

impl Ty {
    pub const VOID: Ty = Ty { s: S::Void, n: 1 };
    pub const BOOL: Ty = Ty { s: S::Bool, n: 1 };
    pub const U16: Ty = Ty { s: S::U16, n: 1 };
    pub const U32: Ty = Ty { s: S::U32, n: 1 };
    pub const U64: Ty = Ty { s: S::U64, n: 1 };
    pub const F16: Ty = Ty { s: S::F16, n: 1 };
    pub const F32: Ty = Ty { s: S::F32, n: 1 };
    pub const U8: Ty = Ty { s: S::U8, n: 1 };
    pub fn vec(s: S, n: u8) -> Ty {
        Ty { s, n }
    }
    pub fn elem(self) -> Ty {
        Ty { s: self.s, n: 1 }
    }
}

/// Built-in inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Builtin {
    VertexIndex,
    InstanceIndex,
    BaseInstance,
    FragCoord(u8),
    FrontFacing,
    SampleId,
    SampleMaskIn,
    Helper,
    WorkgroupId(u8),
    LocalInvocationId(u8),
    LocalInvocationIndex,
    SubgroupId,
    SubgroupLocalId,
    PrimitiveId,
    /// BaryCoordKHR / BaryCoordNoPerspKHR (component), centroid, sample.
    Bary { perspective: bool, centroid: bool, sample: bool, comp: u8 },
}

/// Interpolation of a pixel input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interp {
    Flat,
    /// perspective (false: linear), centroid, sample
    Smooth { perspective: bool, centroid: bool, sample: bool },
}

/// What a helper function the backend writes once does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Helper {
    /// (title address u64) -> device address u64 of `zero` when unmapped.
    Translate,
    /// (title base u64, bytes u32) -> device base u64, or the zero block
    /// when the bytes are not inside one range.
    TranslateRange,
    /// (device address u64, shape u32) -> uvec4: a typed buffer element
    /// decoded (floats as their bits; missing channels 0, alpha 1).
    Typed,
    /// (V# FORMAT u32) -> the shape `Typed` decodes (a table lookup).
    TypedShape,
}

/// Image access kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageKind {
    /// Sample; `lod`: explicit level (`_l`, `_lz`), `bias`, `grad`
    /// (derivatives), `dref` (comparison), `offset`, `min_lod` (`_cl`).
    Sample { lod: bool, bias: bool, grad: bool, dref: bool, offset: bool, min_lod: bool },
    /// Gather of channel `comp` (or comparison).
    Gather { comp: u8, lod: bool, dref: bool, offset: bool },
    /// Load (`image_load`, `image_load_mip`): fetch or storage read by the
    /// T#'s flags.
    Load { mip: bool },
    Store { mip: bool },
    Atomic(u16),
    /// Size of a level (args: lod).
    Size,
    Levels,
    /// `image_get_lod`.
    QueryLod,
}

/// Which sampler an image access uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sampler {
    None,
    /// Immediate sampler n (set 1).
    Static(u32),
    /// From the S# (an arg).
    Dynamic,
}

/// Optional image operands, in argument order.
pub mod img {
    pub const DREF: u16 = 1;
    pub const BIAS: u16 = 2;
    pub const LOD: u16 = 4;
    pub const GRAD: u16 = 8;
    pub const OFFSET: u16 = 16;
    pub const MIN_LOD: u16 = 32;
    pub const SAMPLE: u16 = 64;
    pub const DATA: u16 = 128;
    pub const CMP: u16 = 256;
}

/// The image type a T# is read as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageClass {
    /// SPIR-V Dim: 0 1D, 1 2D, 2 3D, 3 cube.
    pub dim: u8,
    pub arrayed: bool,
    pub ms: bool,
    /// Sampled type unsigned integer (else float).
    pub uint: bool,
    /// Storage image (binding 1), else sampled (binding 0).
    pub storage: bool,
    /// Sampled type and binding chosen at run time from the T#'s word 7
    /// (INTEGER, STORAGE bits).
    pub dynamic: bool,
    /// A cube read as its six-layer 2D array (set 0 binding 2): for
    /// derivatives and offsets, which apply per face.
    pub cube_layers: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Const(u64),
    Undef,
    /// Push-constant dword.
    Push(u32),
    /// A u64 field of the memory map (0 zero block, 1 sink block).
    MapField(u32),
    Builtin(Builtin),
    /// Pixel input at a location, component.
    Input { loc: u8, comp: u8, interp: Interp },
    /// Pixel input read per vertex (`PerVertexKHR`): row, component, vertex.
    PerVertex { loc: u8, comp: u8, vertex: u8 },
    /// A pure SPIR-V core instruction; operands are the args, then `imm`
    /// literals (Extract indices) in [`Node::imm`].
    Spv(u16),
    /// A pure GLSL.std.450 instruction.
    Glsl(u32),
    Call(Helper),
    /// Load of `ty` from a device address (args: address u64).
    Load { align: u32 },
    /// Store (args: address u64, value).
    Store { align: u32 },
    /// Atomic with SPIR-V opcode (args: address, value[, comparator]).
    Atomic(u16),
    /// Workgroup memory: dword index args.
    LdsLoad,
    LdsStore,
    LdsAtomic(u16),
    /// Image access. args: T# word 7, [S# word 3 unless `sampler` is a
    /// static one], coordinates, then one arg per bit of `flags` in bit
    /// order ([`img`]).
    Image { kind: ImageKind, class: ImageClass, sampler: Sampler, flags: u16 },
    /// Subgroup operation with SPIR-V opcode and group operation
    /// (u32::MAX when none).
    Wave(u16, u32),
    /// Output store: location (or built-in code | 0x100), component.
    Output { slot: u16, comp: u8 },
    Demote,
    /// `s_barrier`.
    Barrier,
    /// The two halves of a console wave64 (a group of one wave) exchange a
    /// word uniform in each half: uvec2 (lower half's, upper half's).
    Xchg,
    /// A block or guard phi: args in predecessor order (guards: inner, outer).
    Phi,
    /// Dynamic index into a function-local array of the args (v_movrel).
    Pick,
}

impl Op {
    /// Free of side effects: CSE and DCE may treat it as a value.
    /// Removable when its result is unused.
    pub fn removable(&self) -> bool {
        self.pure()
            || matches!(self, Op::Load { .. } | Op::LdsLoad | Op::Phi)
            || matches!(self, Op::Image { kind, .. } if !matches!(kind, ImageKind::Store { .. } | ImageKind::Atomic(_)))
    }

    pub fn pure(&self) -> bool {
        matches!(
            self,
            Op::Const(_)
                | Op::Undef
                | Op::Push(_)
                | Op::MapField(_)
                | Op::Builtin(_)
                | Op::Input { .. }
                | Op::PerVertex { .. }
                | Op::Spv(_)
                | Op::Glsl(_)
                | Op::Call(_)
                | Op::Wave(..)
                | Op::Pick
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    pub op: Op,
    pub ty: Ty,
    pub args: Vec<V>,
    pub imm: Vec<u32>,
}

#[derive(Debug, Clone)]
pub enum Item {
    N(V),
    Guard { cond: V, body: Vec<Item>, phis: Vec<V> },
}

#[derive(Debug, Clone)]
pub enum Term {
    Jump(usize),
    Branch(V, usize, usize),
    Return,
}

#[derive(Debug, Clone)]
pub struct Block {
    /// Phi nodes, args in `preds` order.
    pub phis: Vec<V>,
    pub preds: Vec<usize>,
    pub body: Vec<Item>,
    pub term: Term,
}

/// A thread program.
#[derive(Debug, Clone, Default)]
pub struct Func {
    pub nodes: Vec<Node>,
    /// Values every block may use (constants, push constants, inputs),
    /// emitted at the function's start.
    pub prelude: Vec<V>,
    pub blocks: Vec<Block>,
    global: HashMap<Node, V>,
}

impl Func {
    pub fn ty(&self, v: V) -> Ty {
        self.nodes[v as usize].ty
    }

    pub fn node(&self, v: V) -> &Node {
        &self.nodes[v as usize]
    }

    pub fn push(&mut self, node: Node) -> V {
        self.nodes.push(node);
        (self.nodes.len() - 1) as V
    }

    /// A value available everywhere (deduplicated).
    pub fn global(&mut self, op: Op, ty: Ty) -> V {
        let node = Node { op, ty, args: Vec::new(), imm: Vec::new() };
        if let Some(&v) = self.global.get(&node) {
            return v;
        }
        let v = self.push(node.clone());
        self.global.insert(node, v);
        if !matches!(self.nodes[v as usize].op, Op::Const(_) | Op::Undef) {
            self.prelude.push(v);
        }
        v
    }

    pub fn konst(&mut self, ty: Ty, bits: u64) -> V {
        self.global(Op::Const(bits), ty)
    }

    pub fn as_const(&self, v: V) -> Option<u64> {
        match self.nodes[v as usize].op {
            Op::Const(c) => Some(c),
            _ => None,
        }
    }

    /// Instructions per block body (guards included), for statistics.
    pub fn count(&self) -> usize {
        fn items(list: &[Item]) -> usize {
            list.iter()
                .map(|i| match i {
                    Item::N(_) => 1,
                    Item::Guard { body, phis, .. } => 1 + items(body) + phis.len(),
                })
                .sum()
        }
        self.prelude.len() + self.blocks.iter().map(|b| b.phis.len() + items(&b.body) + 1).sum::<usize>()
    }
}

/// Removes nodes whose results nobody uses and that have no effect, and
/// guards left empty. (A guard's condition stays live.)
pub fn dce(f: &mut Func) {
    let n = f.nodes.len();
    let mut live = vec![false; n];
    let mut work: Vec<V> = Vec::new();
    fn roots(f: &Func, list: &[Item], work: &mut Vec<V>) {
        for it in list {
            match it {
                Item::N(v) => {
                    if !f.nodes[*v as usize].op.removable() {
                        work.push(*v);
                    }
                }
                Item::Guard { cond, body, .. } => {
                    work.push(*cond);
                    roots(f, body, work);
                }
            }
        }
    }
    for b in &f.blocks {
        roots(f, &b.body, &mut work);
        if let Term::Branch(c, _, _) = b.term {
            work.push(c);
        }
    }
    while let Some(v) = work.pop() {
        if std::mem::replace(&mut live[v as usize], true) {
            continue;
        }
        work.extend(f.nodes[v as usize].args.iter().copied());
    }
    fn sweep(live: &[bool], list: &mut Vec<Item>) {
        list.retain_mut(|it| match it {
            Item::N(v) => live[*v as usize],
            Item::Guard { body, phis, .. } => {
                sweep(live, body);
                phis.retain(|p| live[*p as usize]);
                !body.is_empty() || !phis.is_empty()
            }
        });
    }
    for b in &mut f.blocks {
        sweep(&live, &mut b.body);
        b.phis.retain(|p| live[*p as usize]);
    }
    f.prelude.retain(|v| live[*v as usize]);
}

/// Prints the function (debugging).
pub fn dump(f: &Func) {
    fn items(f: &Func, list: &[Item], depth: usize) {
        for it in list {
            match it {
                Item::N(v) => {
                    let n = f.node(*v);
                    eprintln!("{:w$}%{v} = {:?} {:?} {:?} {:?}", "", n.op, n.ty, n.args, n.imm, w = depth * 2);
                }
                Item::Guard { cond, body, phis } => {
                    eprintln!("{:w$}if %{cond} {{", "", w = depth * 2);
                    items(f, body, depth + 1);
                    eprintln!("{:w$}}} phis {:?}", "", phis.iter().map(|p| (*p, f.node(*p).args.clone())).collect::<Vec<_>>(), w = depth * 2);
                }
            }
        }
    }
    eprintln!("prelude {:?}", f.prelude);
    for (i, b) in f.blocks.iter().enumerate() {
        eprintln!("block {i}: preds {:?}", b.preds);
        for p in &b.phis {
            eprintln!("  %{p} = phi {:?} {:?}", f.node(*p).ty, f.node(*p).args);
        }
        items(f, &b.body, 1);
        eprintln!("  -> {:?}", b.term);
    }
}

fn bits(t: Ty) -> u32 {
    let e = match t.s {
        S::Bool => 1,
        S::U8 => 8,
        S::U16 | S::F16 => 16,
        S::U32 | S::F32 => 32,
        S::U64 => 64,
        S::Void => 0,
    };
    e * u32::from(t.n)
}

/// Structural checks of what the backend will write (types of composite
/// operations, bitcasts, selects, arithmetic, phis): a module that fails
/// them is refused rather than handed to a driver (NVIDIA crashed on an
/// extract from a scalar).
pub fn check(f: &Func) -> Result<(), &'static str> {
    use super::spirv::op;
    let ty = |v: V| f.nodes.get(v as usize).map(|n| n.ty);
    for n in &f.nodes {
        if n.args.iter().any(|&a| a as usize >= f.nodes.len()) {
            return Err("an operand that is no value (a translator bug)");
        }
        // constants are scalars (the backend writes no composite constants)
        if matches!(n.op, Op::Const(_)) && n.ty.n != 1 {
            return Err("a composite constant (a translator bug)");
        }
        let Op::Spv(o) = n.op else { continue };
        let a: Vec<Ty> = n.args.iter().map(|&x| ty(x).unwrap()).collect();
        let ok = match o {
            op::COMPOSITE_EXTRACT => {
                let s = a[0];
                (s.n > 1 || s.s == S::Void) && n.imm.first().is_some_and(|&i| i < u32::from(s.n)) && n.ty.n == 1
            }
            op::COMPOSITE_CONSTRUCT => {
                a.iter().all(|t| t.s == n.ty.s) && a.iter().map(|t| u32::from(t.n)).sum::<u32>() == u32::from(n.ty.n)
            }
            op::BITCAST => bits(a[0]) == bits(n.ty) && a[0].s != S::Bool && n.ty.s != S::Bool,
            op::SELECT => a[0].s == S::Bool && a[1] == n.ty && a[2] == n.ty,
            op::I_ADD | op::I_SUB | op::I_MUL | op::F_ADD | op::F_SUB | op::F_MUL | op::F_DIV | op::BITWISE_AND
            | op::BITWISE_OR | op::BITWISE_XOR => a.iter().all(|&t| t == n.ty),
            op::LOGICAL_AND | op::LOGICAL_OR | op::LOGICAL_EQUAL | op::LOGICAL_NOT_EQUAL | op::LOGICAL_NOT => {
                n.ty == Ty::BOOL && a.iter().all(|&t| t == Ty::BOOL)
            }
            _ => true,
        };
        if !ok {
            return Err("an inconsistent module (a translator bug)");
        }
    }
    for b in &f.blocks {
        for &p in &b.phis {
            let n = &f.nodes[p as usize];
            if n.args.len() != b.preds.len() || n.args.iter().any(|&x| ty(x) != Some(n.ty)) {
                return Err("an inconsistent phi (a translator bug)");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::spirv::op;

    #[test]
    fn the_check_refuses_an_extract_from_a_scalar_and_composite_constants() {
        let mut f = Func::default();
        let _ = f.konst(Ty::vec(S::F16, 2), 0x4000_4000);
        f.blocks.push(Block { phis: vec![], preds: vec![], body: vec![], term: Term::Return });
        assert!(check(&f).is_err());
        let mut f = Func::default();
        let s = f.konst(Ty::F16, 0x4000);
        let x = f.push(Node { op: Op::Spv(op::COMPOSITE_EXTRACT), ty: Ty::F16, args: vec![s], imm: vec![0] });
        f.blocks.push(Block { phis: vec![], preds: vec![], body: vec![Item::N(x)], term: Term::Return });
        assert!(check(&f).is_err());
    }
}

/// Branch facts: below the edge `a -> t` that is `t`'s only way in, the
/// condition `a` branched on is known, and a later branch on the same value
/// (or its negation) goes one way. The console's uniform state machines
/// test the same scalar conditions over and over.
pub fn fold_branches(f: &mut Func) {
    for _round in 0..4 {
        let preds = term_preds(f);
        let order = term_rpo(f);
        let mut facts: Vec<Option<std::rc::Rc<Vec<(V, bool)>>>> = vec![None; f.blocks.len()];
        let mut changed = false;
        for &b in &order {
            let mine = facts[b].clone().unwrap_or_default();
            changed |= fold_term(f, b, &mine);
            for (s, fact) in edge_facts(f, b) {
                if preds[s].len() == 1 && facts[s].is_none() {
                    let mut fs = (*mine).clone();
                    fs.extend(fact);
                    facts[s] = Some(std::rc::Rc::new(fs));
                }
            }
        }
        if !changed {
            break;
        }
    }
}

pub fn term_succ(b: &Block) -> Vec<usize> {
    match b.term {
        Term::Jump(t) => vec![t],
        Term::Branch(_, t, e) => vec![t, e],
        Term::Return => vec![],
    }
}

/// Predecessors as the terminators now say (block preds stay as the phis
/// index them).
fn term_preds(f: &Func) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); f.blocks.len()];
    for (b, blk) in f.blocks.iter().enumerate() {
        for s in term_succ(blk) {
            preds[s].push(b);
        }
    }
    preds
}

/// Reverse postorder from the entry by the terminators.
pub fn term_rpo(f: &Func) -> Vec<usize> {
    let mut order = Vec::new();
    let mut mark = vec![false; f.blocks.len()];
    let mut st = vec![(0usize, 0usize)];
    mark[0] = true;
    while let Some(&mut (b, ref mut k)) = st.last_mut() {
        let ss = term_succ(&f.blocks[b]);
        if *k < ss.len() {
            let x = ss[*k];
            *k += 1;
            if !std::mem::replace(&mut mark[x], true) {
                st.push((x, 0));
            }
        } else {
            order.push(b);
            st.pop();
        }
    }
    order.reverse();
    order
}

/// A condition without its negations: (value, negated).
fn strip_not(f: &Func, c: V) -> (V, bool) {
    use super::spirv::op;
    let mut c = c;
    let mut neg = false;
    while let Op::Spv(op::LOGICAL_NOT) = f.nodes[c as usize].op {
        c = f.nodes[c as usize].args[0];
        neg = !neg;
    }
    (c, neg)
}

/// Folds `b`'s branch if its condition is known; whether it did.
fn fold_term(f: &mut Func, b: usize, facts: &[(V, bool)]) -> bool {
    let Term::Branch(c, t, e) = f.blocks[b].term else { return false };
    let (c0, neg) = strip_not(f, c);
    let known = match f.nodes[c0 as usize].op {
        Op::Const(k) => Some(k != 0),
        _ => facts.iter().rev().find(|x| x.0 == c0).map(|x| x.1),
    };
    let to = match known {
        _ if t == e => t,
        Some(v) => if v != neg { t } else { e },
        None => return false,
    };
    f.blocks[b].term = Term::Jump(to);
    true
}

/// The successors of `b` with the fact each edge adds.
fn edge_facts(f: &Func, b: usize) -> Vec<(usize, Option<(V, bool)>)> {
    match f.blocks[b].term {
        Term::Jump(t) => vec![(t, None)],
        Term::Branch(c, t, e) => {
            let (c0, neg) = strip_not(f, c);
            vec![(t, Some((c0, !neg))), (e, Some((c0, neg)))]
        }
        Term::Return => vec![],
    }
}
