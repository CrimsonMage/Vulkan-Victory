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
    /// (title base u64, bytes u32) -> vec2: device base (as two u32), byte
    /// limit (0 when not fully inside one range).
    TranslateRange,
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
    /// A block or guard phi: args in predecessor order (guards: inner, outer).
    Phi,
    /// Load of a block-phi variable / store to it (lowering of block phis).
    VarLoad(u32),
    VarStore(u32),
    /// Dynamic index into a function-local array of the args (v_movrel).
    Pick,
}

impl Op {
    /// Free of side effects: CSE and DCE may treat it as a value.
    /// Removable when its result is unused.
    pub fn removable(&self) -> bool {
        self.pure()
            || matches!(self, Op::Load { .. } | Op::LdsLoad | Op::VarLoad(_) | Op::Phi)
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
    /// Block-phi variables: their types.
    pub vars: Vec<Ty>,
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

    pub fn new_var(&mut self, ty: Ty) -> u32 {
        self.vars.push(ty);
        (self.vars.len() - 1) as u32
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
        self.prelude.len() + self.blocks.iter().map(|b| items(&b.body) + 1).sum::<usize>()
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
        eprintln!("block {i}:");
        items(f, &b.body, 1);
        eprintln!("  -> {:?}", b.term);
    }
}
