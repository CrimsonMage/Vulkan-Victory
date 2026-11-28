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
    F64,
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
    pub const F64: Ty = Ty { s: S::F64, n: 1 };
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
    /// (title address u64) -> u64: the bytes from it to the end of its
    /// memory map range (0 in none).
    RangeLeft,
    /// (device address u64, shape u32) -> uvec4: a typed buffer element
    /// decoded (floats as their bits; missing channels 0, alpha 1).
    Typed,
    /// (V# FORMAT u32) -> the shape `Typed` decodes (a table lookup).
    TypedShape,
    /// (shape u32, the channel registers uvec4, d16 u32) -> uvec4: a typed
    /// buffer element encoded (its dwords; emit_typed.rs).
    TypedPack,
    /// (node device address u64, kind u32, sort u32, extent, origin xyz,
    /// direction xyz, inverse direction xyz) -> uvec4: the ray/node test
    /// of `image_bvh*_intersect_ray` (emit_bvh.rs).
    BvhRay,
    /// (dword index u32) -> u32: dword `index` of table k of
    /// `Func::tables`, 0 past its end: a constant table in the program's
    /// own code, read through `s_getpc_b64` (lift_mem.rs `code_table`).
    CodeTable(u16),
    /// (addend u32, a u32, b u32) -> u32: the addend plus the products of
    /// the four signed bytes of `a` and `b`, modulo 2^32
    /// (`v_dot4c_i32_i8`, `v_dot4_i32_i8`; lift_pk.rs). A function, so
    /// that a program of thousands of them (a network pass of the
    /// FidelityFX upscaler) is a module of a size drivers compile.
    Dot4,
}

/// The most dwords of code a constant table read through `s_getpc_b64`
/// takes (from its start on).
const CODE_TABLE_DWORDS: usize = 4096;

/// One constant table of a program's code.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CodeTable {
    /// Where it starts in the code (a byte offset).
    pub start: u32,
    /// Its first dword in [`CodeTables::words`].
    pub first: u32,
    /// Its dwords (0 until laid out, and when nothing reads it).
    pub dwords: u32,
}

/// The constant tables of a program's code (`Helper::CodeTable`): memory
/// the host keeps for the program, read by address
/// (`docs/vulkan-abi.md` section 3.1). As variables of the module, NVIDIA
/// kept them on the card for 32,768 threads (one program's nine tables:
/// 1.7 GiB).
#[derive(Debug, Clone, Default)]
pub struct CodeTables {
    pub tables: Vec<CodeTable>,
    /// The tables' bytes, as little-endian dwords.
    pub words: Vec<u32>,
}

impl CodeTables {
    /// The table that starts at code byte `start` (made once a start).
    pub fn at(&mut self, start: u32) -> u16 {
        let k = self.tables.iter().position(|t| t.start == start).unwrap_or_else(|| {
            self.tables.push(CodeTable { start, ..CodeTable::default() });
            self.tables.len() - 1
        });
        k as u16
    }

    /// Lays the tables something reads (`used`, by table) out in `words`:
    /// each is the code from its start to the code's end,
    /// [`CODE_TABLE_DWORDS`] at most; tables that overlap share their
    /// bytes (one run of code) where they start whole dwords apart.
    pub fn lay_out(&mut self, code: &[u8], used: &[bool]) {
        let mut order: Vec<usize> = (0..self.tables.len()).filter(|&k| used.get(k) == Some(&true)).collect();
        order.sort_by_key(|&k| self.tables[k].start);
        let mut bytes: Vec<u8> = Vec::new();
        // the runs of code in `bytes`: (first code byte, end, its place)
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for k in order {
            let from = (self.tables[k].start as usize).min(code.len());
            let to = code.len().min(from + 4 * CODE_TABLE_DWORDS);
            // in a run that holds all of it, else in the last run if it
            // begins there (the run grows), else in a new run
            let begins = |&(s, e, _): &(usize, usize, usize)| from >= s && from <= e && (from - s) % 4 == 0;
            let holds = runs.iter().position(|r| begins(r) && to <= r.1);
            let r = holds.or_else(|| runs.len().checked_sub(1).filter(|&l| begins(&runs[l]))).unwrap_or_else(|| {
                bytes.resize(bytes.len().next_multiple_of(4), 0);
                runs.push((from, from, bytes.len()));
                runs.len() - 1
            });
            let (s, e, at) = runs[r];
            if to > e {
                bytes.extend_from_slice(&code[e..to]);
                runs[r].1 = to;
            }
            let at = at + (from - s);
            self.tables[k].first = (at / 4) as u32;
            self.tables[k].dwords = (to - from).div_ceil(4) as u32;
        }
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        self.words = bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    }

    /// Dword `i` of table `k`, 0 past its end: what the module reads from
    /// the host's copy of `words`.
    pub fn dword(&self, k: u16, i: u32) -> u32 {
        let t = self.tables[usize::from(k)];
        if i < t.dwords { self.words[(t.first + i) as usize] } else { 0 }
    }
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
    /// Constant tables of the program's code (`Helper::CodeTable`).
    pub tables: CodeTables,
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

/// Which nodes the function holds (in a block's phis or body, or the
/// prelude), by node: after [`dce`], the ones something still reads or
/// that have an effect.
pub fn placed(f: &Func) -> Vec<bool> {
    fn items(list: &[Item], out: &mut [bool]) {
        for it in list {
            match it {
                Item::N(v) => out[*v as usize] = true,
                Item::Guard { body, phis, .. } => {
                    items(body, out);
                    phis.iter().for_each(|&p| out[p as usize] = true);
                }
            }
        }
    }
    let mut out = vec![false; f.nodes.len()];
    for b in &f.blocks {
        b.phis.iter().for_each(|&p| out[p as usize] = true);
        items(&b.body, &mut out);
    }
    f.prelude.iter().for_each(|&v| out[v as usize] = true);
    out
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
        S::U64 | S::F64 => 64,
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

    /// Every table reads the code from its start on (0 past the end, a
    /// last dword short of bytes padded with zeros), whatever shares bytes.
    #[test]
    fn code_tables_laid_out_read_the_code_from_their_starts() {
        let code: Vec<u8> = (0..4 * CODE_TABLE_DWORDS as u32 + 4096 + 3).map(|i| (i * 7 + i / 251) as u8).collect();
        let reference = |start: usize, i: usize| -> u32 {
            let at = start + 4 * i;
            if i >= CODE_TABLE_DWORDS || at >= code.len() {
                return 0;
            }
            (0..4).rev().fold(0, |w, b| w << 8 | u32::from(code.get(at + b).copied().unwrap_or(0)))
        };
        // overlapping whole dwords apart, one a byte off, one capped at
        // the most dwords, one nothing reads, one past the code's end
        let starts = [4096u32, 16384, 4104, 16385, 8, 12000, code.len() as u32 + 40, 18000];
        let mut t = CodeTables::default();
        for &s in &starts {
            t.at(s);
        }
        assert_eq!(t.at(4104), 2);
        let mut used = vec![true; starts.len()];
        used[5] = false;
        t.lay_out(&code, &used);
        for (k, &s) in starts.iter().enumerate() {
            for i in (0..CODE_TABLE_DWORDS + 8).step_by(if k == 0 { 1 } else { 37 }).chain([CODE_TABLE_DWORDS - 1, CODE_TABLE_DWORDS]) {
                let want = if used[k] { reference(s as usize, i) } else { 0 };
                assert_eq!(t.dword(k as u16, i as u32), want, "table {k} dword {i}");
            }
        }
        assert_eq!(t.dword(0, u32::MAX), 0);
        // one run from 8 to the code's end (4096, 4104, 16384 and 18000
        // begin in it), and 16385 apart
        let shared = (code.len() - 8).div_ceil(4);
        let apart = (code.len() - 16385).div_ceil(4);
        assert_eq!(t.words.len(), shared + apart);
        assert_eq!(t.tables[2].first, t.tables[0].first + 2);
        assert_eq!(t.tables[5].dwords, 0);
        assert_eq!(t.tables[6].dwords, 0);
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
