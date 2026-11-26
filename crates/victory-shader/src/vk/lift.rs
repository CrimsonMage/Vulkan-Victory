//! Wave SSA (`crate::ir`) to the thread IR ([`super::tir`]).
//!
//! Every invocation of a host subgroup walks every block of the wave's
//! graph, as the console wave does: scalar branches are uniform. EXEC is the
//! boolean `active` of each invocation; `s_cbranch_execz`/`execnz` test
//! `subgroupAny(active)`. Vector instructions with effects on lanes run in
//! a guard `if (active) { ... }` per run of instructions under one EXEC
//! value, and the VGPRs they write meet the old values in phis at its end.
//! Instructions that write SGPRs (compares, `v_readfirstlane`...) and
//! cross-lane ones run unguarded with EXEC applied explicitly.
//!
//! A register dword's value is data (`u32`/`f32`), a lane mask (a boolean
//! per invocation: the invocation's own bit) or SCC (a boolean). Masks turn
//! into numbers by a ballot where the code reads them so.

use std::collections::{HashMap, HashSet};

use super::tir::{self, Builtin, Func, Helper, ImageClass, ImageKind, Interp, Item, Node, Op, S, Term, Ty, V};
use super::spirv::{glsl, op};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, Operand, Valu};
use crate::io::{self, Input, Target};
use crate::ir::{self, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, RootType, StageIo};

/// Which half of a console wave64 mask a register dword holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Half {
    Lo,
    Hi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum R {
    /// u32 or f32 data (or f16 halves in a u32).
    D(V),
    /// A lane mask: this invocation's bit.
    M(V, Half),
    /// SCC.
    B(V),
}

/// What the stage's interface needs, found while lifting.
#[derive(Debug, Clone, Default)]
pub struct Interface {
    /// Pixel inputs: (location, interpolation).
    pub inputs: Vec<(u8, Interp)>,
    /// Outputs written: location (or built-in | 0x100) and components.
    pub outputs: Vec<(u16, u8)>,
    /// Immediate samplers (four constant S# words).
    pub samplers: Vec<[u32; 4]>,
    /// Workgroup memory in dwords.
    pub lds_dwords: u32,
    pub approximate: Vec<&'static str>,
    /// Pixel: depth / stencil / sample mask exported.
    pub depth: bool,
    pub demotes: bool,
    /// Reads barycentrics or per-vertex values (fragmentShaderBarycentric).
    pub per_vertex: bool,
}

/// How each value is read: above 0 mostly as a float (so loads and phis
/// make it one and the bitcasts go), below as an integer.
fn float_uses(p: &Program) -> HashMap<Value, i32> {
    let mut score: HashMap<Value, i32> = HashMap::new();
    let mut moves: Vec<(Value, Value)> = Vec::new();
    for b in &p.blocks {
        for phi in &b.phis {
            moves.extend(phi.args.iter().map(|&a| (phi.value, a)));
        }
        for inst in &b.insts {
            score_inst(p, inst, &mut score, &mut moves);
        }
    }
    // a copy's source is read the way its destination is
    for _ in 0..3 {
        for &(d, s) in moves.iter().rev() {
            let x = score.get(&d).copied().unwrap_or(0);
            if x != 0 {
                *score.entry(s).or_default() += x.signum();
            }
        }
    }
    score
}

/// One instruction's reads: +1 per float read, -1 per integer read; moves
/// and selects are copies (scored by their destinations).
fn score_inst(p: &Program, inst: &Inst, score: &mut HashMap<Value, i32>, moves: &mut Vec<(Value, Value)>) {
    let name = inst.name();
    let mut regs = ir::operand_reads(p, inst);
    if name.contains("mac_") {
        regs.extend(inst.writes.iter().map(|w| w.0).filter(|r| matches!(r, Reg::V(_))));
    }
    let reads = regs.into_iter().filter(|r| matches!(r, Reg::V(_) | Reg::S(_))).filter_map(|r| inst.read(r));
    let w = match &inst.fields {
        Fields::Valu(_) if matches!(name, "v_mov_b32" | "v_cndmask_b32") => {
            if let Some(&(_, d)) = inst.writes.iter().find(|w| matches!(w.0, Reg::V(_))) {
                moves.extend(reads.map(|v| (d, v)));
            }
            return;
        }
        Fields::Valu(_) => valu_read_kind(name),
        Fields::Image { addr, .. } if name.starts_with("image_sample") || name.starts_with("image_gather") => {
            // sample coordinates are floats
            for a in addr {
                let Operand::Vector { reg, dwords } = *a else { continue };
                for v in (0..dwords).filter_map(|k| inst.read(Reg::V(reg + k))) {
                    *score.entry(v).or_default() += 1;
                }
            }
            return;
        }
        _ => 0,
    };
    if w != 0 {
        for v in reads {
            *score.entry(v).or_default() += w;
        }
    }
}

/// A VALU instruction's source type by its name: 1 float, -1 integer.
fn valu_read_kind(name: &str) -> i32 {
    let src = match name.strip_prefix("v_cvt_") {
        Some(rest) => rest.rsplit('_').next().unwrap_or(""),
        None => name.rsplit('_').next().unwrap_or(""),
    };
    match src {
        "f32" => 1,
        "u32" | "i32" | "b32" | "u24" | "i24" | "u64" | "i64" | "b64" | "ubyte0" | "ubyte1" | "ubyte2" | "ubyte3" => -1,
        _ => 0,
    }
}

/// The blocks in dominator-tree preorder, and the immediate dominators.
fn dom_preorder(p: &Program) -> (Vec<usize>, Vec<usize>) {
    let order = ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; p.blocks.len()];
    for (i, &b) in order.iter().enumerate() {
        index[b] = i;
    }
    let idom = ir::idoms(&p.blocks, &order, &index);
    let mut children = vec![Vec::new(); p.blocks.len()];
    for &b in order.iter().skip(1) {
        if idom[b] != usize::MAX {
            children[idom[b]].push(b);
        }
    }
    let mut pre = Vec::new();
    let mut stack = vec![order[0]];
    while let Some(b) = stack.pop() {
        pre.push(b);
        stack.extend(children[b].iter().rev());
    }
    (pre, idom)
}

/// A 32-bit operation of two constants.
fn fold_u32_op(o: u16, x: u32, y: u32) -> Option<u32> {
    Some(match o {
        op::I_ADD => x.wrapping_add(y),
        op::I_SUB => x.wrapping_sub(y),
        op::I_MUL => x.wrapping_mul(y),
        op::BITWISE_AND => x & y,
        op::BITWISE_OR => x | y,
        op::BITWISE_XOR => x ^ y,
        op::SHIFT_LEFT_LOGICAL if y < 32 => x << y,
        op::SHIFT_RIGHT_LOGICAL if y < 32 => x >> y,
        op::SHIFT_RIGHT_ARITHMETIC if y < 32 => ((x as i32) >> y) as u32,
        _ => return None,
    })
}

/// Built-in output slots.
pub const OUT_POSITION: u16 = 0x100;
pub const OUT_DEPTH: u16 = 0x101;
pub const OUT_STENCIL: u16 = 0x102;
pub const OUT_SAMPLE_MASK: u16 = 0x103;
/// Mesh: the primitive export (three 10-bit indices, bit 31 culls).
pub const OUT_PRIMITIVE: u16 = 0x104;

/// The old value of a lane mask written in a guard: 0 outside it.
const MASK_FALSE: Value = Value(u32::MAX - 1);

struct Open {
    cond: V,
    items: Vec<Item>,
    /// (register, new ir value, old ir value) of the last write per register.
    defs: Vec<(Reg, Value, Value)>,
    defined: HashSet<Value>,
    cse: HashMap<Node, V>,
    /// Thread-IR values made inside.
    inner: HashSet<V>,
}

pub struct Lift<'a> {
    p: &'a Program,
    native: &'a NativeProgram,
    io: &'a io::Io,
    pub f: Func,
    pub iface: Interface,
    vals: HashMap<Value, R>,
    cur: Vec<Item>,
    cse: HashMap<Node, V>,
    open: Option<Open>,
    guarded: bool,
    /// Unguarded VALU writes meet the old values by this EXEC.
    sel: Option<V>,
    /// Float or integer use of each value ([`float_uses`]).
    floaty: HashMap<Value, i32>,
    /// Bumped by every effect and block: loads share within one epoch.
    epoch: u32,
    /// Undo log of the CSE table (dominator-tree scopes).
    undo: Vec<(Node, Option<V>)>,
    no_cse: bool,
    /// Phi nodes per block (made when the block is lifted; a back edge's
    /// target is lifted before its source).
    all_phis: Vec<Vec<V>>,
    made: Vec<bool>,
    /// Blocks lifted so far.
    done: Vec<Option<tir::Block>>,
    interps: HashMap<(usize, usize), &'a io::Interp>,
    site: (usize, usize),
    wave64: bool,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// The current image address as 16-bit components (a16).
    a16: bool,
    a16_vals: Vec<V>,
    /// 16-bit interface rows: (row, half) -> (row, column) ([`super::pack16`]).
    pub halves: HashMap<(u8, u8), (u8, u8)>,
    /// Mesh: the row of the parameter entry the compiler left unnamed.
    pub unnamed_row: Option<u8>,
    /// Vertex inputs: whether each input-signature row is a float.
    pub vs_float: HashMap<u8, bool>,
    /// Pixel input rows some site reads per vertex.
    per_vertex: HashSet<u8>,
}

type Res<T> = Result<T, ShaderError>;

fn unsupported(offset: usize, what: &'static str) -> ShaderError {
    ShaderError::Unsupported { offset, what }
}

/// The user-data window of a stage in push-constant dwords
/// (`docs/vulkan-abi.md`, section 2).
pub fn push_window(stage: HwStage) -> u32 {
    match stage {
        HwStage::Ps => 32,
        _ => 4,
    }
}

/// Inline constant of operand code `c` as 32-bit data (`f16`: half bits for
/// the float constants).
fn inline_const(c: u8, f16: bool) -> Option<u32> {
    Some(match c {
        128..=192 => u32::from(c) - 128,
        193..=208 => (-(i32::from(c) - 192)) as u32,
        240..=248 if f16 => {
            [0x3800u32, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118][usize::from(c - 240)]
        }
        240..=248 => [
            0.5f32.to_bits(),
            (-0.5f32).to_bits(),
            1.0f32.to_bits(),
            (-1.0f32).to_bits(),
            2.0f32.to_bits(),
            (-2.0f32).to_bits(),
            4.0f32.to_bits(),
            (-4.0f32).to_bits(),
            0x3E22_F983,
        ][usize::from(c - 240)],
        _ => return None,
    })
}

impl<'a> Lift<'a> {
    pub fn new(p: &'a Program, native: &'a NativeProgram, io: &'a io::Io) -> Self {
        let mut l = Lift {
            p,
            native,
            io,
            f: Func::default(),
            iface: Interface::default(),
            vals: HashMap::new(),
            cur: Vec::new(),
            cse: HashMap::new(),
            open: None,
            guarded: false,
            sel: None,
            undo: Vec::new(),
            epoch: 0,
            floaty: float_uses(p),
            no_cse: false,
            all_phis: vec![Vec::new(); p.blocks.len()],
            made: vec![false; p.blocks.len()],
            done: Vec::new(),
            interps: io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            site: (0, 0),
            wave64: p.wave64,
            mesh_stride: 0,
            a16: false,
            a16_vals: Vec::new(),
            halves: HashMap::new(),
            unnamed_row: None,
            vs_float: HashMap::new(),
            per_vertex: io
                .interps
                .iter()
                .filter_map(|(_, i)| match i {
                    io::Interp::Vertex { attribute, vertex: 0 | 1 } | io::Interp::Weighted { attribute } => Some(attribute.row),
                    _ => None,
                })
                .collect(),
        };
        l.iface.lds_dwords = match &native.io {
            StageIo::Cs(cs) => cs.lds_bytes() / 4,
            StageIo::Ms(ms) => ms.lds_bytes / 4,
            _ => 0,
        };
        l
    }

    // ---- emission ----

    fn k32(&mut self, v: u32) -> V {
        self.f.konst(Ty::U32, u64::from(v))
    }
    fn kf(&mut self, v: f32) -> V {
        self.f.konst(Ty::F32, u64::from(v.to_bits()))
    }
    fn kb(&mut self, v: bool) -> V {
        self.f.konst(Ty::BOOL, u64::from(v))
    }

    fn emit(&mut self, op: Op, ty: Ty, args: Vec<V>) -> V {
        self.emit_imm(op, ty, args, Vec::new())
    }

    fn emit_imm(&mut self, op: Op, ty: Ty, args: Vec<V>, imm: Vec<u32>) -> V {
        let node = Node { op, ty, args, imm };
        if let Some(v) = self.fold(&node) {
            return v;
        }
        let pure = node.op.pure();
        // Loads are shared like pure values while nothing in the block has
        // stored since (the key carries the store epoch).
        let load = matches!(node.op, Op::Load { .. } | Op::LdsLoad);
        let key = if load {
            let mut k = node.clone();
            k.imm.push(self.epoch);
            Some(k)
        } else if pure {
            Some(node.clone())
        } else {
            None
        };
        if let Some(k) = &key {
            if self.guarded
                && let Some(o) = &self.open
                && let Some(&v) = o.cse.get(k)
            {
                return v;
            }
            if !self.no_cse
                && let Some(&v) = self.cse.get(k)
            {
                return v;
            }
        }
        if !node.op.removable() {
            self.epoch += 1;
        }
        // Pure values and loads of values from outside the guard go outside
        // it, where the other guards of the block share them.
        let hoist = self.guarded
            && key.is_some()
            && node.args.iter().all(|a| !self.open.as_ref().unwrap().inner.contains(a));
        let v = self.f.push(node);
        if self.guarded && !hoist {
            let o = self.open.as_mut().expect("an open guard");
            o.items.push(Item::N(v));
            o.inner.insert(v);
            if let Some(k) = key {
                o.cse.insert(k, v);
            }
        } else {
            self.cur.push(Item::N(v));
            if let Some(k) = key
                && !self.no_cse
            {
                let old = self.cse.insert(k.clone(), v);
                self.undo.push((k, old));
            }
        }
        v
    }

    fn rollback(&mut self, mark: usize) {
        while self.undo.len() > mark {
            let (node, old) = self.undo.pop().unwrap();
            match old {
                Some(v) => {
                    self.cse.insert(node, v);
                }
                None => {
                    self.cse.remove(&node);
                }
            }
        }
    }

    fn spv(&mut self, opcode: u16, ty: Ty, args: &[V]) -> V {
        self.emit(Op::Spv(opcode), ty, args.to_vec())
    }
    fn gl(&mut self, inst: u32, ty: Ty, args: &[V]) -> V {
        self.emit(Op::Glsl(inst), ty, args.to_vec())
    }
    fn extract(&mut self, v: V, i: u32) -> V {
        let t = self.f.ty(v).elem();
        self.emit_imm(Op::Spv(op::COMPOSITE_EXTRACT), t, vec![v], vec![i])
    }

    /// Constant folding of the common pure forms.
    fn fold(&mut self, n: &Node) -> Option<V> {
        let Op::Spv(o) = n.op else { return None };
        let c: Vec<Option<u64>> = n.args.iter().map(|&a| self.f.as_const(a)).collect();
        let a = c.first().copied().flatten();
        let b = c.get(1).copied().flatten();
        let ty = n.ty;
        let r = match o {
            _ if ty == Ty::U32 && n.args.len() == 2 => self.fold_u32(o, n, a, b),
            op::I_ADD if ty == Ty::U64 && n.args.len() == 2 => self.fold_add64(n, a, b),
            _ if ty == Ty::BOOL => self.fold_bool(o, n, a, b),
            _ => None,
        };
        if r.is_some() {
            return r;
        }
        match o {
            op::U_CONVERT | op::S_CONVERT => a.map(|x| self.fold_convert(o, n, x)),
            op::SELECT if a.is_some() => Some(if a != Some(0) { n.args[1] } else { n.args[2] }),
            op::SELECT if n.args[1] == n.args[2] => Some(n.args[1]),
            op::BITCAST if n.args.len() == 1 => self.fold_bitcast(n),
            _ => None,
        }
    }

    /// 32-bit integer operations of constants, and identities.
    fn fold_u32(&mut self, o: u16, n: &Node, a: Option<u64>, b: Option<u64>) -> Option<V> {
        if let (Some(x), Some(y)) = (a, b) {
            let r = fold_u32_op(o, x as u32, y as u32)?;
            return Some(self.f.konst(Ty::U32, u64::from(r)));
        }
        let (l, r) = (n.args[0], n.args[1]);
        let zero = |s: &mut Self| Some(s.f.konst(Ty::U32, 0));
        match (o, a.map(|x| x as u32), b.map(|x| x as u32)) {
            (op::I_ADD | op::BITWISE_OR | op::BITWISE_XOR, Some(0), _) => Some(r),
            (op::I_ADD | op::BITWISE_OR | op::BITWISE_XOR | op::I_SUB, _, Some(0)) => Some(l),
            (op::SHIFT_LEFT_LOGICAL | op::SHIFT_RIGHT_LOGICAL | op::SHIFT_RIGHT_ARITHMETIC, _, Some(0)) => Some(l),
            (op::BITWISE_AND, Some(0), _) | (op::BITWISE_AND, _, Some(0)) => zero(self),
            (op::BITWISE_AND, Some(u32::MAX), _) => Some(r),
            (op::BITWISE_AND, _, Some(u32::MAX)) => Some(l),
            (op::I_MUL, Some(1), _) => Some(r),
            (op::I_MUL, _, Some(1)) => Some(l),
            (op::I_MUL, Some(0), _) | (op::I_MUL, _, Some(0)) => zero(self),
            _ => None,
        }
    }

    fn fold_add64(&mut self, n: &Node, a: Option<u64>, b: Option<u64>) -> Option<V> {
        match (a, b) {
            (Some(x), Some(y)) => Some(self.f.konst(Ty::U64, x.wrapping_add(y))),
            (_, Some(0)) => Some(n.args[0]),
            (Some(0), _) => Some(n.args[1]),
            _ => None,
        }
    }

    /// Logic and compares of constants; and/or with a constant or twice
    /// the same value; double negation.
    fn fold_bool(&mut self, o: u16, n: &Node, a: Option<u64>, b: Option<u64>) -> Option<V> {
        let same = n.args.len() == 2 && n.args[0] == n.args[1];
        match o {
            op::LOGICAL_AND | op::LOGICAL_OR => {
                // the absorbing constant: 0 for and, 1 for or
                let absorb = u64::from(o == op::LOGICAL_OR);
                match (a, b) {
                    (Some(x), _) | (_, Some(x)) if x == absorb => Some(self.kb(absorb != 0)),
                    (Some(_), _) => Some(n.args[1]),
                    (_, Some(_)) => Some(n.args[0]),
                    _ if same => Some(n.args[0]),
                    _ => None,
                }
            }
            op::LOGICAL_NOT => match (a, &self.f.node(n.args[0]).op) {
                (Some(x), _) => Some(self.kb(x == 0)),
                (_, Op::Spv(op::LOGICAL_NOT)) => Some(self.f.node(n.args[0]).args[0]),
                _ => None,
            },
            op::LOGICAL_NOT_EQUAL => match (a, b) {
                (Some(x), Some(y)) => Some(self.kb(x != y)),
                (Some(0), _) => Some(n.args[1]),
                (_, Some(0)) => Some(n.args[0]),
                _ => None,
            },
            op::I_EQUAL | op::I_NOT_EQUAL | op::U_LESS_THAN | op::U_GREATER_THAN_EQUAL => {
                let (x, y) = (a?, b?);
                let r = match o {
                    op::I_EQUAL => x == y,
                    op::I_NOT_EQUAL => x != y,
                    op::U_LESS_THAN => x < y,
                    _ => x >= y,
                };
                Some(self.kb(r))
            }
            _ => None,
        }
    }

    /// A width conversion of a constant.
    fn fold_convert(&mut self, o: u16, n: &Node, x: u64) -> V {
        let bits = |s: S| match s {
            S::U8 => 8,
            S::U16 => 16,
            S::U32 => 32,
            _ => 64,
        };
        let from = bits(self.f.ty(n.args[0]).s);
        let x = if o == op::S_CONVERT && from < 64 && (x >> (from - 1)) & 1 != 0 { x | (!0u64 << from) } else { x };
        let to = bits(n.ty.s);
        let mask = if to == 64 { u64::MAX } else { (1u64 << to) - 1 };
        self.f.konst(n.ty, x & mask)
    }

    /// A bitcast of a value of the same type, of a scalar constant, or of a
    /// bitcast back.
    fn fold_bitcast(&mut self, n: &Node) -> Option<V> {
        let ty = n.ty;
        let src = self.f.node(n.args[0]).clone();
        if self.f.ty(n.args[0]) == ty {
            return Some(n.args[0]);
        }
        if let Op::Const(bits) = src.op
            && ty.n == 1
            && self.f.ty(n.args[0]).n == 1
        {
            return Some(self.f.konst(ty, bits));
        }
        if src.op == Op::Spv(op::BITCAST) && self.f.ty(src.args[0]) == ty {
            return Some(src.args[0]);
        }
        None
    }

    // ---- conversions ----

    fn u32v(&mut self, v: V) -> V {
        match self.f.ty(v) {
            Ty::U32 => v,
            Ty::F32 => self.spv(op::BITCAST, Ty::U32, &[v]),
            Ty::BOOL => {
                let (one, zero) = (self.k32(1), self.k32(0));
                self.spv(op::SELECT, Ty::U32, &[v, one, zero])
            }
            t => panic!("u32 of {t:?}"),
        }
    }
    fn f32v(&mut self, v: V) -> V {
        match self.f.ty(v) {
            Ty::F32 => v,
            Ty::U32 => self.spv(op::BITCAST, Ty::F32, &[v]),
            t => panic!("f32 of {t:?}"),
        }
    }

    fn lane(&mut self) -> V {
        self.f.global(Op::Builtin(Builtin::SubgroupLocalId), Ty::U32)
    }

    /// The console wave of this invocation: a console wave64 of a group is
    /// two host subgroups.
    fn console_wave(&mut self) -> V {
        let sg = self.f.global(Op::Builtin(Builtin::SubgroupId), Ty::U32);
        if self.wave64 {
            let one = self.k32(1);
            self.global_bin(op::SHIFT_RIGHT_LOGICAL, Ty::U32, sg, one)
        } else {
            sg
        }
    }

    /// The console lane of this invocation (wave64: the odd subgroup of a
    /// pair is the upper half).
    fn console_lane(&mut self) -> V {
        let lane = self.lane();
        if !self.wave64 || !matches!(self.native.stage, HwStage::Cs | HwStage::Ms) {
            return lane;
        }
        let sg = self.f.global(Op::Builtin(Builtin::SubgroupId), Ty::U32);
        let one = self.k32(1);
        let half = self.global_bin(op::BITWISE_AND, Ty::U32, sg, one);
        let k5 = self.k32(5);
        let up = self.global_bin(op::SHIFT_LEFT_LOGICAL, Ty::U32, half, k5);
        self.global_bin(op::BITWISE_OR, Ty::U32, up, lane)
    }

    fn ballot(&mut self, b: V) -> V {
        if let Some(c) = self.f.as_const(b) {
            return self.k32(if c != 0 { u32::MAX } else { 0 });
        }
        let v4 = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_BALLOT, u32::MAX), Ty::vec(S::U32, 4), vec![b]);
        self.extract(v4, 0)
    }
    fn any(&mut self, b: V) -> V {
        if self.f.as_const(b).is_some() {
            return b;
        }
        self.emit(Op::Wave(op::GROUP_NON_UNIFORM_ANY, u32::MAX), Ty::BOOL, vec![b])
    }

    fn data(&mut self, r: R) -> V {
        match r {
            R::D(v) => v,
            R::M(b, Half::Lo) => self.ballot(b),
            R::M(_, Half::Hi) => {
                self.approx("a wave64 mask's high half read as a number");
                self.k32(0)
            }
            R::B(b) => self.u32v(b),
        }
    }

    fn approx(&mut self, what: &'static str) {
        if !self.iface.approximate.contains(&what) {
            self.iface.approximate.push(what);
        }
    }

    /// A data dword as a lane mask half.
    fn data_mask(&mut self, v: V) -> V {
        let v = self.u32v(v);
        if let Some(c) = self.f.as_const(v) {
            if c == 0 || c == 0xFFFF_FFFF {
                return self.kb(c != 0);
            }
        }
        let lane = self.lane();
        let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[v, lane]);
        let one = self.k32(1);
        let bit = self.spv(op::BITWISE_AND, Ty::U32, &[s, one]);
        let zero = self.k32(0);
        self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[bit, zero])
    }

    fn mask_of(&mut self, r: R) -> V {
        match r {
            R::M(b, _) => b,
            R::D(v) => self.data_mask(v),
            R::B(b) => {
                self.approx("SCC read as a lane mask");
                b
            }
        }
    }

    fn scc_of(&mut self, r: R) -> V {
        match r {
            R::B(b) => b,
            R::D(v) => {
                let v = self.u32v(v);
                let z = self.k32(0);
                self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[v, z])
            }
            R::M(b, _) => b,
        }
    }

    // ---- register values ----

    fn val(&mut self, inst: &Inst, reg: Reg) -> Res<R> {
        let v = inst.read(reg).ok_or(unsupported(inst.machine.offset, "a register read the IR does not list"))?;
        self.value(v, inst.machine.offset)
    }

    fn value(&mut self, v: Value, offset: usize) -> Res<R> {
        if let Some(&r) = self.vals.get(&v) {
            return Ok(r);
        }
        // An input.
        let p = self.p;
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == v) {
            let r = self.input(reg, v, offset)?;
            self.vals.insert(v, r);
            return Ok(r);
        }
        Err(unsupported(offset, "a value read before it is defined"))
    }

    fn global_node(&mut self, op: Op, ty: Ty, args: Vec<V>) -> V {
        // Pure operations on prelude values are prelude values too.
        let node = Node { op, ty, args, imm: Vec::new() };
        if let Some(v) = self.fold(&node) {
            return v;
        }
        if let Some(&v) = self.cse.get(&node) {
            return v;
        }
        let v = self.f.push(node);
        self.f.prelude.push(v);
        v
    }
    fn global_bin(&mut self, o: u16, ty: Ty, a: V, b: V) -> V {
        self.global_node(Op::Spv(o), ty, vec![a, b])
    }

    fn def(&mut self, inst: &Inst, reg: Reg, r: R) {
        let r = match (self.sel, reg, r) {
            (Some(a), Reg::V(_), R::D(x)) => {
                let old = inst.read(reg).and_then(|o| self.value(o, inst.machine.offset).ok());
                match old {
                    Some(o) => {
                        let o = self.data(o);
                        let (x, o) = if self.f.ty(x) == Ty::F32 { (x, self.f32v(o)) } else { (self.u32v(x), self.u32v(o)) };
                        let t = self.f.ty(x);
                        R::D(self.spv(op::SELECT, t, &[a, x, o]))
                    }
                    None => r,
                }
            }
            _ => r,
        };
        if let Some(v) = inst.write(reg) {
            self.vals.insert(v, r);
            if self.guarded
                && (matches!(reg, Reg::V(_)) || matches!(r, R::M(..)))
            {
                let old = inst.read(reg);
                let o = self.open.as_mut().unwrap();
                o.defined.insert(v);
                if let Some(e) = o.defs.iter_mut().find(|d| d.0 == reg) {
                    e.1 = v;
                } else {
                    // a lane mask written here: lanes outside the guard get 0
                    let old = if matches!(reg, Reg::S(_)) { MASK_FALSE } else { old.unwrap_or(Value(u32::MAX)) };
                    o.defs.push((reg, v, old));
                }
            }
        }
    }

    // ---- operands ----

    fn opnd(&mut self, inst: &Inst, o: &Operand) -> Res<V> {
        self.opnd_dw(inst, o, 0, false)
    }

    /// Dword `k` of operand `o` as data (`f16` picks half-precision inline
    /// float constants).
    fn opnd_dw(&mut self, inst: &Inst, o: &Operand, k: u8, f16: bool) -> Res<V> {
        let off = inst.machine.offset;
        Ok(match *o {
            Operand::Scalar { code, .. } if code == crate::gfx10::NULL => self.k32(0),
            Operand::Scalar { code, .. } => {
                let r = self.val(inst, Reg::S(code + k))?;
                self.data(r)
            }
            Operand::Vector { reg, .. } => {
                let r = self.val(inst, Reg::V(reg + k))?;
                self.data(r)
            }
            Operand::Const(c) => {
                if k > 0 {
                    // 64-bit operand: sign extension of integer constants.
                    let lo = inline_const(c, f16).ok_or(unsupported(off, "a special operand"))?;
                    return Ok(self.k32(if (lo as i32) < 0 && c >= 193 { u32::MAX } else { 0 }));
                }
                match c {
                    251 => {
                        let r = self.val(inst, Reg::S(ir::VCC))?;
                        let m = self.mask_of(r);
                        let a = self.any(m);
                        let n = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]);
                        self.u32v(n)
                    }
                    252 => {
                        let r = self.val(inst, Reg::S(ir::EXEC))?;
                        let m = self.mask_of(r);
                        let a = self.any(m);
                        let n = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]);
                        self.u32v(n)
                    }
                    253 => {
                        let r = self.val(inst, Reg::Scc)?;
                        let b = self.scc_of(r);
                        self.u32v(b)
                    }
                    _ => {
                        let x = inline_const(c, f16).ok_or(unsupported(off, "a special operand"))?;
                        self.k32(x)
                    }
                }
            }
            Operand::Literal(l) => self.k32(if k > 0 { 0 } else { l }),
            Operand::Attr { .. } => return Err(unsupported(off, "an attribute operand")),
        })
    }

    fn opnd_mask(&mut self, inst: &Inst, o: &Operand) -> Res<V> {
        match *o {
            Operand::Scalar { code, .. } if code != crate::gfx10::NULL => {
                let r = self.val(inst, Reg::S(code))?;
                Ok(self.mask_of(r))
            }
            _ => {
                let v = self.opnd(inst, o)?;
                Ok(self.data_mask(v))
            }
        }
    }

    fn opnd_u64(&mut self, inst: &Inst, o: &Operand) -> Res<V> {
        let lo = self.opnd_dw(inst, o, 0, false)?;
        let hi = self.opnd_dw(inst, o, 1, false)?;
        Ok(self.u64_of(lo, hi))
    }

    fn u64_of(&mut self, lo: V, hi: V) -> V {
        let lo = self.u32v(lo);
        let hi = self.u32v(hi);
        if let (Some(a), Some(b)) = (self.f.as_const(lo), self.f.as_const(hi)) {
            return self.f.konst(Ty::U64, a | (b << 32));
        }
        if self.f.as_const(hi) == Some(0) {
            return self.spv(op::U_CONVERT, Ty::U64, &[lo]);
        }
        let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, 2), &[lo, hi]);
        self.spv(op::BITCAST, Ty::U64, &[v2])
    }

    fn split64(&mut self, v: V) -> (V, V) {
        if let Some(c) = self.f.as_const(v) {
            return (self.k32(c as u32), self.k32((c >> 32) as u32));
        }
        let v2 = self.spv(op::BITCAST, Ty::vec(S::U32, 2), &[v]);
        (self.extract(v2, 0), self.extract(v2, 1))
    }

    /// Float source `i` of a VALU instruction with its modifiers.
    fn fsrc(&mut self, inst: &Inst, v: &Valu, i: usize) -> Res<V> {
        let o = v.src.get(i).ok_or(unsupported(inst.machine.offset, "a missing operand"))?;
        let x = self.opnd(inst, o)?;
        let mut x = self.f32v(x);
        if v.abs >> i & 1 != 0 {
            x = self.gl(glsl::F_ABS, Ty::F32, &[x]);
        }
        if v.neg >> i & 1 != 0 {
            x = self.spv(op::F_NEGATE, Ty::F32, &[x]);
        }
        Ok(x)
    }

    fn usrc(&mut self, inst: &Inst, v: &Valu, i: usize) -> Res<V> {
        let o = v.src.get(i).ok_or(unsupported(inst.machine.offset, "a missing operand"))?;
        let mut x = self.opnd(inst, o)?;
        x = self.u32v(x);
        if let Some(s) = v.sdwa {
            let sel = s.src_sel.get(i).copied().unwrap_or(6);
            let sext = s.sext >> i & 1 != 0;
            x = self.sdwa_sel(x, sel, sext);
        }
        Ok(x)
    }

    fn sdwa_sel(&mut self, x: V, sel: u8, sext: bool) -> V {
        let (off, w) = match sel {
            0..=3 => (8 * u32::from(sel), 8),
            4 | 5 => (16 * u32::from(sel - 4), 16),
            _ => return x,
        };
        let (o, wv) = (self.k32(off), self.k32(w));
        let opc = if sext { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
        self.spv(opc, Ty::U32, &[x, o, wv])
    }

    /// Writes a VALU result to its destination (output modifiers, clamp,
    /// SDWA destination select).
    fn vdst(&mut self, inst: &Inst, v: &Valu, x: V) -> Res<()> {
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err(unsupported(inst.machine.offset, "a VALU destination"));
        };
        let mut x = x;
        if self.f.ty(x) == Ty::F32 {
            match v.omod {
                1 => {
                    let k = self.kf(2.0);
                    x = self.spv(op::F_MUL, Ty::F32, &[x, k]);
                }
                2 => {
                    let k = self.kf(4.0);
                    x = self.spv(op::F_MUL, Ty::F32, &[x, k]);
                }
                3 => {
                    let k = self.kf(0.5);
                    x = self.spv(op::F_MUL, Ty::F32, &[x, k]);
                }
                _ => {}
            }
            if v.clamp {
                let (z, o) = (self.kf(0.0), self.kf(1.0));
                x = self.gl(glsl::N_CLAMP, Ty::F32, &[x, z, o]);
            }
        }
        if let Some(s) = v.sdwa
            && s.dst_sel != 6
        {
            let (off, w) = match s.dst_sel {
                0..=3 => (8 * u32::from(s.dst_sel), 8),
                _ => (16 * u32::from(s.dst_sel - 4), 16),
            };
            let xu = self.u32v(x);
            let base = if s.dst_unused == 2 {
                let r = self.val(inst, Reg::V(reg))?;
                self.data(r)
            } else {
                self.k32(0)
            };
            let base = self.u32v(base);
            let (o, wv) = (self.k32(off), self.k32(w));
            x = self.spv(op::BIT_FIELD_INSERT, Ty::U32, &[base, xu, o, wv]);
        }
        self.def(inst, Reg::V(reg), R::D(x));
        Ok(())
    }

    fn sdst(&mut self, inst: &Inst, d: &Option<Operand>, k: u8, r: R) {
        if let Some(Operand::Scalar { code, .. }) = *d
            && code != crate::gfx10::NULL
        {
            self.def(inst, Reg::S(code + k), r);
        }
    }

    /// The current EXEC as `active`.
    fn active(&mut self, inst: &Inst) -> Res<V> {
        let r = self.val(inst, Reg::S(ir::EXEC))?;
        Ok(self.mask_of(r))
    }

    fn mask_def(&mut self, inst: &Inst, d: &Option<Operand>, b: V) {
        if let Some(Operand::Scalar { code, .. }) = *d
            && code != crate::gfx10::NULL
        {
            self.def(inst, Reg::S(code), R::M(b, Half::Lo));
            if self.wave64 {
                self.def(inst, Reg::S(code + 1), R::M(b, Half::Hi));
            }
        }
    }

    // ---- guards ----

    fn close(&mut self) {
        let Some(mut o) = self.open.take() else { return };
        let mut phis = Vec::new();
        let mut masks: HashMap<V, V> = HashMap::new();
        for (_, new, old) in o.defs {
            if let Some(&R::M(inner, half)) = self.vals.get(&new) {
                let _ = old;
                let phi = match masks.get(&inner) {
                    Some(&p) => p,
                    None => {
                        let f = self.kb(false);
                        let p = self.f.push(Node { op: Op::Phi, ty: Ty::BOOL, args: vec![inner, f], imm: Vec::new() });
                        phis.push(p);
                        masks.insert(inner, p);
                        p
                    }
                };
                self.vals.insert(new, R::M(phi, half));
                continue;
            }
            let mut inner = match self.vals.get(&new) {
                Some(R::D(v)) => *v,
                _ => continue,
            };
            // the type its readers want
            let want = match self.floaty.get(&new).copied().unwrap_or(0) {
                x if x > 0 => Ty::F32,
                x if x < 0 => Ty::U32,
                _ => self.f.ty(inner),
            };
            if want != self.f.ty(inner) && matches!(self.f.ty(inner), Ty::F32 | Ty::U32) {
                inner = match self.f.as_const(inner) {
                    Some(c) => self.f.konst(want, c),
                    None => {
                        let c = self.f.push(Node { op: Op::Spv(op::BITCAST), ty: want, args: vec![inner], imm: Vec::new() });
                        o.items.push(Item::N(c));
                        c
                    }
                };
            }
            let ty = self.f.ty(inner);
            let outer = match self.value(old, 0) {
                Ok(r) => {
                    let d = self.data(r);
                    match ty {
                        Ty::F32 => self.f32v(d),
                        _ => self.u32v(d),
                    }
                }
                Err(_) => self.f.global(Op::Undef, ty),
            };
            let phi = self.f.push(Node { op: Op::Phi, ty, args: vec![inner, outer], imm: Vec::new() });
            phis.push(phi);
            self.vals.insert(new, R::D(phi));
        }
        self.cur.push(Item::Guard { cond: o.cond, body: o.items, phis });
    }

    /// Before an unguarded instruction: closes the open guard if it reads a
    /// value written there.
    fn need(&mut self, inst: &Inst) {
        if let Some(o) = &self.open
            && inst.reads.iter().any(|(_, v)| o.defined.contains(v))
        {
            self.close();
        }
    }

    /// Opens (or keeps) a guard for `inst`; `false` when every lane runs.
    fn guard(&mut self, inst: &Inst) -> Res<bool> {
        let a = self.active(inst)?;
        if self.f.as_const(a) == Some(1) {
            self.need(inst);
            return Ok(false);
        }
        if let Some(o) = &self.open
            && o.cond != a
        {
            self.close();
        }
        if self.open.is_none() {
            self.open =
                Some(Open {
                cond: a,
                items: Vec::new(),
                defs: Vec::new(),
                defined: HashSet::new(),
                cse: HashMap::new(),
                inner: HashSet::new(),
            });
        }
        Ok(true)
    }

    // ---- program ----

    pub fn run(mut self) -> Res<(Func, Interface)> {
        let p = self.p;
        // Blocks in dominator-tree preorder: a block's dominators first, so
        // pure values are shared down the tree (one CSE scope per block).
        let (pre, idom) = dom_preorder(p);
        self.done = vec![None; p.blocks.len()];
        let mut scopes: Vec<(usize, usize)> = Vec::new();
        for &b in &pre {
            while let Some(&(top, mark)) = scopes.last() {
                if top == idom[b] {
                    break;
                }
                self.rollback(mark);
                scopes.pop();
            }
            scopes.push((b, self.undo.len()));
            self.lift_block(b)?;
        }
        self.fill_unreached();
        // Unreachable blocks (none from reverse postorder) become returns.
        self.f.blocks = std::mem::take(&mut self.done)
            .into_iter()
            .map(|b| b.unwrap_or(tir::Block { phis: Vec::new(), preds: Vec::new(), body: Vec::new(), term: Term::Return }))
            .collect();
        Ok((self.f, self.iface))
    }

    fn lift_block(&mut self, b: usize) -> Res<()> {
        let p = self.p;
        self.cur = Vec::new();
        // loads are not shared across blocks
        self.epoch += 1;
        self.block_phis(b);
        self.made[b] = true;
        // Edges from blocks lifted before: their values, converted at
        // the end of each source.
        for (k, &q) in p.blocks[b].preds.iter().enumerate() {
            if self.done[q].is_some() {
                let saved_cur = std::mem::take(&mut self.cur);
                self.cur = std::mem::take(&mut self.done[q].as_mut().unwrap().body);
                self.no_cse = true;
                let r = self.edge_k(q, b, k);
                self.no_cse = false;
                self.done[q].as_mut().unwrap().body = std::mem::replace(&mut self.cur, saved_cur);
                r?;
            }
        }
        let n = p.blocks[b].insts.len();
        let branch = matches!(p.blocks[b].exit, Exit::Branch { .. });
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            self.site = (b, i);
            if i + 1 == n && branch {
                break;
            }
            self.inst(inst)?;
        }
        self.close();
        let term = self.terminator(b)?;
        let body = std::mem::take(&mut self.cur);
        self.done[b] = Some(tir::Block { phis: self.all_phis[b].clone(), preds: p.blocks[b].preds.clone(), body, term });
        Ok(())
    }

    /// Phis: lane masks and SCC as booleans, data as f32 when every value
    /// known so far is f32 (back edges convert), else u32.
    fn block_phis(&mut self, b: usize) {
        let mut phis = Vec::new();
        let p = self.p;
        for phi in &p.blocks[b].phis {
            let mask = matches!(phi.reg, Reg::S(c) if c == ir::EXEC || c == ir::EXEC + 1 || c == ir::VCC || c == ir::VCC + 1)
                || phi.reg == Reg::Scc;
            let known: Vec<R> = phi.args.iter().filter_map(|a| self.vals.get(a).copied()).collect();
            let mask = mask || (!known.is_empty() && known.iter().all(|r| matches!(r, R::M(..))));
            let ty = if mask {
                Ty::BOOL
            } else if self.floaty.get(&phi.value).copied().unwrap_or(0) > 0 {
                Ty::F32
            } else {
                Ty::U32
            };
            let x = self.f.push(Node { op: Op::Phi, ty, args: vec![u32::MAX; phi.args.len()], imm: Vec::new() });
            phis.push(x);
            let half = match phi.reg {
                Reg::S(c) if self.wave64 && (c == ir::VCC + 1 || c == ir::EXEC + 1) => Half::Hi,
                _ => Half::Lo,
            };
            let r = match (phi.reg == Reg::Scc, mask) {
                (true, _) => R::B(x),
                (false, true) => R::M(x, half),
                _ => R::D(x),
            };
            self.vals.insert(phi.value, r);
        }
        self.all_phis[b] = phis;
    }

    /// A block's terminator, after the values its edges give lifted targets.
    fn terminator(&mut self, b: usize) -> Res<Term> {
        let p = self.p;
        Ok(match p.blocks[b].exit {
            Exit::Jump(t) => {
                self.edge(b, t)?;
                Term::Jump(t)
            }
            Exit::Branch { taken, not_taken } => {
                let inst = p.blocks[b].insts.last().unwrap();
                self.site = (b, p.blocks[b].insts.len() - 1);
                let c = self.branch_cond(inst)?;
                self.edge(b, taken)?;
                self.edge(b, not_taken)?;
                Term::Branch(c, taken, not_taken)
            }
            Exit::End => {
                // pixels: lanes not in EXEC at the end are killed
                let last = p.blocks[b].insts.last();
                if self.native.stage == HwStage::Ps
                    && let Some(inst) = last.filter(|i| i.read(Reg::S(ir::EXEC)).is_some())
                {
                    let a = self.active(inst)?;
                    self.kill_inactive(a);
                }
                Term::Return
            }
            Exit::Indirect => return Err(unsupported(p.blocks[b].offset, "an indirect jump (s_setpc_b64)")),
        })
    }

    /// Phi args of edges from blocks never reached: undefined.
    fn fill_unreached(&mut self) {
        for b in 0..self.p.blocks.len() {
            for j in 0..self.all_phis[b].len() {
                let node = self.all_phis[b][j];
                let ty = self.f.ty(node);
                let undef = self.f.global(Op::Undef, ty);
                for a in &mut self.f.nodes[node as usize].args {
                    if *a == u32::MAX {
                        *a = undef;
                    }
                }
            }
        }
    }

    fn kill_inactive(&mut self, a: V) {
        if self.f.as_const(a) == Some(1) {
            return;
        }
        let na = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]);
        let d = self.f.push(Node { op: Op::Demote, ty: Ty::VOID, args: vec![], imm: vec![] });
        self.cur.push(Item::Guard { cond: na, body: vec![Item::N(d)], phis: vec![] });
        self.iface.demotes = true;
    }

    /// At the end of `from`: the values the edge gives the phis of a block
    /// lifted already (a back edge); other targets take theirs when they
    /// are lifted.
    fn edge(&mut self, from: usize, to: usize) -> Res<()> {
        let p = self.p;
        if !self.made[to] {
            return Ok(());
        }
        for (k, &q) in p.blocks[to].preds.iter().enumerate() {
            if q == from {
                self.edge_k(from, to, k)?;
            }
        }
        Ok(())
    }

    /// The values edge `k` (from `from`) gives `to`'s phis, converted to
    /// their types where the current list is.
    fn edge_k(&mut self, from: usize, to: usize, k: usize) -> Res<()> {
        let p = self.p;
        for (j, phi) in p.blocks[to].phis.iter().enumerate() {
            let node = self.all_phis[to][j];
            let ty = self.f.ty(node);
            let x = match self.value(phi.args[k], p.blocks[from].offset) {
                Ok(r) => match ty {
                    Ty::BOOL if phi.reg == Reg::Scc => self.scc_of(r),
                    Ty::BOOL => self.mask_of(r),
                    Ty::F32 => {
                        let d = self.data(r);
                        self.f32v(d)
                    }
                    _ => {
                        let d = self.data(r);
                        self.u32v(d)
                    }
                },
                Err(_) => self.f.global(Op::Undef, ty),
            };
            self.f.nodes[node as usize].args[k] = x;
        }
        Ok(())
    }

    fn branch_cond(&mut self, inst: &Inst) -> Res<V> {
        let name = inst.name();
        Ok(match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let r = self.val(inst, Reg::Scc)?;
                let b = self.scc_of(r);
                if name == "s_cbranch_scc0" { self.spv(op::LOGICAL_NOT, Ty::BOOL, &[b]) } else { b }
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") { ir::VCC } else { ir::EXEC };
                let r = self.val(inst, Reg::S(reg))?;
                let m = self.mask_of(r);
                let a = self.any(m);
                if name.ends_with('z') && !name.ends_with("nz") { self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]) } else { a }
            }
            _ => return Err(unsupported(inst.machine.offset, "a branch")),
        })
    }

    // ---- instructions ----

    fn inst(&mut self, inst: &Inst) -> Res<()> {
        let name = inst.name();
        let off = inst.machine.offset;
        match &inst.fields {
            Fields::Sop { dst, src, imm } => {
                let (dst, src, imm) = (*dst, src.clone(), *imm);
                self.need(inst);
                self.sop(inst, name, &dst, &src, imm)
            }
            Fields::Sopp { .. } => match name {
                "s_barrier" => {
                    self.close();
                    self.emit(Op::Barrier, Ty::VOID, vec![]);
                    Ok(())
                }
                // Mesh output counts: the module declares the maxima at its
                // start (primitives culled until exported).
                "s_sendmsg" | "s_sendmsghalt" => Ok(()),
                "s_endpgm" | "s_branch" | "s_icache_inv" | "s_round_mode" | "s_denorm_mode" | "s_wakeup" => Ok(()),
                _ => Err(unsupported(off, name)),
            },
            Fields::Smem { data, base, offset, soffset, .. } => {
                let (data, base, offset, soffset) = (*data, *base, *offset, *soffset);
                self.need(inst);
                self.smem(inst, name, data, base, offset, soffset)
            }
            Fields::Valu(v) => {
                let v = v.clone();
                self.valu(inst, name, &v)
            }
            Fields::Vintrp { dst, .. } => {
                let dst = *dst;
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.interp(inst, dst);
                self.guarded = false;
                r
            }
            Fields::Buffer { .. } => {
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.buffer(inst, name);
                self.guarded = false;
                r
            }
            Fields::Ds { .. } => {
                if name == "ds_swizzle_b32" || name == "ds_bpermute_b32" || name == "ds_permute_b32" {
                    self.need(inst);
                    return self.ds_cross(inst, name);
                }
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.ds(inst, name);
                self.guarded = false;
                r
            }
            Fields::Image { .. } => {
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.image(inst, name);
                self.guarded = false;
                r
            }
            Fields::Flat { .. } => {
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.flat(inst, name);
                self.guarded = false;
                r
            }
            Fields::Exp { .. } => {
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.export(inst);
                self.guarded = false;
                r
            }
        }
    }

    // ---- scalar ALU ----

    // ---- scalar memory ----

    // ---- buffers ----

    // ---- LDS ----

    // ---- flat / global ----

    // ---- export ----

    // ---- interpolation ----

}







fn int_cmp(rel: &str, signed: bool) -> Option<u16> {
    Some(match (rel, signed) {
        ("eq", _) => op::I_EQUAL,
        ("lg" | "ne", _) => op::I_NOT_EQUAL,
        ("gt", false) => op::U_GREATER_THAN,
        ("gt", true) => op::S_GREATER_THAN,
        ("ge", false) => op::U_GREATER_THAN_EQUAL,
        ("ge", true) => op::S_GREATER_THAN_EQUAL,
        ("lt", false) => op::U_LESS_THAN,
        ("lt", true) => op::S_LESS_THAN,
        ("le", false) => op::U_LESS_THAN_EQUAL,
        ("le", true) => op::S_LESS_THAN_EQUAL,
        _ => return None,
    })
}

fn atomic_op(name: &str) -> Option<u16> {
    let stem = name.split("_atomic_").nth(1).or_else(|| name.strip_prefix("ds_")).unwrap_or("");
    let stem = stem.trim_end_matches("_rtn_u32").trim_end_matches("_rtn_b32").trim_end_matches("_rtn_i32");
    Some(match stem {
        "add" | "add_u32" => op::ATOMIC_I_ADD,
        "sub" | "sub_u32" => op::ATOMIC_I_SUB,
        "smin" | "min_i32" => op::ATOMIC_S_MIN,
        "umin" | "min_u32" => op::ATOMIC_U_MIN,
        "smax" | "max_i32" => op::ATOMIC_S_MAX,
        "umax" | "max_u32" => op::ATOMIC_U_MAX,
        "and" | "and_b32" => op::ATOMIC_AND,
        "or" | "or_b32" => op::ATOMIC_OR,
        "xor" | "xor_b32" => op::ATOMIC_XOR,
        "swap" | "wrxchg_rtn_b32" | "wrxchg" => op::ATOMIC_EXCHANGE,
        "cmpswap" | "cmpst" | "cmpst_rtn_b32" => op::ATOMIC_COMPARE_EXCHANGE,
        _ => return None,
    })
}

#[allow(dead_code)]
fn root_window(stage: HwStage, t: RootType) -> u32 {
    let _ = t;
    push_window(stage)
}

// Instruction families live in their own files.
#[path = "lift_image.rs"]
mod image;
#[path = "lift_io.rs"]
mod io_;
#[path = "lift_mem.rs"]
mod mem;
#[path = "lift_scalar.rs"]
mod scalar;
#[path = "lift_valu.rs"]
mod valu;
#[path = "lift_lane.rs"]
mod lane;

