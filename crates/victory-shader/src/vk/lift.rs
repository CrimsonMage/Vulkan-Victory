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
}

/// How each value is read: above 0 mostly as a float (so loads and phis
/// make it one and the bitcasts go), below as an integer.
fn float_uses(p: &Program) -> HashMap<Value, i32> {
    let mut score: HashMap<Value, i32> = HashMap::new();
    let mut moves: Vec<(Value, Value)> = Vec::new();
    for b in &p.blocks {
        for phi in &b.phis {
            for &a in &phi.args {
                moves.push((phi.value, a));
            }
        }
        for inst in &b.insts {
            let name = inst.name();
            let mut regs = ir::operand_reads(p, inst);
            if name.contains("mac_") {
                regs.extend(inst.writes.iter().map(|w| w.0).filter(|r| matches!(r, Reg::V(_))));
            }
            let val = |r: Reg| inst.read(r);
            let w = match &inst.fields {
                Fields::Valu(_) => {
                    if matches!(name, "v_mov_b32" | "v_cndmask_b32") {
                        if let Some((_, d)) = inst.writes.iter().find(|w| matches!(w.0, Reg::V(_))) {
                            for &r in &regs {
                                if let (Reg::V(_) | Reg::S(_), Some(v)) = (r, val(r)) {
                                    moves.push((*d, v));
                                }
                            }
                        }
                        continue;
                    }
                    let src = if let Some(rest) = name.strip_prefix("v_cvt_") {
                        rest.rsplit('_').next().unwrap_or("")
                    } else {
                        name.rsplit('_').next().unwrap_or("")
                    };
                    match src {
                        "f32" => 1,
                        "u32" | "i32" | "b32" | "u24" | "i24" | "u64" | "i64" | "b64" | "ubyte0" | "ubyte1" | "ubyte2"
                        | "ubyte3" => -1,
                        _ => 0,
                    }
                }
                Fields::Image { addr, .. } if name.starts_with("image_sample") || name.starts_with("image_gather") => {
                    for a in addr {
                        if let Operand::Vector { reg, dwords } = *a {
                            for k in 0..dwords {
                                if let Some(v) = val(Reg::V(reg + k)) {
                                    *score.entry(v).or_default() += 1;
                                }
                            }
                        }
                    }
                    continue;
                }
                _ => 0,
            };
            if w == 0 {
                continue;
            }
            for r in regs {
                if matches!(r, Reg::V(_) | Reg::S(_))
                    && let Some(v) = val(r)
                {
                    *score.entry(v).or_default() += w;
                }
            }
        }
    }
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
        if pure {
            if self.guarded
                && let Some(o) = &self.open
                && let Some(&v) = o.cse.get(&node)
            {
                return v;
            }
            if !self.no_cse
                && let Some(&v) = self.cse.get(&node)
            {
                return v;
            }
        }
        // Conversions of values from outside the guard go outside it, where
        // the other guards of the block share them.
        let hoist = self.guarded
            && matches!(node.op, Op::Spv(op::BITCAST | op::U_CONVERT | op::S_CONVERT | op::COMPOSITE_EXTRACT))
            && node.args.iter().all(|a| !self.open.as_ref().unwrap().inner.contains(a));
        let v = self.f.push(node.clone());
        if self.guarded && !hoist {
            let o = self.open.as_mut().expect("an open guard");
            o.items.push(Item::N(v));
            o.inner.insert(v);
            if pure {
                o.cse.insert(node, v);
            }
        } else {
            self.cur.push(Item::N(v));
            if pure && !self.no_cse {
                let old = self.cse.insert(node.clone(), v);
                self.undo.push((node, old));
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
        let u32r = |s: &mut Self, x: u64| Some(s.f.konst(Ty::U32, x & 0xFFFF_FFFF));
        if ty == Ty::U32 && n.args.len() == 2 {
            if let (Some(x), Some(y)) = (a, b) {
                let (x, y) = (x as u32, y as u32);
                let r = match o {
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
                };
                return u32r(self, u64::from(r));
            }
            // identities
            let (l, r) = (n.args[0], n.args[1]);
            match (o, a.map(|x| x as u32), b.map(|x| x as u32)) {
                (op::I_ADD | op::BITWISE_OR | op::BITWISE_XOR, Some(0), _) => return Some(r),
                (op::I_ADD | op::BITWISE_OR | op::BITWISE_XOR | op::I_SUB, _, Some(0)) => return Some(l),
                (op::SHIFT_LEFT_LOGICAL | op::SHIFT_RIGHT_LOGICAL | op::SHIFT_RIGHT_ARITHMETIC, _, Some(0)) => {
                    return Some(l);
                }
                (op::BITWISE_AND, Some(0), _) | (op::BITWISE_AND, _, Some(0)) => return u32r(self, 0),
                (op::BITWISE_AND, Some(u32::MAX), _) => return Some(r),
                (op::BITWISE_AND, _, Some(u32::MAX)) => return Some(l),
                (op::I_MUL, Some(1), _) => return Some(r),
                (op::I_MUL, _, Some(1)) => return Some(l),
                (op::I_MUL, Some(0), _) | (op::I_MUL, _, Some(0)) => return u32r(self, 0),
                _ => {}
            }
        }
        if ty == Ty::U64 && n.args.len() == 2 && o == op::I_ADD {
            if let (Some(x), Some(y)) = (a, b) {
                return Some(self.f.konst(Ty::U64, x.wrapping_add(y)));
            }
            if b == Some(0) {
                return Some(n.args[0]);
            }
            if a == Some(0) {
                return Some(n.args[1]);
            }
        }
        if ty == Ty::BOOL {
            match o {
                op::LOGICAL_AND => match (a, b) {
                    (Some(0), _) | (_, Some(0)) => return Some(self.kb(false)),
                    (Some(_), _) => return Some(n.args[1]),
                    (_, Some(_)) => return Some(n.args[0]),
                    _ if n.args[0] == n.args[1] => return Some(n.args[0]),
                    _ => {}
                },
                op::LOGICAL_OR => match (a, b) {
                    (Some(1), _) | (_, Some(1)) => return Some(self.kb(true)),
                    (Some(_), _) => return Some(n.args[1]),
                    (_, Some(_)) => return Some(n.args[0]),
                    _ if n.args[0] == n.args[1] => return Some(n.args[0]),
                    _ => {}
                },
                op::LOGICAL_NOT => {
                    if let Some(x) = a {
                        return Some(self.kb(x == 0));
                    }
                    if let Op::Spv(op::LOGICAL_NOT) = self.f.node(n.args[0]).op {
                        return Some(self.f.node(n.args[0]).args[0]);
                    }
                }
                op::LOGICAL_NOT_EQUAL => {
                    if let (Some(x), Some(y)) = (a, b) {
                        return Some(self.kb(x != y));
                    }
                    if a == Some(0) {
                        return Some(n.args[1]);
                    }
                    if b == Some(0) {
                        return Some(n.args[0]);
                    }
                }
                op::I_EQUAL | op::I_NOT_EQUAL | op::U_LESS_THAN | op::U_GREATER_THAN_EQUAL => {
                    if let (Some(x), Some(y)) = (a, b) {
                        let r = match o {
                            op::I_EQUAL => x == y,
                            op::I_NOT_EQUAL => x != y,
                            op::U_LESS_THAN => x < y,
                            _ => x >= y,
                        };
                        return Some(self.kb(r));
                    }
                }
                _ => {}
            }
        }
        if (o == op::U_CONVERT || o == op::S_CONVERT)
            && let Some(x) = a
        {
            let from = self.f.ty(n.args[0]);
            let bits = match from.s {
                S::U8 => 8,
                S::U16 => 16,
                S::U32 => 32,
                _ => 64,
            };
            let x = if o == op::S_CONVERT && bits < 64 && (x >> (bits - 1)) & 1 != 0 { x | (!0u64 << bits) } else { x };
            let mask = match ty.s {
                S::U8 => 0xFF,
                S::U16 => 0xFFFF,
                S::U32 => 0xFFFF_FFFF,
                _ => u64::MAX,
            };
            return Some(self.f.konst(ty, x & mask));
        }
        if o == op::SELECT {
            if let Some(cv) = a {
                return Some(if cv != 0 { n.args[1] } else { n.args[2] });
            }
            if n.args[1] == n.args[2] {
                return Some(n.args[1]);
            }
        }
        if o == op::BITCAST && n.args.len() == 1 {
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

    fn input(&mut self, reg: Reg, v: Value, offset: usize) -> Res<R> {
        let stage = self.native.stage;
        if let Some(&(_, input)) = self.io.inputs.iter().find(|i| i.0 == v) {
            let u = |l: &mut Self, b| l.f.global(Op::Builtin(b), Ty::U32);
            let x = match input {
                Input::VertexId => u(self, Builtin::VertexIndex),
                // A vertex attribute the console's fetch shader loads: a
                // vertex input at the location of its signature row.
                Input::Attribute { row, channel } => {
                    let ty = if self.vs_float.get(&row).copied().unwrap_or(true) { Ty::F32 } else { Ty::U32 };
                    self.f.global(Op::Input { loc: row, comp: channel, interp: Interp::Flat }, ty)
                }
                Input::InstanceId => {
                    let i = u(self, Builtin::InstanceIndex);
                    let b = u(self, Builtin::BaseInstance);
                    self.global_bin(op::I_SUB, Ty::U32, i, b)
                }
                Input::PrimitiveId => u(self, Builtin::PrimitiveId),
                Input::ThreadIdInGroup(c) => u(self, Builtin::LocalInvocationId(c)),
                Input::GroupId(c) => u(self, Builtin::WorkgroupId(c)),
                Input::Position(c) => self.f.global(Op::Builtin(Builtin::FragCoord(c)), Ty::F32),
                Input::FrontFace => {
                    let b = self.f.global(Op::Builtin(Builtin::FrontFacing), Ty::BOOL);
                    let (one, zero) = (self.k32(1), self.k32(0));
                    self.global_node(Op::Spv(op::SELECT), Ty::U32, vec![b, one, zero])
                }
                Input::Ancillary => {
                    // sample id in bits 8-11
                    let s = u(self, Builtin::SampleId);
                    let k = self.k32(8);
                    self.global_bin(op::SHIFT_LEFT_LOGICAL, Ty::U32, s, k)
                }
                Input::SampleCoverage => u(self, Builtin::SampleMaskIn),
                Input::Barycentric { .. } => {
                    // Only read by interpolation (handled at the v_interp sites).
                    self.f.konst(Ty::U32, 0)
                }
                Input::MeshGroup => {
                    let g = u(self, Builtin::WorkgroupId(0));
                    let k = self.k32(self.mesh_stride);
                    self.global_bin(op::I_MUL, Ty::U32, g, k)
                }
                Input::MergedWaveInfo | Input::GroupInfo => {
                    // the wave's index in the group in bits 24-27 (mesh)
                    let w = self.console_wave();
                    let k = self.k32(if input == Input::GroupInfo { 6 } else { 24 });
                    self.global_bin(op::SHIFT_LEFT_LOGICAL, Ty::U32, w, k)
                }
                _ => return Err(unsupported(offset, "a system input")),
            };
            return Ok(R::D(x));
        }
        match reg {
            Reg::S(c) if c == ir::EXEC || c == ir::EXEC + 1 => {
                let half = if c == ir::EXEC { Half::Lo } else { Half::Hi };
                let b = if stage == HwStage::Ps {
                    let h = self.f.global(Op::Builtin(Builtin::Helper), Ty::BOOL);
                    self.global_node(Op::Spv(op::LOGICAL_NOT), Ty::BOOL, vec![h])
                } else {
                    self.kb(true)
                };
                Ok(R::M(b, half))
            }
            Reg::S(c) => {
                // user data
                let base = match stage {
                    HwStage::Ms | HwStage::Gs | HwStage::Es | HwStage::Hs | HwStage::Ls => 8,
                    _ => 0,
                };
                let users = match &self.native.io {
                    StageIo::Cs(cs) => u32::from(cs.user_sgprs()),
                    _ => 32,
                };
                let c = u32::from(c);
                if c >= base && c - base < users && c - base < 60 {
                    let k = push_window(stage) + (c - base);
                    return Ok(R::D(self.f.global(Op::Push(k), Ty::U32)));
                }
                Ok(R::D(self.f.konst(Ty::U32, 0)))
            }
            _ => Ok(R::D(self.f.global(Op::Undef, Ty::U32))),
        }
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
            for &c in children[b].iter().rev() {
                stack.push(c);
            }
        }
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
            self.cur = Vec::new();
            // Phis: lane masks and SCC as booleans, data as f32 when every
            // value known so far is f32 (back edges convert), else u32.
            let mut phis = Vec::new();
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
                let r = if phi.reg == Reg::Scc {
                    R::B(x)
                } else if mask {
                    let half = match phi.reg {
                        Reg::S(c) if c == ir::VCC + 1 || c == ir::EXEC + 1 => Half::Hi,
                        _ => Half::Lo,
                    };
                    R::M(x, if self.wave64 { half } else { Half::Lo })
                } else {
                    R::D(x)
                };
                self.vals.insert(phi.value, r);
            }
            self.all_phis[b] = phis;
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
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                self.site = (b, i);
                if i + 1 == n && matches!(p.blocks[b].exit, Exit::Branch { .. }) {
                    break;
                }
                self.inst(inst)?;
            }
            self.close();
            // terminator
            let term = match p.blocks[b].exit {
                Exit::Jump(t) => {
                    self.edge(b, t)?;
                    Term::Jump(t)
                }
                Exit::Branch { taken, not_taken } => {
                    let inst = p.blocks[b].insts.last().unwrap();
                    self.site = (b, n - 1);
                    let c = self.branch_cond(inst)?;
                    self.edge(b, taken)?;
                    self.edge(b, not_taken)?;
                    Term::Branch(c, taken, not_taken)
                }
                Exit::End => {
                    if self.native.stage == HwStage::Ps && !self.f.blocks.is_empty() || self.native.stage == HwStage::Ps
                    {
                        // Lanes not in EXEC at the end are killed.
                        if let Some(inst) = p.blocks[b].insts.last() {
                            if inst.read(Reg::S(ir::EXEC)).is_some() {
                                let a = self.active(inst)?;
                                self.kill_inactive(a);
                            }
                        }
                    }
                    Term::Return
                }
                Exit::Indirect => {
                    return Err(unsupported(p.blocks[b].offset, "an indirect jump (s_setpc_b64)"));
                }
            };
            let body = std::mem::take(&mut self.cur);
            self.done[b] = Some(tir::Block { phis: self.all_phis[b].clone(), preds: p.blocks[b].preds.clone(), body, term });
        }
        // Phi args of edges from blocks never reached.
        for b in 0..p.blocks.len() {
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
        // Unreachable blocks (none from reverse postorder) become returns.
        self.f.blocks = std::mem::take(&mut self.done)
            .into_iter()
            .map(|b| b.unwrap_or(tir::Block { phis: Vec::new(), preds: Vec::new(), body: Vec::new(), term: Term::Return }))
            .collect();
        Ok((self.f, self.iface))
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

    fn sop(&mut self, inst: &Inst, name: &'static str, dst: &Option<Operand>, src: &[Operand], imm: Option<u16>) -> Res<()> {
        let off = inst.machine.offset;
        let b64 = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        let mask_like = |l: &Self, o: &Operand| -> bool {
            if let Operand::Scalar { code, .. } = *o
                && code != crate::gfx10::NULL
                && let Some(v) = inst.read(Reg::S(code))
            {
                return matches!(l.vals.get(&v), Some(R::M(..)))
                    || (code == ir::EXEC && !l.vals.contains_key(&v));
            }
            false
        };
        let dst_exec = matches!(dst, Some(Operand::Scalar { code, .. }) if *code == ir::EXEC || *code == ir::VCC);
        let stem = name.trim_start_matches("s_");
        let mut base = stem;
        while let Some((a, t)) = base.rsplit_once('_') {
            if matches!(t, "b32" | "b64" | "i32" | "u32" | "i64" | "u64" | "b16" | "i16" | "u16") {
                base = a;
            } else {
                break;
            }
        }
        let scc_def = |l: &mut Self, b: V| {
            if inst.write(Reg::Scc).is_some() {
                l.def(inst, Reg::Scc, R::B(b));
            }
        };
        // saveexec / wrexec family
        if name.contains("saveexec") || name.contains("wrexec") {
            let a = self.active(inst)?;
            let s = self.opnd_mask(inst, &src[0])?;
            let opname = base.trim_end_matches("_saveexec").trim_end_matches("_wrexec");
            let new = self.mask_bin(opname, s, a).ok_or(unsupported(off, name))?;
            let wr = name.contains("wrexec");
            self.mask_def(inst, dst, if wr { new } else { a });
            self.def(inst, Reg::S(ir::EXEC), R::M(new, Half::Lo));
            if self.wave64 {
                self.def(inst, Reg::S(ir::EXEC + 1), R::M(new, Half::Hi));
            }
            let any = self.any(new);
            scc_def(self, any);
            return Ok(());
        }
        match base {
            "mov" | "cmov" | "movk" | "cmovk" => {
                let cond = if base.starts_with("cmov") {
                    let r = self.val(inst, Reg::Scc)?;
                    Some(self.scc_of(r))
                } else {
                    None
                };
                if let Some(k) = imm {
                    let x = self.k32(i32::from(k as i16) as u32);
                    let x = if let Some(c) = cond {
                        let old = self.opnd(inst, &dst.unwrap())?;
                        let old = self.u32v(old);
                        self.spv(op::SELECT, Ty::U32, &[c, x, old])
                    } else {
                        x
                    };
                    self.sdst(inst, dst, 0, R::D(x));
                    return Ok(());
                }
                let n = if b64 { 2 } else { 1 };
                if b64 && (mask_like(self, &src[0]) || dst_exec) && cond.is_none() {
                    let m = self.opnd_mask(inst, &src[0])?;
                    self.mask_def(inst, dst, m);
                    return Ok(());
                }
                for k in 0..n {
                    let r = match src[0] {
                        Operand::Scalar { code, .. } if code != crate::gfx10::NULL => self.val(inst, Reg::S(code + k))?,
                        _ => R::D(self.opnd_dw(inst, &src[0], k, false)?),
                    };
                    let r = if let Some(c) = cond {
                        let x = self.data(r);
                        let x = self.u32v(x);
                        let old = self.opnd_dw(inst, &dst.unwrap(), k, false)?;
                        let old = self.u32v(old);
                        R::D(self.spv(op::SELECT, Ty::U32, &[c, x, old]))
                    } else {
                        r
                    };
                    self.sdst(inst, dst, k, r);
                }
                Ok(())
            }
            "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not" | "wqm" => {
                let unary = matches!(base, "not" | "wqm");
                let maskish = src.iter().any(|o| mask_like(self, o)) || dst_exec;
                if maskish || base == "wqm" {
                    let a = self.opnd_mask(inst, &src[0])?;
                    let r = if base == "not" {
                        self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a])
                    } else if base == "wqm" {
                        self.wqm(a)
                    } else {
                        let b = self.opnd_mask(inst, &src[1])?;
                        self.mask_bin(base, a, b).ok_or(unsupported(off, name))?
                    };
                    self.mask_def(inst, dst, r);
                    let any = self.any(r);
                    scc_def(self, any);
                    return Ok(());
                }
                let n = if b64 { 2 } else { 1 };
                let mut nz = None;
                for k in 0..n {
                    let a = self.opnd_dw(inst, &src[0], k, false)?;
                    let a = self.u32v(a);
                    let r = if unary {
                        self.spv(op::NOT, Ty::U32, &[a])
                    } else {
                        let b = self.opnd_dw(inst, &src[1], k, false)?;
                        let b = self.u32v(b);
                        self.int_bin(base, a, b).ok_or(unsupported(off, name))?
                    };
                    self.sdst(inst, dst, k, R::D(r));
                    nz = Some(match nz {
                        None => r,
                        Some(p) => self.spv(op::BITWISE_OR, Ty::U32, &[p, r]),
                    });
                }
                let z = self.k32(0);
                let b = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[nz.unwrap(), z]);
                scc_def(self, b);
                Ok(())
            }
            "cselect" => {
                let r = self.val(inst, Reg::Scc)?;
                let c = self.scc_of(r);
                if b64 && (src.iter().any(|o| mask_like(self, o)) || dst_exec) {
                    let a = self.opnd_mask(inst, &src[0])?;
                    let b = self.opnd_mask(inst, &src[1])?;
                    let m = self.spv(op::SELECT, Ty::BOOL, &[c, a, b]);
                    self.mask_def(inst, dst, m);
                    return Ok(());
                }
                for k in 0..if b64 { 2 } else { 1 } {
                    let a = self.opnd_dw(inst, &src[0], k, false)?;
                    let b = self.opnd_dw(inst, &src[1], k, false)?;
                    let (a, b) = (self.u32v(a), self.u32v(b));
                    let x = self.spv(op::SELECT, Ty::U32, &[c, a, b]);
                    self.sdst(inst, dst, k, R::D(x));
                }
                Ok(())
            }
            "bcnt1" | "bcnt0" => {
                let mut sum = self.k32(0);
                for k in 0..if b64 { 2 } else { 1 } {
                    let a = self.opnd_dw(inst, &src[0], k, false)?;
                    let a = self.u32v(a);
                    let a = if base == "bcnt0" { self.spv(op::NOT, Ty::U32, &[a]) } else { a };
                    let c = self.spv(op::BIT_COUNT, Ty::U32, &[a]);
                    sum = self.spv(op::I_ADD, Ty::U32, &[sum, c]);
                }
                if b64 && self.wave64 {
                    self.approx("a wave64 mask counted (s_bcnt1_i32_b64)");
                }
                self.sdst(inst, dst, 0, R::D(sum));
                let z = self.k32(0);
                let b = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[sum, z]);
                scc_def(self, b);
                Ok(())
            }
            "ff1" | "ff0" => {
                let a = self.opnd_dw(inst, &src[0], 0, false)?;
                let mut a = self.u32v(a);
                if base == "ff0" {
                    a = self.spv(op::NOT, Ty::U32, &[a]);
                }
                if b64 && self.wave64 {
                    self.approx("a wave64 mask searched (s_ff1_i32_b64)");
                }
                let x = self.gl(glsl::FIND_I_LSB, Ty::U32, &[a]);
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "flbit" => {
                let a = self.opnd(inst, &src[0])?;
                let a = self.u32v(a);
                let x = if name.ends_with("i32_i32") {
                    // count of leading sign bits
                    let m = self.gl(glsl::FIND_S_MSB, Ty::U32, &[a]);
                    let k = self.k32(31);
                    let r = self.spv(op::I_SUB, Ty::U32, &[k, m]);
                    let neg = self.k32(u32::MAX);
                    let is = self.spv(op::I_EQUAL, Ty::BOOL, &[m, neg]);
                    self.spv(op::SELECT, Ty::U32, &[is, neg, r])
                } else {
                    self.clz(a)
                };
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "brev" => {
                let a = self.opnd(inst, &src[0])?;
                let a = self.u32v(a);
                let x = self.spv(op::BIT_REVERSE, Ty::U32, &[a]);
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "abs" => {
                let a = self.opnd(inst, &src[0])?;
                let a = self.u32v(a);
                let x = self.gl(glsl::S_ABS, Ty::U32, &[a]);
                self.sdst(inst, dst, 0, R::D(x));
                let z = self.k32(0);
                let b = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                scc_def(self, b);
                Ok(())
            }
            "add" | "sub" | "addc" | "subb" | "addk" | "mul" | "mulk" | "mul_hi" | "min" | "max" | "lshl" | "lshr"
            | "ashr" | "bfe" | "bfm" | "lshl1_add" | "lshl2_add" | "lshl3_add" | "lshl4_add" | "absdiff" | "pack_ll"
            | "pack_lh" | "pack_hh" | "pack_hl" => self.salu_arith(inst, name, base, dst, src, imm, b64),
            "cmp_eq" | "cmp_lg" | "cmp_gt" | "cmp_ge" | "cmp_lt" | "cmp_le" | "cmpk_eq" | "cmpk_lg" | "cmpk_gt"
            | "cmpk_ge" | "cmpk_lt" | "cmpk_le" | "bitcmp0" | "bitcmp1" => {
                let signed = name.ends_with("i32");
                let (a, b) = if let Some(k) = imm {
                    let a = self.opnd(inst, &src[0])?;
                    let k = if signed { i32::from(k as i16) as u32 } else { u32::from(k) };
                    (self.u32v(a), self.k32(k))
                } else if b64 {
                    (self.opnd_u64(inst, &src[0])?, self.opnd_u64(inst, &src[1])?)
                } else {
                    let a = self.opnd(inst, &src[0])?;
                    let b = self.opnd(inst, &src[1])?;
                    (self.u32v(a), self.u32v(b))
                };
                let rel = base.trim_start_matches("cmpk_").trim_start_matches("cmp_");
                let c = if rel.starts_with("bitcmp") {
                    let one = self.k32(1);
                    let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, b]);
                    let bit = self.spv(op::BITWISE_AND, Ty::U32, &[s, one]);
                    let z = self.k32(if rel == "bitcmp0" { 0 } else { 1 });
                    self.spv(op::I_EQUAL, Ty::BOOL, &[bit, z])
                } else {
                    let o = int_cmp(rel, signed).ok_or(unsupported(off, name))?;
                    self.spv(o, Ty::BOOL, &[a, b])
                };
                self.def(inst, Reg::Scc, R::B(c));
                Ok(())
            }
            "bitset0" | "bitset1" => {
                let bit = self.opnd(inst, &src[0])?;
                let bit = self.u32v(bit);
                let old = self.opnd(inst, &dst.unwrap())?;
                let old = self.u32v(old);
                let one = self.k32(1);
                let k31 = self.k32(31);
                let bit = self.spv(op::BITWISE_AND, Ty::U32, &[bit, k31]);
                let m = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[one, bit]);
                let x = if base == "bitset1" {
                    self.spv(op::BITWISE_OR, Ty::U32, &[old, m])
                } else {
                    let nm = self.spv(op::NOT, Ty::U32, &[m]);
                    self.spv(op::BITWISE_AND, Ty::U32, &[old, nm])
                };
                self.sdst(inst, dst, 0, R::D(x));
                Ok(())
            }
            "getpc" => {
                let x = self.k32((off + 4) as u32);
                self.sdst(inst, dst, 0, R::D(x));
                let z = self.k32(0);
                self.sdst(inst, dst, 1, R::D(z));
                Ok(())
            }
            // the call of the vertex fetch shader: its results are inputs;
            // a far jump (the block's exit already goes there)
            "swappc" | "setpc" => Ok(()),
            "setreg" | "setreg_imm32" | "setprio" | "version" | "waitcnt_vscnt" | "waitcnt_vmcnt" | "waitcnt_expcnt"
            | "waitcnt_lgkmcnt" => Ok(()),
            "getreg" => {
                let z = self.k32(0);
                self.sdst(inst, dst, 0, R::D(z));
                Ok(())
            }
            _ => Err(unsupported(off, name)),
        }
    }

    fn wqm(&mut self, a: V) -> V {
        if self.f.as_const(a).is_some() {
            return a;
        }
        // any lane of my quad
        let b = self.ballot(a);
        let lane = self.lane();
        let m3 = self.k32(!3);
        let q = self.spv(op::BITWISE_AND, Ty::U32, &[lane, m3]);
        let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, q]);
        let f = self.k32(0xF);
        let n = self.spv(op::BITWISE_AND, Ty::U32, &[s, f]);
        let z = self.k32(0);
        self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[n, z])
    }

    fn mask_bin(&mut self, base: &str, a: V, b: V) -> Option<V> {
        Some(match base {
            "and" => self.spv(op::LOGICAL_AND, Ty::BOOL, &[a, b]),
            "or" => self.spv(op::LOGICAL_OR, Ty::BOOL, &[a, b]),
            "xor" => self.spv(op::LOGICAL_NOT_EQUAL, Ty::BOOL, &[a, b]),
            "xnor" => self.spv(op::LOGICAL_EQUAL, Ty::BOOL, &[a, b]),
            "andn2" => {
                let nb = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[b]);
                self.spv(op::LOGICAL_AND, Ty::BOOL, &[a, nb])
            }
            "orn2" => {
                let nb = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[b]);
                self.spv(op::LOGICAL_OR, Ty::BOOL, &[a, nb])
            }
            "nand" => {
                let x = self.spv(op::LOGICAL_AND, Ty::BOOL, &[a, b]);
                self.spv(op::LOGICAL_NOT, Ty::BOOL, &[x])
            }
            "nor" => {
                let x = self.spv(op::LOGICAL_OR, Ty::BOOL, &[a, b]);
                self.spv(op::LOGICAL_NOT, Ty::BOOL, &[x])
            }
            _ => return None,
        })
    }

    fn int_bin(&mut self, base: &str, a: V, b: V) -> Option<V> {
        Some(match base {
            "and" => self.spv(op::BITWISE_AND, Ty::U32, &[a, b]),
            "or" => self.spv(op::BITWISE_OR, Ty::U32, &[a, b]),
            "xor" => self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]),
            "xnor" => {
                let x = self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]);
                self.spv(op::NOT, Ty::U32, &[x])
            }
            "andn2" => {
                let nb = self.spv(op::NOT, Ty::U32, &[b]);
                self.spv(op::BITWISE_AND, Ty::U32, &[a, nb])
            }
            "orn2" => {
                let nb = self.spv(op::NOT, Ty::U32, &[b]);
                self.spv(op::BITWISE_OR, Ty::U32, &[a, nb])
            }
            "nand" => {
                let x = self.spv(op::BITWISE_AND, Ty::U32, &[a, b]);
                self.spv(op::NOT, Ty::U32, &[x])
            }
            "nor" => {
                let x = self.spv(op::BITWISE_OR, Ty::U32, &[a, b]);
                self.spv(op::NOT, Ty::U32, &[x])
            }
            _ => return None,
        })
    }

    fn clz(&mut self, a: V) -> V {
        let m = self.gl(glsl::FIND_U_MSB, Ty::U32, &[a]);
        let k = self.k32(31);
        let r = self.spv(op::I_SUB, Ty::U32, &[k, m]);
        let neg = self.k32(u32::MAX);
        let z = self.k32(0);
        let is = self.spv(op::I_EQUAL, Ty::BOOL, &[a, z]);
        self.spv(op::SELECT, Ty::U32, &[is, neg, r])
    }

    #[allow(clippy::too_many_arguments)]
    fn salu_arith(
        &mut self,
        inst: &Inst,
        name: &'static str,
        base: &str,
        dst: &Option<Operand>,
        src: &[Operand],
        imm: Option<u16>,
        b64: bool,
    ) -> Res<()> {
        let off = inst.machine.offset;
        let signed = name.ends_with("i32") || name.ends_with("i64");
        let a0 = |l: &mut Self| -> Res<V> {
            if imm.is_some() {
                let d = l.opnd(inst, &dst.unwrap())?;
                Ok(l.u32v(d))
            } else {
                let x = l.opnd(inst, &src[0])?;
                Ok(l.u32v(x))
            }
        };
        let b0 = |l: &mut Self| -> Res<V> {
            if let Some(k) = imm {
                Ok(l.k32(i32::from(k as i16) as u32))
            } else {
                let x = l.opnd(inst, &src[1])?;
                Ok(l.u32v(x))
            }
        };
        let z = self.k32(0);
        if b64 && matches!(base, "lshl" | "lshr" | "ashr" | "bfe") {
            let a = self.opnd_u64(inst, &src[0])?;
            let s = self.opnd(inst, &src[1])?;
            let s = self.u32v(s);
            let k63 = self.k32(63);
            let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k63]);
            let r = match base {
                "lshl" => self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U64, &[a, s]),
                "lshr" => self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U64, &[a, s]),
                "ashr" => self.spv(op::SHIFT_RIGHT_ARITHMETIC, Ty::U64, &[a, s]),
                _ => return Err(unsupported(off, name)),
            };
            let (lo, hi) = self.split64(r);
            self.sdst(inst, dst, 0, R::D(lo));
            self.sdst(inst, dst, 1, R::D(hi));
            let zz = self.f.konst(Ty::U64, 0);
            let nz = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[r, zz]);
            self.def(inst, Reg::Scc, R::B(nz));
            return Ok(());
        }
        if b64 && base == "bfm" {
            return Err(unsupported(off, name));
        }
        let a = a0(self)?;
        let b = b0(self)?;
        let (r, scc) = match base {
            "add" | "addk" | "addc" => {
                let mut x = self.spv(op::I_ADD, Ty::U32, &[a, b]);
                let mut carry = self.spv(op::U_LESS_THAN, Ty::BOOL, &[x, a]);
                if base == "addc" {
                    let r = self.val(inst, Reg::Scc)?;
                    let c = self.scc_of(r);
                    let cu = self.u32v(c);
                    let y = self.spv(op::I_ADD, Ty::U32, &[x, cu]);
                    let c2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[y, x]);
                    carry = self.spv(op::LOGICAL_OR, Ty::BOOL, &[carry, c2]);
                    x = y;
                }
                if signed {
                    // overflow: sign of a == sign of b != sign of result
                    let ax = self.spv(op::BITWISE_XOR, Ty::U32, &[a, x]);
                    let bx = self.spv(op::BITWISE_XOR, Ty::U32, &[b, x]);
                    let both = self.spv(op::BITWISE_AND, Ty::U32, &[ax, bx]);
                    carry = self.spv(op::S_LESS_THAN, Ty::BOOL, &[both, z]);
                }
                (x, Some(carry))
            }
            "sub" | "subb" => {
                let mut x = self.spv(op::I_SUB, Ty::U32, &[a, b]);
                let mut borrow = self.spv(op::U_LESS_THAN, Ty::BOOL, &[a, b]);
                if base == "subb" {
                    let r = self.val(inst, Reg::Scc)?;
                    let c = self.scc_of(r);
                    let cu = self.u32v(c);
                    let b2 = self.spv(op::U_LESS_THAN, Ty::BOOL, &[x, cu]);
                    x = self.spv(op::I_SUB, Ty::U32, &[x, cu]);
                    borrow = self.spv(op::LOGICAL_OR, Ty::BOOL, &[borrow, b2]);
                }
                if signed {
                    let ab = self.spv(op::BITWISE_XOR, Ty::U32, &[a, b]);
                    let ax = self.spv(op::BITWISE_XOR, Ty::U32, &[a, x]);
                    let both = self.spv(op::BITWISE_AND, Ty::U32, &[ab, ax]);
                    borrow = self.spv(op::S_LESS_THAN, Ty::BOOL, &[both, z]);
                }
                (x, Some(borrow))
            }
            "mul" | "mulk" => (self.spv(op::I_MUL, Ty::U32, &[a, b]), None),
            "mul_hi" => {
                let o = if signed { op::S_MUL_EXTENDED } else { op::U_MUL_EXTENDED };
                let s = self.spv(o, Ty::vec(S::U32, 2), &[a, b]);
                (self.extract(s, 1), None)
            }
            "min" | "max" => {
                let g = match (base, signed) {
                    ("min", false) => glsl::U_MIN,
                    ("min", true) => glsl::S_MIN,
                    ("max", false) => glsl::U_MAX,
                    _ => glsl::S_MAX,
                };
                let x = self.gl(g, Ty::U32, &[a, b]);
                // SCC: a was picked
                let c = self.spv(op::I_EQUAL, Ty::BOOL, &[x, a]);
                (x, Some(c))
            }
            "lshl" | "lshr" | "ashr" => {
                let k31 = self.k32(31);
                let s = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                let o = match base {
                    "lshl" => op::SHIFT_LEFT_LOGICAL,
                    "lshr" => op::SHIFT_RIGHT_LOGICAL,
                    _ => op::SHIFT_RIGHT_ARITHMETIC,
                };
                let x = self.spv(o, Ty::U32, &[a, s]);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "lshl1_add" | "lshl2_add" | "lshl3_add" | "lshl4_add" => {
                let n = u32::from(base.as_bytes()[4] - b'0');
                let k = self.k32(n);
                let s = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[a, k]);
                let x = self.spv(op::I_ADD, Ty::U32, &[s, b]);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "bfe" => {
                let k31 = self.k32(31);
                let o = self.spv(op::BITWISE_AND, Ty::U32, &[b, k31]);
                let k16 = self.k32(16);
                let w = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, k16]);
                let k7f = self.k32(0x7F);
                let w = self.spv(op::BITWISE_AND, Ty::U32, &[w, k7f]);
                let x = self.bfe(a, o, w, signed);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "bfm" => {
                let x = self.bfm(a, b);
                (x, None)
            }
            "absdiff" => {
                let d = self.spv(op::I_SUB, Ty::U32, &[a, b]);
                let x = self.gl(glsl::S_ABS, Ty::U32, &[d]);
                let c = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[x, z]);
                (x, Some(c))
            }
            "pack_ll" | "pack_lh" | "pack_hh" | "pack_hl" => {
                let k16 = self.k32(16);
                let m = self.k32(0xFFFF);
                let lo = if base.as_bytes()[5] == b'l' {
                    self.spv(op::BITWISE_AND, Ty::U32, &[a, m])
                } else {
                    self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, k16])
                };
                let hi = if base.as_bytes()[6] == b'l' {
                    self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[b, k16])
                } else {
                    let mh = self.k32(0xFFFF_0000);
                    self.spv(op::BITWISE_AND, Ty::U32, &[b, mh])
                };
                (self.spv(op::BITWISE_OR, Ty::U32, &[lo, hi]), None)
            }
            _ => return Err(unsupported(off, name)),
        };
        self.sdst(inst, dst, 0, R::D(r));
        if imm.is_some() && dst.is_none() {
            return Ok(());
        }
        if let Some(c) = scc {
            self.def(inst, Reg::Scc, R::B(c));
        }
        Ok(())
    }

    /// Bit-field extract with the hardware's semantics for any offset and
    /// width (masked to 5 bits / clamped).
    fn bfe(&mut self, a: V, o: V, w: V, signed: bool) -> V {
        if let (Some(oc), Some(wc)) = (self.f.as_const(o), self.f.as_const(w))
            && oc + wc <= 32
            && wc > 0
        {
            let opc = if signed { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
            return self.spv(opc, Ty::U32, &[a, o, w]);
        }
        if let Some(0) = self.f.as_const(w) {
            return self.k32(0);
        }
        // (a >> o) & mask(w), or the arithmetic form.
        let k32 = self.k32(32);
        let wc = self.gl(glsl::U_MIN, Ty::U32, &[w, k32]);
        if signed {
            // (a << (32 - o - w)) >> (32 - w)
            let ow = self.spv(op::I_ADD, Ty::U32, &[o, wc]);
            let ow = self.gl(glsl::U_MIN, Ty::U32, &[ow, k32]);
            let l = self.spv(op::I_SUB, Ty::U32, &[k32, ow]);
            let k31 = self.k32(31);
            let l = self.gl(glsl::U_MIN, Ty::U32, &[l, k31]);
            let sh = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[a, l]);
            let r = self.spv(op::I_SUB, Ty::U32, &[k32, wc]);
            let r = self.gl(glsl::U_MIN, Ty::U32, &[r, k31]);
            let r2 = self.spv(op::I_ADD, Ty::U32, &[r, k32]);
            let _ = r2;
            self.spv(op::SHIFT_RIGHT_ARITHMETIC, Ty::U32, &[sh, r])
        } else {
            let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[a, o]);
            let m = self.mask_bits(wc);
            self.spv(op::BITWISE_AND, Ty::U32, &[s, m])
        }
    }

    /// (1 << w) - 1 for w in 0..=32.
    fn mask_bits(&mut self, w: V) -> V {
        if let Some(c) = self.f.as_const(w) {
            return self.k32(if c >= 32 { u32::MAX } else { (1u32 << c) - 1 });
        }
        let one = self.k32(1);
        let k31 = self.k32(31);
        let wm = self.spv(op::BITWISE_AND, Ty::U32, &[w, k31]);
        let s = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[one, wm]);
        let m = self.spv(op::I_SUB, Ty::U32, &[s, one]);
        let k32 = self.k32(32);
        let big = self.spv(op::U_GREATER_THAN_EQUAL, Ty::BOOL, &[w, k32]);
        let all = self.k32(u32::MAX);
        self.spv(op::SELECT, Ty::U32, &[big, all, m])
    }

    fn bfm(&mut self, w: V, o: V) -> V {
        let k31 = self.k32(31);
        let w = self.spv(op::BITWISE_AND, Ty::U32, &[w, k31]);
        let o = self.spv(op::BITWISE_AND, Ty::U32, &[o, k31]);
        let m = self.mask_bits(w);
        self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[m, o])
    }

    // ---- scalar memory ----

    fn translate(&mut self, addr: V) -> V {
        self.emit(Op::Call(Helper::Translate), Ty::U64, vec![addr])
    }

    fn smem(&mut self, inst: &Inst, name: &'static str, data: Option<Operand>, base: Operand, offset: i32, soffset: Operand) -> Res<()> {
        let off = inst.machine.offset;
        if matches!(name, "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb") {
            return Ok(());
        }
        if name == "s_memtime" || name == "s_memrealtime" {
            let z = self.k32(0);
            self.sdst(inst, &data, 0, R::D(z));
            self.sdst(inst, &data, 1, R::D(z));
            return Ok(());
        }
        if !name.contains("_load_") {
            return Err(unsupported(off, name));
        }
        let Some(Operand::Scalar { code: d, dwords: n }) = data else {
            return Err(unsupported(off, name));
        };
        let so = self.opnd(inst, &soffset)?;
        let so = self.u32v(so);
        let k = self.k32(offset as u32);
        let o = self.spv(op::I_ADD, Ty::U32, &[so, k]);
        let words: Vec<V> = if name.starts_with("s_buffer_") {
            let vs = self.vsharp(inst, &base)?;
            let mut out = Vec::new();
            let mut at = 0u32;
            while at < u32::from(n) {
                let m = (u32::from(n) - at).min(4);
                let ka = self.k32(at * 4);
                let oo = self.spv(op::I_ADD, Ty::U32, &[o, ka]);
                let fl = self.floats(inst, (at..at + m).map(|k| Reg::S(d + k as u8)));
                let v = self.buffer_load_raw(&vs, oo, m, fl)?;
                for i in 0..m {
                    out.push(if m == 1 { v } else { self.extract(v, i) });
                }
                at += m;
            }
            out
        } else {
            let a = self.opnd_u64(inst, &base)?;
            let o64 = self.spv(op::U_CONVERT, Ty::U64, &[o]);
            let addr = self.spv(op::I_ADD, Ty::U64, &[a, o64]);
            let mut out = Vec::new();
            let mut at = 0u32;
            while at < u32::from(n) {
                let m = (u32::from(n) - at).min(4);
                let ka = self.f.konst(Ty::U64, u64::from(at * 4));
                let aa = self.spv(op::I_ADD, Ty::U64, &[addr, ka]);
                let dev = self.translate(aa);
                let fl = self.floats(inst, (at..at + m).map(|k| Reg::S(d + k as u8)));
                let ty = Ty::vec(if fl { S::F32 } else { S::U32 }, m as u8);
                let v = self.emit(Op::Load { align: 4 }, ty, vec![dev]);
                for i in 0..m {
                    out.push(if m == 1 { v } else { self.extract(v, i) });
                }
                at += m;
            }
            out
        };
        for (i, w) in words.into_iter().enumerate() {
            self.def(inst, Reg::S(d + i as u8), R::D(w));
        }
        Ok(())
    }

    // ---- buffers ----

    /// A V#'s parts: device base (u64), byte limit, stride.
    fn vsharp(&mut self, inst: &Inst, rsrc: &Operand) -> Res<VSharp> {
        let w: Vec<V> = (0..4)
            .map(|k| {
                let x = self.opnd_dw(inst, rsrc, k, false)?;
                Ok(self.u32v(x))
            })
            .collect::<Res<_>>()?;
        Ok(self.vsharp_of(&w))
    }

    fn vsharp_of(&mut self, w: &[V]) -> VSharp {
        // CSE across instructions in a block: the nodes repeat.
        let m16 = self.k32(0xFFFF);
        let hi = self.spv(op::BITWISE_AND, Ty::U32, &[w[1], m16]);
        let base = self.u64_of(w[0], hi);
        let k16 = self.k32(16);
        let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[w[1], k16]);
        let k = self.k32(0x3FFF);
        let stride = self.spv(op::BITWISE_AND, Ty::U32, &[s, k]);
        let one = self.k32(1);
        let st1 = self.gl(glsl::U_MAX, Ty::U32, &[stride, one]);
        let nr64 = self.spv(op::U_CONVERT, Ty::U64, &[w[2]]);
        let st64 = self.spv(op::U_CONVERT, Ty::U64, &[st1]);
        let bytes64 = self.spv(op::I_MUL, Ty::U64, &[nr64, st64]);
        let max = self.f.konst(Ty::U64, 0xFFFF_FFFF);
        let bytes64 = self.gl(glsl::U_MIN, Ty::U64, &[bytes64, max]);
        let bytes = self.spv(op::U_CONVERT, Ty::U32, &[bytes64]);
        // the device base, or the zero block when the bytes are in no range
        let dev = self.emit(Op::Call(Helper::TranslateRange), Ty::U64, vec![base, bytes]);
        let zero = self.fallback(false);
        let mapped = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[dev, zero]);
        let z = self.k32(0);
        let limit = self.spv(op::SELECT, Ty::U32, &[mapped, bytes, z]);
        VSharp { dev, limit, stride, w3: w[3] }
    }

    /// Device address for `bytes` at byte offset `o` of a V#, or the zero
    /// (loads) / sink (stores) block when out of bounds.
    fn buffer_addr(&mut self, vs: &VSharp, o: V, bytes: u32, store: bool) -> V {
        // in bounds: o + bytes <= limit, as o < (limit >= bytes ? limit - bytes + 1 : 0)
        // (the bound is shared by the accesses of a V#)
        let kb = self.k32(bytes);
        let room = self.spv(op::U_GREATER_THAN_EQUAL, Ty::BOOL, &[vs.limit, kb]);
        let k1 = self.k32(bytes.wrapping_sub(1));
        let lim = self.spv(op::I_SUB, Ty::U32, &[vs.limit, k1]);
        let z = self.k32(0);
        let lim = self.spv(op::SELECT, Ty::U32, &[room, lim, z]);
        let ok = self.spv(op::U_LESS_THAN, Ty::BOOL, &[o, lim]);
        let o64 = self.spv(op::U_CONVERT, Ty::U64, &[o]);
        let a = self.spv(op::I_ADD, Ty::U64, &[vs.dev, o64]);
        let fallback = self.fallback(store);
        self.spv(op::SELECT, Ty::U64, &[ok, a, fallback])
    }

    fn fallback(&mut self, store: bool) -> V {
        // the memory map's zero and sink blocks
        self.f.global(Op::MapField(u32::from(store)), Ty::U64)
    }

    /// As [`buffer_addr`](Self::buffer_addr) with a computed size.
    fn buffer_addr_dyn(&mut self, vs: &VSharp, o: V, kb: V, store: bool) -> V {
        let room = self.spv(op::U_GREATER_THAN_EQUAL, Ty::BOOL, &[vs.limit, kb]);
        let lim = self.spv(op::I_SUB, Ty::U32, &[vs.limit, kb]);
        let le = self.spv(op::U_LESS_THAN_EQUAL, Ty::BOOL, &[o, lim]);
        let ok = self.spv(op::LOGICAL_AND, Ty::BOOL, &[room, le]);
        let o64 = self.spv(op::U_CONVERT, Ty::U64, &[o]);
        let a = self.spv(op::I_ADD, Ty::U64, &[vs.dev, o64]);
        let fallback = self.fallback(store);
        self.spv(op::SELECT, Ty::U64, &[ok, a, fallback])
    }

    /// d16 results: values (f32 bits) as f16, two per dword.
    fn pack_d16(&mut self, vals: &[V]) -> Vec<V> {
        vals.chunks(2)
            .map(|pair| {
                let mut h = Vec::new();
                for &x in pair {
                    let f = self.f32v(x);
                    let hf = self.spv(op::F_CONVERT, Ty::F16, &[f]);
                    h.push(hf);
                }
                if h.len() == 1 {
                    h.push(self.f.konst(Ty::F16, 0));
                }
                let v2 = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::F16, 2), &h);
                self.spv(op::BITCAST, Ty::U32, &[v2])
            })
            .collect()
    }

    fn buffer_load_raw(&mut self, vs: &VSharp, o: V, n: u32, float: bool) -> Res<V> {
        let a = self.buffer_addr(vs, o, 4 * n, false);
        Ok(self.emit(Op::Load { align: 4 }, Ty::vec(if float { S::F32 } else { S::U32 }, n as u8), vec![a]))
    }

    /// Whether the values `inst` writes to `regs` are read as floats.
    fn floats(&self, inst: &Inst, regs: impl Iterator<Item = Reg>) -> bool {
        let mut any = false;
        for r in regs {
            match inst.write(r).and_then(|v| self.floaty.get(&v)) {
                Some(&x) if x > 0 => any = true,
                _ => return false,
            }
        }
        any
    }

    fn buffer(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Buffer { data, addr, rsrc, soffset, offset, offen, idxen, format, lds, tfe, .. } = inst.fields.clone()
        else {
            unreachable!()
        };
        if lds || tfe {
            return Err(unsupported(off, "a buffer access to LDS or with tfe"));
        }
        let vs = self.vsharp(inst, &rsrc)?;
        let so = self.opnd(inst, &soffset)?;
        let so = self.u32v(so);
        let ko = self.k32(u32::from(offset));
        let mut o = self.spv(op::I_ADD, Ty::U32, &[so, ko]);
        let mut k = 0u8;
        if idxen {
            let idx = self.opnd_dw(inst, &addr, 0, false)?;
            let idx = self.u32v(idx);
            let b = self.spv(op::I_MUL, Ty::U32, &[idx, vs.stride]);
            o = self.spv(op::I_ADD, Ty::U32, &[o, b]);
            k = 1;
        }
        if offen {
            let vo = self.opnd_dw(inst, &addr, k, false)?;
            let vo = self.u32v(vo);
            o = self.spv(op::I_ADD, Ty::U32, &[o, vo]);
        }
        let Some(Operand::Vector { reg, dwords }) = data else {
            return Err(unsupported(off, name));
        };
        let n = u32::from(dwords);
        if name.contains("_atomic_") {
            let glc = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let opc = atomic_op(name).ok_or(unsupported(off, name))?;
            let a = self.buffer_addr(&vs, o, 4, true);
            let x = self.opnd_dw(inst, &data.unwrap(), 0, false)?;
            let x = self.u32v(x);
            let mut args = vec![a, x];
            if name.contains("cmpswap") {
                let c = self.opnd_dw(inst, &data.unwrap(), 1, false)?;
                let c = self.u32v(c);
                args.push(c);
            }
            let r = self.emit(Op::Atomic(opc), Ty::U32, args);
            if glc {
                self.def(inst, Reg::V(reg), R::D(r));
            }
            return Ok(());
        }
        // element size and conversion
        let fmt: Option<u8> = format;
        let load = name.contains("_load_");
        let (size, kind) = if name.contains("_format_") {
            let ch = name.rsplit('_').next().unwrap().len() as u32; // x, xy, xyz, xyzw
            let d16 = name.contains("_d16_");
            match fmt {
                Some(f) => {
                    let Some((bytes, kind)) = tformat(f) else {
                        if std::env::var_os("VK_FMT").is_some() {
                            eprintln!("typed buffer format {f} ({name})");
                        }
                        return Err(unsupported(off, "a typed buffer format"));
                    };
                    (bytes, Kind::T(kind, ch, d16))
                }
                None if load => (0, Kind::VFormat(ch, d16)),
                None => {
                    // stores through a V# format: 32-bit channels
                    self.approx("a typed buffer store written as 32-bit channels");
                    if d16 {
                        (4 * ch, Kind::Halves(ch))
                    } else {
                        (4 * ch, Kind::Dwords(ch))
                    }
                }
            }
        } else if name.ends_with("_ubyte") || name.ends_with("_sbyte") || name.ends_with("_byte") {
            (1, Kind::Byte(name.ends_with("_sbyte")))
        } else if name.ends_with("_ushort") || name.ends_with("_sshort") || name.ends_with("_short") {
            (2, Kind::Short(name.ends_with("_sshort")))
        } else if name.contains("_d16") {
            return Err(unsupported(off, "a d16 buffer access"));
        } else {
            (4 * n, Kind::Dwords(n))
        };
        if let Kind::VFormat(ch, d16) = kind {
            // the format from the V#, decoded by a helper; bounds by the stride
            let four = self.k32(4);
            let bytes = self.gl(glsl::U_MAX, Ty::U32, &[vs.stride, four]);
            let a = self.buffer_addr_dyn(&vs, o, bytes, false);
            let k12 = self.k32(12);
            let f = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[vs.w3, k12]);
            let k7f = self.k32(0x7F);
            let f = self.spv(op::BITWISE_AND, Ty::U32, &[f, k7f]);
            let r = self.emit(Op::Call(Helper::Typed), Ty::vec(S::U32, 4), vec![a, f]);
            let vals: Vec<V> = (0..ch).map(|i| self.extract(r, i)).collect();
            let vals = if d16 { self.pack_d16(&vals) } else { vals };
            for (i, v) in vals.into_iter().enumerate() {
                self.def(inst, Reg::V(reg + i as u8), R::D(v));
            }
            return Ok(());
        }
        if load {
            let a = self.buffer_addr(&vs, o, size, false);
            let kind = match kind {
                Kind::Dwords(n) if self.floats(inst, (0..n).map(|k| Reg::V(reg + k as u8))) => Kind::Floats(n),
                k => k,
            };
            let vals = self.load_kind(a, kind)?;
            for (i, v) in vals.into_iter().enumerate() {
                self.def(inst, Reg::V(reg + i as u8), R::D(v));
            }
        } else {
            let a = self.buffer_addr(&vs, o, size, true);
            let vals: Vec<V> =
                (0..n).map(|i| self.opnd_dw(inst, &data.unwrap(), i as u8, false)).collect::<Res<_>>()?;
            self.store_kind(a, kind, &vals)?;
        }
        Ok(())
    }

    fn load_kind(&mut self, a: V, kind: Kind) -> Res<Vec<V>> {
        Ok(match kind {
            Kind::Dwords(n) | Kind::Floats(n) => {
                let s = if matches!(kind, Kind::Floats(_)) { S::F32 } else { S::U32 };
                let v = self.emit(Op::Load { align: 4 }, Ty::vec(s, n as u8), vec![a]);
                if n == 1 { vec![v] } else { (0..n).map(|i| self.extract(v, i)).collect() }
            }
            Kind::Byte(s) => {
                let v = self.emit(Op::Load { align: 1 }, Ty::U8, vec![a]);
                let o = if s { op::S_CONVERT } else { op::U_CONVERT };
                vec![self.spv(o, Ty::U32, &[v])]
            }
            Kind::Short(s) => {
                let v = self.emit(Op::Load { align: 2 }, Ty::U16, vec![a]);
                let o = if s { op::S_CONVERT } else { op::U_CONVERT };
                vec![self.spv(o, Ty::U32, &[v])]
            }
            Kind::T(t, ch, d16) => {
                let vals = self.typed_load(a, t, ch);
                if d16 { self.pack_d16(&vals) } else { vals }
            }
            Kind::VFormat(..) | Kind::Halves(_) => return Err(unsupported(0, "a typed buffer access")),
        })
    }

    fn store_kind(&mut self, a: V, kind: Kind, vals: &[V]) -> Res<()> {
        match kind {
            Kind::Dwords(n) => {
                // floats stay floats when every dword is one
                let float = vals.iter().all(|&x| self.f.ty(x) == Ty::F32);
                let vals: Vec<V> = vals.iter().map(|&x| if float { x } else { self.u32v(x) }).collect();
                let v = if n == 1 {
                    vals[0]
                } else {
                    self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(if float { S::F32 } else { S::U32 }, n as u8), &vals)
                };
                self.emit(Op::Store { align: 4 }, Ty::VOID, vec![a, v]);
            }
            Kind::Byte(_) => {
                let x = self.u32v(vals[0]);
                let v = self.spv(op::U_CONVERT, Ty::U8, &[x]);
                self.emit(Op::Store { align: 1 }, Ty::VOID, vec![a, v]);
            }
            Kind::Short(_) => {
                let x = self.u32v(vals[0]);
                let v = self.spv(op::U_CONVERT, Ty::U16, &[x]);
                self.emit(Op::Store { align: 2 }, Ty::VOID, vec![a, v]);
            }
            Kind::Halves(n) => {
                // two f16 per data dword, written as 32-bit floats
                let mut out = Vec::new();
                for k in 0..n {
                    let w = self.u32v(vals[(k / 2) as usize]);
                    let f = self.half_f32(w, k % 2 == 1);
                    out.push(self.u32v(f));
                }
                let v = if n == 1 { out[0] } else { self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, n as u8), &out) };
                self.emit(Op::Store { align: 4 }, Ty::VOID, vec![a, v]);
            }
            Kind::T(..) | Kind::Floats(_) | Kind::VFormat(..) => return Err(unsupported(0, "a typed buffer store")),
        }
        Ok(())
    }

    /// A typed element read from its bytes (the format in the instruction).
    fn typed_load(&mut self, a: V, t: TFormat, ch: u32) -> Vec<V> {
        if t.bits == 0 {
            return self.packed_load(a, t, ch);
        }
        let (comps, bits, num) = (t.comps, t.bits, t.num);
        let mut out = Vec::new();
        let raw: Vec<V> = match bits {
            32 => {
                let v = self.emit(Op::Load { align: 4 }, Ty::vec(S::U32, comps), vec![a]);
                if comps == 1 { vec![v] } else { (0..u32::from(comps)).map(|i| self.extract(v, i)).collect() }
            }
            16 => {
                let v = self.emit(Op::Load { align: 2 }, Ty::vec(S::U16, comps), vec![a]);
                (0..u32::from(comps))
                    .map(|i| {
                        let e = if comps == 1 { v } else { self.extract(v, i) };
                        self.spv(if num == Num::Sint || num == Num::Snorm { op::S_CONVERT } else { op::U_CONVERT }, Ty::U32, &[e])
                    })
                    .collect()
            }
            _ => {
                let v = self.emit(Op::Load { align: 1 }, Ty::vec(S::U8, comps), vec![a]);
                (0..u32::from(comps))
                    .map(|i| {
                        let e = if comps == 1 { v } else { self.extract(v, i) };
                        self.spv(if num == Num::Sint || num == Num::Snorm { op::S_CONVERT } else { op::U_CONVERT }, Ty::U32, &[e])
                    })
                    .collect()
            }
        };
        for c in 0..ch {
            let x = if (c as usize) < raw.len() {
                let r = raw[c as usize];
                match num {
                    Num::Uint | Num::Sint => r,
                    Num::Float if bits == 32 => r,
                    Num::Float => {
                        // f16 bits to f32
                        let h = self.spv(op::U_CONVERT, Ty::U16, &[r]);
                        let hf = self.spv(op::BITCAST, Ty::F16, &[h]);
                        let f = self.spv(op::F_CONVERT, Ty::F32, &[hf]);
                        self.u32v(f)
                    }
                    Num::Unorm => {
                        let f = self.spv(op::CONVERT_U_TO_F, Ty::F32, &[r]);
                        let k = self.kf(1.0 / ((1u64 << bits) - 1) as f32);
                        let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                        self.u32v(m)
                    }
                    Num::Snorm => {
                        let f = self.spv(op::CONVERT_S_TO_F, Ty::F32, &[r]);
                        let k = self.kf(1.0 / ((1u64 << (bits - 1)) - 1) as f32);
                        let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                        let lo = self.kf(-1.0);
                        let c = self.gl(glsl::N_MAX, Ty::F32, &[m, lo]);
                        self.u32v(c)
                    }
                }
            } else {
                // missing channels: 0, alpha 1
                if c == 3 {
                    match num {
                        Num::Uint | Num::Sint => self.k32(1),
                        _ => self.k32(1.0f32.to_bits()),
                    }
                } else {
                    self.k32(0)
                }
            };
            out.push(x);
        }
        out
    }

    /// A 2_10_10_10 element (x in the low bits): one dword.
    fn packed_load(&mut self, a: V, t: TFormat, ch: u32) -> Vec<V> {
        let w = self.emit(Op::Load { align: 4 }, Ty::U32, vec![a]);
        let widths = [10u32, 10, 10, 2];
        let mut out = Vec::new();
        let mut at = 0u32;
        for (c, &bits) in widths.iter().enumerate() {
            let (o, wv) = (self.k32(at), self.k32(bits));
            at += bits;
            if c as u32 >= ch {
                continue;
            }
            let signed = matches!(t.num, Num::Snorm | Num::Sint);
            let opc = if signed { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
            let r = self.spv(opc, Ty::U32, &[w, o, wv]);
            let x = match t.num {
                Num::Uint | Num::Sint => r,
                Num::Unorm => {
                    let f = self.spv(op::CONVERT_U_TO_F, Ty::F32, &[r]);
                    let k = self.kf(1.0 / ((1u64 << bits) - 1) as f32);
                    let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                    self.u32v(m)
                }
                _ => {
                    let f = self.spv(op::CONVERT_S_TO_F, Ty::F32, &[r]);
                    let k = self.kf(1.0 / ((1u64 << (bits - 1)) - 1) as f32);
                    let m = self.spv(op::F_MUL, Ty::F32, &[f, k]);
                    let lo = self.kf(-1.0);
                    let c = self.gl(glsl::N_MAX, Ty::F32, &[m, lo]);
                    self.u32v(c)
                }
            };
            out.push(x);
        }
        out
    }

    // ---- LDS ----

    fn lds_index(&mut self, inst: &Inst, addr: &Option<Operand>, off: u32) -> Res<V> {
        let a = match addr {
            Some(o) => {
                let x = self.opnd(inst, o)?;
                self.u32v(x)
            }
            None => self.k32(0),
        };
        let k = self.k32(off);
        let b = self.spv(op::I_ADD, Ty::U32, &[a, k]);
        let two = self.k32(2);
        Ok(self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, two]))
    }

    fn ds(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Ds { dst, addr, data, offset0, offset1, gds } = inst.fields.clone() else { unreachable!() };
        if gds {
            return Err(unsupported(off, "GDS"));
        }
        let o16 = u32::from(offset0) | u32::from(offset1) << 8;
        let dreg = |d: &Option<Operand>| match d {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        match name {
            "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" => {
                let n = match name {
                    "ds_read_b32" => 1,
                    "ds_read_b64" => 2,
                    "ds_read_b96" => 3,
                    _ => 4,
                };
                let i = self.lds_index(inst, &addr, o16)?;
                let d = dreg(&dst).ok_or(unsupported(off, name))?;
                for k in 0..n {
                    let kk = self.k32(k);
                    let ii = self.spv(op::I_ADD, Ty::U32, &[i, kk]);
                    let v = self.emit(Op::LdsLoad, Ty::U32, vec![ii]);
                    self.def(inst, Reg::V(d + k as u8), R::D(v));
                }
            }
            "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
                let st = if name.contains("st64") { 64 } else { 1 };
                let w = if name.ends_with("b64") { 2 } else { 1 };
                let d = dreg(&dst).ok_or(unsupported(off, name))?;
                for (j, o) in [offset0, offset1].into_iter().enumerate() {
                    let i = self.lds_index(inst, &addr, u32::from(o) * 4 * w * st)?;
                    for k in 0..w {
                        let kk = self.k32(k);
                        let ii = self.spv(op::I_ADD, Ty::U32, &[i, kk]);
                        let v = self.emit(Op::LdsLoad, Ty::U32, vec![ii]);
                        self.def(inst, Reg::V(d + (j as u32 * w + k) as u8), R::D(v));
                    }
                }
            }
            "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" => {
                let i = self.lds_index(inst, &addr, o16)?;
                let Some(Operand::Vector { dwords, .. }) = data.first().copied() else {
                    return Err(unsupported(off, name));
                };
                for k in 0..dwords {
                    let x = self.opnd_dw(inst, &data[0], k, false)?;
                    let x = self.u32v(x);
                    let kk = self.k32(u32::from(k));
                    let ii = self.spv(op::I_ADD, Ty::U32, &[i, kk]);
                    self.emit(Op::LdsStore, Ty::VOID, vec![ii, x]);
                }
            }
            "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
                let st = if name.contains("st64") { 64 } else { 1 };
                let w: u32 = if name.ends_with("b64") { 2 } else { 1 };
                for (j, o) in [offset0, offset1].into_iter().enumerate() {
                    let i = self.lds_index(inst, &addr, u32::from(o) * 4 * w * st)?;
                    for k in 0..w {
                        let x = self.opnd_dw(inst, &data[j], k as u8, false)?;
                        let x = self.u32v(x);
                        let kk = self.k32(k);
                        let ii = self.spv(op::I_ADD, Ty::U32, &[i, kk]);
                        self.emit(Op::LdsStore, Ty::VOID, vec![ii, x]);
                    }
                }
            }
            "ds_read_u16" | "ds_read_u8" | "ds_read_i16" | "ds_read_i8" => {
                let a = match &addr {
                    Some(o) => {
                        let x = self.opnd(inst, o)?;
                        self.u32v(x)
                    }
                    None => self.k32(0),
                };
                let k = self.k32(o16);
                let b = self.spv(op::I_ADD, Ty::U32, &[a, k]);
                let two = self.k32(2);
                let i = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, two]);
                let w = self.emit(Op::LdsLoad, Ty::U32, vec![i]);
                let three = self.k32(3);
                let bo = self.spv(op::BITWISE_AND, Ty::U32, &[b, three]);
                let sh = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[bo, three]);
                let width = self.k32(if name.ends_with("16") { 16 } else { 8 });
                let opc = if name.contains("_i") { op::BIT_FIELD_S_EXTRACT } else { op::BIT_FIELD_U_EXTRACT };
                let v = self.spv(opc, Ty::U32, &[w, sh, width]);
                let d = dreg(&dst).ok_or(unsupported(off, name))?;
                self.def(inst, Reg::V(d), R::D(v));
            }
            "ds_write_b16" | "ds_write_b8" => {
                // a half or byte of a dword: clear it, then set it (two
                // atomics, so lanes writing the other bytes are kept)
                let a = match &addr {
                    Some(o) => {
                        let x = self.opnd(inst, o)?;
                        self.u32v(x)
                    }
                    None => self.k32(0),
                };
                let k = self.k32(o16);
                let b = self.spv(op::I_ADD, Ty::U32, &[a, k]);
                let two = self.k32(2);
                let i = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[b, two]);
                let three = self.k32(3);
                let bo = self.spv(op::BITWISE_AND, Ty::U32, &[b, three]);
                let sh = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[bo, three]);
                let m = self.k32(if name.ends_with("16") { 0xFFFF } else { 0xFF });
                let x = self.opnd_dw(inst, &data[0], 0, false)?;
                let x = self.u32v(x);
                let x = self.spv(op::BITWISE_AND, Ty::U32, &[x, m]);
                let xs = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[x, sh]);
                let ms = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[m, sh]);
                let keep = self.spv(op::NOT, Ty::U32, &[ms]);
                self.emit(Op::LdsAtomic(op::ATOMIC_AND), Ty::U32, vec![i, keep]);
                self.emit(Op::LdsAtomic(op::ATOMIC_OR), Ty::U32, vec![i, xs]);
            }
            _ if name.starts_with("ds_") && atomic_op(name).is_some() => {
                let opc = atomic_op(name).unwrap();
                let i = self.lds_index(inst, &addr, o16)?;
                let x = self.opnd_dw(inst, &data[0], 0, false)?;
                let x = self.u32v(x);
                let mut args = vec![i, x];
                if name.contains("cmpst") {
                    // ds_cmpst: data0 = compare, data1 = new
                    let y = self.opnd_dw(inst, &data[1], 0, false)?;
                    let y = self.u32v(y);
                    args = vec![i, y, x];
                }
                let r = self.emit(Op::LdsAtomic(opc), Ty::U32, args);
                if let Some(d) = dreg(&dst) {
                    self.def(inst, Reg::V(d), R::D(r));
                }
            }
            _ => return Err(unsupported(off, name)),
        }
        Ok(())
    }

    fn ds_cross(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Ds { dst, addr, data, offset0, offset1, .. } = inst.fields.clone() else { unreachable!() };
        let Some(Operand::Vector { reg: d, .. }) = dst else { return Err(unsupported(off, name)) };
        let a = self.active(inst)?;
        let lane = self.lane();
        let (src, from) = match name {
            "ds_swizzle_b32" => {
                let x = self.opnd(inst, addr.as_ref().or(data.first()).ok_or(unsupported(off, name))?)?;
                let x = self.u32v(x);
                let pat = u32::from(offset0) | u32::from(offset1) << 8;
                let from = if pat & 0x8000 != 0 {
                    // quad permute
                    let k3 = self.k32(3);
                    let base = self.k32(!3);
                    let qb = self.spv(op::BITWISE_AND, Ty::U32, &[lane, base]);
                    let q = self.spv(op::BITWISE_AND, Ty::U32, &[lane, k3]);
                    // selector by q: (pat >> (2q)) & 3
                    let one = self.k32(1);
                    let q2 = self.spv(op::SHIFT_LEFT_LOGICAL, Ty::U32, &[q, one]);
                    let kp = self.k32(pat & 0xFF);
                    let s = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[kp, q2]);
                    let s = self.spv(op::BITWISE_AND, Ty::U32, &[s, k3]);
                    self.spv(op::BITWISE_OR, Ty::U32, &[qb, s])
                } else {
                    // bitmask mode within groups of 32
                    let and = pat & 0x1F;
                    let or = pat >> 5 & 0x1F;
                    let xor = pat >> 10 & 0x1F;
                    let ka = self.k32(and | !0x1F);
                    let ko = self.k32(or);
                    let kx = self.k32(xor);
                    let t = self.spv(op::BITWISE_AND, Ty::U32, &[lane, ka]);
                    let t = self.spv(op::BITWISE_OR, Ty::U32, &[t, ko]);
                    self.spv(op::BITWISE_XOR, Ty::U32, &[t, kx])
                };
                (x, from)
            }
            _ => {
                // ds_bpermute_b32: dst = src[addr / 4]
                let x = self.opnd(inst, &data[0])?;
                let x = self.u32v(x);
                let ad = self.opnd(inst, addr.as_ref().unwrap())?;
                let ad = self.u32v(ad);
                let o = self.k32(u32::from(offset0) | u32::from(offset1) << 8);
                let ad = self.spv(op::I_ADD, Ty::U32, &[ad, o]);
                let two = self.k32(2);
                let l = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[ad, two]);
                let k31 = self.k32(31);
                if name == "ds_permute_b32" {
                    return Err(unsupported(off, name));
                }
                (x, self.spv(op::BITWISE_AND, Ty::U32, &[l, k31]))
            }
        };
        let k31 = self.k32(31);
        let from = self.spv(op::BITWISE_AND, Ty::U32, &[from, k31]);
        let v = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![src, from]);
        // an inactive source lane reads 0
        let src_active = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::BOOL, vec![a, from]);
        let z = self.k32(0);
        let v = self.spv(op::SELECT, Ty::U32, &[src_active, v, z]);
        let old = self.val(inst, Reg::V(d))?;
        let old = self.data(old);
        let old = self.u32v(old);
        let x = self.spv(op::SELECT, Ty::U32, &[a, v, old]);
        self.def(inst, Reg::V(d), R::D(x));
        Ok(())
    }

    // ---- flat / global ----

    fn flat(&mut self, inst: &Inst, name: &'static str) -> Res<()> {
        let off = inst.machine.offset;
        let Fields::Flat { dst, addr, data, saddr, offset, lds, .. } = inst.fields.clone() else { unreachable!() };
        if lds || name.starts_with("scratch_") {
            return Err(unsupported(off, name));
        }
        let mut a = match (saddr, addr) {
            (Some(s @ Operand::Scalar { code, .. }), Some(v)) if code != crate::gfx10::NULL => {
                let base = self.opnd_u64(inst, &s)?;
                let vo = self.opnd(inst, &v)?;
                let vo = self.u32v(vo);
                let v64 = self.spv(op::U_CONVERT, Ty::U64, &[vo]);
                self.spv(op::I_ADD, Ty::U64, &[base, v64])
            }
            (_, Some(v)) => self.opnd_u64(inst, &v)?,
            _ => return Err(unsupported(off, name)),
        };
        let ko = self.f.konst(Ty::U64, i64::from(offset) as u64);
        a = self.spv(op::I_ADD, Ty::U64, &[a, ko]);
        let dev = self.translate(a);
        let kind = if name.ends_with("_ubyte") {
            Kind::Byte(false)
        } else if name.ends_with("_sbyte") {
            Kind::Byte(true)
        } else if name.ends_with("_ushort") {
            Kind::Short(false)
        } else if name.ends_with("_sshort") {
            Kind::Short(true)
        } else if name.contains("_atomic_") {
            let opc = atomic_op(name).ok_or(unsupported(off, name))?;
            let x = self.opnd_dw(inst, &data.unwrap(), 0, false)?;
            let x = self.u32v(x);
            let r = self.emit(Op::Atomic(opc), Ty::U32, vec![dev, x]);
            if let Some(Operand::Vector { reg, .. }) = dst {
                self.def(inst, Reg::V(reg), R::D(r));
            }
            return Ok(());
        } else {
            let n = match name.rsplit('_').next().unwrap() {
                "dword" => 1,
                "dwordx2" => 2,
                "dwordx3" => 3,
                "dwordx4" => 4,
                _ => return Err(unsupported(off, name)),
            };
            Kind::Dwords(n)
        };
        if name.contains("_load_") {
            let Some(Operand::Vector { reg, .. }) = dst else { return Err(unsupported(off, name)) };
            let vals = self.load_kind(dev, kind)?;
            for (i, v) in vals.into_iter().enumerate() {
                self.def(inst, Reg::V(reg + i as u8), R::D(v));
            }
        } else {
            let Some(d @ Operand::Vector { dwords, .. }) = data else { return Err(unsupported(off, name)) };
            let vals: Vec<V> = (0..dwords).map(|i| self.opnd_dw(inst, &d, i, false)).collect::<Res<_>>()?;
            self.store_kind(dev, kind, &vals)?;
        }
        Ok(())
    }

    // ---- export ----

    fn export(&mut self, inst: &Inst) -> Res<()> {
        let off = inst.machine.offset;
        let e = self
            .io
            .exports
            .iter()
            .find(|e| e.site == self.site)
            .cloned()
            .ok_or(unsupported(off, "an export io does not list"))?;
        if e.compressed {
            return Err(unsupported(off, "a compressed export"));
        }
        let slot: u16 = match e.target {
            Target::Color(n) => u16::from(n),
            Target::Position(0) => OUT_POSITION,
            Target::Position(_) => {
                self.approx("clip and cull distances (pos1+ exports) left out");
                return Ok(());
            }
            Target::Param { row, .. } => u16::from(row),
            Target::Null | Target::Unread => return Ok(()),
            Target::Depth => {
                self.iface.depth = true;
                // channel 0 depth, 1 stencil, 2 sample mask
                for (c, v) in e.values.iter().enumerate() {
                    if let Some(v) = v {
                        let r = self.value(*v, off)?;
                        let x = self.data(r);
                        let slot = match c {
                            0 => OUT_DEPTH,
                            1 => OUT_STENCIL,
                            2 => OUT_SAMPLE_MASK,
                            _ => continue,
                        };
                        self.emit(Op::Output { slot, comp: 0 }, Ty::VOID, vec![x]);
                        self.note_output(slot, 0);
                    }
                }
                return Ok(());
            }
            Target::Primitive => OUT_PRIMITIVE,
            Target::PrimitiveParam { .. } => return Err(unsupported(off, "a per-primitive output")),
        };
        // mesh: vertex and primitive n are written by thread n of the group
        let index = (self.native.stage == HwStage::Ms)
            .then(|| self.f.global(Op::Builtin(Builtin::LocalInvocationIndex), Ty::U32));
        let slot = match e.target {
            Target::Param { usage: crate::native::Usage::Other(0), .. } if self.native.stage == HwStage::Ms => {
                u16::from(self.unnamed_row.ok_or(unsupported(off, "a mesh parameter without a row"))?)
            }
            _ => slot,
        };
        // 16-bit outputs: each dword carries two halves, each a column of a row
        if matches!(e.target, Target::Param { .. }) && self.halves.keys().any(|k| u16::from(k.0) == slot) {
            for (c, v) in e.values.iter().enumerate() {
                let Some(v) = v else { continue };
                let r = self.value(*v, off)?;
                let x = self.data(r);
                let x = self.u32v(x);
                for high in [false, true] {
                    let Some(&(row, col)) = self.halves.get(&(slot as u8, 2 * c as u8 + u8::from(high))) else {
                        continue;
                    };
                    let f = self.half_f32(x, high);
                    let mut args = vec![f];
                    args.extend(index);
                    self.emit(Op::Output { slot: u16::from(row), comp: col }, Ty::VOID, args);
                    self.note_output(u16::from(row), col);
                }
            }
            return Ok(());
        }
        for (c, v) in e.values.iter().enumerate() {
            if let Some(v) = v {
                let r = self.value(*v, off)?;
                let x = self.data(r);
                let mut args = vec![x];
                args.extend(index);
                self.emit(Op::Output { slot, comp: c as u8 }, Ty::VOID, args);
                self.note_output(slot, c as u8);
                if slot == OUT_PRIMITIVE {
                    break;
                }
            }
        }
        Ok(())
    }

    fn note_output(&mut self, slot: u16, comp: u8) {
        if let Some(o) = self.iface.outputs.iter_mut().find(|o| o.0 == slot) {
            o.1 |= 1 << comp;
        } else {
            self.iface.outputs.push((slot, 1 << comp));
        }
    }

    // ---- interpolation ----

    fn interp(&mut self, inst: &Inst, dst: Operand) -> Res<()> {
        let off = inst.machine.offset;
        let Operand::Vector { reg, .. } = dst else { return Err(unsupported(off, "an interpolation")) };
        let Some(&interp) = self.interps.get(&self.site) else {
            // first half of a pair: nothing needs its value
            let z = self.k32(0);
            self.def(inst, Reg::V(reg), R::D(z));
            return Ok(());
        };
        let (attr, mode) = match interp {
            io::Interp::Interpolated { attribute, barycentric, .. } => {
                use io::Barycentric as B;
                let (perspective, centroid, sample) = match barycentric {
                    B::PerspectiveCenter => (true, false, false),
                    B::PerspectiveCentroid => (true, true, false),
                    B::PerspectiveSample => (true, false, true),
                    B::LinearCenter => (false, false, false),
                    B::LinearCentroid => (false, true, false),
                    B::LinearSample => (false, false, true),
                    B::PerspectivePullModel => return Err(unsupported(off, "pull-model interpolation")),
                };
                (*attribute, Interp::Smooth { perspective, centroid, sample })
            }
            io::Interp::Vertex { attribute, vertex: 2 } => (*attribute, Interp::Flat),
            _ => return Err(unsupported(off, "an interpolation of vertex values")),
        };
        let name = inst.name();
        let f16 = name.contains("f16");
        // A 16-bit value: half 2c (+1 high) of a packed parameter, one
        // column of one signature row.
        let (loc, comp) = if f16 {
            let h = 2 * attr.channel + u8::from(attr.high);
            *self.halves.get(&(attr.row, h)).ok_or(unsupported(off, "a 16-bit read of a 32-bit input"))?
        } else if attr.high {
            return Err(unsupported(off, "a high half of a 32-bit input"));
        } else {
            (attr.row, attr.channel)
        };
        if let Some(e) = self.iface.inputs.iter().find(|e| e.0 == loc) {
            if e.1 != mode {
                return Err(unsupported(off, "one attribute interpolated two ways"));
            }
        } else {
            self.iface.inputs.push((loc, mode));
        }
        let x = self.f.global(Op::Input { loc, comp, interp: mode }, Ty::F32);
        if f16 {
            // an f16 result into a half of the destination
            let h = self.spv(op::F_CONVERT, Ty::F16, &[x]);
            if let Fields::Valu(v) = &inst.fields {
                let v = v.clone();
                return self.write_half(inst, &v, h);
            }
            let hb = self.spv(op::BITCAST, Ty::U16, &[h]);
            let x = self.spv(op::U_CONVERT, Ty::U32, &[hb]);
            self.def(inst, Reg::V(reg), R::D(x));
            return Ok(());
        }
        self.def(inst, Reg::V(reg), R::D(x));
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct VSharp {
    dev: V,
    limit: V,
    stride: V,
    w3: V,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Num {
    Unorm,
    Snorm,
    Uint,
    Sint,
    Float,
}

#[derive(Debug, Clone, Copy)]
struct TFormat {
    comps: u8,
    /// Bits per component; 0: a packed 2_10_10_10 dword.
    bits: u8,
    num: Num,
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    /// A typed load through the V#'s format (channels, d16).
    VFormat(u32, bool),
    /// d16 typed store data written as 32-bit floats.
    Halves(u32),
    Dwords(u32),
    Floats(u32),
    Byte(bool),
    Short(bool),
    T(TFormat, u32, bool),
}

/// GFX10 buffer format codes (the unified FORMAT field of MTBUF / the V#):
/// element bytes and decoding.
fn tformat(f: u8) -> Option<(u32, TFormat)> {
    use Num::*;
    let t = |comps, bits, num| Some((u32::from(comps) * u32::from(bits) / 8, TFormat { comps, bits, num }));
    let packed = |num| Some((4, TFormat { comps: 4, bits: 0, num }));
    match f {
        50 => packed(Unorm),
        51 => packed(Snorm),
        54 => packed(Uint),
        55 => packed(Sint),
        1 => t(1, 8, Unorm),
        2 => t(1, 8, Snorm),
        5 => t(1, 8, Uint),
        6 => t(1, 8, Sint),
        7 => t(1, 16, Unorm),
        8 => t(1, 16, Snorm),
        11 => t(1, 16, Uint),
        12 => t(1, 16, Sint),
        13 => t(1, 16, Float),
        14 => t(2, 8, Unorm),
        15 => t(2, 8, Snorm),
        18 => t(2, 8, Uint),
        19 => t(2, 8, Sint),
        20 => t(1, 32, Uint),
        21 => t(1, 32, Sint),
        22 => t(1, 32, Float),
        23 => t(2, 16, Unorm),
        24 => t(2, 16, Snorm),
        27 => t(2, 16, Uint),
        28 => t(2, 16, Sint),
        29 => t(2, 16, Float),
        56 => t(4, 8, Unorm),
        57 => t(4, 8, Snorm),
        60 => t(4, 8, Uint),
        61 => t(4, 8, Sint),
        62 => t(2, 32, Uint),
        63 => t(2, 32, Sint),
        64 => t(2, 32, Float),
        65 => t(4, 16, Unorm),
        66 => t(4, 16, Snorm),
        69 => t(4, 16, Uint),
        70 => t(4, 16, Sint),
        71 => t(4, 16, Float),
        72 => t(3, 32, Uint),
        73 => t(3, 32, Sint),
        74 => t(3, 32, Float),
        75 => t(4, 32, Uint),
        76 => t(4, 32, Sint),
        77 => t(4, 32, Float),
        _ => None,
    }
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

// VALU and images live in their own files.
#[path = "lift_valu.rs"]
mod valu;
#[path = "lift_image.rs"]
mod image;
