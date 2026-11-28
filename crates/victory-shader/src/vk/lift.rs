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
use crate::native::{HwStage, NativeProgram, StageIo};

/// Which half of a console wave64 mask a register dword holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// An approximation the translation took (`docs/vulkan-abi.md` section 9):
/// what, and the opcode of the instruction it was taken at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Approx {
    pub note: &'static str,
    /// Decoder name; empty when no instruction is being lifted.
    pub opcode: &'static str,
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
    pub approximate: Vec<Approx>,
    /// Approximations that hold unless the value read (the first) is
    /// wave-uniform (then the translation is exact: [`super::uniform`]
    /// decides) or the value made of it (the second) is read by nothing
    /// the program does (a step of a whole-wave section whose result
    /// became a subgroup reduction).
    pub unless_uniform: Vec<(V, V, Approx)>,
    /// Pixel: depth / stencil / sample mask exported.
    pub depth: bool,
    pub demotes: bool,
    /// Reads barycentrics or per-vertex values (fragmentShaderBarycentric).
    pub per_vertex: bool,
    /// Clip and cull distances (`PA_CL_VS_OUT_CNTL`): how many of each.
    pub clip_cull: (u8, u8),
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
/// Clip and cull distances (component: the index in the array).
pub const OUT_CLIP: u16 = 0x105;
pub const OUT_CULL: u16 = 0x106;
pub const OUT_POINT_SIZE: u16 = 0x107;
pub const OUT_LAYER: u16 = 0x108;
pub const OUT_VIEWPORT: u16 = 0x109;

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
    /// A store (or other effect) is inside: memory reads stay after it.
    effects: bool,
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
    /// A console wave64 run as two host subgroups that exchange whole-wave
    /// values (lift_wave.rs).
    xwave: bool,
    /// Scalar values whose bits at the lanes that hold no work are all 0
    /// or all 1, where one host subgroup stands for the wave (lift_hi.rs).
    idle: HashMap<Value, bool>,
    /// By lane mask and dword: those bits for the values read so far,
    /// where they agree and are known.
    idle_of: HashMap<(V, Half), Option<bool>>,
    /// Approximations taken so far (every one, also those listed before).
    approx_taken: u32,
    /// The exchanged dwords of masks, per block.
    words: HashMap<V, (V, V)>,
    /// An LDS access since the halves last met (lift_wave.rs).
    lds_dirty: bool,
    /// The block whose body `cur` is.
    cur_block: usize,
    /// The block each value made in a block body belongs to (prelude values
    /// have none).
    born: HashMap<V, usize>,
    /// Depth of each block in the dominator tree.
    depth: Vec<usize>,
    /// Each V#'s derived values (by its four words), made once in the
    /// deepest block defining a word, for every access below it.
    vsharps: HashMap<Vec<V>, mem::VSharp>,
    /// `s_getpc_b64` results: the node and the code offset it holds.
    pcs: HashMap<V, u32>,
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
    /// Per block: phis of a wave64 lane-mask pair (low dword's index, high
    /// dword's), which share the low one's node.
    mask_pairs: HashMap<usize, Vec<(usize, usize)>>,
    /// Mesh: the row of the parameter entry the compiler left unnamed.
    pub unnamed_row: Option<u8>,
    /// Vertex inputs: whether each input-signature row is a float.
    pub vs_float: HashMap<u8, bool>,
    /// Pixel input rows some site reads per vertex.
    per_vertex: HashSet<u8>,
    /// The whole-wave sections of each block, by the index of the
    /// instruction that restores EXEC: the values they leave (lift_wave.rs
    /// `section_values`).
    sections: Vec<HashMap<usize, Vec<(Value, crate::wave::Shape)>>>,
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
            xwave: Self::paired_halves(native),
            idle: if Self::idle_lanes(native) { hi::idle_bits(p) } else { HashMap::new() },
            idle_of: HashMap::new(),
            approx_taken: 0,
            words: HashMap::new(),
            lds_dirty: false,
            cur_block: 0,
            born: HashMap::new(),
            depth: vec![0; p.blocks.len()],
            vsharps: HashMap::new(),
            pcs: HashMap::new(),
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
            mask_pairs: HashMap::new(),
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
            sections: Vec::new(),
        };
        l.sections = l.find_sections();
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
        if let Some(v) = key.as_ref().and_then(|k| self.shared(k)) {
            return v;
        }
        if !node.op.removable() {
            self.epoch += 1;
        }
        let hoist = self.hoisted(&node, key.is_some(), load);
        let effect = !node.op.removable();
        let v = self.f.push(node);
        // (inside a guard: not visible after it)
        self.born.insert(v, if self.guarded && !hoist { usize::MAX } else { self.cur_block });
        if self.guarded && !hoist {
            let o = self.open.as_mut().expect("an open guard");
            o.items.push(Item::N(v));
            o.inner.insert(v);
            o.effects |= effect;
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

    /// A value equal to `k` made before: in the open guard, or shared in
    /// the block.
    fn shared(&self, k: &Node) -> Option<V> {
        if self.guarded
            && let Some(o) = &self.open
            && let Some(&v) = o.cse.get(k)
        {
            return Some(v);
        }
        if self.no_cse { None } else { self.cse.get(k).copied() }
    }

    /// Whether `node` (made inside a guard) goes outside it: pure values
    /// and loads of values from outside it, where the other guards of the
    /// block share them; loads only while no store is inside. Memory reads
    /// and effects keep their order with the open guard's effects: an
    /// unguarded one closes it first.
    fn hoisted(&mut self, node: &Node, keyed: bool, load: bool) -> bool {
        let memory = !node.op.removable() || matches!(node.op, Op::Load { .. } | Op::LdsLoad | Op::Image { .. });
        if memory && !self.guarded && self.open.as_ref().is_some_and(|o| o.effects) {
            self.close();
        }
        self.guarded && keyed && {
            let o = self.open.as_ref().unwrap();
            !(load && o.effects) && node.args.iter().all(|a| !o.inner.contains(a))
        }
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
        let half = self.half_index();
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
        if let Some(a) = self.wave_any(b) {
            return a;
        }
        self.emit(Op::Wave(op::GROUP_NON_UNIFORM_ANY, u32::MAX), Ty::BOOL, vec![b])
    }

    fn data(&mut self, r: R) -> V {
        match r {
            R::D(v) => v,
            R::M(b, half) => self.mask_word(b, half),
            R::B(b) => self.u32v(b),
        }
    }

    /// A wave64 program whose halves do not exchange (graphics stages, a
    /// group of at most 32 threads): cross-half values are per half.
    fn wave64_halves(&self) -> bool {
        self.wave64 && !self.xwave
    }

    fn approx(&mut self, note: &'static str) {
        self.approx_taken += 1;
        let a = self.approx_here(note);
        if !self.iface.approximate.contains(&a) {
            self.iface.approximate.push(a);
        }
    }

    fn approx_here(&self, note: &'static str) -> Approx {
        let (b, i) = self.site;
        let opcode = self.p.blocks.get(b).and_then(|b| b.insts.get(i)).map_or("", |i| i.name());
        Approx { note, opcode }
    }

    /// An approximation taken in computing `r` from `v`, unless `v` turns
    /// out wave-uniform or nothing reads `r`.
    fn approx_unless_uniform(&mut self, note: &'static str, v: V, r: V) {
        let a = self.approx_here(note);
        self.iface.unless_uniform.push((v, r, a));
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
            return Ok(self.settled(v, r));
        }
        // An input.
        let p = self.p;
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == v) {
            let r = self.input(reg, v, offset)?;
            self.vals.insert(v, r);
            return Ok(self.settled(v, r));
        }
        Err(unsupported(offset, "a value read before it is defined"))
    }

    /// A register value as read: of a lane mask's dword, what is known of
    /// its bits at the lanes that hold no work (lift_hi.rs) is kept by its
    /// mask (`mask_word` reads it; two values of one mask that differ in
    /// it: nothing known).
    fn settled(&mut self, v: Value, r: R) -> R {
        if let R::M(b, half) = r
            && Self::idle_lanes(self.native)
        {
            let known = self.idle.get(&v).copied();
            self.idle_of.entry((b, half)).and_modify(|k| *k = k.filter(|_| *k == known)).or_insert(known);
        }
        r
    }

    /// Whether the console wave has lanes that hold no work the host
    /// subgroup has no invocation for: a pixel program's subgroup (partly
    /// filled; the whole upper half of a wave64), and any other wave64
    /// that one subgroup stands for.
    fn idle_lanes(native: &NativeProgram) -> bool {
        native.stage == HwStage::Ps || (native.wave64 && !Self::paired_halves(native))
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
                        let m = self.mask_reg(inst, ir::VCC)?;
                        let a = self.any(m);
                        let n = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]);
                        self.u32v(n)
                    }
                    252 => {
                        let m = self.mask_reg(inst, ir::EXEC)?;
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
            // 64-bit: a signed operand's literal sign-extends, others zero-extend
            Operand::Literal(l) if k > 0 => self.k32(if inst.name().ends_with("_i64") && (l as i32) < 0 { u32::MAX } else { 0 }),
            Operand::Literal(l) => self.k32(l),
            Operand::Attr { .. } => return Err(unsupported(off, "an attribute operand")),
        })
    }

    fn opnd_mask(&mut self, inst: &Inst, o: &Operand) -> Res<V> {
        match *o {
            Operand::Scalar { code, .. } if code != crate::gfx10::NULL => self.mask_reg(inst, code),
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
        self.mask_reg(inst, ir::EXEC)
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

    // ---- program ----

    // ---- instructions ----

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
        // 32-bit float min/max (compare-exchanges: emit_ops.rs).
        "fmin" | "min_f32" | "min_rtn_f32" => op::ATOMIC_F_MIN_EXT,
        "fmax" | "max_f32" | "max_rtn_f32" => op::ATOMIC_F_MAX_EXT,
        _ => return None,
    })
}


// Instruction families live in their own files.
#[path = "lift_image.rs"]
mod image;
#[cfg(test)]
pub(crate) use image::TEST_BVH_BUILDER;
#[path = "lift_io.rs"]
mod io_;
#[path = "lift_mem.rs"]
mod mem;
#[path = "lift_scalar.rs"]
mod scalar;
#[path = "lift_valu.rs"]
mod valu;
#[path = "lift_vops.rs"]
mod vops;
#[path = "lift_pk.rs"]
mod pk;
#[path = "lift_flow.rs"]
mod flow;
#[path = "lift_wave.rs"]
mod wave;
#[path = "lift_hi.rs"]
mod hi;
#[path = "lift_fold.rs"]
mod fold;
#[path = "lift_lane.rs"]
mod lane;

