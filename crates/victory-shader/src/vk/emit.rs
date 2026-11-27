//! Thread IR to SPIR-V: types, interface, helper functions, and structured
//! control flow.
//!
//! The block graph (reducible) is placed by its dominator tree as in
//! Ramsey's "Beyond Relooper" (JFP 2022), as `hlsl/cfg.rs` does for HLSL:
//! a loop header opens a loop construct; a node with several forward
//! predecessors is the merge of a one-iteration loop construct that its
//! predecessors break out of. A jump past more than the innermost construct
//! sets `jmp` and breaks; each construct passes it on. Block phis are
//! function variables (the driver's SSA construction removes them); guard
//! phis are real `OpPhi`s.

use std::collections::{HashMap, HashSet};

use super::lift::{
    Interface, OUT_CLIP, OUT_CULL, OUT_DEPTH, OUT_LAYER, OUT_POINT_SIZE, OUT_POSITION, OUT_PRIMITIVE, OUT_SAMPLE_MASK, OUT_STENCIL,
    OUT_VIEWPORT,
};
use super::graph::Graph;
use super::spirv::{Id, Module, en, glsl, op};
use super::tir::{Builtin, Func, Helper, ImageClass, ImageKind, Interp, Item, Op, S, Sampler, Term, Ty, V, img};
use crate::native::HwStage;

/// What the module needs to know about the stage.
#[derive(Clone)]
pub struct Stage {
    pub stage: HwStage,
    pub local_size: [u32; 3],
    /// Output types per location: 0 float, 1 uint, 2 int.
    pub out_types: HashMap<u16, u8>,
    /// Interface rows and channels to (location, component): the fixed
    /// location of each semantic (the semantic registry).
    pub locs: HashMap<(u8, u8), (u8, u8)>,
    /// Mesh: maximum vertices and primitives (the console's).
    pub mesh: (u32, u32),
    /// Mesh: the console group run as several host groups.
    pub mesh_split: Option<MeshSplit>,
    /// Iterations after which a loop gives up (the loop guard).
    pub loop_limit: u32,
    /// What the loop guard writes to identify the program.
    pub guard_id: u32,
    /// Branch blocks whose join is placed as a block construct, not a
    /// selection (found by earlier attempts).
    pub no_sel: HashSet<usize>,
}

/// The specialization constant (u32, default 0) of a program with code
/// tables: where its tables' bytes lie in the host's tables buffer, in
/// bytes from the buffer's start (`docs/vulkan-abi.md` section 3.1).
/// (0: the memory map has one range; 100 + n: immediate sampler n.)
pub const SPEC_CODE_TABLES: u32 = 1;

pub struct Emitted {
    /// Selections a jump had to leave through a dispatch (the attempt is
    /// redone with them as block constructs).
    pub bad_sel: Vec<usize>,
    pub words: Vec<u32>,
    /// Vertex programs: the input rows read as vertex attributes.
    pub vertex_inputs: Vec<u8>,
    /// (location, word index of the Location literal) of pixel inputs.
    pub input_locations: Vec<(u8, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctx {
    Loop(usize),
    /// A one-iteration loop whose merge is the node (a merge reached from
    /// several constructs).
    Block(usize),
    /// A selection whose merge is the node (the branch's join).
    Sel(usize),
    /// A dispatch loop over the node's merge children.
    Disp(usize),
}

struct Construct {
    node: usize,
    /// A selection's header block.
    head: usize,
    header: Id,
    merge: Id,
    cont: Id,
    /// Jumps passing out through this construct's merge: they bring the
    /// jump's code and the values for its target's phis (a ladder of phis,
    /// no variables).
    arrivals: Vec<Arrival>,
    /// A dispatch loop's cases.
    disp: Option<Box<Disp>>,
}

/// A dispatch loop: the merge children of a block with many as cases of one
/// loop, chosen by a state phi; a jump to one sets the state and its phis'
/// values (in slots by type) and continues the loop.
struct Disp {
    /// The cases' blocks (state j + 1) and per phi its slot.
    cases: Vec<(usize, Vec<usize>)>,
    slot_tys: Vec<Id>,
    /// The header's phis: the state, then the slots.
    state: Id,
    slots: Vec<Id>,
    /// The continue target's phis (made when it closes) and its edges:
    /// (state, slot values, block).
    cstate: Id,
    cslots: Vec<Id>,
    conts: Vec<(Id, Vec<Id>, Id)>,
}

/// A jump to an outer construct arriving at a merge it passes.
struct Arrival {
    from: Id,
    code: Id,
    /// The codes it may carry.
    possible: Vec<u32>,
    /// (target block, its phi values)
    vals: Vec<(usize, Vec<Id>)>,
}


#[derive(Default)]
struct Types {
    void: Id,
    bool_: Id,
    u8_: Id,
    u16_: Id,
    u32_: Id,
    u64_: Id,
    f16_: Id,
    f32_: Id,
}

pub struct Emit<'a> {
    m: Module,
    f: &'a Func,
    iface: &'a Interface,
    st: &'a Stage,
    t: Types,
    ids: Vec<Id>,
    /// The graph block whose code defined an id (values only: the phis a
    /// ladder makes are not here).
    def_at: HashMap<Id, usize>,
    /// The graph block being emitted.
    cur: usize,
    block: Id,
    interface: Vec<Id>,
    push: Id,
    map: Id,
    builtins: HashMap<Builtin, Id>,
    inputs: HashMap<u8, Id>,
    outputs: HashMap<u16, (Id, Id)>,
    lds: Id,
    bad_sel: Vec<usize>,
    /// Open two-way branches with unreachable merges: the construct depth
    /// at each.
    base: Vec<usize>,
    map_var: Id,
    /// Mesh split: this part's set of blocks.
    mesh_part: Id,
    /// Inside a flat guard: its condition (effects go to the sink).
    pred: Option<Id>,
    sink: Id,
    /// Values a forward edge / back edge gives a block's phis, and the
    /// SPIR-V block it comes from.
    fwd: HashMap<usize, Vec<(Vec<Id>, Id)>>,
    back: HashMap<usize, Vec<(Vec<Id>, Id)>>,
    /// A block's phi values when its code begins.
    pending: HashMap<usize, Vec<Id>>,
    /// Phis of each loop's continue target, made at its header.
    cphi: HashMap<usize, Vec<Id>>,
    /// Every edge into a construct's merge so far.
    merge_preds: HashMap<Id, Vec<Id>>,
    images: HashMap<(u8, bool, bool, bool, u32, bool, u32), Id>,
    /// R64ui storage images of set 0 binding 3 by (dim, arrayed, ms).
    images64: HashMap<(u8, bool, bool), Id>,
    samplers: Id,
    spec_samplers: HashMap<u32, Id>,
    /// Specialization constant [`SPEC_CODE_TABLES`] (0: not declared).
    spec_tables: Id,
    helpers: HashMap<Helper, Id>,
    /// Merges of the selections `open_if` opened.
    if_merge: Vec<Id>,
    /// The invocation's loop iterations so far (the loop guard; 0: none yet).
    spent_var: Id,
    /// The guard function (0: no loop gave up a way to it yet).
    guard_fn: Id,
    /// The wave64 halves' exchange (emit_xchg.rs).
    xv: xchg::XchgVars,
    /// Counted console barriers (emit_bar.rs), when `bar_counted`.
    bar: bar::BarVars,
    bar_counted: bool,
    g: &'a Graph,
    ctx: Vec<Ctx>,
    cons: Vec<Construct>,
    entry_vars: Vec<u32>,
    input_locations: Vec<(u8, usize)>,
    frag: bool,
}

/// Memory semantics of a group barrier and of the wave64 halves' meetings:
/// workgroup, buffer and image memory (`s_waitcnt_vscnt` before
/// `s_barrier` is the console's device-memory barrier with group sync).
const SEM_GROUP: u32 = en::SEM_ACQUIRE_RELEASE | en::SEM_WORKGROUP_MEMORY | en::SEM_UNIFORM_MEMORY | en::SEM_IMAGE_MEMORY;

/// Jump codes: construct at stack position i; continue (loop) or exit.
fn code(i: usize, cont: bool) -> u32 {
    (2 * i + 1 + usize::from(cont)) as u32
}

/// Jump codes into case j of the dispatch loop at stack position i.
fn code_case(i: usize, j: usize) -> u32 {
    (1 << 31) | ((j as u32) << 16) | i as u32
}

/// Blocks with at least this many merge children place them as the cases
/// of a dispatch loop (nested block constructs otherwise).
const DISPATCH_MIN: usize = 8;

impl<'a> Emit<'a> {
    pub fn new(f: &'a Func, g: &'a Graph, iface: &'a Interface, st: &'a Stage) -> Self {
        let mut m = Module::new();
        let t = Types {
            void: m.t_void(),
            bool_: m.t_bool(),
            u8_: 0,
            u16_: 0,
            u32_: m.t_int(32),
            u64_: m.t_int(64),
            f16_: 0,
            f32_: m.t_float(32),
        };
        Emit {
            m,
            f,
            iface,
            st,
            t,
            ids: vec![0; f.nodes.len()],
            def_at: HashMap::new(),
            cur: 0,
            block: 0,
            interface: Vec::new(),
            push: 0,
            map: 0,
            builtins: HashMap::new(),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            lds: 0,
            bad_sel: Vec::new(),
            base: Vec::new(),
            map_var: 0,
            mesh_part: 0,
            pred: None,
            sink: 0,
            fwd: HashMap::new(),
            back: HashMap::new(),
            pending: HashMap::new(),
            cphi: HashMap::new(),
            merge_preds: HashMap::new(),
            images: HashMap::new(),
            images64: HashMap::new(),
            samplers: 0,
            spec_samplers: HashMap::new(),
            spec_tables: 0,
            helpers: HashMap::new(),
            if_merge: Vec::new(),
            spent_var: 0,
            guard_fn: 0,
            xv: Default::default(),
            bar: Default::default(),
            bar_counted: false,
            g,
            ctx: Vec::new(),
            cons: Vec::new(),
            entry_vars: Vec::new(),
            input_locations: Vec::new(),
            frag: st.stage == HwStage::Ps,
        }
    }

    fn scalar(&mut self, s: S) -> Id {
        match s {
            S::Void => self.t.void,
            S::Bool => self.t.bool_,
            S::U8 => {
                if self.t.u8_ == 0 {
                    self.t.u8_ = self.m.t_int(8);
                    self.m.capability(en::CAP_STORAGE_BUFFER_8BIT_ACCESS);
                }
                self.t.u8_
            }
            S::U16 => {
                if self.t.u16_ == 0 {
                    self.t.u16_ = self.m.t_int(16);
                    self.m.capability(en::CAP_STORAGE_BUFFER_16BIT_ACCESS);
                }
                self.t.u16_
            }
            S::U32 => self.t.u32_,
            S::U64 => self.t.u64_,
            S::F16 => {
                if self.t.f16_ == 0 {
                    self.t.f16_ = self.m.t_float(16);
                }
                self.t.f16_
            }
            S::F32 => self.t.f32_,
            // 64-bit floats: a handful of water programs add two
            S::F64 => self.m.t_float(64),
        }
    }

    fn ty(&mut self, t: Ty) -> Id {
        if t.s == S::Void && t.n == 2 {
            // frexp result
            let (f, u) = (self.t.f32_, self.t.u32_);
            return self.m.t_struct(&[f, u], 1);
        }
        let s = self.scalar(t.s);
        self.m.t_vec(s, u32::from(t.n))
    }

    fn id(&self, v: V) -> Id {
        let id = self.ids[v as usize];
        assert!(id != 0, "value {v} ({:?}) used before its definition", self.f.node(v).op);
        id
    }

    fn c32(&mut self, v: u32) -> Id {
        self.m.c_u32(v)
    }

    // ---- interface ----

    // ---- helpers ----

    // ---- values ----

    fn emit_node(&mut self, v: V) {
        let n = self.f.node(v).clone();
        let ty = n.ty;
        let tid = if ty == Ty::VOID { self.t.void } else { self.ty(ty) };
        let args: Vec<Id> = n.args.iter().map(|&a| self.id(a)).collect();
        let id = match n.op {
            Op::Const(bits) => self.constant(ty, bits),
            Op::Undef => self.m.undef(tid),
            Op::Push(k) => self.push_load(k),
            Op::MapField(k) => {
                let q = self.t.u64_;
                self.map_load(q, &[k])
            }
            Op::Builtin(b) => self.builtin(b),
            Op::Input { loc, comp, interp } => self.input(loc, comp, interp, ty),
            Op::PerVertex { loc, comp, vertex } => self.per_vertex_input(loc, comp, vertex),
            // a vector of constants is a constant (texel offsets must be)
            Op::Spv(op::COMPOSITE_CONSTRUCT) if n.imm.is_empty() && args.iter().all(|&a| self.m.is_constant(a)) => {
                self.m.c_composite(tid, &args)
            }
            Op::Spv(o) => {
                let mut ops = args.clone();
                ops.extend_from_slice(&n.imm);
                let rt = match o {
                    op::U_MUL_EXTENDED | op::S_MUL_EXTENDED => self.m.t_struct(&[self.t.u32_, self.t.u32_], 5),
                    _ => tid,
                };
                self.m.op(o, rt, &ops)
            }
            Op::Glsl(g) => self.m.ext(g, tid, &args),
            Op::Call(h) => {
                let fid = self.helper(h);
                let mut ops = vec![fid];
                ops.extend_from_slice(&args);
                self.m.op(57, tid, &ops) // OpFunctionCall
            }
            Op::Load { .. } | Op::Store { .. } | Op::Atomic(_) => self.memory_op(&n, &args, tid),
            Op::LdsLoad | Op::LdsStore | Op::LdsAtomic(_) => self.lds_op(&n, &args),
            Op::Image { kind, class, sampler, flags } => self.image(kind, class, sampler, flags, &args, tid),
            Op::Wave(o, g) => self.wave_op(o, g, &args, tid),
            Op::Output { slot, comp } => self.output_op(&n, slot, comp, &args),
            Op::Demote => {
                self.m.capability(en::CAP_DEMOTE_TO_HELPER_INVOCATION);
                self.m.op_void(op::DEMOTE_TO_HELPER_INVOCATION, &[]);
                0
            }
            Op::Xchg => self.xchg(args[0], v),
            Op::Barrier => {
                self.barrier();
                0
            }
            Op::Phi => unreachable!("guard phis are written by their guard"),
            Op::Pick => self.pick(&args),
        };
        self.set_id(v, id);
    }

    /// The id of a value, defined in the block being emitted.
    fn set_id(&mut self, v: V, id: Id) {
        self.ids[v as usize] = id;
        if id != 0 {
            self.def_at.entry(id).or_insert(self.cur);
        }
    }

    fn constant(&mut self, ty: Ty, bits: u64) -> Id {
        match ty.s {
            S::Bool => self.m.c_bool(bits != 0),
            S::U32 => self.m.c_u32(bits as u32),
            S::U64 => self.m.c_u64(bits),
            S::F32 => self.m.c_f32(bits as u32),
            S::U16 => {
                let _ = self.scalar(S::U16);
                self.m.c_u16(bits as u16)
            }
            S::F16 => {
                let _ = self.scalar(S::F16);
                self.m.c_f16(bits as u16)
            }
            S::U8 => {
                let t = self.scalar(S::U8);
                self.m.c_typed(t, bits as u32)
            }
            S::F64 => {
                let t = self.scalar(S::F64);
                self.m.c_typed64(t, bits)
            }
            S::Void => 0,
        }
    }

    // ---- items ----

    fn items(&mut self, list: &[Item]) {
        for it in list {
            match it {
                Item::N(v) => self.emit_node(*v),
                Item::Guard { cond, body, phis } if self.predicable(body) => {
                    // flat: the body runs for every lane, selects keep the
                    // old values of the lanes outside the guard
                    let c = self.id(*cond);
                    let outer = self.pred;
                    self.pred = Some(match outer {
                        Some(o) => self.m.op(op::LOGICAL_AND, self.t.bool_, &[o, c]),
                        None => c,
                    });
                    self.items(body);
                    self.pred = outer;
                    for &p in phis {
                        let n = self.f.node(p);
                        let (inner, outer) = (n.args[0], n.args[1]);
                        let t = self.ty(n.ty);
                        let (a, b) = (self.id(inner), self.id(outer));
                        let id = self.m.op(op::SELECT, t, &[c, a, b]);
                        self.set_id(p, id);
                    }
                }
                Item::Guard { cond, body, phis } => {
                    let c = self.id(*cond);
                    let (then, merge) = (self.m.id(), self.m.id());
                    let header = self.block;
                    self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[c, then, merge]);
                    self.m.label(then);
                    self.block = then;
                    self.items(body);
                    let end = self.block;
                    self.m.op_void(op::BRANCH, &[merge]);
                    self.m.label(merge);
                    self.block = merge;
                    for &p in phis {
                        let n = self.f.node(p);
                        let (inner, outer) = (n.args[0], n.args[1]);
                        let t = self.ty(n.ty);
                        let (a, b) = (self.id(inner), self.id(outer));
                        let id = self.m.op(op::PHI, t, &[a, end, b, header]);
                        self.set_id(p, id);
                    }
                }
            }
        }
    }

    /// A guard whose body has no effects but stores (their addresses go to
    /// the sink outside the guard) is emitted flat, whatever its size.
    fn predicable(&self, body: &[Item]) -> bool {
        body.iter().all(|it| {
            matches!(it, Item::N(v) if {
                let op = &self.f.node(*v).op;
                (op.removable() && *op != Op::Phi)
                    || matches!(op, Op::Store { .. } | Op::Atomic(_) | Op::LdsStore | Op::LdsAtomic(_))
            })
        })
    }

    /// A store's address: the sink where the lanes outside a flat guard
    /// write.
    fn predicated_addr(&mut self, a: Id) -> Id {
        match self.pred {
            Some(c) => {
                let q = self.t.u64_;
                self.m.op(op::SELECT, q, &[c, a, self.sink])
            }
            None => a,
        }
    }

    fn branch_to(&mut self, target: Id) {
        self.m.op_void(op::BRANCH, &[target]);
    }

    // ---- structure ----

    /// A function variable (declared at the entry later).

    pub fn run(mut self) -> Result<Emitted, &'static str> {

        let fid = self.m.id();
        let void = self.t.void;
        // Body first into a separate buffer: variables go first in the entry.
        let saved = std::mem::take(&mut self.m.body);
        let first = self.m.id();
        self.m.label(first);
        self.block = first;
        // constants and prelude
        for i in 0..self.f.nodes.len() {
            if matches!(self.f.nodes[i].op, Op::Const(_) | Op::Undef) {
                self.emit_node(i as V);
            }
        }
        for v in self.f.prelude.clone() {
            self.emit_node(v);
        }
        if self.st.stage == HwStage::Ms {
            self.mesh_prologue();
        }
        self.xchg_init();
        if self.f.nodes.iter().any(|n| matches!(n.op, Op::Store { .. } | Op::Atomic(_))) {
            let q = self.t.u64_;
            self.sink = self.map_load(q, &[1]);
        }
        // record header labels for continue targets lazily: we need the
        // header of each construct when closing it; keep a map cont -> header
        self.tree_root();
        let body = std::mem::replace(&mut self.m.body, saved);
        // function header
        let fty = self.m.t_func(void);
        self.m.op_to(op::FUNCTION, void, fid, &[0, fty]);
        let entry = self.m.id();
        self.m.label(entry);
        for pair in self.entry_vars.clone().chunks(2) {
            self.m.op_to(op::VARIABLE, pair[0], pair[1], &[en::SC_FUNCTION]);
        }
        self.m.op_void(op::BRANCH, &[first]);
        self.m.body.extend(body);
        self.m.op_void(op::FUNCTION_END, &[]);
        self.write_helpers();
        self.write_guard_fn();
        self.write_xchg_fn();
        self.write_bar_fn();
        // entry point and modes
        let (model, name) = match self.st.stage {
            HwStage::Ps => (en::EXEC_FRAGMENT, "main"),
            HwStage::Vs => (en::EXEC_VERTEX, "main"),
            HwStage::Ms => (en::EXEC_MESH_EXT, "main"),
            _ => (en::EXEC_GL_COMPUTE, "main"),
        };
        let interface = self.interface.clone();
        self.m.entry_point(model, fid, name, &interface);
        match self.st.stage {
            HwStage::Ps => {
                self.m.execution_mode(fid, en::EM_ORIGIN_UPPER_LEFT, &[]);
                let (depth, stencil) = (self.outputs.contains_key(&OUT_DEPTH), self.outputs.contains_key(&OUT_STENCIL));
                if depth {
                    self.m.execution_mode(fid, en::EM_DEPTH_REPLACING, &[]);
                }
                if stencil {
                    self.m.execution_mode(fid, en::EM_STENCIL_REF_REPLACING, &[]);
                }
                // The console's depth and stencil tests (and writes) before
                // the program (`Interface::early_depth`): a pixel they
                // reject does not run, so nothing it would store lands,
                // and a kill keeps the depth already written. Never for a
                // program that gives the tests their depth or stencil.
                if self.iface.early_depth && !depth && !stencil {
                    self.m.execution_mode(fid, en::EM_EARLY_FRAGMENT_TESTS, &[]);
                }
            }
            HwStage::Cs => {
                let [x, y, z] = self.st.local_size;
                self.m.execution_mode(fid, en::EM_LOCAL_SIZE, &[x.max(1), y.max(1), z.max(1)]);
            }
            HwStage::Ms => {
                self.m.extension("SPV_EXT_mesh_shader");
                self.m.capability(en::CAP_MESH_SHADING_EXT);
                let [x, y, z] = self.st.local_size;
                self.m.execution_mode(fid, en::EM_LOCAL_SIZE, &[x.max(1), y.max(1), z.max(1)]);
                self.m.execution_mode(fid, en::EM_OUTPUT_VERTICES, &[self.mesh_vertices()]);
                self.m.execution_mode(fid, en::EM_OUTPUT_PRIMITIVES_EXT, &[self.st.mesh.1.max(1)]);
                self.m.execution_mode(fid, en::EM_OUTPUT_TRIANGLES_EXT, &[]);
            }
            _ => {}
        }
        // Signed zeros, infinities and NaNs as written, for every float
        // width the module uses: the console's rules we spell out (legacy
        // multiplies, med3 and class tests on NaN, min/max of zeros) are
        // not the compiler's to fold. A device without them for a width
        // takes the module with that mode left out (victory-vk
        // `limits::fit_float_controls`).
        self.m.capability(en::CAP_SIGNED_ZERO_INF_NAN_PRESERVE);
        for (w, cap) in [(16, Some(en::CAP_FLOAT16)), (32, None), (64, Some(en::CAP_FLOAT64))] {
            if cap.is_none_or(|c| self.m.has_capability(c)) {
                self.m.execution_mode(fid, en::EM_SIGNED_ZERO_INF_NAN_PRESERVE, &[w]);
            }
        }
        let (words, at) = self.m.finish();
        let input_locations = self.input_locations.iter().map(|&(l, w)| (l, at + w)).collect();
        let vertex_inputs = if self.st.stage == HwStage::Vs { self.inputs.keys().copied().collect() } else { Vec::new() };
        Ok(Emitted { bad_sel: self.bad_sel, words, input_locations, vertex_inputs })
    }

    fn tree_root(&mut self) {
        self.tree(0);
    }
}

#[path = "emit_cfg.rs"]
mod cfg;
#[path = "emit_iface.rs"]
mod iface;
#[path = "emit_image.rs"]
mod image;
#[path = "emit_disp.rs"]
mod disp;
#[path = "emit_ops.rs"]
mod ops;
#[path = "emit_guard.rs"]
mod guard;
#[path = "emit_mesh.rs"]
mod mesh;
pub use mesh::MeshSplit;
#[path = "emit_xchg.rs"]
mod xchg;
#[path = "emit_bar.rs"]
mod bar;
#[path = "emit_typed.rs"]
mod typed;
pub(crate) use typed::format_table;
#[path = "emit_bvh.rs"]
mod bvh;

