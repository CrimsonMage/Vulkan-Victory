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

use super::lift::{Interface, OUT_DEPTH, OUT_POSITION, OUT_PRIMITIVE, OUT_SAMPLE_MASK, OUT_STENCIL};
use super::spirv::{Id, Module, en, glsl, op};
use super::tir::{Builtin, Func, Helper, ImageClass, ImageKind, Interp, Item, Op, S, Sampler, Term, Ty, V, img};
use crate::native::HwStage;

/// What the module needs to know about the stage.
pub struct Stage {
    pub stage: HwStage,
    pub local_size: [u32; 3],
    /// Output types per location: 0 float, 1 uint, 2 int.
    pub out_types: HashMap<u16, u8>,
    pub loop_limit: u32,
    /// Interface rows and channels to (location, component): the fixed
    /// location of each semantic (`super::SEMANTIC_LOCATIONS`).
    pub locs: HashMap<(u8, u8), (u8, u8)>,
    /// Mesh: maximum vertices and primitives.
    pub mesh: (u32, u32),
    /// Branch blocks whose join is placed as a block construct, not a
    /// selection (found by earlier attempts).
    pub no_sel: HashSet<usize>,
}

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
}

struct Construct {
    node: usize,
    /// A selection's header block.
    head: usize,
    header: Id,
    merge: Id,
    cont: Id,
    passes: bool,
    /// Jumps from inner constructs that arrive here: bit 0 exit, bit 1 continue.
    wanted: u8,
    /// Loop constructs: the iteration count (a phi at the header) and its
    /// next value (made at the continue target).
    counter: Option<(Id, Id)>,
    /// Jumps passing out through this construct's merge: they bring the
    /// jump's code and the values for its target's phis (a ladder of phis,
    /// no variables).
    arrivals: Vec<Arrival>,
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

struct Graph {
    rpo: Vec<usize>,
    header: Vec<bool>,
    merge: Vec<bool>,
    children: Vec<Vec<usize>>,
    /// Per loop header: the exit placed at the loop construct's merge (a
    /// merge child of the header outside the loop, the most entered), so
    /// leaving the loop for it is a plain break.
    exit: Vec<Option<usize>>,
    /// Per loop header: its merge children outside the loop placed after
    /// the exit (their blocks wrap the loop).
    after_exit: Vec<Vec<usize>>,
}

impl Graph {
    fn new(f: &Func) -> Result<Self, &'static str> {
        let n = f.blocks.len();
        let succ: Vec<Vec<usize>> = f
            .blocks
            .iter()
            .map(|b| match b.term {
                Term::Jump(t) => vec![t],
                Term::Branch(_, t, e) => vec![t, e],
                Term::Return => vec![],
            })
            .collect();
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
        // natural loops: what reaches a back edge's source without the header
        let mut exit = vec![None; n];
        let mut after_exit = vec![Vec::new(); n];
        for &h in &order {
            if !header[h] {
                continue;
            }
            let mut inside = vec![false; n];
            inside[h] = true;
            let mut work: Vec<usize> = preds[h].iter().copied().filter(|&p| rpo[p] >= rpo[h]).collect();
            while let Some(b) = work.pop() {
                if std::mem::replace(&mut inside[b], true) {
                    continue;
                }
                work.extend(preds[b].iter().copied());
            }
            let mut best: Option<(usize, usize)> = None;
            for &c in &children[h] {
                if !merge[c] || inside[c] {
                    continue;
                }
                let edges = preds[c].iter().filter(|&&p| inside[p]).count();
                if edges > 0 && best.is_none_or(|b| edges > b.1) {
                    best = Some((c, edges));
                }
            }
            exit[h] = best.map(|b| b.0);
            if let Some((y, _)) = best {
                let mut after: Vec<usize> =
                    children[h].iter().copied().filter(|&c| merge[c] && !inside[c] && rpo[c] > rpo[y]).collect();
                after.sort_by_key(|&c| std::cmp::Reverse(rpo[c]));
                after_exit[h] = after;
            }
        }
        Ok(Graph { rpo, header, merge, children, exit, after_exit })
    }
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
    cull: Id,
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
    images: HashMap<(u8, bool, bool, bool, bool, bool, u32), Id>,
    samplers: Id,
    spec_samplers: HashMap<u32, Id>,
    helpers: HashMap<Helper, Id>,
    picks: HashMap<V, Id>,
    g: Option<Graph>,
    ctx: Vec<Ctx>,
    cons: Vec<Construct>,
    entry_vars: Vec<u32>,
    input_locations: Vec<(u8, usize)>,
    frag: bool,
}

/// Function control of a helper: DontInline (2) when `VK_NOINLINE` names it
/// (experiment).
fn helper_control(which: &str) -> u32 {
    match std::env::var("VK_NOINLINE") {
        Ok(v) if v == "all" || v == which => 2,
        _ => 0,
    }
}

/// Jump codes: construct at stack position i; continue (loop) or exit.
fn code(i: usize, cont: bool) -> u32 {
    (2 * i + 1 + usize::from(cont)) as u32
}

impl<'a> Emit<'a> {
    pub fn new(f: &'a Func, iface: &'a Interface, st: &'a Stage) -> Self {
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
            cull: 0,
            pred: None,
            sink: 0,
            fwd: HashMap::new(),
            back: HashMap::new(),
            pending: HashMap::new(),
            cphi: HashMap::new(),
            merge_preds: HashMap::new(),
            images: HashMap::new(),
            samplers: 0,
            spec_samplers: HashMap::new(),
            helpers: HashMap::new(),
            picks: HashMap::new(),
            g: None,
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

    fn push_var(&mut self) -> Id {
        if self.push == 0 {
            let u = self.t.u32_;
            let arr = self.m.t_array(u, 64);
            self.m.decorate(arr, en::DEC_ARRAY_STRIDE, &[4]);
            let s = self.m.t_struct(&[arr], 2);
            self.m.decorate(s, en::DEC_BLOCK, &[]);
            self.m.member_decorate(s, 0, en::DEC_OFFSET, &[0]);
            let p = self.m.t_ptr(en::SC_PUSH_CONSTANT, s);
            self.push = self.m.variable(p, en::SC_PUSH_CONSTANT);
            self.interface.push(self.push);
        }
        self.push
    }

    /// The memory map's struct type (`docs/vulkan-abi.md` section 3).
    fn map_type(&mut self) -> Id {
        if self.map == 0 {
            let (u, q) = (self.t.u32_, self.t.u64_);
            let range = self.m.t_struct(&[q, q, q, q], 3);
            for (k, o) in [0u32, 8, 16, 24].iter().enumerate() {
                self.m.member_decorate(range, k as u32, en::DEC_OFFSET, &[*o]);
            }
            let arr = self.m.t_array(range, 64);
            self.m.decorate(arr, en::DEC_ARRAY_STRIDE, &[32]);
            let st = self.m.t_struct(&[q, q, u, arr], 4);
            for (k, o) in [0u32, 8, 16, 32].iter().enumerate() {
                self.m.member_decorate(st, k as u32, en::DEC_OFFSET, &[*o]);
            }
            self.m.decorate(st, en::DEC_BLOCK, &[]);
            self.map = st;
        }
        self.map
    }

    /// Experiment: no binary search code (one range only).
    fn no_search() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("VK_SEARCH").is_ok_and(|v| v == "0"))
    }

    /// The memory map: a uniform buffer, set 3 binding 0 (read with uniform
    /// loads; NVIDIA compiled a pointer-loaded map ~35x slower).
    fn map_ptr(&mut self) -> Id {
        let st = self.map_type();
        if self.map_var == 0 {
            let p = self.m.t_ptr(en::SC_UNIFORM, st);
            self.map_var = self.m.variable(p, en::SC_UNIFORM);
            self.m.decorate(self.map_var, en::DEC_DESCRIPTOR_SET, &[3]);
            self.m.decorate(self.map_var, en::DEC_BINDING, &[0]);
            self.interface.push(self.map_var);
        }
        self.map_var
    }

    /// Loads a member of the memory map (u64 or u32 by `ty`); `path` ids.
    fn map_load_at(&mut self, map: Id, ty: Id, path: &[Id]) -> Id {
        let p = self.m.t_ptr(en::SC_UNIFORM, ty);
        let mut ops = vec![map];
        ops.extend_from_slice(path);
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &ops);
        self.m.op(op::LOAD, ty, &[ptr])
    }

    fn map_load(&mut self, ty: Id, path: &[u32]) -> Id {
        let map = self.map_ptr();
        let ids: Vec<Id> = path.iter().map(|&k| self.c32(k)).collect();
        self.map_load_at(map, ty, &ids)
    }

    fn builtin(&mut self, b: Builtin) -> Id {
        let (bi, comps, sty): (u32, u32, Id) = match b {
            Builtin::VertexIndex => (en::BI_VERTEX_INDEX, 1, self.t.u32_),
            Builtin::InstanceIndex => (en::BI_INSTANCE_INDEX, 1, self.t.u32_),
            Builtin::BaseInstance => (en::BI_BASE_INSTANCE, 1, self.t.u32_),
            Builtin::FragCoord(_) => (en::BI_FRAG_COORD, 4, self.t.f32_),
            Builtin::FrontFacing => (en::BI_FRONT_FACING, 1, self.t.bool_),
            Builtin::SampleId => (en::BI_SAMPLE_ID, 1, self.t.u32_),
            Builtin::SampleMaskIn => (en::BI_SAMPLE_MASK, 1, self.t.u32_),
            Builtin::Helper => (en::BI_HELPER_INVOCATION, 1, self.t.bool_),
            Builtin::WorkgroupId(_) => (en::BI_WORKGROUP_ID, 3, self.t.u32_),
            Builtin::LocalInvocationId(_) => (en::BI_LOCAL_INVOCATION_ID, 3, self.t.u32_),
            Builtin::LocalInvocationIndex => (en::BI_LOCAL_INVOCATION_INDEX, 1, self.t.u32_),
            Builtin::SubgroupId => (en::BI_SUBGROUP_ID, 1, self.t.u32_),
            Builtin::SubgroupLocalId => (en::BI_SUBGROUP_LOCAL_INVOCATION_ID, 1, self.t.u32_),
            Builtin::PrimitiveId => (en::BI_PRIMITIVE_ID, 1, self.t.u32_),
            Builtin::Bary { perspective, .. } => (if perspective { en::BI_BARY_COORD_KHR } else { 5287 }, 3, self.t.f32_),
        };
        let key = match b {
            Builtin::FragCoord(_) => Builtin::FragCoord(0),
            Builtin::WorkgroupId(_) => Builtin::WorkgroupId(0),
            Builtin::LocalInvocationId(_) => Builtin::LocalInvocationId(0),
            Builtin::Bary { perspective, centroid, sample, .. } => Builtin::Bary { perspective, centroid, sample, comp: 0 },
            x => x,
        };
        let var = if let Some(&v) = self.builtins.get(&key) {
            v
        } else {
            let ty = if b == Builtin::SampleMaskIn {
                let arr = self.m.t_array(sty, 1);
                arr
            } else {
                self.m.t_vec(sty, comps)
            };
            let p = self.m.t_ptr(en::SC_INPUT, ty);
            let v = self.m.variable(p, en::SC_INPUT);
            self.m.decorate(v, en::DEC_BUILTIN, &[bi]);
            if self.frag && sty != self.t.f32_ && sty != self.t.bool_ && b != Builtin::Helper {
                self.m.decorate(v, en::DEC_FLAT, &[]);
            }
            if let Builtin::Bary { centroid, sample, .. } = b {
                self.m.extension("SPV_KHR_fragment_shader_barycentric");
                self.m.capability(en::CAP_FRAGMENT_BARYCENTRIC_KHR);
                if centroid {
                    self.m.decorate(v, en::DEC_CENTROID, &[]);
                }
                if sample {
                    self.m.capability(en::CAP_SAMPLE_RATE_SHADING);
                    self.m.decorate(v, en::DEC_SAMPLE, &[]);
                }
            }
            if b == Builtin::BaseInstance {
                self.m.capability(4427); // DrawParameters
            }
            if b == Builtin::SampleId {
                self.m.capability(en::CAP_SAMPLE_RATE_SHADING);
            }
            if matches!(b, Builtin::SubgroupLocalId | Builtin::SubgroupId) {
                self.m.capability(en::CAP_GROUP_NON_UNIFORM);
            }
            self.interface.push(v);
            self.builtins.insert(key, v);
            v
        };
        let comp = match b {
            Builtin::FragCoord(c) | Builtin::WorkgroupId(c) | Builtin::LocalInvocationId(c) => Some(u32::from(c)),
            Builtin::Bary { comp, .. } => Some(u32::from(comp)),
            Builtin::SampleMaskIn => Some(0),
            _ => None,
        };
        match comp {
            Some(c) => {
                let p = self.m.t_ptr(en::SC_INPUT, sty);
                let k = self.c32(c);
                let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, k]);
                self.m.op(op::LOAD, sty, &[ptr])
            }
            None => {
                let ty = self.m.t_vec(sty, comps);
                self.m.op(op::LOAD, ty, &[var])
            }
        }
    }

    fn input(&mut self, row: u8, chan: u8, interp: Interp, ty: Ty) -> Id {
        if self.st.stage == HwStage::Vs {
            return self.vertex_input(row, chan, ty);
        }
        let (loc, comp) = self.st.locs.get(&(row, chan)).copied().unwrap_or((31, chan));
        let f = self.t.f32_;
        let var = if let Some(&v) = self.inputs.get(&loc) {
            v
        } else {
            let v4 = self.m.t_vec(f, 4);
            let p = self.m.t_ptr(en::SC_INPUT, v4);
            let v = self.m.variable(p, en::SC_INPUT);
            let at = self.m.decorate_at(v, en::DEC_LOCATION, u32::from(loc));
            self.input_locations.push((loc, at));
            match interp {
                Interp::Flat => self.m.decorate(v, en::DEC_FLAT, &[]),
                Interp::Smooth { perspective, centroid, sample } => {
                    if !perspective {
                        self.m.decorate(v, en::DEC_NO_PERSPECTIVE, &[]);
                    }
                    if centroid {
                        self.m.decorate(v, en::DEC_CENTROID, &[]);
                    }
                    if sample {
                        self.m.capability(en::CAP_SAMPLE_RATE_SHADING);
                        self.m.decorate(v, en::DEC_SAMPLE, &[]);
                    }
                }
            }
            self.interface.push(v);
            self.inputs.insert(loc, v);
            v
        };
        let p = self.m.t_ptr(en::SC_INPUT, f);
        let k = self.c32(u32::from(comp));
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, k]);
        self.m.op(op::LOAD, f, &[ptr])
    }

    /// A mesh output array (per vertex, or per primitive for the indices).
    fn mesh_output(&mut self, slot: u16) -> (Id, Id) {
        if let Some(&o) = self.outputs.get(&slot) {
            return o;
        }
        let (u, f) = (self.t.u32_, self.t.f32_);
        let (mv, mp) = self.st.mesh;
        let (elem, n, var) = match slot {
            OUT_PRIMITIVE => {
                let v3 = self.m.t_vec(u, 3);
                let arr = self.m.t_array(v3, mp.max(1));
                let p = self.m.t_ptr(en::SC_OUTPUT, arr);
                let v = self.m.variable(p, en::SC_OUTPUT);
                self.m.decorate(v, en::DEC_BUILTIN, &[en::BI_PRIMITIVE_TRIANGLE_INDICES_EXT]);
                (u, mp, v)
            }
            _ => {
                let v4 = self.m.t_vec(f, 4);
                let arr = self.m.t_array(v4, mv.max(1));
                let p = self.m.t_ptr(en::SC_OUTPUT, arr);
                let v = self.m.variable(p, en::SC_OUTPUT);
                if slot == OUT_POSITION {
                    self.m.decorate(v, en::DEC_BUILTIN, &[en::BI_POSITION]);
                } else {
                    self.m.decorate(v, en::DEC_LOCATION, &[u32::from(slot)]);
                }
                (f, mv, v)
            }
        };
        let _ = n;
        self.interface.push(var);
        self.outputs.insert(slot, (var, elem));
        (var, elem)
    }

    /// Mesh: every primitive culled until its export (so the counts can be
    /// the maxima, declared at the start).
    fn mesh_prologue(&mut self) {
        let (mv, mp) = self.st.mesh;
        let (u, b) = (self.t.u32_, self.t.bool_);
        let (kv, kp) = (self.c32(mv), self.c32(mp));
        self.m.op_void(op::SET_MESH_OUTPUTS, &[kv, kp]);
        let arr = self.m.t_array(b, mp.max(1));
        let p = self.m.t_ptr(en::SC_OUTPUT, arr);
        let cull = self.m.variable(p, en::SC_OUTPUT);
        self.m.decorate(cull, en::DEC_BUILTIN, &[en::BI_CULL_PRIMITIVE_EXT]);
        self.m.decorate(cull, en::DEC_PER_PRIMITIVE_EXT, &[]);
        self.interface.push(cull);
        self.cull = cull;
        let me = self.builtin(Builtin::LocalInvocationIndex);
        let threads = self.st.local_size[0].max(1);
        let pb = self.m.t_ptr(en::SC_OUTPUT, b);
        let t = self.m.c_bool(true);
        let mut k = 0;
        while k < mp {
            let kk = self.c32(k);
            let i = self.m.op(op::I_ADD, u, &[me, kk]);
            let inside = self.m.op(op::U_LESS_THAN, b, &[i, kp]);
            let (lt, lm) = (self.m.id(), self.m.id());
            self.m.op_void(op::SELECTION_MERGE, &[lm, 0]);
            self.m.op_void(op::BRANCH_CONDITIONAL, &[inside, lt, lm]);
            self.m.label(lt);
            let ptr = self.m.op(op::ACCESS_CHAIN, pb, &[cull, i]);
            self.m.op_void(op::STORE, &[ptr, t]);
            self.m.op_void(op::BRANCH, &[lm]);
            self.m.label(lm);
            self.block = lm;
            k += threads;
        }
    }

    /// A pixel input read per vertex: `PerVertexKHR` array of three vec4s at
    /// its semantic's location.
    fn per_vertex_input(&mut self, row: u8, chan: u8, vertex: u8) -> Id {
        let (loc, comp) = self.st.locs.get(&(row, chan)).copied().unwrap_or((31, chan));
        let f = self.t.f32_;
        let var = if let Some(&v) = self.inputs.get(&(0x80 | loc)) {
            v
        } else {
            self.m.extension("SPV_KHR_fragment_shader_barycentric");
            self.m.capability(en::CAP_FRAGMENT_BARYCENTRIC_KHR);
            let v4 = self.m.t_vec(f, 4);
            let arr = self.m.t_array(v4, 3);
            let p = self.m.t_ptr(en::SC_INPUT, arr);
            let v = self.m.variable(p, en::SC_INPUT);
            self.m.decorate(v, en::DEC_LOCATION, &[u32::from(loc)]);
            self.m.decorate(v, 5285, &[]); // PerVertexKHR
            self.interface.push(v);
            self.inputs.insert(0x80 | loc, v);
            v
        };
        let p = self.m.t_ptr(en::SC_INPUT, f);
        let (kv, kc) = (self.c32(u32::from(vertex)), self.c32(u32::from(comp)));
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, kv, kc]);
        self.m.op(op::LOAD, f, &[ptr])
    }

    /// A vertex attribute: Location = its input-signature row, a vec4 of
    /// floats or unsigned integers.
    fn vertex_input(&mut self, row: u8, chan: u8, ty: Ty) -> Id {
        let e = if ty == Ty::F32 { self.t.f32_ } else { self.t.u32_ };
        let var = if let Some(&v) = self.inputs.get(&row) {
            v
        } else {
            let v4 = self.m.t_vec(e, 4);
            let p = self.m.t_ptr(en::SC_INPUT, v4);
            let v = self.m.variable(p, en::SC_INPUT);
            self.m.decorate(v, en::DEC_LOCATION, &[u32::from(row)]);
            self.interface.push(v);
            self.inputs.insert(row, v);
            v
        };
        let p = self.m.t_ptr(en::SC_INPUT, e);
        let k = self.c32(u32::from(chan));
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, k]);
        self.m.op(op::LOAD, e, &[ptr])
    }

    fn output(&mut self, slot: u16) -> (Id, Id) {
        if self.st.stage == HwStage::Ms {
            return self.mesh_output(slot);
        }
        if let Some(&o) = self.outputs.get(&slot) {
            return o;
        }
        let (u, f) = (self.t.u32_, self.t.f32_);
        let (ty, elem) = match slot {
            OUT_POSITION => (self.m.t_vec(f, 4), f),
            OUT_DEPTH => (f, f),
            OUT_STENCIL => (u, u),
            OUT_SAMPLE_MASK => (self.m.t_array(u, 1), u),
            _ => {
                let e = match self.st.out_types.get(&slot).copied().unwrap_or(0) {
                    0 => f,
                    _ => u,
                };
                (self.m.t_vec(e, 4), e)
            }
        };
        let p = self.m.t_ptr(en::SC_OUTPUT, ty);
        let v = self.m.variable(p, en::SC_OUTPUT);
        match slot {
            OUT_POSITION => self.m.decorate(v, en::DEC_BUILTIN, &[en::BI_POSITION]),
            OUT_DEPTH => self.m.decorate(v, en::DEC_BUILTIN, &[en::BI_FRAG_DEPTH]),
            OUT_STENCIL => {
                self.m.extension("SPV_EXT_shader_stencil_export");
                self.m.capability(en::CAP_STENCIL_EXPORT);
                self.m.decorate(v, en::DEC_BUILTIN, &[en::BI_FRAG_STENCIL_REF]);
            }
            OUT_SAMPLE_MASK => self.m.decorate(v, en::DEC_BUILTIN, &[en::BI_SAMPLE_MASK]),
            _ => {
                self.m.decorate(v, en::DEC_LOCATION, &[u32::from(slot)]);
                if self.st.stage == HwStage::Ps && elem == u {
                    // integer outputs need no interpolation decoration
                }
            }
        }
        self.interface.push(v);
        self.outputs.insert(slot, (v, elem));
        (v, elem)
    }

    fn lds_var(&mut self) -> Id {
        if self.lds == 0 {
            let u = self.t.u32_;
            let arr = self.m.t_array(u, self.iface.lds_dwords.max(1) + 1);
            let p = self.m.t_ptr(en::SC_WORKGROUP, arr);
            self.lds = self.m.variable(p, en::SC_WORKGROUP);
            self.interface.push(self.lds);
        }
        self.lds
    }

    /// The image array variable for a class; `depth` for comparisons.
    fn image_var(&mut self, c: ImageClass, uint: bool, storage: bool, depth: bool) -> (Id, Id) {
        self.image_var_f(c, uint, storage, depth, 0)
    }

    /// The same with an image format (R32ui = 33 for atomics).
    fn image_var_f(&mut self, c: ImageClass, uint: bool, storage: bool, depth: bool, format: u32) -> (Id, Id) {
        // cube images cannot be fetched or stored: as 2D arrays then
        let (dim, arrayed) = if c.dim == 3 && (storage || !c.storage && c.dynamic) { (1, true) } else { (c.dim, c.arrayed) };
        let key = (dim, arrayed, c.ms, uint, storage, depth, format);
        let sty = if uint { self.t.u32_ } else { self.t.f32_ };
        let ity = self.m.t_image_df(sty, u32::from(dim), depth, arrayed, c.ms, if storage { 2 } else { 1 }, format);
        if let Some(&v) = self.images.get(&key) {
            return (v, ity);
        }
        if dim == 0 {
            self.m.capability(if storage { en::CAP_IMAGE_1D } else { en::CAP_SAMPLED_1D });
        }
        if dim == 3 && arrayed {
            self.m.capability(en::CAP_SAMPLED_CUBE_ARRAY);
        }
        if storage && c.ms {
            self.m.capability(en::CAP_STORAGE_IMAGE_MULTISAMPLE);
        }
        if storage {
            self.m.capability(en::CAP_STORAGE_IMAGE_READ_WITHOUT_FORMAT);
            self.m.capability(en::CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT);
        }
        let arr = self.m.t_array(ity, 1 << 20);
        let p = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, arr);
        let v = self.m.variable(p, en::SC_UNIFORM_CONSTANT);
        self.m.decorate(v, en::DEC_DESCRIPTOR_SET, &[0]);
        self.m.decorate(v, en::DEC_BINDING, &[u32::from(storage)]);
        self.interface.push(v);
        self.images.insert(key, v);
        (v, ity)
    }

    fn load_image(&mut self, c: ImageClass, uint: bool, storage: bool, depth: bool, tw: Id) -> (Id, Id) {
        let (var, ity) = self.image_var(c, uint, storage, depth);
        let k = self.c32(0xFF_FFFF);
        let u = self.t.u32_;
        let slot = self.m.op(op::BITWISE_AND, u, &[tw, k]);
        let p = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, ity);
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, slot]);
        (self.m.op(op::LOAD, ity, &[ptr]), ptr)
    }

    /// Samplers: set 1, binding 0, `SAMPLER[4096]`. A heap sampler is at
    /// the slot in its S#; immediate sampler n at the index specialization
    /// constant 100 + n gives (the host's reserved slots).
    fn load_sampler(&mut self, s: Sampler, sw: Option<Id>) -> Id {
        let sty = self.m.t_sampler();
        let p = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, sty);
        if self.samplers == 0 {
            let arr = self.m.t_array(sty, 4096);
            let pa = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, arr);
            self.samplers = self.m.variable(pa, en::SC_UNIFORM_CONSTANT);
            self.m.decorate(self.samplers, en::DEC_DESCRIPTOR_SET, &[1]);
            self.m.decorate(self.samplers, en::DEC_BINDING, &[0]);
            self.interface.push(self.samplers);
        }
        let u = self.t.u32_;
        let idx = match s {
            Sampler::Static(n) => match self.spec_samplers.get(&n) {
                Some(&id) => id,
                None => {
                    let id = self.m.spec_u32(u, 0);
                    self.m.decorate(id, en::DEC_SPEC_ID, &[100 + n]);
                    self.spec_samplers.insert(n, id);
                    id
                }
            },
            _ => {
                let k = self.c32(0xFFF);
                self.m.op(op::BITWISE_AND, u, &[sw.unwrap(), k])
            }
        };
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[self.samplers, idx]);
        self.m.op(op::LOAD, sty, &[ptr])
    }

    // ---- helpers ----

    fn helper(&mut self, h: Helper) -> Id {
        if let Some(&id) = self.helpers.get(&h) {
            return id;
        }
        let id = self.m.id();
        self.helpers.insert(h, id);
        id
    }

    /// Writes the helper functions' bodies (after main).
    fn write_helpers(&mut self) {
        let mut helpers: Vec<(Helper, Id)> = self.helpers.iter().map(|(h, i)| (*h, *i)).collect();
        if let Some(k) = helpers.iter().position(|h| h.0 == Helper::Typed) {
            let (_, fid) = helpers.remove(k);
            self.write_typed(fid);
        }
        if helpers.is_empty() {
            return;
        }
        // Specialization constant 0: the map has one range (the default).
        let b = self.t.bool_;
        let one_range = self.m.spec_true(b);
        self.m.decorate(one_range, en::DEC_SPEC_ID, &[0]);
        for (h, fid) in helpers {
            let (u, q) = (self.t.u32_, self.t.u64_);
            let range = h == Helper::TranslateRange;
            let (ret, params): (Id, Vec<Id>) = if range { (q, vec![q, u]) } else { (q, vec![q]) };
            let fty = self.m.global_fn_type(ret, &params);
            self.m.op_to(op::FUNCTION, ret, fid, &[helper_control("map"), fty]);
            let ps: Vec<Id> = params
                .iter()
                .map(|&t| {
                    let id = self.m.id();
                    self.m.op_to(55, t, id, &[]); // OpFunctionParameter
                    id
                })
                .collect();
            let entry = self.m.id();
            self.m.label(entry);
            let map = self.map_ptr();
            let z = self.c32(0);
            let zero = self.map_load_at(map, q, &[z]);
            let bytes = if range { self.m.op(op::U_CONVERT, q, &[ps[1]]) } else { 0 };
            // device address and byte limit through range `i` (an id)
            let through = |e: &mut Self, i: Id| -> (Id, Id) {
                let (k3, k0, k1, k2) = (e.c32(3), e.c32(0), e.c32(1), e.c32(2));
                let start = e.map_load_at(map, q, &[k3, i, k0]);
                let size = e.map_load_at(map, q, &[k3, i, k1]);
                let base = e.map_load_at(map, q, &[k3, i, k2]);
                let d = e.m.op(op::I_SUB, q, &[ps[0], start]);
                let mut hit = e.m.op(op::U_LESS_THAN, b, &[d, size]);
                let mut lim = 0;
                if range {
                    let left = e.m.op(op::I_SUB, q, &[size, d]);
                    let fits = e.m.op(op::U_GREATER_THAN_EQUAL, b, &[left, bytes]);
                    hit = e.m.op(op::LOGICAL_AND, b, &[hit, fits]);
                    let z = e.c32(0);
                    lim = e.m.op(op::SELECT, u, &[hit, ps[1], z]);
                }
                let dev = e.m.op(op::I_ADD, q, &[base, d]);
                (e.m.op(op::SELECT, q, &[hit, dev, zero]), lim)
            };
            if Self::no_search() {
                let r0 = through(self, z);
                self.m.op_void(254, &[r0.0]);
                self.m.op_void(op::FUNCTION_END, &[]);
                continue;
            }
            let (lf, ls, lm) = (self.m.id(), self.m.id(), self.m.id());
            self.m.op_void(op::SELECTION_MERGE, &[lm, 0]);
            self.m.op_void(op::BRANCH_CONDITIONAL, &[one_range, lf, ls]);
            self.m.label(lf);
            let r0 = through(self, z);
            self.m.op_void(op::BRANCH, &[lm]);
            self.m.label(ls);
            // binary search over the sorted ranges
            let k2 = self.c32(2);
            let count = self.map_load_at(map, u, &[k2]);
            let mut lo = self.c32(0);
            for step in [32u32, 16, 8, 4, 2, 1] {
                let ks = self.c32(step);
                let mid = self.m.op(op::I_ADD, u, &[lo, ks]);
                let inside = self.m.op(op::U_LESS_THAN, b, &[mid, count]);
                let k3 = self.c32(3);
                let k63 = self.c32(63);
                let midc = self.m.ext(glsl::U_MIN, u, &[mid, k63]);
                let start = self.map_load_at(map, q, &[k3, midc, z]);
                let le = self.m.op(op::U_LESS_THAN_EQUAL, b, &[start, ps[0]]);
                let ok = self.m.op(op::LOGICAL_AND, b, &[inside, le]);
                lo = self.m.op(op::SELECT, u, &[ok, mid, lo]);
            }
            let r1 = through(self, lo);
            self.m.op_void(op::BRANCH, &[lm]);
            self.m.label(lm);
            let res = self.m.op(op::PHI, q, &[r0.0, lf, r1.0, ls]);
            self.m.op_void(254, &[res]); // OpReturnValue
            self.m.op_void(op::FUNCTION_END, &[]);
        }
    }

    /// The typed-buffer decoder: a switch over the GFX10 FORMAT codes the
    /// title's views use; others read one 32-bit channel.
    fn write_typed(&mut self, fid: Id) {
        let (u, q, f) = (self.t.u32_, self.t.u64_, self.t.f32_);
        let v4 = self.m.t_vec(u, 4);
        let fty = self.m.global_fn_type(v4, &[q, u]);
        self.m.op_to(op::FUNCTION, v4, fid, &[helper_control("typed"), fty]);
        let (addr, fmt) = (self.m.id(), self.m.id());
        self.m.op_to(55, q, addr, &[]);
        self.m.op_to(55, u, fmt, &[]);
        let entry = self.m.id();
        self.m.label(entry);
        // (codes, components, bits, num: 0 float 1 unorm 2 snorm 3 uint 4 sint)
        let table: &[(&[u32], u32, u32, u32)] = &[
            (&[22], 1, 32, 0),
            (&[20, 21], 1, 32, 3),
            (&[64], 2, 32, 0),
            (&[62, 63], 2, 32, 3),
            (&[74], 3, 32, 0),
            (&[72, 73], 3, 32, 3),
            (&[77], 4, 32, 0),
            (&[75, 76], 4, 32, 3),
            (&[13], 1, 16, 0),
            (&[29], 2, 16, 0),
            (&[71], 4, 16, 0),
            (&[7], 1, 16, 1),
            (&[23], 2, 16, 1),
            (&[65], 4, 16, 1),
            (&[8], 1, 16, 2),
            (&[24], 2, 16, 2),
            (&[66], 4, 16, 2),
            (&[11], 1, 16, 3),
            (&[27], 2, 16, 3),
            (&[69], 4, 16, 3),
            (&[12], 1, 16, 4),
            (&[28], 2, 16, 4),
            (&[70], 4, 16, 4),
            (&[1], 1, 8, 1),
            (&[14], 2, 8, 1),
            (&[56], 4, 8, 1),
            (&[2], 1, 8, 2),
            (&[15], 2, 8, 2),
            (&[57], 4, 8, 2),
            (&[5], 1, 8, 3),
            (&[18], 2, 8, 3),
            (&[60], 4, 8, 3),
            (&[6], 1, 8, 4),
            (&[19], 2, 8, 4),
            (&[61], 4, 8, 4),
            (&[50], 4, 0, 1),
        ];
        let merge = self.m.id();
        let default = self.m.id();
        let labels: Vec<Id> = table.iter().map(|_| self.m.id()).collect();
        let mut sw = vec![fmt, default];
        for (i, e) in table.iter().enumerate() {
            for &c in e.0 {
                sw.push(c);
                sw.push(labels[i]);
            }
        }
        self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
        self.m.op_void(251, &sw); // OpSwitch
        let one_f = self.m.c_u32(1.0f32.to_bits());
        let one_i = self.c32(1);
        let z = self.c32(0);
        let mut incoming = Vec::new();
        for (i, &(_, comps, bits, num)) in table.iter().enumerate() {
            self.m.label(labels[i]);
            let mut ch = Vec::new();
            if bits == 0 {
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[addr]);
                let w = self.m.op(op::LOAD, u, &[ptr, en::MEM_ALIGNED, 4]);
                let mut at = 0;
                for width in [10u32, 10, 10, 2] {
                    let (o, wv) = (self.c32(at), self.c32(width));
                    at += width;
                    let r = self.m.op(op::BIT_FIELD_U_EXTRACT, u, &[w, o, wv]);
                    let fl = self.m.op(op::CONVERT_U_TO_F, f, &[r]);
                    let k = self.m.c_f32((1.0 / ((1u64 << width) - 1) as f32).to_bits());
                    let m = self.m.op(op::F_MUL, f, &[fl, k]);
                    ch.push(self.m.op(op::BITCAST, u, &[m]));
                }
            } else {
                let st = match bits {
                    32 => S::U32,
                    16 => S::U16,
                    _ => S::U8,
                };
                let et = self.scalar(st);
                let lt = self.m.t_vec(et, comps);
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, lt);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[addr]);
                let v = self.m.op(op::LOAD, lt, &[ptr, en::MEM_ALIGNED, (bits / 8).max(1)]);
                for c in 0..comps {
                    let e = if comps == 1 { v } else { self.m.op(op::COMPOSITE_EXTRACT, et, &[v, c]) };
                    let x = match (bits, num) {
                        (32, _) => e,
                        (16, 0) => {
                            let h = self.scalar(S::F16);
                            let hf = self.m.op(op::BITCAST, h, &[e]);
                            let fl = self.m.op(op::F_CONVERT, f, &[hf]);
                            self.m.op(op::BITCAST, u, &[fl])
                        }
                        (_, 1) => {
                            let i = self.m.op(op::U_CONVERT, u, &[e]);
                            let fl = self.m.op(op::CONVERT_U_TO_F, f, &[i]);
                            let k = self.m.c_f32((1.0 / ((1u64 << bits) - 1) as f32).to_bits());
                            let m = self.m.op(op::F_MUL, f, &[fl, k]);
                            self.m.op(op::BITCAST, u, &[m])
                        }
                        (_, 2) => {
                            let i = self.m.op(op::S_CONVERT, u, &[e]);
                            let fl = self.m.op(op::CONVERT_S_TO_F, f, &[i]);
                            let k = self.m.c_f32((1.0 / ((1u64 << (bits - 1)) - 1) as f32).to_bits());
                            let m = self.m.op(op::F_MUL, f, &[fl, k]);
                            let lo = self.m.c_f32((-1.0f32).to_bits());
                            let c = self.m.ext(glsl::N_MAX, f, &[m, lo]);
                            self.m.op(op::BITCAST, u, &[c])
                        }
                        (_, 3) => self.m.op(op::U_CONVERT, u, &[e]),
                        _ => self.m.op(op::S_CONVERT, u, &[e]),
                    };
                    ch.push(x);
                }
                while ch.len() < 4 {
                    let alpha = ch.len() == 3;
                    ch.push(if !alpha {
                        z
                    } else if num >= 3 {
                        one_i
                    } else {
                        one_f
                    });
                }
            }
            let r = self.m.op(op::COMPOSITE_CONSTRUCT, v4, &ch);
            incoming.push((r, labels[i]));
            self.m.op_void(op::BRANCH, &[merge]);
        }
        self.m.label(default);
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
        let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[addr]);
        let w = self.m.op(op::LOAD, u, &[ptr, en::MEM_ALIGNED, 4]);
        let r = self.m.op(op::COMPOSITE_CONSTRUCT, v4, &[w, z, z, one_f]);
        incoming.push((r, default));
        self.m.op_void(op::BRANCH, &[merge]);
        self.m.label(merge);
        let mut ops = Vec::new();
        for (r, l) in incoming {
            ops.push(r);
            ops.push(l);
        }
        let res = self.m.op(op::PHI, v4, &ops);
        self.m.op_void(254, &[res]);
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    // ---- values ----

    fn emit_node(&mut self, v: V) {
        let n = self.f.node(v).clone();
        let ty = n.ty;
        let tid = if ty == Ty::VOID { self.t.void } else { self.ty(ty) };
        let args: Vec<Id> = n.args.iter().map(|&a| self.id(a)).collect();
        let id = match n.op {
            Op::Const(bits) => self.constant(ty, bits),
            Op::Undef => self.m.undef(tid),
            Op::Push(k) => {
                let var = self.push_var();
                let u = self.t.u32_;
                let p = self.m.t_ptr(en::SC_PUSH_CONSTANT, u);
                let (z, kk) = (self.c32(0), self.c32(k));
                let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, z, kk]);
                self.m.op(op::LOAD, u, &[ptr])
            }
            Op::MapField(k) => {
                let q = self.t.u64_;
                self.map_load(q, &[k])
            }
            Op::Builtin(b) => self.builtin(b),
            Op::Input { loc, comp, interp } => self.input(loc, comp, interp, ty),
            Op::PerVertex { loc, comp, vertex } => self.per_vertex_input(loc, comp, vertex),
            Op::Spv(o) => {
                let mut ops = args.clone();
                ops.extend_from_slice(&n.imm);
                match o {
                    op::U_MUL_EXTENDED | op::S_MUL_EXTENDED => {
                        let u = self.t.u32_;
                        let st = self.m.t_struct(&[u, u], 5);
                        self.m.op(o, st, &ops)
                    }
                    _ => self.m.op(o, tid, &ops),
                }
            }
            Op::Glsl(g) => self.m.ext(g, tid, &args),
            Op::Call(h) => {
                let fid = self.helper(h);
                let mut ops = vec![fid];
                ops.extend_from_slice(&args);
                self.m.op(57, tid, &ops) // OpFunctionCall
            }
            Op::Load { align } => {
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, tid);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[args[0]]);
                self.m.op(op::LOAD, tid, &[ptr, en::MEM_ALIGNED, align])
            }
            Op::Store { align } => {
                let vt = self.f.ty(n.args[1]);
                let vtid = self.ty(vt);
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, vtid);
                let a = self.predicated_addr(args[0]);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[a]);
                self.m.op_void(op::STORE, &[ptr, args[1], en::MEM_ALIGNED, align]);
                0
            }
            Op::Atomic(o) => {
                let u = self.t.u32_;
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
                let a = self.predicated_addr(args[0]);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[a]);
                let (scope, sem) = (self.c32(en::SCOPE_DEVICE), self.c32(0));
                if o == op::ATOMIC_COMPARE_EXCHANGE {
                    self.m.op(o, u, &[ptr, scope, sem, sem, args[1], args[2]])
                } else {
                    self.m.op(o, u, &[ptr, scope, sem, args[1]])
                }
            }
            Op::LdsLoad | Op::LdsStore | Op::LdsAtomic(_) => {
                let lds = self.lds_var();
                let u = self.t.u32_;
                let p = self.m.t_ptr(en::SC_WORKGROUP, u);
                let max = self.c32(self.iface.lds_dwords.max(1) - 1);
                let mut idx = self.m.ext(glsl::U_MIN, u, &[args[0], max]);
                if let Some(c) = self.pred
                    && n.op != Op::LdsLoad
                {
                    // the lanes outside a flat guard write the spare dword
                    let spare = self.c32(self.iface.lds_dwords.max(1));
                    idx = self.m.op(op::SELECT, u, &[c, idx, spare]);
                }
                let ptr = self.m.op(op::ACCESS_CHAIN, p, &[lds, idx]);
                match n.op {
                    Op::LdsLoad => self.m.op(op::LOAD, u, &[ptr]),
                    Op::LdsStore => {
                        self.m.op_void(op::STORE, &[ptr, args[1]]);
                        0
                    }
                    Op::LdsAtomic(o) => {
                        let (scope, sem) = (self.c32(en::SCOPE_WORKGROUP), self.c32(0));
                        if o == op::ATOMIC_COMPARE_EXCHANGE {
                            self.m.op(o, u, &[ptr, scope, sem, sem, args[1], args[2]])
                        } else {
                            self.m.op(o, u, &[ptr, scope, sem, args[1]])
                        }
                    }
                    _ => unreachable!(),
                }
            }
            Op::Image { kind, class, sampler, flags } => self.image(v, kind, class, sampler, flags, &args, tid),
            Op::Wave(o, g) => {
                self.m.capability(en::CAP_GROUP_NON_UNIFORM);
                match o {
                    op::GROUP_NON_UNIFORM_BALLOT => self.m.capability(en::CAP_GROUP_NON_UNIFORM_BALLOT),
                    op::GROUP_NON_UNIFORM_ANY | op::GROUP_NON_UNIFORM_ALL => {
                        self.m.capability(en::CAP_GROUP_NON_UNIFORM_VOTE)
                    }
                    op::GROUP_NON_UNIFORM_BROADCAST_FIRST | op::GROUP_NON_UNIFORM_BROADCAST => {
                        self.m.capability(en::CAP_GROUP_NON_UNIFORM_BALLOT)
                    }
                    op::GROUP_NON_UNIFORM_SHUFFLE | op::GROUP_NON_UNIFORM_SHUFFLE_XOR => {
                        self.m.capability(en::CAP_GROUP_NON_UNIFORM_SHUFFLE)
                    }
                    _ => self.m.capability(en::CAP_GROUP_NON_UNIFORM_ARITHMETIC),
                }
                let scope = self.c32(en::SCOPE_SUBGROUP);
                let mut ops = vec![scope];
                if g != u32::MAX {
                    ops.push(g);
                }
                ops.extend_from_slice(&args);
                self.m.op(o, tid, &ops)
            }
            Op::Output { slot, comp } => {
                // user outputs of vertex and mesh programs at their semantics' locations
                let (slot, comp) = if slot < 0x100 && self.st.stage != HwStage::Ps {
                    match self.st.locs.get(&(slot as u8, comp)) {
                        Some(&(l, c)) => (u16::from(l), c),
                        None => {
                            self.ids[v as usize] = 0;
                            return;
                        }
                    }
                } else {
                    (slot, comp)
                };
                let (var, elem) = self.output(slot);
                if self.st.stage == HwStage::Ms {
                    // [index] of the arrays (clamped), then the component
                    let (u, b) = (self.t.u32_, self.t.bool_);
                    let max = if slot == OUT_PRIMITIVE { self.st.mesh.1 } else { self.st.mesh.0 };
                    let km = self.c32(max.max(1) - 1);
                    let i = self.m.ext(glsl::U_MIN, u, &[args[1], km]);
                    if slot == OUT_PRIMITIVE {
                        let x = if self.f.ty(n.args[0]) == Ty::F32 { self.m.op(op::BITCAST, u, &[args[0]]) } else { args[0] };
                        let m = self.c32(0x3FF);
                        let (k10, k20, k31) = (self.c32(10), self.c32(20), self.c32(31));
                        let i0 = self.m.op(op::BITWISE_AND, u, &[x, m]);
                        let s1 = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[x, k10]);
                        let i1 = self.m.op(op::BITWISE_AND, u, &[s1, m]);
                        let s2 = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[x, k20]);
                        let i2 = self.m.op(op::BITWISE_AND, u, &[s2, m]);
                        let v3 = self.m.t_vec(u, 3);
                        let tri = self.m.op(op::COMPOSITE_CONSTRUCT, v3, &[i0, i1, i2]);
                        let p3 = self.m.t_ptr(en::SC_OUTPUT, v3);
                        let ptr = self.m.op(op::ACCESS_CHAIN, p3, &[var, i]);
                        self.m.op_void(op::STORE, &[ptr, tri]);
                        let c = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[x, k31]);
                        let z = self.c32(0);
                        let culled = self.m.op(op::I_NOT_EQUAL, b, &[c, z]);
                        let pb = self.m.t_ptr(en::SC_OUTPUT, b);
                        let cp = self.m.op(op::ACCESS_CHAIN, pb, &[self.cull, i]);
                        self.m.op_void(op::STORE, &[cp, culled]);
                    } else {
                        let x = if self.f.ty(n.args[0]) == Ty::F32 { args[0] } else { self.m.op(op::BITCAST, elem, &[args[0]]) };
                        let p = self.m.t_ptr(en::SC_OUTPUT, elem);
                        let k = self.c32(u32::from(comp));
                        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, i, k]);
                        self.m.op_void(op::STORE, &[ptr, x]);
                    }
                    self.ids[v as usize] = 0;
                    return;
                }
                let at = self.f.ty(n.args[0]);
                let at = self.ty(at);
                let x = if at != elem { self.m.op(op::BITCAST, elem, &[args[0]]) } else { args[0] };
                let ptr = match slot {
                    OUT_DEPTH | OUT_STENCIL => var,
                    _ => {
                        let p = self.m.t_ptr(en::SC_OUTPUT, elem);
                        let k = self.c32(u32::from(comp));
                        self.m.op(op::ACCESS_CHAIN, p, &[var, k])
                    }
                };
                self.m.op_void(op::STORE, &[ptr, x]);
                0
            }
            Op::Demote => {
                self.m.capability(en::CAP_DEMOTE_TO_HELPER_INVOCATION);
                self.m.op_void(op::DEMOTE_TO_HELPER_INVOCATION, &[]);
                0
            }
            Op::Barrier => {
                let wg = self.c32(en::SCOPE_WORKGROUP);
                let sem = self.c32(en::SEM_ACQUIRE_RELEASE | en::SEM_WORKGROUP_MEMORY);
                self.m.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
                0
            }
            Op::Phi => unreachable!("guard phis are written by their guard"),
            Op::Pick => {
                let var = self.picks[&v];
                let u = self.t.u32_;
                let p = self.m.t_ptr(en::SC_FUNCTION, u);
                for (i, &a) in args[1..].iter().enumerate() {
                    let k = self.c32(i as u32);
                    let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, k]);
                    self.m.op_void(op::STORE, &[ptr, a]);
                }
                let max = self.c32((args.len() - 2) as u32);
                let idx = self.m.ext(glsl::U_MIN, u, &[args[0], max]);
                let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, idx]);
                self.m.op(op::LOAD, u, &[ptr])
            }
        };
        self.ids[v as usize] = id;
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
            S::Void => 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn image(&mut self, _v: V, kind: ImageKind, class: ImageClass, sampler: Sampler, flags: u16, args: &[Id], tid: Id) -> Id {
        let (u, f) = (self.t.u32_, self.t.f32_);
        let v4u = self.m.t_vec(u, 4);
        let v4f = self.m.t_vec(f, 4);
        let tw = args[0];
        let mut i = 1;
        let sw = if sampler == Sampler::Dynamic {
            i += 1;
            Some(args[1])
        } else {
            None
        };
        let coord = args[i];
        i += 1;
        let mut opt = |bit: u16| -> Option<Id> {
            if flags & bit != 0 {
                let x = args[i];
                i += 1;
                Some(x)
            } else {
                None
            }
        };
        let dref = opt(img::DREF);
        let bias = opt(img::BIAS);
        let lod = opt(img::LOD);
        let grad = if flags & img::GRAD != 0 {
            let (a, b) = (args[i], args[i + 1]);
            i += 2;
            Some((a, b))
        } else {
            None
        };
        let mut opt = |bit: u16| -> Option<Id> {
            if flags & bit != 0 {
                let x = args[i];
                i += 1;
                Some(x)
            } else {
                None
            }
        };
        let offset = opt(img::OFFSET);
        let min_lod = opt(img::MIN_LOD);
        let sample = opt(img::SAMPLE);
        let data = opt(img::DATA);
        let cmp = opt(img::CMP);
        match kind {
            ImageKind::Sample { .. } | ImageKind::Gather { .. } | ImageKind::QueryLod => {
                let depth = dref.is_some();
                let (im, _) = self.load_image(class, false, false, depth, tw);
                let s = self.load_sampler(sampler, sw);
                let ity = self.m.t_image_d(f, u32::from(class.dim), depth, class.arrayed, class.ms, 1);
                let sity = self.m.t_sampled_image(ity);
                let si = self.m.op(op::SAMPLED_IMAGE, sity, &[im, s]);
                if kind == ImageKind::QueryLod {
                    self.m.capability(en::CAP_IMAGE_QUERY);
                    if !self.frag {
                        return self.m.c_null(tid);
                    }
                    let v2 = self.m.t_vec(f, 2);
                    return self.m.op(op::IMAGE_QUERY_LOD, v2, &[si, coord]);
                }
                let mut mask = 0u32;
                let mut ops = Vec::new();
                let implicit = lod.is_none() && grad.is_none();
                let explicit_lod = if implicit && !self.frag {
                    // no derivatives outside pixel shaders: level 0
                    Some(self.m.c_f32(0))
                } else {
                    lod
                };
                if let Some(b) = bias {
                    if self.frag && implicit {
                        mask |= en::IMG_BIAS;
                        ops.push(b);
                    }
                }
                if let Some(l) = explicit_lod {
                    mask |= en::IMG_LOD;
                    ops.push(l);
                }
                if let Some((dx, dy)) = grad {
                    mask |= en::IMG_GRAD;
                    ops.push(dx);
                    ops.push(dy);
                }
                if let Some(o) = offset {
                    mask |= en::IMG_OFFSET;
                    self.m.capability(en::CAP_IMAGE_GATHER_EXTENDED);
                    ops.push(o);
                }
                if let Some(m) = min_lod {
                    if matches!(kind, ImageKind::Sample { .. }) && (explicit_lod.is_none() || grad.is_some()) {
                        mask |= en::IMG_MIN_LOD;
                        self.m.capability(en::CAP_MIN_LOD);
                        ops.push(m);
                    }
                }
                let lodded = mask & (en::IMG_LOD | en::IMG_GRAD) != 0;
                if let ImageKind::Gather { comp, .. } = kind {
                    let mut o = vec![si, coord];
                    let gm = mask & !(en::IMG_LOD | en::IMG_GRAD | en::IMG_BIAS | en::IMG_MIN_LOD);
                    let gops: Vec<Id> = if gm & en::IMG_OFFSET != 0 { vec![offset.unwrap()] } else { vec![] };
                    let r = if let Some(d) = dref {
                        o.push(d);
                        if gm != 0 {
                            o.push(gm);
                            o.extend(gops);
                        }
                        self.m.op(op::IMAGE_DREF_GATHER, v4f, &o)
                    } else {
                        let c = self.c32(u32::from(comp));
                        o.push(c);
                        if gm != 0 {
                            o.push(gm);
                            o.extend(gops);
                        }
                        self.m.op(op::IMAGE_GATHER, v4f, &o)
                    };
                    return r;
                }
                let mut o = vec![si, coord];
                if let Some(d) = dref {
                    o.push(d);
                }
                if mask != 0 {
                    o.push(mask);
                    o.extend(ops);
                }
                if let Some(_d) = dref {
                    let opc = if lodded { op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD } else { op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD };
                    let r = self.m.op(opc, f, &o);
                    // the TIR reads a vec4 and takes component 0
                    let z = self.m.c_f32(0);
                    return self.m.op(op::COMPOSITE_CONSTRUCT, v4f, &[r, z, z, z]);
                }
                let opc = if lodded { op::IMAGE_SAMPLE_EXPLICIT_LOD } else { op::IMAGE_SAMPLE_IMPLICIT_LOD };
                self.m.op(opc, v4f, &o)
            }
            ImageKind::Load { .. } | ImageKind::Store { .. } | ImageKind::Size => {
                // runtime choice: storage bit 25, integer bit 24
                let b = self.t.bool_;
                let k25 = self.c32(1 << 25);
                let k24 = self.c32(1 << 24);
                let z = self.c32(0);
                let sbit = self.m.op(op::BITWISE_AND, u, &[tw, k25]);
                let is_storage = self.m.op(op::I_NOT_EQUAL, b, &[sbit, z]);
                let ibit = self.m.op(op::BITWISE_AND, u, &[tw, k24]);
                let is_int = self.m.op(op::I_NOT_EQUAL, b, &[ibit, z]);
                let store = matches!(kind, ImageKind::Store { .. });
                let size = kind == ImageKind::Size;
                let variants: Vec<(bool, bool)> = if store {
                    vec![(true, true), (true, false)]
                } else if size {
                    vec![(true, false), (false, false)]
                } else {
                    vec![(true, true), (true, false), (false, true), (false, false)]
                };
                let result_ty = if store { 0 } else if size { tid } else { v4u };
                let merge = self.m.id();
                let mut incoming = Vec::new();
                // a chain of selections: if (storage) { if (int) A else B } else { ... }
                let emit_variant = |e: &mut Self, storage: bool, int: bool| -> Id {
                    // a size query keeps a cube a cube (loads read it as an array)
                    let c = if size && !storage { ImageClass { dynamic: false, ..class } } else { class };
                    let (im, _) = e.load_image(c, int, storage, false, tw);
                    if size {
                        e.m.capability(en::CAP_IMAGE_QUERY);
                        if storage || class.ms {
                            return e.m.op(op::IMAGE_QUERY_SIZE, tid, &[im]);
                        }
                        return e.m.op(op::IMAGE_QUERY_SIZE_LOD, tid, &[im, lod.unwrap()]);
                    }
                    if store {
                        let texel = if int { data.unwrap() } else { e.m.op(op::BITCAST, v4f, &[data.unwrap()]) };
                        let mut o = vec![im, coord, texel];
                        if let Some(s) = sample {
                            o.push(en::IMG_SAMPLE);
                            o.push(s);
                        }
                        e.m.op_void(op::IMAGE_WRITE, &o);
                        return 0;
                    }
                    let rt = if int { v4u } else { v4f };
                    let r = if storage {
                        let mut o = vec![im, coord];
                        if let Some(s) = sample {
                            o.push(en::IMG_SAMPLE);
                            o.push(s);
                        }
                        e.m.op(op::IMAGE_READ, rt, &o)
                    } else {
                        let mut o = vec![im, coord];
                        if let Some(s) = sample {
                            o.push(en::IMG_SAMPLE);
                            o.push(s);
                        } else {
                            let l = lod.unwrap_or(z);
                            o.push(en::IMG_LOD);
                            o.push(l);
                        }
                        e.m.op(op::IMAGE_FETCH, rt, &o)
                    };
                    if int { r } else { e.m.op(op::BITCAST, v4u, &[r]) }
                };
                if variants.len() == 2 {
                    let cond = if store { is_int } else { is_storage };
                    let (lt, le) = (self.m.id(), self.m.id());
                    self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[cond, lt, le]);
                    for (lab, (s, i)) in [(lt, variants[0]), (le, variants[1])] {
                        self.m.label(lab);
                        self.block = lab;
                        let r = emit_variant(self, s, i);
                        incoming.push((r, self.block));
                        self.m.op_void(op::BRANCH, &[merge]);
                    }
                } else {
                    let (ls, ln) = (self.m.id(), self.m.id());
                    self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[is_storage, ls, ln]);
                    for (lab, storage) in [(ls, true), (ln, false)] {
                        self.m.label(lab);
                        let inner = self.m.id();
                        let (li, lf) = (self.m.id(), self.m.id());
                        self.m.op_void(op::SELECTION_MERGE, &[inner, 0]);
                        self.m.op_void(op::BRANCH_CONDITIONAL, &[is_int, li, lf]);
                        let mut inc = Vec::new();
                        for (l2, int) in [(li, true), (lf, false)] {
                            self.m.label(l2);
                            self.block = l2;
                            let r = emit_variant(self, storage, int);
                            inc.push((r, l2));
                            self.m.op_void(op::BRANCH, &[inner]);
                        }
                        self.m.label(inner);
                        let r = self.m.op(op::PHI, v4u, &[inc[0].0, inc[0].1, inc[1].0, inc[1].1]);
                        incoming.push((r, inner));
                        self.m.op_void(op::BRANCH, &[merge]);
                    }
                }
                self.m.label(merge);
                self.block = merge;
                if store {
                    return 0;
                }
                self.m.op(op::PHI, result_ty, &[incoming[0].0, incoming[0].1, incoming[1].0, incoming[1].1])
            }
            ImageKind::Levels => {
                self.m.capability(en::CAP_IMAGE_QUERY);
                if class.ms {
                    return self.c32(1);
                }
                let c = ImageClass { dynamic: false, ..class };
                let (im, _) = self.load_image(c, false, false, false, tw);
                self.m.op(op::IMAGE_QUERY_LEVELS, u, &[im])
            }
            ImageKind::Atomic(o) => {
                let (var, ity) = self.image_var_f(class, true, true, false, 33);
                let k = self.c32(0xFF_FFFF);
                let slot = self.m.op(op::BITWISE_AND, u, &[tw, k]);
                let pi = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, ity);
                let ptr = self.m.op(op::ACCESS_CHAIN, pi, &[var, slot]);
                let pt = self.m.t_ptr(11, u); // StorageClass Image
                let s = sample.unwrap_or_else(|| self.c32(0));
                let tp = self.m.op(60, pt, &[ptr, coord, s]); // OpImageTexelPointer
                let (scope, sem) = (self.c32(en::SCOPE_DEVICE), self.c32(0));
                if o == op::ATOMIC_COMPARE_EXCHANGE {
                    self.m.op(o, u, &[tp, scope, sem, sem, data.unwrap(), cmp.unwrap()])
                } else {
                    self.m.op(o, u, &[tp, scope, sem, data.unwrap()])
                }
            }
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
                        self.ids[p as usize] = id;
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
                        self.ids[p as usize] = id;
                    }
                }
            }
        }
    }

    /// A guard whose body has no effects and is small enough is emitted
    /// flat (`VICTORY_VK_PRED`-free policy: every such guard; the experiment
    /// switch `VK_PRED=n` caps its size, 0 keeps every guard a branch).
    fn predicable(&self, body: &[Item]) -> bool {
        static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let cap = *CAP.get_or_init(|| std::env::var("VK_PRED").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX));
        body.len() <= cap
            && body.iter().all(|it| {
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

    /// The values the edge `from -> to` gives `to`'s phis.
    fn edge_vals(&self, from: usize, to: usize) -> Vec<Id> {
        let b = &self.f.blocks[to];
        let k = b.preds.iter().position(|&q| q == from).expect("an edge of the graph");
        b.phis.iter().map(|&ph| self.id(self.f.node(ph).args[k])).collect()
    }

    fn phi_tys(&mut self, x: usize) -> Vec<Id> {
        let tys: Vec<Ty> = self.f.blocks[x].phis.iter().map(|&p| self.f.ty(p)).collect();
        tys.into_iter().map(|t| self.ty(t)).collect()
    }

    fn tree(&mut self, x: usize) {
        let g = self.g.as_ref().unwrap();
        let exit = if g.header[x] { g.exit[x] } else { None };
        let after: Vec<usize> = if exit.is_some() { g.after_exit[x].clone() } else { Vec::new() };
        let mut merges: Vec<usize> = g.children[x]
            .iter()
            .copied()
            .filter(|&c| g.merge[c] && Some(c) != exit && !after.contains(&c))
            .collect();
        merges.sort_by_key(|&c| std::cmp::Reverse(g.rpo[c]));
        let header = g.header[x];
        let entry = self.pending.remove(&x).unwrap_or_default();
        if header {
            self.loop_tree(x, &after, &merges, exit, entry);
        } else {
            let phis = self.f.blocks[x].phis.clone();
            for (k, &ph) in phis.iter().enumerate() {
                self.ids[ph as usize] = entry[k];
            }
            let b = self.f.blocks[x].body.clone();
            self.items(&b);
            self.within(x, &merges);
        }
    }

    /// A loop headed by `x`: the blocks of the merges that follow its exit
    /// wrap it; the exit comes at the loop construct's merge.
    fn loop_tree(&mut self, x: usize, after: &[usize], merges: &[usize], exit: Option<usize>, entry: Vec<Id>) {
        if let Some((&z, rest)) = after.split_first() {
            let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
            self.branch_to(h);
            self.m.label(h);
            self.block = h;
            self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
            let body = self.m.id();
            self.branch_to(body);
            self.m.label(body);
            self.block = body;
            self.ctx.push(Ctx::Block(z));
            self.cons.push(Construct { node: z, head: usize::MAX, header: h, merge, cont, passes: false, wanted: 0, counter: None, arrivals: Vec::new() });
            self.loop_tree(x, rest, merges, exit, entry);
            self.close();
            self.tree(z);
            return;
        }
        let phis = self.f.blocks[x].phis.clone();
        let z = self.c32(0);
        let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
        let pre = self.block;
        self.branch_to(h);
        self.m.label(h);
        self.block = h;
        // the iteration count
        let u = self.t.u32_;
        let next = self.m.id();
        let count = self.m.op(op::PHI, u, &[z, pre, next, cont]);
        let counter = (count, next);
        let tys = self.phi_tys(x);
        let cids: Vec<Id> = phis.iter().map(|_| self.m.id()).collect();
        for (k, &ph) in phis.iter().enumerate() {
            let id = self.m.op(op::PHI, tys[k], &[entry[k], pre, cids[k], cont]);
            self.ids[ph as usize] = id;
        }
        self.cphi.insert(x, cids);
        self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
        let body = self.m.id();
        self.branch_to(body);
        self.m.label(body);
        self.block = body;
        self.ctx.push(Ctx::Loop(x));
        self.cons.push(Construct { node: x, head: usize::MAX, header: h, merge, cont, passes: false, wanted: 0, counter: Some(counter), arrivals: Vec::new() });
        let b = self.f.blocks[x].body.clone();
        self.items(&b);
        self.within(x, merges);
        self.close();
        match exit {
            // the loop's exit follows it
            Some(y) => self.tree(y),
            // a loop is left only by jumps to outer constructs
            None => self.m.op_void(op::UNREACHABLE, &[]),
        }
    }

    fn within(&mut self, x: usize, merges: &[usize]) {
        match merges.split_first() {
            None => match self.f.blocks[x].term.clone() {
                Term::Jump(t) => self.jump(x, t),
                Term::Branch(c, t, e) => {
                    // both arms leave (an unreachable merge): jumps out of it
                    // to a selection's merge are not allowed
                    let c = self.id(c);
                    let (lt, le, sm) = (self.m.id(), self.m.id(), self.m.id());
                    self.m.op_void(op::SELECTION_MERGE, &[sm, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
                    self.base.push(self.ctx.len());
                    self.m.label(lt);
                    self.block = lt;
                    self.jump(x, t);
                    self.m.label(le);
                    self.block = le;
                    self.jump(x, e);
                    self.base.pop();
                    self.m.label(sm);
                    self.block = sm;
                    self.m.op_void(op::UNREACHABLE, &[]);
                }
                Term::Return => self.m.op_void(op::RETURN, &[]),
            },
            Some((&y, [])) if matches!(self.f.blocks[x].term, Term::Branch(..)) && !self.st.no_sel.contains(&x) => {
                // the branch's join: a selection construct merging at it
                let Term::Branch(c, t, e) = self.f.blocks[x].term.clone() else { unreachable!() };
                let c = self.id(c);
                let merge = self.m.id();
                self.ctx.push(Ctx::Sel(y));
                self.cons.push(Construct { node: y, head: x, header: 0, merge, cont: 0, passes: false, wanted: 0, counter: None, arrivals: Vec::new() });
                let header = self.block;
                let arm = |e: &mut Self, to: usize| -> Id { if to == y { merge } else { e.m.id() } };
                let (lt, le) = (arm(self, t), arm(self, e));
                self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
                for (to, l) in [(t, lt), (e, le)] {
                    if to == y {
                        // straight to the join
                        let vals = self.edge_vals(x, y);
                        self.fwd.entry(y).or_default().push((vals, header));
                        self.merge_preds.entry(merge).or_default().push(header);
                    } else {
                        self.m.label(l);
                        self.block = l;
                        self.jump(x, to);
                    }
                }
                self.close();
                self.tree(y);
            }
            Some((&y, rest)) => {
                let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
                self.branch_to(h);
                self.m.label(h);
                self.block = h;
                self.m.op_void(op::LOOP_MERGE, &[merge, cont, 0]);
                let body = self.m.id();
                self.branch_to(body);
                self.m.label(body);
                self.block = body;
                self.ctx.push(Ctx::Block(y));
                self.cons.push(Construct { node: y, head: usize::MAX, header: h, merge, cont, passes: false, wanted: 0, counter: None, arrivals: Vec::new() });
                self.within(x, rest);
                self.close();
                self.tree(y);
            }
        }
    }

    /// Branches to the merge of construct `k`, noting the edge.
    fn to_merge(&mut self, k: usize) {
        let m = self.cons[k].merge;
        let blk = self.block;
        self.merge_preds.entry(m).or_default().push(blk);
        self.branch_to(m);
    }

    /// Phis at a merge: per predecessor, the value it brings or undef.
    fn phis_at(&mut self, tys: &[Id], preds: &[Id], given: &dyn Fn(Id, usize) -> Option<Id>) -> Vec<Id> {
        let mut out = Vec::new();
        for (k, &t) in tys.iter().enumerate() {
            let mut ops = Vec::new();
            for &p in preds {
                let v = match given(p, k) {
                    Some(v) => v,
                    None => self.m.undef(t),
                };
                ops.push(v);
                ops.push(p);
            }
            out.push(self.m.op(op::PHI, t, &ops));
        }
        out
    }

    /// Ends the innermost construct: writes its continue target and opens
    /// its merge block, where a jump to an outer construct passes on.
    fn close(&mut self) {
        let c = self.cons.pop().unwrap();
        let kind = self.ctx.pop().unwrap();
        let (u, b) = (self.t.u32_, self.t.bool_);
        let h = c.header;
        if matches!(kind, Ctx::Sel(_)) {
            // no continue target
        } else {
        // continue target
        self.m.label(c.cont);
        self.block = c.cont;
        match c.counter {
            Some((count, next)) => {
                // the loop's back edges meet here
                let x = c.node;
                let cids = self.cphi.remove(&x).unwrap_or_default();
                let inc = self.back.remove(&x).unwrap_or_default();
                let tys = self.phi_tys(x);
                for (k, &t) in tys.iter().enumerate() {
                    let mut ops = Vec::new();
                    for (vals, blk) in &inc {
                        ops.push(vals[k]);
                        ops.push(*blk);
                    }
                    if ops.is_empty() {
                        let u = self.m.undef(t);
                        ops.push(u);
                        ops.push(h);
                    }
                    self.m.op_to(op::PHI, t, cids[k], &ops);
                }
                let one = self.c32(1);
                self.m.op_to(op::I_ADD, u, next, &[count, one]);
                let lim = self.c32(self.st.loop_limit);
                let over = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[next, lim]);
                // past the limit: leave through the merge, then return
                self.merge_preds.entry(c.merge).or_default().push(c.cont);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[over, c.merge, h]);
            }
            None => self.branch_to(h),
        }
        }
        self.m.label(c.merge);
        self.block = c.merge;
        let preds = self.merge_preds.remove(&c.merge).unwrap_or_default();
        // the block placed at this merge: a Block's node, or a loop's exit
        let at_merge = match kind {
            Ctx::Block(y) | Ctx::Sel(y) => Some(y),
            Ctx::Loop(x) => self.g.as_ref().unwrap().exit[x],
        };
        if let Some(y) = at_merge
            && !self.f.blocks[y].phis.is_empty()
        {
            let inc = self.fwd.remove(&y).unwrap_or_default();
            let tys = self.phi_tys(y);
            let ids = self.phis_at(&tys, &preds, &|p, k| inc.iter().find(|e| e.1 == p).map(|e| e.0[k]));
            self.pending.insert(y, ids);
        }
        // the ladder: which jump passes, with what values
        let mut code_id = 0;
        let mut ladder: Vec<(usize, Vec<Id>)> = Vec::new();
        let mut possible: Vec<u32> = c.arrivals.iter().flat_map(|a| a.possible.iter().copied()).collect();
        possible.sort_unstable();
        possible.dedup();
        if !c.arrivals.is_empty() {
            let zero = self.c32(0);
            let arr = &c.arrivals;
            let codes = self.phis_at(&[u], &preds, &|p, _| Some(arr.iter().find(|a| a.from == p).map_or(zero, |a| a.code)));
            code_id = codes[0];
            let mut targets: Vec<usize> = arr.iter().flat_map(|a| a.vals.iter().map(|v| v.0)).collect();
            targets.sort_unstable();
            targets.dedup();
            for t in targets {
                let tys = self.phi_tys(t);
                if tys.is_empty() {
                    ladder.push((t, Vec::new()));
                    continue;
                }
                let ids = self.phis_at(&tys, &preds, &|p, k| {
                    arr.iter().find(|a| a.from == p).and_then(|a| a.vals.iter().find(|v| v.0 == t)).map(|v| v.1[k])
                });
                ladder.push((t, ids));
            }
        }
        if c.counter.is_some() {
            // the iteration limit: arrived from the continue target
            let (t, f) = (self.m.c_bool(true), self.m.c_bool(false));
            let cont = c.cont;
            let over = self.phis_at(&[b], &preds, &|p, _| Some(if p == cont { t } else { f }))[0];
            let (lr, lc) = (self.m.id(), self.m.id());
            self.m.op_void(op::SELECTION_MERGE, &[lc, 0]);
            self.m.op_void(op::BRANCH_CONDITIONAL, &[over, lr, lc]);
            self.m.label(lr);
            self.m.op_void(op::RETURN, &[]);
            self.m.label(lc);
            self.block = lc;
        }
        let i = self.ctx.len();
        if i > 0 && code_id != 0 {
            let outer = i - 1;
            let z = self.c32(0);
            let outer_kind = self.ctx[outer];
            let codes: Vec<(u32, bool)> = match outer_kind {
                Ctx::Loop(_) => vec![(code(outer, true), true), (code(outer, false), false)],
                Ctx::Block(_) | Ctx::Sel(_) => vec![(code(outer, false), false)],
            };
            let of = |l: &Vec<(usize, Vec<Id>)>, t: usize| l.iter().find(|v| v.0 == t).map(|v| v.1.clone()).unwrap_or_default();
            let mut rest = possible.clone();
            for (k, cont) in codes {
                if !possible.contains(&k) {
                    continue;
                }
                rest.retain(|&x| x != k);
                let kk = self.c32(k);
                let is = self.m.op(op::I_EQUAL, b, &[code_id, kk]);
                let (lt, lf) = (self.m.id(), self.m.id());
                self.m.op_void(op::SELECTION_MERGE, &[lf, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[is, lt, lf]);
                self.m.label(lt);
                self.block = lt;
                let node = self.cons[outer].node;
                if cont {
                    let vals = of(&ladder, node);
                    self.back.entry(node).or_default().push((vals, lt));
                    let t = self.cons[outer].cont;
                    self.branch_to(t);
                } else {
                    let exit = match outer_kind {
                        Ctx::Block(y) | Ctx::Sel(y) => Some(y),
                        Ctx::Loop(x) => self.g.as_ref().unwrap().exit[x],
                    };
                    if let Some(y) = exit {
                        let vals = of(&ladder, y);
                        self.fwd.entry(y).or_default().push((vals, lt));
                    }
                    self.to_merge(outer);
                }
                self.m.label(lf);
                self.block = lf;
            }
            if !rest.is_empty() {
                // further out: pass the code and the values on
                let nz = self.m.op(op::I_NOT_EQUAL, b, &[code_id, z]);
                let (lt, lf) = (self.m.id(), self.m.id());
                self.m.op_void(op::SELECTION_MERGE, &[lf, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[nz, lt, lf]);
                self.m.label(lt);
                self.block = lt;
                self.cons[outer].arrivals.push(Arrival { from: lt, code: code_id, possible: rest, vals: ladder.clone() });
                self.to_merge(outer);
                self.m.label(lf);
                self.block = lf;
            }
        }
    }

    fn jump(&mut self, from: usize, to: usize) {
        let g = self.g.as_ref().unwrap();
        let back = g.rpo[to] <= g.rpo[from];
        let merge = g.merge[to];
        let vals = self.edge_vals(from, to);
        let exit_of = (!back && merge)
            .then(|| self.ctx.iter().rposition(|c| matches!(*c, Ctx::Loop(h) if g.exit[h] == Some(to))))
            .flatten();
        let block_at = (!back && merge)
            .then(|| self.ctx.iter().rposition(|c| *c == Ctx::Block(to) || *c == Ctx::Sel(to)))
            .flatten();
        if back || merge {
            // the innermost of the construct ending at `to`: a Block, or a Loop it exits
            let i = if back {
                self.ctx.iter().rposition(|c| *c == Ctx::Loop(to)).expect("a back edge inside its loop")
            } else {
                match (exit_of, block_at) {
                    (Some(a), Some(b)) => a.max(b),
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => panic!("a merge after its block"),
                }
            };
            let inner = self.ctx.len() - 1;
            // direct: the innermost construct, or a loop or block construct
            // with only selections inside it (break and continue of the
            // innermost loop are allowed from selections)
            let direct = i == inner
                || (!matches!(self.ctx[i], Ctx::Sel(_)) && self.ctx[i + 1..].iter().all(|c| matches!(c, Ctx::Sel(_))));
            // leaving a selection with no merge of its own (the arms of a
            // two-way branch) for a selection's merge is not allowed either
            let into = if direct { i } else { inner };
            if matches!(self.ctx[into], Ctx::Sel(_)) && self.base.iter().any(|&m| m > into) {
                self.bad_sel.push(self.cons[into].head);
            }
            if direct {
                let blk = self.block;
                let list = if back { self.back.entry(to).or_default() } else { self.fwd.entry(to).or_default() };
                list.push((vals, blk));
                if back {
                    let t = self.cons[i].cont;
                    self.branch_to(t);
                } else {
                    self.to_merge(i);
                }
                return;
            }
            // out of several constructs: the code and the values ride a
            // ladder of phis out to the target. A selection cannot take a
            // dispatched branch out of a nested one: such a selection's
            // header becomes a block construct in the next attempt.
            for k in i..inner {
                if matches!(self.ctx[k], Ctx::Sel(_)) {
                    self.bad_sel.push(self.cons[k].head);
                }
            }
                for k in i + 1..=inner {
                self.cons[k].passes = true;
            }
            self.cons[i].wanted |= 1 << u8::from(back);
            let k = code(i, back);
            let code = self.c32(k);
            let blk = self.block;
            self.cons[inner].arrivals.push(Arrival { from: blk, code, possible: vec![k], vals: vec![(to, vals)] });
            self.to_merge(inner);
        } else {
            self.pending.insert(to, vals);
            self.tree(to);
        }
    }

    /// A function variable (declared at the entry later).
    fn entry_var(&mut self, t: Ty) -> Id {
        let tid = self.ty(t);
        let p = self.m.t_ptr(en::SC_FUNCTION, tid);
        let id = self.m.id();
        self.entry_vars.extend_from_slice(&[p, id]);
        id
    }

    pub fn run(mut self) -> Result<Emitted, &'static str> {
        self.g = Some(Graph::new(self.f)?);
        let fid = self.m.id();
        let void = self.t.void;
        // pick arrays
        let mut picks = Vec::new();
        fn find(list: &[Item], f: &Func, out: &mut Vec<V>) {
            for it in list {
                match it {
                    Item::N(v) if f.node(*v).op == Op::Pick => out.push(*v),
                    Item::Guard { body, .. } => find(body, f, out),
                    _ => {}
                }
            }
        }
        for b in &self.f.blocks {
            find(&b.body, self.f, &mut picks);
        }
        for v in picks {
            let n = (self.f.node(v).args.len() - 1) as u32;
            let u = self.t.u32_;
            let arr = self.m.t_array(u, n.max(1));
            let p = self.m.t_ptr(en::SC_FUNCTION, arr);
            let id = self.m.id();
            self.entry_vars.extend_from_slice(&[p, id]);
            self.picks.insert(v, id);
        }
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
                if self.outputs.contains_key(&OUT_DEPTH) {
                    self.m.execution_mode(fid, en::EM_DEPTH_REPLACING, &[]);
                }
                if self.outputs.contains_key(&OUT_STENCIL) {
                    self.m.execution_mode(fid, en::EM_STENCIL_REF_REPLACING, &[]);
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
                self.m.execution_mode(fid, en::EM_OUTPUT_VERTICES, &[self.st.mesh.0.max(1)]);
                self.m.execution_mode(fid, en::EM_OUTPUT_PRIMITIVES_EXT, &[self.st.mesh.1.max(1)]);
                self.m.execution_mode(fid, en::EM_OUTPUT_TRIANGLES_EXT, &[]);
            }
            _ => {}
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
