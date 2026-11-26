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

use std::collections::HashMap;

use super::lift::{Interface, OUT_DEPTH, OUT_POSITION, OUT_SAMPLE_MASK, OUT_STENCIL};
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
}

pub struct Emitted {
    pub words: Vec<u32>,
    /// (location, word index of the Location literal) of pixel inputs.
    pub input_locations: Vec<(u8, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctx {
    Loop(usize),
    Block(usize),
}

struct Construct {
    node: usize,
    header: Id,
    merge: Id,
    cont: Id,
    passes: bool,
    /// Jumps from inner constructs that arrive here: bit 0 exit, bit 1 continue.
    wanted: u8,
    /// Loop constructs: the iteration counter variable.
    counter: Option<Id>,
}

struct Graph {
    rpo: Vec<usize>,
    header: Vec<bool>,
    merge: Vec<bool>,
    children: Vec<Vec<usize>>,
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
        let merge = forward.iter().map(|&k| k >= 2).collect();
        let mut children = vec![Vec::new(); n];
        for &b in order.iter().skip(1) {
            children[idom[b]].push(b);
        }
        Ok(Graph { rpo, header, merge, children })
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
    /// Inside a flat guard: its condition (effects go to the sink).
    pred: Option<Id>,
    sink: Id,
    /// Values a forward edge / back edge gives a block's phis, and the
    /// SPIR-V block it comes from.
    fwd: HashMap<usize, Vec<(Vec<Id>, Id)>>,
    back: HashMap<usize, Vec<(Vec<Id>, Id)>>,
    /// A block's phi values when its code begins.
    pending: HashMap<usize, Vec<Id>>,
    /// Phi variables of blocks reached by a jump past several constructs.
    pvars: HashMap<usize, Vec<Id>>,
    /// Phis of each loop's continue target, made at its header.
    cphi: HashMap<usize, Vec<Id>>,
    /// Blocks that pass a jump on into a construct's merge (no value).
    pass_in: HashMap<Id, Vec<Id>>,
    jmp: Id,
    images: HashMap<(u8, bool, bool, bool, bool, bool), Id>,
    samplers: Id,
    static_samplers: Id,
    helpers: HashMap<Helper, Id>,
    picks: HashMap<V, Id>,
    g: Option<Graph>,
    ctx: Vec<Ctx>,
    cons: Vec<Construct>,
    uses_jmp: bool,
    entry_vars: Vec<u32>,
    input_locations: Vec<(u8, usize)>,
    frag: bool,
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
            pred: None,
            sink: 0,
            fwd: HashMap::new(),
            back: HashMap::new(),
            pending: HashMap::new(),
            pvars: HashMap::new(),
            cphi: HashMap::new(),
            pass_in: HashMap::new(),
            jmp: 0,
            images: HashMap::new(),
            samplers: 0,
            static_samplers: 0,
            helpers: HashMap::new(),
            picks: HashMap::new(),
            g: None,
            ctx: Vec::new(),
            cons: Vec::new(),
            uses_jmp: false,
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

    /// The memory map's struct type (`docs/vulkan-abi.md` section 3), read
    /// only, behind a PhysicalStorageBuffer pointer.
    fn map_type(&mut self) -> Id {
        if self.map == 0 {
            let (u, q) = (self.t.u32_, self.t.u64_);
            let range = self.m.t_struct(&[q, q, q, q], 3);
            for (k, o) in [0u32, 8, 16, 24].iter().enumerate() {
                self.m.member_decorate(range, k as u32, en::DEC_OFFSET, &[*o]);
                self.m.member_decorate(range, k as u32, 24, &[]); // NonWritable
            }
            let arr = self.m.t_array(range, 64);
            self.m.decorate(arr, en::DEC_ARRAY_STRIDE, &[32]);
            let st = self.m.t_struct(&[q, q, u, arr], 4);
            for (k, o) in [0u32, 8, 16, 32].iter().enumerate() {
                self.m.member_decorate(st, k as u32, en::DEC_OFFSET, &[*o]);
                self.m.member_decorate(st, k as u32, 24, &[]);
            }
            self.m.decorate(st, en::DEC_BLOCK, &[]);
            self.map = st;
        }
        self.map
    }

    /// A pointer to the memory map (push dwords 0-1), made here.
    fn map_ptr(&mut self) -> Id {
        let st = self.map_type();
        let var = self.push_var();
        let u = self.t.u32_;
        let p = self.m.t_ptr(en::SC_PUSH_CONSTANT, u);
        let z = self.c32(0);
        let one = self.c32(1);
        let a = self.m.op(op::ACCESS_CHAIN, p, &[var, z, z]);
        let b = self.m.op(op::ACCESS_CHAIN, p, &[var, z, one]);
        let lo = self.m.op(op::LOAD, u, &[a]);
        let hi = self.m.op(op::LOAD, u, &[b]);
        let v2 = self.m.t_vec(u, 2);
        let pair = self.m.op(op::COMPOSITE_CONSTRUCT, v2, &[lo, hi]);
        let q = self.t.u64_;
        let addr = self.m.op(op::BITCAST, q, &[pair]);
        let pt = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, st);
        self.m.op(op::CONVERT_U_TO_PTR, pt, &[addr])
    }

    /// Loads a member of the memory map (u64 or u32 by `ty`); `path` ids.
    fn map_load_at(&mut self, map: Id, ty: Id, path: &[Id]) -> Id {
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, ty);
        let mut ops = vec![map];
        ops.extend_from_slice(path);
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &ops);
        let align = if ty == self.t.u64_ { 8 } else { 4 };
        self.m.op(op::LOAD, ty, &[ptr, en::MEM_ALIGNED, align])
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
        };
        let key = match b {
            Builtin::FragCoord(_) => Builtin::FragCoord(0),
            Builtin::WorkgroupId(_) => Builtin::WorkgroupId(0),
            Builtin::LocalInvocationId(_) => Builtin::LocalInvocationId(0),
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

    fn input(&mut self, loc: u8, comp: u8, interp: Interp) -> Id {
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

    fn output(&mut self, slot: u16) -> (Id, Id) {
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
        // cube images cannot be fetched or stored: as 2D arrays then
        let (dim, arrayed) = if c.dim == 3 && (storage || !c.storage && c.dynamic) { (1, true) } else { (c.dim, c.arrayed) };
        let key = (dim, arrayed, c.ms, uint, storage, depth);
        let sty = if uint { self.t.u32_ } else { self.t.f32_ };
        let ity = self.m.t_image_d(sty, u32::from(dim), depth, arrayed, c.ms, if storage { 2 } else { 1 });
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
        self.m.capability(en::CAP_RUNTIME_DESCRIPTOR_ARRAY);
        let arr = self.m.t_runtime_array(ity);
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

    fn load_sampler(&mut self, s: Sampler, sw: Option<Id>) -> Id {
        let sty = self.m.t_sampler();
        let p = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, sty);
        match s {
            Sampler::Static(n) => {
                if self.static_samplers == 0 {
                    let count = self.iface.samplers.len().max(1) as u32;
                    let arr = self.m.t_array(sty, count);
                    let pa = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, arr);
                    self.static_samplers = self.m.variable(pa, en::SC_UNIFORM_CONSTANT);
                    self.m.decorate(self.static_samplers, en::DEC_DESCRIPTOR_SET, &[1]);
                    let b = u32::from(self.st.stage == HwStage::Ps);
                    self.m.decorate(self.static_samplers, en::DEC_BINDING, &[b]);
                    self.interface.push(self.static_samplers);
                }
                let k = self.c32(n);
                let ptr = self.m.op(op::ACCESS_CHAIN, p, &[self.static_samplers, k]);
                self.m.op(op::LOAD, sty, &[ptr])
            }
            _ => {
                if self.samplers == 0 {
                    let arr = self.m.t_runtime_array(sty);
                    let pa = self.m.t_ptr(en::SC_UNIFORM_CONSTANT, arr);
                    self.samplers = self.m.variable(pa, en::SC_UNIFORM_CONSTANT);
                    self.m.decorate(self.samplers, en::DEC_DESCRIPTOR_SET, &[0]);
                    self.m.decorate(self.samplers, en::DEC_BINDING, &[2]);
                    self.interface.push(self.samplers);
                    self.m.capability(en::CAP_RUNTIME_DESCRIPTOR_ARRAY);
                }
                let u = self.t.u32_;
                let k = self.c32(0xFFF);
                let idx = self.m.op(op::BITWISE_AND, u, &[sw.unwrap(), k]);
                let ptr = self.m.op(op::ACCESS_CHAIN, p, &[self.samplers, idx]);
                self.m.op(op::LOAD, sty, &[ptr])
            }
        }
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
        let helpers: Vec<(Helper, Id)> = self.helpers.iter().map(|(h, i)| (*h, *i)).collect();
        if helpers.is_empty() {
            return;
        }
        // Specialization constant 0: the map has one range (the default).
        let b = self.t.bool_;
        let one_range = self.m.spec_true(b);
        self.m.decorate(one_range, en::DEC_SPEC_ID, &[0]);
        for (h, fid) in helpers {
            let (u, q) = (self.t.u32_, self.t.u64_);
            let v3 = self.m.t_vec(u, 3);
            let range = h == Helper::TranslateRange;
            let (ret, params): (Id, Vec<Id>) = if range { (v3, vec![q, u]) } else { (q, vec![q]) };
            let fty = self.m.global_fn_type(ret, &params);
            self.m.op_to(op::FUNCTION, ret, fid, &[0, fty]);
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
            let out = if range {
                let lim = self.m.op(op::PHI, u, &[r0.1, lf, r1.1, ls]);
                let v2 = self.m.t_vec(u, 2);
                let pair = self.m.op(op::BITCAST, v2, &[res]);
                let lo = self.m.op(op::COMPOSITE_EXTRACT, u, &[pair, 0]);
                let hi = self.m.op(op::COMPOSITE_EXTRACT, u, &[pair, 1]);
                self.m.op(op::COMPOSITE_CONSTRUCT, v3, &[lo, hi, lim])
            } else {
                res
            };
            self.m.op_void(254, &[out]); // OpReturnValue
            self.m.op_void(op::FUNCTION_END, &[]);
        }
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
            Op::Input { loc, comp, interp } => self.input(loc, comp, interp),
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
                let (var, elem) = self.output(slot);
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
                    let (im, _) = e.load_image(class, int, storage, false, tw);
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
                let (im, _) = self.load_image(class, false, false, false, tw);
                self.m.op(op::IMAGE_QUERY_LEVELS, u, &[im])
            }
            ImageKind::Atomic(o) => {
                let (_, ptr) = self.load_image(class, true, true, false, tw);
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
        let mut merges: Vec<usize> = g.children[x].iter().copied().filter(|&c| g.merge[c]).collect();
        merges.sort_by_key(|&c| std::cmp::Reverse(g.rpo[c]));
        let header = g.header[x];
        let entry = self.pending.remove(&x).unwrap_or_default();
        let phis = self.f.blocks[x].phis.clone();
        if header {
            let counter = self.entry_var(Ty::U32);
            let z = self.c32(0);
            self.m.op_void(op::STORE, &[counter, z]);
            let (h, merge, cont) = (self.m.id(), self.m.id(), self.m.id());
            let pre = self.block;
            self.branch_to(h);
            self.m.label(h);
            self.block = h;
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
            self.cons.push(Construct { node: x, header: h, merge, cont, passes: false, wanted: 0, counter: Some(counter) });
            let b = self.f.blocks[x].body.clone();
            self.items(&b);
            self.within(x, &merges);
            self.close();
            // a loop is left only by jumps to outer constructs
            self.m.op_void(op::UNREACHABLE, &[]);
        } else {
            for (k, &ph) in phis.iter().enumerate() {
                self.ids[ph as usize] = entry[k];
            }
            let b = self.f.blocks[x].body.clone();
            self.items(&b);
            self.within(x, &merges);
        }
    }

    fn within(&mut self, x: usize, merges: &[usize]) {
        match merges.split_first() {
            None => match self.f.blocks[x].term.clone() {
                Term::Jump(t) => self.jump(x, t),
                Term::Branch(c, t, e) => {
                    let c = self.id(c);
                    let (lt, le, sm) = (self.m.id(), self.m.id(), self.m.id());
                    self.m.op_void(op::SELECTION_MERGE, &[sm, 0]);
                    self.m.op_void(op::BRANCH_CONDITIONAL, &[c, lt, le]);
                    self.m.label(lt);
                    self.block = lt;
                    self.jump(x, t);
                    self.m.label(le);
                    self.block = le;
                    self.jump(x, e);
                    self.m.label(sm);
                    self.block = sm;
                    self.m.op_void(op::UNREACHABLE, &[]);
                }
                Term::Return => self.m.op_void(op::RETURN, &[]),
            },
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
                self.cons.push(Construct { node: y, header: h, merge, cont, passes: false, wanted: 0, counter: None });
                self.within(x, rest);
                self.close();
                self.tree(y);
            }
        }
    }

    /// Loads the phi variables of `x` here and records them as an incoming
    /// edge (a jump that passed several constructs arrives).
    fn arrive(&mut self, x: usize, back: bool) {
        if let Some(vars) = self.pvars.get(&x).cloned() {
            let tys = self.phi_tys(x);
            let vals: Vec<Id> = vars.iter().zip(&tys).map(|(&v, &t)| self.m.op(op::LOAD, t, &[v])).collect();
            let blk = self.block;
            let list = if back { self.back.entry(x).or_default() } else { self.fwd.entry(x).or_default() };
            list.push((vals, blk));
        }
    }

    /// Phis of a merge from what reached it.
    fn merge_phis(&mut self, x: usize, label: Id, incoming: Vec<(Vec<Id>, Id)>) -> Vec<Id> {
        let tys = self.phi_tys(x);
        let passed = self.pass_in.remove(&label).unwrap_or_default();
        let mut out = Vec::new();
        for (k, &t) in tys.iter().enumerate() {
            let mut ops = Vec::new();
            for (vals, blk) in &incoming {
                ops.push(vals[k]);
                ops.push(*blk);
            }
            for &blk in &passed {
                let u = self.m.undef(t);
                ops.push(u);
                ops.push(blk);
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
        // continue target
        self.m.label(c.cont);
        self.block = c.cont;
        let h = c.header;
        match c.counter {
            Some(counter) => {
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
                let u = self.t.u32_;
                let b = self.t.bool_;
                let n = self.m.op(op::LOAD, u, &[counter]);
                let one = self.c32(1);
                let n1 = self.m.op(op::I_ADD, u, &[n, one]);
                self.m.op_void(op::STORE, &[counter, n1]);
                let lim = self.c32(self.st.loop_limit);
                let over = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[n1, lim]);
                // past the limit: leave through the merge with jmp = return
                let jmp = self.jmp_var();
                let ret = self.c32(u32::MAX);
                let cur = self.m.op(op::LOAD, u, &[jmp]);
                let nj = self.m.op(op::SELECT, u, &[over, ret, cur]);
                self.m.op_void(op::STORE, &[jmp, nj]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[over, c.merge, h]);
            }
            None => self.branch_to(h),
        }
        self.m.label(c.merge);
        self.block = c.merge;
        if let Ctx::Block(y) = kind
            && !self.f.blocks[y].phis.is_empty()
        {
            let inc = self.fwd.remove(&y).unwrap_or_default();
            let ids = self.merge_phis(y, c.merge, inc);
            self.pending.insert(y, ids);
        } else {
            self.pass_in.remove(&c.merge);
        }
        let i = self.ctx.len();
        if c.counter.is_some() {
            // the iteration limit
            let (u, b) = (self.t.u32_, self.t.bool_);
            let jmp = self.jmp_var();
            let cur = self.m.op(op::LOAD, u, &[jmp]);
            let ret = self.c32(u32::MAX);
            let is = self.m.op(op::I_EQUAL, b, &[cur, ret]);
            let (lr, lc) = (self.m.id(), self.m.id());
            self.m.op_void(op::SELECTION_MERGE, &[lc, 0]);
            self.m.op_void(op::BRANCH_CONDITIONAL, &[is, lr, lc]);
            self.m.label(lr);
            self.m.op_void(op::RETURN, &[]);
            self.m.label(lc);
            self.block = lc;
        }
        if c.passes && i > 0 {
            let outer = i - 1;
            let (u, b) = (self.t.u32_, self.t.bool_);
            let jmp = self.jmp_var();
            let cur = self.m.op(op::LOAD, u, &[jmp]);
            let z = self.c32(0);
            let outer_kind = self.ctx[outer];
            let codes: Vec<(u32, bool)> = match outer_kind {
                Ctx::Loop(_) => vec![(code(outer, true), true), (code(outer, false), false)],
                Ctx::Block(_) => vec![(code(outer, false), false)],
            };
            for (k, cont) in codes {
                if self.cons[outer].wanted & (1 << u8::from(cont)) == 0 {
                    continue;
                }
                let kk = self.c32(k);
                let is = self.m.op(op::I_EQUAL, b, &[cur, kk]);
                let (lt, lf) = (self.m.id(), self.m.id());
                self.m.op_void(op::SELECTION_MERGE, &[lf, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[is, lt, lf]);
                self.m.label(lt);
                self.block = lt;
                self.m.op_void(op::STORE, &[jmp, z]);
                let node = self.cons[outer].node;
                let target = if cont {
                    self.arrive(node, true);
                    self.cons[outer].cont
                } else {
                    if matches!(outer_kind, Ctx::Block(_)) {
                        self.arrive(node, false);
                    } else {
                        let m = self.cons[outer].merge;
                        let blk = self.block;
                        self.pass_in.entry(m).or_default().push(blk);
                    }
                    self.cons[outer].merge
                };
                self.branch_to(target);
                self.m.label(lf);
                self.block = lf;
            }
            if self.cons[outer].passes {
                let nz = self.m.op(op::I_NOT_EQUAL, b, &[cur, z]);
                let (lt, lf) = (self.m.id(), self.m.id());
                self.m.op_void(op::SELECTION_MERGE, &[lf, 0]);
                self.m.op_void(op::BRANCH_CONDITIONAL, &[nz, lt, lf]);
                self.m.label(lt);
                let m = self.cons[outer].merge;
                self.pass_in.entry(m).or_default().push(lt);
                self.branch_to(m);
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
        if back || merge {
            let i = if back {
                self.ctx.iter().rposition(|c| *c == Ctx::Loop(to)).expect("a back edge inside its loop")
            } else {
                self.ctx.iter().rposition(|c| *c == Ctx::Block(to)).expect("a merge after its block")
            };
            let inner = self.ctx.len() - 1;
            if i == inner {
                let blk = self.block;
                let list = if back { self.back.entry(to).or_default() } else { self.fwd.entry(to).or_default() };
                list.push((vals, blk));
            } else if !vals.is_empty() {
                // stored here, loaded where the jump arrives
                let vars = match self.pvars.get(&to) {
                    Some(v) => v.clone(),
                    None => {
                        let tys: Vec<Ty> = self.f.blocks[to].phis.iter().map(|&p| self.f.ty(p)).collect();
                        let v: Vec<Id> = tys.into_iter().map(|t| self.entry_var(t)).collect();
                        self.pvars.insert(to, v.clone());
                        v
                    }
                };
                for (v, x) in vars.into_iter().zip(vals) {
                    self.m.op_void(op::STORE, &[v, x]);
                }
            }
            self.leave(i, back);
        } else {
            self.pending.insert(to, vals);
            self.tree(to);
        }
    }

    fn leave(&mut self, i: usize, cont: bool) {
        let inner = self.ctx.len() - 1;
        if i == inner {
            let t = if cont { self.cons[i].cont } else { self.cons[i].merge };
            self.branch_to(t);
            return;
        }
        for k in i + 1..=inner {
            self.cons[k].passes = true;
        }
        self.cons[i].wanted |= 1 << u8::from(cont);
        self.uses_jmp = true;
        let jmp = self.jmp_var();
        let k = self.c32(code(i, cont));
        self.m.op_void(op::STORE, &[jmp, k]);
        let m = self.cons[inner].merge;
        let blk = self.block;
        self.pass_in.entry(m).or_default().push(blk);
        self.branch_to(m);
    }

    fn jmp_var(&mut self) -> Id {
        if self.jmp == 0 {
            self.jmp = self.entry_var(Ty::U32);
        }
        self.jmp
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
            _ => {}
        }
        let (words, at) = self.m.finish();
        let input_locations = self.input_locations.iter().map(|&(l, w)| (l, at + w)).collect();
        Ok(Emitted { words, input_locations })
    }

    fn tree_root(&mut self) {
        self.tree(0);
    }
}
