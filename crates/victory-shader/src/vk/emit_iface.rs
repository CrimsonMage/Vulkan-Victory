//! Interface variables, descriptors, the memory map and the helper functions.

use super::*;

impl Emit<'_> {
    pub(super) fn push_var(&mut self) -> Id {
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
    pub(super) fn map_type(&mut self) -> Id {
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
    pub(super) fn no_search() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("VK_SEARCH").is_ok_and(|v| v == "0"))
    }

    /// The memory map: a uniform buffer, set 3 binding 0 (read with uniform
    /// loads; NVIDIA compiled a pointer-loaded map ~35x slower).
    pub(super) fn map_ptr(&mut self) -> Id {
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
    pub(super) fn map_load_at(&mut self, map: Id, ty: Id, path: &[Id]) -> Id {
        let p = self.m.t_ptr(en::SC_UNIFORM, ty);
        let mut ops = vec![map];
        ops.extend_from_slice(path);
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &ops);
        self.m.op(op::LOAD, ty, &[ptr])
    }

    pub(super) fn map_load(&mut self, ty: Id, path: &[u32]) -> Id {
        let map = self.map_ptr();
        let ids: Vec<Id> = path.iter().map(|&k| self.c32(k)).collect();
        self.map_load_at(map, ty, &ids)
    }

    pub(super) fn builtin(&mut self, b: Builtin) -> Id {
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

    pub(super) fn input(&mut self, row: u8, chan: u8, interp: Interp, ty: Ty) -> Id {
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
    pub(super) fn mesh_output(&mut self, slot: u16) -> (Id, Id) {
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
    pub(super) fn mesh_prologue(&mut self) {
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
    pub(super) fn per_vertex_input(&mut self, row: u8, chan: u8, vertex: u8) -> Id {
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
    pub(super) fn vertex_input(&mut self, row: u8, chan: u8, ty: Ty) -> Id {
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

    pub(super) fn output(&mut self, slot: u16) -> (Id, Id) {
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

    pub(super) fn lds_var(&mut self) -> Id {
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
    pub(super) fn image_var(&mut self, c: ImageClass, uint: bool, storage: bool, depth: bool) -> (Id, Id) {
        self.image_var_f(c, uint, storage, depth, 0)
    }

    /// The same with an image format (R32ui = 33 for atomics).
    pub(super) fn image_var_f(&mut self, c: ImageClass, uint: bool, storage: bool, depth: bool, format: u32) -> (Id, Id) {
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

    pub(super) fn load_image(&mut self, c: ImageClass, uint: bool, storage: bool, depth: bool, tw: Id) -> (Id, Id) {
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
    pub(super) fn load_sampler(&mut self, s: Sampler, sw: Option<Id>) -> Id {
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

    pub(super) fn helper(&mut self, h: Helper) -> Id {
        if let Some(&id) = self.helpers.get(&h) {
            return id;
        }
        let id = self.m.id();
        self.helpers.insert(h, id);
        id
    }

    /// Writes the helper functions' bodies (after main).
    pub(super) fn write_helpers(&mut self) {
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
    pub(super) fn write_typed(&mut self, fid: Id) {
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
}
