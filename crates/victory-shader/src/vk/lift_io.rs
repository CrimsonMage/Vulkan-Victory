//! Stage interface: inputs, exports, interpolation.

use super::*;

impl Lift<'_> {
    pub(super) fn input(&mut self, reg: Reg, v: Value, offset: usize) -> Res<R> {
        if let Some(&(_, input)) = self.io.inputs.iter().find(|i| i.0 == v) {
            return Ok(R::D(self.system_input(input, offset)?));
        }
        Ok(self.register_input(reg))
    }

    /// An input the hardware loads (io.rs names it).
    fn system_input(&mut self, input: Input, offset: usize) -> Res<V> {
        let u = |l: &mut Self, b| l.f.global(Op::Builtin(b), Ty::U32);
        Ok(match input {
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
            // A vertex program has no primitive id on the PC: a third of
            // the vertex index (exact for a triangle list without indices).
            Input::PrimitiveId if self.native.stage == HwStage::Vs => {
                let (v, three) = (u(self, Builtin::VertexIndex), self.k32(3));
                self.global_bin(op::U_DIV, Ty::U32, v, three)
            }
            Input::PrimitiveId => u(self, Builtin::PrimitiveId),
            Input::ThreadIdInGroup(c) => self.thread_id(c),
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
            Input::Barycentric { kind, index } => self.bary_input(kind, index, offset)?,
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
        })
    }

    /// i and j, the weights of vertices 1 and 2, read as numbers.
    fn bary_input(&mut self, kind: io::Barycentric, index: u8, offset: usize) -> Res<V> {
        let (perspective, centroid, sample) = bary_mode(kind).ok_or(unsupported(offset, "pull-model barycentrics"))?;
        if index > 1 {
            return Err(unsupported(offset, "pull-model barycentrics"));
        }
        let b = Builtin::Bary { perspective, centroid, sample, comp: index + 1 };
        self.iface.per_vertex = true;
        Ok(self.f.global(Op::Builtin(b), Ty::F32))
    }

    /// A register read before any write: EXEC, user data, or undefined.
    fn register_input(&mut self, reg: Reg) -> R {
        let stage = self.native.stage;
        match reg {
            Reg::S(c) if c == ir::EXEC || c == ir::EXEC + 1 => {
                let half = if c == ir::EXEC { Half::Lo } else { Half::Hi };
                // pixels: the live lanes (helpers join at s_wqm)
                let b = if stage == HwStage::Ps {
                    let h = self.f.global(Op::Builtin(Builtin::Helper), Ty::BOOL);
                    self.global_node(Op::Spv(op::LOGICAL_NOT), Ty::BOOL, vec![h])
                } else {
                    self.in_group().unwrap_or_else(|| self.kb(true))
                };
                R::M(b, half)
            }
            Reg::S(c) => R::D(self.user_data(u32::from(c))),
            _ => R::D(self.f.global(Op::Undef, Ty::U32)),
        }
    }

    /// User SGPR `c` (merged stages have 8 system SGPRs first): its push
    /// constant, or 0 past the user data.
    fn user_data(&mut self, c: u32) -> V {
        let stage = self.native.stage;
        let base = if matches!(stage, HwStage::Ms | HwStage::Gs | HwStage::Es | HwStage::Hs | HwStage::Ls) { 8 } else { 0 };
        let users = match &self.native.io {
            StageIo::Cs(cs) => u32::from(cs.user_sgprs()),
            _ => 32,
        };
        // A compute program's spilled user-data table (spill.rs): its
        // entries at index 16 and above arrive in push dwords too. Only a
        // compute program: it has the push constants to itself, while a
        // draw's stages share them (a mesh program's entry 28 would be the
        // pixel program's entry 0: run vk63 faulted on such pointers).
        let spilled = |k: u32| {
            stage == HwStage::Cs
                && k >= 16
                && push_window(stage) + k < 64
                && self.native.layout.iter().any(|e| k >= u32::from(e.user_data) && k < u32::from(e.user_data) + e.sgprs)
        };
        if c >= base && (c - base < users || spilled(c - base)) && c - base < 60 {
            return self.f.global(Op::Push(push_window(stage) + (c - base)), Ty::U32);
        }
        self.f.konst(Ty::U32, 0)
    }

    pub(super) fn export(&mut self, inst: &Inst) -> Res<()> {
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
            Target::Position(n) => return self.export_pos_extra(&e, n, off),
            Target::Param { .. } => return self.export_param(&e, off),
            Target::Null | Target::Unread => return Ok(()),
            Target::Depth => return self.export_depth(&e, off),
            Target::Primitive => OUT_PRIMITIVE,
            Target::PrimitiveParam { .. } => return Err(unsupported(off, "a per-primitive output")),
        };
        self.export_values(&e, slot, off)
    }

    /// `exp pos1+`: the vectors `PA_CL_VS_OUT_CNTL` enables after the
    /// position, in order: the misc vector (point size, edge flag, layer
    /// and viewport index), clip/cull distances 0-3, 4-7. Distance d is
    /// clip when CLIP_DIST_ENA bit d is set, cull when CULL_DIST_ENA bit d
    /// is; each kind fills its array in distance order.
    fn export_pos_extra(&mut self, e: &io::Export, n: u8, off: usize) -> Res<()> {
        let cntl = match &self.native.io {
            StageIo::Vs { clip_cntl, .. } => *clip_cntl,
            StageIo::Ms(m) => m.clip_cntl,
            _ => 0,
        };
        let clip = cntl & 0xFF;
        let cull = cntl >> 8 & 0xFF;
        self.iface.clip_cull = (clip.count_ones() as u8, cull.count_ones() as u8);
        let vectors: Vec<u32> = [21, 22, 23].into_iter().filter(|b| cntl >> b & 1 != 0).collect();
        let Some(&vector) = vectors.get(usize::from(n) - 1) else {
            return Err(unsupported(off, "a position export PA_CL_VS_OUT_CNTL does not enable"));
        };
        for (c, v) in e.values.iter().enumerate() {
            let Some(v) = v else { continue };
            let r = self.value(*v, off)?;
            let x = self.data(r);
            let c = c as u32;
            let outs: Vec<(u16, u32, V)> = if vector == 21 {
                self.misc_outputs(cntl, c, x)
            } else {
                let d = 4 * (vector - 22) + c;
                let below = (1u32 << d) - 1;
                let mut o = Vec::new();
                if clip >> d & 1 != 0 {
                    o.push((OUT_CLIP, (clip & below).count_ones(), x));
                }
                if cull >> d & 1 != 0 {
                    o.push((OUT_CULL, (cull & below).count_ones(), x));
                }
                o
            };
            let index = self.mesh_index();
            for (slot, comp, x) in outs {
                let mut args = vec![x];
                args.extend(index);
                self.emit(Op::Output { slot, comp: comp as u8 }, Ty::VOID, args);
                self.note_output(slot, comp as u8);
            }
        }
        Ok(())
    }

    /// Channel `c` of the misc vector: x point size, y edge flag (no
    /// Vulkan counterpart: dropped), z layer in bits 0-10 and viewport
    /// index in bits 16-19, each when its USE_VTX bit is set.
    fn misc_outputs(&mut self, cntl: u32, c: u32, x: V) -> Vec<(u16, u32, V)> {
        let on = |b: u32| cntl >> b & 1 != 0;
        match c {
            0 if on(16) => vec![(OUT_POINT_SIZE, 0, x)],
            1 if on(17) => {
                self.approx("a vertex edge flag (misc vector y) left out");
                vec![]
            }
            // a mesh program's layer and viewport index are per primitive
            2 if self.native.stage == HwStage::Ms && (on(18) || on(19)) => {
                self.approx("a mesh program's per-vertex layer or viewport index left out");
                vec![]
            }
            2 => {
                let u = self.u32v(x);
                let mut o = Vec::new();
                if on(18) {
                    let (z, k11) = (self.k32(0), self.k32(11));
                    o.push((OUT_LAYER, 0, self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[u, z, k11])));
                }
                if on(19) {
                    let (k16, k4) = (self.k32(16), self.k32(4));
                    o.push((OUT_VIEWPORT, 0, self.spv(op::BIT_FIELD_U_EXTRACT, Ty::U32, &[u, k16, k4])));
                }
                o
            }
            _ => vec![],
        }
    }

    /// Mesh: vertex and primitive n are written by thread n of the group.
    fn mesh_index(&mut self) -> Option<V> {
        (self.native.stage == HwStage::Ms).then(|| self.console_thread())
    }

    /// The channels of an export as outputs of a slot.
    fn export_values(&mut self, e: &io::Export, slot: u16, off: usize) -> Res<()> {
        let index = self.mesh_index();
        for (c, v) in e.values.iter().enumerate() {
            let Some(v) = v else { continue };
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
        Ok(())
    }

    /// Depth, stencil and sample mask (channels 0-2).
    fn export_depth(&mut self, e: &io::Export, off: usize) -> Res<()> {
        self.iface.depth = true;
        for (c, slot) in [OUT_DEPTH, OUT_STENCIL, OUT_SAMPLE_MASK].into_iter().enumerate() {
            let Some(v) = e.values[c] else { continue };
            let r = self.value(v, off)?;
            let x = self.data(r);
            self.emit(Op::Output { slot, comp: 0 }, Ty::VOID, vec![x]);
            self.note_output(slot, 0);
        }
        Ok(())
    }

    /// A parameter: its row (the mesh's unnamed entry has the free row);
    /// 16-bit rows carry two halves per dword, each a column of a row.
    fn export_param(&mut self, e: &io::Export, off: usize) -> Res<()> {
        let Target::Param { usage, row } = e.target else { unreachable!() };
        let slot = if usage == crate::native::Usage::Other(0) && self.native.stage == HwStage::Ms {
            self.unnamed_row.ok_or(unsupported(off, "a mesh parameter without a row"))?
        } else {
            row
        };
        if !self.halves.keys().any(|k| k.0 == slot) {
            return self.export_values(e, u16::from(slot), off);
        }
        let index = self.mesh_index();
        for (c, v) in e.values.iter().enumerate() {
            let Some(v) = v else { continue };
            let r = self.value(*v, off)?;
            let x = self.data(r);
            let x = self.u32v(x);
            for high in [false, true] {
                let Some(&(row, col)) = self.halves.get(&(slot, 2 * c as u8 + u8::from(high))) else { continue };
                let f = self.half_f32(x, high);
                let mut args = vec![f];
                args.extend(index);
                self.emit(Op::Output { slot: u16::from(row), comp: col }, Ty::VOID, args);
                self.note_output(u16::from(row), col);
            }
        }
        Ok(())
    }

    pub(super) fn note_output(&mut self, slot: u16, comp: u8) {
        if let Some(o) = self.iface.outputs.iter_mut().find(|o| o.0 == slot) {
            o.1 |= 1 << comp;
        } else {
            self.iface.outputs.push((slot, 1 << comp));
        }
    }

    pub(super) fn interp(&mut self, inst: &Inst, dst: Operand) -> Res<()> {
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
                let (perspective, centroid, sample) =
                    bary_mode(*barycentric).ok_or(unsupported(off, "pull-model interpolation"))?;
                (*attribute, Interp::Smooth { perspective, centroid, sample })
            }
            io::Interp::Vertex { attribute, vertex: 2 } if !self.per_vertex.contains(&attribute.row) => {
                (*attribute, Interp::Flat)
            }
            _ => return self.interp_per_vertex(inst, reg, interp),
        };
        if self.per_vertex.contains(&attr.row) {
            return self.interp_per_vertex(inst, reg, interp);
        }
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

impl Lift<'_> {
    /// An attribute some site reads per vertex: every read of it weighs its
    /// three vertices (VK_KHR_fragment_shader_barycentric), as the
    /// hardware's P0 + i P10 + j P20.
    pub(super) fn interp_per_vertex(&mut self, inst: &Inst, reg: u8, interp: &io::Interp) -> Res<()> {
        let off = inst.machine.offset;
        let name = inst.name();
        if name.contains("f16") {
            return Err(unsupported(off, "a 16-bit interpolation of vertex values"));
        }
        let attribute = match *interp {
            io::Interp::Interpolated { attribute, .. } | io::Interp::Vertex { attribute, .. } | io::Interp::Weighted { attribute } => attribute,
        };
        if attribute.high {
            return Err(unsupported(off, "a high half of a 32-bit input"));
        }
        let (row, c) = (attribute.row, attribute.channel);
        let p = |l: &mut Self, k: u8| l.f.global(Op::PerVertex { loc: row, comp: c, vertex: k }, Ty::F32);
        let (p0, p1, p2) = (p(self, 0), p(self, 1), p(self, 2));
        let p10 = self.spv(op::F_SUB, Ty::F32, &[p1, p0]);
        let p20 = self.spv(op::F_SUB, Ty::F32, &[p2, p0]);
        let x = match *interp {
            io::Interp::Interpolated { barycentric, .. } => {
                let (perspective, centroid, sample) =
                    bary_mode(barycentric).ok_or(unsupported(off, "pull-model interpolation"))?;
                let b = |l: &mut Self, comp| {
                    l.f.global(Op::Builtin(Builtin::Bary { perspective, centroid, sample, comp }), Ty::F32)
                };
                let (i, j) = (b(self, 1), b(self, 2));
                let t = self.gl(glsl::FMA, Ty::F32, &[i, p10, p0]);
                self.gl(glsl::FMA, Ty::F32, &[j, p20, t])
            }
            io::Interp::Vertex { vertex: 2, .. } => p0,
            io::Interp::Vertex { vertex: 0, .. } => p10,
            io::Interp::Vertex { .. } => p20,
            io::Interp::Weighted { .. } => {
                // p1: P0 + i P10; p2: old + j P20 (i, j computed by the program)
                let (src, old) = match &inst.fields {
                    Fields::Vintrp { src, .. } => {
                        let o = self.val(inst, Reg::V(reg))?;
                        (*src, Some(o))
                    }
                    Fields::Valu(v) => (v.src.get(1).copied().ok_or(unsupported(off, "an interpolation"))?, None),
                    _ => return Err(unsupported(off, "an interpolation")),
                };
                let w = self.opnd(inst, &src)?;
                let w = self.f32v(w);
                if name.starts_with("v_interp_p1") {
                    self.gl(glsl::FMA, Ty::F32, &[w, p10, p0])
                } else {
                    let old = match (old, &inst.fields) {
                        (Some(o), _) => self.data(o),
                        (None, Fields::Valu(v)) => self.opnd(inst, &v.src[2])?,
                        _ => return Err(unsupported(off, "an interpolation")),
                    };
                    let old = self.f32v(old);
                    self.gl(glsl::FMA, Ty::F32, &[w, p20, old])
                }
            }
        };
        self.iface.per_vertex = true;
        self.def(inst, Reg::V(reg), R::D(x));
        Ok(())
    }
}

/// (perspective, centroid, sample) of a barycentric kind; none for the pull
/// model.
pub(super) fn bary_mode(kind: io::Barycentric) -> Option<(bool, bool, bool)> {
    use io::Barycentric as B;
    Some(match kind {
        B::PerspectiveCenter => (true, false, false),
        B::PerspectiveCentroid => (true, true, false),
        B::PerspectiveSample => (true, false, true),
        B::LinearCenter => (false, false, false),
        B::LinearCentroid => (false, true, false),
        B::LinearSample => (false, false, true),
        B::PerspectivePullModel => return None,
    })
}
