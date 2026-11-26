//! Stage interface: inputs, exports, interpolation.

use super::*;

impl Lift<'_> {
    pub(super) fn input(&mut self, reg: Reg, v: Value, offset: usize) -> Res<R> {
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
                Input::Barycentric { kind, index } => {
                    // i and j: the weights of vertices 1 and 2
                    use io::Barycentric as B;
                    let (perspective, centroid, sample) = match kind {
                        B::PerspectiveCenter => (true, false, false),
                        B::PerspectiveCentroid => (true, true, false),
                        B::PerspectiveSample => (true, false, true),
                        B::LinearCenter => (false, false, false),
                        B::LinearCentroid => (false, true, false),
                        B::LinearSample => (false, false, true),
                        B::PerspectivePullModel => return Err(unsupported(offset, "pull-model barycentrics")),
                    };
                    if index > 1 {
                        return Err(unsupported(offset, "pull-model barycentrics"));
                    }
                    let b = Builtin::Bary { perspective, centroid, sample, comp: index + 1 };
                    let x = self.f.global(Op::Builtin(b), Ty::F32);
                    self.iface.per_vertex = true;
                    x
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
