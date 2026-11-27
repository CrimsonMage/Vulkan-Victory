//! The Vulkan backend: a console program as a SPIR-V module for the Vulkan
//! host (`docs/vulkan-abi.md`).
//!
//! Stages: the decoder (`gfx10`), the wave SSA (`ir`, `spill`), I/O raising
//! (`io`) as before; then [`lift`] (wave SSA to the thread IR, [`tir`]),
//! dead-code removal, and [`emit`] (thread IR to SPIR-V, written by
//! [`spirv`]). No descriptor tracing, no page directory: descriptors and
//! title memory are read the way the console reads them, through buffer
//! device addresses over the host's imports of title memory.

pub mod emit;
pub mod graph;
pub mod ifconv;
pub mod interp;
#[cfg(test)]
mod oracle;
pub mod lift;
pub mod pack16;
pub mod report;
pub mod semantics;
pub mod spirv;
pub mod tir;
pub mod usage;

use std::collections::HashMap;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io};
pub use lift::Approx;
pub use report::{Report, report};
pub use semantics::Semantics;

/// Version of the interface in `docs/vulkan-abi.md`.
pub const ABI_VERSION: u32 = 5;

/// (row, channel) -> (location, component) for the user-semantic rows of
/// a signature part (`locs`: each row's location, from the registry).
fn locations(rows: &[&SigRow], locs: &[u8]) -> HashMap<(u8, u8), (u8, u8)> {
    let mut out = HashMap::new();
    for (r, &loc) in rows.iter().zip(locs).filter(|(r, _)| r.reg < 32) {
        let first = r.mask.trailing_zeros().min(3) as u8;
        for c in 0..4u8 {
            if r.mask >> c & 1 != 0 {
                out.insert((r.reg as u8, c), (loc, c - first));
            }
        }
    }
    out
}

/// One element of a stage's interface, for linking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interface {
    /// Semantic name and index (from the console signature).
    pub semantic: String,
    pub index: u32,
    pub location: u8,
    /// Input (pixel) or output (vertex/mesh, pixel targets).
    pub input: bool,
    /// Word index of the `Location` literal in [`VkShader::spirv`] (inputs).
    pub word: Option<usize>,
}

/// A translated stage and what the host binds for it.
#[derive(Debug, Clone)]
pub struct VkShader {
    pub stage: HwStage,
    pub spirv: Vec<u32>,
    /// Push-constant dword `index` (absolute) <- root parameter `slot`, dword.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: set 1, binding 0 (binding 1 for pixel programs).
    pub samplers: Vec<[u32; 4]>,
    pub interface: Vec<Interface>,
    pub local_size: [u32; 3],
    /// Approximations taken, once each (`docs/vulkan-abi.md` section 9).
    pub approximate: Vec<Approx>,
    /// Thread-IR instructions after dead-code removal (statistics).
    pub ir_insts: usize,
    /// What the loop guard writes to the guard buffer's dword 1 when a
    /// loop of this program gives up (FNV-1a of the container).
    pub guard_id: u32,
    /// Mesh: host groups per console group (1, or a split's parts: the
    /// host dispatches y × `mesh_parts` groups in y, `docs/vulkan-abi.md`
    /// section 6).
    pub mesh_parts: u32,
}

struct SigRow {
    name: String,
    index: u32,
    sv: u32,
    comp: u32,
    reg: u32,
    mask: u8,
    /// 16-bit (minimum precision or a native 16-bit type).
    half: bool,
}

fn signature(part: &[u8]) -> Result<Vec<SigRow>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        out.push(SigRow {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            sv: u32_at(part, at + 12)?,
            comp: u32_at(part, at + 16)?,
            reg: u32_at(part, at + 20)?,
            mask: *part.get(at + 24).ok_or(ShaderError::Malformed("truncated signature"))?,
            half: matches!(u32_at(part, at + 16)?, 4..=6) || u32_at(part, at + 28)? != 0,
        });
    }
    Ok(out)
}

/// The vertex, pixel, compute or mesh program of a container.
pub fn native(container: &[u8]) -> Result<(Container<'_>, NativeProgram), ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c.part(FourCC::DXIL).ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms))
        .ok_or(ShaderError::Unsupported { offset: 0, what: "a container without a vertex, pixel, compute or mesh program" })?;
    Ok((c, native))
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (the value: a constant, or a descriptor table's heap index).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// Translates a console shader container to SPIR-V, with the process-wide
/// semantic registry.
pub fn translate(container: &[u8]) -> Result<VkShader, ShaderError> {
    translate_with(container, Semantics::global())
}

/// Translates a console shader container to SPIR-V; interface semantics
/// take their locations from `semantics`.
pub fn translate_with(container: &[u8], semantics: &Semantics) -> Result<VkShader, ShaderError> {
    let (c, native) = native(container)?;
    let (f, iface) = thread_ir_of(&c, &native)?;
    let ir_insts = f.count();
    let osg = c.part(FourCC::OSG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
    let isg = c.part(FourCC::ISG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
    let ps = native.stage == HwStage::Ps;
    let rows = if ps { &isg } else { &osg };
    let (local_size, mesh) = match &native.io {
        crate::native::StageIo::Cs(_) => ([lift::Lift::host_threads(&native), 1, 1], (0, 0)),
        crate::native::StageIo::Ms(ms) => ([ms.threads, 1, 1], (ms.max_vertices, ms.max_primitives)),
        _ => ([1, 1, 1], (0, 0)),
    };
    let out_types = if ps { pixel_out_types(&osg) } else { HashMap::new() };
    let user: Vec<&SigRow> = rows.iter().filter(|r| r.sv == 0).collect();
    let names: Vec<(&str, u32)> = user.iter().map(|r| (r.name.as_str(), r.index)).collect();
    let locs = semantics.assign(&names).map_err(|what| ShaderError::Unsupported { offset: 0, what })?;
    let st = emit::Stage {
        stage: native.stage,
        local_size,
        out_types,
        locs: locations(&user, &locs),
        mesh,
        mesh_split: None,
        loop_limit: loop_limit(native.stage),
        guard_id: guard_id(container),
        no_sel: Default::default(),
    };
    let (e, mesh_parts) = emit_mesh_fitting(&f, &iface, st)?;
    let mut interface = Vec::new();
    if native.stage == HwStage::Vs {
        interface = vertex_inputs(&isg, &e.vertex_inputs);
    }
    // the rest is information only: locations are fixed per semantic
    interface.extend(user.iter().zip(&locs).map(|(r, &location)| Interface {
        semantic: r.name.clone(),
        index: r.index,
        location,
        input: ps,
        word: None,
    }));
    Ok(VkShader {
        stage: native.stage,
        spirv: e.words,
        user_data: user_data(&native),
        samplers: iface.samplers.clone(),
        interface,
        local_size,
        approximate: iface.approximate.clone(),
        ir_insts,
        guard_id: guard_id(container),
        mesh_parts,
    })
}

/// Emits the module; a mesh module over the output budget
/// ([`usage::MESH_OUTPUT_BUDGET`]) again as a split with a window of 96
/// vertices, then 64 (`emit_mesh.rs`). Returns the module and the host
/// groups per console group; a module still over is returned whole (the
/// host refuses it with the reason).
fn emit_mesh_fitting(f: &tir::Func, iface: &lift::Interface, st: emit::Stage) -> Result<(emit::Emitted, u32), ShaderError> {
    let budget = usage::MESH_OUTPUT_BUDGET;
    let fits = |e: &emit::Emitted| usage::usage(&e.words).mesh_output_bytes(budget) <= u64::from(budget.bytes);
    let e = emit_retrying(f, iface, st.clone())?;
    // each part runs the whole program: one that writes memory cannot split
    let writes = f.nodes.iter().any(|n| {
        matches!(n.op, tir::Op::Store { .. } | tir::Op::Atomic(_))
            || matches!(n.op, tir::Op::Image { kind: tir::ImageKind::Store { .. } | tir::ImageKind::Atomic(_), .. })
    });
    if st.stage != HwStage::Ms || writes || fits(&e) {
        return Ok((e, 1));
    }
    for window in [96, 64] {
        let Some(split) = emit::MeshSplit::for_window(st.mesh.0, window) else { continue };
        let split_e = emit_retrying(f, iface, emit::Stage { mesh_split: Some(split), ..st.clone() })?;
        if fits(&split_e) {
            return Ok((split_e, split.parts()));
        }
    }
    Ok((e, 1))
}

/// A container's thread program (the IR [`emit`] writes as SPIR-V), as the
/// translation makes it: for [`interp`] and other offline checks.
pub fn thread_ir(container: &[u8]) -> Result<(tir::Func, lift::Interface, NativeProgram), ShaderError> {
    let (c, native) = native(container)?;
    let (f, iface) = thread_ir_of(&c, &native)?;
    Ok((f, iface, native))
}

/// The thread program of a bare compute program (no container: no
/// signatures), for offline tests.
pub fn thread_ir_native(native: &NativeProgram) -> Result<(tir::Func, lift::Interface), ShaderError> {
    thread_ir_parts(native, &[], &[], 0)
}

fn thread_ir_of(c: &Container<'_>, native: &NativeProgram) -> Result<(tir::Func, lift::Interface), ShaderError> {
    let osg = c.part(FourCC::OSG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
    let isg = c.part(FourCC::ISG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
    thread_ir_parts(native, &osg, &isg, mesh_stride(c, native))
}

fn thread_ir_parts(native: &NativeProgram, osg: &[SigRow], isg: &[SigRow], stride: u32) -> Result<(tir::Func, lift::Interface), ShaderError> {
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    // The branches that only skip code when no lane is left for it go: the
    // skipped vector code is guarded by EXEC anyway, and without them the
    // wave's graph is the structured program.
    let mut p = crate::ir::without_skips(&p, true).unwrap_or(p);
    // then a spilled user-data table's loads become entry values (after:
    // the skips' renaming goes by register and would undo it)
    crate::spill::user_data(&mut p, native);
    let io = io::raise(&p, native)?;
    let rows = if native.stage == HwStage::Ps { isg } else { osg };
    let iface_elems = elements(rows);
    let mut lift = lift::Lift::new(&p, native, &io);
    lift.mesh_stride = stride;
    lift.halves = pack16::halves(&native.io, &iface_elems);
    lift.unnamed_row = pack16::unnamed_row(&native.io, &iface_elems);
    if native.stage == HwStage::Vs {
        lift.vs_float = isg.iter().filter(|r| r.reg < 32).map(|r| (r.reg as u8, !matches!(r.comp, 1 | 2 | 4 | 5))).collect();
    }
    let (mut f, iface) = lift.run()?;
    tir::fold_branches(&mut f);
    ifconv::if_convert(&mut f);
    tir::dce(&mut f);
    tir::check(&f).map_err(|what| ShaderError::Unsupported { offset: 0, what })?;
    Ok((f, iface))
}

/// Loop iterations (all loops of an invocation together) after which the
/// invocation gives up: low enough that the programs that give up cannot
/// make a GPU timeout by themselves (a culling dispatch of thousands of
/// waves whose loops each ran to 2^16 took a large part of the ~2 s
/// limit), far above what a loop of the title runs.
fn loop_limit(_stage: HwStage) -> u32 {
    1 << 14
}

/// FNV-1a of the container: the program's id in the guard buffer.
pub fn guard_id(container: &[u8]) -> u32 {
    container.iter().fold(0x811C_9DC5u32, |h, &b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193))
}

/// A mesh group's id arrives multiplied by max(threads, vertices,
/// primitives); the thread count is in PSV0 from runtime info version 2.
fn mesh_stride(c: &Container<'_>, native: &NativeProgram) -> u32 {
    let crate::native::StageIo::Ms(ms) = &native.io else { return 0 };
    let psv = c.part(FourCC::PSV0).map(|p| p.data);
    let threads =
        psv.filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48)).and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
    threads.map_or(ms.threads, |t| t.max(ms.max_vertices).max(ms.max_primitives))
}

/// The interface elements (registers below 32) for 16-bit packing.
fn elements(rows: &[SigRow]) -> Vec<pack16::Elem> {
    rows.iter()
        .filter(|r| r.reg < 32)
        .map(|r| pack16::Elem {
            row: r.reg as u8,
            col: r.mask.trailing_zeros().min(3) as u8,
            cols: r.mask.count_ones().max(1) as u8,
            half: r.half,
            user: r.sv == 0,
        })
        .collect()
}

/// A pixel program's render target types: 0 float, 1 uint, 2 sint.
fn pixel_out_types(osg: &[SigRow]) -> HashMap<u16, u8> {
    let kind = |comp| match comp {
        1 | 4 => 1,
        2 | 5 => 2,
        _ => 0,
    };
    osg.iter().filter(|r| r.sv == 64).map(|r| (r.reg as u16, kind(r.comp))).collect()
}

/// Joins as selections; a selection that a jump had to leave through a
/// dispatch is a block construct in the next attempt (all of them, at worst).
fn emit_retrying(f: &tir::Func, iface: &lift::Interface, mut st: emit::Stage) -> Result<emit::Emitted, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    let g = graph::Graph::new(f).map_err(unsupported)?;
    let run = |st: &emit::Stage| emit::Emit::new(f, &g, iface, st).run().map_err(unsupported);
    let mut e = run(&st)?;
    for round in 0..=8 {
        if e.bad_sel.is_empty() {
            break;
        }
        if round == 8 {
            st.no_sel = (0..f.blocks.len()).collect();
        } else {
            st.no_sel.extend(e.bad_sel.iter().copied());
        }
        e = run(&st)?;
    }
    Ok(e)
}

/// The push constants user data occupies, by root parameter.
fn user_data(native: &NativeProgram) -> Vec<UserData> {
    let window = lift::push_window(native.stage);
    let mut user_data = Vec::new();
    for r in &native.layout {
        if !matches!(r.root, RootType::Constants | RootType::Descriptor | RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) {
            continue;
        }
        for d in 0..r.sgprs {
            user_data.push(UserData { index: window + u32::from(r.user_data) + d, slot: r.slot, root: r.root, dword: d });
        }
    }
    user_data
}

/// Vertex attributes (a vertex program with a fetch shader): Location =
/// the input-signature row; the host matches its input layout by semantic.
fn vertex_inputs(isg: &[SigRow], used: &[u8]) -> Vec<Interface> {
    isg.iter()
        .filter(|r| r.sv == 0 && r.reg < 32 && used.contains(&(r.reg as u8)))
        .map(|r| Interface { semantic: r.name.clone(), index: r.index, location: r.reg as u8, input: true, word: None })
        .collect()
}

