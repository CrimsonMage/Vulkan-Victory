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
pub mod lift;
pub mod pack16;
pub mod spirv;
pub mod tir;

use std::collections::HashMap;

use crate::dxbc::{Container, FourCC};
use crate::emit::UserData;
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io};

/// Version of the interface in `docs/vulkan-abi.md`.
pub const ABI_VERSION: u32 = 1;

/// The location of each interface semantic (name, index), the same in
/// every program of the pack: 37 semantics, at most 23 in one program,
/// coloured so that no two semantics one program uses share a location
/// (found offline from the whole pack). Others take location 31.
pub const SEMANTIC_LOCATIONS: &[(&str, u32, u8)] = &[
    ("TEXCOORD", 0, 0),
    ("TEXCOORD", 1, 1),
    ("TEXCOORD", 3, 2),
    ("COLOR", 0, 3),
    ("TEXCOORD", 4, 4),
    ("COLOR", 1, 5),
    ("TEXCOORD", 2, 6),
    ("TEXCOORD", 9, 7),
    ("TEXCOORD", 6, 8),
    ("TEXCOORD", 5, 9),
    ("TEXCOORD", 10, 10),
    ("TEXCOORD", 7, 11),
    ("TEXCOORD", 8, 12),
    ("COLOR", 2, 13),
    ("TEXCOORD", 15, 14),
    ("TEXCOORD", 11, 15),
    ("TEXCOORD", 20, 16),
    ("TEXCOORD", 21, 17),
    ("TEXCOORD", 13, 18),
    ("TEXCOORD", 17, 19),
    ("TEXCOORD", 14, 20),
    ("COLOR", 3, 21),
    ("TEXCOORD", 19, 22),
    ("TEXCOORD", 22, 23),
    ("TEXCOORD", 12, 24),
    ("TEXCOORD", 16, 25),
    ("TEXCOORD", 23, 26),
    ("TEXCOORD", 18, 7),
    ("NORMAL", 2, 18),
    ("NORMAL", 0, 3),
    ("TANGENT", 0, 10),
    ("TEXCOORD", 24, 15),
    ("NORMAL", 1, 5),
    ("POSITION", 1, 7),
    ("INSTANCEID", 0, 6),
    ("TANGENT", 1, 3),
    ("POSITION", 0, 1),
];

/// The location of a semantic.
pub fn semantic_location(name: &str, index: u32) -> u8 {
    SEMANTIC_LOCATIONS
        .iter()
        .find(|s| s.0.eq_ignore_ascii_case(name) && s.1 == index)
        .map_or(31, |s| s.2)
}

/// (row, channel) -> (location, component) for the user-semantic rows of
/// a signature part.
fn locations(rows: &[SigRow]) -> HashMap<(u8, u8), (u8, u8)> {
    let mut out = HashMap::new();
    for r in rows.iter().filter(|r| r.sv == 0 && r.reg < 32) {
        let first = r.mask.trailing_zeros().min(3) as u8;
        let loc = semantic_location(&r.name, r.index);
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
    pub approximate: Vec<&'static str>,
    /// Thread-IR instructions after dead-code removal (statistics).
    pub ir_insts: usize,
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

/// Translates a console shader container to SPIR-V.
pub fn translate(container: &[u8]) -> Result<VkShader, ShaderError> {
    let (c, native) = native(container)?;
    let p = crate::spill::build(&native)?;
    // The branches that only skip code when no lane is left for it go: the
    // skipped vector code is guarded by EXEC anyway, and without them the
    // wave's graph is the structured program.
    let p = if std::env::var_os("VK_SKIPS").is_some() { p } else { crate::ir::without_skips(&p, true).unwrap_or(p) };
    let io = io::raise(&p, &native)?;
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives); the thread count is in PSV0 from runtime info version 2.
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| t.max(ms.max_vertices).max(ms.max_primitives))
        }
        _ => 0,
    };
    let osg = c.part(FourCC::OSG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
    let isg = c.part(FourCC::ISG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
    let elems = |rows: &[SigRow]| -> Vec<pack16::Elem> {
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
    };
    let iface_elems = if native.stage == HwStage::Ps { elems(&isg) } else { elems(&osg) };
    let mut lift = lift::Lift::new(&p, &native, &io);
    lift.mesh_stride = mesh_stride;
    lift.halves = pack16::halves(&native.io, &iface_elems);
    lift.unnamed_row = pack16::unnamed_row(&native.io, &iface_elems);
    if native.stage == HwStage::Vs {
        lift.vs_float = isg.iter().filter(|r| r.reg < 32).map(|r| (r.reg as u8, !matches!(r.comp, 1 | 2 | 4 | 5))).collect();
    }
    let (mut f, iface) = lift.run()?;
    if std::env::var("VK_DUMP").is_ok_and(|v| v == "2") {
        tir::dump(&f);
    }
    tir::dce(&mut f);
    if std::env::var("VK_DUMP").is_ok_and(|v| v == "1") {
        tir::dump(&f);
    }
    let ir_insts = f.count();
    let mut out_types = HashMap::new();
    if native.stage == HwStage::Ps {
        for r in &osg {
            if r.sv == 64 {
                out_types.insert(r.reg as u16, match r.comp {
                    1 | 4 => 1,
                    2 | 5 => 2,
                    _ => 0,
                });
            }
        }
    }
    let (local_size, mesh) = match &native.io {
        crate::native::StageIo::Cs(cs) => (cs.threads, (0, 0)),
        crate::native::StageIo::Ms(ms) => ([ms.threads, 1, 1], (ms.max_vertices, ms.max_primitives)),
        _ => ([1, 1, 1], (0, 0)),
    };
    let loop_limit = match native.stage {
        HwStage::Ps => 128,
        HwStage::Cs => 4096,
        _ => 256,
    };
    let locs = if native.stage == HwStage::Ps { locations(&isg) } else { locations(&osg) };
    let st = emit::Stage { stage: native.stage, local_size, out_types, loop_limit, locs, mesh };
    let e = emit::Emit::new(&f, &iface, &st).run().map_err(|what| ShaderError::Unsupported { offset: 0, what })?;
    // user data
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
    // interface (information only: locations are fixed per semantic)
    let mut interface = Vec::new();
    // vertex attributes (a vertex program with a fetch shader): Location =
    // the input-signature row; the host matches its input layout by semantic
    if native.stage == HwStage::Vs {
        for r in isg.iter().filter(|r| r.sv == 0 && r.reg < 32) {
            if e.vertex_inputs.contains(&(r.reg as u8)) {
                interface.push(Interface {
                    semantic: r.name.clone(),
                    index: r.index,
                    location: r.reg as u8,
                    input: true,
                    word: None,
                });
            }
        }
    }
    let rows = if native.stage == HwStage::Ps { &isg } else { &osg };
    for r in rows.iter().filter(|r| r.sv == 0) {
        interface.push(Interface {
            semantic: r.name.clone(),
            index: r.index,
            location: semantic_location(&r.name, r.index),
            input: native.stage == HwStage::Ps,
            word: None,
        });
    }
    Ok(VkShader {
        stage: native.stage,
        spirv: e.words,
        user_data,
        samplers: iface.samplers.clone(),
        interface,
        local_size,
        approximate: iface.approximate.clone(),
        ir_insts,
    })
}

