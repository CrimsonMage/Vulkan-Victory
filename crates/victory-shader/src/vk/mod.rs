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
pub mod spirv;
pub mod tir;

use std::collections::HashMap;

use crate::dxbc::{Container, FourCC};
use crate::emit::UserData;
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io};

/// Version of the interface in `docs/vulkan-abi.md`.
pub const ABI_VERSION: u32 = 0;

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
    if native.stage == HwStage::Ms {
        return Err(ShaderError::Unsupported { offset: 0, what: "mesh programs (not yet)" });
    }
    let p = crate::spill::build(&native)?;
    let io = io::raise(&p, &native)?;
    let (mut f, iface) = lift::Lift::new(&p, &native, &io).run()?;
    if std::env::var("VK_DUMP").is_ok_and(|v| v == "2") {
        tir::dump(&f);
    }
    tir::dce(&mut f);
    if std::env::var("VK_DUMP").is_ok_and(|v| v == "1") {
        tir::dump(&f);
    }
    let ir_insts = f.count();
    let osg = c.part(FourCC::OSG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
    let isg = c.part(FourCC::ISG1).map(|p| signature(p.data)).transpose()?.unwrap_or_default();
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
    let local_size = match &native.io {
        crate::native::StageIo::Cs(cs) => cs.threads,
        _ => [1, 1, 1],
    };
    let loop_limit = match native.stage {
        HwStage::Ps => 128,
        HwStage::Cs => 4096,
        _ => 256,
    };
    let st = emit::Stage { stage: native.stage, local_size, out_types, loop_limit };
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
    // interface
    let mut interface = Vec::new();
    for &(loc, word) in &e.input_locations {
        if let Some(r) = isg.iter().find(|r| r.reg == u32::from(loc)) {
            interface.push(Interface { semantic: r.name.clone(), index: r.index, location: loc, input: true, word: Some(word) });
        }
    }
    for &(slot, _) in &iface.outputs {
        if slot >= 0x100 {
            continue;
        }
        if let Some(r) = osg.iter().find(|r| r.reg == u32::from(slot) && (native.stage != HwStage::Ps || r.sv == 64)) {
            interface.push(Interface { semantic: r.name.clone(), index: r.index, location: slot as u8, input: false, word: None });
        }
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

/// Links a pixel module to the vertex (or mesh) module it runs after: each
/// pixel input takes the location of the vertex output with the same
/// semantic; one no output names reads from a location past both.
pub fn link(vertex: &VkShader, pixel: &mut VkShader) {
    let mut spare = vertex.interface.iter().filter(|i| !i.input).map(|i| u32::from(i.location) + 1).max().unwrap_or(0);
    for i in pixel.interface.iter_mut().filter(|i| i.input) {
        let to = vertex
            .interface
            .iter()
            .find(|o| !o.input && o.semantic.eq_ignore_ascii_case(&i.semantic) && o.index == i.index)
            .map(|o| u32::from(o.location))
            .unwrap_or_else(|| {
                spare += 1;
                spare - 1
            });
        if let Some(w) = i.word {
            pixel.spirv[w] = to;
        }
        i.location = to as u8;
    }
}
