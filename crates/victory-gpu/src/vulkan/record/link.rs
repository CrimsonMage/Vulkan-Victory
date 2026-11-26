//! Pipelines for draws: the vertex-input and output parts, fast links of
//! four parts (cached), the parts of the draw tests, and the vertex
//! attributes of fetch-shader programs.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use victory_vk::ash::vk;

use super::super::{Backend, lock};
use super::super::stages::{self, Lookup, Stage};
use super::STATS;

/// Fast-linked pipelines by their parts and keys.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct LinkKey {
    pub(super) pre: vk::Pipeline,
    pub(super) fragment: vk::Pipeline,
    pub(super) colors: [vk::Format; 8],
    pub(super) count: u32,
    pub(super) depth: vk::Format,
    pub(super) inputs: Vec<(u32, u32, vk::Format, u32)>,
}

static LINKS: Mutex<Option<HashMap<LinkKey, vk::Pipeline>>> = Mutex::new(None);

/// What picks a draw's pipeline in a recording: the title's pipeline, the
/// pass's formats, rectangles or not, mesh or not.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::vulkan) struct DrawKey {
    pub(super) pipeline: usize,
    pub(super) formats: [vk::Format; 8],
    pub(super) count: u32,
    pub(super) depth: vk::Format,
    pub(super) rect: bool,
    pub(super) mesh: bool,
}

/// A draw's linked pipeline, its vertex attributes, and the bindings they
/// name in order of first use.
pub(in crate::vulkan) struct Linked {
    pub(super) pipeline: vk::Pipeline,
    pub(super) inputs: Vec<(u32, u32, vk::Format, u32)>,
    pub(super) bindings: Vec<u32>,
}

/// The pipeline of a draw with the stages `pre` and `pixel` (made: the
/// parts and the link, each cached), `None` if a part is missing.
pub(super) fn linked(b: &'static Backend, pre: &Stage, pixel: Option<&Stage>, gs: &crate::pipeline::host::GraphicsState, k: DrawKey) -> Option<Linked> {
    let inputs = attributes(pre, gs);
    let fragment = match pixel {
        _ if draw_test() >= 2 => solid_fragment(b)?,
        Some(p) => p.part?,
        None => empty_fragment(b)?,
    };
    let output = part(b, &[], k.formats, k.count, k.depth, true)?;
    let vertex_input = if k.mesh { None } else { Some(part(b, &inputs, [vk::Format::UNDEFINED; 8], 0, vk::Format::UNDEFINED, false)?) };
    let key = LinkKey {
        pre: pre_part(b, pre, k.rect, k.mesh)?,
        fragment,
        colors: k.formats,
        count: k.count,
        depth: k.depth,
        inputs: inputs.clone(),
    };
    let pipeline = link(b, key, vertex_input, output)?;
    let mut bindings: Vec<u32> = Vec::new();
    for &(_, binding, _, _) in &inputs {
        if !bindings.contains(&(binding & 0xff)) {
            bindings.push(binding & 0xff);
        }
    }
    Some(Linked { pipeline, inputs, bindings })
}

/// The pre-rasterisation part of a draw: the vertex or mesh program's,
/// the rectangle part for rectangle lists, the test's full-screen
/// triangle under `VICTORY_VK_DRAW_TEST=3`.
fn pre_part(b: &'static Backend, pre: &Stage, rect: bool, mesh: bool) -> Option<vk::Pipeline> {
    if draw_test() >= 3 && !mesh {
        return fullscreen_vertex(b);
    }
    if rect && !mesh {
        let part = pre.rect_part(b);
        if part.is_none() {
            super::not_yet(super::NotYet::Topology);
        }
        return part;
    }
    pre.part
}
static PARTS: Mutex<Option<HashMap<(Vec<(u32, u32, vk::Format, u32)>, [vk::Format; 8], u32, vk::Format, bool), vk::Pipeline>>> =
    Mutex::new(None);

pub(super) fn library_flags(b: &Backend) -> vk::PipelineCreateFlags {
    let mut f = vk::PipelineCreateFlags::LIBRARY_KHR;
    if b.host.caps.descriptor_buffer {
        f |= vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT;
    }
    f
}

/// The vertex-input part (`inputs`: location, binding, format, offset) or
/// the fragment-output part (formats), made once each.
pub(super) fn part(b: &Backend, inputs: &[(u32, u32, vk::Format, u32)], colors: [vk::Format; 8], count: u32, depth: vk::Format, output: bool) -> Option<vk::Pipeline> {
    let key = (inputs.to_vec(), colors, count, depth, output);
    let mut guard = lock(&PARTS);
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(&p) = map.get(&key) {
        return Some(p);
    }
    let layout = stages::layout(b)?;
    use vk::DynamicState as D;
    let mut flags = vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE;
    let mut states = vec![D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE, D::VERTEX_INPUT_BINDING_STRIDE];
    if output {
        flags = vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE;
        states = vec![D::BLEND_CONSTANTS];
        if b.host.caps.dynamic_blend {
            states.extend([D::COLOR_BLEND_ENABLE_EXT, D::COLOR_BLEND_EQUATION_EXT, D::COLOR_WRITE_MASK_EXT]);
        }
        if b.host.caps.dynamic_raster {
            states.extend([D::SAMPLE_MASK_EXT, D::ALPHA_TO_COVERAGE_ENABLE_EXT]);
        }
    }
    let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
    let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(flags);
    let formats = &colors[..count as usize];
    let stencil = matches!(depth, vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT);
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(formats)
        .depth_attachment_format(depth)
        .stencil_attachment_format(if stencil { depth } else { vk::Format::UNDEFINED });
    let mut bindings: Vec<vk::VertexInputBindingDescription> = Vec::new();
    for &(_, binding, _, _) in inputs {
        if !bindings.iter().any(|x| x.binding == binding & 0xff) {
            bindings.push(
                vk::VertexInputBindingDescription::default()
                    .binding(binding & 0xff)
                    .stride(0)
                    .input_rate(if binding & 0x100 != 0 { vk::VertexInputRate::INSTANCE } else { vk::VertexInputRate::VERTEX }),
            );
        }
    }
    let attributes: Vec<vk::VertexInputAttributeDescription> = inputs
        .iter()
        .map(|&(location, binding, format, offset)| {
            vk::VertexInputAttributeDescription::default()
                .location(location)
                .binding(binding & 0xff)
                .format(format)
                .offset(offset)
        })
        .collect();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let attachments: Vec<vk::PipelineColorBlendAttachmentState> = (0..count)
        .map(|_| vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA))
        .collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let samples = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .flags(library_flags(b))
        .dynamic_state(&dyn_info)
        .push_next(&mut library);
    if output {
        info = info
            .color_blend_state(&blend)
            .multisample_state(&samples)
            .layout(layout.pipeline)
            .push_next(&mut rendering);
    } else {
        info = info.vertex_input_state(&vertex_input).input_assembly_state(&assembly);
    }
    // SAFETY: valid create infos.
    let p = unsafe { b.host.device.create_graphics_pipelines(stages::cache(b), &[info], None) }.ok()?[0];
    map.insert(key, p);
    Some(p)
}

/// A fragment part without a shader (depth-only pipelines).
pub(super) fn empty_fragment(b: &Backend) -> Option<vk::Pipeline> {
    static EMPTY: std::sync::OnceLock<Option<vk::Pipeline>> = std::sync::OnceLock::new();
    *EMPTY.get_or_init(|| {
        let layout = stages::layout(b)?;
        use vk::DynamicState as D;
        let states = [
            D::DEPTH_TEST_ENABLE,
            D::DEPTH_WRITE_ENABLE,
            D::DEPTH_COMPARE_OP,
            D::DEPTH_BOUNDS_TEST_ENABLE,
            D::DEPTH_BOUNDS,
            D::STENCIL_TEST_ENABLE,
            D::STENCIL_OP,
            D::STENCIL_COMPARE_MASK,
            D::STENCIL_WRITE_MASK,
            D::STENCIL_REFERENCE,
        ];
        let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
        let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER);
        let mut rendering = vk::PipelineRenderingCreateInfo::default();
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let samples = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .flags(library_flags(b))
            .layout(layout.pipeline)
            .dynamic_state(&dyn_info)
            .depth_stencil_state(&depth)
            .multisample_state(&samples)
            .push_next(&mut library)
            .push_next(&mut rendering);
        // SAFETY: valid create infos.
        unsafe { b.host.device.create_graphics_pipelines(stages::cache(b), &[info], None) }.ok().map(|p| p[0])
    })
}

pub(super) fn link(b: &Backend, key: LinkKey, vertex_input: Option<vk::Pipeline>, output: vk::Pipeline) -> Option<vk::Pipeline> {
    let mut guard = lock(&LINKS);
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(&p) = map.get(&key) {
        return Some(p);
    }
    let layout = stages::layout(b)?;
    let mut libs = vec![key.pre, key.fragment, output];
    if let Some(vi) = vertex_input {
        libs.push(vi);
    }
    let mut libraries = vk::PipelineLibraryCreateInfoKHR::default().libraries(&libs);
    let flags = if b.host.caps.descriptor_buffer {
        vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT
    } else {
        vk::PipelineCreateFlags::empty()
    };
    let info = vk::GraphicsPipelineCreateInfo::default()
        .flags(flags)
        .layout(layout.pipeline)
        .push_next(&mut libraries);
    // SAFETY: valid create infos over live parts.
    let p = unsafe { b.host.device.create_graphics_pipelines(stages::cache(b), &[info], None) }.ok()?[0];
    STATS[6].fetch_add(1, Ordering::Relaxed);
    map.insert(key, p);
    Some(p)
}

/// Diagnostics: each stage a draw found not ready (after the frame's
/// waiting budget), once.
fn waiting_once(s: &crate::pipeline::Shader) {
    static SEEN: Mutex<Option<std::collections::HashSet<[u8; 32]>>> = Mutex::new(None);
    if cfg!(feature = "diagnostics") && lock(&SEEN).get_or_insert_with(Default::default).insert(s.full) {
        tracing::info!(stage = s.stage, hash = %s.hash, name = %s.name, "Vulkan: a draw skipped: its stage was not compiled within the frame's waiting budget");
    }
}

pub(super) fn stage_of(b: &'static Backend, shaders: &[Arc<crate::pipeline::Shader>], name: &str) -> Result<Option<Arc<Stage>>, bool> {
    let Some(s) = shaders.iter().find(|s| s.stage == name) else {
        return Ok(None);
    };
    match stages::lookup_waiting(b, s) {
        Lookup::Ready(st) => Ok(Some(st)),
        Lookup::Waiting => {
            waiting_once(s);
            Err(true)
        }
        Lookup::Failed => Err(false),
    }
}

/// The vertex attributes of a fetch-shader program (ABI section 6) from
/// the title's input layout: (location, binding | 0x100 per instance,
/// format, offset). D3D12's "append aligned" offsets follow the elements
/// before them in the same slot.
pub(super) fn attributes(pre: &Stage, gs: &crate::pipeline::host::GraphicsState) -> Vec<(u32, u32, vk::Format, u32)> {
    if pre.inputs.is_empty() {
        return Vec::new();
    }
    // Every element's offset, in layout order.
    let mut ends = [0u32; 32];
    let offsets: Vec<u32> = gs
        .inputs
        .iter()
        .map(|e| {
            let slot = (e.InputSlot as usize).min(31);
            let at = if e.AlignedByteOffset == u32::MAX { ends[slot] } else { e.AlignedByteOffset };
            let bytes = crate::resource::element(e.Format.0 as u32).map_or(4, |(b, _)| b as u32);
            ends[slot] = at + bytes;
            at
        })
        .collect();
    let mut out = Vec::new();
    for (semantic, index, location) in &pre.inputs {
        let found = gs.inputs.iter().zip(&offsets).find(|(e, _)| {
            // SAFETY: the element's name points into the state's names.
            let name = unsafe { e.SemanticName.to_string() }.unwrap_or_default();
            name.eq_ignore_ascii_case(semantic) && e.SemanticIndex == *index
        });
        if let Some((e, &offset)) = found
            && let Some(f) = victory_vk::gfx10::from_dxgi(e.Format.0 as u32)
        {
            let rate = if e.InputSlotClass.0 == 1 { 0x100 } else { 0 };
            out.push((u32::from(*location), e.InputSlot | rate, f, offset));
        }
    }
    out
}

/// Diagnostics: the program's attribute inputs, the title's input layout
/// and the attributes made of them.
pub(super) fn layout_text(pre: &Stage, gs: &crate::pipeline::host::GraphicsState, made: &[(u32, u32, vk::Format, u32)]) -> String {
    let wants: Vec<String> = pre.inputs.iter().map(|(s, i, l)| format!("{s}{i}@{l}")).collect();
    let elements: Vec<String> = gs
        .inputs
        .iter()
        .map(|e| {
            // SAFETY: the element's name points into the state's names.
            let name = unsafe { e.SemanticName.to_string() }.unwrap_or_default();
            format!("{name}{} fmt{} slot{} off{:#x} class{}", e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0)
        })
        .collect();
    let attrs: Vec<String> = made.iter().map(|(l, b, f, o)| format!("loc{l} bind{b:#x} {f:?} +{o}")).collect();
    format!("program wants [{}]; layout [{}]; attributes [{}]", wants.join(", "), elements.join(", "), attrs.join(", "))
}

/// `VICTORY_VK_DRAW_TEST` (diagnostics): 1 draws with depth, stencil,
/// culling and blending off and every channel written; 2 also replaces
/// every pixel program by a constant magenta; 3 also every vertex program
/// by a triangle over the whole target. A capture then tells draws
/// that never rasterise from draws that compute zero.
pub(super) fn draw_test() -> u32 {
    static TEST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TEST.get_or_init(|| {
        let t = if cfg!(feature = "diagnostics") {
            std::env::var("VICTORY_VK_DRAW_TEST").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
        } else {
            0
        };
        if t > 0 {
            tracing::info!(test = t, "Vulkan: draw test (VICTORY_VK_DRAW_TEST)");
        }
        t
    })
}

/// The constant magenta pixel program of draw test 2 (HLSL
/// `float4 main() : SV_Target0 { return float4(1, 0, 1, 1); }`, compiled
/// with `glslc -x hlsl`), as a fragment part.
pub(super) fn solid_fragment(b: &Backend) -> Option<vk::Pipeline> {
    const SOLID: &[u32] = &[
    0x07230203, 0x00010600, 0x000d000b, 0x00000012, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000011, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000005, 0x000001f4, 0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45,
    0x74735f70, 0x5f656c79, 0x656e696c, 0x7269645f, 0x69746365, 0x00006576, 0x00080004, 0x475f4c47,
    0x4c474f4f, 0x6e695f45, 0x64756c63, 0x69645f65, 0x74636572, 0x00657669, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00070005, 0x00000011, 0x746e6540, 0x6f507972, 0x4f746e69, 0x75707475,
    0x00000074, 0x00040047, 0x00000011, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006,
    0x00000004, 0x0004002b, 0x00000006, 0x0000000b, 0x3f800000, 0x0004002b, 0x00000006, 0x0000000c,
    0x00000000, 0x0007002c, 0x00000007, 0x0000000d, 0x0000000b, 0x0000000c, 0x0000000b, 0x0000000b,
    0x00040020, 0x00000010, 0x00000003, 0x00000007, 0x0004003b, 0x00000010, 0x00000011, 0x00000003,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e,
    0x00000011, 0x0000000d, 0x000100fd, 0x00010038
    ];
    static PART: std::sync::OnceLock<Option<vk::Pipeline>> = std::sync::OnceLock::new();
    *PART.get_or_init(|| {
        let layout = stages::layout(b)?;
        let d = &b.host.device;
        // SAFETY: valid create infos; the module is kept with the part.
        unsafe {
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(SOLID), None).ok()?;
            use vk::DynamicState as D;
            let states = [
                D::DEPTH_TEST_ENABLE,
                D::DEPTH_WRITE_ENABLE,
                D::DEPTH_COMPARE_OP,
                D::DEPTH_BOUNDS_TEST_ENABLE,
                D::DEPTH_BOUNDS,
                D::STENCIL_TEST_ENABLE,
                D::STENCIL_OP,
                D::STENCIL_COMPARE_MASK,
                D::STENCIL_WRITE_MASK,
                D::STENCIL_REFERENCE,
            ];
            let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
            let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER);
            let mut rendering = vk::PipelineRenderingCreateInfo::default();
            let depth = vk::PipelineDepthStencilStateCreateInfo::default();
            let samples = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let stage = [vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(module).name(c"main")];
            let info = vk::GraphicsPipelineCreateInfo::default()
                .flags(library_flags(b))
                .stages(&stage)
                .layout(layout.pipeline)
                .dynamic_state(&dyn_info)
                .depth_stencil_state(&depth)
                .multisample_state(&samples)
                .push_next(&mut library)
                .push_next(&mut rendering);
            d.create_graphics_pipelines(stages::cache(b), &[info], None).ok().map(|p| p[0])
        }
    })
}

/// The vertex program of draw test 3: a triangle over the whole target
/// from the vertex index (HLSL `float4 main(uint id : SV_VertexID) :
/// SV_Position { float2 t = float2((id << 1) & 2, id & 2); return
/// float4(t * 2 - 1, 0.5, 1); }`, compiled with `glslc -x hlsl`), as a
/// pre-rasterisation part.
pub(super) fn fullscreen_vertex(b: &Backend) -> Option<vk::Pipeline> {
    const FULL: &[u32] = &[
        0x07230203, 0x00010600, 0x000d000b, 0x00000045, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
        0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
        0x0007000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000002a, 0x0000002d, 0x00030003,
        0x00000005, 0x000001f4, 0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45, 0x74735f70, 0x5f656c79,
        0x656e696c, 0x7269645f, 0x69746365, 0x00006576, 0x00080004, 0x475f4c47, 0x4c474f4f, 0x6e695f45,
        0x64756c63, 0x69645f65, 0x74636572, 0x00657669, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
        0x00030005, 0x0000002a, 0x00006469, 0x00070005, 0x0000002d, 0x746e6540, 0x6f507972, 0x4f746e69,
        0x75707475, 0x00000074, 0x00040047, 0x0000002a, 0x0000000b, 0x0000002a, 0x00040047, 0x0000002d,
        0x0000000b, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
        0x00000006, 0x00000020, 0x00000000, 0x00030016, 0x00000008, 0x00000020, 0x00040017, 0x00000009,
        0x00000008, 0x00000004, 0x00040017, 0x0000000e, 0x00000008, 0x00000002, 0x00040015, 0x00000012,
        0x00000020, 0x00000001, 0x0004002b, 0x00000012, 0x00000013, 0x00000001, 0x0004002b, 0x00000006,
        0x00000015, 0x00000002, 0x0004002b, 0x00000008, 0x0000001d, 0x40000000, 0x0004002b, 0x00000008,
        0x0000001f, 0x3f800000, 0x0004002b, 0x00000008, 0x00000022, 0x3f000000, 0x00040020, 0x00000029,
        0x00000001, 0x00000006, 0x0004003b, 0x00000029, 0x0000002a, 0x00000001, 0x00040020, 0x0000002c,
        0x00000003, 0x00000009, 0x0004003b, 0x0000002c, 0x0000002d, 0x00000003, 0x0005002c, 0x0000000e,
        0x00000044, 0x0000001f, 0x0000001f, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
        0x000200f8, 0x00000005, 0x0004003d, 0x00000006, 0x0000002b, 0x0000002a, 0x000500c4, 0x00000006,
        0x00000036, 0x0000002b, 0x00000013, 0x000500c7, 0x00000006, 0x00000037, 0x00000036, 0x00000015,
        0x00040070, 0x00000008, 0x00000038, 0x00000037, 0x000500c7, 0x00000006, 0x0000003a, 0x0000002b,
        0x00000015, 0x00040070, 0x00000008, 0x0000003b, 0x0000003a, 0x00050050, 0x0000000e, 0x0000003c,
        0x00000038, 0x0000003b, 0x0005008e, 0x0000000e, 0x0000003e, 0x0000003c, 0x0000001d, 0x00050083,
        0x0000000e, 0x00000040, 0x0000003e, 0x00000044, 0x00050051, 0x00000008, 0x00000041, 0x00000040,
        0x00000000, 0x00050051, 0x00000008, 0x00000042, 0x00000040, 0x00000001, 0x00070050, 0x00000009,
        0x00000043, 0x00000041, 0x00000042, 0x00000022, 0x0000001f, 0x0003003e, 0x0000002d, 0x00000043,
        0x000100fd, 0x00010038
    ];
    static PART: std::sync::OnceLock<Option<vk::Pipeline>> = std::sync::OnceLock::new();
    *PART.get_or_init(|| {
        let layout = stages::layout(b)?;
        let d = &b.host.device;
        // SAFETY: valid create infos; the module is kept with the part.
        unsafe {
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(FULL), None).ok()?;
            let states = stages::pre_raster_dynamic(b);
            let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
            let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS);
            let mut rendering = vk::PipelineRenderingCreateInfo::default();
            let viewport = vk::PipelineViewportStateCreateInfo::default();
            let raster = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).line_width(1.0);
            let stage = [vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(module).name(c"main")];
            let info = vk::GraphicsPipelineCreateInfo::default()
                .flags(library_flags(b))
                .stages(&stage)
                .layout(layout.pipeline)
                .dynamic_state(&dyn_info)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .push_next(&mut library)
                .push_next(&mut rendering);
            d.create_graphics_pipelines(stages::cache(b), &[info], None).ok().map(|p| p[0])
        }
    })
}

