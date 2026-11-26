//! M2b step (c): the title's list calls recorded into Vulkan command
//! buffers (`docs/vulkan-host.md`, "Command lists").
//!
//! The title's threads write their list calls down (`list/thin.rs`); the
//! list workers replay each stream here, into one command buffer per part
//! of the list (parts are cut at memory-value waits, `vulkan.rs`).
//!
//! - **Pipelines.** A draw links its stages' library parts (`stages`) with a
//!   vertex-input part and a fragment-output part (the bound targets'
//!   formats) by a fast link, cached per tuple. A draw whose stage is not
//!   compiled yet is skipped and counted (the recording never waits).
//! - **User data** goes into push constants (ABI section 2) at every draw.
//! - **Descriptor buffers** and the memory map are bound once per command
//!   buffer.
//! - **Dynamic rendering** begins at the first draw after the targets
//!   change and ends at anything that is not a draw. Every image lives in
//!   `GENERAL`; a title barrier is a full memory barrier.
//! - **Not yet** (counted in the 10 s line): indirect execution, query
//!   resolves, clears of rectangles (the whole view is cleared), MSAA
//!   draws.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use victory_vk::ash::vk;
use victory_vk::timeline::Recording;
use windows::Win32::Graphics::Direct3D12::*;

use super::images::{self, Image, ViewKey};
use super::stages::{self, Lookup, Stage};
use super::{Backend, lock};
use crate::descriptor::RtvRecord;
use crate::draw::{IndexView, VertexView};
use crate::list::{CopyLocation, Rect};
use crate::object::This;
use victory_shader::{HwStage, RootType};

/// Counters for the 10 s line: draws recorded, dispatches, skipped (stage
/// not ready), skipped (stage failed or no pipeline), skipped (other:
/// indirect, MSAA, unknown topology), command buffers, links.
pub(crate) static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

#[derive(Default, Clone)]
struct Root {
    /// Per root parameter: its dwords (constants), or an address (lo, hi).
    slots: Vec<Vec<u32>>,
}

impl Root {
    fn slot(&mut self, slot: u32) -> &mut Vec<u32> {
        let i = slot as usize;
        if self.slots.len() <= i {
            self.slots.resize(i + 1, Vec::new());
        }
        &mut self.slots[i]
    }

    fn set_address(&mut self, slot: u32, address: u64) {
        let s = self.slot(slot);
        s.clear();
        s.extend([address as u32, (address >> 32) as u32]);
    }

    fn set_constants(&mut self, slot: u32, offset: u32, values: &[u32]) {
        let s = self.slot(slot);
        let end = offset as usize + values.len();
        if s.len() < end {
            s.resize(end, 0);
        }
        s[offset as usize..end].copy_from_slice(values);
    }

    fn word(&self, slot: u8, dword: u32) -> u32 {
        self.slots
            .get(slot as usize)
            .and_then(|s| s.get(dword as usize))
            .copied()
            .unwrap_or(0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PassKey {
    colors: [vk::ImageView; 8],
    count: u32,
    depth: vk::ImageView,
    extent: (u32, u32),
}

/// What a list records with, between its calls.
pub(crate) struct Rec {
    cmd: Option<Recording>,
    /// The statistics query of the command buffer (diagnostics).
    query: Option<u32>,
    pipeline: This,
    graphics: Root,
    compute: Root,
    heaps: [u64; 2],
    viewports: Vec<vk::Viewport>,
    scissors: Vec<vk::Rect2D>,
    topology: u32,
    stencil_ref: u32,
    blend: [f32; 4],
    depth_bounds: (f32, f32),
    index: Option<IndexView>,
    vertex: [Option<VertexView>; 16],
    colors: [Option<RtvRecord>; 8],
    color_count: u32,
    depth: Option<RtvRecord>,
    /// Rendering is open with these attachments, and their formats.
    pass: Option<(PassKey, [vk::Format; 8], vk::Format)>,
    targets_changed: bool,
    bound: vk::Pipeline,
    bound_compute: vk::Pipeline,
    /// Images the recording names (kept alive until the list is reset).
    keep: Vec<Arc<Image>>,
    /// The open pass's attachment images (draws are counted on them).
    attached: Vec<Arc<Image>>,
    /// Diagnostics: the last push constants, and the last draw's vertex
    /// attributes and buffers found.
    last_push: [u32; 64],
    last_inputs: (usize, usize),
    last_user: String,
    last_layout: String,
}

// SAFETY: a list is recorded by one worker at a time; the records hold
// title handles (plain addresses) only.
unsafe impl Send for Rec {}

impl Default for Rec {
    fn default() -> Rec {
        Rec {
            cmd: None,
            query: None,
            pipeline: std::ptr::null(),
            graphics: Root::default(),
            compute: Root::default(),
            heaps: [0; 2],
            viewports: Vec::new(),
            scissors: Vec::new(),
            topology: 4,
            stencil_ref: 0,
            blend: [0.0; 4],
            depth_bounds: (0.0, 1.0),
            index: None,
            vertex: [None; 16],
            colors: [None; 8],
            color_count: 0,
            depth: None,
            pass: None,
            targets_changed: true,
            bound: vk::Pipeline::null(),
            bound_compute: vk::Pipeline::null(),
            keep: Vec::new(),
            attached: Vec::new(),
            last_push: [0; 64],
            last_inputs: (0, 0),
            last_user: String::new(),
            last_layout: String::new(),
        }
    }
}

static RECS: Mutex<Option<HashMap<usize, Rec>>> = Mutex::new(None);

/// The list's recording state, taken out while a worker replays a chunk.
pub(crate) fn take(list: This) -> Rec {
    lock(&RECS)
        .get_or_insert_with(HashMap::new)
        .remove(&(list as usize))
        .unwrap_or_default()
}

pub(crate) fn put(list: This, rec: Rec) {
    lock(&RECS).get_or_insert_with(HashMap::new).insert(list as usize, rec);
}

/// The list is released.
pub(crate) fn forget(list: This) {
    let rec = lock(&RECS).as_mut().and_then(|m| m.remove(&(list as usize)));
    if let (Some(mut rec), Some(b)) = (rec, super::backend()) {
        rec.abandon(b);
    }
}

/// A D3D viewport as Vulkan's, flipped (negative height): the console's
/// programs put +y up, as D3D does.
fn d3d_flip(v: &D3D12_VIEWPORT) -> vk::Viewport {
    let (w, h) = (v.Width.max(1.0), v.Height.max(1.0));
    vk::Viewport {
        x: v.TopLeftX,
        y: v.TopLeftY + h,
        width: w,
        height: -h,
        min_depth: v.MinDepth.clamp(0.0, 1.0),
        max_depth: v.MaxDepth.clamp(0.0, 1.0),
    }
}

fn blend_factor(f: D3D12_BLEND) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match f.0 {
        1 => B::ZERO,
        2 => B::ONE,
        3 => B::SRC_COLOR,
        4 => B::ONE_MINUS_SRC_COLOR,
        5 => B::SRC_ALPHA,
        6 => B::ONE_MINUS_SRC_ALPHA,
        7 => B::DST_ALPHA,
        8 => B::ONE_MINUS_DST_ALPHA,
        9 => B::DST_COLOR,
        10 => B::ONE_MINUS_DST_COLOR,
        11 => B::SRC_ALPHA_SATURATE,
        14 => B::CONSTANT_COLOR,
        15 => B::ONE_MINUS_CONSTANT_COLOR,
        16 => B::SRC1_COLOR,
        17 => B::ONE_MINUS_SRC1_COLOR,
        18 => B::SRC1_ALPHA,
        19 => B::ONE_MINUS_SRC1_ALPHA,
        20 => B::CONSTANT_ALPHA,
        21 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

/// A view of a target the record names, as an attachment.
fn target_view(b: &Backend, r: &RtvRecord, depth: bool) -> Option<(Arc<Image>, vk::ImageView, vk::Format, (u32, u32))> {
    let image = images::of_resource(b, r.resource)?;
    let words = r.words();
    let format_word = words.map_or(0, |w| w[0]);
    let dim = words.map_or(if depth { 3 } else { 4 }, |w| w[1]);
    // RTV: format, dimension, then mip, (first, count) or plane; DSV:
    // format, dimension, flags, then mip, (first, count).
    let w = words.unwrap_or([0; 6]);
    let (mip, layer) = if depth {
        match dim {
            1 | 3 => (w[3], 0),
            2 | 4 => (w[3], w[4]),
            6 => (0, w[3]),
            _ => (0, 0),
        }
    } else {
        match dim {
            2 | 4 => (w[2], 0),
            3 | 5 => (w[2], w[3]),
            7 => (0, w[2]),
            8 => (w[2], w[3]),
            _ => (0, 0),
        }
    };
    let format = if image.depth() {
        image.format
    } else {
        let f = if format_word == 0 { None } else { images::color_format(format_word, image.card_written()) };
        f.filter(|f| images::format_bytes(*f) == images::format_bytes(image.format)).unwrap_or(image.format)
    };
    let mip = mip.min(image.mips - 1);
    let view = image.view(
        b,
        ViewKey {
            format,
            kind: vk::ImageViewType::TYPE_2D,
            aspect: image.aspect(),
            base_mip: mip,
            mips: 1,
            base_layer: layer,
            layers: 1,
            components: [vk::ComponentSwizzle::IDENTITY; 4],
            storage: false,
            attachment: true,
        },
    )?;
    let extent = ((image.extent.width >> mip).max(1), (image.extent.height >> mip).max(1));
    Some((image, view, format, extent))
}

/// Fast-linked pipelines by their parts and keys.
#[derive(Clone, PartialEq, Eq, Hash)]
struct LinkKey {
    pre: vk::Pipeline,
    fragment: vk::Pipeline,
    colors: [vk::Format; 8],
    count: u32,
    depth: vk::Format,
    inputs: Vec<(u32, u32, vk::Format, u32)>,
}

static LINKS: Mutex<Option<HashMap<LinkKey, vk::Pipeline>>> = Mutex::new(None);
static PARTS: Mutex<Option<HashMap<(Vec<(u32, u32, vk::Format, u32)>, [vk::Format; 8], u32, vk::Format, bool), vk::Pipeline>>> =
    Mutex::new(None);

fn library_flags(b: &Backend) -> vk::PipelineCreateFlags {
    let mut f = vk::PipelineCreateFlags::LIBRARY_KHR;
    if b.host.caps.descriptor_buffer {
        f |= vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT;
    }
    f
}

/// The vertex-input part (`inputs`: location, binding, format, offset) or
/// the fragment-output part (formats), made once each.
fn part(b: &Backend, inputs: &[(u32, u32, vk::Format, u32)], colors: [vk::Format; 8], count: u32, depth: vk::Format, output: bool) -> Option<vk::Pipeline> {
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
fn empty_fragment(b: &Backend) -> Option<vk::Pipeline> {
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

fn link(b: &Backend, key: LinkKey, vertex_input: Option<vk::Pipeline>, output: vk::Pipeline) -> Option<vk::Pipeline> {
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

fn stage_of(b: &'static Backend, shaders: &[Arc<crate::pipeline::Shader>], name: &str) -> Result<Option<Arc<Stage>>, bool> {
    let Some(s) = shaders.iter().find(|s| s.stage == name) else {
        return Ok(None);
    };
    match stages::lookup(b, s) {
        Lookup::Ready(st) => Ok(Some(st)),
        Lookup::Waiting => Err(true),
        Lookup::Failed => Err(false),
    }
}

/// The vertex attributes of a fetch-shader program (ABI section 6) from
/// the title's input layout: (location, binding | 0x100 per instance,
/// format, offset). D3D12's "append aligned" offsets follow the elements
/// before them in the same slot.
fn attributes(pre: &Stage, gs: &crate::pipeline::host::GraphicsState) -> Vec<(u32, u32, vk::Format, u32)> {
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
fn layout_text(pre: &Stage, gs: &crate::pipeline::host::GraphicsState, made: &[(u32, u32, vk::Format, u32)]) -> String {
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
fn draw_test() -> u32 {
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
fn solid_fragment(b: &Backend) -> Option<vk::Pipeline> {
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
fn fullscreen_vertex(b: &Backend) -> Option<vk::Pipeline> {
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

/// Counts a skipped draw or dispatch: not ready (`true`) or not to be.
fn skipped(waiting: bool) {
    STATS[if waiting { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
}

impl Rec {
    fn cmd(&mut self, b: &'static Backend) -> Option<vk::CommandBuffer> {
        if self.cmd.is_none() {
            let rec = b.timeline.begin(&b.host).ok()?;
            let heaps = super::descriptors::heaps(b)?;
            heaps.immediates(b);
            let (generation, block) = super::map_block(b, heaps.zero_sink());
            let map = heaps.map_offset(b, generation, &block);
            // SAFETY: recording into our own command buffer.
            unsafe { heaps.bind(b, rec.cmd, map) };
            // SAFETY: outside rendering in our new command buffer.
            self.query = unsafe { super::gpu_stats::begin(b, rec.cmd) };
            STATS[5].fetch_add(1, Ordering::Relaxed);
            self.cmd = Some(rec);
            self.bound = vk::Pipeline::null();
            self.bound_compute = vk::Pipeline::null();
            self.pass = None;
        }
        self.cmd.as_ref().map(|r| r.cmd)
    }

    fn end_pass(&mut self, b: &Backend) {
        if self.pass.take().is_some()
            && let Some(r) = &self.cmd
        {
            // SAFETY: rendering is open in our command buffer.
            unsafe { b.host.device.cmd_end_rendering(r.cmd) };
        }
    }

    /// Hands the command buffer to the list's current part (at a cut and
    /// at the close).
    pub(crate) fn cut(&mut self, list: This, b: &Backend) {
        self.end_pass(b);
        if let Some(rec) = self.cmd.take() {
            if let Some(slot) = self.query.take() {
                // SAFETY: outside rendering, before the buffer ends.
                unsafe { super::gpu_stats::end(b, rec.cmd, slot, true) };
            }
            super::part_commands(list, rec);
        }
    }

    pub(crate) fn abandon(&mut self, b: &Backend) {
        self.end_pass(b);
        if let Some(rec) = self.cmd.take() {
            if let Some(slot) = self.query.take() {
                // SAFETY: outside rendering, before the buffer ends.
                unsafe { super::gpu_stats::end(b, rec.cmd, slot, false) };
            }
            b.timeline.abandon(&b.host, rec);
        }
    }

    pub(crate) fn reset(&mut self, b: &Backend, pipeline: This) {
        self.abandon(b);
        *self = Rec {
            pipeline,
            ..Rec::default()
        };
    }

    // ---- state ----

    pub(crate) fn set_pipeline(&mut self, p: This) {
        self.pipeline = p;
        crate::pipeline::bound(p);
    }
    pub(crate) fn topology(&mut self, t: u32) {
        self.topology = t;
    }
    pub(crate) fn stencil_ref(&mut self, v: u32) {
        self.stencil_ref = v;
    }
    pub(crate) fn depth_bounds(&mut self, min: f32, max: f32) {
        self.depth_bounds = (min, max);
    }
    pub(crate) fn blend_factor(&mut self, f: Option<[f32; 4]>) {
        self.blend = f.unwrap_or([1.0; 4]);
    }
    pub(crate) fn viewports(&mut self, v: &[D3D12_VIEWPORT]) {
        self.viewports = v.iter().map(d3d_flip).collect();
    }
    pub(crate) fn scissors(&mut self, r: &[windows::Win32::Foundation::RECT]) {
        self.scissors = r
            .iter()
            .map(|r| vk::Rect2D {
                offset: vk::Offset2D { x: r.left.max(0), y: r.top.max(0) },
                extent: vk::Extent2D {
                    width: (r.right - r.left.max(0)).max(0) as u32,
                    height: (r.bottom - r.top.max(0)).max(0) as u32,
                },
            })
            .collect();
    }
    pub(crate) fn heaps(&mut self, heaps: &[This]) {
        for &h in heaps {
            if h.is_null() {
                continue;
            }
            let heap = crate::descriptor::heap(h);
            self.heaps[usize::from(heap.samplers())] = heap.start;
            super::touch(heap.start);
        }
    }
    pub(crate) fn table(&mut self, compute: bool, slot: u32, handle: u64) {
        super::touch(handle);
        self.root(compute).set_address(slot, handle);
    }
    pub(crate) fn view(&mut self, compute: bool, slot: u32, address: u64) {
        super::touch(address);
        self.root(compute).set_address(slot, address);
    }
    pub(crate) fn constants(&mut self, compute: bool, slot: u32, offset: u32, values: &[u32]) {
        self.root(compute).set_constants(slot, offset, values);
    }
    fn root(&mut self, compute: bool) -> &mut Root {
        if compute { &mut self.compute } else { &mut self.graphics }
    }
    pub(crate) fn index_buffer(&mut self, v: Option<IndexView>) {
        self.index = v;
    }
    pub(crate) fn vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = Some(*v);
            }
        }
    }
    pub(crate) fn targets(&mut self, colors: &[Option<RtvRecord>], depth: Option<RtvRecord>) {
        self.colors = [None; 8];
        for (i, c) in colors.iter().take(8).enumerate() {
            self.colors[i] = *c;
        }
        self.color_count = colors.len().min(8) as u32;
        self.depth = depth;
        self.targets_changed = true;
    }

    // ---- work ----

    pub(crate) fn barrier(&mut self, b: &'static Backend) {
        self.end_pass(b);
        if let Some(cmd) = self.cmd(b) {
            // SAFETY: outside rendering, in our command buffer.
            unsafe { images::full_barrier(&b.host.device, cmd) };
        }
    }

    pub(crate) fn clear_color(&mut self, b: &'static Backend, r: &RtvRecord, color: [f32; 4]) {
        self.end_pass(b);
        let Some((image, view, format, _)) = target_view(b, r, false) else { return };
        let _ = view;
        let Some(cmd) = self.cmd(b) else { return };
        let value = if victory_vk::gfx10::is_integer(format) {
            vk::ClearColorValue { uint32: color.map(|c| c as u32) }
        } else {
            vk::ClearColorValue { float32: color }
        };
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        // SAFETY: outside rendering; the image is in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_clear_color_image(cmd, image.image, vk::ImageLayout::GENERAL, &value, &[range]);
            images::full_barrier(&b.host.device, cmd);
        }
        image.written[1].fetch_add(1, Ordering::Relaxed);
        self.keep.push(image);
    }

    pub(crate) fn clear_depth(&mut self, b: &'static Backend, r: &RtvRecord, flags: u32, depth: f32, stencil: u8) {
        self.end_pass(b);
        let Some((image, ..)) = target_view(b, r, true) else { return };
        let Some(cmd) = self.cmd(b) else { return };
        let mut aspect = vk::ImageAspectFlags::empty();
        if flags & 1 != 0 {
            aspect |= vk::ImageAspectFlags::DEPTH;
        }
        if flags & 2 != 0 {
            aspect |= vk::ImageAspectFlags::STENCIL;
        }
        aspect &= image.aspect();
        if aspect.is_empty() {
            return;
        }
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        // SAFETY: outside rendering; the image is in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_clear_depth_stencil_image(
                cmd,
                image.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearDepthStencilValue { depth, stencil: u32::from(stencil) },
                &[range],
            );
            images::full_barrier(&b.host.device, cmd);
        }
        self.keep.push(image);
    }

    /// Opens rendering on the bound targets if it is not open on them.
    fn begin_pass(&mut self, b: &'static Backend, cmd: vk::CommandBuffer) -> Option<([vk::Format; 8], u32, vk::Format)> {
        if !self.targets_changed
            && let Some((key, formats, depth)) = &self.pass
        {
            return Some((*formats, key.count, *depth));
        }
        let mut key = PassKey {
            colors: [vk::ImageView::null(); 8],
            count: self.color_count,
            depth: vk::ImageView::null(),
            extent: (u32::MAX, u32::MAX),
        };
        let mut formats = [vk::Format::UNDEFINED; 8];
        let mut att: Vec<Arc<Image>> = Vec::new();
        let mut depth_format = vk::Format::UNDEFINED;
        for i in 0..self.color_count as usize {
            if let Some(r) = self.colors[i]
                && let Some((image, view, format, extent)) = target_view(b, &r, false)
            {
                key.colors[i] = view;
                formats[i] = format;
                key.extent = (key.extent.0.min(extent.0), key.extent.1.min(extent.1));
                att.push(image);
            }
        }
        if let Some(r) = self.depth
            && let Some((image, view, format, extent)) = target_view(b, &r, true)
        {
            key.depth = view;
            depth_format = format;
            key.extent = (key.extent.0.min(extent.0), key.extent.1.min(extent.1));
            att.push(image);
        }
        if key.extent.0 == u32::MAX {
            return None;
        }
        self.keep.extend(att.iter().cloned());
        self.attached = att;
        self.targets_changed = false;
        if let Some((open, ..)) = &self.pass
            && *open == key
        {
            return Some((formats, key.count, depth_format));
        }
        self.end_pass(b);
        let attachment = |view: vk::ImageView| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        };
        let colors: Vec<vk::RenderingAttachmentInfo> = (0..key.count as usize)
            .map(|i| if key.colors[i] == vk::ImageView::null() { vk::RenderingAttachmentInfo::default() } else { attachment(key.colors[i]) })
            .collect();
        let depth = attachment(key.depth);
        let stencil = matches!(depth_format, vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT);
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D { width: key.extent.0, height: key.extent.1 },
            })
            .layer_count(1)
            .color_attachments(&colors);
        if key.depth != vk::ImageView::null() {
            info = info.depth_attachment(&depth);
            if stencil {
                info = info.stencil_attachment(&depth);
            }
        }
        // SAFETY: outside rendering in our command buffer; views of live images.
        unsafe { b.host.device.cmd_begin_rendering(cmd, &info) };
        self.pass = Some((key, formats, depth_format));
        Some((formats, key.count, depth_format))
    }

    /// The push constants of the stages (ABI section 2).
    fn push(&mut self, b: &Backend, cmd: vk::CommandBuffer, stages: &[&Stage], compute: bool) {
        let Some(layout) = stages::layout(b) else { return };
        let mut words = [0u32; 64];
        let root = if compute { &self.compute } else { &self.graphics };
        for st in stages {
            for ud in &st.user_data {
                let i = ud.index as usize;
                if i >= 64 {
                    continue;
                }
                words[i] = match ud.root {
                    RootType::Constants | RootType::Descriptor | RootType::Table => root.word(ud.slot, ud.dword),
                    RootType::ResourceHeap => self.heaps[0] as u32,
                    RootType::SamplerHeap => self.heaps[1] as u32,
                    _ => 0,
                };
            }
        }
        self.last_push = words;
        if cfg!(feature = "diagnostics") {
            self.last_user = stages
                .iter()
                .flat_map(|s| s.user_data.iter().map(|u| format!("{}<-{}.{}{:?}", u.index, u.slot, u.dword, u.root)))
                .collect::<Vec<_>>()
                .join(" ");
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        // SAFETY: recording into our command buffer; 256 bytes of the layout's range.
        unsafe { b.host.device.cmd_push_constants(cmd, layout.pipeline, vk::ShaderStageFlags::ALL, 0, &bytes) };
    }

    /// Everything a draw needs bound; `None` (counted) if it is skipped.
    fn prepare_draw(&mut self, b: &'static Backend, mesh: bool) -> Option<vk::CommandBuffer> {
        let Some((shaders, state)) = crate::pipeline::parts(self.pipeline) else {
            skipped(false);
            return None;
        };
        let Some(gs) = state else {
            skipped(false);
            return None;
        };
        // Stages the backend does not link yet (tessellation, geometry,
        // amplification): counted.
        if shaders.iter().any(|s| matches!(s.stage, "hull" | "domain" | "geometry" | "amplification")) {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let pre = match stage_of(b, shaders, if mesh { "mesh" } else { "vertex" }) {
            Ok(Some(s)) => s,
            Ok(None) => {
                skipped(false);
                return None;
            }
            Err(waiting) => {
                skipped(waiting);
                return None;
            }
        };
        let pixel = match stage_of(b, shaders, "pixel") {
            Ok(s) => s,
            Err(waiting) => {
                skipped(waiting);
                return None;
            }
        };
        if gs.samples.Count > 1 {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let topology = match self.topology {
            1 => vk::PrimitiveTopology::POINT_LIST,
            2 => vk::PrimitiveTopology::LINE_LIST,
            3 => vk::PrimitiveTopology::LINE_STRIP,
            4 => vk::PrimitiveTopology::TRIANGLE_LIST,
            5 => vk::PrimitiveTopology::TRIANGLE_STRIP,
            _ if mesh => vk::PrimitiveTopology::TRIANGLE_LIST,
            _ => {
                STATS[4].fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        let cmd = self.cmd(b)?;
        if let Some(h) = super::descriptors::heaps(b) {
            h.immediates(b);
        }
        self.last_inputs = (0, 0);
        let (formats, count, depth) = self.begin_pass(b, cmd)?;
        let inputs = attributes(&pre, gs);
        if cfg!(feature = "diagnostics") {
            self.last_layout = layout_text(&pre, gs, &inputs);
        }
        let fragment = match &pixel {
            _ if draw_test() >= 2 => solid_fragment(b)?,
            Some(p) => p.part?,
            None => empty_fragment(b)?,
        };
        let output = part(b, &[], formats, count, depth, true)?;
        let vertex_input = if mesh { None } else { Some(part(b, &inputs, [vk::Format::UNDEFINED; 8], 0, vk::Format::UNDEFINED, false)?) };
        let key = LinkKey {
            pre: if draw_test() >= 3 && !mesh { fullscreen_vertex(b)? } else { pre.part? },
            fragment,
            colors: formats,
            count,
            depth,
            inputs: inputs.clone(),
        };
        let Some(pipeline) = link(b, key, vertex_input, output) else {
            skipped(false);
            return None;
        };
        let d = &b.host.device;
        // SAFETY: recording into our command buffer, rendering open.
        unsafe {
            if self.bound != pipeline {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                self.bound = pipeline;
            }
            let mut list: Vec<&Stage> = vec![&pre];
            if let Some(p) = &pixel {
                list.push(p);
            }
            self.push(b, cmd, &list, false);
            self.dynamic(b, cmd, gs, topology, count);
            // Each binding the attributes name, in order of first use.
            let mut order: Vec<u32> = Vec::new();
            for &(_, binding, _, _) in &inputs {
                if !order.contains(&(binding & 0xff)) {
                    order.push(binding & 0xff);
                }
            }
            let mut found = 0;
            let wanted = order.len();
            for slot in order {
                let v = self.vertex.get(slot as usize).copied().flatten();
                if let Some(v) = v
                    && let Some((buf, off)) = super::buffer_of(b, v.address)
                {
                    d.cmd_bind_vertex_buffers2(cmd, slot, &[buf], &[off], Some(&[u64::from(v.size)]), Some(&[u64::from(v.stride)]));
                    found += 1;
                }
            }
            self.last_inputs = (inputs.len() + 1000 * wanted, found);
        }
        Some(cmd)
    }

    /// Sets every dynamic state the parts declare.
    ///
    /// # Safety
    /// Recording into `cmd`.
    unsafe fn dynamic(&self, b: &'static Backend, cmd: vk::CommandBuffer, gs: &crate::pipeline::host::GraphicsState, topology: vk::PrimitiveTopology, count: u32) {
        let d = &b.host.device;
        let r = &gs.rasterizer;
        let z = &gs.depth;
        let viewports = if self.viewports.is_empty() {
            vec![vk::Viewport { x: 0.0, y: 1.0, width: 1.0, height: -1.0, min_depth: 0.0, max_depth: 1.0 }]
        } else {
            self.viewports.clone()
        };
        let scissors = if self.scissors.is_empty() {
            vec![vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: 16384, height: 16384 } }]
        } else {
            self.scissors.clone()
        };
        let n = viewports.len().min(scissors.len()).max(1);
        let stencil_op = |o: &D3D12_DEPTH_STENCILOP_DESC| {
            (
                vk::StencilOp::from_raw((o.StencilFailOp.0 - 1).max(0)),
                vk::StencilOp::from_raw((o.StencilPassOp.0 - 1).max(0)),
                vk::StencilOp::from_raw((o.StencilDepthFailOp.0 - 1).max(0)),
                vk::CompareOp::from_raw((o.StencilFunc.0 - 1).max(0)),
            )
        };
        // SAFETY: as the caller says.
        unsafe {
            d.cmd_set_viewport_with_count(cmd, &viewports[..n]);
            d.cmd_set_scissor_with_count(cmd, &scissors[..n]);
            d.cmd_set_cull_mode(
                cmd,
                match r.CullMode.0 {
                    2 => vk::CullModeFlags::FRONT,
                    3 => vk::CullModeFlags::BACK,
                    _ => vk::CullModeFlags::NONE,
                },
            );
            d.cmd_set_front_face(cmd, if r.FrontCounterClockwise.as_bool() { vk::FrontFace::COUNTER_CLOCKWISE } else { vk::FrontFace::CLOCKWISE });
            let bias = r.DepthBias != 0 || r.SlopeScaledDepthBias != 0.0;
            d.cmd_set_depth_bias_enable(cmd, bias);
            d.cmd_set_depth_bias(cmd, r.DepthBias as f32, r.DepthBiasClamp, r.SlopeScaledDepthBias);
            d.cmd_set_rasterizer_discard_enable(cmd, false);
            d.cmd_set_depth_test_enable(cmd, z.DepthEnable.as_bool());
            d.cmd_set_depth_write_enable(cmd, z.DepthEnable.as_bool() && z.DepthWriteMask.0 == 1);
            d.cmd_set_depth_compare_op(cmd, vk::CompareOp::from_raw((z.DepthFunc.0 - 1).clamp(0, 7)));
            d.cmd_set_depth_bounds_test_enable(cmd, gs.depth_bounds);
            d.cmd_set_depth_bounds(cmd, self.depth_bounds.0.clamp(0.0, 1.0), self.depth_bounds.1.clamp(0.0, 1.0));
            d.cmd_set_stencil_test_enable(cmd, z.StencilEnable.as_bool());
            for (face, o) in [(vk::StencilFaceFlags::FRONT, &z.FrontFace), (vk::StencilFaceFlags::BACK, &z.BackFace)] {
                let (fail, pass, depth_fail, compare) = stencil_op(o);
                d.cmd_set_stencil_op(cmd, face, fail, pass, depth_fail, compare);
            }
            d.cmd_set_stencil_compare_mask(cmd, vk::StencilFaceFlags::FRONT_AND_BACK, u32::from(z.StencilReadMask));
            d.cmd_set_stencil_write_mask(cmd, vk::StencilFaceFlags::FRONT_AND_BACK, u32::from(z.StencilWriteMask));
            d.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::FRONT_AND_BACK, self.stencil_ref);
            d.cmd_set_primitive_topology(cmd, topology);
            let strip = matches!(topology, vk::PrimitiveTopology::LINE_STRIP | vk::PrimitiveTopology::TRIANGLE_STRIP);
            d.cmd_set_primitive_restart_enable(cmd, strip && gs.strip_cut.0 != 0);
            d.cmd_set_blend_constants(cmd, &self.blend);
            let eds3 = super::eds3(b);
            if let Some(e) = eds3 {
                if b.host.caps.dynamic_raster {
                    e.cmd_set_polygon_mode(cmd, if r.FillMode.0 == 2 { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });
                    e.cmd_set_depth_clamp_enable(cmd, !r.DepthClipEnable.as_bool());
                    e.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[gs.sample_mask]);
                    e.cmd_set_alpha_to_coverage_enable(cmd, gs.blend.AlphaToCoverageEnable.as_bool());
                }
                if b.host.caps.dynamic_blend && count > 0 {
                    let independent = gs.blend.IndependentBlendEnable.as_bool();
                    let t = |i: usize| &gs.blend.RenderTarget[if independent { i } else { 0 }];
                    let enables: Vec<u32> = (0..count as usize).map(|i| u32::from(t(i).BlendEnable.as_bool())).collect();
                    let equations: Vec<vk::ColorBlendEquationEXT> = (0..count as usize)
                        .map(|i| {
                            let x = t(i);
                            vk::ColorBlendEquationEXT {
                                src_color_blend_factor: blend_factor(x.SrcBlend),
                                dst_color_blend_factor: blend_factor(x.DestBlend),
                                color_blend_op: vk::BlendOp::from_raw((x.BlendOp.0 - 1).clamp(0, 4)),
                                src_alpha_blend_factor: blend_factor(x.SrcBlendAlpha),
                                dst_alpha_blend_factor: blend_factor(x.DestBlendAlpha),
                                alpha_blend_op: vk::BlendOp::from_raw((x.BlendOpAlpha.0 - 1).clamp(0, 4)),
                            }
                        })
                        .collect();
                    let masks: Vec<vk::ColorComponentFlags> = (0..count as usize)
                        .map(|i| vk::ColorComponentFlags::from_raw(u32::from(t(i).RenderTargetWriteMask) & 0xf))
                        .collect();
                    let enables: Vec<vk::Bool32> = enables;
                    e.cmd_set_color_blend_enable(cmd, 0, &enables);
                    e.cmd_set_color_blend_equation(cmd, 0, &equations);
                    e.cmd_set_color_write_mask(cmd, 0, &masks);
                }
            }
            if draw_test() >= 1 {
                self.test_state(b, cmd, count);
            }
        }
    }

    /// Diagnostics: the first 48 bytes of vertex view 0.
    fn vertex_bytes(&self) -> String {
        let Some(v) = self.vertex[0] else { return String::new() };
        let Some(b) = super::backend() else { return String::new() };
        match super::read_title(b, v.address, 48) {
            Ok(bytes) => bytes.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect::<Vec<_>>().join(" "),
            Err(e) => format!("{e:#}"),
        }
    }

    /// Draw test state: nothing hidden, every channel written.
    ///
    /// # Safety
    /// Recording into `cmd`.
    unsafe fn test_state(&self, b: &'static Backend, cmd: vk::CommandBuffer, count: u32) {
        let d = &b.host.device;
        // SAFETY: as the caller says.
        unsafe {
            d.cmd_set_cull_mode(cmd, vk::CullModeFlags::NONE);
            d.cmd_set_depth_test_enable(cmd, false);
            d.cmd_set_depth_write_enable(cmd, false);
            d.cmd_set_depth_bounds_test_enable(cmd, false);
            d.cmd_set_stencil_test_enable(cmd, false);
            if let Some(e) = super::eds3(b)
                && b.host.caps.dynamic_raster
            {
                e.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[u32::MAX]);
                e.cmd_set_alpha_to_coverage_enable(cmd, false);
                e.cmd_set_polygon_mode(cmd, vk::PolygonMode::FILL);
                e.cmd_set_depth_clamp_enable(cmd, false);
            }
            if let Some(e) = super::eds3(b)
                && b.host.caps.dynamic_blend
                && count > 0
            {
                let n = count as usize;
                e.cmd_set_color_blend_enable(cmd, 0, &vec![vk::FALSE; n]);
                e.cmd_set_color_write_mask(cmd, 0, &vec![vk::ColorComponentFlags::RGBA; n]);
            }
        }
    }

    pub(crate) fn draw(&mut self, b: &'static Backend, vertices: u32, instances: u32, first: u32, first_instance: u32) {
        if let Some(cmd) = self.prepare_draw(b, false) {
            // SAFETY: rendering open, pipeline bound.
            unsafe { b.host.device.cmd_draw(cmd, vertices, instances, first, first_instance) };
            self.drawn(&format!("draw {vertices}x{instances}"));
        }
    }

    /// Counts a recorded draw, on the attachments too; diagnostics builds
    /// describe the first ones.
    fn drawn(&self, what: &str) {
        let n = STATS[0].fetch_add(1, Ordering::Relaxed);
        for i in &self.attached {
            i.written[0].fetch_add(1, Ordering::Relaxed);
        }
        // Diagnostics: the first two draws of each of the first 64 pipelines.
        static SEEN: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
        let describe = cfg!(feature = "diagnostics") && {
            let mut seen = lock(&SEEN);
            let m = seen.get_or_insert_with(HashMap::new);
            let known = m.len();
            let c = m.entry(self.pipeline as usize).or_insert(0);
            *c += 1;
            *c <= 2 && known < 64
        };
        let _ = n;
        if describe {
            let names = crate::pipeline::parts(self.pipeline)
                .map(|(s, _)| s.iter().map(|s| format!("{}:{}", s.stage, s.name)).collect::<Vec<_>>().join(" "))
                .unwrap_or_default();
            let targets: Vec<String> = self.attached.iter().map(|i| format!("{}@{:#x} {:?} {}x{}", i.id, i.key.address, i.format, i.extent.width, i.extent.height)).collect();
            let push: Vec<String> = self.last_push.iter().enumerate().filter(|(_, w)| **w != 0).map(|(k, w)| format!("{k}:{w:#x}")).collect();
            tracing::info!(
                what,
                push = ?push,
                user_data = %self.last_user,
                layout = %self.last_layout,
                vertex_bytes = %self.vertex_bytes(),
                attributes = self.last_inputs.0 % 1000,
                bindings = self.last_inputs.0 / 1000,

                bindings_found = self.last_inputs.1,
                vertex_views = ?self.vertex.iter().take(4).map(|v| v.map(|v| (format!("{:#x}", v.address), v.size, v.stride))).collect::<Vec<_>>(),
                index = ?self.index.map(|i| (format!("{:#x}", i.address), i.size, i.format)),
                pipeline = %names,
                viewports = ?self.viewports.iter().map(|v| (v.x, v.y, v.width, v.height)).collect::<Vec<_>>(),
                scissors = ?self.scissors.iter().map(|s| (s.offset.x, s.offset.y, s.extent.width, s.extent.height)).collect::<Vec<_>>(),
                topology = self.topology,
                sample_mask = format_args!("{:#x}", crate::pipeline::parts(self.pipeline).and_then(|(_, g)| g).map_or(0, |g| g.sample_mask)),
                targets = ?targets,
                "Vulkan: a draw recorded"
            );
        }
    }

    pub(crate) fn draw_indexed(&mut self, b: &'static Backend, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
        let Some(index) = self.index else {
            skipped(false);
            return;
        };
        let Some((buffer, offset)) = super::buffer_of(b, index.address) else {
            skipped(false);
            return;
        };
        if let Some(cmd) = self.prepare_draw(b, false) {
            let ty = if index.format == 57 { vk::IndexType::UINT16 } else { vk::IndexType::UINT32 };
            // SAFETY: rendering open, pipeline bound; a title buffer.
            unsafe {
                b.host.device.cmd_bind_index_buffer(cmd, buffer, offset, ty);
                b.host.device.cmd_draw_indexed(cmd, indices, instances, first, base, first_instance);
            }
            self.drawn(&format!("indexed {indices}x{instances}"));
        }
    }

    pub(crate) fn draw_mesh(&mut self, b: &'static Backend, x: u32, y: u32, z: u32) {
        let Some(mesh) = super::mesh(b) else {
            skipped(false);
            return;
        };
        if let Some(cmd) = self.prepare_draw(b, true) {
            // SAFETY: rendering open, a mesh pipeline bound.
            unsafe { mesh.cmd_draw_mesh_tasks(cmd, x.max(1), y.max(1), z.max(1)) };
            self.drawn(&format!("mesh {x}x{y}x{z}"));
        }
    }

    pub(crate) fn dispatch(&mut self, b: &'static Backend, x: u32, y: u32, z: u32) {
        let Some((shaders, _)) = crate::pipeline::parts(self.pipeline) else {
            skipped(false);
            return;
        };
        let stage = match stage_of(b, shaders, "compute") {
            Ok(Some(s)) => s,
            Ok(None) => {
                skipped(false);
                return;
            }
            Err(waiting) => {
                skipped(waiting);
                return;
            }
        };
        let Some(pipeline) = stage.part.filter(|_| stage.stage == HwStage::Cs) else {
            skipped(false);
            return;
        };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        if let Some(h) = super::descriptors::heaps(b) {
            h.immediates(b);
        }
        // SAFETY: outside rendering, in our command buffer.
        unsafe {
            if self.bound_compute != pipeline {
                b.host.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.bound_compute = pipeline;
            }
            self.push(b, cmd, &[&stage], true);
            b.host.device.cmd_dispatch(cmd, x, y, z);
        }
        STATS[1].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn not_yet(&mut self) {
        STATS[4].fetch_add(1, Ordering::Relaxed);
    }

    // ---- copies ----

    fn copy_buffer(&mut self, b: &'static Backend, dst: u64, src: u64, len: u64) {
        let (Some((db, doff)), Some((sb, soff))) = (super::buffer_of(b, dst), super::buffer_of(b, src)) else {
            return;
        };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: outside rendering; title buffers.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_buffer(cmd, sb, db, &[vk::BufferCopy { src_offset: soff, dst_offset: doff, size: len }]);
            images::full_barrier(&b.host.device, cmd);
        }
    }

    pub(crate) fn copy_memory(&mut self, b: &'static Backend, dst: u64, src: u64, len: u64) {
        self.copy_buffer(b, dst, src, len);
    }

    pub(crate) fn copy_buffer_region(&mut self, b: &'static Backend, dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64) {
        let (Some((d, _)), Some((s, _))) = (crate::resource::buffer_bytes(dst), crate::resource::buffer_bytes(src)) else {
            return;
        };
        self.copy_buffer(b, d as u64 + dst_offset, s as u64 + src_offset, len);
    }

    pub(crate) fn copy_resource(&mut self, b: &'static Backend, dst: This, src: This) {
        if let (Some((d, len)), Some((s, _))) = (crate::resource::buffer_bytes(dst), crate::resource::buffer_bytes(src)) {
            self.copy_buffer(b, d as u64, s as u64, len);
            return;
        }
        let (Some(di), Some(si)) = (images::of_resource(b, dst), images::of_resource(b, src)) else {
            return;
        };
        let mut regions = Vec::new();
        for mip in 0..di.mips.min(si.mips) {
            let ext = vk::Extent3D {
                width: (di.extent.width.min(si.extent.width) >> mip).max(1),
                height: (di.extent.height.min(si.extent.height) >> mip).max(1),
                depth: (di.extent.depth.min(si.extent.depth) >> mip).max(1),
            };
            let layers = |i: &Image| vk::ImageSubresourceLayers {
                aspect_mask: i.aspect(),
                mip_level: mip,
                base_array_layer: 0,
                layer_count: di.layers.min(si.layers),
            };
            regions.push(vk::ImageCopy::default().src_subresource(layers(&si)).dst_subresource(layers(&di)).extent(ext));
        }
        self.copy_images(b, &si, &di, &regions);
        di.written[2].fetch_add(1, Ordering::Relaxed);
        self.keep.extend([si, di]);
    }

    fn copy_images(&mut self, b: &'static Backend, src: &Image, dst: &Image, regions: &[vk::ImageCopy]) {
        if src.aspect() != dst.aspect() || regions.is_empty() {
            return;
        }
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: outside rendering; images in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_image(cmd, src.image, vk::ImageLayout::GENERAL, dst.image, vk::ImageLayout::GENERAL, regions);
            images::full_barrier(&b.host.device, cmd);
        }
    }

    pub(crate) fn copy_texture(&mut self, b: &'static Backend, dst: &CopyLocation, x: u32, y: u32, z: u32, src: &CopyLocation, boxed: Option<[u32; 6]>) {
        let sub = |l: &CopyLocation, i: &Image| {
            let index = l.footprint.offset as u32;
            (index % i.mips, index / i.mips)
        };
        let dst_image = if dst.kind == 0 { images::of_resource(b, dst.resource) } else { None };
        let src_image = if src.kind == 0 { images::of_resource(b, src.resource) } else { None };
        match (dst_image, src_image) {
            (Some(di), Some(si)) => {
                let (dm, dl) = sub(dst, &di);
                let (sm, sl) = sub(src, &si);
                let [l, t, f, r, bt, k] = boxed.unwrap_or([0, 0, 0, (si.extent.width >> sm).max(1), (si.extent.height >> sm).max(1), (si.extent.depth >> sm).max(1)]);
                let region = vk::ImageCopy::default()
                    .src_subresource(vk::ImageSubresourceLayers { aspect_mask: si.aspect(), mip_level: sm, base_array_layer: sl, layer_count: 1 })
                    .src_offset(vk::Offset3D { x: l as i32, y: t as i32, z: f as i32 })
                    .dst_subresource(vk::ImageSubresourceLayers { aspect_mask: di.aspect(), mip_level: dm, base_array_layer: dl, layer_count: 1 })
                    .dst_offset(vk::Offset3D { x: x as i32, y: y as i32, z: z as i32 })
                    .extent(vk::Extent3D { width: r.saturating_sub(l).max(1), height: bt.saturating_sub(t).max(1), depth: k.saturating_sub(f).max(1) });
                self.copy_images(b, &si, &di, &[region]);
                di.written[2].fetch_add(1, Ordering::Relaxed);
                self.keep.extend([si, di]);
            }
            (Some(di), None) if src.kind == 1 => {
                // Buffer rows (a placed footprint) into a texture.
                let Some((data, _)) = crate::resource::buffer_bytes(src.resource) else { return };
                let fp = src.footprint;
                let Some((buffer, offset)) = super::buffer_of(b, data as u64 + fp.offset) else { return };
                let (dm, dl) = sub(dst, &di);
                let (element, block) = crate::resource::element(fp.format).unwrap_or((4, 1));
                let row_texels = (u64::from(fp.row_pitch) / element * block) as u32;
                let [l, t, f, r, bt, k] = boxed.unwrap_or([0, 0, 0, fp.width, fp.height, fp.depth]);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset + u64::from(t / block as u32) * u64::from(fp.row_pitch) + u64::from(l / block as u32) * element)
                    .buffer_row_length(row_texels)
                    .buffer_image_height(fp.height.next_multiple_of(block as u32))
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: di.aspect() & !vk::ImageAspectFlags::STENCIL, mip_level: dm, base_array_layer: dl, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: z as i32 })
                    .image_extent(vk::Extent3D { width: r.saturating_sub(l).max(1), height: bt.saturating_sub(t).max(1), depth: k.saturating_sub(f).max(1) });
                self.end_pass(b);
                let Some(cmd) = self.cmd(b) else { return };
                // SAFETY: outside rendering; a title buffer and an image in GENERAL.
                unsafe {
                    images::full_barrier(&b.host.device, cmd);
                    b.host.device.cmd_copy_buffer_to_image(cmd, buffer, di.image, vk::ImageLayout::GENERAL, &[region]);
                    images::full_barrier(&b.host.device, cmd);
                }
                di.written[2].fetch_add(1, Ordering::Relaxed);
                self.keep.push(di);
            }
            (None, Some(si)) if dst.kind == 1 => {
                let Some((data, _)) = crate::resource::buffer_bytes(dst.resource) else { return };
                let fp = dst.footprint;
                let Some((buffer, offset)) = super::buffer_of(b, data as u64 + fp.offset) else { return };
                let (sm, sl) = sub(src, &si);
                let (element, block) = crate::resource::element(fp.format).unwrap_or((4, 1));
                let [l, t, f, r, bt, k] = boxed.unwrap_or([0, 0, 0, (si.extent.width >> sm).max(1), (si.extent.height >> sm).max(1), (si.extent.depth >> sm).max(1)]);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset + u64::from(y / block as u32) * u64::from(fp.row_pitch) + u64::from(x / block as u32) * element)
                    .buffer_row_length((u64::from(fp.row_pitch) / element * block) as u32)
                    .buffer_image_height(fp.height.next_multiple_of(block as u32))
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: si.aspect() & !vk::ImageAspectFlags::STENCIL, mip_level: sm, base_array_layer: sl, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: l as i32, y: t as i32, z: f as i32 })
                    .image_extent(vk::Extent3D { width: r.saturating_sub(l).max(1), height: bt.saturating_sub(t).max(1), depth: k.saturating_sub(f).max(1) });
                self.end_pass(b);
                let Some(cmd) = self.cmd(b) else { return };
                // SAFETY: outside rendering; an image in GENERAL and a title buffer.
                unsafe {
                    images::full_barrier(&b.host.device, cmd);
                    b.host.device.cmd_copy_image_to_buffer(cmd, si.image, vk::ImageLayout::GENERAL, buffer, &[region]);
                    images::full_barrier(&b.host.device, cmd);
                }
                self.keep.push(si);
            }
            _ => {}
        }
    }

    pub(crate) fn resolve(&mut self, b: &'static Backend, dst: This, dst_sub: u32, src: This, src_sub: u32) {
        let (Some(di), Some(si)) = (images::of_resource(b, dst), images::of_resource(b, src)) else { return };
        if si.samples <= 1 || di.samples != 1 {
            return;
        }
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        let layers = |i: &Image, s: u32| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: s % i.mips,
            base_array_layer: s / i.mips,
            layer_count: 1,
        };
        let region = vk::ImageResolve::default()
            .src_subresource(layers(&si, src_sub))
            .dst_subresource(layers(&di, dst_sub))
            .extent(vk::Extent3D { width: si.extent.width.min(di.extent.width), height: si.extent.height.min(di.extent.height), depth: 1 });
        // SAFETY: outside rendering; images in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_resolve_image(cmd, si.image, vk::ImageLayout::GENERAL, di.image, vk::ImageLayout::GENERAL, &[region]);
            images::full_barrier(&b.host.device, cmd);
        }
        self.keep.extend([si, di]);
    }
}

/// Clear rectangles are not taken yet (the whole view is cleared).
pub(crate) fn rects_ignored(rects: &[Rect]) {
    if !rects.is_empty() {
        STATS[4].fetch_add(1, Ordering::Relaxed);
    }
}

/// The recording's part of the 10 s line.
pub(crate) fn report() {
    let s: Vec<u64> = STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let v: Vec<u64> = super::descriptors::STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let i: Vec<u64> = images::STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    tracing::info!(
        draws = s[0],
        dispatches = s[1],
        skipped_not_ready = s[2],
        skipped_no_pipeline = s[3],
        not_yet = s[4],
        command_buffers = s[5],
        links = s[6],
        views = v[0],
        samplers = v[1],
        views_without_image = v[2],
        images = images::count(),
        images_made = i[0],
        fills = i[1],
        filled_mb = i[2] >> 20,
        images_refused = i[3],
        refills_by_fingerprint = i[4],
        gpu_vs = super::gpu_stats::SUMS[0].load(Ordering::Relaxed),
        gpu_clip_in = super::gpu_stats::SUMS[1].load(Ordering::Relaxed),
        gpu_clip_out = super::gpu_stats::SUMS[2].load(Ordering::Relaxed),
        gpu_fs = super::gpu_stats::SUMS[3].load(Ordering::Relaxed),
        gpu_counted_buffers = super::gpu_stats::SUMS[4].load(Ordering::Relaxed),
        "Vulkan: draws recorded"
    );
}
