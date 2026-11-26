//! M2b step (d): the title's display planes drawn into the swapchain image
//! (`docs/vulkan-host.md`, "Presentation"), as `crate::compose` does on
//! the D3D12 path: each plane's rectangle scaled (linear filter) to its
//! place, the first as it is, later ones over it with premultiplied alpha
//! (one, one minus source alpha).
//!
//! The pass reads the plane through the global descriptor buffers (its
//! image view in a set-0 slot, sampler slot 0, linear) and takes its
//! rectangles in push constants. Its two programs are HLSL, compiled
//! offline with `glslc -x hlsl --target-env=vulkan1.3` (entry points
//! `corner`, vertex, and `texel`, fragment) and embedded as words:
//!
//! ```hlsl
//! struct Push { float4 shown; float4 place; uint slot; uint pad0; uint pad1; uint pad2; };
//! [[vk::push_constant]] ConstantBuffer<Push> pc;
//! [[vk::binding(0, 0)]] Texture2D<float4> images[1048576];
//! [[vk::binding(0, 1)]] SamplerState samplers[4096];
//! struct Corner { float4 position : SV_Position; float2 uv : TEXCOORD0; };
//! Corner corner(uint i : SV_VertexID) {
//!     float2 t = float2(float(i & 1), float(i >> 1)) * 2.0;
//!     Corner c;
//!     float2 at = lerp(pc.place.xy, pc.place.zw, t);
//!     c.position = float4(at * 2.0 - 1.0, 0.0, 1.0);
//!     c.uv = lerp(pc.shown.xy, pc.shown.zw, t);
//!     return c;
//! }
//! float4 texel(Corner c) : SV_Target {
//!     return images[pc.slot].SampleLevel(samplers[0], c.uv, 0.0);
//! }
//! ```

use std::sync::Mutex;

use victory_vk::ash::vk;

use super::images::{self, ViewKey};
use super::{Backend, lock};
use crate::object::This;

const VERTEX: &[u32] = &[
    0x07230203, 0x00010600, 0x000d000b, 0x00000090, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0009000f, 0x00000000, 0x00000004, 0x6e726f63, 0x00007265, 0x00000021, 0x00000047, 0x0000004e,
    0x00000052, 0x00030003, 0x00000005, 0x000001f4, 0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45,
    0x74735f70, 0x5f656c79, 0x656e696c, 0x7269645f, 0x69746365, 0x00006576, 0x00080004, 0x475f4c47,
    0x4c474f4f, 0x6e695f45, 0x64756c63, 0x69645f65, 0x74636572, 0x00657669, 0x00040005, 0x00000004,
    0x6e726f63, 0x00007265, 0x00030005, 0x0000001f, 0x00006370, 0x00050006, 0x0000001f, 0x00000000,
    0x776f6873, 0x0000006e, 0x00050006, 0x0000001f, 0x00000001, 0x63616c70, 0x00000065, 0x00050006,
    0x0000001f, 0x00000002, 0x746f6c73, 0x00000000, 0x00050006, 0x0000001f, 0x00000003, 0x30646170,
    0x00000000, 0x00050006, 0x0000001f, 0x00000004, 0x31646170, 0x00000000, 0x00050006, 0x0000001f,
    0x00000005, 0x32646170, 0x00000000, 0x00030005, 0x00000021, 0x00006370, 0x00030005, 0x00000047,
    0x00000069, 0x00090005, 0x0000004e, 0x746e6540, 0x6f507972, 0x4f746e69, 0x75707475, 0x6f702e74,
    0x69746973, 0x00006e6f, 0x00080005, 0x00000052, 0x746e6540, 0x6f507972, 0x4f746e69, 0x75707475,
    0x76752e74, 0x00000000, 0x00030047, 0x0000001f, 0x00000002, 0x00050048, 0x0000001f, 0x00000000,
    0x00000023, 0x00000000, 0x00050048, 0x0000001f, 0x00000001, 0x00000023, 0x00000010, 0x00050048,
    0x0000001f, 0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x0000001f, 0x00000003, 0x00000023,
    0x00000024, 0x00050048, 0x0000001f, 0x00000004, 0x00000023, 0x00000028, 0x00050048, 0x0000001f,
    0x00000005, 0x00000023, 0x0000002c, 0x00040047, 0x00000047, 0x0000000b, 0x0000002a, 0x00040047,
    0x0000004e, 0x0000000b, 0x00000000, 0x00040047, 0x00000052, 0x0000001e, 0x00000000, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000,
    0x00030016, 0x00000008, 0x00000020, 0x00040017, 0x00000009, 0x00000008, 0x00000004, 0x00040017,
    0x0000000a, 0x00000008, 0x00000002, 0x0004002b, 0x00000006, 0x00000013, 0x00000001, 0x00040015,
    0x00000017, 0x00000020, 0x00000001, 0x0004002b, 0x00000017, 0x00000018, 0x00000001, 0x0004002b,
    0x00000008, 0x0000001c, 0x40000000, 0x0008001e, 0x0000001f, 0x00000009, 0x00000009, 0x00000006,
    0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x00000020, 0x00000009, 0x0000001f, 0x0004003b,
    0x00000020, 0x00000021, 0x00000009, 0x00040020, 0x00000022, 0x00000009, 0x00000009, 0x0004002b,
    0x00000017, 0x0000002d, 0x00000000, 0x0004002b, 0x00000008, 0x00000030, 0x3f800000, 0x0004002b,
    0x00000008, 0x00000033, 0x00000000, 0x00040020, 0x00000046, 0x00000001, 0x00000006, 0x0004003b,
    0x00000046, 0x00000047, 0x00000001, 0x00040020, 0x0000004d, 0x00000003, 0x00000009, 0x0004003b,
    0x0000004d, 0x0000004e, 0x00000003, 0x00040020, 0x00000051, 0x00000003, 0x0000000a, 0x0004003b,
    0x00000051, 0x00000052, 0x00000003, 0x0005002c, 0x0000000a, 0x0000008f, 0x00000030, 0x00000030,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
    0x00000006, 0x00000048, 0x00000047, 0x000500c7, 0x00000006, 0x00000065, 0x00000048, 0x00000013,
    0x00040070, 0x00000008, 0x00000066, 0x00000065, 0x000500c2, 0x00000006, 0x00000068, 0x00000048,
    0x00000018, 0x00040070, 0x00000008, 0x00000069, 0x00000068, 0x00050050, 0x0000000a, 0x0000006a,
    0x00000066, 0x00000069, 0x0005008e, 0x0000000a, 0x0000006b, 0x0000006a, 0x0000001c, 0x00050041,
    0x00000022, 0x0000006c, 0x00000021, 0x00000018, 0x0004003d, 0x00000009, 0x0000006d, 0x0000006c,
    0x0007004f, 0x0000000a, 0x0000006e, 0x0000006d, 0x0000006d, 0x00000000, 0x00000001, 0x00050041,
    0x00000022, 0x0000006f, 0x00000021, 0x00000018, 0x0004003d, 0x00000009, 0x00000070, 0x0000006f,
    0x0007004f, 0x0000000a, 0x00000071, 0x00000070, 0x00000070, 0x00000002, 0x00000003, 0x0008000c,
    0x0000000a, 0x00000073, 0x00000001, 0x0000002e, 0x0000006e, 0x00000071, 0x0000006b, 0x0005008e,
    0x0000000a, 0x00000075, 0x00000073, 0x0000001c, 0x00050083, 0x0000000a, 0x00000077, 0x00000075,
    0x0000008f, 0x00050051, 0x00000008, 0x00000078, 0x00000077, 0x00000000, 0x00050051, 0x00000008,
    0x00000079, 0x00000077, 0x00000001, 0x00070050, 0x00000009, 0x0000007a, 0x00000078, 0x00000079,
    0x00000033, 0x00000030, 0x00050041, 0x00000022, 0x0000007c, 0x00000021, 0x0000002d, 0x0004003d,
    0x00000009, 0x0000007d, 0x0000007c, 0x0007004f, 0x0000000a, 0x0000007e, 0x0000007d, 0x0000007d,
    0x00000000, 0x00000001, 0x00050041, 0x00000022, 0x0000007f, 0x00000021, 0x0000002d, 0x0004003d,
    0x00000009, 0x00000080, 0x0000007f, 0x0007004f, 0x0000000a, 0x00000081, 0x00000080, 0x00000080,
    0x00000002, 0x00000003, 0x0008000c, 0x0000000a, 0x00000083, 0x00000001, 0x0000002e, 0x0000007e,
    0x00000081, 0x0000006b, 0x0003003e, 0x0000004e, 0x0000007a, 0x0003003e, 0x00000052, 0x00000083,
    0x000100fd, 0x00010038,
];

const FRAGMENT: &[u32] = &[
    0x07230203, 0x00010600, 0x000d000b, 0x0000004e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000a000f, 0x00000004, 0x00000004, 0x65786574, 0x0000006c, 0x00000014, 0x00000017, 0x00000024,
    0x0000003a, 0x0000003e, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000005, 0x000001f4,
    0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45, 0x74735f70, 0x5f656c79, 0x656e696c, 0x7269645f,
    0x69746365, 0x00006576, 0x00080004, 0x475f4c47, 0x4c474f4f, 0x6e695f45, 0x64756c63, 0x69645f65,
    0x74636572, 0x00657669, 0x00040005, 0x00000004, 0x65786574, 0x0000006c, 0x00040005, 0x00000014,
    0x67616d69, 0x00007365, 0x00030005, 0x00000015, 0x00006370, 0x00050006, 0x00000015, 0x00000000,
    0x776f6873, 0x0000006e, 0x00050006, 0x00000015, 0x00000001, 0x63616c70, 0x00000065, 0x00050006,
    0x00000015, 0x00000002, 0x746f6c73, 0x00000000, 0x00050006, 0x00000015, 0x00000003, 0x30646170,
    0x00000000, 0x00050006, 0x00000015, 0x00000004, 0x31646170, 0x00000000, 0x00050006, 0x00000015,
    0x00000005, 0x32646170, 0x00000000, 0x00030005, 0x00000017, 0x00006370, 0x00050005, 0x00000024,
    0x706d6173, 0x7372656c, 0x00000000, 0x00040005, 0x0000003a, 0x76752e63, 0x00000000, 0x00070005,
    0x0000003e, 0x746e6540, 0x6f507972, 0x4f746e69, 0x75707475, 0x00000074, 0x00040047, 0x00000014,
    0x00000021, 0x00000000, 0x00040047, 0x00000014, 0x00000022, 0x00000000, 0x00030047, 0x00000015,
    0x00000002, 0x00050048, 0x00000015, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000015,
    0x00000001, 0x00000023, 0x00000010, 0x00050048, 0x00000015, 0x00000002, 0x00000023, 0x00000020,
    0x00050048, 0x00000015, 0x00000003, 0x00000023, 0x00000024, 0x00050048, 0x00000015, 0x00000004,
    0x00000023, 0x00000028, 0x00050048, 0x00000015, 0x00000005, 0x00000023, 0x0000002c, 0x00040047,
    0x00000024, 0x00000021, 0x00000000, 0x00040047, 0x00000024, 0x00000022, 0x00000001, 0x00040047,
    0x0000003a, 0x0000001e, 0x00000000, 0x00040047, 0x0000003e, 0x0000001e, 0x00000000, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
    0x00000007, 0x00000006, 0x00000004, 0x00040017, 0x00000008, 0x00000006, 0x00000002, 0x00090019,
    0x0000000f, 0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000,
    0x00040015, 0x00000010, 0x00000020, 0x00000000, 0x0004002b, 0x00000010, 0x00000011, 0x00100000,
    0x0004001c, 0x00000012, 0x0000000f, 0x00000011, 0x00040020, 0x00000013, 0x00000000, 0x00000012,
    0x0004003b, 0x00000013, 0x00000014, 0x00000000, 0x0008001e, 0x00000015, 0x00000007, 0x00000007,
    0x00000010, 0x00000010, 0x00000010, 0x00000010, 0x00040020, 0x00000016, 0x00000009, 0x00000015,
    0x0004003b, 0x00000016, 0x00000017, 0x00000009, 0x00040015, 0x00000018, 0x00000020, 0x00000001,
    0x0004002b, 0x00000018, 0x00000019, 0x00000002, 0x00040020, 0x0000001a, 0x00000009, 0x00000010,
    0x00040020, 0x0000001d, 0x00000000, 0x0000000f, 0x0002001a, 0x00000020, 0x0004002b, 0x00000010,
    0x00000021, 0x00001000, 0x0004001c, 0x00000022, 0x00000020, 0x00000021, 0x00040020, 0x00000023,
    0x00000000, 0x00000022, 0x0004003b, 0x00000023, 0x00000024, 0x00000000, 0x0004002b, 0x00000018,
    0x00000025, 0x00000000, 0x00040020, 0x00000026, 0x00000000, 0x00000020, 0x0003001b, 0x00000029,
    0x0000000f, 0x0004002b, 0x00000006, 0x0000002f, 0x00000000, 0x00040020, 0x00000039, 0x00000001,
    0x00000008, 0x0004003b, 0x00000039, 0x0000003a, 0x00000001, 0x00040020, 0x0000003d, 0x00000003,
    0x00000007, 0x0004003b, 0x0000003d, 0x0000003e, 0x00000003, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000008, 0x0000003b, 0x0000003a,
    0x00050041, 0x0000001a, 0x00000044, 0x00000017, 0x00000019, 0x0004003d, 0x00000010, 0x00000045,
    0x00000044, 0x00050041, 0x0000001d, 0x00000046, 0x00000014, 0x00000045, 0x0004003d, 0x0000000f,
    0x00000047, 0x00000046, 0x00050041, 0x00000026, 0x00000048, 0x00000024, 0x00000025, 0x0004003d,
    0x00000020, 0x00000049, 0x00000048, 0x00050056, 0x00000029, 0x0000004a, 0x00000047, 0x00000049,
    0x00070058, 0x00000007, 0x0000004d, 0x0000004a, 0x0000003b, 0x00000002, 0x0000002f, 0x0003003e,
    0x0000003e, 0x0000004d, 0x000100fd, 0x00010038,
];

/// A plane the title shows: its texture, the rectangle of it shown, and
/// where it goes (fractions of the screen).
pub(crate) struct Shown {
    pub(crate) resource: This,
    pub(crate) source: Option<[i32; 4]>,
    pub(crate) placement: Option<[f32; 4]>,
}

/// The pipelines per swapchain format: the first plane, and planes over it.
static PIPELINES: Mutex<Vec<(vk::Format, [vk::Pipeline; 2])>> = Mutex::new(Vec::new());

fn pipelines(b: &Backend, format: vk::Format) -> Option<[vk::Pipeline; 2]> {
    let mut list = lock(&PIPELINES);
    if let Some((_, p)) = list.iter().find(|(f, _)| *f == format) {
        return Some(*p);
    }
    let layout = super::stages::layout(b)?;
    let d = &b.host.device;
    // SAFETY: valid create infos; the modules go once the pipelines exist.
    unsafe {
        let vs = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(VERTEX), None).ok()?;
        let fs = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(FRAGMENT), None).ok()?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"corner"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"texel"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let samples = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
        let formats = [format];
        let mut made = [vk::Pipeline::null(); 2];
        for (i, over) in [false, true].into_iter().enumerate() {
            let attachment = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(over)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachment);
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
            let flags = if b.host.caps.descriptor_buffer {
                vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT
            } else {
                vk::PipelineCreateFlags::empty()
            };
            let info = vk::GraphicsPipelineCreateInfo::default()
                .flags(flags)
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&samples)
                .depth_stencil_state(&depth)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout.pipeline)
                .push_next(&mut rendering);
            made[i] = d.create_graphics_pipelines(super::stages::cache(b), &[info], None).ok()?[0];
        }
        d.destroy_shader_module(vs, None);
        d.destroy_shader_module(fs, None);
        list.push((format, made));
        Some(made)
    }
}

/// Records the planes into the swapchain image `target` (in
/// `TRANSFER_DST_OPTIMAL` before and after); whether any plane was drawn.
///
/// # Safety
/// Recording into `cmd` (the frame's command buffer), outside rendering.
pub(crate) unsafe fn record(b: &'static Backend, cmd: vk::CommandBuffer, target: &victory_vk::swapchain::Target, planes: &[Shown]) -> bool {
    let Some(heaps) = super::descriptors::heaps(b) else { return false };
    let Some([first, over]) = pipelines(b, target.format) else { return false };
    let mut drawn = Vec::new();
    for p in planes {
        let Some(image) = images::of_resource(b, p.resource) else { continue };
        if image.depth() {
            continue;
        }
        let key = ViewKey {
            format: image.format,
            kind: vk::ImageViewType::TYPE_2D,
            aspect: vk::ImageAspectFlags::COLOR,
            base_mip: 0,
            mips: 1,
            base_layer: 0,
            layers: 1,
            components: [vk::ComponentSwizzle::IDENTITY; 4],
            storage: false,
            attachment: false,
        };
        let Some(slot) = heaps.plane_slot(b, &image, key) else { continue };
        let (w, h) = (image.extent.width as f32, image.extent.height as f32);
        let shown = match p.source {
            Some([l, t, r, bt]) if 0 <= l && l < r && r as f32 <= w && 0 <= t && t < bt && bt as f32 <= h => {
                [l as f32 / w, t as f32 / h, r as f32 / w, bt as f32 / h]
            }
            _ => [0.0, 0.0, 1.0, 1.0],
        };
        let place = match p.placement {
            Some([l, t, r, bt]) if (0.0..1.0).contains(&l) && l < r && r <= 1.0 && (0.0..1.0).contains(&t) && t < bt && bt <= 1.0 => [l, t, r, bt],
            _ => [0.0, 0.0, 1.0, 1.0],
        };
        drawn.push((slot, shown, place, image));
    }
    if drawn.is_empty() {
        return false;
    }
    let d = &b.host.device;
    let range = victory_vk::swapchain::color_range();
    let (generation, block) = super::map_block(b, heaps.zero_sink());
    let map = heaps.map_offset(b, generation, &block);
    // SAFETY: as the caller says.
    unsafe {
        images::to_general(d, cmd, target.image, range, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        heaps.bind(b, cmd, map);
        let attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(target.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: target.extent };
        d.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&attachment));
        let (tw, th) = (target.extent.width as f32, target.extent.height as f32);
        d.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: tw, height: th, min_depth: 0.0, max_depth: 1.0 }]);
        let layout = super::stages::layout(b).map(|l| l.pipeline).unwrap_or_default();
        for (i, (slot, shown, place, _)) in drawn.iter().enumerate() {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, if i == 0 { first } else { over });
            let x0 = (place[0] * tw) as i32;
            let y0 = (place[1] * th) as i32;
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: x0, y: y0 },
                extent: vk::Extent2D { width: ((place[2] * tw) as i32 - x0).max(1) as u32, height: ((place[3] * th) as i32 - y0).max(1) as u32 },
            };
            d.cmd_set_scissor(cmd, 0, &[scissor]);
            let mut push = [0u32; 12];
            for k in 0..4 {
                push[k] = shown[k].to_bits();
                push[4 + k] = place[k].to_bits();
            }
            push[8] = *slot;
            let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_le_bytes()).collect();
            d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::ALL, 0, &bytes);
            d.cmd_draw(cmd, 3, 1, 0, 0);
        }
        d.cmd_end_rendering(cmd);
        let barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(target.image)
            .subresource_range(range)];
        d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&barrier));
    }
    true
}
