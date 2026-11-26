//! Graphics pipelines without per-state compiles (`docs/vulkan-host.md`,
//! "Pipelines"): each stage compiled once, a draw links them.
//!
//! Two ways, as the driver offers them:
//!
//! - **Pipeline libraries** (`VK_EXT_graphics_pipeline_library`): four
//!   parts (vertex input, pre-rasterisation, fragment shader, fragment
//!   output) compiled apart; a fast link at the first draw, an optimised
//!   link later in the background. Keeps `VkPipelineCache`.
//! - **Shader objects** (`VK_EXT_shader_object`): each stage a
//!   `VkShaderEXT`, no link at all, every state set per command buffer.
//!   No `VkPipelineCache`; binaries come from `vkGetShaderBinaryDataEXT`.
//!
//! The probe compiles a triangle both ways, times every part, and draws
//! with each into a 64x64 target whose centre pixel it reads back.
//!
//! The programs (`glslc --target-env=vulkan1.3 -O`, `spirv-val` clean):
//!
//! ```glsl
//! // vertex
//! #version 460
//! layout(location = 0) out vec4 color;
//! void main() {
//!     vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//!     gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
//!     color = vec4(0.25, 0.5, 0.75, 1.0);
//! }
//! // fragment
//! #version 460
//! layout(location = 0) in vec4 color;
//! layout(location = 0) out vec4 target;
//! void main() { target = color; }
//! ```

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use ash::vk;

use crate::host::Host;
use crate::memory::{Buffer, Place};

#[rustfmt::skip]
pub const VERTEX: &[u32] = &[
    0x07230203, 0x00010600, 0x000d000b, 0x00000031, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000c, 0x0000001d, 0x0000002b,
    0x00040047, 0x0000000c, 0x0000000b, 0x0000002a, 0x00030047, 0x0000001b, 0x00000002, 0x00050048,
    0x0000001b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000001b, 0x00000001, 0x0000000b,
    0x00000001, 0x00050048, 0x0000001b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000001b,
    0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x0000002b, 0x0000001e, 0x00000000, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
    0x00000007, 0x00000006, 0x00000002, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x00040020,
    0x0000000b, 0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004002b,
    0x0000000a, 0x0000000e, 0x00000001, 0x0004002b, 0x0000000a, 0x00000010, 0x00000002, 0x00040017,
    0x00000017, 0x00000006, 0x00000004, 0x00040015, 0x00000018, 0x00000020, 0x00000000, 0x0004002b,
    0x00000018, 0x00000019, 0x00000001, 0x0004001c, 0x0000001a, 0x00000006, 0x00000019, 0x0006001e,
    0x0000001b, 0x00000017, 0x00000006, 0x0000001a, 0x0000001a, 0x00040020, 0x0000001c, 0x00000003,
    0x0000001b, 0x0004003b, 0x0000001c, 0x0000001d, 0x00000003, 0x0004002b, 0x0000000a, 0x0000001e,
    0x00000000, 0x0004002b, 0x00000006, 0x00000020, 0x40000000, 0x0004002b, 0x00000006, 0x00000022,
    0x3f800000, 0x0004002b, 0x00000006, 0x00000025, 0x00000000, 0x00040020, 0x00000029, 0x00000003,
    0x00000017, 0x0004003b, 0x00000029, 0x0000002b, 0x00000003, 0x0004002b, 0x00000006, 0x0000002c,
    0x3e800000, 0x0004002b, 0x00000006, 0x0000002d, 0x3f000000, 0x0004002b, 0x00000006, 0x0000002e,
    0x3f400000, 0x0007002c, 0x00000017, 0x0000002f, 0x0000002c, 0x0000002d, 0x0000002e, 0x00000022,
    0x0005002c, 0x00000007, 0x00000030, 0x00000022, 0x00000022, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c,
    0x000500c4, 0x0000000a, 0x0000000f, 0x0000000d, 0x0000000e, 0x000500c7, 0x0000000a, 0x00000011,
    0x0000000f, 0x00000010, 0x0004006f, 0x00000006, 0x00000012, 0x00000011, 0x000500c7, 0x0000000a,
    0x00000014, 0x0000000d, 0x00000010, 0x0004006f, 0x00000006, 0x00000015, 0x00000014, 0x00050050,
    0x00000007, 0x00000016, 0x00000012, 0x00000015, 0x0005008e, 0x00000007, 0x00000021, 0x00000016,
    0x00000020, 0x00050083, 0x00000007, 0x00000024, 0x00000021, 0x00000030, 0x00050051, 0x00000006,
    0x00000026, 0x00000024, 0x00000000, 0x00050051, 0x00000006, 0x00000027, 0x00000024, 0x00000001,
    0x00070050, 0x00000017, 0x00000028, 0x00000026, 0x00000027, 0x00000025, 0x00000022, 0x00050041,
    0x00000029, 0x0000002a, 0x0000001d, 0x0000001e, 0x0003003e, 0x0000002a, 0x00000028, 0x0003003e,
    0x0000002b, 0x0000002f, 0x000100fd, 0x00010038,
];

#[rustfmt::skip]
pub const FRAGMENT: &[u32] = &[
    0x07230203, 0x00010600, 0x000d000b, 0x0000000d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00030010,
    0x00000004, 0x00000007, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000b,
    0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
    0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040020, 0x0000000a,
    0x00000001, 0x00000007, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000007, 0x0000000c,
    0x0000000b, 0x0003003e, 0x00000009, 0x0000000c, 0x000100fd, 0x00010038,
];

/// The target's side and format.
pub const SIZE: u32 = 64;
pub const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// The colour the programs draw, as stored.
pub const EXPECT: [u8; 4] = [64, 128, 191, 255];

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// The dynamic states a part of a pipeline takes: everything the title
/// varies per draw, so that no state change ever costs a compile.
fn dynamic(host: &Host, part: vk::GraphicsPipelineLibraryFlagsEXT) -> Vec<vk::DynamicState> {
    use vk::DynamicState as D;
    use vk::GraphicsPipelineLibraryFlagsEXT as P;
    let c = &host.caps;
    let mut out = Vec::new();
    if part.contains(P::VERTEX_INPUT_INTERFACE) {
        out.extend([D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE]);
    }
    if part.contains(P::PRE_RASTERIZATION_SHADERS) {
        out.extend([
            D::VIEWPORT_WITH_COUNT,
            D::SCISSOR_WITH_COUNT,
            D::CULL_MODE,
            D::FRONT_FACE,
            D::DEPTH_BIAS,
            D::DEPTH_BIAS_ENABLE,
            D::RASTERIZER_DISCARD_ENABLE,
        ]);
        if c.dynamic_raster {
            out.extend([D::POLYGON_MODE_EXT, D::DEPTH_CLAMP_ENABLE_EXT]);
        }
    }
    if part.contains(P::FRAGMENT_SHADER) {
        out.extend([
            D::DEPTH_TEST_ENABLE,
            D::DEPTH_WRITE_ENABLE,
            D::DEPTH_COMPARE_OP,
            D::STENCIL_TEST_ENABLE,
            D::STENCIL_OP,
            D::STENCIL_COMPARE_MASK,
            D::STENCIL_WRITE_MASK,
            D::STENCIL_REFERENCE,
        ]);
    }
    if part.contains(P::FRAGMENT_OUTPUT_INTERFACE) {
        out.push(D::BLEND_CONSTANTS);
        if c.dynamic_blend {
            out.extend([
                D::COLOR_BLEND_ENABLE_EXT,
                D::COLOR_BLEND_EQUATION_EXT,
                D::COLOR_WRITE_MASK_EXT,
            ]);
        }
        if c.dynamic_raster {
            out.extend([
                D::RASTERIZATION_SAMPLES_EXT,
                D::SAMPLE_MASK_EXT,
                D::ALPHA_TO_COVERAGE_ENABLE_EXT,
            ]);
        }
    }
    out
}

/// Sets the dynamic states of [`dynamic`] for every part.
///
/// # Safety
/// `cmd` is recording inside a rendering pass of one colour target.
unsafe fn set_pipeline_state(host: &Host, cmd: vk::CommandBuffer) {
    let d = &host.device;
    let c = &host.caps;
    unsafe {
        d.cmd_set_primitive_topology(cmd, vk::PrimitiveTopology::TRIANGLE_LIST);
        d.cmd_set_primitive_restart_enable(cmd, false);
        d.cmd_set_viewport_with_count(cmd, &[viewport()]);
        d.cmd_set_scissor_with_count(cmd, &[scissor()]);
        d.cmd_set_cull_mode(cmd, vk::CullModeFlags::NONE);
        d.cmd_set_front_face(cmd, vk::FrontFace::COUNTER_CLOCKWISE);
        d.cmd_set_depth_bias(cmd, 0.0, 0.0, 0.0);
        d.cmd_set_depth_bias_enable(cmd, false);
        d.cmd_set_rasterizer_discard_enable(cmd, false);
        d.cmd_set_depth_test_enable(cmd, false);
        d.cmd_set_depth_write_enable(cmd, false);
        d.cmd_set_depth_compare_op(cmd, vk::CompareOp::ALWAYS);
        d.cmd_set_stencil_test_enable(cmd, false);
        let both = vk::StencilFaceFlags::FRONT_AND_BACK;
        d.cmd_set_stencil_op(
            cmd,
            both,
            vk::StencilOp::KEEP,
            vk::StencilOp::KEEP,
            vk::StencilOp::KEEP,
            vk::CompareOp::ALWAYS,
        );
        d.cmd_set_stencil_compare_mask(cmd, both, 0xff);
        d.cmd_set_stencil_write_mask(cmd, both, 0xff);
        d.cmd_set_stencil_reference(cmd, both, 0);
        d.cmd_set_blend_constants(cmd, &[0.0; 4]);
        if let Some(o) = &host.shader_objects {
            // EDS3 states share their entry points with shader objects.
            if c.dynamic_raster {
                o.cmd_set_polygon_mode(cmd, vk::PolygonMode::FILL);
                o.cmd_set_depth_clamp_enable(cmd, false);
                o.cmd_set_rasterization_samples(cmd, vk::SampleCountFlags::TYPE_1);
                o.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[!0]);
                o.cmd_set_alpha_to_coverage_enable(cmd, false);
            }
            if c.dynamic_blend {
                set_blend(o, cmd);
            }
        } else if c.dynamic_raster || c.dynamic_blend {
            let eds3 = ash::ext::extended_dynamic_state3::Device::new(&host.instance, d);
            if c.dynamic_raster {
                eds3.cmd_set_polygon_mode(cmd, vk::PolygonMode::FILL);
                eds3.cmd_set_depth_clamp_enable(cmd, false);
                eds3.cmd_set_rasterization_samples(cmd, vk::SampleCountFlags::TYPE_1);
                eds3.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[!0]);
                eds3.cmd_set_alpha_to_coverage_enable(cmd, false);
            }
            if c.dynamic_blend {
                eds3.cmd_set_color_blend_enable(cmd, 0, &[vk::FALSE]);
                eds3.cmd_set_color_blend_equation(cmd, 0, &[blend_equation()]);
                eds3.cmd_set_color_write_mask(cmd, 0, &[vk::ColorComponentFlags::RGBA]);
            }
        }
    }
}

fn blend_equation() -> vk::ColorBlendEquationEXT {
    vk::ColorBlendEquationEXT::default()
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
}

unsafe fn set_blend(o: &ash::ext::shader_object::Device, cmd: vk::CommandBuffer) {
    unsafe {
        o.cmd_set_color_blend_enable(cmd, 0, &[vk::FALSE]);
        o.cmd_set_color_blend_equation(cmd, 0, &[blend_equation()]);
        o.cmd_set_color_write_mask(cmd, 0, &[vk::ColorComponentFlags::RGBA]);
    }
}

fn viewport() -> vk::Viewport {
    vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: SIZE as f32,
        height: SIZE as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

fn scissor() -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent: vk::Extent2D {
            width: SIZE,
            height: SIZE,
        },
    }
}

/// Milliseconds each part took.
#[derive(Debug, Default)]
pub struct LibraryTimes {
    pub vertex_input: f64,
    pub pre_raster: f64,
    pub fragment: f64,
    pub output: f64,
    pub fast_link: f64,
    pub optimised_link: f64,
    pub monolithic: f64,
}

/// Pipelines made from libraries, and one made whole for comparison.
pub struct Libraries {
    pub parts: Vec<vk::Pipeline>,
    pub fast: vk::Pipeline,
    pub optimised: vk::Pipeline,
    pub monolithic: vk::Pipeline,
    pub modules: [vk::ShaderModule; 2],
    pub times: LibraryTimes,
}

impl Libraries {
    pub fn new(
        host: &Host,
        layout: vk::PipelineLayout,
        cache: vk::PipelineCache,
    ) -> Result<Libraries> {
        ensure!(
            host.caps.pipeline_library,
            "no VK_EXT_graphics_pipeline_library"
        );
        use vk::GraphicsPipelineLibraryFlagsEXT as P;
        let d = &host.device;
        // SAFETY: valid create infos; every object is destroyed by `destroy`.
        unsafe {
            let vs =
                d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(VERTEX), None)?;
            let fs = d.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(FRAGMENT),
                None,
            )?;
            let formats = [FORMAT];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default();
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0);
            let depth = vk::PipelineDepthStencilStateCreateInfo::default();
            let samples = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let attachments = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
            let stage = |stage, module| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage)
                    .module(module)
                    .name(c"main")
            };
            let vs_stage = [stage(vk::ShaderStageFlags::VERTEX, vs)];
            let fs_stage = [stage(vk::ShaderStageFlags::FRAGMENT, fs)];
            let library_flags = vk::PipelineCreateFlags::LIBRARY_KHR
                | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT;

            let part = |flags: P| -> Result<(vk::Pipeline, f64)> {
                let states = dynamic(host, flags);
                let dyn_info =
                    vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
                let mut rendering =
                    vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
                let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(flags);
                let mut info = vk::GraphicsPipelineCreateInfo::default()
                    .flags(library_flags)
                    .dynamic_state(&dyn_info)
                    .push_next(&mut library);
                if flags == P::VERTEX_INPUT_INTERFACE {
                    info = info
                        .vertex_input_state(&vertex_input)
                        .input_assembly_state(&assembly);
                } else {
                    info = info.layout(layout).push_next(&mut rendering);
                }
                if flags == P::PRE_RASTERIZATION_SHADERS {
                    info = info
                        .stages(&vs_stage)
                        .viewport_state(&viewport)
                        .rasterization_state(&raster);
                }
                if flags == P::FRAGMENT_SHADER {
                    info = info
                        .stages(&fs_stage)
                        .depth_stencil_state(&depth)
                        .multisample_state(&samples);
                }
                if flags == P::FRAGMENT_OUTPUT_INTERFACE {
                    info = info.color_blend_state(&blend).multisample_state(&samples);
                }
                let start = Instant::now();
                let p = d
                    .create_graphics_pipelines(cache, &[info], None)
                    .map_err(|(_, e)| e)
                    .with_context(|| format!("library {flags:?}"))?[0];
                Ok((p, ms(start)))
            };
            let mut times = LibraryTimes::default();
            let (vi, t) = part(P::VERTEX_INPUT_INTERFACE)?;
            times.vertex_input = t;
            let (pr, t) = part(P::PRE_RASTERIZATION_SHADERS)?;
            times.pre_raster = t;
            let (fr, t) = part(P::FRAGMENT_SHADER)?;
            times.fragment = t;
            let (out, t) = part(P::FRAGMENT_OUTPUT_INTERFACE)?;
            times.output = t;
            let parts = vec![vi, pr, fr, out];

            let link = |flags: vk::PipelineCreateFlags| -> Result<(vk::Pipeline, f64)> {
                let mut libraries = vk::PipelineLibraryCreateInfoKHR::default().libraries(&parts);
                let info = vk::GraphicsPipelineCreateInfo::default()
                    .flags(flags)
                    .layout(layout)
                    .push_next(&mut libraries);
                let start = Instant::now();
                let p = d
                    .create_graphics_pipelines(cache, &[info], None)
                    .map_err(|(_, e)| e)
                    .context("link")?[0];
                Ok((p, ms(start)))
            };
            let (fast, t) = link(vk::PipelineCreateFlags::empty())?;
            times.fast_link = t;
            let (optimised, t) = link(vk::PipelineCreateFlags::LINK_TIME_OPTIMIZATION_EXT)?;
            times.optimised_link = t;

            // The same pipeline made whole, as vkd3d makes every one.
            let all = P::VERTEX_INPUT_INTERFACE
                | P::PRE_RASTERIZATION_SHADERS
                | P::FRAGMENT_SHADER
                | P::FRAGMENT_OUTPUT_INTERFACE;
            let states = dynamic(host, all);
            let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
            let mut rendering =
                vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
            let stages = [vs_stage[0], fs_stage[0]];
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&samples)
                .depth_stencil_state(&depth)
                .color_blend_state(&blend)
                .dynamic_state(&dyn_info)
                .layout(layout)
                .push_next(&mut rendering);
            let start = Instant::now();
            let monolithic = d
                .create_graphics_pipelines(cache, &[info], None)
                .map_err(|(_, e)| e)
                .context("whole pipeline")?[0];
            times.monolithic = ms(start);
            Ok(Libraries {
                parts,
                fast,
                optimised,
                monolithic,
                modules: [vs, fs],
                times,
            })
        }
    }

    /// # Safety
    /// No submitted work still uses the pipelines.
    pub unsafe fn destroy(&self, host: &Host) {
        let d = &host.device;
        unsafe {
            for &p in self
                .parts
                .iter()
                .chain([&self.fast, &self.optimised, &self.monolithic])
            {
                d.destroy_pipeline(p, None);
            }
            for m in self.modules {
                d.destroy_shader_module(m, None);
            }
        }
    }
}

/// Records a draw with a pipeline.
///
/// # Safety
/// `cmd` is recording inside [`Target::draw`]'s pass.
pub unsafe fn draw_pipeline(host: &Host, cmd: vk::CommandBuffer, pipeline: vk::Pipeline) {
    unsafe {
        host.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        set_pipeline_state(host, cmd);
        host.device.cmd_draw(cmd, 3, 1, 0, 0);
    }
}

/// The two stages as shader objects.
pub struct Objects {
    pub shaders: Vec<vk::ShaderEXT>,
    pub ms: f64,
    pub binary_bytes: usize,
}

impl Objects {
    pub fn new(host: &Host) -> Result<Objects> {
        let Some(o) = &host.shader_objects else {
            bail!("no VK_EXT_shader_object");
        };
        let bytes = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (vs, fs) = (bytes(VERTEX), bytes(FRAGMENT));
        let infos = [
            vk::ShaderCreateInfoEXT::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .next_stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(&vs)
                .name(c"main"),
            vk::ShaderCreateInfoEXT::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(&fs)
                .name(c"main"),
        ];
        let start = Instant::now();
        // SAFETY: valid create infos; destroyed by `destroy`.
        let shaders = unsafe { o.create_shaders(&infos, None) }.map_err(|(made, e)| {
            for s in made {
                if s != vk::ShaderEXT::null() {
                    unsafe { o.destroy_shader(s, None) };
                }
            }
            anyhow::anyhow!("vkCreateShadersEXT: {e}")
        })?;
        let ms = ms(start);
        let binary_bytes = shaders
            .iter()
            .map(|&s| unsafe { o.get_shader_binary_data(s) }.map_or(0, |b| b.len()))
            .sum();
        Ok(Objects {
            shaders,
            ms,
            binary_bytes,
        })
    }

    /// Records a draw with the shader objects and every state they need.
    ///
    /// # Safety
    /// `cmd` is recording inside [`Target::draw`]'s pass.
    pub unsafe fn draw(&self, host: &Host, cmd: vk::CommandBuffer) {
        let o = host.shader_objects.as_ref().expect("shader objects");
        let c = &host.caps;
        let core = &c.core;
        let mut stages = vec![vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT];
        let mut shaders = self.shaders.clone();
        // Every stage the device has enabled is bound, the unused ones to
        // nothing.
        let mut unused = Vec::new();
        if core.tessellation_shader == vk::TRUE {
            unused.extend([
                vk::ShaderStageFlags::TESSELLATION_CONTROL,
                vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            ]);
        }
        if core.geometry_shader == vk::TRUE {
            unused.push(vk::ShaderStageFlags::GEOMETRY);
        }
        if c.mesh_shader {
            unused.extend([
                vk::ShaderStageFlags::TASK_EXT,
                vk::ShaderStageFlags::MESH_EXT,
            ]);
        }
        for s in unused {
            stages.push(s);
            shaders.push(vk::ShaderEXT::null());
        }
        unsafe {
            o.cmd_bind_shaders(cmd, &stages, &shaders);
            set_pipeline_state(host, cmd);
            o.cmd_set_vertex_input(cmd, &[], &[]);
            if !c.dynamic_raster {
                // Shader objects make these dynamic whatever EDS3 says.
                o.cmd_set_polygon_mode(cmd, vk::PolygonMode::FILL);
                o.cmd_set_rasterization_samples(cmd, vk::SampleCountFlags::TYPE_1);
                o.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[!0]);
                o.cmd_set_alpha_to_coverage_enable(cmd, false);
            }
            if !c.dynamic_blend {
                set_blend(o, cmd);
            }
            if core.depth_clamp == vk::TRUE {
                o.cmd_set_depth_clamp_enable(cmd, false);
            }
            if core.alpha_to_one == vk::TRUE {
                o.cmd_set_alpha_to_one_enable(cmd, false);
            }
            if core.logic_op == vk::TRUE {
                o.cmd_set_logic_op_enable(cmd, false);
            }
            if core.depth_bounds == vk::TRUE {
                host.device.cmd_set_depth_bounds_test_enable(cmd, false);
            }
            if c.has(ash::ext::depth_clip_enable::NAME) {
                o.cmd_set_depth_clip_enable(cmd, true);
            }
            if c.has(ash::ext::conservative_rasterization::NAME) {
                o.cmd_set_conservative_rasterization_mode(
                    cmd,
                    vk::ConservativeRasterizationModeEXT::DISABLED,
                );
            }
            host.device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    /// # Safety
    /// No submitted work still uses the shaders.
    pub unsafe fn destroy(&self, host: &Host) {
        if let Some(o) = &host.shader_objects {
            for &s in &self.shaders {
                unsafe { o.destroy_shader(s, None) };
            }
        }
    }
}

/// A 64x64 colour target, its view and a read-back buffer.
pub struct Target {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub readback: Buffer,
    pub pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl Target {
    pub fn new(host: &Host) -> Result<Target> {
        let d = &host.device;
        // SAFETY: valid create infos; destroyed by `destroy`.
        unsafe {
            let image = d.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(FORMAT)
                    .extent(vk::Extent3D {
                        width: SIZE,
                        height: SIZE,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    ),
                None,
            )?;
            let needs = d.get_image_memory_requirements(image);
            let memory_type = host
                .memory_type(
                    needs.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                )
                .context("no memory type for the target")?;
            let memory = d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(memory_type),
                None,
            )?;
            d.bind_image_memory(image, memory, 0)?;
            let view = d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(FORMAT)
                    .subresource_range(range()),
                None,
            )?;
            let readback = Buffer::new(
                host,
                u64::from(SIZE * SIZE * 4),
                vk::BufferUsageFlags::TRANSFER_DST,
                Place::Readback,
            )?;
            let pool = d.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(host.queues.graphics.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            Ok(Target {
                image,
                memory,
                view,
                readback,
                pool,
                cmd,
                fence,
            })
        }
    }

    /// Clears the target to black, runs `record` inside a dynamic
    /// rendering pass, and returns the centre pixel.
    pub fn draw(&self, host: &Host, record: impl FnOnce(vk::CommandBuffer)) -> Result<[u8; 4]> {
        let d = &host.device;
        let cmd = self.cmd;
        // SAFETY: the command buffer is idle (the last run's fence was
        // waited for); every handle is alive.
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let to_target = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.image)
                .subresource_range(range())];
            d.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_target),
            );
            let attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(self.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                })];
            d.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(scissor())
                    .layer_count(1)
                    .color_attachments(&attachments),
            );
            record(cmd);
            d.cmd_end_rendering(cmd);
            let to_copy = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(self.image)
                .subresource_range(range())];
            d.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_copy),
            );
            d.cmd_copy_image_to_buffer(
                cmd,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback.buffer,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: SIZE,
                        height: SIZE,
                        depth: 1,
                    })],
            );
            let to_host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            d.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
            d.end_command_buffer(cmd)?;
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            d.queue_submit2(
                host.queues.graphics.queue,
                &[vk::SubmitInfo2::default().command_buffer_infos(&cmds)],
                self.fence,
            )?;
            d.wait_for_fences(&[self.fence], true, 5_000_000_000)?;
            d.reset_fences(&[self.fence])?;
            let at = ((SIZE / 2) * SIZE + SIZE / 2) as usize * 4;
            let px = std::slice::from_raw_parts(self.readback.mapped.add(at), 4);
            Ok([px[0], px[1], px[2], px[3]])
        }
    }

    /// # Safety
    /// No submitted work still uses the target.
    pub unsafe fn destroy(&self, host: &Host) {
        let d = &host.device;
        unsafe {
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            self.readback.destroy(host);
            d.destroy_image_view(self.view, None);
            d.destroy_image(self.image, None);
            d.free_memory(self.memory, None);
        }
    }
}

fn range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

/// Whether a read-back pixel is the colour the programs draw.
pub fn right(px: [u8; 4]) -> bool {
    px.iter().zip(EXPECT).all(|(&a, b)| a.abs_diff(b) <= 1)
}
