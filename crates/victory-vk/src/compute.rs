//! The probe's compute program and the pipeline around it
//! (`docs/vulkan-host.md`, "Pipelines").
//!
//! The program reads title memory through a device-address pointer, writes
//! it back changed through another, and writes a copy and a sum through a
//! storage buffer bound from a descriptor buffer (or the fallback set): the
//! three ways translated programs reach memory. Push constants carry the
//! pointers, as they will carry the title's user data.
//!
//! Built from this GLSL with `glslc --target-env=vulkan1.3 -O` (SPIR-V 1.6,
//! checked by `spirv-val --target-env vulkan1.3`):
//!
//! ```glsl
//! #version 460
//! #extension GL_EXT_buffer_reference : require
//! #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
//! layout(local_size_x = 64) in;
//! layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint w[]; };
//! layout(push_constant, std430) uniform Push { Words src; Words dst; uint count; uint salt; } pc;
//! layout(set = 0, binding = 0, std430) buffer Result { uint sum; uint words[]; } result;
//! void main() {
//!     uint i = gl_GlobalInvocationID.x;
//!     if (i >= pc.count) return;
//!     uint v = pc.src.w[i];
//!     pc.dst.w[i] = v ^ pc.salt;
//!     result.words[i] = v + 1u;
//!     atomicAdd(result.sum, v);
//! }
//! ```

use anyhow::{Context, Result};
use ash::vk;

use crate::host::Host;

/// The program's push constants.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Push {
    pub src: u64,
    pub dst: u64,
    pub count: u32,
    pub salt: u32,
}

/// Threads per group.
pub const GROUP: u32 = 64;

/// The SPIR-V words.
#[rustfmt::skip]
pub const SPIRV: &[u32] = &[
    0x07230203, 0x00010600, 0x000d000b, 0x00000045, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x0008000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00000016, 0x00000037, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001,
    0x00040047, 0x0000000b, 0x0000000b, 0x0000001c, 0x00030047, 0x00000012, 0x00000002, 0x00050048,
    0x00000012, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000012, 0x00000001, 0x00000023,
    0x00000008, 0x00050048, 0x00000012, 0x00000002, 0x00000023, 0x00000010, 0x00050048, 0x00000012,
    0x00000003, 0x00000023, 0x00000014, 0x00040047, 0x00000013, 0x00000006, 0x00000004, 0x00030047,
    0x00000014, 0x00000002, 0x00050048, 0x00000014, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x00000034, 0x00000006, 0x00000004, 0x00030047, 0x00000035, 0x00000002, 0x00050048, 0x00000035,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000035, 0x00000001, 0x00000023, 0x00000004,
    0x00040047, 0x00000037, 0x00000021, 0x00000000, 0x00040047, 0x00000037, 0x00000022, 0x00000000,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001,
    0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c,
    0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x00030027, 0x00000011, 0x000014e5,
    0x0006001e, 0x00000012, 0x00000011, 0x00000011, 0x00000006, 0x00000006, 0x0003001d, 0x00000013,
    0x00000006, 0x0003001e, 0x00000014, 0x00000013, 0x00040020, 0x00000011, 0x000014e5, 0x00000014,
    0x00040020, 0x00000015, 0x00000009, 0x00000012, 0x0004003b, 0x00000015, 0x00000016, 0x00000009,
    0x00040015, 0x00000017, 0x00000020, 0x00000001, 0x0004002b, 0x00000017, 0x00000018, 0x00000002,
    0x00040020, 0x00000019, 0x00000009, 0x00000006, 0x00020014, 0x0000001c, 0x0004002b, 0x00000017,
    0x00000022, 0x00000000, 0x00040020, 0x00000023, 0x00000009, 0x00000011, 0x00040020, 0x00000027,
    0x000014e5, 0x00000006, 0x0004002b, 0x00000017, 0x0000002a, 0x00000001, 0x0004002b, 0x00000017,
    0x0000002f, 0x00000003, 0x0003001d, 0x00000034, 0x00000006, 0x0004001e, 0x00000035, 0x00000006,
    0x00000034, 0x00040020, 0x00000036, 0x0000000c, 0x00000035, 0x0004003b, 0x00000036, 0x00000037,
    0x0000000c, 0x0004002b, 0x00000006, 0x0000003a, 0x00000001, 0x00040020, 0x0000003c, 0x0000000c,
    0x00000006, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x000300f7, 0x00000043, 0x00000000, 0x000300fb, 0x0000000c, 0x00000044, 0x000200f8, 0x00000044,
    0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f,
    0x0000000e, 0x00050041, 0x00000019, 0x0000001a, 0x00000016, 0x00000018, 0x0004003d, 0x00000006,
    0x0000001b, 0x0000001a, 0x000500ae, 0x0000001c, 0x0000001d, 0x0000000f, 0x0000001b, 0x000300f7,
    0x0000001f, 0x00000000, 0x000400fa, 0x0000001d, 0x0000001e, 0x0000001f, 0x000200f8, 0x0000001e,
    0x000200f9, 0x00000043, 0x000200f8, 0x0000001f, 0x00050041, 0x00000023, 0x00000024, 0x00000016,
    0x00000022, 0x0004003d, 0x00000011, 0x00000025, 0x00000024, 0x00060041, 0x00000027, 0x00000028,
    0x00000025, 0x00000022, 0x0000000f, 0x0006003d, 0x00000006, 0x00000029, 0x00000028, 0x00000002,
    0x00000004, 0x00050041, 0x00000023, 0x0000002b, 0x00000016, 0x0000002a, 0x0004003d, 0x00000011,
    0x0000002c, 0x0000002b, 0x00050041, 0x00000019, 0x00000030, 0x00000016, 0x0000002f, 0x0004003d,
    0x00000006, 0x00000031, 0x00000030, 0x000500c6, 0x00000006, 0x00000032, 0x00000029, 0x00000031,
    0x00060041, 0x00000027, 0x00000033, 0x0000002c, 0x00000022, 0x0000000f, 0x0005003e, 0x00000033,
    0x00000032, 0x00000002, 0x00000004, 0x00050080, 0x00000006, 0x0000003b, 0x00000029, 0x0000003a,
    0x00060041, 0x0000003c, 0x0000003d, 0x00000037, 0x0000002a, 0x0000000f, 0x0003003e, 0x0000003d,
    0x0000003b, 0x00050041, 0x0000003c, 0x0000003e, 0x00000037, 0x00000022, 0x000700ea, 0x00000006,
    0x00000040, 0x0000003e, 0x0000003a, 0x0000000c, 0x00000029, 0x000200f9, 0x00000043, 0x000200f8,
    0x00000043, 0x000100fd, 0x00010038,
];

/// The compute pipeline and what it was made with.
pub struct Probe {
    pub module: vk::ShaderModule,
    pub layout: vk::PipelineLayout,
    pub cache: vk::PipelineCache,
    pub pipeline: vk::Pipeline,
    /// Milliseconds the driver took to make the pipeline.
    pub ms: f64,
}

impl Probe {
    /// Makes the pipeline over `set_layout`; `in_memory` when that layout
    /// is for a descriptor buffer. The cache starts empty (the backend
    /// proper loads it from the run's cache folder).
    pub fn new(host: &Host, set_layout: vk::DescriptorSetLayout, in_memory: bool) -> Result<Probe> {
        let device = &host.device;
        // SAFETY: valid create infos; everything is destroyed by `destroy`.
        unsafe {
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(SPIRV), None)
                .context("vkCreateShaderModule")?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size_of::<Push>() as u32)];
            let sets = [set_layout];
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&sets)
                        .push_constant_ranges(&ranges),
                    None,
                )
                .context("vkCreatePipelineLayout")?;
            let cache = device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                .context("vkCreatePipelineCache")?;
            let flags = if in_memory {
                vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT
            } else {
                vk::PipelineCreateFlags::empty()
            };
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = [vk::ComputePipelineCreateInfo::default()
                .flags(flags)
                .stage(stage)
                .layout(layout)];
            let start = std::time::Instant::now();
            let pipeline = device
                .create_compute_pipelines(cache, &info, None)
                .map_err(|(_, e)| e)
                .context("vkCreateComputePipelines")?[0];
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(Probe {
                module,
                layout,
                cache,
                pipeline,
                ms,
            })
        }
    }

    /// Bytes the pipeline cache holds now.
    pub fn cache_bytes(&self, host: &Host) -> usize {
        // SAFETY: the cache is alive.
        unsafe { host.device.get_pipeline_cache_data(self.cache) }.map_or(0, |d| d.len())
    }

    /// # Safety
    /// No submitted work still uses the pipeline.
    pub unsafe fn destroy(&self, host: &Host) {
        unsafe {
            host.device.destroy_pipeline(self.pipeline, None);
            host.device.destroy_pipeline_cache(self.cache, None);
            host.device.destroy_pipeline_layout(self.layout, None);
            host.device.destroy_shader_module(self.module, None);
        }
    }
}
