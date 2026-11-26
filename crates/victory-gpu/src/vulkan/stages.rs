//! M2b step 1: each shader stage translated to SPIR-V once and compiled
//! once (`docs/vulkan-host.md`, "Pipelines"), off the title's threads.
//!
//! At the title's creation of a pipeline an earlier run used (the catalog,
//! as the D3D12 path queues them), each of its shader containers not seen
//! before is queued for the stage workers: [`victory_shader::vk::translate`],
//! then a `VkShaderModule`. When the translator speaks the host's ABI
//! ([`HOST_ABI`]), the stage is also compiled: a graphics-pipeline-library
//! part (pre-rasterisation for vertex programs, fragment shader for pixel
//! programs) or a compute pipeline, over the one pipeline layout of
//! `docs/vulkan-abi.md` v1. Until then (the translator's code is at ABI 0
//! while the document is at 1) only modules are made: the run counts what
//! translates, what fails and why, and what it costs.
//!
//! The D3D12 path's translation to DXIL does not run under
//! `VICTORY_GPU=vulkan` (`pipeline::make::created` hands over here).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use victory_shader::HwStage;
use victory_vk::ash::vk;

use super::Backend;
use crate::pipeline::Shader;

/// The ABI version the host binds for (`docs/vulkan-abi.md`).
pub(crate) const HOST_ABI: u32 = 1;

/// Stage workers: few and in the background (the machine's load is kept
/// low; the title's own threads come first).
const WORKERS: usize = 4;

/// Images and samplers the global descriptor sets hold (ABI v1).
const IMAGE_SLOTS: u32 = 1 << 20;
const SAMPLER_SLOTS: u32 = 4096;
/// Push constants: one 256-byte block for every stage (ABI v1).
const PUSH_BYTES: u32 = 256;

/// Where translated programs read the memory map (`docs/vulkan-abi.md`
/// section 3): a uniform buffer at set [`MAP_SET`] binding 0, one small
/// block per map version, bound once per command buffer. The one place
/// that says how the map is reached: a change of the ABI (a push-constant
/// pointer instead) is an edit here and in the binding of the block.
/// (Read through the push-constant pointer, NVIDIA compiled the largest
/// program in ~9-10 s against 0.26 s from a uniform buffer.)
pub(crate) const MAP_SET: usize = 3;

fn map_binding() -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::ALL)
}

/// The one pipeline layout of ABI v1 (sets 0 and 1 as descriptor buffers).
#[allow(dead_code)] // the sets are bound from M2b's draws on
pub(crate) struct Layout {
    /// Set 0 images, set 1 samplers (heap and immutable, proposal 8), set
    /// 2 empty (no per-pipeline set any more), set [`MAP_SET`] the map.
    pub(crate) sets: [vk::DescriptorSetLayout; 4],
    pub(crate) pipeline: vk::PipelineLayout,
    /// Bytes of each set in its descriptor buffer.
    pub(crate) set_bytes: [u64; 4],
}

fn make_layout(b: &Backend) -> anyhow::Result<Layout> {
    let d = &b.host.device;
    let flags = if b.host.caps.descriptor_buffer {
        vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT
    } else {
        vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
    };
    let all = vk::ShaderStageFlags::ALL;
    let images = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(IMAGE_SLOTS)
            .stage_flags(all),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(IMAGE_SLOTS)
            .stage_flags(all),
    ];
    let samplers = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .descriptor_count(SAMPLER_SLOTS)
        .stage_flags(all)];
    let partly = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 2];
    // SAFETY: valid create infos; the layout lives as long as the backend.
    unsafe {
        let set = |bindings: &[vk::DescriptorSetLayoutBinding]| {
            let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&partly[..bindings.len()]);
            let mut info = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(flags)
                .bindings(bindings);
            if !b.host.caps.descriptor_buffer {
                info = info.push_next(&mut binding_flags);
            }
            d.create_descriptor_set_layout(&info, None)
        };
        let s0 = set(&images)?;
        let s1 = set(&samplers)?;
        let s2 = set(&[])?;
        let s3 = set(&[map_binding()])?;
        let mut sets = [s2; 4];
        sets[0] = s0;
        sets[1] = s1;
        sets[MAP_SET] = s3;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(all)
            .offset(0)
            .size(PUSH_BYTES)];
        let pipeline = d.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&sets)
                .push_constant_ranges(&ranges),
            None,
        )?;
        let set_bytes = match &b.host.descriptor_buffer {
            Some(db) => [
                db.get_descriptor_set_layout_size(s0),
                db.get_descriptor_set_layout_size(s1),
                db.get_descriptor_set_layout_size(s2),
                db.get_descriptor_set_layout_size(s3),
            ],
            None => [0; 4],
        };
        Ok(Layout {
            sets,
            pipeline,
            set_bytes,
        })
    }
}

/// The ABI v1 layout, made once (`None` if the driver refuses it).
pub(crate) fn layout(b: &Backend) -> Option<&'static Layout> {
    static LAYOUT: OnceLock<Option<Layout>> = OnceLock::new();
    LAYOUT
        .get_or_init(|| match make_layout(b) {
            Ok(l) => {
                tracing::info!(
                    images = IMAGE_SLOTS,
                    samplers = SAMPLER_SLOTS,
                    image_set_mb = l.set_bytes[0] >> 20,
                    sampler_set_kb = l.set_bytes[1] >> 10,
                    descriptor_buffer = b.host.caps.descriptor_buffer,
                    "Vulkan: the ABI v1 pipeline layout (set 0 images, set 1 samplers, set 3 the memory map, 256 bytes of push constants)"
                );
                Some(l)
            }
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "Vulkan: the ABI v1 pipeline layout was refused");
                None
            }
        })
        .as_ref()
}

/// What became of a stage.
#[allow(dead_code)] // the stages are linked from M2b's draws on
enum State {
    Queued,
    Done(Stage),
    Failed,
}

#[allow(dead_code)] // kept for M2b's draws
struct Stage {
    stage: HwStage,
    module: vk::ShaderModule,
    /// The compiled part (library part or compute pipeline), once the
    /// translator speaks the host's ABI.
    part: Option<vk::Pipeline>,
    words: usize,
}

struct Queue {
    states: HashMap<[u8; 32], State>,
    todo: VecDeque<Arc<Shader>>,
}

static QUEUE: Mutex<Option<Queue>> = Mutex::new(None);
static WORK: Condvar = Condvar::new();

fn queue() -> std::sync::MutexGuard<'static, Option<Queue>> {
    QUEUE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Totals since the start: translated, failed, unsupported, modules,
/// parts, translate µs, module µs, part µs, SPIR-V words.
static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
/// Failure reasons and how often.
static REASONS: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());

/// The title created a pipeline with `shaders`: those not seen go to the
/// stage workers.
pub(crate) fn created(b: &'static Backend, shaders: &[Arc<Shader>]) {
    let mut q = queue();
    let q = q.get_or_insert_with(|| {
        start(b);
        Queue {
            states: HashMap::new(),
            todo: VecDeque::new(),
        }
    });
    for s in shaders {
        if let std::collections::hash_map::Entry::Vacant(e) = q.states.entry(s.full) {
            e.insert(State::Queued);
            q.todo.push_back(s.clone());
        }
    }
    WORK.notify_all();
}

fn start(b: &'static Backend) {
    for i in 0..WORKERS {
        let _ = std::thread::Builder::new()
            .name(format!("victory-vk-stage-{i}"))
            .spawn(move || worker(b));
    }
}

fn worker(b: &'static Backend) {
    loop {
        let shader = {
            let mut q = queue();
            loop {
                if let Some(s) = q.as_mut().and_then(|q| q.todo.pop_front()) {
                    break s;
                }
                q = WORK.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        let state = make(b, &shader);
        if let Some(q) = queue().as_mut() {
            q.states.insert(shader.full, state);
        }
    }
}

fn fail(why: String) -> State {
    STATS[1].fetch_add(1, Ordering::Relaxed);
    let mut r = REASONS.lock().unwrap_or_else(|e| e.into_inner());
    match r.iter_mut().find(|(w, _)| *w == why) {
        Some((_, n)) => *n += 1,
        None => r.push((why, 1)),
    }
    State::Failed
}

fn make(b: &Backend, shader: &Shader) -> State {
    let t = Instant::now();
    let translated = std::panic::catch_unwind(|| victory_shader::vk::translate(shader.container()));
    STATS[5].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    let vs = match translated {
        Ok(Ok(vs)) => vs,
        Ok(Err(e)) => {
            if matches!(e, victory_shader::ShaderError::Unsupported { .. }) {
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            let why = match &e {
                victory_shader::ShaderError::Unsupported { what, .. } => {
                    format!("unsupported: {what}")
                }
                other => other.to_string(),
            };
            return fail(why);
        }
        Err(_) => return fail("the translator panicked".into()),
    };
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[8].fetch_add(vs.spirv.len() as u64, Ordering::Relaxed);
    let t = Instant::now();
    // SAFETY: the translator's words, a SPIR-V module.
    let module = unsafe {
        b.host
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs.spirv), None)
    };
    STATS[6].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    let module = match module {
        Ok(m) => m,
        Err(e) => return fail(format!("vkCreateShaderModule: {e}")),
    };
    STATS[3].fetch_add(1, Ordering::Relaxed);
    #[allow(clippy::absurd_extreme_comparisons)]
    let speaks = victory_shader::vk::ABI_VERSION >= HOST_ABI;
    let part = if speaks {
        let t = Instant::now();
        let p = compile(b, vs.stage, module);
        STATS[7].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        match p {
            Ok(p) => {
                STATS[4].fetch_add(1, Ordering::Relaxed);
                Some(p)
            }
            Err(e) => {
                fail(format!("stage compile: {e:#}"));
                None
            }
        }
    } else {
        None
    };
    State::Done(Stage {
        stage: vs.stage,
        module,
        part,
        words: vs.spirv.len(),
    })
}

/// Dynamic states of the pre-rasterisation and fragment parts
/// (`victory_vk::graphics` uses the same lists for the probe).
fn dynamic(b: &Backend, fragment: bool) -> Vec<vk::DynamicState> {
    use vk::DynamicState as D;
    let mut out = if fragment {
        vec![
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
        ]
    } else {
        vec![
            D::VIEWPORT_WITH_COUNT,
            D::SCISSOR_WITH_COUNT,
            D::CULL_MODE,
            D::FRONT_FACE,
            D::DEPTH_BIAS,
            D::DEPTH_BIAS_ENABLE,
            D::RASTERIZER_DISCARD_ENABLE,
        ]
    };
    if !fragment && b.host.caps.dynamic_raster {
        out.extend([D::POLYGON_MODE_EXT, D::DEPTH_CLAMP_ENABLE_EXT]);
    }
    out
}

/// Compiles one stage: a library part (vertex: pre-rasterisation; pixel:
/// fragment shader) or a compute pipeline.
fn compile(b: &Backend, stage: HwStage, module: vk::ShaderModule) -> anyhow::Result<vk::Pipeline> {
    let layout = layout(b).ok_or_else(|| anyhow::anyhow!("no pipeline layout"))?;
    let d = &b.host.device;
    let db = if b.host.caps.descriptor_buffer {
        vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT
    } else {
        vk::PipelineCreateFlags::empty()
    };
    let name = c"main";
    // SAFETY: valid create infos over a live module and layout.
    unsafe {
        if stage == HwStage::Cs {
            let info = vk::ComputePipelineCreateInfo::default()
                .flags(db)
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(name),
                )
                .layout(layout.pipeline);
            return d
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map(|p| p[0])
                .map_err(|(_, e)| anyhow::anyhow!("vkCreateComputePipelines: {e}"));
        }
        anyhow::ensure!(
            b.host.caps.pipeline_library,
            "no graphics pipeline libraries"
        );
        let (flags, vk_stage, fragment) = match stage {
            HwStage::Vs => (
                vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS,
                vk::ShaderStageFlags::VERTEX,
                false,
            ),
            HwStage::Ps => (
                vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER,
                vk::ShaderStageFlags::FRAGMENT,
                true,
            ),
            other => anyhow::bail!("no library part for a {other:?} program"),
        };
        let states = dynamic(b, fragment);
        let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
        let stages = [vk::PipelineShaderStageCreateInfo::default()
            .stage(vk_stage)
            .module(module)
            .name(name)];
        let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(flags);
        let mut rendering = vk::PipelineRenderingCreateInfo::default();
        let viewport = vk::PipelineViewportStateCreateInfo::default();
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let samples = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .flags(
                db | vk::PipelineCreateFlags::LIBRARY_KHR
                    | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT,
            )
            .stages(&stages)
            .layout(layout.pipeline)
            .dynamic_state(&dyn_info)
            .push_next(&mut library)
            .push_next(&mut rendering);
        info = if fragment {
            info.depth_stencil_state(&depth).multisample_state(&samples)
        } else {
            info.viewport_state(&viewport).rasterization_state(&raster)
        };
        d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map(|p| p[0])
            .map_err(|(_, e)| anyhow::anyhow!("vkCreateGraphicsPipelines (library part): {e}"))
    }
}

/// The stage workers' part of the 10 s line.
pub(crate) fn report() {
    let s: Vec<u64> = STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    if s[0] + s[1] == 0 {
        return;
    }
    let queued = queue().as_ref().map_or(0, |q| q.todo.len());
    let mut reasons = REASONS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    reasons.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let top: Vec<String> = reasons
        .iter()
        .take(6)
        .map(|(w, n)| format!("{n}x {w}"))
        .collect();
    let per = |us: u64, n: u64| {
        if n == 0 {
            0.0
        } else {
            us as f64 / n as f64 / 1000.0
        }
    };
    tracing::info!(
        translated = s[0],
        failed = s[1],
        unsupported = s[2],
        modules = s[3],
        parts = s[4],
        queued,
        translate_ms_avg = format_args!("{:.2}", per(s[5], s[0] + s[1])),
        module_ms_avg = format_args!("{:.2}", per(s[6], s[3])),
        part_ms_avg = format_args!("{:.2}", per(s[7], s[4])),
        spirv_kwords = s[8] / 1000,
        translator_abi = victory_shader::vk::ABI_VERSION,
        host_abi = HOST_ABI,
        top_failures = %top.join("; "),
        "Vulkan: stages translated once and compiled once"
    );
}
