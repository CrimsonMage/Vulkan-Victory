//! M2b step 1: each shader stage translated to SPIR-V once and compiled
//! once (`docs/vulkan-host.md`, "Pipelines"), off the title's threads.
//!
//! At the title's creation of a pipeline whose stages an earlier run
//! translated (kept on disk, `stage_cache.rs`), each of its shader
//! containers not seen before is queued for the stage workers:
//! [`victory_shader::vk::translate`],
//! then a `VkShaderModule`. When the translator speaks the host's ABI
//! ([`HOST_ABI`]), the stage is also compiled: a graphics-pipeline-library
//! part (pre-rasterisation for vertex programs, fragment shader for pixel
//! programs) or a compute pipeline, over the one pipeline layout of
//! `docs/vulkan-abi.md` v1. Until then (the translator's code is at ABI 0
//! while the document is at 1) only modules are made: the run counts what
//! translates, what fails and why, and what it costs.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use victory_shader::HwStage;
use victory_vk::ash::vk;

use super::Backend;
use crate::pipeline::Shader;

/// The ABI version the host binds for (`docs/vulkan-abi.md`).
pub(crate) const HOST_ABI: u32 = 4;

/// Stage workers: half the logical processors, 4 to 8, below the title's
/// threads (`SetThreadPriority` below normal): draws wait for their stages.
fn workers() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get() / 2).clamp(4, 8)
}

/// SPIR-V words compiling at once at most: the driver's memory for a
/// compile grows with the module (seven mesh modules of ~140,000 words
/// compiling together took 14 GiB and the bake was killed for it; about
/// 2 GiB a module of that size).
const WORDS_IN_FLIGHT: u64 = 300_000;

/// Stages bigger than this go to the slow lane (the CS_RaytracingResolve
/// variants, 1.2-1.5 M words, ran over 25 minutes each in NVIDIA's
/// compiler; the next biggest stages are well under).
pub(super) const SLOW_WORDS: usize = 600_000;

thread_local! {
    /// This stage worker handed its place to a new one: it ends after its
    /// slow compile.
    static RETIRING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// This stage worker's compile is slow and a new worker took its place:
/// it ends after it.
fn retire() {
    RETIRING.with(|r| r.set(true));
}

/// Stage `hash` (32 hex digits) of `words` is known to be slow: over
/// [`SLOW_WORDS`], or written down slow by an earlier bake or run.
pub(super) fn known_slow(b: &Backend, hash: &str, words: usize) -> bool {
    words > SLOW_WORDS || super::slow::known(b).contains(hash)
}

/// Stages queued for the stage workers now (the background bake leaves
/// known-slow stages alone while any wait).
pub(super) fn waiting() -> usize {
    queue().as_ref().map_or(0, |q| q.todo.len())
}
static IN_FLIGHT: Mutex<u64> = Mutex::new(0);
static ROOM: Condvar = Condvar::new();

/// Waits until `words` more may compile; hands them back on drop.
pub(super) struct Flight(u64);

impl Flight {
    pub(super) fn take(words: u64) -> Flight {
        let mut n = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
        while *n > 0 && *n + words > WORDS_IN_FLIGHT {
            n = ROOM.wait(n).unwrap_or_else(|e| e.into_inner());
        }
        *n += words;
        Flight(words)
    }
}

impl Drop for Flight {
    fn drop(&mut self) {
        *IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()) -= self.0;
        ROOM.notify_all();
    }
}

/// CPU time of the calling thread, in µs (the driver compiles on it: CPU
/// time against wall time tells a slow compile from a starved one).
fn thread_cpu_us() -> u64 {
    use windows::Win32::Foundation::FILETIME;
    use windows::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let (mut a, mut b, mut k, mut u) = (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
    // SAFETY: a plain query of our own thread.
    if unsafe { GetThreadTimes(GetCurrentThread(), &mut a, &mut b, &mut k, &mut u) }.is_err() {
        return 0;
    }
    let t = |f: FILETIME| (u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime)) / 10;
    t(k) + t(u)
}

/// Images and samplers the global descriptor sets hold (ABI v1).
const IMAGE_SLOTS: u32 = 1 << 20;
pub(crate) const SAMPLER_SLOTS: u32 = 4096;
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
        // ABI v2: cube images as six-layer 2D arrays, at the T#'s slot.
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(IMAGE_SLOTS)
            .stage_flags(all),
    ];
    let samplers = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .descriptor_count(SAMPLER_SLOTS)
        .stage_flags(all)];
    let partly = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 3];
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
enum State {
    Queued,
    Done(Arc<Stage>),
    Failed,
}

/// A stage translated and compiled, as draws link it.
pub(crate) struct Stage {
    pub(crate) stage: HwStage,
    #[allow(dead_code)] // kept while the part lives
    module: vk::ShaderModule,
    /// The compiled part (library part or compute pipeline).
    pub(crate) part: Option<vk::Pipeline>,
    /// Push dword <- root parameter (ABI section 2).
    pub(crate) user_data: Vec<victory_shader::vk::UserData>,
    /// Vertex attributes read as inputs (fetch-shader programs, ABI
    /// section 6): semantic, index, location.
    pub(crate) inputs: Vec<(String, u32, u8)>,
    /// Immediate samplers (the part's specialisation).
    samplers: Vec<[u32; 4]>,
    /// The vertex part with the rectangle geometry stage behind it, made
    /// at the first rectangle-list draw (`rect.rs`).
    rect: std::sync::OnceLock<Option<vk::Pipeline>>,
    #[allow(dead_code)]
    words: usize,
    /// The program's loop-guard id (ABI v3), for diagnostics.
    pub(crate) guard_id: u32,
    /// Mesh: host groups per console group (ABI v4 section 6), 1 otherwise.
    pub(crate) mesh_parts: u32,
}

impl Stage {
    /// The pre-rasterisation part that draws rectangle lists (console
    /// topology 17): this vertex program, then the rectangle geometry
    /// stage. Made once, on the recording thread (a small compile).
    pub(crate) fn rect_part(&self, b: &Backend) -> Option<vk::Pipeline> {
        *self.rect.get_or_init(|| {
            if self.stage != HwStage::Vs {
                return None;
            }
            let gs = super::rect::module(b)?;
            compile(b, self.stage, self.module, &self.samplers, Some(gs))
                .map_err(|e| tracing::warn!(error = %format!("{e:#}"), "Vulkan: the rectangle part does not compile"))
                .ok()
        })
    }
}

/// What a draw finds for a stage.
pub(crate) enum Lookup {
    Ready(Arc<Stage>),
    /// Queued or being compiled: the draw is skipped and counted.
    Waiting,
    /// It does not translate or compile: draws with it are skipped.
    Failed,
}

/// The stage of `shader` for a draw: queued at the front if nothing asked
/// for it before (pipelines the catalog did not know at creation).
pub(crate) fn lookup(b: &'static Backend, shader: &Arc<Shader>) -> Lookup {
    let mut guard = queue();
    let q = guard.get_or_insert_with(|| {
        start(b);
        Queue {
            states: HashMap::new(),
            todo: VecDeque::new(),
        }
    });
    match q.states.get(&shader.full) {
        Some(State::Done(s)) if s.part.is_some() => Lookup::Ready(s.clone()),
        Some(State::Done(_) | State::Failed) => Lookup::Failed,
        Some(State::Queued) => {
            // Queued at creation, behind the catalog: a draw wants it now.
            if let Some(i) = q.todo.iter().position(|s| s.full == shader.full)
                && i > 0
                && let Some(s) = q.todo.remove(i)
            {
                q.todo.push_front(s);
            }
            Lookup::Waiting
        }
        None => {
            q.states.insert(shader.full, State::Queued);
            q.todo.push_front(shader.clone());
            WORK.notify_one();
            Lookup::Waiting
        }
    }
}

/// The one `VkPipelineCache` of the run, loaded from the cache folder
/// (one file per `pipelineCacheUUID`), saved at the end ([`save_cache`]).
pub(crate) fn cache(b: &Backend) -> vk::PipelineCache {
    static CACHE: OnceLock<vk::PipelineCache> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let data = cache_file(b).and_then(|f| std::fs::read(f).ok()).unwrap_or_default();
        // SAFETY: valid create info; a blob of another driver is ignored by
        // the driver (its header names the device).
        let made = unsafe {
            b.host.device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(&data),
                None,
            )
        };
        match made {
            Ok(c) => {
                tracing::info!(kb = data.len() >> 10, "Vulkan: the pipeline cache");
                c
            }
            Err(e) => {
                tracing::warn!(error = %e, "Vulkan: no pipeline cache");
                vk::PipelineCache::null()
            }
        }
    })
}

fn cache_file(b: &Backend) -> Option<std::path::PathBuf> {
    let uuid: String = b.host.caps.pipeline_cache_uuid.iter().map(|x| format!("{x:02x}")).collect();
    Some(crate::pipeline::cache_dir()?.join(format!("vulkan-{uuid}.bin")))
}

/// Saves the pipeline cache (at the end of a run).
pub(crate) fn save_cache(b: &Backend) {
    let c = cache(b);
    if c == vk::PipelineCache::null() {
        return;
    }
    // SAFETY: a live cache.
    if let (Ok(data), Some(f)) = (unsafe { b.host.device.get_pipeline_cache_data(c) }, cache_file(b)) {
        let tmp = f.with_extension("tmp");
        if std::fs::write(&tmp, &data).is_ok() && std::fs::rename(&tmp, &f).is_ok() {
            tracing::info!(kb = data.len() >> 10, "Vulkan: the pipeline cache saved");
        }
    }
}

/// Every 60 s, when parts were compiled since the last save: the pipeline
/// cache is saved (runs end by being killed, so the save at the end is
/// rarely reached).
pub(crate) fn save_cache_now_and_then(b: &Backend) {
    static LAST: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
    let parts = STATS[4].load(Ordering::Relaxed);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let (when, saved) = *last.get_or_insert((Instant::now(), parts));
    if when.elapsed() < std::time::Duration::from_secs(60) || parts == saved {
        return;
    }
    *last = Some((Instant::now(), parts));
    drop(last);
    save_cache(b);
}

struct Queue {
    states: HashMap<[u8; 32], State>,
    todo: VecDeque<Arc<Shader>>,
}

static QUEUE: Mutex<Option<Queue>> = Mutex::new(None);
static WORK: Condvar = Condvar::new();
/// A stage was made (draws waiting for one look again).
static DONE: Condvar = Condvar::new();

/// How long a frame's draws and dispatches may wait for their stages, and
/// how long they waited this frame (µs). The title's GPU watchdog fires
/// after ~30 s without progress; 2 s a frame keeps far from it.
const WAIT_BUDGET_US: u64 = 2_000_000;
static WAITED_US: AtomicU64 = AtomicU64::new(0);
static WAITED_TOTAL_US: AtomicU64 = AtomicU64::new(0);

/// A frame was presented: its waiting budget starts again.
pub(crate) fn new_frame() {
    WAITED_US.store(0, Ordering::Relaxed);
}

/// As [`lookup`], but a stage being compiled is waited for while the
/// frame's budget lasts: a draw or dispatch skipped is lost (one-off
/// passes such as the LUT bakes, a clear drawn as a rectangle), so the
/// recording waits as a PC game's first use of a pipeline does.
pub(crate) fn lookup_waiting(b: &'static Backend, shader: &Arc<Shader>) -> Lookup {
    loop {
        let found = lookup(b, shader);
        if !matches!(found, Lookup::Waiting) || WAITED_US.load(Ordering::Relaxed) >= WAIT_BUDGET_US {
            return found;
        }
        let t = Instant::now();
        let q = queue();
        if q.as_ref().is_some_and(|q| matches!(q.states.get(&shader.full), Some(State::Queued))) {
            drop(DONE.wait_timeout(q, std::time::Duration::from_millis(20)));
        }
        WAITED_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        WAITED_TOTAL_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

fn queue() -> std::sync::MutexGuard<'static, Option<Queue>> {
    QUEUE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Totals since the start: translated, failed, unsupported, modules,
/// parts, translate µs, module µs, part µs, SPIR-V words, from the disk
/// cache.
static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
/// Per kind (pre-rasterisation, fragment, compute): parts, µs, slowest µs.
static KINDS: [[AtomicU64; 3]; 3] = [const { [const { AtomicU64::new(0) }; 3] }; 3];
/// Failure reasons and how often.
static REASONS: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());

/// The title created a pipeline with `shaders`: those not seen go to the
/// stage workers.
pub(crate) fn created(b: &'static Backend, shaders: &[Arc<Shader>]) {
    let mut guard = queue();
    let q = guard.get_or_insert_with(|| {
        start(b);
        Queue {
            states: HashMap::new(),
            todo: VecDeque::new(),
        }
    });
    let inline = mode() == Mode::Inline;
    let mut mine = Vec::new();
    for s in shaders {
        if let std::collections::hash_map::Entry::Vacant(e) = q.states.entry(s.full) {
            e.insert(State::Queued);
            if inline {
                mine.push(s.clone());
            } else {
                q.todo.push_back(s.clone());
            }
        }
    }
    WORK.notify_all();
    drop(guard);
    // Diagnostics: compiled here, on the title's thread, one at a time.
    for s in mine {
        let state = make(b, &s);
        if let Some(q) = queue().as_mut() {
            q.states.insert(s.full, state);
        }
    }
}

/// How stages are compiled (`VICTORY_VK_STAGES`, diagnostics builds only):
/// on the stage workers (default), on the title's thread that created the
/// pipeline, one at a time (`inline`), or not at all, modules only
/// (`modules`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Workers,
    Inline,
    Modules,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if !cfg!(feature = "diagnostics") {
            return Mode::Workers;
        }
        let m = match std::env::var("VICTORY_VK_STAGES").as_deref() {
            Ok("inline") => Mode::Inline,
            Ok("modules") => Mode::Modules,
            _ => Mode::Workers,
        };
        tracing::info!(mode = ?m, "Vulkan: how stages compile (VICTORY_VK_STAGES)");
        m
    })
}

/// Diagnostics builds: the folder each module's words are written to before
/// it is compiled (`VICTORY_VK_DUMP`), as `<hash>.<stage>.spv`.
fn dump_dir() -> Option<&'static std::path::Path> {
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        if !cfg!(feature = "diagnostics") {
            return None;
        }
        let dir = std::path::PathBuf::from(std::env::var_os("VICTORY_VK_DUMP")?);
        let _ = std::fs::create_dir_all(&dir);
        Some(dir)
    })
    .as_deref()
}

/// The device limits modules are checked against, read once.
pub(crate) fn limits(b: &Backend) -> &'static victory_vk::limits::Limits {
    static LIMITS: OnceLock<victory_vk::limits::Limits> = OnceLock::new();
    LIMITS.get_or_init(|| {
        let l = victory_vk::limits::Limits::of(&b.host);
        tracing::info!(
            compute_invocations = l.compute_invocations,
            compute_shared = l.compute_shared,
            mesh_invocations = l.mesh_invocations,
            mesh_shared = l.mesh_shared,
            mesh_output_memory = l.mesh_output_memory,
            mesh_vertices = l.mesh_vertices,
            mesh_primitives = l.mesh_primitives,
            mesh_components = l.mesh_components,
            stencil_export = l.stencil_export,
            mesh_groups = ?l.mesh_groups,
            mesh_groups_total = l.mesh_groups_total,
            "Vulkan: the limits modules are checked against before they compile"
        );
        l
    })
}

fn hash_hex(full: &[u8; 32]) -> String {
    full[..8].iter().map(|b| format!("{b:02x}")).collect()
}

fn start(b: &'static Backend) {
    super::baking::in_background(b);
    let n = workers();
    tracing::info!(workers = n, words_in_flight = WORDS_IN_FLIGHT, slow_words = SLOW_WORDS, "Vulkan: stage workers");
    for i in 0..n {
        spawn_worker(b, i);
    }
    super::slow::start_watch(b, |b| spawn_worker(b, usize::MAX));
}

fn spawn_worker(b: &'static Backend, i: usize) {
    let name = if i == usize::MAX { "victory-vk-stage-extra".to_string() } else { format!("victory-vk-stage-{i}") };
    let _ = std::thread::Builder::new().name(name).spawn(move || worker(b));
}

fn worker(b: &'static Backend) {
    WORKER.with(|w| w.set(true));
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
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
        DONE.notify_all();
        if RETIRING.with(std::cell::Cell::get) {
            return;
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

/// The translation of `shader`: from the disk cache, else translated now
/// (and kept).
fn translate(shader: &Shader) -> std::thread::Result<Result<victory_shader::vk::VkShader, victory_shader::ShaderError>> {
    if let Some(vs) = super::stage_cache::read(&shader.full) {
        STATS[9].fetch_add(1, Ordering::Relaxed);
        return Ok(Ok(vs));
    }
    let t = Instant::now();
    let translated = std::panic::catch_unwind(|| victory_shader::vk::translate(shader.container()));
    STATS[5].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    if let Ok(Ok(vs)) = &translated {
        super::stage_cache::write(&shader.full, vs);
    }
    translated
}

fn make(b: &Backend, shader: &Shader) -> State {
    let vs = match translate(shader) {
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
    let hash = hash_hex(&shader.full);
    super::guard::register(vs.guard_id, &shader.name, &hash);
    super::stage_cache::note_name(&shader.full, &shader.name);
    let usage = victory_vk::limits::usage(&vs.spirv);
    if let Some(dir) = dump_dir() {
        let bytes: Vec<u8> = vs.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        let _ = std::fs::write(dir.join(format!("{hash}.{:?}.spv", vs.stage)), bytes);
    }
    if let Some(why) = limits(b).refuses(&usage) {
        if cfg!(feature = "diagnostics") {
            tracing::info!(stage = ?vs.stage, hash = %hash, why = %why, "Vulkan part: refused before compiling");
        }
        return fail(format!("over the device: {why}"));
    }
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
    let speaks = victory_shader::vk::ABI_VERSION >= HOST_ABI && mode() != Mode::Modules;
    let part = if speaks {
        // Diagnostics builds name each part before the driver sees it, and
        // flush: a fault inside the driver ends the process without a word
        // (winevulkan's ExitProcess(3)), so the last such line names it.
        if cfg!(feature = "diagnostics") {
            tracing::info!(
                stage = ?vs.stage,
                hash = %hash,
                words = vs.spirv.len(),
                samplers = vs.samplers.len(),
                local = ?usage.local_size,
                workgroup_bytes = usage.workgroup_bytes,
                out_vertices = usage.out_vertices,
                out_primitives = usage.out_primitives,
                out_locations = usage.out_locations,
                mesh_out_bytes = usage.mesh_output_bytes(limits(b).mesh_budget()),
                thread = std::thread::current().name().unwrap_or("?"),
                "Vulkan part: compiling"
            );
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }
        let (p, t, cpu_ms) = compile_timed(b, shader, &vs, module);
        if cfg!(feature = "diagnostics") {
            tracing::info!(
                stage = ?vs.stage,
                hash = %hash,
                ms = t.elapsed().as_millis() as u64,
                cpu_ms,
                ok = p.is_ok(),
                "Vulkan part: compiled"
            );
        }
        let us = t.elapsed().as_micros() as u64;
        STATS[7].fetch_add(us, Ordering::Relaxed);
        let kind = match vs.stage {
            HwStage::Ps => 1,
            HwStage::Cs => 2,
            _ => 0,
        };
        KINDS[kind][0].fetch_add(1, Ordering::Relaxed);
        KINDS[kind][1].fetch_add(us, Ordering::Relaxed);
        KINDS[kind][2].fetch_max(us, Ordering::Relaxed);
        if us > 2_000_000 {
            tracing::info!(stage = ?vs.stage, hash = %hash, ms = us / 1000, words = vs.spirv.len(), "Vulkan: a slow part compile");
        }
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
    let inputs = vs
        .interface
        .iter()
        .filter(|i| i.input && vs.stage != HwStage::Ps)
        .map(|i| (i.semantic.clone(), i.index, i.location))
        .collect();
    State::Done(Arc::new(Stage {
        stage: vs.stage,
        module,
        part,
        user_data: vs.user_data.clone(),
        inputs,
        samplers: vs.samplers.clone(),
        rect: std::sync::OnceLock::new(),
        words: vs.spirv.len(),
        guard_id: vs.guard_id,
        mesh_parts: vs.mesh_parts.max(1),
    }))
}

/// Compiles a stage's part, registered as running (`slow.rs`): a stage
/// known to be slow waits for its turn on the slow lane and hands its
/// worker's place to a new one first; one found slow while compiling had
/// its place handed over by the watch. Either way this worker ends after
/// it. (part, start, CPU ms.)
fn compile_timed(b: &Backend, shader: &Shader, vs: &victory_shader::vk::VkShader, module: vk::ShaderModule) -> (anyhow::Result<vk::Pipeline>, Instant, u64) {
    use super::slow::{self, Owner};
    let hash: String = shader.full[..16].iter().map(|x| format!("{x:02x}")).collect();
    let worker = WORKER.with(std::cell::Cell::get);
    let known = known_slow(b, &hash, vs.spirv.len());
    if known {
        tracing::info!(words = vs.spirv.len(), hash = %hash, name = %shader.name, "Vulkan: a stage known to be slow waits for the slow lane");
        slow::lane();
        if worker && !RETIRING.with(std::cell::Cell::get)
            && let Some(b) = super::backend()
        {
            retire();
            spawn_worker(b, usize::MAX);
        }
    }
    let flight = (!known).then(|| Flight::take(vs.spirv.len() as u64));
    let owner = if worker { Owner::Worker } else { Owner::Background };
    let running = slow::compiling(&hash, &shader.name, &vs.spirv, owner, known);
    let (t, cpu) = (Instant::now(), thread_cpu_us());
    let p = compile(b, vs.stage, module, &vs.samplers, None);
    let cpu_ms = thread_cpu_us().saturating_sub(cpu) / 1000;
    if running.finish() && worker {
        retire();
    }
    drop(flight);
    (p, t, cpu_ms)
}

/// The first of the sampler heap's slots kept for immediate samplers
/// (proposal 8): set 1 holds 4,096 samplers, the last 256 are these.
pub(crate) const IMMEDIATE_FIRST: u32 = SAMPLER_SLOTS - 256;

/// Immediate samplers by their words, and the heap slot each has.
pub(crate) static IMMEDIATE: Mutex<Vec<[u32; 4]>> = Mutex::new(Vec::new());

/// The heap slot of an immediate sampler (one per distinct words; the
/// sampler itself is written into set 1 when descriptors are made, step
/// (a)). Past 256 distinct ones the last slot is shared (logged once).
fn immediate_slot(words: [u32; 4]) -> u32 {
    static LOADED: std::sync::Once = std::sync::Once::new();
    let mut list = IMMEDIATE.lock().unwrap_or_else(|e| e.into_inner());
    // The slots of earlier runs first, so that a stage's specialisation
    // (and the driver's cached pipeline) is the same in every run.
    LOADED.call_once(|| *list = super::stage_cache::load_immediates());
    let at = match list.iter().position(|w| *w == words) {
        Some(i) => i,
        None if list.len() < 256 => {
            list.push(words);
            super::stage_cache::save_immediates(&list);
            list.len() - 1
        }
        None => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("Vulkan: more than 256 distinct immediate samplers; the rest share the last slot"));
            255
        }
    };
    IMMEDIATE_FIRST + at as u32
}

/// The dynamic states of pre-rasterisation parts (also the test parts').
pub(crate) fn pre_raster_dynamic(b: &Backend) -> Vec<vk::DynamicState> {
    dynamic(b, false)
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

/// Compiles one stage: a library part (vertex: pre-rasterisation, with
/// `geometry` behind it if given; pixel: fragment shader) or a compute
/// pipeline.
pub(super) fn compile(
    b: &Backend,
    stage: HwStage,
    module: vk::ShaderModule,
    samplers: &[[u32; 4]],
    geometry: Option<vk::ShaderModule>,
) -> anyhow::Result<vk::Pipeline> {
    let layout = layout(b).ok_or_else(|| anyhow::anyhow!("no pipeline layout"))?;
    let d = &b.host.device;
    let db = if b.host.caps.descriptor_buffer {
        vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT
    } else {
        vk::PipelineCreateFlags::empty()
    };
    let name = c"main";
    // Specialisation (ABI v1): constant 0 false (the map has more than one
    // range), constant 100 + n the heap slot of immediate sampler n.
    let mut data: Vec<u32> = vec![vk::FALSE];
    let mut entries = vec![
        vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(4),
    ];
    for (n, words) in samplers.iter().enumerate() {
        entries.push(
            vk::SpecializationMapEntry::default()
                .constant_id(100 + n as u32)
                .offset(4 * data.len() as u32)
                .size(4),
        );
        data.push(immediate_slot(*words));
    }
    let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    let spec = vk::SpecializationInfo::default()
        .map_entries(&entries)
        .data(&bytes);
    // SAFETY: valid create infos over a live module and layout.
    unsafe {
        if stage == HwStage::Cs {
            let info = vk::ComputePipelineCreateInfo::default()
                .flags(db)
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(name)
                        .specialization_info(&spec),
                )
                .layout(layout.pipeline);
            return d
                .create_compute_pipelines(cache(b), &[info], None)
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
            HwStage::Ms => (
                vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS,
                vk::ShaderStageFlags::MESH_EXT,
                false,
            ),
            other => anyhow::bail!("no library part for a {other:?} program"),
        };
        let states = dynamic(b, fragment);
        let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
        let mut stages = vec![vk::PipelineShaderStageCreateInfo::default()
            .stage(vk_stage)
            .module(module)
            .name(name)
            .specialization_info(&spec)];
        if let Some(gs) = geometry {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::GEOMETRY).module(gs).name(name));
        }
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
                // No RETAIN_LINK_TIME_OPTIMIZATION_INFO: no optimised link is
                // made yet, and keeping that information costs the driver
                // a full compile per part.
                db | vk::PipelineCreateFlags::LIBRARY_KHR,
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
        d.create_graphics_pipelines(cache(b), &[info], None)
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
        pre_raster_ms_avg = format_args!("{:.1}", per(KINDS[0][1].load(Ordering::Relaxed), KINDS[0][0].load(Ordering::Relaxed))),
        fragment_ms_avg = format_args!("{:.1}", per(KINDS[1][1].load(Ordering::Relaxed), KINDS[1][0].load(Ordering::Relaxed))),
        compute_ms_avg = format_args!("{:.1}", per(KINDS[2][1].load(Ordering::Relaxed), KINDS[2][0].load(Ordering::Relaxed))),
        slowest_ms = [0, 1, 2].map(|k| KINDS[k][2].load(Ordering::Relaxed) / 1000).iter().max().copied().unwrap_or(0),
        spirv_kwords = s[8] / 1000,
        from_disk = s[9],
        draws_waited_ms = WAITED_TOTAL_US.load(Ordering::Relaxed) / 1000,
        translator_abi = victory_shader::vk::ABI_VERSION,
        host_abi = HOST_ABI,
        top_failures = %top.join("; "),
        "Vulkan: stages translated once and compiled once"
    );
}
