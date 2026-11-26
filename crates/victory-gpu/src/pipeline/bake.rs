//! The bake: every pipeline in the catalog (`super::catalog`, the
//! pipelines earlier runs drew or dispatched with) is made before the
//! title wants it, like a PC game's "compiling shaders" phase, in the
//! order an earlier run first used them. One function, used two ways:
//!
//! - in the background as the title starts (`start_background`): on at
//!   most `make::BACKGROUND_WORKERS` workers at the lowest priority, after
//!   anything the title asks for, its compiles in flight under 1 GiB;
//! - standalone (`victory bake`, [`standalone`]): before the title, on its
//!   own D3D12 device, with more workers, then the process ends.
//!
//! A baked pipeline is made only to fill the caches (the host's pipeline
//! library, vkd3d-proton's shader cache, the NVIDIA driver's disk cache)
//! and let go: holding thousands costs card memory (code, and local memory
//! reserved per pipeline). The title's creation of it then loads it from
//! those caches. A pipeline the title has created already is made as the
//! title's own and kept.
//!
//! What was made is recorded in the catalog with a [`stamp`] of what it
//! was made with (the translator, the adapter and its driver, the host
//! D3D12 layer): a bake after one with the same stamp skips it without
//! looking further, so a repeat bake takes seconds.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::catalog::{self, Desc};
use super::host::{GraphicsState, HostPipeline};
use super::{Shader, make};

/// One pipeline of the catalog to bake.
pub(crate) struct Job {
    key: String,
    desc: Arc<Desc>,
}

impl Job {
    #[cfg(test)]
    pub(crate) fn test() -> Job {
        Job {
            key: String::new(),
            desc: Arc::new(Desc {
                shaders: Vec::new(),
                graphics: None,
            }),
        }
    }
}

/// The standalone bake's worker count, when this process is one.
static STANDALONE: OnceLock<usize> = OnceLock::new();

/// Whether this process is the standalone bake.
pub(crate) fn is_standalone() -> bool {
    STANDALONE.get().is_some()
}

/// The standalone bake's worker count, when this process is one.
pub(crate) fn standalone_jobs() -> Option<usize> {
    STANDALONE.get().copied()
}

/// What host pipelines are made with: the translator, the adapter and its
/// driver, and the host's D3D12 layer beside our executable (vkd3d-proton
/// under Proton). A pipeline made with the same stamp is in the caches.
pub(crate) fn stamp() -> [u8; 32] {
    static STAMP: OnceLock<[u8; 32]> = OnceLock::new();
    *STAMP.get_or_init(|| {
        use windows::Win32::Graphics::Dxgi::IDXGIDevice;
        use windows::core::Interface;
        let mut h = blake3::Hasher::new();
        h.update(b"victory bake stamp 1\0");
        h.update(&crate::shader_cache::translator());
        if let Some(adapter) = crate::host::host().and_then(|host| host.adapter.as_ref()) {
            // SAFETY: plain queries.
            unsafe {
                if let Ok(d) = adapter.GetDesc1() {
                    for v in [d.VendorId, d.DeviceId, d.SubSysId, d.Revision] {
                        h.update(&v.to_le_bytes());
                    }
                    h.update(&d.Description.map(|c| c as u8));
                }
                if let Ok(umd) = adapter.CheckInterfaceSupport(&IDXGIDevice::IID) {
                    h.update(&umd.to_le_bytes());
                }
            }
        }
        let beside = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(std::path::Path::to_path_buf));
        for dll in ["d3d12.dll", "d3d12core.dll"] {
            let meta = beside
                .as_ref()
                .and_then(|d| std::fs::metadata(d.join(dll)).ok());
            if let Some(m) = meta {
                h.update(&m.len().to_le_bytes());
                let at = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                h.update(&at.map_or(0, |d| d.as_secs()).to_le_bytes());
            }
        }
        *h.finalize().as_bytes()
    })
}

/// The bake's progress: pipelines listed to bake, skipped because a bake
/// or a run made them with this [`stamp`] already, done (made, refused,
/// without a translation, or made as the title's own), made, refused by
/// the host, and without a translation in the cache.
static TOTAL: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicU64 = AtomicU64::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static UNTRANSLATED: AtomicU64 = AtomicU64::new(0);
/// When the bake began (`crate::host::now_ms`), and whether it ended.
static BEGAN_MS: AtomicU64 = AtomicU64::new(0);
static ENDED: AtomicBool = AtomicBool::new(false);

/// Bakes `job` on a worker taking background work: whether a host
/// pipeline was made (or loaded) for it now.
pub(crate) fn run(job: &Job) -> bool {
    let made = bake(job);
    let done = DONE.fetch_add(1, Ordering::Relaxed) + 1;
    if done == TOTAL.load(Ordering::Relaxed) {
        finished();
    }
    made
}

fn bake(job: &Job) -> bool {
    let stamp = stamp();
    // The title created it meanwhile: made as its own, and kept.
    if let Some(p) = make::title_pipeline(&job.key) {
        let made = make::make_for_bake(&p);
        let counter = if made { &MADE } else { &REFUSED };
        counter.fetch_add(1, Ordering::Relaxed);
        return made;
    }
    if catalog::made_with(&job.key, &stamp) {
        return false;
    }
    let shaders: Option<Vec<Arc<Shader>>> = job
        .desc
        .shaders
        .iter()
        .map(|(stage, full, name)| {
            crate::shader_cache::by_hash(full)
                .map(|host| Arc::new(Shader::baked(stage, *full, name.clone(), host)))
        })
        .collect();
    let Some(shaders) = shaders else {
        UNTRANSLATED.fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let graphics = match &job.desc.graphics {
        None => None,
        Some(bytes) => match GraphicsState::decode(bytes) {
            Some(g) => Some(g),
            None => {
                REFUSED.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        },
    };
    let made = {
        let _room = make::reserve(make::estimate(
            shaders.iter().filter_map(|s| s.translated()),
        ));
        HostPipeline::new(&shaders, graphics.as_ref())
    };
    // Let go at once: the caches hold it now.
    let ok = made.is_some();
    drop(made);
    if ok {
        catalog::note_made(&job.key, &stamp);
        MADE.fetch_add(1, Ordering::Relaxed);
    } else {
        REFUSED.fetch_add(1, Ordering::Relaxed);
    }
    ok
}

/// The bake's part of the `frame time:` line.
pub(crate) fn summary() -> String {
    let total = TOTAL.load(Ordering::Relaxed);
    if total == 0 && SKIPPED.load(Ordering::Relaxed) == 0 {
        return "no bake".to_owned();
    }
    format!(
        "bake {} of {} ({} made, {} refused, {} without a translation; {} cached before)",
        DONE.load(Ordering::Relaxed),
        total,
        MADE.load(Ordering::Relaxed),
        REFUSED.load(Ordering::Relaxed),
        UNTRANSLATED.load(Ordering::Relaxed),
        SKIPPED.load(Ordering::Relaxed),
    )
}

/// Readies the translator's facts (the host's wave size and view limit),
/// which the translated-shader cache's keys depend on; `false` without a
/// host GPU.
fn ready_host() -> bool {
    if crate::host::host().is_none() {
        return false;
    }
    crate::descriptor::host::probe_view_limit();
    true
}

/// Lists the catalog's pipelines to bake and queues them; whether there is
/// anything to bake.
fn begin() -> bool {
    let stamp = stamp();
    let (work, skipped) = catalog::to_bake(&stamp);
    SKIPPED.store(skipped as u64, Ordering::Relaxed);
    TOTAL.store(work.len() as u64, Ordering::Relaxed);
    BEGAN_MS.store(crate::host::now_ms(), Ordering::Relaxed);
    tracing::info!(
        to_bake = work.len(),
        cached_before = skipped,
        workers = make::threads().1,
        ceiling_mb = make::ceiling() >> 20,
        "compiling shaders: {} pipelines to make, {skipped} made before with these shaders, driver and layer",
        work.len()
    );
    if work.is_empty() {
        finished();
        return false;
    }
    let jobs = work
        .into_iter()
        .map(|(key, desc)| Arc::new(Job { key, desc }))
        .collect();
    if !make::queue_bake(jobs) {
        tracing::warn!("compiling shaders: no pipeline workers; nothing baked");
        finished();
        return false;
    }
    true
}

/// The bake ended: logged once, with its time, and the caches saved.
fn finished() {
    if ENDED.swap(true, Ordering::Relaxed) {
        return;
    }
    let s = crate::host::now_ms().saturating_sub(BEGAN_MS.load(Ordering::Relaxed)) as f64 / 1000.0;
    tracing::info!(
        seconds = s,
        "compiling shaders: done in {s:.1} s; {}; {}",
        summary(),
        resources()
    );
    super::library::save(true);
}

/// The bake's use of the PC: workers busy (the most at once), the memory
/// reckoned for compiles in flight (the most at once) against the
/// ceiling, the process's resident memory (its peak) and the card memory
/// it uses.
fn resources() -> String {
    let (workers, background) = make::threads();
    let (rss, hwm) = crate::release::resident().unwrap_or((0, 0));
    let card = crate::host::host()
        .and_then(|h| h.video_memory_raw())
        .map_or(0, |v| v[1]);
    format!(
        "workers busy {} (peak {}) of {workers}, {background} taking the bake; compiles in flight {} MB (peak {} MB) of a {} MB ceiling; process memory {} MB (peak {} MB); card memory {} MB",
        make::BUSY.load(Ordering::Relaxed),
        make::BUSY_PEAK.load(Ordering::Relaxed),
        make::reserved() >> 20,
        make::RESERVED_PEAK.load(Ordering::Relaxed) >> 20,
        make::ceiling() >> 20,
        rss >> 20,
        hwm >> 20,
        card >> 20,
    )
}

/// One progress line: done of total, the ETA at the rate so far, and
/// [`resources`].
fn progress() {
    let (done, total) = (DONE.load(Ordering::Relaxed), TOTAL.load(Ordering::Relaxed));
    let elapsed =
        crate::host::now_ms().saturating_sub(BEGAN_MS.load(Ordering::Relaxed)) as f64 / 1000.0;
    let eta = if done == 0 {
        "unknown".to_owned()
    } else {
        let s = (elapsed / done as f64 * total.saturating_sub(done) as f64) as u64;
        format!("{}:{:02}", s / 60, s % 60)
    };
    tracing::info!(
        "compiling shaders: {done} of {total} ({:.0}%), ETA {eta}, {elapsed:.0} s so far; {}; {}",
        if total == 0 {
            100.0
        } else {
            done as f64 * 100.0 / total as f64
        },
        summary(),
        resources()
    );
}

/// In the game, as the title starts: bakes the catalog in the background
/// (see the module's notes), with a progress line every 10 s.
pub(crate) fn start_background() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("victory-bake".into())
            .spawn(|| {
                catalog::begin_run();
                if !ready_host() || !begin() {
                    return;
                }
                while !ENDED.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_secs(10));
                    if !ENDED.load(Ordering::Relaxed) {
                        progress();
                    }
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "compiling shaders: no bake thread");
        }
    });
}

/// The container hashes of the catalog's shaders that the translated-shader
/// cache has no translation of for this translator and host (to translate
/// from the title's files before a standalone bake). Readies the host.
pub fn missing_shaders() -> Vec<[u8; 32]> {
    if !ready_host() {
        return Vec::new();
    }
    catalog::shaders()
        .into_iter()
        .filter(|h| !crate::shader_cache::has(h))
        .collect()
}

/// What a standalone bake did: pipelines made, refused, without a
/// translation, made before (skipped), and its seconds.
pub struct Baked {
    pub made: u64,
    pub refused: u64,
    pub untranslated: u64,
    pub cached_before: u64,
    pub seconds: f64,
}

/// The standalone bake (`victory bake`): makes every pipeline of the
/// catalog not made before with this stamp on `jobs` workers (default:
/// half the logical processors), with a progress line every 2 s, saves
/// the caches and returns. Needs no title: its own D3D12 device, the same
/// caches as the game.
pub fn standalone(jobs: Option<usize>) -> Result<Baked, String> {
    let jobs = jobs.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1))
    });
    if STANDALONE.set(jobs).is_err() {
        return Err("the bake runs once per process".into());
    }
    if !ready_host() {
        return Err("no host D3D12 device".into());
    }
    let began = Instant::now();
    if begin() {
        while !ENDED.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(250));
            static LAST: AtomicUsize = AtomicUsize::new(0);
            let tick = (began.elapsed().as_millis() / 2000) as usize;
            if LAST.swap(tick, Ordering::Relaxed) != tick && !ENDED.load(Ordering::Relaxed) {
                progress();
            }
        }
    }
    Ok(Baked {
        made: MADE.load(Ordering::Relaxed),
        refused: REFUSED.load(Ordering::Relaxed),
        untranslated: UNTRANSLATED.load(Ordering::Relaxed),
        cached_before: SKIPPED.load(Ordering::Relaxed),
        seconds: began.elapsed().as_secs_f64(),
    })
}
