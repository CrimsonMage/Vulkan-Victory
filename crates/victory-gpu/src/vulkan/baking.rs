//! `victory bake` and the game's background bake: every translation the
//! stage cache holds (this translator's key) is compiled as the game
//! compiles it (`stages::compile`: the same device, layout, flags,
//! specialisation and pipeline cache), so that the driver's caches and
//! ours hold it before a draw waits for it.
//!
//! - `victory bake` runs on half the logical processors with at most a
//!   million SPIR-V words compiling at once, saves the pipeline cache every
//!   60 s and at the end, prints progress with an ETA, and writes the
//!   pipeline index (`--index`, default `pipeline-index.md` in the cache
//!   folder): a summary, one row per stage, the stages by refusal or
//!   failure reason, the 30 slowest compiles. No timestamps; rows sorted
//!   by hash, so two runs diff.
//! - The game starts one background thread at the lowest priority when
//!   its stage workers start, which compiles what no bake compiled yet.
//! - What a bake compiled (or refused) is listed per translator key and
//!   device (`baked-<key>-<device>.txt`) and skipped by the next one.
//!
//! The title pipelines that pair stages are not known here: the Vulkan
//! host keeps no catalog of them.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use victory_shader::HwStage;
use victory_vk::ash::vk;

use super::{Backend, backend, stage_cache, stages};

/// A compile at most this long found the stage in a cache.
const HIT_MS: u64 = 20;

#[derive(Clone)]
enum Status {
    Compiled(u64),
    Cached(u64),
    BakedBefore,
    Refused(String),
    Failed(String),
}

struct Row {
    hash: String,
    name: String,
    kind: &'static str,
    words: usize,
    status: Status,
}

fn kind_of(stage: HwStage) -> &'static str {
    match stage {
        HwStage::Ps => "fragment",
        HwStage::Cs => "compute",
        _ => "pre-raster/mesh",
    }
}

fn device_key(b: &Backend) -> String {
    b.host.caps.pipeline_cache_uuid.iter().map(|x| format!("{x:02x}")).collect()
}

/// Compiles the translation in `file` as the game would; its row.
fn bake_one(b: &Backend, hash: &str, file: &Path, name: String) -> Row {
    let Some(vs) = stage_cache::read_file(file) else {
        return Row { hash: hash.into(), name, kind: "?", words: 0, status: Status::Failed("unreadable translation".into()) };
    };
    let row = |status| Row { hash: hash.into(), name: name.clone(), kind: kind_of(vs.stage), words: vs.spirv.len(), status };
    if let Some(why) = victory_vk::limits::usage(&vs.spirv).refused(stages::limits(b)) {
        return row(Status::Refused(format!("over the device: {why}")));
    }
    let d = &b.host.device;
    // SAFETY: the translator's words, a SPIR-V module.
    let module = match unsafe { d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs.spirv), None) } {
        Ok(m) => m,
        Err(e) => return row(Status::Failed(format!("vkCreateShaderModule: {e}"))),
    };
    let flight = stages::Flight::take(vs.spirv.len() as u64);
    let t = Instant::now();
    let made = stages::compile(b, vs.stage, module, &vs.samplers, None);
    let ms = t.elapsed().as_millis() as u64;
    drop(flight);
    // SAFETY: ours, used by nothing (the caches keep what was compiled).
    unsafe {
        if let Ok(p) = &made {
            d.destroy_pipeline(*p, None);
        }
        d.destroy_shader_module(module, None);
    }
    match made {
        Ok(_) if ms <= HIT_MS => row(Status::Cached(ms)),
        Ok(_) => row(Status::Compiled(ms)),
        Err(e) => row(Status::Failed(format!("{e:#}"))),
    }
}

/// What is left to bake: (hash, file), and the program names.
fn todo(b: &Backend) -> (VecDeque<(String, PathBuf)>, Vec<Row>, BTreeMap<String, String>) {
    let names: BTreeMap<String, String> = stage_cache::names().into_iter().collect();
    let baked = stage_cache::baked(&device_key(b));
    let mut done = Vec::new();
    let mut left = VecDeque::new();
    for (hash, file) in stage_cache::all() {
        if baked.contains(&hash) {
            let name = names.get(&hash).cloned().unwrap_or_default();
            done.push(Row { hash, name, kind: "", words: 0, status: Status::BakedBefore });
        } else {
            left.push_back((hash, file));
        }
    }
    (left, done, names)
}

/// Bakes the queue on `jobs` threads at `priority`; each row as it is done
/// goes to `rows` and is marked baked.
fn work(b: &'static Backend, queue: &Mutex<VecDeque<(String, PathBuf)>>, names: &BTreeMap<String, String>, rows: &Mutex<Vec<Row>>, (jobs, priority): (usize, i32)) {
    let device = device_key(b);
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                set_priority(priority);
                loop {
                    let Some((hash, file)) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front() else { break };
                    let row = bake_one(b, &hash, &file, names.get(&hash).cloned().unwrap_or_default());
                    if !matches!(row.status, Status::Failed(_)) {
                        stage_cache::mark_baked(&device, &hash);
                    }
                    rows.lock().unwrap_or_else(|e| e.into_inner()).push(row);
                }
            });
        }
    });
}

fn set_priority(level: i32) {
    use windows::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY};
    // SAFETY: a plain call on our own thread.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY(level));
    }
}

/// `victory bake`.
pub(super) fn run(jobs: Option<usize>, index: Option<PathBuf>) -> anyhow::Result<()> {
    let b = backend().ok_or_else(|| anyhow::anyhow!("no Vulkan device"))?;
    stages::layout(b).ok_or_else(|| anyhow::anyhow!("no pipeline layout"))?;
    let (left, mut done, names) = todo(b);
    let total = left.len();
    let jobs = jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() / 2)).max(1);
    println!("bake: {total} stages to compile, {} baked before, {jobs} threads", done.len());
    let queue = Mutex::new(left);
    let rows = Mutex::new(Vec::new());
    let started = Instant::now();
    std::thread::scope(|s| {
        s.spawn(|| work(b, &queue, &names, &rows, (jobs, 0)));
        progress(b, &rows, total, started);
    });
    stages::save_cache(b);
    done.extend(rows.into_inner().unwrap_or_else(|e| e.into_inner()));
    done.sort_by(|a, c| a.hash.cmp(&c.hash));
    let path = index.or_else(|| crate::pipeline::cache_dir().map(|d| d.with_file_name("pipeline-index.md"))).unwrap_or_else(|| "pipeline-index.md".into());
    std::fs::write(&path, index_text(&done))?;
    println!("bake: done in {:.0} s; pipeline index {}", started.elapsed().as_secs_f64(), path.display());
    Ok(())
}

/// Progress every 5 s with an ETA; the pipeline cache saved every 60 s.
fn progress(b: &Backend, rows: &Mutex<Vec<Row>>, total: usize, started: Instant) {
    let mut saved = Instant::now();
    loop {
        std::thread::sleep(Duration::from_secs(5));
        let n = rows.lock().unwrap_or_else(|e| e.into_inner()).len();
        let secs = started.elapsed().as_secs_f64();
        let eta = if n == 0 { 0.0 } else { secs / n as f64 * (total - n) as f64 };
        println!("bake: {n} of {total} ({:.0}%), {:.0} s so far, ETA {:.0} s", 100.0 * n as f64 / total.max(1) as f64, secs, eta);
        if saved.elapsed() >= Duration::from_secs(60) {
            stages::save_cache(b);
            saved = Instant::now();
        }
        if n >= total {
            return;
        }
    }
}

/// The game's background bake: one thread at the lowest priority bakes
/// what no bake did yet, from the stage workers' start on.
pub(super) fn in_background(b: &'static Backend) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-vk-bake".into()).spawn(move || {
            let (left, done, names) = todo(b);
            tracing::info!(left = left.len(), baked_before = done.len(), "Vulkan: background bake of the stage cache");
            let rows = Mutex::new(Vec::new());
            work(b, &Mutex::new(left), &names, &rows, (1, -2));
            tracing::info!(baked = rows.lock().map_or(0, |r| r.len()), "Vulkan: background bake done");
        });
    });
}

// ---- the pipeline index ----

fn ms_of(s: &Status) -> Option<u64> {
    match s {
        Status::Compiled(ms) | Status::Cached(ms) => Some(*ms),
        _ => None,
    }
}

fn status_text(s: &Status) -> String {
    match s {
        Status::Compiled(_) => "compiled".into(),
        Status::Cached(_) => "cached".into(),
        Status::BakedBefore => "baked before".into(),
        Status::Refused(w) => format!("refused: {w}"),
        Status::Failed(w) => format!("failed: {w}"),
    }
}

fn percentile(sorted: &[u64], p: usize) -> u64 {
    sorted.get((sorted.len().saturating_sub(1)) * p / 100).copied().unwrap_or(0)
}

fn summary(rows: &[Row]) -> String {
    let mut by: BTreeMap<&str, [usize; 5]> = BTreeMap::new();
    for r in rows {
        let k = by.entry(if r.kind.is_empty() { "(baked before)" } else { r.kind }).or_default();
        let i = match r.status {
            Status::Compiled(_) => 0,
            Status::Cached(_) | Status::BakedBefore => 1,
            Status::Refused(_) => 2,
            Status::Failed(_) => 3,
        };
        k[i] += 1;
        k[4] += 1;
    }
    let mut compiled: Vec<u64> = rows.iter().filter(|r| matches!(r.status, Status::Compiled(_))).filter_map(|r| ms_of(&r.status)).collect();
    compiled.sort_unstable();
    let cached = rows.iter().filter(|r| matches!(r.status, Status::Cached(_))).count();
    let mut out = format!("Stages known: {}\n\n| kind | stages | compiled | cached | refused | failed |\n|---|---|---|---|---|---|\n", rows.len());
    for (k, c) in &by {
        out += &format!("| {k} | {} | {} | {} | {} | {} |\n", c[4], c[0], c[1], c[2], c[3]);
    }
    out += &format!(
        "\nCompile ms (compiled this run): total {}, p50 {}, p90 {}, max {}. Cache hits this run: {} of {} compiles (a compile of {HIT_MS} ms or less).\n",
        compiled.iter().sum::<u64>(),
        percentile(&compiled, 50),
        percentile(&compiled, 90),
        compiled.last().copied().unwrap_or(0),
        cached,
        cached + compiled.len()
    );
    out
}

fn index_text(rows: &[Row]) -> String {
    let mut out = String::from("# Pipeline index\n\nGenerated by `victory bake` from the Vulkan stage cache (this translator's key). Not edited by hand.\n\n## Summary\n\n");
    out += &summary(rows);
    out += "\nThe title pipelines that pair stages are not listed: the Vulkan host keeps no catalog of them.\n\n## Stages\n\n| hash | program | kind | words | compile ms | status |\n|---|---|---|---|---|---|\n";
    for r in rows {
        let ms = ms_of(&r.status).map_or_else(|| "-".into(), |m| m.to_string());
        out += &format!("| {} | {} | {} | {} | {ms} | {} |\n", &r.hash[..16.min(r.hash.len())], r.name, r.kind, r.words, status_text(&r.status));
    }
    let mut reasons: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for r in rows {
        if let Status::Refused(_) | Status::Failed(_) = r.status {
            reasons.entry(status_text(&r.status)).or_default().insert(format!("{} {}", &r.hash[..16.min(r.hash.len())], r.name));
        }
    }
    out += "\n## Refusals and failures\n";
    for (why, stages) in &reasons {
        out += &format!("\n### {why}\n\n");
        for s in stages {
            out += &format!("- {s}\n");
        }
    }
    let mut slow: Vec<&Row> = rows.iter().filter(|r| matches!(r.status, Status::Compiled(_))).collect();
    slow.sort_by(|a, c| ms_of(&c.status).cmp(&ms_of(&a.status)).then(a.hash.cmp(&c.hash)));
    out += "\n## The 30 slowest compiles\n\n| hash | program | kind | words | ms |\n|---|---|---|---|---|\n";
    for r in slow.iter().take(30) {
        out += &format!("| {} | {} | {} | {} | {} |\n", &r.hash[..16.min(r.hash.len())], r.name, r.kind, r.words, ms_of(&r.status).unwrap_or(0));
    }
    out
}
