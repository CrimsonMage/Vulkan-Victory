//! `victory bake` and the game's background bake: every translation the
//! stage cache holds (this translator's key) is compiled as the game
//! compiles it (`stages::compile`: the same device, layout, flags,
//! specialisation and pipeline cache), so that the driver's caches and
//! ours hold it before a draw waits for it.
//!
//! - First, containers the title used (a translation under any key, or a
//!   name noted) with none under this key are translated from the title's
//!   packs (`victory bake` in the app finds them; [`translate`]), exactly
//!   as the game would translate them; old keys' files with a current
//!   twin are deleted afterwards.
//! - Stages compile smallest first, stages known to be slow last, on half
//!   the logical processors with at most a million SPIR-V words compiling
//!   at once. A compile that runs a minute turns slow (`slow.rs`): a new
//!   bake thread takes its place. When only slow compiles are left and the
//!   oldest has run [`GIVE_UP`], the bake ends without them (listed with
//!   their time so far, and written down slow for the next bake and the
//!   game).
//! - The game's background bake leaves stages known to be slow alone while
//!   the title's own stages wait.
//! - Progress every 5 s with an ETA, and what is compiling when nothing
//!   finished since; the pipeline cache saved and the pipeline index
//!   rewritten every 60 s and at the end.
//! - What each stage came to (compiled, cached, refused or failed with its
//!   reason, slow lane) is kept per translator key and device
//!   (`baked-<key>-<device>.txt`): the next bake skips it (failures too:
//!   they are tried again when the translator changes) and its index still
//!   lists it.
//! - The game starts one background thread at the lowest priority when
//!   its stage workers start, which compiles what no bake did yet.
//!
//! The title pipelines that pair stages are not known here: the Vulkan
//! host keeps no catalog of them.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use victory_shader::HwStage;
use victory_vk::ash::vk;

use super::slow::{self, Owner};
use super::{Backend, backend, stage_cache, stages};

/// A compile at most this long found the stage in a cache.
const HIT_MS: u64 = 20;
/// The slow lane's longest compile the bake waits for once nothing else
/// is left.
const GIVE_UP: Duration = Duration::from_secs(15 * 60);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone)]
enum Status {
    Compiled(u64),
    Cached(u64),
    Refused(String),
    Failed(String),
    /// On the slow lane when the bake ended: seconds so far.
    GaveUp(u64),
    /// An earlier bake's outcome, as it wrote it.
    Before(String),
}

#[derive(Clone)]
struct Row {
    hash: String,
    name: String,
    kind: &'static str,
    words: usize,
    slow: bool,
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

/// The line kept for a row: `words|kind|lane|status`.
fn outcome(r: &Row) -> String {
    let lane = if r.slow { "slow lane" } else { "" };
    format!("{}|{}|{lane}|{}", r.words, r.kind, status_text(&r.status))
}

fn row_before(hash: String, name: String, line: &str) -> Row {
    let mut f = line.splitn(4, '|');
    let words = f.next().and_then(|w| w.parse().ok()).unwrap_or(0);
    let kind = match f.next() {
        Some("fragment") => "fragment",
        Some("compute") => "compute",
        Some("pre-raster/mesh") => "pre-raster/mesh",
        _ => "",
    };
    let slow = f.next() == Some("slow lane");
    let status = Status::Before(f.next().unwrap_or(line).to_string());
    Row { hash, name, kind, words, slow, status }
}

/// What a bake works on.
struct Shared {
    device: String,
    names: BTreeMap<String, String>,
    queue: Mutex<VecDeque<(String, PathBuf)>>,
    rows: Mutex<Vec<Row>>,
    total: usize,
}

/// The stages left (smallest first) and the rows of earlier bakes.
fn prepare(b: &Backend) -> (Shared, Vec<Row>) {
    let names: BTreeMap<String, String> = stage_cache::names().into_iter().collect();
    let device = device_key(b);
    let before = stage_cache::baked(&device);
    let mut done = Vec::new();
    let mut left = Vec::new();
    for (hash, file) in stage_cache::all() {
        let name = names.get(&hash).cloned().unwrap_or_default();
        match before.get(&hash) {
            Some(line) if !slow::is_slow(line) => done.push(row_before(hash, name, line)),
            // Slow before: compiled again, last.
            Some(_) => left.push((true, std::fs::metadata(&file).map_or(0, |m| m.len()), hash, file)),
            None => left.push((false, std::fs::metadata(&file).map_or(0, |m| m.len()), hash, file)),
        }
    }
    left.sort();
    let total = left.len();
    let queue = Mutex::new(left.into_iter().map(|(_, _, h, f)| (h, f)).collect());
    (Shared { device, names, queue, rows: Mutex::new(Vec::new()), total }, done)
}

/// Compiles the translation in `file` as the game would; its row, and
/// whether it turned slow (its thread was replaced: it ends).
fn bake_one(b: &Backend, hash: &str, file: &Path, name: String, owner: Owner) -> (Row, bool) {
    let Some(vs) = stage_cache::read_file(file) else {
        return (Row { hash: hash.into(), name, kind: "?", words: 0, slow: false, status: Status::Failed("unreadable translation".into()) }, false);
    };
    let words = vs.spirv.len();
    let known = stages::known_slow(b, hash, words);
    if known && owner == Owner::Background {
        // The title's own stages first.
        while stages::waiting() > 0 {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    let row = |status, slow| Row { hash: hash.into(), name: name.clone(), kind: kind_of(vs.stage), words, slow, status };
    if let Some(why) = stages::limits(b).refuses(&victory_vk::limits::usage(&vs.spirv)) {
        return (row(Status::Refused(format!("over the device: {why}")), false), false);
    }
    let d = &b.host.device;
    // SAFETY: the translator's words, a SPIR-V module.
    let module = match unsafe { d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs.spirv), None) } {
        Ok(m) => m,
        Err(e) => return (row(Status::Failed(format!("vkCreateShaderModule: {e}")), false), false),
    };
    if known {
        slow::lane();
    }
    let flight = (!known).then(|| stages::Flight::take(words as u64));
    let running = slow::compiling(hash, &name, &vs.spirv, owner, known);
    let t = Instant::now();
    let made = stages::compile(b, vs.stage, module, &vs.samplers, None);
    let ms = t.elapsed().as_millis() as u64;
    let turned = running.finish() && !known;
    drop(flight);
    // SAFETY: ours, used by nothing (the caches keep what was compiled).
    unsafe {
        if let Ok(p) = &made {
            d.destroy_pipeline(*p, None);
        }
        d.destroy_shader_module(module, None);
    }
    let slow = known || turned;
    let status = match made {
        Ok(_) if ms <= HIT_MS => Status::Cached(ms),
        Ok(_) => Status::Compiled(ms),
        Err(e) => Status::Failed(format!("{e:#}")),
    };
    (row(status, slow), turned)
}

/// One bake thread: stages from the queue until it is empty, or until a
/// compile of its turned slow (a new thread took its place).
fn work(b: &'static Backend, s: &Shared, priority: i32, owner: Owner) {
    set_priority(priority);
    loop {
        let Some((hash, file)) = lock(&s.queue).pop_front() else { return };
        let (row, turned) = bake_one(b, &hash, &file, s.names.get(&hash).cloned().unwrap_or_default(), owner);
        stage_cache::mark_baked(&s.device, &hash, &outcome(&row));
        lock(&s.rows).push(row);
        if turned {
            return;
        }
    }
}

fn set_priority(level: i32) {
    use windows::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY};
    // SAFETY: a plain call on our own thread.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY(level));
    }
}

/// The containers to translate from the title's packs first.
pub(super) fn wanted() -> BTreeSet<String> {
    stage_cache::wanted()
}

/// Translates a container from the title's packs as the game would, and
/// keeps its translation and name; the error, if it does not translate.
pub(super) fn translate(container: &[u8]) -> Result<(), String> {
    let full = *blake3::hash(container).as_bytes();
    stage_cache::note_name(&full, &crate::pipeline::program_name(container));
    match std::panic::catch_unwind(|| victory_shader::vk::translate(container)) {
        Ok(Ok(vs)) => {
            stage_cache::write(&full, &vs);
            Ok(())
        }
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("the translator panicked".into()),
    }
}

/// Old keys' translations with a current twin, deleted; how many.
pub(super) fn prune() -> usize {
    stage_cache::prune_old_keys()
}

/// `victory bake`.
pub(super) fn run(jobs: Option<usize>, index: Option<PathBuf>) -> anyhow::Result<()> {
    let b = backend().ok_or_else(|| anyhow::anyhow!("no Vulkan device"))?;
    stages::layout(b).ok_or_else(|| anyhow::anyhow!("no pipeline layout"))?;
    let (shared, before) = prepare(b);
    let shared = Arc::new(shared);
    let jobs = jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() / 2)).max(1);
    println!("bake: {} stages to compile (smallest first), {} done by earlier bakes, {jobs} threads", shared.total, before.len());
    for _ in 0..jobs {
        let s = shared.clone();
        std::thread::spawn(move || work(b, &s, 0, Owner::Bake));
    }
    let path = index.or_else(|| crate::pipeline::cache_dir().map(|d| d.with_file_name("pipeline-index.md"))).unwrap_or_else(|| "pipeline-index.md".into());
    let started = Instant::now();
    watch(b, &shared, &before, &path, started);
    if let Some(dir) = crate::pipeline::cache_dir() {
        println!("bake: slow stages' SPIR-V (diagnostics builds) in {}", dir.with_file_name("slow-stages").display());
    }
    let rows = all_rows(&shared, &before);
    std::fs::write(&path, index_text(&rows))?;
    println!("bake: done in {:.0} s; pipeline index {}", started.elapsed().as_secs_f64(), path.display());
    Ok(())
}

/// Every row: earlier bakes', this one's, and the slow-lane compiles still
/// running (given up), sorted by hash.
fn all_rows(s: &Shared, before: &[Row]) -> Vec<Row> {
    let mut rows: Vec<Row> = before.to_vec();
    rows.extend(lock(&s.rows).iter().cloned());
    for (hash, busy) in lock(&slow::BUSY).iter() {
        rows.push(Row { hash: hash.clone(), name: busy.name.clone(), kind: "", words: busy.words, slow: busy.slow, status: Status::GaveUp(busy.since.elapsed().as_secs()) });
    }
    rows.sort_by(|a, c| a.hash.cmp(&c.hash));
    rows
}

/// Whether the bake is over: every stage done, or only slow compiles left
/// and the oldest has run past [`GIVE_UP`].
fn over(s: &Shared) -> bool {
    if lock(&s.rows).len() >= s.total {
        return true;
    }
    let busy = lock(&slow::BUSY);
    lock(&s.queue).is_empty() && !busy.is_empty() && busy.values().all(|c| c.slow) && busy.values().any(|c| c.since.elapsed() >= GIVE_UP)
}

/// Once a second, compiles past a minute turn slow and a bake thread takes
/// each one's place; progress every 5 s (with what is compiling when
/// nothing finished); the pipeline cache and the index every 60 s;
/// returns when the bake is over.
fn watch(b: &'static Backend, s: &Arc<Shared>, before: &[Row], path: &Path, started: Instant) {
    let (mut saved, mut last) = (Instant::now(), 0usize);
    let mut tick = 0u64;
    while !over(s) {
        std::thread::sleep(Duration::from_secs(1));
        slow::watch(b, &|owner| {
            if owner == Owner::Bake {
                let s = s.clone();
                std::thread::spawn(move || work(b, &s, 0, Owner::Bake));
            }
        });
        tick += 1;
        if !tick.is_multiple_of(5) {
            continue;
        }
        let n = lock(&s.rows).len();
        let secs = started.elapsed().as_secs_f64();
        let eta = if n == 0 { 0.0 } else { secs / n as f64 * s.total.saturating_sub(n) as f64 };
        println!("bake: {n} of {} ({:.0}%), {secs:.0} s so far, ETA {eta:.0} s", s.total, 100.0 * n as f64 / s.total.max(1) as f64);
        if n == last {
            for (hash, c) in lock(&slow::BUSY).iter() {
                let lane = if c.slow { " (slow)" } else { "" };
                println!("bake:   compiling {} {} ({} words) for {} s{lane}", short(hash), c.name, c.words, c.since.elapsed().as_secs());
            }
        }
        last = n;
        if saved.elapsed() >= Duration::from_secs(60) {
            let _ = std::fs::write(path, index_text(&all_rows(s, before)));
            saved = Instant::now();
        }
    }
}

/// The game's background bake: one thread at the lowest priority bakes
/// what no bake did yet, from the stage workers' start on.
pub(super) fn in_background(b: &'static Backend) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-vk-bake".into()).spawn(move || {
            let (shared, before) = prepare(b);
            tracing::info!(left = shared.total, baked_before = before.len(), "Vulkan: background bake of the stage cache");
            work(b, &shared, -2, Owner::Background);
            tracing::info!(baked = lock(&shared.rows).len(), "Vulkan: background bake done");
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
        Status::Compiled(ms) => format!("compiled {ms} ms"),
        Status::Cached(ms) => format!("cached {ms} ms"),
        Status::Refused(w) => format!("refused: {w}"),
        Status::Failed(w) => format!("failed: {w}"),
        Status::GaveUp(secs) => format!("not compiled: still compiling after {secs} s"),
        Status::Before(t) => format!("{t} (earlier bake)"),
    }
}

/// The outcome's class: compiled, cached, refused, failed, slow (given
/// up), from this bake's status or an earlier bake's text.
fn class(s: &Status) -> usize {
    let text = status_text(s);
    ["compiled", "cached", "refused", "failed", "not compiled"].iter().position(|w| text.starts_with(w)).unwrap_or(0)
}

fn percentile(sorted: &[u64], p: usize) -> u64 {
    sorted.get((sorted.len().saturating_sub(1)) * p / 100).copied().unwrap_or(0)
}

fn summary(rows: &[Row]) -> String {
    let mut by: BTreeMap<&str, [usize; 6]> = BTreeMap::new();
    for r in rows {
        let k = by.entry(if r.kind.is_empty() { "?" } else { r.kind }).or_default();
        k[class(&r.status)] += 1;
        k[5] += 1;
    }
    let mut compiled: Vec<u64> = rows.iter().filter(|r| matches!(r.status, Status::Compiled(_))).filter_map(|r| ms_of(&r.status)).collect();
    compiled.sort_unstable();
    let cached = rows.iter().filter(|r| matches!(r.status, Status::Cached(_))).count();
    let mut out = format!("Stages known: {}\n\n| kind | stages | compiled | cached | refused | failed | slow lane, not compiled |\n|---|---|---|---|---|---|---|\n", rows.len());
    for (k, c) in &by {
        out += &format!("| {k} | {} | {} | {} | {} | {} | {} |\n", c[5], c[0], c[1], c[2], c[3], c[4]);
    }
    out += &format!(
        "\nCompile ms (compiled by this bake): total {}, p50 {}, p90 {}, max {}. Cache hits in this bake: {} of {} compiles (a compile of {HIT_MS} ms or less). Slow lane: stages over {} SPIR-V words compile one at a time.\n",
        compiled.iter().sum::<u64>(),
        percentile(&compiled, 50),
        percentile(&compiled, 90),
        compiled.last().copied().unwrap_or(0),
        cached,
        cached + compiled.len(),
        stages::SLOW_WORDS
    );
    out
}

fn short(h: &str) -> &str {
    &h[..16.min(h.len())]
}

fn index_text(rows: &[Row]) -> String {
    let mut out = String::from("# Pipeline index\n\nGenerated by `victory bake` from the Vulkan stage cache (this translator's key). Not edited by hand.\n\n## Summary\n\n");
    out += &summary(rows);
    out += "\nThe title pipelines that pair stages are not listed: the Vulkan host keeps no catalog of them.\n\n## Stages\n\n| hash | program | kind | words | lane | status |\n|---|---|---|---|---|---|\n";
    for r in rows {
        let lane = if r.slow { "slow lane" } else { "" };
        out += &format!("| {} | {} | {} | {} | {lane} | {} |\n", short(&r.hash), r.name, r.kind, r.words, status_text(&r.status));
    }
    let mut reasons: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for r in rows.iter().filter(|r| matches!(class(&r.status), 2..=4)) {
        let why = status_text(&r.status).trim_end_matches(" (earlier bake)").to_string();
        let why = if class(&r.status) == 4 { "slow lane: not compiled when the bake ended".to_string() } else { why };
        reasons.entry(why).or_default().insert(format!("{} {} ({} words)", short(&r.hash), r.name, r.words));
    }
    out += "\n## Refusals, failures and the slow lane\n";
    for (why, stages) in &reasons {
        out += &format!("\n### {why}\n\n");
        for s in stages {
            out += &format!("- {s}\n");
        }
    }
    let mut slow: Vec<&Row> = rows.iter().filter(|r| ms_of(&r.status).is_some()).collect();
    slow.sort_by(|a, c| ms_of(&c.status).cmp(&ms_of(&a.status)).then(a.hash.cmp(&c.hash)));
    out += "\n## The 30 slowest compiles (this bake)\n\n| hash | program | kind | words | ms |\n|---|---|---|---|---|\n";
    for r in slow.iter().take(30) {
        out += &format!("| {} | {} | {} | {} | {} |\n", short(&r.hash), r.name, r.kind, r.words, ms_of(&r.status).unwrap_or(0));
    }
    out
}
