//! Slow stage compiles, by time (game and bake).
//!
//! Some stages keep NVIDIA's compiler busy for many minutes whatever their
//! size (CS_ProjDecalReduceClassification_Vectorized, 34,840 words, and
//! CS_RaytracingResolve variants of ~48,000 words ran over 24 minutes each
//! in the bake, while finished compiles take seconds). The rule:
//!
//! - Every compile is registered while it runs ([`compiling`]). One that
//!   has run [`SLOW_AFTER`] is marked slow by the watch ([`watch`], once a
//!   second): its thread is replaced at once (a stage worker or a bake
//!   thread starts in its place and the slow one ends after its compile),
//!   so the others' throughput never drops. The stage is written down as
//!   slow in the bake's outcome file (`stage_cache::mark_baked`), and in
//!   diagnostics builds its SPIR-V goes to `slow-stages/<hash>.spv` beside
//!   the caches.
//! - Stages known to be slow (written down so, or over
//!   `stages::SLOW_WORDS` words) start last and on the slow lane: at most
//!   [`LANE`] slow compiles are alive at a time, those found slow while
//!   compiling included; the others wait their turn ([`lane`]).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::{Backend, stage_cache};

/// A compile running this long is slow (the bake's p99 is far below).
pub(super) const SLOW_AFTER: Duration = Duration::from_secs(60);
/// Slow compiles alive at a time (in the game).
const LANE: usize = 2;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Who runs a compile: a stage worker of the game, a bake thread, the
/// game's background bake.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Owner {
    Worker,
    Bake,
    Background,
}

pub(super) struct Busy {
    pub(super) name: String,
    pub(super) words: usize,
    pub(super) since: Instant,
    pub(super) owner: Owner,
    /// Found slow, or started on the slow lane.
    pub(super) slow: bool,
    /// Diagnostics: its SPIR-V, dumped when it turns slow.
    spirv: Option<Vec<u32>>,
}

/// The compiles running now, by hash (32 hex digits).
pub(super) static BUSY: Mutex<BTreeMap<String, Busy>> = Mutex::new(BTreeMap::new());
/// Slow compiles alive.
static ALIVE: Mutex<usize> = Mutex::new(0);
static FREED: Condvar = Condvar::new();

/// A compile running; on drop it leaves [`BUSY`] (and the slow lane).
pub(super) struct Compiling {
    hash: String,
}

impl Compiling {
    /// Whether it was slow (its thread was replaced: it ends now).
    pub(super) fn finish(self) -> bool {
        let gone = lock(&BUSY).remove(&self.hash);
        std::mem::forget(self);
        let slow = gone.is_some_and(|b| b.slow);
        if slow {
            *lock(&ALIVE) -= 1;
            FREED.notify_all();
        }
        slow
    }
}

impl Drop for Compiling {
    fn drop(&mut self) {
        if lock(&BUSY).remove(&self.hash).is_some_and(|b| b.slow) {
            *lock(&ALIVE) -= 1;
            FREED.notify_all();
        }
    }
}

/// Registers a compile of `hash` (`slow`: on the slow lane from the start,
/// its turn already taken with [`lane`]).
pub(super) fn compiling(hash: &str, name: &str, spirv: &[u32], owner: Owner, slow: bool) -> Compiling {
    let busy = Busy {
        name: name.to_string(),
        words: spirv.len(),
        since: Instant::now(),
        owner,
        slow,
        spirv: cfg!(feature = "diagnostics").then(|| spirv.to_vec()),
    };
    lock(&BUSY).insert(hash.to_string(), busy);
    Compiling { hash: hash.to_string() }
}

/// Waits for a turn on the slow lane (at most [`LANE`] slow compiles
/// alive), then takes it.
pub(super) fn lane() {
    let mut alive = lock(&ALIVE);
    while *alive >= LANE {
        alive = FREED.wait(alive).unwrap_or_else(|e| e.into_inner());
    }
    *alive += 1;
}

/// The stages written down as slow for this device (by an earlier bake or
/// game run), by hash.
pub(super) fn known(b: &Backend) -> &'static BTreeSet<String> {
    static KNOWN: OnceLock<BTreeSet<String>> = OnceLock::new();
    KNOWN.get_or_init(|| {
        stage_cache::baked(&device_key(b))
            .into_iter()
            .filter(|(_, o)| is_slow(o))
            .map(|(h, _)| h)
            .collect()
    })
}

/// Whether an outcome line says slow.
pub(super) fn is_slow(outcome: &str) -> bool {
    outcome.contains("slow")
}

pub(super) fn device_key(b: &Backend) -> String {
    b.host.caps.pipeline_cache_uuid.iter().map(|x| format!("{x:02x}")).collect()
}

/// Once a second: compiles past [`SLOW_AFTER`] turn slow; `replace` is
/// called with the owner of each (to start a thread in its place).
pub(super) fn watch(b: &Backend, replace: &dyn Fn(Owner)) {
    let device = device_key(b);
    let mut turned = Vec::new();
    for (hash, busy) in lock(&BUSY).iter_mut() {
        if busy.slow || busy.since.elapsed() < SLOW_AFTER {
            continue;
        }
        busy.slow = true;
        turned.push((hash.clone(), busy.name.clone(), busy.words, busy.owner, busy.spirv.take()));
    }
    for (hash, name, words, owner, spirv) in turned {
        *lock(&ALIVE) += 1;
        tracing::warn!(hash = %hash, name = %name, words, ?owner, secs = SLOW_AFTER.as_secs(), "Vulkan: a stage compile is slow; its thread is replaced and it goes on alone");
        stage_cache::mark_baked(&device, &hash, &format!("{words}||slow lane|slow: still compiling after {} s", SLOW_AFTER.as_secs()));
        if let Some(words) = spirv {
            dump(&hash, &words);
        }
        replace(owner);
    }
}

/// Diagnostics: the SPIR-V of a slow stage, beside the caches.
fn dump(hash: &str, words: &[u32]) {
    let Some(dir) = crate::pipeline::cache_dir().map(|d| d.with_file_name("slow-stages")) else { return };
    if std::fs::create_dir_all(&dir).is_ok() {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let _ = std::fs::write(dir.join(format!("{hash}.spv")), bytes);
    }
}

/// The game's watch: from the stage workers' start, once a second; a slow
/// stage worker's place goes to a new worker.
pub(super) fn start_watch(b: &'static Backend, spawn_worker: fn(&'static Backend)) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-vk-slow".into()).spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(1));
                watch(b, &|owner| {
                    if owner == Owner::Worker {
                        spawn_worker(b);
                    }
                });
            }
        });
    });
}
