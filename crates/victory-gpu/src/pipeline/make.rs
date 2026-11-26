//! Host pipelines are made by workers, not by the threads that record the
//! title's lists. Translating a shader and the host driver's compile take
//! from milliseconds to seconds each (0.5 to 10 s of the NVIDIA compiler
//! for the biggest translated programs), and the world wants 2,000 to
//! 3,000 pipelines (decision 0007, "Pipelines made off the title's
//! threads").
//!
//! One queue, three priorities:
//!
//! - **For draws**: a draw or dispatch found its pipeline not made. These
//!   go first, on any worker.
//! - **Created**: the title created a pipeline an earlier run used (the
//!   catalog, `super::catalog`): it creates them while it loads, ahead of
//!   use.
//! - **Bake**: the catalog's pipelines the title has not created yet, in
//!   the order an earlier run first used them (`super::bake`). Made to
//!   fill the caches (the library, vkd3d-proton's and the driver's) and
//!   let go: the title's creation then loads it from them.
//!
//! The last two are background work: only [`BACKGROUND_WORKERS`] of the
//! workers take it (all of them in the standalone bake), at the lowest
//! priority, and only while no pipeline for a draw is queued; their
//! compiles in flight stay under a memory ceiling ([`reserve`]).
//!
//! A draw whose pipeline is not made waits for it, up to [`WAIT_PER_FRAME`]
//! a frame over all threads together ([`wait`]); only then is it dropped.
//! A wait holds the list, the submission and the title's progress on the
//! card: bounded so that the title's own watchdog (about 30 s without
//! progress) never fires. Dropping after 3 ms instead (efa4886) skipped
//! one-off and temporal passes (the exposure chain) and left the world
//! black.
//!
//! No lock of ours is held across a compile: the queue's lock is taken to
//! queue and to take a pipeline only, a made pipeline is handed out before
//! it is stored in the library, and a store never waits for a save
//! (`library::pipeline`).

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use super::Shader;
use super::bake;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

/// What the thread making a pipeline is doing it for ([`MAKING_NS`]).
const ON_FRAME_THREAD: u8 = 0;
const FOR_DRAWS: u8 = 1;
const FOR_CREATED: u8 = 2;
const FOR_BAKE: u8 = 3;

thread_local! {
    static MAKING_FOR: Cell<u8> = const { Cell::new(ON_FRAME_THREAD) };
}

/// Since the last frame line, per [`ON_FRAME_THREAD`], [`FOR_DRAWS`],
/// [`FOR_CREATED`], [`FOR_BAKE`]: nanoseconds making pipelines
/// (translation, library load or driver compile) and pipelines made.
static MAKING_NS: [[AtomicU64; 2]; 4] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; 4];

/// Counts a pipeline made by this thread since `began`.
fn count_making(began: Instant) {
    let sums = &MAKING_NS[usize::from(MAKING_FOR.with(Cell::get))];
    sums[0].fetch_add(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
    sums[1].fetch_add(1, Ordering::Relaxed);
}

/// Whether this thread is making a pipeline as background work.
fn in_background() -> bool {
    matches!(MAKING_FOR.with(Cell::get), FOR_CREATED | FOR_BAKE)
}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(
            graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        Pending {
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        let began = Instant::now();
        let mut ran = false;
        // Translated first (or taken from the cache): the compile's memory
        // is reckoned from the code's size. Background work waits for room
        // before it takes the pipeline, so that a draw's worker that wants
        // the same pipeline never waits behind that.
        let room = || reserve(estimate(self.shaders.iter().filter_map(|s| s.translated())));
        let mut early = (in_background() && self.made.get().is_none()).then(room);
        self.made.get_or_init(|| {
            ran = true;
            let room = early.take().unwrap_or_else(room);
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            drop(room);
            if made.is_some() {
                super::catalog::note_made(&self.key, &bake::stamp());
            }
            made
        });
        if ran {
            count_making(began);
        }
        let (lock, done) = &DONE;
        let _held = super::lock(lock);
        done.notify_all();
    }

    /// What the catalog keeps of it (`super::catalog::Desc`).
    fn desc(&self) -> Option<super::catalog::Desc> {
        if self.shaders.is_empty() {
            return None;
        }
        Some(super::catalog::Desc {
            shaders: self
                .shaders
                .iter()
                .map(|s| (s.stage, s.full, s.name.clone()))
                .collect(),
            graphics: self.graphics.as_ref().map(|g| g.encode()),
        })
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads together, for the
/// pipelines its draws want: a draw whose pipeline is not made by then is
/// dropped. A frame that does not end gets this much again only after
/// [`WAIT_REST`] without waiting, so the card makes progress at least a
/// third of the time.
const WAIT_PER_FRAME: Duration = Duration::from_secs(2);
const WAIT_REST: Duration = Duration::from_secs(1);

/// Pipelines of the title queued or being made, not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Since the last frame line: nanoseconds and times frame threads waited
/// for a pipeline ([`wait`]); draws and dispatches dropped because their
/// pipeline was still being made, and because it has none.
static WAITED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static DROPPED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The host pipeline, made now or within the frame's waiting
/// ([`WAIT_PER_FRAME`]); queues it first for draws.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // The first use this run: listed in the catalog (with when), so the
    // next launch bakes it in this order.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::catalog::note_used(&p.key, crate::frame::current(), || p.desc());
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// A draw or dispatch with `p` was dropped for want of a host pipeline.
pub(crate) fn dropped(p: &Pending) {
    DROPPED[usize::from(p.made.get().is_some())].fetch_add(1, Ordering::Relaxed);
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// The title's pipelines an earlier run used, by key, as the title created
/// them: the bake makes one of these as the title's own (kept) rather than
/// a copy of its own (let go).
static CREATED: Mutex<Option<HashMap<String, Weak<Pending>>>> = Mutex::new(None);

/// At the title's creation of a pipeline: queued at once, ahead of the
/// bake, if an earlier run used it (the title creates them while it loads,
/// ahead of use). Others are made at their first draw, if any comes: the
/// title creates tens of thousands it never draws with.
pub(crate) fn created(p: &Arc<Pending>) {
    if p.shaders.is_empty() || !super::catalog::known(&p.key) {
        return;
    }
    super::lock(&CREATED)
        .get_or_insert_with(HashMap::new)
        .insert(p.key.clone(), Arc::downgrade(p));
    super::library::EARLY.fetch_add(1, Ordering::Relaxed);
    start(p, CREATED_EARLY);
}

/// The title's own pipeline with `key`, if it created one ([`created`]).
pub(crate) fn title_pipeline(key: &str) -> Option<Arc<Pending>> {
    super::lock(&CREATED).as_ref()?.get(key)?.upgrade()
}

/// Makes the title's pipeline `p` for the bake (on a worker taking
/// background work); whether it has a host pipeline.
pub(crate) fn make_for_bake(p: &Arc<Pending>) -> bool {
    now(p)
}

/// Waits for `p` while frame `frame` has waiting left ([`WAIT_PER_FRAME`]).
fn wait(p: &Pending, frame: u64) {
    // The frame the waiting belongs to, and when its waiting ends.
    static WINDOW: Mutex<(u64, Option<Instant>)> = Mutex::new((u64::MAX, None));
    let deadline = {
        let mut window = super::lock(&WINDOW);
        let now = Instant::now();
        let fresh = window.0 != frame || window.1.is_none_or(|end| now >= end + WAIT_REST);
        if fresh {
            *window = (frame, Some(now + WAIT_PER_FRAME));
        }
        window.1.unwrap_or(now)
    };
    if Instant::now() >= deadline {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = deadline.saturating_duration_since(Instant::now());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    drop(held);
    WAITED[0].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    WAITED[1].fetch_add(1, Ordering::Relaxed);
}

/// The memory a compile of `translated` may take in the host driver's
/// compiler, reckoned from the code's size: 48 MiB plus 512 bytes a byte
/// of DXIL (the biggest translated programs, 600 to 800 KiB, take some
/// hundreds of MB in the NVIDIA compiler).
pub(crate) fn estimate<'a>(translated: impl Iterator<Item = &'a super::Translated>) -> u64 {
    (48 << 20)
        + translated
            .map(|t| t.host.dxil.len() as u64 * 512)
            .sum::<u64>()
}

/// Memory reckoned for the compiles in flight ([`estimate`]), the most at
/// once, and the ceiling background work keeps under: [`IN_GAME_CEILING`],
/// or [`STANDALONE_CEILING`] in the standalone bake.
static RESERVED: Mutex<u64> = Mutex::new(0);
static ROOM: Condvar = Condvar::new();
pub(crate) static RESERVED_PEAK: AtomicU64 = AtomicU64::new(0);
const IN_GAME_CEILING: u64 = 1 << 30;
const STANDALONE_CEILING: u64 = 3 << 30;

pub(crate) fn ceiling() -> u64 {
    if bake::is_standalone() {
        STANDALONE_CEILING
    } else {
        IN_GAME_CEILING
    }
}

/// Memory reckoned for one compile in flight, given back when dropped.
pub(crate) struct Reservation(u64);

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut reserved = super::lock(&RESERVED);
        *reserved = reserved.saturating_sub(self.0);
        ROOM.notify_all();
    }
}

/// Reckons `bytes` for a compile about to start. Background work waits
/// until it fits under the [`ceiling`] (or nothing else is in flight); a
/// compile a draw waits for never waits here, though it counts.
pub(crate) fn reserve(bytes: u64) -> Reservation {
    let mut reserved = super::lock(&RESERVED);
    if in_background() {
        let ceiling = ceiling();
        while *reserved > 0 && *reserved + bytes > ceiling {
            reserved = ROOM.wait(reserved).unwrap_or_else(|e| e.into_inner());
        }
    }
    *reserved += bytes;
    RESERVED_PEAK.fetch_max(*reserved, Ordering::Relaxed);
    Reservation(bytes)
}

/// The memory reckoned for the compiles in flight now.
pub(crate) fn reserved() -> u64 {
    *super::lock(&RESERVED)
}

const NOT_QUEUED: u8 = 0;
const CREATED_EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// A worker's next job.
enum Job {
    /// A title pipeline; whether it is background work.
    Title(Arc<Pending>, bool),
    /// A catalog pipeline the bake makes.
    Bake(Arc<bake::Job>),
}

struct Queue {
    created: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
    bake: VecDeque<Arc<bake::Job>>,
    /// Pipelines queued at creation and not promoted or taken yet
    /// (`created` also holds promoted ones, passed over when they come up).
    created_queued: usize,
}

/// Mirrors of the queue's lengths for draws, created and bake, read
/// without the lock (the frame line).
static NOW_QUEUED: AtomicU64 = AtomicU64::new(0);
static CREATED_QUEUED: AtomicU64 = AtomicU64::new(0);
static BAKE_QUEUED: AtomicU64 = AtomicU64::new(0);

impl Queue {
    const fn new() -> Self {
        Self {
            created: VecDeque::new(),
            requested: VecDeque::new(),
            bake: VecDeque::new(),
            created_queued: 0,
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED {
            if prev == CREATED_EARLY {
                self.created_queued -= 1;
            }
            self.requested.push_back(p.clone());
        } else {
            self.created_queued += 1;
            self.created.push_back(p.clone());
        }
        self.publish();
        Some(prev == NOT_QUEUED)
    }

    /// The next job: pipelines for draws first; background work (created,
    /// then bake) only for a worker that takes it, and only when none is
    /// queued for a draw.
    fn pop(&mut self, background: bool) -> Option<Job> {
        let job = if let Some(p) = self.requested.pop_front() {
            Job::Title(p, false)
        } else if !background {
            return None;
        } else if let Some(p) = std::iter::from_fn(|| self.created.pop_front())
            .find(|p| p.queue_state.load(Ordering::Relaxed) == CREATED_EARLY)
        {
            // (An entry can have been promoted and claimed already.)
            self.created_queued -= 1;
            Job::Title(p, true)
        } else {
            Job::Bake(self.bake.pop_front()?)
        };
        if let Job::Title(p, _) = &job {
            p.queue_state.store(MAKING, Ordering::Relaxed);
        }
        self.publish();
        Some(job)
    }

    fn publish(&self) {
        NOW_QUEUED.store(self.requested.len() as u64, Ordering::Relaxed);
        CREATED_QUEUED.store(self.created_queued as u64, Ordering::Relaxed);
        BAKE_QUEUED.store(self.bake.len() as u64, Ordering::Relaxed);
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
/// Waited on by the workers that take only pipelines for draws, and by
/// those that take background work too.
static FOR_DRAWS_ONLY: Condvar = Condvar::new();
static FOR_ANY: Condvar = Condvar::new();

/// Queues or promotes `p`. Queue changes and worker claims happen under
/// the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = super::lock(&QUEUE);
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(
                WAITING.fetch_add(1, Ordering::Relaxed) + 1,
                Ordering::Relaxed,
            );
        }
        // One waiting worker of each kind may take a pipeline for a draw;
        // only the background kind takes the rest.
        if priority == REQUESTED {
            FOR_DRAWS_ONLY.notify_one();
        }
        FOR_ANY.notify_one();
    }
}

/// Queues the bake's jobs, in order, behind everything else.
pub(crate) fn queue_bake(jobs: Vec<Arc<bake::Job>>) -> bool {
    if !workers() {
        return false;
    }
    let mut queue = super::lock(&QUEUE);
    queue.bake.extend(jobs);
    queue.publish();
    FOR_ANY.notify_all();
    true
}

fn next(background: bool) -> Job {
    let available = if background {
        &FOR_ANY
    } else {
        &FOR_DRAWS_ONLY
    };
    let mut queue = super::lock(&QUEUE);
    loop {
        if let Some(job) = queue.pop(background) {
            return job;
        }
        queue = available.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Workers that take background work in the game (the others only
/// pipelines for draws).
const BACKGROUND_WORKERS: usize = 4;

/// The workers started, those that take background work, and those
/// making a pipeline now (and the most at once).
static THREADS: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];
pub(crate) static BUSY: AtomicUsize = AtomicUsize::new(0);
pub(crate) static BUSY_PEAK: AtomicUsize = AtomicUsize::new(0);

/// The workers started, and of them those that take background work.
pub(crate) fn threads() -> (usize, usize) {
    (
        THREADS[0].load(Ordering::Relaxed),
        THREADS[1].load(Ordering::Relaxed),
    )
}

/// Runs the calling worker below the title's threads and the list workers;
/// those that take background work lower still. In the standalone bake,
/// at normal priority.
fn set_priority(background: bool) {
    use windows::Win32::System::Threading::*;
    if bake::is_standalone() {
        return;
    }
    let level = if background {
        THREAD_PRIORITY_LOWEST
    } else {
        THREAD_PRIORITY_BELOW_NORMAL
    };
    // SAFETY: a plain call on our own thread.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), level);
    }
}

/// Starts the workers on first use: in the game half the logical
/// processors, 2 to 8, [`BACKGROUND_WORKERS`] of them (at most all but
/// one) taking background work; in the standalone bake its job count, all
/// taking everything.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let (workers, background) = match bake::standalone_jobs() {
            Some(jobs) => (jobs.max(1), jobs.max(1)),
            None => {
                let workers =
                    std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
                (workers, BACKGROUND_WORKERS.min(workers - 1))
            }
        };
        for i in 0..workers {
            let takes_background = i < background;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    super::compile_thread();
                    // Once: a thread's nice value raised again needs a
                    // limit Linux does not grant by default.
                    set_priority(takes_background);
                    loop {
                        let job = next(takes_background);
                        let busy = BUSY.fetch_add(1, Ordering::Relaxed) + 1;
                        BUSY_PEAK.fetch_max(busy, Ordering::Relaxed);
                        match job {
                            Job::Title(p, background) => {
                                MAKING_FOR.with(|m| {
                                    m.set(if background { FOR_CREATED } else { FOR_DRAWS })
                                });
                                p.make();
                                if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                                    drained();
                                } else {
                                    super::library::save(false);
                                }
                            }
                            Job::Bake(job) => {
                                MAKING_FOR.with(|m| m.set(FOR_BAKE));
                                let began = Instant::now();
                                if bake::run(&job) {
                                    count_making(began);
                                }
                                super::library::save(false);
                            }
                        }
                        BUSY.fetch_sub(1, Ordering::Relaxed);
                    }
                });
            match spawned {
                Err(e) => tracing::warn!(error = %e, "no pipeline worker"),
                Ok(_) => {
                    THREADS[0].fetch_add(1, Ordering::Relaxed);
                    if takes_background {
                        THREADS[1].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        let started = THREADS.each_ref().map(|t| t.load(Ordering::Relaxed));
        tracing::info!(
            workers = started[0],
            background = started[1],
            "pipeline workers started"
        );
        if started[0] > 0 && started[1] == 0 {
            // No worker would take background work: what is queued so is
            // made when a draw asks for it.
            tracing::warn!("no pipeline worker takes background work");
        }
        started[0] > 0
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// Host pipelines made (or loaded) with per-lane arrays since the start,
/// and the largest such array in bytes a lane. NVIDIA keeps these in local
/// memory reserved for every lane the card can hold at once (36 SMs x 1,536
/// lanes on the RTX 5060 Ti: 18 KiB a lane is about 1 GB), outside the
/// card memory the budget counts.
static LOCAL: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A host pipeline was made whose largest stage keeps `local_array_bytes`
/// a lane ([`LOCAL`]).
pub(crate) fn made_with(local_array_bytes: u32) {
    if local_array_bytes > 0 {
        LOCAL[0].fetch_add(1, Ordering::Relaxed);
        LOCAL[1].fetch_max(u64::from(local_array_bytes), Ordering::Relaxed);
    }
}

/// The workers ran out of the title's pipelines: logged after a batch of
/// 10 or more, and the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            queued_at_creation = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

/// The pipelines' part of the `frame time:` line, per frame of `frames`
/// since the last line.
pub(crate) fn frame_text(frames: f64) -> String {
    use super::library::{LOADED, MADE};
    let per = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / frames;
    let ms = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames;
    let making = MAKING_NS.each_ref().map(|m| (ms(&m[0]), per(&m[1])));
    let [frame_locks, worker_locks] = super::lock_waits(frames);
    let (waited_ms, waits) = (ms(&WAITED[0]), per(&WAITED[1]));
    let (being_made, refused) = (per(&DROPPED[0]), per(&DROPPED[1]));
    // Driver compiles and library loads since the last line.
    static SEEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let since = |total: &AtomicU64, seen: &AtomicU64| {
        let now = total.load(Ordering::Relaxed);
        now.saturating_sub(seen.swap(now, Ordering::Relaxed))
    };
    let (made, loaded) = (since(&MADE, &SEEN[0]), since(&LOADED, &SEEN[1]));
    let part = |i: u8| making[usize::from(i)];
    format!(
        "pipelines: {} being made ({} queued for draws, {} queued at creation, {} to bake; {} workers, {} take background work; {}); \
         making ms/frame summed (n/frame): for draws {:.1} ({:.2}), at creation {:.1} ({:.2}), bake {:.1} ({:.2}), on frame threads {:.1} ({:.2}); \
         {made} compiled by the driver and {loaded} loaded from the library since the last line; \
         frame threads waited for pipelines {waited_ms:.1} ms/frame ({waits:.2} waits), for pipeline locks {:.2} ms/frame ({:.2}); \
         workers waited for pipeline locks {:.2} ms/frame ({:.2}); \
         draws and dispatches dropped without a host pipeline: {being_made:.1}/frame still being made (after waiting), {refused:.1}/frame with none; \
         made with per-lane arrays (local memory): {} pipelines, the largest {} bytes a lane",
        WAITING.load(Ordering::Relaxed),
        NOW_QUEUED.load(Ordering::Relaxed),
        CREATED_QUEUED.load(Ordering::Relaxed),
        BAKE_QUEUED.load(Ordering::Relaxed),
        THREADS[0].load(Ordering::Relaxed),
        THREADS[1].load(Ordering::Relaxed),
        bake::summary(),
        part(FOR_DRAWS).0,
        part(FOR_DRAWS).1,
        part(FOR_CREATED).0,
        part(FOR_CREATED).1,
        part(FOR_BAKE).0,
        part(FOR_BAKE).1,
        part(ON_FRAME_THREAD).0,
        part(ON_FRAME_THREAD).1,
        frame_locks.0,
        frame_locks.1,
        worker_locks.0,
        worker_locks.1,
        LOCAL[0].load(Ordering::Relaxed),
        LOCAL[1].load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and gets the worker's answer (here:
    /// no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        let _ = ready(&p);
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn draws_promote_created_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, CREATED_EARLY), Some(true));
        assert_eq!(queue.push(&promoted, CREATED_EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert_eq!(queue.created_queued, 1);

        let pop = |q: &mut Queue| match q.pop(true) {
            Some(Job::Title(p, background)) => Some((p, background)),
            _ => None,
        };
        let (p, background) = pop(&mut queue).unwrap();
        assert!(Arc::ptr_eq(&p, &requested) && !background);
        let (p, background) = pop(&mut queue).unwrap();
        assert!(Arc::ptr_eq(&p, &promoted) && !background);
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        let (p, background) = pop(&mut queue).unwrap();
        assert!(Arc::ptr_eq(&p, &earlier) && background);
        assert!(queue.pop(true).is_none());
        assert_eq!(queue.created_queued, 0);
    }

    /// A worker that takes only pipelines for draws leaves background work
    /// to the others, and background work waits while one for a draw is
    /// queued; the bake comes after pipelines the title created.
    #[test]
    fn background_work_goes_to_its_workers_after_draws() {
        let created = Arc::new(Pending::new(Vec::new(), None));
        let drawn = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();
        queue.bake.push_back(Arc::new(bake::Job::test()));
        queue.push(&created, CREATED_EARLY);
        assert!(
            queue.pop(false).is_none(),
            "a worker for draws took background work"
        );
        queue.push(&drawn, REQUESTED);
        assert!(matches!(queue.pop(true), Some(Job::Title(p, false)) if Arc::ptr_eq(&p, &drawn)));
        assert!(matches!(queue.pop(true), Some(Job::Title(p, true)) if Arc::ptr_eq(&p, &created)));
        assert!(matches!(queue.pop(true), Some(Job::Bake(_))));
        assert!(queue.pop(true).is_none());
    }

    /// Background compiles keep under the ceiling; one alone always runs.
    #[test]
    fn background_compiles_keep_under_the_ceiling() {
        MAKING_FOR.with(|m| m.set(FOR_BAKE));
        let first = reserve(ceiling() * 2);
        assert!(
            reserved() >= ceiling() * 2,
            "one compile alone runs whatever its size"
        );
        let waiter = std::thread::spawn(|| {
            MAKING_FOR.with(|m| m.set(FOR_BAKE));
            let _r = reserve(1 << 20);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !waiter.is_finished(),
            "a second compile went past the ceiling"
        );
        drop(first);
        waiter.join().unwrap();
        MAKING_FOR.with(|m| m.set(ON_FRAME_THREAD));
    }
}
