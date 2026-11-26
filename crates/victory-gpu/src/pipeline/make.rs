//! Host pipelines are made by workers, not by the threads that record the
//! title's lists. Translating a shader and the host driver's compile take
//! from milliseconds to seconds each (structurisation of the biggest
//! compute programs ~0.4 s, some driver compiles 3.4 s), and the title's
//! first world frame wants over 1,200 new pipelines (decision 0007,
//! "Pipelines made off the title's threads").
//!
//! One queue, two priorities:
//!
//! - **For draws**: a draw or dispatch found its pipeline not made. These
//!   go first, on any worker.
//! - **Early**: pipelines an earlier run drew with, queued when the title
//!   creates them (during loading). Only [`EARLY_WORKERS`] of the workers
//!   take these, and only while no pipeline for a draw is queued, so the
//!   others are free the moment a draw needs one; early work goes on in
//!   the background, a few at a time, through loading and into the world.
//!
//! Workers run below the title's threads and the list workers (those that
//! take early work lower still), so a compile takes only the processors
//! they leave.
//!
//! A draw whose pipeline is not made is dropped (a brief pop-in), never
//! waited for beyond [`WAIT_BUDGET`] per frame: a list worker that waits
//! stalls its list, the submission waits for the list, and the frame with
//! it. With half a second per frame (as before), the world's start ran at
//! 300 to 400 ms a frame while the list workers took about 1% of the
//! process's samples each: standing still, not short of processors (the
//! frame line's "waited for pipelines" shows how long).
//!
//! No lock of ours is held across a compile: the queue's lock is taken to
//! queue and to take a pipeline only, a made pipeline is handed out before
//! it is stored in the library, and a store never waits for a save
//! (`library::pipeline`).

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
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
const FOR_EARLY: u8 = 2;

thread_local! {
    static MAKING_FOR: Cell<u8> = const { Cell::new(ON_FRAME_THREAD) };
}

/// Since the last frame line, per [`ON_FRAME_THREAD`], [`FOR_DRAWS`],
/// [`FOR_EARLY`]: nanoseconds making pipelines (translation, library load
/// or driver compile) and pipelines made.
static MAKING_NS: [[AtomicU64; 2]; 3] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; 3];

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
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
        self.made.get_or_init(|| {
            ran = true;
            HostPipeline::new(&self.shaders, self.graphics.as_deref())
        });
        if ran {
            let sums = &MAKING_NS[usize::from(MAKING_FOR.with(Cell::get))];
            sums[0].fetch_add(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
            sums[1].fetch_add(1, Ordering::Relaxed);
        }
        let (lock, done) = &DONE;
        let _held = super::lock(lock);
        done.notify_all();
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads together, for a
/// pipeline a draw wants that a worker is making or is about to: enough
/// for one loaded from the library or the driver's cache, so that menus
/// and one-off passes whose pipelines are quick to make are not lost; a
/// draw whose pipeline takes longer is dropped. (Half a second, as before,
/// held the frame for as long whenever pipelines were being made.)
const WAIT_BUDGET: Duration = Duration::from_millis(3);

/// Pipelines queued or being made, not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Since the last frame line: nanoseconds and times frame threads waited
/// for a pipeline ([`wait`]); draws and dispatches dropped because their
/// pipeline was still being made, and because it has none.
static WAITED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static DROPPED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        // Only when a worker is on it or will be at once: one far down the
        // queue cannot be made within the budget.
        if p.queue_state.load(Ordering::Relaxed) == MAKING
            || NOW_QUEUED.load(Ordering::Relaxed) <= THREADS[0].load(Ordering::Relaxed) as u64
        {
            wait(p, crate::frame::current());
        }
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

/// At the title's creation of a pipeline: queues it as early work if an
/// earlier run drew with it (during loading, so the world's first frames
/// find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let budget = WAIT_BUDGET;
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now).
    static SPENT: Mutex<(u64, Duration)> = Mutex::new((u64::MAX, Duration::ZERO));
    let left = {
        let mut spent = super::lock(&SPENT);
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1);
        spent.1 += left;
        left
    };
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed();
    WAITED[0].fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
    WAITED[1].fetch_add(1, Ordering::Relaxed);
    let waited = waited.min(left);
    {
        let mut spent = super::lock(&SPENT);
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            spent.1 = spent.1.saturating_sub(left - waited);
        }
    }
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

struct Queue {
    early: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
    /// Pipelines queued early and not promoted or taken yet (`early` also
    /// holds promoted ones, passed over when they come up).
    early_queued: usize,
}

/// Mirrors of the queue's lengths for draws and early, read without the
/// lock (the frame line, [`ready`]).
static NOW_QUEUED: AtomicU64 = AtomicU64::new(0);
static EARLY_QUEUED: AtomicU64 = AtomicU64::new(0);

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            requested: VecDeque::new(),
            early_queued: 0,
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
            if prev == EARLY {
                self.early_queued -= 1;
            }
            self.requested.push_back(p.clone());
        } else {
            self.early_queued += 1;
            self.early.push_back(p.clone());
        }
        self.publish();
        Some(prev == NOT_QUEUED)
    }

    /// The next pipeline to make, and whether it is early work: those for
    /// draws first; early ones only for a worker that takes them, and only
    /// when none is queued for a draw.
    fn pop(&mut self, takes_early: bool) -> Option<(Arc<Pending>, bool)> {
        let job = if let Some(p) = self.requested.pop_front() {
            (p, false)
        } else if takes_early {
            // An early entry can have been promoted and claimed already.
            let p = std::iter::from_fn(|| self.early.pop_front())
                .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)?;
            self.early_queued -= 1;
            (p, true)
        } else {
            return None;
        };
        job.0.queue_state.store(MAKING, Ordering::Relaxed);
        self.publish();
        Some(job)
    }

    fn publish(&self) {
        NOW_QUEUED.store(self.requested.len() as u64, Ordering::Relaxed);
        EARLY_QUEUED.store(self.early_queued as u64, Ordering::Relaxed);
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
/// Waited on by the workers that take only pipelines for draws, and by
/// those that take early ones too.
static FOR_DRAWS_ONLY: Condvar = Condvar::new();
static FOR_ANY: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
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
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // One waiting worker of each kind may take a pipeline for a draw;
        // only the early kind takes early work.
        if priority == REQUESTED {
            FOR_DRAWS_ONLY.notify_one();
        }
        FOR_ANY.notify_one();
    }
}

fn next(takes_early: bool) -> (Arc<Pending>, bool) {
    let available = if takes_early { &FOR_ANY } else { &FOR_DRAWS_ONLY };
    let mut queue = super::lock(&QUEUE);
    loop {
        if let Some(job) = queue.pop(takes_early) {
            return job;
        }
        queue = available.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Workers that take early work too (the others only pipelines for draws).
const EARLY_WORKERS: usize = 3;

/// The workers started, and of them those that take early work.
static THREADS: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

/// Runs the calling worker below the title's threads and the list workers;
/// the workers that take early work lower still.
fn set_priority(early: bool) {
    use windows::Win32::System::Threading::*;
    let level = if early { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
    // SAFETY: a plain call on our own thread.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), level);
    }
}

/// Starts the workers (half the logical processors, 2 to 8) on first use;
/// [`EARLY_WORKERS`] of them (at most all but one) take early work.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        let early = EARLY_WORKERS.min(workers - 1);
        for i in 0..workers {
            let takes_early = i < early;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    super::compile_thread();
                    // Once: a thread's nice value raised again needs a
                    // limit Linux does not grant by default.
                    set_priority(takes_early);
                    loop {
                        let (p, early) = next(takes_early);
                        MAKING_FOR.with(|m| m.set(if early { FOR_EARLY } else { FOR_DRAWS }));
                        p.make();
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            match spawned {
                Err(e) => tracing::warn!(error = %e, "no pipeline worker"),
                Ok(_) => {
                    THREADS[0].fetch_add(1, Ordering::Relaxed);
                    if takes_early {
                        THREADS[1].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        let started = THREADS.each_ref().map(|t| t.load(Ordering::Relaxed));
        tracing::info!(workers = started[0], early = started[1], "pipeline workers started");
        if started[0] > 0 && started[1] == 0 {
            // No worker would take early work: what is queued so is made
            // when a draw asks for it.
            tracing::warn!("no pipeline worker takes early work");
        }
        started[0] > 0
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
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
    format!(
        "pipelines: {} being made ({} queued for draws, {} queued early; {} workers, {} take early work); \
         making ms/frame summed (n/frame): for draws {:.1} ({:.2}), early {:.1} ({:.2}), on frame threads {:.1} ({:.2}); \
         {made} compiled by the driver and {loaded} loaded from the library since the last line; \
         frame threads waited for pipelines {waited_ms:.1} ms/frame ({waits:.2} waits), for pipeline locks {:.2} ms/frame ({:.2}); \
         workers waited for pipeline locks {:.2} ms/frame ({:.2}); \
         draws and dispatches dropped without a host pipeline: {being_made:.1}/frame still being made, {refused:.1}/frame with none",
        WAITING.load(Ordering::Relaxed),
        NOW_QUEUED.load(Ordering::Relaxed),
        EARLY_QUEUED.load(Ordering::Relaxed),
        THREADS[0].load(Ordering::Relaxed),
        THREADS[1].load(Ordering::Relaxed),
        making[usize::from(FOR_DRAWS)].0,
        making[usize::from(FOR_DRAWS)].1,
        making[usize::from(FOR_EARLY)].0,
        making[usize::from(FOR_EARLY)].1,
        making[usize::from(ON_FRAME_THREAD)].0,
        making[usize::from(ON_FRAME_THREAD)].1,
        frame_locks.0,
        frame_locks.1,
        worker_locks.0,
        worker_locks.1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert_eq!(queue.early_queued, 1);

        let (p, early) = queue.pop(true).unwrap();
        assert!(Arc::ptr_eq(&p, &requested) && !early);
        let (p, early) = queue.pop(true).unwrap();
        assert!(Arc::ptr_eq(&p, &promoted) && !early);
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        let (p, early) = queue.pop(true).unwrap();
        assert!(Arc::ptr_eq(&p, &earlier) && early);
        assert!(queue.pop(true).is_none());
        assert_eq!(queue.early_queued, 0);
    }

    /// A worker that takes only pipelines for draws leaves early work to
    /// the others, and early work waits while one for a draw is queued.
    #[test]
    fn early_work_goes_to_its_workers_after_draws() {
        let early = Arc::new(Pending::new(Vec::new(), None));
        let drawn = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();
        queue.push(&early, EARLY);
        assert!(queue.pop(false).is_none(), "a worker for draws took early work");
        queue.push(&drawn, REQUESTED);
        let (p, is_early) = queue.pop(true).unwrap();
        assert!(Arc::ptr_eq(&p, &drawn) && !is_early, "early work went before a draw's");
        let (p, is_early) = queue.pop(true).unwrap();
        assert!(Arc::ptr_eq(&p, &early) && is_early);
    }
}
