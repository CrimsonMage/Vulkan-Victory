//! Diagnostics (`VICTORY_VK_GPU_TIME=<frame>[,<frames>[,counted]]`): the
//! card's time by program over a few frames (four unless said), told as
//! time a frame.
//! Every time a recording binds another pipeline for a draw, and before
//! every dispatch, a timestamp is written; the time to the next timestamp
//! (all lists together, in the order the card ran them) is the bound
//! program's. A dispatch ends with a timestamp too: what the list does
//! from there to the next program (`after: <program>`: the title's copies
//! and clears, the host's own passes, the next program's preparation) is
//! not the dispatch's; the title's copies, clears and fills are marked
//! themselves (`title: a buffer copy`, ...). With `counted` each
//! dispatch's invocations are counted too (a pipeline statistics query
//! around it; the lists' own statistics, `gpu_stats.rs`, stand back in
//! those frames), so that a program's time can be held against its work
//! and a replay of a captured dispatch against the game's; the query
//! costs 30 to 100 us a dispatch on the test card, so the times of a
//! counted run are not the game's (run cb1 against bh1). Two
//! presents after the last frame the sums are told, the largest first, by
//! program and by family (the name without its variant hashes). A list's
//! end is a timestamp too (`host: between lists`): what follows it until
//! the next list's first program, the card waiting for a list included, is
//! not charged to a program. One frame
//! alone is not a whole frame of the title's: its lists are recorded while
//! the frame before is still on the card (run am1: 107 of 410 ms), so the
//! window is several presents and the sums are divided by their number.
//! Lists that run on other queues at the same time blur the split a
//! little; it is meant to show where seconds go, not microseconds.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_vk::ash::vk;

use super::{Backend, lock};

/// Timestamps the window takes at most.
const MOST: u32 = 1 << 19;

static POOL: OnceLock<Option<vk::QueryPool>> = OnceLock::new();
static NEXT: AtomicU32 = AtomicU32::new(0);
static MARKS: Mutex<Vec<(u32, String)>> = Mutex::new(Vec::new());
/// Dispatches counted at most.
const COUNTED: u32 = 1 << 16;
static COUNT_POOL: OnceLock<Option<vk::QueryPool>> = OnceLock::new();
static COUNT_NEXT: AtomicU32 = AtomicU32::new(0);
static COUNTS: Mutex<Vec<(u32, String)>> = Mutex::new(Vec::new());
// The program the recording thread marked last (lists are recorded side
// by side, each by one thread).
thread_local! {
    static LAST: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static STRAYED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the recording thread's last mark was not a program's (a copy
/// or a clear of the title's, a dispatch's end): a draw with the pipeline
/// still bound marks its program again.
pub(super) fn strayed() -> bool {
    STRAYED.with(std::cell::Cell::get)
}

/// The first frame timed and how many frames, if any.
fn timed() -> Option<(u64, u64)> {
    static FRAME: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *FRAME.get_or_init(|| {
        let v = std::env::var("VICTORY_VK_GPU_TIME").ok()?;
        let mut fields = v.split(',');
        let (first, frames) = (fields.next().unwrap_or(""), fields.next().unwrap_or("4"));
        cfg!(feature = "diagnostics").then(|| (first.trim().parse().unwrap_or(0), frames.trim().parse().unwrap_or(4).max(1)))
    })
}

/// Whether `frame` is one of the frames timed.
fn within(frame: u64) -> bool {
    timed().is_some_and(|(f, n)| frame >= f && frame < f + n)
}

fn pool(b: &Backend) -> Option<vk::QueryPool> {
    *POOL.get_or_init(|| {
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(MOST);
        // SAFETY: a plain query pool of the device.
        unsafe { b.host.device.create_query_pool(&info, None) }.ok()
    })
}

fn count_pool(b: &Backend) -> Option<vk::QueryPool> {
    *COUNT_POOL.get_or_init(|| {
        if b.host.caps.core.pipeline_statistics_query != vk::TRUE {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::PIPELINE_STATISTICS).query_count(COUNTED).pipeline_statistics(vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS);
        // SAFETY: a plain query pool of the device.
        unsafe { b.host.device.create_query_pool(&info, None) }.ok()
    })
}

/// Whether the frame being recorded is timed and its dispatches counted
/// (the lists' own statistics queries stand back: one statistics query at
/// a time in a list).
pub(super) fn counting() -> bool {
    static COUNTED_RUN: OnceLock<bool> = OnceLock::new();
    *COUNTED_RUN.get_or_init(|| std::env::var("VICTORY_VK_GPU_TIME").is_ok_and(|v| v.split(',').nth(2).is_some_and(|w| w.trim() == "counted")))
        && within(super::present::FRAME.load(Ordering::Relaxed))
}

/// Before a dispatch of the program marked last: its invocations are
/// counted from here (the slot to end with, [`dispatched`]).
///
/// # Safety
/// Recording into `cmd`, outside rendering, no statistics query active.
pub(super) unsafe fn dispatching(b: &Backend, cmd: vk::CommandBuffer) -> Option<u32> {
    if !counting() {
        return None;
    }
    let pool = count_pool(b)?;
    let n = COUNT_NEXT.fetch_add(1, Ordering::Relaxed);
    if n >= COUNTED {
        return None;
    }
    // SAFETY: as the caller says; the query was reset a present ago.
    unsafe { b.host.device.cmd_begin_query(cmd, pool, n, vk::QueryControlFlags::empty()) };
    lock(&COUNTS).push((n, LAST.with(|l| l.borrow().clone())));
    Some(n)
}

/// After the dispatch: its count ends, and what follows is not its time.
///
/// # Safety
/// Recording into `cmd`, outside rendering; `slot` from [`dispatching`].
pub(super) unsafe fn dispatched(b: &Backend, cmd: vk::CommandBuffer, slot: Option<u32>) {
    if let (Some(n), Some(pool)) = (slot, count_pool(b)) {
        // SAFETY: as the caller says.
        unsafe { b.host.device.cmd_end_query(cmd, pool, n) };
    }
    // SAFETY: as the caller says.
    unsafe { mark(b, cmd, || format!("after: {}", LAST.with(|l| l.borrow().clone()))) };
}

/// In the present before the first timed frame: the pool is reset. In a
/// present of the window: a mark, so that the time to the next list is
/// not the last program's of the frame before.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
pub(super) unsafe fn arm(b: &Backend, cmd: vk::CommandBuffer, frame: u64) {
    if timed().is_some_and(|(f, _)| frame + 1 == f)
        && let Some(pool) = pool(b)
    {
        // SAFETY: as the caller says.
        unsafe { b.host.device.cmd_reset_query_pool(cmd, pool, 0, MOST) };
        if let Some(counts) = count_pool(b) {
            // SAFETY: as the caller says.
            unsafe { b.host.device.cmd_reset_query_pool(cmd, counts, 0, COUNTED) };
        }
    }
    // SAFETY: as the caller says.
    unsafe { mark(b, cmd, || "host: present".into()) };
}

/// A recording binds the pipeline of `name` (its programs' names), or
/// dispatches with it.
///
/// # Safety
/// Recording into `cmd`.
pub(super) unsafe fn mark(b: &Backend, cmd: vk::CommandBuffer, name: impl FnOnce() -> String) {
    if !within(super::present::FRAME.load(Ordering::Relaxed)) {
        return;
    }
    let Some(pool) = pool(b) else { return };
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    if n >= MOST {
        return;
    }
    // SAFETY: as the caller says; the query was reset a present ago.
    unsafe { b.host.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, n) };
    let name = name();
    let stray = ["after: ", "title: ", "host: "].iter().any(|p| name.starts_with(p));
    STRAYED.with(|s| s.set(stray));
    if !stray {
        LAST.with(|l| l.borrow_mut().clone_from(&name));
    }
    lock(&MARKS).push((n, name));
}

/// A program's name without its variant hashes: `CS_Name_0123abcd` and
/// `PS_Name_89ef4567` pairs become `CS_Name` and `PS_Name`.
fn family(name: &str) -> String {
    let plain = |part: &str| {
        let hash = |w: &&str| w.len() >= 8 && w.bytes().all(|c| c.is_ascii_hexdigit());
        part.split('_').filter(|w| !hash(w)).collect::<Vec<_>>().join("_")
    };
    name.split('+').map(plain).collect::<Vec<_>>().join("+")
}

/// The sums of `at` (timestamps in ns with their programs, in the card's
/// order) by `key`, the largest first, with how often each was bound.
fn sums(at: &[(f64, &str)], key: impl Fn(&str) -> String) -> Vec<(String, f64, u32)> {
    let mut by: HashMap<String, (f64, u32)> = HashMap::new();
    for w in at.windows(2) {
        let e = by.entry(key(w[0].1)).or_default();
        e.0 += w[1].0 - w[0].0;
        e.1 += 1;
    }
    let mut top: Vec<(String, f64, u32)> = by.into_iter().map(|(k, (t, n))| (k, t, n)).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top
}

/// The counted dispatches' invocations and their number, by program.
fn work(b: &Backend) -> HashMap<String, (u64, u32)> {
    let mut work: HashMap<String, (u64, u32)> = HashMap::new();
    let counts = std::mem::take(&mut *lock(&COUNTS));
    let Some(pool) = count_pool(b).filter(|_| !counts.is_empty()) else { return work };
    let mut got = vec![[0u64; 2]; COUNT_NEXT.load(Ordering::Relaxed).min(COUNTED) as usize];
    // SAFETY: that many queries of a 64-bit value and availability each.
    let _ = unsafe { b.host.device.get_query_pool_results(pool, 0, &mut got, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY) };
    for (i, name) in counts.iter().filter(|(i, _)| got[*i as usize][1] != 0) {
        let e = work.entry(name.clone()).or_default();
        e.0 += got[*i as usize][0];
        e.1 += 1;
    }
    work
}

/// Two presents after the last timed frame: the sums are told, a frame.
pub(super) fn tell(b: &Backend, frame: u64) {
    let Some((first, frames)) = timed() else { return };
    if frame != first + frames + 1 {
        return;
    }
    let (Some(pool), marks) = (pool(b), std::mem::take(&mut *lock(&MARKS))) else { return };
    let n = NEXT.load(Ordering::Relaxed).min(MOST);
    if n == 0 {
        return;
    }
    let mut out = vec![[0u64; 2]; n as usize];
    // SAFETY: `n` queries of 64-bit value and availability each.
    let r = unsafe { b.host.device.get_query_pool_results(pool, 0, &mut out, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY) };
    if let Err(e) = r {
        tracing::warn!(error = ?e, "Vulkan: the window's timestamps could not all be read");
    }
    let period = f64::from(b.host.caps.timestamp_period);
    let mut at: Vec<(f64, &str)> =
        marks.iter().filter(|(i, _)| out[*i as usize][1] != 0).map(|(i, name)| (out[*i as usize][0] as f64 * period, name.as_str())).collect();
    at.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total = at.first().zip(at.last()).map_or(0.0, |(a, z)| z.0 - a.0);
    let a_frame = 1e6 * frames as f64;
    tracing::info!(
        marks = at.len(),
        lost = marks.len() - at.len(),
        full = n >= MOST,
        frames,
        ms_a_frame = format_args!("{:.1}", total / a_frame),
        "Vulkan: the card's time by program follows (ms a frame over the frames timed)"
    );
    let work = work(b);
    let worked = |name: &str| work.get(name).copied().unwrap_or_default();
    for (name, t, binds) in sums(&at, family).into_iter().take(40) {
        tracing::info!(ms = format_args!("{:.2}", t / a_frame), share = format_args!("{:.1}%", 100.0 * t / total.max(1.0)), binds, family = name, "Vulkan: card time by family");
    }
    for (name, t, binds) in sums(&at, str::to_owned).into_iter().take(80) {
        let (invocations, dispatches) = worked(&name);
        tracing::info!(
            ms = format_args!("{:.2}", t / a_frame),
            share = format_args!("{:.1}%", 100.0 * t / total.max(1.0)),
            binds,
            dispatches,
            invocations_a_frame = invocations / frames,
            program = name,
            "Vulkan: card time"
        );
    }
}
