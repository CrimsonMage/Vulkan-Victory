//! Diagnostics (`VICTORY_VK_GPU_TIME=<frame>[,<frames>]`): the card's time by
//! program over a few frames (four unless said), told as time a frame.
//! Every time a recording binds another pipeline in those frames a
//! timestamp is written; the time to the next timestamp (all lists
//! together, in the order the card ran them) is the bound program's. Two
//! presents after the last frame the sums are told, the largest first, by
//! program and by family (the name without its variant hashes). One frame
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

/// The first frame timed and how many frames, if any.
fn timed() -> Option<(u64, u64)> {
    static FRAME: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *FRAME.get_or_init(|| {
        let v = std::env::var("VICTORY_VK_GPU_TIME").ok()?;
        let (first, frames) = v.split_once(',').unwrap_or((&v, "4"));
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
    }
    // SAFETY: as the caller says.
    unsafe { mark(b, cmd, || "host: present".into()) };
}

/// A recording binds the pipeline of `name` (its programs' names).
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
    lock(&MARKS).push((n, name()));
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
    for (name, t, binds) in sums(&at, family).into_iter().take(40) {
        tracing::info!(ms = format_args!("{:.2}", t / a_frame), share = format_args!("{:.1}%", 100.0 * t / total.max(1.0)), binds, family = name, "Vulkan: card time by family");
    }
    for (name, t, binds) in sums(&at, str::to_owned).into_iter().take(60) {
        tracing::info!(ms = format_args!("{:.2}", t / a_frame), share = format_args!("{:.1}%", 100.0 * t / total.max(1.0)), binds, program = name, "Vulkan: card time");
    }
}
