//! Diagnostics (`VICTORY_VK_GPU_TIME=<frame>`): the card's time by program
//! for one frame. Every time a recording binds another pipeline during that
//! frame a timestamp is written; the time to the next timestamp (all lists
//! together, in the order the card ran them) is the bound program's. Two
//! presents later the sums are told, the largest first. Lists that run on
//! other queues at the same time blur the split a little; it is meant to
//! show where seconds go, not microseconds.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_vk::ash::vk;

use super::{Backend, lock};

/// Timestamps a frame takes at most.
const MOST: u32 = 65536;

static POOL: OnceLock<Option<vk::QueryPool>> = OnceLock::new();
static NEXT: AtomicU32 = AtomicU32::new(0);
static MARKS: Mutex<Vec<(u32, String)>> = Mutex::new(Vec::new());

/// The frame timed, if any.
fn timed() -> Option<u64> {
    static FRAME: OnceLock<Option<u64>> = OnceLock::new();
    *FRAME.get_or_init(|| {
        let v = std::env::var("VICTORY_VK_GPU_TIME").ok()?;
        cfg!(feature = "diagnostics").then(|| v.parse().unwrap_or(0))
    })
}

fn pool(b: &Backend) -> Option<vk::QueryPool> {
    *POOL.get_or_init(|| {
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(MOST);
        // SAFETY: a plain query pool of the device.
        unsafe { b.host.device.create_query_pool(&info, None) }.ok()
    })
}

/// In the present before the timed frame: the pool is reset.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
pub(super) unsafe fn arm(b: &Backend, cmd: vk::CommandBuffer, frame: u64) {
    if timed().is_some_and(|f| frame + 1 == f)
        && let Some(pool) = pool(b)
    {
        // SAFETY: as the caller says.
        unsafe { b.host.device.cmd_reset_query_pool(cmd, pool, 0, MOST) };
    }
}

/// A recording binds the pipeline of `name` (its programs' names).
///
/// # Safety
/// Recording into `cmd`.
pub(super) unsafe fn mark(b: &Backend, cmd: vk::CommandBuffer, name: impl FnOnce() -> String) {
    if !timed().is_some_and(|f| super::present::FRAME.load(Ordering::Relaxed) == f) {
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

/// Two presents after the timed frame, the queue idle: the sums are told.
pub(super) fn tell(b: &Backend, frame: u64) {
    if !timed().is_some_and(|f| frame == f + 2) {
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
        tracing::warn!(error = ?e, "Vulkan: the frame's timestamps could not all be read");
    }
    let mut at: Vec<(u64, &str)> = marks.iter().filter(|(i, _)| out[*i as usize][1] != 0).map(|(i, name)| (out[*i as usize][0], name.as_str())).collect();
    at.sort_unstable();
    let ns = |ticks: u64| ticks as f64 * f64::from(b.host.caps.timestamp_period);
    let mut by: HashMap<&str, (f64, u32)> = HashMap::new();
    for w in at.windows(2) {
        let e = by.entry(w[0].1).or_default();
        e.0 += ns(w[1].0 - w[0].0);
        e.1 += 1;
    }
    let total = at.first().zip(at.last()).map_or(0.0, |(a, z)| ns(z.0 - a.0));
    let mut top: Vec<(&str, (f64, u32))> = by.into_iter().collect();
    top.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    tracing::info!(marks = at.len(), total_ms = format_args!("{:.1}", total / 1e6), "Vulkan: the card's time by program for one frame follows");
    for (name, (t, binds)) in top.into_iter().take(40) {
        tracing::info!(ms = format_args!("{:.2}", t / 1e6), share = format_args!("{:.1}%", 100.0 * t / total.max(1.0)), binds, program = name, "Vulkan: card time");
    }
}
