//! Diagnostics: pipeline statistics per recorded command buffer (vertex
//! shader invocations, primitives into and out of clipping, fragment
//! shader invocations), summed into the 10 s line. They tell draws the
//! card never runs from draws whose primitives are all clipped, from draws
//! that write fragments somewhere unexpected.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_vk::ash::vk;

use super::{Backend, lock};

const SLOTS: u32 = 4096;

struct Pool {
    pool: vk::QueryPool,
    next: u32,
    /// Slots ended in submitted command buffers, not read yet.
    pending: Vec<u32>,
}

static POOL: Mutex<Option<Option<Pool>>> = Mutex::new(None);
/// Vertex invocations, clipping invocations, clipping primitives, fragment
/// invocations, command buffers counted.
pub(crate) static SUMS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn on() -> bool {
    cfg!(feature = "diagnostics")
}

fn flags() -> vk::QueryPipelineStatisticFlags {
    vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS
}

/// Begins a statistics query in a fresh command buffer (outside rendering);
/// the slot to end it with.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
pub(crate) unsafe fn begin(b: &Backend, cmd: vk::CommandBuffer) -> Option<u32> {
    if !on() || b.host.caps.core.pipeline_statistics_query != vk::TRUE {
        return None;
    }
    let mut guard = lock(&POOL);
    let pool = guard
        .get_or_insert_with(|| {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::PIPELINE_STATISTICS)
                .query_count(SLOTS)
                .pipeline_statistics(flags());
            // SAFETY: valid create info; kept for the run.
            unsafe { b.host.device.create_query_pool(&info, None) }.ok().map(|pool| Pool {
                pool,
                next: 0,
                pending: Vec::new(),
            })
        })
        .as_mut()?;
    let slot = pool.next % SLOTS;
    pool.next += 1;
    pool.pending.retain(|&s| s != slot);
    // SAFETY: as the caller says.
    unsafe {
        b.host.device.cmd_reset_query_pool(cmd, pool.pool, slot, 1);
        b.host.device.cmd_begin_query(cmd, pool.pool, slot, vk::QueryControlFlags::empty());
    }
    Some(slot)
}

/// Ends the query of `slot` (outside rendering, before the command buffer
/// ends); `submitted`: its results are to be read.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
pub(crate) unsafe fn end(b: &Backend, cmd: vk::CommandBuffer, slot: u32, submitted: bool) {
    let mut guard = lock(&POOL);
    let Some(Some(pool)) = guard.as_mut() else { return };
    // SAFETY: as the caller says.
    unsafe { b.host.device.cmd_end_query(cmd, pool.pool, slot) };
    if submitted {
        pool.pending.push(slot);
    }
}

/// Reads the queries whose command buffers have run into [`SUMS`].
pub(crate) fn collect(b: &Backend) {
    let mut guard = lock(&POOL);
    let Some(Some(pool)) = guard.as_mut() else { return };
    let mut left = Vec::new();
    for &slot in &pool.pending {
        // One query: its four counters as one element (ash takes the
        // element as the stride and the slice length as the query count).
        let mut out = [[0u64; 4]; 1];
        // SAFETY: a slot of our pool; no wait, so an unfinished one says
        // NOT_READY and is read later.
        let r = unsafe {
            b.host.device.get_query_pool_results(pool.pool, slot, &mut out, vk::QueryResultFlags::TYPE_64)
        };
        match r {
            Ok(()) => {
                for (k, v) in out[0].iter().enumerate() {
                    SUMS[k].fetch_add(*v, Ordering::Relaxed);
                }
                SUMS[4].fetch_add(1, Ordering::Relaxed);
            }
            Err(vk::Result::NOT_READY) => left.push(slot),
            Err(_) => {}
        }
    }
    pool.pending = left;
}
