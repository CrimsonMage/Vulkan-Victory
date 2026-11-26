//! Bottom-of-pipe writes made by the card (`docs/vulkan-host.md`, "Queues
//! and fences"): at the end of their part, after a full barrier, a value
//! goes into the imported title memory by `vkCmdUpdateBuffer`, a GPU
//! timestamp by `vkCmdWriteTimestamp2` and `vkCmdCopyQueryPoolResults`.
//! The CPU sees them in place, since title memory is imported where it
//! lies. Nothing waits for the part to land.
//!
//! The CPU still makes the writes the card cannot: into the pool's GPU view
//! (the CPU reads its home, not the card's copy), into memory not imported
//! yet, unaligned ones, and the CPU's own timestamps.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use victory_vk::ash::vk;
use victory_vk::timeline::Recording;

use super::engine::Write;
use super::{Backend, backend, images, memory};
use crate::queue::Written;

/// Timestamp queries, used in turn (a slot is reused after this many).
const SLOTS: u32 = 1024;

/// The buffer and offset the card writes `bytes` at title address `at`
/// through, if it can.
fn target(b: &Backend, at: u64, what: Written, bytes: usize) -> Option<(vk::Buffer, u64)> {
    let aligned = match what {
        Written::Value(_) => at.is_multiple_of(4),
        // `vkCmdCopyQueryPoolResults` with 64-bit results.
        Written::GpuTimestamp => bytes == 8 && at.is_multiple_of(8),
        Written::SystemTimestamp => false,
    };
    if !aligned {
        return None;
    }
    let m = memory(b);
    if m.pools.iter().any(|p| p.contains(at)) {
        return None;
    }
    m.buffer_of(at)
}

fn timestamps(b: &Backend) -> Option<vk::QueryPool> {
    static POOL: OnceLock<Option<vk::QueryPool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(SLOTS);
        // SAFETY: valid create info; kept for the run.
        (b.host.caps.timestamp_period > 0.0)
            .then(|| unsafe { b.host.device.create_query_pool(&info, None) }.ok())
            .flatten()
    })
}

/// Records the writes the card can make at the end of the part `cmds`
/// (into its last command buffer, or a new one); returns the writes left
/// to the CPU.
pub(super) fn record(b: &Backend, cmds: &mut Vec<Recording>, writes: &[Write]) -> Vec<Write> {
    let mut cpu = Vec::new();
    let mut card = Vec::new();
    for &(at, what, bytes) in writes {
        match target(b, at, what, bytes) {
            Some(t) if !matches!(what, Written::GpuTimestamp) || timestamps(b).is_some() => {
                card.push((t, what, bytes))
            }
            _ => cpu.push((at, what, bytes)),
        }
    }
    if card.is_empty() {
        return cpu;
    }
    if cmds.is_empty() {
        match b.timeline.begin(&b.host) {
            Ok(r) => cmds.push(r),
            Err(_) => return writes.to_vec(),
        }
    }
    let cmd = cmds.last().expect("command buffer").cmd;
    let d = &b.host.device;
    // SAFETY: the part's last command buffer, handed over by its list and
    // still open, outside rendering; title buffers (imported memory).
    unsafe {
        images::full_barrier(d, cmd);
        for ((buffer, offset), what, bytes) in card {
            match what {
                Written::Value(v) => {
                    d.cmd_update_buffer(cmd, buffer, offset, &v.to_le_bytes()[..bytes])
                }
                _ => timestamp(b, cmd, buffer, offset),
            }
        }
        let host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&host));
    }
    cpu
}

/// The card's clock into title memory: a timestamp written once
/// everything before has run, copied into the buffer.
///
/// # Safety
/// Recording into `cmd`, outside rendering; `offset` 8-byte aligned.
unsafe fn timestamp(b: &Backend, cmd: vk::CommandBuffer, buffer: vk::Buffer, offset: u64) {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let Some(pool) = timestamps(b) else { return };
    let slot = NEXT.fetch_add(1, Ordering::Relaxed) % SLOTS;
    let d = &b.host.device;
    // SAFETY: as the caller says; a slot of our pool.
    unsafe {
        d.cmd_reset_query_pool(cmd, pool, slot, 1);
        d.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, slot);
        d.cmd_copy_query_pool_results(
            cmd,
            pool,
            slot,
            1,
            buffer,
            offset,
            8,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        );
    }
}

/// Ticks per second of the card's timestamps (what the title's queues
/// report under Vulkan).
pub(crate) fn timestamp_frequency() -> Option<u64> {
    let period = f64::from(backend()?.host.caps.timestamp_period);
    (period > 0.0).then(|| (1e9 / period) as u64)
}
