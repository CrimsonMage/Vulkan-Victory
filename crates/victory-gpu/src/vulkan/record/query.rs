//! Queries measured by the card: each title query heap of occlusion or
//! timestamp queries is a `VkQueryPool`, made at its first use.
//!
//! - `BeginQuery`/`EndQuery` of occlusion (precise if the device has it) and
//!   binary occlusion queries reset and begin/end the pool's query, outside
//!   rendering. The console lets occlusion queries nest, Vulkan does not:
//!   a query begun while another is active ends that one, which then
//!   resolves as visible (as XCL does), and so does a query still active
//!   when its command buffer ends.
//! - `EndQuery` of a timestamp writes it (`vkCmdWriteTimestamp2`, when the
//!   work before it is done).
//! - `ResolveQueryData` copies the results into title memory through its
//!   buffer (`vkCmdCopyQueryPoolResults`, 64 bits each, waiting for them).
//!   An occlusion query never issued in the run resolves as visible; a
//!   timestamp never written is left as it is.
//! - Other query types (pipeline and stream-output statistics) keep the
//!   plausible values of `list/console.rs`.
//!
//! Timestamps are the card's ticks: the queue's frequency says how many a
//! second (`timestamp_frequency`).

use std::collections::HashMap;
use std::sync::Mutex;

use victory_vk::ash::vk;

use super::super::{Backend, images, lock};
use super::Rec;
use crate::object::This;

/// Query types (`D3D12_QUERY_TYPE`).
const OCCLUSION: u32 = 0;
const BINARY_OCCLUSION: u32 = 1;
const TIMESTAMP: u32 = 2;

struct Pool {
    pool: vk::QueryPool,
    timestamps: bool,
    count: u32,
    /// The heap's description, to tell a new heap at the same address.
    desc: (u32, u32),
    /// Queries begun or written in the run (recorded so far).
    issued: Vec<bool>,
    /// Occlusion queries ended early (nesting, a cut): resolve as visible.
    forced: Vec<bool>,
}

/// Pools by title heap; replaced ones are kept (command buffers in flight
/// may name them).
static POOLS: Mutex<Option<HashMap<usize, Pool>>> = Mutex::new(None);
static RETIRED: Mutex<Vec<vk::QueryPool>> = Mutex::new(Vec::new());

/// Runs `f` on the pool of title heap `heap` (made the first time); `None`
/// for heaps of other query types.
fn with_pool<R>(b: &Backend, heap: This, f: impl FnOnce(&mut Pool) -> R) -> Option<R> {
    let (kind, count) = crate::query::desc(heap)?;
    // `D3D12_QUERY_HEAP_TYPE`: occlusion 0, timestamp 1, copy-queue timestamp 5.
    let timestamps = match kind {
        0 => false,
        1 | 5 => true,
        _ => return None,
    };
    let mut pools = lock(&POOLS);
    let pools = pools.get_or_insert_with(HashMap::new);
    if pools.get(&(heap as usize)).is_some_and(|p| p.desc != (kind, count)) {
        let old = pools.remove(&(heap as usize)).map(|p| p.pool);
        lock(&RETIRED).extend(old);
    }
    if !pools.contains_key(&(heap as usize)) {
        let ty = if timestamps { vk::QueryType::TIMESTAMP } else { vk::QueryType::OCCLUSION };
        let info = vk::QueryPoolCreateInfo::default().query_type(ty).query_count(count.max(1));
        // SAFETY: valid create info; kept for the run.
        let pool = unsafe { b.host.device.create_query_pool(&info, None) }.ok()?;
        let n = count as usize;
        pools.insert(heap as usize, Pool { pool, timestamps, count, desc: (kind, count), issued: vec![false; n], forced: vec![false; n] });
    }
    pools.get_mut(&(heap as usize)).map(f)
}

/// Marks query `index` of `heap` issued (and not forced visible) or forced.
fn mark(b: &Backend, heap: usize, index: u32, forced: bool) {
    let _ = with_pool(b, heap as This, |p| {
        if let (Some(i), Some(f)) = (p.issued.get_mut(index as usize), p.forced.get_mut(index as usize)) {
            *i = true;
            *f = forced;
        }
    });
}

/// Ticks a second of the card's timestamps (`None`: the Vulkan backend
/// does not run).
pub(crate) fn timestamp_frequency() -> Option<u64> {
    if !super::super::on() {
        return None;
    }
    let period = f64::from(super::super::backend()?.host.caps.timestamp_period);
    (period > 0.0).then(|| (1e9 / period).round() as u64)
}

/// The ring of our own timestamp queries (top-of-pipe timestamp writes).
const OWN_TIMESTAMPS: u32 = 1024;

fn own_timestamp(b: &Backend) -> Option<(vk::QueryPool, u32)> {
    static RING: Mutex<Option<(Option<vk::QueryPool>, u32)>> = Mutex::new(None);
    let mut ring = lock(&RING);
    let (pool, next) = ring.get_or_insert_with(|| {
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(OWN_TIMESTAMPS);
        // SAFETY: valid create info; kept for the run.
        (unsafe { b.host.device.create_query_pool(&info, None) }.ok(), 0)
    });
    let slot = *next % OWN_TIMESTAMPS;
    *next += 1;
    pool.map(|p| (p, slot))
}

impl Rec {
    /// Ends the active occlusion query (outside rendering); `forced`: it
    /// ended early and resolves as visible.
    fn end_active(&mut self, b: &Backend, cmd: vk::CommandBuffer, forced: bool) {
        if let Some((heap, index, pool)) = self.console.query.take() {
            // SAFETY: outside rendering; the query is active in `cmd`.
            unsafe { b.host.device.cmd_end_query(cmd, pool, index) };
            if forced {
                mark(b, heap, index, true);
            }
        }
    }

    /// The command buffer ends: an active query ends with it.
    pub(super) fn queries_cut(&mut self, b: &Backend, cmd: vk::CommandBuffer) {
        self.end_active(b, cmd, true);
    }

    pub(crate) fn begin_query(&mut self, b: &'static Backend, heap: This, kind: u32, index: u32) {
        if kind != OCCLUSION && kind != BINARY_OCCLUSION {
            return;
        }
        let Some(pool) = with_pool(b, heap, |p| (!p.timestamps && index < p.count).then_some(p.pool)).flatten() else { return };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        self.end_active(b, cmd, true);
        let precise = kind == OCCLUSION && b.host.caps.core.occlusion_query_precise == vk::TRUE;
        let flags = if precise { vk::QueryControlFlags::PRECISE } else { vk::QueryControlFlags::empty() };
        // SAFETY: outside rendering; a query of our pool.
        unsafe {
            b.host.device.cmd_reset_query_pool(cmd, pool, index, 1);
            b.host.device.cmd_begin_query(cmd, pool, index, flags);
        }
        self.console.query = Some((heap as usize, index, pool));
        mark(b, heap as usize, index, false);
    }

    pub(crate) fn end_query(&mut self, b: &'static Backend, heap: This, kind: u32, index: u32) {
        match kind {
            TIMESTAMP => {
                let Some(pool) = with_pool(b, heap, |p| (p.timestamps && index < p.count).then_some(p.pool)).flatten() else { return };
                self.end_pass(b);
                let Some(cmd) = self.cmd(b) else { return };
                // SAFETY: outside rendering; a query of our pool.
                unsafe {
                    b.host.device.cmd_reset_query_pool(cmd, pool, index, 1);
                    b.host.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, index);
                }
                mark(b, heap as usize, index, false);
            }
            OCCLUSION | BINARY_OCCLUSION if self.console.query.is_some_and(|(h, i, _)| h == heap as usize && i == index) => {
                self.end_pass(b);
                if let Some(cmd) = self.cmd(b) {
                    self.end_active(b, cmd, false);
                }
            }
            _ => {}
        }
    }

    /// `ResolveQueryData(heap, type, first, count, buffer, offset)`.
    #[allow(clippy::too_many_arguments)] // the method's own arguments
    pub(crate) fn resolve_queries(&mut self, b: &'static Backend, list: This, heap: This, kind: u32, first: u32, count: u32, buffer: This, offset: u64) {
        let fake = || crate::list::resolve_queries(list, heap, kind, first, count, buffer, offset);
        let wanted_timestamps = match kind {
            OCCLUSION | BINARY_OCCLUSION => false,
            TIMESTAMP => true,
            _ => return fake(),
        };
        let Some((data, size)) = crate::resource::buffer_bytes(buffer) else { return };
        if offset + u64::from(count) * 8 > size || count == 0 {
            return;
        }
        let found = with_pool(b, heap, |p| {
            let end = (first + count) as usize;
            (p.timestamps == wanted_timestamps && end <= p.issued.len())
                .then(|| (p.pool, p.issued[first as usize..end].to_vec(), p.forced[first as usize..end].to_vec()))
        });
        let (Some(Some((pool, issued, forced))), Some((dst, at))) = (found, super::super::buffer_of(b, data as u64 + offset)) else {
            return fake();
        };
        if self.console.query.is_some_and(|(h, i, _)| h == heap as usize && (first..first + count).contains(&i)) {
            // Resolving the active query: it ends here.
            self.end_pass(b);
            if let Some(cmd) = self.cmd(b) {
                self.end_active(b, cmd, true);
            }
        }
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        let visible = (!wanted_timestamps).then(|| issued.iter().zip(&forced).map(|(&i, &f)| !i || f).collect::<Vec<bool>>());
        // SAFETY: outside rendering; our pool, title memory through its buffer.
        unsafe { copy_results(&b.host.device, cmd, (pool, first, &issued), (dst, at), visible.as_deref()) };
    }

    /// The card's clock now, into title memory at `buffer` + `offset`.
    pub(super) fn timestamp_into(&mut self, b: &'static Backend, buffer: vk::Buffer, offset: u64) {
        let Some((pool, slot)) = own_timestamp(b) else { return };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        let d = &b.host.device;
        // SAFETY: outside rendering; our pool, title memory through its buffer.
        unsafe {
            d.cmd_reset_query_pool(cmd, pool, slot, 1);
            d.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, pool, slot);
            d.cmd_copy_query_pool_results(cmd, pool, slot, 1, buffer, offset, 8, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT);
            images::full_barrier(d, cmd);
        }
    }
}

/// The results of the `issued` queries from `first` of `pool` into `dst`
/// at `at`, 8 bytes each; then a nonzero count for those `visible` says.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
unsafe fn copy_results(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, (pool, first, issued): (vk::QueryPool, u32, &[bool]), (dst, at): (vk::Buffer, u64), visible: Option<&[bool]>) {
    let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
    // SAFETY: as the caller says.
    unsafe {
        images::full_barrier(d, cmd);
        for (k, n) in runs(issued, true) {
            d.cmd_copy_query_pool_results(cmd, pool, first + k, n, dst, at + u64::from(k) * 8, 8, flags);
        }
        images::full_barrier(d, cmd);
        if let Some(visible) = visible {
            for (k, n) in runs(visible, true) {
                d.cmd_fill_buffer(cmd, dst, at + u64::from(k) * 8, u64::from(n) * 8, 1);
            }
            images::full_barrier(d, cmd);
        }
    }
}

/// The runs of `want` in `flags`: (first, count) each.
fn runs(flags: &[bool], want: bool) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &f) in flags.iter().chain(std::iter::once(&!want)).enumerate() {
        match (f == want, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s as u32, (i - s) as u32));
                start = None;
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn runs_of_flags() {
        assert_eq!(super::runs(&[true, true, false, true], true), vec![(0, 2), (3, 1)]);
        assert_eq!(super::runs(&[false, false], true), vec![]);
        assert_eq!(super::runs(&[false, true, true], false), vec![(0, 1)]);
    }
}
