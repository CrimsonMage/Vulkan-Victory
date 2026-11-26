//! Card memory: one rule, and what tells where it goes.
//!
//! **The rule** (KISS, Series S-like): everything of ours on the card
//! stays under [`CEILING`], and never grows into the last [`ROOM`] of the
//! card's budget (`VK_EXT_memory_budget`: what the driver, the spills of
//! shaders and other programs leave us).
//!
//! - Images come first. A new image block that would cross the limit by
//!   more than half the room is refused (counted as `images_refused`).
//! - The pool's GPU view takes what the images leave: it grows a block
//!   while ours stays under the limit, and past it evicts its least
//!   recently touched page (a map never fails). Every frame, a view over
//!   what the images leave gives blocks back ([`PoolView::shrink`]: the
//!   coldest blocks), down to [`LEAST`].
//!
//! **What tells** (diagnostics builds): every card allocation of 64 MiB or
//! more as it happens (what, size, ours after it), and the card-memory
//! line whenever ours or the heap's use moves by 256 MiB, besides every
//! 10 s.
//!
//! [`PoolView::shrink`]: victory_vk::pool::PoolView::shrink

use std::sync::atomic::{AtomicU64, Ordering};

use victory_vk::account::{self, Kind};
use victory_vk::ash::vk;
use victory_vk::host::Host;
use victory_vk::pool::LEAST;

use super::{Backend, images, memory};

/// Ours on the card at most.
pub(super) const CEILING: u64 = 4608 << 20;
/// The card's budget we never grow into.
const ROOM: u64 = 1 << 30;
/// A move of this much prints the card-memory line.
const STEP: u64 = 256 << 20;

/// The card heap's budget and use, if the driver tells them.
fn heap(host: &Host) -> Option<(u64, u64)> {
    host.heaps()
        .iter()
        .filter(|(h, _)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .filter_map(|(_, bu)| *bu)
        .max_by_key(|(budget, _)| *budget)
}

/// Bytes of card memory ours may take now: the ceiling, less what the
/// budget leaves past the others' use and the room.
pub(super) fn limit(host: &Host) -> u64 {
    let Some((budget, usage)) = heap(host) else {
        return CEILING;
    };
    let others = usage.saturating_sub(account::card_total());
    CEILING.min(budget.saturating_sub(others).saturating_sub(ROOM))
}

/// Whether `bytes` more of ours (an image block) may be allocated.
pub(super) fn fits(host: &Host, bytes: u64) -> bool {
    let fits = account::card_total() + bytes <= limit(host) + ROOM / 2;
    if !fits {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::warn!(
                mb = bytes >> 20,
                ours_mb = account::card_total() >> 20,
                limit_mb = limit(host) >> 20,
                "Vulkan: card memory for images refused (the card-memory ceiling)"
            );
        }
    }
    fits
}

/// Every frame: each pool view's ceiling is the limit; ours over it
/// gives pool blocks back now, down to [`LEAST`] a view.
pub(super) fn follow_budget(b: &Backend) {
    let limit = limit(&b.host);
    let mut m = memory(b);
    let mut over = account::card_total().saturating_sub(limit);
    for p in &mut m.pools {
        p.ceiling = limit;
        let has = p.card_bytes();
        let target = has.saturating_sub(over).max(LEAST);
        over = over.saturating_sub(has - target.min(has));
        if has > target
            && let Err(e) = p.shrink(&b.host, &b.timeline, target)
        {
            tracing::warn!(error = %format!("{e:#}"), "Vulkan: the pool view could not give card memory back");
        }
    }
}

/// All the card memory we allocated, by kind, against the driver's own
/// count of the card heaps' use; the difference is the driver's (shader
/// local memory for spills, internal heaps). Host-heap allocations and
/// imports are listed apart: they are system memory.
pub(super) fn card_memory(b: &Backend) {
    let (card, host) = account::totals_mb();
    let (heap_budget, heap_used) = heap(&b.host).map_or((0, 0), |(t, u)| (t >> 20, u >> 20));
    let ours: u64 = card.iter().sum();
    tracing::info!(
        images_mb = card[0],
        images_bound_mb = images::used_mb(),
        pool_mb = card[1],
        buffers_mb = card[3],
        other_mb = card[4],
        ours_mb = ours,
        limit_mb = limit(&b.host) >> 20,
        heap_used_mb = heap_used,
        heap_budget_mb = heap_budget,
        driver_mb = heap_used.saturating_sub(ours),
        host_buffers_mb = host[3],
        imports_mb = host[2],
        "Vulkan: card memory"
    );
}

/// Diagnostics: a card allocation of 64 MiB or more, as it happens.
fn big(kind: Kind, bytes: u64, total: u64) {
    tracing::info!(
        ?kind,
        mb = bytes >> 20,
        ours_mb = total >> 20,
        "Vulkan: a big card allocation"
    );
}

/// Diagnostics: the watcher of big allocations is set.
pub(super) fn watch() {
    if cfg!(feature = "diagnostics") {
        account::watch(big);
    }
}

/// Diagnostics, polled often: the card-memory line when ours or the
/// heap's use moved by [`STEP`] since it was last printed.
pub(super) fn steps(b: &Backend) {
    static LAST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let now = [account::card_total(), heap(&b.host).map_or(0, |(_, u)| u)];
    let moved = now
        .iter()
        .zip(&LAST)
        .any(|(&n, l)| n.abs_diff(l.load(Ordering::Relaxed)) >= STEP);
    if moved {
        for (n, l) in now.iter().zip(&LAST) {
            l.store(*n, Ordering::Relaxed);
        }
        card_memory(b);
    }
}
