//! Every `vkAllocateMemory` of ours, by kind and by heap (card or host), so
//! one line can account for all the card memory the process holds: what
//! we allocated against `VK_EXT_memory_budget`'s heap usage. The rest is
//! the driver's own (shader local memory for spills, internal heaps).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use ash::vk;

use crate::host::Host;

/// What an allocation is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Card images (blocks and dedicated allocations).
    Images,
    /// The pool's GPU view's card pages.
    Pool,
    /// Title memory imported by host pointer (host memory the card reads).
    Imports,
    /// Buffers of ours: staging, descriptor buffers, memory maps, uploads.
    Buffers,
    /// Small fixed things (null images, the probe's).
    Other,
}

const KINDS: usize = 5;

struct Book {
    by_memory: HashMap<vk::DeviceMemory, (Kind, u64, bool)>,
    /// Bytes by kind, on card heaps and on host heaps.
    card: [u64; KINDS],
    host: [u64; KINDS],
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);

/// Allocations of card memory at least this big are told to the watcher.
pub const BIG: u64 = 64 << 20;

/// Told each allocation of card memory of [`BIG`] or more: what, bytes,
/// and ours on card heaps after it.
pub type Watcher = fn(Kind, u64, u64);

static WATCHER: OnceLock<Watcher> = OnceLock::new();

/// Sets the watcher of big card allocations (once).
pub fn watch(f: Watcher) {
    let _ = WATCHER.set(f);
}

fn book() -> std::sync::MutexGuard<'static, Option<Book>> {
    BOOK.lock().unwrap_or_else(|e| e.into_inner())
}

/// `memory` of `bytes` and memory type `type_index` was allocated for `kind`.
pub fn allocated(host: &Host, memory: vk::DeviceMemory, type_index: u32, bytes: u64, kind: Kind) {
    let heap = host.memory.memory_types.get(type_index as usize).map_or(0, |t| t.heap_index as usize);
    let card = host.memory.memory_heaps.get(heap).is_some_and(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        && kind != Kind::Imports;
    let mut g = book();
    let b = g.get_or_insert_with(|| Book { by_memory: HashMap::new(), card: [0; KINDS], host: [0; KINDS] });
    b.by_memory.insert(memory, (kind, bytes, card));
    if card {
        b.card[kind as usize] += bytes;
    } else {
        b.host[kind as usize] += bytes;
    }
    let total: u64 = b.card.iter().sum();
    drop(g);
    if card
        && bytes >= BIG
        && let Some(f) = WATCHER.get()
    {
        f(kind, bytes, total);
    }
}

/// Bytes of ours on card heaps.
pub fn card_total() -> u64 {
    book().as_ref().map_or(0, |b| b.card.iter().sum())
}

/// `memory` is about to be freed.
pub fn freed(memory: vk::DeviceMemory) {
    let mut g = book();
    let Some(b) = g.as_mut() else { return };
    if let Some((kind, bytes, card)) = b.by_memory.remove(&memory) {
        let n = if card { &mut b.card[kind as usize] } else { &mut b.host[kind as usize] };
        *n = n.saturating_sub(bytes);
        let total: u64 = b.card.iter().sum();
        drop(g);
        if card
            && bytes >= BIG
            && let Some(f) = FREE_WATCHER.get()
        {
            f(kind, bytes, total);
        }
    }
}

static FREE_WATCHER: OnceLock<Watcher> = OnceLock::new();

/// Sets the watcher of big card frees (once).
pub fn watch_frees(f: Watcher) {
    let _ = FREE_WATCHER.set(f);
}

/// MiB by kind (images, pool, imports, buffers, other) on card heaps and
/// on host heaps.
pub fn totals_mb() -> ([u64; KINDS], [u64; KINDS]) {
    book().as_ref().map_or(([0; KINDS], [0; KINDS]), |b| (b.card.map(|x| x >> 20), b.host.map(|x| x >> 20)))
}
