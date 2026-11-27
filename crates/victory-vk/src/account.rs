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
    /// The recordings' scratch buffers (`timeline.rs`).
    Scratch,
}

const KINDS: usize = 6;

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

/// MiB by kind (images, pool, imports, buffers, other, scratch) on card
/// heaps and on host heaps.
pub fn totals_mb() -> ([u64; KINDS], [u64; KINDS]) {
    book().as_ref().map_or(([0; KINDS], [0; KINDS]), |b| (b.card.map(|x| x >> 20), b.host.map(|x| x >> 20)))
}

/// The card heap's budget and use in bytes (`VK_EXT_memory_budget`), if
/// the driver tells them. On NVIDIA the budget is the heap's use plus what
/// is free on the card, so `budget - use` is the card's free memory.
pub fn card_heap(host: &Host) -> Option<(u64, u64)> {
    host.heaps()
        .iter()
        .filter(|(h, _)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .filter_map(|(_, bu)| *bu)
        .max_by_key(|(budget, _)| *budget)
}

/// The card's free memory when the device was made, before anything of
/// ours was allocated on it.
static FREE_AT_START: OnceLock<u64> = OnceLock::new();

/// Notes the card's free memory now (once, right after the device is
/// made): what the process takes on the card later is measured against it.
pub fn start(host: &Host) {
    if let Some((budget, usage)) = card_heap(host) {
        let _ = FREE_AT_START.set(budget.saturating_sub(usage));
    }
}

/// Bytes the whole process holds on the card, as `nvidia-smi` counts it:
/// what left the card's free memory since the device was made (other
/// programs' moves are in it too), and the part of it that is in no
/// Vulkan heap's use: the driver's own objects.
pub fn process(host: &Host) -> Option<(u64, u64)> {
    let (budget, usage) = card_heap(host)?;
    let taken = FREE_AT_START.get()?.saturating_sub(budget.saturating_sub(usage));
    Some((taken, taken.saturating_sub(usage)))
}

/// What one kind of driver call took: calls, bytes outside every heap's
/// use (budget lost), bytes of heap use that are not allocations of ours.
type Took = (&'static str, u64, i64, i64);

static TRACE: Mutex<Vec<Took>> = Mutex::new(Vec::new());

fn tracing_calls() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_CARD_TRACE").is_ok_and(|v| v.trim() == "1"))
}

thread_local! {
    static TRACING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Diagnostics (`VICTORY_CARD_TRACE=1`): runs `f`, a driver call of kind
/// `what`, alone (traced calls wait for each other) and books what the
/// card lost over it beside our own allocations. Returns the bytes
/// outside the heap's use and the heap use not ours it took.
pub fn traced_with<T>(host: &Host, what: &'static str, f: impl FnOnce() -> T) -> (T, i64, i64) {
    static ONE: Mutex<()> = Mutex::new(());
    if !tracing_calls() || TRACING.with(std::cell::Cell::get) {
        return (f(), 0, 0);
    }
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    TRACING.with(|t| t.set(true));
    let before = card_heap(host).unwrap_or((0, 0));
    let ours = card_total();
    let r = f();
    let after = card_heap(host).unwrap_or((0, 0));
    let ours = card_total() as i64 - ours as i64;
    TRACING.with(|t| t.set(false));
    let hidden = before.0 as i64 - after.0 as i64;
    let driver = after.1 as i64 - before.1 as i64 - ours;
    let mut t = TRACE.lock().unwrap_or_else(|e| e.into_inner());
    match t.iter_mut().find(|k| k.0 == what) {
        Some(k) => {
            k.1 += 1;
            k.2 += hidden;
            k.3 += driver;
        }
        None => t.push((what, 1, hidden, driver)),
    }
    (r, hidden, driver)
}

/// [`traced_with`] without the figures.
pub fn traced<T>(host: &Host, what: &'static str, f: impl FnOnce() -> T) -> T {
    traced_with(host, what, f).0
}

/// The traced kinds: "what n=calls outside=MiB heap=MiB", biggest first;
/// empty when nothing is traced.
pub fn traced_line() -> String {
    let mut t = TRACE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    t.sort_by_key(|k| std::cmp::Reverse(k.2.abs() + k.3.abs()));
    t.iter()
        .map(|k| format!("{} n={} outside={}MB heap={}MB", k.0, k.1, k.2 >> 20, k.3 >> 20))
        .collect::<Vec<_>>()
        .join("; ")
}
