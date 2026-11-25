//! Counts of the host views victory maps and unmaps while the title runs.
//! Under Proton every map and every unmap of a view is a wineserver request
//! that walks the list of all the process's views (`server/mapping.c`), on
//! the one wineserver thread, while the calling thread waits: a map costs
//! more the more views are alive. One line every 10 s says how many were
//! made, where, and how many are alive.

use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};

/// Who asked for a view change.
#[derive(Clone, Copy)]
pub enum Site {
    /// The title: `XMemMapPhysicalPages`.
    Pages = 0,
    /// The title: `XMemMapPhysicalPagesScatter`.
    Scatter = 1,
    /// The title's GPU page-mapping copies (`CopyPageMappingsBatchX`).
    Alias = 2,
    /// What stays of a view an unmap cut, mapped again.
    Remnant = 3,
    /// One view of a whole section, for victory's own writes into pages
    /// (DirectStorage), mapped once and kept.
    Section = 4,
    /// Unified memory's canary page, once per section opened.
    Canary = 5,
}

const SITES: usize = 6;
const NAMES: [&str; SITES] = ["pages", "scatter", "alias", "remnant", "section", "canary"];

/// Views mapped, 64 KiB pages they show, views unmapped, requests that
/// found the memory already as asked (no host call), stand-ins made for a
/// filler page (no host view), per site.
static MAPS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static PAGES: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static UNMAPS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static KEPT: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static FILLS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
/// Placeholder splits, placeholder merges, maps the host refused, views of
/// pages already shown elsewhere, title writes into stand-ins, title
/// threads that met a page in the middle of a change and waited.
static OTHER: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Views of ours alive now; of them, views of one 64 KiB page; pages held
/// by stand-ins now.
static ALIVE: AtomicI64 = AtomicI64::new(0);
static SINGLE: AtomicI64 = AtomicI64::new(0);
static FILLED: AtomicI64 = AtomicI64::new(0);

/// A view of `bytes` mapped.
pub fn mapped(site: Site, bytes: usize) {
    MAPS[site as usize].fetch_add(1, Relaxed);
    PAGES[site as usize].fetch_add(bytes.div_ceil(64 << 10) as u64, Relaxed);
    ALIVE.fetch_add(1, Relaxed);
    if bytes <= 64 << 10 {
        SINGLE.fetch_add(1, Relaxed);
    }
}

pub fn unmapped(site: Site, bytes: usize) {
    UNMAPS[site as usize].fetch_add(1, Relaxed);
    ALIVE.fetch_sub(1, Relaxed);
    if bytes <= 64 << 10 {
        SINGLE.fetch_sub(1, Relaxed);
    }
}

/// A stand-in of `bytes` for a filler page made (no host view).
pub fn filled(site: Site, bytes: usize) {
    FILLS[site as usize].fetch_add(1, Relaxed);
    FILLED.fetch_add(bytes.div_ceil(64 << 10) as i64, Relaxed);
}

/// A stand-in of `bytes` let go.
pub fn unfilled(bytes: usize) {
    FILLED.fetch_sub(bytes.div_ceil(64 << 10) as i64, Relaxed);
}

/// A request found what it asks for already mapped: no host call.
pub fn kept(site: Site) {
    KEPT[site as usize].fetch_add(1, Relaxed);
}

pub fn split() {
    OTHER[0].fetch_add(1, Relaxed);
}

pub fn merged() {
    OTHER[1].fetch_add(1, Relaxed);
}

pub fn failed() {
    OTHER[2].fetch_add(1, Relaxed);
}

/// A view whose pages already show somewhere else.
pub fn aliased() {
    OTHER[3].fetch_add(1, Relaxed);
}

/// The title wrote into a stand-in for a filler page.
pub fn filler_written() {
    OTHER[4].fetch_add(1, Relaxed);
}

/// A title thread met a page in the middle of a change and waited for it.
pub fn waited() {
    OTHER[5].fetch_add(1, Relaxed);
}

/// Called with each line (`xmem` checks its filler pages there).
static HOOK: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

pub fn on_tick(hook: fn()) {
    let _ = HOOK.set(hook);
}

/// The line, at most every 10 s; cheap to call often.
pub fn tick() {
    static START: std::sync::LazyLock<std::time::Instant> = std::sync::LazyLock::new(std::time::Instant::now);
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = START.elapsed().as_millis() as u64;
    let last = LAST.load(Relaxed);
    if now < last + 10_000 || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    if let Some(hook) = HOOK.get() {
        hook();
    }
    const N: usize = 5 * SITES + 6;
    // Totals at the last line, to print what happened since.
    static BEFORE: Mutex<[u64; N]> = Mutex::new([0; N]);
    let mut counts = [0u64; N];
    for i in 0..SITES {
        counts[i] = MAPS[i].load(Relaxed);
        counts[SITES + i] = PAGES[i].load(Relaxed);
        counts[2 * SITES + i] = UNMAPS[i].load(Relaxed);
        counts[3 * SITES + i] = KEPT[i].load(Relaxed);
        counts[4 * SITES + i] = FILLS[i].load(Relaxed);
    }
    for i in 0..6 {
        counts[5 * SITES + i] = OTHER[i].load(Relaxed);
    }
    let d: Vec<u64> = {
        let mut before = BEFORE.lock().unwrap_or_else(|e| e.into_inner());
        let d = counts.iter().zip(before.iter()).map(|(n, b)| n.saturating_sub(*b)).collect();
        *before = counts;
        d
    };
    let per_site = |from: usize| {
        let total: u64 = d[from..from + SITES].iter().sum();
        let parts: Vec<String> = (0..SITES).filter(|&i| d[from + i] > 0).map(|i| format!("{} {}", NAMES[i], d[from + i])).collect();
        if parts.is_empty() { total.to_string() } else { format!("{total} ({})", parts.join(", ")) }
    };
    let o = 5 * SITES;
    tracing::info!(
        "views: in {:.0} s mapped {} showing {} pages ({} of pages shown elsewhere too), unmapped {}, already as asked {}, \
         filler stand-ins {}; placeholder splits {}, merges {}, maps refused {}; title writes into stand-ins {}, \
         title threads that waited for a change {}; views alive {} ({} of one page), stand-in pages {}",
        (now - last) as f64 / 1000.0,
        per_site(0),
        d[SITES..2 * SITES].iter().sum::<u64>(),
        d[o + 3],
        per_site(2 * SITES),
        per_site(3 * SITES),
        per_site(4 * SITES),
        d[o],
        d[o + 1],
        d[o + 2],
        d[o + 4],
        d[o + 5],
        ALIVE.load(Relaxed),
        SINGLE.load(Relaxed),
        FILLED.load(Relaxed),
    );
}
