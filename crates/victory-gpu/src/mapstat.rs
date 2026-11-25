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
    /// The title: `XMemMapPhysicalPages` and `XMemMapPhysicalPagesScatter`.
    Pages = 0,
    /// The title's GPU page-mapping copies (`CopyPageMappingsBatchX`).
    Alias = 1,
    /// What stays of a view an unmap cut, mapped again.
    Remnant = 2,
    /// One view of a whole section, for victory's own writes into pages
    /// (DirectStorage), mapped once and kept.
    Section = 3,
    /// Unified memory's canary page, once per section opened.
    Canary = 4,
}

const SITES: usize = 5;
const NAMES: [&str; SITES] = ["pages", "alias", "remnant", "section", "canary"];

/// Views mapped, 64 KiB pages they show, views unmapped, requests that
/// found the memory already as asked (no host call), per site.
static MAPS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static PAGES: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static UNMAPS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static KEPT: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
/// Placeholder splits, placeholder merges, maps the host refused.
static PLACEHOLDERS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Views of ours alive now.
static ALIVE: AtomicI64 = AtomicI64::new(0);
/// Of them, views of one 64 KiB page.
static SINGLE: AtomicI64 = AtomicI64::new(0);

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

/// A request found what it asks for already mapped: no host call.
pub fn kept(site: Site) {
    KEPT[site as usize].fetch_add(1, Relaxed);
}

pub fn split() {
    PLACEHOLDERS[0].fetch_add(1, Relaxed);
}

pub fn merged() {
    PLACEHOLDERS[1].fetch_add(1, Relaxed);
}

pub fn failed() {
    PLACEHOLDERS[2].fetch_add(1, Relaxed);
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
    // Totals at the last line, to print what happened since.
    static BEFORE: Mutex<[u64; 4 * SITES + 3]> = Mutex::new([0; 4 * SITES + 3]);
    let mut now_counts = [0u64; 4 * SITES + 3];
    for i in 0..SITES {
        now_counts[i] = MAPS[i].load(Relaxed);
        now_counts[SITES + i] = PAGES[i].load(Relaxed);
        now_counts[2 * SITES + i] = UNMAPS[i].load(Relaxed);
        now_counts[3 * SITES + i] = KEPT[i].load(Relaxed);
    }
    for i in 0..3 {
        now_counts[4 * SITES + i] = PLACEHOLDERS[i].load(Relaxed);
    }
    let d: Vec<u64> = {
        let mut before = BEFORE.lock().unwrap_or_else(|e| e.into_inner());
        let d = now_counts.iter().zip(before.iter()).map(|(n, b)| n.saturating_sub(*b)).collect();
        *before = now_counts;
        d
    };
    let per_site = |from: usize| {
        let total: u64 = d[from..from + SITES].iter().sum();
        let parts: Vec<String> = (0..SITES).filter(|&i| d[from + i] > 0).map(|i| format!("{} {}", NAMES[i], d[from + i])).collect();
        if parts.is_empty() { total.to_string() } else { format!("{total} ({})", parts.join(", ")) }
    };
    tracing::info!(
        "views: in {:.0} s mapped {} showing {} pages, unmapped {}, already as asked {}; placeholder splits {}, merges {}, maps refused {}; views alive {} ({} of one page)",
        (now - last) as f64 / 1000.0,
        per_site(0),
        d[SITES..2 * SITES].iter().sum::<u64>(),
        per_site(2 * SITES),
        per_site(3 * SITES),
        d[4 * SITES],
        d[4 * SITES + 1],
        d[4 * SITES + 2],
        ALIVE.load(Relaxed),
        SINGLE.load(Relaxed),
    );
}
