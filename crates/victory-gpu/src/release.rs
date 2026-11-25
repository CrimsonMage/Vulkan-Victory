//! Victory's memory follows the title's. Content is streamed in once and
//! stays until the title unloads it; when it does, at the API point that
//! says so, victory lets go of what it holds for that memory, at once:
//!
//! - the title frees or decommits memory (`VirtualFree`, the `xmem`
//!   allocator's frees), or unmaps physical pages (`XMemMapPhysicalPages`
//!   with no pages, the GPU's page-mapping copies to nothing):
//!   [`title_memory_released`] drops the host copies of textures the title
//!   fills whose memory starts there and their staging
//!   (`resource::released`; a copy a list or a slot still holds goes as
//!   soon as none does, and a submission still running keeps it until it
//!   has landed, `host::retire`), and the pieces of title memory opened as
//!   heaps there (`unified::released`; a heap goes with the last buffer
//!   placed in it). A section whose pages are all freed lets go of its
//!   pieces when its handle closes (`unified::title_section_gone`);
//! - the title releases a resource: its host copy goes at once
//!   (`Drop for Resource`), counted here.
//!
//! The idle sweep (`resource::let_go_idle`, `texture::let_go_idle`) stays,
//! as the net under the card's budget. One line every ten seconds says
//! what went for which reason ([`tick`]).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Why host memory was let go.
#[derive(Clone, Copy)]
pub(crate) enum Why {
    /// The title freed or unmapped the memory a host copy comes from.
    Freed = 0,
    /// The title released the resource.
    Resource = 1,
}

/// Since the start, per [`Why`]: host copies let go and their bytes; then
/// staging bytes let go because the title freed memory, and host copies
/// that were held when their memory was freed (left for the sweep).
static GONE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Counts a host copy of `bytes` let go for `why`.
pub(crate) fn count(why: Why, bytes: u64) {
    GONE[why as usize * 2].fetch_add(1, Relaxed);
    GONE[why as usize * 2 + 1].fetch_add(bytes, Relaxed);
}

/// The title freed, decommitted or unmapped `[at, at + len)`: what victory
/// holds for that memory goes now (see the module comment).
pub fn title_memory_released(at: usize, len: usize) {
    if len == 0 || (at as u64) >= 1 << 40 {
        return;
    }
    let [n, bytes, staged, held] = crate::resource::released(at, len);
    GONE[0].fetch_add(n, Relaxed);
    GONE[1].fetch_add(bytes, Relaxed);
    GONE[4].fetch_add(staged, Relaxed);
    GONE[5].fetch_add(held, Relaxed);
    crate::unified::released(at, len);
}

/// Every ten seconds, from the sweep: the release line.
pub(crate) fn tick() {
    watch_process_memory();
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Relaxed);
    if now < last + 10_000 || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let g: Vec<u64> = GONE.iter().map(|g| g.load(Relaxed)).collect();
    let k: Vec<u64> = crate::resource::KINDS.iter().map(|k| k.load(Relaxed)).collect();
    let (staging_idle, staging_idle_bytes) = crate::texture::let_go();
    let (staged, _) = crate::texture::staging_bytes();
    let alive = [
        crate::resource::HOST_TEXTURES[0].load(Relaxed),
        crate::resource::HOST_TEXTURES[1].load(Relaxed) + crate::tiles::given_total(),
    ];
    tracing::info!(
        "memory release since the start: because the title freed or unmapped memory {} MB (host copies {}, {} MB; staging {} MB; \
         title memory opened as heaps {} MB; {} copies held then, let go once free), because the title released resources {} MB \
         ({} host copies); by idle eviction {} MB (host copies {}, {} MB; staging {}, {} MB); made again after either {} ({} MB); \
         host copies alive {} ({} MB), staging alive {} MB",
        (g[1] + g[4] + crate::unified::released_bytes()) >> 20,
        g[0],
        g[1] >> 20,
        g[4] >> 20,
        crate::unified::released_bytes() >> 20,
        g[5],
        g[3] >> 20,
        g[2],
        (k[3] + staging_idle_bytes) >> 20,
        k[2],
        k[3] >> 20,
        staging_idle,
        staging_idle_bytes >> 20,
        k[4],
        k[5] >> 20,
        alive[0],
        alive[1] >> 20,
        staged >> 20,
    );
}

/// Every thirty seconds, on a thread of its own: where the process's
/// resident memory is, as Linux counts it (`/proc/self/smaps`, read through
/// Wine's `Z:` drive; off where that does not exist). Split by title
/// addresses (below 1 TiB: the title's regions, its section views, its own
/// heaps) and host addresses (above: victory, vkd3d-proton, Wine), anonymous
/// against shared (sections, `memfd`), the NVIDIA driver's mappings, and
/// files; then the largest mappings. Run p7 had 21.6 GB resident against
/// about 14 GB before, with title memory opened as heaps the same.
pub(crate) fn watch_process_memory() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-procmem".into()).spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(30));
            let Some(pid) = unix_pid() else {
                tracing::info!("process memory: this process is not found under /proc (not Wine?); no process memory line");
                return;
            };
            let path = format!(r"Z:\proc\{pid}\smaps");
            loop {
                match std::fs::read_to_string(&path) {
                    Ok(text) => tracing::info!(pid, "{}", process_memory(&text)),
                    Err(e) => {
                        tracing::info!(pid, error = %e, "process memory: smaps cannot be read; no process memory line");
                        return;
                    }
                }
                std::thread::sleep(std::time::Duration::from_secs(30));
            }
        });
    });
}

/// This process's Linux pid: the process under `/proc` (Wine's `Z:`) that
/// maps our own executable over the address of our own code. (`/proc/self`
/// read from the Windows side named another, small process in run p8: 34
/// MB in 30 mappings against a VmRSS of 21 GB.)
fn unix_pid() -> Option<u32> {
    let marker = unix_pid as *const () as usize as u64;
    let exe = std::env::current_exe().ok()?.file_name()?.to_string_lossy().to_ascii_lowercase();
    for entry in std::fs::read_dir(r"Z:\proc").ok()?.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else { continue };
        let Ok(maps) = std::fs::read_to_string(format!(r"Z:\proc\{pid}\maps")) else { continue };
        let ours = maps.lines().any(|line| {
            let mut words = line.split_whitespace();
            let range = words.next().and_then(|r| r.split_once('-')).and_then(|(a, b)| {
                Some((u64::from_str_radix(a, 16).ok()?, u64::from_str_radix(b, 16).ok()?))
            });
            range.is_some_and(|(a, b)| a <= marker && marker < b) && line.to_ascii_lowercase().contains(&exe)
        });
        if ours {
            return Some(pid);
        }
    }
    None
}

/// A mapping of `/proc/self/smaps` being read: start, bytes, path, resident
/// kB, anonymous kB.
type Mapping = (u64, u64, String, u64, u64);

/// The process memory line from the text of `/proc/self/smaps`.
fn process_memory(text: &str) -> String {
    const TIB: u64 = 1 << 40;
    // Resident kB: title anonymous, title shared, host anonymous, host
    // shared, driver, files.
    let mut kb = [0u64; 6];
    let mut all = 0u64;
    let mut maps = 0u64;
    let mut top: Vec<(u64, u64, u64, String)> = Vec::new();
    let mut cur: Option<Mapping> = None;
    let mut finish = |cur: Option<Mapping>, kb: &mut [u64; 6], all: &mut u64| {
        let Some((start, bytes, path, rss, anon)) = cur else { return };
        *all += rss;
        let shared = rss.saturating_sub(anon);
        if path.starts_with("/dev/nvidia") || path.starts_with("/dev/dri") {
            kb[4] += rss;
        } else if start < TIB {
            kb[0] += anon;
            kb[1] += shared;
        } else if path.is_empty() || path.starts_with('[') || path.starts_with("/memfd:") || path.ends_with("(deleted)") {
            kb[2] += anon;
            kb[3] += shared;
        } else {
            kb[5] += rss;
        }
        if rss >= 64 << 10 {
            top.push((rss, start, bytes, path));
        }
    };
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let Some(first) = words.next() else { continue };
        if let Some((a, b)) = first.split_once('-')
            && let (Ok(a), Ok(b)) = (u64::from_str_radix(a, 16), u64::from_str_radix(b, 16))
        {
            finish(cur.take(), &mut kb, &mut all);
            maps += 1;
            let path = words.skip(4).collect::<Vec<_>>().join(" ");
            cur = Some((a, b.saturating_sub(a), path, 0, 0));
            continue;
        }
        let value = words.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        match (first, cur.as_mut()) {
            ("Rss:", Some(c)) => c.3 = value,
            ("Anonymous:", Some(c)) => c.4 = value,
            _ => {}
        }
    }
    finish(cur.take(), &mut kb, &mut all);
    drop(finish);
    top.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let largest: Vec<String> = top
        .iter()
        .take(8)
        .map(|(rss, start, bytes, path)| {
            format!("{start:#x} {} of {} MB {}", rss >> 10, bytes >> 20, if path.is_empty() { "anon" } else { path })
        })
        .collect();
    format!(
        "process memory (Linux): resident {} MB in {} mappings; title addresses: anonymous {} MB, shared {} MB; host addresses: \
         anonymous {} MB, shared {} MB; NVIDIA driver mappings {} MB; files {} MB; of ours in the PC's memory: title memory opened \
         as heaps {} MB, staging {} MB, spare staging {} MB, buffers pooled {} MB; largest (resident of size): {}",
        all >> 10,
        maps,
        kb[0] >> 10,
        kb[1] >> 10,
        kb[2] >> 10,
        kb[3] >> 10,
        kb[4] >> 10,
        kb[5] >> 10,
        crate::unified::opened_bytes() >> 20,
        crate::texture::staging_bytes().0 >> 20,
        crate::texture::staging_bytes().1 >> 20,
        crate::memcap::pooled() >> 20,
        largest.join("; ")
    )
}
