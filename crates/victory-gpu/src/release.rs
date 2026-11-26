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
    crate::procmem::start();
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
