//! A hard ceiling on the host memory the layer makes the graphics card
//! hold, and a pool that uses the PC-memory buffers it lets go again.
//!
//! Why (runs of 2026-11-25 on an 8 GB card under vkd3d-proton): the card's
//! "budget" the system reports is no fixed line there. vkd3d reports the
//! Vulkan heap budget, which on that driver is about our use plus what is
//! free on the card, so it shrank with everything else on the card (6,780
//! MB at the start, 5,527 MB = our use at the end) and the rules that make
//! host copies "up to nine tenths of the budget" chased it down to a card
//! with nothing free: 7.7 of 8.1 GB in use. Beside that the layer made a
//! separate GPU-visible buffer in the PC's memory for every window of
//! title memory, title-buffer copy and upload (vkd3d gives each buffer of
//! 2 MiB or more an allocation and a GPU mapping of its own) and let them
//! go again: 21,848 upload buffers and 7,486 shared buffers (15.7 GB)
//! made in seven minutes, 5.7 GB of the PC's memory mapped for the card at
//! once. The NVIDIA kernel driver then failed to insert GPU mappings
//! (`intermapRegisterDmaMapping`), failed to allocate, and took the whole
//! machine down on an unmap.
//!
//! What this does:
//! - the card's budget every rule of the layer sees ([`clamp_budget`],
//!   through `Host::video_memory`) is at most a ceiling made from the
//!   adapter's real dedicated memory ([`card_cap`]), and always leaves
//!   [`CARD_RESERVE`] of the system's own budget free;
//! - buffers in the PC's memory that the card maps (upload and shared
//!   ones) are not made past [`shared_cap`]: the pool is emptied first,
//!   then the buffer is refused (the caller's path for a failed creation:
//!   the copy, upload or window is not made this time);
//! - those buffers, when their owner lets them go, go to a pool ([`give`])
//!   and are handed out again for the same size ([`take`]) once nothing
//!   holds them and the submissions that ran meanwhile have finished, so
//!   the driver sees far fewer allocations, mappings and unmappings.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

const MB: u64 = 1 << 20;

fn env_mb(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|mb| mb * MB)
}

/// The adapter's dedicated memory in bytes (0: not known yet).
static DEDICATED: AtomicU64 = AtomicU64::new(0);

/// Notes the adapter's dedicated memory (`DXGI_ADAPTER_DESC1`), once the
/// host device is made, and says what the ceilings are.
pub(crate) fn set_dedicated(bytes: u64) {
    DEDICATED.store(bytes, Relaxed);
    tracing::info!(
        "memory ceilings: the adapter has {} MB of its own memory; the layer's use of it stays under {} MB \
         (VICTORY_CARD_CAP_MB) and {} MB under the system's budget; the PC's memory mapped for the card \
         stays under {} MB (VICTORY_SHARED_CAP_MB); buffers let go are pooled up to {} MB",
        bytes / MB,
        card_cap().map_or(0, |c| c / MB),
        CARD_RESERVE / MB,
        shared_cap().map_or(0, |c| c / MB),
        POOL_MOST / MB,
    );
}

/// The most of the card's own memory the layer's resources may take:
/// `VICTORY_CARD_CAP_MB`, else the adapter's dedicated memory less the
/// larger of 3.5 GiB and three tenths of it (8 GB: 4,608 MB; 12 GB: 8,602
/// MB; 16 GB: 11,469 MB), at least a quarter of it. What the layer does
/// not count (the desktop, the driver's and vkd3d's own allocations, the
/// render targets made without a room test) took 2.2 GB on the 8 GB card
/// beside 5.5 GB counted. `None`: not known (no adapter), no ceiling.
pub(crate) fn card_cap() -> Option<u64> {
    if let Some(cap) = env_mb("VICTORY_CARD_CAP_MB") {
        return Some(cap.max(256 * MB));
    }
    let d = DEDICATED.load(Relaxed);
    if d == 0 {
        return None;
    }
    let headroom = (3584 * MB).max(d / 10 * 3);
    Some(d.saturating_sub(headroom).max(d / 4))
}

/// What of the system's own budget for the card stays free. Under vkd3d
/// that budget is about our use plus what is free on the card, so this
/// keeps that much free.
const CARD_RESERVE: u64 = 1024 * MB;

/// The card budget the layer's rules see for the system's `budget`: at
/// most [`card_cap`], and [`CARD_RESERVE`] under the system's.
pub(crate) fn clamp_budget(budget: u64) -> u64 {
    if budget == 0 {
        return budget;
    }
    let mut b = budget.saturating_sub(CARD_RESERVE).max(budget / 4);
    if let Some(cap) = card_cap() {
        b = b.min(cap);
    }
    b
}

/// The most of the PC's memory mapped for the card (upload and shared
/// buffers, as the system counts this process's use of it):
/// `VICTORY_SHARED_CAP_MB`, else half the adapter's dedicated memory, at
/// most 4,096 MB. `None`: not known, no ceiling.
pub(crate) fn shared_cap() -> Option<u64> {
    if let Some(cap) = env_mb("VICTORY_SHARED_CAP_MB") {
        return Some(cap.max(512 * MB));
    }
    let d = DEDICATED.load(Relaxed);
    (d != 0).then(|| (d / 2).min(4096 * MB).max(1024 * MB))
}

/// The PC's memory mapped for the card as last asked of the system, when
/// (ms), and the bytes of buffers made since.
static SHARED_SEEN: Mutex<(u64, u64)> = Mutex::new((0, 0));
static SHARED_SINCE: AtomicU64 = AtomicU64::new(0);

/// Buffers refused at the ceiling.
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// Whether a new buffer of `size` bytes in the PC's memory that the card
/// maps may be made: under [`shared_cap`] (plus `over`, for the upload
/// buffers whose callers cannot do without), counting what was made since
/// the system was last asked (every tenth of a second at most). At the
/// ceiling the pool is emptied first; refused only if that is not enough.
pub(crate) fn shared_room(host: &crate::host::Host, size: u64, over: u64) -> bool {
    let Some(cap) = shared_cap() else {
        return true;
    };
    let used = |fresh: bool| -> Option<u64> {
        let mut seen = SHARED_SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let now = crate::host::now_ms();
        if fresh || seen.0 == 0 || now >= seen.0 + 100 {
            let [_, _, _, shared] = host.video_memory_raw()?;
            *seen = (now.max(1), shared);
            SHARED_SINCE.store(0, Relaxed);
        }
        Some(seen.1 + SHARED_SINCE.load(Relaxed))
    };
    let Some(now) = used(false) else {
        return true;
    };
    if now + size <= cap + over {
        SHARED_SINCE.fetch_add(size, Relaxed);
        return true;
    }
    // Free what the pool keeps and ask again.
    if drain() > 0 {
        if let Some(now) = used(true)
            && now + size <= cap + over
        {
            SHARED_SINCE.fetch_add(size, Relaxed);
            return true;
        }
    }
    let n = REFUSED.fetch_add(1, Relaxed) + 1;
    if n <= 16 || n.is_multiple_of(256) {
        tracing::warn!(
            "memory ceiling: a buffer of {} KiB in the PC's memory for the card refused: {} MB of it in use, the ceiling {} MB \
             (VICTORY_SHARED_CAP_MB); {} refused so far",
            size >> 10,
            now / MB,
            (cap + over) / MB,
            n
        );
    }
    false
}

/// Which kind of buffer a pooled one is: they are made differently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// Upload heap, write-combined, `GENERIC_READ`.
    Upload,
    /// Custom heap in the PC's memory, write-back, unordered access,
    /// common state (`Host::shared_buffer`).
    Shared,
}

/// The size a buffer of `size` bytes asked for is made with, so that
/// buffers of near sizes can be used for one another: upload buffers to
/// 64 KiB below 2 MiB and to 1 MiB above; shared buffers as asked (their
/// owners' sizes are on 64 KiB or 4 MiB lines already, and views may take
/// a shared buffer's width as the copy's).
pub(crate) fn class(kind: Kind, size: u64) -> u64 {
    let size = size.max(1);
    match kind {
        Kind::Upload if size < 2 * MB => size.next_multiple_of(64 << 10),
        Kind::Upload => size.next_multiple_of(MB),
        Kind::Shared => size,
    }
}

/// The most bytes the pool keeps. Nothing bigger than 64 MiB is kept.
const POOL_MOST: u64 = 512 * MB;

/// How long a pooled buffer is kept unused, in milliseconds.
const POOL_IDLE_MS: u64 = 20_000;

struct Entry {
    kind: Kind,
    size: u64,
    buffer: ID3D12Resource,
    /// When it was given (ms), and the submissions started then: it is
    /// handed out once they have all finished.
    since: u64,
    started: u64,
}

struct Pool {
    entries: Vec<Entry>,
    bytes: u64,
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Pool {}

static POOL: Mutex<Pool> = Mutex::new(Pool { entries: Vec::new(), bytes: 0 });

/// A pooled buffer of `kind` and exactly `size` bytes (a [`class`]) that
/// nothing else holds and no running submission can still use, with its
/// mapping; zeroed over `zero` bytes, as a new one reads.
pub(crate) fn take(kind: Kind, size: u64, zero: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let finished = crate::host::submissions_finished();
    let buffer = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let at = pool.entries.iter().rposition(|e| {
            e.kind == kind && e.size == size && e.started <= finished && crate::host::references(&e.buffer) == 1
        });
        match at {
            Some(at) => {
                pool.bytes -= size;
                pool.entries.swap_remove(at).buffer
            }
            None => return None,
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: a live CPU-visible buffer; mapping it again gives the same
    // persistent mapping.
    if unsafe { buffer.Map(0, None, Some(&mut data)) }.is_err() || data.is_null() {
        return None;
    }
    // SAFETY: the buffer's mapping holds `size` bytes, and nothing else
    // holds the buffer.
    unsafe { std::ptr::write_bytes(data.cast::<u8>(), 0, zero.min(size) as usize) };
    Some((buffer, data.cast()))
}

/// The owner of `buffer` (made by `Host::upload_buffer` or
/// `Host::shared_buffer`, `kind`) lets it go: kept for [`take`] if the
/// pool has room, else freed (when its last reference goes, as before).
pub(crate) fn give(kind: Kind, buffer: ID3D12Resource) {
    let most = POOL_MOST;
    // SAFETY: a plain query.
    let size = unsafe { buffer.GetDesc() }.Width;
    if size > 64 * MB || size > most {
        return;
    }
    let mut out = Vec::new();
    {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.bytes += size;
        pool.entries.push(Entry {
            kind,
            size,
            buffer,
            since: crate::host::now_ms(),
            started: crate::host::submissions_started(),
        });
        // The oldest first.
        while pool.bytes > most && !pool.entries.is_empty() {
            let e = pool.entries.remove(0);
            pool.bytes -= e.size;
            out.push(e);
        }
    }
    // Freed outside the lock (a buffer still held elsewhere lives on).
    drop(out);
}

/// Lets go of pooled buffers unused for [`POOL_IDLE_MS`]. Once a second,
/// with the host's sweep.
pub(crate) fn trim() {
    let now = crate::host::now_ms();
    let out: Vec<Entry> = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let (old, keep): (Vec<Entry>, Vec<Entry>) =
            std::mem::take(&mut pool.entries).into_iter().partition(|e| now >= e.since + POOL_IDLE_MS);
        pool.entries = keep;
        pool.bytes = pool.entries.iter().map(|e| e.size).sum();
        old
    };
    drop(out);
}

/// Lets go of everything pooled; the bytes.
pub(crate) fn drain() -> u64 {
    let out: Vec<Entry> = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.bytes = 0;
        std::mem::take(&mut pool.entries)
    };
    let bytes = out.iter().map(|e| e.size).sum();
    drop(out);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_sizes_are_classed() {
        assert_eq!(class(Kind::Upload, 1), 64 << 10);
        assert_eq!(class(Kind::Upload, (64 << 10) + 1), 128 << 10);
        assert_eq!(class(Kind::Upload, 2 * MB), 2 * MB);
        assert_eq!(class(Kind::Upload, 2 * MB + 1), 3 * MB);
        assert_eq!(class(Kind::Shared, 4 * MB), 4 * MB);
        assert_eq!(class(Kind::Shared, 300), 300);
    }
}
