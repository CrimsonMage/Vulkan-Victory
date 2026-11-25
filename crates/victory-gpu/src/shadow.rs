//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,

    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// The PC-memory buffer (`buffer`, or the card side's `staging`) came
    /// from `Host::shared_buffer`: when the copy goes it is given to the
    /// buffer pool (`crate::memcap`) to be used again.
    pooled: bool,
    /// With unified memory (`crate::unified`): not a copy at all. `buffer`
    /// is placed over the title's own memory, so shaders read what the
    /// title wrote and write where the title reads; nothing is refreshed,
    /// snapshotted or copied back, and no write watch is asked.
    in_place: bool,
    /// Bytes from `buffer`'s first byte to `start` (an in-place buffer
    /// starts on the 64 KiB line below the title memory); 0 for a copy.
    bias: u64,
    /// In place over a view of a section: what the title maps there may
    /// change ([`in_place_remapped`]), and then `stale` is set and the
    /// next ask places a new buffer.
    stale: std::sync::atomic::AtomicBool,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

// The host copies of title buffers are in the card's own memory (the BVH
// skinning list went from 50.5 to 4.5 ms on the card). The title's shaders
// read almost all their buffer data from a few big title buffers, and in
// the PC's memory every load crosses the bus.
//
// A title buffer of `card_limits`' size or more has two sides. Shaders see
// `Shadow::buffer`, in the card's memory. The CPU's side (`Card::staging`,
// what `Shadow::data` maps) takes title memory, and the ranges that changed in it are copied to the card's side
// before each submission (`Host::card_before`). What shaders wrote is
// found on the card: after the submission's lists a compute pass of ours
// compares each declared range, by 64 KiB piece, with the card side's
// bytes from before (`Card::before`), and only the pieces that differ are
// fetched and merged into title memory, byte by byte as
// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
// No snapshot is taken and nothing is compared on the CPU for such a copy.
//
// What it assumes: shaders write a copy only in ranges declared as
// written. A write outside them is seen by later shaders, and goes back to
// the title the next time a declared range holds its piece.

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes: from 1 MiB up (the
/// shader writes of the smaller buffers are then found by the card's
/// hashes too, and with that the first room's walls were drawn in every
/// run), and at most an eighth of the card's budget, between 256 and
/// 1,024 MB (1,024 MB on a card of 12 GB, about 690 on one of 8 GB).
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let most = crate::host::host()
            .and_then(|h| h.video_memory())
            .map_or(1024, |[budget, ..]| (budget >> 20) / 8)
            .clamp(256, 1024);
        (1 << 20, most << 20)
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory copies of title buffers take, for the
/// memory line.
pub(crate) fn card_bytes() -> u64 {
    CARD_BYTES[0].load(Ordering::Relaxed) + CARD_BYTES[1].load(Ordering::Relaxed)
}

// The texture queue leaves room on the card for the card copies of title
// buffers that are not made yet ([`card_room_kept`]). The copies are made in
// one burst when the world loads, and the texture queue fills the card
// while the world loads: its rule (`resource::room`) counts the part of
// [`card_limits`]' second number that no copy has taken yet as used, and a
// copy may be made up to 97 hundredths of the budget (its room was kept).
// The room of a copy turned down is given up ([`CARD_REFUSED`]): a copy
// turned down is not asked for again (`lookup` asks once).

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet.
pub(crate) fn card_room_kept() -> u64 {
    card_limits().1.saturating_sub(CARD_BYTES[0].load(Ordering::Relaxed) + CARD_REFUSED[1].load(Ordering::Relaxed))
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: the size is within [`card_limits`]
/// and below 1 GiB (the compare counts in 32 bits), and the host has what
/// it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // The texture queue kept this copy's room free ([`card_room_kept`]), and
    // the copy may go up to the queue's own line.
    let hundredths = 97;
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}


/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow
            .card
            .as_ref()
            .map(|c| {
                let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let ranges = std::mem::take(&mut *dirty);
                let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                sent.retain(|s| s.2 > fetched);
                sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
                ranges
            })
            .unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
        true
    });
    out
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}


/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined};

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

// The copy-back of what shaders wrote runs on a thread of its own, beside
// the submitting thread and outside the submission lock (`crate::host`'s
// relay). Here that changes what a copy keeps:
//
// - the ranges shaders may be writing are counted per submission
//   ([`Flights`]): a range two flights declared stays closed to refreshes
//   until both have been copied back;
// - the merge of what the card wrote never stores into memory the card
//   may be writing ([`merge_piece`]): what it learns goes into title
//   memory and into the copy's reference (the snapshot, or the CPU's side
//   of a copy in the card's memory), under that reference's lock;
// - what goes up to a copy in the card's memory is numbered
//   ([`card_dirty`]), so that a merge can tell the title's bytes the card
//   had not received from bytes the card wrote.

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
pub(crate) fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

// A submission's snapshots of its shader-written ranges are noted as it
// is prepared
// A submission's snapshots of its shader-written ranges are noted as it is prepared
// and taken together when its refreshes are done, just before it goes to
// the card ([`take_snapshots`]), and the compares of everything declared
// run together after the card ([`find_all`]), both cut into pieces of
// work of about 1 MiB for a few threads that are kept ([`wide`]). The
// bytes taken and the bytes compared are the same ones: a range's bytes
// cannot change between the moment it is declared and the end of the
// preparing (a declared range is left alone by every refresh, the title's
// own copies write title memory and not the copy, and the card runs the
// lists only afterwards).

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        let n = std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(2, 6);
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                let workers = WORKERS.get().expect("made above");
                loop {
                    let job = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    job();
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let left = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

#[cfg(test)]
mod merge_tests {
    use super::{PIECE, back_piece, changed_runs};

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if self.in_place {
            IN_PLACE[0].fetch_sub(1, Ordering::Relaxed);
            IN_PLACE[1].fetch_sub(self.len, Ordering::Relaxed);
        }
        if self.pooled {
            let shared = self.card.as_ref().map_or_else(|| self.buffer.clone(), |card| card.staging.clone());
            crate::memcap::give(crate::memcap::Kind::Shared, shared);
        }
        if let Some(card) = &mut self.card {
            CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}


/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        idle.into_iter().filter_map(|key| map.remove(&key)).collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        !self.in_place && self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        self.bias + address - self.start
    }

    /// Whether the buffer is placed over the title's memory itself
    /// (unified memory): nothing to copy back.
    pub(crate) fn in_place(&self) -> bool {
        self.in_place
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.in_place {
            // Nothing to bring up to date: only kept alive while the list
            // may run (a remapped view drops it from the tables).
            if list.first_use(Arc::as_ptr(self) as usize | 1) {
                list.keep(&self.buffer);
                crate::frametime::add(crate::frametime::Sum::InPlaceBytes, self.len);
            }
            return;
        }
        if self.card.is_some() && list.first_use(Arc::as_ptr(self) as usize | 1) {
            crate::frametime::add(crate::frametime::Sum::CardBytes, self.len);
        }
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            pooled: false,
            in_place: false,
            bias: 0,
            stale: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        if self.in_place {
            // The shaders write the title's memory itself: declared all the
            // same, so that what is prepared on the CPU while the
            // submission runs waits where it writes (`host.rs`
            // `late_blocked`); at submission it is noted
            // ([`Shadow::writing`]) and announced once the submission has
            // finished (`crate::unified::settle`).
            self.refresh(list, address, len);
        }
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.in_place {
            // Nothing kept to compare: the title's bytes are the GPU's.
            crate::unified::gpu_wrote(address, len);
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // The copy-back is on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        // Noted, and taken with the submission's other snapshots when its
        // refreshes are done (`take_snapshots`): the range is declared from
        // here on, so nothing changes its bytes before then. The snapshot's
        // memory stays where it is until the range is done with
        // (`copy_back_done`).
        let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
        let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
        snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                let at = at + first;
                crate::host::shaders_write(start + at, &title[..n.min(title.len())]);
                crate::watch::writing(start + at, n);
                // Only the GPU's bytes (upstream #2009): the rest of the
                // stretch is the title's as read, and a title write landing
                // there since stays.
                for &(off, len) in &runs {
                    crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                    written += len;
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        crate::watch::writing(start + at, n);
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// Several submissions may hold a range: `ranges` are those of one of
    /// them (each once), and a range is done with when the last submission
    /// that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.in_place {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
        pending.retain(|r| !free.contains(r));
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }


    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Where a range shaders may be writing holds a page the title wrote
    /// since the copy last took it: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays. For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let mut title = vec![0u8; PAGE];
        for (a, e) in busy {
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                return true;
            };
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied, a, e, &written)
            };
            for page in stale {
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    continue;
                }
                let n = hi - lo;
                let title = &mut title[..n];
                // The lock of the copy's reference (the snapshot, or the
                // CPU's side of a copy in the card's memory), from before
                // title memory is read until the title's bytes are in: with
                // the copy-back on its own thread ([`relay`]) that thread
                // puts what the card wrote into title memory and into the
                // reference under the same lock, and a page read before
                // such a merge and compared after it would take the card's
                // bytes for the title's.
                let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
                // A page that is not there is left as it is, and looked at
                // again next time.
                if !crate::host::read_into(lo as *const u8, title) {
                    continue;
                }
                MERGED[0].fetch_add(1, Ordering::Relaxed);
                let at = lo - start;
                if self.card.is_some() {
                    // SAFETY: inside the mapping of the copy's CPU side,
                    // which only the submitting thread writes.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let runs = changed_runs(title, held);
                    if runs.len() > *runs_left {
                        return true;
                    }
                    *runs_left -= runs.len();
                    for (r, m) in runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        self.card_changed(at + r, at + r + m);
                        MERGED[1].fetch_add(1, Ordering::Relaxed);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                } else {
                    let Some(snapshot) = before.as_mut() else {
                        return true;
                    };
                    // SAFETY: the snapshot's allocation holds the copy's
                    // bytes at the copy's offsets (`writing`), and these
                    // are in a declared range.
                    let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                    for (r, m) in changed_runs(title, held) {
                        // SAFETY: inside the copy's mapping. The card may be
                        // writing the copy beside these bytes; they are
                        // written here and nothing around them.
                        unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        MERGED[1].fetch_add(1, Ordering::Relaxed);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                }
                drop(before);
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(at) = copied.get_mut(page - base) {
                        *at = now;
                    }
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.card_changed(from - start, to - start);
        to - from
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                crate::host::shaders_write(start + at, &title[..]);
                crate::watch::writing(start + at, n);
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                .filter(|&(a, b)| a < at + n && b > at)
                .collect()
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        crate::watch::writing(start + at, n);
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }
}

/// In-place buffers alive, and their bytes of title memory.
static IN_PLACE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// See [`IN_PLACE`].
pub(crate) fn in_place_live() -> (u64, u64) {
    (IN_PLACE[0].load(Ordering::Relaxed), IN_PLACE[1].load(Ordering::Relaxed))
}

/// In-place buffers over views of sections, by (start, id): end and the
/// buffer; and the longest one's bytes.
#[allow(clippy::type_complexity)]
static VIEWED: Mutex<(BTreeMap<(u64, u64), (u64, std::sync::Weak<Shadow>)>, u64)> = Mutex::new((BTreeMap::new(), 0));

/// A buffer placed over title memory `[start, start + len)`, if
/// `crate::unified` can open it; `None` leaves the caller to copy.
fn in_place(start: u64, len: u64, window: bool) -> Option<Arc<Shadow>> {
    let placed = crate::unified::place(start, len)?;
    let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
    shadow.in_place = true;
    shadow.bias = placed.bias;
    shadow.window = window;
    IN_PLACE[0].fetch_add(1, Ordering::Relaxed);
    IN_PLACE[1].fetch_add(len, Ordering::Relaxed);
    let shadow = Arc::new(shadow);
    if placed.view {
        let mut viewed = VIEWED.lock().unwrap_or_else(|e| e.into_inner());
        viewed.1 = viewed.1.max(len);
        viewed.0.insert((start, shadow.id), (start + len, Arc::downgrade(&shadow)));
    }
    Some(shadow)
}

/// The runtime mapped title memory `[at, at + len)` anew: in-place buffers
/// over views there show the old pages, so the next ask places new ones
/// (lists and slots that hold the old ones keep them alive until done).
pub(crate) fn in_place_remapped(at: usize, len: usize) {
    let (at, end) = (at as u64, (at + len) as u64);
    let mut guard = VIEWED.lock().unwrap_or_else(|e| e.into_inner());
    let (viewed, longest) = &mut *guard;
    if viewed.is_empty() {
        return;
    }
    let from = at.saturating_sub(*longest);
    let hit: Vec<(u64, u64)> = viewed.range((from, 0)..(end, 0)).filter(|(_, (e, _))| *e > at).map(|(k, _)| *k).collect();
    for key in hit {
        if let Some((_, weak)) = viewed.remove(&key)
            && let Some(shadow) = weak.upgrade()
        {
            shadow.stale.store(true, Ordering::Relaxed);
        }
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len)
}
fn lookup(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.as_ref().is_some_and(|s| s.stale.load(Ordering::Relaxed)) {
        entry.shadow = None;
    }
    if entry.shadow.is_none()
        && crate::unified::on()
        && entry.end - key.0 < crate::unified::BIG
        && let Some(shadow) = in_place(key.0, entry.end - key.0, false)
    {
        entry.shadow = Some(shadow);
    }
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        let (buffer, data) = host()?.shared_buffer(size)?;
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the one just made is the CPU's side.
        let (buffer, card) = match card_for(size) {
            Some(on_card) => {
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging: buffer,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    size,
                };
                (on_card, Some(card))
            }
            None => (buffer, None),
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        shadow.pooled = true;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned 4 MiB windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read. An ask
/// without a length (a root view) reads from its address to its window's
/// end, and mesh streams bound that way reach megabytes.
const WINDOW: u64 = 4 << 20;

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)`.
fn window_of(address: u64, end: u64) -> (u64, u64) {
    let start = address & !(WINDOW - 1);
    (start, end.next_multiple_of(WINDOW) - start)
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool) -> Option<Arc<Shadow>> {
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let (start, len) = window_of(address, end);
    // Under the lock, with the handing out: see `let_go_idle`.
    let note = |shadow: &Shadow| {
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    if map.get(&(start, len)).is_some_and(|s| s.stale.load(Ordering::Relaxed)) {
        map.remove(&(start, len));
    }
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        return Some(shadow.clone());
    }
    if let Some(shadow) = in_place(start, len, true) {
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // In a section's views the 4 MiB window often is not one run (the
    // views are smaller, or end, or show pages of another section): then
    // the run of the view the ask lies in, from the ask's 64 KiB page to
    // where the run or the window ends, if it holds the ask. Without this
    // every such ask took a 4 MiB copy in the PC's memory, and at the world
    // start thousands of them hit the ceiling in one frame (147,000 refused
    // in the run of 10:33).
    if crate::unified::on() {
        let from = address & !0xffff;
        let reach = crate::views::run_reach(from as usize, (start + len - from) as usize) as u64;
        if reach >= end - from {
            if let Some(shadow) = map.get(&(from, reach)).filter(|s| !s.stale.load(Ordering::Relaxed)) {
                note(shadow);
                return Some(shadow.clone());
            }
            if let Some(shadow) = in_place(from, reach, true) {
                note(&shadow);
                map.insert((from, reach), shadow.clone());
                return Some(shadow);
            }
        }
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.pooled = true;
    let shadow = Arc::new(shadow);
    note(&shadow);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `own_bytes`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, address, len);
    shadow.pooled = true;
    let shadow = Arc::new(shadow);
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, window_of};

    /// A window holds its ask and is cut on the 4 MiB grid.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end), (0xd_1200_0000, 4 << 20));
        // An ask across a grid line takes both cells.
        let end = 0xd_1240_0010;
        assert_eq!(window_of(address, end), (0xd_1200_0000, 2 * WINDOW));
    }
}

