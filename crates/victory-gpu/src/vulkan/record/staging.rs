//! Store staging (`docs/vulkan-host.md`, "Store staging"): a dispatch's
//! store views get card memory for the dispatch.
//!
//! Title memory is system RAM the card reads and writes in place, and a
//! program's scattered stores into it are what compute costs (a store
//! into cached system RAM about 250 times one into card memory; a bulk
//! copy there and back 0.24 ms a MiB). So, for a dispatch that scatters
//! enough:
//!
//! 1. each buffer view it scatters stores over gets a place in a piece
//!    of the command buffer's scratch (one piece, used by one dispatch
//!    after the other), and the view's bytes are copied in;
//! 2. the dispatch runs with a memory map of its own, in which exactly
//!    those bytes lie in the piece (the map's range cut around them);
//! 3. the bytes are copied back, and the command buffer's map is bound
//!    again.
//!
//! What a dispatch may stage is told by the program's views
//! ([`victory_shader::vk::views`], kept beside the translations:
//! `stage_cache::views`) and their V#s as the title's tables hold them
//! at record time:
//!
//! - **A cut is a whole view, to the byte**: a V# must lie whole in one
//!   range of the map, or it reads zeros and its stores are dropped. So
//!   every view of the program is held against a cut, not only the ones
//!   it stores through (views lie 12 bytes apart): one that reaches into
//!   the cut without lying inside it leaves the view in place.
//! - **Scattered stores only**: a view is staged when the program's
//!   stores through it land where data says (`View::scattered`). Stores
//!   that fill a view in the dispatch's order cost little in place (run
//!   ss2 staged those too: 95 MiB copied a frame, 22 ms of copies).
//! - **The floor**: a scattering invocation costs about 100 ns in place,
//!   a staged view 0.24 ms a MiB (both copies) and 0.1 ms; a view is
//!   staged where the first exceeds the second. The largest views first,
//!   as the map has ranges left (a cut takes up to two more).
//! - A program whose stores or views are found only at run time (a
//!   descriptor index from memory, a computed address) is left in place,
//!   counted, and named once.
//!
//! Between the copies the view's home is stale; nothing else of the
//! command buffer runs there, and what another queue does at that time
//! had no order with the dispatch before either.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_vk::ash::vk;
use victory_vk::map::MAX_RANGES;

use super::super::stages::{Stage, Stores};
use super::super::{Backend, descriptors, gputime, images, imported_over, lock, map_block, map_now, tables};
use super::Rec;

/// What a scattering invocation costs in place (ns; the pass bench),
/// what a staged view costs a MiB, both copies (the game: runs ss2 and
/// ss4; the bench had 0.16 ms), and a staged dispatch besides
/// (`docs/kiss.md`, 2026-11-28).
const IN_PLACE_NS: u64 = 100;
const COPY_NS_A_MIB: u64 = 240_000;
const STAGED_NS: u64 = 100_000;
/// Bytes after a view that are copied in with it (never back): a load of
/// several dwords at a computed address that begins in the view's last
/// bytes reads what lies behind it, as in place.
const PAD: u64 = 16;

/// For the 10 s line: dispatches with a staged view, views staged, their
/// bytes; and dispatches or views left in place: the program's stores are
/// not found, a V# was not read, below the floor, another view reaches
/// into the view, the map has no ranges left (or none holds the view),
/// the view has no home in one imported buffer (or no scratch).
static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
const NOT_FOUND: usize = 3;
const UNREAD: usize = 4;
const FLOOR: usize = 5;
const REACHED: usize = 6;
const NO_RANGE: usize = 7;
const NO_HOME: usize = 8;

fn count(what: usize) {
    STATS[what].fetch_add(1, Ordering::Relaxed);
}

/// Diagnostics: the programs staged, by loop-guard id: name, dispatches,
/// bytes.
static PROGRAMS: Mutex<Vec<(u32, String, u64, u64)>> = Mutex::new(Vec::new());

/// Diagnostics: a dispatch of the pipeline's program staged `bytes`.
fn staged_by(rec: &Rec, id: u32, bytes: u64) {
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let mut all = lock(&PROGRAMS);
    let k = all.iter().position(|p| p.0 == id).unwrap_or_else(|| {
        all.push((id, name(rec), 0, 0));
        all.len() - 1
    });
    all[k].2 += 1;
    all[k].3 += bytes;
}

/// The names of the pipeline's programs.
fn name(rec: &Rec) -> String {
    crate::pipeline::parts(rec.pipeline).map(|(s, _)| s.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join("+")).unwrap_or_default()
}

/// The 10 s line.
pub(super) fn report() {
    let n = |i: usize| STATS[i].load(Ordering::Relaxed);
    tracing::info!(
        dispatches = n(0),
        views = n(1),
        staged_mb = n(2) >> 20,
        programs_not_found = n(NOT_FOUND),
        views_unread = n(UNREAD),
        below_floor = n(FLOOR),
        reached_by_another_view = n(REACHED),
        no_range = n(NO_RANGE),
        no_home = n(NO_HOME),
        programs_read_now = super::super::stages::VIEWS_READ.load(Ordering::Relaxed),
        "Vulkan: store staging (dispatches whose store views ran on the card; left in place by reason; programs whose views no file held)"
    );
    if cfg!(feature = "diagnostics") {
        let mut all = lock(&PROGRAMS).clone();
        all.sort_by_key(|p| std::cmp::Reverse(p.3));
        let most: Vec<String> = all.iter().take(16).map(|p| format!("{} x{} {} MiB", p.1, p.2, p.3 >> 20)).collect();
        tracing::info!(programs = all.len(), most = most.join("; "), "Vulkan: store staging by program (dispatches staged, bytes copied each way; since the start)");
    }
}

/// Says once a program (and reason) why its stores stay in place.
fn once(rec: &Rec, id: u32, why: &'static str) {
    static SAID: Mutex<Option<HashSet<(u32, &'static str)>>> = Mutex::new(None);
    if lock(&SAID).get_or_insert_with(HashSet::new).insert((id, why)) {
        tracing::info!(program = name(rec), why, "Vulkan: a program's stores stay in title memory");
    }
}

/// A view's bytes in title memory, from its V# at `at` (`docs/vulkan-abi.md`
/// section 4): (base, bytes). `None`: the V# is not read.
fn view_at(at: u64) -> Option<(u64, u64)> {
    let mut bytes = [0u8; 16];
    if !crate::host::read_into(at as *const u8, &mut bytes) {
        return None;
    }
    let w = |k: usize| u32::from_le_bytes([bytes[4 * k], bytes[4 * k + 1], bytes[4 * k + 2], bytes[4 * k + 3]]);
    let base = u64::from(w(0)) | u64::from(w(1) & 0xffff) << 32;
    let stride = u64::from((w(1) >> 16) & 0x3fff).max(1);
    Some((base, (u64::from(w(2)) * stride).min(0xffff_ffff)))
}

/// A staged view: its home, its place in the command buffer's staging
/// piece, its bytes.
pub(super) struct Staged {
    home: vk::Buffer,
    offset: u64,
    at: u64,
    bytes: u64,
}

/// A view chosen for staging: its title bytes, its home, the bytes copied
/// in behind it, its place in the piece.
struct Chosen {
    base: u64,
    bytes: u64,
    home: (vk::Buffer, u64),
    pad: u64,
    at: u64,
}

impl Rec {
    /// Before a dispatch of `stage` over `groups` work groups, its
    /// pipeline bound and its constants pushed: the store views worth it
    /// are copied to the card and a map that shows them there is bound.
    /// What it returns goes to [`Rec::unstage`] after the dispatch.
    pub(super) fn stage_stores(&mut self, b: &'static Backend, cmd: vk::CommandBuffer, stage: &Stage, groups: u64) -> Vec<Staged> {
        let chosen = self.choose(b, stage, groups);
        let (Some(last), Some(heaps)) = (chosen.last(), descriptors::heaps(b)) else { return Vec::new() };
        // one piece a command buffer, as large as its largest dispatch needs
        let needed = last.at + last.bytes + last.pad + 16;
        if self.staging.is_none_or(|p| p.1 < needed) {
            self.staging = self.cmd.as_mut().and_then(|r| r.scratch(&b.host, &b.timeline, needed)).map(|p| (p, needed));
        }
        let Some((piece, _)) = self.staging else {
            count(NO_HOME);
            return Vec::new();
        };
        let mut map = map_now(b);
        let mut staged = Vec::new();
        let mut regions = Vec::new();
        for c in &chosen {
            // the piece keeps the view's place in a 16-byte line
            let at = c.at + c.base % 16;
            if map.cut(c.base, c.bytes, piece.address + at).is_err() {
                count(NO_RANGE);
                continue;
            }
            count(1);
            STATS[2].fetch_add(c.bytes, Ordering::Relaxed);
            regions.push((c.home.0, vk::BufferCopy { src_offset: c.home.1, dst_offset: piece.offset + at, size: c.bytes + c.pad }));
            staged.push(Staged { home: c.home.0, offset: c.home.1, at: piece.offset + at, bytes: c.bytes });
        }
        if staged.is_empty() {
            return staged;
        }
        count(0);
        staged_by(self, stage.guard_id, staged.iter().map(|s| s.bytes).sum());
        let (zero, sink) = heaps.zero_sink();
        let at = heaps.staged_map_offset(b, &map.block(zero, sink, tables::address(b)));
        // SAFETY: outside rendering in our command buffer; the homes are
        // imports that hold the bytes, the piece our scratch. The program
        // is marked again for the timing.
        unsafe {
            gputime::mark(b, cmd, || "host: store views to the card".into());
            for (home, region) in &regions {
                b.host.device.cmd_copy_buffer(cmd, *home, piece.buffer, std::slice::from_ref(region));
            }
            images::full_barrier(&b.host.device, cmd);
            heaps.bind(b, cmd, at);
            gputime::mark(b, cmd, || name(self));
        }
        staged
    }

    /// The store views of a dispatch that are staged, the largest first,
    /// each with its place in the piece (the module's rules).
    fn choose(&mut self, b: &'static Backend, stage: &Stage, groups: u64) -> Vec<Chosen> {
        let views = match &stage.stores {
            Stores::None => return Vec::new(),
            Stores::Unfound(why) => {
                count(NOT_FOUND);
                once(self, stage.guard_id, why);
                return Vec::new();
            }
            Stores::Views(v) => v,
        };
        let in_place = groups.saturating_mul(u64::from(stage.threads)).saturating_mul(IN_PLACE_NS);
        let worth = |bytes: u64| in_place > STAGED_NS + ((bytes * COPY_NS_A_MIB) >> 20);
        if !worth(0) {
            count(FLOOR);
            return Vec::new();
        }
        // every view as the title's tables hold it now: (base, bytes), and
        // whether the program scatters stores over it
        let found: Option<Vec<(u64, u64, bool)>> = views.views.iter().map(|v| view_at(v.at.address(&self.last_push)).map(|(base, bytes)| (base, bytes, v.scattered))).collect();
        let Some(found) = found else {
            count(UNREAD);
            once(self, stage.guard_id, "a view's V# is not read");
            return Vec::new();
        };
        let mut wanted: Vec<(u64, u64)> = found.iter().filter(|v| v.2 && v.1 != 0).map(|v| (v.0, v.1)).collect();
        wanted.sort_unstable_by_key(|v| std::cmp::Reverse(v.1));
        wanted.dedup();
        let mut ranges = None;
        let mut chosen: Vec<Chosen> = Vec::new();
        for (base, bytes) in wanted {
            if chosen.iter().any(|c| base >= c.base && base + bytes <= c.base + c.bytes) {
                continue;
            }
            if !worth(bytes) {
                count(FLOOR);
                continue;
            }
            let reaches = |v: &(u64, u64, bool)| v.1 != 0 && v.0 < base + bytes && v.0 + v.1 > base && !(v.0 >= base && v.0 + v.1 <= base + bytes);
            if found.iter().any(reaches) {
                count(REACHED);
                once(self, stage.guard_id, "another of its views reaches into a store view");
                continue;
            }
            // a cut takes up to two more ranges
            let map = ranges.get_or_insert_with(|| map_now(b));
            let held = map.range_of(base).is_some_and(|r| base + bytes <= r.title_start + r.title_size);
            if !held || map.ranges().len() + 2 * (chosen.len() + 1) > MAX_RANGES {
                count(NO_RANGE);
                once(self, stage.guard_id, if held { "the memory map has no ranges left for a store view" } else { "a store view lies in no one range of the memory map" });
                continue;
            }
            // the view's home: system RAM in one buffer (the pool's view is card memory already)
            let home = imported_over(b, base, bytes + PAD).map(|h| (h, PAD)).or_else(|| imported_over(b, base, bytes).map(|h| (h, 0)));
            let Some((home, pad)) = home else {
                count(NO_HOME);
                once(self, stage.guard_id, "a store view's home is no one imported buffer");
                continue;
            };
            let at = chosen.last().map_or(0, |c| (c.at + c.bytes + c.pad + 16).next_multiple_of(16));
            chosen.push(Chosen { base, bytes, home, pad, at });
        }
        chosen
    }

    /// After the dispatch: the staged views go back to their homes and
    /// the command buffer's map is bound again.
    pub(super) fn unstage(&mut self, b: &'static Backend, cmd: vk::CommandBuffer, staged: &[Staged]) {
        let (Some(heaps), Some((piece, _)), false) = (descriptors::heaps(b), self.staging, staged.is_empty()) else { return };
        let piece = piece.buffer;
        let (generation, block) = map_block(b, heaps.zero_sink());
        let map = heaps.map_offset(b, generation, &block);
        // SAFETY: outside rendering in our command buffer; as in
        // `stage_stores`, the other way.
        unsafe {
            gputime::mark(b, cmd, || "host: store views back".into());
            images::full_barrier(&b.host.device, cmd);
            for s in staged {
                let region = [vk::BufferCopy { src_offset: s.at, dst_offset: s.offset, size: s.bytes }];
                b.host.device.cmd_copy_buffer(cmd, piece, s.home, &region);
            }
            images::full_barrier(&b.host.device, cmd);
            heaps.bind(b, cmd, map);
            gputime::mark(b, cmd, || "host: after store views".into());
        }
        self.map_gen = generation;
    }
}
