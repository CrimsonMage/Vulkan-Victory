//! The card's page cache for the pool (`docs/kiss.md`, "Unified memory on
//! split hardware"; `docs/unified-memory.md`, "The card's page cache").
//!
//! Pool pages are streamed once and then read by the card, over and over:
//! their home on a PC is the card's memory. Read through the directory
//! from the opened sections they cross the bus (12.6 GB/s against about
//! 199 GB/s from the card's own memory). So a **physical** pool page that a
//! list names (a pool view, the first 128 KiB of a root view) gets a copy
//! in a block of the card's memory, one copy however many title pages
//! show it, and the directory entry of every title page that shows it
//! names the copy instead of the section (`Memory::record_head`: the
//! entries stay the section's in the mirror and are translated when the
//! blocks go to the card).
//!
//! - **Blocks.** 64 MiB buffers in the card's memory, 1,024 slots of
//!   64 KiB each, each block one copy of the directory (a raw view, as a
//!   section is). Blocks are made while the card has room (under 90% of the
//!   layer's budget, `crate::memcap`) and one is given back per submission
//!   while it is over 95%, its pages evicted first.
//! - **Taken in once.** A page is copied in at the head of the first
//!   submission that names it (views of the heap that shaders picking
//!   buffers by index may read: only into room the cache has, never
//!   evicting for them, and they do not count as named), and again only
//!   when the title announces
//!   that it changed: DirectStorage completion, file reads, title copies
//!   on the CPU and page mappings (`crate::unified::written` -> [`written`]).
//!   An announced write into a page that no list named yet but that a
//!   title page in the directory shows takes it in ahead, into room the
//!   cache has (no eviction for it).
//! - **Shader writes** land in the copy. Each page that a list's shaders
//!   write by directory (declared: `crate::memory::named` with writes) has
//!   a **twin**, a second slot holding what the copy held before the list.
//!   Right after the list, a pass of ours compares the copy with its twin
//!   word by word and stores only the words that changed into the page
//!   itself (and into the twin): what the title's CPU wrote into the page
//!   meanwhile, announced or not, is never overwritten by a word the
//!   shaders left alone. So a page is never dirty past its list: CPU
//!   reads, windows (`crate::gather`) and in-place buffers of later lists
//!   see the page as the shaders left it, and eviction needs no copy back.
//!   Without the pass (it did not compile), pages that shaders write are
//!   not cached.
//! - **Other writes into a cached page** (shaders through an in-place
//!   buffer, title copies on the card: they write the page's own memory)
//!   copy the page in again right after the list.
//! - **Writes nobody declares view by view**: shaders that pick buffers
//!   from the heap by index and write them may write any pool page an
//!   unordered access view of the heap names. Those pages are evicted
//!   before such a list and not cached for [`HOLD`] frames after the last
//!   one: their writes go to the pages themselves, as before the cache.
//! - **Eviction**, when no slot is free and no block can be made: the least
//!   recently named page (a clock over the slots, by the frame it was last
//!   named), never one named in this frame or the last (so a working set
//!   larger than the cache keeps what it has and reads the rest from the
//!   pages, rather than churning). Evicted entries name the section again
//!   in the same head list. A slot is reused only in a later head list:
//!   queue order puts it after every list that may still read it.
//!
//! shortcut: writes that raw-address code makes into pool pages are not
//! declared (as before for windows): with the cache they land in the copy
//! and are lost when it is taken in again or evicted (the ray-tracing
//! builder is off by default).
//! shortcut: CPU writes the title makes into pool pages without saying so
//! reach the card's copy only with the next announced change of the page,
//! or with a write of the card's into it (as for windows).
//! shortcut: inside one list, a page written by directory and read
//! through an in-place buffer or a window shows the bytes from before the
//! list (copied back after it).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use super::{BLOCK, COPIES, Copy, HostHeap, MEMORY_PAGES, Memory, Named, PAGE, Running, own_list};
use crate::fast::{FastMap, FastSet};
use crate::unified::Home;

/// Bytes of a block of the cache.
pub(super) const BLOCK_BYTES: u64 = 64 << 20;
/// Slots of a block.
const PER_BLOCK: u32 = (BLOCK_BYTES / PAGE) as u32;
/// A slot's home when it is a twin.
const TWIN: u64 = u64::MAX;
/// No slot.
const NONE: u32 = u32::MAX;
/// Blocks are made while the layer's use of the card stays under this
/// share of memcap's ceiling (percent; `memcap::card_cap`, from the
/// adapter's real memory: not the system's budget figure, which under
/// vkd3d falls as the card fills).
const GROW: u64 = 90;
/// Over this share of the ceiling a block is given back per submission.
const SHRINK: u64 = 95;
/// What of the card stays free besides (the driver's budget less its use):
/// blocks are made only while this much would be left, and given back
/// below half of it. 1 GiB: under vkd3d the driver keeps ~3 GB of its own
/// (pipelines, local memory) outside the budget figure, so less left the
/// card full (run p26: 7.2 GB at frame 990).
const FREE_KEEP: u64 = 1 << 30;
/// Blocks at most (6 GiB; the Series S pool is 3,977 MiB, plus twins).
const MOST_BLOCKS: usize = 96;
/// Blocks at most when the card's ceiling is not known.
const BLOCKS_UNKNOWN: usize = 16;
/// Pages newly taken in by one submission at most (the rest is read from
/// the pages this time); pages taken in again because they changed always
/// are.
const NEW_MOST: u64 = 512 << 20;
/// Pages taken in ahead of their first naming by one submission at most.
const AHEAD_MOST: u64 = 64 << 20;
/// Announced pages kept to be taken in ahead at most.
const WANTED_MOST: usize = 1 << 16;
/// Announced writes kept between two submissions at most; past it every
/// cached page is taken in again.
const QUEUED_MOST: usize = 1 << 20;
/// Frames a pool page stays uncached after shaders that pick buffers from
/// the heap by index may have written it.
const HOLD: u64 = 120;

#[derive(Clone, Copy)]
struct Slot {
    /// The directory entry of the section page it holds (0: free; [`TWIN`]).
    home: u64,
    /// The frame (+2) it was last named in.
    used: u64,
    /// Its twin's slot; for a twin, the slot it is the twin of.
    twin: u32,
    /// Copied in at the next head list.
    due: bool,
}

impl Slot {
    const FREE: Slot = Slot { home: 0, used: 0, twin: NONE, due: false };
}

struct Block {
    copy: u32,
    buffer: ID3D12Resource,
    gpu: u64,
}

#[derive(Default)]
pub(super) struct Cache {
    blocks: Vec<Option<Block>>,
    /// Slot `s` is slot `s % PER_BLOCK` of block `s / PER_BLOCK`.
    slots: Vec<Slot>,
    free: Vec<u32>,
    /// By section page (its directory entry): the slot holding it.
    by_home: FastMap<u64, u32>,
    /// By section page: the directory pages that show it.
    aliases: FastMap<u64, Vec<u32>>,
    /// Slots to copy in at the next head list (checked against `due`).
    due: Vec<u32>,
    /// Section pages announced written that no slot holds yet.
    wanted: FastSet<u64>,
    /// The clock's hand.
    hand: usize,
    /// Copy numbers of blocks given back, with the submission after which
    /// no list uses them; then numbers free for another block.
    released: Vec<(u32, u64)>,
    spare: Vec<u32>,
    /// Twins alive.
    twins: u64,
    /// Section pages not cached until a frame (+2): shaders that pick
    /// buffers from the heap by index write there.
    held: FastMap<u64, u64>,
    /// No block could be made in this submission's plan; and none is tried
    /// before this time (ms) after the card refused one.
    no_grow: bool,
    grow_after: u64,
}

impl Cache {
    /// Directory page `page` showed section page `old` and shows `new` now.
    pub(super) fn moved(&mut self, page: u32, old: u64, new: u64) {
        if old != 0
            && let Some(pages) = self.aliases.get_mut(&old)
        {
            pages.retain(|&p| p != page);
            if pages.is_empty() {
                self.aliases.remove(&old);
            }
        }
        if new != 0 {
            self.aliases.entry(new).or_default().push(page);
        }
    }

    /// Whether the cache ever had a block (the directory then must follow
    /// it at every submission).
    pub(super) fn active(&self) -> bool {
        !self.blocks.is_empty()
    }

    /// What the card's directory holds for mirror entry `entry`: the
    /// cache's copy if it holds the page.
    pub(super) fn entry(&self, entry: u64) -> u64 {
        if entry == 0 || self.by_home.is_empty() {
            return entry;
        }
        self.by_home.get(&entry).map_or(entry, |&s| self.card_entry(s))
    }

    fn card_entry(&self, s: u32) -> u64 {
        let copy = self.blocks[(s / PER_BLOCK) as usize].as_ref().map_or(0, |b| b.copy);
        u64::from(copy) | (u64::from(s % PER_BLOCK) * PAGE) << 32
    }

    /// Slot `s`'s buffer, byte offset in it and GPU address.
    fn at(&self, s: u32) -> Option<(ID3D12Resource, u64, u64)> {
        let b = self.blocks.get((s / PER_BLOCK) as usize)?.as_ref()?;
        let offset = u64::from(s % PER_BLOCK) * PAGE;
        Some((b.buffer.clone(), offset, b.gpu + offset))
    }
}

/// A page's copy from one buffer into another.
struct CopyOp {
    dst: ID3D12Resource,
    dst_at: u64,
    src: ID3D12Resource,
    src_at: u64,
}

/// Records `ops`, joining those that continue one another.
fn record_copies(list: &ID3D12GraphicsCommandList, ops: &[CopyOp]) {
    let same = |a: &ID3D12Resource, b: &ID3D12Resource| a.as_raw() == b.as_raw();
    let mut i = 0;
    while i < ops.len() {
        let mut len = PAGE;
        let mut j = i + 1;
        while j < ops.len()
            && same(&ops[j].dst, &ops[i].dst)
            && same(&ops[j].src, &ops[i].src)
            && ops[j].dst_at == ops[i].dst_at + len
            && ops[j].src_at == ops[i].src_at + len
        {
            len += PAGE;
            j += 1;
        }
        // SAFETY: an open list; live buffers, each copy inside both.
        unsafe { list.CopyBufferRegion(&ops[i].dst, ops[i].dst_at, &ops[i].src, ops[i].src_at, len) };
        i = j;
    }
}

/// What the head list of a submission copies for the cache, and what is
/// recorded after each of its lists.
#[derive(Default)]
pub(super) struct Plan {
    /// From the sections into slots.
    uploads: Vec<CopyOp>,
    /// From slots into their twins, after the uploads.
    twins: Vec<CopyOp>,
    pub(super) kept: Vec<ID3D12Resource>,
    /// Per list of the submission, or empty if none has anything.
    pub(super) after: Vec<After>,
}

impl Plan {
    pub(super) fn is_empty(&self) -> bool {
        self.uploads.is_empty() && self.twins.is_empty()
    }

    /// Into the head list, after a barrier and before the directory.
    pub(super) fn record_in(&self, list: &ID3D12GraphicsCommandList, list7: Option<&ID3D12GraphicsCommandList7>) {
        record_copies(list, &self.uploads);
        if !self.twins.is_empty() {
            crate::host::order_writes(list, list7);
            record_copies(list, &self.twins);
        }
    }
}

/// What is recorded after one list: the comparisons of written pages with
/// their twins (GPU addresses: copy, twin, page), pages copied in again
/// (written in place) and their twins refreshed.
#[derive(Default)]
pub(super) struct After {
    diffs: Vec<[u64; 3]>,
    uploads: Vec<CopyOp>,
    twins: Vec<CopyOp>,
    kept: Vec<ID3D12Resource>,
}

impl After {
    fn is_empty(&self) -> bool {
        self.diffs.is_empty() && self.uploads.is_empty()
    }
}

/// Holds `buffer` in `kept` once.
fn keep(kept: &mut Vec<ID3D12Resource>, seen: &mut FastSet<usize>, buffer: &ID3D12Resource) {
    if seen.insert(buffer.as_raw() as usize) {
        kept.push(buffer.clone());
    }
}

const SOURCE: &str = r#"
#define RS "UAV(u0), UAV(u1), UAV(u2)"
RWByteAddressBuffer card : register(u0);
RWByteAddressBuffer twin : register(u1);
RWByteAddressBuffer page : register(u2);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // 64 groups of 64 threads, 16 bytes each: one 64 KiB page.
    uint at = id.x * 16;
    uint4 now = card.Load4(at);
    uint4 was = twin.Load4(at);
    if (any(now != was)) {
        twin.Store4(at, now);
        if (now.x != was.x) page.Store(at, now.x);
        if (now.y != was.y) page.Store(at + 4, now.y);
        if (now.z != was.z) page.Store(at + 8, now.z);
        if (now.w != was.w) page.Store(at + 12, now.w);
    }
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// The pass that stores the words shaders changed in a cached page.
fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let made = (|| -> windows::core::Result<Pass> {
            let device = &crate::host::host().ok_or_else(windows::core::Error::empty)?.device;
            let code = crate::host::compile(SOURCE, s!("main"), s!("cs_5_1"))?;
            // SAFETY: our compiled shader, its root signature inside.
            unsafe {
                let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
                let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                    ..Default::default()
                })?;
                Ok(Pass { root, pso })
            }
        })();
        match made {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(error = %e, "card page cache: no pass to copy shader writes back; pool pages shaders write stay uncached");
                None
            }
        }
    })
    .as_ref()
}

/// The list of ours to run right after a list (`after`): `None` if there
/// is nothing.
pub(super) fn record_after(after: &After) -> Option<ID3D12CommandList> {
    if after.is_empty() {
        return None;
    }
    let mut c = own_list()?;
    let pass = pass();
    // SAFETY: our list, not executing; our root signature and pipeline;
    // every address is a page inside a live buffer the list keeps.
    let recorded = unsafe {
        c.allocator.Reset().and_then(|()| c.list.Reset(&c.allocator, None)).map(|()| {
            crate::host::order_writes(&c.list, c.list7.as_ref());
            if let (false, Some(pass)) = (after.diffs.is_empty(), pass) {
                c.list.SetComputeRootSignature(&pass.root);
                c.list.SetPipelineState(&pass.pso);
                for &[card, twin, page] in &after.diffs {
                    c.list.SetComputeRootUnorderedAccessView(0, card);
                    c.list.SetComputeRootUnorderedAccessView(1, twin);
                    c.list.SetComputeRootUnorderedAccessView(2, page);
                    c.list.Dispatch((PAGE / 16 / 64) as u32, 1, 1);
                }
                crate::host::order_writes(&c.list, c.list7.as_ref());
            }
            if !after.uploads.is_empty() {
                record_copies(&c.list, &after.uploads);
                crate::host::order_writes(&c.list, c.list7.as_ref());
                if !after.twins.is_empty() {
                    record_copies(&c.list, &after.twins);
                    crate::host::order_writes(&c.list, c.list7.as_ref());
                }
            }
        })
    }
    // SAFETY: our list, recorded above.
    .and_then(|()| unsafe { c.list.Close() });
    if let Err(e) = recorded {
        crate::host::fail(format_args!("the card page cache's copies back did not record ({e})"));
    }
    c.kept.extend(after.kept.iter().cloned());
    let list = c.list.cast().ok();
    crate::host::retire(Running(Some(c)));
    list
}

// Announced writes, between submissions.

/// Whether the cache has a block: before that (and whenever it has none)
/// lists name nothing for it and announced writes are not kept.
static LIVE: AtomicBool = AtomicBool::new(false);

/// Whether the cache has a block ([`LIVE`]).
pub(super) fn live() -> bool {
    LIVE.load(Ordering::Relaxed)
}

/// Bytes of idle host copies of textures the cache asks the sweep to let go
/// of for its room (`crate::resource::let_go_idle`); taken by the sweep.
static ROOM_WANTED: AtomicU64 = AtomicU64::new(0);

pub(super) fn room_wanted() -> u64 {
    ROOM_WANTED.swap(0, Ordering::Relaxed)
}

/// The card's room for the cache: whether a block may be made, whether one
/// must be given back, and what is short for a block (bytes); `None` if the
/// card's memory is not known.
fn room() -> Option<(bool, bool, u64)> {
    let host = crate::host::host()?;
    let cap = crate::memcap::card_cap()?;
    let (used, _) = crate::memcap::card_used(host)?;
    let [budget, ..] = host.video_memory_raw()?;
    // What the driver still has (its budget is our use plus what is free).
    let free = budget.saturating_sub(used);
    let under = (cap / 100 * GROW).saturating_sub(used);
    let left = free.saturating_sub(FREE_KEEP);
    let grow = under >= BLOCK_BYTES && left >= BLOCK_BYTES;
    let shrink = used > cap / 100 * SHRINK || free < FREE_KEEP / 2;
    let short = if grow { 0 } else { BLOCK_BYTES.saturating_sub(under.min(left)) };
    Some((grow, shrink, short))
}
/// Announced writes since the last submission: section, offset, bytes.
static QUEUED: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());
/// More were announced than kept: every cached page is taken in again.
static OVERFLOW: AtomicBool = AtomicBool::new(false);

/// Title memory `[at, at + len)` was written by a writer that says so: the
/// section pages under it, for the next submission's head list. Takes the
/// views' and units' locks (`crate::unified::spans`), then the queue's
/// alone; not the directory's.
pub(super) fn written(at: u64, len: u64) {
    if !live() || len == 0 {
        return;
    }
    let mut spans = Vec::new();
    crate::unified::spans(at, len, &mut spans);
    if spans.is_empty() {
        return;
    }
    let mut q = QUEUED.lock().unwrap_or_else(|e| e.into_inner());
    for (home, offset, bytes) in spans {
        if let Home::Section(handle) = home {
            q.push((handle, offset, bytes));
        }
    }
    if q.len() > QUEUED_MOST {
        q.clear();
        OVERFLOW.store(true, Ordering::Relaxed);
    }
}

// Statistics for the frame line.

/// Since the last line: bytes copied in, bytes of pages compared after
/// their lists, bytes evicted, named pages served from the card, named
/// pages that could be, lists whose heap-picking writers had no known
/// views.
static SUMS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Now: pages cached, twins, blocks, pages held off.
static NOW: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A shader that picks buffers from the heap by index writes, and the
/// heap's unordered access views are not known yet.
pub(super) fn heap_written_unknown() {
    if live() && SUMS[5].fetch_add(1, Ordering::Relaxed) == 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!("card page cache: a shader writes buffers it picks from the heap by index, whose views are not known yet: what it writes into cached pool pages may be lost")
        });
    }
}

pub(super) fn frame_text(frames: f64) -> String {
    let take = |i: usize| SUMS[i].swap(0, Ordering::Relaxed);
    let now = |i: usize| NOW[i].load(Ordering::Relaxed);
    let (up, back, out, served, named, unknown) = (take(0), take(1), take(2), take(3), take(4), take(5));
    let mb = |b: u64| b as f64 / frames / 1e6;
    format!(
        "pool page cache: {} MB cached (twins {} MB; {} blocks, {} MB of the card; {} MB held off for heap-picked writes), taken in {:.1} MB, \
         written back {:.1} MB (pages compared, changed words stored), evicted {:.1} MB, pool pages named served from the card {:.0}%{}",
        (now(0) * PAGE) >> 20,
        (now(1) * PAGE) >> 20,
        now(2),
        (now(2) * BLOCK_BYTES) >> 20,
        (now(3) * PAGE) >> 20,
        mb(up),
        mb(back),
        mb(out),
        if named > 0 { served as f64 * 100.0 / named as f64 } else { 0.0 },
        if unknown > 0 { format!(", heap-picked writes with views not known {:.1}/frame", unknown as f64 / frames) } else { String::new() },
    )
}

/// Directory page numbers of `[address, address + len)`.
fn pages(address: u64, len: u64) -> std::ops::Range<u64> {
    let last = u64::from(MEMORY_PAGES) - 1;
    (address / PAGE).min(last)..address.saturating_add(len.max(1)).div_ceil(PAGE).min(last)
}

impl Memory {
    /// The section buffer and byte offset of section page `home`, if the
    /// cache may hold it: a page of an opened section, on a 64 KiB line.
    fn section_of(&self, home: u64) -> Option<(ID3D12Resource, u64)> {
        let (n, offset) = (home as u32, home >> 32);
        if home == 0 || home == TWIN || n == 0 || offset % PAGE != 0 {
            return None;
        }
        match self.copies.get(n as usize - 1)? {
            Copy::Section(_, Some(buffer), _) => Some((buffer.clone(), offset)),
            _ => None,
        }
    }

    /// The directory blocks of the pages that show section page `home`
    /// go to the card again.
    fn aliases_dirty(&mut self, home: u64) {
        if let Some(pages) = self.cache.aliases.get(&home) {
            for &p in pages {
                let block = p as usize / BLOCK;
                self.dirty[block / 64] |= 1 << (block % 64);
                self.any_dirty = true;
            }
        }
    }

    /// Slot `s` holds section page `home` from now on, copied in at this
    /// head list.
    fn assign(&mut self, s: u32, home: u64, used: u64) {
        self.cache.slots[s as usize] = Slot { home, used, twin: NONE, due: true };
        self.cache.by_home.insert(home, s);
        self.cache.due.push(s);
        self.aliases_dirty(home);
    }

    /// Slot `s` (a page) and its twin are let go; the pages that show it
    /// name its section again.
    fn evict(&mut self, s: u32) {
        let slot = self.cache.slots[s as usize];
        if slot.home == 0 || slot.home == TWIN {
            return;
        }
        self.cache.by_home.remove(&slot.home);
        if slot.twin != NONE {
            self.cache.slots[slot.twin as usize] = Slot::FREE;
            self.cache.free.push(slot.twin);
            self.cache.twins -= 1;
        }
        self.cache.slots[s as usize] = Slot::FREE;
        self.cache.free.push(s);
        self.aliases_dirty(slot.home);
        SUMS[2].fetch_add(PAGE, Ordering::Relaxed);
    }

    /// A free slot: one free, else one of a new block if the card has room,
    /// else (with `evict`) the least recently named page's.
    fn take_slot(&mut self, now: u64, evict: bool) -> Option<u32> {
        if let Some(s) = self.cache.free.pop() {
            return Some(s);
        }
        if !self.cache.no_grow {
            if self.grow() {
                return self.cache.free.pop();
            }
            self.cache.no_grow = true;
        }
        if !evict {
            return None;
        }
        let n = self.cache.slots.len();
        for _ in 0..n {
            let s = self.cache.hand % n;
            self.cache.hand = s + 1;
            let slot = self.cache.slots[s];
            if slot.home != 0 && slot.home != TWIN && slot.used + 2 <= now {
                self.evict(s as u32);
                return self.cache.free.pop();
            }
        }
        None
    }

    /// Makes a block if the card has room for it.
    fn grow(&mut self) -> bool {
        let live = self.cache.blocks.iter().filter(|b| b.is_some()).count();
        if live >= MOST_BLOCKS {
            return false;
        }
        let Some(host) = crate::host::host() else {
            return false;
        };
        if crate::host::now_ms() < self.cache.grow_after {
            return false;
        }
        match room() {
            Some((true, _, _)) => {}
            Some((false, _, short)) => {
                // Idle host copies of textures make room first (the sweep,
                // once a second); asked again after it.
                ROOM_WANTED.fetch_max(short, Ordering::Relaxed);
                self.cache.grow_after = crate::host::now_ms() + 1_000;
                return false;
            }
            None if live >= BLOCKS_UNKNOWN => return false,
            None => {}
        }
        let number = match self.cache.spare.last() {
            Some(&n) => Some(n),
            None if self.copies.len() < COPIES as usize => None,
            None => return false,
        };
        let Some(buffer) = host.uav_buffer(BLOCK_BYTES) else {
            self.cache.grow_after = crate::host::now_ms() + 2_000;
            return false;
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = match number {
            Some(n) => {
                self.cache.spare.pop();
                self.copies[n as usize - 1] = Copy::Card(Some(buffer.clone()));
                self.rewrite_copy(n);
                n
            }
            None => {
                self.copies.push(Copy::Card(Some(buffer.clone())));
                self.unmade = true;
                self.copies.len() as u32
            }
        };
        let b = match self.cache.blocks.iter().position(Option::is_none) {
            Some(b) => b,
            None => {
                self.cache.blocks.push(None);
                self.cache.slots.extend(std::iter::repeat_n(Slot::FREE, PER_BLOCK as usize));
                self.cache.blocks.len() - 1
            }
        };
        self.cache.blocks[b] = Some(Block { copy, buffer, gpu });
        LIVE.store(true, Ordering::Relaxed);
        let first = b as u32 * PER_BLOCK;
        self.cache.free.extend((first..first + PER_BLOCK).rev());
        if live == 0 {
            tracing::info!(
                block_mb = BLOCK_BYTES >> 20,
                grow_until_percent = GROW,
                shrink_over_percent = SHRINK,
                "card page cache: pool pages lists name get a copy in the card's memory, in blocks made while the card has room"
            );
        }
        true
    }

    /// Writes the view of copy `n` again in every heap that has made it
    /// (a block made under a number given back, or the number let go).
    fn rewrite_copy(&self, n: u32) {
        for (&key, run) in &self.heaps {
            if run.made < n {
                continue;
            }
            // SAFETY: a heap named here is alive: it leaves the map under
            // this lock before it is freed (`heap_gone`).
            let heap = unsafe { &*(key as *const HostHeap) };
            let slot = run.base + n;
            match &self.copies[n as usize - 1] {
                Copy::Card(Some(buffer)) => heap.raw_uav_at(slot, buffer, 0, BLOCK_BYTES),
                _ => heap.null_raw_uav_at(slot),
            }
        }
    }

    /// Over [`SHRINK`]: the block whose newest naming is oldest is given
    /// back, its pages evicted (`submission`: the head list that names them
    /// in their sections again).
    fn shrink(&mut self, submission: u64) {
        if self.cache.blocks.iter().all(Option::is_none) {
            return;
        }
        if !matches!(room(), Some((_, true, _))) {
            return;
        }
        let per = PER_BLOCK as usize;
        let Some(b) = (0..self.cache.blocks.len())
            .filter(|&b| self.cache.blocks[b].is_some())
            .min_by_key(|&b| self.cache.slots[b * per..(b + 1) * per].iter().map(|s| s.used).max().unwrap_or(0))
        else {
            return;
        };
        for s in b * per..(b + 1) * per {
            let slot = self.cache.slots[s];
            if slot.home == TWIN {
                // The twin of a page in another block: that page has none now.
                self.cache.slots[slot.twin as usize].twin = NONE;
                self.cache.slots[s] = Slot::FREE;
                self.cache.twins -= 1;
            } else if slot.home != 0 {
                self.evict(s as u32);
            }
        }
        self.cache.free.retain(|&s| s as usize / per != b);
        if let Some(block) = self.cache.blocks[b].take() {
            self.copies[block.copy as usize - 1] = Copy::Card(None);
            // The heaps' views hold the buffer until no list can use it.
            self.cache.released.push((block.copy, submission));
            tracing::info!(
                blocks = self.cache.blocks.iter().filter(|b| b.is_some()).count(),
                "card page cache: a block given back, the card is short of room"
            );
        }
        LIVE.store(self.cache.blocks.iter().any(Option::is_some), Ordering::Relaxed);
    }

    /// While the cache has no block: makes one if the card has room (lists
    /// name pool memory for the cache from then on). Whether it has one.
    pub(super) fn cache_wake(&mut self) -> bool {
        if self.sections.is_empty() {
            return false;
        }
        self.cache.no_grow = false;
        self.grow();
        live()
    }

    /// Numbers of blocks given back whose last user has finished: their
    /// views go (the buffer with them), and the number may serve again.
    fn let_go_released(&mut self) {
        let finished = crate::host::submissions_finished();
        let done: Vec<u32> = self.cache.released.iter().filter(|r| r.1 <= finished).map(|r| r.0).collect();
        if done.is_empty() {
            return;
        }
        self.cache.released.retain(|r| r.1 > finished);
        for n in done {
            self.rewrite_copy(n);
            self.cache.spare.push(n);
        }
    }

    /// Announced writes since the last submission: cached pages are copied
    /// in again, pages the directory shows are wanted ahead.
    fn take_written(&mut self) {
        let queued = std::mem::take(&mut *QUEUED.lock().unwrap_or_else(|e| e.into_inner()));
        if OVERFLOW.swap(false, Ordering::Relaxed) {
            for s in 0..self.cache.slots.len() {
                let slot = &mut self.cache.slots[s];
                if slot.home != 0 && slot.home != TWIN && !slot.due {
                    slot.due = true;
                    self.cache.due.push(s as u32);
                }
            }
        }
        for (handle, offset, len) in queued {
            let Some(&n) = self.sections.get(&handle) else {
                continue;
            };
            let mut page = offset & !(PAGE - 1);
            while page < offset + len {
                let home = u64::from(n) | page << 32;
                match self.cache.by_home.get(&home) {
                    Some(&s) => {
                        let slot = &mut self.cache.slots[s as usize];
                        if !slot.due {
                            slot.due = true;
                            self.cache.due.push(s);
                        }
                    }
                    None if self.cache.aliases.contains_key(&home) && self.cache.wanted.len() < WANTED_MOST => {
                        self.cache.wanted.insert(home);
                    }
                    None => {}
                }
                page += PAGE;
            }
        }
    }

    /// At a submission: what the cache copies at its head and after each of
    /// its lists (`named`, one per list), its residency settled first
    /// (pages taken in, twins made, blocks made or given back, pages
    /// evicted; entries that change marked for the directory).
    pub(super) fn cache_plan(&mut self, named: &[Named], submission: u64) -> Plan {
        let mut plan = Plan::default();
        if self.sections.is_empty() {
            return plan;
        }
        self.cache.no_grow = false;
        // Frames from 2 on: a page taken in ahead (named at "0") may go at
        // once.
        let now = crate::frame::current() + 2;
        self.take_written();
        self.let_go_released();
        self.shrink(submission);
        // Pages shaders that pick buffers by index may write: held off.
        for n in named {
            for &(from, to) in &n.heap {
                for page in pages(from, to.saturating_sub(from)) {
                    let home = self.mirror[page as usize];
                    if self.section_of(home).is_none() {
                        continue;
                    }
                    self.cache.held.insert(home, now + HOLD);
                    if let Some(&s) = self.cache.by_home.get(&home) {
                        self.evict(s);
                    }
                }
            }
        }
        if submission.is_multiple_of(64) {
            self.cache.held.retain(|_, until| *until > now);
            NOW[3].store(self.cache.held.len() as u64, Ordering::Relaxed);
        }
        let held = |c: &Cache, home: u64| c.held.get(&home).is_some_and(|&until| until > now);
        // Pages named: copies of their own, the least recently named
        // evicted for them.
        let mut seen: FastSet<u64> = FastSet::default();
        // Lists of a submission name the same views over and over.
        let mut ranges: FastSet<(u64, u64)> = FastSet::default();
        let (mut fresh, mut served, mut all) = (NEW_MOST, 0u64, 0u64);
        for n in named {
            for &(address, len) in &n.reads {
                if !ranges.insert((address, len)) {
                    continue;
                }
                for page in pages(address, len) {
                    if !seen.insert(page) {
                        continue;
                    }
                    let home = self.mirror[page as usize];
                    if let Some(&s) = self.cache.by_home.get(&home) {
                        self.cache.slots[s as usize].used = now;
                        served += 1;
                        all += 1;
                        continue;
                    }
                    if self.section_of(home).is_none() || held(&self.cache, home) {
                        continue;
                    }
                    all += 1;
                    if fresh < PAGE {
                        continue;
                    }
                    if let Some(s) = self.take_slot(now, true) {
                        self.assign(s, home, now);
                        fresh -= PAGE;
                        served += 1;
                    }
                }
            }
        }
        // Pages announced written that the directory shows: ahead, into
        // room the cache has (before the twins: a page taken in ahead may be
        // one this submission's shaders write).
        let mut ahead = AHEAD_MOST;
        for home in std::mem::take(&mut self.cache.wanted) {
            if ahead < PAGE {
                break;
            }
            if self.cache.by_home.contains_key(&home) || self.section_of(home).is_none() || held(&self.cache, home) {
                continue;
            }
            match self.take_slot(now, false) {
                Some(s) => {
                    self.assign(s, home, 0);
                    ahead -= PAGE;
                }
                None => break,
            }
        }
        // Pool views of the heap that shaders picking by index may read:
        // into room the cache has, as if taken in ahead.
        let mut maybe = AHEAD_MOST;
        'maybe: for n in named {
            for &(address, len) in &n.maybe {
                for page in pages(address, len) {
                    let home = self.mirror[page as usize];
                    if maybe < PAGE {
                        break 'maybe;
                    }
                    if self.cache.by_home.contains_key(&home) || self.section_of(home).is_none() || held(&self.cache, home) {
                        continue;
                    }
                    match self.take_slot(now, false) {
                        Some(s) => {
                            self.assign(s, home, 0);
                            maybe -= PAGE;
                        }
                        None => break 'maybe,
                    }
                }
            }
        }
        // Pages shaders write by directory: a twin each (or no copy: their
        // writes then go to the pages).
        let mut written: Vec<u32> = Vec::new();
        let mut marked: FastSet<u32> = FastSet::default();
        ranges.clear();
        for n in named {
            for &(address, len) in &n.writes {
                if !ranges.insert((address, len)) {
                    continue;
                }
                for page in pages(address, len) {
                    if let Some(&s) = self.cache.by_home.get(&self.mirror[page as usize])
                        && marked.insert(s)
                    {
                        self.cache.slots[s as usize].used = now;
                        written.push(s);
                    }
                }
            }
        }
        let can_write_back = !written.is_empty() && pass().is_some();
        let mut new_twins = Vec::new();
        for s in written {
            let slot = self.cache.slots[s as usize];
            if slot.home == 0 || slot.home == TWIN || slot.twin != NONE {
                continue;
            }
            match can_write_back.then(|| self.take_slot(now, true)).flatten() {
                Some(t) => {
                    self.cache.slots[t as usize] = Slot { home: TWIN, used: now, twin: s, due: false };
                    self.cache.slots[s as usize].twin = t;
                    self.cache.twins += 1;
                    new_twins.push(s);
                }
                None => self.evict(s),
            }
        }
        // The head list's copies: pages in, then their twins.
        let mut seen_buffers = FastSet::default();
        let mut twinned: FastSet<u32> = FastSet::default();
        let mut uploaded = 0u64;
        for s in std::mem::take(&mut self.cache.due) {
            let slot = self.cache.slots[s as usize];
            if !slot.due || slot.home == 0 || slot.home == TWIN {
                continue;
            }
            self.cache.slots[s as usize].due = false;
            let (Some((src, src_at)), Some((dst, dst_at, _))) = (self.section_of(slot.home), self.cache.at(s)) else {
                self.evict(s);
                continue;
            };
            keep(&mut plan.kept, &mut seen_buffers, &src);
            keep(&mut plan.kept, &mut seen_buffers, &dst);
            plan.uploads.push(CopyOp { dst: dst.clone(), dst_at, src, src_at });
            uploaded += PAGE;
            if slot.twin != NONE
                && let Some((twin, twin_at, _)) = self.cache.at(slot.twin)
            {
                twinned.insert(s);
                plan.twins.push(CopyOp { dst: twin, dst_at: twin_at, src: dst, src_at: dst_at });
            }
        }
        for s in new_twins {
            let slot = self.cache.slots[s as usize];
            if slot.home == 0 || slot.home == TWIN || slot.twin == NONE || twinned.contains(&s) {
                continue;
            }
            if let (Some((dst, dst_at, _)), Some((twin, twin_at, _))) = (self.cache.at(s), self.cache.at(slot.twin)) {
                keep(&mut plan.kept, &mut seen_buffers, &twin);
                plan.twins.push(CopyOp { dst: twin, dst_at: twin_at, src: dst, src_at: dst_at });
            }
        }
        // After each list: what its shaders wrote into cached pages back to
        // the pages.
        let mut afters = Vec::with_capacity(named.len());
        let mut any = false;
        let mut compared = 0u64;
        for n in named {
            let mut after = After::default();
            let mut seen_buffers = FastSet::default();
            let mut done: FastSet<u32> = FastSet::default();
            for &(address, len) in &n.writes {
                for page in pages(address, len) {
                    let home = self.mirror[page as usize];
                    let Some(&s) = self.cache.by_home.get(&home) else {
                        continue;
                    };
                    let twin = self.cache.slots[s as usize].twin;
                    if twin == NONE || !done.insert(s) {
                        continue;
                    }
                    let (Some((card, _, card_gpu)), Some((twins, _, twin_gpu)), Some((section, offset))) =
                        (self.cache.at(s), self.cache.at(twin), self.section_of(home))
                    else {
                        continue;
                    };
                    // SAFETY: a live buffer.
                    let page_gpu = unsafe { section.GetGPUVirtualAddress() } + offset;
                    after.diffs.push([card_gpu, twin_gpu, page_gpu]);
                    keep(&mut after.kept, &mut seen_buffers, &card);
                    keep(&mut after.kept, &mut seen_buffers, &twins);
                    keep(&mut after.kept, &mut seen_buffers, &section);
                    compared += PAGE;
                }
            }
            if !n.other.is_empty() && !self.cache.by_home.is_empty() {
                let mut spans = Vec::new();
                for &(address, len) in &n.other {
                    crate::unified::spans(address, len, &mut spans);
                }
                let mut again: FastSet<u32> = FastSet::default();
                for (home, offset, len) in spans {
                    let Home::Section(handle) = home else {
                        continue;
                    };
                    let Some(&number) = self.sections.get(&handle) else {
                        continue;
                    };
                    let mut at = offset & !(PAGE - 1);
                    while at < offset + len {
                        let key = u64::from(number) | at << 32;
                        at += PAGE;
                        let Some(&s) = self.cache.by_home.get(&key) else {
                            continue;
                        };
                        if !again.insert(s) {
                            continue;
                        }
                        let (Some((src, src_at)), Some((dst, dst_at, _))) = (self.section_of(key), self.cache.at(s)) else {
                            continue;
                        };
                        keep(&mut after.kept, &mut seen_buffers, &src);
                        keep(&mut after.kept, &mut seen_buffers, &dst);
                        after.uploads.push(CopyOp { dst: dst.clone(), dst_at, src, src_at });
                        uploaded += PAGE;
                        let twin = self.cache.slots[s as usize].twin;
                        if twin != NONE
                            && let Some((t, t_at, _)) = self.cache.at(twin)
                        {
                            keep(&mut after.kept, &mut seen_buffers, &t);
                            after.twins.push(CopyOp { dst: t, dst_at: t_at, src: dst, src_at: dst_at });
                        }
                    }
                }
            }
            any |= !after.is_empty();
            afters.push(after);
        }
        if any {
            plan.after = afters;
        }
        SUMS[0].fetch_add(uploaded, Ordering::Relaxed);
        SUMS[1].fetch_add(compared, Ordering::Relaxed);
        SUMS[3].fetch_add(served, Ordering::Relaxed);
        SUMS[4].fetch_add(all, Ordering::Relaxed);
        NOW[0].store(self.cache.by_home.len() as u64, Ordering::Relaxed);
        NOW[1].store(self.cache.twins, Ordering::Relaxed);
        NOW[2].store(self.cache.blocks.iter().filter(|b| b.is_some()).count() as u64, Ordering::Relaxed);
        plan
    }
}
