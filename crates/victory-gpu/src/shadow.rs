//! Title buffers on the host GPU (`docs/unified-memory.md`, `docs/kiss.md`
//! "Unified memory on split hardware"). A title buffer, or a window of
//! title memory outside buffers, is a host buffer placed over the title's
//! own bytes ([`crate::unified`]): shaders read what the title wrote and
//! write where the title reads, with nothing copied either way, nothing
//! watched and nothing compared. A title GPU address inside the buffer is
//! the same offset in the host buffer (plus [`Shadow::offset`]'s bias).
//!
//! Raw, structured and constant views over the title's page pool do not
//! come here: shaders reach them by directory ([`crate::memory`]). For
//! what the directory cannot express (index, vertex and indirect buffers,
//! card copies, typed views), a window over the title's scattered pages (a
//! 4 MiB window that is not one run of one section) is a buffer on the
//! card gathered from the pages' opened heaps by GPU copies, again only
//! for pages the title remaps or announces written ([`crate::gather`]);
//! what shaders write into it stays there.
//!
//! What cannot be placed so has one fallback, a buffer of its own in the
//! PC's memory that each submission naming it fills from title memory
//! (`HostList::defer_upload`) and that nothing ever copies back:
//! - a view whose bytes do not start on an element of its buffer ([`exact`]);
//! - memory `crate::unified` refuses to open (logged there, and here once
//!   per buffer of 64 KiB or more), and windows with pages that lie in no
//!   unit.
//!
//! Shader writes into such a buffer stay in it until the next fill and
//! never reach title memory ([`Shadow::gpu_writes`] says so, counted).
//! Every other kind of title memory has one home, the title's own pages.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};

/// A title buffer (or window of title memory) as the host GPU sees it.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    /// The CPU address of `buffer`'s byte for `start`: the title's own
    /// memory when in place, else our mapping of the buffer of its own.
    data: *mut u8,
    gpu: u64,
    /// Where the title memory starts and how long it is.
    start: u64,
    len: u64,
    /// When it was last asked for (`crate::host::moment`): a window outside
    /// buffers nothing asked for in a while, and nothing holds, is let go
    /// ([`let_go_idle`]).
    used: AtomicU64,
    /// A number no other one has had. Its address is no name: one let go
    /// and another made may share it.
    id: u64,
    /// `buffer` came from `Host::shared_buffer`: when it goes it is given
    /// to the buffer pool (`crate::memcap`) to be used again.
    pooled: bool,
    /// Placed over the title's memory itself (the normal case). Else a
    /// buffer of its own, filled at each submission that names it.
    in_place: bool,
    /// Bytes from `buffer`'s first byte to `start` (an in-place buffer
    /// starts on the 64 KiB line below the title memory); 0 for a buffer of
    /// its own.
    bias: u64,
    /// In place over a view of a section: what the title maps there may
    /// change ([`in_place_remapped`]), and then `stale` is set and the
    /// next ask places a new buffer.
    stale: AtomicBool,
    /// A window over scattered pages, gathered on the card from them
    /// ([`crate::gather`]).
    gather: Option<crate::gather::Gather>,
}

// SAFETY: D3D12 resources are free-threaded; `data` is title memory or a
// persistent mapping written only at submission.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.in_place {
            IN_PLACE[0].fetch_sub(1, Ordering::Relaxed);
            IN_PLACE[1].fetch_sub(self.len, Ordering::Relaxed);
        }
        if self.pooled {
            crate::memcap::give(crate::memcap::Kind::Shared, self.buffer.clone());
        }
        if let Some(gather) = &self.gather {
            crate::gather::forget(self.id, self.start, gather);
        }
    }
}

impl Shadow {
    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            used: AtomicU64::new(crate::host::moment()),
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            pooled: false,
            in_place: false,
            bias: 0,
            stale: AtomicBool::new(false),
            gather: None,
        }
    }

    /// A window over scattered pages `[start, start + len)`, gathered on
    /// the card from the pages' heaps ([`crate::gather`]), if every page of
    /// it the title has lies in a unit.
    fn gathered(start: u64, len: u64) -> Option<Arc<Shadow>> {
        let gather = crate::gather::Gather::new(start, len)?;
        let (buffer, pooled) = crate::gather::window_buffer(len)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut shadow = Shadow::new(buffer, std::ptr::null_mut(), gpu, start, len);
        shadow.pooled = pooled;
        shadow.gather = Some(gather);
        let shadow = Arc::new(shadow);
        crate::gather::register(&shadow);
        Some(shadow)
    }

    /// The window's pages and their state, for a window over scattered
    /// pages.
    pub(crate) fn gather(&self) -> Option<&crate::gather::Gather> {
        self.gather.as_ref()
    }

    /// A buffer of its own in the PC's memory for `[start, start + len)`
    /// (`size` bytes, at least `len`), filled at each submission that names
    /// it.
    fn own(start: u64, len: u64, size: u64) -> Option<Shadow> {
        let (buffer, data) = host()?.shared_buffer(size)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut shadow = Shadow::new(buffer, data, gpu, start, len);
        shadow.pooled = true;
        OWN[0].fetch_add(1, Ordering::Relaxed);
        OWN[1].fetch_add(len, Ordering::Relaxed);
        Some(shadow)
    }

    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory it holds: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the title memory it holds, if it
    /// lies there.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Kept for the callers of the time of copies: nothing is write-watched
    /// any more.
    pub(crate) fn unwatched(&self) -> bool {
        false
    }

    /// Its own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the host buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        self.bias + address - self.start
    }

    /// Whether the buffer is placed over the title's memory itself.
    pub(crate) fn in_place(&self) -> bool {
        self.in_place
    }

    /// `list` reads `len` bytes at `address`: the buffer is kept alive
    /// until the list is reset; a buffer of its own is filled from title
    /// memory when the list is submitted.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if list.first_use(Arc::as_ptr(self) as usize | 1) {
            list.keep(&self.buffer);
            let sum = if self.in_place {
                crate::frametime::Sum::InPlaceBytes
            } else if self.gather.is_some() {
                // Its pages the title changed are copied in on the card
                // before the list runs.
                list.name_window(self);
                crate::frametime::Sum::WindowBytes
            } else {
                crate::frametime::Sum::CopyBytes
            };
            crate::frametime::add(sum, self.len);
        }
        if self.in_place || self.gather.is_some() {
            return;
        }
        let address = address.clamp(self.start, self.start + self.len);
        let len = len.min(self.start + self.len - address);
        if len > 0 {
            // SAFETY: inside the buffer's mapping.
            let to = unsafe { self.data.add((address - self.start) as usize) };
            list.defer_upload(self.id, to, address as *const u8, len as usize);
        }
    }

    /// Shaders of `list` may write `[address, address + len)`. In place
    /// they write the title's memory itself: declared all the same, so that
    /// what is prepared on the CPU while the submission runs waits where it
    /// writes (`host.rs` `late_blocked`), and announced once the submission
    /// has finished (`crate::unified::settle`: textures over it detile
    /// again). Into a buffer of its own the writes stay there.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        self.refresh(list, address, len);
        if self.in_place {
            list.defer_gpu_write(address as usize, len as usize);
            return;
        }
        let n = LOST.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 8 || n.is_power_of_two() {
            tracing::warn!(
                at = format_args!("{address:#x}"),
                len,
                start = format_args!("{:#x}", self.start),
                n,
                "shaders write a buffer that is not in place: the writes stay on the host and never reach title memory"
            );
        }
    }
}

/// Buffers of their own made since the start, and their bytes.
static OWN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Lists whose shaders write a buffer of its own, since the start.
static LOST: AtomicU64 = AtomicU64::new(0);

/// Windows outside buffers let go since the start, and their bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Lets go of the windows outside buffers that nothing names any more (the
/// world's streaming keeps asking for new ones). A window goes when only
/// the table of windows holds it (no list not yet reset, no pass over a
/// heap), only it holds its host buffer (no descriptor slot views it), and
/// nothing asked for it for a while. Called before a submission starts.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands one out under
        // the same lock, so none is handed out in between.
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
        crate::host::trail(|| format!("sweep: {went} windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10));
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        crate::host::retire(shadow);
    }
}

/// For the unified line: buffers of their own made since the start and
/// their bytes, lists whose shaders wrote one, windows let go.
pub(crate) fn own_made() -> [u64; 4] {
    [
        OWN[0].load(Ordering::Relaxed),
        OWN[1].load(Ordering::Relaxed),
        LOST.load(Ordering::Relaxed),
        LET_GO[0].load(Ordering::Relaxed),
    ]
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
/// `crate::unified` can open it.
fn in_place(start: u64, len: u64) -> Option<Arc<Shadow>> {
    let placed = crate::unified::place(start, len)?;
    let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
    shadow.in_place = true;
    shadow.bias = placed.bias;
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
    // Windows over these addresses look their pages up again, and whether a
    // window opens in place whole is asked again.
    crate::gather::remapped(at, end - at);
    {
        let mut split = SPLIT.lock().unwrap_or_else(|e| e.into_inner());
        if !split.is_empty() {
            // (A window of two cells starts a cell lower.)
            let gone: Vec<u64> = split.range((at & !(WINDOW - 1)).saturating_sub(WINDOW)..end).copied().collect();
            for start in gone {
                split.remove(&start);
            }
        }
    }
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

/// Memory `crate::unified` would not open: a buffer of its own, said once
/// per address for 64 KiB or more.
fn not_in_place(start: u64, len: u64, size: u64, what: &str) -> Option<Arc<Shadow>> {
    if len >= 64 << 10 {
        tracing::info!(
            start = format_args!("{start:#x}"),
            len,
            what,
            "title memory did not open in place: a buffer of its own, filled at each submission that names it, never copied back"
        );
    }
    Shadow::own(start, len, size).map(Arc::new)
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their host buffer once
/// made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let mut map = buffers();
    map.insert((start, resource), Entry { end: start + len, shadow: None });
    BUFFERS_CHANGED.fetch_add(1, Ordering::Release);
}

/// The title released it; lists that still hold its host buffer keep it
/// alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let mut map = buffers();
    map.remove(&(start, resource));
    BUFFERS_CHANGED.fetch_add(1, Ordering::Release);
}

/// Changes of [`BUFFERS`]' keys (buffers registered and released), made
/// under its lock: what each thread's recent answers ([`Hit`]) hold for.
static BUFFERS_CHANGED: AtomicU64 = AtomicU64::new(1);

/// One answer of [`find`] a thread had lately, and the asks it holds for:
/// an ask `[a, b)` gets the same buffer while `from <= a < until` (the walk
/// down from `a` then meets the same keys in the same order) and `above < b
/// <= end` (the keys met before this buffer end below `b`, and this one
/// holds it), as long as no key changed. Kept weakly.
struct Hit {
    changed: u64,
    from: u64,
    until: u64,
    above: u64,
    end: u64,
    shadow: std::sync::Weak<Shadow>,
}

/// Answers a thread keeps, replaced in turn.
const RECENT: usize = 16;

#[derive(Default)]
struct Recent {
    hits: [Option<Hit>; RECENT],
    next: usize,
}

thread_local! {
    /// The title's recording threads ask for the same few buffers draw
    /// after draw: the lock and the walk for each ask were a third of their
    /// time in the profile of 2026-11-25.
    static RECENT_HITS: std::cell::RefCell<Recent> = std::cell::RefCell::default();
}

/// [`find`] answered from the thread's recent answers, if one holds the
/// ask, no key changed since, and its buffer is alive and not stale.
fn recent(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let changed = BUFFERS_CHANGED.load(Ordering::Acquire);
    RECENT_HITS.with(|recent| {
        let recent = recent.try_borrow().ok()?;
        recent.hits.iter().flatten().find_map(|h| {
            if h.changed != changed || address < h.from || address >= h.until || end <= h.above || end > h.end {
                return None;
            }
            let shadow = h.shadow.upgrade().filter(|s| !s.stale.load(Ordering::Relaxed))?;
            // In use (a window outside buffers is let go when idle,
            // [`let_go_idle`]).
            shadow.used.store(crate::host::moment(), Ordering::Relaxed);
            Some(shadow)
        })
    })
}

/// Keeps `hit` among the thread's recent answers.
fn remember(hit: Hit) {
    RECENT_HITS.with(|recent| {
        let Ok(mut recent) = recent.try_borrow_mut() else {
            return;
        };
        if recent.hits.iter().flatten().any(|h| h.changed != hit.changed) {
            // Answers from before a change hold for nothing now.
            *recent = Recent::default();
        }
        let at = recent.next;
        recent.hits[at] = Some(hit);
        recent.next = (at + 1) % RECENT;
    });
}

/// The host buffer of the innermost title buffer holding `[address,
/// address + len)`, made on first use. A `len` of 0 or 1 is an ask without
/// a length (a root view: the caller reads on to the buffer's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    if let Some(shadow) = recent(address, end) {
        return Some(shadow);
    }
    let mut map = buffers();
    // The walk down from the ask's address, noting the first key met and
    // the highest end of those passed over ([`Hit`]).
    let (mut first, mut above, mut key) = (None, 0u64, None);
    for (k, e) in map.range(..=(address, usize::MAX)).rev().take(64) {
        first.get_or_insert(k.0);
        if e.end >= end {
            key = Some(*k);
            break;
        }
        above = above.max(e.end);
    }
    let Some(key) = key else {
        // Memory outside buffers: the window's answer holds for the asks
        // inside it that the same walk would find no buffer for ([`Hit`]).
        let changed = BUFFERS_CHANGED.load(Ordering::Relaxed);
        let until = map.range((address, usize::MAX)..).find(|(k, _)| k.0 > address).map_or(u64::MAX, |(k, _)| k.0);
        drop(map);
        let found = loose(address, end)?;
        let (start, len) = found.span();
        remember(Hit {
            changed,
            from: first.unwrap_or(0).max(start),
            until: until.min(start + len),
            above,
            end: start + len,
            shadow: Arc::downgrade(&found),
        });
        return Some(found);
    };
    let changed = BUFFERS_CHANGED.load(Ordering::Relaxed);
    let until = map.range((address, usize::MAX)..).find(|(k, _)| k.0 > address).map_or(u64::MAX, |(k, _)| k.0);
    let from = first.unwrap_or(key.0);
    let entry = map.get_mut(&key)?;
    if entry.shadow.as_ref().is_some_and(|s| s.stale.load(Ordering::Relaxed)) {
        entry.shadow = None;
    }
    let end = entry.end;
    let found = match entry.shadow.clone() {
        Some(found) => {
            drop(map);
            found
        }
        None => {
            // Made without the lock: opening memory may wait for the card
            // (its check), and the other recording threads look up their
            // buffers meanwhile.
            drop(map);
            let len = end - key.0;
            let made = match in_place(key.0, len) {
                Some(shadow) => shadow,
                // Rounded up so that views ending in the buffer's last 256
                // bytes fit.
                None => not_in_place(key.0, len, len.next_multiple_of(1 << 16), "buffer")?,
            };
            let mut map = buffers();
            match map.get_mut(&key) {
                // Another thread made one first.
                Some(entry) if entry.shadow.as_ref().is_some_and(|s| !s.stale.load(Ordering::Relaxed)) => entry.shadow.clone()?,
                Some(entry) => {
                    entry.shadow = Some(made.clone());
                    made
                }
                // Released meanwhile: the lists that name it hold it.
                None => return Some(made),
            }
        }
    };
    remember(Hit { changed, from, until, above, end, shadow: Arc::downgrade(&found) });
    Some(found)
}

/// Titles also use memory outside any buffer object as GPU memory (the
/// title's interface vertices, from its own allocator). Such memory is
/// placed by the aligned 4 MiB windows it touches (or the run of the view
/// the ask lies in). A window nothing names any more is let go
/// ([`let_go_idle`]).
const WINDOW: u64 = 4 << 20;

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)`.
fn window_of(address: u64, end: u64) -> (u64, u64) {
    let start = address & !(WINDOW - 1);
    (start, end.next_multiple_of(WINDOW) - start)
}

/// Windows of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let (start, len) = window_of(address, end);
    let take = |map: &mut BTreeMap<(u64, u64), Arc<Shadow>>, key: (u64, u64)| -> Option<Arc<Shadow>> {
        if map.get(&key).is_some_and(|s| s.stale.load(Ordering::Relaxed)) {
            map.remove(&key);
        }
        let shadow = map.get(&key)?;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        Some(shadow.clone())
    };
    // The whole window: in place, or over its scattered pages, made before
    // (it holds every ask in it, also a root view's, whose length is not
    // known).
    if let Some(shadow) = take(&mut map, (start, len)) {
        return Some(shadow);
    }
    // The GPU view of the pool has no host view: never one run, never in
    // place (`crate::memory`).
    if crate::memory::gpu_view(start, len) {
        let shadow = Shadow::gathered(start, len)?;
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // Asked once per window until the title maps there anew (it was asked
    // at every ask: a walk of the views and, every two seconds, of the
    // pages' protections, all under the units' lock, 34,553 refusals in run
    // p13).
    if !SPLIT.lock().unwrap_or_else(|e| e.into_inner()).contains(&start) {
        if let Some(shadow) = in_place(start, len) {
            map.insert((start, len), shadow.clone());
            return Some(shadow);
        }
        SPLIT.lock().unwrap_or_else(|e| e.into_inner()).insert(start);
    }
    // In a section's views the 4 MiB window often is not one run (the
    // views are smaller, or end, or show pages of another section): then
    // the run of the view the ask lies in, from the ask's 64 KiB page to
    // where the run or the window ends, if it holds the ask: one made
    // before, else a new one.
    let from = address & !0xffff;
    let held = map
        .range((start, 0)..=(from, u64::MAX))
        .rev()
        .take(64)
        .find(|((at, bytes), s)| s.in_place && at + bytes >= end && !s.stale.load(Ordering::Relaxed))
        .map(|(_, s)| s.clone());
    if let Some(shadow) = held {
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow);
    }
    let reach = crate::views::run_reach(from as usize, (start + len - from) as usize) as u64;
    if reach >= end - from {
        if let Some(shadow) = take(&mut map, (from, reach)) {
            return Some(shadow);
        }
        if let Some(shadow) = in_place(from, reach) {
            map.insert((from, reach), shadow.clone());
            return Some(shadow);
        }
    }
    // The window over its scattered pages, gathered on the card from them
    // (`crate::gather`); a buffer of its own if some page lies in no unit.
    let shadow = match Shadow::gathered(start, len) {
        Some(shadow) => shadow,
        None => not_in_place(start, len, len, "window outside buffers")?,
    };
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Windows (by start) known not to open in place whole ([`loose`]), until
/// the title maps there anew ([`in_place_remapped`]).
static SPLIT: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());

/// Buffers of their own for single buffer views' bytes ([`exact`]), by
/// (start, bytes), each with the asking that last took it; and the askings
/// so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// A buffer of its own for the title memory `[address, address + len)`
/// that starts at `address` itself: for a buffer view whose bytes do not
/// start on an element of the host buffer its buffer has
/// (`descriptor/host.rs` `own_bytes`). A host view counts its first
/// element from its buffer's start, a placed buffer starts on a 64 KiB
/// line, and a console view starts at any address (the title places views
/// at ring positions inside its big GPU heap: its meshlet culling
/// constants and colour-bake parameter blocks, upstream VIctory 86a5ed6).
///
/// Filled at each submission that names it; never copied back. Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Kept before the idle ones are looked for, and the askings an idle
    /// one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
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
        // Held by the map alone (no list), its buffer by it alone (no slot
        // shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now);
    }
    let shadow = Arc::new(Shadow::own(address, len, len.next_multiple_of(256))?);
    map.insert((address, len), (shadow.clone(), now));
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
