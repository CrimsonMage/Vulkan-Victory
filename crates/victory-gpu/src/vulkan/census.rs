//! Diagnostics (`VICTORY_VK_WRITE_CENSUS=<frame>,<frames>`): which pages of
//! the title memory the card reads in place (the imported ranges of the
//! memory map) change from one present to the next, whoever wrote them.
//! Each 4 KiB page is hashed at the presents of the window; at its end
//! `write-census.txt` in the capture folder lists every page that changed
//! (`<range start> <page> <mask>`: bit k set when it changed between
//! presents k and k + 1), and a line a range is logged.
//!
//! **The touch probe** follows for as many presents again: the pages that
//! changed in the window and lie under an unordered-access view (what the
//! card may store into) are made inaccessible at each present. The CPU's
//! first read or write of such a page in a frame faults, is counted (by
//! kind, and whether the instruction is victory's own) and lets the page
//! go until the next present; the host's own guarded reads let pages go
//! beforehand ([`host_reads`]). A page that changed over a frame without a
//! CPU write is one the card stored into. `touch-census.txt` lists each
//! probed page (`<range start> <page> <read faults> <write faults> <of
//! them victory's> <frames changed> <frames changed with no CPU write>
//! <read faults at a byte under an unordered-access view of 1 MiB or more>
//! <write faults there>`), and a line a range is logged. Buffer memory
//! that holds an unordered-access view of 1 MiB or more is probed whole,
//! changed or not (`buffer-resources.txt`: `<first address> <bytes>
//! <its largest such view>`): what a card home for it would have to
//! follow. So is every other address the CPU sees such a buffer's pages
//! at (loose memory is views of physical pages, and the title maps them
//! twice): a line `<range start> <address> <read faults> <write faults>
//! <of them victory's> alias <the buffer's page> <read faults at a byte
//! under a large view> <write faults there>`. Faults between a page's hash and its
//! re-arming at a present are missed, and a page whose protection write
//! tracking changes meanwhile may be given back writable once: the frames
//! of the probe are not to be judged as pictures.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use super::{Backend, lock, memory};

const SMALL: usize = 4096;
const CHUNK: usize = 1 << 20;

fn window() -> Option<(u64, u64)> {
    static W: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *W.get_or_init(|| {
        let v = std::env::var("VICTORY_VK_WRITE_CENSUS").ok()?;
        let (a, n) = v.split_once(',')?;
        Some((a.trim().parse().ok()?, n.trim().parse::<u64>().ok()?.clamp(1, 31)))
    })
}

#[derive(Default)]
struct State {
    /// Range start -> (bytes, the pages' hashes at the last present, masks).
    ranges: BTreeMap<u64, (u64, Vec<u64>, Vec<u32>)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Buffer views the title made: (kind, address, bytes) -> how often. Kind
/// 0 a constant buffer, 1 a shader resource, 2 unordered access (those
/// from the start of the run; the others in the window only).
static VIEWS: Mutex<BTreeMap<(u8, u64, u64), u32>> = Mutex::new(BTreeMap::new());

/// The title made a buffer view of `[at, at + len)`.
pub(super) fn view(kind: u8, at: u64, len: u64) {
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let Some((first, frames)) = window() else { return };
    let frame = super::present::FRAME.load(std::sync::atomic::Ordering::Relaxed);
    if kind != 2 && (frame < first || frame > first + frames) {
        return;
    }
    let mut v = lock(&VIEWS);
    if v.len() < 1 << 20 {
        *v.entry((kind, at, len)).or_insert(0) += 1;
    }
}

/// Buffer memory the title made unordered-access views of: (first
/// address, bytes) of the resource (or of a placed view's memory) -> the
/// bytes of its largest such view.
static RESOURCES: Mutex<BTreeMap<(u64, u64), u64>> = Mutex::new(BTreeMap::new());

/// The title made an unordered-access view of `view` bytes in the buffer
/// memory `[base, base + len)`.
pub(super) fn resource(base: u64, len: u64, view: u64) {
    if cfg!(feature = "diagnostics") && window().is_some() {
        let mut r = lock(&RESOURCES);
        let e = r.entry((base, len)).or_insert(0);
        *e = (*e).max(view);
    }
}

/// Sorted spans joined where they touch.
fn merged(mut spans: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    spans.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (a, b) in spans {
        match out.last_mut() {
            Some(m) if a <= m.1 => m.1 = m.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// Whether one of the sorted, disjoint `spans` holds a byte of `[at, at + len)`.
fn within(spans: &[(u64, u64)], at: u64, len: u64) -> bool {
    let i = spans.partition_point(|m| m.0 < at + len);
    i > 0 && spans[i - 1].1 > at
}

pub(super) fn hash(page: &[u8]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for c in page.chunks_exact(8) {
        let w = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h = (h ^ w).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(23);
    }
    h
}

/// The hashes of the pages of `[at, at + len)`; a page that cannot be read
/// hashes as 0.
fn hashes(at: u64, len: u64) -> Vec<u64> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut out = Vec::with_capacity((len as usize) / SMALL);
    let mut buf = vec![0u8; CHUNK];
    let mut done = 0usize;
    while done < len as usize {
        let n = CHUNK.min(len as usize - done);
        let mut copied = 0;
        // SAFETY: the kernel checks the source; the destination is ours.
        let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), (at as usize + done) as *const _, buf.as_mut_ptr().cast(), n, &mut copied) } != 0;
        for p in buf[..n].chunks(SMALL) {
            out.push(if ok { hash(p) } else { 0 });
        }
        done += n;
    }
    out
}

/// At a present.
pub(super) fn frame(b: &Backend, frame: u64) {
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let Some((first, frames)) = window() else { return };
    if frame > first + frames && frame <= first + 2 * frames {
        probe_frame(frame == first + 2 * frames, frames);
    }
    if frame < first || frame > first + frames {
        return;
    }
    let t = std::time::Instant::now();
    let k = (frame - first) as u32;
    let loose = memory(b).loose_spans();
    let imported: Vec<(u64, u64)> = {
        let m = memory(b);
        let pools: Vec<(u64, u64)> = m.pools.iter().map(|p| (p.title_start, p.len)).collect();
        m.map.ranges().iter().map(|r| (r.title_start, r.title_size)).filter(|r| !pools.contains(r)).collect()
    };
    let mut guard = lock(&STATE);
    let st = guard.get_or_insert_with(State::default);
    let mut bytes = 0u64;
    for (at, len) in imported {
        bytes += len;
        let now = hashes(at, len);
        match st.ranges.get_mut(&at) {
            Some((l, last, masks)) if *l == len && k > 0 => {
                for (i, (a, c)) in last.iter().zip(&now).enumerate() {
                    if a != c {
                        masks[i] |= 1 << (k - 1);
                    }
                }
                *last = now;
            }
            _ => {
                let n = now.len();
                st.ranges.insert(at, (len, now, vec![0; n]));
            }
        }
    }
    tracing::info!(frame, mb = bytes >> 20, ms = t.elapsed().as_millis() as u64, recordings_open = b.timeline.open(), "Vulkan: write census, a present hashed");
    if frame < first + frames {
        return;
    }
    probe_start(st, &loose);
    let all = (1u32 << frames) - 1;
    let mut text = Vec::new();
    for (at, (len, _, masks)) in &st.ranges {
        let changed = masks.iter().filter(|m| **m != 0).count();
        let every = masks.iter().filter(|m| **m == all).count();
        let per: f64 = masks.iter().map(|m| f64::from(m.count_ones())).sum::<f64>() / frames as f64;
        tracing::info!(range = format_args!("{at:#x}+{len:#x}"), mb = len >> 20, pages_changed = changed, changed_every_frame = every, pages_a_frame = format_args!("{per:.0}"), "Vulkan: write census (4 KiB pages)");
        for (i, m) in masks.iter().enumerate().filter(|(_, m)| **m != 0) {
            let _ = writeln!(text, "{at:x} {:x} {m:x}", at + (i * SMALL) as u64);
        }
    }
    if let Some(folder) = crate::capture::folder() {
        let _ = std::fs::write(folder.join("write-census.txt"), text);
        write_views(&folder);
    }
    *guard = None;
}

/// `buffer-views.txt` and `buffer-resources.txt` into the capture folder.
fn write_views(folder: &std::path::Path) {
    let mut text = Vec::new();
    for ((kind, at, len), n) in lock(&VIEWS).iter() {
        let _ = writeln!(text, "{kind} {at:x} {len:x} {n}");
    }
    let _ = std::fs::write(folder.join("buffer-views.txt"), text);
    let mut text = Vec::new();
    for ((at, len), view) in lock(&RESOURCES).iter() {
        let _ = writeln!(text, "{at:x} {len:x} {view:x}");
    }
    let _ = std::fs::write(folder.join("buffer-resources.txt"), text);
}

// ---------------------------------------------------------------------------
// The touch probe

const ARMED: u32 = 1 << 31;
/// In a probed page's second field: the page is another address of the
/// buffer page in the low bits.
const ALIAS: u64 = 1 << 63;

struct Probe {
    /// (page, the start of its range), by page.
    pages: Vec<(u64, u64)>,
    /// [`ARMED`] and the protection to give back.
    state: Vec<AtomicU32>,
    reads: Vec<AtomicU32>,
    writes: Vec<AtomicU32>,
    /// Faults whose instruction lies in victory's image, and pages the
    /// host's guarded reads let go.
    ours: Vec<AtomicU32>,
    /// Of the first touches, those at a byte under an unordered-access
    /// view of 1 MiB or more: reads, writes.
    view_reads: Vec<AtomicU32>,
    view_writes: Vec<AtomicU32>,
    /// The bytes of those views, joined.
    large: Vec<(u64, u64)>,
    /// The CPU wrote the page since the last present.
    wrote: Vec<AtomicBool>,
    /// Presents at which the page had changed; of those, without a CPU
    /// write in between.
    changed: Vec<AtomicU32>,
    card: Vec<AtomicU32>,
    last: Mutex<Vec<u64>>,
    /// victory's own image.
    image: (usize, usize),
}

static PROBE: OnceLock<Probe> = OnceLock::new();

fn counters(n: usize) -> Vec<AtomicU32> {
    (0..n).map(|_| AtomicU32::new(0)).collect()
}

/// victory's image in memory: (start, end).
fn own_image() -> (usize, usize) {
    // SAFETY: the process's own module; its headers are mapped.
    unsafe {
        let base = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) as usize;
        if base == 0 {
            return (0, 0);
        }
        let pe = (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
        let size = (base as *const u8).add(pe + 0x50).cast::<u32>().read_unaligned() as usize;
        (base, base + size)
    }
}

fn hash_at(page: u64) -> Option<u64> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut buf = [0u8; SMALL];
    let mut copied = 0;
    // SAFETY: the kernel checks the source; the destination is ours.
    let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), page as *const _, buf.as_mut_ptr().cast(), SMALL, &mut copied) } != 0;
    ok.then(|| hash(&buf))
}

/// Protects `page` as `protection`; what it was.
fn protect(page: u64, protection: u32) -> Option<u32> {
    let mut old = 0;
    // SAFETY: a page of title memory; a failure changes nothing.
    (unsafe { windows_sys::Win32::System::Memory::VirtualProtect(page as *const _, SMALL, protection, &mut old) } != 0).then_some(old)
}

/// Arms page `i`: inaccessible, to be given back as it was.
fn arm(p: &Probe, i: usize) {
    use windows_sys::Win32::System::Memory::PAGE_NOACCESS;
    let before = p.state[i].load(Ordering::Relaxed) & !ARMED;
    p.state[i].store(ARMED | before, Ordering::SeqCst);
    match protect(p.pages[i].0, PAGE_NOACCESS) {
        // someone set another protection meanwhile: that one is given back
        Some(now) if now != PAGE_NOACCESS && now != before => {
            let _ = p.state[i].compare_exchange(ARMED | before, ARMED | (now & 0xfff), Ordering::SeqCst, Ordering::SeqCst);
        }
        Some(_) => {}
        None => p.state[i].store(before, Ordering::SeqCst),
    }
}

/// Lets page `i` go if it is armed; whether it was.
fn let_go(p: &Probe, i: usize) -> bool {
    let s = p.state[i].fetch_and(!ARMED, Ordering::SeqCst);
    if s & ARMED != 0 {
        protect(p.pages[i].0, s & 0xfff);
    }
    s & ARMED != 0
}

/// The census is over: the pages it saw change under unordered-access
/// views are probed from here.
fn probe_start(st: &State, loose: &[(u64, u64)]) {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    // the views' spans, joined; those of 1 MiB or more; and the buffer
    // memory that holds such a view, whole
    let merged = merged(lock(&VIEWS).keys().filter(|k| k.0 == 2).map(|k| (k.1, k.1 + k.2)).collect());
    let large = self::merged(lock(&VIEWS).keys().filter(|k| k.0 == 2 && k.2 >= 1 << 20).map(|k| (k.1, k.1 + k.2)).collect());
    let whole = self::merged(lock(&RESOURCES).iter().filter(|(_, view)| **view >= 1 << 20).map(|(k, _)| (k.0, k.0 + k.1)).collect());
    let mut pages: Vec<(u64, u64)> = Vec::new();
    for (at, (_, _, masks)) in &st.ranges {
        let probed = |i: usize, m: u32| {
            let page = at + (i * SMALL) as u64;
            (m != 0 && within(&merged, page, SMALL as u64)) || within(&whole, page, SMALL as u64)
        };
        pages.extend(masks.iter().enumerate().filter(|(i, m)| probed(*i, **m)).map(|(i, _)| (at + (i * SMALL) as u64, *at)));
    }
    // the other addresses the CPU sees those buffers' pages at, for the
    // buffers in loose memory that homes take (run hm1 probed every
    // buffer's: 456,000 pages, and the title crashed at once)
    let mut aliases = 0usize;
    for &(lo, hi) in whole.iter().filter(|w| loose.iter().any(|l| l.0 <= w.0 && w.1 <= l.1)) {
        for (page, others) in crate::track::elsewhere(lo, hi - lo) {
            for (other, k) in others.iter().flat_map(|&o| (0..crate::track::PAGE / SMALL).map(move |k| (o, k * SMALL))) {
                pages.push(((other + k) as u64, ALIAS | (page + k) as u64));
                aliases += 1;
            }
        }
    }
    pages.sort_unstable();
    pages.dedup_by_key(|p| p.0);
    let n = pages.len();
    let last: Vec<u64> = pages.iter().map(|p| hash_at(p.0).unwrap_or(0)).collect();
    let probe = Probe {
        pages,
        state: (0..n).map(|_| AtomicU32::new(PAGE_READWRITE)).collect(),
        reads: counters(n),
        writes: counters(n),
        ours: counters(n),
        view_reads: counters(n),
        view_writes: counters(n),
        large,
        wrote: (0..n).map(|_| AtomicBool::new(false)).collect(),
        changed: counters(n),
        card: counters(n),
        last: Mutex::new(last),
        image: own_image(),
    };
    if PROBE.set(probe).is_err() {
        return;
    }
    let p = PROBE.get().expect("set above");
    // SAFETY: a handler of the process's own, never removed.
    unsafe { AddVectoredExceptionHandler(1, Some(on_touch)) };
    for i in 0..n {
        arm(p, i);
    }
    tracing::info!(pages = n, mb = (n * SMALL) >> 20, views = merged.len(), other_addresses = aliases, "Vulkan: touch probe, the pages that changed under unordered-access views are inaccessible until touched");
}

/// At a present of the probe: each page's change is noted and the page
/// armed again (`end`: let go for good, and the result told).
fn probe_frame(end: bool, frames: u64) {
    let Some(p) = PROBE.get() else { return };
    let t = std::time::Instant::now();
    let mut last = lock(&p.last);
    for i in 0..p.pages.len() {
        let_go(p, i);
        if let Some(h) = hash_at(p.pages[i].0) {
            if h != last[i] {
                p.changed[i].fetch_add(1, Ordering::Relaxed);
                if !p.wrote[i].load(Ordering::Relaxed) {
                    p.card[i].fetch_add(1, Ordering::Relaxed);
                }
            }
            last[i] = h;
        }
        p.wrote[i].store(false, Ordering::Relaxed);
        if !end {
            arm(p, i);
        }
    }
    tracing::info!(ms = t.elapsed().as_millis() as u64, end, "Vulkan: touch probe, a present");
    if end {
        probe_tell(p, frames);
    }
}

fn probe_tell(p: &Probe, frames: u64) {
    #[derive(Default)]
    struct Sum {
        pages: u64,
        read: u64,
        written: u64,
        read_faults: u64,
        write_faults: u64,
        ours: u64,
        card: u64,
        card_read: u64,
        card_untouched: u64,
        card_frames: u64,
        /// Other addresses of its buffers' pages: pages, read faults,
        /// write faults, and of those at a byte under a large view.
        alias: [u64; 4],
    }
    let get = |v: &[AtomicU32], i: usize| u64::from(v[i].load(Ordering::Relaxed));
    let mut by: BTreeMap<u64, Sum> = BTreeMap::new();
    let mut text = Vec::new();
    for (i, &(page, range)) in p.pages.iter().enumerate() {
        let (r, w, o, c, k) = (get(&p.reads, i), get(&p.writes, i), get(&p.ours, i), get(&p.changed, i), get(&p.card, i));
        if range & ALIAS != 0 {
            // counted with the range of the buffer page it is another address of
            let of = range & !ALIAS;
            let range = p.pages.binary_search_by_key(&of, |q| q.0).map_or(0, |j| p.pages[j].1);
            let s = by.entry(range).or_default();
            let under = get(&p.view_writes, i);
            s.alias = [s.alias[0] + 1, s.alias[1] + r, s.alias[2] + w, s.alias[3] + under];
            let _ = writeln!(text, "{range:x} {page:x} {r} {w} {o} alias {of:x} {} {under}", get(&p.view_reads, i));
            continue;
        }
        let s = by.entry(range).or_default();
        s.pages += 1;
        s.read += u64::from(r != 0);
        s.written += u64::from(w != 0);
        s.read_faults += r;
        s.write_faults += w;
        s.ours += o;
        s.card += u64::from(k != 0);
        s.card_read += u64::from(k != 0 && r != 0);
        s.card_untouched += u64::from(k != 0 && r == 0 && w == 0);
        s.card_frames += k;
        let _ = writeln!(text, "{range:x} {page:x} {r} {w} {o} {c} {k} {} {}", get(&p.view_reads, i), get(&p.view_writes, i));
    }
    for (range, s) in &by {
        tracing::info!(
            range = format_args!("{range:#x}"),
            pages = s.pages,
            cpu_read_pages = s.read,
            cpu_written_pages = s.written,
            read_faults_a_frame = format_args!("{:.0}", s.read_faults as f64 / frames as f64),
            write_faults_a_frame = format_args!("{:.0}", s.write_faults as f64 / frames as f64),
            of_them_victorys = s.ours,
            card_stored_pages = s.card,
            card_stored_a_frame = format_args!("{:.0}", s.card_frames as f64 / frames as f64),
            card_stored_and_cpu_read = s.card_read,
            card_stored_cpu_never_touched = s.card_untouched,
            other_address_pages = s.alias[0],
            read_faults_there = s.alias[1],
            write_faults_there = s.alias[2],
            of_them_under_a_large_view = s.alias[3],
            "Vulkan: touch probe, a range"
        );
    }
    if let Some(folder) = crate::capture::folder() {
        let _ = std::fs::write(folder.join("touch-census.txt"), text);
    }
}

/// The host is about to read `[at, at + len)` of title memory through the
/// kernel (which fails on an inaccessible page instead of faulting): the
/// probed pages there are let go, counted as victory's reads.
pub(crate) fn host_reads(at: u64, len: u64) {
    let Some(p) = PROBE.get() else { return };
    let first = p.pages.partition_point(|q| q.0 + (SMALL as u64) <= at);
    for i in first..p.pages.len() {
        if p.pages[i].0 >= at + len {
            break;
        }
        if let_go(p, i) {
            p.reads[i].fetch_add(1, Ordering::Relaxed);
            p.ours[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Whether `page` takes a read (or a `write`) as it is protected now.
fn takes(page: u64, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS, PAGE_READWRITE, PAGE_WRITECOPY, VirtualQuery};
    // SAFETY: plain query of a page.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(page as *const _, &mut m, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return false;
    }
    let writable = m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY) != 0;
    m.Protect != 0 && m.Protect & PAGE_NOACCESS == 0 && (!write || writable)
}

/// The first touch of probed page `i` since the last present, by the
/// instruction at `ip`.
fn touched(p: &Probe, i: usize, write: bool, ip: usize, at: u64) {
    // another address of a buffer page: the byte it is in the buffer
    let at = if p.pages[i].1 & ALIAS != 0 { (p.pages[i].1 & !ALIAS) + at % SMALL as u64 } else { at };
    let under = within(&p.large, at, 1);
    if write {
        p.writes[i].fetch_add(1, Ordering::Relaxed);
        p.view_writes[i].fetch_add(u32::from(under), Ordering::Relaxed);
        p.wrote[i].store(true, Ordering::Relaxed);
    } else {
        p.reads[i].fetch_add(1, Ordering::Relaxed);
        p.view_reads[i].fetch_add(u32::from(under), Ordering::Relaxed);
    }
    if ip >= p.image.0 && ip < p.image.1 {
        p.ours[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// An access violation: if it is the first touch of a probed page since
/// the last present, it is counted and the page let go.
unsafe extern "system" fn on_touch(info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> i32 {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH};
    // SAFETY: the system passes the exception being dispatched.
    let r = unsafe { &*(*info).ExceptionRecord };
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let Some(p) = PROBE.get() else { return EXCEPTION_CONTINUE_SEARCH };
    let (write, at) = (r.ExceptionInformation[0] == 1, r.ExceptionInformation[1] as u64);
    let page = at & !(SMALL as u64 - 1);
    let Ok(i) = p.pages.binary_search_by_key(&page, |q| q.0) else { return EXCEPTION_CONTINUE_SEARCH };
    let s = p.state[i].fetch_and(!ARMED, Ordering::SeqCst);
    if s & ARMED != 0 {
        touched(p, i, write, r.ExceptionAddress as usize, at);
        protect(page, s & 0xfff);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    // Another thread let the page go a moment ago: the access is tried
    // again if the page now takes it; else the fault is someone else's.
    if takes(page, write) { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH }
}
