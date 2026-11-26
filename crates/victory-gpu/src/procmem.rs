//! Where the process's memory is: one line every ten seconds, made on a
//! thread of its own (nothing on the title's or the frame's threads).
//!
//! Three sources, joined per mapping:
//!
//! - **Linux's view** (`/proc/<pid>/smaps` through Wine's `Z:`): resident
//!   bytes, unique bytes (`Pss`: a page mapped at two addresses counts
//!   once; Linux's resident count, the one `top` shows, counts it at each),
//!   swapped bytes, per mapping, and what backs it (anonymous, a section, a
//!   file, the NVIDIA driver). The file is opened by wineserver, so
//!   `/proc/self` would be wineserver's: our pid is the process that shows
//!   a page of ours, read-only between two reserved ones, as a mapping of
//!   its own ([`unix_pid`]).
//! - **Windows' view** (`VirtualQuery` over the whole address space, one
//!   call a region): which mappings are Windows memory (committed: private,
//!   a section's view, a DLL's image) and which only Linux knows (the
//!   driver's and Wine's Unix half). Title memory lies below 1 TiB (its
//!   regions and its pool's section views); everything of victory,
//!   vkd3d-proton and Wine above.
//! - **Our own counters**: victory's heap by owner (the counting allocator
//!   of the diagnostics build, `victory_core::heap`), staging and pooled
//!   buffers (driver memory), title memory opened as heaps.
//!
//! Costs: the walk is a few thousand `VirtualQuery` calls; reading smaps
//! makes the kernel walk the page tables (about 4 million entries for 16
//! GB, a mapping at a time). The line says how long both took.

// The reading of smaps is tested on any host; the rest is Windows'.
#![cfg_attr(not(windows), allow(dead_code))]

use std::io::BufRead;

const TIB: u64 = 1 << 40;
// `MEMORY_BASIC_INFORMATION` states and types.
const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_FREE: u32 = 0x1_0000;
const MEM_MAPPED: u32 = 0x4_0000;
const MEM_IMAGE: u32 = 0x100_0000;

/// Starts the line's thread, once.
#[cfg(windows)]
pub(crate) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-procmem".into()).spawn(|| {
            let pid = unix_pid();
            if pid.is_none() {
                tracing::info!(
                    "process memory: this process is not found under Z:\\proc (not Wine?); the line has Windows' figures only"
                );
            }
            loop {
                std::thread::sleep(std::time::Duration::from_secs(10));
                tracing::info!("{}", line(pid));
            }
        });
    });
}

/// This process's Linux pid: the one whose maps show the middle page of
/// three we reserve, committed read-only, as a mapping of its own (its
/// neighbours are reserved, no access, so Linux keeps it apart). The
/// executable's own mapping does not tell: Wine reads images whose sections
/// are not page-aligned into anonymous memory, so no mapping is named after
/// it (run p8 found no process that way).
#[cfg(windows)]
fn unix_pid() -> Option<u32> {
    use std::ffi::c_void;
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, VirtualAlloc};
    // SAFETY: a reservation of ours, kept for the run, and one page of it
    // committed.
    let at = unsafe { VirtualAlloc(std::ptr::null(), 3 * 4096, MEM_RESERVE, PAGE_NOACCESS) } as u64;
    if at == 0 || unsafe { VirtualAlloc((at + 4096) as *const c_void, 4096, MEM_COMMIT, PAGE_READONLY) }.is_null() {
        return None;
    }
    let want = format!("{:08x}-{:08x} r-", at + 4096, at + 8192);
    for entry in std::fs::read_dir(r"Z:\proc").ok()?.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else { continue };
        let Ok(file) = std::fs::File::open(format!(r"Z:\proc\{pid}\maps")) else { continue };
        if std::io::BufReader::new(file).lines().map_while(Result::ok).any(|l| l.starts_with(&want)) {
            return Some(pid);
        }
    }
    None
}

/// A Windows region that is not free: its addresses, state and type.
#[derive(Clone, Copy)]
struct Region {
    start: u64,
    end: u64,
    state: u32,
    kind: u32,
}

/// Every region of the address space that is not free, in order.
#[cfg(windows)]
fn walk() -> Vec<Region> {
    use std::ffi::c_void;
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut out = Vec::new();
    let mut at: u64 = 0x1_0000;
    // The user address space of a 64-bit process.
    while at < 0x7FFF_FFFF_0000 {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query about our own address space into a local.
        let n = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        if n == 0 {
            break;
        }
        let start = info.BaseAddress as u64;
        let end = start.saturating_add(info.RegionSize as u64);
        if end <= at {
            break;
        }
        if info.State != MEM_FREE {
            out.push(Region { start, end, state: info.State, kind: info.Type });
        }
        at = end;
    }
    out
}

/// What a mapping is, for the line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Title addresses, anonymous: the title's regions.
    TitleRegions,
    /// Title addresses, shared: views of the pool's sections.
    TitleViews,
    /// Above, Windows views of sections (victory's whole views of the
    /// pool's sections, for its own writes).
    HostViews,
    /// Above, Windows private memory: the heaps of victory, vkd3d-proton
    /// and Wine's DLLs, and their own big allocations.
    Private,
    /// DLL and executable images.
    Images,
    /// Windows views of files (pipeline library, shader cache).
    Files,
    /// The NVIDIA driver's mappings (upload, read-back and shared buffers
    /// the card maps; the driver's own).
    Driver,
    /// Memory only Linux knows: the driver's, Vulkan's and Wine's Unix
    /// half's heaps.
    UnixAnon,
    /// Unix libraries and other files only Linux maps.
    UnixFiles,
}

const KINDS: usize = 9;

/// Resident, unique and swapped kB per kind.
#[derive(Default)]
struct Sums {
    rss: [u64; KINDS],
    pss: [u64; KINDS],
    swap: [u64; KINDS],
}

/// A mapping being read from smaps.
#[derive(Default)]
struct Vma {
    start: u64,
    path: String,
    rss: u64,
    pss: u64,
    /// Anonymous kB: private pages of ours, mapped once (smaps has no
    /// per-mapping split of `Pss`; only the rollup has).
    anon: u64,
    swap: u64,
}

fn file_path(path: &str) -> bool {
    path.starts_with('/') && !path.starts_with("/memfd:") && !path.starts_with("/dev/") && !path.ends_with("(deleted)")
}

fn classify(vma: &Vma, regions: &[Region]) -> Kind {
    let path = vma.path.as_str();
    if path.starts_with("/dev/nvidia") || path.starts_with("/dev/dri") {
        return Kind::Driver;
    }
    let at = regions.partition_point(|r| r.start <= vma.start);
    let region = at.checked_sub(1).map(|i| regions[i]).filter(|r| vma.start < r.end);
    match region {
        Some(r) if r.state == MEM_COMMIT => match r.kind {
            MEM_IMAGE => Kind::Images,
            MEM_MAPPED if file_path(path) => Kind::Files,
            MEM_MAPPED => Kind::HostViews,
            _ => Kind::Private,
        },
        _ if file_path(path) => Kind::UnixFiles,
        _ => Kind::UnixAnon,
    }
}

/// Adds a mapping read from smaps to the sums (and the files).
fn finish(vma: Option<Vma>, regions: &[Region], sums: &mut Sums, files: &mut Vec<(u64, String)>) {
    let Some(v) = vma else { return };
    if v.rss == 0 && v.swap == 0 {
        return;
    }
    if v.start < TIB && !v.path.starts_with("/dev/") {
        let (a, s) = (Kind::TitleRegions as usize, Kind::TitleViews as usize);
        sums.rss[a] += v.anon;
        sums.pss[a] += v.anon;
        sums.rss[s] += v.rss.saturating_sub(v.anon);
        sums.pss[s] += v.pss.saturating_sub(v.anon);
        sums.swap[a] += v.swap;
        return;
    }
    let k = classify(&v, regions);
    sums.rss[k as usize] += v.rss;
    sums.pss[k as usize] += v.pss;
    sums.swap[k as usize] += v.swap;
    if k == Kind::Files && v.rss >= 1024 {
        match files.iter_mut().find(|(_, p)| *p == v.path) {
            Some(f) => f.0 += v.rss,
            None => files.push((v.rss, v.path)),
        }
    }
}

/// Reads smaps into per-kind sums: mappings, the largest files, and how
/// many mappings. `None` if the file cannot be read.
#[cfg(windows)]
fn smaps(pid: u32, regions: &[Region]) -> Option<(Sums, Vec<(u64, String)>, u64)> {
    let file = std::fs::File::open(format!(r"Z:\proc\{pid}\smaps")).ok()?;
    Some(read_smaps(std::io::BufReader::with_capacity(1 << 20, file), regions))
}

/// Sums the mappings of smaps text: per kind, the largest files, and how
/// many mappings.
fn read_smaps(mut reader: impl BufRead, regions: &[Region]) -> (Sums, Vec<(u64, String)>, u64) {
    let mut sums = Sums::default();
    let mut files: Vec<(u64, String)> = Vec::new();
    let mut maps = 0u64;
    let mut cur: Option<Vma> = None;
    let mut text = String::new();
    loop {
        text.clear();
        match reader.read_line(&mut text) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let mut words = text.split_whitespace();
        let Some(first) = words.next() else { continue };
        if !first.ends_with(':') {
            // A mapping's header: `start-end perms offset dev inode path`.
            let Some((a, _)) = first.split_once('-') else { continue };
            let Ok(start) = u64::from_str_radix(a, 16) else { continue };
            finish(cur.take(), regions, &mut sums, &mut files);
            maps += 1;
            let path = words.skip(4).collect::<Vec<_>>().join(" ");
            cur = Some(Vma { start, path, ..Vma::default() });
            continue;
        }
        let Some(v) = cur.as_mut() else { continue };
        let mut kb = || words.next().and_then(|w| w.parse::<u64>().ok()).unwrap_or(0);
        match first {
            "Rss:" => v.rss = kb(),
            "Pss:" => v.pss = kb(),
            "Anonymous:" => v.anon = kb(),
            "Swap:" => v.swap = kb(),
            _ => {}
        }
    }
    finish(cur.take(), regions, &mut sums, &mut files);
    files.sort_unstable_by_key(|f| std::cmp::Reverse(f.0));
    (sums, files, maps)
}

/// The system's own count of our working set (under Wine, Linux's resident
/// count): bytes.
#[cfg(windows)]
fn working_set() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: a query about our own process into a local of the size given.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return 0;
    }
    c.WorkingSetSize as u64
}

/// The line.
#[cfg(windows)]
fn line(pid: Option<u32>) -> String {
    let began = std::time::Instant::now();
    let regions = walk();
    // Windows' committed and reserved bytes: title regions, title views,
    // title reserved only; above, private, views of sections or files,
    // images.
    let mut c = [0u64; 6];
    for r in &regions {
        let bytes = r.end - r.start;
        let i = match (r.start < TIB, r.state, r.kind) {
            (true, MEM_COMMIT, MEM_MAPPED) => 1,
            (true, MEM_COMMIT, _) => 0,
            (true, _, _) => 2,
            (false, MEM_COMMIT, MEM_MAPPED) => 4,
            (false, MEM_COMMIT, MEM_IMAGE) => 5,
            (false, MEM_COMMIT, _) => 3,
            _ => continue,
        };
        c[i] += bytes;
    }
    let walked = began.elapsed();
    let linux = pid.and_then(|pid| smaps(pid, &regions));
    let took = began.elapsed();
    let mb = |b: u64| b >> 20;
    let kmb = |kb: u64| kb >> 10;
    let k = |kind: Kind| kind as usize;

    let heap = match victory_core::heap::live() {
        Some(live) => {
            let total: u64 = live.iter().sum();
            let parts: Vec<String> = victory_core::heap::OWNERS
                .iter()
                .map(|o| format!("{} {}", o.name(), mb(live[*o as usize])))
                .chain(std::iter::once(format!("small blocks {}", mb(live[victory_core::heap::OWNERS.len()]))))
                .collect();
            Some((total, parts.join(", ")))
        }
        None => None,
    };
    let (staged, spare) = crate::texture::staging_bytes();
    let pooled = crate::memcap::pooled();

    let mut out = String::new();
    use std::fmt::Write;
    match &linux {
        Some((s, _, maps)) => {
            let rss: u64 = s.rss.iter().sum();
            let pss: u64 = s.pss.iter().sum();
            let swap: u64 = s.swap.iter().sum();
            let _ = write!(
                out,
                "process memory: resident {} MB by Linux's count (a page mapped at two addresses counted twice; the system says {} MB), \
                 unique {} MB, swapped {} MB; {} mappings, read in {} ms (walk {} ms)",
                kmb(rss),
                mb(working_set()),
                kmb(pss),
                kmb(swap),
                maps,
                took.as_millis(),
                walked.as_millis()
            );
        }
        None => {
            let _ = write!(
                out,
                "process memory: working set {} MB (the system's count); walk {} ms",
                mb(working_set()),
                walked.as_millis()
            );
        }
    }
    let _ = write!(
        out,
        " | title memory (below 1 TiB): regions committed {} MB, pool section views {} MB (each view counted), reserved only {} MB; \
         opened as heaps {} MB",
        mb(c[0]),
        mb(c[1]),
        mb(c[2]),
        mb(crate::unified::opened_bytes())
    );
    if let Some((s, _, _)) = &linux {
        let _ = write!(
            out,
            "; resident: regions {} MB, section views {} MB by Linux's count, the sections {} MB unique with victory's own views of them \
             ({} MB resident there); swapped {} MB",
            kmb(s.rss[k(Kind::TitleRegions)]),
            kmb(s.rss[k(Kind::TitleViews)]),
            kmb(s.pss[k(Kind::TitleViews)] + s.pss[k(Kind::HostViews)]),
            kmb(s.rss[k(Kind::HostViews)]),
            kmb(s.swap[k(Kind::TitleRegions)])
        );
    }
    let private_rss = linux.as_ref().map(|(s, _, _)| s.rss[k(Kind::Private)] << 10);
    let _ = write!(
        out,
        " | Windows private memory above (victory, vkd3d-proton, Wine's DLLs): committed {} MB",
        mb(c[3])
    );
    if let Some(r) = private_rss {
        let _ = write!(out, ", resident {} MB", mb(r));
    }
    match &heap {
        Some((total, parts)) => {
            let _ = write!(
                out,
                "; victory's heap live {} MB ({parts}); the rest about {} MB (vkd3d-proton, Wine, heap slack)",
                mb(*total),
                mb(private_rss.unwrap_or(c[3]).saturating_sub(*total))
            );
        }
        None => out.push_str("; victory's heap not counted (diagnostics build only)"),
    }
    let _ = write!(out, " | images committed {} MB, section views above {} MB", mb(c[5]), mb(c[4]));
    if let Some((s, files, _)) = &linux {
        let top: Vec<String> =
            files.iter().take(3).map(|(kb, p)| format!("{} {} MB", p.rsplit('/').next().unwrap_or(p), kmb(*kb))).collect();
        let _ = write!(
            out,
            "; resident: images {} MB, files mapped {} MB ({}) | NVIDIA driver mappings {} MB | Unix side only (driver, Vulkan, \
             Wine's Unix half): anonymous {} MB, libraries and files {} MB",
            kmb(s.rss[k(Kind::Images)]),
            kmb(s.rss[k(Kind::Files)]),
            if top.is_empty() { "none".to_string() } else { top.join(", ") },
            kmb(s.rss[k(Kind::Driver)]),
            kmb(s.rss[k(Kind::UnixAnon)]),
            kmb(s.rss[k(Kind::UnixFiles)])
        );
    }
    let _ = write!(
        out,
        " | of ours in driver memory: staging {} MB, spare staging {} MB, buffers pooled {} MB; host lists kept idle {}, lists \
         of ours for copies spare {} (their command memory is the driver's)",
        mb(staged),
        mb(spare),
        mb(pooled),
        crate::list::idle_lists(),
        crate::memory::spare_lists()
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mappings_are_told_apart() {
        let regions = [
            Region { start: 0x7f00_0000_0000, end: 0x7f00_0010_0000, state: MEM_COMMIT, kind: 0x2_0000 },
            Region { start: 0x7f00_0010_0000, end: 0x7f00_0020_0000, state: MEM_COMMIT, kind: MEM_MAPPED },
            Region { start: 0x7f00_0020_0000, end: 0x7f00_0030_0000, state: MEM_COMMIT, kind: MEM_MAPPED },
            Region { start: 0x7f00_0030_0000, end: 0x7f00_0040_0000, state: MEM_COMMIT, kind: MEM_IMAGE },
            Region { start: 0x7f10_0000_0000, end: 0x7f20_0000_0000, state: MEM_RESERVE, kind: 0 },
        ];
        let text = "\
100000000-100100000 rw-p 00000000 00:00 0
Size:               1024 kB
Rss:                 400 kB
Pss:                 400 kB
Pss_Anon:            400 kB
Anonymous:           400 kB
Swap:                 64 kB
VmFlags: rd wr mr mw me ac
200000000-200100000 rw-s 00000000 00:01 77                         /memfd:wine-mapping (deleted)
Rss:                1000 kB
Pss:                 500 kB
Pss_Anon:              0 kB
Anonymous:             0 kB
Swap:                  0 kB
7f0000000000-7f0000100000 rw-p 00000000 00:00 0
Rss:                 300 kB
Pss:                 300 kB
Anonymous:           300 kB
7f0000100000-7f0000200000 rw-s 00000000 00:01 77                         /memfd:wine-mapping (deleted)
Rss:                1000 kB
Pss:                 500 kB
Pss_Anon:              0 kB
Anonymous:             0 kB
7f0000200000-7f0000300000 r--s 00000000 08:01 12                         /mnt/data/victory/cache/pipelines.bin
Rss:                2048 kB
Pss:                2048 kB
7f0000300000-7f0000400000 r-xp 00000000 00:00 0
Rss:                 128 kB
Pss:                 128 kB
Anonymous:           128 kB
7f1000000000-7f1000100000 rw-p 00000000 00:00 0
Rss:                 700 kB
Pss:                 700 kB
Anonymous:           700 kB
7f1000100000-7f1000200000 r-xp 00000000 08:01 13                         /usr/lib/libnvidia-glcore.so.580
Rss:                 900 kB
Pss:                 900 kB
7f3000000000-7f3000100000 rw-s 00000000 00:05 9                          /dev/nvidiactl
Rss:                 256 kB
Pss:                 256 kB
";
        let (s, files, maps) = read_smaps(std::io::Cursor::new(text), &regions);
        assert_eq!(maps, 9);
        let k = |kind: Kind| kind as usize;
        assert_eq!((s.rss[k(Kind::TitleRegions)], s.pss[k(Kind::TitleRegions)], s.swap[k(Kind::TitleRegions)]), (400, 400, 64));
        assert_eq!((s.rss[k(Kind::TitleViews)], s.pss[k(Kind::TitleViews)]), (1000, 500));
        assert_eq!(s.rss[k(Kind::Private)], 300);
        assert_eq!((s.rss[k(Kind::HostViews)], s.pss[k(Kind::HostViews)]), (1000, 500));
        assert_eq!(s.rss[k(Kind::Files)], 2048);
        assert_eq!(files, vec![(2048, "/mnt/data/victory/cache/pipelines.bin".to_string())]);
        assert_eq!(s.rss[k(Kind::Images)], 128);
        assert_eq!(s.rss[k(Kind::UnixAnon)], 700);
        assert_eq!(s.rss[k(Kind::UnixFiles)], 900);
        assert_eq!(s.rss[k(Kind::Driver)], 256);
    }
}
