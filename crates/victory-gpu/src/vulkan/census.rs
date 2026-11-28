//! Diagnostics (`VICTORY_VK_WRITE_CENSUS=<frame>,<frames>`): which pages of
//! the title memory the card reads in place (the imported ranges of the
//! memory map) change from one present to the next, whoever wrote them.
//! Each 4 KiB page is hashed at the presents of the window; at its end
//! `write-census.txt` in the capture folder lists every page that changed
//! (`<range start> <page> <mask>`: bit k set when it changed between
//! presents k and k + 1), and a line a range is logged.

use std::collections::BTreeMap;
use std::io::Write as _;
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

fn hash(page: &[u8]) -> u64 {
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
    if frame < first || frame > first + frames {
        return;
    }
    let t = std::time::Instant::now();
    let k = (frame - first) as u32;
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
    tracing::info!(frame, mb = bytes >> 20, ms = t.elapsed().as_millis() as u64, "Vulkan: write census, a present hashed");
    if frame < first + frames {
        return;
    }
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
        let mut text = Vec::new();
        for ((kind, at, len), n) in lock(&VIEWS).iter() {
            let _ = writeln!(text, "{kind} {at:x} {len:x} {n}");
        }
        let _ = std::fs::write(folder.join("buffer-views.txt"), text);
    }
    *guard = None;
}
