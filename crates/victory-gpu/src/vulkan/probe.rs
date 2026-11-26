//! Diagnostics: where title memory the card reads lives (the pool's GPU
//! view, an import in place, a big view of a section, nothing), and
//! whether the card sees what the CPU wrote there: a probe copies 64 bytes
//! at each address, as the card sees them, into a read-back buffer in the
//! command buffer that reads them, and the landing compares them with the
//! CPU's bytes at recording and at landing.

use std::sync::Mutex;

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::{Backend, memory, read_title};

/// Bytes probed at each address.
const SPAN: u64 = 64;
/// The marker written after the copies: the probe has run.
const DONE: u32 = 0x600d_f00d;
/// Probes in a run at most.
const MOST: usize = 64;

/// What the CPU's page query says of `at`: state, protection, type, and
/// the allocation's base.
fn page(at: u64) -> String {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut i = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a query into our own structure.
    if unsafe { VirtualQuery(at as *const _, &mut i, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return "no page".into();
    }
    let kind = match i.Type {
        0x2_0000 => "private",
        0x4_0000 => "mapped",
        0x100_0000 => "image",
        _ => "?",
    };
    format!("state={:#x} protect={:#x} {kind} allocation={:#x}", i.State, i.Protect, i.AllocationBase as usize)
}

/// Where the card reads title address `at`.
pub(super) fn describe(b: &Backend, at: u64) -> String {
    let m = memory(b);
    let map = m.map.range_of(at).map_or_else(
        || "no map range (the card reads zeros)".to_string(),
        |r| format!("map range {:#x}+{:#x} -> device {:#x}", r.title_start, r.title_size, r.device_base + (at - r.title_start)),
    );
    let home = if let Some(p) = m.pools.iter().find(|p| p.contains(at)) {
        match p.page_at(at) {
            Some((key, off)) => format!("pool view, physical page {:?} +{off:#x} (card copy)", key),
            None => "pool view, nothing mapped there".into(),
        }
    } else if let Some((&a, r)) = m.regions.range(..=at as usize).next_back().filter(|(a, r)| at < (**a + r.len) as u64) {
        format!(
            "region {a:#x}+{:#x} loose={} import={}",
            r.len,
            r.loose,
            r.import.as_ref().map_or("none".into(), |i| format!("device {:#x}", i.device_base))
        )
    } else if let Some((&a, &(len, section, _))) = m.views.range(..=at as usize).next_back().filter(|(a, v)| at < (**a + v.0) as u64) {
        format!(
            "big view {a:#x}+{len:#x} of section {section:#x}, imported={}",
            m.sections.get(&section).is_some_and(|s| s.import.is_some())
        )
    } else {
        "in no region, view or pool".into()
    };
    format!("{home}; {map}; {}", page(at))
}

/// The CPU's dwords at `at` (`SPAN` bytes), if it reads them.
fn cpu_words(b: &Backend, at: u64) -> Option<Vec<u32>> {
    let bytes = read_title(b, at, SPAN).ok()?;
    Some(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

struct Probe {
    buf: Buffer,
    targets: Vec<(String, u64)>,
    cpu: Vec<Option<Vec<u32>>>,
}

static PROBES: Mutex<Vec<Probe>> = Mutex::new(Vec::new());
/// The last culling dispatch's targets, described again at a guard hit.
static LAST: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());

pub(super) fn remember(targets: &[(String, u64)]) {
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = targets.to_vec();
}

/// At a guard hit: where the last culling dispatch's targets live and the
/// CPU's dwords there now.
pub(super) fn describe_last(b: &Backend) {
    static TIMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if TIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 8 {
        return;
    }
    let last = LAST.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for (what, at) in last {
        tracing::warn!(
            what = %what,
            at = format_args!("{at:#x}"),
            home = %describe(b, at),
            cpu = format_args!("{:08x?}", cpu_words(b, at).map(|c| c[..8].to_vec())),
            "Vulkan: at a guard hit, the last culling dispatch's memory"
        );
    }
}
static MADE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Records, into `cmd` (outside rendering), a copy of `SPAN` bytes at each
/// target as the card sees them; logs where each lives. At most [`MOST`]
/// probes a run.
pub(super) fn record(b: &Backend, cmd: vk::CommandBuffer, targets: Vec<(String, u64)>) {
    use std::sync::atomic::Ordering;
    if !cfg!(feature = "diagnostics") || targets.is_empty() || MADE.fetch_add(1, Ordering::Relaxed) >= MOST {
        return;
    }
    let n = targets.len() as u64;
    let Ok(buf) = Buffer::new(&b.host, n * SPAN + 16, vk::BufferUsageFlags::TRANSFER_DST, Place::Readback) else { return };
    // SAFETY: our mapped buffer, `n * SPAN + 16` bytes.
    unsafe { std::ptr::write_bytes(buf.mapped, 0xee, (n * SPAN + 16) as usize) };
    let d = &b.host.device;
    // SAFETY: outside rendering in the caller's command buffer; copies from
    // title buffers the card reads into our read-back buffer.
    unsafe {
        super::images::full_barrier(d, cmd);
        for (k, (_, at)) in targets.iter().enumerate() {
            if let Some((src, off)) = super::buffer_of(b, *at) {
                let region = vk::BufferCopy::default().src_offset(off).dst_offset(k as u64 * SPAN).size(SPAN);
                d.cmd_copy_buffer(cmd, src, buf.buffer, &[region]);
            }
        }
        super::images::full_barrier(d, cmd);
        d.cmd_update_buffer(cmd, buf.buffer, n * SPAN, &DONE.to_le_bytes());
    }
    let cpu = targets.iter().map(|(_, at)| cpu_words(b, *at)).collect();
    for (what, at) in &targets {
        tracing::info!(what = %what, at = format_args!("{at:#x}"), home = %describe(b, *at), "Vulkan probe: where the card reads it");
    }
    PROBES.lock().unwrap_or_else(|e| e.into_inner()).push(Probe { buf, targets, cpu });
}

/// After a landing: each probe that has run logs the card's dwords against
/// the CPU's at recording and now.
pub(super) fn check(b: &Backend) {
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let mut probes = PROBES.lock().unwrap_or_else(|e| e.into_inner());
    if probes.is_empty() {
        return;
    }
    let mut left = Vec::new();
    for p in probes.drain(..) {
        let n = p.targets.len();
        // SAFETY: our mapped read-back buffer; the marker after the copies.
        let marker = unsafe { p.buf.mapped.add(n * SPAN as usize).cast::<u32>().read_volatile() };
        if marker != DONE {
            left.push(p);
            continue;
        }
        for (k, (what, at)) in p.targets.iter().enumerate() {
            // SAFETY: as above, `SPAN` bytes at `k * SPAN`.
            let card: Vec<u32> = (0..SPAN as usize / 4).map(|w| unsafe { p.buf.mapped.add(k * SPAN as usize + 4 * w).cast::<u32>().read_volatile() }).collect();
            let now = cpu_words(b, *at);
            let at_record = p.cpu[k].as_ref();
            tracing::info!(
                what = %what,
                at = format_args!("{at:#x}"),
                same_as_cpu_at_record = at_record.is_some_and(|c| *c == card),
                same_as_cpu_now = now.as_ref().is_some_and(|c| *c == card),
                card = format_args!("{:08x?}", &card[..8]),
                cpu_at_record = format_args!("{:08x?}", at_record.map(|c| &c[..8])),
                cpu_now = format_args!("{:08x?}", now.as_ref().map(|c| &c[..8])),
                "Vulkan probe: what the card read (first 8 dwords; 0xeeeeeeee: not copied)"
            );
        }
        // SAFETY: the card has run the copies into it.
        unsafe { p.buf.destroy(&b.host) };
    }
    *probes = left;
}
