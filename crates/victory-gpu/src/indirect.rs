//! Indirect execution's records, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver (run of 2026-11-25: Xid 109 in the
//! credits, after an untranslated compute pass).
//!
//! A record no draw or dispatch of the title asks for is run empty:
//! - draws of 2^23 vertices or more, or of more than 2^26 vertex
//!   instances;
//! - dispatches with a group count past 65,535 (the host's limit), or more
//!   than 2^21 groups in all;
//! - mesh dispatches likewise, with the second and third counts cut to at
//!   most 1 (the console's mesh programs are one-dimensional).
//!
//! Only records in buffers shaders may write are bounded (the title made
//! the buffer for unordered access, or a copy on the card wrote it): what
//! the CPU writes the title meant as it is.
//!
//! The records are copied, bounded, into a buffer of the list's own by a
//! compute pass of ours, not one execution at a time but a **window** of
//! up to [`WINDOW`] records from the first execution's on: the title runs
//! its draws one record at a time, each its own execution, at successive
//! records. A list keeps up to [`WINDOWS`] windows at once, each in its own
//! part of one copy buffer made once per list. Later executions within a
//! window read the bounded copy, with
//! nothing recorded for them, until something recorded since may write the
//! window's records: a shader or a copy declared to write memory over them
//! ([`HostList::wrote`]), or a shader whose writes are not known by address
//! (title memory by address, buffers picked from the heap by index). A
//! barrier alone ends no window (the windows of run p15 were one execution
//! long, a barrier between every two). The window's records past the ones
//! the title runs are bounded too and never read.
//!
//! The pass sets a compute root signature and pipeline of its own; the
//! title's compute root signature and arguments are set back right after
//! it (`crate::draw::State::restore_compute`), so the next dispatch binds
//! nothing anew.
//!
//! The copy buffer is moved from common (or indirect arguments) to
//! unordered access before a pass and to indirect arguments after it, and
//! back to common once, when the list closes.
//!
//! shortcut: the bounds cut legitimate work beyond them; raise them if the
//! title needs more. A list of thousands of records each just inside them
//! still runs for long (there is no budget per list any more).

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), SRV(t0), UAV(u0)"
cbuffer Window : register(b0) {
    uint kind;
    uint stride;
    uint records;
};
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
static const uint MOST_VERTICES = 1u << 23;
static const uint MOST_DRAWN = 1u << 26;
static const uint MOST_GROUPS = 1u << 21;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) {
        return;
    }
    uint at = id.x * stride;
    uint head;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 d = src.Load2(at);
        if (d.x >= MOST_VERTICES || (d.x != 0 && d.y > MOST_DRAWN / d.x)) {
            d = uint2(0, 0);
        }
        dst.Store2(at, d);
        head = 8;
    } else {
        uint3 g = src.Load3(at);
        if (kind != 2) {
            g.yz = min(g.yz, uint2(1, 1));
        }
        // Each count is at most 65,535 here, so x * y fits 32 bits.
        if (any(g > 65535u) || (g.z != 0 && g.x * g.y > MOST_GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        dst.Store3(at, g);
        head = 12;
    }
    for (uint off = head; off < stride; off += 4) {
        dst.Store(at + off, src.Load(at + off));
    }
}
"#;

/// Records a window bounds at most, beyond what one execution asks for.
const WINDOW: u32 = 512;
/// The copy buffer's smallest size.
const SMALLEST: u64 = 128 << 10;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn pass() -> Option<&'static Pass> {
    static PASS: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
    PASS.get_or_init(|| {
        let made = (|| -> windows::core::Result<Pass> {
            let device = &host().ok_or_else(windows::core::Error::empty)?.device;
            let code = crate::host::compile(SOURCE, s!("main"), s!("cs_5_1"))?;
            // SAFETY: our compiled shader, its root signature inside.
            unsafe {
                let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
                let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })?;
                Ok(Pass { root, pso })
            }
        })();
        match made {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(error = %e, "no bound for indirect records: indirect executions dropped");
                None
            }
        }
    })
    .as_ref()
}

/// Title memory a copy on the card wrote since the start, by 64 KiB page
/// (hashed onto 65,536 bits; a page that shares a bit is bounded too):
/// records copied there by the GPU from a buffer shaders write are bounded
/// although the title's buffer is not one shaders write.
static COPIED: [std::sync::atomic::AtomicU64; 1024] = [const { std::sync::atomic::AtomicU64::new(0) }; 1024];

fn pages(address: u64, len: u64) -> impl Iterator<Item = (usize, u64)> {
    (address >> 16..=(address + len.max(1) - 1) >> 16).map(|page| {
        let bit = (page.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 48) as usize;
        (bit / 64, 1u64 << (bit % 64))
    })
}

/// A copy on the card writes `[address, address + len)`.
pub(crate) fn copied_onto(address: u64, len: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    for (word, bit) in pages(address, len) {
        if COPIED[word].load(Relaxed) & bit == 0 {
            COPIED[word].fetch_or(bit, Relaxed);
        }
    }
}

/// Whether a copy on the card may have written `[address, address + len)`.
pub(crate) fn copied(address: u64, len: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    pages(address, len).any(|(word, bit)| COPIED[word].load(Relaxed) & bit != 0)
}

/// What ends a window of bounded records, for the frame line.
#[derive(Clone, Copy)]
pub(crate) enum Cause {
    /// A shader's root unordered access view (to its buffer's end).
    RootUav = 0,
    /// An unordered access view in a shader's table.
    TableUav = 1,
    /// A shader that writes memory not known by address (title memory by
    /// address, buffers picked from the heap by index).
    Untracked = 2,
    /// A title copy run on the card.
    Copy = 3,
}

/// Since the last frame line: bounding passes and the recording time they
/// took (ticks), copy buffers made, executions bounded and not, and
/// windows ended by each [`Cause`].
static PASSES: AtomicU64 = AtomicU64::new(0);
static PASS_TICKS: AtomicU64 = AtomicU64::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
static BOUNDED: AtomicU64 = AtomicU64::new(0);
static UNBOUNDED: AtomicU64 = AtomicU64::new(0);
static ENDED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// An indirect execution ran from its records as they are.
pub(crate) fn unbounded() {
    UNBOUNDED.fetch_add(1, Relaxed);
}

/// The frame line's part, per frame over `frames` frames.
pub(crate) fn frame_text(frames: f64) -> String {
    let take = |c: &AtomicU64| c.swap(0, Relaxed) as f64 / frames;
    let ended = ENDED.each_ref().map(take);
    let ms = crate::fast::ticks_to_ns(PASS_TICKS.swap(0, Relaxed)) as f64 / 1e6 / frames;
    format!(
        "indirect bounding per frame: passes {:.1} ({ms:.1} ms), copy buffers made {:.1}, executions bounded {:.0} unbounded {:.0}, windows ended by root UAV {:.1} table UAV {:.1} by address or heap {:.1} card copy {:.1}",
        take(&PASSES),
        take(&MADE),
        take(&BOUNDED),
        take(&UNBOUNDED),
        ended[0],
        ended[1],
        ended[2],
        ended[3]
    )
}

/// Windows a list keeps at once (one per argument buffer the title
/// interleaves).
const WINDOWS: usize = 8;

/// A bounded window: kind, stride, records, the GPU address of the title
/// records' first, their title memory (from, to), and where their copy
/// starts in the list's copy buffer.
type Bounded = (u32, u32, u32, u64, (u64, u64), u64);

/// A list's copy buffer and the windows bounded into it.
#[derive(Default)]
pub(crate) struct Window {
    /// The copy buffer and its size, kept across the list's recordings.
    copy: Option<(ID3D12Resource, u64)>,
    /// A pass in this recording left the buffer readable as indirect
    /// arguments (else it is common).
    readable: bool,
    /// Where the next window's copy goes in the buffer.
    next: u64,
    /// The windows bounded in this recording, still valid.
    bounded: Vec<Bounded>,
}

impl Window {
    /// Something recorded from now on writes `[address, address + len)`.
    pub(crate) fn wrote(&mut self, address: u64, len: u64, cause: Cause) {
        let end = address.saturating_add(len);
        let before = self.bounded.len();
        self.bounded.retain(|&(.., (from, to), _)| address >= to || end <= from);
        let ended = before - self.bounded.len();
        if ended > 0 {
            ENDED[cause as usize].fetch_add(ended as u64, Relaxed);
        }
    }

    /// Something recorded from now on may write anywhere.
    pub(crate) fn wrote_somewhere(&mut self) {
        if !self.bounded.is_empty() {
            ENDED[Cause::Untracked as usize].fetch_add(self.bounded.len() as u64, Relaxed);
            self.bounded.clear();
        }
    }

    /// The list is reset: its buffer was left common when it closed.
    pub(crate) fn reset(&mut self) {
        self.readable = false;
        self.next = 0;
        self.bounded.clear();
    }

    /// Before the list closes: the buffer goes back to common, where the
    /// next recording's first pass finds it.
    pub(crate) fn close(&mut self, list: &ID3D12GraphicsCommandList) {
        if let (true, Some((copy, _))) = (self.readable, &self.copy) {
            // SAFETY: an open list; the buffer is in `INDIRECT_ARGUMENT`.
            unsafe {
                list.ResourceBarrier(&[transition(
                    copy,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )])
            };
        }
        self.reset();
    }
}

/// The bounded copy of `max` records of `kind` (PC numbering) `stride`
/// bytes apart at GPU address `src` (4-byte aligned, `left` bytes of its
/// host buffer from there on; title address `title`): the buffer and the
/// offset to execute from, and whether a pass of ours was recorded (it set
/// its own compute root signature and pipeline: the caller sets the
/// title's back). `None` if it cannot be bounded: the execution is
/// dropped.
///
/// The copy buffer is made once per list and kept; windows take the next
/// part of it, and when it is full the next window starts again at its
/// beginning (the pass's move of the buffer out of `INDIRECT_ARGUMENT`
/// orders it after every execution before) and the windows before it are
/// forgotten.
pub(crate) fn bounded(
    list: &HostList,
    (kind, stride, max): (u32, u32, u32),
    (src, title): (u64, u64),
    left: u64,
) -> Option<(ID3D12Resource, u64, bool)> {
    let max = max.max(1);
    let mut window = list.indirect();
    let hit = window.bounded.iter().find_map(|&(k, st, records, first, _, at)| {
        ((k, st) == (kind, stride)
            && src >= first
            && (src - first) % u64::from(stride) == 0
            && (src - first) / u64::from(stride) + u64::from(max) <= u64::from(records))
        .then_some(at + (src - first))
    });
    if let (Some(from), Some((copy, _))) = (hit, &window.copy) {
        BOUNDED.fetch_add(1, Relaxed);
        return Some((copy.clone(), from, false));
    }
    let began = crate::fast::ticks();
    let records = (u64::from(max).max(u64::from(WINDOW)))
        .min(left / u64::from(stride))
        .max(u64::from(max));
    let bytes = records * u64::from(stride);
    if bytes > u64::from(u32::MAX) {
        return None;
    }
    let pass = pass()?;
    if window.copy.as_ref().is_none_or(|(_, size)| *size < bytes) {
        let size = bytes.max(SMALLEST).next_power_of_two();
        let copy = host()?.uav_buffer(size)?;
        MADE.fetch_add(1, Relaxed);
        if let Some((old, _)) = window.copy.replace((copy, size)) {
            // The list ends with every copy buffer common.
            if window.readable {
                // SAFETY: an open list; the buffer is in `INDIRECT_ARGUMENT`.
                unsafe {
                    list.list.ResourceBarrier(&[transition(
                        &old,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_COMMON,
                    )])
                };
            }
            list.keep(&old);
        }
        window.readable = false;
        window.next = 0;
        window.bounded.clear();
    }
    let (copy, size) = window.copy.clone()?;
    // The next part of the buffer, records on 16 bytes; from the start
    // again when full (the windows there are forgotten).
    let mut at = window.next.next_multiple_of(16);
    if at + bytes > size {
        at = 0;
        window.bounded.clear();
    }
    if window.bounded.len() >= WINDOWS {
        window.bounded.remove(0);
    }
    let before = if window.readable {
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    } else {
        D3D12_RESOURCE_STATE_COMMON
    };
    let constants = [kind, stride, records as u32];
    // SAFETY: an open list; our root signature and pipeline; the title's
    // records lie in a live host buffer (`left` bytes from `src`), the copy
    // buffer is the list's and `at + bytes` lies inside it. Moving it out
    // of `INDIRECT_ARGUMENT` orders this pass's writes after the reads of
    // the executions before.
    unsafe {
        let l = &list.list;
        l.ResourceBarrier(&[transition(&copy, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src);
        l.SetComputeRootUnorderedAccessView(2, copy.GetGPUVirtualAddress() + at);
        l.Dispatch((records as u32).div_ceil(64), 1, 1);
        l.ResourceBarrier(&[transition(
            &copy,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
    window.readable = true;
    window.next = at + bytes;
    window.bounded.push((kind, stride, records as u32, src, (title, title + bytes), at));
    PASSES.fetch_add(1, Relaxed);
    BOUNDED.fetch_add(1, Relaxed);
    PASS_TICKS.fetch_add(crate::fast::ticks() - began, Relaxed);
    Some((copy, at, true))
}
