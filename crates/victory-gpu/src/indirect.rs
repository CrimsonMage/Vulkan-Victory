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
//! window read the bounded copy, with nothing recorded for them, until
//! something recorded since may write the window's records: a shader or a
//! copy declared to write memory over them ([`HostList::wrote`]), a shader
//! that writes title memory by address where the records have a copy it
//! reaches, or one that writes buffers it picks from the heap by index. A
//! barrier alone ends no window (the windows of run p15 were one execution
//! long, a barrier between every two). The window's records past the ones
//! the title runs are bounded too and never read.
//!
//! A window that executions read and a write ended is bounded again at the
//! title's next barrier that orders writes ([`rebound`]): the barrier
//! splits the host's render pass anyway, and the executions after it, in
//! the title's next render pass, read the copy with no pass of ours
//! between them (run p18: 1,522 passes a frame at executions, and ~135 µs
//! of recording per indirect execution outside them, which a pass between
//! two draws costs the host in ending and starting its render pass again).
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
    /// A shader that writes title memory by address, where the window's
    /// records have a copy it reaches (`crate::memory::reachable`).
    ByAddress = 2,
    /// A shader that writes buffers it picks from the heap by index.
    Heap = 3,
    /// A title copy run on the card.
    Copy = 4,
}

/// Since the last frame line: bounding passes (at executions, at the
/// title's barriers) and the recording time they took (ticks), copy
/// buffers made, executions bounded and not, windows ended by each
/// [`Cause`], and the recording time of an indirect execution's parts
/// ([`Part`]).
static PASSES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static PASS_TICKS: AtomicU64 = AtomicU64::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
static BOUNDED: AtomicU64 = AtomicU64::new(0);
static UNBOUNDED: AtomicU64 = AtomicU64::new(0);
static ENDED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static PARTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The parts of an indirect execution's recording, timed for the frame
/// line.
#[derive(Clone, Copy)]
pub(crate) enum Part {
    /// Finding its records' window or bounding them (passes included).
    Bound = 0,
    /// Binding the title's pipeline and arguments (the draw's preparation).
    Bind = 1,
    /// The host's `ExecuteIndirect`.
    Execute = 2,
    /// Finding the records' buffers and the signature, before.
    Find = 3,
    /// From the call's timing on to the list's draw state taken (its lock).
    Lock = 4,
    /// Letting go of what the execution held (the copy buffer, the
    /// signature, the host copies), after.
    After = 5,
}

/// Adds the ticks since `began` to `part`; the ticks now.
pub(crate) fn timed(part: Part, began: u64) -> u64 {
    let now = crate::fast::ticks();
    PARTS[part as usize].fetch_add(now - began, Relaxed);
    now
}

/// An indirect execution ran from its records as they are.
pub(crate) fn unbounded() {
    UNBOUNDED.fetch_add(1, Relaxed);
}

/// The frame line's part, per frame over `frames` frames.
pub(crate) fn frame_text(frames: f64) -> String {
    let take = |c: &AtomicU64| c.swap(0, Relaxed) as f64 / frames;
    let ms = |c: &AtomicU64| crate::fast::ticks_to_ns(c.swap(0, Relaxed)) as f64 / 1e6 / frames;
    let ended = ENDED.each_ref().map(take);
    let parts = PARTS.each_ref().map(ms);
    format!(
        "indirect bounding per frame: passes at executions {:.1} at barriers {:.1} ({:.1} ms), copy buffers made {:.1}, executions bounded {:.0} unbounded {:.0}, \
         windows ended by root UAV {:.1} table UAV {:.1} by address {:.1} heap {:.1} card copy {:.1}; indirect call parts ms: lock {:.1} find {:.1} bound {:.1} bind {:.1} ExecuteIndirect {:.1} after {:.1}",
        take(&PASSES[0]),
        take(&PASSES[1]),
        ms(&PASS_TICKS),
        take(&MADE),
        take(&BOUNDED),
        take(&UNBOUNDED),
        ended[0],
        ended[1],
        ended[2],
        ended[3],
        ended[4],
        parts[4],
        parts[3],
        parts[0],
        parts[1],
        parts[2],
        parts[5]
    )
}

/// Windows a list keeps at once (one per argument buffer the title
/// interleaves), and windows ended by writes that are bounded again at the
/// title's next barrier.
const WINDOWS: usize = 8;

/// What a window bounds: kind, stride, records, the GPU address of the
/// title records' first, and their title memory (from, to).
type Spec = (u32, u32, u32, u64, (u64, u64));

/// A bounded window: what it bounds, where its copy starts in the list's
/// copy buffer, and whether an execution read it since it was bounded.
type Bounded = (Spec, u64, bool);

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
    /// Windows that executions read and writes ended since: bounded again
    /// at the title's next barrier ([`rebound`]).
    stale: Vec<Spec>,
}

impl Window {
    /// Ends the windows `ends` says, by `cause`; the ones executions read
    /// are bounded again at the title's next barrier.
    fn end(&mut self, cause: Cause, mut ends: impl FnMut(&Spec) -> bool) {
        let mut ended = 0;
        let mut k = 0;
        while k < self.bounded.len() {
            let (spec, _, read) = self.bounded[k];
            if !ends(&spec) {
                k += 1;
                continue;
            }
            self.bounded.remove(k);
            ended += 1;
            if read && !self.stale.contains(&spec) {
                if self.stale.len() >= WINDOWS {
                    self.stale.remove(0);
                }
                self.stale.push(spec);
            }
        }
        if ended > 0 {
            ENDED[cause as usize].fetch_add(ended, Relaxed);
        }
    }

    /// Something recorded from now on writes `[address, address + len)`.
    pub(crate) fn wrote(&mut self, address: u64, len: u64, cause: Cause) {
        let end = address.saturating_add(len);
        self.end(cause, |&(.., (from, to))| address < to && end > from);
    }

    /// A shader recorded from now on writes title memory by address: the
    /// windows whose records it reaches end.
    pub(crate) fn wrote_by_address(&mut self) {
        if self.bounded.is_empty() {
            return;
        }
        self.end(Cause::ByAddress, |&(.., (from, to))| crate::memory::reachable(from, to));
    }

    /// A shader recorded from now on writes buffers it picks from the heap:
    /// the windows over memory an unordered access view of the heap names
    /// end (`uavs`: their title memory, sorted and joined; every window
    /// without it).
    pub(crate) fn wrote_heap(&mut self, uavs: Option<&[(u64, u64)]>) {
        if self.bounded.is_empty() {
            return;
        }
        self.end(Cause::Heap, |&(.., (from, to))| {
            let Some(uavs) = uavs else {
                return true;
            };
            // The first span that ends past `from`: it holds some of the
            // window if it starts before `to`.
            let k = uavs.partition_point(|&(_, end)| end <= from);
            uavs.get(k).is_some_and(|&(start, _)| start < to)
        });
    }

    /// The list is reset: its buffer was left common when it closed.
    pub(crate) fn reset(&mut self) {
        self.readable = false;
        self.next = 0;
        self.bounded.clear();
        self.stale.clear();
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

    /// Records the pass that bounds `spec` into the next part of the copy
    /// buffer (made or grown first if it must be): where the copy starts.
    /// The buffer is made once per list and kept; when it is full the next
    /// window starts again at its beginning (the pass's move of the buffer
    /// out of `INDIRECT_ARGUMENT` orders it after every execution before)
    /// and the windows before it are forgotten.
    fn record(&mut self, list: &HostList, spec: Spec, read: bool) -> Option<u64> {
        let began = crate::fast::ticks();
        let (kind, stride, records, src, _) = spec;
        let bytes = u64::from(records) * u64::from(stride);
        let pass = pass()?;
        if self.copy.as_ref().is_none_or(|(_, size)| *size < bytes) {
            let size = bytes.max(SMALLEST).next_power_of_two();
            let copy = host()?.uav_buffer(size)?;
            MADE.fetch_add(1, Relaxed);
            if let Some((old, _)) = self.copy.replace((copy, size)) {
                // The list ends with every copy buffer common.
                if self.readable {
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
            self.readable = false;
            self.next = 0;
            self.bounded.clear();
        }
        let (copy, size) = self.copy.clone()?;
        // The next part of the buffer, records on 16 bytes; from the start
        // again when full (the windows there are forgotten).
        let mut at = self.next.next_multiple_of(16);
        if at + bytes > size {
            at = 0;
            self.bounded.clear();
        }
        if self.bounded.len() >= WINDOWS {
            self.bounded.remove(0);
        }
        let before = if self.readable {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };
        let constants = [kind, stride, records];
        // SAFETY: an open list; our root signature and pipeline; the title's
        // records lie in a live host buffer (`bytes` from `src`), the copy
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
            l.Dispatch(records.div_ceil(64), 1, 1);
            l.ResourceBarrier(&[transition(
                &copy,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
        self.readable = true;
        self.next = at + bytes;
        self.bounded.push((spec, at, read));
        PASS_TICKS.fetch_add(crate::fast::ticks() - began, Relaxed);
        Some(at)
    }
}

/// The bounded copy of `max` records of `kind` (PC numbering) `stride`
/// bytes apart at GPU address `src` (4-byte aligned, `left` bytes of its
/// host buffer from there on; title address `title`): the buffer and the
/// offset to execute from, and whether a pass of ours was recorded (it set
/// its own compute root signature and pipeline: the caller sets the
/// title's back). `None` if it cannot be bounded: the execution is
/// dropped.
pub(crate) fn bounded(
    list: &HostList,
    (kind, stride, max): (u32, u32, u32),
    (src, title): (u64, u64),
    left: u64,
) -> Option<(ID3D12Resource, u64, bool)> {
    let max = max.max(1);
    let mut window = list.indirect();
    let hit = window.bounded.iter_mut().find_map(|((k, st, records, first, _), at, read)| {
        let inside = (*k, *st) == (kind, stride)
            && src >= *first
            && (src - *first) % u64::from(stride) == 0
            && (src - *first) / u64::from(stride) + u64::from(max) <= u64::from(*records);
        inside.then(|| {
            *read = true;
            *at + (src - *first)
        })
    });
    if let (Some(from), Some((copy, _))) = (hit, &window.copy) {
        BOUNDED.fetch_add(1, Relaxed);
        return Some((copy.clone(), from, false));
    }
    let records = (u64::from(max).max(u64::from(WINDOW)))
        .min(left / u64::from(stride))
        .max(u64::from(max));
    let bytes = records * u64::from(stride);
    if bytes > u64::from(u32::MAX) {
        return None;
    }
    let spec = (kind, stride, records as u32, src, (title, title + bytes));
    window.stale.retain(|s| *s != spec);
    let at = window.record(list, spec, true)?;
    PASSES[0].fetch_add(1, Relaxed);
    BOUNDED.fetch_add(1, Relaxed);
    let (copy, _) = window.copy.clone()?;
    Some((copy, at, true))
}

/// At a barrier of the title's that orders writes (its passes' writes are
/// done for what follows): the windows executions read and writes ended
/// since are bounded again here, where the barrier already splits the
/// host's render pass, so that the executions after it read them with no
/// pass of ours in between. Whether a pass was recorded (the caller sets
/// the title's compute bindings back).
pub(crate) fn rebound(list: &HostList) -> bool {
    let mut window = list.indirect();
    if window.stale.is_empty() {
        return false;
    }
    let stale = std::mem::take(&mut window.stale);
    let mut any = false;
    for spec in stale {
        if window.record(list, spec, false).is_some() {
            PASSES[1].fetch_add(1, Relaxed);
            any = true;
        }
    }
    any
}
