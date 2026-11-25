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
//! records. Later executions within the window read the bounded copy, with
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
const WINDOW: u32 = 1024;
/// The copy buffer's smallest size.
const SMALLEST: u64 = 64 << 10;

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

/// A list's copy buffer and the window bounded into it.
#[derive(Default)]
pub(crate) struct Window {
    /// The copy buffer and its size, kept across the list's recordings.
    copy: Option<(ID3D12Resource, u64)>,
    /// A pass in this recording left the buffer readable as indirect
    /// arguments (else it is common).
    readable: bool,
    /// What is bounded in it: kind, stride, records, the GPU address of
    /// the title records' first, and their title memory (from, to).
    bounded: Option<(u32, u32, u32, u64, (u64, u64))>,
}

impl Window {
    /// Something recorded from now on writes `[address, address + len)`.
    pub(crate) fn wrote(&mut self, address: u64, len: u64) {
        if let Some((.., (from, to))) = self.bounded
            && address < to
            && address.saturating_add(len) > from
        {
            self.bounded = None;
        }
    }

    /// Something recorded from now on may write anywhere.
    pub(crate) fn wrote_somewhere(&mut self) {
        self.bounded = None;
    }

    /// The list is reset: its buffer was left common when it closed.
    pub(crate) fn reset(&mut self) {
        self.readable = false;
        self.bounded = None;
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
        self.readable = false;
        self.bounded = None;
    }
}

/// The bounded copy of `max` records of `kind` (PC numbering) `stride`
/// bytes apart at GPU address `src` (4-byte aligned, `left` bytes of its
/// host buffer from there on; title address `title`): the buffer and the offset to execute from,
/// and whether a pass of ours was recorded (it set its own compute root
/// signature and pipeline: the caller binds the title's again). `None` if
/// it cannot be bounded: the execution is dropped.
pub(crate) fn bounded(
    list: &HostList,
    (kind, stride, max): (u32, u32, u32),
    (src, title): (u64, u64),
    left: u64,
) -> Option<(ID3D12Resource, u64, bool)> {
    let max = max.max(1);
    let mut window = list.indirect();
    if let (Some((k, st, records, first, _)), Some((copy, _))) = (window.bounded, &window.copy)
        && (k, st) == (kind, stride)
        && src >= first
        && (src - first) % u64::from(stride) == 0
        && (src - first) / u64::from(stride) + u64::from(max) <= u64::from(records)
    {
        return Some((copy.clone(), src - first, false));
    }
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
            window.readable = false;
        }
    }
    let (copy, _) = window.copy.clone()?;
    let before = if window.readable {
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    } else {
        D3D12_RESOURCE_STATE_COMMON
    };
    let constants = [kind, stride, records as u32];
    // SAFETY: an open list; our root signature and pipeline; the title's
    // records lie in a live host buffer (`left` bytes from `src`), the copy
    // buffer is the list's. Moving it out of `INDIRECT_ARGUMENT` orders this
    // pass's writes after the reads of the executions before.
    unsafe {
        let l = &list.list;
        l.ResourceBarrier(&[transition(&copy, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src);
        l.SetComputeRootUnorderedAccessView(2, copy.GetGPUVirtualAddress());
        l.Dispatch((records as u32).div_ceil(64), 1, 1);
        l.ResourceBarrier(&[transition(
            &copy,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
    window.readable = true;
    window.bounded = Some((kind, stride, records as u32, src, (title, title + bytes)));
    Some((copy, 0, true))
}
