//! `ExecuteIndirect` (`docs/vulkan-host.md`, "Indirect"). A signature holds
//! one argument (draw, indexed draw, dispatch, mesh dispatch), whose
//! records are already Vulkan's command structs.
//!
//! 1. A compute pass of ours (`vulkan/bound.rs`) copies the records into
//!    the command buffer's scratch (freed with the command buffer, once its
//!    line has passed it), bounded as the D3D12 path bounds them, with the
//!    count (the title's, at most `max`).
//! 2. A barrier to `DRAW_INDIRECT`.
//! 3. `vkCmdDraw(Indexed)Indirect(Count)`, `vkCmdDrawMeshTasksIndirect(Count)`
//!    or `vkCmdDispatchIndirect` from the scratch. Draws bind as direct
//!    draws do, so indirect rectangles take the rectangle part.
//!
//! Executions without a count read a **window**: a pass bounds up to
//! [`WINDOW`] records from the first execution's on, and the executions at
//! the next records read them with no pass of ours, until a dispatch or a
//! copy may have written records ([`Windows::clear`]) or the command buffer
//! ends.
//!
//! The cap: at most [`MOST`] records an execution (and the title's buffer),
//! at most [`MOST_DISPATCHES`] for dispatches (one command each).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};
use victory_vk::timeline::ScratchRange;

use super::super::bound::{self, DISPATCH, DRAW, DRAW_INDEXED, MESH};
use super::super::{Backend, buffer_over, device_of, mesh, stages};
use super::{NotYet, Rec, not_yet, skipped};
use crate::object::This;

/// Records a window bounds beyond what one execution asks for.
const WINDOW: u32 = 512;
/// Records one execution runs at most.
const MOST: u32 = 1 << 16;
/// Dispatch records one execution runs at most.
const MOST_DISPATCHES: u32 = 256;
/// Windows a command buffer keeps.
const WINDOWS: usize = 8;
/// Bytes before the bounded records: the count.
const HEAD: u64 = 16;

/// Executions, bounding passes, executions read from a window, executions
/// skipped (records out of reach, no pass), for the 10 s line.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn tally(what: usize) {
    COUNTS[what].fetch_add(1, Ordering::Relaxed);
}

/// The 10 s line's part.
pub(super) fn text() -> String {
    let n = |i: usize| COUNTS[i].load(Ordering::Relaxed);
    format!(
        "executions={} passes={} window_hits={} skipped={}",
        n(0),
        n(1),
        n(2),
        n(3)
    )
}

/// One execution, as the title asked for it.
#[derive(Clone, Copy)]
struct Spec {
    kind: u32,
    /// Words a record's command takes (the bounded copy's stride is four
    /// times that).
    words: u32,
    stride: u32,
    /// Title address of the first record.
    src: u64,
    max: u32,
    /// Records the title's buffer holds from `src` on (at most [`MOST`]).
    room: u32,
    /// Title address of the count.
    count: Option<u64>,
    /// Mesh: host groups per console group of the program (the bounded
    /// y count is multiplied by it), 1 otherwise.
    parts: u32,
}

impl Spec {
    /// The kind the bounding pass and the windows take: the parts in the
    /// high half.
    fn bound_kind(&self) -> u32 {
        self.kind | self.parts << 16
    }
}

/// Bounded records in the command buffer's scratch.
#[derive(Clone, Copy)]
struct Window {
    kind: u32,
    stride: u32,
    first: u64,
    records: u32,
    at: ScratchRange,
}

/// The windows of the command buffer being recorded.
#[derive(Default)]
pub(super) struct Windows(Vec<Window>);

impl Windows {
    /// Something recorded from now on may write records (a dispatch, a
    /// copy), or the command buffer is new: no window holds.
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }

    /// Where `spec`'s records lie bounded in a window, if one holds them.
    fn find(&self, spec: &Spec) -> Option<(vk::Buffer, u64)> {
        self.0.iter().find_map(|w| {
            let step = u64::from(spec.stride);
            let inside = (w.kind, w.stride) == (spec.bound_kind(), spec.stride)
                && spec.src >= w.first
                && (spec.src - w.first).is_multiple_of(step)
                && (spec.src - w.first) / step + u64::from(spec.max) <= u64::from(w.records);
            inside.then(|| {
                (
                    w.at.buffer,
                    w.at.offset + HEAD + (spec.src - w.first) / step * u64::from(4 * spec.words),
                )
            })
        })
    }
}

/// The execution's argument: kind and words of its command.
fn argument(signature: This) -> Option<(u32, u32, u32)> {
    const DISPATCH_SINGLE: u32 = 105;
    let (stride, args) = crate::signature::layout(signature)?;
    let [[kind, ..]] = args else { return None };
    let kind = if *kind == DISPATCH_SINGLE {
        DISPATCH
    } else {
        *kind
    };
    let words = match kind {
        DRAW => 4,
        DRAW_INDEXED => 5,
        DISPATCH | MESH => 3,
        _ => return None,
    };
    (stride >= 4 * words && stride.is_multiple_of(4)).then_some((kind, words, stride))
}

fn spec(
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) -> Option<Spec> {
    let (kind, words, stride) = argument(signature)?;
    let (data, size) = crate::resource::buffer_bytes(arguments)?;
    let src = data as u64 + offset;
    let room = (size.saturating_sub(offset) / u64::from(stride)).min(u64::from(MOST)) as u32;
    let count = match crate::resource::buffer_bytes(count) {
        Some((c, len)) if count_offset + 4 <= len => Some(c as u64 + count_offset),
        Some(_) => return None,
        None => None,
    };
    let most = if kind == DISPATCH {
        MOST_DISPATCHES
    } else {
        MOST
    };
    src.is_multiple_of(4).then_some(Spec {
        kind,
        words,
        stride,
        src,
        max: max.min(room).min(most),
        room,
        count,
        parts: 1,
    })
}

/// A word of `u32::MAX`: the count of executions without one.
fn no_count(b: &Backend) -> Option<u64> {
    static WORD: OnceLock<Option<Buffer>> = OnceLock::new();
    WORD.get_or_init(|| {
        let word = Buffer::new(
            &b.host,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Place::Upload,
        )
        .ok()?;
        // SAFETY: a mapped buffer of 16 bytes, ours.
        unsafe { word.mapped.cast::<[u32; 4]>().write([u32::MAX; 4]) };
        Some(word)
    })
    .as_ref()
    .map(|w| w.address)
}

/// The bounding pass's pipeline (over the ABI's layout: its push
/// constants are the first 40 bytes of the 256).
fn pipeline(b: &Backend) -> Option<vk::Pipeline> {
    static PIPELINE: OnceLock<Option<vk::Pipeline>> = OnceLock::new();
    *PIPELINE.get_or_init(|| {
        let layout = stages::layout(b)?;
        let d = &b.host.device;
        let words = bound::module();
        let flags = if b.host.caps.descriptor_buffer { vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT } else { vk::PipelineCreateFlags::empty() };
        // SAFETY: valid create infos; kept for the run.
        let made = unsafe {
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None).ok()?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let info = vk::ComputePipelineCreateInfo::default().flags(flags).stage(stage).layout(layout.pipeline);
            let p = d.create_compute_pipelines(stages::cache(b), &[info], None);
            d.destroy_shader_module(module, None);
            p
        };
        match made {
            Ok(p) => Some(p[0]),
            Err((_, e)) => {
                tracing::error!(error = %e, "Vulkan: no bounding pass; indirect executions are skipped");
                None
            }
        }
    })
}

/// Diagnostics, to bisect a hang: `VICTORY_DIAG_NO_INDIRECT=1` skips
/// every execution (counted as skipped), as before indirect ran.
fn skip_all() -> bool {
    static SKIP: OnceLock<bool> = OnceLock::new();
    *SKIP.get_or_init(|| {
        let skip = cfg!(feature = "diagnostics")
            && std::env::var("VICTORY_DIAG_NO_INDIRECT").is_ok_and(|v| v.trim() == "1");
        if skip {
            tracing::warn!("Vulkan: indirect executions are skipped (VICTORY_DIAG_NO_INDIRECT=1, diagnostics)");
        }
        skip
    })
}

fn has_count_draws(b: &Backend) -> bool {
    b.host
        .caps
        .features
        .iter()
        .any(|f| f.name == "drawIndirectCount" && f.present)
}

impl Rec {
    /// `ExecuteIndirect(signature, max, (arguments, offset), (count,
    /// count offset))`.
    pub(crate) fn execute_indirect(
        &mut self,
        b: &'static Backend,
        signature: This,
        max: u32,
        (arguments, offset): (This, u64),
        (count, count_offset): (This, u64),
    ) {
        let Some(mut spec) = spec(signature, max, arguments, offset, count, count_offset) else {
            not_yet(NotYet::Indirect);
            return;
        };
        if spec.max == 0 {
            return;
        }
        tally(0);
        if skip_all() {
            tally(3);
            skipped(false);
            return;
        }
        if spec.kind == MESH {
            let Some(parts) = self.mesh_parts(b) else {
                tally(3);
                return;
            };
            spec.parts = parts;
        }
        let Some((buffer, at, counted)) = self.bounded(b, &spec) else {
            tally(3);
            skipped(false);
            return;
        };
        self.note_as(
            || {
                format!(
                    "indirect kind={} max={} count_at={:?} src={:#x}",
                    spec.kind, spec.max, spec.count, spec.src
                )
            },
            true,
        );
        super::super::passes::note(|| format!("indirect kind {} max {} count_at {:x?} src {:x}", spec.kind, spec.max, spec.count, spec.src));
        let size = 4 * spec.words;
        match spec.kind {
            DISPATCH => {
                self.dispatch_indirect(b, buffer, at, spec.max);
                self.watch_cull(b, None, Some((spec.src, spec.count)));
                // The captured program ends here (left open, the next
                // program's push and dispatch lines went into its capture);
                // its capture opened in `dispatch_indirect`, after the
                // `indirect` line above, so its arguments are named again.
                super::super::passes::note(|| format!("dispatch-indirect src {:x}", spec.src));
                super::super::passes::close();
            }
            MESH => {
                self.last_indirect = Some((spec.src, spec.count));
                self.mesh_indirect(b, buffer, at, counted, spec.max);
            }
            _ => self.draw_indirect(
                b,
                spec.kind == DRAW_INDEXED,
                (buffer, at, counted),
                spec.max,
                size,
            ),
        }
    }

    /// The bounded records (buffer, offset) and where their count lies,
    /// from a window or a pass recorded now.
    fn bounded(
        &mut self,
        b: &'static Backend,
        spec: &Spec,
    ) -> Option<(vk::Buffer, u64, Option<u64>)> {
        if spec.count.is_none()
            && let Some((buffer, at)) = self.indirect.find(spec)
        {
            tally(2);
            return Some((buffer, at, None));
        }
        let records = if spec.count.is_some() {
            spec.max
        } else {
            spec.max.max(WINDOW).min(spec.room)
        };
        let span = u64::from(records) * u64::from(spec.stride);
        let src = device_of(b, spec.src, span)?;
        let count = match spec.count {
            Some(at) => device_of(b, at, 4)?,
            None => no_count(b)?,
        };
        let pipeline = pipeline(b)?;
        let layout = stages::layout(b)?;
        self.end_pass(b);
        let cmd = self.cmd(b)?;
        let at = self
            .cmd
            .as_mut()?
            .scratch(&b.host, &b.timeline, HEAD + u64::from(records * 4 * spec.words))?;
        let mut push = Vec::with_capacity(bound::PUSH_BYTES);
        for v in [src, at.address, count] {
            push.extend(v.to_le_bytes());
        }
        for v in [spec.words, spec.stride, records, spec.bound_kind()] {
            push.extend(v.to_le_bytes());
        }
        let d = &b.host.device;
        // SAFETY: outside rendering in our command buffer; the pass reads
        // the title's records by device address (inside an import, checked
        // above) and writes the scratch range it was given.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_push_constants(cmd, layout.pipeline, vk::ShaderStageFlags::ALL, 0, &push);
            d.cmd_dispatch(cmd, (records * spec.words).div_ceil(bound::GROUP), 1, 1);
        }
        self.note(|| format!("bounding pass records={records}"));
        // SAFETY: as above.
        unsafe {
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)];
            d.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&barrier),
            );
        }
        // The title's compute pipeline and push constants are bound again
        // at its next draw or dispatch.
        self.bound_compute = vk::Pipeline::null();
        self.pushed = false;
        tally(1);
        if spec.count.is_none() {
            if self.indirect.0.len() >= WINDOWS {
                self.indirect.0.remove(0);
            }
            self.indirect.0.push(Window {
                kind: spec.bound_kind(),
                stride: spec.stride,
                first: spec.src,
                records,
                at,
            });
        }
        Some((at.buffer, at.offset + HEAD, spec.count.map(|_| at.offset)))
    }

    fn draw_indirect(
        &mut self,
        b: &'static Backend,
        indexed: bool,
        (buffer, at, counted): (vk::Buffer, u64, Option<u64>),
        max: u32,
        size: u32,
    ) {
        let index = if indexed {
            let Some(found) = self
                .index
                .and_then(|i| buffer_over(b, i.address, u64::from(i.size)).map(|(buf, off)| (buf, off, i.format)))
            else {
                skipped(false);
                return;
            };
            Some(found)
        } else {
            None
        };
        let Some(cmd) = self.prepare_draw(b, false) else {
            return;
        };
        let d = &b.host.device;
        let count = counted.filter(|_| has_count_draws(b));
        // SAFETY: rendering open, pipeline bound; our scratch and a title
        // index buffer.
        unsafe {
            if let Some((buf, off, format)) = index {
                let ty = if format == 57 {
                    vk::IndexType::UINT16
                } else {
                    vk::IndexType::UINT32
                };
                d.cmd_bind_index_buffer(cmd, buf, off, ty);
            }
            match (indexed, count) {
                (true, Some(c)) => {
                    d.cmd_draw_indexed_indirect_count(cmd, buffer, at, buffer, c, max, size)
                }
                (false, Some(c)) => {
                    d.cmd_draw_indirect_count(cmd, buffer, at, buffer, c, max, size)
                }
                (true, None) => d.cmd_draw_indexed_indirect(cmd, buffer, at, max, size),
                (false, None) => d.cmd_draw_indirect(cmd, buffer, at, max, size),
            }
        }
        self.drawn(&format!("indirect {max} indexed {indexed}"));
    }

    /// Host groups per console group of the bound pipeline's mesh program
    /// (`VkShader::mesh_parts`, for direct and indirect mesh draws);
    /// `None`, counted as skipped, while it is not ready.
    pub(super) fn mesh_parts(&self, b: &'static Backend) -> Option<u32> {
        let Some((shaders, _)) = crate::pipeline::parts(self.pipeline) else {
            skipped(false);
            return None;
        };
        match super::link::stage_of(b, shaders, "mesh") {
            Ok(Some(s)) => Some(s.mesh_parts.max(1)),
            Ok(None) => {
                skipped(false);
                None
            }
            Err(waiting) => {
                skipped(waiting);
                None
            }
        }
    }

    fn mesh_indirect(
        &mut self,
        b: &'static Backend,
        buffer: vk::Buffer,
        at: u64,
        counted: Option<u64>,
        max: u32,
    ) {
        let Some(m) = mesh(b) else {
            skipped(false);
            return;
        };
        let Some(cmd) = self.prepare_draw(b, true) else {
            return;
        };
        // SAFETY: rendering open, a mesh pipeline bound; our scratch.
        unsafe {
            match counted.filter(|_| has_count_draws(b)) {
                Some(c) => {
                    m.cmd_draw_mesh_tasks_indirect_count(cmd, buffer, at, buffer, c, max, 12)
                }
                None => m.cmd_draw_mesh_tasks_indirect(cmd, buffer, at, max, 12),
            }
        }
        self.drawn(&format!("mesh indirect {max}"));
    }

    /// One `vkCmdDispatchIndirect` a record (records past the count were
    /// bounded to no groups).
    fn dispatch_indirect(&mut self, b: &'static Backend, buffer: vk::Buffer, at: u64, max: u32) {
        let Some(cmd) = self.prepare_dispatch(b) else {
            return;
        };
        // SAFETY: outside rendering, in our command buffer.
        let watched = if super::super::watch::on() { unsafe { super::super::watch::read(b, cmd, None) } } else { Vec::new() };
        for k in 0..u64::from(max) {
            // SAFETY: outside rendering, the title's compute pipeline bound;
            // our scratch.
            unsafe {
                b.host
                    .device
                    .cmd_dispatch_indirect(cmd, buffer, at + 12 * k)
            };
        }
        self.watched_dispatch(b, cmd, &watched, "dispatch indirect");
        self.ambient(b, cmd);
        super::STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
