//! The console's own list methods past the PC's (`list/console.rs`),
//! recorded as plain Vulkan:
//!
//! - UAV clears: `vkCmdClearColorImage` of the view's mip and layers (from
//!   the T# at the handle), `vkCmdFillBuffer` of a buffer view's range
//!   (from the V#).
//! - Memory fills: `vkCmdFillBuffer` (one repeated dword), else the pattern
//!   by `vkCmdUpdateBuffer` and doubling copies, into title memory through
//!   its buffer (`buffer_of`).
//! - `DrawIndexedX`: an indexed draw from the given indices; the bound index
//!   buffer stays.
//! - Predication: conditional rendering (`VK_EXT_conditional_rendering`) on
//!   the predicate's low dword, copied when it is set.
//! - Top-of-pipe and bottom-of-shader writes: done by the card
//!   (`vkCmdUpdateBuffer`, timestamps by `vkCmdWriteTimestamp2`) into title
//!   memory; on the CPU when the list has run if the card cannot reach it.
//! - `KickoffX`: the command buffer so far goes to the list's part.
//! - `RSSetDepthBiasX`: the depth bias of the draws after it.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::super::{Backend, images, lock};
use super::Rec;
use crate::draw::IndexView;
use crate::list::Rect;
use crate::object::This;
use crate::queue::Written;

/// What the console methods keep between calls of a recording.
#[derive(Default)]
pub(crate) struct State {
    /// `RSSetDepthBiasX`: bias, clamp, slope for the draws after it.
    pub(super) depth_bias: Option<[f32; 3]>,
    /// Conditional rendering is on in the command buffer.
    predicated: bool,
    /// The active occlusion query: its heap, index and pool (`query.rs`).
    pub(super) query: Option<(usize, u32, vk::QueryPool)>,
}

/// When a value the card writes is written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ValueWrite {
    /// When the command is reached (`Write*BitValueTopOfPipeX`).
    Top = 0,
    /// The card's clock, when reached.
    TopTimestamp = 1,
    /// The CPU's clock (left to the CPU).
    TopSystemTimestamp = 2,
    /// When the shaders before it are done (`Write32BitValueBottomOfShaderX`).
    BottomOfShader = 3,
}

impl ValueWrite {
    pub(crate) fn from_word(w: u64) -> ValueWrite {
        match w {
            1 => ValueWrite::TopTimestamp,
            2 => ValueWrite::TopSystemTimestamp,
            3 => ValueWrite::BottomOfShader,
            _ => ValueWrite::Top,
        }
    }
}

/// A shortcut taken, logged once by name.
fn once(what: &'static str) {
    static SEEN: Mutex<Vec<&str>> = Mutex::new(Vec::new());
    let mut seen = lock(&SEEN);
    if !seen.contains(&what) {
        seen.push(what);
        tracing::info!(what, "Vulkan: a console list method done approximately");
    }
}

/// Predicates copied at `SetPredicationBufferX`, a dword each, used in turn
/// (far more than the command buffers in flight use).
const PREDICATES: u64 = 4096;

fn predicates(b: &Backend) -> Option<(vk::Buffer, u64)> {
    static RING: Mutex<Option<(Option<Buffer>, u64)>> = Mutex::new(None);
    let mut ring = lock(&RING);
    let (buffer, next) = ring.get_or_insert_with(|| {
        let usage = vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT | vk::BufferUsageFlags::TRANSFER_DST;
        (Buffer::new(&b.host, PREDICATES * 4, usage, Place::Card).ok(), 0)
    });
    let at = *next % PREDICATES * 4;
    *next += 1;
    buffer.as_ref().map(|buf| (buf.buffer, at))
}

fn conditional(b: &Backend) -> Option<&'static victory_vk::ash::ext::conditional_rendering::Device> {
    static C: std::sync::OnceLock<Option<victory_vk::ash::ext::conditional_rendering::Device>> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        b.host
            .caps
            .conditional_rendering
            .then(|| victory_vk::ash::ext::conditional_rendering::Device::new(&b.host.instance, &b.host.device))
    })
    .as_ref()
}

impl Rec {
    /// The card buffer and offset of title memory the CPU also sees there
    /// (an import), for the card's writes the CPU may read; `None` for the
    /// pool's GPU view (card pages) and memory not imported.
    fn shared_buffer(b: &'static Backend, at: u64) -> Option<(vk::Buffer, u64)> {
        if super::super::memory(b).pools.iter().any(|p| p.contains(at)) {
            return None;
        }
        super::super::buffer_of(b, at)
    }

    /// Ends what the console methods have open in the command buffer: at
    /// its end (predication does not carry into the next one).
    pub(super) fn console_cut(&mut self, b: &Backend, cmd: vk::CommandBuffer) {
        if std::mem::take(&mut self.console.predicated)
            && let Some(c) = conditional(b)
        {
            once("predication ends at a cut of the list (a memory wait or KickoffX)");
            // SAFETY: outside rendering, conditional rendering active.
            unsafe { (c.fp().cmd_end_conditional_rendering_ext)(cmd) };
        }
        self.queries_cut(b, cmd);
    }

    /// `ClearUnorderedAccessView{Uint,Float}`: the view's mip and layers of
    /// a texture, or the view's range of a buffer, as the descriptor at the
    /// handle (`words`) gives them.
    pub(crate) fn clear_unordered(&mut self, b: &'static Backend, resource: This, words: [u32; 8], values: [u32; 4], float: bool, rects: &[Rect]) {
        if let Some((data, len)) = crate::resource::buffer_bytes(resource) {
            // V#: base (48 bits), stride, records (bytes with stride 0).
            let base = u64::from(words[0]) | u64::from(words[1] & 0xffff) << 32;
            let stride = u64::from(words[1] >> 16 & 0x3fff);
            let bytes = u64::from(words[2]) * stride.max(1);
            let (at, bytes) = if base != 0 && bytes != 0 { (base, bytes) } else { (data as u64, len) };
            // (A float clear's value comes as its bits: what a raw or
            // 32-bit view holds.)
            self.fill(b, at, bytes & !3, crate::list::pattern(&values[0].to_le_bytes()), 4);
            return;
        }
        let Some(image) = images::of_resource(b, resource) else { return };
        if image.depth() {
            return;
        }
        if !rects.is_empty() {
            once("UAV clear of rectangles: the whole view is cleared");
        }
        // T#: base level (word 3 bits 12-15); last and first layer (word 4).
        let described = words[6] == words[7] ^ victory_vk::gfx10::T_CHECK;
        let (mip, first, last) = if described { (words[3] >> 12 & 0xf, words[4] >> 16 & 0x1fff, words[4] & 0x1fff) } else { (0, 0, image.layers - 1) };
        let mip = mip.min(image.mips - 1);
        let first = first.min(image.layers - 1);
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: mip,
            level_count: 1,
            base_array_layer: first,
            layer_count: last.clamp(first, image.layers - 1) - first + 1,
        };
        // A float clear of integers converts; else the bits are the value.
        let bits = if float && victory_vk::gfx10::is_integer(image.format) { values.map(|v| f32::from_bits(v) as u32) } else { values };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: outside rendering; a live image in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_clear_color_image(cmd, image.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { uint32: bits }, &[range]);
            images::full_barrier(&b.host.device, cmd);
        }
        image.written[1].fetch_add(1, Ordering::Relaxed);
        self.keep.push(image);
    }

    /// `FillMemoryWith{32,64,128}BitValueX`: `size` bytes at title address
    /// `at`, the 16-byte `pattern` repeating every `width` bytes.
    pub(crate) fn fill(&mut self, b: &'static Backend, at: u64, size: u64, pattern: [u8; 16], width: usize) {
        if size == 0 {
            return;
        }
        let Some((buffer, offset)) = super::super::buffer_of(b, at) else {
            once("a fill of title memory the card does not reach: not done");
            return;
        };
        if offset % 4 != 0 || size % 4 != 0 {
            once("a fill not in whole dwords: not done");
            return;
        }
        let words: [u32; 4] = std::array::from_fn(|k| u32::from_le_bytes(std::array::from_fn(|i| pattern[k * 4 + i])));
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        let d = &b.host.device;
        // SAFETY: outside rendering; title memory through its buffer.
        unsafe {
            images::full_barrier(d, cmd);
            if words[..width.clamp(4, 16) / 4].iter().all(|&w| w == words[0]) {
                d.cmd_fill_buffer(cmd, buffer, offset, size, words[0]);
            } else {
                // The pattern into the first 64 KiB (or all), then copied
                // after itself, doubling.
                let first = size.min(65536 / 16 * 16) as usize;
                let data: Vec<u8> = (0..first).map(|k| pattern[k % 16]).collect();
                d.cmd_update_buffer(cmd, buffer, offset, &data);
                let mut done = first as u64;
                while done < size {
                    let n = done.min(size - done);
                    images::full_barrier(d, cmd);
                    d.cmd_copy_buffer(cmd, buffer, buffer, &[vk::BufferCopy { src_offset: offset, dst_offset: offset + done, size: n }]);
                    done += n;
                }
            }
            images::full_barrier(d, cmd);
        }
    }

    /// `DrawIndexedX`: `count` indices of `format` read at `at`, one instance.
    pub(crate) fn draw_indexed_x(&mut self, b: &'static Backend, at: u64, format: u32, count: u32) {
        const R16_UINT: u32 = 57;
        const R32_UINT: u32 = 42;
        let stride = match format {
            R16_UINT => 2,
            R32_UINT => 4,
            _ => {
                once("DrawIndexedX with indices of another format: not drawn");
                return;
            }
        };
        let bound = self.index;
        self.index = Some(IndexView { address: at, size: count.saturating_mul(stride), format });
        self.touched = true;
        self.draw_indexed(b, count, 1, 0, 0, 0);
        self.index = bound;
    }

    /// `SetPredicationBufferX`: draws and dispatches after it run only if
    /// the predicate is not zero (operation 0) or is zero (1); `None`: off.
    pub(crate) fn predication(&mut self, b: &'static Backend, predicate: Option<(u64, u32)>) {
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        let Some(c) = conditional(b) else {
            if predicate.is_some() {
                once("predication without VK_EXT_conditional_rendering: predicated work always runs");
            }
            return;
        };
        if std::mem::take(&mut self.console.predicated) {
            // SAFETY: outside rendering, conditional rendering active.
            unsafe { (c.fp().cmd_end_conditional_rendering_ext)(cmd) };
        }
        let Some((at, op)) = predicate else { return };
        let (Some((src, offset)), Some((ring, slot))) = (super::super::buffer_of(b, at), predicates(b)) else {
            once("a predicate in memory the card does not reach: predicated work always runs");
            return;
        };
        // shortcut: the predicate's low dword (Vulkan reads 32 bits),
        // read when it is set, not at each draw.
        let flags = if op == 1 { vk::ConditionalRenderingFlagsEXT::INVERTED } else { vk::ConditionalRenderingFlagsEXT::empty() };
        // SAFETY: outside rendering; our ring and a title buffer.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_buffer(cmd, src, ring, &[vk::BufferCopy { src_offset: offset, dst_offset: slot, size: 4 }]);
            images::full_barrier(&b.host.device, cmd);
            let info = vk::ConditionalRenderingBeginInfoEXT::default().buffer(ring).offset(slot).flags(flags);
            (c.fp().cmd_begin_conditional_rendering_ext)(cmd, &info);
        }
        self.console.predicated = true;
    }

    /// A top-of-pipe or bottom-of-shader write of the low `bytes` of
    /// `value` at title address `at`, by the card.
    pub(crate) fn write_value(&mut self, b: &'static Backend, list: This, at: u64, value: u64, bytes: usize, when: ValueWrite) {
        let target = Self::shared_buffer(b, at).filter(|_| at % 4 == 0 && when != ValueWrite::TopSystemTimestamp);
        let Some((buffer, offset)) = target else {
            // On the CPU when the list has run.
            let what = match when {
                ValueWrite::TopTimestamp => Written::GpuTimestamp,
                ValueWrite::TopSystemTimestamp => Written::SystemTimestamp,
                _ => Written::Value(value),
            };
            super::super::list_write(list, at, what, bytes);
            return;
        };
        if when == ValueWrite::TopTimestamp {
            self.timestamp_into(b, buffer, offset);
            return;
        }
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        let d = &b.host.device;
        // SAFETY: outside rendering; title memory through its buffer.
        unsafe {
            if when == ValueWrite::BottomOfShader {
                images::full_barrier(d, cmd);
            }
            d.cmd_update_buffer(cmd, buffer, offset, &value.to_le_bytes()[..bytes.clamp(4, 8) / 4 * 4]);
            images::full_barrier(d, cmd);
        }
    }

    /// `KickoffX`: what is recorded so far goes to the list's part (it runs
    /// at the list's submission, behind nothing new).
    pub(crate) fn kickoff(&mut self, list: This, b: &Backend) {
        self.cut(list, b);
    }

    /// `RSSetDepthBiasX`: the depth bias of the draws after it.
    pub(crate) fn depth_bias(&mut self, bias: [f32; 3]) {
        self.console.depth_bias = Some(bias.map(|v| if v.is_finite() { v } else { 0.0 }));
    }
}
