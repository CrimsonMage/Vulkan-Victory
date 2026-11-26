//! Diagnostics (`VICTORY_VK_DUMP_TARGETS=<frame>`): at the first three
//! captured frames from that one on, the
//! images the card writes (render targets, depth, compute outputs) are
//! read back and saved beside the frame, so a wrong final picture can be
//! followed back to the pass that lost it. The most drawn images first,
//! level 0 and layer 0 of each, at half size like the frame itself.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::images::{self, Image};
use super::{Backend, lock};

/// Images a dump holds at most, and its bytes.
const MOST: usize = 64;
const ROOM: u64 = 768 << 20;

static BUFFER: Mutex<Option<Buffer>> = Mutex::new(None);

pub(super) struct Plan {
    buffer: vk::Buffer,
    mapped: usize,
    items: Vec<(Arc<Image>, u64)>,
}

/// Dumps a run makes at most.
const DUMPS: u64 = 3;
static DONE: AtomicU64 = AtomicU64::new(0);

/// The first frame dumped (`VICTORY_VK_DUMP_TARGETS=<frame>`), if any.
fn from() -> Option<u64> {
    static FROM: OnceLock<Option<u64>> = OnceLock::new();
    *FROM.get_or_init(|| {
        let v = std::env::var("VICTORY_VK_DUMP_TARGETS").ok()?;
        cfg!(feature = "diagnostics").then(|| v.parse().unwrap_or(0))
    })
}

/// Bytes a texel of the copy takes (a depth image: its depth aspect).
fn texel(img: &Image) -> u64 {
    match img.format {
        vk::Format::D16_UNORM => 2,
        _ if img.depth() => 4,
        f => u64::from(images::format_bytes(f)),
    }
}

/// What a dump at this frame reads back, if dumps are on.
pub(super) fn plan(b: &Backend, frame: u64) -> Option<Plan> {
    if from().is_none_or(|f| frame < f) || DONE.fetch_add(1, Ordering::Relaxed) >= DUMPS {
        return None;
    }
    let mut buffer = lock(&BUFFER);
    if buffer.is_none() {
        *buffer = Buffer::new(&b.host, ROOM, vk::BufferUsageFlags::TRANSFER_DST, Place::Readback).ok();
    }
    let buf = buffer.as_ref()?;
    let (mut items, mut at) = (Vec::new(), 0u64);
    for img in images::card_written_images() {
        let bytes = u64::from(img.extent.width) * u64::from(img.extent.height) * texel(&img);
        if img.samples > 1 || img.extent.width < 128 || texel(&img) == 0 || at + bytes > ROOM {
            continue;
        }
        items.push((img, at));
        at += bytes.next_multiple_of(16);
        if items.len() == MOST {
            break;
        }
    }
    Some(Plan { buffer: buf.buffer, mapped: buf.mapped as usize, items })
}

/// # Safety
/// Recording into `cmd`, outside rendering; the images are in GENERAL.
pub(super) unsafe fn record(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, p: &Plan) {
    for (img, at) in &p.items {
        let aspect = if img.depth() { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
        let region = vk::BufferImageCopy::default()
            .buffer_offset(*at)
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width: img.extent.width, height: img.extent.height, depth: 1 });
        // SAFETY: as the caller says.
        unsafe { d.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, p.buffer, &[region]) };
    }
}

/// Saves what [`record`] copied (the queue is idle).
pub(super) fn save(frame: u64, p: &Plan) {
    for (n, (img, at)) in p.items.iter().enumerate() {
        let (w, h, t) = (img.extent.width as usize, img.extent.height as usize, texel(img) as usize);
        // SAFETY: the mapped readback buffer; the copy into it has finished.
        let src = unsafe { std::slice::from_raw_parts((p.mapped as *const u8).add(*at as usize), w * h * t) };
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in src.chunks_exact(t) {
            rgba.extend(shown(img, px));
        }
        let name = format!(
            "frame-{frame}-t{n:02}-{:?}-{w}x{h}-{:x}-draws{}-clears{}",
            img.format,
            img.key.address,
            img.written[0].load(Ordering::Relaxed),
            img.written[1].load(Ordering::Relaxed)
        );
        crate::capture::save_named(&name, w as u32, h as u32, w * 4, false, &rgba);
    }
    tracing::info!(frame, images = p.items.len(), "Vulkan: the card-written images dumped beside the frame");
}

fn half(x: u16) -> f32 {
    let (e, m) = (i32::from(x >> 10 & 0x1f), f32::from(x & 0x3ff));
    let v = match e {
        0 => m / 1024.0 * 2f32.powi(-14),
        31 => f32::MAX,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if x & 0x8000 != 0 { -v } else { v }
}

/// An unsigned float of `bits` mantissa bits and a 5-bit exponent above.
fn ufloat(x: u32, bits: u32) -> f32 {
    let (e, m) = ((x >> bits) as i32, (x & ((1 << bits) - 1)) as f32 / (1u32 << bits) as f32);
    match e {
        0 => m * 2f32.powi(-14),
        31 => f32::MAX,
        _ => (1.0 + m) * 2f32.powi(e - 15),
    }
}

/// A linear value as a byte: compressed so highlights stay in range.
fn tone(v: f32) -> u8 {
    let v = v.abs();
    ((v / (1.0 + v)).sqrt() * 255.0) as u8
}

fn unit(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0) as u8
}

/// One texel as RGBA bytes one can look at.
fn shown(img: &Image, p: &[u8]) -> [u8; 4] {
    use vk::Format as F;
    let u16_at = |i: usize| u16::from_le_bytes([p[2 * i], p[2 * i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]]);
    match img.format {
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => [p[2], p[1], p[0], 255],
        F::A2B10G10R10_UNORM_PACK32 => {
            let x = u32_at(0);
            [(x >> 2) as u8, (x >> 12) as u8, (x >> 22) as u8, 255]
        }
        F::B10G11R11_UFLOAT_PACK32 => {
            let x = u32_at(0);
            [tone(ufloat(x & 0x7ff, 6)), tone(ufloat(x >> 11 & 0x7ff, 6)), tone(ufloat(x >> 22, 5)), 255]
        }
        F::R16G16B16A16_SFLOAT => [tone(half(u16_at(0))), tone(half(u16_at(1))), tone(half(u16_at(2))), 255],
        F::R16G16_SFLOAT => [tone(half(u16_at(0))), tone(half(u16_at(1))), 0, 255],
        F::R16_SFLOAT => [tone(half(u16_at(0))); 4],
        F::R16_UNORM | F::D16_UNORM => [p[1]; 4],
        F::R16G16_UNORM | F::R16G16_SNORM => [p[1], p[3], 0, 255],
        F::R16G16B16A16_UNORM => [p[1], p[3], p[5], 255],
        F::R32_SFLOAT => [tone(f32::from_bits(u32_at(0))); 4],
        F::R32G32_SFLOAT => [tone(f32::from_bits(u32_at(0))), tone(f32::from_bits(u32_at(1))), 0, 255],
        F::R32G32B32A32_SFLOAT => [tone(f32::from_bits(u32_at(0))), tone(f32::from_bits(u32_at(1))), tone(f32::from_bits(u32_at(2))), 255],
        // depth: the fourth root spreads a reversed or a plain depth
        _ if img.depth() => [unit(f32::from_bits(u32_at(0)).sqrt().sqrt()); 4],
        _ => match p.len() {
            1 => [p[0]; 4],
            2 => [p[0], p[1], 0, 255],
            3 => [p[0], p[1], p[2], 255],
            // whatever it is, its low bytes: enough to see whether it holds anything
            _ => [p[0], p[1], p[2], 255],
        },
    }
}
