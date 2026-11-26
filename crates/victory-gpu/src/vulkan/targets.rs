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
const ROOM: u64 = 512 << 20;

pub(super) struct Plan {
    /// Its own readback buffer, destroyed after the save.
    held: Buffer,
    buffer: vk::Buffer,
    mapped: usize,
    items: Vec<(Arc<Image>, u64)>,
    /// What it is named after: "" at a present, else the program it was
    /// taken before.
    label: String,
    /// The frame it was recorded in.
    born: u64,
}

// SAFETY: the mapped pointer is read only at the save, under `PENDING`'s lock or by the presenter.
unsafe impl Send for Plan {}

/// Dumps taken in the middle of a frame, saved at the next present.
static PENDING: Mutex<Vec<Plan>> = Mutex::new(Vec::new());

/// `VICTORY_VK_DUMP_AT=<program>[,<program>...]`: the first time each of
/// these programs is used in a frame the dump covers, the images are
/// read back just before it (what the pass is about to read).
fn at_names() -> &'static Mutex<Vec<String>> {
    static AT: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    AT.get_or_init(|| Mutex::new(std::env::var("VICTORY_VK_DUMP_AT").map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default()))
}

/// Before a draw or dispatch of program `name`: if a dump was asked for
/// there, it is recorded now (the caller is outside rendering when this
/// returns true: it ends its pass first when `wants` says so).
pub(super) fn wants(name: &str) -> bool {
    if !cfg!(feature = "diagnostics") {
        return false;
    }
    let Some(f) = from() else { return false };
    let frame = super::present::FRAME.load(Ordering::Relaxed);
    if frame == f {
        let mut order = lock(&ORDER);
        if !order.iter().any(|n| n == name) {
            order.push(name.to_string());
        }
    }
    frame >= f && lock(at_names()).iter().any(|n| n == name)
}

/// The programs of the first dumped frame in the order they were first
/// used (told with the dump: which names `VICTORY_VK_DUMP_AT` can take).
static ORDER: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// # Safety
/// Recording into `cmd`, outside rendering.
pub(super) unsafe fn before(b: &Backend, cmd: vk::CommandBuffer, name: &str) {
    lock(at_names()).retain(|n| n != name);
    let Some(mut p) = make(b) else { return };
    p.label = format!("before-{name}-");
    // SAFETY: as the caller says.
    unsafe {
        images::full_barrier(&b.host.device, cmd);
        record(&b.host.device, cmd, &p);
        images::full_barrier(&b.host.device, cmd);
    }
    lock(&PENDING).push(p);
}

/// Whether a mid-frame dump of an earlier frame waits to be saved (its
/// list has been executed by now).
pub(super) fn pending(frame: u64) -> bool {
    lock(&PENDING).iter().any(|p| p.born < frame)
}

/// At a present with the queue idle: the mid-frame dumps of earlier
/// frames are saved.
pub(super) fn save_pending(b: &Backend, frame: u64) {
    let (ready, later): (Vec<Plan>, Vec<Plan>) = std::mem::take(&mut *lock(&PENDING)).into_iter().partition(|p| p.born < frame);
    lock(&PENDING).extend(later);
    for p in ready {
        let born = p.born;
        save(b, born, p);
    }
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
    make(b)
}

fn make(b: &Backend) -> Option<Plan> {
    let buf = Buffer::new(&b.host, ROOM, vk::BufferUsageFlags::TRANSFER_DST, Place::Readback).ok()?;
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
    Some(Plan { buffer: buf.buffer, mapped: buf.mapped as usize, held: buf, items, label: String::new(), born: super::present::FRAME.load(Ordering::Relaxed) })
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

/// Saves what [`record`] copied and frees the dump (the queue is idle).
pub(super) fn save(b: &Backend, frame: u64, p: Plan) {
    for (n, (img, at)) in p.items.iter().enumerate() {
        let (w, h, t) = (img.extent.width as usize, img.extent.height as usize, texel(img) as usize);
        // SAFETY: the mapped readback buffer; the copy into it has finished.
        let src = unsafe { std::slice::from_raw_parts((p.mapped as *const u8).add(*at as usize), w * h * t) };
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in src.chunks_exact(t) {
            rgba.extend(shown(img, px));
        }
        let name = format!(
            "frame-{frame}-{}t{n:02}-{:?}-{w}x{h}-{:x}-draws{}-clears{}",
            p.label,
            img.format,
            img.key.address,
            img.written[0].load(Ordering::Relaxed),
            img.written[1].load(Ordering::Relaxed)
        );
        crate::capture::save_named(&name, w as u32, h as u32, w * 4, false, &rgba);
    }
    for (img, at) in p.items.iter().filter(|(i, _)| i.depth() && texel(i) == 4) {
        let n = img.extent.width as usize * img.extent.height as usize;
        // SAFETY: as above.
        let src = unsafe { std::slice::from_raw_parts((p.mapped as *const u8).add(*at as usize), n * 4) };
        let (mut lo, mut hi, mut some) = (f32::MAX, f32::MIN, 0u64);
        for c in src.chunks_exact(4) {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            (lo, hi) = (lo.min(v), hi.max(v));
            some += u64::from(v != 0.0);
        }
        tracing::info!(frame, at = format_args!("{:#x}", img.key.address), width = img.extent.width, height = img.extent.height, min = lo, max = hi, not_zero = some, "Vulkan: a depth image's values at the dump");
    }
    for (n, (img, at)) in p.items.iter().enumerate().filter(|(_, (i, _))| i.format == vk::Format::R16G16B16A16_SFLOAT && i.extent.width >= 480) {
        let count = img.extent.width as usize * img.extent.height as usize;
        // SAFETY: as above.
        let src = unsafe { std::slice::from_raw_parts((p.mapped as *const u8).add(*at as usize), count * 8) };
        let (mut lo, mut hi, mut odd) = ([f32::MAX; 4], [f32::MIN; 4], [0u64; 4]);
        for px in src.chunks_exact(8) {
            for c in 0..4 {
                let bits = u16::from_le_bytes([px[2 * c], px[2 * c + 1]]);
                if bits & 0x7c00 == 0x7c00 {
                    odd[c] += 1;
                } else {
                    let v = half(bits);
                    (lo[c], hi[c]) = (lo[c].min(v), hi[c].max(v));
                }
            }
        }
        let mid = &src[(count / 2 + img.extent.width as usize / 2) * 8..][..8];
        tracing::info!(
            frame,
            label = %p.label,
            n,
            at = format_args!("{:#x}", img.key.address),
            width = img.extent.width,
            min = ?lo,
            max = ?hi,
            inf_or_nan = ?odd,
            centre = %mid.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "Vulkan: a half-float image's channels at the dump"
        );
    }
    tell_depth_uses();
    let order = std::mem::take(&mut *lock(&ORDER));
    for (n, part) in order.chunks(40).enumerate() {
        tracing::info!(part = n, programs = %part.join(" "), "Vulkan: the programs of the first dumped frame, in the order of their first use");
    }
    tracing::info!(frame, images = p.items.len(), label = %p.label, "Vulkan: the card-written images dumped beside the frame");
    // SAFETY: the queue is idle and nothing else names the buffer.
    unsafe { p.held.destroy(&b.host) };
}

/// Draws by depth target and the state that decides depth (the dump tells
/// the commonest): a description -> draws.
static DEPTH_USES: Mutex<Option<std::collections::HashMap<String, u64>>> = Mutex::new(None);

/// A draw into depth image `at` with this state (counted while dumps are
/// asked for).
pub(super) fn depth_use(at: u64, state: impl FnOnce() -> String) {
    if from().is_none() {
        return;
    }
    *lock(&DEPTH_USES).get_or_insert_with(Default::default).entry(format!("{at:#x} {}", state())).or_insert(0) += 1;
}

fn tell_depth_uses() {
    let mut uses: Vec<(String, u64)> = lock(&DEPTH_USES).as_ref().map(|m| m.iter().map(|(k, &v)| (k.clone(), v)).collect()).unwrap_or_default();
    uses.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (state, draws) in uses.into_iter().take(32) {
        tracing::info!(state, draws, "Vulkan: draws by depth target and depth state");
    }
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
