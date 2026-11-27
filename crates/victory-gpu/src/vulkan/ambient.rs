//! A stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder off (`VICTORY_BVH_BUILDER` unset)
//! every ray of the title misses. Its program that resolves the diffuse
//! light of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) then
//! leaves nothing indoors and, in blocks of tiles, what the probes' memory
//! held: walls the sun does not reach are black, and all the colour a frame
//! has is the sun's (amber at a low sun). Right after that program the
//! light it wrote (shared-exponent words in a 32_UINT image the tiled
//! lighting samples) is raised to an estimate.
//!
//! The estimate is worked out from the frame, so it follows the hour, the
//! room and its lamps by itself: the light the visible surfaces send out
//! (the title keeps it for its own screen traces:
//! `CS_CacheScreenSpaceRadiance`, the frame before, half size) is averaged
//! in 32x18 cells, and each of 160x90 points gets the mean of the cells
//! weighted by the share of its surroundings they cover, with a direction
//! per colour; a texel takes the points around it at its own depth, along
//! its normal (from the depth), shaded in creases. One host compute
//! program in three dispatches (`ambient.comp`), one buffer of 0.7 MB, no
//! image of its own: depth, albedo and the light of the frame before are
//! the title's, named in the resolve's own tables.
//!
//! It is one bounce of the light the title's own lighting gives: the
//! stand-in's light of the frame before, as the lighting added it (about
//! 0.85 x albedo x light: run am1, a wall), is taken out of the kept light,
//! so its own light does not come back to it and no share runs away.
//!
//! What it assumes: what is not on screen sends out what is (a fifth of
//! the light is the slow mean of the whole picture, so that a wall whose
//! light comes from behind the camera is not black); a field of view of 48
//! degrees; no occlusion but the creases'.
//!
//! `VICTORY_AMBIENT`:
//! - unset: the estimate while the structure builder is off; nothing
//!   with the builder on (the rays find the light).
//! - a number: the estimate times it ([`SHARE`] if unset). 1 is the mean
//!   of the one bounce around; more stands for the further bounces and
//!   for what lights the room from where the picture does not show (2.5:
//!   the walls of the opening shot, lit from behind the camera, come near
//!   the console's; the one scene it was set on).
//! - `constant:<level>[,<red>,<green>,<blue>]`: the same light everywhere
//!   (the image cleared to `level` times the tint, 1 / 0.98 / 0.94 if none
//!   is given): flat, the same indoors, outdoors and at night.
//! - `0` or `off`: the title's own output.
//!
//! That a stand-in can be put there is beloooo's find
//! (VIctory-Belobug-AMD, GPL-3.0-or-later); the estimate is
//! VIctory-NVIDIA's (`crates/victory-gpu/src/ambient.rs`), its method taken
//! as a Vulkan compute pass.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::images::{self, Image, ViewKey};
use super::{Backend, lock, read_title};

mod words;

/// Where the resolve's tables hold the T#s of the diffuse light it writes,
/// of that of the frame before (its history), of the scene's depth and of
/// its albedo, and where the program that keeps the light of visible
/// surfaces holds the T# of what it writes: root parameter and dword.
const LIGHT: (usize, u64) = (1, 176);
const BEFORE: (usize, u64) = (1, 72);
const DEPTH: (usize, u64) = (0, 688);
const ALBEDO: (usize, u64) = (0, 560);
const RADIANCE: (usize, u64) = (1, 16);
/// The share of albedo x light the title's lighting adds.
const BACK: f32 = 0.85;
/// What the estimate is multiplied by when no number is given.
const SHARE: f32 = 2.5;
/// A T#'s check word (`docs/vulkan-abi.md` 5.2).
const CHECK: u32 = 0x7A3C_5E91;
/// The estimate's cells and points (`ambient.comp`), and its buffer: a
/// light and a place per cell, the slow mean, three values a point.
const CELLS: [u32; 2] = [32, 18];
const POINTS: [u32; 2] = [160, 90];
const KEPT_BYTES: u64 = (2 * (CELLS[0] * CELLS[1]) as u64 + 1 + 3 * (POINTS[0] * POINTS[1]) as u64) * 16;

/// The title's programs the stand-in follows.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Role {
    /// Resolves the diffuse light.
    Resolve,
    /// Keeps the light visible surfaces send out.
    Radiance,
}

/// The part the program `name` has in this.
pub(super) fn role(name: &str) -> Option<Role> {
    if name.starts_with("CS_ScreenProbes_TemporalResolve") {
        Some(Role::Resolve)
    } else if name.starts_with("CS_CacheScreenSpaceRadiance") {
        Some(Role::Radiance)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant([f32; 3]),
}

fn parse(set: &str) -> Set {
    let set = set.trim();
    let number = |t: &str| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0);
    let Some(constant) = set.strip_prefix("constant:") else {
        return match set {
            "" => Set::Estimate(SHARE),
            _ => number(set).filter(|v| *v > 0.0).map_or(Set::Off, Set::Estimate),
        };
    };
    let mut n = constant.split(',').map(number);
    let Some(level) = n.next().flatten().filter(|v| *v > 0.0) else { return Set::Off };
    let tint = match (n.next().flatten(), n.next().flatten(), n.next().flatten()) {
        (Some(r), Some(g), Some(b)) => [r, g, b],
        _ => [1.0, 0.98, 0.94],
    };
    Set::Constant(tint.map(|t| t * level))
}

fn set() -> Set {
    static SET: OnceLock<Set> = OnceLock::new();
    *SET.get_or_init(|| {
        if super::record::bvh_builder() {
            return Set::Off;
        }
        parse(&std::env::var("VICTORY_AMBIENT").unwrap_or_default())
    })
}

/// Whether the stand-in is on.
pub(super) fn on() -> bool {
    set() != Set::Off
}

/// A colour as a shared-exponent word (R9G9B9E5).
fn pack(c: [f32; 3]) -> u32 {
    let c = c.map(|v| v.clamp(0.0, 65408.0));
    let top = c[0].max(c[1]).max(c[2]);
    let mut e = (top.max(1e-9).log2().floor() as i32 + 16).clamp(0, 31);
    let mut scale = 2f32.powi(24 - e);
    if top * scale >= 511.5 {
        e = (e + 1).min(31);
        scale = 2f32.powi(24 - e);
    }
    let m = c.map(|v| ((v * scale + 0.5) as u32).min(511));
    m[0] | m[1] << 9 | m[2] << 18 | (e as u32) << 27
}

/// The image slot the T# at dword `at.1` of the table of root parameter
/// `at.0` names.
fn named(b: &Backend, roots: &[Vec<u32>], at: (usize, u64)) -> Option<u32> {
    let &table = roots.get(at.0).and_then(|r| r.first())?;
    let bytes = read_title(b, u64::from(table) + at.1 * 4, 32).ok()?;
    let w = |i: usize| u32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
    (w(6) ^ w(7) == CHECK).then(|| w(7) & 0xff_ffff)
}

/// The first thing that keeps the stand-in from its work is told.
fn warn_once(why: impl FnOnce() -> String) {
    static TOLD: AtomicBool = AtomicBool::new(false);
    if !TOLD.swap(true, Ordering::Relaxed) {
        tracing::warn!("Vulkan: the ambient stand-in {}; the title's own light stays", why());
    }
}

/// What the estimate takes of an image the title names.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Use {
    /// The depth plane, sampled.
    Depth,
    /// Colour as floats, sampled (a storage view's slot holds the sampled
    /// view of its format too).
    Colour,
    /// A 32_UINT storage view of shared-exponent words.
    Words,
}

impl Use {
    fn takes(self, image: &Image, key: &ViewKey) -> bool {
        use vk::Format as F;
        key.kind == vk::ImageViewType::TYPE_2D
            && match self {
                Use::Depth => image.depth() && key.aspect == vk::ImageAspectFlags::DEPTH && !key.storage,
                Use::Colour => matches!(key.format, F::R8G8B8A8_SRGB | F::R8G8B8A8_UNORM | F::E5B9G9R9_UFLOAT_PACK32 | F::B10G11R11_UFLOAT_PACK32 | F::R16G16B16A16_SFLOAT),
                Use::Words => key.storage && key.format == F::R32_UINT && image.format == F::R32_UINT,
            }
    }
}

/// The title's image slot `slot` if the estimate can use it so, and the
/// image's size (looked up once per image: the slot names it as long as it
/// lives).
fn slot_for(slot: u32, used: Use) -> Option<(u32, [u32; 2])> {
    type Found = (u32, Use, Weak<Image>, [u32; 2]);
    static FOUND: Mutex<Vec<Found>> = Mutex::new(Vec::new());
    let mut found = lock(&FOUND);
    if let Some(&(.., size)) = found.iter().find(|(s, u, i, _)| *s == slot && *u == used && i.strong_count() > 0) {
        return Some((slot, size));
    }
    found.retain(|(.., i, _)| i.strong_count() > 0);
    let (image, key) = super::descriptors::slot_image(slot)?;
    if !used.takes(&image, &key) {
        warn_once(|| format!("cannot use the {:?} image at {:#x} ({:?} view, storage {}) as {used:?}", image.format, image.key.address, key.format, key.storage));
        return None;
    }
    let size = [image.extent.width, image.extent.height];
    tracing::info!(?used, address = format_args!("{:#x}", image.key.address), format = ?key.format, ?size, slot, "Vulkan: the ambient stand-in takes an image of the title's");
    found.push((slot, used, Arc::downgrade(&image), size));
    Some((slot, size))
}

struct Pass {
    pipeline: vk::Pipeline,
    kept: Buffer,
}

/// The estimate's program and buffer, made at its first use.
fn pass(b: &'static Backend) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| match make(b) {
        Ok(p) => Some(p),
        Err(e) => {
            tracing::error!("Vulkan: the ambient estimate is not made, the title's own light stays: {e:#}");
            None
        }
    })
    .as_ref()
}

fn make(b: &'static Backend) -> anyhow::Result<Pass> {
    let layout = super::stages::layout(b).ok_or_else(|| anyhow::anyhow!("no pipeline layout"))?;
    let kept = Buffer::new(&b.host, KEPT_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Place::Card)?;
    let buffer = kept.buffer;
    b.timeline.record(&b.host, |d, cmd| {
        // SAFETY: recording into the timeline's one-shot buffer.
        unsafe { d.cmd_fill_buffer(cmd, buffer, 0, vk::WHOLE_SIZE, 0) };
    })?;
    let d = &b.host.device;
    // SAFETY: valid create infos; the module goes once the pipeline exists.
    let pipeline = unsafe {
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words::PROGRAM), None)?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let flags = if b.host.caps.descriptor_buffer { vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT } else { vk::PipelineCreateFlags::empty() };
        let info = vk::ComputePipelineCreateInfo::default().flags(flags).stage(stage).layout(layout.pipeline);
        let made = d.create_compute_pipelines(super::stages::cache(b), &[info], None);
        d.destroy_shader_module(module, None);
        made.map_err(|(_, e)| e)?[0]
    };
    tracing::info!(bytes = KEPT_BYTES, "Vulkan: the ambient estimate's program and buffer are made");
    Ok(Pass { pipeline, kept })
}

/// The slot of the light of visible surfaces the title last kept (0: none yet).
static RADIANCE_SLOT: AtomicU32 = AtomicU32::new(0);

/// Right after a dispatch of the title's program of `role`, whose root
/// parameters are `roots`: the keeping program's image is noted; the
/// resolve's diffuse light is raised to the estimate, or is the constant.
/// True if the compute pipeline and the push constants are no longer the
/// title's.
///
/// # Safety
/// Recording into `cmd`, outside rendering, the descriptor buffers bound.
pub(super) unsafe fn after(b: &'static Backend, cmd: vk::CommandBuffer, role: Role, roots: &[Vec<u32>]) -> bool {
    if role == Role::Radiance {
        if let Some(slot) = named(b, roots, RADIANCE) {
            RADIANCE_SLOT.store(slot, Ordering::Relaxed);
        }
        return false;
    }
    let Some(light) = named(b, roots, LIGHT).and_then(|s| slot_for(s, Use::Words)) else {
        warn_once(|| "found no 32_UINT image where the resolve writes its light".into());
        return false;
    };
    match set() {
        Set::Off => false,
        Set::Constant(c) => {
            // SAFETY: as the caller says.
            unsafe { constant(b, cmd, light.0, c) };
            false
        }
        // SAFETY: as the caller says.
        Set::Estimate(level) => unsafe { estimate(b, cmd, light, level, roots) },
    }
}

/// The light is `c` everywhere.
///
/// # Safety
/// As [`after`].
unsafe fn constant(b: &Backend, cmd: vk::CommandBuffer, light: u32, c: [f32; 3]) {
    static TOLD: AtomicBool = AtomicBool::new(false);
    let Some((image, _)) = super::descriptors::slot_image(light) else { return };
    let word = pack(c);
    if !TOLD.swap(true, Ordering::Relaxed) {
        tracing::info!(address = format_args!("{:#x}", image.key.address), light = ?c, word = format_args!("{word:#x}"), "Vulkan: the ambient stand-in fills the resolve's diffuse light");
    }
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: image.layers.max(1) };
    // SAFETY: as the caller says; the image is kept by its descriptor slot.
    unsafe {
        images::full_barrier(&b.host.device, cmd);
        b.host.device.cmd_clear_color_image(cmd, image.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { uint32: [word, 0, 0, 0] }, &[range]);
        images::full_barrier(&b.host.device, cmd);
    }
}

/// The light (its slot and size) is raised to `level` times the estimate;
/// false (nothing recorded) until the title has kept a frame's light, or
/// if an image is not where or what it is taken to be (told once).
///
/// # Safety
/// As [`after`].
unsafe fn estimate(b: &'static Backend, cmd: vk::CommandBuffer, light: (u32, [u32; 2]), level: f32, roots: &[Vec<u32>]) -> bool {
    let kept = RADIANCE_SLOT.load(Ordering::Relaxed);
    let taken = |at, used| named(b, roots, at).and_then(|s| slot_for(s, used)).map(|(slot, _)| slot);
    let radiance = (kept != 0).then(|| slot_for(kept, Use::Colour)).flatten().map(|(slot, _)| slot);
    let (Some(depth), Some(albedo), Some(before), Some(radiance)) = (taken(DEPTH, Use::Depth), taken(ALBEDO, Use::Colour), taken(BEFORE, Use::Colour), radiance) else {
        return false;
    };
    let (Some(pass), Some(layout)) = (pass(b), super::stages::layout(b)) else { return false };
    // The buffer; the part; the five slots; the level; the creases'
    // strength and radius in metres; the near plane and the tangent of
    // half the field of view; how fast the slow mean follows; the most of
    // the title's own light kept, as a multiple of the estimate; the share
    // of albedo x light that comes back.
    let floats = [level, 1.0, 0.6, 0.1, 0.45, 0.04, 4.0, BACK];
    let mut push = [0u32; 16];
    push[0] = pass.kept.address as u32;
    push[1] = (pass.kept.address >> 32) as u32;
    push[3..8].copy_from_slice(&[depth, radiance, light.0, albedo, before]);
    for (to, v) in push[8..].iter_mut().zip(floats) {
        *to = v.to_bits();
    }
    let d = &b.host.device;
    // SAFETY: as the caller says; the title's programs have just used the
    // images (every image is in GENERAL), the buffer lives on.
    unsafe {
        super::gputime::mark(b, cmd, || "host: the ambient estimate".into());
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pass.pipeline);
        for (part, groups) in [CELLS, POINTS, light.1].into_iter().enumerate() {
            push[2] = part as u32;
            let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_le_bytes()).collect();
            images::full_barrier(d, cmd);
            d.cmd_push_constants(cmd, layout.pipeline, vk::ShaderStageFlags::ALL, 0, &bytes);
            d.cmd_dispatch(cmd, groups[0].div_ceil(8), groups[1].div_ceil(8), 1);
        }
        images::full_barrier(d, cmd);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack(v: u32) -> [f32; 3] {
        let s = 2f32.powi((v >> 27) as i32 - 24);
        [(v & 511) as f32 * s, (v >> 9 & 511) as f32 * s, (v >> 18 & 511) as f32 * s]
    }

    #[test]
    fn a_colour_survives_its_shared_exponent_word() {
        for c in [[1.5, 1.47, 1.41], [0.02, 0.5, 0.0], [300.0, 2.0, 0.1], [0.0, 0.0, 0.0], [0.999, 0.999, 0.999]] {
            let back = unpack(pack(c));
            let top = c[0].max(c[1]).max(c[2]);
            for k in 0..3 {
                assert!((back[k] - c[k]).abs() <= top / 256.0 + 1e-6, "{c:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn the_setting_is_a_share_of_the_estimate_or_a_constant() {
        assert_eq!(parse(""), Set::Estimate(SHARE));
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("constant:0"), Set::Off);
        assert_eq!(parse("constant:nan"), Set::Off);
        assert_eq!(parse("constant:2"), Set::Constant([2.0, 1.96, 1.88]));
        assert_eq!(parse("constant:2,1,0.5,0.25"), Set::Constant([2.0, 1.0, 0.5]));
    }

    #[test]
    fn the_programs_it_follows() {
        assert_eq!(role("CS_ScreenProbes_TemporalResolve_ReflectiveCaustics"), Some(Role::Resolve));
        assert_eq!(role("CS_CacheScreenSpaceRadiance"), Some(Role::Radiance));
        assert_eq!(role("CS_TemporalResolve2_Opaque_FrameStatistics"), None);
    }

    #[test]
    fn the_program_is_spir_v() {
        assert_eq!(words::PROGRAM[0], 0x0723_0203);
    }
}
