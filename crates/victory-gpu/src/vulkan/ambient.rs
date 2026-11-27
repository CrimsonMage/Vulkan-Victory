//! A stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder off every ray of the title
//! misses. Its program that resolves the diffuse light of its screen
//! probes (`CS_ScreenProbes_TemporalResolve_*`) then leaves nothing
//! indoors and, in blocks of tiles, what the probes' memory held: walls
//! the sun does not reach are black, and all the colour a frame has is the
//! sun's (amber at a low sun). Run col1: that image (0x9bb910000,
//! shared-exponent words in a 32_UINT image) is 0 on the cell's walls and
//! window frame and 0.7 / 0.34 / 0.30 on the bunk, the only thing in the
//! room that shows.
//!
//! `VICTORY_AMBIENT=constant:<level>[,<red>,<green>,<blue>]` puts the same
//! light everywhere right after that program: the image is cleared to
//! `level` times the tint (1, 0.98, 0.94 if none is given). The tiled
//! lighting adds albedo x light / pi (measured on the bunk), so a level of
//! 1.5 gives a wall of albedo 0.4 the 0.2 a lit interior has. It is a
//! stand-in, not the title's light: no occlusion, no bounce colour, the
//! same indoors and outdoors.
//!
//! That a stand-in can be put there is beloooo's find
//! (VIctory-Belobug-AMD, GPL-3.0-or-later), carried by VIctory-NVIDIA
//! (`crates/victory-gpu/src/ambient.rs`, which estimates the light from
//! the frame before: the next step here, a compute pass of the host's).
//! Unset: the title's own output.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use victory_vk::ash::vk;

use super::{Backend, images, read_title};

/// Where the resolve's table holds the T# of the diffuse light it
/// writes: root parameter and dword (the same place as on the D3D12 path).
const LIGHT: (usize, u64) = (1, 176);
/// A T#'s check word (`docs/vulkan-abi.md` 5.2).
const CHECK: u32 = 0x7A3C_5E91;

fn parse(set: &str) -> Option<[f32; 3]> {
    let mut n = set.trim().strip_prefix("constant:")?.split(',').map(|t| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0));
    let level = n.next().flatten().filter(|v| *v > 0.0)?;
    let tint = match (n.next().flatten(), n.next().flatten(), n.next().flatten()) {
        (Some(r), Some(g), Some(b)) => [r, g, b],
        _ => [1.0, 0.98, 0.94],
    };
    Some(tint.map(|t| t * level))
}

fn light() -> Option<[f32; 3]> {
    static SET: OnceLock<Option<[f32; 3]>> = OnceLock::new();
    *SET.get_or_init(|| parse(&std::env::var("VICTORY_AMBIENT").unwrap_or_default()))
}

/// Whether the stand-in is asked for.
pub(super) fn on() -> bool {
    light().is_some()
}

/// Whether `name` is the program the stand-in follows.
pub(super) fn follows(name: &str) -> bool {
    name.starts_with("CS_ScreenProbes_TemporalResolve")
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

/// Right after the resolve's dispatch: the diffuse light it wrote is the
/// constant. `roots` are the root parameters as the title set them.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
pub(super) unsafe fn after(b: &Backend, cmd: vk::CommandBuffer, roots: &[Vec<u32>]) {
    static TOLD: AtomicBool = AtomicBool::new(false);
    let Some(c) = light() else { return };
    let Some(&table) = roots.get(LIGHT.0).and_then(|r| r.first()) else { return };
    let Ok(bytes) = read_title(b, u64::from(table) + LIGHT.1 * 4, 32) else { return };
    let w: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let found = (w[6] ^ w[7] == CHECK).then(|| super::descriptors::slot_image(w[7] & 0xff_ffff)).flatten();
    let Some((image, _)) = found.filter(|(i, _)| i.format == vk::Format::R32_UINT) else {
        if !TOLD.swap(true, Ordering::Relaxed) {
            tracing::warn!(table = format_args!("{table:#x}"), "Vulkan: the ambient stand-in found no 32_UINT image where the resolve writes its light");
        }
        return;
    };
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
    fn the_setting_is_a_level_and_an_optional_tint() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("0.8"), None);
        assert_eq!(parse("constant:0"), None);
        assert_eq!(parse("constant:2"), Some([2.0, 1.96, 1.88]));
        assert_eq!(parse("constant:2,1,0.5,0.25"), Some([2.0, 1.0, 0.5]));
    }
}
