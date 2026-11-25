//! Texture quality, as a PC game has it: `VICTORY_TEXTURE_QUALITY=high`,
//! `medium` or `low`. Default: `medium` on a card with less than 10 GB of
//! its own memory, else `high`.
//!
//! At `medium` the host copy of a texture the title fills leaves out its
//! largest level, at `low` its two largest: a quarter or a sixteenth of the
//! card's memory, and of the detiling. The title sees no difference: its
//! descriptions, sizes and queries are the console's. Only the host copy is
//! smaller (`resource.rs` `hosted`), the detiling skips the levels left out
//! (`texture.rs`), host views start at the same level of the console's
//! chain one or two levels lower in the host's ([`shift_view`]), and the
//! title's copies into or out of the texture name the host level that
//! holds the console level ([`host_level`]).
//!
//! Left at full size: render targets, depth and unordered-access textures
//! (the host writes them), textures of one level, textures under 256
//! texels on a side, volumes, and textures read through a view in another
//! format (`resource.rs` `alias_of`).

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

/// The levels left out, plus one (0: not decided yet).
static DROPPED: AtomicU32 = AtomicU32::new(0);

/// A card with less of its own memory than this gets `medium` by default.
const HIGH_FROM: u64 = 10 << 30;

/// The smallest side, in texels, of a texture whose host copy may be made
/// smaller.
pub(crate) const SMALLEST: u64 = 256;

/// The levels `VICTORY_TEXTURE_QUALITY` (`value`) leaves out on a card with
/// `dedicated` bytes of its own memory, and the name of that quality.
fn decide(value: Option<&str>, dedicated: u64) -> (u32, &'static str, bool) {
    let by_default = || {
        if dedicated != 0 && dedicated < HIGH_FROM {
            (1, "medium")
        } else {
            (0, "high")
        }
    };
    let (levels, name) = match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("high") => (0, "high"),
        Some("medium") => (1, "medium"),
        Some("low") => (2, "low"),
        Some(other) if !other.is_empty() => {
            tracing::warn!(
                value = other,
                "VICTORY_TEXTURE_QUALITY is high, medium or low: the default is used"
            );
            by_default()
        }
        _ => by_default(),
    };
    (levels, name, value.is_none_or(|v| v.trim().is_empty()))
}

/// Decides the quality once the adapter's own memory is known (`dedicated`
/// bytes), and says so.
pub(crate) fn set(dedicated: u64) {
    let value = std::env::var("VICTORY_TEXTURE_QUALITY").ok();
    let (levels, name, by_default) = decide(value.as_deref(), dedicated);
    DROPPED.store(levels + 1, Relaxed);
    tracing::info!(
        "texture quality {name}{}: host copies of textures the title fills leave out their {} (VICTORY_TEXTURE_QUALITY=high|medium|low)",
        if by_default {
            format!(
                " (the default for a card with {} MB of its own memory)",
                dedicated >> 20
            )
        } else {
            String::new()
        },
        match levels {
            0 => "largest level never",
            1 => "largest level: a quarter of the memory",
            _ => "two largest levels: a sixteenth of the memory",
        },
    );
}

/// The levels the quality leaves out of a texture that may be made
/// smaller (0 until [`set`]).
pub(crate) fn levels() -> u32 {
    DROPPED.load(Relaxed).saturating_sub(1)
}

/// The quality's name, for the memory line.
pub(crate) fn name() -> &'static str {
    match levels() {
        0 => "high",
        1 => "medium",
        _ => "low",
    }
}

/// The levels to leave out of a two-dimensional texture of `width` by
/// `height` texels and `mips` levels, in a format of `block`-texel elements
/// (4: block compressed): at most [`levels`], one level always kept, both
/// sides at least [`SMALLEST`], and a block-compressed level 0 of whole
/// blocks (the host requires it).
pub(crate) fn dropped_for(width: u64, height: u64, mips: u32, block: u64) -> u32 {
    if width < SMALLEST || height < SMALLEST || mips < 2 {
        return 0;
    }
    let mut dropped = levels().min(mips - 1);
    while dropped > 0
        && block > 1
        && (!(width >> dropped).is_multiple_of(block) || !(height >> dropped).is_multiple_of(block))
    {
        dropped -= 1;
    }
    dropped
}

/// The host level that holds console level `level` of a texture whose
/// host copy leaves out `dropped` levels; `None` for a level left out.
pub(crate) fn host_level(level: u32, dropped: u32) -> Option<u32> {
    level.checked_sub(dropped)
}

/// The host subresource that holds console subresource `index` of a texture
/// of `mips` console levels whose host copy leaves out `dropped` (levels
/// first, then array slices, as D3D12 numbers them); `None` for a level
/// left out.
pub(crate) fn host_subresource(index: u32, mips: u32, dropped: u32) -> Option<u32> {
    if dropped == 0 || mips == 0 {
        return Some(index);
    }
    let (level, slice) = (index % mips, index / mips);
    Some(host_level(level, dropped)? + slice * (mips - dropped))
}

/// A view of a texture whose host copy leaves out `dropped` of the
/// console's levels and keeps `kept`: it starts that many levels lower in
/// the host's chain, at the same level of the texture (at the host's level
/// 0 if the title's view started at a level left out, with as many levels
/// fewer), and its minimum level of detail moves with it. A shader that
/// samples level 0 gets the largest level kept.
pub(crate) fn shift_view(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, dropped: u32, kept: u32) {
    if dropped == 0 || kept == 0 {
        return;
    }
    let shift = |most: &mut u32, mips: &mut u32, min_lod: &mut f32| {
        let below = dropped.saturating_sub(*most);
        *most = most.saturating_sub(dropped).min(kept - 1);
        if *mips != u32::MAX {
            *mips = mips.saturating_sub(below).clamp(1, kept - *most);
        }
        *min_lod = (*min_lod - dropped as f32).max(0.0);
    };
    // SAFETY: each arm reads the union member its dimension selects.
    unsafe {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let v = &mut desc.Anonymous.Texture2D;
                shift(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let v = &mut desc.Anonymous.Texture2DArray;
                shift(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                let v = &mut desc.Anonymous.TextureCube;
                shift(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let v = &mut desc.Anonymous.TextureCubeArray;
                shift(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_follows_the_cards_memory() {
        assert_eq!(decide(None, 8151 << 20).0, 1);
        assert_eq!(decide(None, 12 << 30).0, 0);
        assert_eq!(decide(Some(""), 8151 << 20).0, 1);
        assert_eq!(decide(Some("LOW"), 16 << 30).0, 2);
        assert_eq!(decide(Some("high"), 4 << 30).0, 0);
        assert_eq!(decide(Some("ultra"), 4 << 30).0, 1);
    }

    #[test]
    fn subresources_move_down_by_the_levels_left_out() {
        // Ten levels, three slices, one level left out.
        assert_eq!(host_subresource(0, 10, 1), None);
        assert_eq!(host_subresource(1, 10, 1), Some(0));
        assert_eq!(host_subresource(9, 10, 1), Some(8));
        assert_eq!(host_subresource(10, 10, 1), None);
        assert_eq!(host_subresource(12, 10, 1), Some(10));
        assert_eq!(host_subresource(25, 10, 2), Some(2 * 8 + 3));
        assert_eq!(host_subresource(7, 10, 0), Some(7));
    }

    fn view(most: u32, mips: u32, min_lod: f32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: most,
                    MipLevels: mips,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: min_lod,
                },
            },
            ..Default::default()
        }
    }

    fn of(d: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> (u32, u32, f32) {
        // SAFETY: made as a 2D view.
        let v = unsafe { d.Anonymous.Texture2D };
        (v.MostDetailedMip, v.MipLevels, v.ResourceMinLODClamp)
    }

    #[test]
    fn views_start_at_the_same_level_of_the_texture() {
        // Twelve console levels, two left out: ten kept.
        let mut all = view(0, u32::MAX, 0.0);
        shift_view(&mut all, 2, 10);
        assert_eq!(of(&all), (0, u32::MAX, 0.0));
        let mut top = view(0, 1, 0.0);
        shift_view(&mut top, 2, 10);
        assert_eq!(of(&top), (0, 1, 0.0));
        let mut from_one = view(1, 4, 1.5);
        shift_view(&mut from_one, 2, 10);
        assert_eq!(of(&from_one), (0, 3, 0.0));
        let mut from_three = view(3, 9, 3.5);
        shift_view(&mut from_three, 2, 10);
        assert_eq!(of(&from_three), (1, 9, 1.5));
        let mut past = view(4, 20, 0.0);
        shift_view(&mut past, 2, 10);
        assert_eq!(of(&past), (2, 8, 0.0));
    }

    #[test]
    fn small_one_level_and_unaligned_textures_keep_their_levels() {
        DROPPED.store(3, Relaxed);
        assert_eq!(dropped_for(2048, 2048, 12, 4), 2);
        assert_eq!(dropped_for(2048, 2048, 2, 1), 1);
        assert_eq!(dropped_for(255, 2048, 12, 1), 0);
        assert_eq!(dropped_for(2048, 2048, 1, 1), 0);
        // 1000 / 4 = 250 is no whole number of blocks; 1000 / 2 = 500 is.
        assert_eq!(dropped_for(1000, 1000, 10, 4), 1);
        assert_eq!(dropped_for(1000, 1000, 10, 1), 2);
        DROPPED.store(0, Relaxed);
    }
}
