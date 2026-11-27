//! Texture quality, as a PC game has it: `VICTORY_TEXTURE_QUALITY=high`,
//! `medium` or `low`. Default: `low` on a card with less than 10 GB of
//! its own memory, else `high`.
//!
//! At `medium` the card image of a texture the title fills leaves out its
//! largest level, at `low` its two largest: a quarter or a sixteenth of the
//! card's memory, and of the detiling. The title sees no difference: its
//! descriptions, sizes and queries are the console's. Only the card image
//! is smaller (`vulkan/images/levels.rs`, [`dropped_for`]).
//!
//! Only block-compressed textures: those are the colour, normal and
//! parameter textures a surface is shaded from, where a level less is a
//! softer picture. An uncompressed texture the title fills is data as
//! often as colour (a map of material numbers a program reads at an
//! explicit level, a glyph cache): a smaller image of it blends numbers
//! (the pale seams on the character's suit, `docs/kiss.md` 2026-11-27),
//! and there are few of them (9 MB in the test scene).

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// The levels left out, plus one (0: not decided yet).
static DROPPED: AtomicU32 = AtomicU32::new(0);

/// A card with less of its own memory than this gets `low` by default (vk40:
/// at `medium` an 8 GB card ran past its ceiling in the world).
const HIGH_FROM: u64 = 10 << 30;

/// The smallest side, in texels, of a texture whose host copy may be made
/// smaller.
pub(crate) const SMALLEST: u64 = 256;

/// The levels `VICTORY_TEXTURE_QUALITY` (`value`) leaves out on a card with
/// `dedicated` bytes of its own memory, and the name of that quality.
fn decide(value: Option<&str>, dedicated: u64) -> (u32, &'static str, bool) {
    let by_default = || {
        if dedicated != 0 && dedicated < HIGH_FROM {
            (2, "low")
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

/// The levels to leave out of a two-dimensional texture of `width` by
/// `height` texels and `mips` levels, in a format of `block`-texel elements
/// (4: block compressed; none of a texture of single texels): at most
/// [`levels`], one level always kept, both sides at least [`SMALLEST`],
/// and a level 0 of whole blocks.
pub(crate) fn dropped_for(width: u64, height: u64, mips: u32, block: u64) -> u32 {
    if width < SMALLEST || height < SMALLEST || mips < 2 || block < 2 {
        return 0;
    }
    let mut dropped = levels().min(mips - 1);
    while dropped > 0 && (!(width >> dropped).is_multiple_of(block) || !(height >> dropped).is_multiple_of(block)) {
        dropped -= 1;
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_follows_the_cards_memory() {
        assert_eq!(decide(None, 8151 << 20).0, 2);
        assert_eq!(decide(None, 12 << 30).0, 0);
        assert_eq!(decide(Some(""), 8151 << 20).0, 2);
        assert_eq!(decide(Some("LOW"), 16 << 30).0, 2);
        assert_eq!(decide(Some("high"), 4 << 30).0, 0);
        assert_eq!(decide(Some("ultra"), 4 << 30).0, 2);
    }

    #[test]
    fn small_one_level_uncompressed_and_unaligned_textures_keep_their_levels() {
        DROPPED.store(3, Relaxed);
        assert_eq!(dropped_for(2048, 2048, 12, 4), 2);
        assert_eq!(dropped_for(2048, 2048, 2, 4), 1);
        assert_eq!(dropped_for(255, 2048, 12, 4), 0);
        assert_eq!(dropped_for(2048, 2048, 1, 4), 0);
        // 1000 / 4 = 250 is no whole number of blocks; 1000 / 2 = 500 is.
        assert_eq!(dropped_for(1000, 1000, 10, 4), 1);
        // single texels: data as often as colour
        assert_eq!(dropped_for(1000, 1000, 10, 1), 0);
        assert_eq!(dropped_for(256, 256, 7, 1), 0);
        DROPPED.store(0, Relaxed);
    }
}
