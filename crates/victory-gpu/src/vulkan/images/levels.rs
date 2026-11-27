//! Which levels of a texture the title fills are on the card, and which
//! bytes of its console layout they are.
//!
//! The title streams a texture's levels (runs tx0 to tx3, `docs/kiss.md`).
//! It places every texture with its whole description, all levels, in a
//! piece of its texture pool sized for the levels it holds: the console
//! stores the block of the small levels first, then the others from the
//! smallest up, level 0 last, so the levels held are the head of the
//! layout, and the rest of the layout's address range is the next
//! textures' memory (a 1024x1024 texture held from level 3 has the
//! first 0x3000 of its 0xab000 bytes; its neighbour starts 0x4000 on).
//! **Which levels are there is the view's word**: the title writes the
//! texture's view into its descriptor slot from the first level it holds
//! (`MostDetailedMip`), and writes it again when levels come or go (in
//! place while the piece has the room, else at the texture's new address).
//!
//! So the image of such a texture has the levels from the one its latest
//! view names on, less those texture quality cuts, and is filled from the
//! head of the layout that holds them; when a view names another first
//! level the image is made again (`fill::renamed`).

use victory_formats::texture::Layout;

use crate::resource::ResourceDesc;

const TEXTURE_2D: u32 = 3;

/// The levels of one image: what the title's view named, what the image
/// leaves out, and the heads of the console layout that go with both.
pub(super) struct Levels {
    /// The first console level the title's view names.
    pub(super) named: u32,
    /// The console's largest levels the image leaves out.
    pub(super) dropped: u32,
    /// Bytes of the layout the image's levels are read from.
    pub(super) bytes: u64,
    /// Bytes of the layout through the level named: the texture's own.
    pub(super) claim: u64,
}

/// The texels a side of an element of a texture whose image may leave
/// its largest levels out: two-dimensional, of one sample, in a console
/// layout the detiling reads.
fn trimmed(desc: &ResourceDesc) -> Option<u64> {
    if desc.dimension != TEXTURE_2D || desc.sample_count > 1 {
        return None;
    }
    let (_, block) = crate::resource::element(crate::resource::pc_format(desc.format))?;
    let readable = crate::resource::texture_layout(desc).is_some_and(|t| !t.planes[0].layout.desc.swizzle.xor());
    readable.then_some(block)
}

/// The levels of the image of `desc` (`mips` levels) whose view names
/// `first` as its first level. `whole`: a texture the card writes, a depth
/// texture or an image over another's memory, which keeps every level.
pub(super) fn of(desc: &ResourceDesc, whole: bool, mips: u32, first: u32) -> Levels {
    let size = crate::resource::texture_layout(desc).map_or(0, |t| t.size);
    let Some(block) = trimmed(desc).filter(|_| !whole) else {
        return Levels { named: 0, dropped: 0, bytes: size, claim: size };
    };
    let named = first.min(mips - 1);
    let quality = crate::quality::dropped_for(desc.width, u64::from(desc.height), mips, block);
    let dropped = named.max(quality);
    let head = |level: u32| match crate::resource::texture_layout(desc) {
        Some(t) if level > 0 => head(&t.planes[0].layout, level),
        _ => size,
    };
    Levels { named, dropped, bytes: head(dropped), claim: head(named) }
}

/// The first level a view names, as [`of`] takes it for the image of
/// `desc`: 0 for a texture that keeps every level.
pub(super) fn named(desc: &ResourceDesc, mips: u32, first: u32) -> u32 {
    if trimmed(desc).is_some() { first.min(mips.max(1) - 1) } else { 0 }
}

/// The bytes of a console layout through level `level`: the head that
/// holds that level and every smaller one. All of it for a layout of
/// several slices (each slice has its own chain).
pub(super) fn head(layout: &Layout, level: u32) -> u64 {
    if layout.slices != 1 || level == 0 {
        return layout.size;
    }
    if level >= layout.first_tail_mip {
        return layout.alignment.min(layout.size);
    }
    layout.mips.get(level as usize - 1).map_or(layout.size, |m| m.offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bc1(width: u64, mips: u16) -> ResourceDesc {
        // DXGI_FORMAT_BC1_UNORM, the console's pick of layout
        ResourceDesc { dimension: TEXTURE_2D, alignment: 0, width, height: width as u32, depth_or_array_size: 1, mip_levels: mips, format: 71, sample_count: 1, sample_quality: 0, layout: 0, flags: 0 }
    }

    #[test]
    fn the_levels_held_are_the_head_of_the_layout() {
        let t = crate::resource::texture_layout(&bc1(1024, 9)).unwrap();
        let l = &t.planes[0].layout;
        assert_eq!(l.size, 0xab000);
        // the block of the small levels, then 128x128, 256x256, 512x512
        assert_eq!(head(l, 8), 0x1000);
        assert_eq!(head(l, 4), 0x1000);
        assert_eq!(head(l, 3), 0x3000);
        assert_eq!(head(l, 2), 0xb000);
        assert_eq!(head(l, 1), 0x2b000);
        assert_eq!(head(l, 0), 0xab000);
    }

    #[test]
    fn an_image_leaves_out_what_the_view_does_not_name() {
        let d = bc1(1024, 9);
        let l = of(&d, false, 9, 3);
        assert_eq!((l.named, l.dropped, l.bytes, l.claim), (3, 3, 0x3000, 0x3000));
        // a level past the last is the last
        assert_eq!(of(&d, false, 9, 15).dropped, 8);
        // a texture the card writes keeps every level
        let w = of(&d, true, 9, 3);
        assert_eq!((w.named, w.dropped, w.bytes), (0, 0, 0xab000));
        assert_eq!(named(&d, 9, 3), 3);
    }
}
