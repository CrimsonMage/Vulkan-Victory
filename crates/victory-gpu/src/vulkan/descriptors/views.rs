//! The title's descriptors written as the console's: V# for buffer views,
//! T# for texture views (their slots made here), S# for samplers.

use std::sync::atomic::Ordering;

use victory_vk::ash::vk;
use victory_vk::gfx10;

use super::super::images::{self, Image, ViewKey};
use super::super::Backend;
use super::{Heaps, STATS, SamplerDesc, heaps};
use crate::descriptor::View;


/// The D3D12 component mapping as Vulkan's (3 bits a channel: source 0-3,
/// 4 zero, 5 one).
fn components(mapping: u32) -> [vk::ComponentSwizzle; 4] {
    use vk::ComponentSwizzle as C;
    let mapping = if mapping == 0 { 0x1688 } else { mapping };
    [0, 3, 6, 9].map(|at| match mapping >> at & 7 {
        0 => C::R,
        1 => C::G,
        2 => C::B,
        3 => C::A,
        4 => C::ZERO,
        _ => C::ONE,
    })
}

/// An A8_UNORM view made with R8_UNORM: the title's channels read the one
/// channel as alpha, and 0 for red, green and blue (D3D's A8), not R8's
/// (r, 0, 0, 1).
fn alpha_only(c: [vk::ComponentSwizzle; 4]) -> [vk::ComponentSwizzle; 4] {
    use vk::ComponentSwizzle as C;
    c.map(|s| match s {
        C::A => C::R,
        C::R | C::G | C::B => C::ZERO,
        other => other,
    })
}

fn swizzle_of(c: [vk::ComponentSwizzle; 4], fmt: gfx10::Swizzle) -> gfx10::Swizzle {
    use vk::ComponentSwizzle as C;
    c.map(|s| match s {
        C::R => fmt[0],
        C::G => fmt[1],
        C::B => fmt[2],
        C::A => fmt[3],
        C::ZERO => gfx10::SEL_0,
        _ => gfx10::SEL_1,
    })
}

/// Writes the descriptor of `view` into the title's slot at `handle`.
pub(crate) fn write(handle: u64, view: &View) {
    if handle == 0 {
        return;
    }
    let Some(b) = super::super::backend() else { return };
    let Some(h) = heaps(b) else { return };
    let started = std::time::Instant::now();
    // The heap the slot is in, and the memory a buffer view names, are read
    // by the card: the buffer whole (a view names its first address, and
    // may begin anywhere in it).
    super::super::touch(handle);
    match view {
        View::Constant(address, size) => {
            super::super::census::view(0, *address, u64::from(*size));
            super::super::touch_range(*address, u64::from(*size))
        }
        View::Shader(r, _) | View::Unordered(r, _, _) => {
            if let Some((data, len)) = crate::resource::buffer_bytes(*r as crate::object::This) {
                super::super::touch_range(data as u64, len);
            }
        }
        View::Placed(_, rdesc, desc) => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            // SAFETY: the resource description is these 14 words.
            let r: crate::resource::ResourceDesc = unsafe { std::ptr::read_unaligned(rdesc.as_ptr().cast()) };
            super::super::touch_range(address, if rdesc[0] == 1 { r.width } else { 1 });
        }
        _ => {}
    }
    if images::census_on() {
        census(b, handle, view);
    }
    // The texture the slot names from now on (`images/named.rs`).
    let mut shown = None;
    let slot = Slot { handle, heaps: h };
    let words: Vec<u32> = match view {
        View::Sampler(w) => {
            let d = SamplerDesc::from_words(w);
            d.encode(h.sampler_slot(b, &d)).to_vec()
        }
        View::ConsoleSampler(w) => {
            let s = crate::sampler::console(w);
            let mut words = [0u32; 13];
            // SAFETY: D3D12_SAMPLER_DESC is these 13 words.
            unsafe { std::ptr::copy_nonoverlapping((&raw const s).cast::<u32>(), words.as_mut_ptr(), 13) };
            let d = SamplerDesc::from_words(&words);
            d.encode(h.sampler_slot(b, &d)).to_vec()
        }
        View::Constant(address, size) => {
            // As the console's CBVs: records of 16 bytes (float4). Its code
            // indexes constant arrays with idxen and the V#'s stride (the
            // title's Scaleform batch programs: a stride of 0 read every
            // row from row 0, their quads collapsed onto the diagonal).
            let cbv = gfx10::Buffer {
                address: *address,
                stride: 16,
                records: size.div_ceil(16),
                format: 77,
                swizzle: gfx10::IDENTITY,
                oob: gfx10::Oob::Structured,
            };
            let mut w = cbv.encode().to_vec();
            w.resize(8, 0);
            w
        }
        View::Shader(r, desc) | View::Unordered(r, _, desc) => {
            let unordered = matches!(view, View::Unordered(..));
            let r = *r as crate::object::This;
            match crate::resource::buffer_bytes(r) {
                Some((data, len)) => buffer_words(data as u64, len, desc.as_ref(), unordered),
                None => match crate::resource::texture_memory(r) {
                    Some((data, rdesc)) => texture_words(b, &slot, data as u64, &rdesc, desc.as_ref(), unordered, &mut shown),
                    None => null_texture(desc.as_ref().map_or(4, |d| d[1]), unordered),
                },
            }
        }
        View::Placed(unordered, rdesc, desc) => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            // SAFETY: the resource description is these 14 words.
            let r: crate::resource::ResourceDesc = unsafe { std::ptr::read_unaligned(rdesc.as_ptr().cast()) };
            let d10: [u32; 10] = desc[..10].try_into().unwrap_or([0; 10]);
            if rdesc[0] == 1 {
                buffer_words(address, r.width, Some(&d10), *unordered)
            } else {
                texture_words(b, &slot, address, &r, Some(&d10), *unordered, &mut shown)
            }
        }
    };
    if shown.is_none() && !matches!(view, View::Sampler(_) | View::ConsoleSampler(_)) {
        images::slot_shows(handle, None);
    }
    // SAFETY: the handle is a descriptor slot (views 32 bytes, samplers 16).
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), handle as *mut u32, words.len()) };
    super::super::split(super::super::Split::Views, started);
}

/// Diagnostics (`images::census`): the texture view the title writes.
fn census(b: &Backend, handle: u64, view: &View) {
    match view {
        View::Shader(r, desc) | View::Unordered(r, _, desc) => {
            if let Some((data, rdesc)) = crate::resource::texture_memory(*r as crate::object::This) {
                let words = desc.map_or_else(Vec::new, |d| d.to_vec());
                images::census_view(b, handle, data as u64, &rdesc, &words, matches!(view, View::Unordered(..)));
            }
        }
        View::Placed(unordered, rdesc, desc) if rdesc[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            // SAFETY: the resource description is these 14 words.
            let r: crate::resource::ResourceDesc = unsafe { std::ptr::read_unaligned(rdesc.as_ptr().cast()) };
            images::census_view(b, handle, address, &r, desc, *unordered);
        }
        _ => {}
    }
}

/// A V# for a buffer view of `len` bytes at `base` (SRV or UAV words).
fn buffer_words(base: u64, len: u64, desc: Option<&[u32; 10]>, unordered: bool) -> Vec<u32> {
    let out = match desc {
        None => gfx10::Buffer::raw(base, len.min(u64::from(u32::MAX)) as u32),
        Some(d) => {
            // SRV: format, dimension, mapping, pad, first (2), count, stride, flags.
            // UAV: format, dimension, first (2), count, stride, counter (2), flags.
            let (format, first, count, stride, flags) = if unordered {
                (d[0], u64::from(d[2]) | u64::from(d[3]) << 32, d[4], d[5], d[8])
            } else {
                (d[0], u64::from(d[4]) | u64::from(d[5]) << 32, d[6], d[7], d[8])
            };
            if stride > 0 {
                gfx10::Buffer {
                    address: base + first * u64::from(stride),
                    stride,
                    records: count,
                    format: 0,
                    swizzle: gfx10::IDENTITY,
                    oob: gfx10::Oob::Structured,
                }
            } else if flags & 1 != 0 || format == 0 {
                gfx10::Buffer::raw(base + first * 4, count.saturating_mul(4))
            } else {
                let element = crate::resource::element(format).map_or(4, |(e, _)| e) as u32;
                let (code, swizzle) = victory_vk::gfx10::from_dxgi(format)
                    .and_then(gfx10::format_of)
                    .unwrap_or((0, gfx10::IDENTITY));
                gfx10::Buffer {
                    address: base + first * u64::from(element),
                    stride: element,
                    records: count,
                    format: if code < 128 { code } else { 0 },
                    swizzle,
                    oob: gfx10::Oob::Structured,
                }
            }
        }
    }
    ;
    let bytes = u64::from(out.stride.max(1)) * u64::from(out.records);
    super::super::census::view(1 + u8::from(unordered), out.address, bytes);
    if unordered {
        super::super::census::resource(base, len, bytes);
        if let Some(b) = super::super::backend() {
            super::super::homes::view(b, base, len, bytes);
        }
    }
    let mut out = out.encode().to_vec();
    out.resize(8, 0);
    out
}

fn null_texture(dimension: u32, unordered: bool) -> Vec<u32> {
    let kind = match dimension {
        2 => gfx10::ImageType::D1,
        3 => gfx10::ImageType::D1Array,
        5 => gfx10::ImageType::D2Array,
        8 => gfx10::ImageType::D3,
        9 | 10 => gfx10::ImageType::Cube,
        _ => gfx10::ImageType::D2,
    };
    let (code, swizzle) = gfx10::format_of(vk::Format::R8G8B8A8_UNORM).unwrap_or((0, gfx10::IDENTITY));
    gfx10::Image {
        address: 0,
        format: code,
        kind,
        width: 1,
        height: 1,
        depth: 1,
        first_layer: 0,
        layer_count: 1,
        first_mip: 0,
        mip_count: 1,
        resource_mips: 1,
        samples: 1,
        swizzle,
        min_lod: 0.0,
    }
    .encode(gfx10::ImageSlot {
        slot: 0,
        integer: false,
        storage: unordered,
    })
    .to_vec()
}

/// The title's slot a view is written into, and the host's heaps.
struct Slot<'a> {
    handle: u64,
    heaps: &'a Heaps,
}

/// The first level a texture view's description names (`shape`'s
/// `base_mip`, before there is an image to read it for).
fn first_level(dimension: u32, desc: Option<&[u32; 10]>, unordered: bool) -> u32 {
    match (desc, dimension) {
        (Some(d), 2..=5 | 8..=10) if !unordered => d[4],
        _ => 0,
    }
}

/// What a texture view shows: its kind and ranges.
struct Shape {
    kind: vk::ImageViewType,
    base_mip: u32,
    mips: u32,
    base_layer: u32,
    layers: u32,
}

/// The view's kind and ranges from the description (SRV fields from word
/// 4, UAV fields from word 2); without one, the whole image.
fn shape(image: &Image, dimension: u32, desc: Option<&[u32; 10]>, unordered: bool) -> Shape {
    let d = desc.copied().unwrap_or([0; 10]);
    let f = |i: usize| if unordered { d[i - 2] } else { d[i] };
    let all = u32::MAX;
    let one = |n: u32| if unordered { 1 } else { n };
    use vk::ImageViewType as T;
    let (kind, base_mip, mips, base_layer, layers) = match (dimension, unordered) {
        (2, _) => (T::TYPE_1D, f(4), one(f(5)), 0, 1),
        (3, false) => (T::TYPE_1D_ARRAY, f(4), f(5), f(6), f(7)),
        (3, true) => (T::TYPE_1D_ARRAY, f(4), 1, f(5), f(6)),
        (4, _) => (T::TYPE_2D, f(4), one(f(5)), 0, 1),
        (5, false) => (T::TYPE_2D_ARRAY, f(4), f(5), f(6), f(7)),
        (5, true) => (T::TYPE_2D_ARRAY, f(4), 1, f(5), f(6)),
        (6, _) => (T::TYPE_2D, 0, 1, 0, 1),
        (7, _) => (T::TYPE_2D_ARRAY, 0, 1, f(4), f(5)),
        (8, _) => (T::TYPE_3D, f(4), one(f(5)), 0, 1),
        (9, _) => (T::CUBE, f(4), f(5), 0, 6),
        (10, _) => (T::CUBE_ARRAY, f(4), f(5), f(6) * 6, f(7) * 6),
        _ if image.kind == vk::ImageType::TYPE_3D => (T::TYPE_3D, 0, all, 0, 1),
        _ if image.kind == vk::ImageType::TYPE_1D => (T::TYPE_1D_ARRAY, 0, all, 0, all),
        _ => (T::TYPE_2D_ARRAY, 0, all, 0, all),
    };
    let fill = |n: u32, total: u32, base: u32| if n == 0 || n == all { total.saturating_sub(base).max(1) } else { n };
    Shape {
        kind,
        base_mip,
        mips: fill(mips, image.mips, base_mip),
        base_layer,
        layers: fill(layers, image.layers, base_layer),
    }
}

/// Diagnostics: each distinct unordered view of a 64-bit-class format
/// (the alpha prepass's V-buffer, whose texels take 64-bit atomics, ABI v5),
/// once: the resource's and the view's DXGI formats and the image's.
fn note_wide_uav(image: &Image, resource_format: u32, view_format: u32, format: vk::Format) {
    if !cfg!(feature = "diagnostics") || images::format_bytes(format) != 8 {
        return;
    }
    static SEEN: std::sync::Mutex<Vec<(u32, u32, vk::Format)>> = std::sync::Mutex::new(Vec::new());
    let key = (resource_format, view_format, image.format);
    let mut seen = SEEN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if seen.contains(&key) || seen.len() >= 32 {
        return;
    }
    seen.push(key);
    tracing::info!(
        resource_dxgi = resource_format,
        view_dxgi = view_format,
        image_format = ?image.format,
        view_format = ?format,
        width = image.extent.width,
        height = image.extent.height,
        usage = ?image.usage,
        "Vulkan: an unordered view of a 64-bit-class format"
    );
}

/// The view's format and aspect for the title's `view_format`, and
/// whether it reads stencil.
fn view_format(image: &Image, view_format: u32, storage: bool) -> (vk::Format, vk::ImageAspectFlags, bool) {
    if image.depth() {
        let stencil = images::reads_stencil(view_format);
        let aspect = if stencil { vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
        return (image.format, aspect, stencil);
    }
    let f = images::color_format(view_format, image.card_written()).unwrap_or(image.format);
    let f = if images::format_bytes(f) == images::format_bytes(image.format) {
        f
    } else if !storage && crate::resource::pc_format(view_format) == 67 && images::format_bytes(image.format) == 4 {
        // DXGI_FORMAT_R9G9B9E5_SHAREDEXP read from a 4-byte image the card
        // writes as packed words (a compute program packs them into a
        // 32_UINT view): sampled as what the words are, not as integers
        // (the fog inscatter read as R32_UINT gave 2e9 in red).
        vk::Format::E5B9G9R9_UFLOAT_PACK32
    } else {
        image.format
    };
    (f, vk::ImageAspectFlags::COLOR, false)
}

fn image_type(kind: vk::ImageViewType, msaa: bool) -> gfx10::ImageType {
    use vk::ImageViewType as T;
    match kind {
        T::TYPE_1D => gfx10::ImageType::D1,
        T::TYPE_1D_ARRAY => gfx10::ImageType::D1Array,
        T::TYPE_3D => gfx10::ImageType::D3,
        T::CUBE | T::CUBE_ARRAY => gfx10::ImageType::Cube,
        T::TYPE_2D_ARRAY if msaa => gfx10::ImageType::D2MsaaArray,
        T::TYPE_2D_ARRAY => gfx10::ImageType::D2Array,
        _ if msaa => gfx10::ImageType::D2Msaa,
        _ => gfx10::ImageType::D2,
    }
}

/// A T# for a texture view, its slot made now.
/// A storage view of a depth buffer: the stores go to the colour image
/// over that plane's memory (`images::written_plane`); the image and the
/// title format the view is made with.
fn stored_plane(
    b: &Backend,
    image: std::sync::Arc<Image>,
    address: u64,
    rdesc: &crate::resource::ResourceDesc,
    desc: Option<&[u32; 10]>,
    title_format: u32,
) -> (std::sync::Arc<Image>, u32) {
    // The stencil plane by its format, or as plane slice 1 of a 2D
    // storage view (D3D12_TEX2D_UAV: mip slice, plane slice).
    let stencil = images::reads_stencil(title_format) || desc.is_some_and(|d| d[1] == 4 && d[3] == 1);
    static TOLD: std::sync::Mutex<Vec<(u64, u32)>> = std::sync::Mutex::new(Vec::new());
    let mut told = TOLD.lock().unwrap_or_else(|e| e.into_inner());
    let what = (address, title_format | u32::from(stencil) << 31);
    if told.len() < 64 && !told.contains(&what) {
        told.push(what);
        tracing::warn!(
            at = format_args!("{address:#x}"),
            resource_format = rdesc.format,
            view_format = title_format,
            plane = if stencil { "stencil" } else { "depth" },
            view = ?desc.map(|d| [d[1], d[2], d[3]]),
            "Vulkan: a storage view of a depth buffer: its stores go to a colour image, copied in before the buffer is attached"
        );
    }
    drop(told);
    match images::written_plane(b, &image, address, rdesc, stencil) {
        Some(p) => {
            let format = p.desc.format;
            (p, format)
        }
        None => (image, title_format),
    }
}

fn texture_words(
    b: &Backend,
    slot: &Slot,
    address: u64,
    rdesc: &crate::resource::ResourceDesc,
    desc: Option<&[u32; 10]>,
    unordered: bool,
    shown: &mut Option<images::Key>,
) -> Vec<u32> {
    let h = slot.heaps;
    let dimension = desc.map_or(0, |d| d[1]);
    let Some(image) = images::at_view(b, slot.handle, address, rdesc, first_level(dimension, desc, unordered)) else {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return null_texture(dimension, unordered);
    };
    *shown = Some(image.key);
    let title_format = desc.map_or(0, |d| d[0]);
    let title_format = if title_format == 0 { rdesc.format } else { title_format };
    let (image, title_format) = if unordered && image.depth() && image.samples == 1 {
        stored_plane(b, image, address, rdesc, desc, title_format)
    } else {
        (image, title_format)
    };
    let mapping = if unordered { 0 } else { desc.map_or(0, |d| d[2]) };
    let shape = shape(&image, dimension, desc, unordered);
    let (format, aspect, stencil) = view_format(&image, title_format, unordered);
    if unordered {
        note_wide_uav(&image, rdesc.format, title_format, format);
    }
    let comps = components(mapping);
    // DXGI_FORMAT_A8_UNORM
    let comps = if !unordered && crate::resource::pc_format(title_format) == 65 && format == vk::Format::R8_UNORM {
        alpha_only(comps)
    } else {
        comps
    };
    // A stencil view holds the stencil in every channel, as the console's
    // does (the title's programs load it from the first and gather it from
    // the second: CS_Tiling_Generate_Opaque's gather of the material
    // classes read the second channel of the card's view, which is 0).
    let comps = if stencil && !unordered { [vk::ComponentSwizzle::R; 4] } else { comps };
    let key = ViewKey {
        format,
        kind: shape.kind,
        aspect,
        base_mip: shape.base_mip,
        mips: shape.mips,
        base_layer: shape.base_layer,
        layers: shape.layers,
        components: comps,
        storage: unordered,
        attachment: false,
    };
    let Some(slot) = h.image_slot(b, &image, key) else {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return null_texture(dimension, unordered);
    };
    let sampled_format = match (image.depth(), stencil) {
        (true, true) => vk::Format::R8_UINT,
        (true, false) => vk::Format::R32_SFLOAT,
        _ => format,
    };
    let (code, fmt_swizzle) = gfx10::format_of(sampled_format).unwrap_or((0, gfx10::IDENTITY));
    gfx10::Image {
        address,
        format: code,
        kind: image_type(shape.kind, image.samples > 1),
        // The console's shape (texture quality may leave the image's
        // largest levels out).
        width: if image.dropped > 0 { image.desc.width as u32 } else { image.extent.width },
        height: if image.dropped > 0 { image.desc.height } else { image.extent.height },
        depth: image.extent.depth,
        first_layer: shape.base_layer,
        layer_count: shape.layers,
        first_mip: shape.base_mip,
        mip_count: shape.mips,
        resource_mips: image.mips + image.dropped,
        samples: image.samples,
        swizzle: swizzle_of(comps, fmt_swizzle),
        min_lod: 0.0,
    }
    .encode(gfx10::ImageSlot {
        slot,
        integer: stencil || gfx10::is_integer(format),
        storage: unordered,
    })
    .to_vec()
}
