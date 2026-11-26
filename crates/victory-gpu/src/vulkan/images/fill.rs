//! Filling images from title memory: queued images set up and detiled
//! into one staging buffer per flush, refilled at announced writes and
//! when their memory's fingerprint changes; images freed with their memory.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::super::{Backend, lock};
use super::{Image, Key, PLACES, STATS, USERS, full_barrier, once_warn, registry, release, to_general};


/// Announced write over `[at, at + len)`: images there are filled again.
pub(crate) fn written(at: u64, len: u64) {
    let mut guard = registry();
    let Some(r) = guard.as_mut() else { return };
    // Images start at most 4 GiB before the write.
    let from = Key {
        address: at.saturating_sub(4 << 30),
        dimension: 0,
        width: 0,
        height: 0,
        depth_or_array: 0,
        mips: 0,
        element: 0,
        samples: 0,
        depth: false,
    };
    let mut hit = Vec::new();
    for (k, img) in r.images.range(from..) {
        if k.address >= at + len {
            break;
        }
        if !img.card_written() && k.address + img.bytes.max(1) > at && !img.pending.swap(true, Ordering::Relaxed) {
            hit.push(img.clone());
        }
    }
    r.pending.extend(hit);
}

/// The title freed, unmapped or remapped `[at, at + len)`: images over
/// any of it go once the card is done (counted by `why`, [`Dropped`]). An
/// image made there later is filled from what the memory holds then.
pub(crate) fn released(b: &Backend, at: u64, len: u64, why: super::Dropped) {
    let mut guard = registry();
    let Some(r) = guard.as_mut() else { return };
    // Images start at most 4 GiB before the range.
    let from = Key { address: at.saturating_sub(4 << 30), dimension: 0, width: 0, height: 0, depth_or_array: 0, mips: 0, element: 0, samples: 0, depth: false };
    let gone: Vec<Key> = r
        .images
        .range(from..)
        .take_while(|(k, _)| k.address < at + len)
        .filter(|(k, i)| k.address + i.bytes.max(1) > at)
        .map(|(k, _)| *k)
        .collect();
    let value = b.timeline.last();
    for k in gone {
        if let Some(img) = r.images.remove(&k) {
            super::super::descriptors::image_gone(img.id);
            r.dropped.push((value, img));
            super::DROPPED[why as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Fills queued images and destroys what the card no longer uses; called
/// before work is submitted (the fills go first on the timeline).
pub(crate) fn flush(b: &Backend) {
    let (pending, done) = {
        let mut guard = registry();
        let Some(r) = guard.as_mut() else { return };
        let completed = b.timeline.completed(&b.host);
        free_staging(b, r, completed);
        let done = finished(b, r, completed);
        (std::mem::take(&mut r.pending), done)
    };
    for img in done {
        destroy(b, &img);
    }
    if !pending.is_empty() {
        set_up(b, &pending);
    }
}

/// Staging buffers the card is done with go.
fn free_staging(b: &Backend, r: &mut super::Registry, completed: u64) {
    let (done, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut r.staging).into_iter().partition(|(v, _)| *v <= completed);
    r.staging = kept;
    for (_, buf) in done {
        // SAFETY: the card has passed its last use.
        unsafe { buf.destroy(&b.host) };
    }
}

/// Dropped images the card is done with and nothing holds; their card
/// memory goes with the last image bound to it.
fn finished(b: &Backend, r: &mut super::Registry, completed: u64) -> Vec<Arc<Image>> {
    let (done, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut r.dropped)
        .into_iter()
        .partition(|(v, i)| *v <= completed && Arc::strong_count(i) == 1);
    r.dropped = kept;
    let done: Vec<Arc<Image>> = done.into_iter().map(|(_, i)| i).collect();
    for img in &done {
        let Some((memory, at, size, _)) = lock(&PLACES).remove(&img.id) else { continue };
        let mut users = lock(&USERS);
        let n = users.entry((memory, at)).or_insert(1);
        *n -= 1;
        if *n == 0 {
            users.remove(&(memory, at));
            release(b, &mut r.blocks, memory, at, size);
        }
    }
    done
}

fn destroy(b: &Backend, img: &Image) {
    // SAFETY: nothing uses it any more.
    unsafe {
        for (_, v) in lock(&img.views).drain(..) {
            b.host.device.destroy_image_view(v, None);
        }
        b.host.device.destroy_image(img.image, None);
    }
}

/// One staging buffer and one submission for everything queued: new
/// images into GENERAL (cleared), then the fills.
fn set_up(b: &Backend, pending: &[Arc<Image>]) {
    let (jobs, total) = plan_all(pending);
    let staging = staging_buffer(b, total);
    let copies = staging.as_ref().map(|st| detile_all(b, st, &jobs, total)).unwrap_or_default();
    let fresh: Vec<(vk::Image, vk::ImageAspectFlags, bool)> = pending
        .iter()
        .filter(|i| i.fresh.swap(false, Ordering::Relaxed))
        .map(|i| (i.image, i.aspect(), !i.alias))
        .collect();
    let buffer = staging.as_ref().map(|s| s.buffer);
    let value = b.timeline.record(&b.host, |d, cmd| {
        // SAFETY: recording into the timeline's one-shot buffer.
        unsafe { record_set_up(d, cmd, &fresh, buffer, &copies) }
    });
    match (value, staging) {
        (Ok(v), Some(st)) => {
            if let Some(r) = registry().as_mut() {
                r.staging.push((v, st));
            }
        }
        (Err(e), st) => {
            once_warn(format!("Vulkan: image set-up was not submitted: {e:#}"));
            if let Some(st) = st {
                b.timeline.wait(&b.host, b.timeline.last(), 5_000_000_000);
                // SAFETY: nothing was submitted with it.
                unsafe { st.destroy(&b.host) };
            }
        }
        _ => {}
    }
}

/// The fills to do: (image, subresources, offset in staging), and the bytes.
fn plan_all(pending: &[Arc<Image>]) -> (Vec<(Arc<Image>, Vec<Sub>, u64)>, u64) {
    let mut jobs = Vec::new();
    let mut total = 0u64;
    for img in pending {
        img.pending.store(false, Ordering::Relaxed);
        if img.card_written() && !img.cpu_changed.load(Ordering::Relaxed) {
            continue;
        }
        img.cpu_changed.store(false, Ordering::Relaxed);
        match plan(img) {
            Ok(subs) => {
                let start = total;
                total += subs.iter().map(|s| s.bytes.next_multiple_of(16)).sum::<u64>();
                jobs.push((img.clone(), subs, start));
            }
            Err(e) => once_warn(format!("Vulkan: a texture is not filled: {e:#}")),
        }
    }
    (jobs, total)
}

fn staging_buffer(b: &Backend, total: u64) -> Option<Buffer> {
    if total == 0 {
        return None;
    }
    Buffer::new(&b.host, total, vk::BufferUsageFlags::TRANSFER_SRC, Place::Upload)
        .map_err(|e| once_warn(format!("Vulkan: no staging buffer of {} MiB: {e:#}", total >> 20)))
        .ok()
}

/// Detiles every job into the staging buffer; the copies to record.
fn detile_all(b: &Backend, st: &Buffer, jobs: &[(Arc<Image>, Vec<Sub>, u64)], total: u64) -> Vec<(vk::Image, Vec<vk::BufferImageCopy>)> {
    // SAFETY: the staging buffer is mapped and `total` bytes long.
    let linear = unsafe { std::slice::from_raw_parts_mut(st.mapped, total as usize) };
    let mut copies = Vec::new();
    for (img, subs, start) in jobs {
        match detile(b, img, subs, &mut linear[*start as usize..], *start) {
            Ok(regions) => {
                img.written[3].fetch_add(1, Ordering::Relaxed);
                STATS[1].fetch_add(1, Ordering::Relaxed);
                STATS[2].fetch_add(subs.iter().map(|s| s.bytes).sum::<u64>(), Ordering::Relaxed);
                copies.push((img.image, regions));
            }
            Err(e) => once_warn(format!("Vulkan: a texture is not filled: {e:#}")),
        }
    }
    copies
}

/// # Safety
/// Recording into `cmd`.
unsafe fn record_set_up(
    d: &victory_vk::ash::Device,
    cmd: vk::CommandBuffer,
    fresh: &[(vk::Image, vk::ImageAspectFlags, bool)],
    buffer: Option<vk::Buffer>,
    copies: &[(vk::Image, Vec<vk::BufferImageCopy>)],
) {
    // SAFETY: as the caller says.
    unsafe {
        for &(image, aspect, clear) in fresh {
            set_up_image(d, cmd, image, aspect, clear);
        }
        full_barrier(d, cmd);
        let Some(buffer) = buffer else { return };
        for (image, regions) in copies.iter().filter(|(_, r)| !r.is_empty()) {
            d.cmd_copy_buffer_to_image(cmd, buffer, *image, vk::ImageLayout::GENERAL, regions);
        }
        full_barrier(d, cmd);
    }
}

/// A new image into GENERAL, cleared unless it shares another's memory.
///
/// # Safety
/// Recording into `cmd`.
unsafe fn set_up_image(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, image: vk::Image, aspect: vk::ImageAspectFlags, clear: bool) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    };
    // SAFETY: as the caller says.
    unsafe {
        to_general(d, cmd, image, range, vk::ImageLayout::UNDEFINED);
        if !clear {
            return;
        }
        if aspect == vk::ImageAspectFlags::COLOR {
            d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
        } else {
            d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue::default(), &[range]);
        }
    }
}

/// One subresource of a fill: where it goes in the staging buffer.
struct Sub {
    slice: u32,
    /// The image's level, and the console's (texture quality leaves the
    /// largest out).
    mip: u32,
    console: u32,
    extent: vk::Extent3D,
    row: u64,
    rows: u64,
    bytes: u64,
}

/// The subresources a fill of `img` copies (plane 0, every mip and slice).
fn plan(img: &Image) -> anyhow::Result<Vec<Sub>> {
    let Some(texture) = crate::resource::texture_layout(&img.desc) else {
        anyhow::bail!("no console layout for format {}", img.desc.format);
    };
    let layout = &texture.planes[0].layout;
    let (element, block) = crate::resource::element(crate::resource::pc_format(img.desc.format))
        .ok_or_else(|| anyhow::anyhow!("no element size"))?;
    let vk_bytes = format_bytes(img.format);
    if vk_bytes != 0 && vk_bytes != element as u32 {
        anyhow::bail!("stand-in format {:?}", img.format);
    }
    let slices = if img.kind == vk::ImageType::TYPE_3D { 1 } else { img.layers };
    let mut out = Vec::new();
    for slice in 0..slices {
        for mip in 0..img.mips.min((layout.mips.len() as u32).saturating_sub(img.dropped)) {
            let w = (img.extent.width >> mip).max(1);
            let h = (img.extent.height >> mip).max(1);
            let dd = (img.extent.depth >> mip).max(1);
            let row = u64::from(w).div_ceil(block) * element;
            let rows = u64::from(h).div_ceil(block);
            out.push(Sub {
                slice,
                mip,
                console: mip + img.dropped,
                extent: vk::Extent3D { width: w, height: h, depth: dd },
                row,
                rows,
                bytes: row * rows * u64::from(dd),
            });
        }
    }
    Ok(out)
}

/// Detiles `img`'s title memory into `linear` (at `base` in the staging
/// buffer) and gives the copy regions.
fn detile(b: &Backend, img: &Image, subs: &[Sub], linear: &mut [u8], base: u64) -> anyhow::Result<Vec<vk::BufferImageCopy>> {
    let texture = crate::resource::texture_layout(&img.desc).ok_or_else(|| anyhow::anyhow!("no layout"))?;
    let layout = &texture.planes[0].layout;
    let tiled = super::super::read_title(b, img.key.address, layout.size)?;
    // The baseline the fingerprint check compares with (a first check
    // would only store it, missing a change made before it).
    img.hash.store(fingerprint(b, img.key.address, img.bytes).unwrap_or(0), Ordering::Relaxed);
    let mut regions = Vec::new();
    let mut at = 0usize;
    for s in subs {
        let out = &mut linear[at..at + s.bytes as usize];
        if layout.detile(&tiled, s.console, s.slice, out, s.row as usize, (s.row * s.rows) as usize).is_ok() {
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(base + at as u64)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: img.aspect() & !vk::ImageAspectFlags::STENCIL,
                        mip_level: s.mip,
                        base_array_layer: s.slice,
                        layer_count: 1,
                    })
                    .image_extent(s.extent),
            );
        }
        at += s.bytes.next_multiple_of(16) as usize;
    }
    Ok(regions)
}

/// Bytes per texel (or block) of the formats images take; 0 if unknown.
pub(crate) fn format_bytes(f: vk::Format) -> u32 {
    use vk::Format as F;
    match f {
        F::R32G32B32A32_SFLOAT | F::R32G32B32A32_UINT | F::R32G32B32A32_SINT => 16,
        F::R32G32B32_SFLOAT | F::R32G32B32_UINT | F::R32G32B32_SINT => 12,
        F::R16G16B16A16_SFLOAT
        | F::R16G16B16A16_UNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_SINT
        | F::R32G32_SFLOAT
        | F::R32G32_UINT
        | F::R32G32_SINT
        | F::D32_SFLOAT_S8_UINT => 8,
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGBA_SRGB_BLOCK | F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => 8,
        F::BC2_UNORM_BLOCK
        | F::BC2_SRGB_BLOCK
        | F::BC3_UNORM_BLOCK
        | F::BC3_SRGB_BLOCK
        | F::BC5_UNORM_BLOCK
        | F::BC5_SNORM_BLOCK
        | F::BC6H_UFLOAT_BLOCK
        | F::BC6H_SFLOAT_BLOCK
        | F::BC7_UNORM_BLOCK
        | F::BC7_SRGB_BLOCK => 16,
        F::R16G16_SFLOAT
        | F::R16G16_UNORM
        | F::R16G16_UINT
        | F::R16G16_SNORM
        | F::R16G16_SINT
        | F::R32_SFLOAT
        | F::R32_UINT
        | F::R32_SINT
        | F::D32_SFLOAT
        | F::D24_UNORM_S8_UINT
        | F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SRGB
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_SINT
        | F::B8G8R8A8_UNORM
        | F::B8G8R8A8_SRGB
        | F::A2B10G10R10_UNORM_PACK32
        | F::A2B10G10R10_UINT_PACK32
        | F::B10G11R11_UFLOAT_PACK32
        | F::E5B9G9R9_UFLOAT_PACK32 => 4,
        F::R8G8_UNORM
        | F::R8G8_UINT
        | F::R8G8_SNORM
        | F::R8G8_SINT
        | F::R16_UNORM
        | F::R16_SFLOAT
        | F::R16_UINT
        | F::R16_SNORM
        | F::R16_SINT
        | F::D16_UNORM
        | F::R5G6B5_UNORM_PACK16
        | F::A1R5G5B5_UNORM_PACK16
        | F::A4R4G4B4_UNORM_PACK16 => 2,
        F::R8_UNORM | F::R8_UINT | F::R8_SNORM | F::R8_SINT => 1,
        _ => 0,
    }
}

/// Textures the title fills that changed without an API call saying so
/// (movie frames, glyph atlases: the CPU writes them in place). At each
/// present their memory's fingerprint (three 16 KiB windows read through
/// the CPU view, [`fingerprint`]) is compared: every texture that changed
/// in the last 120 frames, and up to [`ROUND`] others in turn. A changed
/// one is filled again at the next flush. Pool memory is read through the
/// pool's CPU view (the GPU view's pages have no CPU mapping at their
/// address: the Bink planes, vk17's green frames). Images the card writes
/// as well are checked too: the card never changes title memory under an
/// image, so a changed fingerprint is the CPU's write. (The D3D12 path
/// watched pages for this; a fingerprint is the cheap form of the check.)
pub(crate) fn refresh(b: &Backend, frame: u64) {
    const ROUND: usize = 256;
    static CURSOR: Mutex<Option<Key>> = Mutex::new(None);
    let candidates: Vec<Arc<Image>> = {
        let guard = registry();
        let Some(r) = guard.as_ref() else { return };
        let mut cursor = lock(&CURSOR);
        let mut out: Vec<Arc<Image>> = r
            .images
            .values()
            .filter(|i| !i.attachment() && i.hot_until.load(Ordering::Relaxed) >= frame)
            .cloned()
            .collect();
        let start = cursor.unwrap_or(Key {
            address: 0,
            dimension: 0,
            width: 0,
            height: 0,
            depth_or_array: 0,
            mips: 0,
            element: 0,
            samples: 0,
            depth: false,
        });
        let mut last = None;
        for (k, i) in r.images.range(start..).filter(|(_, i)| !i.attachment()).take(ROUND) {
            out.push(i.clone());
            last = Some(*k);
        }
        *cursor = match last {
            Some(mut k) => {
                k.samples += 1;
                Some(k)
            }
            None => None,
        };
        out
    };
    let mut changed = Vec::new();
    for img in candidates {
        if img.bytes == 0 || img.pending.load(Ordering::Relaxed) {
            continue;
        }
        let Some(h) = fingerprint(b, img.key.address, img.bytes) else { continue };
        let was = img.hash.swap(h, Ordering::Relaxed);
        if was != 0 && was != h {
            img.hot_until.store(frame + 120, Ordering::Relaxed);
            img.cpu_changed.store(true, Ordering::Relaxed);
            changed_once(&img);
            if !img.pending.swap(true, Ordering::Relaxed) {
                changed.push(img);
            }
        }
    }
    if !changed.is_empty() {
        STATS[4].fetch_add(changed.len() as u64, Ordering::Relaxed);
        if let Some(r) = registry().as_mut() {
            r.pending.extend(changed);
        }
    }
}


/// A fingerprint (never 0) of three 16 KiB windows of `[at, at + len)`
/// read through its CPU view (`read_title`: pool pages through the pool's
/// CPU view); `None` where there is none.
fn fingerprint(b: &Backend, at: u64, len: u64) -> Option<u64> {
    const WINDOW: u64 = 16 << 10;
    if !super::super::cpu_readable(b, at, len) {
        return None;
    }
    let windows = if len <= 3 * WINDOW { vec![(0, len)] } else { vec![(0, WINDOW), ((len / 2) & !0xfff, WINDOW), (len - WINDOW, WINDOW)] };
    let mut hasher = blake3::Hasher::new();
    for (from, n) in windows {
        hasher.update(&super::super::read_title(b, at + from, n).ok()?);
    }
    let mut first = [0u8; 8];
    first.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    Some(u64::from_le_bytes(first).max(1))
}

/// Diagnostics: the first images the CPU changed, once each.
fn changed_once(img: &Image) {
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if cfg!(feature = "diagnostics") && img.hot_until.load(Ordering::Relaxed) != 0 && img.written[3].load(Ordering::Relaxed) <= 1 && LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
        tracing::info!(
            image = img.id,
            at = format_args!("{:#x}", img.key.address),
            format = ?img.format,
            width = img.extent.width,
            height = img.extent.height,
            card_written = img.card_written(),
            "Vulkan: a texture the CPU changed (filled again)"
        );
    }
}
