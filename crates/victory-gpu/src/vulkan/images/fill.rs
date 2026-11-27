//! Filling images from title memory: queued images set up and detiled
//! into one staging buffer per flush, refilled at announced writes and
//! when the CPU stores into their memory (write tracking; a fingerprint
//! where that cannot be had); images freed with their memory.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::super::{Backend, lock};
use super::{Image, Key, PLACES, STATS, USERS, full_barrier, once_warn, registry, release, to_general};


/// A write over `[at, at + len)` (announced, or found by write tracking):
/// images there are filled again. Whether any image filled from title
/// memory lies there.
pub(crate) fn written(at: u64, len: u64) -> bool {
    let mut guard = registry();
    let Some(r) = guard.as_mut() else { return false };
    let (mut hit, mut any) = (Vec::new(), false);
    for img in (at >> 16..=(at + len.max(1) - 1) >> 16).filter_map(|page| r.over.get(&page)).flatten() {
        let k = &img.key;
        if img.card_written() || k.address >= at + len || k.address + img.bytes.max(1) <= at {
            continue;
        }
        any = true;
        if !img.pending.swap(true, Ordering::Relaxed) {
            hit.push(img.clone());
        }
    }
    r.pending.extend(hit);
    any
}

/// The title freed, unmapped or remapped `[at, at + len)`: images whose
/// bytes lie over any of it go once the card is done (counted by `why`,
/// [`Dropped`]). An image made there later is filled from what the memory
/// holds then. Where the title only showed other pages (`Remapped`), a
/// texture it fills stays: it is filled again from the pages there now
/// (the caller has [`written`] do that once they are shown), and goes
/// when no slot of the title names it; an image the card writes goes, as
/// its content was the old pages'.
pub(crate) fn released(b: &Backend, at: u64, len: u64, why: super::Dropped) {
    let mut guard = registry();
    let Some(r) = guard.as_mut() else { return };
    let from = r.first_over(at);
    let remapped = matches!(why, super::Dropped::Remapped);
    let gone: Vec<Key> = r
        .images
        .range(from..)
        .take_while(|(k, _)| k.address < at + len)
        .filter(|(k, i)| k.address + i.bytes.max(1) > at && (!remapped || i.card_written()))
        .map(|(k, _)| *k)
        .collect();
    for k in gone {
        r.drop_image(b, &k, why, at);
    }
}

impl super::Registry {
    /// The image of `key` goes once the card is done with it.
    pub(super) fn drop_image(&mut self, b: &Backend, key: &Key, why: super::Dropped, by: u64) {
        if let Some(img) = self.take(key) {
            self.retire(b, img, why, by);
        }
    }

    /// `img`, taken out of the registry, goes once the card is done.
    fn retire(&mut self, b: &Backend, img: Arc<Image>, why: super::Dropped, by: u64) {
        super::census::dropped(["freed", "unmapped", "remapped", "resource", "unnamed", "renamed"][why as usize], img.key.address, img.bytes, by);
        super::super::descriptors::image_gone(&img);
        self.dropped.push((b.timeline.last(), img));
        super::DROPPED[why as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Textures whose latest view names another first level than their image
/// has (`levels.rs`: levels the title streamed in, or let go of): an
/// image with the levels named is made and filled at this flush. The
/// texture's slots keep showing the old image until the card has filled
/// the new one ([`settled`]): work in flight reads the slots as it runs.
/// Without card memory for it the old image stays.
fn renamed(b: &Backend, r: &mut super::Registry) {
    for key in std::mem::take(&mut r.renamed) {
        let Some(old) = r.images.get(&key).cloned() else { continue };
        let first = old.wanted.load(Ordering::Relaxed);
        let on_its_way = r.remade.iter().any(|(n, _)| n.key == key && n.named == first);
        if first == old.named || old.card_written() || on_its_way {
            continue;
        }
        let Some(new) = super::make(b, r, key, &old.desc, None, first) else { continue };
        let new = Arc::new(new);
        r.queue(&new);
        r.remade.push((new, old.written[3].load(Ordering::Relaxed)));
    }
}

/// The images made again that the card has set up and filled take their
/// textures' places: the slots show them, and the old images go once the
/// card is done. One whose texture went meanwhile, or whose title has
/// since named yet another level, goes unused.
fn settled(b: &Backend, r: &mut super::Registry, completed: u64) {
    let ready = |n: &Arc<Image>| !n.pending.load(Ordering::Relaxed) && (1..=completed).contains(&n.last_use.load(Ordering::Relaxed));
    let (ready, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut r.remade).into_iter().partition(|(n, _)| ready(n));
    r.remade = waiting;
    for (new, fills) in ready {
        let wanted = r.images.get(&new.key).is_some_and(|old| !old.card_written() && old.wanted.load(Ordering::Relaxed) == new.named);
        let Some(old) = wanted.then(|| r.take(&new.key)).flatten() else {
            r.dropped.push((b.timeline.last(), new));
            continue;
        };
        r.place(&new);
        // A write over the texture since reached the old image only.
        if (old.written[3].load(Ordering::Relaxed) != fills || old.pending.load(Ordering::Relaxed)) && !new.pending.swap(true, Ordering::Relaxed) {
            r.pending.push(new.clone());
        }
        r.retire(b, old, super::Dropped::Renamed, u64::from(new.named));
    }
}

/// Textures the title fills that none of its descriptor slots names any
/// more (`named.rs`) go.
fn unnamed(b: &Backend, r: &mut super::Registry) {
    for key in super::named::let_go() {
        if r.images.get(&key).is_some_and(|i| !i.card_written()) {
            r.drop_image(b, &key, super::Dropped::Unnamed, 0);
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
        renamed(b, r);
        settled(b, r, completed);
        unnamed(b, r);
        let done = finished(b, r, completed);
        // (an image that went since it was queued is not set up or filled)
        let live = |i: &Arc<Image>| r.images.get(&i.key).is_some_and(|j| Arc::ptr_eq(i, j)) || r.remade.iter().any(|(n, _)| Arc::ptr_eq(i, n));
        let pending: Vec<Arc<Image>> = std::mem::take(&mut r.pending).into_iter().filter(live).collect();
        (pending, done)
    };
    for img in done {
        destroy(b, &img);
    }
    // At most FILL_BATCH bytes of fills a submission: one submission of a
    // world load's thousands of fills held the card for over a second
    // (vk60, vk61: the title's next submission waited on it).
    let mut batch: Vec<Arc<Image>> = Vec::new();
    let mut bytes = 0u64;
    for img in pending {
        bytes += img.bytes;
        batch.push(img);
        if bytes >= FILL_BATCH {
            set_up(b, &std::mem::take(&mut batch));
            bytes = 0;
        }
    }
    if !batch.is_empty() {
        set_up(b, &batch);
    }
}

/// Bytes of fills one set-up submission takes at most (one image may be
/// bigger).
const FILL_BATCH: u64 = 32 << 20;

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
        // Past the drop and past its last fill: a fill submitted after the
        // drop (the image was still pending then) must not land in freed
        // memory (vk61: the copy engine faulted on an unmapped page).
        .partition(|(v, i)| (*v).max(i.last_use.load(Ordering::Relaxed)) <= completed && Arc::strong_count(i) == 1);
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
    if let Ok(v) = value {
        for img in pending {
            img.last_use.fetch_max(v, Ordering::Relaxed);
        }
    }
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
    // Protect, then copy: a store into its memory from here on faults and
    // has it filled again (`crate::track`). Memory that cannot be tracked
    // is checked by fingerprint ([`refresh`]).
    // (the head of the layout that holds the image's levels: what follows
    // is other textures' memory, or not mapped)
    let tracked = crate::track::protect(img.key.address, img.bytes, true);
    img.tracked.store(tracked, Ordering::Relaxed);
    let tiled = super::super::read_head(b, img.key.address, img.bytes, layout.size)?;
    if !tracked {
        // The baseline the fingerprint check compares with (a first check
        // would only store it, missing a change made before it).
        img.hash.store(fingerprint(b, img.key.address, img.bytes).unwrap_or(0), Ordering::Relaxed);
    }
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

/// Textures whose memory write tracking does not cover (`crate::track`:
/// memory `xmem` does not map, pages left open as hot, and textures the
/// card writes too, whose title memory is not their content until the CPU
/// stores into it) that changed without an API call saying so. At each
/// present the images that are due have their memory's [`fingerprint`]
/// compared with the one of their last fill; a changed one is filled again
/// at the next flush. An image is due one present after its fill or a
/// change, then twice as late each time nothing changed, up to `SETTLED`
/// presents (`DYNAMIC` for one that ever changed: a glyph cache gets new
/// letters all the time). The old rule checked 256 images a present in
/// turn: with 35,000 live images new letters showed as stale boxes for a
/// hundred presents and more (vk47).
///
/// Pool memory is read through the pool's CPU view (the GPU view's pages
/// have no CPU mapping at their address: the Bink planes, vk17's green
/// frames). Images the card writes as well are checked too: the card never
/// changes title memory under an image, so a changed fingerprint is the
/// CPU's write.
pub(crate) fn refresh(b: &Backend, frame: u64) {
    /// Title bytes fingerprinted a present at most.
    const BUDGET: u64 = 16 << 20;
    /// Presents between the checks of an image that never changed, and of
    /// one that did, at most.
    const SETTLED: u32 = 256;
    const DYNAMIC: u32 = 4;
    let mut due: Vec<Arc<Image>> = {
        let guard = registry();
        let Some(r) = guard.as_ref() else { return };
        r.images.values().filter(|i| !i.attachment() && !i.tracked.load(Ordering::Relaxed) && i.next_check.load(Ordering::Relaxed) <= frame).cloned().collect()
    };
    // the longest overdue first: what the budget leaves waits a present
    due.sort_by_key(|i| i.next_check.load(Ordering::Relaxed));
    let (mut spent, mut changed) = (0, Vec::new());
    for img in due {
        if spent >= BUDGET {
            break;
        }
        if img.bytes == 0 || img.pending.load(Ordering::Relaxed) {
            continue;
        }
        let Some(h) = fingerprint(b, img.key.address, img.bytes) else {
            img.next_check.store(frame + u64::from(SETTLED), Ordering::Relaxed);
            continue;
        };
        spent += fingerprinted(img.bytes);
        let was = img.hash.swap(h, Ordering::Relaxed);
        let every = if was != 0 && was != h {
            img.dynamic.store(true, Ordering::Relaxed);
            img.cpu_changed.store(true, Ordering::Relaxed);
            changed_once(&img);
            if !img.pending.swap(true, Ordering::Relaxed) {
                changed.push(img.clone());
            }
            1
        } else {
            let most = if img.dynamic.load(Ordering::Relaxed) { DYNAMIC } else { SETTLED };
            (img.check_every.load(Ordering::Relaxed) * 2).min(most)
        };
        img.check_every.store(every, Ordering::Relaxed);
        img.next_check.store(frame + u64::from(every), Ordering::Relaxed);
    }
    CHECKED.fetch_add(spent, Ordering::Relaxed);
    if !changed.is_empty() {
        STATS[4].fetch_add(changed.len() as u64, Ordering::Relaxed);
        if let Some(r) = registry().as_mut() {
            r.pending.extend(changed);
        }
    }
}

/// Title bytes fingerprinted by [`refresh`] so far (diagnostics).
pub(crate) static CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Textures up to this size are fingerprinted whole; bigger ones in three
/// windows.
const WHOLE: u64 = 4 << 20;
const WINDOW: u64 = 16 << 10;

/// Bytes [`fingerprint`] reads of an image of `len` bytes.
fn fingerprinted(len: u64) -> u64 {
    if len <= WHOLE { len } else { 3 * WINDOW }
}

/// A fingerprint (never 0) of `[at, at + len)` read through its CPU view
/// (`read_title`: pool pages through the pool's CPU view); `None` where
/// there is none. Textures up to [`WHOLE`] are hashed whole: a glyph cache
/// the CPU updates in small rectangles anywhere (vk40: UI text with letters
/// missing) changes outside any window. Bigger ones are hashed in three
/// 16 KiB windows (movie frames, rewritten whole).
fn fingerprint(b: &Backend, at: u64, len: u64) -> Option<u64> {
    if !super::super::cpu_readable(b, at, len) {
        return None;
    }
    let windows = if len <= WHOLE { vec![(0, len)] } else { vec![(0, WINDOW), ((len / 2) & !0xfff, WINDOW), (len - WINDOW, WINDOW)] };
    let mut hasher = blake3::Hasher::new();
    for (from, n) in windows {
        hasher.update(&super::super::read_title(b, at + from, n).ok()?);
    }
    let mut first = [0u8; 8];
    first.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    Some(u64::from_le_bytes(first).max(1))
}

/// Diagnostics: live images filled from title memory that write tracking
/// covers and that are checked by fingerprint, and the six filled most
/// often.
pub(super) fn tracking() -> String {
    let guard = registry();
    let Some(r) = guard.as_ref() else { return String::new() };
    let filled: Vec<&Arc<Image>> = r.images.values().filter(|i| i.written[3].load(Ordering::Relaxed) > 0).collect();
    let tracked = filled.iter().filter(|i| i.tracked.load(Ordering::Relaxed)).count();
    let polled = r.images.values().filter(|i| !i.attachment() && !i.tracked.load(Ordering::Relaxed)).count();
    let mut most = filled;
    most.sort_by_key(|i| std::cmp::Reverse(i.written[3].load(Ordering::Relaxed)));
    let most: Vec<String> = most
        .iter()
        .take(6)
        .map(|i| format!("{}x at {:#x} {:?} {}x{} {}K", i.written[3].load(Ordering::Relaxed), i.key.address, i.format, i.extent.width, i.extent.height, i.bytes >> 10))
        .collect();
    format!("tracked {tracked}, checked by fingerprint {polled}; most filled: {}", most.join("; "))
}

/// Diagnostics: the first images the CPU changed, once each.
fn changed_once(img: &Image) {
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if cfg!(feature = "diagnostics") && img.written[3].load(Ordering::Relaxed) <= 1 && LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
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
