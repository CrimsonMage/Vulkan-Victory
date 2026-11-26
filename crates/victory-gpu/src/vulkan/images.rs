//! M2b step (b): card images for title textures and render targets
//! (`docs/vulkan-host.md`, "Images over title memory: no").
//!
//! A title texture lives in title memory in the console's tiled layout; the
//! card samples a `VkImage` of its own. Images are keyed by the title
//! address and the texture's shape, as the console knows a texture by its
//! memory: a render target's resource and a placed view of the same memory
//! name one image.
//!
//! - **Made** when the first view, target or copy names the texture.
//! - **Filled** (detiled on the CPU with `victory_formats`' layouts into a
//!   staging buffer, then copied on the timeline) when made, and again at
//!   every announced write over its memory (DirectStorage completion, file
//!   reads, page mappings: `vulkan::title_memory_written`). Textures the
//!   card writes (targets, depth, unordered access) are not filled: the
//!   card's copy is theirs.
//! - **Layouts**: every image lives in `GENERAL` from its creation on
//!   (attachments, sampling, storage and copies all take it). The title's
//!   barriers become global memory barriers; there is no per-image state
//!   to track (shortcut, KISS: NVIDIA does not lose speed in `GENERAL`).
//! - **Freed** when the title releases the memory under it
//!   (`vulkan::title_memory_released`), after the timeline passes what may
//!   still use it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::{Backend, lock};
use crate::object::This;
use crate::resource::ResourceDesc;

/// D3D12 resource flags.
const RENDER_TARGET: u32 = 1;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED: u32 = 4;
const BUFFER: u32 = 1;
const TEXTURE_1D: u32 = 2;
const TEXTURE_3D: u32 = 4;

/// The shape that, with the address, names an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Key {
    pub(crate) address: u64,
    dimension: u32,
    width: u32,
    height: u32,
    depth_or_array: u16,
    mips: u16,
    /// Bytes per element (views may name another format of the same size).
    element: u32,
    samples: u32,
    depth: bool,
}

/// What view of an image a descriptor or attachment wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ViewKey {
    pub(crate) format: vk::Format,
    pub(crate) kind: vk::ImageViewType,
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) base_mip: u32,
    pub(crate) mips: u32,
    pub(crate) base_layer: u32,
    pub(crate) layers: u32,
    pub(crate) components: [vk::ComponentSwizzle; 4],
    /// For storage descriptors (the view's usage is narrowed to storage).
    pub(crate) storage: bool,
    pub(crate) attachment: bool,
}

pub(crate) struct Image {
    pub(crate) id: u64,
    pub(crate) key: Key,
    pub(crate) image: vk::Image,
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent3D,
    pub(crate) mips: u32,
    pub(crate) layers: u32,
    pub(crate) kind: vk::ImageType,
    pub(crate) samples: u32,
    pub(crate) usage: vk::ImageUsageFlags,
    /// The title's description (format, layout).
    pub(crate) desc: ResourceDesc,
    /// The card writes it: never filled from title memory.
    pub(crate) card_written: bool,
    /// Bytes of its console layout.
    pub(crate) bytes: u64,
    views: Mutex<Vec<(ViewKey, vk::ImageView)>>,
    /// A fill is queued.
    pending: AtomicBool,
    /// Not yet in GENERAL (the next flush moves and clears it).
    fresh: AtomicBool,
    /// Diagnostics: draws into it as an attachment, clears, copies into it,
    /// fills.
    pub(crate) written: [AtomicU64; 4],
    /// The fingerprint of its title memory when last checked (0: never),
    /// and the frame until which it is checked every present (it changed
    /// lately: a movie frame, a glyph atlas).
    hash: AtomicU64,
    hot_until: AtomicU64,
}

impl Image {
    pub(crate) fn depth(&self) -> bool {
        self.key.depth
    }

    pub(crate) fn aspect(&self) -> vk::ImageAspectFlags {
        aspect_of(self.format)
    }

    /// The image view `v` (made the first time).
    pub(crate) fn view(&self, b: &Backend, v: ViewKey) -> Option<vk::ImageView> {
        let mut views = lock(&self.views);
        if let Some(&(_, view)) = views.iter().find(|(k, _)| *k == v) {
            return Some(view);
        }
        let usage = if v.storage {
            vk::ImageUsageFlags::STORAGE
        } else if v.attachment {
            self.usage
                & (vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        } else {
            vk::ImageUsageFlags::SAMPLED
        };
        if !self.usage.contains(usage) || usage.is_empty() {
            return None;
        }
        let mut narrowed = vk::ImageViewUsageCreateInfo::default().usage(usage);
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(v.kind)
            .format(v.format)
            .components(vk::ComponentMapping {
                r: v.components[0],
                g: v.components[1],
                b: v.components[2],
                a: v.components[3],
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: v.aspect,
                base_mip_level: v.base_mip.min(self.mips - 1),
                level_count: v.mips.clamp(1, self.mips - v.base_mip.min(self.mips - 1)),
                base_array_layer: v.base_layer.min(self.layers - 1),
                layer_count: v.layers.clamp(1, self.layers - v.base_layer.min(self.layers - 1)),
            })
            .push_next(&mut narrowed);
        // SAFETY: a live image of ours; the view goes with it.
        match unsafe { b.host.device.create_image_view(&info, None) } {
            Ok(view) => {
                views.push((v, view));
                Some(view)
            }
            Err(e) => {
                once_warn(format!("Vulkan: an image view was refused ({e}): {v:?} of {:?}", self.format));
                None
            }
        }
    }
}

fn once_warn(text: String) {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
        tracing::warn!("{text}");
    }
}

pub(crate) fn aspect_of(format: vk::Format) -> vk::ImageAspectFlags {
    use vk::Format as F;
    match format {
        F::D32_SFLOAT | F::D16_UNORM | F::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        F::D24_UNORM_S8_UINT | F::D32_SFLOAT_S8_UINT | F::D16_UNORM_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        F::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// The depth format of a DXGI depth family (typeless, depth or the
/// shader views of it), if `dxgi` is one.
pub(crate) fn depth_format(dxgi: u32) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match dxgi {
        19..=22 => F::D32_SFLOAT_S8_UINT,
        39..=41 => F::D32_SFLOAT,
        44..=47 | 118..=120 => F::D24_UNORM_S8_UINT,
        53 | 55 | 56 => F::D16_UNORM,
        _ => return None,
    })
}

/// Whether a view format of a depth family reads stencil.
pub(crate) fn reads_stencil(dxgi: u32) -> bool {
    matches!(dxgi, 21 | 47 | 120)
}

/// The Vulkan format of a title (colour) format; for textures the card
/// writes (`stand_in`), with the stand-ins the card needs to render to
/// them (`resource::target_format`). Textures the title fills keep their
/// own format, so their texels copy as they are.
pub(crate) fn color_format(dxgi: u32, stand_in: bool) -> Option<vk::Format> {
    let pc = crate::resource::pc_format(dxgi);
    let pc = match pc {
        86 | 115 if stand_in => 28,
        67 if stand_in => 10,
        115 => return Some(vk::Format::A4R4G4B4_UNORM_PACK16),
        other => other,
    };
    victory_vk::gfx10::from_dxgi(pc)
}

#[derive(Default)]
struct Registry {
    images: BTreeMap<Key, Arc<Image>>,
    /// Destroyed once the timeline passes the value.
    dropped: Vec<(u64, Arc<Image>)>,
    staging: Vec<(u64, Buffer)>,
    pending: Vec<Arc<Image>>,
    blocks: Vec<Block>,
}

// SAFETY: the staging buffers' mapped pointers are used only under the lock.
unsafe impl Send for Registry {}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Images made, fills done, bytes filled, images that could not be made,
/// refills found by fingerprint.
pub(crate) static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn registry() -> std::sync::MutexGuard<'static, Option<Registry>> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

// ---- card memory: 256 MiB blocks with first-fit free lists ----

const BLOCK: u64 = 256 << 20;

struct Block {
    memory: vk::DeviceMemory,
    type_index: u32,
    free: Vec<(u64, u64)>,
}

fn allocate(
    b: &Backend,
    blocks: &mut Vec<Block>,
    needs: vk::MemoryRequirements,
) -> Option<(vk::DeviceMemory, u64)> {
    let type_index = b.host.memory_type(
        needs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    if needs.size > BLOCK / 4 {
        // SAFETY: valid allocate info.
        let m = unsafe {
            b.host.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(type_index),
                None,
            )
        }
        .ok()?;
        return Some((m, u64::MAX));
    }
    let take = |blk: &mut Block| -> Option<u64> {
        let i = blk.free.iter().position(|&(at, len)| {
            let start = at.next_multiple_of(needs.alignment.max(1));
            start + needs.size <= at + len
        })?;
        let (at, len) = blk.free[i];
        let start = at.next_multiple_of(needs.alignment.max(1));
        let end = start + needs.size;
        blk.free.remove(i);
        if start > at {
            blk.free.push((at, start - at));
        }
        if end < at + len {
            blk.free.push((end, at + len - end));
        }
        Some(start)
    };
    for blk in blocks.iter_mut().filter(|k| k.type_index == type_index) {
        if let Some(at) = take(blk) {
            return Some((blk.memory, at));
        }
    }
    // SAFETY: valid allocate info.
    let memory = unsafe {
        b.host.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(BLOCK)
                .memory_type_index(type_index),
            None,
        )
    }
    .ok()?;
    let mut blk = Block {
        memory,
        type_index,
        free: vec![(0, BLOCK)],
    };
    let at = take(&mut blk)?;
    blocks.push(blk);
    Some((memory, at))
}

fn release(b: &Backend, blocks: &mut [Block], memory: vk::DeviceMemory, at: u64, size: u64) {
    if at == u64::MAX {
        // SAFETY: a dedicated allocation nothing uses any more.
        unsafe { b.host.device.free_memory(memory, None) };
        return;
    }
    if let Some(blk) = blocks.iter_mut().find(|k| k.memory == memory) {
        blk.free.push((at, size));
        // Merge neighbours.
        blk.free.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(blk.free.len());
        for &(a, l) in &blk.free {
            match merged.last_mut() {
                Some((pa, pl)) if *pa + *pl == a => *pl += l,
                _ => merged.push((a, l)),
            }
        }
        blk.free = merged;
    }
}

/// Where an image's memory is: (memory, offset or MAX for dedicated, size).
static PLACES: Mutex<BTreeMap<u64, (vk::DeviceMemory, u64, u64)>> = Mutex::new(BTreeMap::new());

// ---- making ----

fn key_of(address: u64, desc: &ResourceDesc) -> Option<Key> {
    if desc.dimension == BUFFER || address == 0 {
        return None;
    }
    let element = crate::resource::element(crate::resource::pc_format(desc.format)).map_or(0, |(e, _)| e as u32);
    Some(Key {
        address,
        dimension: desc.dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height.max(1),
        depth_or_array: desc.depth_or_array_size.max(1),
        mips: desc.mip_levels,
        element,
        samples: desc.sample_count.max(1),
        depth: desc.flags & DEPTH_STENCIL != 0,
    })
}

/// The image of title texture `resource` (made the first time).
pub(crate) fn of_resource(b: &Backend, resource: This) -> Option<Arc<Image>> {
    let (data, desc) = crate::resource::texture_memory(resource)?;
    at(b, data as u64, &desc)
}

/// The image of the texture described by `desc` at title address
/// `address` (made the first time).
pub(crate) fn at(b: &Backend, address: u64, desc: &ResourceDesc) -> Option<Arc<Image>> {
    let key = key_of(address, desc)?;
    let mut guard = registry();
    let r = guard.get_or_insert_with(Registry::default);
    if let Some(i) = r.images.get(&key) {
        return Some(i.clone());
    }
    let made = make(b, r, key, desc);
    match made {
        Some(img) => {
            let img = Arc::new(img);
            // Into GENERAL and cleared at the next flush (ahead of anything
            // that uses it); filled after that unless the card writes it.
            img.pending.store(true, Ordering::Relaxed);
            r.pending.push(img.clone());
            r.images.insert(key, img.clone());
            STATS[0].fetch_add(1, Ordering::Relaxed);
            Some(img)
        }
        None => {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

fn make(b: &Backend, r: &mut Registry, key: Key, desc: &ResourceDesc) -> Option<Image> {
    let depth = key.depth;
    let card_written = desc.flags & (RENDER_TARGET | DEPTH_STENCIL | UNORDERED) != 0;
    let format = if depth {
        depth_format(desc.format)?
    } else {
        color_format(desc.format, card_written)?
    };
    let (kind, extent, layers) = match desc.dimension {
        TEXTURE_1D => (vk::ImageType::TYPE_1D, vk::Extent3D { width: key.width, height: 1, depth: 1 }, u32::from(key.depth_or_array)),
        TEXTURE_3D => (
            vk::ImageType::TYPE_3D,
            vk::Extent3D { width: key.width, height: key.height, depth: u32::from(key.depth_or_array) },
            1,
        ),
        _ => (vk::ImageType::TYPE_2D, vk::Extent3D { width: key.width, height: key.height, depth: 1 }, u32::from(key.depth_or_array)),
    };
    let full = 32 - key.width.max(key.height).max(extent.depth).max(1).leading_zeros();
    let mips = match desc.mip_levels {
        0 => full,
        n => u32::from(n).min(full),
    };
    let mut usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if depth {
        usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    } else {
        if desc.flags & RENDER_TARGET != 0 {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if desc.flags & UNORDERED != 0 {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
    }
    let samples = vk::SampleCountFlags::from_raw(key.samples);
    let mut flags = vk::ImageCreateFlags::empty();
    if !depth {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE;
    }
    if kind == vk::ImageType::TYPE_2D && layers >= 6 && key.width == key.height && key.samples == 1 {
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    if kind == vk::ImageType::TYPE_3D && !depth {
        flags |= vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE;
    }
    // Drop what the driver does not take for the format, storage first.
    let supported = |usage: vk::ImageUsageFlags, flags: vk::ImageCreateFlags| {
        // SAFETY: a plain query.
        unsafe {
            b.host.instance.get_physical_device_image_format_properties(
                b.host.physical,
                format,
                kind,
                vk::ImageTiling::OPTIMAL,
                usage,
                flags,
            )
        }
        .is_ok()
    };
    if !supported(usage, flags) {
        usage &= !vk::ImageUsageFlags::STORAGE;
        if !supported(usage, flags) {
            usage &= !vk::ImageUsageFlags::COLOR_ATTACHMENT;
            if !supported(usage, flags) {
                once_warn(format!("Vulkan: no image for format {format:?} ({:?}, {:?})", kind, flags));
                return None;
            }
        }
    }
    let info = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(kind)
        .format(format)
        .extent(extent)
        .mip_levels(mips)
        .array_layers(layers.max(1))
        .samples(samples)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let d = &b.host.device;
    // SAFETY: valid create info; destroyed with the registry entry.
    let image = unsafe { d.create_image(&info, None) }.ok()?;
    // SAFETY: a live image.
    let needs = unsafe { d.get_image_memory_requirements(image) };
    let Some((memory, offset)) = allocate(b, &mut r.blocks, needs) else {
        // SAFETY: never bound or used.
        unsafe { d.destroy_image(image, None) };
        once_warn(format!("Vulkan: no card memory for an image of {} KiB", needs.size >> 10));
        return None;
    };
    // SAFETY: fresh memory of a type the image takes.
    if unsafe { d.bind_image_memory(image, memory, if offset == u64::MAX { 0 } else { offset }) }.is_err() {
        release(b, &mut r.blocks, memory, offset, needs.size);
        // SAFETY: never used.
        unsafe { d.destroy_image(image, None) };
        return None;
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    lock(&PLACES).insert(id, (memory, offset, needs.size));
    let bytes = crate::resource::texture_layout(desc).map_or(0, |t| t.size);
    Some(Image {
        id,
        key,
        image,
        format,
        extent,
        mips,
        layers: layers.max(1),
        kind,
        samples: key.samples,
        usage,
        desc: *desc,
        card_written,
        bytes,
        views: Mutex::new(Vec::new()),
        pending: AtomicBool::new(false),
        fresh: AtomicBool::new(true),
        written: [const { AtomicU64::new(0) }; 4],
        hash: AtomicU64::new(0),
        hot_until: AtomicU64::new(0),
    })
}

/// # Safety
/// Recording into `cmd`.
pub(crate) unsafe fn to_general(
    d: &victory_vk::ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    from: vk::ImageLayout,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(from)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(image)
        .subresource_range(range);
    let barriers = [barrier];
    // SAFETY: as the caller says.
    unsafe { d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&barriers)) };
}

/// Everything before against everything after.
///
/// # Safety
/// Recording into `cmd`, outside a render pass.
pub(crate) unsafe fn full_barrier(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    // SAFETY: as the caller says.
    unsafe { d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&barrier)) };
}

// ---- filling ----

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
        if !img.card_written && k.address + img.bytes.max(1) > at && !img.pending.swap(true, Ordering::Relaxed) {
            hit.push(img.clone());
        }
    }
    r.pending.extend(hit);
}

/// The title freed `[at, at + len)`: images there go once the card is done.
pub(crate) fn released(b: &Backend, at: u64, len: u64) {
    let mut guard = registry();
    let Some(r) = guard.as_mut() else { return };
    let gone: Vec<Key> = r
        .images
        .keys()
        .filter(|k| k.address >= at && k.address < at + len)
        .copied()
        .collect();
    let value = b.timeline.last();
    for k in gone {
        if let Some(img) = r.images.remove(&k) {
            super::descriptors::image_gone(img.id);
            r.dropped.push((value, img));
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
        // Staging and images the card is done with.
        let mut i = 0;
        while i < r.staging.len() {
            if r.staging[i].0 <= completed {
                let (_, buf) = r.staging.swap_remove(i);
                // SAFETY: the card has passed its last use.
                unsafe { buf.destroy(&b.host) };
            } else {
                i += 1;
            }
        }
        let mut done = Vec::new();
        let mut i = 0;
        while i < r.dropped.len() {
            if r.dropped[i].0 <= completed && Arc::strong_count(&r.dropped[i].1) == 1 {
                done.push(r.dropped.swap_remove(i).1);
            } else {
                i += 1;
            }
        }
        for img in &done {
            if let Some((memory, at, size)) = lock(&PLACES).remove(&img.id) {
                release(b, &mut r.blocks, memory, at, size);
            }
        }
        (std::mem::take(&mut r.pending), done)
    };
    for img in done {
        // SAFETY: nothing uses it any more.
        unsafe {
            for (_, v) in lock(&img.views).drain(..) {
                b.host.device.destroy_image_view(v, None);
            }
            b.host.device.destroy_image(img.image, None);
        }
    }
    if pending.is_empty() {
        return;
    }
    // One staging buffer and one submission for everything queued: new
    // images into GENERAL (cleared), then the fills.
    let mut jobs = Vec::new();
    let mut total = 0u64;
    for img in &pending {
        img.pending.store(false, Ordering::Relaxed);
        if img.card_written {
            continue;
        }
        match plan(img) {
            Ok(subs) => {
                let start = total;
                total += subs.iter().map(|s| s.bytes.next_multiple_of(16)).sum::<u64>();
                jobs.push((img.clone(), subs, start));
            }
            Err(e) => once_warn(format!("Vulkan: a texture is not filled: {e:#}")),
        }
    }
    let staging = if total > 0 {
        match Buffer::new(&b.host, total, vk::BufferUsageFlags::TRANSFER_SRC, Place::Upload) {
            Ok(s) => Some(s),
            Err(e) => {
                once_warn(format!("Vulkan: no staging buffer of {} MiB: {e:#}", total >> 20));
                None
            }
        }
    } else {
        None
    };
    let mut copies: Vec<(vk::Image, Vec<vk::BufferImageCopy>)> = Vec::new();
    if let Some(st) = &staging {
        // SAFETY: the staging buffer is mapped and `total` bytes long.
        let linear = unsafe { std::slice::from_raw_parts_mut(st.mapped, total as usize) };
        for (img, subs, start) in &jobs {
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
    }
    let fresh: Vec<(vk::Image, vk::ImageAspectFlags)> = pending
        .iter()
        .filter(|i| i.fresh.swap(false, Ordering::Relaxed))
        .map(|i| (i.image, i.aspect()))
        .collect();
    let buffer = staging.as_ref().map(|s| s.buffer);
    let value = b.timeline.record(&b.host, |d, cmd| {
        // SAFETY: recording into the timeline's one-shot buffer.
        unsafe {
            for &(image, aspect) in &fresh {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                };
                to_general(d, cmd, image, range, vk::ImageLayout::UNDEFINED);
                if aspect == vk::ImageAspectFlags::COLOR {
                    d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
                } else {
                    d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue::default(), &[range]);
                }
            }
            full_barrier(d, cmd);
            if let Some(buffer) = buffer {
                for (image, regions) in &copies {
                    if !regions.is_empty() {
                        d.cmd_copy_buffer_to_image(cmd, buffer, *image, vk::ImageLayout::GENERAL, regions);
                    }
                }
                full_barrier(d, cmd);
            }
        }
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

/// One subresource of a fill: where it goes in the staging buffer.
struct Sub {
    slice: u32,
    mip: u32,
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
        for mip in 0..img.mips.min(layout.mips.len() as u32) {
            let w = (img.extent.width >> mip).max(1);
            let h = (img.extent.height >> mip).max(1);
            let dd = (img.extent.depth >> mip).max(1);
            let row = u64::from(w).div_ceil(block) * element;
            let rows = u64::from(h).div_ceil(block);
            out.push(Sub {
                slice,
                mip,
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
    let tiled = super::read_title(b, img.key.address, layout.size)?;
    let mut regions = Vec::new();
    let mut at = 0usize;
    for s in subs {
        let out = &mut linear[at..at + s.bytes as usize];
        if layout.detile(&tiled, s.mip, s.slice, out, s.row as usize, (s.row * s.rows) as usize).is_ok() {
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
/// present their memory's fingerprint (three 16 KiB windows,
/// `texture::fingerprint`) is compared: every texture that changed in the
/// last 120 frames, and up to [`ROUND`] others in turn. A changed one is
/// filled again at the next flush. (The D3D12 path watched pages for this;
/// a fingerprint of a few windows is the cheap form of the same check.)
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
            .filter(|i| !i.card_written && i.hot_until.load(Ordering::Relaxed) >= frame)
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
        for (k, i) in r.images.range(start..).filter(|(_, i)| !i.card_written).take(ROUND) {
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
        // Only memory with a CPU view at its own address.
        if !super::cpu_readable(b, img.key.address, img.bytes) {
            continue;
        }
        let h = crate::texture::fingerprint(img.key.address, img.bytes).max(1);
        let was = img.hash.swap(h, Ordering::Relaxed);
        if was != 0 && was != h {
            img.hot_until.store(frame + 120, Ordering::Relaxed);
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

/// Diagnostics: the image of `resource` and what wrote it.
pub(crate) fn describe(b: &Backend, resource: This) -> String {
    match of_resource(b, resource) {
        Some(i) => format!(
            "image {} at {:#x} {:?} {}x{} card_written={} draws={} clears={} copies={} fills={}",
            i.id,
            i.key.address,
            i.format,
            i.extent.width,
            i.extent.height,
            i.card_written,
            i.written[0].load(Ordering::Relaxed),
            i.written[1].load(Ordering::Relaxed),
            i.written[2].load(Ordering::Relaxed),
            i.written[3].load(Ordering::Relaxed),
        ),
        None => "no image".into(),
    }
}

/// Images alive, for the 10 s line.
pub(crate) fn count() -> usize {
    registry().as_ref().map_or(0, |r| r.images.len())
}
