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
use victory_vk::memory::Buffer;

use super::{Backend, lock};

mod alloc;
mod fill;

use alloc::{Block, PLACES, USERS, allocate, memory_type_of, release};
pub(crate) use fill::{flush, format_bytes, refresh, released, written};
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
    /// The card writes it (or an image over the same memory): never filled
    /// from title memory.
    card_written: AtomicBool,
    /// It shares the memory of an earlier image at its address (one memory,
    /// several shapes, as on the console): made without a clear.
    alias: bool,
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
    /// Its title memory changed under the CPU since the last fill (the
    /// fingerprint): filled again even over memory the card also writes
    /// (the CPU's write is the newer content, as on the console).
    cpu_changed: AtomicBool,
}

impl Image {
    /// Drawn into (a target or depth): its title memory is never its
    /// content, so it is not fingerprinted.
    pub(crate) fn attachment(&self) -> bool {
        self.usage.intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
    }
}

impl Image {
    pub(crate) fn card_written(&self) -> bool {
        self.card_written.load(Ordering::Relaxed)
    }

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
    // One memory, several shapes: an image over memory another image has
    // shares its card memory (the largest one there).
    let alias = r
        .images
        .range(Key { dimension: 0, width: 0, height: 0, depth_or_array: 0, mips: 0, element: 0, samples: 0, depth: false, ..key }..)
        .take_while(|(k, _)| k.address == address)
        .map(|(_, i)| i.clone())
        .max_by_key(|i| lock(&PLACES).get(&i.id).map_or(0, |p| p.2));
    let made = make(b, r, key, desc, alias.as_ref());
    if let (Some(new), Some(_)) = (&made, &alias)
        && new.card_written()
    {
        // The card writes this memory: no image over it is filled from it.
        for (_, i) in r.images.range(Key { dimension: 0, width: 0, height: 0, depth_or_array: 0, mips: 0, element: 0, samples: 0, depth: false, ..key }..).take_while(|(k, _)| k.address == address) {
            i.card_written.store(true, Ordering::Relaxed);
        }
    }
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

fn make(b: &Backend, r: &mut Registry, key: Key, desc: &ResourceDesc, alias: Option<&Arc<Image>>) -> Option<Image> {
    let depth = key.depth;
    let card_written = desc.flags & (RENDER_TARGET | DEPTH_STENCIL | UNORDERED) != 0
        || alias.is_some_and(|a| a.card_written());
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
    // Over an earlier image's memory when it fits there.
    let shared = alias.and_then(|a| lock(&PLACES).get(&a.id).copied()).filter(|&(_, _, size, ty)| {
        needs.size <= size && needs.memory_type_bits & (1 << ty) != 0
    });
    if alias.is_some() && shared.is_none() {
        once_warn(format!("Vulkan: an image over the same title memory does not fit the first one's card memory ({} KiB)", needs.size >> 10));
    }
    let Some((memory, offset)) = shared.map(|(m, o, _, _)| (m, o)).or_else(|| allocate(b, &mut r.blocks, needs)) else {
        // SAFETY: never bound or used.
        unsafe { d.destroy_image(image, None) };
        once_warn(format!("Vulkan: no card memory for an image of {} KiB", needs.size >> 10));
        return None;
    };
    // SAFETY: fresh memory of a type the image takes.
    if unsafe { d.bind_image_memory(image, memory, if offset == u64::MAX { 0 } else { offset }) }.is_err() {
        if shared.is_none() {
            release(b, &mut r.blocks, memory, offset, needs.size);
        }
        // SAFETY: never used.
        unsafe { d.destroy_image(image, None) };
        return None;
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let ty = shared.map_or_else(|| memory_type_of(b, needs), |s| s.3);
    let size = shared.map_or(needs.size, |s| s.2);
    lock(&PLACES).insert(id, (memory, offset, size, ty));
    *lock(&USERS).entry((memory, offset)).or_insert(0) += 1;
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
        card_written: AtomicBool::new(card_written),
        alias: alias.is_some(),
        bytes,
        views: Mutex::new(Vec::new()),
        pending: AtomicBool::new(false),
        fresh: AtomicBool::new(true),
        written: [const { AtomicU64::new(0) }; 4],
        hash: AtomicU64::new(0),
        hot_until: AtomicU64::new(0),
        cpu_changed: AtomicBool::new(false),
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
            i.card_written(),
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
