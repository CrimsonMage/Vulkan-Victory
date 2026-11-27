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
use std::sync::{Arc, Mutex, OnceLock};

use victory_vk::ash::vk;
use victory_vk::memory::Buffer;

use super::{Backend, lock};

mod alloc;
mod fill;

use alloc::{Block, PLACES, USERS, allocate, memory_type_of, release};
pub(crate) use alloc::used_mb;
pub(crate) use fill::{flush, format_bytes, refresh, released, written};
use crate::object::This;
use crate::resource::ResourceDesc;

/// D3D12 resource flags.
const RENDER_TARGET: u32 = 1;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED: u32 = 4;
const BUFFER: u32 = 1;
const TEXTURE_1D: u32 = 2;
const TEXTURE_2D: u32 = 3;
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
    /// The console's largest levels the image leaves out (texture quality,
    /// `crate::quality`): its level 0 is the console's level `dropped`.
    pub(crate) dropped: u32,
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
    /// The first bytes of that layout known to be this texture's own, and
    /// the most detailed level they reach. The console stores a texture's
    /// small levels first and level 0 last, and a title that streams
    /// levels in places a texture whose large levels are not there yet so
    /// that only its small ones lie in its allocation: the rest of the
    /// layout is other textures' memory (the title's views say from which
    /// level on it is to be read). All of it for a texture a resource use
    /// made; through the most detailed level a view has named otherwise.
    pub(crate) own: AtomicU64,
    own_level: std::sync::atomic::AtomicU32,
    views: Mutex<Vec<(ViewKey, vk::ImageView)>>,
    /// A depth image the title also writes through a storage view (a
    /// compute program storing into a depth buffer's memory, plain memory
    /// on the console): the colour image over the same memory that takes
    /// those stores ([`written_plane`]), copied into the depth plane
    /// before the depth image is next attached.
    plane: OnceLock<Arc<Image>>,
    /// The same for its stencil plane (an 8-bit integer image).
    plane_stencil: OnceLock<Arc<Image>>,
    /// A fill is queued.
    pending: AtomicBool,
    /// The ticket of the last submission of ours that names it (its
    /// set-up and fills): it is destroyed only once the card is past both
    /// that and the ticket it was dropped at.
    pub(crate) last_use: AtomicU64,
    /// Not yet in GENERAL (the next flush moves and clears it).
    fresh: AtomicBool,
    /// Diagnostics: draws into it as an attachment, clears, copies into it,
    /// fills.
    pub(crate) written: [AtomicU64; 4],
    /// The fingerprint of its title memory when last filled or checked
    /// (0: never).
    hash: AtomicU64,
    /// The present its fingerprint is checked at next, and the presents
    /// between checks: 1 after a fill or a change, doubling while nothing
    /// changes (`fill::refresh`).
    next_check: AtomicU64,
    check_every: std::sync::atomic::AtomicU32,
    /// The CPU changed it after its first fill (a glyph cache, a movie
    /// plane): checked often from then on.
    dynamic: AtomicBool,
    /// Its last fill protected its title memory against stores
    /// (`crate::track`): a store has it filled again, and no fingerprint
    /// is taken.
    tracked: AtomicBool,
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
    /// The colour images that take the stores into this depth image's
    /// depth plane and its stencil plane.
    pub(crate) fn planes(&self) -> [Option<&Arc<Image>>; 2] {
        [self.plane.get(), self.plane_stencil.get()]
    }

    pub(crate) fn card_written(&self) -> bool {
        self.card_written.load(Ordering::Relaxed)
    }

    /// An unordered view of the image was made: the card writes it from
    /// now on, whatever its resource said.
    pub(crate) fn written_by_card(&self) {
        self.card_written.store(true, Ordering::Relaxed);
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
        // The console's levels to the image's: a view from a level left out
        // starts at the image's level 0, with as many levels fewer.
        let base = v.base_mip.saturating_sub(self.dropped).min(self.mips - 1);
        let count = v.mips.saturating_sub(self.dropped.saturating_sub(v.base_mip));
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
                base_mip_level: base,
                level_count: count.clamp(1, self.mips - base),
                base_array_layer: v.base_layer.min(self.layers - 1),
                layer_count: v.layers.clamp(1, self.layers - v.base_layer.min(self.layers - 1)),
            })
            .push_next(&mut narrowed);
        // SAFETY: a live image of ours; the view goes with it.
        match victory_vk::account::traced(&b.host, "image view", || unsafe { b.host.device.create_image_view(&info, None) }) {
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

pub(crate) fn once_warn(text: String) {
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
        // The console's 24-bit depth formats are 32-bit float planes too
        // (its hardware has Z16 and Z32F): the title writes one with a
        // 32_FLOAT view of the same memory (the material classes the
        // tiled lighting tests EQUAL against).
        44..=47 | 118..=120 => F::D32_SFLOAT_S8_UINT,
        53 | 55 | 56 => F::D16_UNORM,
        _ => return None,
    })
}

/// Whether a view format of a depth family reads stencil.
pub(crate) fn reads_stencil(dxgi: u32) -> bool {
    matches!(dxgi, 22 | 47 | 120)
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
    /// The most title bytes any image made so far covers: how far before
    /// an address an image over it can start.
    widest: u64,
    /// The images the title fills, by each 64 KiB page of title memory
    /// they lie over: a write names its pages ([`fill::written`]).
    over: std::collections::HashMap<u64, Vec<Arc<Image>>>,
}

impl Registry {
    /// Puts `img` into, or takes it out of, the pages it lies over.
    fn page(&mut self, img: &Arc<Image>, add: bool) {
        // (the pool's view keeps card pages for what programs read by
        // address before it keeps them for a texture's memory)
        super::paging::texture_over(img.key.address, img.bytes.max(1), add);
        for page in img.key.address >> 16..=(img.key.address + img.bytes.max(1) - 1) >> 16 {
            let there = self.over.entry(page).or_default();
            there.retain(|i| !Arc::ptr_eq(i, img));
            if add {
                there.push(img.clone());
            } else if there.is_empty() {
                self.over.remove(&page);
            }
        }
    }

    /// Takes the image of `key` out, for its dropping.
    fn take(&mut self, key: &Key) -> Option<Arc<Image>> {
        let img = self.images.remove(key)?;
        if !img.card_written() || self.over.contains_key(&(key.address >> 16)) {
            self.page(&img, false);
        }
        Some(img)
    }

    /// The key every image that may cover `at` sorts at or after.
    fn first_over(&self, at: u64) -> Key {
        Key { address: at.saturating_sub(self.widest), dimension: 0, width: 0, height: 0, depth_or_array: 0, mips: 0, element: 0, samples: 0, depth: false }
    }

    /// The title's memory under a new image held other textures: one
    /// memory holds one texture at a time on the console, so filled images
    /// that start elsewhere and whose own bytes ([`Image::own`]) overlap
    /// `[at, at + len)` (the new texture's own bytes) are textures the
    /// title has since replaced (it reuses its texture pool without
    /// unmapping it). They go once the card is done with them; without
    /// this they piled up to the card-memory ceiling (vk46: 35,467 live).
    fn drop_overlaid(&mut self, b: &Backend, at: u64, len: u64) {
        let gone: Vec<Key> = self
            .images
            .range(self.first_over(at)..)
            .take_while(|(k, _)| k.address < at + len)
            .filter(|(k, i)| k.address != at && !i.card_written() && k.address + i.own.load(Ordering::Relaxed).max(1) > at)
            .map(|(k, _)| *k)
            .collect();
        let value = b.timeline.last();
        for k in gone {
            if let Some(img) = self.take(&k) {
                if cfg!(feature = "diagnostics") && DROPPED[Dropped::Overlaid as usize].load(Ordering::Relaxed) < 12 {
                    tracing::info!(
                        new_at = %format_args!("{at:#x}+{len:#x}"),
                        old_at = %format_args!("{:#x}+{:#x} (layout {:#x})", k.address, img.own.load(Ordering::Relaxed), img.bytes),
                        old = %format_args!("{:?} {}x{} mips={}", img.format, img.extent.width, img.extent.height, img.mips),
                        "Vulkan: a texture made over an older one's memory (the older image goes)"
                    );
                }
                super::descriptors::image_gone(img.id);
                self.dropped.push((value, img));
                DROPPED[Dropped::Overlaid as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// SAFETY: the staging buffers' mapped pointers are used only under the lock.
unsafe impl Send for Registry {}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Images made, fills done, bytes filled, images that could not be made,
/// refills found by fingerprint.
pub(crate) static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Why images went: the title freed the memory, unmapped it, remapped its
/// pages, released the resource.
#[derive(Clone, Copy)]
pub(crate) enum Dropped {
    Freed,
    Unmapped,
    Remapped,
    Resource,
    /// A newer texture was made over part of its memory.
    Overlaid,
}

pub(crate) static DROPPED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Images made by what named them first (a view, a resource: target, copy,
/// plane), and by 256 MiB of title address (diagnostics: where they come from).
pub(crate) static MADE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static MADE_AT: Mutex<Option<std::collections::HashMap<u64, u64>>> = Mutex::new(None);

/// The images' part of the 10 s line: made and dropped by reason, the
/// address ranges most images were made in, images on memory of their own
/// over another's (aliases that did not fit).
pub(crate) fn report() {
    let n = |a: &[AtomicU64], i: usize| a[i].load(Ordering::Relaxed);
    let mut top: Vec<(u64, u64)> = lock(&MADE_AT).as_ref().map(|m| m.iter().map(|(&k, &v)| (k, v)).collect()).unwrap_or_default();
    top.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
    let top: Vec<String> = top.iter().take(6).map(|(k, v)| format!("{:#x}+256M:{v}", k << 28)).collect();
    tracing::info!(
        live = count(),
        made_by_view = n(&MADE, 0),
        made_by_resource = n(&MADE, 1),
        dropped_freed = n(&DROPPED, 0),
        dropped_unmapped = n(&DROPPED, 1),
        dropped_remapped = n(&DROPPED, 2),
        dropped_resource = n(&DROPPED, 3),
        dropped_overlaid = n(&DROPPED, 4),
        not_fitting_aliases = UNFIT.load(Ordering::Relaxed),
        fingerprinted_mb = fill::CHECKED.load(Ordering::Relaxed) >> 20,
        cpu_changed_fills = n(&STATS, 4),
        made_at = %top.join(" "),
        "Vulkan: images"
    );
    if cfg!(feature = "diagnostics") {
        let (biggest, by_kind, sums) = biggest();
        tracing::info!(sums = %sums, biggest = %biggest, by_kind = %by_kind, "Vulkan: where image memory goes");
        tracing::info!(tracked = %fill::tracking(), "Vulkan: images by how a CPU store reaches them, and the ones filled most often");
    }
}

/// Diagnostics: the 8 live images with the most card memory, card memory
/// by format and whether the card writes them (each piece of memory
/// counted once, at its first image), and the sums: card memory of the
/// images the card writes against the title memory under them (more card
/// than title memory: images of memory the title has since reused), of
/// the textures the title fills, and of images waiting to be destroyed.
fn biggest() -> (String, String, String) {
    // The registry before the places, as `at` takes them.
    let guard = registry();
    let Some(r) = guard.as_ref() else { return (String::new(), String::new(), String::new()) };
    let places = lock(&PLACES);
    let mut seen = std::collections::HashSet::new();
    let mut sized: Vec<(u64, &Arc<Image>)> = Vec::new();
    let mut kinds: std::collections::HashMap<(vk::Format, bool), (u64, u64)> = std::collections::HashMap::new();
    // [card-written, filled]: images, card bytes, title bytes under them
    // (overlaps once; the registry is in address order).
    let mut sums = [[0u64; 3]; 2];
    let mut end = [0u64; 2];
    for img in r.images.values() {
        let Some(&(memory, offset, size, _)) = places.get(&img.id) else { continue };
        let s = usize::from(!img.card_written());
        sums[s][0] += 1;
        let (from, to) = (img.key.address.max(end[s]), img.key.address + img.bytes);
        sums[s][2] += to.saturating_sub(from);
        end[s] = end[s].max(to);
        if !seen.insert((memory, offset)) {
            continue;
        }
        sums[s][1] += size;
        sized.push((size, img));
        let k = kinds.entry((img.format, img.card_written())).or_insert((0, 0));
        k.0 += size;
        k.1 += 1;
    }
    sized.sort_by_key(|&(size, _)| std::cmp::Reverse(size));
    let biggest: Vec<String> = sized
        .iter()
        .take(8)
        .map(|(size, i)| {
            format!(
                "{}MB {:?} {}x{}x{} mips={} dropped={} card_written={} at={:#x}",
                size >> 20, i.format, i.extent.width, i.extent.height, i.extent.depth.max(i.layers), i.mips, i.dropped, i.card_written(), i.key.address
            )
        })
        .collect();
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort_by_key(|&(_, (size, _))| std::cmp::Reverse(size));
    let by_kind: Vec<String> = kinds
        .iter()
        .take(8)
        .map(|((f, w), (size, n))| format!("{:?}{} {}MB n={n}", f, if *w { " card-written" } else { "" }, size >> 20))
        .collect();
    let waiting: u64 = r.dropped.iter().filter_map(|(_, i)| places.get(&i.id)).map(|p| p.2).sum();
    let sums = format!(
        "card-written n={} {}MB over {}MB of title memory; filled n={} {}MB over {}MB; dropped, waiting for the card n={} {}MB; blocks {}",
        sums[0][0],
        sums[0][1] >> 20,
        sums[0][2] >> 20,
        sums[1][0],
        sums[1][1] >> 20,
        sums[1][2] >> 20,
        r.dropped.len(),
        waiting >> 20,
        r.blocks.len(),
    );
    (biggest.join("; "), by_kind.join("; "), sums)
}

static UNFIT: AtomicU64 = AtomicU64::new(0);

fn registry() -> std::sync::MutexGuard<'static, Option<Registry>> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title released a texture resource: its image goes once the card is
/// done with it (its card memory when no alias uses it any more). A later
/// resource of the same shape at the address gets an image of its own,
/// filled, not this one's stale contents.
pub(crate) fn resource_gone(b: &Backend, address: u64, desc: &ResourceDesc) {
    let Some(key) = key_of(address, desc) else { return };
    let mut guard = registry();
    let Some(r) = guard.as_mut() else { return };
    if let Some(img) = r.take(&key) {
        super::descriptors::image_gone(img.id);
        r.dropped.push((b.timeline.last(), img));
        DROPPED[Dropped::Resource as usize].fetch_add(1, Ordering::Relaxed);
    }
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
    at(b, data as u64, &desc, false)
}

/// The colour image that takes storage writes into a plane of depth image
/// `depth` (over `address`, the resource `desc`), made the first time: a
/// 32-bit float image of its shape for the depth plane, an 8-bit integer
/// one for the stencil plane. A depth image has no storage views.
pub(crate) fn written_plane(b: &Backend, depth: &Arc<Image>, address: u64, desc: &ResourceDesc, stencil: bool) -> Option<Arc<Image>> {
    let cell = if stencil { &depth.plane_stencil } else { &depth.plane };
    if let Some(p) = cell.get() {
        return Some(p.clone());
    }
    // DXGI_FORMAT_R8_UINT; DXGI_FORMAT_R16_UNORM or DXGI_FORMAT_R32_FLOAT,
    // as the depth plane's own bits
    let format = match (stencil, depth.format) {
        (true, _) => 62,
        (false, vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT) => 56,
        (false, _) => 41,
    };
    let plain = ResourceDesc { format, flags: (desc.flags & !DEPTH_STENCIL) | UNORDERED, sample_count: 1, mip_levels: 1, ..*desc };
    // The stencil plane's image is keyed one byte on: images over one
    // address share card memory, and the two planes are two memories.
    let p = at(b, address + u64::from(stencil), &plain, true)?;
    Some(cell.get_or_init(|| p).clone())
}

/// The image over `address` of `desc`'s shape, made the first time
/// (`by_view`: a descriptor named it first, else a resource did).
pub(crate) fn at(b: &Backend, address: u64, desc: &ResourceDesc, by_view: bool) -> Option<Arc<Image>> {
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
        // a depth image's card memory holds nothing a colour image reads
        .filter(|(k, _)| k.depth == key.depth)
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
            if by_view && !img.card_written() {
                // (its views say how far up its levels are there: `names_level`)
                img.own.store(crate::resource::bytes_through(desc, img.bytes, u32::MAX), Ordering::Relaxed);
                img.own_level.store(u32::MAX, Ordering::Relaxed);
            }
            let img = Arc::new(img);
            r.widest = r.widest.max(img.bytes);
            if !img.card_written() {
                r.drop_overlaid(b, address, img.own.load(Ordering::Relaxed).max(1));
                r.page(&img, true);
            }
            // Into GENERAL and cleared at the next flush (ahead of anything
            // that uses it); filled after that unless the card writes it.
            img.pending.store(true, Ordering::Relaxed);
            r.pending.push(img.clone());
            r.images.insert(key, img.clone());
            STATS[0].fetch_add(1, Ordering::Relaxed);
            MADE[usize::from(!by_view)].fetch_add(1, Ordering::Relaxed);
            *lock(&MADE_AT).get_or_insert_with(Default::default).entry(address >> 28).or_insert(0) += 1;
            Some(img)
        }
        None => {
            // Which images a run goes without (the first ones; the 10 s
            // line counts them all): a draw into one is left out, a
            // program that reads one reads zeros.
            if cfg!(feature = "diagnostics") && STATS[3].fetch_add(1, Ordering::Relaxed) < 48 {
                tracing::warn!(
                    at = format_args!("{address:#x}"),
                    format = desc.format,
                    size = format_args!("{}x{}x{}", desc.width, desc.height, desc.depth_or_array_size),
                    mips = desc.mip_levels,
                    flags = format_args!("{:#x}", desc.flags),
                    "Vulkan: no image was made (no card memory under the ceiling, or no format for it)"
                );
            } else if !cfg!(feature = "diagnostics") {
                STATS[3].fetch_add(1, Ordering::Relaxed);
            }
            None
        }
    }
}

/// A view shows `img` from level `level` on: the texture's own bytes reach
/// through that level now, and filled images of other textures that start
/// in them are textures the title has replaced.
pub(crate) fn names_level(b: &Backend, img: &Arc<Image>, level: u32) {
    if level >= img.own_level.load(Ordering::Relaxed) || img.card_written() {
        return;
    }
    let own = crate::resource::bytes_through(&img.desc, img.bytes, level);
    let mut guard = registry();
    let Some(r) = guard.as_mut() else { return };
    img.own_level.fetch_min(level, Ordering::Relaxed);
    if img.own.fetch_max(own, Ordering::Relaxed) < own {
        r.drop_overlaid(b, img.key.address, own.max(1));
    }
}

/// The format of the colour image of a resource.
fn made_format(desc: &ResourceDesc, card_written: bool) -> Option<vk::Format> {
    // DXGI_FORMAT_R9G9B9E5_SHAREDEXP that is no render target: the card
    // can neither store to nor render to that format, and the title writes
    // such textures as packed words through 32_UINT views (the screen
    // probes' ray radiance, the final gather's output), whether or not the
    // resource says it takes unordered access. A 32_UINT image takes the
    // stores; its sampled views read the words as what they are
    // (`views::view_format`).
    if crate::resource::pc_format(desc.format) == 67 && desc.flags & RENDER_TARGET == 0 {
        return Some(vk::Format::R32_UINT);
    }
    color_format(desc.format, card_written)
}

fn make(b: &Backend, r: &mut Registry, key: Key, desc: &ResourceDesc, alias: Option<&Arc<Image>>) -> Option<Image> {
    let depth = key.depth;
    let card_written = desc.flags & (RENDER_TARGET | DEPTH_STENCIL | UNORDERED) != 0
        || alias.is_some_and(|a| a.card_written());
    let format = if depth {
        depth_format(desc.format)?
    } else {
        made_format(desc, card_written)?
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
    let dropped = dropped_of(desc, card_written || depth || alias.is_some(), mips);
    let extent = vk::Extent3D { width: extent.width >> dropped, height: extent.height >> dropped, ..extent };
    let mips = mips - dropped;
    let mut usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if depth {
        usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    } else {
        if desc.flags & RENDER_TARGET != 0 {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        // Storage for every colour image the driver gives it to (it is
        // dropped below where it does not): the console takes an unordered
        // view of a resource made without the flag, and an image without
        // the usage has no storage view, so the stores went to the null
        // descriptor (CS_ScreenProbes_Inject's ray radiance, runs gi5 to
        // gi8: the probes' radiance was all zeros and no indirect light
        // reached the lighting).
        if key.samples == 1 {
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
    let image = victory_vk::account::traced(&b.host, "image", || unsafe { d.create_image(&info, None) }).ok()?;
    // SAFETY: a live image.
    let needs = unsafe { d.get_image_memory_requirements(image) };
    // Over an earlier image's memory when it fits there.
    let shared = alias.and_then(|a| lock(&PLACES).get(&a.id).copied()).filter(|&(_, _, size, ty)| {
        needs.size <= size && needs.memory_type_bits & (1 << ty) != 0
    });
    if alias.is_some() && shared.is_none() {
        UNFIT.fetch_add(1, Ordering::Relaxed);
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
        dropped,
        layers: layers.max(1),
        kind,
        samples: key.samples,
        usage,
        desc: *desc,
        card_written: AtomicBool::new(card_written),
        alias: alias.is_some(),
        bytes,
        own: AtomicU64::new(bytes),
        own_level: std::sync::atomic::AtomicU32::new(0),
        views: Mutex::new(Vec::new()),
        plane: OnceLock::new(),
        plane_stencil: OnceLock::new(),
        pending: AtomicBool::new(false),
        last_use: AtomicU64::new(0),
        fresh: AtomicBool::new(true),
        written: [const { AtomicU64::new(0) }; 4],
        hash: AtomicU64::new(0),
        next_check: AtomicU64::new(0),
        check_every: std::sync::atomic::AtomicU32::new(1),
        dynamic: AtomicBool::new(false),
        tracked: AtomicBool::new(false),
        cpu_changed: AtomicBool::new(false),
    })
}

/// The largest levels texture quality leaves out of the image of a texture
/// the title fills (`crate::quality`, as the D3D12 path's
/// `resource::dropped_levels`): two-dimensional, of one sample, in a
/// console layout the detiling reads. None of a texture the card writes,
/// a depth texture or an image over another's memory (`kept_whole`).
fn dropped_of(desc: &ResourceDesc, kept_whole: bool, mips: u32) -> u32 {
    if kept_whole || desc.dimension != TEXTURE_2D || desc.sample_count > 1 {
        return 0;
    }
    let Some((_, block)) = crate::resource::element(crate::resource::pc_format(desc.format)) else {
        return 0;
    };
    let readable = crate::resource::texture_layout(desc).is_some_and(|t| !t.planes[0].layout.desc.swizzle.xor());
    if !readable {
        return 0;
    }
    crate::quality::dropped_for(desc.width, u64::from(desc.height), mips, block)
}

/// The image's level and layer of the console's subresource `index`
/// (levels first, then layers); `None` for a level the image leaves out.
pub(crate) fn subresource(img: &Image, index: u32) -> Option<(u32, u32)> {
    let console = img.mips + img.dropped;
    Some(((index % console).checked_sub(img.dropped)?, index / console))
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
/// Diagnostics: the images the card writes, the most drawn first.
pub(crate) fn card_written_images() -> Vec<Arc<Image>> {
    let mut out: Vec<Arc<Image>> = registry().as_ref().map_or_else(Vec::new, |r| r.images.values().filter(|i| i.card_written()).cloned().collect());
    out.sort_by_key(|i| std::cmp::Reverse(i.written[0].load(Ordering::Relaxed) + i.written[1].load(Ordering::Relaxed)));
    out
}

pub(crate) fn count() -> usize {
    registry().as_ref().map_or(0, |r| r.images.len())
}
