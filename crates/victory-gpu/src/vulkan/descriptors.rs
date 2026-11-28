//! M2b step (a): the title's descriptors as real RDNA2 descriptors
//! (`docs/vulkan-abi.md` sections 4 and 5).
//!
//! At `Create*View` / `CreateSampler` the host writes the descriptor the
//! console would: a V# for buffer views (title address, stride, records,
//! format), a T# for texture views (WORD7 the host's slot), an S# for
//! samplers (WORD3 the host's slot). Shaders read them from title memory as
//! the console's code does; nothing is mirrored or looked up per draw.
//!
//! A T#'s slot names a Vulkan image descriptor in the set-0 descriptor
//! buffer (sampled at binding 0, storage at binding 1); an S#'s names a
//! sampler in the set-1 descriptor buffer. Slot 0 of both is a null image
//! (one transparent black texel) and a plain linear sampler. Set 3 holds
//! the memory map's uniform-buffer descriptor, one per map generation.
//!
//! **A slot belongs to a texture and a view of it** (the title address
//! and shape that name the image, `images::Key`, and the view's ranges),
//! as the console's descriptor names memory and a shape, **and stays
//! theirs**: the title keeps its descriptors (programs pick textures by
//! heap index, and nothing writes those descriptors again), so a slot
//! given to another texture showed that texture through every descriptor
//! that still carried it (the character's suit took its colours from
//! other materials' textures, differently in every run). While the
//! texture has no image its slots read the null image; when it has one
//! again (made again for other levels, or after its memory came back)
//! they show it, and the old descriptors work again. Slots are given in
//! order. The last 65,536 textures that went keep theirs ([`KEPT`]: what
//! the bookkeeping of a long session may cost); past that, and when all
//! 2^20 are given, a new view takes the slots of the texture that has
//! been without an image longest.
//!
//! shortcut: a descriptor older than that shows the slot's new texture.
//! The clean form is a generation in the T# that the translated code
//! checks (`docs/vulkan-host.md`, proposals for the ABI).
//!
//! **Every slot holds a valid descriptor at all times** (vk41: a graphics
//! shader faulted with "Illegal Instruction Parameter"): every slot of
//! bindings 0-2 and of the samplers starts as slot 0's null descriptor;
//! a slot taken for a view is set back to null on all three bindings
//! before its own are written; and a slot whose image goes is set back to
//! null at once (title descriptors may still carry it).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_vk::ash::vk;
use victory_vk::gfx10;
use victory_vk::memory::{Buffer, Place};

use super::images::{self, Image, Key, ViewKey};
use super::stages;
use super::{Backend, lock};

const IMAGE_SLOTS: u32 = 1 << 20;
/// Textures without an image that keep their slots.
const KEPT: usize = 1 << 16;
/// Map blocks kept (one per generation, reused round).
const MAPS: u64 = 256;
const MAP_BLOCK: u64 = 4096;

pub(crate) struct Heaps {
    resource: Buffer,
    sampler: Buffer,
    sampled_at: u64,
    storage_at: u64,
    /// Set 0 binding 2: 2D-array views of cube images (ABI v2).
    cube_at: u64,
    /// Set 0 binding 3: R64_UINT storage views of two-dword images (ABI v5).
    wide_at: u64,
    sampler_at: u64,
    map_at: u64,
    map_stride: u64,
    sizes: [usize; 4],
    maps: Buffer,
    zero_sink: Buffer,
    state: Mutex<State>,
    /// Slot 0's descriptor bytes: sampled, storage, cube, sampler, wide.
    nulls: OnceLock<[Vec<u8>; 5]>,
}

// SAFETY: the mapped pointers are written at distinct slots, under the
// state lock for anything shared.
unsafe impl Send for Heaps {}
// SAFETY: as above.
unsafe impl Sync for Heaps {}

#[derive(Default)]
struct State {
    next: u32,
    /// Slots taken back from textures long without an image.
    free: Vec<u32>,
    /// Textures whose image went, the longest ago first, with the
    /// timeline's value then.
    gone: VecDeque<(u64, Key)>,
    /// The slot of each view of each texture.
    views: HashMap<(Key, ViewKey), u32>,
    by_texture: HashMap<Key, Vec<(ViewKey, u32)>>,
    /// The image the slots of a texture show now.
    shown: HashMap<Key, Arc<Image>>,
    samplers: HashMap<[u32; 8], u32>,
    sampler_objects: Vec<vk::Sampler>,
    /// Map generation and the block it went into.
    map: Option<(u64, u64)>,
    map_round: u64,
    immediate: usize,
}

/// Views made, samplers made, view writes that found no image.
pub(crate) static STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The R64_UINT storage view of a storage view of a two-dword format
/// (R32G32 UINT, SINT or SFLOAT: one compatibility class with R64_UINT, the
/// image is made MUTABLE_FORMAT), for 64-bit image atomics (ABI v5); `None`
/// on a device without `shaderImageInt64Atomics` or R64_UINT storage atomics.
fn wide_view(b: &Backend, image: &Image, key: ViewKey) -> Option<vk::ImageView> {
    let two_dwords = matches!(key.format, vk::Format::R32G32_UINT | vk::Format::R32G32_SINT | vk::Format::R32G32_SFLOAT);
    if !(two_dwords && b.host.caps.image_int64_atomics && r64_atomics(b)) {
        return None;
    }
    let identity = [vk::ComponentSwizzle::IDENTITY; 4];
    image.view(b, ViewKey { format: vk::Format::R64_UINT, components: identity, ..key })
}

/// Whether R64_UINT storage images take atomics (optimal tiling).
fn r64_atomics(b: &Backend) -> bool {
    static R64: OnceLock<bool> = OnceLock::new();
    *R64.get_or_init(|| {
        // SAFETY: a plain query.
        let p = unsafe { b.host.instance.get_physical_device_format_properties(b.host.physical, vk::Format::R64_UINT) };
        let has = p.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC);
        tracing::info!(r64_uint_storage_atomics = has, int64_image_atomics = b.host.caps.image_int64_atomics, "Vulkan: 64-bit image atomics");
        has
    })
}

/// The image bindings of set 0.
#[derive(Clone, Copy)]
enum Binding {
    Sampled,
    Storage,
    /// 2D-array views of cube images (ABI v2).
    Cube,
    /// R64_UINT storage views of two-dword images (ABI v5).
    Wide,
}

pub(crate) fn heaps(b: &'static Backend) -> Option<&'static Heaps> {
    static HEAPS: OnceLock<Option<Heaps>> = OnceLock::new();
    HEAPS
        .get_or_init(|| match make(b) {
            Ok(h) => Some(h),
            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "Vulkan: no descriptor buffers; nothing draws");
                None
            }
        })
        .as_ref()
}

fn make(b: &'static Backend) -> anyhow::Result<Heaps> {
    let layout = stages::layout(b).ok_or_else(|| anyhow::anyhow!("no pipeline layout"))?;
    let db = b.host.descriptor_buffer.as_ref().ok_or_else(|| anyhow::anyhow!("no VK_EXT_descriptor_buffer"))?;
    let props = victory_vk::descriptors::properties(&b.host).ok_or_else(|| anyhow::anyhow!("no descriptor buffer properties"))?;
    let align = props.descriptor_buffer_offset_alignment.max(1);
    // SAFETY: live layouts.
    let (sampled_at, storage_at, cube_at, wide_at, sampler_at) = unsafe {
        (
            db.get_descriptor_set_layout_binding_offset(layout.sets[0], 0),
            db.get_descriptor_set_layout_binding_offset(layout.sets[0], 1),
            db.get_descriptor_set_layout_binding_offset(layout.sets[0], 2),
            db.get_descriptor_set_layout_binding_offset(layout.sets[0], 3),
            db.get_descriptor_set_layout_binding_offset(layout.sets[1], 0),
        )
    };
    let map_stride = layout.set_bytes[stages::MAP_SET].max(1).next_multiple_of(align);
    let map_at = layout.set_bytes[0].next_multiple_of(align);
    let resource = Buffer::new(
        &b.host,
        map_at + MAPS * map_stride,
        vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
        Place::Upload,
    )?;
    let sampler = Buffer::new(
        &b.host,
        layout.set_bytes[1].max(64),
        vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
        Place::Upload,
    )?;
    let maps = Buffer::new(&b.host, MAPS * MAP_BLOCK, vk::BufferUsageFlags::UNIFORM_BUFFER, Place::Upload)?;
    let zero_sink = Buffer::new(
        &b.host,
        64 << 10,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        Place::Card,
    )?;
    let zs = zero_sink.buffer;
    let _ = b.timeline.record(&b.host, |d, cmd| {
        // SAFETY: recording into the timeline's one-shot buffer.
        unsafe { d.cmd_fill_buffer(cmd, zs, 0, vk::WHOLE_SIZE, 0) };
    });
    let heaps = Heaps {
        resource,
        sampler,
        sampled_at,
        storage_at,
        cube_at,
        wide_at,
        sampler_at,
        map_at,
        map_stride,
        sizes: [
            props.sampled_image_descriptor_size,
            props.storage_image_descriptor_size,
            props.sampler_descriptor_size,
            props.uniform_buffer_descriptor_size,
        ],
        maps,
        zero_sink,
        state: Mutex::new(State {
            next: 1,
            ..State::default()
        }),
        nulls: OnceLock::new(),
    };
    heaps.make_null(b)?;
    heaps.fill_null();
    tracing::info!(
        sampled_bytes = heaps.sizes[0],
        storage_bytes = heaps.sizes[1],
        sampler_bytes = heaps.sizes[2],
        map_stride,
        "Vulkan: descriptor buffers (set 0 images, set 1 samplers, set 3 the memory map)"
    );
    Ok(heaps)
}

impl Heaps {
    /// Slot 0: a 1x1 image of zeros, sampled and storage; sampler slot 0.
    fn make_null(&self, b: &Backend) -> anyhow::Result<()> {
        let d = &b.host.device;
        let format = vk::Format::R8G8B8A8_UNORM;
        // SAFETY: valid create infos; kept for the run.
        unsafe {
            let image = d.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST),
                None,
            )?;
            let needs = d.get_image_memory_requirements(image);
            let ty = b
                .host
                .memory_type(needs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty())
                .ok_or_else(|| anyhow::anyhow!("no memory for the null image"))?;
            let memory = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(needs.size).memory_type_index(ty), None)?;
            victory_vk::account::allocated(&b.host, memory, ty, needs.size, victory_vk::account::Kind::Other);
            d.bind_image_memory(image, memory, 0)?;
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let view = d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(range),
                None,
            )?;
            b.timeline.record(&b.host, |d, cmd| {
                images::to_general(d, cmd, image, range, vk::ImageLayout::UNDEFINED);
                d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
            })?;
            let array = d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                    .format(format)
                    .subresource_range(range),
                None,
            )?;
            // Slot 0 is the null image on bindings 0-3 (ABI v5; the wide
            // binding takes only null descriptors and R64_UINT views).
            self.write_image(b, 0, view, Binding::Sampled);
            self.write_image(b, 0, view, Binding::Storage);
            self.write_image(b, 0, array, Binding::Cube);
            self.write_image(b, 0, view, Binding::Wide);
            // Null descriptors fit every image type a shader declares (a 2D
            // view of one texel does not fit a 3D, cube or arrayed
            // variable: vk42's Illegal Instruction Parameter).
            if b.host.caps.null_descriptor {
                for binding in [Binding::Sampled, Binding::Storage, Binding::Cube, Binding::Wide] {
                    self.write_null(b, binding);
                }
            }
            let s = d.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?;
            self.write_sampler(b, 0, s);
        }
        Ok(())
    }

    /// Slot 0's descriptors copied into every other slot of bindings 0-3
    /// and of the samplers.
    fn fill_null(&self) {
        let take = |base: *mut u8, at: u64, size: usize| -> Vec<u8> {
            // SAFETY: slot 0 of a mapped descriptor range, just written.
            unsafe { std::slice::from_raw_parts(base.add(at as usize), size).to_vec() }
        };
        let nulls = [
            take(self.resource.mapped, self.sampled_at, self.sizes[0]),
            take(self.resource.mapped, self.storage_at, self.sizes[1]),
            take(self.resource.mapped, self.cube_at, self.sizes[0]),
            take(self.sampler.mapped, self.sampler_at, self.sizes[2]),
            take(self.resource.mapped, self.wide_at, self.sizes[1]),
        ];
        let spread = |base: *mut u8, at: u64, bytes: &[u8], slots: u32| {
            for slot in 1..slots as usize {
                // SAFETY: the slot lies inside the mapped range of its binding.
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(at as usize + slot * bytes.len()), bytes.len()) };
            }
        };
        spread(self.resource.mapped, self.sampled_at, &nulls[0], IMAGE_SLOTS);
        spread(self.resource.mapped, self.storage_at, &nulls[1], IMAGE_SLOTS);
        spread(self.resource.mapped, self.cube_at, &nulls[2], IMAGE_SLOTS);
        spread(self.sampler.mapped, self.sampler_at, &nulls[3], stages::SAMPLER_SLOTS);
        spread(self.resource.mapped, self.wide_at, &nulls[4], IMAGE_SLOTS);
        let _ = self.nulls.set(nulls);
    }

    /// `slot` back to the null image on bindings 0-3.
    fn null_slot(&self, slot: u32) {
        for binding in [Binding::Sampled, Binding::Storage, Binding::Cube, Binding::Wide] {
            self.null_at(slot, binding);
        }
    }

    /// `slot` back to the null image on one binding.
    fn null_at(&self, slot: u32, binding: Binding) {
        let Some(n) = self.nulls.get() else { return };
        if slot == 0 {
            return;
        }
        let (bytes, at) = match binding {
            Binding::Sampled => (&n[0], self.sampled_at),
            Binding::Storage => (&n[1], self.storage_at),
            Binding::Cube => (&n[2], self.cube_at),
            Binding::Wide => (&n[4], self.wide_at),
        };
        // SAFETY: the slot lies inside the mapped set-0 range.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.resource.mapped.add(at as usize + slot as usize * bytes.len()), bytes.len()) };
    }

    /// Slot 0 of `binding`: a null descriptor (`nullDescriptor`).
    fn write_null(&self, b: &Backend, binding: Binding) {
        let Some(db) = b.host.descriptor_buffer.as_ref() else { return };
        let (ty, data, at, size) = match binding {
            Binding::Storage | Binding::Wide => (
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorDataEXT { p_storage_image: std::ptr::null() },
                if matches!(binding, Binding::Wide) { self.wide_at } else { self.storage_at },
                self.sizes[1],
            ),
            Binding::Sampled | Binding::Cube => (
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorDataEXT { p_sampled_image: std::ptr::null() },
                if matches!(binding, Binding::Cube) { self.cube_at } else { self.sampled_at },
                self.sizes[0],
            ),
        };
        // SAFETY: slot 0 lies inside the mapped set-0 range.
        unsafe {
            let out = std::slice::from_raw_parts_mut(self.resource.mapped.add(at as usize), size);
            db.get_descriptor(&vk::DescriptorGetInfoEXT::default().ty(ty).data(data), out);
        }
    }

    fn write_image(&self, b: &Backend, slot: u32, view: vk::ImageView, binding: Binding) {
        let Some(db) = b.host.descriptor_buffer.as_ref() else { return };
        let info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL);
        let (ty, data, at, size) = match binding {
            Binding::Storage | Binding::Wide => (
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorDataEXT { p_storage_image: &info },
                if matches!(binding, Binding::Wide) { self.wide_at } else { self.storage_at },
                self.sizes[1],
            ),
            Binding::Sampled | Binding::Cube => (
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorDataEXT { p_sampled_image: &info },
                if matches!(binding, Binding::Cube) { self.cube_at } else { self.sampled_at },
                self.sizes[0],
            ),
        };
        let offset = at as usize + slot as usize * size;
        // SAFETY: the slot lies inside the mapped set-0 range.
        unsafe {
            let out = std::slice::from_raw_parts_mut(self.resource.mapped.add(offset), size);
            db.get_descriptor(&vk::DescriptorGetInfoEXT::default().ty(ty).data(data), out);
        }
    }

    fn write_sampler(&self, b: &Backend, slot: u32, sampler: vk::Sampler) {
        let Some(db) = b.host.descriptor_buffer.as_ref() else { return };
        let size = self.sizes[2];
        let offset = self.sampler_at as usize + slot as usize * size;
        // SAFETY: the slot lies inside the mapped set-1 range.
        unsafe {
            let out = std::slice::from_raw_parts_mut(self.sampler.mapped.add(offset), size);
            db.get_descriptor(
                &vk::DescriptorGetInfoEXT::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .data(vk::DescriptorDataEXT { p_sampler: &sampler }),
                out,
            );
        }
    }

    /// The slot of `view` of `image`'s texture (given and written the
    /// first time).
    fn image_slot(&self, b: &Backend, image: &Arc<Image>, key: ViewKey) -> Option<u32> {
        let mut s = lock(&self.state);
        self.show(b, &mut s, image);
        if let Some(&slot) = s.views.get(&(image.key, key)) {
            return Some(slot);
        }
        let image = &s.shown.get(&image.key).cloned().unwrap_or_else(|| image.clone());
        let slot = loop {
            if let Some(slot) = s.free.pop() {
                break slot;
            }
            if s.next < IMAGE_SLOTS {
                s.next += 1;
                break s.next - 1;
            }
            if !self.take_back(b, &mut s) {
                return None;
            }
        };
        if !self.write_slot(b, slot, image, key) {
            self.null_slot(slot);
            s.free.push(slot);
            return None;
        }
        s.views.insert((image.key, key), slot);
        s.by_texture.entry(image.key).or_default().push((key, slot));
        STATS[0].fetch_add(1, Ordering::Relaxed);
        Some(slot)
    }

    /// The descriptors of `key`'s view of `image` into `slot`; whether
    /// the image has such a view.
    fn write_slot(&self, b: &Backend, slot: u32, image: &Arc<Image>, key: ViewKey) -> bool {
        let sampled = image.view(b, ViewKey { storage: false, ..key });
        let storage = if key.storage { image.view(b, key) } else { None };
        if key.storage {
            match storage {
                Some(_) => image.written_by_card(),
                // (never silently: the program's stores would go to the null descriptor)
                None => super::images::once_warn(format!("Vulkan: an unordered view without a storage view: {:?} of an image of {:?} at {:#x}, usage {:?}", key.format, image.format, image.key.address, image.usage)),
            }
        }
        let Some(sampled) = sampled.or(storage) else { return false };
        let wide = storage.and_then(|_| wide_view(b, image, key));
        let cube = matches!(key.kind, vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY)
            .then(|| image.view(b, ViewKey { kind: vk::ImageViewType::TYPE_2D_ARRAY, storage: false, ..key }))
            .flatten();
        // Each binding straight to its new descriptor (work in flight may
        // read the slot as this runs), or to null: nothing of the slot's
        // last image is left on any binding.
        for (view, binding) in [(Some(sampled), Binding::Sampled), (storage, Binding::Storage), (wide, Binding::Wide), (cube, Binding::Cube)] {
            match view {
                Some(view) => self.write_image(b, slot, view, binding),
                None => self.null_at(slot, binding),
            }
        }
        true
    }

    /// `image` is its texture's image now: the slots the texture has show
    /// it (after it was made again, or came back).
    fn show(&self, b: &Backend, s: &mut State, image: &Arc<Image>) {
        // (ids grow: a caller that still holds the image before the one
        // shown changes nothing)
        if s.shown.get(&image.key).is_some_and(|i| i.id >= image.id) {
            return;
        }
        s.shown.insert(image.key, image.clone());
        for &(view, slot) in s.by_texture.get(&image.key).map_or(&[][..], |v| v) {
            if !self.write_slot(b, slot, image, view) {
                self.null_slot(slot);
            }
        }
    }

    /// The slots of the texture that has been without an image longest
    /// are given again (not one that has an image again). False when
    /// there is no such texture the card is past.
    fn take_back(&self, b: &Backend, s: &mut State) -> bool {
        let Some(&(value, key)) = s.gone.front() else { return false };
        if value > b.timeline.completed(&b.host) {
            return false;
        }
        s.gone.pop_front();
        if !s.shown.contains_key(&key) {
            for (view, slot) in s.by_texture.remove(&key).unwrap_or_default() {
                s.views.remove(&(key, view));
                s.free.push(slot);
            }
        }
        true
    }

    /// The slot of a sampler (made the first time).
    fn sampler_slot(&self, b: &Backend, desc: &SamplerDesc) -> u32 {
        let key = desc.key();
        let mut s = lock(&self.state);
        if let Some(&slot) = s.samplers.get(&key) {
            return slot;
        }
        let slot = s.samplers.len() as u32 + 1;
        if slot >= stages::IMMEDIATE_FIRST {
            return 0;
        }
        // SAFETY: valid create info; kept for the run (deduplicated).
        match unsafe { b.host.device.create_sampler(&desc.info(), None) } {
            Ok(sampler) => {
                self.write_sampler(b, slot, sampler);
                s.samplers.insert(key, slot);
                s.sampler_objects.push(sampler);
                STATS[1].fetch_add(1, Ordering::Relaxed);
                slot
            }
            Err(_) => 0,
        }
    }

    /// Writes the immediate samplers the stages named since the last call.
    pub(crate) fn immediates(&self, b: &Backend) {
        let list = {
            let all = lock(&stages::IMMEDIATE);
            if all.len() == lock(&self.state).immediate {
                return;
            }
            all.clone()
        };
        let mut s = lock(&self.state);
        while s.immediate < list.len() {
            let n = s.immediate;
            let desc = SamplerDesc::from_static(&victory_shader::sampler::StaticSampler::from_words(list[n]));
            // SAFETY: valid create info; kept for the run.
            if let Ok(sampler) = unsafe { b.host.device.create_sampler(&desc.info(), None) } {
                self.write_sampler(b, stages::IMMEDIATE_FIRST + n as u32, sampler);
                s.sampler_objects.push(sampler);
            }
            s.immediate += 1;
        }
    }

    /// The set-3 offset of the descriptor of the map `block` of
    /// `generation` (written into the next block when it is new).
    pub(crate) fn map_offset(&self, b: &Backend, generation: u64, block: &[u8]) -> u64 {
        let mut s = lock(&self.state);
        if let Some((g, at)) = s.map
            && g == generation
        {
            return self.map_at + at * self.map_stride;
        }
        let at = s.map_round % MAPS;
        s.map_round += 1;
        let Some(db) = b.host.descriptor_buffer.as_ref() else { return 0 };
        let len = block.len().min(MAP_BLOCK as usize);
        // SAFETY: the block lies inside the mapped map buffer; a block is
        // reused only 256 generations later.
        unsafe {
            std::ptr::copy_nonoverlapping(block.as_ptr(), self.maps.mapped.add((at * MAP_BLOCK) as usize), len);
            let info = vk::DescriptorAddressInfoEXT::default()
                .address(self.maps.address + at * MAP_BLOCK)
                .range(MAP_BLOCK);
            let offset = (self.map_at + at * self.map_stride) as usize;
            let out = std::slice::from_raw_parts_mut(self.resource.mapped.add(offset), self.sizes[3]);
            db.get_descriptor(
                &vk::DescriptorGetInfoEXT::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .data(vk::DescriptorDataEXT { p_uniform_buffer: &info }),
                out,
            );
        }
        s.map = Some((generation, at));
        self.map_at + at * self.map_stride
    }

    /// The zero and sink addresses of the map (ABI section 3).
    pub(crate) fn zero_sink(&self) -> (u64, u64) {
        (self.zero_sink.address, self.zero_sink.address + (32 << 10))
    }

    /// Binds the descriptor buffers into `cmd` for both bind points, set 3
    /// at `map`.
    ///
    /// # Safety
    /// Recording into `cmd`.
    pub(crate) unsafe fn bind(&self, b: &Backend, cmd: vk::CommandBuffer, map: u64) {
        let Some(db) = b.host.descriptor_buffer.as_ref() else { return };
        let Some(layout) = stages::layout(b) else { return };
        let infos = [
            vk::DescriptorBufferBindingInfoEXT::default()
                .address(self.resource.address)
                .usage(vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT),
            vk::DescriptorBufferBindingInfoEXT::default()
                .address(self.sampler.address)
                .usage(vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT),
        ];
        let indices = [0u32, 1, 0, 0];
        let offsets = [0u64, 0, 0, map];
        // SAFETY: as the caller says.
        unsafe {
            db.cmd_bind_descriptor_buffers(cmd, &infos);
            for point in [vk::PipelineBindPoint::GRAPHICS, vk::PipelineBindPoint::COMPUTE] {
                db.cmd_set_descriptor_buffer_offsets(cmd, point, layout.pipeline, 0, &indices, &offsets);
            }
        }
    }
}

impl Heaps {
    /// The slot of a plane's view (the composition pass reads it).
    pub(crate) fn plane_slot(&self, b: &Backend, image: &Arc<Image>, key: ViewKey) -> Option<u32> {
        self.image_slot(b, image, key)
    }
}

/// Diagnostics: what the host wrote at image slot `slot` (view type,
/// format, storage, image), or that it holds the null descriptor.
pub(crate) fn slot_view(slot: u32) -> String {
    let Some(b) = super::backend() else { return "?".into() };
    let Some(h) = heaps(b) else { return "?".into() };
    let s = lock(&h.state);
    let found = s.views.iter().find(|(_, v)| **v == slot).map(|((key, k), _)| {
        let image = match s.shown.get(key) {
            Some(i) => format!("image {} at {:#x} {:?} {}x{}x{}", i.id, i.key.address, i.format, i.extent.width, i.extent.height, i.extent.depth),
            None => format!("null (its texture at {:#x} has no image)", key.address),
        };
        format!("{image} view {:?} {:?} mips {}+{} layers {}+{} storage={}", k.kind, k.format, k.base_mip, k.mips, k.base_layer, k.layers, k.storage)
    });
    found.unwrap_or_else(|| if slot < s.next { "null (taken back)".into() } else { "null (never given)".into() })
}

/// Diagnostics: the image and view the host wrote at image slot `slot`.
pub(crate) fn slot_image(slot: u32) -> Option<(Arc<Image>, ViewKey)> {
    let h = heaps(super::backend()?)?;
    let s = lock(&h.state);
    let (&(key, view), _) = s.views.iter().find(|(_, v)| **v == slot)?;
    Some((s.shown.get(&key)?.clone(), view))
}

/// `image` is gone: the slots of its texture read the null image until
/// the texture has an image again (title descriptors may still carry
/// them: never a view of an image that is gone, never another texture).
pub(crate) fn image_gone(image: &Image) {
    let Some(b) = super::backend() else { return };
    let Some(h) = heaps(b) else { return };
    let mut s = lock(&h.state);
    // (made again since: the slots show the new image already)
    if s.shown.get(&image.key).is_none_or(|i| i.id != image.id) {
        return;
    }
    s.shown.remove(&image.key);
    let slots = s.by_texture.get(&image.key).map_or(0, Vec::len);
    for &(_, slot) in s.by_texture.get(&image.key).map_or(&[][..], |v| v) {
        h.null_slot(slot);
    }
    if slots > 0 {
        s.gone.push_back((b.timeline.last(), image.key));
    }
    while s.gone.len() > KEPT && h.take_back(b, &mut s) {}
}

/// `image` was made: the slots its texture has from an earlier image
/// show it.
pub(crate) fn image_made(image: &Arc<Image>) {
    let Some(b) = super::backend() else { return };
    let Some(h) = heaps(b) else { return };
    let mut s = lock(&h.state);
    if s.by_texture.contains_key(&image.key) {
        h.show(b, &mut s, image);
    }
}

/// The title's shader-visible view heaps: first address and bytes.
static TITLE_HEAPS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// The title made a shader-visible heap of views at `start`.
pub(crate) fn title_heap(start: u64, size: u64) {
    lock(&TITLE_HEAPS).push((start, size));
}

/// Diagnostics: the texture descriptors in the title's view heaps (T#s
/// with the host's check word and a slot), those of them whose slot
/// belongs to a texture without an image (they read the null image: a
/// black or missing surface if a program picks them), and the first of
/// those (heap index, slot, the texture's address). Descriptors the
/// title copied itself are among them; so are ones it no longer uses.
pub(crate) fn reading_null() -> (usize, usize, String) {
    let Some(h) = super::backend().and_then(heaps) else { return (0, 0, String::new()) };
    let waiting: HashMap<u32, u64> = {
        let s = lock(&h.state);
        s.by_texture.iter().filter(|(k, _)| !s.shown.contains_key(*k)).flat_map(|(k, v)| v.iter().map(|&(_, slot)| (slot, k.address))).collect()
    };
    let heaps_now = lock(&TITLE_HEAPS).clone();
    let (mut all, mut null, mut first) = (0, 0, Vec::new());
    let mut page = vec![0u8; 64 << 10];
    for (start, size) in heaps_now {
        for at in (start..start + size).step_by(page.len()) {
            let n = page.len().min((start + size - at) as usize);
            if !read_own(at, &mut page[..n]) {
                continue;
            }
            for (i, d) in page[..n].chunks_exact(32).enumerate() {
                let word = |w: usize| u32::from_le_bytes([d[4 * w], d[4 * w + 1], d[4 * w + 2], d[4 * w + 3]]);
                let slot = word(7) & 0xff_ffff;
                if word(6) ^ word(7) != gfx10::T_CHECK || slot == 0 {
                    continue;
                }
                all += 1;
                let Some(texture) = waiting.get(&slot) else { continue };
                null += 1;
                if first.len() < 6 {
                    first.push(format!("heap {:#x} slot {slot:#x} texture at {texture:#x}", (at - start) as usize / 32 + i));
                }
            }
        }
    }
    (all, null, first.join("; "))
}

/// Reads our own memory at `at` through the kernel (a page of a heap the
/// title has not committed is no fault).
fn read_own(at: u64, out: &mut [u8]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut copied = 0;
    // SAFETY: the destination is `out`; the kernel checks the source.
    super::census::host_reads(at as u64, out.len() as u64);
    unsafe { ReadProcessMemory(GetCurrentProcess(), at as *const _, out.as_mut_ptr().cast(), out.len(), &mut copied) != 0 }
}

/// For the 10 s line: slots given, slots of textures without an image.
pub(crate) fn slots() -> (u32, usize) {
    let Some(h) = super::backend().and_then(heaps) else { return (0, 0) };
    let s = lock(&h.state);
    let waiting = s.by_texture.iter().filter(|(k, _)| !s.shown.contains_key(*k)).map(|(_, v)| v.len()).sum();
    (s.next - 1, waiting)
}

// ---- samplers ----

/// A sampler as D3D12 describes it (the form all three sources reach).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SamplerDesc {
    filter: u32,
    address: [u32; 3],
    bias: f32,
    anisotropy: u32,
    compare: u32,
    border: [f32; 4],
    min_lod: f32,
    max_lod: f32,
}

impl SamplerDesc {
    fn from_words(w: &[u32; 13]) -> SamplerDesc {
        let f = |i: usize| f32::from_bits(w[i]);
        SamplerDesc {
            filter: w[0],
            address: [w[1], w[2], w[3]],
            bias: f(4),
            anisotropy: w[5],
            compare: w[6],
            border: [f(7), f(8), f(9), f(10)],
            min_lod: f(11),
            max_lod: f(12),
        }
    }

    fn from_static(s: &victory_shader::sampler::StaticSampler) -> SamplerDesc {
        let border = match s.border {
            1 => [0.0, 0.0, 0.0, 1.0],
            2 => [1.0; 4],
            _ => [0.0; 4],
        };
        SamplerDesc {
            filter: s.filter,
            address: s.address,
            bias: s.mip_lod_bias,
            anisotropy: s.max_anisotropy,
            compare: s.comparison,
            border,
            min_lod: s.min_lod,
            max_lod: s.max_lod,
        }
    }

    fn key(&self) -> [u32; 8] {
        let border = match self.border_color() {
            vk::BorderColor::FLOAT_OPAQUE_BLACK => 1,
            vk::BorderColor::FLOAT_OPAQUE_WHITE => 2,
            _ => 0,
        };
        [
            self.filter,
            self.address[0] | self.address[1] << 8 | self.address[2] << 16 | border << 24,
            self.bias.to_bits(),
            self.anisotropy,
            self.compare,
            self.min_lod.to_bits(),
            self.max_lod.to_bits(),
            0,
        ]
    }

    fn border_color(&self) -> vk::BorderColor {
        match self.border {
            [r, g, b, a] if r >= 0.5 && g >= 0.5 && b >= 0.5 && a >= 0.5 => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            [_, _, _, a] if a >= 0.5 => vk::BorderColor::FLOAT_OPAQUE_BLACK,
            _ => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        }
    }

    fn reduction(&self) -> u32 {
        self.filter >> 7 & 3
    }

    fn info(&self) -> vk::SamplerCreateInfo<'static> {
        let filter = |linear: bool| if linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let aniso = self.filter & 0x40 != 0;
        let address = |m: u32| match m {
            2 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            3 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            4 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            5 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            _ => vk::SamplerAddressMode::REPEAT,
        };
        let compares = self.reduction() == 1;
        vk::SamplerCreateInfo::default()
            .mag_filter(filter(aniso || self.filter & 0x04 != 0))
            .min_filter(filter(aniso || self.filter & 0x10 != 0))
            .mipmap_mode(if aniso || self.filter & 0x01 != 0 {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            })
            .address_mode_u(address(self.address[0]))
            .address_mode_v(address(self.address[1]))
            .address_mode_w(address(self.address[2]))
            .mip_lod_bias(self.bias.clamp(-15.99, 15.99))
            .anisotropy_enable(aniso && self.anisotropy > 1)
            .max_anisotropy(self.anisotropy.clamp(1, 16) as f32)
            .compare_enable(compares)
            .compare_op(vk::CompareOp::from_raw(self.compare.clamp(1, 8) as i32 - 1))
            .min_lod(self.min_lod.max(0.0))
            .max_lod(if self.max_lod >= 1000.0 { vk::LOD_CLAMP_NONE } else { self.max_lod.max(self.min_lod.max(0.0)) })
            .border_color(self.border_color())
    }

    /// The S# the console would hold, its slot in WORD3.
    fn encode(&self, slot: u32) -> [u32; 4] {
        let aniso = self.filter & 0x40 != 0;
        let xy = |linear: bool| match (aniso, linear) {
            (true, _) => 3,
            (false, true) => 1,
            (false, false) => 0,
        };
        let clamp = |m: u32| match m {
            2 => 1,
            3 => 2,
            4 => 6,
            5 => 3,
            _ => 0,
        };
        let border = match self.border_color() {
            vk::BorderColor::FLOAT_OPAQUE_BLACK => 1,
            vk::BorderColor::FLOAT_OPAQUE_WHITE => 2,
            _ => 0,
        };
        gfx10::Sampler {
            address: self.address.map(clamp),
            mag: xy(self.filter & 0x04 != 0),
            min: xy(self.filter & 0x10 != 0),
            mip: if self.filter & 0x01 != 0 || aniso { 2 } else { 1 },
            max_anisotropy: if aniso { self.anisotropy.clamp(1, 16) } else { 1 },
            compare: if self.reduction() == 1 { self.compare.clamp(1, 8) - 1 } else { 0 },
            min_lod: self.min_lod.max(0.0),
            max_lod: self.max_lod.clamp(0.0, 15.0),
            lod_bias: self.bias,
            border,
            filter_mode: match self.reduction() {
                2 => 1,
                3 => 2,
                _ => 0,
            },
        }
        .encode(slot)
    }
}

mod views;

pub(crate) use views::write;
