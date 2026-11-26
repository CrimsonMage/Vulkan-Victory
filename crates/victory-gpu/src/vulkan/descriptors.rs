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
//! sampler in the set-1 descriptor buffer. Slots are given per distinct
//! (image, view) and sampler, and freed when the image goes (after the
//! timeline passes what may read them). Slot 0 of both is a null image
//! (one transparent black texel) and a plain linear sampler. Set 3 holds
//! the memory map's uniform-buffer descriptor, one per map generation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_vk::ash::vk;
use victory_vk::gfx10;
use victory_vk::memory::{Buffer, Place};

use super::images::{self, Image, ViewKey};
use super::stages;
use super::{Backend, lock};
use crate::descriptor::View;

const IMAGE_SLOTS: u32 = 1 << 20;
/// Map blocks kept (one per generation, reused round).
const MAPS: u64 = 256;
const MAP_BLOCK: u64 = 4096;

pub(crate) struct Heaps {
    resource: Buffer,
    sampler: Buffer,
    sampled_at: u64,
    storage_at: u64,
    sampler_at: u64,
    map_at: u64,
    map_stride: u64,
    sizes: [usize; 4],
    maps: Buffer,
    zero_sink: Buffer,
    state: Mutex<State>,
}

// SAFETY: the mapped pointers are written at distinct slots, under the
// state lock for anything shared.
unsafe impl Send for Heaps {}
// SAFETY: as above.
unsafe impl Sync for Heaps {}

#[derive(Default)]
struct State {
    next: u32,
    free: Vec<u32>,
    deferred: Vec<(u64, u32)>,
    views: HashMap<(u64, ViewKey), u32>,
    by_image: HashMap<u64, Vec<u32>>,
    keep: HashMap<u64, Arc<Image>>,
    samplers: HashMap<[u32; 8], u32>,
    sampler_objects: Vec<vk::Sampler>,
    /// Map generation and the block it went into.
    map: Option<(u64, u64)>,
    map_round: u64,
    immediate: usize,
}

/// Views made, samplers made, view writes that found no image.
pub(crate) static STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

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
    let (sampled_at, storage_at, sampler_at) = unsafe {
        (
            db.get_descriptor_set_layout_binding_offset(layout.sets[0], 0),
            db.get_descriptor_set_layout_binding_offset(layout.sets[0], 1),
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
    };
    heaps.make_null(b)?;
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
            self.write_image(b, 0, view, false);
            self.write_image(b, 0, view, true);
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

    fn write_image(&self, b: &Backend, slot: u32, view: vk::ImageView, storage: bool) {
        let Some(db) = b.host.descriptor_buffer.as_ref() else { return };
        let info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL);
        let (ty, data, at, size) = if storage {
            (
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorDataEXT { p_storage_image: &info },
                self.storage_at,
                self.sizes[1],
            )
        } else {
            (
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorDataEXT { p_sampled_image: &info },
                self.sampled_at,
                self.sizes[0],
            )
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

    /// The slot of `view` of `image` (written the first time).
    fn image_slot(&self, b: &Backend, image: &Arc<Image>, key: ViewKey) -> Option<u32> {
        let mut s = lock(&self.state);
        if let Some(&slot) = s.views.get(&(image.id, key)) {
            return Some(slot);
        }
        let sampled = image.view(b, ViewKey { storage: false, ..key });
        let storage = if key.storage { image.view(b, key) } else { None };
        let sampled = sampled.or(storage)?;
        let completed = b.timeline.completed(&b.host);
        let mut i = 0;
        while i < s.deferred.len() {
            if s.deferred[i].0 <= completed {
                let (_, slot) = s.deferred.swap_remove(i);
                s.free.push(slot);
            } else {
                i += 1;
            }
        }
        let slot = match s.free.pop() {
            Some(slot) => slot,
            None if s.next < IMAGE_SLOTS => {
                s.next += 1;
                s.next - 1
            }
            None => return None,
        };
        self.write_image(b, slot, sampled, false);
        if let Some(st) = storage {
            self.write_image(b, slot, st, true);
        }
        s.views.insert((image.id, key), slot);
        s.by_image.entry(image.id).or_default().push(slot);
        s.keep.entry(image.id).or_insert_with(|| image.clone());
        STATS[0].fetch_add(1, Ordering::Relaxed);
        Some(slot)
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
            let desc = SamplerDesc::from_static(&victory_shader::emit::StaticSampler::from_words(list[n]));
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

/// The image `id` is gone: its slots are reused once the card is done.
pub(crate) fn image_gone(id: u64) {
    let Some(b) = super::backend() else { return };
    let Some(h) = heaps(b) else { return };
    let value = b.timeline.last();
    let mut s = lock(&h.state);
    if let Some(slots) = s.by_image.remove(&id) {
        for slot in slots {
            s.deferred.push((value, slot));
        }
    }
    s.views.retain(|(i, _), _| *i != id);
    s.keep.remove(&id);
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

    fn from_static(s: &victory_shader::emit::StaticSampler) -> SamplerDesc {
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

// ---- writing the title's descriptors ----

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
    let Some(b) = super::backend() else { return };
    let Some(h) = heaps(b) else { return };
    let started = std::time::Instant::now();
    // The heap the slot is in, and the memory a buffer view names, are read
    // by the card.
    super::touch(handle);
    match view {
        View::Constant(address, _) => super::touch(*address),
        View::Shader(r, _) | View::Unordered(r, _, _) => {
            if let Some((data, _)) = crate::resource::buffer_bytes(*r as crate::object::This) {
                super::touch(data as u64);
            }
        }
        View::Placed(_, _, desc) => super::touch(u64::from(desc[10]) | u64::from(desc[11]) << 32),
        _ => {}
    }
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
            let mut w = gfx10::Buffer::raw(*address, *size).encode().to_vec();
            w.resize(8, 0);
            w
        }
        View::Shader(r, desc) | View::Unordered(r, _, desc) => {
            let unordered = matches!(view, View::Unordered(..));
            let r = *r as crate::object::This;
            match crate::resource::buffer_bytes(r) {
                Some((data, len)) => buffer_words(data as u64, len, desc.as_ref(), unordered),
                None => match crate::resource::texture_memory(r) {
                    Some((data, rdesc)) => texture_words(b, h, data as u64, &rdesc, desc.as_ref(), unordered),
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
                texture_words(b, h, address, &r, Some(&d10), *unordered)
            }
        }
    };
    // SAFETY: the handle is a descriptor slot (views 32 bytes, samplers 16).
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), handle as *mut u32, words.len()) };
    super::split(super::Split::Views, started);
}

/// A V# for a buffer view of `len` bytes at `base` (SRV or UAV words).
fn buffer_words(base: u64, len: u64, desc: Option<&[u32; 10]>, unordered: bool) -> Vec<u32> {
    let mut out = match desc {
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
    .encode()
    .to_vec();
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

/// A T# for a texture view, its slot made now.
fn texture_words(
    b: &Backend,
    h: &Heaps,
    address: u64,
    rdesc: &crate::resource::ResourceDesc,
    desc: Option<&[u32; 10]>,
    unordered: bool,
) -> Vec<u32> {
    let dimension = desc.map_or(0, |d| d[1]);
    let Some(image) = images::at(b, address, rdesc) else {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return null_texture(dimension, unordered);
    };
    let view_format = desc.map_or(0, |d| d[0]);
    let view_format = if view_format == 0 { rdesc.format } else { view_format };
    let mapping = if unordered { 0 } else { desc.map_or(0, |d| d[2]) };
    // The view's kind and ranges from the description (SRV fields from
    // word 4, UAV fields from word 2).
    let d = desc.copied().unwrap_or([0; 10]);
    let f = |i: usize| if unordered { d[i - 2] } else { d[i] };
    let all = u32::MAX;
    let is_3d = image.kind == vk::ImageType::TYPE_3D;
    let is_1d = image.kind == vk::ImageType::TYPE_1D;
    let msaa = image.samples > 1;
    let (kind, base_mip, mips, base_layer, layers) = match (dimension, unordered) {
        (2, _) => (vk::ImageViewType::TYPE_1D, f(4), if unordered { 1 } else { f(5) }, 0, 1),
        (3, false) => (vk::ImageViewType::TYPE_1D_ARRAY, f(4), f(5), f(6), f(7)),
        (3, true) => (vk::ImageViewType::TYPE_1D_ARRAY, f(4), 1, f(5), f(6)),
        (4, false) => (vk::ImageViewType::TYPE_2D, f(4), f(5), 0, 1),
        (4, true) => (vk::ImageViewType::TYPE_2D, f(4), 1, 0, 1),
        (5, false) => (vk::ImageViewType::TYPE_2D_ARRAY, f(4), f(5), f(6), f(7)),
        (5, true) => (vk::ImageViewType::TYPE_2D_ARRAY, f(4), 1, f(5), f(6)),
        (6, _) => (vk::ImageViewType::TYPE_2D, 0, 1, 0, 1),
        (7, false) => (vk::ImageViewType::TYPE_2D_ARRAY, 0, 1, f(4), f(5)),
        (7, true) => (vk::ImageViewType::TYPE_2D_ARRAY, 0, 1, f(4), f(5)),
        (8, false) => (vk::ImageViewType::TYPE_3D, f(4), f(5), 0, 1),
        (8, true) => (vk::ImageViewType::TYPE_3D, f(4), 1, 0, 1),
        (9, _) => (vk::ImageViewType::CUBE, f(4), f(5), 0, 6),
        (10, _) => (vk::ImageViewType::CUBE_ARRAY, f(4), f(5), f(6) * 6, f(7) * 6),
        _ if is_3d => (vk::ImageViewType::TYPE_3D, 0, all, 0, 1),
        _ if is_1d => (vk::ImageViewType::TYPE_1D_ARRAY, 0, all, 0, all),
        _ => (vk::ImageViewType::TYPE_2D_ARRAY, 0, all, 0, all),
    };
    let mips = if mips == 0 || mips == all { image.mips.saturating_sub(base_mip).max(1) } else { mips };
    let layers = if layers == 0 || layers == all { image.layers.saturating_sub(base_layer).max(1) } else { layers };
    let (format, aspect, stencil) = if image.depth() {
        let stencil = images::reads_stencil(view_format);
        let aspect = if stencil { vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
        (image.format, aspect, stencil)
    } else {
        let f = images::color_format(view_format, image.card_written).unwrap_or(image.format);
        let f = if images::format_bytes(f) == images::format_bytes(image.format) { f } else { image.format };
        (f, vk::ImageAspectFlags::COLOR, false)
    };
    let comps = components(mapping);
    let key = ViewKey {
        format,
        kind,
        aspect,
        base_mip,
        mips,
        base_layer,
        layers,
        components: comps,
        storage: unordered,
        attachment: false,
    };
    let Some(slot) = h.image_slot(b, &image, key) else {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return null_texture(dimension, unordered);
    };
    let sampled_format = if image.depth() {
        if stencil { vk::Format::R8_UINT } else { vk::Format::R32_SFLOAT }
    } else {
        format
    };
    let (code, fmt_swizzle) = gfx10::format_of(sampled_format).unwrap_or((0, gfx10::IDENTITY));
    let tkind = match kind {
        vk::ImageViewType::TYPE_1D => gfx10::ImageType::D1,
        vk::ImageViewType::TYPE_1D_ARRAY => gfx10::ImageType::D1Array,
        vk::ImageViewType::TYPE_3D => gfx10::ImageType::D3,
        vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => gfx10::ImageType::Cube,
        vk::ImageViewType::TYPE_2D_ARRAY if msaa => gfx10::ImageType::D2MsaaArray,
        vk::ImageViewType::TYPE_2D_ARRAY => gfx10::ImageType::D2Array,
        _ if msaa => gfx10::ImageType::D2Msaa,
        _ => gfx10::ImageType::D2,
    };
    gfx10::Image {
        address,
        format: code,
        kind: tkind,
        width: image.extent.width,
        height: image.extent.height,
        depth: image.extent.depth,
        first_layer: base_layer,
        layer_count: layers,
        first_mip: base_mip,
        mip_count: mips,
        resource_mips: image.mips,
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
