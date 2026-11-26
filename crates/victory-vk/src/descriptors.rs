//! Descriptors in memory (`docs/vulkan-host.md`, "Descriptors").
//!
//! With `VK_EXT_descriptor_buffer` a descriptor is bytes the backend writes
//! (`vkGetDescriptorEXT`) into a buffer the shaders read by address, as the
//! console's T#, V# and S# are bytes in the title's heaps. Without it, one
//! big descriptor-indexing set stands in (`Fallback`).

use anyhow::{Context, Result};
use ash::vk;

use crate::host::Host;

/// The driver's descriptor sizes and limits (`None` without the extension).
pub fn properties(host: &Host) -> Option<vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'static>> {
    if !host.caps.descriptor_buffer {
        return None;
    }
    let mut db = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
    {
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut db);
        // SAFETY: a plain query on a live physical device.
        unsafe {
            host.instance
                .get_physical_device_properties2(host.physical, &mut p2)
        };
    }
    db.p_next = std::ptr::null_mut();
    Some(db)
}

/// A set layout of one storage buffer at binding 0 (the probe's result),
/// laid out for a descriptor buffer when `in_memory`.
pub fn storage_layout(host: &Host, in_memory: bool) -> Result<vk::DescriptorSetLayout> {
    let binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let flags = if in_memory {
        vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT
    } else {
        vk::DescriptorSetLayoutCreateFlags::empty()
    };
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(flags)
        .bindings(&binding);
    // SAFETY: valid create info.
    unsafe { host.device.create_descriptor_set_layout(&info, None) }
        .context("vkCreateDescriptorSetLayout")
}

/// A layout's size in a descriptor buffer and the offset of binding 0.
pub fn layout_size(host: &Host, layout: vk::DescriptorSetLayout) -> Option<(u64, u64)> {
    let db = host.descriptor_buffer.as_ref()?;
    // SAFETY: the layout was made with the descriptor-buffer flag.
    unsafe {
        Some((
            db.get_descriptor_set_layout_size(layout),
            db.get_descriptor_set_layout_binding_offset(layout, 0),
        ))
    }
}

/// Writes a storage-buffer descriptor for `range` bytes at device address
/// `address` into `out` (the descriptor's bytes in a descriptor buffer).
pub fn write_storage_buffer(host: &Host, address: u64, range: u64, out: &mut [u8]) {
    let db = host
        .descriptor_buffer
        .as_ref()
        .expect("descriptor buffer enabled");
    let info = vk::DescriptorAddressInfoEXT::default()
        .address(address)
        .range(range)
        .format(vk::Format::UNDEFINED);
    let get = vk::DescriptorGetInfoEXT::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .data(vk::DescriptorDataEXT {
            p_storage_buffer: &info,
        });
    // SAFETY: `out` is the size the driver reports for storage buffers.
    unsafe { db.get_descriptor(&get, out) };
}

/// The fallback: a pool and one set (in the backend proper, one
/// heap-sized set of mutable descriptors with update-after-bind).
pub struct Fallback {
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

impl Fallback {
    pub fn new(
        host: &Host,
        layout: vk::DescriptorSetLayout,
        buffer: vk::Buffer,
        range: u64,
    ) -> Result<Fallback> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: valid create and allocate infos; the set is written
        // before any use.
        unsafe {
            let pool = host.device.create_descriptor_pool(&info, None)?;
            let layouts = [layout];
            let set = host.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?[0];
            let buffers = [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(range)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffers);
            host.device.update_descriptor_sets(&[write], &[]);
            Ok(Fallback { pool, set })
        }
    }

    /// # Safety
    /// No submitted work still uses the set.
    pub unsafe fn destroy(&self, host: &Host) {
        unsafe { host.device.destroy_descriptor_pool(self.pool, None) };
    }
}
