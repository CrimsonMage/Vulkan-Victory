//! Title memory and the backend's own buffers (`docs/vulkan-host.md`,
//! "Resources and title memory").
//!
//! Title memory is *imported*, never allocated: a range the title already
//! has (an `XMemVirtualAlloc` region, the whole view victory keeps of each
//! physical-page section) becomes a `VkDeviceMemory` by
//! `VK_EXT_external_memory_host` and one buffer spanning it, whose device
//! address is the range's base on the card. A raw title address then is
//! `device_base + (address - title_base)`: no copies, no page directory.
//! The host pointer must stay mapped while the import lives (the spec's
//! rule; vkd3d-proton unmaps its view right after import and relies on the
//! driver's pinning, which the backend does not).

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::host::Host;

/// Buffer usages every title-memory buffer gets: it is read and written
/// as raw memory, used as index, vertex and indirect data, copied, and
/// addressed by pointer.
pub const TITLE_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

/// A range of title memory imported in place.
pub struct Imported {
    pub memory: vk::DeviceMemory,
    pub buffer: vk::Buffer,
    /// Where the title sees the range.
    pub title_base: u64,
    pub len: u64,
    /// Where the card sees it.
    pub device_base: u64,
    pub memory_type: u32,
}

impl Imported {
    /// Imports `len` bytes at `at` (both on the driver's import alignment)
    /// with `usage` in addition to [`TITLE_USAGE`].
    ///
    /// # Safety
    /// `at..at + len` is committed, readable and writable memory of this
    /// process that stays mapped until [`Imported::destroy`].
    pub unsafe fn new(
        host: &Host,
        at: *mut core::ffi::c_void,
        len: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Imported> {
        let align = host.caps.min_imported_host_pointer_alignment.max(1);
        if !(at as u64).is_multiple_of(align) || !len.is_multiple_of(align) {
            bail!("import of {len:#x} bytes at {at:p} is not on the driver's {align:#x} alignment");
        }
        let device = &host.device;
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut pointer = vk::MemoryHostPointerPropertiesEXT::default();
        unsafe {
            (host.host_memory.fp().get_memory_host_pointer_properties_ext)(
                device.handle(),
                handle,
                at,
                &mut pointer,
            )
        }
        .result()
        .context("vkGetMemoryHostPointerPropertiesEXT")?;

        let mut external_props = vk::ExternalBufferProperties::default();
        unsafe {
            host.instance
                .get_physical_device_external_buffer_properties(
                    host.physical,
                    &vk::PhysicalDeviceExternalBufferInfo::default()
                        .usage(TITLE_USAGE | usage)
                        .handle_type(handle),
                    &mut external_props,
                )
        };
        let m = external_props.external_memory_properties;
        if !m
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            || !m.compatible_handle_types.contains(handle)
        {
            bail!(
                "the driver does not import host memory into buffers of usage {:?}",
                TITLE_USAGE | usage
            );
        }
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(TITLE_USAGE | usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let buffer =
            unsafe { device.create_buffer(&info, None) }.context("vkCreateBuffer (import)")?;
        let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let bits = needs.memory_type_bits & pointer.memory_type_bits;
        let Some(memory_type) = host
            .memory_type(
                bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::empty(),
            )
            .or_else(|| {
                host.memory_type(
                    bits,
                    vk::MemoryPropertyFlags::empty(),
                    vk::MemoryPropertyFlags::empty(),
                )
            })
        else {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!(
                "no memory type takes both the buffer (types {:#x}) and the host pointer (types {:#x})",
                needs.memory_type_bits,
                pointer.memory_type_bits
            );
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle)
            .host_pointer(at);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(len)
            .memory_type_index(memory_type)
            .push_next(&mut import)
            .push_next(&mut flags);
        let memory = match unsafe { device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                bail!("vkAllocateMemory (import of {len:#x} bytes at {at:p}): {e}");
            }
        };
        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            bail!("vkBindBufferMemory (import): {e}");
        }
        let device_base = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok(Imported {
            memory,
            buffer,
            title_base: at as u64,
            len,
            device_base,
            memory_type,
        })
    }

    /// The card's address of a title address in this range.
    pub fn device_address(&self, title: u64) -> Option<u64> {
        let offset = title.checked_sub(self.title_base)?;
        (offset < self.len).then(|| self.device_base + offset)
    }

    /// # Safety
    /// No submitted work still uses the import.
    pub unsafe fn destroy(&self, host: &Host) {
        unsafe {
            host.device.destroy_buffer(self.buffer, None);
            host.device.free_memory(self.memory, None);
        }
    }
}

/// Where a buffer of the backend's own lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    /// Card memory, not mapped (streamed-once copies, render scratch).
    Card,
    /// PC memory the CPU writes and the card reads (descriptors, constants).
    Upload,
    /// PC memory the card writes and the CPU reads.
    Readback,
}

/// A buffer of the backend's own, with its device address and mapping.
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub address: u64,
    /// The CPU's view (`Upload` and `Readback`).
    pub mapped: *mut u8,
}

// SAFETY: Vulkan handles and a mapping of memory that lives as long as the
// buffer; callers order their own reads and writes of the mapping.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(
        host: &Host,
        size: u64,
        usage: vk::BufferUsageFlags,
        place: Place,
    ) -> Result<Buffer> {
        let device = &host.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        // SAFETY: valid create info; the buffer is destroyed by `destroy`.
        let buffer = unsafe { device.create_buffer(&info, None) }.context("vkCreateBuffer")?;
        let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
        use vk::MemoryPropertyFlags as M;
        let (want, avoid) = match place {
            Place::Card => (M::DEVICE_LOCAL, M::HOST_VISIBLE),
            Place::Upload => (M::HOST_VISIBLE | M::HOST_COHERENT, M::DEVICE_LOCAL),
            Place::Readback => (
                M::HOST_VISIBLE | M::HOST_COHERENT | M::HOST_CACHED,
                M::DEVICE_LOCAL,
            ),
        };
        let memory_type = host
            .memory_type(needs.memory_type_bits, want, avoid)
            .or_else(|| match place {
                Place::Card => host.memory_type(needs.memory_type_bits, M::empty(), M::empty()),
                _ => host.memory_type(
                    needs.memory_type_bits,
                    M::HOST_VISIBLE | M::HOST_COHERENT,
                    M::empty(),
                ),
            })
            .context("no memory type for a buffer")?;
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(memory_type)
            .push_next(&mut flags);
        let memory = unsafe { device.allocate_memory(&alloc, None) }.context("vkAllocateMemory")?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
        let mapped = if place == Place::Card {
            std::ptr::null_mut()
        } else {
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }?
                .cast()
        };
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok(Buffer {
            buffer,
            memory,
            size,
            address,
            mapped,
        })
    }

    /// # Safety
    /// No submitted work still uses the buffer.
    pub unsafe fn destroy(&self, host: &Host) {
        unsafe {
            host.device.destroy_buffer(self.buffer, None);
            host.device.free_memory(self.memory, None);
        }
    }
}
