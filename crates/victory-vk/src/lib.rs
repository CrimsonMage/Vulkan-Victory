//! The Vulkan host backend (`docs/vulkan-host.md`).
//!
//! Victory turns the title's D3D12.X into Vulkan directly: title memory is
//! imported in place and addressed by device address (a raw console
//! address is `base + offset`), descriptors live in memory as on the
//! console (`VK_EXT_descriptor_buffer`), translated programs are SPIR-V,
//! and frames are presented on the title's window through winevulkan.
//!
//! Milestone 1 is the foundation and its self-test, `victory vkprobe`:
//! [`host`] (instance, device, queues, what is missing), [`memory`] (title
//! memory imports, the backend's own buffers), [`descriptors`],
//! [`compute`] (a pipeline over a SPIR-V program) and, on Windows,
//! presentation and the probe itself.
//!
//! A separate crate, not a module of `victory-gpu`: the backend shares no
//! code with the D3D12 path it replaces, builds in seconds on its own, and
//! keeps `ash` out of the D3D12 path until the switch-over (M2 onward wires
//! it in behind `VICTORY_GPU`).

pub use ash;

pub mod account;
pub mod compute;
pub mod descriptors;
pub mod gfx10;
pub mod graphics;
pub mod host;
pub mod limits;
pub mod map;
pub mod memory;
pub mod pool;
#[cfg(windows)]
pub mod present;
#[cfg(windows)]
mod probe;
pub mod swapchain;
pub mod timeline;

/// `victory vkprobe`: runs every step and prints one line per step.
/// `Ok(true)` when every required step passed.
pub fn vkprobe() -> anyhow::Result<bool> {
    #[cfg(windows)]
    {
        probe::run()
    }
    #[cfg(not(windows))]
    {
        anyhow::bail!("vkprobe runs on the Windows build (under Proton on Linux)")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn spirv_header() {
        let w = crate::compute::SPIRV;
        assert_eq!(w[0], 0x0723_0203, "SPIR-V magic");
        assert!(w[1] >= 0x0001_0500, "SPIR-V 1.5 or later (Vulkan 1.3)");
        assert!(w.len() > 5);
    }

    #[test]
    fn push_layout() {
        assert_eq!(size_of::<crate::compute::Push>(), 24);
    }
}
