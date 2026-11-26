//! D3D12.X on host Vulkan.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, written down and recorded into
//! Vulkan command buffers by [`vulkan`] (`docs/vulkan-host.md`). The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees.

#[cfg(windows)]
mod capture;
pub mod console;
mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
mod dxgi;
mod fence;
mod frame;
#[cfg(windows)]
mod cores;
#[cfg(windows)]
mod host;
#[cfg(windows)]
pub mod mapstat;
mod layout;
mod list;
pub mod object;
pub mod pipeline;
mod query;
#[cfg(windows)]
mod quality;
mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
mod signature;
#[cfg(windows)]
pub mod upscale;
#[cfg(windows)]
mod vulkan;
#[cfg(windows)]
pub use vulkan::{bake, bake_prune, bake_translate, bake_wanted};
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use vulkan::{
    GpuPage, title_memory_released, title_memory_unmapped, title_memory_view, title_page_reservation,
    title_pages_aliased, title_region, title_section, title_section_gone,
};
#[cfg(windows)]
pub use watch::{
    mapping_lock_held, remapping_over, title_memory_reading, title_memory_remapping,
};

/// The bytes of a write the title's own memory views do not show the card
/// are there: title memory `[at, at + len)` (DirectStorage and file reads
/// into title memory, page writes through the runtime's views). The
/// Vulkan backend fills the card pages that show them again.
#[cfg(windows)]
pub fn title_memory_written(at: usize, len: usize) {
    if len > 0 {
        vulkan::title_memory_written(at, len);
    }
}
