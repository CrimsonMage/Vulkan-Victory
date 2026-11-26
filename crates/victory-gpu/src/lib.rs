//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(windows)]
mod ambient;
#[cfg(windows)]
mod compose;
pub mod console;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod drvwatch;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg_attr(not(windows), allow(dead_code))]
mod fast;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod frametime;
#[cfg(windows)]
pub mod cores;
#[cfg(windows)]
mod gather;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(all(windows, test))]
mod warp_tests;
#[cfg(windows)]
mod views;
#[cfg(windows)]
pub mod mapstat;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
#[cfg(windows)]
mod quality;
mod procmem;
#[cfg(windows)]
mod release;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod memcap;
#[cfg(windows)]
mod shadow;
#[cfg(windows)]
mod tiles;
#[cfg(windows)]
mod unified;
#[cfg(windows)]
mod vulkan;
mod unified_map;
#[cfg(windows)]
mod unified_probe;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
pub mod upscale;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use release::title_memory_released;
#[cfg(windows)]
pub use unified::{on as unified_on, title_region, title_section, title_section_gone};
#[cfg(windows)]
pub use views::title_memory_view;
#[cfg(windows)]
pub use memory::{GpuPage, page_reservation as title_page_reservation, title_pages_aliased};
#[cfg(windows)]
pub use watch::{
    mapping_lock_held, remapping_over, title_memory_reading, title_memory_remapped,
    title_memory_remapping, title_memory_written,
    writing as kernel_will_write,
};

/// Whether the Vulkan backend runs (`VICTORY_GPU=vulkan`). The runtime then
/// sets up none of the D3D12 path's memory machinery: no write watching,
/// no canary pages, no guards opened for kernel writes.
#[cfg(windows)]
pub fn vulkan_on() -> bool {
    vulkan::on()
}

/// `xmem` took what `[at, at + len)` showed away: a view of physical pages
/// unmapped (back to a placeholder) or a filler stand-in released. What
/// still shows there afterwards is announced again (`title_memory_view`).
/// The Vulkan backend prunes its memory map and imports here; a no-op until
/// it does (the D3D12 path learns of it from `title_memory_remapped`).
#[cfg(windows)]
pub fn title_memory_unmapped(at: usize, len: usize) {
    #[cfg(windows)]
    if vulkan::on() {
        vulkan::title_memory_unmapped(at, len);
    }
    #[cfg(not(windows))]
    let _ = (at, len);
}
