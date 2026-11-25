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
pub mod fence;
pub mod frame;
#[cfg(windows)]
pub mod cores;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
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
pub use unified::{on as unified_on, title_region, title_section, title_section_gone};
#[cfg(windows)]
pub use views::title_memory_view;
#[cfg(windows)]
pub use watch::{
    mapping_lock_held, remapping_over, title_memory_reading, title_memory_remapped,
    title_memory_remapping, title_memory_written,
    writing as kernel_will_write,
};
