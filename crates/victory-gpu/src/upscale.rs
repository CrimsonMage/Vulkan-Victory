//! The title's upscale (AMD FidelityFX, `ffxDispatch` with an upscale
//! description).
//!
//! On Series S the title renders the world at 1280x720 and has the AMD
//! upscaler (FSR4, `amd_fidelityfx_upscaler_xbox.dll`) bring it to
//! 1920x1080; everything after (post-processing, tone mapping, the scene
//! plane) reads the upscaler's output. Series X uses the title's own
//! temporal pass instead, which runs.
//!
//! The upscaler's passes are compute programs like the title's, made by
//! another compiler: console code in an `ACCC` part with no `DXIL` part
//! (`victory_shader::native`, `native/fspx.rs`). Its
//! `DeserializeComputePipelineStateX` path fails here and its fallback
//! hands the same containers to `CreateComputePipelineState`, so they go
//! the way of every compute stage. **When every pass translates
//! ([`passes_translate`]) and none has failed on the card since
//! ([`pass_failed`]), the title's own `ffxDispatch` runs the upscale** and
//! nothing here does.
//!
//! Otherwise the stand-in below: the render-resolution colour scaled into
//! the output with a linear filter, in the title's list where it asked for
//! the upscale (recorded by the Vulkan backend,
//! `vulkan/record/upscale.rs`). Before the passes translated none of them
//! ran and the output was never written: the world showed black, the
//! tone-mapper's dither on zero (run s4, 2026-11-25).
//!
//! shortcut (the stand-in): no temporal accumulation and no sharpening;
//! the scene keeps its sub-pixel jitter.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::object::This;

/// `FFX_API_DISPATCH_DESC_TYPE_UPSCALE`.
pub const UPSCALE: u64 = 0x0001_0001;

/// `FfxApiResource` (FidelityFX API `ffx_api_types.h`): the resource, its
/// description (type, format, width, height, depth, mips, flags, usage)
/// and its state.
#[repr(C)]
pub struct ApiResource {
    pub resource: *mut std::ffi::c_void,
    pub description: [u32; 8],
    pub state: u32,
}

/// `ffxDispatchDescUpscale` as the FidelityFX API's `ffx_upscale.h` lays
/// it out (the same from FSR 3.1, the API's first upscaler, to FSR 4; the
/// caller checks the upscaler's version).
#[repr(C)]
pub struct DispatchDescUpscale {
    pub kind: u64,
    pub next: *const std::ffi::c_void,
    pub command_list: *mut std::ffi::c_void,
    pub color: ApiResource,
    pub depth: ApiResource,
    pub motion_vectors: ApiResource,
    pub exposure: ApiResource,
    pub reactive: ApiResource,
    pub transparency_and_composition: ApiResource,
    pub output: ApiResource,
    pub jitter_offset: [f32; 2],
    pub motion_vector_scale: [f32; 2],
    pub render_size: [u32; 2],
    pub upscale_size: [u32; 2],
    pub enable_sharpening: bool,
    pub sharpness: f32,
    pub frame_time_delta: f32,
    pub pre_exposure: f32,
    pub reset: bool,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub flags: u32,
}
const _: () = {
    use std::mem::offset_of;
    assert!(size_of::<ApiResource>() == 0x30);
    assert!(offset_of!(DispatchDescUpscale, color) == 0x18);
    assert!(offset_of!(DispatchDescUpscale, output) == 0x138);
    assert!(offset_of!(DispatchDescUpscale, render_size) == 0x178);
    assert!(offset_of!(DispatchDescUpscale, flags) == 0x1ac);
    assert!(size_of::<DispatchDescUpscale>() == 0x1b0);
};

/// How many passes the upscaler in `dll` (its file's bytes) has, if every
/// one translates: its containers whose console code is in an `ACCC`
/// part, each through the translator as the stage workers will put it.
/// Else the first that does not, and why (or that there is none).
pub fn passes_translate(dll: &[u8]) -> Result<usize, String> {
    victory_shader::vk::report::accc_passes(dll)
}

/// A pass that translated did not become a pipeline (refused for the
/// device's limits, or the driver's compile failed).
static PASS_FAILED: AtomicBool = AtomicBool::new(false);

/// The stage workers' word that the stage of `container` failed: if it is
/// a pass of the upscaler, the upscales are the stand-in's from now on.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn stage_failed(container: &[u8], hash: &str) {
    if crate::pipeline::is_pass(container) && !PASS_FAILED.swap(true, Ordering::Relaxed) {
        tracing::error!(hash, "a pass of the title's upscaler is no pipeline on this card: its upscales run as a linear scale from now on");
    }
}

/// Whether a pass of the upscaler failed on the card.
pub fn pass_failed() -> bool {
    PASS_FAILED.load(Ordering::Relaxed)
}

/// `width` x `height` if both are set and fit in `w` x `h`, else all of it.
pub(crate) fn within((width, height): (u32, u32), w: u32, h: u32) -> (u32, u32) {
    if width == 0 || height == 0 || width > w || height > h {
        (w, h)
    } else {
        (width, height)
    }
}

/// Runs the upscale `desc` describes as a linear scale (the stand-in).
/// False, recording nothing, if its list or textures are not ours or not
/// of a shape the pass takes: the title's upscaler then runs as before.
///
/// # Safety
///
/// `desc` is an `ffxDispatchDescUpscale` (its type checked by the caller),
/// as the title passes it to `ffxDispatch`, on the thread recording its
/// command list.
pub unsafe fn dispatch(desc: *const u8) -> bool {
    // SAFETY: as the caller says.
    let d = unsafe { &*desc.cast::<DispatchDescUpscale>() };
    let (list, color, output) = (
        d.command_list as This,
        d.color.resource as This,
        d.output.resource as This,
    );
    let (render, upscale) = ((d.render_size[0], d.render_size[1]), (d.upscale_size[0], d.upscale_size[1]));
    if !crate::resource::is_resource(color) || !crate::resource::is_resource(output) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::error!("the title's upscale is left to its upscaler: its colour or output is no resource of ours"));
        return false;
    }
    crate::vulkan::record::upscale::dispatch(list, color, output, render, upscale)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_file_without_passes_has_no_upscaler_to_run() {
        assert!(super::passes_translate(b"MZ and no shader").is_err());
        assert!(!super::pass_failed());
    }

    #[test]
    fn sizes_outside_the_texture_mean_all_of_it() {
        assert_eq!(super::within((0, 0), 1280, 720), (1280, 720));
        assert_eq!(super::within((1920, 1080), 1280, 720), (1280, 720));
        assert_eq!(super::within((1152, 648), 1280, 720), (1152, 648));
    }
}
