//! The title's upscale (AMD FidelityFX `ffxDispatch`, `crate::upscale`) on
//! the Vulkan backend: the render-resolution colour scaled into the output
//! with a linear filter (`vkCmdBlitImage`), recorded into the title's list
//! where it asked for the upscale.
//!
//! shortcut: as on the D3D12 path, no temporal accumulation and no
//! sharpening; and the filter at the rendered rectangle's edge may read the
//! texels just past it when the title renders into part of its colour.

use std::sync::atomic::Ordering;

use victory_vk::ash::vk;

use super::super::{Backend, images};
use super::Rec;
use crate::object::This;

/// Whether `resource` is a one-sample 2D texture of one slice.
fn plain(resource: This) -> bool {
    const TEXTURE2D: u32 = 3;
    crate::resource::texture_memory(resource)
        .is_some_and(|(_, d)| d.dimension == TEXTURE2D && d.sample_count <= 1 && d.depth_or_array_size == 1)
}

/// Records the upscale into `list`; false (nothing recorded) if the list
/// or the textures are not of a kind the scale takes: the title's
/// upscaler then runs as before.
pub(crate) fn dispatch(list: This, color: This, output: This, render: (u32, u32), upscale: (u32, u32)) -> bool {
    if !plain(color) || !plain(output) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::error!("Vulkan: the title's upscale is left to its upscaler: textures of another shape"));
        return false;
    }
    crate::list::record_vulkan(list, Box::new(move |rec: &mut Rec, b: &'static Backend| rec.scale(b, color, output, render, upscale)))
}

impl Rec {
    fn scale(&mut self, b: &'static Backend, color: This, output: This, render: (u32, u32), upscale: (u32, u32)) {
        let (Some(src), Some(dst)) = (images::of_resource(b, color), images::of_resource(b, output)) else { return };
        // SAFETY: plain queries on the live physical device.
        let features = |f: vk::Format| unsafe { b.host.instance.get_physical_device_format_properties(b.host.physical, f) }.optimal_tiling_features;
        let (from, to) = (features(src.format), features(dst.format));
        if !from.contains(vk::FormatFeatureFlags::BLIT_SRC) || !to.contains(vk::FormatFeatureFlags::BLIT_DST) || src.depth() || dst.depth() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::error!(from = ?src.format, to = ?dst.format, "Vulkan: the title's upscale cannot be scaled between these formats: not done"));
            return;
        }
        let filter = if from.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let (rw, rh) = crate::upscale::within(render, src.extent.width, src.extent.height);
        let (ow, oh) = crate::upscale::within(upscale, dst.extent.width, dst.extent.height);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                from = format_args!("{rw}x{rh} of {}x{} {:?}", src.extent.width, src.extent.height, src.format),
                to = format_args!("{ow}x{oh} of {}x{} {:?}", dst.extent.width, dst.extent.height, dst.format),
                ?filter,
                "Vulkan: the title's upscale (AMD FidelityFX) runs as a linear scale"
            );
        });
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let corner = |w: u32, h: u32| [vk::Offset3D::default(), vk::Offset3D { x: w as i32, y: h as i32, z: 1 }];
        let region = vk::ImageBlit { src_subresource: layers, src_offsets: corner(rw, rh), dst_subresource: layers, dst_offsets: corner(ow, oh) };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: outside rendering; live images in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_blit_image(cmd, src.image, vk::ImageLayout::GENERAL, dst.image, vk::ImageLayout::GENERAL, &[region], filter);
            images::full_barrier(&b.host.device, cmd);
        }
        dst.written[2].fetch_add(1, Ordering::Relaxed);
        self.keep.extend([src, dst]);
    }
}
