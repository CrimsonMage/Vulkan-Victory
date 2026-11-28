//! Copies and resolves recorded into a list's command buffer.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_vk::ash::vk;

use super::super::Backend;
use super::super::images::{self, Image};
use super::Rec;
use crate::list::CopyLocation;
use crate::object::This;

/// The format of the plane a depth-to-colour copy moves: a depth image's
/// depth plane, a colour image's own.
fn plane_format(i: &Image) -> vk::Format {
    match i.format {
        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D32_SFLOAT => vk::Format::R32_SFLOAT,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 => vk::Format::R32_UINT,
        vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => vk::Format::R16_UNORM,
        f => f,
    }
}

impl Rec {
    /// A depth buffer the title also stores into through storage views
    /// has a colour image a plane for those (`images::written_plane`):
    /// the planes are copied into the depth image before it is attached
    /// (`into_depth`), and back once draws have written it.
    ///
    /// # Safety
    /// Recording into `cmd`, outside rendering.
    pub(super) unsafe fn copy_planes(&mut self, b: &Backend, cmd: vk::CommandBuffer, depth: &std::sync::Arc<Image>, into_depth: bool) {
        if !b.host.caps.maintenance8 {
            return;
        }
        for (plane, aspect) in depth.planes().into_iter().zip([vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL]) {
            let Some(plane) = plane.filter(|_| depth.aspect().contains(aspect)) else { continue };
            let layers = |aspect| vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 };
            let extent = vk::Extent3D { width: depth.extent.width.min(plane.extent.width), height: depth.extent.height.min(plane.extent.height), depth: 1 };
            let (src, dst, from, to) = if into_depth {
                (plane.image, depth.image, vk::ImageAspectFlags::COLOR, aspect)
            } else {
                (depth.image, plane.image, aspect, vk::ImageAspectFlags::COLOR)
            };
            let region = vk::ImageCopy::default().src_subresource(layers(from)).dst_subresource(layers(to)).extent(extent);
            // SAFETY: as the caller says; live images in GENERAL.
            unsafe {
                images::full_barrier(&b.host.device, cmd);
                b.host.device.cmd_copy_image(cmd, src, vk::ImageLayout::GENERAL, dst, vk::ImageLayout::GENERAL, &[region]);
                images::full_barrier(&b.host.device, cmd);
            }
            super::PLANE_COPIES.fetch_add(1, Ordering::Relaxed);
            self.keep.push(plane.clone());
        }
    }

    // ---- copies ----

    pub(super) fn copy_buffer(&mut self, b: &'static Backend, dst: u64, src: u64, len: u64) {
        let (Some((db, doff)), Some((sb, soff))) = (super::super::buffer_over(b, dst, len), super::super::buffer_over(b, src, len)) else {
            return;
        };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: recording into our command buffer.
        unsafe { super::super::gputime::mark(b, cmd, || "title: a buffer copy".into()) };
        // It may write indirect records: no window holds.
        self.indirect.clear();
        self.planes_synced.clear();
        // SAFETY: outside rendering; title buffers.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_buffer(cmd, sb, db, &[vk::BufferCopy { src_offset: soff, dst_offset: doff, size: len }]);
            images::full_barrier(&b.host.device, cmd);
        }
        self.note(|| format!("copy {len} bytes {src:#x} -> {dst:#x}"));
    }

    pub(crate) fn copy_memory(&mut self, b: &'static Backend, dst: u64, src: u64, len: u64) {
        self.copy_buffer(b, dst, src, len);
    }

    pub(crate) fn copy_buffer_region(&mut self, b: &'static Backend, dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64) {
        let (Some((d, _)), Some((s, _))) = (crate::resource::buffer_bytes(dst), crate::resource::buffer_bytes(src)) else {
            return;
        };
        self.copy_buffer(b, d as u64 + dst_offset, s as u64 + src_offset, len);
    }

    pub(crate) fn copy_resource(&mut self, b: &'static Backend, dst: This, src: This) {
        if let (Some((d, len)), Some((s, _))) = (crate::resource::buffer_bytes(dst), crate::resource::buffer_bytes(src)) {
            self.copy_buffer(b, d as u64, s as u64, len);
            return;
        }
        let (Some(di), Some(si)) = (images::of_resource(b, dst), images::of_resource(b, src)) else {
            return;
        };
        let mut regions = Vec::new();
        // Level by console level (texture quality may leave out the largest
        // of either image).
        for mip in 0..di.mips {
            let Some(sm) = (mip + di.dropped).checked_sub(si.dropped).filter(|&m| m < si.mips) else { continue };
            let ext = vk::Extent3D {
                width: (di.extent.width >> mip).min(si.extent.width >> sm).max(1),
                height: (di.extent.height >> mip).min(si.extent.height >> sm).max(1),
                depth: (di.extent.depth >> mip).min(si.extent.depth >> sm).max(1),
            };
            let layers = |i: &Image, m: u32| vk::ImageSubresourceLayers {
                aspect_mask: i.aspect(),
                mip_level: m,
                base_array_layer: 0,
                layer_count: di.layers.min(si.layers),
            };
            regions.push(vk::ImageCopy::default().src_subresource(layers(&si, sm)).dst_subresource(layers(&di, mip)).extent(ext));
        }
        self.copy_images(b, &si, &di, &regions);
        self.note(|| format!("copy image {} -> image {} ({} levels)", si.id, di.id, regions.len()));
        di.written[2].fetch_add(1, Ordering::Relaxed);
        self.keep.extend([si, di]);
    }

    pub(super) fn copy_images(&mut self, b: &'static Backend, src: &Image, dst: &Image, regions: &[vk::ImageCopy]) {
        if regions.is_empty() {
            return;
        }
        // The console copies a depth plane into a colour texture of the
        // same bits and back (the title keeps a copy of its depth and
        // writes it back with PS_Copy_SRV_To_Depth): one aspect of each
        // side, which the card takes with VK_KHR_maintenance8. Skipping
        // these left the depth buffer empty for everything after.
        let crossed: Vec<vk::ImageCopy>;
        let regions = if src.depth() == dst.depth() {
            regions
        } else {
            static TOLD: AtomicU64 = AtomicU64::new(0);
            let can = b.host.caps.maintenance8 && images::format_bytes(plane_format(src)) == images::format_bytes(plane_format(dst));
            if TOLD.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(from = ?src.format, to = ?dst.format, copied = can, "Vulkan: a copy between a depth image and a colour image");
            }
            if !can {
                return;
            }
            let one = |i: &Image| if i.depth() { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
            crossed = regions
                .iter()
                .map(|r| {
                    let mut r = *r;
                    r.src_subresource.aspect_mask = one(src);
                    r.dst_subresource.aspect_mask = one(dst);
                    r
                })
                .collect();
            &crossed
        };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: recording into our command buffer.
        unsafe { super::super::gputime::mark(b, cmd, || "title: an image copy".into()) };
        // SAFETY: outside rendering; images in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_image(cmd, src.image, vk::ImageLayout::GENERAL, dst.image, vk::ImageLayout::GENERAL, regions);
            images::full_barrier(&b.host.device, cmd);
        }
    }

    pub(crate) fn copy_texture(&mut self, b: &'static Backend, dst: &CopyLocation, x: u32, y: u32, z: u32, src: &CopyLocation, boxed: Option<[u32; 6]>) {
        // A level texture quality left out is not copied.
        let sub = |l: &CopyLocation, i: &Image| images::subresource(i, l.footprint.offset as u32);
        let dst_image = if dst.kind == 0 { images::of_resource(b, dst.resource) } else { None };
        let src_image = if src.kind == 0 { images::of_resource(b, src.resource) } else { None };
        match (dst_image, src_image) {
            (Some(di), Some(si)) => {
                let (Some((dm, dl)), Some((sm, sl))) = (sub(dst, &di), sub(src, &si)) else { return };
                let [l, t, f, r, bt, k] = boxed.unwrap_or([0, 0, 0, (si.extent.width >> sm).max(1), (si.extent.height >> sm).max(1), (si.extent.depth >> sm).max(1)]);
                let region = vk::ImageCopy::default()
                    .src_subresource(vk::ImageSubresourceLayers { aspect_mask: si.aspect(), mip_level: sm, base_array_layer: sl, layer_count: 1 })
                    .src_offset(vk::Offset3D { x: l as i32, y: t as i32, z: f as i32 })
                    .dst_subresource(vk::ImageSubresourceLayers { aspect_mask: di.aspect(), mip_level: dm, base_array_layer: dl, layer_count: 1 })
                    .dst_offset(vk::Offset3D { x: x as i32, y: y as i32, z: z as i32 })
                    .extent(vk::Extent3D { width: r.saturating_sub(l).max(1), height: bt.saturating_sub(t).max(1), depth: k.saturating_sub(f).max(1) });
                self.copy_images(b, &si, &di, &[region]);
                self.note(|| format!("copy region image {} -> image {}", si.id, di.id));
                di.written[2].fetch_add(1, Ordering::Relaxed);
                self.keep.extend([si, di]);
            }
            (Some(di), None) if src.kind == 1 => {
                // Buffer rows (a placed footprint) into a texture.
                let Some((data, _)) = crate::resource::buffer_bytes(src.resource) else { return };
                let fp = src.footprint;
                let Some((buffer, offset)) = super::super::buffer_of(b, data as u64 + fp.offset) else { return };
                let Some((dm, dl)) = sub(dst, &di) else { return };
                let (element, block) = crate::resource::element(fp.format).unwrap_or((4, 1));
                let row_texels = (u64::from(fp.row_pitch) / element * block) as u32;
                let [l, t, f, r, bt, k] = boxed.unwrap_or([0, 0, 0, fp.width, fp.height, fp.depth]);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset + u64::from(t / block as u32) * u64::from(fp.row_pitch) + u64::from(l / block as u32) * element)
                    .buffer_row_length(row_texels)
                    .buffer_image_height(fp.height.next_multiple_of(block as u32))
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: di.aspect() & !vk::ImageAspectFlags::STENCIL, mip_level: dm, base_array_layer: dl, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: z as i32 })
                    .image_extent(vk::Extent3D { width: r.saturating_sub(l).max(1), height: bt.saturating_sub(t).max(1), depth: k.saturating_sub(f).max(1) });
                self.end_pass(b);
                let Some(cmd) = self.cmd(b) else { return };
                // SAFETY: recording into our command buffer.
                unsafe { super::super::gputime::mark(b, cmd, || "title: a texture copy".into()) };
                // SAFETY: outside rendering; a title buffer and an image in GENERAL.
                unsafe {
                    images::full_barrier(&b.host.device, cmd);
                    b.host.device.cmd_copy_buffer_to_image(cmd, buffer, di.image, vk::ImageLayout::GENERAL, &[region]);
                    images::full_barrier(&b.host.device, cmd);
                }
                di.written[2].fetch_add(1, Ordering::Relaxed);
                self.keep.push(di);
            }
            (None, Some(si)) if dst.kind == 1 => {
                let Some((data, _)) = crate::resource::buffer_bytes(dst.resource) else { return };
                let fp = dst.footprint;
                let Some((buffer, offset)) = super::super::buffer_of(b, data as u64 + fp.offset) else { return };
                let Some((sm, sl)) = sub(src, &si) else { return };
                let (element, block) = crate::resource::element(fp.format).unwrap_or((4, 1));
                let [l, t, f, r, bt, k] = boxed.unwrap_or([0, 0, 0, (si.extent.width >> sm).max(1), (si.extent.height >> sm).max(1), (si.extent.depth >> sm).max(1)]);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset + u64::from(y / block as u32) * u64::from(fp.row_pitch) + u64::from(x / block as u32) * element)
                    .buffer_row_length((u64::from(fp.row_pitch) / element * block) as u32)
                    .buffer_image_height(fp.height.next_multiple_of(block as u32))
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: si.aspect() & !vk::ImageAspectFlags::STENCIL, mip_level: sm, base_array_layer: sl, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: l as i32, y: t as i32, z: f as i32 })
                    .image_extent(vk::Extent3D { width: r.saturating_sub(l).max(1), height: bt.saturating_sub(t).max(1), depth: k.saturating_sub(f).max(1) });
                self.end_pass(b);
                let Some(cmd) = self.cmd(b) else { return };
                // SAFETY: recording into our command buffer.
                unsafe { super::super::gputime::mark(b, cmd, || "title: a texture copy".into()) };
                // SAFETY: outside rendering; an image in GENERAL and a title buffer.
                unsafe {
                    images::full_barrier(&b.host.device, cmd);
                    b.host.device.cmd_copy_image_to_buffer(cmd, si.image, vk::ImageLayout::GENERAL, buffer, &[region]);
                    images::full_barrier(&b.host.device, cmd);
                }
                self.keep.push(si);
            }
            _ => {}
        }
    }

    pub(crate) fn resolve(&mut self, b: &'static Backend, dst: This, dst_sub: u32, src: This, src_sub: u32) {
        let (Some(di), Some(si)) = (images::of_resource(b, dst), images::of_resource(b, src)) else { return };
        if si.samples <= 1 || di.samples != 1 {
            return;
        }
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: recording into our command buffer.
        unsafe { super::super::gputime::mark(b, cmd, || "title: a resolve".into()) };
        let layers = |i: &Image, s: u32| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: s % i.mips,
            base_array_layer: s / i.mips,
            layer_count: 1,
        };
        let region = vk::ImageResolve::default()
            .src_subresource(layers(&si, src_sub))
            .dst_subresource(layers(&di, dst_sub))
            .extent(vk::Extent3D { width: si.extent.width.min(di.extent.width), height: si.extent.height.min(di.extent.height), depth: 1 });
        // SAFETY: outside rendering; images in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_resolve_image(cmd, si.image, vk::ImageLayout::GENERAL, di.image, vk::ImageLayout::GENERAL, &[region]);
            images::full_barrier(&b.host.device, cmd);
        }
        self.keep.extend([si, di]);
    }
}
