//! Copies and resolves recorded into a list's command buffer.

use std::sync::atomic::Ordering;

use victory_vk::ash::vk;

use super::super::Backend;
use super::super::images::{self, Image};
use super::Rec;
use crate::list::CopyLocation;
use crate::object::This;

impl Rec {
    // ---- copies ----

    pub(super) fn copy_buffer(&mut self, b: &'static Backend, dst: u64, src: u64, len: u64) {
        let (Some((db, doff)), Some((sb, soff))) = (super::super::buffer_of(b, dst), super::super::buffer_of(b, src)) else {
            return;
        };
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: outside rendering; title buffers.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_buffer(cmd, sb, db, &[vk::BufferCopy { src_offset: soff, dst_offset: doff, size: len }]);
            images::full_barrier(&b.host.device, cmd);
        }
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
        for mip in 0..di.mips.min(si.mips) {
            let ext = vk::Extent3D {
                width: (di.extent.width.min(si.extent.width) >> mip).max(1),
                height: (di.extent.height.min(si.extent.height) >> mip).max(1),
                depth: (di.extent.depth.min(si.extent.depth) >> mip).max(1),
            };
            let layers = |i: &Image| vk::ImageSubresourceLayers {
                aspect_mask: i.aspect(),
                mip_level: mip,
                base_array_layer: 0,
                layer_count: di.layers.min(si.layers),
            };
            regions.push(vk::ImageCopy::default().src_subresource(layers(&si)).dst_subresource(layers(&di)).extent(ext));
        }
        self.copy_images(b, &si, &di, &regions);
        di.written[2].fetch_add(1, Ordering::Relaxed);
        self.keep.extend([si, di]);
    }

    pub(super) fn copy_images(&mut self, b: &'static Backend, src: &Image, dst: &Image, regions: &[vk::ImageCopy]) {
        if src.aspect() != dst.aspect() || regions.is_empty() {
            return;
        }
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        // SAFETY: outside rendering; images in GENERAL.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_image(cmd, src.image, vk::ImageLayout::GENERAL, dst.image, vk::ImageLayout::GENERAL, regions);
            images::full_barrier(&b.host.device, cmd);
        }
    }

    pub(crate) fn copy_texture(&mut self, b: &'static Backend, dst: &CopyLocation, x: u32, y: u32, z: u32, src: &CopyLocation, boxed: Option<[u32; 6]>) {
        let sub = |l: &CopyLocation, i: &Image| {
            let index = l.footprint.offset as u32;
            (index % i.mips, index / i.mips)
        };
        let dst_image = if dst.kind == 0 { images::of_resource(b, dst.resource) } else { None };
        let src_image = if src.kind == 0 { images::of_resource(b, src.resource) } else { None };
        match (dst_image, src_image) {
            (Some(di), Some(si)) => {
                let (dm, dl) = sub(dst, &di);
                let (sm, sl) = sub(src, &si);
                let [l, t, f, r, bt, k] = boxed.unwrap_or([0, 0, 0, (si.extent.width >> sm).max(1), (si.extent.height >> sm).max(1), (si.extent.depth >> sm).max(1)]);
                let region = vk::ImageCopy::default()
                    .src_subresource(vk::ImageSubresourceLayers { aspect_mask: si.aspect(), mip_level: sm, base_array_layer: sl, layer_count: 1 })
                    .src_offset(vk::Offset3D { x: l as i32, y: t as i32, z: f as i32 })
                    .dst_subresource(vk::ImageSubresourceLayers { aspect_mask: di.aspect(), mip_level: dm, base_array_layer: dl, layer_count: 1 })
                    .dst_offset(vk::Offset3D { x: x as i32, y: y as i32, z: z as i32 })
                    .extent(vk::Extent3D { width: r.saturating_sub(l).max(1), height: bt.saturating_sub(t).max(1), depth: k.saturating_sub(f).max(1) });
                self.copy_images(b, &si, &di, &[region]);
                di.written[2].fetch_add(1, Ordering::Relaxed);
                self.keep.extend([si, di]);
            }
            (Some(di), None) if src.kind == 1 => {
                // Buffer rows (a placed footprint) into a texture.
                let Some((data, _)) = crate::resource::buffer_bytes(src.resource) else { return };
                let fp = src.footprint;
                let Some((buffer, offset)) = super::super::buffer_of(b, data as u64 + fp.offset) else { return };
                let (dm, dl) = sub(dst, &di);
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
                let (sm, sl) = sub(src, &si);
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
