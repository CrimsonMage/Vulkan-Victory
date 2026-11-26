//! Clears as D3D12 clears: only the view's mip and slice, only the given
//! rectangles, in a rendering of their own (`vkCmdClearAttachments`).

use std::sync::atomic::Ordering;

use victory_vk::ash::vk;

use super::super::Backend;
use super::{Rec, target_view};
use crate::descriptor::RtvRecord;
use crate::list::Rect;

impl Rec {
    /// A clear of a target view, of `rects` (none: all of it), as D3D12
    /// clears: only the view's mip and slice, only the rectangles.
    pub(crate) fn clear_color(&mut self, b: &'static Backend, r: &RtvRecord, color: [f32; 4], rects: &[Rect]) {
        let Some((image, view, format, extent)) = target_view(b, r, false) else { return };
        let value = if victory_vk::gfx10::is_integer(format) {
            vk::ClearColorValue { uint32: color.map(|c| c as u32) }
        } else {
            vk::ClearColorValue { float32: color }
        };
        let clear = vk::ClearAttachment {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            color_attachment: 0,
            clear_value: vk::ClearValue { color: value },
        };
        self.clear_view(b, view, extent, None, clear, rects);
        image.written[1].fetch_add(1, Ordering::Relaxed);
        self.keep.push(image);
    }

    pub(crate) fn clear_depth(&mut self, b: &'static Backend, r: &RtvRecord, flags: u32, depth: f32, stencil: u8, rects: &[Rect]) {
        let Some((image, view, _, extent)) = target_view(b, r, true) else { return };
        let mut aspect = vk::ImageAspectFlags::empty();
        if flags & 1 != 0 {
            aspect |= vk::ImageAspectFlags::DEPTH;
        }
        if flags & 2 != 0 {
            aspect |= vk::ImageAspectFlags::STENCIL;
        }
        aspect &= image.aspect();
        if aspect.is_empty() {
            return;
        }
        let clear = vk::ClearAttachment {
            aspect_mask: aspect,
            color_attachment: 0,
            clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth, stencil: u32::from(stencil) } },
        };
        self.clear_view(b, view, extent, Some(image.aspect()), clear, rects);
        self.keep.push(image);
    }

    /// Clears `view` (a colour view, or a depth view of `depth` aspects) in
    /// a rendering of its own.
    fn clear_view(&mut self, b: &'static Backend, view: vk::ImageView, extent: (u32, u32), depth: Option<vk::ImageAspectFlags>, clear: vk::ClearAttachment, rects: &[Rect]) {
        self.end_pass(b);
        let Some(cmd) = self.cmd(b) else { return };
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } };
        let rects: Vec<vk::ClearRect> = if rects.is_empty() {
            vec![vk::ClearRect { rect: area, base_array_layer: 0, layer_count: 1 }]
        } else {
            rects.iter().filter_map(|r| clip_rect(r, extent)).map(|rect| vk::ClearRect { rect, base_array_layer: 0, layer_count: 1 }).collect()
        };
        let attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let mut info = vk::RenderingInfo::default().render_area(area).layer_count(1);
        info = match depth {
            None => info.color_attachments(&attachment),
            Some(a) if a.contains(vk::ImageAspectFlags::STENCIL) => info.depth_attachment(&attachment[0]).stencil_attachment(&attachment[0]),
            Some(_) => info.depth_attachment(&attachment[0]),
        };
        // SAFETY: outside rendering in our command buffer; a view of a live
        // image in GENERAL.
        unsafe {
            b.host.device.cmd_begin_rendering(cmd, &info);
            if !rects.is_empty() {
                b.host.device.cmd_clear_attachments(cmd, &[clear], &rects);
            }
            b.host.device.cmd_end_rendering(cmd);
        }
        self.targets_changed = true;
    }
}

/// A D3D12 rectangle cut to `extent`, if anything of it is left.
fn clip_rect(r: &Rect, extent: (u32, u32)) -> Option<vk::Rect2D> {
    let (l, t) = (r.left.max(0), r.top.max(0));
    let (rt, bt) = (r.right.min(extent.0 as i32), r.bottom.min(extent.1 as i32));
    (rt > l && bt > t).then(|| vk::Rect2D {
        offset: vk::Offset2D { x: l, y: t },
        extent: vk::Extent2D { width: (rt - l) as u32, height: (bt - t) as u32 },
    })
}
