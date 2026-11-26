//! The dynamic state of a draw, from the title's pipeline state and the
//! list's state.

use victory_vk::ash::vk;
use windows::Win32::Graphics::Direct3D12::*;

use super::super::Backend;
use super::{Rec, draw_test};
use crate::pipeline::host::GraphicsState;

/// # Safety
/// Recording into `cmd`.
unsafe fn raster_state(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, r: &D3D12_RASTERIZER_DESC) {
    let cull = match r.CullMode.0 {
        2 => vk::CullModeFlags::FRONT,
        3 => vk::CullModeFlags::BACK,
        _ => vk::CullModeFlags::NONE,
    };
    let front = if r.FrontCounterClockwise.as_bool() { vk::FrontFace::COUNTER_CLOCKWISE } else { vk::FrontFace::CLOCKWISE };
    // SAFETY: as the caller says.
    unsafe {
        d.cmd_set_cull_mode(cmd, cull);
        d.cmd_set_front_face(cmd, front);
        d.cmd_set_depth_bias_enable(cmd, r.DepthBias != 0 || r.SlopeScaledDepthBias != 0.0);
        d.cmd_set_depth_bias(cmd, r.DepthBias as f32, r.DepthBiasClamp, r.SlopeScaledDepthBias);
        d.cmd_set_rasterizer_discard_enable(cmd, false);
    }
}

/// # Safety
/// Recording into `cmd`.
unsafe fn extended_raster(e: &victory_vk::ash::ext::extended_dynamic_state3::Device, cmd: vk::CommandBuffer, gs: &GraphicsState) {
    let r = &gs.rasterizer;
    // SAFETY: as the caller says.
    unsafe {
        e.cmd_set_polygon_mode(cmd, if r.FillMode.0 == 2 { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL });
        e.cmd_set_depth_clamp_enable(cmd, !r.DepthClipEnable.as_bool());
        e.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[gs.sample_mask]);
        e.cmd_set_alpha_to_coverage_enable(cmd, gs.blend.AlphaToCoverageEnable.as_bool());
    }
}

/// # Safety
/// Recording into `cmd`.
unsafe fn blend_state(e: &victory_vk::ash::ext::extended_dynamic_state3::Device, cmd: vk::CommandBuffer, gs: &GraphicsState, count: usize) {
    let independent = gs.blend.IndependentBlendEnable.as_bool();
    let t = |i: usize| &gs.blend.RenderTarget[if independent { i } else { 0 }];
    let enables: Vec<vk::Bool32> = (0..count).map(|i| u32::from(t(i).BlendEnable.as_bool())).collect();
    let equations: Vec<vk::ColorBlendEquationEXT> = (0..count)
        .map(|i| {
            let x = t(i);
            vk::ColorBlendEquationEXT {
                src_color_blend_factor: blend_factor(x.SrcBlend),
                dst_color_blend_factor: blend_factor(x.DestBlend),
                color_blend_op: vk::BlendOp::from_raw((x.BlendOp.0 - 1).clamp(0, 4)),
                src_alpha_blend_factor: blend_factor(x.SrcBlendAlpha),
                dst_alpha_blend_factor: blend_factor(x.DestBlendAlpha),
                alpha_blend_op: vk::BlendOp::from_raw((x.BlendOpAlpha.0 - 1).clamp(0, 4)),
            }
        })
        .collect();
    let masks: Vec<vk::ColorComponentFlags> = (0..count)
        .map(|i| vk::ColorComponentFlags::from_raw(u32::from(t(i).RenderTargetWriteMask) & 0xf))
        .collect();
    // SAFETY: as the caller says.
    unsafe {
        e.cmd_set_color_blend_enable(cmd, 0, &enables);
        e.cmd_set_color_blend_equation(cmd, 0, &equations);
        e.cmd_set_color_write_mask(cmd, 0, &masks);
    }
}

/// A D3D viewport as Vulkan's, flipped (negative height): the console's
/// programs put +y up, as D3D does.
pub(super) fn d3d_flip(v: &D3D12_VIEWPORT) -> vk::Viewport {
    let (w, h) = (v.Width.max(1.0), v.Height.max(1.0));
    vk::Viewport {
        x: v.TopLeftX,
        y: v.TopLeftY + h,
        width: w,
        height: -h,
        min_depth: v.MinDepth.clamp(0.0, 1.0),
        max_depth: v.MaxDepth.clamp(0.0, 1.0),
    }
}

fn blend_factor(f: D3D12_BLEND) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match f.0 {
        1 => B::ZERO,
        2 => B::ONE,
        3 => B::SRC_COLOR,
        4 => B::ONE_MINUS_SRC_COLOR,
        5 => B::SRC_ALPHA,
        6 => B::ONE_MINUS_SRC_ALPHA,
        7 => B::DST_ALPHA,
        8 => B::ONE_MINUS_DST_ALPHA,
        9 => B::DST_COLOR,
        10 => B::ONE_MINUS_DST_COLOR,
        11 => B::SRC_ALPHA_SATURATE,
        14 => B::CONSTANT_COLOR,
        15 => B::ONE_MINUS_CONSTANT_COLOR,
        16 => B::SRC1_COLOR,
        17 => B::ONE_MINUS_SRC1_COLOR,
        18 => B::SRC1_ALPHA,
        19 => B::ONE_MINUS_SRC1_ALPHA,
        20 => B::CONSTANT_ALPHA,
        21 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

impl Rec {
    /// A digest of everything [`dynamic`](Self::dynamic) sets from: the
    /// state is set again only when it changes within a command buffer
    /// (binding a pipeline keeps dynamic state).
    pub(super) fn state_key(&self, gs: &GraphicsState, topology: vk::PrimitiveTopology, count: u32) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (std::ptr::from_ref(gs) as usize, topology.as_raw(), count, self.stencil_ref).hash(&mut h);
        for v in &self.viewports {
            [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth].map(f32::to_bits).hash(&mut h);
        }
        for r in &self.scissors {
            (r.offset.x, r.offset.y, r.extent.width, r.extent.height).hash(&mut h);
        }
        (self.blend.map(f32::to_bits), self.depth_bounds.0.to_bits(), self.depth_bounds.1.to_bits()).hash(&mut h);
        h.finish()
    }

    /// Sets every dynamic state the parts declare.
    ///
    /// # Safety
    /// Recording into `cmd`.
    pub(super) unsafe fn dynamic(&self, b: &'static Backend, cmd: vk::CommandBuffer, gs: &GraphicsState, topology: vk::PrimitiveTopology, count: u32) {
        let d = &b.host.device;
        // SAFETY: as the caller says.
        unsafe {
            self.viewport_state(d, cmd);
            raster_state(d, cmd, &gs.rasterizer);
            self.depth_stencil_state(d, cmd, gs);
            d.cmd_set_primitive_topology(cmd, topology);
            let strip = matches!(topology, vk::PrimitiveTopology::LINE_STRIP | vk::PrimitiveTopology::TRIANGLE_STRIP);
            d.cmd_set_primitive_restart_enable(cmd, strip && gs.strip_cut.0 != 0);
            d.cmd_set_blend_constants(cmd, &self.blend);
            if let Some(e) = super::super::eds3(b) {
                if b.host.caps.dynamic_raster {
                    extended_raster(e, cmd, gs);
                }
                if b.host.caps.dynamic_blend && count > 0 {
                    blend_state(e, cmd, gs, count as usize);
                }
            }
            if draw_test() >= 1 {
                self.test_state(b, cmd, count);
            }
        }
    }

    /// # Safety
    /// Recording into `cmd`.
    unsafe fn viewport_state(&self, d: &victory_vk::ash::Device, cmd: vk::CommandBuffer) {
        let viewports = if self.viewports.is_empty() {
            vec![vk::Viewport { x: 0.0, y: 1.0, width: 1.0, height: -1.0, min_depth: 0.0, max_depth: 1.0 }]
        } else {
            self.viewports.clone()
        };
        let scissors = if self.scissors.is_empty() {
            vec![vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: 16384, height: 16384 } }]
        } else {
            self.scissors.clone()
        };
        let n = viewports.len().min(scissors.len()).max(1);
        // SAFETY: as the caller says.
        unsafe {
            d.cmd_set_viewport_with_count(cmd, &viewports[..n]);
            d.cmd_set_scissor_with_count(cmd, &scissors[..n]);
        }
    }

    /// # Safety
    /// Recording into `cmd`.
    unsafe fn depth_stencil_state(&self, d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, gs: &GraphicsState) {
        let z = &gs.depth;
        let stencil_op = |o: &D3D12_DEPTH_STENCILOP_DESC| {
            (
                vk::StencilOp::from_raw((o.StencilFailOp.0 - 1).max(0)),
                vk::StencilOp::from_raw((o.StencilPassOp.0 - 1).max(0)),
                vk::StencilOp::from_raw((o.StencilDepthFailOp.0 - 1).max(0)),
                vk::CompareOp::from_raw((o.StencilFunc.0 - 1).max(0)),
            )
        };
        // SAFETY: as the caller says.
        unsafe {
            d.cmd_set_depth_test_enable(cmd, z.DepthEnable.as_bool());
            d.cmd_set_depth_write_enable(cmd, z.DepthEnable.as_bool() && z.DepthWriteMask.0 == 1);
            d.cmd_set_depth_compare_op(cmd, vk::CompareOp::from_raw((z.DepthFunc.0 - 1).clamp(0, 7)));
            d.cmd_set_depth_bounds_test_enable(cmd, gs.depth_bounds);
            d.cmd_set_depth_bounds(cmd, self.depth_bounds.0.clamp(0.0, 1.0), self.depth_bounds.1.clamp(0.0, 1.0));
            d.cmd_set_stencil_test_enable(cmd, z.StencilEnable.as_bool());
            for (face, o) in [(vk::StencilFaceFlags::FRONT, &z.FrontFace), (vk::StencilFaceFlags::BACK, &z.BackFace)] {
                let (fail, pass, depth_fail, compare) = stencil_op(o);
                d.cmd_set_stencil_op(cmd, face, fail, pass, depth_fail, compare);
            }
            d.cmd_set_stencil_compare_mask(cmd, vk::StencilFaceFlags::FRONT_AND_BACK, u32::from(z.StencilReadMask));
            d.cmd_set_stencil_write_mask(cmd, vk::StencilFaceFlags::FRONT_AND_BACK, u32::from(z.StencilWriteMask));
            d.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::FRONT_AND_BACK, self.stencil_ref);
        }
    }

    /// Draw test state: nothing hidden, every channel written.
    ///
    /// # Safety
    /// Recording into `cmd`.
    pub(super) unsafe fn test_state(&self, b: &'static Backend, cmd: vk::CommandBuffer, count: u32) {
        let d = &b.host.device;
        // SAFETY: as the caller says.
        unsafe {
            d.cmd_set_cull_mode(cmd, vk::CullModeFlags::NONE);
            d.cmd_set_depth_test_enable(cmd, false);
            d.cmd_set_depth_write_enable(cmd, false);
            d.cmd_set_depth_bounds_test_enable(cmd, false);
            d.cmd_set_stencil_test_enable(cmd, false);
            if let Some(e) = super::super::eds3(b)
                && b.host.caps.dynamic_raster
            {
                e.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[u32::MAX]);
                e.cmd_set_alpha_to_coverage_enable(cmd, false);
                e.cmd_set_polygon_mode(cmd, vk::PolygonMode::FILL);
                e.cmd_set_depth_clamp_enable(cmd, false);
            }
            if let Some(e) = super::super::eds3(b)
                && b.host.caps.dynamic_blend
                && count > 0
            {
                let n = count as usize;
                e.cmd_set_color_blend_enable(cmd, 0, &vec![vk::FALSE; n]);
                e.cmd_set_color_write_mask(cmd, 0, &vec![vk::ColorComponentFlags::RGBA; n]);
            }
        }
    }

}
