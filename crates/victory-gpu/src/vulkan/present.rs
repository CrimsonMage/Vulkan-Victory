//! The present: queued work submitted (held waits passed, as the D3D12 path
//! does), the swapchain made on the title's window the first time, the
//! planes composed into the acquired image (`compose`), and at captured
//! frames the image read back and saved (`crate::capture`). Diagnostics
//! builds also read a few texels of each plane at those frames.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};
use victory_vk::swapchain::{Swapchain, Target};

use super::{Backend, Split, backend, compose, engine, images, lock, lost_once, memory, split};

/// Presents that showed at least one plane.
pub(crate) static PLANES_SHOWN: AtomicU64 = AtomicU64::new(0);
/// The last frame presented, for the 10 s line.
pub(crate) static FRAME: AtomicU64 = AtomicU64::new(0);
/// The buffer captures are read back into, and plane texels (diagnostics).
static READBACK: Mutex<Option<Buffer>> = Mutex::new(None);
static PROBE: Mutex<Option<Buffer>> = Mutex::new(None);

/// `VICTORY_VK_COMPOSE_TEST=1` (diagnostics): the swapchain is cleared
/// magenta under the planes, so a capture tells undrawn planes (magenta)
/// from black plane images.
fn compose_test() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = cfg!(feature = "diagnostics") && std::env::var("VICTORY_VK_COMPOSE_TEST").is_ok_and(|v| v == "1");
        if on {
            tracing::info!("Vulkan: the compose test colour (magenta) is cleared under the planes");
        }
        on
    })
}

pub(crate) fn present(frame: u64, size: Option<(u32, u32)>, planes: &[compose::Shown]) {
    let Some(b) = backend() else { return };
    let t = Instant::now();
    images::flush(b);
    split(Split::Flush, t);
    let t = Instant::now();
    engine(b).pump(b, true);
    split(Split::Submit, t);
    images::refresh(b, frame);
    super::stages::new_frame();
    super::follow_budget(b);
    let presented = Instant::now();
    memory(b).tick = frame;
    super::remap::each_frame(b);
    let mut sc = lock(&b.present);
    if sc.is_none() {
        *sc = make_swapchain(b, size);
    }
    if let Some(s) = sc.as_mut() {
        if b.timeline.lost() {
            engine(b).report_lost(b);
            lost_once();
        } else {
            show(b, s, frame, planes);
        }
    }
    drop(sc);
    split(Split::Present, presented);
    FRAME.store(frame, Ordering::Relaxed);
}

/// A swapchain on the title's window (or one of ours, at the planes' size).
fn make_swapchain(b: &Backend, size: Option<(u32, u32)>) -> Option<Swapchain> {
    let (w, h) = size.unwrap_or((1920, 1080));
    let hwnd = crate::host::title_window().map(Ok).unwrap_or_else(|| crate::host::window(w, h));
    // SAFETY: a plain query of this module's instance handle.
    let hinstance = unsafe { windows::Win32::System::LibraryLoader::GetModuleHandleW(None) };
    let (Ok(hwnd), Ok(inst)) = (hwnd, hinstance) else {
        tracing::error!("Vulkan: no window to present to");
        return None;
    };
    // SAFETY: the title's window and our module's instance.
    match unsafe { Swapchain::new(&b.host, inst.0 as isize, hwnd.0 as isize) } {
        Ok(s) => {
            tracing::info!(width = s.extent.width, height = s.extent.height, format = ?s.format, mode = ?s.mode, "Vulkan: presenting to a window");
            Some(s)
        }
        Err(e) => {
            tracing::error!(error = %format!("{e:#}"), "Vulkan: no swapchain");
            None
        }
    }
}

/// Waits until the queue has run everything given.
fn idle(b: &Backend) -> bool {
    let r = b.timeline.with_queue(|q, _| {
        // SAFETY: waits for our queue, under its lock.
        (unsafe { b.host.device.queue_wait_idle(q) }, false)
    });
    matches!(r, Ok(Ok(())))
}

/// The readback buffer for a capture of `bytes` (remade at a new size).
fn readback(b: &Backend, bytes: u64) -> Option<(vk::Buffer, usize, u64)> {
    let mut rb = lock(&READBACK);
    if rb.as_ref().is_none_or(|r| r.size != bytes) {
        if let Some(old) = rb.take() {
            idle(b);
            // SAFETY: the queue is idle.
            unsafe { old.destroy(&b.host) };
        }
        *rb = Buffer::new(&b.host, bytes, vk::BufferUsageFlags::TRANSFER_DST, Place::Readback).ok();
    }
    rb.as_ref().map(|r| (r.buffer, r.mapped as usize, r.size))
}

fn probe_buffer(b: &Backend) -> Option<(vk::Buffer, usize)> {
    let mut pb = lock(&PROBE);
    if pb.is_none() {
        *pb = Buffer::new(&b.host, 256, vk::BufferUsageFlags::TRANSFER_DST, Place::Readback).ok();
    }
    pb.as_ref().map(|p| (p.buffer, p.mapped as usize))
}

/// One frame: cleared, the planes composed, read back if captured.
fn show(b: &'static Backend, s: &mut Swapchain, frame: u64, planes: &[compose::Shown]) {
    let wanted = crate::capture::wanted(frame);
    let capture = wanted.then(|| readback(b, u64::from(s.extent.width) * u64::from(s.extent.height) * 4)).flatten();
    let probes: Vec<Arc<images::Image>> = if cfg!(feature = "diagnostics") && wanted {
        for (n, p) in planes.iter().enumerate() {
            tracing::info!(frame, plane = n, what = %images::describe(b, p.resource), "Vulkan: a plane at a captured frame");
        }
        planes.iter().filter_map(|p| images::of_resource(b, p.resource)).filter(|i| !i.depth()).take(4).collect()
    } else {
        Vec::new()
    };
    let probe = (!probes.is_empty()).then(|| probe_buffer(b)).flatten();
    let (extent, format) = (s.extent, s.format);
    let r = s.frame(&b.host, &b.timeline, |d, cmd, t| {
        // SAFETY: the image is in TRANSFER_DST_OPTIMAL while recording;
        // everything below leaves it so.
        unsafe {
            clear(d, cmd, t);
            if compose::record(b, cmd, t, planes) {
                PLANES_SHOWN.fetch_add(1, Ordering::Relaxed);
            }
            if let Some((pbuf, _)) = probe {
                copy_texels(d, cmd, &probes, pbuf);
            }
            if let Some((buffer, _, _)) = capture {
                copy_target(d, cmd, t, buffer);
            }
        }
    });
    if let Err(e) = r {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 4 {
            tracing::error!(error = %format!("{e:#}"), "Vulkan: a present failed");
        }
        return;
    }
    if let Some((_, mapped, size)) = capture
        && idle(b)
    {
        // SAFETY: the readback buffer is mapped, `size` bytes, and the copy
        // into it has finished.
        let rows = unsafe { std::slice::from_raw_parts(mapped as *const u8, size as usize) };
        let bgra = matches!(format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB);
        crate::capture::save(frame, extent.width, extent.height, extent.width as usize * 4, bgra, rows);
        if let Some((_, mapped)) = probe {
            log_texels(frame, &probes, mapped);
        }
    }
}

/// # Safety
/// Recording into `cmd`, the image in TRANSFER_DST_OPTIMAL.
unsafe fn clear(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, t: &Target) {
    let color = vk::ClearColorValue {
        float32: if compose_test() { [1.0, 0.0, 1.0, 1.0] } else { [0.0, 0.0, 0.0, 1.0] },
    };
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    // SAFETY: as the caller says.
    unsafe {
        d.cmd_clear_color_image(cmd, t.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &color, &[victory_vk::swapchain::color_range()]);
        d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&barrier));
    }
}

/// Four texels of each plane (the centre and three quarter points), 64
/// bytes a plane into `pbuf`.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
unsafe fn copy_texels(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, probes: &[Arc<images::Image>], pbuf: vk::Buffer) {
    for (pn, img) in probes.iter().enumerate() {
        let (w, h) = (img.extent.width, img.extent.height);
        let at = [(w / 2, h / 2), (w / 4, h / 4), (3 * w / 4, h / 4), (w / 4, 3 * h / 4)];
        let regions: Vec<vk::BufferImageCopy> = at
            .iter()
            .enumerate()
            .map(|(k, &(x, y))| {
                vk::BufferImageCopy::default()
                    .buffer_offset((pn * 4 + k) as u64 * 16)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                    .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
            })
            .collect();
        // SAFETY: as the caller says; the plane's image is in GENERAL.
        unsafe { d.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, pbuf, &regions) };
    }
}

fn log_texels(frame: u64, probes: &[Arc<images::Image>], mapped: usize) {
    for (pn, img) in probes.iter().enumerate() {
        // SAFETY: 64 bytes a plane of the mapped probe buffer, copied into.
        let texels = unsafe { std::slice::from_raw_parts((mapped as *const u8).add(pn * 64), 64) };
        let bytes = images::format_bytes(img.format).clamp(1, 16) as usize;
        let hex: Vec<String> = texels.chunks(16).map(|c| c[..bytes].iter().map(|x| format!("{x:02x}")).collect()).collect();
        tracing::info!(frame, plane = pn, image = img.id, format = ?img.format, texels = ?hex, "Vulkan: plane texels (centre, then three quarter points)");
    }
}

/// The swapchain image into `buffer`.
///
/// # Safety
/// Recording into `cmd`, the image in TRANSFER_DST_OPTIMAL (and left so).
unsafe fn copy_target(d: &victory_vk::ash::Device, cmd: vk::CommandBuffer, t: &Target, buffer: vk::Buffer) {
    let range = victory_vk::swapchain::color_range();
    let to = |from: vk::ImageLayout, to: vk::ImageLayout| {
        [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(from)
            .new_layout(to)
            .image(t.image)
            .subresource_range(range)]
    };
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
        .image_extent(vk::Extent3D { width: t.extent.width, height: t.extent.height, depth: 1 });
    // SAFETY: as the caller says.
    unsafe {
        let b1 = to(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b1));
        d.cmd_copy_image_to_buffer(cmd, t.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[region]);
        let b2 = to(vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b2));
    }
}
