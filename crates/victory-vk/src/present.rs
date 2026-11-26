//! Presentation (`docs/vulkan-host.md`, "Presentation"): a swapchain on a
//! Win32 window through winevulkan's `VK_KHR_win32_surface`, frames
//! cleared with dynamic rendering. The backend proper uses the title's
//! window (`victory-gpu` host.rs makes it today); the probe makes its own.

use anyhow::{Context, Result, bail};
use ash::vk;
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DestroyWindow,
    DispatchMessageW, IDC_ARROW, LoadCursorW, MSG, PM_REMOVE, PeekMessageW, RegisterClassExW,
    TranslateMessage, WNDCLASSEXW, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

use crate::host::Host;

/// A window of the probe's own.
pub struct Window {
    pub hwnd: HWND,
    pub hinstance: *mut core::ffi::c_void,
}

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    // SAFETY: forwarding the message as received.
    unsafe { DefWindowProcW(hwnd, msg, w, l) }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

impl Window {
    pub fn new(title: &str, width: i32, height: i32) -> Result<Window> {
        let class = wide("victory-vkprobe");
        let title = wide(title);
        // SAFETY: plain Win32 calls with valid, NUL-terminated strings.
        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let wc = WNDCLASSEXW {
                cbSize: size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(window_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinstance,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                width,
                height,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                bail!("CreateWindowExW failed");
            }
            Ok(Window { hwnd, hinstance })
        }
    }

    /// Handles the window's pending messages.
    pub fn pump(&self) {
        // SAFETY: a standard message loop step on this thread's window.
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // SAFETY: the window is ours and alive.
        unsafe { DestroyWindow(self.hwnd) };
    }
}

/// What a presentation run did.
pub struct Presented {
    pub frames: u32,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub mode: vk::PresentModeKHR,
    pub images: usize,
    pub ms_per_frame: f64,
    pub suboptimal: u32,
}

/// Presents `frames` cleared frames on `window` (one frame in flight).
pub fn clear_frames(host: &Host, window: &Window, frames: u32) -> Result<Presented> {
    let (Some(surface_fn), Some(win32)) = (&host.surface, &host.win32_surface) else {
        bail!("the loader offers no VK_KHR_win32_surface");
    };
    let device = &host.device;
    let queue = host.queues.graphics;
    // SAFETY: every object is made, used and destroyed on this thread in
    // order, waiting for the device before destruction.
    unsafe {
        let surface = win32
            .create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(window.hinstance as isize)
                    .hwnd(window.hwnd as isize),
                None,
            )
            .context("vkCreateWin32SurfaceKHR")?;
        let result = (|| -> Result<Presented> {
            if !surface_fn.get_physical_device_surface_support(
                host.physical,
                queue.family,
                surface,
            )? {
                bail!("the graphics queue cannot present to the window");
            }
            let caps =
                surface_fn.get_physical_device_surface_capabilities(host.physical, surface)?;
            let formats = surface_fn.get_physical_device_surface_formats(host.physical, surface)?;
            let format = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
                .or(formats.first())
                .copied()
                .context("the surface offers no format")?;
            let mode = vk::PresentModeKHR::FIFO;
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: 1280,
                    height: 720,
                }
            };
            let mut count = (caps.min_image_count + 1).max(2);
            if caps.max_image_count != 0 {
                count = count.min(caps.max_image_count);
            }
            let swapchain = host
                .swapchain
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(count)
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(caps.current_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(mode)
                        .clipped(true),
                    None,
                )
                .context("vkCreateSwapchainKHR")?;
            let images = host.swapchain.get_swapchain_images(swapchain)?;
            let views: Vec<vk::ImageView> = images
                .iter()
                .map(|&image| {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(color_range()),
                        None,
                    )
                })
                .collect::<Result<_, _>>()?;
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let acquired = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            // One per image: a present may still wait on the last one.
            let rendered: Vec<vk::Semaphore> = images
                .iter()
                .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<_, _>>()?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            let start = std::time::Instant::now();
            let mut suboptimal = 0;
            let mut shown = 0;
            let mut failure = None;
            for frame in 0..frames {
                window.pump();
                let (index, sub) = match host.swapchain.acquire_next_image(
                    swapchain,
                    1_000_000_000,
                    acquired,
                    vk::Fence::null(),
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        failure = Some(format!("vkAcquireNextImageKHR: {e}"));
                        break;
                    }
                };
                suboptimal += sub as u32;
                let i = index as usize;
                let t = frame as f32 / frames.max(1) as f32;
                record_clear(
                    host,
                    cmd,
                    images[i],
                    views[i],
                    extent,
                    [0.1, 0.2 + 0.6 * t, 0.4, 1.0],
                )?;
                let waits = [vk::SemaphoreSubmitInfo::default()
                    .semaphore(acquired)
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
                let signals = [vk::SemaphoreSubmitInfo::default()
                    .semaphore(rendered[i])
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
                let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
                let submit = vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&waits)
                    .command_buffer_infos(&cmds)
                    .signal_semaphore_infos(&signals);
                device.queue_submit2(queue.queue, &[submit], fence)?;
                let wait = [rendered[i]];
                let swapchains = [swapchain];
                let indices = [index];
                match host.swapchain.queue_present(
                    queue.queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&wait)
                        .swapchains(&swapchains)
                        .image_indices(&indices),
                ) {
                    Ok(sub) => suboptimal += sub as u32,
                    Err(e) => failure = Some(format!("vkQueuePresentKHR: {e}")),
                }
                device.wait_for_fences(&[fence], true, 2_000_000_000)?;
                device.reset_fences(&[fence])?;
                if failure.is_some() {
                    break;
                }
                shown += 1;
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            device.device_wait_idle()?;
            device.destroy_fence(fence, None);
            for s in rendered {
                device.destroy_semaphore(s, None);
            }
            device.destroy_semaphore(acquired, None);
            device.destroy_command_pool(pool, None);
            for v in views {
                device.destroy_image_view(v, None);
            }
            host.swapchain.destroy_swapchain(swapchain, None);
            if let Some(f) = failure {
                bail!("{f} after {shown} frames");
            }
            Ok(Presented {
                frames: shown,
                extent,
                format: format.format,
                mode,
                images: images.len(),
                ms_per_frame: elapsed / f64::from(shown.max(1)),
                suboptimal,
            })
        })();
        surface_fn.destroy_surface(surface, None);
        result
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

/// Records one frame: undefined → attachment, a cleared dynamic-rendering
/// pass, attachment → present.
fn record_clear(
    host: &Host,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    view: vk::ImageView,
    extent: vk::Extent2D,
    color: [f32; 4],
) -> Result<()> {
    let device = &host.device;
    // SAFETY: the command buffer is not in use (the last frame's fence was
    // waited for) and every handle is alive.
    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let to_target = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(color_range())];
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&to_target),
        );
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            })];
        device.cmd_begin_rendering(
            cmd,
            &vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                })
                .layer_count(1)
                .color_attachments(&attachments),
        );
        device.cmd_end_rendering(cmd);
        let to_present = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(color_range())];
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&to_present),
        );
        device.end_command_buffer(cmd)?;
    }
    Ok(())
}
