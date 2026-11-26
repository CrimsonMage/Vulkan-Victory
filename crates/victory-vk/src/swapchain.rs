//! A swapchain on a Win32 window (winevulkan's `VK_KHR_win32_surface`):
//! acquire, record into the image, present; made again when the window's
//! size changes. Mailbox when offered, else immediate, else FIFO: the
//! title paces its frames itself (`victory-gpu` frame.rs), as the D3D12
//! path's `Present(0)` lets it.

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::host::Host;
use crate::timeline::Timeline;

pub struct Swapchain {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub mode: vk::PresentModeKHR,
    acquired: Vec<vk::Semaphore>,
    rendered: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,
    pools: Vec<(vk::CommandPool, vk::CommandBuffer)>,
    frame: usize,
}

/// What a frame records into: the image, its view and size. The image is
/// in `TRANSFER_DST_OPTIMAL` when the closure runs and must be left so.
pub struct Target {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

const IN_FLIGHT: usize = 2;

impl Swapchain {
    /// # Safety
    /// `hinstance` and `hwnd` name a live window of this process.
    pub unsafe fn new(host: &Host, hinstance: isize, hwnd: isize) -> Result<Swapchain> {
        let win32 = host
            .win32_surface
            .as_ref()
            .context("the loader offers no VK_KHR_win32_surface")?;
        let surface = unsafe {
            win32.create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(hinstance)
                    .hwnd(hwnd),
                None,
            )
        }
        .context("vkCreateWin32SurfaceKHR")?;
        let d = &host.device;
        let mut s = Swapchain {
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            extent: vk::Extent2D::default(),
            format: vk::Format::UNDEFINED,
            mode: vk::PresentModeKHR::FIFO,
            acquired: Vec::new(),
            rendered: Vec::new(),
            fences: Vec::new(),
            pools: Vec::new(),
            frame: 0,
        };
        unsafe {
            for _ in 0..IN_FLIGHT {
                s.acquired
                    .push(d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
                s.fences.push(d.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
                let pool = d.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(host.queues.graphics.family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )?;
                let cmd = d.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0];
                s.pools.push((pool, cmd));
            }
            s.make(host)?;
        }
        Ok(s)
    }

    unsafe fn make(&mut self, host: &Host) -> Result<()> {
        let surface_fn = host.surface.as_ref().context("no surface functions")?;
        let d = &host.device;
        unsafe {
            if !surface_fn.get_physical_device_surface_support(
                host.physical,
                host.queues.graphics.family,
                self.surface,
            )? {
                bail!("the graphics queue cannot present to the window");
            }
            let caps =
                surface_fn.get_physical_device_surface_capabilities(host.physical, self.surface)?;
            let formats =
                surface_fn.get_physical_device_surface_formats(host.physical, self.surface)?;
            let modes = surface_fn
                .get_physical_device_surface_present_modes(host.physical, self.surface)?;
            let format = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
                .or(formats.first())
                .copied()
                .context("the surface offers no format")?;
            let mode = [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
                .into_iter()
                .find(|m| modes.contains(m))
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: 1920,
                    height: 1080,
                }
            };
            let mut count = (caps.min_image_count + 1).max(3);
            if caps.max_image_count != 0 {
                count = count.min(caps.max_image_count);
            }
            let old = self.swapchain;
            let swapchain = host
                .swapchain
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(self.surface)
                        .min_image_count(count)
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::TRANSFER_SRC,
                        )
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(caps.current_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(mode)
                        .clipped(true)
                        .old_swapchain(old),
                    None,
                )
                .context("vkCreateSwapchainKHR")?;
            self.drop_images(host);
            if old != vk::SwapchainKHR::null() {
                host.swapchain.destroy_swapchain(old, None);
            }
            self.swapchain = swapchain;
            self.images = host.swapchain.get_swapchain_images(swapchain)?;
            for &image in &self.images {
                self.views.push(
                    d.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(color_range()),
                        None,
                    )?,
                );
                self.rendered
                    .push(d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            }
            self.extent = extent;
            self.format = format.format;
            self.mode = mode;
        }
        Ok(())
    }

    unsafe fn drop_images(&mut self, host: &Host) {
        let d = &host.device;
        unsafe {
            // The frames in flight (not the whole device: waiting for it
            // idle would touch every queue without their lock).
            if !self.fences.is_empty() {
                let _ = d.wait_for_fences(&self.fences, true, 2_000_000_000);
            }
            for v in self.views.drain(..) {
                d.destroy_image_view(v, None);
            }
            for s in self.rendered.drain(..) {
                d.destroy_semaphore(s, None);
            }
        }
        self.images.clear();
    }

    /// One frame: acquires an image, runs `record` with the image in
    /// `TRANSFER_DST_OPTIMAL`, presents. `wait` is a timeline value of
    /// `timeline` the frame's commands wait for (what the planes show).
    pub fn frame(
        &mut self,
        host: &Host,
        timeline: &Timeline,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer, &Target),
    ) -> Result<()> {
        let d = &host.device;
        let f = self.frame % IN_FLIGHT;
        self.frame += 1;
        // SAFETY: objects of ours; the frame's fence guards its command
        // buffer and semaphore.
        unsafe {
            d.wait_for_fences(&[self.fences[f]], true, 2_000_000_000)?;
            let (index, _) = match host.swapchain.acquire_next_image(
                self.swapchain,
                1_000_000_000,
                self.acquired[f],
                vk::Fence::null(),
            ) {
                Ok(r) => r,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                    self.make(host)?;
                    return Ok(());
                }
                Err(e) => bail!("vkAcquireNextImageKHR: {e}"),
            };
            d.reset_fences(&[self.fences[f]])?;
            let i = index as usize;
            let (_, cmd) = self.pools[f];
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let image = self.images[i];
            barrier(
                d,
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            let target = Target {
                image,
                view: self.views[i],
                extent: self.extent,
                format: self.format,
            };
            record(d, cmd, &target);
            barrier(
                d,
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            d.end_command_buffer(cmd)?;
            let signals = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.rendered[i])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let fence = self.fences[f];
            let acquired = self.acquired[f];
            let rendered = [self.rendered[i]];
            let swapchains = [self.swapchain];
            let indices = [index];
            // Behind everything the queue was given, under its lock.
            let presented = timeline.with_queue(|queue, last| {
                let waits = [
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(acquired)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(timeline.semaphore)
                        .value(last)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                ];
                let submitted = timeline.check(
                    d.queue_submit2(
                        queue,
                        &[vk::SubmitInfo2::default()
                            .wait_semaphore_infos(&waits)
                            .command_buffer_infos(&cmds)
                            .signal_semaphore_infos(&signals)],
                        fence,
                    ),
                );
                if let Err(e) = submitted {
                    return (Err(e), false);
                }
                let r = timeline.check(
                    host.swapchain.queue_present(
                        queue,
                        &vk::PresentInfoKHR::default()
                            .wait_semaphores(&rendered)
                            .swapchains(&swapchains)
                            .image_indices(&indices),
                    ),
                );
                (r, false)
            })?;
            match presented {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.make(host)?,
                Err(e) => bail!("vkQueueSubmit2 or vkQueuePresentKHR: {e}"),
            }
        }
        Ok(())
    }

    /// # Safety
    /// Nothing of the swapchain is in use.
    pub unsafe fn destroy(&mut self, host: &Host) {
        let d = &host.device;
        unsafe {
            self.drop_images(host);
            for s in self.acquired.drain(..) {
                d.destroy_semaphore(s, None);
            }
            for f in self.fences.drain(..) {
                d.destroy_fence(f, None);
            }
            for (pool, _) in self.pools.drain(..) {
                d.destroy_command_pool(pool, None);
            }
            host.swapchain.destroy_swapchain(self.swapchain, None);
            if let Some(s) = &host.surface {
                s.destroy_surface(self.surface, None);
            }
        }
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

/// A whole-image layout change ordering everything before against
/// everything after (frames are few; precision does not matter here).
pub fn barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .image(image)
        .subresource_range(color_range())];
    // SAFETY: `cmd` is recording.
    unsafe {
        d.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&b),
        )
    };
}
