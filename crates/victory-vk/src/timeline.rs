//! The one host queue's line of work (`docs/vulkan-host.md`, "Queues and
//! fences"): every submission and every sparse bind goes to the graphics
//! queue behind the one before it and signals the next value of one
//! timeline semaphore. Values say how far the card has got; nothing is
//! waited for by fences.

use std::sync::Mutex;

use anyhow::{Context, Result};
use ash::vk;

use crate::host::Host;

/// A one-shot command buffer and the value after which it may be reused.
struct Recorder {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    /// The value its last submission signals (0: never submitted).
    done_at: u64,
}

struct State {
    /// The value the last operation signals.
    last: u64,
    recorders: Vec<Recorder>,
}

pub struct Timeline {
    pub semaphore: vk::Semaphore,
    queue: vk::Queue,
    family: u32,
    state: Mutex<State>,
}

/// A sparse bind of one buffer: `(buffer, binds)`.
pub type BufferBinds = (vk::Buffer, Vec<vk::SparseMemoryBind>);

impl Timeline {
    pub fn new(host: &Host) -> Result<Timeline> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        // SAFETY: valid create info; destroyed by `destroy`.
        let semaphore = unsafe {
            host.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .context("timeline semaphore")?;
        Ok(Timeline {
            semaphore,
            queue: host.queues.graphics.queue,
            family: host.queues.graphics.family,
            state: Mutex::new(State {
                last: 0,
                recorders: Vec::new(),
            }),
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The value the last operation given to the queue signals.
    pub fn last(&self) -> u64 {
        self.state().last
    }

    /// How far the card has got.
    pub fn completed(&self, host: &Host) -> u64 {
        // SAFETY: a live timeline semaphore.
        unsafe { host.device.get_semaphore_counter_value(self.semaphore) }.unwrap_or(0)
    }

    /// Waits until the card has reached `value`, at most `timeout_ns`.
    pub fn wait(&self, host: &Host, value: u64, timeout_ns: u64) -> bool {
        let sems = [self.semaphore];
        let values = [value];
        // SAFETY: a live timeline semaphore.
        unsafe {
            host.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&sems)
                    .values(&values),
                timeout_ns,
            )
        }
        .is_ok()
    }

    /// Submits `cmds` (none: a marker that only signals) behind everything
    /// given before; returns the value it signals.
    pub fn submit(&self, host: &Host, cmds: &[vk::CommandBuffer]) -> Result<u64> {
        let mut s = self.state();
        unsafe { self.submit_locked(host, &mut s, cmds) }
    }

    unsafe fn submit_locked(
        &self,
        host: &Host,
        s: &mut State,
        cmds: &[vk::CommandBuffer],
    ) -> Result<u64> {
        let value = s.last + 1;
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(s.last)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let infos: Vec<vk::CommandBufferSubmitInfo> = cmds
            .iter()
            .map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c))
            .collect();
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&infos)
            .signal_semaphore_infos(&signals);
        // SAFETY: the queue is used under the state's lock only.
        unsafe {
            host.device
                .queue_submit2(self.queue, &[submit], vk::Fence::null())
        }
        .context("vkQueueSubmit2")?;
        s.last = value;
        Ok(value)
    }

    /// Records a one-shot command buffer with `f` and submits it; returns
    /// the value it signals.
    pub fn record(
        &self,
        host: &Host,
        f: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<u64> {
        let completed = self.completed(host);
        let mut s = self.state();
        let device = &host.device;
        let at = s.recorders.iter().position(|r| r.done_at <= completed);
        let i = match at {
            Some(i) => i,
            None => {
                // SAFETY: valid create and allocate infos.
                let (pool, cmd) = unsafe {
                    let pool = device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(self.family)
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                        None,
                    )?;
                    let cmd = device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )?[0];
                    (pool, cmd)
                };
                s.recorders.push(Recorder {
                    pool,
                    cmd,
                    done_at: 0,
                });
                s.recorders.len() - 1
            }
        };
        let (pool, cmd) = (s.recorders[i].pool, s.recorders[i].cmd);
        // SAFETY: the recorder's last submission has completed.
        unsafe {
            device.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            f(device, cmd);
            device.end_command_buffer(cmd)?;
            let value = self.submit_locked(host, &mut s, &[cmd])?;
            s.recorders[i].done_at = value;
            Ok(value)
        }
    }

    /// Binds sparse memory behind everything given before; returns the
    /// value it signals.
    pub fn bind_sparse(&self, host: &Host, binds: &[BufferBinds]) -> Result<u64> {
        let mut s = self.state();
        let value = s.last + 1;
        let infos: Vec<vk::SparseBufferMemoryBindInfo> = binds
            .iter()
            .map(|(buffer, b)| {
                vk::SparseBufferMemoryBindInfo::default()
                    .buffer(*buffer)
                    .binds(b)
            })
            .collect();
        let wait_values = [s.last];
        let signal_values = [value];
        let mut values = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let sems = [self.semaphore];
        let info = vk::BindSparseInfo::default()
            .wait_semaphores(&sems)
            .signal_semaphores(&sems)
            .buffer_binds(&infos)
            .push_next(&mut values);
        // SAFETY: the queue is used under the state's lock only; the binds
        // name live buffers and memory.
        unsafe {
            host.device
                .queue_bind_sparse(self.queue, &[info], vk::Fence::null())
        }
        .context("vkQueueBindSparse")?;
        s.last = value;
        Ok(value)
    }

    /// # Safety
    /// The device is idle.
    pub unsafe fn destroy(&self, host: &Host) {
        let s = self.state();
        unsafe {
            for r in &s.recorders {
                host.device.destroy_command_pool(r.pool, None);
            }
            host.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
