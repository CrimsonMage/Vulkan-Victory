//! The host's lines of work (`docs/vulkan-host.md`, "Queues and fences").
//!
//! A **line** is a timeline semaphore on one queue of the graphics family.
//! Each operation given to a line signals its next value.
//!
//! - **The host line** ([`HOST`], the first queue) takes our own work:
//!   fills, sparse binds and one-shot copies. Each of its operations waits
//!   for everything given before on every line, so the host's work keeps
//!   the order it always had.
//! - **A title line** ([`Timeline::line`]) belongs to one title queue.
//!   Its submissions wait for the line's previous value, the host line's
//!   last value, and the waits the caller names (title fences signalled on
//!   other lines). Title queues of different kinds run on different queues
//!   ([`crate::host::LANES`]), so their work overlaps on the card.
//!
//! Every operation also takes a **ticket**, one count over all lines.
//! [`Timeline::last`], [`Timeline::completed`] and [`Timeline::wait`] speak
//! in tickets: "everything given until then". Release, refill and recorder
//! reuse go by them, as they went by the one timeline before.
//!
//! Every `vkQueue*` call is made under this module's lock: Vulkan requires
//! a queue's calls to be externally synchronised.
//!
//! The command buffers of recordings carry a GPU timestamp at their start
//! and their end. A line's busy time is the sum over its submissions, read
//! once they complete ([`Timeline::take_stats`]).

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use ash::vk;

use crate::host::Host;
use crate::memory::{Buffer, Place};

/// A line: an index into the timeline's lines.
pub type Line = usize;
/// The host's own line.
pub const HOST: Line = 0;

/// What kind of title queue a line serves (the queue it runs on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Direct,
    Compute,
    Copy,
}

/// Timestamp pairs for the busy figure (two queries each).
const PAIRS: u32 = 8192;
/// A scratch buffer's least size.
const SCRATCH: u64 = 256 << 10;
/// Scratch buffers kept for the next recordings at most; what comes back
/// past it is freed.
const SPARE: u64 = 32 << 20;

/// A command buffer's scratch: card memory its commands write and read
/// (bounded indirect records, the store views a dispatch has staged).
/// The buffers are the timeline's, lent to one recording at a time and
/// handed back once its command buffer has run: the card holds what the
/// recordings in flight use, not the most each command buffer ever used
/// (vk103: 478 MiB and growing).
#[derive(Default)]
struct Scratch {
    /// The last one is being filled.
    buffers: Vec<Buffer>,
    /// Bytes used in it.
    used: u64,
}

/// A part of a recording's scratch.
#[derive(Clone, Copy, Debug)]
pub struct ScratchRange {
    pub buffer: vk::Buffer,
    pub offset: u64,
    /// Its device address.
    pub address: u64,
}

impl Scratch {
    fn take(&mut self, host: &Host, timeline: &Timeline, bytes: u64) -> Option<ScratchRange> {
        let offset = self.used.next_multiple_of(16);
        if let Some(b) = self.buffers.last().filter(|b| offset + bytes <= b.size) {
            self.used = offset + bytes;
            return Some(ScratchRange {
                buffer: b.buffer,
                offset,
                address: b.address + offset,
            });
        }
        let b = timeline.lend(host, bytes)?;
        let range = ScratchRange {
            buffer: b.buffer,
            offset: 0,
            address: b.address,
        };
        self.buffers.push(b);
        self.used = bytes;
        Some(range)
    }
}

/// A one-shot command buffer and the ticket after which it may be reused.
struct Recorder {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    /// The ticket its last submission took (0: never submitted;
    /// `u64::MAX`: being recorded).
    done_at: u64,
    scratch: Scratch,
    /// The serial of its current recording ([`Mark`]).
    serial: u64,
}

/// An operation given to a line and not yet seen complete.
struct Pending {
    value: u64,
    ticket: u64,
    /// Its timestamp queries (start, end), if timed.
    timing: Option<(u32, u32)>,
}

struct LineState {
    semaphore: vk::Semaphore,
    queue: vk::Queue,
    label: String,
    /// The value the last operation signals.
    last: u64,
    pending: VecDeque<Pending>,
    /// Since the last [`Timeline::take_stats`].
    submissions: u64,
    busy_ns: f64,
}

struct State {
    /// The last ticket given.
    ticket: u64,
    lines: Vec<LineState>,
    recorders: Vec<Recorder>,
    /// Scratch buffers no recording holds.
    spare: Vec<Buffer>,
    next_pair: u32,
    /// The serial the next recording begins with ([`Mark`]).
    next_serial: u64,
}

/// A point in the recordings ([`Timeline::mark`]): something that left use
/// there may still be named by a command buffer begun before it and not
/// yet submitted; it is safe to free once every such one is submitted (or
/// abandoned) and the card has run everything given by then
/// ([`Timeline::passed`]).
pub struct Mark {
    serial: u64,
    ticket: Option<u64>,
    since: std::time::Instant,
}

/// A command buffer being recorded ([`Timeline::begin`]).
pub struct Recording {
    index: usize,
    pub cmd: vk::CommandBuffer,
    /// Its start timestamp's query (the end's is the next).
    pair: Option<u32>,
    scratch: Scratch,
    /// Diagnostics: what was recorded into it, for a hang's report.
    pub trail: Trail,
}

/// Diagnostics: the last [`Trail::MOST`] notes of a command buffer (what
/// it draws and dispatches) and how many went before them.
#[derive(Default)]
pub struct Trail {
    pub notes: VecDeque<String>,
    /// Beside each note: the descriptor table addresses its draw or
    /// dispatch was given (empty for others).
    pub tables: VecDeque<Vec<u32>>,
    pub earlier: u64,
}

impl Trail {
    pub const MOST: usize = 64;

    pub fn push(&mut self, note: String) {
        self.push_with(note, Vec::new());
    }

    pub fn push_with(&mut self, note: String, tables: Vec<u32>) {
        if self.notes.len() == Self::MOST {
            self.notes.pop_front();
            self.tables.pop_front();
            self.earlier += 1;
        }
        self.notes.push_back(note);
        self.tables.push_back(tables);
    }
}

impl Recording {
    /// `bytes` of the command buffer's scratch, 16-byte aligned.
    pub fn scratch(&mut self, host: &Host, timeline: &Timeline, bytes: u64) -> Option<ScratchRange> {
        self.scratch.take(host, timeline, bytes)
    }
}

/// A line's figures since the last [`Timeline::take_stats`].
pub struct LineStats {
    pub label: String,
    pub submissions: u64,
    /// GPU time its submissions took, start to end.
    pub busy_ms: f64,
}

pub struct Timeline {
    queues: Vec<vk::Queue>,
    family: u32,
    timing: Option<vk::QueryPool>,
    period_ns: f64,
    state: Mutex<State>,
    /// The device was lost: nothing is given to a queue any more.
    lost: AtomicBool,
}

/// A sparse bind of one buffer: `(buffer, binds)`.
pub type BufferBinds = (vk::Buffer, Vec<vk::SparseMemoryBind>);

fn semaphore(host: &Host) -> Result<vk::Semaphore> {
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    // SAFETY: valid create info; destroyed by `destroy`.
    unsafe {
        host.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .context("timeline semaphore")
}

fn line_state(semaphore: vk::Semaphore, queue: vk::Queue, label: String) -> LineState {
    LineState {
        semaphore,
        queue,
        label,
        last: 0,
        pending: VecDeque::new(),
        submissions: 0,
        busy_ns: 0.0,
    }
}

impl Timeline {
    pub fn new(host: &Host) -> Result<Timeline> {
        let host_line = line_state(semaphore(host)?, host.queues.graphics.queue, "host".into());
        let period_ns = f64::from(host.caps.timestamp_period);
        // SAFETY: valid create info; destroyed by `destroy`.
        let timing = (period_ns > 0.0)
            .then(|| unsafe {
                host.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2 * PAIRS),
                    None,
                )
            })
            .and_then(|r| r.ok());
        let mut queues = host.queues.lanes.clone();
        if queues.is_empty() {
            queues.push(host.queues.graphics.queue);
        }
        Ok(Timeline {
            queues,
            family: host.queues.graphics.family,
            timing,
            period_ns,
            state: Mutex::new(State {
                ticket: 0,
                lines: vec![host_line],
                recorders: Vec::new(),
                spare: Vec::new(),
                next_pair: 0,
                next_serial: 0,
            }),
            lost: AtomicBool::new(false),
        })
    }

    /// A new title line for a queue of `kind`.
    pub fn line(&self, host: &Host, kind: Kind) -> Result<Line> {
        let lane = match kind {
            Kind::Direct => 0,
            Kind::Compute => 1,
            Kind::Copy => 2,
        }
        .min(self.queues.len() - 1);
        let sem = semaphore(host)?;
        let mut s = self.state();
        let n = s
            .lines
            .iter()
            .filter(|l| l.label.starts_with(kind_name(kind)))
            .count();
        let label = format!("{}{n}", kind_name(kind));
        s.lines.push(line_state(sem, self.queues[lane], label));
        Ok(s.lines.len() - 1)
    }

    /// Whether the device was lost (every later operation fails at once).
    pub fn lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }

    /// Notes a queue call's result: a lost device is remembered.
    pub fn check<T>(&self, r: ash::prelude::VkResult<T>) -> ash::prelude::VkResult<T> {
        if let Err(vk::Result::ERROR_DEVICE_LOST) = r {
            self.lost.store(true, Ordering::Relaxed);
        }
        r
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` with the host queue and the host line's last value, under
    /// the lock every queue call takes; `f` returns whether it gave the
    /// queue an operation signalling the host line's `last + 1`.
    pub fn with_queue<R>(&self, f: impl FnOnce(vk::Queue, u64) -> (R, bool)) -> Result<R> {
        anyhow::ensure!(!self.lost(), "the Vulkan device was lost");
        let mut s = self.state();
        let (r, signalled) = f(self.queues[0], s.lines[HOST].last);
        if signalled {
            let value = s.lines[HOST].last + 1;
            commit(&mut s, HOST, value, None);
        }
        Ok(r)
    }

    /// Runs `f` with the host queue and waits for everything given on every
    /// line (a present), under the lock every queue call takes.
    pub fn with_queue_joined<R>(
        &self,
        f: impl FnOnce(vk::Queue, &[vk::SemaphoreSubmitInfo<'_>]) -> R,
    ) -> Result<R> {
        anyhow::ensure!(!self.lost(), "the Vulkan device was lost");
        let s = self.state();
        let waits = submit_infos(&waits(&s, HOST, &[]));
        Ok(f(self.queues[0], &waits))
    }

    /// The last ticket given.
    pub fn last(&self) -> u64 {
        self.state().ticket
    }

    /// The ticket up to which everything has run on the card.
    pub fn completed(&self, host: &Host) -> u64 {
        let mut s = self.state();
        self.completed_locked(host, &mut s)
    }

    fn completed_locked(&self, host: &Host, s: &mut State) -> u64 {
        let lost = self.lost();
        let mut done = s.ticket;
        for l in &mut s.lines {
            let reached = if lost {
                u64::MAX
            } else {
                // SAFETY: a live timeline semaphore.
                unsafe { host.device.get_semaphore_counter_value(l.semaphore) }.unwrap_or(0)
            };
            while l.pending.front().is_some_and(|p| p.value <= reached) {
                let p = l.pending.pop_front().expect("pending");
                l.busy_ns += p.timing.map_or(0.0, |t| self.elapsed_ns(host, t));
            }
            if let Some(p) = l.pending.front() {
                done = done.min(p.ticket - 1);
            }
        }
        done
    }

    /// Nanoseconds between two timestamps of a run submission.
    fn elapsed_ns(&self, host: &Host, (start, end): (u32, u32)) -> f64 {
        let Some(pool) = self.timing else { return 0.0 };
        let mut a = [0u64; 1];
        let mut b = [0u64; 1];
        // SAFETY: slots of our pool, written by a submission that has run.
        let read = unsafe {
            host.device
                .get_query_pool_results(pool, start, &mut a, vk::QueryResultFlags::TYPE_64)
                .is_ok()
                && host
                    .device
                    .get_query_pool_results(pool, end, &mut b, vk::QueryResultFlags::TYPE_64)
                    .is_ok()
        };
        if read && b[0] >= a[0] {
            (b[0] - a[0]) as f64 * self.period_ns
        } else {
            0.0
        }
    }

    /// The last value given on `line` and its label.
    pub fn line_given(&self, line: Line) -> (u64, String) {
        let s = self.state();
        (s.lines[line].last, s.lines[line].label.clone())
    }

    /// A mark at this point of the recordings.
    pub fn mark(&self) -> Mark {
        Mark {
            serial: self.state().next_serial,
            ticket: None,
            since: std::time::Instant::now(),
        }
    }

    /// Whether what left use at `mark` is free of the card: every command
    /// buffer begun before it is submitted or abandoned (or 10 s passed: a
    /// recording never handed back would hold everything forever), and the
    /// card has run everything given when that was first seen.
    pub fn passed(&self, host: &Host, mark: &mut Mark) -> bool {
        if mark.ticket.is_none() {
            let s = self.state();
            let open = s
                .recorders
                .iter()
                .any(|r| r.done_at == u64::MAX && r.serial < mark.serial);
            if open && mark.since.elapsed() < std::time::Duration::from_secs(10) {
                return false;
            }
            mark.ticket = Some(s.ticket);
        }
        self.lost() || self.completed(host) >= mark.ticket.unwrap_or(0)
    }

    /// Recordings begun and not yet submitted or abandoned: what may
    /// still run with what was bound when each began.
    pub fn open(&self) -> usize {
        self.state().recorders.iter().filter(|r| r.done_at == u64::MAX).count()
    }

    /// The value `line` has reached on the card.
    pub fn line_completed(&self, host: &Host, line: Line) -> u64 {
        let sem = self.state().lines[line].semaphore;
        // SAFETY: a live timeline semaphore.
        unsafe { host.device.get_semaphore_counter_value(sem) }.unwrap_or(0)
    }

    /// Waits until everything given up to `ticket` has run, at most
    /// `timeout_ns`.
    pub fn wait(&self, host: &Host, ticket: u64, timeout_ns: u64) -> bool {
        let targets: Vec<(vk::Semaphore, u64)> = {
            let s = self.state();
            s.lines
                .iter()
                .filter_map(|l| {
                    let last = l.pending.iter().take_while(|p| p.ticket <= ticket).last()?;
                    Some((l.semaphore, last.value))
                })
                .collect()
        };
        wait_semaphores(host, &targets, vk::SemaphoreWaitFlags::empty(), timeout_ns)
    }

    /// Waits until any of `values` (line, value) is reached, at most
    /// `timeout_ns`.
    pub fn wait_any(&self, host: &Host, values: &[(Line, u64)], timeout_ns: u64) -> bool {
        let targets: Vec<(vk::Semaphore, u64)> = {
            let s = self.state();
            values
                .iter()
                .map(|&(l, v)| (s.lines[l].semaphore, v))
                .collect()
        };
        wait_semaphores(host, &targets, vk::SemaphoreWaitFlags::ANY, timeout_ns)
    }

    /// A recorder whose last submission has run (or a new one), taken.
    fn take_recorder(&self, host: &Host, s: &mut State) -> Result<usize> {
        let completed = self.completed_locked(host, s);
        if let Some(i) = s
            .recorders
            .iter()
            .position(|r| r.done_at != u64::MAX && r.done_at <= completed)
        {
            s.recorders[i].done_at = u64::MAX;
            return Ok(i);
        }
        let device = &host.device;
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
            done_at: u64::MAX,
            scratch: Scratch::default(),
            serial: 0,
        });
        Ok(s.recorders.len() - 1)
    }

    /// A command buffer to record into on this thread (the list workers
    /// record parts in parallel), begun, with its start timestamp. Hand it
    /// back with [`Timeline::submit_on`] or [`Timeline::abandon`].
    pub fn begin(&self, host: &Host) -> Result<Recording> {
        let mut freed = Vec::new();
        let (index, pool, cmd, pair) = {
            let mut s = self.state();
            let i = self.take_recorder(host, &mut s)?;
            // Its last recording has run: its scratch is lent again.
            let mut kept: u64 = s.spare.iter().map(|b| b.size).sum();
            for b in std::mem::take(&mut s.recorders[i].scratch).buffers {
                if kept + b.size <= SPARE {
                    kept += b.size;
                    s.spare.push(b);
                } else {
                    freed.push(b);
                }
            }
            let pair = self.timing.map(|_| {
                let p = 2 * (s.next_pair % PAIRS);
                s.next_pair = s.next_pair.wrapping_add(1);
                p
            });
            let serial = s.next_serial;
            s.next_serial += 1;
            let r = &mut s.recorders[i];
            r.serial = serial;
            (i, r.pool, r.cmd, pair)
        };
        for b in freed {
            // SAFETY: the command buffer that used it has run.
            unsafe { b.destroy(host) };
        }
        let device = &host.device;
        // SAFETY: the recorder's last submission has run and it is ours.
        unsafe {
            device.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if let (Some(qp), Some(p)) = (self.timing, pair) {
                device.cmd_reset_query_pool(cmd, qp, p, 2);
                device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, qp, p);
            }
        }
        Ok(Recording {
            index,
            cmd,
            pair,
            scratch: Scratch::default(),
            trail: Trail::default(),
        })
    }

    /// Records a one-shot command buffer with `f` and submits it on the
    /// host line; returns its ticket.
    pub fn record(
        &self,
        host: &Host,
        f: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<u64> {
        let rec = self.begin(host)?;
        f(&host.device, rec.cmd);
        Ok(self.submit(host, HOST, vec![rec], &[])?.1)
    }

    /// Ends `recordings` (begun by [`Timeline::begin`]) and submits them in
    /// order as one submission on `line` (none: a marker that only waits and
    /// signals), behind the line's previous value, the host line's last
    /// value and `waits` (line, value); on the host line, behind everything
    /// given before. Returns the line value it signals.
    pub fn submit_on(
        &self,
        host: &Host,
        line: Line,
        recordings: Vec<Recording>,
        waits: &[(Line, u64)],
    ) -> Result<u64> {
        Ok(self.submit(host, line, recordings, waits)?.0)
    }

    /// [`Timeline::submit_on`]: the line value and the ticket.
    fn submit(
        &self,
        host: &Host,
        line: Line,
        recordings: Vec<Recording>,
        waits: &[(Line, u64)],
    ) -> Result<(u64, u64)> {
        let device = &host.device;
        let mut cmds = Vec::with_capacity(recordings.len());
        for (k, r) in recordings.iter().enumerate() {
            // SAFETY: begun by `begin`, recorded by its owner only.
            unsafe {
                if let (Some(qp), Some(p), true) = (self.timing, r.pair, k + 1 == recordings.len())
                {
                    device.cmd_write_timestamp2(
                        r.cmd,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        qp,
                        p + 1,
                    );
                }
                device.end_command_buffer(r.cmd)?;
            }
            cmds.push(r.cmd);
        }
        let timing = match (
            recordings.first().and_then(|r| r.pair),
            recordings.last().and_then(|r| r.pair),
        ) {
            (Some(a), Some(b)) => Some((a, b + 1)),
            _ => None,
        };
        let mut s = self.state();
        let r = self.submit_locked(host, &mut s, line, &cmds, waits, timing);
        let done = if r.is_ok() { s.ticket } else { 0 };
        for rec in recordings {
            let recorder = &mut s.recorders[rec.index];
            recorder.done_at = done;
            recorder.scratch = rec.scratch;
        }
        r.map(|value| (value, done))
    }

    fn submit_locked(
        &self,
        host: &Host,
        s: &mut State,
        line: Line,
        cmds: &[vk::CommandBuffer],
        extra: &[(Line, u64)],
        timing: Option<(u32, u32)>,
    ) -> Result<u64> {
        anyhow::ensure!(!self.lost(), "the Vulkan device was lost");
        let l = &s.lines[line];
        let value = l.last + 1;
        let waits = submit_infos(&waits(s, line, extra));
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(l.semaphore)
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
        // SAFETY: queues are used under the state's lock only.
        self.check(crate::account::traced(host, "submit", || unsafe {
            host.device
                .queue_submit2(l.queue, &[submit], vk::Fence::null())
        }))
        .context("vkQueueSubmit2")?;
        commit(s, line, value, timing);
        Ok(value)
    }

    /// Hands back a recording that is not submitted.
    pub fn abandon(&self, host: &Host, recording: Recording) {
        // SAFETY: begun by `begin`; never submitted.
        let _ = unsafe { host.device.end_command_buffer(recording.cmd) };
        let mut s = self.state();
        let r = &mut s.recorders[recording.index];
        r.done_at = 0;
        r.scratch = recording.scratch;
    }

    /// A scratch buffer of at least `bytes` for a recording: the smallest
    /// spare one that holds them, else a new one.
    fn lend(&self, host: &Host, bytes: u64) -> Option<Buffer> {
        {
            let mut s = self.state();
            let fit = s.spare.iter().enumerate().filter(|(_, b)| b.size >= bytes).min_by_key(|(_, b)| b.size).map(|(i, _)| i);
            if let Some(i) = fit {
                return Some(s.spare.swap_remove(i));
            }
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        Buffer::new(host, bytes.next_power_of_two().max(SCRATCH), usage, Place::Scratch).ok()
    }

    /// Binds sparse memory on the host line, behind everything given
    /// before; returns its ticket.
    pub fn bind_sparse(&self, host: &Host, binds: &[BufferBinds]) -> Result<u64> {
        anyhow::ensure!(!self.lost(), "the Vulkan device was lost");
        let mut s = self.state();
        let value = s.lines[HOST].last + 1;
        let infos: Vec<vk::SparseBufferMemoryBindInfo> = binds
            .iter()
            .map(|(buffer, b)| {
                vk::SparseBufferMemoryBindInfo::default()
                    .buffer(*buffer)
                    .binds(b)
            })
            .collect();
        let (wait_sems, wait_values): (Vec<_>, Vec<_>) = waits(&s, HOST, &[]).into_iter().unzip();
        let signal_values = [value];
        let mut values = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let sems = [s.lines[HOST].semaphore];
        let info = vk::BindSparseInfo::default()
            .wait_semaphores(&wait_sems)
            .signal_semaphores(&sems)
            .buffer_binds(&infos)
            .push_next(&mut values);
        // SAFETY: queues are used under the state's lock only; the binds
        // name live buffers and memory.
        self.check(crate::account::traced(host, "sparse bind", || unsafe {
            host.device
                .queue_bind_sparse(self.queues[0], &[info], vk::Fence::null())
        }))
        .context("vkQueueBindSparse")?;
        commit(&mut s, HOST, value, None);
        Ok(s.ticket)
    }

    /// Each line's submissions and GPU busy time since the last call.
    pub fn take_stats(&self, host: &Host) -> Vec<LineStats> {
        let mut s = self.state();
        self.completed_locked(host, &mut s);
        s.lines
            .iter_mut()
            .map(|l| LineStats {
                label: l.label.clone(),
                submissions: std::mem::take(&mut l.submissions),
                busy_ms: std::mem::take(&mut l.busy_ns) / 1e6,
            })
            .collect()
    }

    /// # Safety
    /// The device is idle.
    pub unsafe fn destroy(&self, host: &Host) {
        let s = self.state();
        unsafe {
            for r in &s.recorders {
                host.device.destroy_command_pool(r.pool, None);
                for b in &r.scratch.buffers {
                    b.destroy(host);
                }
            }
            for b in &s.spare {
                b.destroy(host);
            }
            for l in &s.lines {
                host.device.destroy_semaphore(l.semaphore, None);
            }
            if let Some(qp) = self.timing {
                host.device.destroy_query_pool(qp, None);
            }
        }
    }
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Direct => "direct",
        Kind::Compute => "compute",
        Kind::Copy => "copy",
    }
}

/// Notes an operation given to `line` that signals `value`.
fn commit(s: &mut State, line: Line, value: u64, timing: Option<(u32, u32)>) {
    s.ticket += 1;
    let ticket = s.ticket;
    let l = &mut s.lines[line];
    l.last = value;
    l.submissions += 1;
    l.pending.push_back(Pending {
        value,
        ticket,
        timing,
    });
}

/// What an operation on `line` waits for (semaphore, value): the line's
/// previous value; on the host line every line's last value; on a title
/// line the host line's last value and `extra`.
fn waits(s: &State, line: Line, extra: &[(Line, u64)]) -> Vec<(vk::Semaphore, u64)> {
    let mut w: Vec<(Line, u64)> = Vec::new();
    let mut add = |l: Line, v: u64| {
        if v == 0 {
            return;
        }
        match w.iter_mut().find(|(x, _)| *x == l) {
            Some(e) => e.1 = e.1.max(v),
            None => w.push((l, v)),
        }
    };
    if line == HOST {
        for (i, l) in s.lines.iter().enumerate() {
            add(i, l.last);
        }
    } else {
        add(line, s.lines[line].last);
        add(HOST, s.lines[HOST].last);
        for &(l, v) in extra {
            add(l, v);
        }
    }
    w.into_iter()
        .map(|(l, v)| (s.lines[l].semaphore, v))
        .collect()
}

fn submit_infos(waits: &[(vk::Semaphore, u64)]) -> Vec<vk::SemaphoreSubmitInfo<'static>> {
    waits
        .iter()
        .map(|&(sem, v)| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(sem)
                .value(v)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        })
        .collect()
}

fn wait_semaphores(
    host: &Host,
    targets: &[(vk::Semaphore, u64)],
    flags: vk::SemaphoreWaitFlags,
    timeout_ns: u64,
) -> bool {
    if targets.is_empty() {
        return true;
    }
    let (sems, values): (Vec<_>, Vec<_>) = targets.iter().copied().unzip();
    // SAFETY: live timeline semaphores.
    unsafe {
        host.device.wait_semaphores(
            &vk::SemaphoreWaitInfo::default()
                .flags(flags)
                .semaphores(&sems)
                .values(&values),
            timeout_ns,
        )
    }
    .is_ok()
}
