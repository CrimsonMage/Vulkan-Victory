//! First uploads of textures the title fills, on a copy queue of their own,
//! and slots that show such a texture only once its upload has run.
//!
//! After VIctory-7900XTX (github.com/avatarmanz/VIctory-7900XTX,
//! GPL-3.0-or-later: `crates/victory-gpu/src/texture/ahead.rs`, its
//! `VICTORY_UPLOAD_AHEAD`, and the slots shown after their upload,
//! `VICTORY_SHOW_AFTER`, in `descriptor/host.rs`), reduced to this tree.
//! There, a slot that showed a texture before its upload had run was read
//! by lists already on the card while the texture was still being made: a
//! device loss every ~15 minutes, gone with it. Here a slot made from a view
//! of a texture whose contents never went up showed whatever the new host
//! texture held (black, or garbage) until a list uploaded it, and uploads
//! waited for room and were detiled on the thread that submits.
//!
//! Now: when a slot is made from a view of a title-filled texture that no
//! upload ever reached ([`show_later`]: the moment the title announces the
//! texture by writing its view), its first upload is handed to two workers
//! of ours. They detile it into its staging buffer, copy it on the copy
//! queue and wait for the queue's fence. Until then the slot shows a 1x1
//! grey texture ([`fallback`]); then it is made from the view again
//! (`crate::descriptor::make_landed`, before the next submission runs).
//! Every later upload of the texture (the title changed it) goes into the
//! lists as before, in order with the draws that read it: the slot keeps
//! showing the old contents until the list's copy has run.
//!
//! Taken only for a texture nothing else has touched: title-filled, its
//! host copy with memory of its own (not made level by level), not written
//! by the card, not an alias of a texture the card writes, in the common or
//! the copy-destination state, and never asked for an upload. While its
//! upload is on its way no list names it ([`pending`]: the draws that read
//! it skip it, a copy from or into it waits for the upload). The copy
//! queue's copy leaves it in the common state, from which the direct
//! queue's reads promote it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::Staging;
use crate::object::This;
use crate::resource::HostTexture;

/// [`HostTexture::ahead`]: no upload asked yet; one on the copy queue; it
/// has run; one recorded into a list (the old way, from then on).
const NONE: u8 = 0;
const PENDING: u8 = 1;
const DONE: u8 = 2;
const LISTED: u8 = 3;

/// Bytes of texture on their way at most (staging memory): past it a new
/// texture's slot is made as before and a list uploads it.
const MOST: u64 = 128 << 20;
/// Bytes and textures of one copy-queue submission at most.
const BATCH: (u64, usize) = (16 << 20, 64);

/// One first upload.
struct Job {
    resource: usize,
    copy: Arc<HostTexture>,
    bytes: u64,
    asked: std::time::Instant,
}

/// Jobs to run, and the host copies (their addresses) on their way.
static QUEUE: (Mutex<VecDeque<Job>>, Condvar) = (Mutex::new(VecDeque::new()), Condvar::new());
static ON_THEIR_WAY: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static LANDED: Condvar = Condvar::new();
static BYTES: AtomicU64 = AtomicU64::new(0);
/// Title textures whose upload ran (or was given back to the lists) since
/// [`take_landed`]: their slots are made again.
static FLIPS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Since the start: uploads, bytes, microseconds from the asking until the
/// copy had run, the longest; turned down for room; lists that waited.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// A slot at `at` is made from a view (`desc`: the title's description,
/// `None`: all of it) of title texture `resource`, whose host copy is
/// `texture`: if the texture's first upload is on its way, or starts now,
/// the slot shows the fallback until it has run, and that is made here (the
/// texture to keep comes back). `None`: the slot shows the texture itself
/// (its contents went up, the lists upload it, or the view is of a kind the
/// fallback cannot stand for).
pub(crate) fn show_later(
    resource: This,
    texture: &ID3D12Resource,
    desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<ID3D12Resource> {
    if !super::title_written(texture) {
        return None;
    }
    let dimension = fallback_dimension(texture, desc)?;
    let copy = crate::resource::made_host_copy(resource)?;
    let later = match copy.ahead.load(Relaxed) {
        PENDING => true,
        NONE => fallback().is_some() && start(resource, copy),
        _ => false,
    };
    if later { fallback_view(dimension, at) } else { None }
}

/// Whether the first upload of title texture `resource` is on its way: no
/// list may name it until it has run.
pub(crate) fn pending(resource: This) -> bool {
    crate::resource::made_host_copy(resource).is_some_and(|copy| copy.ahead.load(Relaxed) == PENDING)
}

/// A list is about to record an upload of, a copy from or a copy into
/// `resource`: the copy queue takes no first upload of it from now on, and
/// one on its way runs first.
pub(crate) fn listed(resource: This) {
    let Some(copy) = crate::resource::made_host_copy(resource) else {
        return;
    };
    if copy.ahead.compare_exchange(NONE, LISTED, Relaxed, Relaxed) != Err(PENDING) {
        return;
    }
    STATS[5].fetch_add(1, Relaxed);
    let key = Arc::as_ptr(&copy) as usize;
    let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::time::Instant::now();
    while way.contains(&key) && started.elapsed().as_secs() < 2 {
        way = LANDED.wait_timeout(way, std::time::Duration::from_millis(20)).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// Starts the first upload of `resource`, if it may go to the copy queue.
fn start(resource: This, copy: Arc<HostTexture>) -> bool {
    const COPY_DEST: u32 = 0x400;
    let state = copy.state.load(Relaxed);
    if copy.tiled.is_some()
        || (state != 0 && state != COPY_DEST)
        || super::host_wrote(resource as usize)
        || crate::resource::alias_of(resource).is_some()
        || workers().is_none()
    {
        return false;
    }
    let bytes = crate::resource::host_bytes(&copy.texture);
    if BYTES.load(Relaxed).saturating_add(bytes) > MOST {
        STATS[4].fetch_add(1, Relaxed);
        return false;
    }
    if copy.ahead.compare_exchange(NONE, PENDING, Relaxed, Relaxed).is_err() {
        return copy.ahead.load(Relaxed) == PENDING;
    }
    BYTES.fetch_add(bytes, Relaxed);
    ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::as_ptr(&copy) as usize);
    let job = Job { resource: resource as usize, copy, bytes, asked: std::time::Instant::now() };
    QUEUE.0.lock().unwrap_or_else(|e| e.into_inner()).push_back(job);
    QUEUE.1.notify_one();
    true
}

/// The title textures whose slots wait for nothing any more.
pub(crate) fn take_landed() -> Vec<usize> {
    std::mem::take(&mut *FLIPS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The copy queue and its workers, made on first use.
fn workers() -> Option<&'static ID3D12CommandQueue> {
    struct Queue(ID3D12CommandQueue);
    // SAFETY: a command queue is free-threaded.
    unsafe impl Send for Queue {}
    // SAFETY: as above.
    unsafe impl Sync for Queue {}
    static WORKERS: OnceLock<Option<Queue>> = OnceLock::new();
    WORKERS
        .get_or_init(|| {
            let host = crate::host::host()?;
            // SAFETY: plain object creation.
            let queue: ID3D12CommandQueue = unsafe {
                host.device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })
                    .ok()?
            };
            for n in 0..2 {
                let worker = Worker::new(&host.device, queue.clone()).ok()?;
                std::thread::Builder::new().name(format!("victory-uploads-{n}")).spawn(move || worker.work()).ok()?;
            }
            tracing::info!("first uploads of textures go on a copy queue of their own");
            Some(Queue(queue))
        })
        .as_ref()
        .map(|q| &q.0)
}

struct Worker {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
}

// SAFETY: a worker's objects are its thread's own; the queue is free-threaded.
unsafe impl Send for Worker {}

impl Worker {
    fn new(device: &ID3D12Device, queue: ID3D12CommandQueue) -> windows::core::Result<Worker> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            Ok(Worker { queue, allocator, list, fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? })
        }
    }

    fn work(self) {
        victory_core::heap::enter_for_thread(victory_core::heap::Owner::Textures);
        let mut value = 0;
        loop {
            let mut jobs = Vec::new();
            {
                let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
                while queue.is_empty() {
                    queue = QUEUE.1.wait(queue).unwrap_or_else(|e| e.into_inner());
                }
                let mut bytes = 0;
                while bytes < BATCH.0
                    && jobs.len() < BATCH.1
                    && let Some(job) = queue.pop_front()
                {
                    bytes += job.bytes;
                    jobs.push(job);
                }
            }
            // The staging buffers, detiled here; `None`: the texture is the
            // lists' again (no console layout, or the host copy changed).
            let staged: Vec<Option<Arc<Staging>>> = jobs.iter().map(stage).collect();
            value += 1;
            if staged.iter().any(Option::is_some)
                && let Err(e) = self.copy(&jobs, &staged, value)
            {
                tracing::warn!(error = %e, "first texture uploads on the copy queue did not run");
            }
            finish(jobs, staged);
        }
    }

    fn copy(&self, jobs: &[Job], staged: &[Option<Arc<Staging>>], value: u64) -> windows::core::Result<()> {
        // SAFETY: this worker's allocator and list, whose last submission has
        // run; the jobs hold what the list names until after the wait.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            for (job, staging) in jobs.iter().zip(staged) {
                if let Some(staging) = staging {
                    record(&self.list, &job.copy.texture, staging);
                }
            }
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.queue.Signal(&self.fence, value)?;
            // (Without an event: this waits.)
            self.fence.SetEventOnCompletion(value, windows::Win32::Foundation::HANDLE::default())?;
        }
        Ok(())
    }
}

/// The job's staging buffer, detiled from title memory now.
fn stage(job: &Job) -> Option<Arc<Staging>> {
    let resource = job.resource as This;
    // The host copy the job was asked for is still the resource's.
    let now = crate::resource::made_host_copy(resource)?;
    if !Arc::ptr_eq(&now, &job.copy) {
        return None;
    }
    let staging = super::staging(resource, &job.copy.texture).filter(|s| !s.on_demand)?;
    let began = std::time::Instant::now();
    staging.refresh();
    crate::frametime::add_since(crate::frametime::Sum::AheadNs, began);
    Some(staging)
}

/// Records the copy of every subresource of `staging` into `texture`.
///
/// # Safety
/// `list` is an open copy list; both live until it has run.
unsafe fn record(list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, staging: &Staging) {
    for (i, footprint) in staging.footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        // SAFETY: the caller's.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// The jobs' copies have run (or were not made: those textures are the
/// lists' again, uploaded as before).
fn finish(jobs: Vec<Job>, staged: Vec<Option<Arc<Staging>>>) {
    const COPY_DEST: u32 = 0x400;
    for (job, staging) in jobs.iter().zip(&staged) {
        match staging {
            Some(staging) => {
                staging.uploaded.store(true, Relaxed);
                staging.dirty.store(false, Relaxed);
                // The copy queue leaves it in the common state.
                let _ = job.copy.state.compare_exchange(COPY_DEST, 0, Relaxed, Relaxed);
                job.copy.ahead.store(DONE, Relaxed);
                let us = job.asked.elapsed().as_micros() as u64;
                STATS[0].fetch_add(1, Relaxed);
                STATS[1].fetch_add(job.bytes, Relaxed);
                STATS[2].fetch_add(us, Relaxed);
                STATS[3].fetch_max(us, Relaxed);
            }
            None => job.copy.ahead.store(LISTED, Relaxed),
        }
        BYTES.fetch_sub(job.bytes, Relaxed);
    }
    {
        let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        way.retain(|key| !jobs.iter().any(|job| Arc::as_ptr(&job.copy) as usize == *key));
        LANDED.notify_all();
    }
    FLIPS.lock().unwrap_or_else(|e| e.into_inner()).extend(jobs.iter().map(|job| job.resource));
    // The staging buffers go with the jobs, unless a list holds them.
    drop(staged);
}

/// For the levels line: first uploads on the copy queue since the start,
/// their MB, the ms from the asking until they had run (average, longest),
/// those turned down for room, lists that waited for one.
pub(crate) fn line() -> String {
    let n = STATS[0].load(Relaxed);
    format!(
        "first uploads on the copy queue {n} ({} MB, {:.1} ms from the asking on average, longest {:.0}; {} left to the lists for room, {} waited for by lists)",
        STATS[1].load(Relaxed) >> 20,
        STATS[2].load(Relaxed) as f64 / n.max(1) as f64 / 1e3,
        STATS[3].load(Relaxed) as f64 / 1e3,
        STATS[4].load(Relaxed),
        STATS[5].load(Relaxed)
    )
}

/// A 1x1 grey texture (all four channels 0x80), in the place of one whose
/// first upload has not run; made once, through the copy queue.
fn fallback() -> Option<&'static ID3D12Resource> {
    struct Made(ID3D12Resource);
    // SAFETY: a D3D12 resource is free-threaded; it is never written again.
    unsafe impl Send for Made {}
    // SAFETY: as above.
    unsafe impl Sync for Made {}
    static TEXTURE: OnceLock<Option<Made>> = OnceLock::new();
    TEXTURE
        .get_or_init(|| {
            let made = make_fallback();
            if made.is_none() {
                tracing::warn!("no fallback texture: slots show textures before their first upload has run");
            }
            made.map(Made)
        })
        .as_ref()
        .map(|m| &m.0)
}

fn make_fallback() -> Option<ID3D12Resource> {
    let host = crate::host::host()?;
    let queue = workers()?;
    let device = &host.device;
    let (from, data) = host.upload_buffer(256)?;
    // SAFETY: the buffer's own mapping; plain creation and recording, waited
    // for before anything here goes.
    unsafe {
        std::ptr::write_bytes(data, 0x80, 256);
        let mut made: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 1,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut made,
            )
            .ok()?;
        let made = made?;
        let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
        let list: ID3D12GraphicsCommandList =
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None).ok()?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&made),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let source = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&from),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 1,
                        Height: 1,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &source, None);
        list.Close().ok()?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        queue.ExecuteCommandLists(&[Some(list.cast().ok()?)]);
        queue.Signal(&fence, 1).ok()?;
        // (Without an event: this waits.)
        fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).ok()?;
        Some(made)
    }
}

/// The dimension of the fallback's view for a view `desc` (the title's, or
/// none: the whole texture `texture`): a 2D texture or a 2D array; `None`
/// for other views (they show their texture as before).
fn fallback_dimension(
    texture: &ID3D12Resource,
    desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>,
) -> Option<D3D12_SRV_DIMENSION> {
    let dimension = match desc {
        Some(d) => d.ViewDimension,
        None => {
            // SAFETY: a live resource.
            let d = unsafe { texture.GetDesc() };
            match (d.Dimension, d.DepthOrArraySize) {
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1) => D3D12_SRV_DIMENSION_TEXTURE2D,
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, _) => D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                _ => return None,
            }
        }
    };
    matches!(dimension, D3D12_SRV_DIMENSION_TEXTURE2D | D3D12_SRV_DIMENSION_TEXTURE2DARRAY).then_some(dimension)
}

/// Makes the fallback's view of `dimension` at `at`: the texture to keep.
fn fallback_view(dimension: D3D12_SRV_DIMENSION, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    let anonymous = if dimension == D3D12_SRV_DIMENSION_TEXTURE2D {
        D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } }
    } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
        }
    };
    let (fallback, host) = (fallback()?, crate::host::host()?);
    let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: dimension,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: anonymous,
    };
    // SAFETY: a live texture and a slot of our heap.
    unsafe { host.device.CreateShaderResourceView(fallback, Some(&view), at) };
    Some(fallback.clone())
}
