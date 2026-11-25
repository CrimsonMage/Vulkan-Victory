//! Unified memory (`docs/unified-memory.md`): the title's GPU-visible
//! memory is host heaps that the GPU reads and writes where they are, with
//! no host copy of title buffers and no write detection. This file holds
//! the one switch and the startup probe that says what this PC's driver
//! allows; the stages that use it are described in the document.
//!
//! `VICTORY_UNIFIED`: 0 (unset) off, nothing changes. 1: the probe, once,
//! at the first sweep, on objects of its own (its own queue, list, heaps
//! and memory; nothing of the title's is touched); its lines start with
//! "unified probe:" and end with a VERDICT line. 2: also the tile-mapping
//! test (a reserved buffer mapped into a heap opened on a section), which
//! a driver may answer by removing the device: run 2 on its own, and only
//! after 1 has been read.
//!
//! `VICTORY_UNIFIED_PROBE_MB` (256, 16 to 2048): the size of the bandwidth
//! measurements. `VICTORY_UNIFIED_PROBE_BIG_MB` (0: left out): one section
//! of that size opened whole, as the title's 6.2 GB page pool would be
//! (6400 is the pool's size; it pins that much of the PC's memory for the
//! moment of the test).
//!
//! Note for reading the run: vkd3d-proton turns off its promotion of upload
//! heaps into the card's memory once a GPU-upload heap has been made. On an
//! 8 GB card it does not promote them anyway (it wants 9 GB), so the run's
//! frame rate is not changed by the probe there; on a bigger card compare
//! frame rates only between runs without the probe.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};
use windows_sys::Win32::System::Memory as mem;

/// The card's tile, the console's page and the placement unit.
const K64: usize = 64 << 10;
/// Title addresses lie below this (40-bit texture descriptors, the
/// ray-tracing builder; `xmem` keeps them there).
const TIB: u64 = 1 << 40;

/// `VICTORY_UNIFIED`, see the module comment.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("VICTORY_UNIFIED").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    })
}

fn env_mb(name: &str, default: usize, most: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default).min(most)
}

/// Called at every sweep: runs the probe once when [`level`] asks for it.
pub(crate) fn tick() {
    if level() == 0 {
        return;
    }
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    let Some(host) = crate::host::host() else {
        DONE.store(false, Ordering::Relaxed);
        return;
    };
    let mb = env_mb("VICTORY_UNIFIED_PROBE_MB", 256, 2048).max(16);
    let big = env_mb("VICTORY_UNIFIED_PROBE_BIG_MB", 0, 16384);
    tracing::info!(
        "unified probe: starting (VICTORY_UNIFIED={}, bandwidth over {mb} MB, big section {big} MB)",
        level()
    );
    let took = std::time::Instant::now();
    let memory = || host.video_memory_raw();
    let lines = probe(&host.device, &memory, mb, big, level() >= 2);
    for line in &lines {
        tracing::info!("unified probe: {line}");
    }
    tracing::info!("unified probe: took {:.0} ms", took.elapsed().as_secs_f64() * 1000.0);
}

fn hr<T>(result: &windows::core::Result<T>) -> String {
    match result {
        Ok(_) => "ok".to_owned(),
        Err(e) => format!("REFUSED {:#010x} ({})", e.code().0, e.message().trim()),
    }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// The probe's texture: 256 by 256 RGBA8, one level (256 KiB).
const SIDE: u32 = 256;
const TEXTURE_BYTES: usize = (SIDE * SIDE * 4) as usize;

fn texture_desc(layout: D3D12_TEXTURE_LAYOUT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(SIDE),
        Height: SIDE,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: layout,
        Flags: flags,
        ..Default::default()
    }
}

fn footprint() -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: 0,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Width: SIDE,
            Height: SIDE,
            Depth: 1,
            RowPitch: SIDE * 4,
        },
    }
}

fn gpu_upload() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
}

/// What D3D12 calls the same memory without the GPU-upload heap type:
/// write-combined pages in the card's own memory.
fn custom_l1() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L1,
        ..Default::default()
    }
}

fn plain(kind: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() }
}

fn make_heap(device: &ID3D12Device, properties: D3D12_HEAP_PROPERTIES, size: u64, flags: D3D12_HEAP_FLAGS) -> windows::core::Result<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC { SizeInBytes: size, Properties: properties, Alignment: K64 as u64, Flags: flags };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain creation of an object of our own.
    unsafe { device.CreateHeap(&desc, &mut heap)? };
    heap.ok_or_else(windows::core::Error::empty)
}

fn place(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    offset: u64,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let mut placed: Option<ID3D12Resource> = None;
    // SAFETY: plain creation in a live heap.
    unsafe { device.CreatePlacedResource(heap, offset, desc, state, None, &mut placed)? };
    placed.ok_or_else(windows::core::Error::empty)
}

fn map(resource: &ID3D12Resource) -> windows::core::Result<*mut u8> {
    let mut data = std::ptr::null_mut();
    // SAFETY: a buffer in a heap the CPU may map; the mapping lives as
    // long as the buffer (the whole range may be read).
    unsafe { resource.Map(0, None, Some(&mut data))? };
    Ok(data.cast())
}

fn describe(heap: &ID3D12Heap) -> String {
    // SAFETY: a plain query.
    let d = unsafe { heap.GetDesc() };
    format!(
        "heap of {} KiB: type {} (4 custom, 5 GPU upload), CPU pages {} (2 write-combine, 3 write-back), pool {} (1 the PC's memory, 2 the card's), flags {:#x}",
        d.SizeInBytes >> 10,
        d.Properties.Type.0,
        d.Properties.CPUPageProperty.0,
        d.Properties.MemoryPoolPreference.0,
        d.Flags.0
    )
}

fn fill(at: *mut u8, len: usize, seed: u8) {
    for i in 0..len {
        // SAFETY: the caller's own memory of `len` bytes.
        unsafe { at.add(i).write((i as u8).wrapping_mul(31).wrapping_add(seed) ^ (i >> 16) as u8) };
    }
}

fn same(at: *const u8, bytes: &[u8]) -> bool {
    // SAFETY: the caller's own memory of that length.
    unsafe { std::slice::from_raw_parts(at, bytes.len()) == bytes }
}

fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer the barrier never releases.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn subresource(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    }
}

fn rows(buffer: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint() },
    }
}

/// The process's working set in MB.
fn working_set_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    let mut counters = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut counters, counters.cb) };
    counters.WorkingSetSize as u64 >> 20
}

/// A queue, a list, a fence and two timestamps of the probe's own.
struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
    stamps: ID3D12QueryHeap,
    stamps_back: ID3D12Resource,
    frequency: u64,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

impl Rig {
    fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let mut stamps: Option<ID3D12QueryHeap> = None;
            device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut stamps)?;
            let stamps = stamps.ok_or_else(windows::core::Error::empty)?;
            let frequency = queue.GetTimestampFrequency()?;
            let mut rig = Rig {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence,
                done: 0,
                event,
                stamps,
                stamps_back: committed(device, plain(D3D12_HEAP_TYPE_READBACK), 16, D3D12_RESOURCE_STATE_COPY_DEST)?,
                frequency,
            };
            // A first empty run: the queue is up before anything is timed.
            rig.run(|_| ())?;
            Ok(rig)
        }
    }

    /// Records with `record`, runs it and waits (ten seconds at most).
    fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            record(&self.list);
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.wait()
        }
    }

    /// Waits for everything sent to the queue so far.
    fn wait(&mut self) -> windows::core::Result<()> {
        // SAFETY: our own queue, fence and event.
        unsafe {
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            self.fence.SetEventOnCompletion(self.done, self.event)?;
            if WaitForSingleObject(self.event, 10_000).0 != 0 {
                return Err(windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL));
            }
            self.device.GetDeviceRemovedReason()
        }
    }

    /// What the GPU reads of `size` bytes of `resource` from `offset`.
    fn fetch(&mut self, resource: &ID3D12Resource, offset: u64, size: usize) -> windows::core::Result<Vec<u8>> {
        let back = committed(&self.device, plain(D3D12_HEAP_TYPE_READBACK), size as u64, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(&back, 0, resource, offset, size as u64) })?;
        let data = map(&back)?;
        // SAFETY: the mapping holds `size` bytes while `back` lives.
        Ok(unsafe { std::slice::from_raw_parts(data, size) }.to_vec())
    }

    /// The GPU writes `bytes` at `offset` of `resource` (a copy).
    fn store(&mut self, resource: &ID3D12Resource, offset: u64, bytes: &[u8]) -> windows::core::Result<()> {
        let up = committed(&self.device, plain(D3D12_HEAP_TYPE_UPLOAD), bytes.len() as u64, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let data = map(&up)?;
        // SAFETY: the mapping holds the buffer's bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(resource, offset, &up, 0, bytes.len() as u64) })
    }

    /// A compute shader stores, through a root unordered-access view at
    /// `gpu`, the sum of the buffer's first two words in its third.
    fn shade(&mut self, gpu: u64) -> windows::core::Result<()> {
        const SOURCE: &str = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n[RootSignature(RS)]\n[numthreads(1, 1, 1)]\nvoid main() { b.Store(8, b.Load(0) + b.Load(4)); }\n";
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; then the
        // blob's own bytes, which hold the root signature too.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("unified"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)?;
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            (root, pso)
        };
        // SAFETY: an open list of ours; the view names a live buffer.
        self.run(|list| unsafe {
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootUnorderedAccessView(0, gpu);
            list.Dispatch(1, 1, 1);
        })
    }

    /// Milliseconds of the card's clock to copy `bytes` of `from` into a
    /// buffer in its own memory, the best of three.
    fn copy_ms(&mut self, from: &ID3D12Resource, bytes: u64) -> windows::core::Result<f64> {
        let to = committed(&self.device, plain(D3D12_HEAP_TYPE_DEFAULT), bytes, D3D12_RESOURCE_STATE_COMMON)?;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let (stamps, back) = (self.stamps.clone(), self.stamps_back.clone());
            // SAFETY: live buffers of `bytes` bytes or more; our query heap.
            self.run(|list| unsafe {
                list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                list.CopyBufferRegion(&to, 0, from, 0, bytes);
                list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
            })?;
            let data = map(&self.stamps_back)?.cast::<[u64; 2]>();
            // SAFETY: the read-back buffer holds two timestamps.
            let [a, b] = unsafe { data.read_unaligned() };
            best = best.min(b.saturating_sub(a) as f64 * 1000.0 / self.frequency.max(1) as f64);
        }
        Ok(best)
    }

    /// The GPU writes a pattern into `texture` (in the copy-destination
    /// state) through a copy and reads it back through another: whether the
    /// texture keeps what is written to it.
    fn texture_round_trip(&mut self, texture: &ID3D12Resource) -> windows::core::Result<bool> {
        let up = committed(&self.device, plain(D3D12_HEAP_TYPE_UPLOAD), TEXTURE_BYTES as u64, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let back = committed(&self.device, plain(D3D12_HEAP_TYPE_READBACK), TEXTURE_BYTES as u64, D3D12_RESOURCE_STATE_COPY_DEST)?;
        let data = map(&up)?;
        fill(data, TEXTURE_BYTES, 9);
        // SAFETY: the mapping holds the buffer's bytes.
        let want = unsafe { std::slice::from_raw_parts(data, TEXTURE_BYTES) }.to_vec();
        // SAFETY: live resources; footprints fit the texture and buffers.
        self.run(|list| unsafe {
            list.CopyTextureRegion(&subresource(texture), 0, 0, 0, &rows(&up), None);
            list.ResourceBarrier(&[transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyTextureRegion(&rows(&back), 0, 0, 0, &subresource(texture), None);
        })?;
        Ok(same(map(&back)?, &want))
    }
}

fn committed(
    device: &ID3D12Device,
    heap: D3D12_HEAP_PROPERTIES,
    size: u64,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE), state, None, &mut buffer)? };
    buffer.ok_or_else(windows::core::Error::empty)
}

/// What the probe found, for the verdict.
#[derive(Default)]
struct Found {
    gpu_upload: bool,
    /// The CPU address of a GPU-upload buffer's mapping.
    upload_at: Option<u64>,
    upload_reads: bool,
    upload_writes: bool,
    upload_texture: bool,
    custom_texture: bool,
    /// Two views of one section (aliases, as `xmem` maps physical pages)
    /// below 1 TiB, and a heap opened on the section: the GPU reads what
    /// the CPU wrote through either view and the CPU reads what the GPU
    /// wrote through both.
    alias_reads: bool,
    alias_writes: bool,
    alias_texture: bool,
    private_reads: bool,
    /// GB/s the card reads at: from an opened section, a GPU-upload heap,
    /// its own memory.
    read_gbps: [Option<f64>; 3],
    /// GB/s the CPU writes and reads a GPU-upload mapping at.
    cpu_gbps: [Option<f64>; 2],
    big: Option<bool>,
    tiles: Option<bool>,
}

/// Runs every test on objects of the probe's own on `device`; `memory` is
/// the card's memory account (`Host::video_memory_raw`).
fn probe(device: &ID3D12Device, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize, big: usize, tiles: bool) -> Vec<String> {
    let mut lines = Vec::new();
    let mut f = Found::default();
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => {
            lines.push(format!("no queue of its own: {e}"));
            return lines;
        }
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let mut options16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: plain queries that fill our structures.
    let asked16 = unsafe {
        let _ = device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut options).cast(), size_of_val(&options) as u32);
        device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut options16).cast(), size_of_val(&options16) as u32)
    };
    f.gpu_upload = options16.GPUUploadHeapSupported.as_bool();
    let card = memory().map_or_else(|| "?".to_owned(), |m| format!("budget {} MB, in use {} MB; the PC's memory shared with it {} MB", m[0] >> 20, m[1] >> 20, m[3] >> 20));
    lines.push(format!(
        "the device: GPU-upload heaps {} (options16 {}); heap tier {}; ID3D12Device3 {}, ID3D12Device13 {}; timestamps at {} Hz; the card: {card}",
        f.gpu_upload,
        hr(&asked16),
        options.ResourceHeapTier.0,
        device.cast::<ID3D12Device3>().is_ok(),
        device.cast::<ID3D12Device13>().is_ok(),
        rig.frequency
    ));
    let bytes = (mb << 20) as u64;
    upload_heaps(&mut lines, &mut f, &mut rig, bytes);
    textures(&mut lines, &mut f, &mut rig);
    // SAFETY: memory of the probe's own, made and released in each.
    unsafe {
        aliases(&mut lines, &mut f, &mut rig, tiles);
        private(&mut lines, &mut f, &mut rig);
        section_bandwidth(&mut lines, &mut f, &mut rig, bytes);
        if big > 0 {
            big_section(&mut lines, &mut f, &mut rig, memory, big);
        }
    }
    if let Ok(default) = committed(device, plain(D3D12_HEAP_TYPE_DEFAULT), bytes, D3D12_RESOURCE_STATE_COMMON) {
        f.read_gbps[2] = rig.copy_ms(&default, bytes).ok().map(|ms| gbps(bytes, ms));
    }
    let g = |v: Option<f64>| v.map_or_else(|| "?".to_owned(), |v| format!("{v:.1}"));
    lines.push(format!(
        "the card reads {mb} MB at: {} GB/s from a heap opened on a section (the PC's memory), {} GB/s from a GPU-upload heap, {} GB/s from its own memory; \
         the CPU writes a GPU-upload mapping at {} GB/s and reads it at {} GB/s",
        g(f.read_gbps[0]),
        g(f.read_gbps[1]),
        g(f.read_gbps[2]),
        g(f.cpu_gbps[0]),
        g(f.cpu_gbps[1])
    ));
    let low = f.upload_at.map(|at| at < TIB);
    lines.push(format!(
        "VERDICT: title memory as opened sections (the PC's memory): aliases seen by the GPU {} and GPU writes seen through every alias {}, \
         private memory {}, a texture placed in it {}; GPU-upload heap: supported {}, works for buffers {}, mapped below 1 TiB {} (at {}), \
         holds a texture {} (custom L1 heap: {}); big section {}; tile mappings into an opened section {}. \
         Title memory can be a GPU-upload heap only if it is mapped below 1 TiB AND holds textures AND aliases are not needed; \
         it can be opened sections if both alias answers are true.",
        f.alias_reads,
        f.alias_writes,
        f.private_reads,
        f.alias_texture,
        f.gpu_upload,
        f.upload_reads && f.upload_writes,
        low.map_or_else(|| "?".to_owned(), |l| l.to_string()),
        f.upload_at.map_or_else(|| "?".to_owned(), |at| format!("{at:#x}")),
        f.upload_texture,
        f.custom_texture,
        f.big.map_or_else(|| "not tried".to_owned(), |b| b.to_string()),
        f.tiles.map_or_else(|| "not tried (VICTORY_UNIFIED=2)".to_owned(), |b| b.to_string()),
    ));
    lines
}

fn gbps(bytes: u64, ms: f64) -> f64 {
    bytes as f64 / (ms.max(1e-6) / 1000.0) / 1e9
}

/// A GPU-upload heap and a custom heap in the card's memory with
/// write-combined pages: a buffer placed in each, where the CPU sees it,
/// whether the bytes go both ways, and the bandwidths.
fn upload_heaps(lines: &mut Vec<String>, f: &mut Found, rig: &mut Rig, bytes: u64) {
    let device = rig.device.clone();
    for (name, properties) in [("GPU-upload heap", gpu_upload()), ("custom heap, card memory, write-combined", custom_l1())] {
        let heap = make_heap(&device, properties, bytes, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
        lines.push(format!("{name} of {} MB for buffers: {}", bytes >> 20, hr(&heap)));
        let Ok(heap) = heap else { continue };
        lines.push(format!("  {}", describe(&heap)));
        let placed = place(&device, &heap, 0, &buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON);
        let mapped = placed.as_ref().map_err(Clone::clone).and_then(map);
        lines.push(format!("  a buffer placed over all of it: {}; mapped: {}", hr(&placed), hr(&mapped)));
        let (Ok(buffer), Ok(at)) = (placed, mapped) else { continue };
        let first = name.starts_with("GPU");
        lines.push(format!(
            "  the CPU sees it at {:p}: {} 1 TiB (title addresses must lie below; xmem puts them there)",
            at,
            if (at as u64) < TIB { "below" } else { "ABOVE" }
        ));
        fill(at, 4 * K64, 1);
        // SAFETY: the mapping holds `bytes` bytes, more than 4 pages.
        let wrote = unsafe { std::slice::from_raw_parts(at, 4 * K64) }.to_vec();
        let reads = rig.fetch(&buffer, 0, 4 * K64).is_ok_and(|got| got == wrote);
        let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x5a).collect();
        let copied = rig.store(&buffer, 0, &new).is_ok() && same(at, &new);
        // SAFETY: the first three words of the mapping.
        unsafe { at.cast::<[u32; 3]>().write_unaligned([40, 2, 0]) };
        // SAFETY: a live buffer.
        let shaded = rig.shade(unsafe { buffer.GetGPUVirtualAddress() });
        // SAFETY: as above.
        let shader = shaded.is_ok() && unsafe { at.cast::<[u32; 3]>().read_unaligned() }[2] == 42;
        lines.push(format!(
            "  the GPU reads what the CPU wrote {reads}; the CPU reads what the GPU wrote: by a copy {copied}, by a shader through unordered access {shader} ({})",
            hr(&shaded)
        ));
        if first {
            f.upload_at = Some(at as u64);
            f.upload_reads = reads;
            f.upload_writes = copied && shader;
            // The CPU's bandwidths into and out of the card's memory.
            let src = vec![0x3cu8; bytes as usize];
            let started = std::time::Instant::now();
            // SAFETY: the mapping holds `bytes` bytes.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), at, src.len()) };
            f.cpu_gbps[0] = Some(gbps(bytes, started.elapsed().as_secs_f64() * 1000.0));
            let n = (bytes as usize).min(32 << 20);
            let mut back = vec![0u8; n];
            let started = std::time::Instant::now();
            // SAFETY: as above, `n` bytes of it.
            unsafe { std::ptr::copy_nonoverlapping(at, back.as_mut_ptr(), n) };
            f.cpu_gbps[1] = Some(gbps(n as u64, started.elapsed().as_secs_f64() * 1000.0));
            std::hint::black_box(&back);
            f.read_gbps[1] = rig.copy_ms(&buffer, bytes).ok().map(|ms| gbps(bytes, ms));
        }
    }
    // The same memory as a committed buffer, the way a heap of the title's
    // own device calls would be made.
    let committed = committed(&device, gpu_upload(), K64 as u64, D3D12_RESOURCE_STATE_COMMON);
    lines.push(format!("a committed buffer in a GPU-upload heap: {}", hr(&committed)));
}

/// Textures (256 by 256 RGBA8, the card's own layout) placed in heaps of
/// each kind, and a committed one in a GPU-upload heap; each that is made
/// is written and read back by the GPU.
fn textures(lines: &mut Vec<String>, f: &mut Found, rig: &mut Rig) {
    let device = rig.device.clone();
    let kinds = [("the card's memory (control)", plain(D3D12_HEAP_TYPE_DEFAULT)), ("GPU-upload", gpu_upload()), ("custom, card memory, write-combined", custom_l1())];
    for (i, (name, properties)) in kinds.into_iter().enumerate() {
        let heap = make_heap(&device, properties, 4 * K64 as u64 * 4, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES);
        let placed = heap
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|heap| place(&device, heap, 0, &texture_desc(D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST));
        let kept = placed.as_ref().ok().map(|t| rig.texture_round_trip(t));
        let ok = matches!(kept, Some(Ok(true)));
        lines.push(format!(
            "texture heap in {name}: {}; a texture placed in it: {}; written and read back by the GPU: {}",
            hr(&heap),
            hr(&placed),
            match &kept {
                None => "not tried".to_owned(),
                Some(Ok(k)) => k.to_string(),
                Some(Err(e)) => format!("failed ({e})"),
            }
        ));
        match i {
            1 => f.upload_texture = ok,
            2 => f.custom_texture = ok,
            _ => {}
        }
    }
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    let made = unsafe {
        device.CreateCommittedResource(
            &gpu_upload(),
            D3D12_HEAP_FLAG_NONE,
            &texture_desc(D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut texture,
        )
    };
    lines.push(format!("a committed texture in a GPU-upload heap: {}", hr(&made)));
}

/// A reservation below 1 TiB (from 4 GiB on), as `xmem` makes the title's:
/// a placeholder of `bytes`, or null.
unsafe fn placeholder(bytes: usize) -> *mut u8 {
    let mut requirements = mem::MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 32) as *mut c_void,
        HighestEndingAddress: (TIB as usize - 1) as *mut c_void,
        Alignment: 0,
    };
    let mut parameter = mem::MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = mem::MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: the parameter points at `requirements`, alive for the call.
    unsafe {
        mem::VirtualAlloc2(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            std::ptr::null(),
            bytes,
            mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
            mem::PAGE_NOACCESS,
            &mut parameter,
            1,
        )
        .cast()
    }
}

/// Maps `len` bytes of `section` from `offset` over the placeholder part
/// at `at`; whether it landed there.
unsafe fn view_at(section: windows_sys::Win32::Foundation::HANDLE, at: *mut u8, offset: u64, len: usize) -> bool {
    // SAFETY: `at` starts a placeholder of exactly `len` bytes.
    unsafe {
        mem::MapViewOfFile3(
            section,
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            at.cast(),
            offset,
            len,
            mem::MEM_REPLACE_PLACEHOLDER,
            mem::PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
        .Value
        .cast::<u8>()
            == at
    }
}

unsafe fn unmap(at: *mut u8) {
    // SAFETY: a view we mapped; it goes, and its range with it.
    unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at.cast() }) };
}

fn open_section(device: &ID3D12Device, section: windows_sys::Win32::Foundation::HANDLE) -> windows::core::Result<ID3D12Heap> {
    // SAFETY: a live section handle; a refusal is an error code.
    unsafe { device.cast::<ID3D12Device3>()?.OpenExistingHeapFromFileMapping(HANDLE(section)) }
}

unsafe fn section(bytes: usize) -> windows_sys::Win32::Foundation::HANDLE {
    // SAFETY: plain creation of a pagefile-backed section, as `xmem` makes
    // the title's physical pages under Wine (committed).
    unsafe {
        mem::CreateFileMappingW(
            windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
            std::ptr::null(),
            mem::PAGE_READWRITE | mem::SEC_COMMIT,
            (bytes as u64 >> 32) as u32,
            bytes as u32,
            std::ptr::null(),
        )
    }
}

/// A section of 64 pages, two views of its first 32 pages below 1 TiB
/// (aliases, as `xmem` maps one physical page at several addresses), a
/// heap opened on the section and a buffer placed over it: the GPU sees
/// the CPU's writes through either view, the CPU sees the GPU's through
/// both. Also a texture placed in it, and with `tiles` a reserved buffer
/// mapped into it page by page in reverse.
unsafe fn aliases(lines: &mut Vec<String>, f: &mut Found, rig: &mut Rig, tiles: bool) {
    const PAGES: usize = 64;
    const SHOWN: usize = 32 * K64;
    let device = rig.device.clone();
    // SAFETY (the whole function): the section, placeholder and views made
    // here; every address used lies inside them.
    unsafe {
        let handle = section(PAGES * K64);
        let hole = placeholder(2 * SHOWN);
        if handle.is_null() || hole.is_null() {
            lines.push("aliases: no section or reservation to try with".to_owned());
            return;
        }
        mem::VirtualFree(hole.cast(), SHOWN, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
        let (a, b) = (hole, hole.add(SHOWN));
        let mapped = (view_at(handle, a, 0, SHOWN), view_at(handle, b, 0, SHOWN));
        let opened = open_section(&device, handle);
        lines.push(format!(
            "aliases: two views of one section at {a:p} and {b:p} (below 1 TiB: {}), mapped {mapped:?}; the section opened as a heap (OpenExistingHeapFromFileMapping): {}",
            (b as u64) < TIB,
            hr(&opened)
        ));
        if let (Ok(heap), (true, true)) = (&opened, mapped) {
            lines.push(format!("  {}", describe(heap)));
            let flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let placed = place(&device, heap, 0, &buffer_desc((PAGES * K64) as u64, flags), D3D12_RESOURCE_STATE_COMMON);
            lines.push(format!("  a buffer placed over the whole section: {}", hr(&placed)));
            if let Ok(buffer) = &placed {
                fill(a, SHOWN / 2, 5);
                fill(b.add(SHOWN / 2), SHOWN / 2, 6);
                let got = rig.fetch(buffer, 0, SHOWN);
                // The CPU wrote half through each view; both views and the
                // GPU must show the same bytes.
                let views_agree = std::slice::from_raw_parts(a, SHOWN) == std::slice::from_raw_parts(b, SHOWN);
                f.alias_reads = views_agree && got.as_ref().is_ok_and(|g| same(a, g));
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x33).collect();
                let copied = rig.store(buffer, 3 * K64 as u64, &new).is_ok() && same(a.add(3 * K64), &new) && same(b.add(3 * K64), &new);
                a.cast::<[u32; 3]>().write_unaligned([40, 2, 0]);
                let shaded = rig.shade(buffer.GetGPUVirtualAddress());
                let shader = shaded.is_ok() && b.cast::<[u32; 3]>().read_unaligned()[2] == 42;
                f.alias_writes = copied && shader;
                lines.push(format!(
                    "  the views agree {views_agree}; the GPU reads what the CPU wrote through both {}{}; the CPU reads through both views what the GPU wrote: \
                     by a copy {copied}, by a shader {shader} ({})",
                    f.alias_reads,
                    if !f.alias_reads && views_agree && got.is_ok() {
                        " (THE HEAP IS NOT THE MEMORY: the driver gave memory of its own, vkd3d-proton's silent fallback)"
                    } else {
                        ""
                    },
                    hr(&shaded)
                ));
            }
            for (name, layout) in [("the card's own layout", D3D12_TEXTURE_LAYOUT_UNKNOWN), ("row-major", D3D12_TEXTURE_LAYOUT_ROW_MAJOR)] {
                let t = place(&device, heap, 32 * K64 as u64, &texture_desc(layout, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER), D3D12_RESOURCE_STATE_COPY_DEST);
                let kept = t.as_ref().ok().map(|t| rig.texture_round_trip(t));
                if matches!(kept, Some(Ok(true))) {
                    f.alias_texture = true;
                }
                lines.push(format!("  a texture placed in it, {name}: {}; written and read back {:?}", hr(&t), kept.map(|k| k.map_err(|e| e.code().0))));
            }
            if tiles {
                f.tiles = Some(tile_test(lines, rig, heap, a));
            }
        }
        drop(opened);
        unmap(a);
        unmap(b);
        windows_sys::Win32::Foundation::CloseHandle(handle);
    }
}

/// A reserved buffer of 16 pages whose page `i` is mapped to page `31 - i`
/// of the opened section (`heap`; its first 32 pages show at `a`): the
/// GPU page table the console has, made of host tile mappings.
unsafe fn tile_test(lines: &mut Vec<String>, rig: &mut Rig, heap: &ID3D12Heap, a: *mut u8) -> bool {
    const N: u32 = 16;
    lines.push("tiles: mapping a reserved buffer into a heap opened on a section (a driver may remove the device here)".to_owned());
    // Each line is logged at once, should the next call not return.
    tracing::info!("unified probe: tiles: about to map a reserved buffer into a heap opened on a section");
    let device = rig.device.clone();
    let mut tried = Vec::new();
    for flags in [D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER] {
        let mut reserved: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe { device.CreateReservedResource(&buffer_desc(u64::from(N) * K64 as u64, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved) };
        tried.push(format!("flags {:#x}: {}", flags.0, hr(&made)));
        if let (Ok(()), Some(r)) = (made, reserved) {
            let starts: Vec<u32> = (0..N).map(|i| 31 - i).collect();
            let counts = vec![1u32; N as usize];
            let coordinate = D3D12_TILED_RESOURCE_COORDINATE::default();
            let size = D3D12_TILE_REGION_SIZE { NumTiles: N, ..Default::default() };
            // SAFETY: one region of the buffer's 16 tiles, 16 ranges of one
            // tile each, all inside the heap.
            unsafe {
                rig.queue.UpdateTileMappings(
                    &r,
                    1,
                    Some(&coordinate),
                    Some(&size),
                    heap,
                    N,
                    None,
                    Some(starts.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                )
            };
            let waited = rig.wait();
            tracing::info!("unified probe: tiles: the mapping returned; the device: {}", hr(&waited));
            for i in 0..N as usize {
                fill(unsafe { a.add((31 - i) * K64) }, K64, i as u8);
            }
            let got = rig.fetch(&r, 0, N as usize * K64);
            let ok = got.as_ref().is_ok_and(|g| (0..N as usize).all(|i| same(unsafe { a.add((31 - i) * K64) }, &g[i * K64..(i + 1) * K64])));
            lines.push(format!(
                "tiles: reserved buffer {}; mapped and waited: {}; the GPU reads each page through the mapping: {ok} ({})",
                tried.join(", "),
                hr(&waited),
                hr(&got)
            ));
            return ok;
        }
    }
    lines.push(format!("tiles: no reserved buffer ({})", tried.join(", ")));
    false
}

/// Private memory below 1 TiB, committed whole as `xmem` would commit the
/// title's heaps, opened whole (`OpenExistingHeapFromAddress1`).
unsafe fn private(lines: &mut Vec<String>, f: &mut Found, rig: &mut Rig) {
    const SIZE: usize = 32 * K64;
    let device = rig.device.clone();
    // SAFETY (the whole function): the allocation made here.
    unsafe {
        let base = placeholder(SIZE);
        if base.is_null() {
            return;
        }
        // The placeholder becomes plain committed memory at that address.
        mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        let at = mem::VirtualAlloc(base.cast(), SIZE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE).cast::<u8>();
        if at.is_null() {
            lines.push("private memory: none to try with".to_owned());
            return;
        }
        let opened = device.cast::<ID3D12Device13>().and_then(|d| d.OpenExistingHeapFromAddress1::<ID3D12Heap>(at.cast(), SIZE));
        let flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let placed = opened.as_ref().map_err(Clone::clone).and_then(|h| place(&device, h, 0, &buffer_desc(SIZE as u64, flags), D3D12_RESOURCE_STATE_COMMON));
        if let Ok(buffer) = &placed {
            fill(at, SIZE, 11);
            f.private_reads = rig.fetch(buffer, 0, SIZE).is_ok_and(|g| same(at, &g));
        }
        lines.push(format!(
            "private memory at {at:p} opened whole (OpenExistingHeapFromAddress1): {}; a buffer over it: {}; the GPU reads what the CPU wrote: {}",
            hr(&opened),
            hr(&placed),
            f.private_reads
        ));
        drop(placed);
        drop(opened);
        mem::VirtualFree(at.cast(), 0, mem::MEM_RELEASE);
    }
}

/// How fast the card reads a heap opened on a section (the PC's memory
/// over the bus): what every title buffer read would cost.
unsafe fn section_bandwidth(lines: &mut Vec<String>, f: &mut Found, rig: &mut Rig, bytes: u64) {
    let device = rig.device.clone();
    // SAFETY (the whole function): the section made here.
    unsafe {
        let handle = section(bytes as usize);
        if handle.is_null() {
            return;
        }
        let view = mem::MapViewOfFile(handle, mem::FILE_MAP_WRITE, 0, 0, bytes as usize);
        if !view.Value.is_null() {
            std::ptr::write_bytes(view.Value.cast::<u8>(), 0x11, bytes as usize);
        }
        let opened = open_section(&device, handle);
        let placed = opened
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|h| place(&device, h, 0, &buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER), D3D12_RESOURCE_STATE_COMMON));
        if let Ok(buffer) = &placed {
            f.read_gbps[0] = rig.copy_ms(buffer, bytes).ok().map(|ms| gbps(bytes, ms));
        }
        lines.push(format!("bandwidth: a {} MB section opened: {}, a buffer over it: {}", bytes >> 20, hr(&opened), hr(&placed)));
        drop(placed);
        drop(opened);
        if !view.Value.is_null() {
            mem::UnmapViewOfFile(view);
        }
        windows_sys::Win32::Foundation::CloseHandle(handle);
    }
}

/// One section of `mb` MB, mapped whole below 1 TiB and opened whole, as
/// the title's page pool would be: the time, the PC's memory it pins, and
/// whether a buffer over all of it shows its last page.
unsafe fn big_section(lines: &mut Vec<String>, f: &mut Found, rig: &mut Rig, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) {
    let bytes = mb << 20;
    let device = rig.device.clone();
    let shared = || memory().map_or(0, |m| m[3] >> 20);
    // SAFETY (the whole function): the section and view made here.
    unsafe {
        let handle = section(bytes);
        let hole = placeholder(bytes);
        if handle.is_null() || hole.is_null() || !view_at(handle, hole, 0, bytes) {
            lines.push(format!("big section of {mb} MB: none to try with"));
            if !handle.is_null() {
                windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            return;
        }
        let last = hole.add(bytes - K64);
        fill(last, K64, 13);
        let (set, account, started) = (working_set_mb(), shared(), std::time::Instant::now());
        let opened = open_section(&device, handle);
        let took = started.elapsed().as_secs_f64() * 1000.0;
        let (set_opened, account_opened) = (working_set_mb(), shared());
        let placed = opened
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|h| place(&device, h, 0, &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER), D3D12_RESOURCE_STATE_COMMON));
        let ok = placed.as_ref().ok().map(|b| rig.fetch(b, (bytes - K64) as u64, K64).is_ok_and(|g| same(last, &g)));
        f.big = Some(ok == Some(true));
        lines.push(format!(
            "big section of {mb} MB at {hole:p}: opened {} in {took:.0} ms; working set {set} -> {set_opened} MB; the PC's memory shared with the card {account} -> {account_opened} MB; \
             a buffer over all of it {}; the GPU reads its last page {:?}",
            hr(&opened),
            hr(&placed),
            ok
        ));
        drop(placed);
        drop(opened);
        unmap(hole);
        windows_sys::Win32::Foundation::CloseHandle(handle);
    }
}
