# The Vulkan host

Victory turns the title's D3D12.X into **Vulkan directly** (the user's
decision, 2026-11-25). Since 2026-11-26 this is the only GPU back end: the
D3D12 path (D3D12 on vkd3d-proton, with the DXIL translator) is deleted
(section "M4: D3D12 removed"). The console GPU is closer to Vulkan than to
D3D12:

- its shaders use raw 64-bit addresses. With buffer device address, a
  title address becomes `base + offset`, with no 64 KiB page directory in
  every shader (`docs/unified-memory.md` section 12);
- its descriptors are bytes in memory, and so are
  `VK_EXT_descriptor_buffer`'s;
- one translation step less (no DXIL → SPIR-V), and no vkd3d overhead.

This document maps each D3D12.X concept the title uses to Vulkan, records
what Proton's winevulkan allows, and plans the milestones. The ABI between
the translator's SPIR-V and the host is in `docs/vulkan-abi.md` (owned
together with the translator, branch `worktree-agent-a34d12784adf81093`).
The code is `crates/victory-vk`. Its self-test is `victory vkprobe`.

The KISS rule (`docs/kiss.md`) holds unchanged. Each Xbox behaviour turns
into the plain PC API at the moment the title asks for it. Under Vulkan
that API is simply one level lower.

## 1. What the host allows (Proton's winevulkan, Wine 11)

Facts from `~/src/wine-cachyos` (`dlls/winevulkan/make_vulkan`,
`dlls/win32u/vulkan.c`; `VK_HEADER_VERSION` 357):

| Feature | winevulkan | Consequence |
|---|---|---|
| `VK_EXT_external_memory_host` | Exposed and passed through. `win32u_vkAllocateMemory` leaves `VkImportMemoryHostPointerInfoEXT` in the chain. In a 64-bit process the host pointer is the process's own address, so the driver pins the title's pages. This is the driver path vkd3d-proton's `OpenExistingHeapFromAddress1` takes, and the unified-memory probe showed it works on this card (aliases agree, the GPU writes back through every alias). Wine uses the extension itself only for WoW64 mappings. | **Title memory is imported in place.** |
| `VK_EXT_map_memory_placed` | Internal only (`UNEXPOSED_EXTENSIONS`), used for WoW64. | Card memory cannot be mapped at a title address, so title memory cannot live in card memory at its own address. This is the same conclusion as under D3D12. |
| `VK_KHR_external_memory_win32` (opaque Win32 handles) | Only for Wine's own d3dkmt shared resources, i.e. handles exported from Vulkan or D3D allocations. `NtCreateSection` handles are not such resources. | A title section cannot be imported by handle. The importer uses the host pointer of the whole view victory keeps of each section. |
| `VK_EXT_descriptor_buffer` | Exposed. `vkGetDescriptorEXT` is in `PERF_CRITICAL_FUNCTIONS` (no tracing). | Descriptors in memory. |
| `VK_EXT_descriptor_heap` | In the tables (`USE_VK_EXT`), thunked. | A later option (section 3.6). Whether the driver offers it, vkprobe says. |
| Buffer device address | Core 1.2, passed through. | Raw title addresses become `base + offset`. |
| `VK_KHR_win32_surface`, swapchain | Core WSI. winevulkan makes the host surface for the HWND. | Presenting on the title's existing window. |
| `VK_EXT_mesh_shader`, `VK_EXT_extended_dynamic_state3`, `VK_EXT_graphics_pipeline_library`, `VK_EXT_robustness2`, sparse binding | Passed through (not in the unsupported list). | Used as the driver offers them. |

Two rules that differ from vkd3d-proton:

- **The host pointer stays mapped while the import lives.** The spec
  requires it. vkd3d-proton maps a view, imports it and unmaps it again,
  relying on the driver's pinning. Victory imports the whole view it
  already keeps per section (`memory::read`/`write` and DirectStorage
  writes use that view today).
- **There is no silent fallback.** A refused import is a Vulkan error,
  not a heap of the driver's own. The canary check of `unified.rs`
  becomes unnecessary; vkprobe checks the behaviour once.

## 2. Device

One `VkInstance` and `VkDevice` on the first discrete GPU, Vulkan 1.3
(`victory-vk/src/host.rs`). Features are enabled as the driver reports
them, minus a few: robust buffer access, capture/replay and protected
memory. Paths pick by `Caps`. Each extension and feature carries its
purpose and a need:

- **Required:** the swapchain, host-memory import, buffer device address,
  `shaderInt64`, timeline semaphores, synchronization2 and dynamic
  rendering.
- **Wanted:** descriptor buffer, mesh shaders, robustness2, EDS3, GPL,
  mutable descriptors, maintenance5, border colours, depth clip,
  conservative raster, memory budget, and the core features the
  translator uses.
- **Reported only** in this milestone: ray tracing, VRS, conditional
  rendering, memory priority, present wait, device-generated commands,
  descriptor heap, untyped pointers, 64-bit indexing.

What is missing is logged with what it is needed for. Only a missing
required item stops device creation.

## 3. D3D12.X → Vulkan

### 3.1 Queues and fences

| D3D12.X | Vulkan |
|---|---|
| Direct queue(s) | the graphics family's first queue (also presents), a line (timeline semaphore) per title queue |
| Compute queue(s) | the graphics family's second queue, a line per title queue (section "Lines and indirect": same family, no ownership transfers) |
| Copy queue(s) | the graphics family's third queue, a line per title queue; our own uploads and fills go on the host line |
| `ExecuteCommandLists` | `vkQueueSubmit2` of the lists' command buffers, in order, signalling the queue's **timeline semaphore** |
| Title fence (`fence.rs`: a 64-bit value at a GPU address) | A queue `Signal` becomes a `vkCmdFillBuffer`/`vkCmdUpdateBuffer` of the value into title memory (the fence's address in the title window, section 3.4) at the end of the submission, after a full barrier. The CPU sees it in place, since title memory is coherent. The submission's timeline value is recorded beside it for CPU waits and events. |
| Queue `Wait` on a fence another queue signals | a timeline-semaphore wait in the submit (the signal's timeline value is known when the signal was submitted) |
| `SetEventOnCompletion` | one waiter thread on `vkWaitSemaphores` over all queues' timelines. It replaces the relay's landing of flights, except for the CPU steps that remain. |
| Bottom-of-pipe writes, GPU timestamps (`write32`/`write64`) | a fill/update into title memory after a sync2 barrier; timestamps by `vkCmdWriteTimestamp2` plus `vkCmdCopyQueryPoolResults` into title memory. The GPU writes them itself. Nothing waits for landing. |
| `WaitMemoryValue` inside a list | Vulkan has no wait on a memory value. The cut into parts (`list.rs`, `queue.rs` held parts) stays. A part whose value a known signal or write meets becomes a timeline wait (today's `late_met`). Only values nothing in the line meets are held on the CPU, as today. |

### 3.2 Command lists

- The title's threads keep writing list calls down (`list/thin.rs`,
  unchanged). The existing 8 workers translate each stream **straight
  into a `VkCommandBuffer`**. Each worker owns one `VkCommandPool` per
  frame in flight, since pools are externally synchronised. A list is
  recorded by one worker. Splitting a very large list into secondary
  command buffers is an M5 option.
- **State.** The per-list state of `draw.rs` stays. Binding a draw becomes:
  1. the pipeline;
  2. push constants: the user data of `docs/vulkan-abi.md` section 2, i.e.
     root constants and title addresses as they are;
  3. descriptor buffers, bound once per command buffer;
  4. index buffers as the title window buffer plus an offset.

  There are no vertex input bindings (ABI section 6), no per-draw
  descriptor work, and no host copies to bring up to date.
- **Barriers.** Title barriers become sync2 barriers. Title-memory
  buffers have no state, so a title barrier batch is one global memory
  barrier. Host images (render targets, depth, UAV textures, detiled
  copies) get layout transitions: `GENERAL` for anything a shader writes,
  `*_ATTACHMENT_OPTIMAL` and `SHADER_READ_ONLY_OPTIMAL` otherwise.
- **Render passes: dynamic rendering.** `vkCmdBeginRendering` starts at
  the first draw after a target change. It ends at a barrier, copy,
  dispatch, resolve or target change. A clear just before the first draw
  becomes `LOAD_OP_CLEAR`. Other clears become `vkCmdClearAttachments`
  inside a pass, or `vkCmdClearColorImage`/`DepthStencilImage` outside.
- **Copies.**
  - Title buffer copies become `vkCmdCopyBuffer` between offsets of the
    title window.
  - Texture copies become `vkCmdCopyImage` or `vkCmdCopyBufferToImage`.
  - `CopyMemoryToMemoryX` runs on the card from the window, no longer on
    the CPU.
- **Indirect.** `ExecuteIndirect` becomes
  `vkCmdDraw(Indexed)Indirect(Count)`, `vkCmdDispatchIndirect` or
  `vkCmdDrawMeshTasksIndirect(Count)EXT` on the window buffer plus the
  arguments' offset.
  - **Bounding stays.** Records the title's own compute passes write are
    garbage when such a pass is dropped or mistranslated, and garbage
    draws hung the card (Xid 109, `indirect.rs`). Bounding is a compute
    pass reading the records by address and writing bounded copies.
  - **Simpler than today:** its arguments are push constants
    (`src, dst, kind, stride, count`), with no root signature and no
    buffer views. Vulkan's compute state is separate from graphics
    state, so there is nothing to restore but the title's compute
    pipeline and push constants.
  - Windows as in `indirect.rs`, kept per command buffer and ended by any
    dispatch or copy (section "Lines and indirect"); no re-bounding at
    barriers.

### 3.3 Pipelines

**The constraint (measured).** Baking the 2,024 pipelines one run used
took ~7.7 s of NVIDIA compile per pipeline: 4.3 CPU-hours on the
D3D12/vkd3d path, where every combination of shaders and state is a
full compile. Under Vulkan, **each shader stage is compiled once, and a
draw links stages cheaply.** State the title varies is dynamic, so a
state change never costs a compile.

**What is compiled when** (graphics pipeline libraries, the default):

| When | What | Where | Cost |
|---|---|---|---|
| The title creates a PSO (`CreateGraphicsPipelineState`, which carries the stages' code) | Each stage is translated to SPIR-V (`shader_cache.rs`, keyed by content). Each stage not seen before becomes a **library part**: a *pre-rasterisation* part per vertex (or task+mesh) program, and a *fragment shader* part per pixel program. They are queued on the `pipeline/make.rs` workers, background threads below the title's. A stage shared by many PSOs is compiled once. | background | one driver compile per distinct stage, not per PSO |
| The same moment | The *vertex input* part: one for every pipeline, since there are no vertex inputs (ABI section 6) and topology and restart are dynamic. The *fragment output* part: one per attachment format set and sample count (blend is dynamic with EDS3), so a few dozen in all. | background | negligible |
| The first draw with a PSO | **Fast link** of its four parts, with no link-time optimisation, on the recording worker. It is cached per tuple of parts, and later draws find it with one lock-free lookup. | recording worker | ~0.1 ms with `graphicsPipelineLibraryFastLinking` (vkprobe measures it) |
| Right after | The **optimised link** (`LINK_TIME_OPTIMIZATION`, parts made with `RETAIN_LINK_TIME_OPTIMIZATION_INFO`) is queued on the workers at low priority. When it is done it replaces the fast one in the lookup (one atomic store); draws recorded after that use it. | background | a full compile, but nothing waits for it |
| Compute PSO | One compute pipeline per compute program (a compute PSO is one stage). | background | one compile per program |

- **If a stage is not ready at a draw.** The draw raises its parts to the
  front of the workers' queue (draw priority, as `make.rs` does today)
  and is **skipped for that frame**. The frame line counts it. The
  recording worker never waits: one wait would stall the whole list and
  every list behind it.
  - With compiles per distinct stage, and started at PSO creation (the
    title creates its PSOs at load, ahead of use), skips should be rare
    and short.
  - A second run starts warm: `VkPipelineCache`, the driver's disk cache
    and the early-start list of `pipeline/library.rs` all carry over.
- **Dynamic state on everything the title varies:**
  - core 1.3: viewport and scissor with count, cull mode, front face,
    topology, primitive restart, depth test, write and compare, depth bias
    and its enable, stencil test, ops, masks and reference, blend
    constants, rasteriser discard, depth bounds;
  - with EDS3 (`Caps::dynamic_blend`, `dynamic_raster`): blend enables,
    equations and write masks, polygon mode, depth clamp, sample count,
    sample mask, alpha to coverage, and depth clip and conservative mode
    where offered.

  vkprobe's library parts declare exactly these, per part
  (`graphics.rs` `dynamic`).
- **The rest is static and lives in parts with few variants:**
  - attachment formats and sample count (the output part);
  - sample-rate shading (the fragment part);
  - line width (lines are rare).

  A title state the host cannot make dynamic on the driver joins the key
  of the part it belongs to, never the key of a stage compile.
- **One pipeline layout for everything** (the ABI's global sets plus 256
  bytes of push constants), so parts never differ by layout and need no
  `INDEPENDENT_SETS`.
- **Stages must compile without knowing each other.** ABI v0 links pixel
  inputs to vertex outputs by *rewriting* the pixel module's locations
  per pair. That ties a fragment part to one vertex program, so parts
  would multiply by pairs. Host proposal 6 (section 6) fixes each
  semantic's location instead. Pixel inputs then also need
  `graphicsPipelineLibraryIndependentInterpolationDecoration` (vkprobe
  reports it), or the vertex outputs must carry the same interpolation
  decorations.
- **Shader objects** (`VK_EXT_shader_object`) are the alternative model:
  one `VkShaderEXT` per stage, compiled at PSO creation, no link at all,
  every state set per command buffer (about 30 calls at a pipeline
  change, cheap, and mostly already set by the draw state).
  - They take no `VkPipelineCache`. Their binaries
    (`vkGetShaderBinaryDataEXT`, keyed by SPIR-V hash and the driver's
    `shaderBinaryUUID`) would go to victory's own cache file, beside the
    shader cache.
  - NVIDIA implements both natively. vkprobe compiles the same triangle
    both ways, times every part and the links, and draws with each.
- **The choice:**
  1. libraries when the driver offers fast linking (they keep the driver
     cache and get optimised code later);
  2. else shader objects;
  3. else whole pipelines with the same dynamic state, one compile per
     (stages, formats). Still one compile per state-free combination,
     not per state.
- **Caches.**
  - **`VkPipelineCache`** is one blob per `pipelineCacheUUID` in the run's
    cache folder, loaded at start and saved as `pipeline.rs`
    `save_pipeline_cache` saves today, with parts and optimised pipelines
    alike. It replaces `ID3D12PipelineLibrary` (`pipeline/library.rs`).
  - **The driver's disk cache** sits beneath it. Under Proton, the
    launcher should keep NVIDIA's cache on (`__GL_SHADER_DISK_CACHE=1`),
    large (`__GL_SHADER_DISK_CACHE_SIZE`, several GiB), never cleaned
    (`__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`), and at a path per build of
    victory. The launcher is outside the repository (open question 5).
- **Smaller programs compile faster.** The directory lookups that the
  window removes are a large part of today's translated code: without
  the directory, the shaders of run p27 had 16.3M SPIR-V instructions
  instead of 28.0M. So each stage compile should also get cheaper.
- **The shader cache** (`shader_cache.rs`) keeps its key scheme. The
  payload becomes SPIR-V.
- **Internal programs.** The indirect bounder, display-plane compose,
  upscale stand-in, ambient stand-in and twin pass are HLSL compiled at
  run time with FXC today (`host::compile`). They stay HLSL, which the
  repository policy allows (it does not allow `.glsl` or `.spv` files).
  They are compiled offline with `dxc -spirv` or `glslc -x hlsl`, and
  their SPIR-V words are embedded in Rust, as the probe's program is
  (`victory-vk/src/compute.rs`).

### 3.4 Resources and title memory

The unified table of `docs/kiss.md` holds; only the mechanism changes.

**One window over all title memory (the design to try first).**

- **The window.** A single sparse buffer (`SPARSE_BINDING |
  SPARSE_RESIDENCY`, every title usage, device address) spans the title's
  GPU address range. Its pages are bound to imported title memory at the
  offsets of the title's own addresses, so every title address is
  `window + address`.
  - The ABI's memory map has one range: `title_start` 0,
    `device_base` the window.
  - Unbound pages read zeros and drop writes when the driver reports
    `residencyNonResidentStrict`, which is the ABI's `zero`/`sink` rule
    for free.
  - The window can cover title addresses from 0 to the top of what xmem
    hands out. xmem keeps GPU memory below 2^40 today, and can pack it
    lower (e.g. under 64 GiB) so that the window fits the driver's
    `sparseAddressSpaceSize`.
- **Regions** (`XMemVirtualAlloc`, `title_region`) are imported whole by
  host pointer. Their pages are bound at their addresses in one
  `vkQueueBindSparse`.
- **Sections** (`XMemAllocatePhysicalPages`, at most 512 MiB each,
  `title_section`): the whole view victory keeps is imported once. A
  section has no fixed address. Its pages are bound wherever the title
  shows them:
  - `XMemMapPhysicalPages` and its scatter form bind the section's pages
    at the title address;
  - `CopyPageMappingsBatchX` (the pool's GPU view, which has no CPU view
    at all) binds the same way;
  - unmapping binds to nothing.

  `vkQueueBindSparse` is a queue operation, ordered by the timeline
  semaphores. So **a remap lands in queue order, exactly as on the
  console.** The window is the console's GPU page table. One physical
  page shown at several addresses is several binds of one memory offset.
- **What it replaces.** The directory (`memory.rs`), pool tables,
  windows over scattered pages (`gather.rs`), in-place buffers
  (`shadow.rs`), heap opening (`unified.rs`) and the "not in place"
  buffers.
- **The probe's test.** vkprobe's "title window" step binds imported
  section pages out of order, and one page twice, 48 GiB into a 64 GiB
  sparse buffer, then reads them by address. If it passes, this is the
  design.

**Fallback, if the driver refuses sparse binds of imported memory.**

- Each import has the device address the driver chose. The memory map
  holds one range per import (up to 64, sorted; ABI proposal below).
- The pool's GPU view then needs CPU views of its pages so they can be
  imported. That costs one wineserver map per page change, ~17 maps/s
  after workstream 11.
- Otherwise a reduced directory stays for that view alone.

**Device-local copies for streamed-once data** (class 2 of the unified
table).

- **Which data:** what the title streams once and the GPU only reads:
  - DirectStorage destinations in write-combined heaps (protection
    0x2404);
  - static geometry.
- **When the data has arrived**, at the API point that says so
  (DirectStorage completion, a title copy):
  1. its 64 KiB pages are copied into card memory;
  2. their window pages are **re-bound to that card memory**.

  The address does not change, so shaders do not change and read at
  ~199 GB/s instead of the bus's 12.6 GB/s.
- **An announced CPU write** copies a page in again.
- **Unloading or freeing** binds the pages back to the import, and the
  card memory goes. Eviction under the card budget does the same.
- **Pages the GPU writes** are never promoted, so a CPU read after a
  fence always sees them in place. These are classes 3 and 4 of the
  unified table.
- This is the job of the pool page cache (`memory/cache.rs`), without
  slots, twins or compares.

**Images over title memory: no.**

- Host-pointer memory does not take optimally tiled images on this
  stack (vkprobe's "images over title memory" line reports linear and
  optimal).
- The console's tiled layouts are not the card's anyway.

So title textures keep a host `VkImage` in card memory:

- **Detiling** happens at the announced points, with `texture.rs`'s
  logic unchanged.
- **Linear title textures** are copied straight from the window buffer
  (`vkCmdCopyBufferToImage` with the title's row pitch), with no CPU
  work.
- **Tiled ones** are detiled on the CPU into a staging ring, as today.
  The window later makes a GPU detiler trivial: it reads title memory by
  address.
- **Render targets, depth and UAV textures** are card images. A T#'s
  base address stays the title's.
- **Twins stay** where the title reads a target's memory as a buffer, or
  uses a depth texture as a UAV (no storage depth formats on NVIDIA).
- **Levels on demand** (`tiles.rs`) become sparse-residency images.

**Victory's own memory.** A few 64 MiB blocks per memory type with a
simple suballocator. The M1 probe makes a handful of dedicated
allocations. `gpu-allocator` is not added for M1–M2, because title memory
is imported, not allocated, and is the bulk. Reconsider it when host
textures move over (M2): it handles images, dedicated allocations and
defragmentation, and the backend would need all three.

### 3.5 Descriptors

Per `docs/vulkan-abi.md` v0:

- **Buffers need no host descriptor at all.** The host writes real V#s
  into the title's heap slots at `Create*View`. Shaders read the V# and
  reach the bytes by window address.
  - This removes the host views made per slot, the dynamic slots and the
    per-draw re-mirroring (`descriptor/host.rs`, 3,690 lines, almost
    all).
  - It also removes the D3D12 view rules: CBVs on 256 bytes, structured
    offsets on the stride, null views past the end.
- **Images and samplers** are real T#/S# in the title's slots. A slot
  index (T# WORD7, S# border-colour pointer) names a host descriptor. The
  host keeps those descriptors in:
  - **`VK_EXT_descriptor_buffer`** (preferred): one resource descriptor
    buffer of image descriptors (2^20 slots) and one sampler descriptor
    buffer (4,096 slots), both upload-class memory.
    - Making a view writes its slot's bytes (`vkGetDescriptorEXT` plus a
      copy). Nothing happens per draw.
    - The buffers are bound once per command buffer.
    - Static samplers are the set layout's embedded immutable samplers.
  - **the fallback, one descriptor-indexing set**: `UPDATE_AFTER_BIND`,
    partially bound and variable count, written by
    `vkUpdateDescriptorSets` at view creation. The shader side is the
    same.
- **Slots** are allocated per distinct (image, view description) and
  freed when the resource is released (`release.rs`). The title copies
  descriptors bytewise, so a slot cannot be freed when one heap slot is
  overwritten. Deduplication bounds the count by the distinct views.
- **Samplers** are deduplicated by description. NVIDIA's
  `maxSamplerAllocationCount` is about 4,000.
- vkprobe prints the driver's descriptor sizes, offset alignment and
  descriptor address space. On this card they should fit inside the
  console's slots (32-byte T#, 16-byte V#/S#). That would also allow
  host descriptors *in* title memory (the probe's "descriptor heap in
  title memory" step tries it), but ABI v0 does not need it.

### 3.6 Presentation

- **The swapchain** is made on the title's existing window (the one
  `host.rs` makes today) through `VK_KHR_win32_surface`: FIFO by default,
  MAILBOX if the user turns vsync off.
- **The display planes** (`compose.rs`: up to 4 planes, scaled, later ones
  over the first) become one small graphics pass in dynamic rendering
  into the acquired image, from embedded SPIR-V.
- **Frame pacing** (`frame.rs`) is unchanged. `VK_KHR_present_wait` is an
  option for M5.
- **Upscale stand-in** (`upscale.rs`) and **ambient stand-in**
  (`ambient.rs`) are compute passes as today.

### 3.7 What moves and what disappears

**Disappears**, because it exists only for D3D12 or vkd3d:

- root-signature emulation: our root signatures, root views by directory
  and the `with_root_windows` fallback;
- host descriptor heaps mirroring title heaps;
- the 64 KiB directory, pool tables and `MEMORY_RUN_POOL`;
- the pool page cache's slots, twins and compares;
- windows over scattered pages;
- in-place buffers and buffers filled at submission;
- heap opening and the canary;
- the unified probe and the WARP tests;
- the relay's landing of deferred writes, since the GPU writes values
  itself;
- `D3D12SerializeRootSignature` forwarding. The blob is opaque to the
  title, so victory serialises its own.

**Moves** (logic kept, re-targeted):

- indirect bounding;
- rectangle lists: a geometry stage, or CPU expansion
  (`primitive.rs`);
- display planes, the upscale and ambient stand-ins;
- depth and UAV twins;
- detiling and uploads;
- levels on demand (sparse images);
- queries and timestamps (`VkQueryPool`);
- capture (readback);
- the memory ceiling (`VK_EXT_memory_budget`);
- memory-value waits: a CPU hold, but more of them pass as timeline
  waits;
- `CopyPageMappingsBatchX` (sparse binds);
- title fence values at GPU addresses (fills into title memory);
- 64-lane waves (translator, unchanged).

**Modules of `victory-gpu` under Vulkan** (43,650 lines today):

| Verdict | Modules (lines) |
|---|---|
| Dead | `memory.rs` (1,231), `memory/cache.rs` (1,069), `gather.rs` (462), `unified_probe.rs` (962), `warp_tests.rs` (838); most of `descriptor/host.rs` (3,690). About 8,000 lines. |
| Much simpler | `shadow.rs` (711 → an address-to-(window, offset) helper); `unified.rs` (995 → imports and binds); `draw.rs` (3,555: table walks, root emulation and host buffer copies go); `pipeline/host.rs` (1,316: root signatures go); `host.rs` (4,407: the prelude, windows and relay landing go); `resource.rs` (3,941: buffer copies go); `memcap.rs` (519) |
| Moves to Vulkan objects | `list.rs`, `queue.rs` submission, `indirect.rs`, `compose.rs`, `upscale.rs`, `ambient.rs`, `texture.rs` uploads, `texture/ahead.rs`, `tiles.rs`, `frametime.rs`, `capture.rs`, `pipeline/library.rs`, `sampler.rs`, `root.rs` |
| Unchanged | the title-facing objects (`device.rs`, `dxgi.rs`, `object.rs`, `fence.rs`, `pipeline.rs`, `signature.rs`, `query.rs`), `list/thin.rs`, `pipeline/make.rs`, `frame.rs`, `console.rs`, `cores.rs`, `fast.rs`, `primitive.rs`, `quality.rs`, `unified_map.rs`, `watch.rs`, `views.rs`, `release.rs`, `mapstat.rs`, `drvwatch.rs`, `shader_cache.rs` |

Expected end state: `victory-gpu` around 25,000 lines, plus
`victory-vk`. No vkd3d-proton, `d3dcompiler_47` or DXC beside the exe.

## 4. The code (milestone 1)

`crates/victory-vk` is a crate of its own, not a module of `victory-gpu`:

- it shares no code with the D3D12 path it replaces;
- it builds in seconds on its own;
- it keeps `ash` out of the D3D12 path until the switch;
- other agents work inside `victory-gpu` (pipelines, memory), and a new
  crate cannot collide with them.

From M2, `victory-gpu` depends on it and picks the backend at device
creation.

| File | What |
|---|---|
| `host.rs` | instance, device, queues; the extension and feature tables with purpose and need; `Caps`; memory types and budgets |
| `memory.rs` | `Imported` (a title range imported by host pointer, its device base, `device_address(title)`); `Buffer` (victory's own: card, upload, readback) |
| `descriptors.rs` | descriptor-buffer properties, layouts, `vkGetDescriptorEXT` writes; the fallback set |
| `graphics.rs` | graphics pipelines from library parts (dynamic state per part, fast and optimised links) and from shader objects; a 64x64 target drawn and read back |
| `compute.rs` | the probe's SPIR-V program (GLSL in the module notes) and its pipeline with a `VkPipelineCache` |
| `present.rs` | a Win32 window, surface, swapchain, frames cleared by dynamic rendering |
| `probe.rs` | `victory vkprobe` |

`ash` 0.38 loads `vulkan-1.dll` at run time (Proton's winevulkan). There
is no link-time Vulkan dependency.

### `victory vkprobe`

A standalone self-test that touches nothing of a title. Each step prints
one line:

- `[ok]` / `[FAIL]` for required steps;
- `[no]` for a step that tries something the plan may use, whose failure
  fails nothing;
- `[info]` for facts.

The exit code is 1 if a required step failed.

| Step | Checks |
|---|---|
| instance, device | loader version; device, driver, Vulkan version; queue families (sparse flag) |
| limits, memory heap/type | import alignment, buffer and allocation limits, push-constant size, wave sizes; heaps with budgets |
| extensions present / missing | every known extension; missing ones with their purpose (required, wanted, later) |
| present (run first: later steps try things a driver may refuse) | 120 frames cleared on a 960x540 window, FIFO; ms per frame |
| title memory, private region | 64 MiB `VirtualAlloc` at 256 GiB (below 1 TiB) imported; device address; a title address → device address |
| title memory, section | 32 MiB committed section mapped twice; view A imported |
| descriptor buffer (or fallback set) | layout size and offset; every descriptor size; offset alignment; address spaces; whether host descriptors fit console slots |
| compute pipeline | SPIR-V module, pipeline layout (24-byte push constants, one set), pipeline cache, creation time |
| compute, region by address | 1 MiB read by BDA, written back by BDA, copied and summed through the descriptor-buffer storage buffer; timeline-semaphore wait |
| compute, section alias | the card reads through view A's import what the CPU wrote through view B, and writes back what the CPU then reads through view B |
| card copy (streamed once) | 1 MiB copied from the import into card memory, read there by address |
| descriptor heap in title memory (`[no]` allowed) | a descriptor written by the CPU into imported title memory, used as the descriptor buffer |
| title window (`[no]` allowed) | the sparse window of section 3.4 |
| pipeline model (`[info]`) | libraries, fast linking, independent interpolation, shader objects, dynamic blend/raster state |
| pipeline libraries (`[no]` allowed) | a triangle's four parts compiled apart and timed; fast link and optimised link timed; the same pipeline whole timed; both linked pipelines draw the right colour |
| shader objects (`[no]` allowed) | the triangle's two stages as shader objects, timed, binary size; drawn with every state set |
| images over title memory (`[info]`) | linear and optimal RGBA8 importability from host memory |
| pipeline cache | bytes after one pipeline; cache UUID |

Expected output on the RTX 5060 Ti (values illustrative):

```
vkprobe: Vulkan host self-test (docs/vulkan-host.md)
[ok]   instance: loader 1.4.x; VK_KHR_surface, VK_KHR_win32_surface, VK_EXT_debug_utils
[ok]   device: NVIDIA GeForce RTX 5060 Ti (DISCRETE_GPU), NVIDIA 620.73, Vulkan 1.4.x; queues: graphics family 0 (x16, sparse), compute family 2 (x8, sparse), transfer family 1 (x2, sparse)
[info] limits: import alignment 0x1000, max buffer 1024 GiB, ...
[info] memory heap: 0: 7xxx MiB card, budget ... MiB, used ... MiB
[info] missing: VK_EXT_descriptor_heap (later): ...
[ok]   present: 120 frames of 960x540 B8G8R8A8_UNORM, FIFO, 3 images, 16.7 ms/frame
[ok]   title memory, private region: 64 MiB at title 0x4000000000 imported (memory type N); device address 0x...; title 0x4000001234 -> device 0x...
[ok]   title memory, section: 32 MiB section, views at 0x... (imported) and 0x...; device address 0x...
[ok]   descriptor buffer: layout 16 B (binding 0 at 0); descriptor sizes: sampler .., sampled image .., ... storage buffer 16 ...
[info] descriptor sizes against the console's: ... true ...
[ok]   compute pipeline: SPIR-V 347 words, made in .. ms
[ok]   compute, region by address: 262144 words read at device 0x... and written back at 0x...; the CPU sees them in place; .. ms
[ok]   compute, section alias: the card read through view A what the CPU wrote through view B, and the CPU read through view B what the card wrote through view A; .. ms
[ok]   card copy (streamed once): 1 MiB copied from title memory into card memory and read there by address (.. ms)
[ok]   descriptor heap in title memory: ...            (or [no] with the reason)
[ok]   title window (sparse buffer over imports): ...   (or [no] with the reason)
[info] pipeline model: graphics pipeline libraries true, fast linking true, independent interpolation true; shader objects true; blend state dynamic true, raster state dynamic true
[ok]   pipeline libraries: parts compiled in .. (vertex input) + .. (pre-rasterisation) + .. (fragment) + .. (output) ms; fast link 0.0.. ms, optimised link .. ms, the same pipeline whole .. ms; both linked pipelines drew right
[ok]   shader objects: both stages compiled in .. ms (no link), binaries .. bytes; drew right
[info] images over title memory: linear RGBA8 ...; optimal RGBA8 ...
[ok]   pipeline cache: .. bytes after one pipeline; cache UUID ...
vkprobe: 11 of 11 required steps passed
```

## 5. Milestones

The D3D12 path kept working until the user's decision of 2026-11-26 to
drop it; during the migration `VICTORY_GPU=d3d12|vulkan` picked the
backend. Both are gone (M4 below).

| # | Work | Size | Gate |
|---|---|---|---|
| M1 | Foundation and `victory vkprobe` (this) | ~4,000 lines (rustfmt-expanded tables and embedded SPIR-V included); done | vkprobe passes under Proton; the title window, descriptor-heap and pipeline-model answers recorded here |
| M2a | **Queues, fences, title memory and present on Vulkan, no draws** (built: see "M2a as built"). | ~2,100 lines | the title boots and runs to the menus under `VICTORY_GPU=vulkan` with no hang; cleared frames presented (the planes need M2b's draws) |
| M2b | **Menus' draws and dispatches** through the new translator (ABI v0). The workers record `VkCommandBuffer`s. Stages compiled once at PSO creation as library parts (or shader objects, per vkprobe), fast link at the first draw, optimised link in the background, the persisted cache; a skipped-draw count for parts not ready. Dynamic rendering, barriers and layouts. Descriptor buffers with T#/S# slots. Host textures (detile and upload as today), clears, copies, resolves, queries. | ~5,000 lines | menus at ≥30 fps (D3D12: ~26), captures match, no driver error in the 5-minute gate run up to the world's door |
| M3 | **The world.** Bounded indirect, mesh shaders at scale, levels on demand as sparse images, streamed-once promotion by re-binding window pages, depth and UAV twins, the upscale and ambient stand-ins, the BVH builder's programs, the frame-time line with Vulkan timestamps, the memory ceiling from the budget. | ~5,000 lines | the gate run reaches the world with no Xid, card memory ≤ D3D12's, frame time ≤ D3D12's (~95 ms in run p28) |
| M4 | **Deletion** (done 2026-11-26, by decision before parity). The D3D12 path goes, and the switch; vkd3d-proton, `d3dcompiler_47` and DXC are no longer needed beside the exe. | −76,500 lines | build green; gate run by the coordinator |
| M5 | **60 fps work.** Async compute on its queue, big lists split into secondary command buffers, GPU detiling from the window, present wait. | measured | `docs/kiss.md` workstream 10 |

### M2a as built (`VICTORY_GPU=vulkan`)

- **No D3D12 device.** `host::host()` answers `None`, so every D3D12 path
  stands down as it does without a GPU. `crate::vulkan` (victory-gpu) and
  `victory-vk` take over through a few hooks: queue execute, signal,
  wait and present; list waits, writes and resets; fence waits; xmem's
  announcements; CPU reads of the pool's GPU view.
- **Queues and fences.**
  - Each title queue has a line of waits, runs and signals.
  - A list is cut into parts at its memory-value waits. A part whose
    waits are met goes to the graphics queue behind everything before
    it, signalling the next timeline value (`victory-vk/src/timeline.rs`).
    A wait counts as met when memory holds the value, or when a write in
    the air will put it there.
  - A landing thread follows the timeline. At each value it does that
    part's bottom-of-pipe writes, then the fence signals queued behind
    it, then runs every line on.
  - CPU fence waits block on it, for at most 10 s.
  - At a present, waits that are still unmet are passed, as the D3D12
    path does. The count is in the 10 s line.
  - M2a parts carry no draws or dispatches. Parts without writes submit
    nothing.
- **The pool's GPU view** (`victory-vk/src/pool.rs`).
  - It is one sparse buffer over the page reservation that
    `CopyPageMappingsBatchX` maps into. Each title page is bound to the
    card page of its physical page, with one card page per physical
    page.
  - A card page is filled from the section's import (its whole view,
    imported by host pointer at first need) when first shown, and again
    at every announced write that reaches the GPU view's addresses.
    Announced writes are DirectStorage completion, file reads, title
    copies and page mappings.
  - Binds and fills go to the queue in call order.
  - Bottom-of-pipe writes into the view go to the page's home in its
    section, and the card page is filled again.
- **Room** (the eviction policy).
  - The view takes card memory in 64 MiB blocks. Its budget is the card
    heap's budget at creation less 2 GiB, and at least 512 MiB.
  - Past the budget, the page touched least recently is evicted. Touched
    means mapped or written; the tick is the frame, and a page touched in
    the current frame is never chosen. Every title page showing it is
    bound to a zero page.
  - An evicted page comes back, filled again, at its next announced
    write or mapping. Shortcut: until then it reads zeros.
- **Memory map.** It is built from the pool views and the large views
  (≥ 16 MiB) of imported sections. Regions are noted, not imported:
  nothing on the card reads them before M2b.
- **Present.** A swapchain goes on the title's window (or one of ours, at
  the planes' size), mailbox if offered. Each present waits for the
  timeline's last value, then clears the image and presents.
- **Gate for M2a (corrected).** The planes' images are what the title's
  draws produce, and draws run from M2b on. So M2a shows cleared frames,
  not the intro or menus. The gate is that the title boots and runs to
  the menus under `VICTORY_GPU=vulkan` with no hang:
  - fences and memory-value waits keep moving;
  - the `frame presented` lines advance;
  - the 10 s `Vulkan: queues, timeline and title memory` line shows the
    timeline reached, few held items and forced waits, and the pool
    view's binds and fills;
  - no Vulkan error lines.

  The visual gate moves to M2b.
- **Shortcuts.**
  - `CopyPageMappingsBatchX` binds at once, possibly ahead of
    submissions its queue still holds (the D3D12 path does the same).
  - A sparse buffer per page reservation covers the whole reservation.
  - Regions freed while imported are not covered, because nothing
    imports regions yet.

### M2b handoff (2026-11-26)

Branch `worktree-agent-a5b010d38d592dbd0`, at f61b493 when this note was
written. It holds integrate 30cd6a5 and the translator's ABI v1
(`worktree-agent-a34d12784adf81093` at 0614732, merged as 76c4e7d).

**Done.**

- **M2a.** Queues, timeline fences, title memory, the pool's GPU view and
  present (see "M2a as built"). Run clean for 300 s.
- **Step 1: stages compiled once** (`victory-gpu/src/vulkan/stages.rs`).
  - At the title's creation of a catalog pipeline, each new container goes
    to 4 stage workers: `victory_shader::vk::translate`, then a shader
    module, then a part.
  - A part is a pre-rasterisation part (vertex or mesh) or a fragment part
    (pixel), with `RETAIN_LINK_TIME_OPTIMIZATION`, or a compute pipeline.
    All of them use the one pipeline layout.
  - The layout is set 0 images (2^20 sampled, 2^20 storage), set 1
    samplers (4,096), set 2 empty, set 3 the map UBO, and 256 bytes of push
    constants, all as descriptor buffers.
  - Specialisation: constant 0 is false (the map has several ranges).
    Constant 100+n is the slot of immediate sampler n, from the top 256
    slots of set 1 (`IMMEDIATE`, `IMMEDIATE_FIRST`).
  - Stats go into a 10 s line (`Vulkan: stages translated once and
    compiled once`).
  - The map's binding is defined in one place (`MAP_SET`, `map_binding`).
- **Encoders.** `victory-vk/src/gfx10.rs` writes T#, S# and V# with the
  host's words (T# WORD7 slot/integer/storage, S# WORD3 slot), holds the
  GFX10 format table, and maps DXGI formats to Vulkan. It is unit-tested.
- **Map block.** `victory-vk/src/map.rs` writes the ABI v1 block: 64
  sorted ranges, zero and sink addresses. `vulkan.rs` `Memory::rebuild_map`
  keeps `MemoryMap` current: the pool view plus big section views. Regions
  are not imported yet.

**Next, in order.**

1. **(a) Descriptors.** In `descriptor.rs` `record(handle, view)`, under
   `crate::vulkan::on()`, write the encoded descriptor into the slot
   instead of victory's record:
   - `View::Constant` → `gfx10::Buffer::raw`;
   - buffer SRV/UAV and `Placed` buffers → V#: raw, structured (stride)
     or typed (format plus `gfx10::format_of`);
   - texture views → T#, with a slot from a vulkan-side table keyed by
     `View`.

   The SRV/UAV description words are D3D12's: format, dimension, 4×3-bit
   component mapping, then the union. `descriptor/host.rs` already parses
   them, for reference.
   - **Slots** are freed when the resource is released (`descriptor::forget`)
     or its memory is released (`vulkan::title_memory_released`), and only
     reused after the timeline passes the last submission that may read
     them.
   - **Descriptor bytes** go into a set-0 descriptor buffer (upload memory,
     `layout.set_bytes[0]` = 8 MiB) at `binding offset + slot × size`
     (`vkGetDescriptorEXT`).
   - **Samplers:** `View::Sampler` (D3D12_SAMPLER_DESC) and
     `View::ConsoleSampler` → `gfx10::Sampler`, deduplicated into set 1.
     Also write the `IMMEDIATE` samplers into their slots. Their words are
     S# words: decode as in Leonida's `vulkan_sampler`.
   - **Keep the records** victory's other code reads, for RTV/DSV (separate
     non-shader heaps, unchanged).
2. **(b) Card images.**
   - One image per title texture or render target. Reuse
     `resource::texture_layout` and the `texture.rs` detile code to fill
     a staging buffer, then `vkCmdCopyBufferToImage` on the timeline. Do
     it at the announced points `texture.rs` already uses.
   - Linear textures copy straight from the map's buffers.
   - Track layouts: `GENERAL` for written ones; attachment and read-only
     otherwise.
   - Views per T#.
3. **(c) Draw recording.**
   - Under Vulkan, turn `thin` recording on (`list.rs`: `thin` is
     `host().is_some()` today). Replay each stream into a command buffer
     per part (`vulkan.rs` `Part` gets `cmds`; `Engine::pump` already
     submits per part).
   - **Pipelines:** the 4 parts (vertex input; pre-raster from
     `stages`; fragment from `stages`; output per format set) are
     fast-linked, cached per tuple, and optimised-linked in the background.
   - **Push constants:** the user data from `VkShader::user_data`. Root
     constants as given, root descriptors as title addresses, tables and
     heaps as the low 32 bits of their title address.
   - **Bind once per command buffer:** the set 0/1/3 descriptor buffers,
     with a map UBO per map generation.
   - **Index and indirect buffers:** title address → `MemoryMap::range_of`
     → its buffer (pool view sparse buffer, section import) plus offset.
   - **Dynamic rendering** on RTV/DSV records → image views. Also clears,
     sync2 barriers, copies, dispatch and draw, and indirect execution
     bounded as `indirect.rs` does.
   - **Regions** must be imported (`Imported::new`) before shaders read
     them, then added to the map. Commit their pages first, as
     `unified.rs` `place` does.
4. **(d) Plane composition.** In `vulkan::present`, blit or draw each
   plane's image (step b) into the swapchain image. The first plane goes
   as is; later ones premultiplied-alpha over it, which needs a small
   graphics pass, unlike a blit.

**Open issues.**

- **Translator coverage** at 0614732 (the translator's numbers): 1,249 of
  1,527 used programs translate, and 134 of 175 used mesh programs. The
  census of step 1 (ABI 0) listed the failure classes: mesh, f16 high
  half, a16/tfe/lwe, swappc, d16.
- **Immediate samplers** past 256 distinct share the last slot (logged).
- **Unbound pool pages** read zeros (`residencyNonResidentStrict` is true
  on this card). Evicted pages stay zeros until announced again.
- **`CopyPageMappingsBatchX`** binds at once, possibly ahead of
  submissions its queue still holds.
- **Stage workers** compile with no `VkPipelineCache` yet. Load and save
  one per `pipelineCacheUUID` in the run's cache folder.

### M2b as built (2026-11-26, branch `worktree-agent-ad4bc2cc3bbe1cff3`)

Steps (a)–(d) of the handoff above are built and compile; none has run
yet. The translator is merged at b252e39 (ABI 1, structural self-check).

- **Crash at the first part compiles (vk4).** It was a translator bug
  (an invalid f16 module, now fixed). winevulkan's thunks end the process
  with `ExitProcess(3)` after a fault inside the driver, and print nothing
  under Proton's `WINEDEBUG=-all`. What the host keeps from that hunt:
  - `victory-vk/src/limits.rs`: every module is checked against the
    device limits before it compiles: mesh and compute group size, shared
    memory, mesh output memory and components, capabilities. A module
    over a limit is refused with a reason (`over the device: ...` in the
    stage line). The module is read, and its mesh output memory counted,
    by the translator's own `victory_shader::vk::usage` (the count the
    translator keeps its modules under, ABI v4 section 6).
  - `VK_EXT_shader_stencil_export` is now enabled; the translator emits
    it. `VK_KHR_fragment_shader_barycentric` is enabled when present.
  - Diagnostics builds log `Vulkan part: compiling` (flushed) and
    `Vulkan part: compiled` around each part. `VICTORY_VK_STAGES=inline`
    compiles on the title's thread; `VICTORY_VK_STAGES=modules` makes
    modules only. `VICTORY_VK_DUMP=<dir>` writes each module before it
    compiles.
- **(a) Descriptors** (`vulkan/descriptors.rs`).
  - `descriptor::record` writes the console's descriptor into the slot
    under Vulkan:
    - a V# for CBVs and buffer views (raw, structured or typed), and for
      placed buffer views;
    - a T# for texture views (slot in WORD7; INTEGER and STORAGE bits);
    - an S# for samplers (slot in WORD3).
  - Descriptor buffers live in upload memory: set 0 (sampled at binding 0,
    storage at binding 1), set 1 (samplers), and set 3 (one map UBO
    descriptor per generation, 256 blocks kept round).
  - Slots are given per (image, view) and freed when the image goes,
    after the timeline passes. Slot 0 is a null image and a linear
    sampler.
  - Samplers from all three forms (D3D12 description, console
    description, immediate S# words) meet as one D3D12-shaped description.
    They are deduplicated, and immediate samplers are written into the
    top 256 slots before draws.
  - Regions are imported when an API call first names an address in
    them (`vulkan::touch`: heap slots, buffer views, root views, tables,
    heaps, index, vertex and copy addresses), then added to the map.
- **(b) Card images** (`vulkan/images.rs`).
  - Images are keyed by title address and shape, so a target's resource
    and a placed view of its memory are one image. One is made when a
    view, target or copy first names the texture.
  - Each flush makes one submission: new images go into `GENERAL` and are
    cleared, then the textures the card does not write are detiled with
    `victory_formats` layouts into one staging buffer and copied in.
  - Fills happen at creation and again at announced writes
    (`title_memory_written`).
  - Every image stays in `GENERAL`, and a title barrier is a full memory
    barrier.
  - Card memory comes in 256 MiB blocks with first-fit free lists; images
    over 64 MiB are dedicated allocations.
  - Images go at `title_memory_released`, once the timeline passes.
- **(c) Recording** (`vulkan/record.rs`, `list/thin.rs`
  `translate_vulkan`).
  - Lists are `thin` under Vulkan, and `ExecuteCommandLists` joins them
    first.
  - Each stream is replayed into one command buffer per part
    (`Timeline::begin`/`submit_recordings`). The command buffer goes to
    its part at a memory-value wait and at the close. Parts with commands
    submit them; parts with only writes submit a marker, as before.
  - Draws fast-link four parts: the stage parts, a vertex-input part (the
    attributes of fetch-shader programs, ABI §6 amended), and an output
    part (the bound attachments' formats). Links are cached per tuple.
    There is no optimised link yet.
  - Push constants come from `VkShader::user_data` and the list's root
    arguments.
  - Descriptor buffers and the map are bound once per command buffer.
  - Dynamic rendering, plus every dynamic state the parts declare. The
    viewport is flipped (negative height), with front face as D3D.
  - Also: clears (the whole view), barriers, buffer, memory, resource and
    texture copies (texture to texture, buffer to texture, texture to
    buffer), resolves, dispatches and mesh draws.
  - A stage that is not ready is queued at the front and its draw is
    skipped. Skipped draws are counted.
  - `VkPipelineCache`: one file per `pipelineCacheUUID` in the pipeline
    cache folder, saved by `save_pipeline_cache`.
- **(d) Planes** (`vulkan/compose.rs`). Each present clears the image,
  then draws each plane's rectangle into place: the first as it is, later
  ones with premultiplied alpha. The pass uses HLSL compiled offline with
  `glslc -x hlsl` and embedded words; the source is in the module notes.
- **The 10 s line `Vulkan: draws recorded`** gives these counters:
  - draws, dispatches, `skipped_not_ready`, `skipped_no_pipeline`,
    `not_yet` (indirect execution, query resolves, MSAA, clear rects,
    tessellation, geometry, amplification), `command_buffers`, `links`;
  - views, samplers, `views_without_image`;
  - images, fills, `filled_mb`, `images_refused`.

  The queue line also gives `presents_with_planes`.

**Open issues (M2b).**

- **Not run yet.** Expect first-run fixes. Likely places: format or usage
  combinations an image view refuses (logged once each), regions that
  shaders reach without an API call naming them (zeros), map generations
  changing within a command buffer (the recording binds the map at its
  start).
- **`AllocateUploadMemory` blocks** (list method 148) are Rust heap
  memory the card cannot reach. Shaders that read addresses there read
  zeros. Fix: hand out such blocks from an imported upload arena.
- **CPU writes to textures without an API call** (movie frames, glyph
  atlases) are not seen. Only announced writes refill. The D3D12 path
  watched pages for this. Measure first; then either fingerprint the
  sampled textures at present, or find the title's own announce point.
- **Indirect execution** is not recorded yet (counted). Bounding as
  `indirect.rs` does is the next step for the world.
- **No optimised link**: fast links only.
- **MSAA draws** are skipped: the fragment part has static samples 1.
- **Clear rectangles** are ignored (the whole view is cleared). **UAV
  clears** (`ClearUnorderedAccessView*`) and **queries** do nothing under
  Vulkan yet.
- **A list executed twice** without a reset runs its commands once: its
  command buffers are one-time and go with the first execution.
- **CPU reads of the pool's GPU view** after card writes still see the
  section's home, not the card page (M2a shortcut).

### M2b after the first runs (vk4–vk16, 2026-11-26)

The UI renders on Vulkan from vk16 (f9b0165) on. What the runs found, in
order:

- **vk4/vk5:** a translator module crashed the driver. winevulkan ends the
  process silently with `ExitProcess(3)`. Fixed in the translator. The host
  keeps per-part logging, the limits check and `VICTORY_VK_DUMP`.
- **vk7–vk13, black planes:** the planes' images were black because draws
  produced no fragments: their positions collapsed.
  - The pipeline statistics showed vertices running, primitives passing
    clipping, and no fragment shaded.
  - The vertex programs' constants were unreachable. The title's placed
    descriptor heap (38 MB) lies across several mappings: an allocation of
    10 MB and a big section view. Only the first piece was in the memory map.
  - Two bugs:
    - Loose memory was imported only from the touched page forward, and an
      overlapping run was refused.
    - An address inside a big section view counted as mapped even when its
      section was never imported.
  - Now:
    - Loose runs start at their allocation's base, and overlapping runs merge.
    - Views of unimported sections are imported as loose memory.
    - Every MiB of a shader-visible heap is imported at its creation.
    - A command buffer binds a newer map before a draw or dispatch if one
      appeared since it began.
- **vk12/vk13, a crash in report():** ash's `get_query_pool_results` takes
  the element type as the stride. Four queries were written into 32 bytes of
  stack. Fixed. The 10 s lines are written by a thread of ours.
- **Images:** one card memory per title memory. An image of another shape
  over memory that already has an image is bound to the first image's
  memory, without a clear, and is never filled if any alias is card-written.
- **CPU-written textures** (movie frames, glyph atlases): fingerprinted at
  each present (three 16 KiB windows). Recently changed ones are checked
  every frame, others 256 a frame in turn. Changed ones are refilled.
- **Clears** clear only the view's mip and slice, and only the given
  rectangles (vk17).
- **Translations** are kept on disk (`vulkan-stages/`). A warm launch
  translates nothing. The pipeline cache is saved every minute.
- **Diagnostics** (diagnostics builds):
  - `VICTORY_VK_COMPOSE_TEST=1` (magenta under the planes);
  - `VICTORY_VK_DRAW_TEST=1|2|3` (state off / constant magenta / full-screen
    vertices);
  - per-pipeline draw descriptions (push constants, user data, input layout,
    vertex bytes, first table and its V#, map reach);
  - plane texels at captured frames;
  - `gpu_*` statistics;
  - `not_yet_reasons`;
  - stages a draw waited for.
- **The unmap hook** `title_memory_unmapped` drops the views and imports in
  the range, and the images over it.

**Open (from the coordinator's audit, in its order):**
- one timeline per title queue, with bottom-of-pipe values and timestamps
  written by the card;
- the leaks: the Vulkan image on resource release, imports of committed runs
  only;
- per-draw CPU cost: link cache per pipeline and formats, state set only on
  change, pool fills batched;
- one mechanism each: queue.rs's held line, cache.rs, the live memory budget,
  the D3D12-only hooks under Vulkan;
- ExecuteIndirect with one bounding pass;
- the compile backlog (compute parts at ~300 ms average);
- `AllocateUploadMemory` blocks the card cannot reach;
- MSAA draws.

### M2b after vk17–vk19 (2026-11-26, branch `worktree-agent-ab7e5143fb2eb77c0`)

vk19 (b08fedc) matches D3D12 at frames 90, 450 and 960: frame 90 is navy
with "ACCOUNT PICKER", frame 450 plays the Bink intro, frame 960 shows the
art in its slanted panel. The run had 1,281 refills by fingerprint, 0
errors and 0 crash reports. What changed and why:

- **CBVs as the console's** (`descriptors/views.rs`).
  - The V# now has a 16-byte stride, `size / 16` records and the
    32_32_32_32_FLOAT format.
  - The Scaleform batch programs (`VBatchVertex*`) read their matrix and
    colour-transform rows with `buffer_load idxen`, indexed by the V#'s
    stride. With stride 0, every row read row 0. Their quads collapsed
    onto the screen's diagonal: the white line in vk17's frame 90, and the
    missing navy backdrop, text and panel frame.
  - Found by disassembling the cached module (`vulkan-stages/*.vks`, the
    SPIR-V after the header) with `spirv-dis`.
- **Rectangle lists** (topology 17) go to a geometry stage
  (`vulkan/rect.rs`, HLSL in the module notes, compiled offline with
  `glslc`).
  - Each draw program's rectangle part (its vertex stage plus that
    geometry stage) is made at its first rectangle draw:
    `Stage::rect_part`.
  - `VS_Blit`, the per-frame clear of the UI plane, is one of these draws;
    it was skipped as `not_yet topology`.
  - Triangle fans (topology 6) are drawn too. Line loops and quads are
    still `not_yet`, each value logged once.
- **CPU-written textures** (`images/fill.rs`).
  - The fingerprint now reads through the CPU view: `read_title`, so pool
    memory goes through the pool's CPU view. `cpu_readable` used to leave
    out the whole pool, so the Bink Y/U/V planes were never refilled. The
    green frame was YUV(0,0,0) through the Bink shader.
  - Every image that is not an attachment is checked. An image over
    memory the card also writes is refilled when the CPU changed it
    (`cpu_changed`).
  - The baseline fingerprint is taken at each fill.
  - CPU reads of the pool's GPU view (`vulkan::read`) go page by page.
- **Draws wait for compiling stages** (`stages::lookup_waiting`), up to
  2 s a frame (`WAIT_BUDGET_US`), instead of being skipped.
  - Cold cache: `draws_waited_ms` was 12,544 over 840 frames.
  - Warm cache (vk19b): 3,677 over 990 frames, with 303 stages from disk
    and 367 translated.
  - The compile backlog (`part_ms_avg` 205, 2,698 queued) is now the
    first cost of a launch.
- **Per-draw cost** (`record.rs`, `record/link.rs`, `record/state.rs`).
  - A recording finds each draw pipeline once per title pipeline and pass
    formats (`DrawKey` → `Linked`): stage lookups, attributes, parts and
    link.
  - Dynamic state is set again only when its digest changes within a
    command buffer (`state_key`). Push constants are pushed again only
    when they change.
  - vk19b frame split: record 29.8 ms, flush 9.0, views 5.2, submit 2.2.
    That run predates these per-draw changes (5d5f92d); measure again.
- **Resource release** (`resource.rs` drop → `vulkan::resource_released`
  → `images::resource_gone`). A released texture's image goes once the
  card is done with it, and its card memory goes with the last alias.
  Before, a later resource of the same shape at the address got the stale
  image, never filled.
- **Live memory budget** (`vulkan::follow_budget`). Every 60 frames the
  pool's GPU view may take the card's budget less everything else in use,
  less 512 MiB (at least 512 MiB), read through `VK_EXT_memory_budget`.

**The crash at victory-image+0x4e8b4be** (vk16 and vk17; none in vk18,
vk19 or vk19b):

- The function at title+0x4e8b4b0 makes a virtual call (slot 1) on an
  engine object whose vtable pointer is 0: `mov rax,[rcx]; mov
  rax,[rax+8]`.
- The caller (title+0x52479f6) takes the object from a handle table while
  streaming loads.
- No GPU result or query feeds it: queries are unused, `not_yet
  queries=0` (queries are measured since; see below).
- Both crashes came about 0.1 s after a 512 MB section import ("a section
  imported in place"), at about frame 1030.
- Crashes at varying title sites appear in D3D12 runs too, before and
  after the stand-in change (e4e2dd2).
- Suspects, in order:
  1. A title race widened by our stalls. The section import pins 512 MB
     under the memory lock that xmem's remaps also take.
  2. xmem's filler stand-ins. A stand-in shows a snapshot of zeros while
     the physical page may be written through other views, until
     `check_fillers` (every 10 s) and the next mapping change. vk17's
     mapstat line showed "title writes into stand-ins 0" before the
     crash, so this is not shown.
     Later (kiss.md, 2026-11-26): it was. The writes came through views
     of the same pages, not the stand-ins: ~1,348 fresh pages were made
     fillers at every startup, and 39-216 of them were found written.
- Next diagnostic: at a crash, report which xmem kind holds `rcx`'s page
  (view, stand-in, section, region).

**Images climbing in the world load** (vk19b 23,127 and vk20 20,822
images, 6.7 GiB of card memory).
- Images were dropped only for memory the title freed or unmapped, and
  only for keys starting inside the range.
- 9ba7212 drops every image overlapping a range the title frees, unmaps
  or maps again in the pool (`title_pages_aliased`, before the memory
  lock).
- A 10 s `Vulkan: images` line gives:
  - live images, made by view or by resource;
  - dropped by reason: freed, unmapped, remapped, resource;
  - aliases that did not fit;
  - the 256 MiB ranges most images were made in.
- If live images still climb, the line shows whether views of new shapes
  over the same memory are the source (fix: one image per memory, views
  by format and mip, the audit's item).
- Watch `dropped_remapped` against frames: a title that maps the same
  pages again every frame would remake images every frame.

**Card memory (vk21: 6,700 MiB per process, pool 233 MiB, 1,739 images).**
- cb051bf adds the 10 s line `Vulkan: card memory`:
  - `images_mb` (allocated) and `images_bound_mb` (live);
  - `pool_mb`, `buffers_mb`, `other_mb` and `ours_mb`;
  - `heap_used_mb` / `heap_budget_mb` (`VK_EXT_memory_budget`);
  - `driver_mb` = heap use less ours: shader local memory for spills and
    internal heaps;
  - host buffers and imports apart.
- Empty 256 MiB image blocks now go back to the driver.
- If `driver_mb` is large, the spills are the translator's (big per-lane
  arrays).

**The memory map is full** (vk21 `map_ranges=64`, the ABI cap).
- The map took 151 loose regions under 1 MiB at 0xad2b00000.. (64 KiB
  title allocations), one import and one range each. Ranges past 64 were
  dropped silently, and shaders read zeros there.
- cb051bf coalesces neighbours and overlaps that share one device offset,
  and logs and counts what the map still cannot take (`map_refused`).
- The loose imports themselves need joining: import the union of
  neighbouring loose allocations when every page between them is
  committed (as grown loose memory is imported again), so the map stays
  far under 64. In the end, the title window: one range.

**ABI v2, host side (4b53b11, with the translator's 5d4ea8d merged):**
- The semantic registry is kept beside the stage cache as
  `vulkan-stages/semantics-<key>.txt`.
  - It is loaded when the cache folder is first opened, before any
    translation.
  - It is saved, if it grew, before a translation is written.
  - A missing or bad registry drops the translations of its key.
  - `SEMANTICS_VERSION` is part of the key.
- Set 0 binding 2 holds a six-layer 2D-array view of each cube image, at
  the T#'s slot.
- Every T# the host writes has WORD6 = WORD7 ^ 0x7A3C5E91
  (`gfx10::T_CHECK`, in `Image::encode`).
- Slot 0 is the null image on bindings 0, 1 and 2.
- The rectangle geometry stage passes locations 0..30.
- `HOST_ABI` is 2.

**Loose imports joined (cff995e).** A loose region the card reads takes
in its loose neighbours when the pages between are committed read-write
memory (gaps up to 16 MiB). If the 64 KiB blocks at 0xad2b00000 have
reserved gaps between them, they still cannot join, and the map still
fills: `map_refused` tells. The way out then is the title window, one
range. The helpers moved to `vulkan/loose.rs`.

**Open, in order** (the coordinator's audit and its comparison list):

1. **One timeline per title queue.** Today `Timeline` is one semaphore on
   the graphics queue, and every submission waits for the one before.
   Design:
   - one `Timeline` per title queue (graphics → graphics queue, compute →
     the compute queue), each with its own semaphore;
   - title fence Signal/Wait become semaphore signals and waits between
     the timelines;
   - bottom-of-pipe writes are done by the card (`vkCmdUpdateBuffer` into
     the imported region, `buffer_of`) instead of `write_now` when a
     flight lands;
   - memory-value waits stay CPU-side cuts (`Part`).
   The engine (`vulkan.rs` `Engine`) keeps its lines. Only `submit_part`
   picks the queue's timeline, and `flights` become per timeline.
2. **ExecuteIndirect** (`thin/vulkan.rs` → `NotYet::Indirect`).
   - Signatures hold one argument: draw (16), indexed (20), dispatch (12;
     type 105 too) or mesh (12). Their records are already Vulkan's
     command structs.
   - Plan: end rendering. One compute pass of ours copies `max` records,
     bounded as `indirect.rs` bounds them, with the count, into a scratch
     buffer of the command buffer's (a ring freed by timeline value).
     Then a barrier to `DRAW_INDIRECT`, rendering again, and
     `vkCmdDraw*IndirectCount` / `vkCmdDispatchIndirect` from it.
   - The pass needs buffer device addresses. glslc's HLSL has no
     `vk::RawBufferLoad`, so write the pass in SPIR-V assembly (source in
     the module notes, `spirv-as`, words embedded) or with the translator's
     `vk::spirv` builder.
   - Indirect rectangles (tiled lighting) take the rectangle part.
3. Done (branch `worktree-agent-a8169a5f3c8e04704`, below): FSR4 on
   Vulkan as a blit, gated on the upscaler DLL and its version.
4. Done (same branch, below): the missing list slots, measured occlusion
   and timestamp queries, the slot layout found at run time, PresentX
   with its plane count (up to 8). Still open from this item: 2D-array
   views of cube images and the T# tag bits, once the translator sets
   them in `docs/vulkan-abi.md`.
5. The rest of the earlier list: imports of committed runs only, one held
   line, `AllocateUploadMemory` blocks, MSAA draws.
1. Done: **one line per title queue** (section "Lines and indirect"
   below).
2. Done: **ExecuteIndirect** (same section).
3. **FSR4 on Vulkan.** `ffx.rs` / `upscale.rs` are D3D12-only, so the
   Series S world will be black on Vulkan. Make it one small linear scale
   pass. Gate it on the DLL and its API version, not this title's offsets.
4. Missing list slots (#53/54 UAV clears, #61–63 FillMemoryWith*, #81
   DrawIndexedX, #85 predication, #88/89/90 top/bottom-of-pipe writes,
   #93 KickoffX, #94 RSSetDepthBiasX). Real occlusion and timestamp
   queries with resolves into title memory. The slot layout detected at
   run time. PresentX with its plane count (up to 8). 2D-array views of
   cube images and the T# tag bits, once the translator sets them in
   `docs/vulkan-abi.md`.
5. The rest of the earlier list: imports of committed runs only,
   `AllocateUploadMemory` blocks, MSAA draws. (One held line: under
   Vulkan it is `vulkan/engine.rs` only; `queue.rs`'s `Held` is the D3D12
   path's and goes with it.)
6. Done: the clears moved to `record/clear.rs` (record.rs at 916 lines).
7. Card memory outside heapUsage (vk23: nvidia-smi 6.8 GB, heapUsage
   2.9 GB). Probably shader local memory for spills, which follows the
   pipelines bound. `VK_KHR_pipeline_executable_properties` reports bogus
   figures. No cheap way to measure it in the process was found; the
   translator should cut the spills (per-lane arrays).

### M2b: console slots, queries, FSR4, layouts, planes (2026-11-26, branch `worktree-agent-a8169a5f3c8e04704`)

Built (`cargo xwin build --profile diag --features diagnostics`), quality
gate clean; not run yet. Commits 295f7e2, d589b17, f4511be.

- **FSR4 on Vulkan** (`vulkan/record/upscale.rs`, `upscale.rs`,
  `victory-hle/src/modules/ffx.rs`).
  - `ffxDispatchDescUpscale` is read as a `#[repr(C)]` struct in the
    FidelityFX API's `ffx_upscale.h` layout (offset asserts:
    colour 0x18, output 0x138, render size 0x178, flags 0x1ac, size
    0x1b0). The title's offsets are no longer named anywhere.
  - Gate: `amd_fidelityfx_upscaler_xbox.dll` beside the loader, with a
    `VS_FIXEDFILEINFO` file version of 3.1.x to 4.x (the FidelityFX API
    upscalers; this title's is 3.1.5). Anything else is logged as an
    error, and the upscale is left to the title's upscaler.
  - Vulkan: one `vkCmdBlitImage` (linear if the format filters, else
    nearest) from the colour's rendered rectangle to the output's
    upscale rectangle, recorded where the title dispatched it. That
    uses a new stream op, `OWN_VULKAN`
    (`list::record_vulkan(list, closure(&mut Rec, &Backend))`).
- **Console list slots** (older numbering; the newer layout adds 2 from
  59 on). The title-side writers are in `list/thin/console.rs`; Vulkan
  recording in `vulkan/record/console.rs`; the D3D12 path in
  `list/console.rs`, where the D3D12 path only does CPU fills and CPU
  writes, and logs the rest once.
  - #53/#54 ClearUnorderedAccessView Uint/Float: the descriptor at the
    CPU handle is copied at the call. For a texture, `vkCmdClearColorImage`
    of the T#'s base mip and its layers. For a buffer, `vkCmdFillBuffer`
    of the V#'s range. Rectangles: the whole view (logged once).
  - #61-63 FillMemoryWith{32,64,128}BitValueX: `vkCmdFillBuffer` when the
    pattern is one repeated dword. Otherwise `vkCmdUpdateBuffer` of the
    first 64 KiB, then doubling copies.
  - #81 DrawIndexedX: an indexed draw from the given address and format;
    the bound index buffer stays.
  - #85 SetPredicationBufferX: `VK_EXT_conditional_rendering`, now
    `Wanted`, with its feature enabled in `victory-vk`
    (`Caps::conditional_rendering`).
    - The predicate's low dword is copied, when it is set, into a ring of
      ours, then `INVERTED` for operation 1. Predication ends at a cut of
      the list.
    - Without the extension, predicated work always runs (logged once).
  - #88/#89 Write*TopOfPipeX and #90 Write32BitValueBottomOfShaderX:
    done by the card.
    - `vkCmdUpdateBuffer` into imported title memory. Bottom-of-shader
      writes take a barrier first.
    - #89's timestamp flag (4) writes `vkCmdWriteTimestamp2` (TOP_OF_PIPE)
      into a ring of our own and copies it in.
    - Pool memory (card pages the CPU does not see) and the system
      timestamp flag (8) are written on the CPU when the list has run
      (`list_write`).
  - #93 KickoffX: the command buffer so far goes to the list's part (it
    still runs at the list's submission).
  - #94 RSSetDepthBiasX: overrides the pipeline's depth bias in the
    dynamic state (`Rec::console.depth_bias`, part of `state_key`).
- **Queries** (`vulkan/record/query.rs`).
  - Each occlusion or timestamp query heap gets a `VkQueryPool` at its
    first use, keyed by heap address and description.
  - Begin/End reset and begin/end the query outside rendering (precise
    for occlusion if the device has it). A query begun while another is
    active ends that one, which then resolves as visible (XCL's nesting
    rule). So does a query still active at a cut.
  - Timestamps: `vkCmdWriteTimestamp2`, ALL_COMMANDS.
  - Resolves: `vkCmdCopyQueryPoolResults` (64-bit, WAIT) into title memory
    through `buffer_of`, for the queries issued in the run. Occlusion
    queries never issued, or ended early, are filled with a nonzero count.
  - Pipeline-statistics and stream-output queries keep their plausible
    values.
  - `GetTimestampFrequency` is the card's (1e9 / `timestamp_period`)
    under Vulkan.
  - Risk: WAIT stalls the queue if a resolve runs before the list that
    ended its query is submitted.
- **Slot layout at run time** (`layout.rs`).
  - `ID3D12Device11` at creation is now only a hint. Calls decide, after
    XCL `real.rs:6972-7030`:
    - a fence passed to the older queue's slot 16 or 17 means newer; to
      the newer queue's 14 or 15, older;
    - a command signature passed to the older list's slot 61 means newer
      (`ExecuteIndirect`, else `FillMemoryWith32BitValueX`);
    - a copy location of a resource passed to the older list's slot 85
      means newer (`CopyTextureRegionX`, else `SetPredicationBufferX`).
  - Pointers are checked readable (`texture::committed`) before they are
    looked at.
  - Live queues and lists are registered with their older/newer table
    pair and take the decided table at once. The older queue's slot 17
    (GetClockCalibration) answers.
- **PresentX planes** (`queue.rs`). The 0x48-byte struct is
  `D3D12XBOX_PRESENT_PLANE_PARAMETERS`, which XCL reads the same way for
  other titles. Up to 8 planes (was 4) on both paths. A change in the
  plane count is logged. The Vulkan compose already draws later planes
  over earlier ones with premultiplied alpha.

**To run** (the coordinator): the gate run on Series S with
`VICTORY_GPU=vulkan VICTORY_CONSOLE=series-s`, plus the usual autorun
arguments. Look for these lines:
- `FidelityFX upscaler: its upscales run as a linear scale` with
  `version=3.1.5`, then once `Vulkan: the title's upscale (AMD
  FidelityFX) runs as a linear scale from=1280x720 ... to=1920x1080`.
  The world captures should be lit, not black. An `error` line naming the
  upscaler means the gate refused.
- `D3D12.X layouts: newer, as the device was created as ID3D12Device11`,
  then `D3D12.X layouts: decided by a call newer=true why="a fence passed
  to queue slot 16"`. A `contradicts` warning would mean a detection is
  wrong.
- `PresentX: the number of planes changed planes=2`.
- `Vulkan: a console list method done approximately`: each shortcut,
  once.
- `Vulkan device lacks ... conditional_rendering` should not appear on
  the 5060 Ti.
- No new validation errors or driver errors; no stall at a query resolve.
  The `not_yet_reasons` line no longer counts queries.
- Windows-target unit tests build (`cargo xwin test --no-run`), not run:
  `layout::tests`, `list::console::tests`, `vulkan::record::query::tests`,
  `modules::ffx::tests`.
### Lines and indirect (2026-11-26, branch `worktree-agent-a53e5cfa3b4ebdc62`)

**One line per title queue** (`victory-vk/src/timeline.rs`,
`vulkan/engine.rs`, `vulkan/writes.rs`).

- A **line** is a timeline semaphore on one queue of the graphics family.
  The device now has three graphics-family queues (`host::LANES`): title
  direct queues run on the first, compute queues on the second, copy
  queues on the third. Each title queue gets a line of its own at its
  first use ("a title queue runs on a line of its own").
  - Why not the compute-only family: images and imported buffers are
    `EXCLUSIVE`, so another family would need ownership transfers or
    `CONCURRENT` everywhere, and command buffers are recorded before the
    queue is known. Same family: no transfers, any command buffer goes to
    any lane.
- **The host line** (line 0, first queue) takes our own work: fills,
  sparse binds, one-shot copies, the present. Each of its operations waits
  for everything given on every line, so it keeps the order it always had.
- **A title submission** waits for its line's previous value, the host
  line's last value (fills and binds before it), and the semaphore waits
  its queue collected: a fence `Wait` whose `Signal` is in the air on
  another line, or a memory-value wait met by a write in the air on
  another line. A `Signal` after such waits first submits a marker that
  carries them.
- **Tickets**: every operation also takes one count over all lines.
  `last`, `completed` and `wait` speak in tickets ("everything given until
  then"), so image release, staging, descriptor and recorder reuse work as
  with the one timeline.
- **Bottom-of-pipe writes by the card**: at the end of their part, after a
  full barrier, `vkCmdUpdateBuffer` into the imported title memory
  (`Memory::buffer_of`, imports only), GPU timestamps by
  `vkCmdWriteTimestamp2` + `vkCmdCopyQueryPoolResults` (64-bit, so 8-byte
  aligned), then a barrier to `HOST`. The title's
  `GetTimestampFrequency` answers the card's clock under Vulkan
  (`1e9 / timestampPeriod`). The CPU still lands writes into the pool's
  GPU view, into memory not imported yet, unaligned ones and CPU
  timestamps.
- Fence signals are still set by the landing thread, which now waits for
  the first flight of every line (`wait_any`) and lands flights per line.
- Memory-value waits nothing in the air meets stay CPU-side cuts.
- **Busy figure**: every recorded command buffer carries a timestamp at its
  start and its end. The 10 s line "queues, timeline and title memory"
  has `lines="host … ms/frame …% n=…/frame; direct0 …; compute0 …"`: GPU
  time per frame of each line's submissions, its share of the wall clock,
  and submissions per frame. Shares that add up past 100% mean the lines
  overlapped.
- What still serialises: each host-line operation (a fill, a sparse
  bind, the present) waits for every line, and the next title submission
  waits for it. Bink frames refill textures every frame, so there is at
  least one such join a frame. Next step if `lines` shows it matters: let
  fills wait only for the lines that last read their images.

**ExecuteIndirect** (`vulkan/record/indirect.rs`, `vulkan/bound.rs`).

- A signature with one argument (draw, indexed, dispatch or type 105,
  mesh) runs; others count `not_yet indirect`.
- The bounding pass is a compute program built in Rust with the
  translator's `vk::spirv` builder (no file, nothing on the policy
  allowlist; `spirv-val --target-env vulkan1.3` passes). One thread per
  record word; it reads the title's records by device address and writes
  them, bounded as `crate::indirect` bounds them, with the count
  (`min(count, max)`; a word of `u32::MAX` when there is none) into the
  **command buffer's scratch**. The scratch belongs to the recorder and is
  reused when the recorder is (its ticket passed): freed by timeline
  value.
- Then a barrier `COMPUTE_SHADER` → `DRAW_INDIRECT`, and
  `vkCmdDraw(Indexed)Indirect(Count)`, `vkCmdDrawMeshTasksIndirect(Count)EXT`
  (count forms when `drawIndirectCount` is on) or one
  `vkCmdDispatchIndirect` per record.
- Draws bind through `prepare_draw`, so indirect rectangles take the
  rectangle part. The pass runs over the ABI layout's push constants, so
  the title's are pushed again at its next draw or dispatch.
- **Windows**: an execution without a count bounds up to 512 records from
  its first on. Later executions inside a window read it with no pass,
  until a dispatch or a copy is recorded (either may write records) or
  the command buffer ends. Draws that write memory do not end a window
  (the D3D12 path ended windows for shaders writing by address). The
  coming UAV clears (#53/54) and FillMemoryWith* should call
  `Rec::indirect.clear()` too.
- **The cap**: at most `max`, at most what the title's buffer holds, at
  most 65,536 records an execution (256 for dispatches), and every record
  bounded (vertices < 2^23, instances ≤ 2^26 / vertices, groups ≤ 65,535
  each and ≤ 2^21 in all; mesh y and z cut to 1). Unbounded counts hung
  the card before (Xid 109).
- `Vulkan: draws recorded` has `indirect="executions=… passes=…
  window_hits=… skipped=…"`.

**Expected effect on frame time.**

- What may now run at once on the card: a title compute queue's work
  beside the direct queue's (async compute), copy-queue work beside both,
  and work of one queue that another queue's fence wait would have held
  up behind unrelated work. Before, every submission waited for the one
  before on one queue.
- Menus, Bink and loading (where the runs reach today): the frame is
  CPU-bound (record ~21 ms summed over workers, fills ~9 ms, join up to
  22 ms), and the title submits ~6 times a frame. Expect no change beyond
  noise. `lines` will say how much GPU time each queue takes; if `direct0`
  is far below the frame time, the GPU was never the limit.
- The world (D3D12 figures, run of 04:53: GPU busy ~112 ms of a ~1,070 ms
  frame, 32 submissions a frame): the gain is bounded by the smaller of
  the direct and compute lines' busy time a frame, less the joins at
  fills and the present, and is only visible once the CPU side gets under
  the GPU time.
- ExecuteIndirect adds work that was skipped: ~780 executions a frame in
  the world (D3D12 figure), so up to ~1,500 passes a frame there (each
  ends and restarts rendering; D3D12's window hits were about a third).
  Watch `passes` against `executions`.

### The card-memory ceiling (vk27, 2026-11-26)

vk27 was killed by the VRAM watchdog at frame ~1,024, when world loading
began: the process went 3,105 → 6,442 MiB in ~3 s.

- **The likely cause: the pool's live budget.** At frame 840
  `follow_budget` gave the pool's GPU view the card's budget less
  everything else in use less 512 MiB: 6,734 − (2,880 − 256) − 512 =
  3,598 MiB, against 256 MiB in use. World loading then showed pages of
  four new 512 MiB sections in the pool ("a section imported in place" at
  10:46:16–19). Each page shown takes a card page, so the pool could grow
  by ~3.3 GB before eviction started, which matches the jump. Images were
  allocating at the same time, against a budget figure up to 60 frames
  old.
- Ruled out: the 4K plane (3840x2160 RGBA8, 33 MB), MSAA (`not_yet
  msaa=0`; no MSAA draws), the descriptor buffers (host memory,
  `host_buffers_mb=32`), imports (host memory; `imports_mb` was already
  3,857 while the process showed 3,105 MiB on the card).
- **The rule** (`vulkan/budget.rs`):
  - everything of ours on the card stays under 4.5 GiB (`CEILING`), and
    never grows into the last 1 GiB of the card's budget (`limit`: the
    budget less the others' use, driver spills included);
  - the pool's GPU view gets a fixed share: at most 1.5 GiB, less what the
    rest of ours takes past its part, at least 256 MiB. It is checked every
    10 frames. A view over its share gives blocks back at once
    (`PoolView::shrink`: the emptiest blocks' pages are evicted and
    unbound, and the memory is freed once the card is past the unbinds);
  - a new image block (256 MiB, or a dedicated allocation over 64 MiB) that
    would cross the limit by more than 512 MiB is refused, counted in
    `images_refused`, with a warning "card memory for images refused".
- **What tells** (diagnostics builds): "Vulkan: a big card allocation"
  (kind, mb, ours_mb) for every allocation of 64 MiB or more, as it
  happens. The "Vulkan: card memory" line (now with `limit_mb`) is
  printed whenever ours or the heap's use moves by 256 MiB, polled every
  250 ms, besides every 10 s.
- Costs to watch: an evicted or given-back pool page reads zeros until the
  title maps or announces it again (shortcut, as before), so a working set
  over 1.5 GiB shows as `pool_evicted` and missing content. A frame that
  maps more than the share of fresh pages fails with "pages of the pool's
  GPU view were not mapped". Idle images are not evicted: dropping one
  would leave the descriptors that name it empty.

### Card memory the driver takes (vm1–vm7, 2026-11-27, branch `worktree-agent-ac0ae83396b163713`)

vk103 ended with 6,869 MiB on the card for the process (`nvidia-smi`),
of which only 3,294 MiB were allocations of ours. The rest is the
driver's, in two places, and both come from pipelines.

**How it is counted.** On NVIDIA the card heap's budget
(`VK_EXT_memory_budget`) is the heap's use plus what is free on the card
(checked against `nvidia-smi`'s `memory.free` to 1 MiB). So:

- the process on the card = free when the device was made − free now
  (`account::start`, `account::process`; other programs' moves are in it);
- **outside** = that − the heap's use: card memory no Vulkan heap counts;
- **driver** = the heap's use − our allocations.

The 10 s line `Vulkan: card memory` adds up from these:
`process_mb = ours_mb + driver_mb + outside_mb`, and
`ours_mb = images_mb + pool_mb + buffers_mb + scratch_mb + other_mb`.
`heaps=` lists every heap as size/use/budget (`d`: the card's; the other
one is system memory: imports and mapped buffers, `imports_mb`,
`host_buffers_mb`).

**What tells which call takes it.** `VICTORY_CARD_TRACE=1`
(`account::traced`): every driver call that can take card memory (stage
compiles by kind, links, shader modules, images, views, imports, buffers,
sparse buffer and binds, submissions) runs alone and the budget is read
before and after; the sums by kind are the line's `traced=` field, and a
stage compile that moves 2 MiB or more is logged. The run is slow (calls
wait for each other); 100 s are enough, the driver's share is taken while
the title makes its pipelines.

`victory vkprobe` with `VICTORY_VK_PROBE_SPV=<folder>` compiles every
`*.spv` (a compute module) and `*.vks` (a translation of the stage cache)
of the folder as the game does and prints what each compile took. One
process per variant: the driver sizes by the worst pipeline so far and
does not shrink.

**Measured** (vm1 traced; vk103, and vm7s on tag `batch-2026-11-27c` with these changes, at the end of the run, MiB):

| What | vk103 | vm7s | How it is known |
|---|---|---|---|
| images (256 MiB blocks) | 2,560 | 3,072 | ours; vk103 refused 46,218 images at its limit, vm7s 1,461 |
| pool's GPU view | 256 | 384 | ours |
| recordings' scratch | 478 | 46 | ours (vk103: `buffers_mb`) |
| driver, in the heap's use | 1,426 | 1,058 | heap use − ours; grows only in stage compiles, in doubling steps (5, 11, 22, … 348 MiB for compute; 1, 3, 6, … 96 for mesh) |
| outside every heap | 2,149 | 844 | process − heap use; traced at start: compute compiles 1,761, one first-frame step 174, the pool's sparse buffer 32, mesh compiles 16, imports 11; then ~+100 over a run |
| **the process** | **6,869** | **5,405** | `nvidia-smi --query-compute-apps` (6,869 / 5,405) |

Not card memory, measured at zero or near it: imports (9.3 GiB of title
memory imported by host pointer cost 11 MiB), descriptor buffers and the
memory maps (host heap), image and view objects (2 MiB for 573 views),
submissions, sparse binds, links, shader modules. There are two heaps
only: the card's and system memory.

**Outside the heaps: per-thread local memory.**

- The driver keeps each program's own variables (SPIR-V `Private` and
  `Function` storage) on the card for 32,768 threads: bytes × 32,768,
  one buffer per kind of stage, as big as the pipeline that asks most.
  It is never given back, not when the pipeline is destroyed either.
- Whether a variable is ever written does not matter. One compute
  program, CS_RaytraceVBuffer_RayBinning_ShaderBinning_Mixed_WithFar
  (6d7fe7745ab9c6d8), has nine constant tables of its code (47.7 KiB,
  `Private` arrays with initialisers, read through `min(index, last)`):
  1,712 MiB. With the tables gone the same module takes 1 MiB.
- No shape of variable helps (probed, one process each): `NonWritable`
  1,751 MiB, a `Function` variable with the initialiser in the function
  that reads it 1,750, the same stored once at the function's head
  1,739. Tables must be memory the program reads by address, not
  variables: **a translator change** (ABI: a place for a program's
  tables), not made here. (Made 2026-11-27, awaiting a bake: the host's
  tables buffer, `vulkan/tables.rs`, `docs/vulkan-abi.md` section 3.1.)
- Next in line: the BC1/BC3/BC7 compressors (CS_CompressBC7Mip0: 44
  `Function` arrays, 11.4 KiB, 357 MiB), then every program's 512-byte
  table (16 MiB). (The arrays were the picks of `v_movrels_b32`:
  selections on the index's bits since 2026-11-27, no variable. The most
  a used module has is then 652 bytes, the 512-byte table of buffer
  formats and a split mesh program's parts.)
- **The limit** (`victory_vk::limits::thread_bytes`, `THREAD_BYTES`
  16 KiB = 512 MiB on the card): a module over it is refused before it
  compiles, like one over a device limit ("N bytes of variables for each
  thread over 16384"); its draws and dispatches are skipped
  (`skipped_no_pipeline`). Today that is the ray-trace program alone
  (its rays find nothing while the structure builder is off). It comes
  back by itself once its tables are not variables.

**In the heap's use: the pipelines themselves.**

- Each kind of stage has an arena that doubles when it is full and is
  never given back (destroying all pipelines of a probe returned 0 of
  191 MiB). After the title's 3,624 stages it holds ~1.06 GiB: the
  translated programs are large (228 M SPIR-V words).
- The game's background bake compiled the 266 stages earlier bakes
  found slow again in every run, beside the title's own pipelines of
  them, and destroyed them: that only pushed the compute arena one
  doubling further (368 MiB). The background bake now leaves stages
  with an outcome alone (`baking::prepare`); `victory bake` still
  retries them.
- Smaller only with smaller translations or fewer pipelines; the title
  binds 2,371 of the 3,624 stages in the test run, so making pipelines
  at first use would save little.

**Ours.**

- **Scratch** (`timeline.rs`): every command buffer kept the most
  scratch it ever used, doubled (478 MiB with ~230 command buffers in
  flight, and growing). Scratch buffers are now the timeline's, lent to
  a recording and handed back when its command buffer has run; 32 MiB
  are kept spare. 46 MiB.
- **Images**: 1,478 images the card writes hold 2,236 MiB over
  2,016 MiB of title memory, 18,802 filled textures 497 MiB over
  3,409 MiB (texture quality low); ~340 MiB of the blocks is free
  space between images. They are the title's own targets, not copies
  kept past use (the line `where image memory goes`, `sums=`).
- **The seven 3840x2160 targets** (233 MiB) are the title's choice. It
  reads the device type (Series S, logged `device type read`), scales
  the world from 1280x720 to a 1920x1080 plane, and draws its interface
  on a second plane of 3840x2160 with two depth-stencil and three colour
  targets of that size, whatever display mode is listed (1280x720 and
  1920x1080 tried). The host cannot make them smaller without scaling
  every viewport, scissor, copy and texel fetch that names them.
- `ours` over `limit` in vk103 was the same thing seen from the other
  side: the limit is what the card's free memory leaves, and the driver
  took 3.5 GiB after the images were made.
- **Storage usage for every colour image** (integrate 899741d) costs no
  card memory: of 168,918 images given the usage by that rule in run
  vm7s, none asked for more memory or another alignment than its twin
  without it (6,731,188 KiB both ways), and `outside_mb` did not move.

**Before and after on tag `batch-2026-11-27c`** (same translator and
driver cache, 420 s runs): int56 (run vmi56) was killed by the watchdog
at frame ~1,830 with 7,707 MiB on the card (the process 6,885; 45,901
images refused). With these changes (run vm7s) the card peaks at 6,238 MiB
(the process 5,405: 5,391 while loading, 5,405 in the world), nothing is
killed and the run reaches frame 2,310 with 1,461 images refused. The
process's resident memory peaks 0.45 GB higher (17.07 against 16.61 GB at
equal times; imports are equal): not traced.

### vk28: the hang at world load, the pool and images (2026-11-26, branch `worktree-agent-a3b7585e2bcbe1a37`)

vk28 (b4465d1): FSR4 ran, the world began loading, then `NVRM: Xid 109
… CTX SWITCH TIMEOUT` at 10:55:12 UTC. The first world frame (1098,
10:55:08: 1,290 first binds, most new compute stages skipped as not
compiled yet) was the last one recorded.

**What the log says about the hang.**

- Indirect: `executions=0` through frame 1023. No 10 s line covers
  1024–1098, so whether indirect ran in frame 1098 is unknown (the world
  issues ~780 executions a frame on D3D12).
- The pool view went wrong before it: 1,487 "pages of the pool's GPU view
  were not mapped" and 28 "an announced write was not taken" (10:54:41 to
  10:55:03). That error path **left the view's books unlike the card**: a
  call evicted and reassigned pages in its books, then returned at the
  first page it found no room for, without submitting the unbinds, fills
  and binds made so far. From then on title pages showed other pages'
  bytes: garbage, not zeros.
- The D3D12 path rendered the world with the same indirect bounds and the
  same skipping of stages not ready, but its translated shaders had loop
  limits. The Vulkan translator removed its per-loop cap (f02fac3).
- Most likely chain: garbage pool pages (c, made worse by the error path)
  feed a loop with no cap (b) in one of the first world frame's shaders.
  (a) is less likely: the bounding pass rejects random words (≥ 2^23
  vertices, > 2^26 vertex instances, > 65,535 or > 2^21 groups), and zeros
  run nothing. (d) does not give Xid 109.
- The bounding pass (iii): the count is `min(count, records)`, records ≤
  `max` ≤ the buffer's records ≤ 65,536 (256 for dispatches); strides are
  checked (≥ 4 × words, multiple of 4); a zero count runs nothing. No
  zero or garbage count or stride makes a huge count. What is left: no
  budget per execution (65,536 records each just inside the bounds, or 256
  dispatches of 2^21 groups, would still run for seconds).

**Built** (commits fcd2d4a, 333ad9e; `cargo xtask quality` clean).

- **The pool view takes what images leave** (`victory-vk/src/pool.rs`,
  `vulkan/paging.rs`, `vulkan/budget.rs`). It grows a 64 MiB block while
  everything of ours stays under the limit (the 4.5 GiB ceiling, less the
  card budget's last 1 GiB), always up to 256 MiB (`pool::LEAST`). Past
  it, it evicts its least recently touched page (a `BTreeSet` LRU; pages
  of the current frame last). **A map never fails** and never stops half
  way. Images come first: they may cross the limit by 512 MiB, and every
  frame a view over what images leave gives back its coldest blocks
  (`follow_budget`, down to 256 MiB). No fixed pool share any more.
- **Evicted pages come back when a recording names them**
  (`PoolView::ensure`): root views (`touch`), vertex, index and indirect
  buffers, indirect counts, buffer copies (`buffer_over`, `device_of`).
  Their binds go to the host line, which the recording's submission waits
  for. Pages read only through descriptors in tables are not seen named;
  they come back at the title's next mapping or announced write.
- **Texture quality on Vulkan** (`vulkan/images.rs`). The setting was
  decided only when the D3D12 host was made, so Vulkan images were always
  full size. Now the backend decides it from the device-local heap
  (medium under 10 GB), and an image of a texture the title fills
  (2D, one sample, a layout the detiling reads, not an alias) leaves out
  its largest level: a quarter of its memory. Fills detile from the
  console's level; views start that many levels lower; copies map console
  subresources and skip levels left out; T#s keep the console's width,
  height and level count. Shortcut, as on D3D12: `image_get_resinfo`
  answers the image's (halved) size.
- **Hang report** (diagnostics builds): every command buffer keeps its
  last 64 notes (draws, dispatches with sizes, indirect executions with
  kind, max, count address and records address, bounding passes, with the
  shader names at each pipeline change). A title submission the card has
  not finished within 1.5 s is reported once: "Vulkan: a submission has
  not finished (a hang?)" with `line`, `value`, `reached`, `host_reached`,
  `host_given`, `waited_ms` and `trail`.
- **Bisect switch** (diagnostics builds only): `VICTORY_DIAG_NO_INDIRECT=1`
  skips every indirect execution (counted as `skipped`).
- **Where image memory goes** (diagnostics): after the 10 s images line,
  "Vulkan: where image memory goes" with the 8 biggest live images and card
  memory by format.

**Images in vk28.** `images_mb` climbed 2,560 → 4,608 at world load
(10:54:37–44); 21,099 live images, 47,976 made by views, 26,859 dropped as
remapped, `images_refused` 19,480, `views_without_image` 40,198,
`not_fitting_aliases` 1,784 (each takes memory of its own). Most were made
at 0x590000000–0x5bfffffff, inside the pool reservation
(0x540b10000 + 16 GiB): streamed textures in pool pages. Why so much: no
texture quality on Vulkan (fixed above), every view's texture is made
whether a draw samples it or not, and alias misfits. The pool pages
under those textures are also held in the pool view, a second copy. Next
step if memory is still short: leave pool pages under live title-filled
images out of the pool view.

**To run** (vk29): the gate run, `VICTORY_GPU=vulkan
VICTORY_CONSOLE=series-s`, captures every 60 frames, 480 s, on this
branch's diag build. If it hangs again, vk29b with
`VICTORY_DIAG_NO_INDIRECT=1` added. Look for:

- `texture quality medium` at the start;
- no "pages of the pool's GPU view were not mapped" (that error cannot
  happen any more) and no "evicted pages … were not brought back";
- the card-memory line: `images_mb` well under vk28's 4,608, `pool_mb`
  growing into what is left; the 10 s line's `pool_shown_mb` (the title's
  working set in the view) against `pool_card_mb`, and `pool_evicted`,
  `pool_refilled`;
- "Vulkan: where image memory goes": the biggest images and formats;
- on a hang: "a submission has not finished (a hang?)". The first such
  line of the line whose `reached` stopped names the submission; the last
  notes of its `trail` name the shader and whether it was an indirect
  execution. An `indirect` note last points at (a), a plain draw or
  dispatch at (b);
- the host line's share and `n=/frame` in `lines` (the record-time
  refills are host-line operations);
- textures half as sharp as vk28's at medium (expected).

### vk29: not indirect; the loop guard, host side (ABI v3)

vk29 (this branch): the pool held (no unmapped pages, 8,901 refills),
texture quality medium applied, images ~4.35 GB at the 4.6 GB limit,
still Xid 109 at frame ~1,024. vk29b with `VICTORY_DIAG_NO_INDIRECT=1`:
Xid 109 at ~1,260, so not indirect. The hang note never printed (autorun
kills the game within 0.5 s of the Xid).

Host side of ABI v3 (translator ab78299):

- `HOST_ABI = 3`. Push dwords 2–3 hold the guard buffer's device address
  (`vulkan/guard.rs`); dwords 0–1 stay 0.
- **One** 8-byte host-visible guard buffer for all lines, not one per
  line: command buffers are recorded before the queue that runs them is
  known, so the pushed address cannot be a line's. The guard id names the
  program, which is what the report needs.
- After every landing the host reads dword 0. If it is not 0, it is
  swapped to 0 and "Vulkan: a shader loop hit its guard" is logged once
  per guard id, with `count`, `guard_id` and the program's name and hash
  (registered as each stage is made).
- The stage cache stores `guard_id` (magic `vkstage2`; old entries are
  dropped).
- The hang note fires at 1.0 s. Once any line's oldest submission is
  1.0 s old, the oldest unfinished submission of every line still running
  is printed with its trail.

### Stage compile speed and the caches (after vk32/vk33)

- spvtime timed the modules against the ABI v0 layout (two sets, a
  sampler binding in set 0, immutable samplers in set 1, **no set 3**).
  Every translated module reads the memory map from set 3 binding 0, so
  its layout was not the module's (undefined; the compiler may fold the
  map's reads and the address search away). It also left
  robustBufferAccess and robustImageAccess on. `target/spvtime-abi3/spvtime3.c`
  (in this worktree, not committed) creates them as the host does: the v3
  layout, the host's device features, specialisation 0 = false and
  100..131, `DESCRIPTOR_BUFFER` flags, optionally a `VkPipelineCache`
  (`SPVTIME_CACHE`) and several threads (`SPVTIME_THREADS`).
- Pipelines differed between runs: immediate sampler slots (constants
  100 + n) were handed out first come. They are now kept in
  `immediate-samplers.txt` beside the translations.
- The `VkPipelineCache` was saved only at the end of a run, which killed
  runs never reach. It is now saved every 60 s while parts compile.
- Stage workers: half the logical processors (4 to 8), below-normal
  priority, at most 1 M SPIR-V words compiling at once. "Vulkan part:
  compiled" has `cpu_ms` beside `ms`.
### M4: D3D12 removed (2026-11-26)

`VICTORY_GPU` and every `vulkan::on()` branch are gone; the Vulkan side
of each stays. What went, and what was kept for the Vulkan path:

- **victory-gpu** (56,679 → 19,065 lines). Deleted: the D3D12 device and
  host (submission line, relay, presenter, retirement), the draw binding
  (`draw.rs` keeps the vertex and index views), `descriptor/host.rs` and
  the descriptor record registry (slots hold the console's descriptors,
  written by `vulkan::write_descriptor`), `memory.rs` and its card page
  cache (`GpuPage` moved to `vulkan/paging.rs`), `unified*`, `views`,
  `shadow`, `gather`, `indirect`, `watch.rs`'s write watch and guards
  (it keeps the remapping lock), `texture*`, `tiles`, `memcap`,
  `compose` (D3D12 planes), `frametime`, `procmem`, `release`, the warp
  tests, `shader_cache` (DXIL), `pipeline/{catalog,bake,library,make}`
  and `pipeline/host.rs` (its graphics state moved to
  `pipeline/state.rs`), the queue's held line (`vulkan/engine.rs` is the
  one line), the D3D12 FSR pass (`upscale.rs` keeps the FidelityFX layout
  and hands the scale to `vulkan/record/upscale.rs`), the D3D12 capture
  read-back. `host.rs` keeps the window, the failure report and title
  memory reads; the list's base table keeps its lifetime, wait and write
  methods; every recording call goes through `list/thin.rs` into
  `list/thin/vulkan.rs`.
- **Pipelines at creation.** The D3D12 catalog decided which created
  pipelines got their stages made ahead of the draws. Now: a pipeline
  whose stage cache (`vulkan-stages`) already holds a translation of one
  of its containers goes to the stage workers at creation
  (`stage_cache::known`, the folder listed once); others are made at
  their first draw, as before.
- **victory-shader** (58,151 → 20,038). Deleted: `emit/` (the DXIL
  encoder, its lowering and tests), `hlsl/` (and the dxc glue),
  `resource.rs`, `structure.rs`, `bvh.rs`. Kept: `gfx10`, `native.rs`,
  `ir.rs`, `io.rs`, `spill.rs`, `vk/`, `dxbc.rs`/`dxil.rs` (the container),
  `sampler.rs` (immediate samplers, read by `vulkan/descriptors.rs`),
  `translation_fingerprint` (the stage cache key, now in `lib.rs`), and
  `wave.rs` (the whole-wave section simulation from `emit/wave.rs`, unused
  until its port into `vk/`).
- **victory-hle.** xmem's write watch, canary pages and kernel-write
  guards went; every region and section goes to the Vulkan backend.
- **apps/victory.** `victory bake` went; `victory translate` runs the
  SPIR-V translator (`--list`, `--jobs`).
- Still on the host's `d3d12.dll`: the title's root signature
  serialization (`root.rs`), which the title calls itself.

### Write tracking for card copies of title memory (2026-11-27, branch `worktree-agent-ae0e3bf350bf8a526`)

Goal (coordinator): the card never reads a stale copy of title memory;
write tracking instead of polling; no extra RAM or VRAM.

**Where the card holds copies.** (1) The pool's GPU view: one card page per
physical 64 KiB page (`victory-vk/src/pool.rs`, `vulkan/paging.rs`). (2)
Images filled from title memory (`images/fill.rs`). Imports (regions,
sections, loose) are zero-copy and need nothing. Before, both were filled
only at mapping and at announced writes, and images also by fingerprint
polling on a backoff schedule; the census (one pool page in 8) found
1,997 of 3,786 sampled pages changed by the CPU after the card's copy.

**As built** (`victory-gpu/src/track.rs`, `victory-hle/.../xmem/tracked.rs`).

- **State**: one byte per 64 KiB page of title address space, in 4 KiB
  leaves made for each 256 MiB stretch that holds a tracked page:
  protected, written, hot, an image may lie over it, counted this frame,
  and a streak of frames (3 bits). No page is mirrored.
- **Protect, then copy.** Before the card copies title memory, the pages
  are made read-only in every CPU view that shows them (`track::protect`):
  at a page mapping of the pool's GPU view (`title_pages_aliased`) and at
  each image fill (`detile`). `xmem` answers which addresses show a page
  (`tracked::places`: the views of a physical page, the page itself in a
  region), under its page-table lock, so no view comes or goes meanwhile.
- **The fault** (`xmem::on_fault` -> `track::write_fault`): the mark goes
  protected -> written by one compare-and-swap, the page is opened
  (`VirtualProtect`) and then listed. It takes only the list's own small
  lock: victory's own writers (bottom-of-pipe writes, copies) fault there
  too. Listing after opening is what makes the race with a flush safe: a
  page that is open is in the list, or will be.
- **The flush** (`paging::flush_written`, before every `ExecuteCommandLists`
  hands work to the engine and at every present): each listed page is
  resolved to its places; if the card holds a copy (a page of the pool's
  GPU view on the card, or an image filled from it) its CPU views are
  protected again and then the pool pages are copied again in one
  submission (`PoolView::stored`) and the images queued to be filled
  again (`images::written`). If the card holds nothing (evicted, dropped,
  unmapped), tracking of the page ends. So data the CPU stored before it
  submitted a list is on the card before that list runs.
- **Hot pages**: a page that 8 flushes running found written (the title
  writes it all the time) stays open for 32 frames; its pool page is
  copied at every flush. An image over a hot page is not filled at every
  flush (a fill detiles the whole texture): while its page is hot it is
  checked by fingerprint, as untracked memory is. Then the page is
  protected once to see whether the title still writes it. In the world
  no page became hot (the pages written are written about once a frame,
  not in every one of its ~19 flushes).
- **Host calls that write title memory from the kernel** do not fault,
  they fail (Wine's `NtReadFile` returns `STATUS_ACCESS_VIOLATION` for a
  read-only buffer; nothing reaches the vectored handler). Such a call
  holds `track::writing(range)`: tracked pages there are opened and
  listed, and no flush protects a page of the range until the guard is
  dropped. Sites: `files.rs` `ReadFile` and `fread`, `dstorage.rs`
  requests with an address destination, and the driver's pinning of a
  region for an import (`import_regions`). A failed `ReadFile` with `ERROR_NOACCESS` is
  logged ("the host could not write the buffer").
- **Victory's own views** (each section's whole view, used by
  DirectStorage page destinations and by the GPU side for pool pages) are
  never protected: every writer through them announces
  (`pages_written`, `write_now`). This is the one place the build differs
  from the handoff's plan ("every CPU view"): streamed data, the bulk of
  what is written, then costs no fault and no protection change at all.
- **Views that come and go** (`xmem`): an unmapped view's marks go, and
  stores not yet flushed are told through another place that shows the
  page; a new view of a page shown elsewhere is listed (it is open). The
  title's own `VirtualProtect`, a commit at an address and a free let go
  of the pages first (`tracked::title_protects`); the next flush protects
  again what the card still has a copy of.
- **Evicted pool pages** are not unprotected at eviction (no host call
  for nothing): the first store into one faults once, the flush finds no
  copy and tracking ends. A page brought back under the memory lock
  (`ensure`, an announced write) is listed as copied, not written
  (`track::copied`): the next flush protects it and copies it once more,
  and leaves the images over it alone.
- **Which images a written page belongs to**: the registry keeps the
  images the title fills by each 64 KiB page they lie over
  (`Registry::over`, a few pointers a page), so a write (found or
  announced) names its images without a walk over every image that could
  reach the address (that walk was most of the flush's time in the first
  run: the widest image is 39 MiB).
- **Other code that looked at protections**: loose imports take a
  read-only page of ours as usable (`loose::usable`), and the console
  instruction emulator writes read-only title memory (the write faults
  inside its handler into ours).

**Deleted**: `vulkan/stale.rs` (the sampling census) and the fingerprint
taken at every fill of a tracked image. Fingerprint polling (`refresh`)
stays only for images whose memory cannot be tracked: memory `xmem` does
not map, and textures the card writes too (their title memory is not
their content unless the CPU stores into it; a page-sized store elsewhere
must not overwrite the card's content).

**The 10 s line** (`Vulkan: write tracking`): faults, faults/frame, pool
pages copied again/frame, pages under images written/frame, hot pages,
flushes/frame, flush ms/frame, fault ms/frame, and what one fault and one
protection change cost on this host (measured once on a page of our own).

**Measured** (on the 8 GB card under Proton):

- Runs wt2 and wt3 (the gate run, 420 s, Series S profile): frames 1,754
  and 1,780, no title crash, no file-read failure, no GPU fault, no
  `ERROR_NOACCESS`.
- In the world (wt3, frames ~1,500 to 1,780, ~19 flushes a frame):
  **130 to 160 faults a frame**, 92% of them on pages under images; **25
  to 105 pool pages copied again a frame**; 115 to 145 written pages a
  frame had images over them (filled again); no page hot.
- **Cost a frame**: the flushes 2.5 to 4.5 ms (about 3.5), the faults 0.3
  to 0.8 ms (2.2 to 5.7 us each, measured in the world with each line; one
  protection change 0.8 to 2.7 us; 14 us and 5.8 us in wt1 while another
  job loaded the machine), and about 3 ms more in the image fills the
  written pages cause (the frame split's `flush_ms` 9.3 against 6.3).
  About 7 ms of a ~600 ms frame; the ~3 ms target is not met. Wine's
  exception dispatch is not what costs: the flushes are (a host-line
  submission for the pool copies and one for the image fills at most
  flushes).
- What is written: the images filled most often are 35 to 260 texel wide
  one-row `R8G8B8A8_SRGB` textures (curves) at unaligned addresses, 450 to
  540 fills each in the run: small textures inside pages the title writes
  every frame.
- Polling left: 2,050 of 19,972 live images are checked by fingerprint
  (textures the card writes too), 17,119 are tracked; 2,149 MB
  fingerprinted in the run against 14,981 MB, 18 fills found by
  fingerprint against 5,731.
- RAM peak 15.2 GB (wt3) and 16.2 GB (wt2) against 16.0 GB; card memory
  peak 6.56 GB and 6.43 GB against 6.6 GB. The marks are 4 KiB for each
  256 MiB that holds a tracked page; the images' page index is a few
  pointers for each page under a filled image (~2 MiB).
- The captures: the text they show (the account prompt, the "NINE MONTHS
  AGO" card) is whole, as it already was in vk80; the world frames are the
  same flat grey as before, so its HUD text is not judged.
- wt1 (the first build) crashed at frame ~1,370 on a store into a page
  read-only with no protected mark: a flush protected an already
  protected second view of a page (mark, then protection) while a store
  into it was in the handler (mark, then opening). Since then a flush
  leaves a protected view alone, and the handler opens any page its mark
  says is open. wt1 also filled the images over every page the pool
  brought back (105 to 280 ms a frame of fills): those are listed as
  copied, not written, now.

**Open.**

- The cost: (1) one submission for a flush's pool copies and image fills
  instead of two; (2) a small image under a written page filled again
  only if its bytes changed (its fingerprint is nearly free; most of the
  130 pages a frame are neighbours' writes); either should bring the
  frame's cost under 3 ms.

- Textures the card also writes (unordered-access textures the CPU fills
  now and then) are still polled; tracking them needs the fingerprint as a
  filter on the store, not a schedule.
- A store by the *card* into title memory under an image filled from it is
  not seen (no CPU fault); none is known.
- Above 2^40 (titles that ask `XMemVirtualAlloc` for more address bits)
  nothing is tracked; images there are polled.
- An overlapped `ReadFile` that completes after the call returns would
  write a page protected again (Wine completes file reads in the call).

## 6. Proposals for `docs/vulkan-abi.md` (from the host)

1. **The memory map holds up to 64 ranges, sorted by `title_start`; the
   translator does a binary search.** With the title window the host
   always writes `count = 1` (`title_start` 0, `device_base` the window),
   so the search is one compare. The fallback (one import per
   region/section) needs more than 4 ranges: Series S has 8 pool sections
   and tens of regions.
2. **The image views stay set 0, but samplers move to a set of their own**
   (set 1 binding 0: heap samplers, the slot in S# WORD3). Immutable
   samplers move to set 2 (bindings 0/1 as v0's set 1). Under
   `VK_EXT_descriptor_buffer` a set lives in one buffer, and samplers
   need a sampler descriptor buffer. Immutable samplers become embedded
   samplers (`DESCRIPTOR_BUFFER_EMBEDDED_IMMUTABLE_SAMPLERS`), which need
   a set without other descriptors. The descriptor-indexing fallback is
   indifferent.
3. **Fixed array sizes for set 0:** 2^20 sampled and 2^20 storage slots
   (SLOT stays 24 bits in the T#, but the host never hands out ≥ 2^20);
   4,096 heap samplers. Descriptor buffers need every binding's size at
   layout creation, and variable counts only on the last binding.
4. **`zero` and `sink`** stay in the map (out-of-bounds V# accesses).
   With the window and `residencyNonResidentStrict`, addresses in no
   mapped page already read zeros. The translator need not branch on
   "in no range" when `count == 1` and the range spans the window.
5. **Descriptor heaps below 4 GiB** (v0 "the code supplies the high
   dword 0"): the host checks at heap creation that the title's heap
   lies below 4 GiB, and logs if not. xmem's 32-bit requests already land
   there.
6. **Fixed locations per semantic; no pixel-module rewrite per pair.**
   v0's `link(vertex, pixel)` rewrites the pixel module's `Location`s
   for each vertex program it meets. That makes a fragment library part
   (or shader object) depend on the vertex program, so parts multiply by
   pairs, which is exactly the per-combination compile this design
   avoids (section 3.3). Proposal: the translator gives each semantic
   (name and index) a fixed location from one table for the whole pack,
   in both the vertex outputs and the pixel inputs. The table can be
   found offline from the pack: the count of distinct semantics must be
   at most 32. A pixel input that no vertex output writes reads an
   undefined value instead of 0. If that matters for a title shader, the
   translator can make the pixel program read 0 for semantics that are
   rarely written. Pixel inputs keep their interpolation decorations.
   Vertex outputs carry none, which relies on the driver's independent
   interpolation (vkprobe reports it).
7. **The map's `device_base` is read per distinct base** (v0 section 3):
   agreed. The host keeps the map in card memory and never rewrites a
   live one. With the window it never changes during a run.

## 7. Open questions

1. **The title window.** If vkprobe's step passes, it is the design for
   all title memory (one range, remaps as sparse binds, promotion by
   re-binding). If the driver refuses sparse binds of imported memory,
   the fallback is the multi-range map, plus either CPU views of the
   pool's GPU view (wineserver cost) or a reduced directory for that view
   only. Which fallback does the user prefer?
2. **Descriptor buffer or descriptor heap.** `VK_EXT_descriptor_heap`
   (Wine 11 carries it) is the newer, D3D12-shaped API. Default:
   descriptor buffer, which the driver certainly has. Revisit if vkprobe
   reports the heap and descriptor buffers misbehave.
3. **Internal GPU programs** stay HLSL, compiled offline to SPIR-V words
   embedded in Rust (the policy allows `.hlsl`, not `.glsl` or `.spv`).
   Is a build step acceptable instead (dxc in `build.rs`)? Proposal: no,
   embedded words plus an xtask to regenerate them.
4. **The switch** `VICTORY_GPU`: gone with the D3D12 path (2026-11-26).
5. **The driver's disk cache under Proton.** The `victory-proton`
   launcher (outside the repository) should set
   `__GL_SHADER_DISK_CACHE=1`, `__GL_SHADER_DISK_CACHE_SIZE` (several
   GiB) and `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`. May it?
6. **Native Windows** (no Proton) is not covered. Host-pointer imports of
   section views should work on Windows' NVIDIA driver too, but nobody
   has tested it.
