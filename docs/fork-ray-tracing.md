# Ray tracing in this fork: what the title does and what has been built

Status, 2026-11-25: **nothing is lit by ray tracing yet.** Most of the
machinery translates and is unit-tested; it has not yet run end to end in the
game. Everything here is off by default.

## What the title does

The title's indirect light (probes, bounce, reflections) and the
shadows of its lamps and sun are ray traced. Without them, anything the sun
does not hit directly is black: interiors, the shadowed side of a face, rooms
lit only by lamps.

The title does not use a ray-tracing API:

- Traversal is ordinary compute-shader code: a per-thread stack in shared
  memory, and loads through raw 40-bit addresses of title memory.
- The static world's acceleration structures ship pre-built in the title's
  data, in the console GPU's native node format (64-byte nodes: box nodes with
  four children and 16-bit float bounds, 128-byte box nodes with 32-bit float
  bounds, triangle nodes, instance nodes). They are unpacked on the GPU.
- A set of builder programs (from the title's `XboxBvhBuilder.dll`, 52 compute
  programs) builds and refits the dynamic structures on the GPU every frame.
- Only the per-node test is hardware: the instructions
  `image_bvh64_intersect_ray` and `image_bvh_intersect_ray`. One call tests a
  ray against a box node's four children (returning the hit children, nearest
  first) or against a triangle node (returning the intersection's numerator,
  denominator and barycentric numerators).

So mapping this onto the host's own ray tracing (DXR) is not practical: the
layer would have to know the geometry behind structures the title never
describes to it, and the traversal loops live in 64 of the title's own
programs. This fork runs the title's builder and traversal as translated
shaders and emulates the one instruction.

## What this fork adds

1. **The builder programs translate** (49 of 52; all that the title creates in
   the scenes tested). They pass buffer addresses in a spilled parameter table
   rather than in descriptors. Added: the spilled user-data table, root
   unordered-access views (translator and host binding), byte and 16-bit buffer
   accesses, and several control-flow shapes.
2. **Title memory by raw address from shaders.** A directory with one entry per
   64 KiB of title address space leads a shader to where the layer has that
   memory (an opened section of the title's page pool, or a buffer placed
   over title memory); loads, stores, atomic add/sub and float min/max go
   through it. Before a dispatch the host makes ready the memory its root
   views and root constants name. The same directory serves buffer views
   over the page pool (`docs/unified-memory.md`, "Title memory by
   directory"); it lies in the card's memory, in each view heap's memory
   run (`victory_shader::emit::MEMORY_RUN_POOL`).
3. **The ray test, emulated in DXIL.** Four slab tests and a sorting network
   for box nodes; a watertight ray/triangle test for triangle nodes; both forms
   of the instruction. A plain Rust reference lives in
   `crates/victory-shader/src/bvh.rs`; tests run the emitted DXIL on the GPU
   over 311 ray/node cases and compare with the reference bit for bit.
4. **The trace and shading programs translate** (59 of 64 trace programs, and
   the irradiance-cache shading program). This needed lane masks treated as
   plain numbers on some paths, and exact forms for programs written for
   64-lane waves running on a GPU with 32-lane waves.
5. **A safety rule on the host side.** The builder's stages run only while
   every builder pipeline the title has created has a host pipeline; otherwise
   none run (`VICTORY_BVH_BUILDER=1` asks for it). A partly working builder
   wrote garbage bounds into title memory and crashed the title (run 98).
   `VICTORY_BVH_QUARANTINE` is gone (removed with the copy-back paths in
   4e70c14): with unified memory the builder's stores land in title memory in
   place, so there is no copy left to keep them in.
6. **Every ray a miss, behind a switch** (`VICTORY_BVH_MISS=1`, off by
   default). The node test returns no child for a box node and a miss for
   anything else, and reads no node memory. Each traversal then ends right
   after its root node, whatever lies where the structures would be:
   indirect-light and reflection rays take the title's own miss path (sky
   probe), shadow rays are unoccluded. It also stops the title's loops from
   walking up to 65,535 steps a wave through garbage. With it the
   constant ambient light (`ambient.rs`) defaults to off, so that the title's
   own probe resolve shows; `VICTORY_AMBIENT=<level>` still sets one. Do not
   combine it with `VICTORY_BVH_BUILDER=1`.
7. **Device-coherent memory for programs that sync across groups.** The
   builder's bottom-up pass reads child bounds other groups wrote (XCL: child
   pointers come out wrong unless title memory is globally coherent in compute;
   Leonida: device-scope acquire/release atomics). A compute program that writes
   memory and either is one of the builder's (a spilled user-data table) or
   invalidates the GPU's L1 (`buffer_gl1_inv`, the console compiler's
   device-scope acquire) is emitted with: buffer UAVs (root views, heap views,
   title memory by address) `globallycoherent`; the UAV fence in its
   `s_barrier`s also when it writes only by address (`global_*`/`flat_*`), so
   the barrier is device-scope; a `DeviceMemoryBarrier` before each buffer or
   global atomic (release; the console's `s_waitcnt_vscnt` is dropped with the
   waits) and at each `buffer_gl0_inv`/`buffer_gl1_inv` (acquire)
   (`emit/shader.rs` `annotate`, `emit/lower.rs` `device_fence`). Programs that
   only read (the trace programs) keep L1 caching.

## Findings, 2026-11-25

- The title talks to the builder through a DXR-shaped interface
  (`XboxBvhBuilder::GetRaytracingAccelerationStructurePrebuildInfo` with
  `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS`,
  `BuildRaytracingAccelerationStructureBatched2X`,
  `RehydrateRaytracingAccelerationStructureBatched2X` for the static world
  shipped in the `*_bvh4` packs, `CopyRaytracingAccelerationStructure`). So
  the dynamic geometry is described to the layer after all. But the trace
  programs' containers carry no DXIL bitcode, only console code: the
  title's inline ray queries are compiled into a traversal loop (stack in
  LDS, parent-pointer restart, instance nodes read at node + 48..108, the
  root pointer read from the structure's header at + 48). Serving them with
  host DXR needs that loop recognised in all 64 programs and replaced by a
  `RayQuery`, with each program's outputs (hit distance, instance, primitive,
  barycentrics) mapped back to its registers, and the static structures
  decoded from console nodes into host geometry. Weeks of work.
- The 35 builder programs the title creates all translate (49 of 52 in the
  DLL; the 3 that do not are wave64 programs it does not use); none needs
  64-bit image atomics, and none keeps 64-lane waves on a 32-lane host. The
  builder stays off only because of `VICTORY_BVH_BUILDER`'s default.
- The title has non-ray-traced variants of most passes
  (`CS_ScreenProbes_TraceRays_Fallback`/`_Voxels`, `CS_DeferredGather_Fallback`,
  `CS_Primary_Gather_Fallback`, `CS_PPR_RaytraceFallback`,
  `CS_LaserLightShadowRaytracingFallback`, `CS_TraceShadowRays_ScreenSpaceOnly*`),
  and its near sun shadows are rasterised cascades (`cascadeshadows_*`,
  `CS_GenerateShadowMasks_Checkerboard_NoHair`). It picked some fallbacks for
  a moment in run r11 (frame 1526); no setting or capability query that
  selects them was found in the executable.

## Known gaps

- With `VICTORY_BVH_BUILDER=1` the builder stayed off in earlier runs: a few
  of its pipelines are refused by the host (they still ask for 64-lane waves on
  a 32-lane GPU) or do not translate.
- Five shadow-mask trace programs do not translate yet.
- An access to a title page that has no host copy is lost (a read gives zero, a
  store is dropped).
- Not publicly documented, and assumed: the instruction's rounding, the order
  of two boxes entered at the same distance, and the result for a ray exactly
  through an edge.
- Cost on the GPU is unmeasured.

## Where to look

- `crates/victory-shader/src/bvh.rs`: the reference for the node test.
- `crates/victory-shader/src/emit/lower.rs`: the lowering of the two
  instructions and of memory by address (`Lower::memory`).
- `crates/victory-shader/src/resource.rs`: address tracking for descriptors
  and raw addresses.
- `crates/victory-gpu/src/pipeline.rs`, `draw.rs`: the builder rule and the
  host side of root views and title memory.
