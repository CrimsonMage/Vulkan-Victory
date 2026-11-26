# Vulkan shader ABI (translator ↔ Vulkan host)

Status: **v3** (2026-11-26: the loop guard buffer in push dwords 2–3; v2
2026-11-26: semantic registry, six-layer 2D-array views of cubes on set 0
binding 2, T# check word; v1 2026-11-26, v0 2026-11-25; v1
amended the same day: immediate samplers in set 1 by specialization
constant, change 8). v1 takes the host's
proposals (`docs/vulkan-host.md` section 6). Owner of the translator side:
`crates/victory-shader/src/vk/`. Owner of the host side: the Vulkan host
(branch `worktree-agent-a5b010d38d592dbd0`). Change this file first, then
both sides; every change bumps `ABI_VERSION` (below) and the shader cache
magic.

The translator turns a console program (the RDNA2 code the exe hands to
D3D12.X at pipeline creation) into one SPIR-V module per stage. The
module reads what the console program reads, in the way the console
reads it:

- **User data** (the user SGPRs the console driver loads per draw or
  dispatch) from push constants.
- **Title memory** by title GPU address, through buffer device addresses
  (PhysicalStorageBuffer pointers) over the host's imports of title
  memory. No page directory.
- **Descriptors** (V#, T#, S#) from title memory, where the title or the
  host's `Create*View` put them, exactly as the console's scalar loads read
  them. A V# is dereferenced as an address; a T# or S# carries the index of
  a Vulkan descriptor the host made for it.

Nothing is traced, guessed or looked up per access beyond that.

`ABI_VERSION = 3`

## 1. Vulkan requirements

Vulkan 1.3 with:

- `bufferDeviceAddress`, `scalarBlockLayout`, `shaderInt64`, `shaderInt16`,
  `shaderFloat16`, `storageBuffer16BitAccess`/`8BitAccess` (byte and short
  buffer access), `vulkanMemoryModel` (not required in v0).
- Descriptor indexing: `runtimeDescriptorArray`,
  `shaderSampledImageArrayNonUniformIndexing`,
  `shaderStorageImageArrayNonUniformIndexing`, `descriptorBindingPartiallyBound`,
  `descriptorBindingUpdateAfterBind` for sampled images, storage images and
  samplers (`VK_EXT_descriptor_buffer` works the same from the shader side).
- `shaderStorageImageReadWithoutFormat`, `shaderStorageImageWriteWithoutFormat`.
- Subgroups: basic, vote, ballot, shuffle, arithmetic, quad (size 32 on
  NVIDIA; the translator assumes `subgroupSize` = 32 and full subgroups,
  `VK_EXT_subgroup_size_control` `requiredSubgroupSize` 32 if available).
- `VK_EXT_mesh_shader` for mesh programs, `VK_KHR_fragment_shader_barycentric`
  (`fragmentShaderBarycentric`) for pixel programs that read an
  attribute per vertex (`v_interp_mov` of P10/P20, interpolation with
  computed weights) or barycentrics as numbers: such an attribute is a
  `PerVertexKHR` input at its semantic's location and every read of it
  weighs its three vertices with `BaryCoordKHR`/`BaryCoordNoPerspKHR`
  (P0 + i (P1 - P0) + j (P2 - P0)); and `shaderDemoteToHelperInvocation`.

SPIR-V 1.6 (Vulkan 1.3), `PhysicalStorageBuffer64` addressing, GLSL450
memory model. Every module passes `spirv-val --target-env vulkan1.3
--scalar-block-layout` offline.

## 2. Push constants (256 bytes, one block for all stages)

| Dwords | Content |
|---|---|
| 0–1 | reserved, 0 (v0 had the memory map's address here; v1 reads the map from a uniform buffer, section 3) |
| 2–3 | the device address of the **guard buffer** (low dword at 2), 0 for none (section 2.1) |
| 4–31 | user data of the first stage: compute, vertex or mesh program (user-data dword k at push dword 4 + k; k < 28) |
| 32–63 | user data of the pixel program (user-data dword k at push dword 32 + k; k < 32) |

A compute program may use dwords 4–63 (k < 60).

### 2.1 The loop guard

Every loop counts its iterations and gives up after 2^18 of them (compute)
or 2^16 (vertex, mesh, pixel): it leaves and the invocation returns, so a
loop that never exits (a mistranslation, or memory that reads zeros) cannot
hang the GPU. Before returning, when push dwords 2–3 hold an address, the
invocation adds 1 to **dword 0** of the guard buffer (`OpAtomicIAdd`,
device scope) and stores the program's **guard id** in **dword 1**
(`VkShader::guard_id`, FNV-1a of the console container). The host gives
one 8-byte host-visible buffer (one per queue is enough), with
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`, zeroed; after a submission it
reads it, and when dword 0 is not 0 it logs the count and the pipeline
whose stage has that guard id, then zeroes it. A program that hits the
guard has a loop to fix, not a limit to raise.

User-data dword k is what the console puts in user SGPR k of the stage
(SGPR k + 8 for mesh programs, which run as a merged stage). The
translated module lists, per stage, which root parameter fills each dword
it reads (`VkShader::user_data`: dword k ← root parameter `slot`, dword
`d` of it, kind):

- 32-bit constants: the constant.
- Root CBV/SRV/UAV (`Descriptor`): the 64-bit **title** GPU address the
  title passed, low dword at k, high at k + 1.
- Descriptor table (`Table`): the low 32 bits of the title address of the
  table's first descriptor (the GPU handle the title passed). The console
  heaps lie below 4 GiB; the code supplies the high dword (0).
- `ResourceHeap` / `SamplerHeap` (direct heap indexing): the low 32 bits of
  the title address of the bound heap's first descriptor.
- `ImmediateSampler`: nothing (section 5.3).

Measured over the pack (25,985 containers): the highest user-data dword
any program reads is 20 (vertex, pixel), 17 (mesh), 14 (compute), so the
per-stage windows above hold every layout. A compute program with a
spilled user-data table (the console's BVH builder) reads entries at
user-data index 16 and above from push dwords 4 + k as well (the translator
rewrites the table loads, `spill.rs`).

## 3. Title memory: the memory map

Title GPU addresses (the console's virtual addresses: what the title puts
in root descriptors, V#s, tables, vertex/index views) are translated by
one subtract and add over the host's imports of title memory:

```
struct MemoryMap {            // uniform buffer, set 3 binding 0 (std140; offsets as written)
    u64 zero;                 // device address of >= 256 bytes of zeros (never written)
    u64 sink;                 // device address of >= 256 writable bytes nobody reads (dropped stores)
    u32 count;                // ranges in use, 1..=64
    u32 pad[3];
    struct {
        u64 title_start;      // first title address of the range
        u64 title_size;       // bytes
        u64 device_base;      // device address that title_start maps to
        u64 pad;
    } range[64];              // sorted by title_start, disjoint
};
```

`device = range.device_base + (title - range.title_start)` for the range
with `title - title_start < title_size` (unsigned). **Specialization
constant 0** (bool, default true) says the map has one range (the host's
title window: `title_start` 0, `device_base` the window): the translator
then does exactly that with `range[0]`, one compare and add. A host that
writes several ranges sets it false at pipeline creation; the module then
does a binary search over the sorted ranges (6 steps). (`count` is read
only then.) An address in no range reads
zeros from `zero` and writes into `sink`; so does any access that is out
of bounds of its V# (section 4). The map must stay valid and unchanged
while any submitted work can read it; a new map is a new block.

The map is a **uniform buffer at set 3, binding 0** (a set of its own, so
the host binds it once per submission), read with uniform loads.
Measured (NVIDIA 620.73, b7a7fb5ba5934a89): the same module reading the
map through a pointer from push constants compiled ~35x slower (8.8 s
against 0.26 s).

Translation happens once per distinct base (a V#'s base, a table
address, a root address): every access then only adds its offset.

## 4. Buffers: V# (4 dwords)

The host writes buffer views as real RDNA2 V#s (`Create*View` for
buffers, root views built by the code itself):

```
WORD0  BASE_ADDRESS[31:0]          title GPU address
WORD1  BASE_ADDRESS_HI[15:0], STRIDE[29:16]
WORD2  NUM_RECORDS                 bytes (stride 0) or records (stride > 0)
WORD3  DST_SEL_X/Y/Z/W[11:0], FORMAT[18:12], OOB_SELECT[29:28], TYPE[31:30] = 0
```

(field positions: Mesa `gfx10-rsrc.json` SQ_BUF_RSRC_WORD0..3, as in
Leonida's `gfx10_descriptors.h`). Raw view: stride 0, NUM_RECORDS =
bytes. Structured: stride, NUM_RECORDS = records. Typed (formatted)
views: FORMAT = the GFX10 buffer format code, stride = element size,
NUM_RECORDS = elements, DST_SEL the format's channel order.

The shader computes the console's address
`base + (stride * index) + voffset + soffset + inst_offset` and its bounds
check from the V# at run time: with stride 0, the access is in bounds when
`offset + size <= NUM_RECORDS`; with stride > 0, when
`index < NUM_RECORDS` (and the offset < stride for swizzled forms, not
used by the title). An out-of-bounds access reads zeros / drops the write
(whole access, not per dword). Typed loads and stores decode FORMAT in
the shader.

## 5. Images and samplers

### 5.1 Descriptor sets

| Set | Binding | Type | Content |
|---|---|---|---|
| 0 | 0 | `SAMPLED_IMAGE[1 << 20]` | image views, indexed by a T#'s slot |
| 0 | 1 | `STORAGE_IMAGE[1 << 20]` | storage views of UAV textures, indexed by the same slot |
| 0 | 2 | `SAMPLED_IMAGE[1 << 20]` | cube and cube-array images as `VK_IMAGE_VIEW_TYPE_2D_ARRAY` views (6 × N layers), indexed by the same slot; other slots may stay unwritten |
| 1 | 0 | `SAMPLER[4096]` | heap samplers, indexed by an S#'s slot; immediate samplers in slots the host reserves |
| 3 | 0 | `UNIFORM_BUFFER` | the memory map (section 3) |

Sets 0, 1 and 3 are global (all stages; 0 and 1 update-after-bind,
partially bound), so every stage of every pipeline has the same layout.
Set 2 is unused.

Specialization constants: **0** (bool, default true) the memory map has
one range (section 3); **100 + n** (u32, default 0) the sampler-heap slot
of immediate sampler n (section 5.3). The array sizes are fixed (descriptor
buffers need them at layout creation); the host never hands out a slot
past them.

The translator declares several variables on bindings 0 and 1, one per
image type it uses (1D, 2D, 3D, cube, arrays, multisampled; float or
uint sampled type), aliasing the same descriptors. The host writes the
view whose `viewType` matches the T#'s TYPE. For a cube (TYPE cube) it also
writes binding 2 with the image's 2D-array view: the translator reads a
cube there as `OpTypeImage %float 2D Depth=0|1 Arrayed=1 MS=0 Sampled=1`
where a cube cannot serve: samples with derivatives (`_d`, `_cd`) or texel
offsets (`_o`), at (s − 1, t − 1, slice · 6 + face) from the console's
(s, t in [1, 2], face = slice · 8 + face), and image loads (texel fetches).
Other cube samples use binding 0's cube view (filtering across faces stays
exact there).

**Slot 0 is the null image** on bindings 0, 1 and 2 (a null descriptor, or
a 1 × 1 black view where `nullDescriptor` is missing); the host never
hands it out. A T# that fails the check word (5.2) reads slot 0.

### 5.2 T# (8 dwords)

The host writes image views as RDNA2 T#s (Mesa's
`ac_build_gfx10_texture_descriptor` field positions, as Leonida's
`gfx10_descriptors.h` `encode(const Image&)`): base address, format,
width, height, depth, levels, layers, swizzle, type are real (the code may
read them as numbers). **WORD6 and WORD7 are the host's:**

```
WORD7  SLOT[23:0]    index into set 0 bindings 0 and 1
       INTEGER[24]   1: the view was made with the UINT twin of its format
                     (integer formats, signed or not); the shader uses a
                     uint sampled type. 0: float sampled type.
       STORAGE[25]   1: binding 1 holds a storage view at SLOT (a UAV);
                     image loads then read it with OpImageRead
                     (coherent with stores in the same dispatch); 0:
                     loads use OpImageFetch on binding 0.
       [31:26]       0
WORD6  WORD7 ^ 0x7A3C5E91 (the check word)
```

The translator uses WORD7 only when `WORD6 ^ WORD7 == 0x7A3C5E91`;
otherwise it reads WORD7 as 0 (slot 0, the null image, float, sampled).
T#s the title builds itself, or garbage, therefore read the null image
instead of an arbitrary slot; a T# the title copies keeps both words.

Sampling (`image_sample*`, `image_gather*`) always uses binding 0; stores
and atomics binding 1; `image_get_resinfo` queries binding 0 (binding 1
when STORAGE). The console code passes the 8 dwords it loaded (or built)
straight to the image instruction, so a T# the title copies from one heap
slot to another keeps its slot.

### 5.3 S# (4 dwords)

Heap samplers: real RDNA2 S# fields (WORD0–2 as Leonida's
`encode(const Sampler&)`); **WORD3 BORDER_COLOR_PTR[11:0] is the slot**
into set 1 binding 0 (the host owns border colours; BORDER_COLOR_TYPE
stays as the console's). At most 4,096 heap samplers.

Immediate samplers (root signature static samplers, whose four words the
console code builds from constants): the translator lists their words
(`VkShader::samplers`, position n) and reads immediate sampler n from
**set 1 binding 0 at the index specialization constant `100 + n`**
(u32, default 0) gives. The host puts each distinct immediate sampler
(`StaticSampler::from_words`, already in `emit/sampler.rs`) into a slot of
the sampler heap it reserves for them, and sets the constants per stage
at pipeline creation.

## 6. Stage interface

- **Locations come from the title's semantic registry.** Every
  vertex/mesh output and pixel input with semantic (name, index) has the
  location `vk::Semantics` assigns it: the location it got the first time
  any program of the title used it, so every vertex/pixel pair agrees and
  no module is rewritten per pair. A new semantic takes a location no
  semantic holds yet (0..30), else (more than 31 in the title) the lowest
  one no other semantic of that program holds; a program two of whose
  semantics share a location is refused ("two semantics of the program
  share a location in the registry"), as is one past 31 locations. The
  registry is seeded with the first title's colouring (37 semantics in 27
  locations) and only grows. **The host keeps it per title next to the
  stage cache** (`Semantics::save` / `Semantics::load` on
  `Semantics::global()`, which `vk::translate` uses; or its own registry
  through `vk::translate_with`), loads it before the first translation and
  saves it when it grew; `vk::semantics::SEMANTICS_VERSION` goes into the
  cache key, and the registry and the cache are kept or dropped together.
  Each semantic is a vec4 at its location, components as the console
  signature's mask columns; a pixel input no vertex output writes reads an
  undefined value.
- **Vertex.** Most vertex programs fetch their vertices themselves with
  buffer loads (no vertex input bindings). A vertex program that calls a
  console fetch shader (`s_swappc_b64` at its start) instead reads
  **vertex attributes as `Input` variables at `Location` = the
  attribute's input-signature row**, a vec4 of 32-bit floats or unsigned
  integers (the signature's component type; signed integers as unsigned
  bits). `VkShader::interface` lists them (`input: true`, semantic,
  index, location); the host makes the vertex input state from the
  title's input layout, matching its elements by semantic name and
  index. `gl_VertexIndex` is the console's vertex id
  (index + base vertex), `gl_InstanceIndex - gl_BaseInstance` its instance
  id. `exp pos0` is `gl_Position`; `exp param<n>` writes the outputs of
  signature row n at their semantics' locations (32-bit float; integer
  values travel as float bits). Outputs carry no interpolation
  decorations.
- **Pixel.** Inputs at their semantics' locations with the interpolation
  the barycentrics say (perspective/linear × center/centroid/sample,
  flat for `v_interp_mov` P0). `exp mrt<n>` is
  `Location = n` with the type of the console signature's `SV_Target<n>`
  (float, uint or int vec4); depth, stencil export and sample mask as
  `FragDepth`, `FragStencilRefEXT`, `SampleMask`. A lane killed by the
  program is demoted (`OpDemoteToHelperInvocation`).
- **Compute.** `LocalSize` from the console's thread-group size; thread
  and group ids from `LocalInvocationId` / `WorkgroupId`.
- **Mesh.** `VK_EXT_mesh_shader`, `OutputVertices`/`OutputPrimitivesEXT`
  the console's maxima, triangles. The module sets the output counts to
  the maxima at its start and culls every primitive
  (`CullPrimitiveEXT`) until the console code exports it (the console's
  `GS_ALLOC_REQ` counts are not used). Thread n writes vertex n and
  primitive n (`LocalInvocationIndex`). Dispatched as `vkCmdDrawMeshTasksEXT(x,
  1, 1)` with the title's group count.

## 7. Waves

The console runs waves of 32 or 64 lanes; the host runs subgroups of 32.
Each host subgroup acts as a console wave: scalar (wave-uniform) values
are uniform per subgroup, lane masks are per-invocation booleans
(`OpGroupNonUniformBallot` where the code reads a mask as a number).
A console wave64 program runs as two subgroups; cross-lane operations
that reach across the two halves (a 64-bit ballot read as a number,
`v_readlane` of a lane ≥ 32, DPP row broadcasts across 32) are taken per
half. Programs where that changes the result are listed by the
translator (`VkShader::approximate`).

## 8. What the translator returns

```rust
pub struct VkShader {
    pub stage: HwStage,
    pub spirv: Vec<u32>,
    pub user_data: Vec<emit::UserData>,  // push dword (stage window + index) <- root slot/dword
    pub samplers: Vec<[u32; 4]>,         // immediate sampler n: set 1 binding 0 at spec constant 100 + n
    pub interface: Vec<Interface>,       // semantics and locations (information only)
    pub local_size: [u32; 3],            // compute / mesh
    pub approximate: Vec<&'static str>,  // wave64 operations taken per half
}
```

## 9. Known gaps and open points

Known gaps (the program translates; the result differs from the console):

- Position exports past the first (`exp pos1..3`: clip and cull
  distances) are dropped.
- Ray/node tests (`image_bvh*_intersect_ray`) miss every node.
- Typed buffer stores through a V#'s format are written as 32-bit
  channels.
- Wave64 programs run as two host subgroups: a 64-bit mask read as a
  number (`s_bcnt1_i32_b64`, `s_ff1_i32_b64`, the high half), and
  `v_readlane`/`v_writelane` of a lane in the other half, are taken per
  half (`VkShader::approximate` lists them).
- Cube samples with derivatives or offsets read the faces as a 2D array:
  no filtering across face edges.

Open points:

- Typed buffer formats decoded in the shader: which formats (the title's
  `buffer_load_format_*` / `tbuffer_*` set) is measured as the backend
  grows.
- Ray-tracing node tests (`image_bvh*_intersect_ray`): the BVH base is a
  title address in a buffer-like T#; v0 misses every node, as
  `VICTORY_BVH_MISS` does today.
- Whether graphics stages should share one user-data window (the layouts
  in the pack put the same root parameter at the same index in every
  stage; v1 keeps one window per stage).
- Heaps below 4 GiB: the host checks it at heap creation (section 2,
  `Table`).
