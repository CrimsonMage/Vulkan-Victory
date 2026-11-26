# Vulkan shader ABI (translator ↔ Vulkan host)

Status: **draft v0** (2026-11-25). Owner of the translator side:
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

`ABI_VERSION = 0`

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
  only if a pixel program reads per-vertex attributes (`v_interp_mov`), and
  `shaderDemoteToHelperInvocation`.

SPIR-V 1.6 (Vulkan 1.3), `PhysicalStorageBuffer64` addressing, GLSL450
memory model. Every module passes `spirv-val --target-env vulkan1.3
--scalar-block-layout` offline.

## 2. Push constants (256 bytes, one block for all stages)

| Dwords | Content |
|---|---|
| 0–1 | `memory_map`: device address (u64) of the memory map block (section 3) |
| 2–3 | reserved, 0 |
| 4–31 | user data of the first stage: compute, vertex or mesh program (user-data dword k at push dword 4 + k; k < 28) |
| 32–63 | user data of the pixel program (user-data dword k at push dword 32 + k; k < 32) |

A compute program may use dwords 4–63 (k < 60).

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
one subtract and add over a small set of host imports of title memory:

```
struct MemoryMap {            // at push dwords 0-1, 16-byte aligned, read-only for shaders
    u64 zero;                 // device address of >= 256 bytes of zeros (never written)
    u64 sink;                 // device address of >= 256 writable bytes nobody reads (dropped stores)
    u32 count;                // ranges in use, <= 4
    u32 pad[3];
    struct {
        u64 title_start;      // first title address of the range
        u64 title_size;       // bytes
        u64 device_base;      // device address that title_start maps to
        u64 pad;
    } range[4];
};
```

`device = range.device_base + (title - range.title_start)` for the range
with `title - title_start < title_size` (unsigned). The translator checks
the ranges in order (unrolled, `count` read at run time). An address in
no range reads zeros from `zero` and writes into `sink`; so does any
access that is out of bounds (section 4). The map must stay valid and
unchanged while any submitted work can read it; a new map is a new block.

Translation happens once per distinct base (a V#'s base, a table address,
a root address): CSE keeps it out of loops; every access then only adds
its offset.

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

### 5.1 Descriptor set 0 (global, all stages, update-after-bind, partially bound)

| Binding | Type | Content |
|---|---|---|
| 0 | `SAMPLED_IMAGE[]` | image views, indexed by a T#'s slot |
| 1 | `STORAGE_IMAGE[]` | storage views of UAV textures, indexed by the same slot |
| 2 | `SAMPLER[]` | samplers, indexed by an S#'s slot |

The translator declares several variables on bindings 0 and 1, one per
image type it uses (1D, 2D, 3D, cube, arrays, multisampled; float or
uint sampled type), aliasing the same descriptors. The host writes the
view whose `viewType` matches the T#'s TYPE.

### 5.2 T# (8 dwords)

The host writes image views as RDNA2 T#s (Mesa's
`ac_build_gfx10_texture_descriptor` field positions, as Leonida's
`gfx10_descriptors.h` `encode(const Image&)`): base address, format,
width, height, depth, levels, layers, swizzle, type are real (the code may
read them as numbers). WORD6 is 0. **WORD7 is the host's:**

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
```

Sampling (`image_sample*`, `image_gather*`) always uses binding 0; stores
and atomics binding 1; `image_get_resinfo` queries binding 0 (binding 1
when STORAGE). The console code passes the 8 dwords it loaded (or built)
straight to the image instruction, so a T# the title copies from one heap
slot to another keeps its slot.

### 5.3 S# (4 dwords)

Heap samplers: real RDNA2 S# fields (WORD0–2 as Leonida's
`encode(const Sampler&)`); **WORD3 BORDER_COLOR_PTR[11:0] is the slot**
into set 0 binding 2 (the host owns border colours; BORDER_COLOR_TYPE
stays as the console's). At most 4,096 heap samplers.

Immediate samplers (root signature static samplers, whose four words the
console code builds from constants): the translator lists their words
(`VkShader::samplers`, position n) and reads them from **set 1, element n
of binding 0** (compute, vertex and mesh programs) **or binding 1** (pixel
programs): per-pipeline arrays of immutable samplers the host builds from
the words (`StaticSampler::from_words`, already in `emit/sampler.rs`). A
stage with no immediate sampler declares nothing in set 1.

## 6. Stage interface

- **Vertex.** No vertex input bindings: the title's code fetches its
  vertices with buffer loads. `gl_VertexIndex` is the console's vertex id
  (index + base vertex), `gl_InstanceIndex - gl_BaseInstance` its instance
  id. `exp pos0` is `gl_Position`; `exp param<n>` an output at
  `Location = n` (vec4 of 32-bit, float or the signature's integer type).
- **Pixel.** Inputs at the location of the attribute the console reads
  (`v_interp_*` attribute index), with the interpolation the barycentrics
  say (perspective/linear × center/centroid/sample). Because the console
  driver links the pixel program's attributes to the vertex program's
  parameters by semantic at pipeline creation, so does the host: each
  module lists its interface (`VkShader::interface`: semantic name, index,
  location, the word offset of the `Location` literal), and
  `victory_shader::vk::link(vertex, pixel)` rewrites the pixel module's
  locations to the vertex module's for the same semantic (a pixel input no
  vertex output names reads 0 from a location past both). `exp mrt<n>` is
  `Location = n` with the type of the console signature's `SV_Target<n>`
  (float, uint or int vec4); depth, stencil export and sample mask as
  `FragDepth`, `FragStencilRefEXT`, `SampleMask`. A lane killed by the
  program is demoted (`OpDemoteToHelperInvocation`).
- **Compute.** `LocalSize` from the console's thread-group size; thread
  and group ids from `LocalInvocationId` / `WorkgroupId`.
- **Mesh.** `VK_EXT_mesh_shader`, `OutputVertices`/`OutputPrimitivesEXT`
  the console's maxima, triangles. Dispatched as `vkCmdDrawMeshTasksEXT(x,
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
    pub samplers: Vec<[u32; 4]>,         // immediate samplers, set 1 binding 0 element n
    pub interface: Vec<Interface>,       // vertex outputs / pixel inputs and outputs
    pub local_size: [u32; 3],            // compute / mesh
    pub approximate: Vec<&'static str>,  // wave64 operations taken per half
}
```

## 9. Open points (v0)

- Typed buffer formats decoded in the shader: which formats (the title's
  `buffer_load_format_*` / `tbuffer_*` set) is measured as the backend
  grows.
- Ray-tracing node tests (`image_bvh*_intersect_ray`): the BVH base is a
  title address in a buffer-like T#; v0 misses every node, as
  `VICTORY_BVH_MISS` does today.
- Whether graphics stages should share one user-data window (the layouts
  in the pack put the same root parameter at the same index in every
  stage; v0 keeps one window per stage).
