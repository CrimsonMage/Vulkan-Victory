# Keep it simple: building for PC from Xbox

## The principle

A studio shipping on Xbox writes its game against the GDK and Direct3D 12,
then turns on the Xbox-specific parts: unified memory, D3D12.X extensions,
precompiled GPU code, hardware audio and storage paths. Victory runs that
in reverse. **Each Xbox-specific behaviour is turned back into the plain PC
API it was ported from, at the moment the title asks for it.** Nothing is
watched, guessed or mirrored when an API call already says what happens.

Much of the GDK exists on PC unchanged (XGameRuntime, GameInput, XAudio2,
XUser, DirectStorage). Where the console API has a PC twin, victory calls
the twin; it implements only what is truly console-only.

## Xbox → PC map

| Xbox side (what the title uses) | PC side (what victory turns it into) | Module | Status |
|---|---|---|---|
| Unified memory: the CPU writes, the GPU reads the same pages | Title memory imported by host pointer into Vulkan (buffer device addresses = title addresses, through the memory map); the pool's GPU view a sparse buffer of card pages bound at each page-mapping call, filled at announced writes and after CPU stores (write tracking: pages the card has a copy of are write-protected, no mirror) | `victory-gpu/vulkan*`, `victory-vk`, `victory-hle/xmem.rs` | works (`docs/vulkan-host.md`) |
| D3D12.X device and command lists (`d3d12_xs`) | Vulkan, recorded from the calls the title's threads write down | `victory-gpu` (`device`, `list`, `vulkan/`) | works; the only GPU back end since 2026-11-26 |
| Console GPU code (RDNA2 machine code, no HLSL or DXIL shipped) | Decoder → IR → **SPIR-V** (`docs/vulkan-abi.md`), translated once per stage and kept on disk | `victory-shader` (`vk/`) | works |
| Console tiled texture layouts | Detiled into card images when the data arrives | `victory-gpu/vulkan/images*` | works |
| Ray tracing (console BVH format, title's own builder DLL) | Not yet on the SPIR-V path (the DXIL builder went with the D3D12 back end) | `victory-shader/vk/` | open |
| 64-lane waves | 32-lane PC waves with emulation in the translator | `victory-shader` | works; keep |
| Console DirectStorage (`XCRDAPI`) | One shared handle per file; each request does one positioned read of exactly its source range when it runs (uncompressed straight into title memory, compressed into a reused worker buffer, then inflate/unpack on the CPU); one FIFO worker pool, no lock per request (`dstorage streaming` line every 10 s) | `victory-hle/dstorage.rs` | built, not yet run |
| XMA audio hardware (`AcpHal`), Opus (`xapu`) | Software decode → host audio (XAudio2 / WASAPI) | `victory-hle/audio*`, `apu.rs` | works; reverb (`chad` XDsp) silent |
| Bink 2 video | The title's own Bink DLL, with XAudio2 passed through | title DLL | works |
| GameInput, window, lifecycle | GameInput on PC; a Win32 window | `gameinput.rs`, `lifecycle.rs` | works |
| XUser, XStore, saves, networking | PC GDK twins or local stand-ins; no network | `xuser.rs`, `xstore.rs`, `xgamesave.rs`, `net.rs` | works |
| Console kernel (`kernelx.dll`, its legacy parts, `minuserx.dll`); runtime classes we lack | The PC's kernel32/kernelbase/advapi32 and user32 by name, our overrides first; kernelx's own exports mapped onto xmem, lifecycle and thread descriptions; unknown XGameRuntime classes to the PC's own `xgameruntime.dll` (Proton ships none: NOT_SUPPORTED, logged once per class) | `modules/mod.rs` (`twins`), `kernelx.rs`, `xgameruntime.rs` | built; this title imports none of them |
| Console CPU instructions the PC lacks (CLZERO, MONITORX, INSERTQ ...) | Rewritten as plain x86 at the first trap (stubs for clzero loops and INSERTQ; PAUSE/NOP for MONITORX/MWAITX); emulated only as fallback | `victory-loader` | works |

## Targets

- **At least 30 fps everywhere**: menus, loading, cutscenes and the world.
  More is better (the title's own cap is 30; see workstream 10 for 60).
- **Low memory**: as little card memory and system memory as the content
  needs, not a mirror of everything.
- **Process memory (the PC's RAM)**: the baseline is a Series S, where the
  title has about 8 GB for everything, CPU and GPU. The title's own memory
  (~7.2 GB, all opened as heaps and so resident) is its budget; what is
  ours (victory, Wine, the driver) comes on top. Target:
  the whole process at **≤ ~12 GB first, then toward ~10 GB**, read as the
  `unique` figure of the `process memory:` line (each page once; Linux's
  resident count, which `top` shows, counts a section page at each view
  that maps it). Last full run: ~16.7 GB resident, ~9.5 GB of it ours.
- **Iterate**: measure (`frame time:` line, card/PC memory), cut the
  biggest cost, repeat.
- **Content is streamed on demand and then stays put**: the title loads it
  once (DirectStorage, file reads) and does not churn it. So do work once,
  at the moment the title announces new data, and never re-check, re-hash,
  re-copy or re-translate it per frame.
- **Memory follows what is loaded**: when the title frees, unmaps or
  destroys something, victory releases its copies, views and heaps for it
  right then (after in-flight work lands). Idle eviction under budget
  pressure is only a safety net.

## Unified memory on split hardware

The console has one memory; the PC has RAM and video memory joined by a bus.
Every piece of title memory falls into one of four kinds, and each kind has
exactly one home, the one a PC game would give it:

| Who writes / who reads | Home on the PC | What victory does |
|---|---|---|
| CPU writes often, GPU reads (constants, dynamic vertices, command data) | system RAM, read in place by the GPU (upload heap) | open the title's memory as a heap; nothing copied, nothing tracked |
| Streamed once, GPU reads (textures, static geometry) | video memory (default heap) | upload once when the title says it arrived; free when it unloads |
| GPU writes, GPU reads (render targets, scratch) | video memory only | never copied back |
| GPU writes, CPU reads (queries, read-backs) | system RAM (read-back heap) | the GPU writes into title memory in place |

No class is mirrored in both places and kept in sync by watching or
comparing. If something seems to need that, it is in the wrong class.

The card forces two exceptions, both copies of memory the CPU may still
write: the pool's GPU view (imported memory cannot back a sparse buffer on
NVIDIA) and textures (the console's tiled layouts are not the card's).
They are kept exact by write tracking, not by comparing: such a page is
write-protected, the first store into it faults once, and its copies are
made again before the next submission (`victory-gpu/src/track.rs`,
`docs/vulkan-host.md` "Write tracking").

## Rules

- Prefer an explicit API point over detecting side effects in memory.
- One behaviour, one code path. No `VICTORY_*` switch for experiments; a
  switch exists only for a user-facing choice (paths, logging, console
  profile, capture for testing, the few listed in the inventories).
- Linux runs exactly the Windows build under stock Proton (its
  winevulkan); no app-local DLLs (no vkd3d-proton, `d3dcompiler_47` or
  DXC since the D3D12 back end went).
- Every change passes the gate before it lands.
- Code shape (`cargo xtask quality`, run by the pre-commit hook): a file
  over 1,000 lines plans a split, over 1,500 must split; per function,
  cognitive complexity (SonarSource) over 15 plans and over 25 must
  simplify; cyclomatic over 10 only warns (flat lookup tables score high on
  it). New code stays under the mandatory numbers. Today's offenders are listed in `xtask/quality-baseline.toml` and
  may only shrink; `cargo xtask quality --update-baseline` lowers them as
  code is split.

## Gate

`VICTORY_CONSOLE=series-s /mnt/data/victory/autorun.sh <name> 300 "$(cat /mnt/data/victory/keys-start.txt)"` (5 minutes)
on the diagnostics build, with the GPU-memory watchdog: it boots, reaches
the world, saves captures and logs frame time and card memory. A change
lands on `integrate` only if the run reaches as far as before, with no
driver errors and no new crash, and the captures look no worse.

## Measures

| | Start (2026-11-25) | Now |
|---|---|---|
| Rust lines | 128,292 | 78,027 (154,557 before the D3D12 back end went) |
| `VICTORY_*` switches | 224 | 43 |
| `diagnostics`-only blocks | 283 | 71 |
| World frame time (Proton, 8 GB card) | 0.5–5 s | ~295 ms (~3.3 fps), Series S (run p20) |

## Workstreams

| # | Work | Owner | State |
|---|---|---|---|
| 1 | Prune `victory-gpu` switches and statistics | subagent | merged (6f7b937): −18,465 lines, 163 → 21 switches; one gate line every 5 s; memory-mirroring switches left for #7; gate g4 running |
| 2 | Prune `victory-shader` switches | subagent | merged (d645fb8): switches 25 → 1, −2,526 lines; all 25,985 containers translate byte-identically; gate g2 running |
| 3 | Prune HLE, loader, formats, app | subagent | merged (08c435a): switches 38 → 7, −1,886 lines; one console-instruction emulator; gate g1 passed |
| 4 | Unified memory design and probe (`VICTORY_UNIFIED`) | subagent | done: `docs/unified-memory.md`; probe run |
| 5 | HLSL + DXC shader backend (`VICTORY_SHADER_BACKEND=hlsl`, off by default) | subagent | merged: 964/1,000 sample shaders compile and validate; ~25% of the lowering code, no DXIL writer; first compiles ~17× slower (median 40 ms, worst ~11 s); GPU A/B parity not yet shown |
| 6 | Merge 1–3, gate run | coordinator | waiting |
| 7 | Unified memory replaces mirroring (stages 1–3, default on; `VICTORY_UNIFIED=0` = old) | subagent | merged (323d262): heaps open, 0 refused, 714 buffers in place; blocked by the world-start VRAM spike; deletion (stage 4) after |
| 9 | Find the world-start card-memory spike (+1.1–1.7 GB in a second) | subagent | running |
| 11 | Cut memory-mapping churn: stock wineserver walks every view on each map/unmap | subagent | merged (e4e2dd2): wineserver 88% → 5–7% of a core in the world; ~17 map calls/s instead of ~200; filler zero page shown by stand-ins |
| 13 | Series S: compose the title's two display planes (compose.rs) | subagent | merged: menus, loading and credits show |
| 14 | Series S world: the title's FSR4 upscale (console-only ACO code) runs as a plain linear scale at `ffxDispatch` | subagent | merged (977868f): Series S world renders lit (run s7, frame 1890) |
| 15 | Frame-time breakdown (`frame time:` line every 5 s) and removal of four needless full card drains | subagent | merged (28ccc38), built, not yet run |
| 17 | Console CPU instructions: patch each site once | subagent | merged: 21,000 traps/s → 8 in a whole run |
| 18 | Per-draw recording cost | subagent | merged: per-thread lookups, lock-free slot check, fast hashers, in-place constant tables, slot chunks; per-call timing in the frame line |
| 21 | Indirect and dispatch recording (~640 ms CPU/frame) | subagent | running |
| 22 | Submission pass-through: last card copies in place, no merge wait, fences to the GPU | subagent | running |
| 23 | Windows over scattered pool pages gathered on the card from the opened sections (GPU copies of runs of pages, again only on remaps and announced writes; written ranges copied back after the list); dispatches with only new root constants set their user data alone; dispatch binding split in the frame line | subagent | built, not yet run |
| 24 | Pool memory by directory: raw, structured and constant views over page reservations read and written through the 64 KiB directory onto the opened sections (pool table per heap slot, uniform branch in the translator, 64 KiB edges split); directory in card memory written by GPU copies at the head of each submission; the pool's GPU view has no host views (entries from the page-mapping copies; the CPU reads it through the sections); windows only for typed views, index/indirect/vertex buffers, card copies, spilled programs' root views; copy-out deleted; frame line `pool by directory MB, windows MB` | subagent | built, not yet run |
| 19 | Memory: levels on demand by default, release on title frees, room-only eviction, big buffers in place | subagent | merged |
| 20 | `globallycoherent` UAVs, device-scope barriers and fenced atomics for compute programs that sync across groups (the BVH builder's, and any with `buffer_gl1_inv`); builder still off by default (`VICTORY_BVH_BUILDER=1` to try). Placed textures picked by index whose memory holds only zeros get no host copy until data arrives | subagent | built, not yet run |
| 16 | Try XCL (github.com/xcl-dev/xcl) for ideas: builds and starts under stock Proton (`/mnt/data/xcl/run.sh`, own prefix); backs title memory with sparse VRAM windows + 512 MB coherent window | coordinator | stalled before its first frame under Proton (title graphics memory 4.5 GB > its 4 GB coherent window); idea study stopped |
| 12 | Texture quality setting (`VICTORY_TEXTURE_QUALITY`, medium under 10 GB) and lighter menus | subagent | merged (be13db8) |
| 24 | Thin recording (after VIctory-7900XTX `VICTORY_THIN`, GPL-3.0): the title's threads only write list calls down (`list/thin.rs`), a worker per fast core translates them, `ExecuteCommandLists` waits for its lists; first uploads of title-filled textures on a copy queue of their own, detiled by its workers, slots show a 1x1 grey fallback until the queue's fence (`texture/ahead.rs`, after their `VICTORY_UPLOAD_AHEAD`/`VICTORY_SHOW_AFTER`) | subagent | built, not yet run |
| 25 | Card page cache for the pool (`memory/cache.rs`): each physical pool page a list names gets one copy in 64 MiB blocks of card memory (a directory copy per block, every alias's entry names it), taken in at first naming or announced change (DirectStorage, file reads, CPU title copies, page mappings), shader writes copied back after each list word by word against a twin (only changed words, never over CPU writes), heap-picked writes held off the cache, LRU eviction by frame inside 90% of memcap's ceiling (idle texture copies let go first for room), naming lock-free once per view and list and off while the cache has no block; frame line `pool page cache: …` | subagent | p21: never opened (gated on the falling system budget), naming cost ~900 ms/frame of workers; fixed, built, not yet run |
| 26 | Process memory: `process memory:` line every 10 s (Linux's smaps joined with a `VirtualQuery` walk and our counters; victory's heap by owner through a counting allocator in the diagnostics build); DirectStorage worker buffers kept up to 1 MiB (was 8 MiB × 3 × 28 workers), texture read buffers up to 16 MiB per thread (was the largest texture) | subagent | built, not yet run |
| 27 | Pipelines like a PC game: a catalog of every pipeline a run drew with (shaders by hash, graphics state, frame of first use; dropped after 5 unused launches) beside the library; baked as the title starts (4 workers, lowest priority, compiles in flight under 1 GiB, made to fill the caches and let go) or before it (`victory bake [--title] [--jobs]`, its own device, half the CPUs, 3 GiB); the title's creation of a catalog pipeline jumps the queue; a draw waits for its pipeline up to 2 s a frame, then drops; a repeat bake skips what was made with the same translator, driver and layer; translated-shader cache keyed by the container's hash; progress lines `compiling shaders: k of n, ETA` | subagent | built, not yet run |
| 10 | 60 fps: hand the title a 16.7 ms frame interval (its own request is 33.3 ms, `frame.rs`) | — | after a steady 30 in the world |
| 8 | HLSL backend replaces the DXIL encoder | — | after 5 and 6 |

## Decisions log

- 2026-11-26: wave-uniform values before emission (`vk/uniform.rs`). The
  card spent ~85% of a world frame in mesh draws whose modules exchanged
  between wave64 halves at every `v_readfirstlane` and counted every
  barrier, though the programs (MS_VS_Transform and kin) have no real
  cross-lane flow: their readfirstlanes read the group id, loads from
  scalar addresses and masks of scalar compares. The thread IR now knows
  which values every invocation of a console wave computes alike; reads
  of one lane of them are the value, exchanges of them and exchanges of
  words made of the invocation's place (thread index compares) are
  computed locally, and programs with no shared state lose their
  meetings and barriers. On the used set: mesh exchange calls 7,607 ->
  1,094 (programs exchanging 240 -> 83), counted barriers 480 -> 0;
  compute exchange calls 6,175 -> 1,732. Interpreter runs of the old and
  new thread IR agree on random memory for all 1,592 programs.

- 2026-11-26: 64-bit image atomics (ABI v5). The 679 alpha prepass
  pixel programs do one `image_atomic_umax` on a two-dword texel (depth
  high, payload low); it ran on the low dword only. Now a u64 atomic on an
  R64_UINT view of the same image (set 0 binding 3 at the T#'s slot, a
  third 2^20 array like the cube binding: 4 MiB of descriptors at
  NVIDIA's 4 bytes); every other program keeps the two-channel views.
  Chosen over a "64-bit texel" T# bit with a split/join typed path: only
  the atomic touches the new binding, the typed load/store path stays as
  it is, and no compact second index is needed at that descriptor size.
  A device without `shaderImageInt64Atomics` refuses those modules with
  the reason (no low-dword fallback: it was wrong output anyway).
- 2026-11-26: shader differences closed (`docs/translation-index.md`):
  clip and cull distances from each program's `PA_CL_VS_OUT_CNTL`
  (stage data; vertex E+0x1a0, mesh E+0x194: the 1,804 `pos1` exports
  are clip distances, two vertex programs' cull distances); typed buffer
  stores encoded per the V#'s FORMAT (`Helper::TypedPack`, checked
  against the interpreter's reference over 23 formats). Wave64 programs
  run as one subgroup (915 / 736 listed) stay per half: exact for masks
  of EXEC and compares, not for constant masks read as numbers or sums
  across halves (`docs/vulkan-abi.md` section 7). The last guard hit (vk48,
  guard id 0x7b3296e7) is the BVH builder's copy program
  c42a31951201f9a0 (`XboxBvhBuilder.dll`, at 0xb4fe0): its spilled
  user-data table read zeros (the skip removal renamed by register after
  spill.rs had made the loads entry values: its stride 0, the copy loop
  never advanced), and its NUM_RECORDS -1 views failed the memory map's
  range test (the count read 0). Both fixed: the table rewrite runs after
  the skips, spilled entries read push dwords, a V# reaches the end of
  its base's range.

- 2026-11-26: the black world: the host refused the world's and the
  characters' mesh programs (`MS_VS_Transform*`, `MS_VS_PedTransform*`)
  before compiling, their output memory over NVIDIA's 32 KiB. Their maxima
  are already the true ones (128/128 in the record and in `PSV0`; one
  vertex per thread), and Vulkan counts outputs by whole locations, so
  narrowing vec4s or fp16 saves nothing. What was done (ABI v4): one count
  of output memory, the specification's (primitive indices left out), in
  `victory_shader::vk::usage`, used by the translator and the host's
  check; culled primitives as degenerate triangles (no `CullPrimitiveEXT`);
  programs still over (position + 16 or more locations: 810 of the 8,765
  mesh programs, 24 of the 29 refused in runs) split into 4 host groups of
  96 vertex slots (35 of 63 slots for the 14 with 21 locations), each
  running the whole console group. Every mesh program now counts at most
  32,768 bytes. Mesh groups of several waves count their console
  barriers per subgroup (the NGG code for a culled group has its first
  wave take an extra `s_barrier`, which an `OpControlBarrier` left waiting
  for waves that had ended: the likely hang of `MS_VS_OctaSLODFire`, vk43b),
  and exchanges wait at most 2^10 polls, once per pair.
- 2026-11-26: loops that give up, traced offline with real data
  (`vk::interp`, a CPU interpreter of the thread IR: one console wave over
  a memory image, missing memory listed; `examples/tir_run.rs`). Found:
  the skip removal (`ir::without_skips`) let a wave with no lane left
  walk a region holding a loop; the console never runs it, and a loop
  whose way out needs a lane never ends there. PS_Textured_Eight's light
  loop (`s59 += 1` until an active lane has `v56 == s59`,
  `s_cbranch_vccz`) ran to the guard in every wave whose pixels had no
  light (`v55 == 0` for all: EXEC empty at its `s_cbranch_execz`); a
  region holding a loop (and no `s_barrier`) now keeps its skip (692 of
  the 1,527 used programs change, 480 of them compute programs whose
  early `if (id >= count) return` skip covers a loop; all translate and
  validate). The batch-culling programs'
  loops (plane, sphere, LOD, meshlet counts) are counted loops over
  scalar counts read from memory; with sane counts the IR and the SPIR-V
  (a Python SPIR-V interpreter in scratch) run them exactly, also with
  EXEC empty. CS_DistanceWindowEmissivesCulling has only the plane and
  sphere loops (counts at view record +0x004 and +0x374), so when it gives
  up the card reads a count that is not small there: the view records
  were never probed (0x400 bytes at V#(ud4+0x500).base + ud7 there; at
  V#(ud4+0xac0).base + batch[0] in the batch programs, 0xbd8fc00 in vk44).
- 2026-11-26: the zeros in streamed resources (vk17, vk35: `m_PageMap`
  0, a vtable 0) traced to xmem's filler pages. Every run made ~1,348
  fresh pages "fillers" at startup (zeros, shown at 8+ places counting
  the GPU views, or twice in a scatter) and showed them by ~2,165
  single-page stand-ins; data then came in through a view of the same
  page and the stand-ins kept showing zeros (39-216 "filler page was
  written" per run, found only at the 10 s check). Now only a page the
  title repeats at consecutive addresses in one call can be a filler;
  never one a DirectStorage request is writing (a filler under a request,
  memory or page destination, becomes views first); fillers are forgotten
  when freed or shown nowhere; section keys are ids, never reused.
  Diagnostics builds compare each memory destination with every other
  place showing its pages; access-violation reports dump 64 bytes at each
  register into title memory, with xmem's view of it. Built, not yet run.
- 2026-11-26: wave64 compute exact on the translator side (branch
  `worktree-agent-a08b3058c5f4bb083`). Found in CS_Tree_BatchCull (and the
  other BatchCull programs with wave prefix sums): `s_mov_b32 exec_lo, 0` /
  `exec_hi, -1` before the step that adds the lower half's total to the
  upper half was read as EXEC's low dword in both halves, so the upper
  half's prefix offsets missed the lower half's count and its lanes wrote
  over the lower lanes' output slots (the slots past them kept stale data).
  The halves also touched LDS unordered (lane 0 initialises, the wave
  ors/mins, lane 0 reads the total). Now: 64-bit masks per half, LDS
  accesses ordered by a meeting of the halves, exchanges in every wave64
  compute/mesh group (counters per pair beyond 64 threads, bounded),
  compute groups padded to whole console waves (identity in the lanes past
  the group), console thread numbering from the subgroup ids, loop guard
  2^14 iterations per invocation over all its loops. No wave64 compute
  approximation left in the used set. CS_Grass/Simple/Slod_BatchCull have
  no cross-lane code, only scalar loops over ranges read from memory: if
  they still give up, their input (constants, indirect group counts) is
  wrong, not the waves.
- 2026-11-26: DECISION (user): **D3D12 removed; victory is Vulkan only.**
  `VICTORY_GPU` and every D3D12 branch are gone; device, queues, lists,
  fences, present, resources, descriptors, views and title memory go
  straight to the Vulkan backend. Deleted: the D3D12 device and host,
  draw binding, descriptor host heaps, the 64 KiB directory and card page
  cache, unified/views/shadow/gather/indirect windows, write watch,
  guards and canary pages (xmem), the DXIL pipeline catalog, bake,
  library and translated-shader cache (`victory bake` with them), D3D12
  FSR, compose and capture paths, the frame-time and process-memory lines,
  and the old translator (emit/ with the DXIL encoder, hlsl/, resource.rs,
  structure.rs, bvh.rs). `emit/wave.rs` moved to `victory-shader/src/wave.rs`
  for its port into `vk/`. A title pipeline's stages are now made at its
  creation when the stage cache on disk holds one of them (was: the
  D3D12 catalog). Rust lines 154,557 → 78,027 (victory-gpu 56,679 →
  19,065; victory-shader 58,151 → 20,038; victory-hle 19,395 → 19,295;
  apps/victory 1,837 → 1,146). The host's `d3d12.dll` is still loaded
  for one thing: the title's `D3D12SerializeRootSignature` calls
  (`root.rs`).

- 2026-11-26: vk28 hung the card (Xid 109) at the first world frame
  (`docs/vulkan-host.md`, "vk28"). The pool view's error path had left its
  pages showing other pages' bytes; the translator has no loop cap since
  f02fac3. The pool view now takes what images leave and never fails a
  map; Vulkan images follow texture quality; a hang report names the
  submission. Built, not yet run.
- 2026-11-26: Vulkan genericity (branch `worktree-agent-a8169a5f3c8e04704`,
  `docs/vulkan-host.md`):
  - FSR4 upscale as one blit, read by the FidelityFX API layout and gated
    on the upscaler DLL's version, not on this title's offsets;
  - the console list slots XCL found (#53/54, #61-63, #81, #85, #88-90,
    #93, #94) as plain Vulkan;
  - measured occlusion and timestamp queries;
  - the queue/list layout decided by call shapes (Device11 only a hint);
  - PresentX takes up to 8 planes.
  - Built, not yet run.
- 2026-11-25: stock Proton install; a current vkd3d-proton beside the exe
  (the bundled converter fails on ~18% of shaders). No patched wineserver.
- 2026-11-25: memory ceilings plus an external GPU-memory watchdog (the
  NVIDIA driver corrupted itself and hung the kernel twice at ~7.7 of 8 GB).
- 2026-11-25: reverted reading DirectStorage page lists at enqueue (loading
  failed 4,922 times with it; 0 without).
- 2026-11-25: unified-memory probe on the RTX 5060 Ti under Proton: title
  memory opened as D3D12 heaps works, and the GPU sees every alias and
  writes back through every alias. Textures cannot be placed in it. The card
  reads it at 12.6 GB/s (PCIe), against 199 GB/s from its own memory.
  GPU-upload (ReBAR) heaps hold buffers and textures at 193 GB/s but are
  mapped above 1 TiB, so they can't be title memory. Direction: what the CPU
  writes often (constants, dynamic vertices) is read in place from RAM; big
  read-mostly data (geometry, textures) is uploaded to the card once, when
  the title says it arrived (DirectStorage completion, title copies), the
  way a PC game uses upload and default heaps.
- 2026-11-25: the Series S profile ends silently right after the title's
  FSR4 pipeline setup (twice); Series X stays the profile. Series X reaches
  the world start but card memory peaks at 7.2 GB there (victory 5.3 GB of
  it, render targets 1.95 GB uncapped).
- 2026-11-25: target is 60 fps in the world eventually (user).
- 2026-11-25: world start pushes the 8 GB card to ~7.7 GB in under 2 s in
  every configuration (Series X; 1080p display mode; rays off; unified on);
  the user chose to keep the 7.2 GB watchdog and find the cause first.
- 2026-11-25: the Series S end was not silent and not FSR4: victory's own
  "host GPU failure: no memory to fetch what shaders wrote" ended both runs
  (as it ended three Series X runs) when the world-start spike filled the
  4,075 MB shared-memory ceiling and a 16 MB fetch buffer was refused. The
  FSR4 lines were only the title's last debug output, printed under the
  crash report. Fetch buffers now skip the ceiling; the report heads the
  title's lines and says the run ends there.
- 2026-11-25: Series S output is black in every run (960x540 captures);
  being traced. Reserving card memory for render targets up front starved
  the menus (menu video froze/flashed); replaced by letting idle copies go.
- 2026-11-25: intro video flashes with unified texture detection; bisect
  run with VICTORY_UNIFIED=0 under way.
- 2026-11-25: world frame breakdown (Series S, 430 ms): GPU busy ~99 ms;
  title threads recording into our lists ~620 ms summed; submissions
  ~100 ms; card waits ~49 ms; landing 72 ms (54 ms fetching shader writes
  back). The process takes ~11,000 signals/s, mostly SIGILL.
- 2026-11-25: XCL study (GPL-2.0-only: ideas only, no code into GPL-3.0
  victory). Worth taking: translate title addresses by arithmetic over
  spanning buffers per opened heap (no per-draw lookup); settle descriptors
  at view creation; never fetch shader writes back; pass fences through to
  the GPU; back render targets with shared pages keyed by title address.
  Not worth taking: its card-memory mirror of title memory (doubles memory;
  wouldn't fit 8 GB), its per-frame write-watch for textures.
- 2026-11-25: run p7 (all fixes merged, stable): render-target page sharing
  would save only ~52 MB, so it is dropped. Found: the 174 MB heap is split
  at a 256 MiB piece edge and stays a card copy (134 ms fetch-back); idle
  eviction churns (18,790 copies re-made); RSS grew to 21.6 GB.
- 2026-11-25: run p10: the 174 MB heap (write-combined view memory) and two
  smaller big buffers open in place; card copies named per frame 14 GB →
  209 MB; fetch-back 140 → 76 ms; card memory 3.7 GB; eviction churn fixed
  (re-made 18,790 → 629). Recording cost is almost all ExecuteIndirect
  (~0.1 ms × 4,550) and Dispatch (~0.05–0.14 ms × 3,190); draws are cheap.
  Submissions still serialise (enter/merge ~120 ms).
- 2026-11-25: run p13 (submission pass-through, bounded indirect windows,
  read-based streaming merged): stable, no GPU errors; RSS 21.6 → 16.7 GB;
  card 4.06 GB; indirect 180–510 → 70 ms. Left: 4.2 GB/frame of windows over
  scattered pool pages refilled at every submission (steps 250 ms) and
  0.43 ms per dispatch. Probe: tile mappings into opened sections read wrong
  data on this stack, so windows will be gathered from the section heaps on
  the GPU, refreshed only on remaps and announced writes.
- 2026-11-25: windows over scattered pool pages are buffers on the card
  that GPU copies gather from the opened section heaps before the list
  that names them, for the pages the title remapped or announced written
  (or shaders wrote through another view) since; what shaders write into
  them is copied back into the sections after the list only with
  `VICTORY_GATHER_COPYOUT=1` (off by default: the copy out of whole
  declared ranges overwrote the title's CPU writes, and the Series S run
  crashed at loading). No CPU fill at
  submission. Whether a window opens in place whole is asked once until
  a remap (it was asked at every ask).
- 2026-11-25: whole-file mapping of archives failed (levels.rpf is 76 GB);
  streaming now does one positioned read of exactly each request's range.
- 2026-11-25: Leonida study (GPL-3.0, code reusable with notice; no world
  frame shown there). Taken: route pool memory through the 64 KiB
  raw-address directory onto section heaps (remap = directory write, in
  queue order), the directory in device-local memory, globallycoherent
  builder UAVs, deferred texture backing in shared 64 MiB blocks. Later:
  descriptors as memory (V#/T# read by shaders), offline pipeline prewarm
  from the title's PSO cache. Not taken: bypassing vkd3d, its windows.
- 2026-11-25: window copy-out corrupted title memory (title crash at frame
  ~900); copy-out off by default (VICTORY_GATHER_COPYOUT).
- 2026-11-25: run p15 (windows gathered on the card, copy-out off): stable;
  steps 250 → 11 ms, in submissions 270 → 39 ms, not-in-place fills 4.2 GB
  → 55 MB; dispatch 555 → 132 ms; but bounded indirect windows cost
  468 ms (short windows), being made nearly free.
- 2026-11-25: exe study (static): the streaming pool has three views of the
  same pages: a CPU view (read-write), a GPU view the CPU can never touch
  (PAGE_NOACCESS | GRAPHICS_NOACCESS) and a pool view mapped whole once.
  Mapping changes are announced once per submission (title 0xF2C470): one
  Scatter for the CPU view, one CopyPageMappingsBatchX of single 64 KiB
  pages for the GPU view. So the GPU view needs no host mapping: a tile →
  pool-page table (the directory) is the GPU view; the pool view is each
  page's one home. Heap classes from the title's own protections: 0x22004
  CPU-coherent (RAM in place), 0x2404 write-combined (streamed once → card
  candidate), 0x42404 the one GPU-writes/CPU-reads page (fences/labels),
  CPU-only shared pools never GPU heaps. Series S is the cheapest profile
  (pool 3,977 MiB, heaps 8.4 GB). Frame interval comes from data (30 fps on
  S; table supports 60/120). Only three command-line switches exist.
- 2026-11-25: indirect bounding: unbounded and "bound only after a drop in
  the same list" both hang the GPU at frame ~1,440 (Xid 109); bounding every
  shader-writable argument buffer never did, but costs ~0.18 ms per call.
  f3bf739 reverted; making the bound cheap instead. The watchdog now also
  stops the game on any NVIDIA Xid.
- 2026-11-25: the title's ray-tracing builder stays off: 3 of its 53 programs
  don't translate (wave64 first-lane reads); being fixed.
- 2026-11-25: the world had rendered black since run p25: draws whose
  pipeline was not made were dropped (3 ms budget), skipping one-off and
  temporal passes (exposure). Never dropping stalled the title, whose own
  watchdog crashed it ~30 s on. Answer: bake used pipelines (first-use
  order, light: <=4 threads in game, ~1 GB RAM, released after caching),
  before launch (`victory bake`) and at start; a bounded wait at draws.
- 2026-11-25: DECISION (user): Vulkan direct. Translator rewritten to
  emit SPIR-V (buffer device address for raw title addresses, descriptor
  buffer), victory-gpu gets a Vulkan host; D3D12 kept behind a switch
  until parity. RAM target: <=12 GB, then ~10 GB (Series S baseline).
- 2026-11-27 (night): the batch bake, and why two of them were lost.
  - **Every pixel classed as sky**: CS_GenerateMaterialMasks tests the main
    depth's stencil with a float compare; the stencil integers are
    denormal bit patterns and the card takes denormals for zero in
    compares (measured with a native test; the device cannot preserve
    them). Float compares and class tests are made on the bits now
    (oracle `vk/oracle/card.rs`; the interpreter has a card-float mode,
    `CARD=1` in the replay). Arithmetic and min/max still flush denormals.
  - **Batch** (tag `batch-2026-11-27`): the twelve audit fixes + the
    compare fix + the fingerprint scope. Phi variables (117f64d) reverted:
    the "smaller" ActivateRays module (2.3M words) made NVIDIA's compiler
    use over 12 GB and run past 6.5 minutes (3 minutes at 3.4M words).
  - **A bake that is killed loses everything**: the driver writes its
    cache index (`*.toc`) at a clean exit only. Bake 1 was OOM-killed by
    its own memory cap (the phi module), bake 2 by a timeout set too
    short; both games after them recompiled every stage in game and
    showed black. A bake needs a timeout far past its hour and must end
    by itself. Also: write tracking's fault-cost probe ended a bake (no
    fault handler there): it runs only once a real store has faulted.
  - Open: in game, on-demand stage loading stalls behind slow compiles
    when the driver cache misses (translated stays at a few hundred).
  - 64-lane approximations in the used set: 76 pixel programs (character
    shading, particles, mirrors); the wave64 and waterfall tests of
    origin/main are not ported yet, and the interpreter runs a wave as
    one, not as the two halves the card runs.
- 2026-11-27 (early): what kept the world grey, found with origin/main as
  the reference and with pass captures replayed offline.
  - **Reference**: origin/main (the D3D12 path, lit on NVIDIA) checked out
    read-only at /mnt/data/victory/scratch-agent/forks/origin-main;
    docs/origin-findings.md and docs/opcode-audit.md compare its lowering
    with ours. Twelve translator fixes came of it, each with an oracle
    test, none baked yet (tag `baked-2026-11-26` is the cache's translator):
    pixel kills were never emitted; multi-dword loads bounded as a whole;
    immediate samplers never compared; whole-wave sections with lanes
    missing; position w; float-to-int saturation; readfirstlane with an
    empty EXEC; gather_l at its level; no hit with the builder off;
    unwritten registers; mad_legacy. Golden: 69 of 1,527 programs changed,
    each explained.
  - **Host, found by captures and replay (no bake needed)**:
    - The fog inscatter is shared-exponent words in a 32_UINT image; its
      R9G9B9E5 view fell back to R32_UINT and read 2e9 in red: now an
      E5B9G9R9 view. The HDR target is finite after the fog pass.
    - One stage still compiling made every dispatch wait out the frame's
      budget (a second a frame): a stage waited out once is skipped until
      ready. 1.7 -> 9 frames a second; the card (105 ms) is the limit.
    - Draws with no targets at all (fog volume and light proxies: storage
      writes only) were dropped; drawn over their scissor now.
    - The tiled lighting selects pixels by depth EQUAL against a depth
      buffer (0xb082a0000) that only CS_GenerateMaterialMasks writes,
      through storage views of its planes. A depth image has none: a
      colour image per plane (32F depth, 8-bit stencil) takes the stores
      and is copied in before the buffer is attached. Still open: what
      the program stores where (integer bits at one view) and that the
      test then passes; the lighting target stays black until it does.
    - Dump readbacks were read (and freed) before the card ran their
      copies: copy-engine faults (vk82, vk87). They settle four frames.
  - **Write tracking** merged (track.rs; pages with a card copy are
    read-only until written; ~7 ms a frame; fingerprint polling only for
    what it cannot track).
  - **Tools**: `cargo xtask shader-golden`, `examples/replay` on
    `pass-<frame>-<program>/` captures, recording time by call on the 10 s
    line.
- 2026-11-26 (rule, the user's): **no bake per translator change.** A
  translator change ends with offline evidence: the instruction oracle
  (vk/oracle), the golden behaviour of every used program under the
  interpreter (which programs changed, and why), spirv-val, and replays of
  captured passes. A bake is made for a verified batch only; host-only
  changes never need one (they leave crates/victory-shader alone).
  - **Pass captures** (vulkan/passes.rs, diagnostics): with
    `VICTORY_VK_DUMP_AT=<program,…>` one run writes, for each program's
    first execution, `pass-<frame>-<program>/` in the capture folder:
    meta.txt (programs, root parameters, push constants, the draw or
    dispatch, state), mem/ (the tables and the memory they point at) and
    img/ (level 0 of every image its T#s name, as the card held it). The
    interpreter replays it: a pass is examined without title or card.
- 2026-11-26 (midnight): the fog pass, tables, float controls.
  - **CS_ApplyCombinedDeferred** (fog apply) has finite inputs (census of
    every float image, volumes whole, just before it: none holds an
    infinity or NaN) and leaves a per-pixel red stipple over the sky.
    Reading: `v_mul_legacy_f32` of inf by 0 (a sky tap's weight) is 0 on
    the console; our select form is only that while the driver keeps the
    `== 0` tests, and no float controls were declared. Every module now
    declares SignedZeroInfNanPreserve (compile time unchanged on NVIDIA).
    To confirm in a run.
  - **Constant tables in a program's own code** (`s_getpc_b64` + offset):
    368 programs of the pack read zeros for them (blur taps, jitter
    angles, filter kernels); now a constant array in the module.
  - **Pool pages stale on the card**: half the sampled pages (1,997 of
    3,786) were changed by the CPU after the card's copy. Write tracking
    (protect pages with a card copy, copy again on write; no extra RAM or
    VRAM; replaces the fingerprint polling) is being built: handoff in
    docs/vulkan-host.md.
  - **Frame**: the card 100 ms, the CPU ~550 ms (recording): profile next.
  - Removed: VICTORY_VK_NO_WIDE, VICTORY_VK_POOL_RESYNC. A first stage's
    user data ends at entry 28 (no layout of the pack goes past 20).
- 2026-11-26 (night, later): measured.
  - **Wave-uniform values need no exchange** (vk/uniform.rs): mesh
    exchange calls 7,607 -> 1,094 over the used set, no counted barriers
    left, 157 of 240 mesh programs exchange nothing. In game (vk80): the
    card's time 670 -> 100 ms a frame, no fault, frame 2,160 reached. The
    frame is now the CPU's: recording takes ~1.4 s of thread time a frame
    in a diagnostics build (join 547 ms): next to measure in a release
    build.
  - **The oracle** (vk/oracle, the other fork's valu tests through our
    interpreter): 6 of 30 passed at first, 21 pass now, 11 ignored with
    reasons. Fixes: a read hoisted above a store in one EXEC guard,
    normalized buffer codes an ulp off, v_med3 with NaN, v_fract returning
    1.0, v_sin/v_cos of large turns, v_bfe_i32 computed fields,
    v_cmp_class, f16 omod, d16 integer loads, v_cvt_pkrtz rounding (inf
    past 65504). Host: A8 views read their channel as alpha (glyph boxes?).
  - **Lighting**: the first non-finite HDR texels are written by
    CS_ApplyCombinedDeferred (fog apply: inscatter + hdr * transmittance);
    the program is exact in the interpreter, so a fog input's R holds the
    NaN: the volumetric-fog writers are next. The glass passes spread it.
  - **Known gaps found**: constant tables read through s_getpc_b64 read
    zeros (being fixed); unbound pages of the pool's sparse view fault the
    card when a shader reaches them (why the range's-end V# rule faulted);
    a mesh program's user-data entries 28-31 overlap the pixel window.
  - **Title crash** (null-based reads, ~1 run in 5): xmem cleared (no
    remap, stand-in or failed map in the failing runs; the null comes from
    title heap memory). The crash report now carries the last mapping
    operations per page.
- 2026-11-26 (late): the world, after the shader piece.
  - **Read another fork** (Belobug/VIctory-Belobug-AMD, same upstream and
    licence; D3D12, a right picture at ~22 fps on an RX 7900 XTX, ~30 GB of
    RAM). Taken as ideas, no code: its instruction-level tests as an
    offline oracle for our translator (being ported), and three facts
    about the title: its indirect light and lamp/sun shadows are software
    ray traced by its own compute programs; a partly working structure
    builder (XboxBvhBuilder.dll) writes garbage into title memory and
    crashes the title; on NVIDIA its frame is ~12x AMD's because of wave64
    on 32-lane subgroups.
  - **The structure builder's dispatches are left out** unless
    `VICTORY_BVH_BUILDER=1` (312,000 a run). Since then: no title crash, no
    guard hit, card time 890 -> 670 ms a frame.
  - **Reverted**: the V# "reaches its range's end" rule (faulted the card
    at world start, bisected); spilled user-data entries as push dwords
    outside compute (a draw's stages share the push constants).
  - **Diagnostics that found things**: `VICTORY_VK_DUMP_TARGETS` /
    `VICTORY_VK_DUMP_AT` (the G-buffer and depth are right mid-frame; the
    HDR target's R and A are inf/NaN after the tiled lighting while G and B
    are sane), `VICTORY_VK_GPU_TIME` (85% of a frame's card time is five
    mesh program pairs).
  - **Mesh cost, the disassembly's answer**: MS_VS_Transform has no real
    cross-lane flow (0 LDS, 5 readfirstlane all of wave-uniform values, 2
    barriers that only order the output count), yet we emit exchanges,
    polls and counted barriers for it. Being fixed by a wave-uniformity
    rule in the lift.
  - **RAM**: 21.8 GiB peak -> 16.5: no pipeline cache of ours (the driver's
    disk cache serves; ours was 2.3 GiB read whole), modules destroyed once
    their part is made. Title memory itself is 9.2 GiB of it.
  - **Open**: the lighting inputs (R/A inf: a shared factor is inf where
    G/B inputs are 0; the shader itself checks out in the interpreter);
    glyph boxes; ~700 pool pages a frame the CPU changes after the card
    copied them (write tracking planned, no extra RAM); 670 ms of card
    time a frame.
- 2026-11-27: write tracking for the card's copies of title memory
  (`track.rs`; `docs/vulkan-host.md` "Write tracking"). Pages under a
  pool page on the card or an image filled from title memory are
  write-protected in every title view; a store faults once into `xmem`'s
  handler, the next flush (before every submission and present) protects
  again and has the copies made again. One byte of state per tracked
  64 KiB page; no mirror. Kernel writes (file reads, the driver's pinning)
  open their range first. Deleted: the stale-page census (`stale.rs`), the
  fingerprint of every tracked image at every fill; fingerprint polling
  stays only for textures the card writes too (2,050 of 19,972 images).
  World, a frame: 130 to 160 faults (2 to 6 us each), 25 to 105 pool
  pages copied again, ~19 flushes costing ~3.5 ms, ~3 ms more of image
  fills; RAM and card memory as before (15.2 to 16.2 GB, 6.4 to 6.6 GB).
  Open: the ~3 ms target (one submission a flush, small images filled
  only if changed).
- 2026-11-26 (night): the shader piece, towards "complete".
  - **Indexes**: `docs/translation-index.md` (xtask, every container of a
    pack: coverage, refusals, approximations) and
    `docs/pipeline-index.md` (`victory bake`: every stage's compile).
  - **`victory bake`** translates what the title ever used straight from
    its packs (so a translator change no longer empties it), compiles
    smallest first, records every outcome, and never hangs: compiles over
    60 s and modules over the words budget go to a one-at-a-time slow lane.
    Words compiling at once: 300,000 (1 M took 14 GiB in NVIDIA's compiler
    and the bake was killed for it).
  - **The 23-minute compiles** (CS_RaytracingResolve, CS_DummyResolve,
    decal classification, CS_GatherAlphaVBuffer): NVIDIA analysed the loop
    guard's constant "+1" counter shared across nested loops. The step is
    now a value the driver cannot see through (always 1): 0.5 s.
  - **Loops on correct data**: `ir::without_skips` removed EXEC-empty
    skips around loops; a wave with no lane walked into a loop it could
    never leave (culling programs, PS_Textured_Eight). Guard hits in a
    run: hundreds of thousands -> 1 program.
  - **Mesh output memory**: every mesh program fits NVIDIA's 32 KiB (ABI
    v4: split parts, counted barriers).
  - **Translator coverage** (25,889 containers): refused 218 -> the 195
    geometry-stage containers the title never requests; invalid 6 -> 0
    (constant texel offsets as constants, VK_KHR_maintenance8 for the
    rest). Added: f64 add/mul/convert, packed 16-bit shifts, d16 LDS
    halves, float atomic min/max as compare-exchanges (NVIDIA lacks the
    extension), POPS wait (ordering not kept), unreachable blocks.
  - **Stage cache key** no longer hashes stage_cache.rs itself.
  - **Images**: a texture made over an older one's memory drops the older
    image (35,000 live -> 20,000; the ceiling stopped refusing); textures
    the CPU changes are found by a backoff schedule, not 256 a present in
    turn (new glyphs showed as stale boxes for a hundred presents).
  - **Open (shader side)**: 64-bit image atomics (679 alpha prepass
    programs run on the low dword only), clip/cull distances (1,804),
    typed buffer stores as 32-bit channels (1,096), wave64 mask leftovers.
  - **Open (world)**: the frame is flat grey; ~2 fps (direct queue 440 ms
    of GPU a frame, record 1.4 s of CPU a frame).
- 2026-11-26 (evening): Vulkan only (D3D12 removed, ~77k lines). Found
  and fixed on the way to the world: xmem treated 1,348 fresh pool pages
  as zero "fillers" (the title's recurring null-pointer crash and stale
  culling inputs); imports freed while recorded copies still named them
  (Xid 31); stale/zero/2D-only descriptor slots (Xid 13 Illegal
  Instruction Parameter; now robustness2 null descriptors); zero-group
  mesh dispatches; images over the ceiling (texture quality low on <10 GB,
  driver memory counted); glyph-cache refills (whole-texture fingerprint
  up to 4 MiB); the stage cache and pipeline cache now hit across runs.
  Translator: real BVH intersect, exact wave64 one-wave exchange, group
  barriers order buffer memory, v_min3/max3_f16, float atomic min/max.
  Open: 9 world/character mesh programs over NVIDIA's 32 KiB mesh output
  memory (refused, so the world is black); culling loops on correct
  inputs (being traced); a GPU hang at world start; compile times of
  seconds per big program (cache makes later runs fast).
- 2026-11-26: the white world (vk32, frame 2,048): the world is mostly not
  drawn: 2.8M draws/dispatches skipped as not compiled against 1.6M run,
  2,376 stages queued. Skipped one-off passes (exposure, histogram, LUT)
  and an empty scene blow auto-exposure to white. In-game compute stages
  compile at 860 ms (one at a time) against ~120 ms for the same modules
  offline (spvtime, with descriptor buffers, spec constants and the disk
  cache), 89% inside NVIDIA's compiler; and the driver disk cache does not
  hit between runs (per-run spec constants suspected). Culling loops
  (*_BatchCull) still hit the guard: the old path's whole-wave section
  lowering (emit/wave.rs) was not carried into vk/.
- 2026-11-26: world load on Vulkan hangs the GPU (Xid 109), also with
  indirect off. Pool garbage fixed (a map failing halfway left pages
  showing other pages' bytes); texture quality now applied on Vulkan. The
  per-loop guard is back (ABI v3: 2^18 compute / 2^16 other iterations,
  a guard buffer naming the program): removing it was a mistake. Run vk31
  named 16 looping programs, led by the *_BatchCull meshlet-culling
  family, PS_Textured_Eight and DefragHeap; suspect the skipped wave64
  cross-half approximations (64-bit mask ops reading one half). Still an
  Xid 109 at ~frame 1,380.
- 2026-11-26: merged into integrate (720b18b): per-title-queue lines on 3
  queues with card-written bottom-of-pipe values and timestamps,
  ExecuteIndirect with a bounding pass built by vk::spirv, FSR4 as a linear
  blit on Vulkan (gated on the FFX DLL version), console list slots
  (UAV clears, fills, DrawIndexedX, predication, top-of-pipe writes,
  KickoffX, depth bias), real occlusion/timestamp queries, run-time
  layout detection, up to 8 present planes, kernelx exports and console
  VirtualAlloc flags. Translator: lever 2 reverted (no general gain, 39f0
  8.8 s vs 145 s); compute sample p50 15 / p90 179 / max 250 ms. Run vk27:
  0 errors, record 20.5 ms; killed by the VRAM watchdog at ~frame 1,024
  (the process jumped 3.1 -> 6.4 GB in 3 s at world-load start while our
  accounting read 2.8 GB): under investigation.
- 2026-11-26: ABI v2 merged (translator 5d4ea8d, host 1f1c2a4): per-title
  semantic registry beside the stage cache, cube 2D-array views on set 0
  binding 2, T# check word, generic DPP. Used set 1,527/1,527 translate.
  Loose imports joined: map ranges 33, refused 0 (was 64/188). Vulkan
  runs vk23/vk23b: account screen, Bink video, loading art with its navy
  bars; 0 errors, 0 crash reports. Card memory of ours steady at ~2.8 GB
  (images 2.56 GB); nvidia-smi per-process 3.9-6.8 GB between runs, a
  driver-internal gap (likely shader local memory) still to explain.
- 2026-11-26: Vulkan run vk19 (b08fedc): the account screen (navy, ACCOUNT
  PICKER/PLAYER) and the Bink intro video render. Fixes: CBV V# stride 16
  (Scaleform batch matrices read row 0), rectangle lists drawn, the
  fingerprint refill reads pool memory through its CPU view (green =
  YUV(0,0,0)), draws wait up to 2 s a frame for a compiling stage. Warm
  launch: 303 stages from disk, waits 3.7 s, record 30 ms + fills 9 ms a
  frame. Fork comparison (7900XTX, XCL, Leonida; notes in
  scratch-agent/compare/notes.md): ours is the smallest (~148k lines,
  ~96k after the D3D12 path goes; XCL 168k, Leonida 274k, 7900XTX 301k)
  and the furthest on this machine. Genericity gaps queued: semantic
  locations fixed to this pack, cube derivatives (6-layer 2D array), generic
  DPP, wave64 cross-half ops, T# slot check, missing console list slots,
  fake queries, slot layout from one signal, 2-plane present, FSR4 with no
  Vulkan path.
- 2026-11-26: HLE genericity (victory-hle): kernelx.dll and its legacy and
  app-model parts resolve as the PC kernel (our modules first, then the
  host by name), with kernelx's own exports mapped onto what we have
  (title memory status from xmem's figures, title physical pages as XMem's,
  allocation hooks, constrained-change registrations, thread names as
  thread descriptions, console type from `VICTORY_CONSOLE`, OS version);
  `GetProcAddress` on it resolves the same way; `VirtualAlloc` with console
  flags goes the XMem way; `minuserx.dll` is user32, `xgameplatform.dll`
  and `xgameruntime*` are ours; unknown XGameRuntime classes go to the
  PC's own runtime where it exists. The routes alone booted unchanged in
  vk24 (29/29 unimplemented-import lines). What remains: runtime areas answer only
  the interface ids of two GDK versions (Minecraft 2022, this title), so
  another GDK's title meets NOT_SUPPORTED; title memory status counts the
  whole process as used (reads full); `VirtualProtect`/`VirtualAllocEx`
  with console bits still reach the host; the console type values are a
  guess; no XGameRuntime fallback without Gaming Services.
- 2026-11-26: FIRST IMAGE ON VULKAN (run vk16, host f9b0165, translator
  b252e39): the loading art renders through victory's own Vulkan path (no
  vkd3d). Black-screen causes found on the way: separate VkImages per view
  of one memory (now one image per memory), vertex attributes at
  APPEND_ALIGNED all read at offset 0, a pipeline-statistics query writing
  4x its buffer (report() stack corruption), and the 38 MB descriptor heap
  imported only in its first committed 10 MB (the tables sat in a section
  view behind it, so every transform read zeros and gpu_fs was 0). Still
  missing against D3D12: the background fill and panel mask (frame 960),
  the star logo (frame 450 shows solid green), early startup content.
  Translated SPIR-V now cached on disk (warm launch: from_disk=376).
- 2026-11-26: KISS audit of the kept "console side" (partly true):
  KISS as is: gfx10 decoder (sp3.rs aside), native.rs, io.rs, spill.rs,
  wave SSA + lift, frame.rs, fence.rs (once its Signal pumps Vulkan),
  xmem's console API, dstorage.rs, d3d12x.rs. Not KISS: the vk/emit.rs
  placement (Ramsey one-iteration loops at every merge -> 220 nested
  constructs, 156 s compiles; replace with post-dominator selection
  merges, ~300-400 lines), the per-loop iteration cap and seven VK_*
  switches; host plumbing still carrying D3D12 machinery (list.rs ~1,600
  D3D12 lines, memory.rs/unified/views/shadow/gather/descriptor-host/
  memory-cache -> ~0 under Vulkan, indirect.rs windows), duplicated
  mechanisms (held line in queue.rs and vulkan.rs; two card-page caches;
  dominators thrice), serial timeline (all queues one after another),
  leaks (descriptor mirror pending, Vulkan images never freed, map not
  pruned on unmap). Vulkan plane 0 black: one VkImage per address+shape
  means views of the same memory get separate images; fix: one image per
  memory, mutable formats. After parity: delete ~37k translator lines
  (emit/, hlsl/, resource.rs, structure.rs, bvh.rs) and ~15k host lines.
  Code-shape limits (user): files 1,000 plan / 1,500 mandatory; cognitive
  15/25 per function (cyclomatic >10 warns only); `cargo xtask quality` + baseline.
- 2026-11-26: real NVIDIA compile times (native Vulkan, cache off, 68
  compute modules by size): new SPIR-V path (pre-OpPhi) p50 25 ms, p90
  538, max 1,043; old path (dxil-spirv) p50 84, p90 414, max 2,447. The
  bake's 7.7 s a pipeline was mostly our re-translation and vkd3d's
  DXIL->SPIR-V conversion, not NVIDIA. Vulkan M2a under Proton (run vk1):
  device, present, sparse pool view and imports work, the title reached
  frame 256, then the device was lost (present waits timed out, then
  vkQueueBindSparse: device lost); being fixed before M2b.
- 2026-11-26: the D3D12 bake measured the real problem: 2,024 used
  pipelines took 1,110 s on 14 threads (~7.7 s of NVIDIA compile each,
  4.3 CPU-hours, 6 GB peak). Bake stopped (user): the translator is the
  cause. Focus: the KISS SPIR-V translator, target <=0.1 s a pipeline.
  `victory vkprobe` under Proton: 11/11 (host-pointer imports with device
  addresses, coherent section aliases, descriptor buffer, descriptor heap
  in title memory, pipeline libraries: fast link 0.004 ms). Imported host
  memory cannot back a sparse buffer, so: a 64-range map over imports for
  CPU-written memory; the pool's GPU view as a sparse buffer of card pages
  bound per page-mapping call and filled on announced writes (console's
  own design). Display mode now the PC's screen (was 4K).
- 2026-11-25: runs p21-p28. Pipeline waits (500 ms a frame) removed:
  menus ~26 fps. The physical-vs-DXGI card gap (~2.2 GB) was NVIDIA local
  memory for per-lane code-constant tables (up to 21 KB a lane x 55k
  lanes); tables trimmed to data only: p28 ran the full 5 minutes, the
  game's card memory peaked at 4.8 GB (was 6.4-7.2, watchdog kills).
  World ~95 ms/frame with ~1,300 draws a frame still dropped for
  pipelines being made (largest compile several seconds each; ~1,100
  queued after 5 min) and GPU busy ~90 ms. Next: one access path per
  buffer access (offline: -59% code), then GPU time.
- 2026-11-25: run p20 (thin recording on 8 workers, pool by directory,
  texture copy queue, builder translating): 294 ms/frame (~3.3 fps, from
  ~560); title threads 1 ms in our layer; windows 0 MB; GPU busy 84 ms;
  workers 458 ms summed, submissions wait 222 ms for translation. Next:
  descriptors read from memory by shaders (per-call binding → constants),
  a VRAM cache of streamed pool pages, then splitting big lists across
  workers. Leonida built here crashed ~20 s in, twice (heap corruption;
  title assert).
- 2026-11-25: dropped the shader control-flow reshaping pass (not needed
  with a current vkd3d-proton).
- 2026-11-25: pool memory goes by directory (workstream 24): the one way
  the GPU sees title memory outside opened buffers. Static analysis of the
  executable: the GPU view of the pool is filled only by
  `CopyPageMappingsBatchX` (one page per range) and never touched by the
  CPU, so it gets no host views at all (the ~62k single-page views); its
  entries come from the batches, and victory's CPU reads of it go through
  the sections' whole views. Windows stay only as the fallback, without
  copy-out.
- 2026-11-25: pool views lowered once (workstream 24): a raw or
  structured access picks the view's slot or the directory's copy by
  selects and makes one access; the directory is looked up once per view
  and dynamic offset (both pages in one load), uniform lookups kept for
  the blocks they dominate; pool views read zeros past their bytes.
  Constant views keep their rows beside the directory's raw read (no raw
  read of a constant buffer view on the host). Shaders run p27 used:
  40.2M → 28.0M SPIR-V instructions (no directory at all: 16.3M); whole
  pack 25,639 translated as before, DXIL 1.94 → 1.52 GB.
- 2026-11-25: VIctory-7900XTX study (GPL-3.0, code reusable with notice).
  Taken: list calls written down on the title's threads and translated by
  workers of ours (their `VICTORY_THIN`; here the only path, into the
  title's own list, the submission waiting for its lists instead of their
  line of submissions); first uploads on a copy queue with slots shown
  after its fence (their `VICTORY_UPLOAD_AHEAD` + `VICTORY_SHOW_AFTER`,
  which ended a device loss every ~15 min there). Not taken: their async
  submit thread and joined parts, relay, compares, card steps (mirroring
  machinery this tree deleted).
- 2026-11-25: process memory (workstream 26). Linux counts a page at each
  address that maps it: the pool's sections are mapped by the title's CPU
  view and by victory's whole view of each section (its own writes), so
  the 16.7 GB resident of the last full run may count up to the
  sections' 4.4 GB twice. The new line gives both counts; the target is
  read on the unique one. Title memory opened as heaps is pinned, so
  resident whole, used or not. Not done, and why: the 1.8 GB of
  "textures the title fills" are the card's memory, not the PC's
  (staging is freed after each upload, ~30 MB alive); pool pages cannot
  be discarded once on the card, since the title's CPU view reads the
  same pages, textures let go for room are detiled again from them (7.4
  GB let go by idle eviction in that run), the card page cache refills
  from them and the GPU reads uncached pages in place;
  translated shaders kept for pipelines (`pipeline.rs`) are counted
  (owner "shaders") but left to the pipeline work.
- 2026-11-25: pipelines baked like a PC game's shader compile (workstream
  26): only pipelines earlier runs drew with (about 3,000; the title
  creates tens of thousands), in first-use order, from the pipeline
  catalog; the title's own data gives the shader containers
  (`shaders.rpf`) but not the pipelines (which shaders pair, what state):
  the title builds those in code at run time, and only the AMD upscaler
  deserializes ready pipelines (ACO code, untranslatable). Draws wait up
  to 2 s a frame for a pipeline (dropping after 3 ms, efa4886, left the
  world black). `CS_DefragHeap_ComputeRelocations`,
  `CS_AlphTransmissionPrefilterMip` and `CS_Visibility` are refused
  (E_INVALIDARG) because their translations declare `WaveSize(64)`,
  which vkd3d-proton refuses on a card of 32-lane waves: wave64 programs
  that share LDS within the console wave inside lane branches (a read
  and a write in one branch region, e.g. one lane updating a counter the
  next loop pass of the whole wave reads) cannot be split into two host
  waves ordered by group barriers. Needs the translator to order the two
  halves of a console wave without the group (pair meetings in
  wave-uniform code); not yet done.
