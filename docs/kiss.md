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
| Streamed textures: placed with every level, the levels held said by the view the title writes into its descriptor slot | A card image of the levels the latest view names, alive while a slot of the title names the texture; a host slot stays its texture's | `victory-gpu/vulkan/images/{levels,named}.rs`, `descriptors.rs` | works |
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
| Rust lines | 128,292 | 100,007 on 2026-11-27, about 10,000 of them tests and tools (78,027 right after the D3D12 back end went; 154,557 just before) |
| `VICTORY_*` switches | 224 | 39 |
| `diagnostics`-only blocks | 283 | 71 |
| World frame time (Proton, 8 GB card) | 0.5–5 s | ~440 ms on the card with the whole scene lit and textured (2026-11-27, run tx7); ~295 ms on 2026-11-25 with most of the picture missing |

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
| 14 | Series S world: the title's FSR4 upscale (console-only ACO code) runs as a plain linear scale at `ffxDispatch` | subagent | merged (977868f): Series S world renders lit (run s7, frame 1890); 2026-11-27: its 29 passes translate (branch `fspx`, offline proof; the scale stands in only while one does not): not yet run |
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

- 2026-11-27 (bench): **where a lighting pass's time went: helper pixels
  spinning to the loop guard. One condition in the translator; 26.6 ms a
  draw become 0.74.** Found with a native pass bench, after three
  readings of the card's time had failed (the map's search, where memory
  lies, the card's power). Branch `passbench`; nothing baked, no game run.
  - **The bench** (scratch `card/passbench`, native Linux Vulkan on the
    test card, built on `victory-vk` and this tree's translator): one
    captured pixel pass as the game runs it. The capture's vertex and
    pixel containers translated now; the host's rectangle stage behind the
    vertex program (the tiled lighting draws 60 rectangles of 128 x 120
    pixels, 180 vertices); library parts and a fast link with the game's
    dynamic state; the capture's images in their descriptor slots; its
    title memory as sparse buffers under the game's own memory map (card
    pages where captured blocks lie); its push constants, targets and
    depth (test EQUAL, no write). Timed by timestamps around each of 20
    draws; the first draw's targets compared with the capture's `after/`.
    - `TL_PS_PBR_MATERIAL_STANDARD` (run tl2): 187,246 pixel invocations
      (the capture's draw changed 187,384 texels); of the 182,351 texels
      both changed, all but 57 within 2% and 92% equal to the bit.
      **26.5 ms a draw against the game's 22.**
    - `TL_PS_NoDir_PBR_MATERIAL_STANDARD` (run cm-a): 322,759 texels,
      every one equal to the capture's to the bit. **34.7 ms.**
    So the pass is as slow by itself as in the game: nothing around it.
  - **What did not move it** (the first pass, 26.5 ms): one pipeline
    instead of parts and a fast link 26.7; no bounds select on each loaded
    dword 26.6; no float controls 26.5; the memory map as one range (the
    one-compare form, title memory under one sparse window) 26.5; votes
    of the lane's quad 28.5; helper lanes without a vote 21.4; maximal
    reconvergence declared 26.5; `spirv-opt -O` 25.1. A flat pixel
    program in its place: 0.01 ms (the draw itself costs nothing). The
    legacy multiplies as plain ones: 23.6. The hit store of the loop
    guard left out: 23.6.
  - **What did**: the loop guard at 256 iterations instead of 16,384:
    **1.1 ms, the same texels**. So the time was loops running to the
    guard in lanes whose work nobody sees: no live pixel's result changed
    and no guard hit was ever stored. With the guard's edge never taken
    the device is lost (those loops do not end). Votes per lane: 0.51.
    Helper invocations returning at the program's start: 0.58, the same
    texels: the lanes are the helpers.
  - **Why.** The tiled lighting walks the lights of the wave: the union
    of the lanes' masks, bit by bit, each bit taken with
    `v_readfirstlane` from the first lane in EXEC. A helper lane (a pixel
    of a quad that failed the early depth test or lies past the
    rectangle's edge; about 11,000 in 6,400 subgroups of this draw) is in
    EXEC at no time in these programs: they have no `s_wqm`.
    `lift_valu.rs` gave **every** helper lane its own value for a
    first-lane read (taken over from origin/main, where a helper lane is
    in EXEC in whole-quad mode and branches are per lane). Here a helper
    lane's own copy is never written (every write is masked), so at the
    loop's scalar branch ("was that the last bit?") it went another way
    than the wave and stayed in the loop alone until the guard: 16,384
    iterations a helper lane, their stores dropped by the card (the
    guard's too: "none fired").
  - **The change** (`lift_valu.rs` `valu_lane0`): a helper lane takes its
    own value for a first-lane read only **where it is in EXEC**; outside
    it takes the wave's, as the console's one register holds it.
    Bench: `TL_PS_PBR_MATERIAL_STANDARD` **26.6 -> 0.74 ms**,
    `TL_PS_NoDir_PBR_MATERIAL_STANDARD` **34.7 -> 0.75 ms**, the same
    texels as before in both (the second still equal to the capture's to
    the bit). `spirv-val` passes on both modules; `cargo test -p
    victory-shader` 186 passed (new:
    `a_pixel_helper_lane_outside_exec_reads_the_waves_first_lane`; the
    interpreter has helper lanes for it, `Machine::helpers`); shader
    golden 1,527 programs, 0 changed behaviour.
    **Bake: 216 of the 414 used pixel programs change** (every one with a
    first-lane read: 130 `TL_PS_*`, the deferred mesh draws' textured
    programs, alpha prepasses, decals); no vertex, mesh or compute
    program does.
  - **Returning helper invocations at the start** of a pixel program
    without `s_wqm` and without implicit derivatives was tried on top
    and left out: 0.58 instead of 0.74 ms in the first pass, 1.12 instead
    of 0.75 in the second.
  - **What is left in the pass** (0.74 ms; one thing at a time, the same
    texels each time): the memory map as one range **0.32** (the search
    of 6 steps at each of 127 address translations is now more than half
    of the pass); legacy multiplies plain 0.50; no bounds selects 0.51;
    no guard store 0.54; votes per lane 0.51; guard never taken 0.72;
    all of them 0.29. No depth test (345,586 pixels instead of 186,453)
    1.27.
  - **Expected in the game, not yet run**: the tiled lighting family
    (104 ms a frame in about 40 passes) to a few milliseconds, so the
    frame from about 286 ms toward 185: 5 fps, not 30. The mesh draws'
    pixel programs (27 ms) have first-lane reads too and may gain. Compute
    (97 ms) has no helper lanes: its time is not explained by this, and
    the bench does not run compute passes yet (the next step: the same
    table for `CS_Primary_GatherCopy`, starting with the memory map as
    one range, which only now shows as a cost).
  - **The other fork's translation of the same container** (VIctory-NVIDIA,
    D3D12/DXIL; scratch `card/tn1` with its own census): 8,048
    instructions in 193 blocks against our 14,864 in 600 blocks and 6
    functions (the second pass: 6,860 against 11,781).
    | | theirs | ours |
    |---|---|---|
    | branches | per lane (the host's own control flow); a wave vote only where every lane must act | flat guards and votes of the subgroup: 26 `Any`, 14 ballots, 28 shuffles (theirs: 31 wave operations in all) |
    | selects, phis | 1,179 (8 of them guards), 250 | 2,438, 819 |
    | buffers | real descriptors from the heap (`ResourceDescriptorHeap`): 97 handles, 149 buffer loads, 36 constant-buffer loads, no address translation in this program (title memory by a 64 KiB page directory only for raw pointers) | 662 loads through 127 address translations, each a 6-step search of the memory map, and a bounds select per dword |
    | legacy multiply | at most one compare and select (none where a factor is known finite or a non-zero constant) | two compares, an or and a select each (829 compares) |
    | loop limit | 2^20, a counter per loop (9) | 2^14, a counter per loop (9), a store on a hit |
    | helper lanes | branch by their own conditions like any lane; own value at a first-lane read (`helper_own`) | outside EXEC, following the votes; the wave's value at a first-lane read (this entry) |
    So theirs never had this cost: a lane there leaves a loop by its own
    condition. What remains between the two after this change is the
    static size (1.8 times) and the address translations.
  - Tools in scratch `card/`: `passbench` (with `rect_gs.spv`, the host's
    rectangle stage; `translator-probes.patch`, the switches the tables
    above were made with, never committed; `probe/` the lane and vote
    probes: on this card a subgroup vote gives helper lanes the live
    lanes' answer, and helper lanes are flagged as such), `trb-new` and
    `trb-base` (module hashes of the used set under two trees), `tn1`.

- 2026-11-27 (night): **the merged translator batch holds 3.1 fps without
  the upscaler; the title's own upscaler is off by default; the card does
  not idle between lists.** Tag `batch-2026-11-27f`, baked in full
  (3,215 s), then runs bf1, bf2 (upscaler on) and bg2 (upscaler off).
  - *The title's upscaler* (FSR 4: 29 translated compute passes, a neural
    model): 0.2 fps, the direct queue busy 3 to 9 s a frame, the picture
    black, `fsr4_model_v07_fp8_no_scale_pass9` at its loop guard every
    frame (count 226,944), 153 "submission has not finished" reports in
    one run. The offline checks could not see this: they prove an
    instruction's result, not a pass's time on the card nor that its waits
    end. The linear scale is the default again; `VICTORY_FFX_OWN=1` runs
    the passes (`modules/ffx.rs`, `docs/switches-hle.md`).
  - *The rest of the batch* (fewer LDS meetings, ordered pixel sections,
    typed-load bounds, the 16-bit image store), run bg2: 31 frames in
    10 s as before, card 289 ms a frame (286 before), no hang, no guard
    hit, no error, the picture right (capture bg2, frame 2340). Correct,
    and no faster.
  - *Is the card computing or waiting for us?* An outside review asked
    whether the frame is card time or the card idle between submissions
    (158 a frame on the direct queue) while the CPU faults, copies and
    serialises. `VICTORY_VK_GPU_TIME` charged the time between two
    timestamps to the earlier program, so idle time between lists would
    have been charged to a program. A list's end is now a timestamp too
    (`host: between lists`): 5.85 ms of 289 ms a frame over six frames
    (909 list ends). The card is inside the programs for the rest. The
    CPU side of the same run: about 280 write faults a frame (0.9 ms in
    our handler), 5.6 ms refreshing, 18 ms in submits, 42 ms joining the
    recording threads, no forced wait.
  - Still not isolated: what in a translated program costs the time. The
    loop guard's own cost was never taken out and timed; the single-pass
    bench (scratch `card/passbench`) is where that and the other pieces
    are removed one at a time.

- 2026-11-27 (late night): **title memory in system RAM is not what the
  card waits for.** The entry below read the card's slowness from a
  benchmark of scattered reads. Measured with the reads the heavy
  programs really make, where the memory lives explains a few
  milliseconds of the 286 ms frame, and the one thing built on it gave
  nothing (branch `card-memory`, runs cm-a to cm-d on the tree of
  `batch-2026-11-27e`; host only, nothing baked; the full account is in
  `docs/vulkan-host.md`, "Where title memory lives: measured").
  - **What the heavy programs read** (captures of frame 1900, run cm-a,
    replayed with `READS=1`; "heap" is the title's shader-visible
    descriptor heap, 38 MiB at 0x56a0000, which it lays over the head of
    the arena it keeps its per-frame tables and constants in):

    | Program (card ms a frame) | Reads | Where | Stores |
    |---|---|---|---|
    | `TL_PS_PBR_MATERIAL_STANDARD` (22) | 663 dwords a pixel, 465 lines | 71% heap (50 lines on 5 pages of 64 KiB: its two root tables); 25% the 1.28 GiB region (per-frame constants and light lists, 16 pages); 4% the rest of the arena | none |
    | `TL_PS_NoDir_PBR_MATERIAL_STANDARD` (27) | 552 a pixel, 227 lines | 71% heap; 27% the region | none |
    | `PS_DeferredPedTexturedNC` (mesh draws) | 82 lines | 68% heap; 15% the rest of the arena; 15% the region; 2% the sampler heap | none |
    | `CS_Primary_GatherCopy` (18) | 171 line accesses a wave, 18,600 lines on 599 pages (a quarter of its groups) | 23% heap; 69% the region; 8% a 348 MiB arena at 0xaf27c0000 | into the region |
    | `CS_ScreenProbes_TraceRays_RayBinning...` (10) | 107 a wave, 28,900 lines on 2,759 pages | 32% heap; 21% that arena; 14% the region; 7% the pool's view; 23% addresses in no range (zeros) | into that arena |
    | `CSComputeSkinningDefaultVelocity` (5) | 134 a wave | 22% heap; 67% the rest of the arena; geometry in the pool's view (card) | skinned vertices into the 348 MiB arena |

    No program stored into the heap. The cloth solvers' replays stop
    early (their captures lack memory).
  - **Who writes it** (`VICTORY_VK_WRITE_CENSUS`, eight presents): of the
    heap's 608 pages of 64 KiB about 116 change a frame (8 in every
    frame), the CPU's work; the 1.28 GiB region changes in 4,600 pages
    of 4 KiB a frame and the title has made unordered-access views over
    197 MB of it, the 348 MiB arena 3,200 pages a frame and 211 MB: CPU
    constants and card scratch share pages there. Whether the CPU reads
    back what the card wrote was not measured.
  - **What those reads cost** (scratch `tracebench` and `wavebench`: the
    same lines, read as often, from memory laid out as the title's
    ranges, native Vulkan on this card): `TL_PS_PBR_MATERIAL_STANDARD`
    **6.6 ms from card memory, 13.1 ms imported**; its heap's lines on
    the card and the rest imported 12.5 ms. `TL_PS_NoDir_...` 2.2 against
    3.1 ms. The ped pixel program 3.0 against 3.2. `CS_Primary_GatherCopy`
    about 0.1 against 1.1 ms, the ray binning 0.1 against 0.5. So with
    every byte of title memory on the card the first lighting pass could
    lose 6.5 of its 22 ms (the 22 scattered reads a pixel of its light
    lists), the others a millisecond or less. Two of about 25 lighting
    programs and three compute programs were measured; taken as typical,
    that is some 10 to 15 ms of the frame, not the bulk of it. The
    benchmark of the entry below picked each read at random among 512
    lines on 512 pages; the passes read 91 lines on 21 pages at every
    pixel, and those cost the same wherever they lie.
  - **Built, measured, not kept: the descriptor heap on the card.** A
    copy of the heap's pages in card memory the CPU writes through a
    mapping (the driver has such memory, type `DEVICE_LOCAL |
    HOST_VISIBLE`), kept current by write tracking (CPU copies at each
    flush, no submission), a range of the memory map at an alias address,
    and the alias handed to programs for descriptor tables and heap
    bases (push constants). Run cm-c: the picture right; the card
    compared with title memory at frame 2350, 607 of 608 pages equal and
    one differing (not explained: a flush between its two steps, or a
    store missed); **card time unchanged** (`TL_PS_PBR_MATERIAL_STANDARD`
    22.5 ms against 22.1, `..._NoDir_...` 27.8 against 26.9, 30 to 31
    frames in 10 s, 40 W), for 38 MiB of card memory, 130 pages copied a
    frame (1.4 ms), 125 more faults a frame and 2.6 ms more in the
    flushes. Patch in scratch `card/heap-copies-with-census.patch`.
  - **A range of the memory map cannot be cut.** The first build (run
    cm-b) put the copy in the heap's own place in the map, cutting the
    import around it: 6 frames a second, and a picture of hair and sky.
    A program translates a base once and adds offsets; the title
    addresses a 174 MiB buffer from 0x61a0000, inside the heap's 38 MiB,
    and everything read past the copy's end read nothing. What the map
    holds as one range (a region, joined loose memory) is one thing to
    programs.
  - **Also not it: victory's own buffers in system RAM.** The descriptor
    buffers, the memory map's blocks and the code tables moved to
    mappable card memory (run cm-d): card time unchanged (27.9 and 21.3
    ms, 32 frames in 10 s, 41 W), and the driver took 512 MiB more of
    the card for it (the pool's view fell from 1,216 to 832 MiB). Not
    kept; patch `card/own-buffers-on-card.patch`.
  - **So** the card's 286 ms are not explained: not by loops, the map's
    search or the program's size (entry below), not by where title
    memory or our own buffers lie. The card's power is no measure
    either: it drew 40 W in run cm-b too, at twice the frame rate. Next
    to try is what the entry below planned and did not do: run one
    captured lighting pass natively (its SPIR-V, its images, its memory)
    and take things away until its 22 ms move.
  - Kept: `VICTORY_VK_WRITE_CENSUS` (`vulkan/census.rs`). In scratch
    `card/`: `readtool` (the replay with `lines.txt`, and `waves.bin`
    with `WAVES=1`), `tracebench`, `wavebench`, `membench` (now with
    mappable card memory: as fast as card memory but for lines spread
    one a page over 64 MiB, 11.8 against 4.1 ms).

- 2026-11-27 (night): **why the card is slow: it reads title memory out
  of system RAM.** After the early depth test (card time 395 -> 286 ms a
  frame, 2.2 -> 3.1 fps; tiled lighting 205 -> 104 ms; bake 201 s) the
  lighting passes that really cover the screen still took 22 to 27 ms
  each, and the card drew 41 W of its 180 W in the scene (11 W idle,
  memory controller 2%): it waits.
  - **Ruled out**: loops at their guard (none fired); the memory map's
    search (sharing the search among translations that add constants to
    one base, 49 of 122 searches gone in `TL_PS_PBR_MATERIAL_STANDARD`:
    104 -> 109 ms, no gain; not kept, patch in scratch `gt/near.patch`);
    the program's size (replay of a captured pass: 10,300 thread-IR
    instructions a wave, loops run 1.2 to 2.3 times).
  - **What the pass reads** (replay with `READS=1`, 7,200 waves): 660
    dword reads a pixel, all from 465 lines of 64 bytes (30 KiB) on 83
    pages of two imported ranges, spread over 1.1 GiB of addresses.
  - **What that costs** (scratch `membench`, native Vulkan on this card:
    1 M invocations, dependent reads of a storage buffer): 512 lines
    spread over 64 MiB, 640 reads each: **4.2 ms from card memory, 27 ms
    from imported shared pages, 50 ms from imported private 4 KiB pages**
    (huge pages: 26 ms; the driver's own host-visible memory: 5.3 ms).
    4,096 lines: 6 ms against 143 ms. Whole 64 MiB at random: 11 ms
    against 364 ms (about 185 M reads/s: the bus).
  - **So**: title memory imported in place costs the CPU nothing and the
    card six to thirty times its own memory's time on every read that is
    not in a small window. What programs read often has to be on the
    card (as the pool's view already is); which ranges, and how they are
    kept current, is the next work (branch `card-memory`).
    **Corrected by the entry above**: the passes' own reads stay in a
    small window, and moving them changes nothing.
  - Tools kept: `replay` writes `visits.txt` (a wave's work by block) and,
    with `READS=1`, `pages.txt` (title memory read and written, by map
    range).

- 2026-11-27 (evening): **where the card's 395 ms a frame go** (run gt1,
  `VICTORY_VK_GPU_TIME=1900,6`, now a window of presents told as time a
  frame, by program and by family; one present alone had covered 107 ms).
  Of the 60 largest programs (328 ms): **tiled lighting pixel programs
  205 ms** (25 of about 40 pipelines `VS_TiledLighting+TL_PS_*`, one per
  material, each bound once a frame, 3 to 28 ms each), compute 88 ms
  (`CS_Primary_GatherCopy` 18.7, screen-probe ray binning 9.9, cloth and
  skinning 5 to 7 each), mesh draws 34 ms, our present 2.4 ms.
  - **Reading, not yet proven**: each lighting pass is a full-screen draw
    the console rejects per pixel by an early stencil/depth test on the
    material; our SPIR-V never asks for early tests (and its stores and
    kills stop the driver doing it unasked), so every pass shades every
    pixel. The early-depth change (from trustno1-coder/VIctory-NVIDIA,
    `DB_SHADER_CONTROL` bit 12; branch `tr-batch`, in work) is the test of
    this: a translator change, so one bake.
  - Also from that fork's comparison (2026-11-27): it leaves out
    `CS_RaytracingResolve*` and `CS_ScreenProbes_ActivateRays*` while the
    builder is off; the title's upscaler (29 `FSPX` programs) translates
    there and is a stand-in here (branch `fspx`, in work); ordered pixel
    sections and fewer half-wave meetings are in `tr-batch` too.
  - Pull request #1 (tbarnavon: regions imported in pieces no larger than
    the driver's largest buffer, for AMD's 2 GiB limit) merged and run
    here (pr1): 59 regions, one piece each, picture and time unchanged.
- 2026-11-27 (translator, from trustno1-coder/VIctory-NVIDIA `0f0ada1`;
  branch `tr-batch`; proven offline, nothing baked or run):
  - **The console's early depth test is the module's.** A pixel program
    whose `DB_SHADER_CONTROL` has DEPTH_BEFORE_SHADER (bit 12: the source
    shader's `[earlydepthstencil]`, the tests and their writes before the
    program) declares `EarlyFragmentTests`, unless it gives the tests
    their depth or stencil (export bits 0-2, an `SV_Depth*` or
    `SV_StencilRef` output, or a module that writes either): theirs
    `early_depth_of` (41b4a05), there behind a switch that is off; here
    always (`vk::early_depth`, `Interface::early_depth`, emit.rs).
    - Pack: 3,009 of the 12,056 pixel programs that translate ask for it
      and all get it (none exports depth or stencil); used set: 223 of
      414, among them all 130 `TL_PS_*` (65 `TL_PS_NoDir_*`; the pack has
      240), the tiled lighting's full-screen draws that the console
      rejects per pixel by the material's stencil. Their word is 0x1010
      (bit 12, Z_ORDER early then late, no kill); the others' 0x1410,
      0x1050, 0x1450 (bit 10, run for no-op colour writes, is set exactly
      in the 919 that store to memory or storage images: 80 used; bit 6,
      kill, in 2,409: 23 used).
    - **Why the driver did not test early by itself:** every one of our
      pixel modules demotes (the kill of the lanes outside EXEC at the
      `done` export), and every one with a loop can store (the loop
      guard's count: an atomic add and a store through a pointer). 140 of
      the 223 have no store but that, 80 store on their own, 3 neither.
      So each `TL_PS_*` draw shaded every pixel of the screen and was
      tested after.
    - **Kills.** With the mode a killed pixel has written its depth and
      stencil before it ran: so it has on the console, whose bit says the
      same of the same shader (Mesa's AMD drivers set it for a shader's
      `early_fragment_tests`: as remembered, their source was not at
      hand; in the pack the neighbouring bits agree with that reading,
      bit 10 being set in exactly the programs that store). Our demotes keep their meaning for colour and for
      what the program stores after them. No program is left out for
      killing.
    - **Stores.** A pixel the tests reject no longer runs, so what it
      would store does not land: the transparent pass's 64-bit atomics
      (glass behind hair took the hair's place in depth and motion), the
      80 used programs above.
    - Proof: `the_consoles_early_depth_test_is_declared` (oracle/nvidia.rs:
      theirs' two tests; fails without the mode); every one of the 3,009
      modules differs from before by that one instruction and passes
      `spirv-val`; no other module of the pack changes; `shader-golden`:
      1,527 programs, 0 changed (the interpreter runs no depth test).
      **Needs the bake:** the 223 used pixel programs.
    - **Left as it is, to decide:** 191 used pixel programs do not ask
      (126 of them 0x10: early then late, no kill, nothing exported, no
      store of their own; the console's hardware tests those early by
      itself). Ours cannot be tested early by the driver either (41 for
      the loop guard's store, all for the demote). Declaring the mode for
      them would be our choice, not the console's word: it would change
      nothing they write unless a draw uses alpha to coverage with them.
  - **An ordered pixel section runs inside the fragment shader interlock.**
    The console's primitive-ordered pixel shading (a pixel wave names its
    packer in `HW_REG_POPS_PACKER`, sleeps until the waves holding earlier
    primitives' samples of its pixels have left, `src_pops_exiting_wave_id`,
    and ends with `s_sendmsg(MSG_ORDERED_PS_DONE)`): until now the wait was
    skipped and nothing ordered ("overlapping pixels of one draw run in any
    order"). Now such a program (`Interface::ordered`: it writes the packer
    register) is a function its entry point calls between
    `OpBeginInvocationInterlockEXT` and `OpEndInvocationInterlockEXT`,
    execution mode `PixelInterlockOrderedEXT`
    (`VK_EXT_fragment_shader_interlock`; theirs `ordered_section`: D3D12's
    rasterizer ordered views, which wait at their first access).
    - **The programs:** 10 in the pack, none in the used set, all
      `PS_PedTexturedCutOutTint_Zero_*` (the hair cards that keep their
      nearest layers), control word 0x10240 (bit 16, primitive ordered,
      in these ten and no other).
    - **The interlock's rules** (one begin and one end, in the entry
      point, outside control flow, begin first, each on every path): all
      ten have the same shape, one section between branches: a wave with
      no pixel left leaves for `exp null` before the section, and one
      does from inside it (a discard between the stores), before the
      `done` message. A begin where the console's is would sit inside
      those branches and its end would be skipped. So the whole program
      is the section: the entry point is `begin; call; end; return` and
      nothing else, in all ten. That orders more than the console does
      (the shading before the section too) and nothing less; what it
      costs on the card is not measured (the programs are in no scene
      run so far).
    - **Host:** `VK_EXT_fragment_shader_interlock` and
      `fragmentShaderPixelInterlock` are enabled where the device has
      them (`victory-vk/src/host.rs`, `Caps::pixel_interlock`; the RTX
      5060 Ti does: `vulkaninfo`). Where not, the module is given
      without its interlock (`limits::fit`, `fit_interlock`: the five
      instructions left out), which is what every such program was
      before. `docs/vulkan-abi.md` is not changed (its section 1 does
      not list the extension yet).
    - Proof: `an_ordered_section_runs_inside_the_interlock`
      (oracle/nvidia.rs: theirs
      `an_ordered_section_uses_rasterizer_ordered_views`, their program;
      fails without the change), `the_interlock_goes_where_the_device_has_none`
      (victory-vk); pack: exactly these 10 modules change (+30 words),
      all pass `spirv-val`, their approximation is gone;
      `shader-golden`: 0 of 1,527 changed; `cargo xwin check` of
      `victory-vk` and `victory-gpu`. **Not run on the card.** Needs no
      bake of a used program.
  - **The halves of a wave64 meet before an LDS access only where the
    meeting orders something** (`vk/meetings.rs`; theirs `LdsOrder`,
    89cfae2, kept exact). The lift puts a meeting before every LDS access
    and says what the access does; when the program is whole (after
    `uniform.rs`) a pass over the block graph takes away the ones whose
    access commutes with everything a half may have done since the halves
    last surely met (a kept meeting, an exchange that is still there, the
    console's barrier): a read after reads; an atomic whose result nothing
    takes after atomics of its family (adds; each minimum, maximum and bit
    operation). A plain write, an atomic that returns the old value, an
    exchange and a compare-exchange keep theirs after anything.
    - **Where it differs from theirs:** theirs cuts the program at
      `s_barrier` and lets every read and atomic go that no plain write
      shares a piece with, reads beside atomics too (it calls those
      unstable: a half may read before the other half's add; the console's
      wave never does, and `interp_group` counts it a race). Ours keeps
      that meeting, and in return goes by what precedes an access on a
      path, so reads after the meeting that follows a write need none.
      `CS_DefragHeap_ComputeRelocations`: 30 meetings in our code before,
      12 now (theirs: 138 to 50 in theirs').
    - **Measured** (meetings in the code): used set 3,419 to 1,803 (96
      programs had some, 90 have); pack 10,530 to 5,250 (356 to 318).
      Most changed of the used set: `LuminanceMinMaxDownsampleFinalCS` and
      `LuminanceMinMaxDownsampleCS` 166 to 81, `CS_IrradianceCache_
      IntegrateIrradianceSH` 126 to 47, `LuminanceMinMaxDownsampleInitCS`
      142 to 66, `CS_AmbientCubeIBR_Paraboloid` 115 to 42 (of the pack:
      `GpuParticleEmit:CS_Emit_Dynamic_Model_Edge` 201 to 85). The
      exchanges are as they were (7,599 used).
    - **Found wrong on the way:** the lift took every exchange for a
      meeting, and `uniform.rs` then took exchanges of uniform words
      away. In `CS_RayBudgeting_Cull_RayBinning` (used) and
      `CS_VolumePortalTiling_Generate` a lane's `ds_write` and the wave's
      `ds_add_u32` after it were left with nothing between them: the half
      that ran first could add before the other half's lane had set the
      counter. Both have a meeting there now (1 to 2 each; the only
      programs that gained one).
    - Proof: `lds_accesses_that_commute_need_no_meeting` (theirs
      `lds_accesses_after_the_last_plain_write_need_no_meeting`, their
      program; fails without the pass), `a_read_after_the_waves_adds_sees_
      them_all` (the case theirs gives up), the unit tests of
      `meetings.rs`; `shader-golden`: 0 of 1,527 changed;
      `shader-golden --halves` (now it also counts an LDS dword the halves
      touched with no meeting between as a difference, and lists the
      programs it cannot compare): program for program the same lines
      before and after: 468 same, 19 same but raced over title memory, 68
      past the block limit, 1 differs before and after alike
      (`CS_ProjDecalReduceClassification_Vectorized`, seed 1: one LDS
      dword both halves write in one instruction, which no meeting
      orders). 361 modules of the pack change (97 used; 23 of them, 8
      used, keep their count and grow by 5 to 37 words: a lane guard now
      ends before every LDS access, also where the meeting goes); all pass
      `spirv-val`. **Needs the bake:** the 97 used compute and mesh
      programs.
  - **Their tests, as oracle tests** (`vk/oracle/nvidia.rs`, through the
    interpreter; they ran them on a D3D12 device):
    `typed_d16_table_uav_roundtrip_uint_sint_and_float` and
    `some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on` pass as
    they are. `a_counter_texture_is_set_by_a_store_of_one_channel`: ours
    refused a `d16` image store; now the 16-bit channels are widened by
    the image's kind (the T#'s INTEGER bit: the integer as it is, else
    the half as a float; no program of the pack has such a store, so no
    module changes: it is the upscaler's). The interpreter ran an image
    atomic of a lane outside its guard; no more.
    `immediate_sampler_chosen_by_control_flow` is there and ignored: it
    fails (also when both arms move the same sampler): an S# that is not
    four constants where it is used is read as a heap sampler, slot 0.
    Open; the fix is a sampler slot chosen by a phi of the immediates'
    slots. Not ported: what their tests say of their own lowering (which
    root views it asks for, the stores its tracing cannot type).
  - **A load in a view's format is bounded by its element** (a commit of
    its own, to take or leave: it is right, and it costs a bake of half
    the used set). Their `typed_non_d16_table_load_preserves_integer_bits`
    failed on the last element of an `R16_UINT` view: a load through the
    V#'s FORMAT was bounded on max(stride, 4) bytes, so the last element
    of a view of 2-byte elements (the last three of 1-byte ones) read 0
    (`CS_BlendshapeTranslationsHSSGraphOutputExtract` reads `R16_UINT`
    so, the ped skinning its bone indices). Now on the stride, as the
    console tests the index against NUM_RECORDS (4 bytes for a view
    without a stride); the decoder reads the element's own dword and
    nothing past it. `shader-golden`: 0 of 1,527 changed (random views
    seldom have a stride under 4); **11,042 modules of the pack change**
    (every program with such a load, 4 words a load), **790 of the used
    set**; all pass `spirv-val`.

- 2026-11-27 (the title's upscaler): **the 29 passes of the title's
  FidelityFX upscaler translate, and the linear scale stands in only while
  one of them does not** (branch `fspx`; offline only: nothing baked, no
  run, so nothing below is seen on the card yet).
  - **What they are.** Compute programs like the title's, made by AMD's
    compiler: a container with an `ACCC` part and no `DXIL` part, the part
    an `FSPX` payload (a directory of dispatch state, the GFX10 code and
    records that say where the root parameters arrive;
    `victory-shader/src/native/fspx.rs`, read as the upscaler's own reader
    reads it). The upscaler is a network (FSR4): a **prepass** (the colour
    as log2(1 + 150 c) / 12.29, the history reprojected, the network's
    first layer), **12 network passes** on tensors of signed bytes in one
    scratch buffer with **13 passes that clear their borders**, a
    **postpass** (the network's last layer, the output and the next
    history), a sharpening pass that decodes that colour space again, and
    an exposure pyramid.
  - **What the translator lacked** (each with a case in
    `vk/oracle/packed.rs`, the hardware's words from ProjectLeonida's
    tests): `v_dot4c_i32_i8` and `v_dot4_i32_i8` (the dot product of four
    signed bytes, 27,271 in the 29 programs; two registers go through one
    function of the module, `Helper::Dot4`, so the largest pass is 275,000
    words where written out it was 607,000), `v_pk_add_u16`,
    `v_pk_sub_u16`, `v_pk_fmac_f16`, an image store of 16-bit data, and two
    things it had wrong: **an image load of 16-bit data from a float view
    took the low 16 bits of the float** (now the float as f16, as samples
    did), and **`s_mov_b64 vcc, <two numbers>` moved a lane mask** (this
    compiler keeps numbers in vcc_lo and vcc_hi: in waves of 32 vcc_hi had
    no value, three passes were refused; in waves of 64 the numbers were
    read as ballots). The loop guard of these programs is 2^18 (they bound
    their own waits at 2^16, `docs/vulkan-abi.md` 2.1).
  - **Proof.** `cargo test -p victory-shader` (173); every one of the 29
    modules passes `spirv-val`; none takes an approximation. The golden
    file: 9 of 1,527 used programs change, each of them one with an image
    load of 16-bit data (six mesh transforms, two compute passes): the fix
    above, meant. In the interpreter (a scratch tool on `vk::interp_group`,
    inputs made up, the network's weights from the user's DLL): the
    prepass's colour is the closed form to five digits; all 27 passes of a
    frame in a row on a static 64x64 scene return and the output is the
    input; **from a black history the output's worst pixel is 0.1615,
    0.1975, 0.19985, 0.19999 of an input of 0.2 after one to four frames**;
    a static ramp is met within 0.001 inside the picture after four; the
    three passes for waves of 64 give the same bytes run as console waves
    and as the card's halves; the sharpening pass gives the colour back
    within 0.6% (it computes in 16 bits).
  - **The host** (`upscale.rs`, `hle/modules/ffx.rs`, `pipeline.rs`,
    `vulkan/record.rs`, `vulkan/stages.rs`; no change of the ABI). The
    title's own `ffxDispatch` runs the upscale when every pass in the
    upscaler's DLL translates (checked once, at the first upscale: about
    0.2 s in a release build); the linear scale takes it otherwise, and from the moment a
    pass fails to become a pipeline on the card. A pass is named by the
    entry function its `PSV0` part names (the recorder left nameless
    compute programs out as the structure builder's: every pass would
    have been skipped). Before a pass's dispatch the recorder puts a
    memory barrier: the passes are made to overlap on the console and wait
    for one another by spinning on flags in the scratch buffer.
  - **Not proven, most important first:** (1) nothing ran on the card: the
    Windows side is compiled and its tests built, not run; (2) the order
    and the group counts of the passes are read from the DLL's code, not
    seen in a run, and in that order **pass 7's first wait is not met in
    the interpreter** (it wants counts of 4 where pass 6 has stored 1) and
    runs to its bound: right output, 65,536 looks; on the card that would
    be time in every frame, and the first run should look at that pass's
    time; (3) the driver's compile of the three largest modules (177,000
    to 275,000 words) is not timed; (4) the exposure pyramid and the
    sharpening pass were only run alone.
  - **Cost, from the code** (1920x1080 out): 27 dispatches a frame (29
    with sharpening and the exposure pyramid): the prepass 8,160 groups of
    256, the network 12 dispatches over tensors of 960x540, 480x270 and
    240x135 (518,400 down to 32,400 threads, about 3.1 billion byte dot
    products a frame in all), the border passes 24 to 94 groups of 32, the
    postpass 2,040 groups of 256.
- 2026-11-27 (streamed textures): **a texture the title places lives
  while one of the title's descriptor slots names it, and has the levels
  its latest view names; a slot stays its texture's** (runs tx0 to tx7 on
  the cache of `batch-2026-11-27d`, branch
  `worktree-agent-a1b8109f1aac12c0f`; host only, nothing baked). The suit
  is orange with its lettering in every run since, the bunk, the window
  frame and the walls have their own textures, and the final frame is the
  reference's picture.
  - **How the title streams textures** (run tx0: the baseline with
    `VICTORY_VK_CENSUS`, 225,686 texture views to the end of the world's
    load; `vulkan/images/levels.rs`):
    - It has no resource for them: 219,180 of the views are views of
      memory it placed itself (`CreatePlaced…View`), each with the
      texture's whole description, every level. The console's four extra
      words of the view are (0, 0, 0, 3) throughout and no view has a
      `ResourceMinLODClamp`; the pages under the layout say little (the
      neighbours' pages are mapped).
    - **Which levels are there is the view's first level.** One slot of
      the title's heap (1,216,000 slots at 0x57b0000; textures in the
      first ~110,000) belongs to one texture object. Slot 0x6990: a view
      of address 0 when the object is made; from level 1 at 0x5aff28000
      when its data is placed (frame 894); from level 1 at 0x69a648000
      when it moves to a piece of 0xc000 bytes, its whole layout (frame
      1290); from level 0 there a frame later (the level came in, in
      place); from level 1 again (frame 1317: let go); from level 1 at
      0x710338000 (frame 1321: moved to a smaller piece).
    - The levels held are the head of the console layout (the block of
      the small levels first, then the others from the smallest up, level
      0 last), and the pieces are packed by it: a 1024x1024 BC1 texture
      (0xab000 bytes) viewed from level 3 holds 0x3000 and its neighbour
      starts 0x4000 on; viewed from level 2 it holds 0xb000 and the next
      starts 0xc000 on; 512x512 ones viewed from level 3 lie 0x1000
      apart. Of the views that name a level above 0 or a texture of one
      level, 123,994 of 124,659 had every byte through the named level
      mapped.
    - **A view from level 0 is not that statement.** While it places a
      texture the title first writes a view of all its levels into a
      scratch slot (eight textures in a row into slot 36,054 in frame
      1005), and the object's own slot gets the view from the level held
      up to five frames later: 43,152 views from level 0 were followed by
      one from a higher level, 33,625 of them in another slot; 30,025 of
      the 53,741 level-0 views of textures with levels claimed bytes that
      were not mapped. The last view written is the one that counts.
      Most textures of the test scene are then cut down again (the
      sequences of first levels 0, 3, 8 and 0, 3, 7 and 0, 1, 6 lead).
  - **What was wrong.** `drop_overlaid` took a texture's whole layout as
    its memory, so every texture placed took the images of its neighbours
    out, and the scratch views of all levels did it for the textures that
    are small on the card too. At the end of the load the title's slots
    named 48,432 textures and 20,428 had an image, another set in every
    run. A fill read the whole layout, so it failed where a page of the
    levels not held was not mapped. A dropped image's slots went to the
    next image made: a descriptor the title kept showed another texture.
  - **The rules as built:**
    - *Levels* (`levels.rs`, `fill::renamed`): the image of a texture the
      title fills has the levels from the one its latest view names on
      (less those texture quality cuts), and is filled from the head of
      the layout that holds them; pages that are not mapped read zeros.
      A view that names another first level has the image made again;
      the texture's slots show the new one once the card has filled it.
    - *Life* (`named.rs`): a slot of the title names the texture of the
      last view call that wrote it (a copy call carries the name along).
      A texture no slot names for four frames goes. Nothing else ends a
      texture the title fills: no neighbour (`drop_overlaid` is gone), no
      remapped page (it is filled again from the pages shown now), no
      resource the title makes over its memory and releases while the
      slot goes on naming it (3,300 a run).
    - *Slots* (`descriptors.rs`): a slot belongs to a texture (title
      address and shape) and a view of it, and stays theirs. Without an
      image it reads the null image; when the texture has an image again
      its slots show it and the old descriptors work. The last 65,536
      textures that went keep their slots; only past that, or with all
      2^20 given, are the longest-gone texture's slots given again.
      The clean form is a generation in the T# that the translated code
      checks: `docs/vulkan-host.md`, proposal 8 (not built).
    - *Texture quality* cuts block-compressed textures only: an
      uncompressed texture the title fills is data as often as colour
      (the suit's 256x256 map of material numbers, read at an explicit
      level: the pale seams), and all of them are 9 MB.
  - **Room on the card.** With the levels the views name, the 25,300
    textures alive when the scene stands take 422 MB (before: 18,900 in
    519 MB, every level under texture quality; the title's levels without
    texture quality would be about 1.1 GB). Image blocks are 64 MiB, the
    pool view's size (256 before; images of 16 MiB and more, the title's
    3840x2160 targets, take exactly their size): 2,913 to 2,977 MiB held
    for 2,664 bound where twelve 256 MiB blocks held 3,072, and a block
    the images need is the pool's to give at once (`budget::fits` counts
    the pool's blocks on their way out and waits a moment for the memory
    lock: before, the image that asked was refused and its descriptor
    stayed null). The pool's view: a page counts as texture memory when
    the textures' own bytes cover it whole (not when one texture's layout
    reaches over it), never when a buffer resource lies over it, and a
    page textures no longer cover takes a card page again
    (`pool_read_by_programs_not_on_card_mb`: 0 once the scene stands).
    The ceiling is as it was (5,504 MiB); nothing was refused.
  - **Numbers** (7 minutes; before: tx0, after: tx3, tx4, tx6, tx7 on one
    build):

    | | before | after |
    |---|---|---|
    | images made | 160,933 | 162,150 to 168,557 |
    | dropped: overlaid / remapped / resource | 107,255 / 32,881 / 369 | 0 / 0 / 2,700 to 3,040 |
    | dropped: no slot names it / made again for levels | - | 52,000 to 55,200 / 79,300 to 82,100 |
    | images refused | 0 | 0 |
    | textures the title's slots name / with an image | 48,432 / 20,428 (end of load) | 26,250 to 26,600 (scene standing) / all |
    | title descriptors reading the null image | not counted | 0 of 74,000 |
    | filled images alive, card MB | 18,925, 519 | 25,070 to 25,410, 422 |
    | image blocks MiB / bound | 3,072 / 2,761 | 2,913 to 2,977 / 2,664 |
    | card peak MB (`vram.txt`) | 6,731 | 6,722 to 6,735 (tx6: one sample of 6,823) |
    | RAM peak MB | 17,125 | 16,215 to 16,818 |
    | card ms a frame (direct0, from frame 1500) | 389 (tx0), 441, 447 (sh2, sh4) | 441 to 445 |

    Images made did not fall: 66,000 placements and 88,000 changes of
    the first level are the title's own streaming in this load (a texture
    is placed with a scratch view of all levels, viewed from level 3,
    then cut to its small levels), and each makes an image. What went is
    the 107,000 images taken out by neighbours. A frame costs the card
    what it cost in the runs where the suit happened to be orange (sh2:
    441 ms): more is drawn (hair, the layered materials' textures).
  - **The picture** (albedo 0xb0a5a0000 at frame 1620, final frames
    1920 to 1950): in tx1 to tx4, tx6 and tx7 the orange suit with INMATE
    across the back, the blue window frame, the pale wall with its
    texture, the bunk's blue frame, the posters, the people in orange in
    the yard. The ponytail is in tx1, tx3, tx4, tx6 and tx7 and not in tx2
    (a few strands at the neck; before the pool's pages that textures no
    longer cover came back to the card). Against the reference the final
    frame is the same picture; seen and not examined: the reference's
    lettering is sharper and its hair denser, with a hair tie, and the
    character's shadow on the left wall is darker there.
  - **Runs**: tx0 the baseline with the census; tx1, tx2 the rule as it
    grew; tx3, tx4, tx6, tx7 the build as it is (one exe, in a row). tx5
    ended 57 s in, before the world: a null read in the title's own code
    (+0x10823c) while a page beside the object it read was being shown
    again; seven logs of the day under other builds end in title faults
    of this kind. The trial branch `sh-texture-own-trial` (the largest
    level any view named) is superseded: the latest view is the word.
  - **Tools**: `VICTORY_VK_CENSUS=<file>` (diagnostics builds): every
    texture view, announced write, page mapping and dropped image as a
    line (`images/census.rs`; the scripts that read it:
    /mnt/data/victory/scratch-agent/tx/py/). The 10 s lines say how many
    textures the title's slots name, how many of them have no image, and
    how many texture descriptors in the title's heaps carry a slot that
    reads the null image.
  - **Open**, most important first: (1) the ponytail was missing in one
    run of six (tx2, the build before pool pages that textures no longer
    cover came back to the card: 76 to 150 MB of pages programs read
    were off the card during the load); the four runs since have it, the
    cause is not proven; (2) the card's time a frame is at the top of
    what the runs before had (441 to 445 ms against 389 to 447): the
    textures are there to be read; (3) a title descriptor
    older than the 65,536 textures gone since shows its slot's new
    texture: the generation in the T# (translator and ABI); (4) a view
    written while no card memory can be had gets the null descriptor and
    is not tried again (none in these runs; a slot given before the image
    is made would let a later flush make it); (5) descriptor bytes the
    title copies itself are not seen by the naming (none read null here);
    (6) the title's scratch views make an image of all levels (under
    texture quality) that is made again a few frames later: made at the
    flush instead, most would never be made; (7) the ceiling: the images
    hold 2.9 GiB where 2.66 are bound, the card-written ones 2.24 of
    them, the seven 3840x2160 interface targets 233 MiB.
- 2026-11-27 (the jumpsuit): **why the character's suit is dark maroon in
  most runs and orange in some: the textures its material picks by index
  are gone from the card, or their slots show other textures** (runs sh1
  to sh6 on the cache of `batch-2026-11-27d`, branch
  `worktree-agent-ad8acb26bd2985605`; host only, nothing baked).
  - **Not the lighting.** The suit's texel of the albedo G-buffer
    (0xb0a5a0000) has one writer a frame (`VICTORY_VK_WATCH`, below): the
    mesh draw MS_VS_PedTransformPreSkinD + PS_DeferredPedTexturedNC
    (94d281151ca58b4f). It writes 229 / 114 / 82 with class 7 (SHEEN; a
    maroon sheen colour 85 / 35 / 21 in the sixth target) in runs sh2 and
    ch8, and 79 / 30 / 16 with class 5 (TRANSLUCENCY) in sh1, sh3, sh4,
    sh5, vk106 and am2 to am4. Nothing overwrites it. The tiled lighting
    then shades what it is given (class 5: TL_PS_PBR_MATERIAL_TRANSLUCENCY;
    class 7: ..._SHEEN). The window frame's class flips the same way (0 in
    sh2, sh3, sh5; 5 in sh1, sh4, sh6), the bunk's albedo too.
  - **What that program reads** (replays of its captures with the
    descriptor heap in them): three textures named by index in the draw's
    record (root table 2 + 0x2014: heap indices; in sh4 a BC1 512x512, an
    R8G8B8A8 256x256 and a BC5 512x512 of two layers) and, through the
    second one, the material records of a 4.9 MB buffer (0x99ae00000, its
    V# in root table 0 + 0x280; the draw's offset at + 0x200c). The second
    texture is a map of material numbers (bytes 0 to 13 a channel, in
    blocks): each texel picks the records whose textures (heap indices
    again) and constants give albedo, class (7 where a record's float is
    under a computed one, else 5 or 0) and the rest. So one wrong texture
    here changes colour and class together.
  - **What it gets on the card.** Run sh4 (maroon): two of the layer
    textures (heap 0x19c5b, 0x19c5c: BC7 1024x1024, shown from level 1 and
    level 3) have slots with no image; the replay without the layers gives
    class 5 and a black albedo, as the card. Run sh5 (maroon, streaked):
    the map's descriptor (heap 0x8bc5: R8G8B8A8 256x256 at 0x57ea7a000)
    carries slot 0x7c96, which the host has given to the image of another
    texture since (BC1 sRGB 512x1024 at 0x765eb4000); the first texture's
    slot 0x61fa shows a BC7 1024x256 of another address; five layer
    textures have no image. **A title descriptor keeps its slot when its
    image goes; the slot reads the null image, and after its reuse another
    texture.** A program that picks descriptors by index never has them
    written again.
  - **Why images go** (the 10 s line, a 7-minute run): 150,000 images made
    for 20,000 alive; 100,000 dropped as overlaid (`drop_overlaid`), 31,000
    for remapped pages, 300 for released resources; 300,000 fills
    (13.5 GB). The overlaid rule takes a texture's whole console layout as
    its memory. The console stores the small levels first and level 0 last
    (`victory_formats::texture`: 1024x1024 BC1, 0xab000 bytes: level 0 at
    0x2b000), and the title places streamed textures 0x4000 apart: each
    new texture drops its neighbours (log: `new_at=0x59c268000+0xab000
    old_at=0x59c264000+0xab000`). Which ones are alive when a material is
    drawn is a matter of order: the run-to-run picture.
  - **Tried, not merged** (local branch `sh-texture-own-trial`, one commit
    on this one, d860421): a
    texture owns the bytes through the most detailed level a view has
    named (`Image::own`), and only those count for `drop_overlaid` and for
    remapped pages. Overlaid drops 100,000 -> 64,000 (most views name
    level 0 of textures 0x4000 apart, so the rule knows no better there),
    filled images alive 19,000 / 505 MB -> 24,800 / 935 MB. Under the
    ceiling that room is not there: sh5 had 19 image blocks refused (the
    view at 1,344 MiB), and with image blocks let through while the view
    can give memory back (tried in `budget::fits`: sh6) the view fell to
    448 MiB and the character's body and the far buildings went (the
    pool-page matter of the entry below). Neither is in. `ResourceMinLODClamp`
    is not how the title says which levels are there (no view of sh6 had
    one).
  - **Tools** (diagnostics builds, `docs/switches-gpu.md`):
    `VICTORY_VK_WATCH=<address>,<x>,<y>[,c][;...]` reads a texel before
    and after every draw that has the image attached and every dispatch of
    the first dumped frame, logs the ones that changed it and captures the
    next execution of the last writers of the texels marked `c` (a draw is
    known again by its pixel program and fourth root parameter). A watch
    capture holds the descriptor heaps as memory and the textures whose
    heap indices stand three in a row in a block near the roots, so a
    replay follows a material's picks. Captures: a block taken first as a
    table and then whole is no longer written over with its first 4 KiB
    (the material buffer came out 4 KiB long); root constants that are
    addresses by themselves are followed. Scripts:
    /mnt/data/victory/scratch-agent/sh/ (tr.py: a node's value and
    definition from a replay's `WATCH=all` trace; deps.py; classes.py).
  - **Open**, most important first: (1) which bytes of its layout a
    streamed texture owns, from the title (its views name level 0 of
    textures whose large levels cannot be there: the residency must come
    from somewhere else: the page mappings under it are the candidate), so
    that no live texture's image is dropped and none is kept; (2) card
    memory for what then stays alive (about 0.45 GB more; image blocks hold
    3.6 GB for 3.2 GB bound: 0.4 GB lies in fragments); (3) a slot must not
    show another texture while title descriptors may carry it (a
    generation in the T#'s host word, checked by the translator, is the
    clean form: translator and ABI); (4) the lettering: no separate decal
    draw writes the suit's texels; it is part of the same layered material
    (not seen in either state: the orange state is the one with fewer
    textures); (5) the material map is sampled at an explicit level and
    loses two levels to texture quality: numbers blend at block edges (the
    pale seams); maps of numbers need level 0 kept; (6) hair and window
    frame: with the estimate of the entry below they are a matter of
    albedo and class as well (the frame's class 0 or 5 by run).
- 2026-11-27 (indirect light, the stand-in): **an estimate from the frame
  instead of the constant, on by default while the structure builder is
  off** (runs am1 to am6 on the cache of `batch-2026-11-27d`, host only;
  branch `worktree-agent-afc0e2c73452079b8`; `vulkan/ambient.rs`,
  `ambient.comp`).
  - **What it is**: VIctory-NVIDIA's method as one host compute program
    (GLSL, compiled offline, its words in `ambient/words.rs`) run in three
    dispatches right after CS_ScreenProbes_TemporalResolve_*: (1) 32x18
    cells: the mean of the light visible surfaces sent out the frame
    before (the half-size image CS_CacheScreenSpaceRadiance keeps,
    0x9a0560000, B10G11R11: the lighting buffer before fog) and where
    each cell is (from the depth); (2) 160x90 points: the mean of the
    cells by the share of the point's surroundings each covers, and its
    first moments, a direction per colour; (3) every texel of the diffuse
    light the resolve wrote: the four points around at its own depth,
    along its normal (from the depth), shaded in creases (12 depth taps
    within 0.6 m), a fifth of it the slow mean of the whole picture,
    stored as the shared-exponent word; what the title left there stays
    up to four times the estimate. Inputs are the title's own images,
    named by T#s in the resolve's tables (root 0: depth dword 688, albedo
    560; root 1: its history 72, its output 176) and in the keeping
    program's (root 1 dword 16). One buffer of 0.7 MB is all it owns.
  - **Two things differ from the model.** (a) The tiled lighting adds
    about 0.85 x albedo x light (run am1, a wall: light 2.39 / 0.72 /
    0.47, albedo 0.36 / 0.30 / 0.25, lit 0.74 / 0.18 / 0.10), not albedo x
    light / pi as the entry below has it. Fed its own reflected light, the
    estimate times pi ran away (am1: walls 180 / 111 / 77, the window
    white), and at the model's 0.8 it settles dark (am2: walls 53 / 39 /
    34). So the cells hold **one bounce**: what the lighting added for the
    stand-in the frame before (0.85 x albedo x the resolve's history) is
    taken out of the kept light, and the number is free: `VICTORY_AMBIENT`
    unset = 2.5 (set on this one shot: the window wall is lit only from
    behind the camera, where the picture shows nothing). (b) 32x18 cells
    instead of 64x36: 0.35 ms a frame instead of 0.71.
  - **Mean R / G / B, final frame (bytes), frame 2010** (places: the
    window frame's face to the room above the head; the wall left of the
    window; the yard; the sky; the bunk; the character's back):

    | | frame | wall | yard | sky | bunk | back |
    |---|---|---|---|---|---|---|
    | no stand-in (am5, `VICTORY_AMBIENT=0`) | 24 / 11 / 3 | 13 / 2 / 0 | 124 / 85 / 52 | 220 / 194 / 176 | 55 / 38 / 30 | 16 / 7 / 7 |
    | constant 0.5 (am6) | 82 / 84 / 95 | 99 / 99 / 88 | 127 / 92 / 58 | 242 / 230 / 223 | 14 / 8 / 4 | 27 / 11 / 9 |
    | estimate, default (am4) | 46 / 41 / 50 | 72 / 64 / 62 | 125 / 96 / 62 | 238 / 223 / 214 | 13 / 6 / 4 | 23 / 12 / 8 |
    | estimate x 2 (am3) | 50 / 38 / 41 | 75 / 60 / 55 | 124 / 92 / 60 | 238 / 223 / 213 | 79 / 61 / 54 | 30 / 14 / 11 |
    | reference | 73 / 66 / 72 | 97 / 92 / 83 | 116 / 80 / 45 | 208 / 178 / 158 | 53 / 54 / 54 | 69 / 27 / 15 |

    The light itself on the wall (am4): 0.31 / 0.24 / 0.28, on the yard
    0.94 / 0.70 / 0.80, none on the sky; the constant is 0.5 / 0.49 / 0.47
    everywhere.
  - **Against the reference** (/mnt/data/victory/capture/am4/frame-2010.png):
    the window frame is blue-grey, the walls grey with the creases and the
    bunk's recess shaded, the yard keeps its gold; darker than the
    console's walls by a quarter. Not the light's doing, and unchanged:
    the **jumpsuit is dark** because its albedo is (0.074 / 0.016 / 0.008
    in the G-buffer, every run); the **bunk and the window frame are black
    in some runs** because their albedo is (bunk 0.0003 in am1, am2 and
    am4 and 0.29 in am3; the frame 0.016 / 0.021 / 0.028 in am1, a tenth
    of the 0.17 / 0.22 / 0.31 of am2 to am4): the albedo that differs between runs is the next
    thing to trace, no light can show it. The **sky is not less exposed**
    with a stand-in but more (220 without, 238 with the estimate, 242 with
    the constant; 208 in the reference): the entry below read that the
    other way. The light indoors is still rather even: most of what the
    window wall gets is the slow mean (its sources are off screen).
  - **Cost**: 0.35 ms of the card a frame (the pass run 100 times a frame
    from frame 1850 on in am3: 426 -> 460 ms a frame; with 64x36 cells in
    am2: 424 -> 495); card time a frame 410 with (am4), 412 without (am5):
    within the noise. Card memory: the 0.7 MB buffer. Card total at peak
    (default ceiling 5,504 MiB): 6,680 (am4), 6,724 (am5), 6,741 (am6),
    6,682 to 6,801 MiB (am1 to am3).
  - **Seen besides**: `VICTORY_VK_GPU_TIME` covered only the first part of
    the frame it was asked for (am1: 107 ms, up to the screen probes' ray
    binning; the resolve and the lighting are not in it); "a submission has not finished (a
    hang?)" 3 to 8 times a run shortly after the world starts, in five of
    the six runs, the one without a stand-in too (none in vk106).
  - **Open**: reflections (CS_RayMiss still gives the sky's light
    indoors; the model's part for it is a fourth dispatch along the
    reflected direction); the air of a room (the model bounds what it
    scatters); the number 2.5 is one shot's; lamp-lit rooms at night and
    open daylight were not seen.

- 2026-11-27 (colour): **where the world's colour goes: stage by stage**
  (runs col1 to col3 on tag `batch-2026-11-27d`, branch
  `worktree-agent-aea8d1272ae4e7c80`; host only, no translator change).
  The opening shot was amber or black. Mean R / G / B of the same places
  at each stage (run col1, exe int57, frame 1600):

  | Stage | window frame, face to the room | wall | yard | sky | bunk |
  |---|---|---|---|---|---|
  | albedo (0xb0a5a0000, linear) | 0.155 / 0.208 / 0.284 (blue) | 0.38 / 0.40 / 0.40 | 0.12 / 0.10 / 0.05 | - | 0.56 / 0.53 / 0.45 |
  | indirect diffuse in (0x9bb910000) | 0 | 0 | 0 | - | 0.71 / 0.34 / 0.30 (leftover) |
  | lighting buffer after the tiled lighting (0x99a1d0000) | 0.91 / 0.24 / 0.020 | 0.000 | 0.83 / 0.17 / 0.010 | 3.5 / 3.4 / 4.9 (before fog) | 0.32 / 0.15 / 0.11 |
  | after fog, = upscaled HDR (0x99e980000) | 0.91 / 0.24 / 0.020 | 0.0020 / 0.0006 / 0.0003 | 0.84 / 0.20 / 0.031 | 4.8 / 2.4 / 1.75 | 0.32 / 0.15 / 0.11 |
  | 10-bit target after the composite (0xb3a8f0000) | 0.66 / 0.42 / 0.14 | 0.055 / 0.016 / 0.002 | 0.70 / 0.45 / 0.20 | 0.93 / 0.86 / 0.82 | 0.47 / 0.35 / 0.28 |
  | final frame (bytes) | 168 / 105 / 35 | 14 / 4 / 0 | 181 / 115 / 50 | 235 / 219 / 209 | 120 / 89 / 71 |
  | reference (bytes) | blue-grey | 95 / 92 / 81 | 132 / 90 / 52 | 206 / 177 / 160 | pale |

  - **Albedo: right.** The G-buffer has the hues (blue frame, grey-beige
    walls, posters, green-brown ground). Texture decode, sRGB and channel
    order are not the fault. (The bunk's and some props' albedo differ
    from run to run, beige, pink, olive: not traced.)
  - **Lighting: the shaders are right, the light is missing.** Captures
    now take a draw's colour targets before and after it; the card's
    result equals the interpreter's texel by texel for the programs
    replayed (TL_PS_PBR_MATERIAL_TRANSLUCENCY 0.5327 / 0.3169 / 0.2705
    against 0.5332 / 0.3175 / 0.2702; NoDir standard 0.7495 against
    0.7512; the standard program over a band of 26,739 texels it wrote:
    26,648 within 2%, means 0.7008 / 0.3487 / 0.2200 against 0.7012 /
    0.3489 / 0.2199). What a surface gets:
    - *the sun only.* Its colour is the title's own: a 16x1 table
      (0x9d2f40000) that CS_DirectionalLightTransmittanceLUT makes from
      the atmosphere's transmittance (replayed: the same), 0.284 / 0.061 /
      0.0039 at the ground, red to blue 73 to 1 at this low sun. Blue paint
      under it is orange: 0.155 x 0.28 against 0.284 x 0.004.
    - *no indirect light.* The diffuse light image is 0 on walls, frame
      and yard (every probe ray misses without structures) and holds
      leftovers on the bunk, the one thing indoors that showed.
    - *no ray-traced sun shadow.* The window frame is of the translucent
      class (albedo alpha 5; 234,227 texels with part of the yard's
      ground), which lets the sun through from behind where the shadow
      mask (0xae6260000) says lit. The mask is 0 on the interior walls
      (the cascades see the building) but 0.80 on the frame, in a
      checkerboard: the frame's face to the room is 0.9 red. Standard
      surfaces facing away from the sun stay dark by their normal. Read
      as: the cascades cannot tell a frame's face from its sunlit back a
      few centimetres behind, and the shadow rays that could find nothing
      (not proven: the rays' part of the mask was not traced).
    - With the interior black the title's exposure sits at its upper
      clamp: the sky and yard are over-exposed (sky 235 against 206).
  - **Post: right.** PS_CompositeNMB replayed from its captured inputs is
    the card's 10-bit target within a step (21,408 texels, worst 2.3 of
    1023); fed neutral greys from 0.01 to 100 with the captured tables it
    returns greys (R and B 9% under G in the shadows, within 2% from 1 up: the
    title's grade, the green-grey shadows of the reference). The back
    buffer is the 10-bit target.
  - **Done (host)**: `VICTORY_AMBIENT=constant:<level>[,<r>,<g>,<b>]`
    (`vulkan/ambient.rs`, off by default) clears the diffuse light image
    to a constant right after CS_ScreenProbes_TemporalResolve_* (the idea
    of VIctory-Belobug-AMD and VIctory-NVIDIA, which is how those D3D12
    forks get their interiors while ray tracing is off). Run col3 (0.5):
    walls 77 / 78 / 66, the frame's edges blue, posters, the window's
    mesh, the bunk, a golden yard; exposure comes down by itself (sky HDR
    4.8 -> 3.7). Still wrong there: the frame's face is orange (the
    shadow rays), the sky too white, no character.
    BCPack: BC7 rotation only in modes 4 and 5 (fxceless's fix; no
    control word of the shipped tables is affected, so no texture of this
    title changes).
  - **Open**, most important first: (1) the stand-in as an estimate from
    the frame before (VIctory-NVIDIA `ambient.rs`: the radiance
    CS_CacheScreenSpaceRadiance keeps, 64x36 cells, gathered at 160x90
    points; a host compute pass), so that level and tint follow the hour
    and nothing glows at night, then on by default while the builder is
    off; (2) sun shadows without rays for the translucent class (the
    builder, or the mask from the cascades); (3) the same for reflections
    (CS_RayMiss gives the sky's light indoors); (4) the albedo that
    differs between runs.

- 2026-11-27 (the character): **why no character was drawn: the memory
  map held 64 KiB of the 50 MB buffer of skinned vertices** (branch
  `worktree-agent-a376a5a004106857f`, host only, shown on the card: runs
  ch1 to ch8 on the cache of `batch-2026-11-27d`).
  - **Where it was lost** (run ch1, exe int57): the albedo G-buffer
    (0xb0a5a0000) has the room, the window frame, the yard and the posters
    and no character and nobody in the yard, before and after every ped
    draw; so nothing downstream (classes, lighting, SSS) is at fault. The
    peds are skinned by compute (CS_PedStandardSkinning_0/1/9 ...) into two
    buffers, 0xad5540000 (50 MB) and 0xad8740000 (25 MB), that the mesh
    programs (MS_VS_PedTransformPreSkinD*) read. Replayed from its capture,
    CS_PedStandardSkinning_0 stores 4,703 dwords at 0xad5c409d0; the
    capture's memory map has `ad5540000 +10000` and nothing else there:
    every store went to the sink and the mesh programs read zeros.
  - **Why**: those buffers lie in memory the title maps page by page
    (views of physical pages: each 64 KiB page is an allocation of its
    own). A buffer view named the buffer's first address only (`touch`),
    and "loose" memory was imported as the run of pages of that one
    allocation: one page. Buffers in `XMemVirtualAlloc` regions were never
    affected (the region is imported whole), which is why the world drew.
  - **Fix** (ae2725f, b6efb7b): a view imports its buffer whole
    (`Memory::want_range`: loose memory to the end of the range named, over
    the allocations that follow, as far as they are committed read-write
    memory and no region, pool or view of their own); the title's buffer
    resources are known by address (`buffer_resource`), so an address named
    inside one (a root view, an index buffer, a copy) imports all of it;
    a range over several regions wants each. In ch3 the map has
    `ad5540000 +4f60000` and about fifty buffers of 0.1 to 8 MB at
    0xaf... come in whole that had one page each before.
  - **Also fixed there**: a loose union with pages freed since (a block of
    the process heap: no call says so) was refused whole and stayed
    refused: in vk105 143 MB of loose memory read zeros from the world's
    load on (`a region was not imported`, also runs gi4 and gi9). Each run
    of pages still there is a region now.
  - **After it, under the ceiling as it is** (ch2, ch3, ch7): the
    character's back and shoulders in the jumpsuit are in the albedo and
    in the final frame, the people in the yard are in the albedo, and the
    window's mesh shows in ch2 and ch7 (it is drawn from such a buffer
    too). No head yet (in ch6 and ch7 its eyes and an ear float where it
    belongs), the hair a tuft at the neck: the next point.
  - **What changes from run to run, and why: the pool's GPU view is three
    quarters unbound.** The title shows 1,770 MB of pool pages in its GPU
    view; under the ceiling the view keeps 372 MB on the card (images take
    3,072 MB, 2,241 MB of them render targets and other card-written
    images, among them 3840x2160 ones), and an unbound page reads zeros.
    Only a recording that names a page brings it back, and what programs
    read through views made once (meshlets, vertices, tables) is never
    named again. Runs on one build differ in buildings, window frame,
    mesh, bunk and hair by what was evicted. With
    `VICTORY_VK_CEILING_MB=5376` (ch5: view 1,144 MB, card 6.55 GB at
    peak, no image refused) the buildings, the watchtower and the bunk
    take the reference's shapes.
  - **The head and the hair were lost there too.** Run ch8 (the rule
    below, `VICTORY_VK_CEILING_MB=5312`: view 1,038 MB on the card, card
    6.49 GB at peak): the albedo at frame 1680 has the whole character
    from behind, head, brown hair, the ponytail down the back, the orange
    jumpsuit, and through the window the building, the watchtower and
    the people in orange at the tables; the final frame 1950 has the
    head, neck and shoulders as a dark silhouette against the yard (no
    indirect light on it: the dark-walls matter), the lit window frame,
    the building at the left, the watchtower, the bunk and posters at the
    right. No window mesh in that frame, no lettering, the left wall's
    albedo black (48 textures of 256 to 1024 texels were not made: the
    images wanted a thirteenth block while the view held the room).
  - **Pool pages by what reads them** (9640112): pages wholly under a
    filled texture are cold (programs read the image, never the page) and
    lose their card page first; a buffer view warms its range; a page
    programs read that lost its card page takes a cold page's. Measured
    (ch7): 737 MB of the 1,779 MB shown are texture pages, none of them
    on the card any more, and the other 1,040 MB still compete for 384 MB
    (124,303 evictions of pages programs read). So the rule is right and
    not enough: **the view needs about 1.05 GB on the card and the images
    3.3 GB**, 0.9 GB more than the ceiling leaves together. To decide
    (card memory): raise `CEILING` from 4,608 to about 5,500 MiB (card at
    about 6.7 GB at peak; the watchdog stands at 7.7), or find the room
    in the 2.2 GB of card-written images.
  - **Tools**: `replay` marks stores outside the capture's memory map
    (`stores.txt`: "OUTSIDE THE MEMORY MAP"); `VICTORY_VK_TRACE_DRAWS=Ped`
    lists every draw of the named programs in the first dumped frame with
    the title's indirect records; the first images not made are named;
    `VICTORY_VK_CEILING_MB` / `VICTORY_VK_ROOM_MB` (diagnostics);
    `examples/ops` lists a program's console instructions.
  - **Open**: the card memory of the pool's view and the images (the
    decision above; until then the head, the mesh and the buildings come
    and go with what is evicted); the character black in the final frame
    though whole in the albedo (indirect light); the window's mesh not in
    every run; the lettering; captures lack the descriptors programs take
    from the resource heap by index (replays of the skinning and culling
    programs read zeros for them) and the indirect records of mesh draws;
    a title crash 44 s into one run's loading (ch8's first try, another
    build running beside it; not seen again).
- 2026-11-27 (code tables): a program's constant tables are memory it
  reads by address, not variables of its module (branch
  `worktree-agent-a237c60fd3ce29af8`; offline only, awaiting a bake).
  - **Why**: NVIDIA keeps a module's own variables (`Private` and
    `Function` storage) on the card for 32,768 threads, sized by the worst
    pipeline of each stage kind, outside every heap's count, for good. The
    tables a console program keeps after its code were private arrays with
    initialisers: CS_RaytraceVBuffer_RayBinning_ShaderBinning_Mixed_WithFar
    (6d7fe7745ab9c6d8) had nine, 47.7 KiB, 1.7 GiB on the card at its
    compile. No shape of variable avoids it.
  - **Now** (`docs/vulkan-abi.md` section 3.1, ABI v5 amended): the
    translator hands the tables' bytes to the host (`VkShader::tables`,
    overlapping tables sharing bytes: that program's are 9.2 KiB); the
    host keeps them in one 2 MiB host-visible buffer (`vulkan/tables.rs`),
    whose address is in the memory map (byte 24, padding until now); the
    module loads a dword at that address + its place in the buffer
    (specialization constant 1) + the table's offset. Modules of programs
    without tables do not change (the stage cache's files do: the tables
    are in them).
  - **The driver's cache**: a specialization constant is part of a cached
    pipeline's key, and a buffer's address differs from process to process:
    as the constant it would have made every run compile every program
    with tables again (155 of the used 3,408; the last bake's 75 stages
    took 403 s). So the constant is the place in the buffer, given once
    and kept beside the translations (`code-tables.txt`, as
    `immediate-samplers.txt`): the same in the bake and in every run.
  - **Measured offline** (the used set, 3,408 modules): the most bytes of
    variables a module has went from 47,748 to 11,460; 155 modules
    changed, all pass spirv-val; the golden file's behaviours not at all.
    The pack: 662 of 25,889 programs have tables, 300 KiB, 139 KiB as 217
    distinct sets.
  - **The next worst were the picks** (a commit of its own): `v_movrels_b32`
    reads the register `m0` names (an array the console program keeps in
    its registers), which was a function array per pick, stored whole and
    read at the index: the 44 picks of CS_CompressBC7Mip0 were 11.4 KiB
    of arrays (357 MiB on the card), BC1's 10.4 KiB, BC3's 7.2 KiB. A pick
    is selections on the index's bits now, no variable; 22 used modules
    change, each smaller. The most variable bytes of a used module is
    then 652 (compute 524, mesh 652, pixel and vertex 516: about 16 to
    20 MiB a stage kind on the card), all but a few bytes of it the
    128-entry table of buffer formats (`Helper::TypedShape`, in 1,429
    modules) and a split mesh program's table of parts.
    Over the whole pack (25,694 modules) the most is 664. The host's
    refusal of a module over 16 KiB a thread (the entry below) stays as
    a guard and refuses nothing now: the ray-trace program compiles again.
  - Open: the format table could lie in the tables buffer too (one fixed
    place for every program, no constant), for a compile of those 1,429
    modules; the tables buffer is PC memory the card reads (as title
    memory is): on the card if a program's table reads show up in frame
    times.

- 2026-11-27 (later): **card memory accounted for** (runs vm1–vm7,
  `docs/vulkan-host.md` "Card memory the driver takes"). vk103's 6.9 GB
  for the process were 3.3 GB of ours (images 2.56, scratch 0.48, pool
  0.26), 1.4 GB of the driver's inside the heap's count and 2.1 GB
  outside any heap's count. Both driver parts come from pipelines:
  - **Outside the heaps** is the driver's per-thread local memory: a
    program's own variables (SPIR-V `Private` and `Function`) are kept
    on the card for 32,768 threads, as big as the worst pipeline of each
    kind of stage, and never given back. The constant tables of a
    program's code (the tone-map fix of tag `batch-2026-11-27b`) are
    `Private` arrays: the nine tables of one ray-trace program (47.7 KiB)
    held 1.7 GiB. That is the ~1.2 GB by which runs on tag b sit above
    runs on the tag before. No shape of variable avoids it (probed);
    **tables must become memory read by address: a translator change,
    open.** Until then a module over 16 KiB a thread is refused before
    it compiles (today that program alone; its dispatches are skipped,
    and with the structure builder off its rays find nothing anyway).
    Next: the BC1/3/7 compressors (357 MiB for 44 scratch arrays).
  - **Inside the heap** are the pipelines themselves, in arenas that
    double and are never given back (~1.06 GiB for 3,624 stages). The
    game's background bake compiled the 266 slow stages a second time
    in every run (368 MiB): it leaves them alone now.
  - Ours: every command buffer kept the most scratch it ever used
    (478 MiB, growing); scratch is lent by the timeline now (39 MiB).
    The images are the title's own targets (2.2 GB over 2.0 GB of title
    memory) and textures (0.5 GB), not copies kept past use. The seven
    3840x2160 targets (233 MiB) are the title's interface plane and its
    depth and colour targets: it reads Series S, renders the world at
    1280x720 into a 1920x1080 plane and still draws its interface at
    3840x2160, whatever display mode is listed.
  - The card-memory line adds up now: `process_mb` (what `nvidia-smi`
    shows for the process) `= ours_mb + driver_mb + outside_mb`.
  - With the driver's share smaller the images get room. On tag
    `batch-2026-11-27c`: int56 (run vmi56) was killed by the watchdog at
    frame ~1,830 with 7.7 GB on the card (the process 6.9 GB, 45,901
    images refused at its limit); with these changes (run vm7s) the card
    peaks at 6.2 GB (the process 5.4 GB), no kill, frame 2,310 in the
    same 420 s, 1,461 images refused: the ground is lit, with shadows.
    RAM peaks 0.45 GB higher (17.07 against 16.61 GB): not traced, open.
  - Storage usage on every colour image (899741d) costs no card memory
    (168,918 images: the same size and alignment with and without it).

- 2026-11-27 (indirect light): why everything the sun does not hit was
  black, found by walking back from the tiled lighting with pass captures
  (runs gi5 to gi9, branch `worktree-agent-a8a715585abfc9ff1`).
  - **The buffers**: the tiled lighting samples indirect diffuse and
    specular from two 1280x720 shared-exponent textures (0x9bb910000,
    0x9be0e0000); both were all zeros, and so was every stage back to the
    screen probes' ray radiance (2048x120, 0x9bb700000 / 0xafa840000) and
    the sky cube (0x9d4eb0000). The miss path itself works: with every ray
    a miss, CS_ScreenProbe_RayMiss writes sky radiance into its ray buffer.
  - **Host, fixed (899741d, shown in run gi9)**: the title stores packed
    R9G9B9E5 words through 32_UINT unordered views of textures whose
    resources do not say they take unordered access. Our image then had no
    storage usage, the view's storage binding stayed null and the stores
    went nowhere (CS_ScreenProbes_Inject, the final gather, the sky cube).
    Every colour image now takes storage where the driver gives it; a
    shared-exponent texture that is no render target is a 32_UINT image;
    an unordered view without a storage view is a warning. After it the
    ray radiance, the final gather's output and both lighting inputs hold
    light (diffuse mean 0.06 / 0.03 / 0.03, specular 0.6 / 0.7 / 1.2), the
    sky cube has its blue (the sky in the lighting buffer is pink-white,
    not brown) and the bunk shows; walls and the character are still far
    too dark.
  - **Translator, awaiting a bake (7756993)**: a gather of an integer
    view went through the float image and read nothing on the card.
    CS_Tiling_Generate_Opaque gathers the stencil of the material classes:
    no tile got a class, its lists stayed empty (0xaf3630000) and every
    tiled pass after it (reflection rays, their filters and resolves) ran
    on no tile. A gather is by the T#'s integer bit now, as a load; the
    host's stencil views hold the stencil in every channel (the programs
    load it from the first and gather it from the second). Replayed: the
    program fills its six lists (681 / 24 / 24 / 657 / 657 / 681 tiles).
  - **Captures**: an indirect dispatch left its capture open (the next
    program's push and dispatch lines went into it) and its own arguments
    were not in it: closed and named now (`dispatch-indirect`). Captured
    memory is what the CPU sees when the pass is recorded: lists and
    counts the card writes in the same frame are a frame old or just
    cleared (open: read them from the card, as the images). Replay writes
    `reads.txt`, `stores.txt`, `images.txt` (`SHOW`, `UNDECODED`, `SPV`).
  - Open: walls still dark with sky light and no occlusion (expected:
    too bright), the builder (docs/fork-ray-tracing.md, record.rs
    `bvh_builder`), the dither pattern.

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
- 2026-11-27 (later): **why the final frame was grey: the tone-map table
  was one value** (branch `cg-composite`; the translator fix awaits a
  batch bake).
  - **Walked back with captures** (runs cg1, cg2): PS_CompositeNMB replayed
    from its captured inputs gives the card's grey (0.19-0.22); the input
    that greys it is the 64x64x64 RGBA16F table at 0x99fd00000, every
    texel 0.2063 (0x35 of 255: the grey on screen). Exposure is sound (the
    title clamps it to -5.5 EV, 0.0221; average luminance 5.76; replay =
    card), so are the 33-cube grading tables and the two 4096x1 curves.
  - **Its writer**, CS_BakeACESWithGamutCompressLUT_QuasiLogShaper (no
    image inputs, dispatch 4x4x64), gives the same flat table in the
    interpreter: it read 28 bytes at address 0x320c nine million times.
    That is its own code: the ACES spline coefficients (-4, -4, -3.157,
    -0.485, 1.848 ...) sit after `s_endpgm` and are read by
    `global_load_dwordx3` at `s_getpc_b64` + a constant + a segment index
    each lane holds (`v_lshlrev_b64`, `v_add_co_u32`, `v_add_co_ci_u32`).
    Only scalar loads knew code tables; a vector load read title memory
    at the code offset: the zero block.
  - **Fix (translator, lift_mem.rs `flat_code_load`)**: a `global_load_dword*`
    whose address is made of an `s_getpc_b64` plus a constant through
    additions, copies, selections and phis reads the code's table
    (`Helper::CodeTable`). Oracle:
    `constant_tables_in_the_code_read_by_vector_loads` (fails before), and
    a vector load elsewhere in such a
    program stays memory. Replay: the table is a tone curve (0 .. 1), no
    missing memory. Golden: 25 of 1,527 programs change, each with its
    table after the code: this bake, 19 CS_InterpolateStrands (hair: a
    table of jitter offsets), CS_MainBlur (depth of field: tap counts and
    offsets), two CS_ReconstructRaymarchClassified*,
    CS_ScreenProbes_DownsampleDepthCoordMinMaxBuffer. The 25 modules pass
    spirv-val.
  - **Shown on the card without a bake** (run cg3, branch `cg-exp-lut`, an
    experiment not to merge: the replayed table copied over the card's
    before each composite): the final frame shows the opening shot (the
    window, the yard, chairs, a long shadow, the low sun). The card's
    10-bit target after the composite equals the interpreter's within one
    step (129,600 pixels, worst 2.2), so the composite and the copy to
    the back buffer are right. What it shows is still wrong upstream:
    the picture is black and amber only (the HDR buffer itself holds
    3.3 / 0.81 / 0.11 in the yard, the sky is brown), the crosshatch
    pattern of the HDR targets, a black interior, dark blocks in the sky.
  - **Tools**: a capture's small memory blocks are written by a thread of
    their own (thirteen thousand files in one folder held the recording
    threads half a minute and the title gave up: run vk100 saved no
    images; six passes now capture in a second). `replay` takes a pixel
    program's inputs per pixel (`INPUTS="loc.comp=a,b,c;..."`: a + b x +
    c y) and prints nodes (`NODES`): PS_CompositeNMB takes its texture
    coordinates and the exposure from VS_PassthroughComposite, and the
    interpreter's made-up inputs replayed it as noise. (For it:
    `INPUTS="1.0=0,0.00052083333,0;1.1=0,0,0.00092592593;1.2=<exposure>;2.0=<EV>"`.)
- 2026-11-27 (late): wave64 and waterfalls, proven offline (branch
  `worktree-agent-af78a0a95bd2278bc`; nothing baked or run).
  - **Tests**: origin/main's wave64 and waterfall cases are in
    `vk/oracle/wave64.rs` (24 run, 2 ignored: a dispatch that counts
    threads, sums of denormals) and `waterfall.rs` (their 5, and a
    waterfall of a wave of 64 across its halves). All passed as ported:
    no translator difference found there.
  - **The interpreter runs a group as the card does**
    (`vk/interp_group.rs`): the two halves of a wave64 as machines of
    their own that meet at the lift's exchanges, and console barriers
    across waves. Every wave64 oracle case runs as one wave and as halves
    in both orders and must agree. Over the used set (`cargo xtask
    shader-golden --halves`, eight seeded runs each): of 1,140 wave64
    compute and mesh programs 1,002 the same, 114 past the block limit
    (nothing to compare), 24 the same in the runs where no half stored to
    a dword the other touched between two meetings (random descriptors
    that collide; 13 of them differ in such a run), none different
    otherwise. The card's transport of an exchange (barriers, counters:
    `emit_xchg.rs`) is not interpreted.
  - **Pixel programs** have no other half: a fragment subgroup is a
    console wave whose other lanes hold no work. `lift_hi.rs` finds what a
    mask read as a number holds for those lanes; the 76 used programs
    listed as approximate are none now. Six ped programs read 0 for the
    upper dword of an inverted mask (the console: all ones); eight
    tiled-light programs ran a section's tree lane by lane for its vector
    result; one program kept a number in `vcc_lo` and read it back through
    a ballot (wrong in a partly filled subgroup: a buffer offset).
    17 modules change (-333 to +114 words; the largest ones not), the
    golden file's 1,527 behaviours not. `docs/translation-index.md` (the
    whole pack) still has the old notes: `cargo xtask translation-index`.
- 2026-11-27 (decision, the user's): the card ceiling is raised from
  4,608 to 5,504 MiB now, and is to come down again as room is found (the
  ~2.2 GiB of card-written images; the seven 3840x2160 interface
  targets). At 4,608 the pool's GPU view was starved and parts of the
  scene came and went.
- 2026-11-27 (12:30): run vk106 (exe int58 = integrate 7c6e839, host
  changes only on the `batch-2026-11-27d` cache; switches
  `VICTORY_AMBIENT=constant:0.5 VICTORY_VK_CEILING_MB=5312`; frame
  /mnt/data/victory/capture/vk106/frame-2100.png): the reference's shot
  in layout and most of its content — the character from behind with the
  ponytail, the window with its chain-link mesh, building, watchtower,
  people in the yard, grey walls, the bunk, the posters. Still wrong: the
  jumpsuit is dark red without its lettering, the window frame black
  (blue-grey in the reference), the light flat (a constant stands in for
  indirect light), the sky too white. Card 6.49 GB at peak (process
  5.65), RAM 16.5 GB, card 423 ms a frame, no fault.
  - Both switches are test settings: the ceiling default (4,608 MiB) is
    the user's to decide (the pool's GPU view needs ~1.05 GB), and the
    constant is to become an estimate from the frame before
    (VIctory-NVIDIA's ambient.rs is the model).
- 2026-11-27 (10:25): run vk105 (tag `batch-2026-11-27d`, exe int57;
  frame /mnt/data/victory/capture/vk105/frame-2430.png): code tables are
  memory read by address (one 2 MiB host buffer, a place per table set
  kept in `code-tables.txt`), picks are selections; with the card-memory
  cuts the process holds 4.95 GB on the card (outside the heaps 392 MiB,
  was 2.1 GB), card total at peak 5.79 GB (was 7.7), no watchdog, RAM
  16.5 GB, card 324 ms a frame. The bake took 545 s (178 stages
  compiled). Picture: lit window frame, pale sky, golden yard with long
  shadows, a tree, buildings, the bunk. Against the reference still: the
  character is not there (hair only in some frames), interior walls
  dark, no window mesh, the image smeared, the bunk too white.
- 2026-11-27 (10:00): run vk104 (tag `batch-2026-11-27c`, exe int56;
  frame /mnt/data/victory/capture/vk104/frame-2400.png): the sky is pale
  and warm as in the reference, the watchtower and buildings stand
  against it, the bunk at the right shows. Still wrong: interior walls
  and the character's body dark, the yard's ground dark, no window mesh,
  dither. What changed: colour images take storage views (stores of
  packed colour into textures made without storage use went nowhere: ray
  radiance, final gather, the sky cube), and gathers of integer views
  read integers (the tile lists).
  - **Card memory**: the driver keeps a module's `Private`/`Function`
    variables for 32,768 threads on the card, outside every heap's count:
    the code tables as private arrays cost 1.7 GB for one program (nine
    tables, 48 KiB). Tables are to become memory read by address (being
    built); host stopgaps on the card-memory agent's branch (process on
    the card 6.9 -> 5.4 GB).
  - The bake after the gather fix: 3163 s (799 stages compiled).
- 2026-11-27 (07:30): **the test scene is on screen** (run vk102, tag
  `batch-2026-11-27b`, exe int55; frame
  /mnt/data/victory/capture/vk102/frame-1380.png): window, yard, building,
  watchtower, the character's hair, low sun. Against the user's reference
  (/mnt/data/victory/reference/opening-shot-target.jpg) still wrong: black
  and amber only (blue is near zero in the HDR buffer already), the
  interior and the character's body black (no indirect light), no window
  mesh, dark blocks in the sky, dither.
  - The bake after the tone-map table fix took 403 s (75 stages compiled,
    the rest from the fresh driver cache).
  - **Card memory**: both runs on this build were ended by the watchdog at
    world load (7.4 and 7.7 GB on the card). Our Vulkan heap use is 4.7 GB
    (images 2.56, buffers 0.4, pool 0.26, driver 1.4) and the budget the
    driver reports has shrunk to 4.7 GB: about 3 GB on the card is the
    process's but not in that heap's count. Not understood; next to trace
    (it also decides whether a run survives the load).
- 2026-11-27 (morning): **the world is lit in the lighting buffer** (run
  vk98, batch `batch-2026-11-27` baked into a fresh driver cache, exe
  int54): class depth written (458k texels with classes), the HDR target
  after the tiled lighting holds the opening shot (window, yard in low
  sun; max 987 / 211 / 99), no infinities; the upscaled 1920x1080 HDR
  (0x99e980000) holds it too at present. Still wrong:
  - **the final frame is grey**: the 10-bit target after the composite
    (0xb3a8f0000) and the back buffers are flat; the picture is lost in
    exposure / colour grading / `PS_CompositeNMB`. Captures of those
    passes are the next step (replay them).
  - interiors black (no ray-traced bounce light: the builder is off), a
    dither pattern in the HDR targets, card 330-470 ms a frame now that
    lighting runs (2-3 frames a second), RAM 21-26 GB in diagnostics runs
    (a full target dump alone adds gigabytes), VRAM 6.5-7.2 GB.
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
  - **The driver's cache file fills up**: after several full bakes into one
    directory its data file passed 4 GiB and its index was no longer
    written (frozen at 01:31 while the data grew to 5.6 GiB): even a bake
    that ended cleanly left the next game compiling everything. A bake
    after a translator change starts a fresh driver cache directory (the
    old one set aside as cache/nv.old-<date>; the launcher is to do it
    when a cache file is past 3 GiB).
  - Open: in game, on-demand stage loading stalls behind slow compiles
    when the driver cache misses (translated stays at a few hundred).
  - 64-lane approximations in the used set: 76 pixel programs (character
    shading, particles, mirrors); the wave64 and waterfall tests of
    origin/main are not ported yet, and the interpreter runs a wave as
    one, not as the two halves the card runs. (Done: the entry above.)
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
