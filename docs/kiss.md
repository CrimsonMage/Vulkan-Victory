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
| Unified memory: the CPU writes, the GPU reads the same pages | One CPU-visible D3D12 heap (ReBAR video memory, spilling to system memory); resources placed in it at the title's offsets; sync at fences, submits and DirectStorage completion | `victory-gpu` (new `unified.rs`), `victory-hle/xmem.rs` | **probe built**: title memory stays in system RAM and is *opened* as D3D12 heaps (`OpenExistingHeapFromAddress1`/`FromFileMapping`); a ReBAR GPU-upload heap can't hold textures or sit at title addresses on this stack. Replaces write-watch, guards, snapshots and copies (~17k lines) |
| D3D12.X device and command lists (`d3d12_xs`) | D3D12, call for call where a PC call exists | `victory-gpu` (`device`, `list`, `draw`) | works; pruning experiments |
| Console GPU code (RDNA2 machine code, no HLSL or DXIL shipped) | Decoder → IR → **HLSL text → DXC** (`dxcompiler.dll` beside the exe) | `victory-shader` (new `hlsl/`) | **prototyping**; today a hand-written DXIL encoder |
| Console tiled texture layouts | Detile once when the data arrives (resource creation, DirectStorage completion, title copies) | `victory-gpu/texture.rs` | works; to move to explicit points |
| Ray tracing (console BVH format, title's own builder DLL) | Victory's builder in DXIL (`VICTORY_BVH_BUILDER`); until then rays miss (`VICTORY_BVH_MISS`) | `victory-shader/bvh.rs`, `intersect.rs` | partial |
| 64-lane waves | 32-lane PC waves with emulation in the translator | `victory-shader` | works; keep |
| Console DirectStorage (`XCRDAPI`) | One shared handle per file; each request does one positioned read of exactly its source range when it runs (uncompressed straight into title memory, compressed into a reused worker buffer, then inflate/unpack on the CPU); one FIFO worker pool, no lock per request (`dstorage streaming` line every 10 s) | `victory-hle/dstorage.rs` | built, not yet run |
| XMA audio hardware (`AcpHal`), Opus (`xapu`) | Software decode → host audio (XAudio2 / WASAPI) | `victory-hle/audio*`, `apu.rs` | works; reverb (`chad` XDsp) silent |
| Bink 2 video | The title's own Bink DLL, with XAudio2 passed through | title DLL | works |
| GameInput, window, lifecycle | GameInput on PC; a Win32 window | `gameinput.rs`, `lifecycle.rs` | works |
| XUser, XStore, saves, networking | PC GDK twins or local stand-ins; no network | `xuser.rs`, `xstore.rs`, `xgamesave.rs`, `net.rs` | works |
| Console CPU instructions the PC lacks (CLZERO, MONITORX, INSERTQ ...) | Rewritten as plain x86 at the first trap (stubs for clzero loops and INSERTQ; PAUSE/NOP for MONITORX/MWAITX); emulated only as fallback | `victory-loader` | works |

## Targets

- **At least 30 fps everywhere**: menus, loading, cutscenes and the world.
  More is better (the title's own cap is 30; see workstream 10 for 60).
- **Low memory**: as little card memory and system memory as the content
  needs, not a mirror of everything.
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

## Rules

- Prefer an explicit API point over detecting side effects in memory.
- One behaviour, one code path. No `VICTORY_*` switch for experiments; a
  switch exists only for a user-facing choice (paths, logging, console
  profile, capture for testing, the few listed in the inventories).
- Linux runs exactly the Windows build under stock Proton, with app-local
  DLLs only: a current vkd3d-proton, `d3dcompiler_47`, DXC.
- Every change passes the gate before it lands.

## Gate

`VICTORY_CONSOLE=series-s /mnt/data/victory/autorun.sh <name> 300 "$(cat /mnt/data/victory/keys-start.txt)"` (5 minutes)
on the diagnostics build, with the GPU-memory watchdog: it boots, reaches
the world, saves captures and logs frame time and card memory. A change
lands on `integrate` only if the run reaches as far as before, with no
driver errors and no new crash, and the captures look no worse.

## Measures

| | Start (2026-11-25) | Now |
|---|---|---|
| Rust lines | 128,292 | 106,472 |
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
| 10 | 60 fps: hand the title a 16.7 ms frame interval (its own request is 33.3 ms, `frame.rs`) | — | after a steady 30 in the world |
| 8 | HLSL backend replaces the DXIL encoder | — | after 5 and 6 |

## Decisions log

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
