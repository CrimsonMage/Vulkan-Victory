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
| Console DirectStorage (`XCRDAPI`) | Plain file reads (or PC DirectStorage), decompression on the CPU | `victory-hle/dstorage.rs` | works |
| XMA audio hardware (`AcpHal`), Opus (`xapu`) | Software decode → host audio (XAudio2 / WASAPI) | `victory-hle/audio*`, `apu.rs` | works; reverb (`chad` XDsp) silent |
| Bink 2 video | The title's own Bink DLL, with XAudio2 passed through | title DLL | works |
| GameInput, window, lifecycle | GameInput on PC; a Win32 window | `gameinput.rs`, `lifecycle.rs` | works |
| XUser, XStore, saves, networking | PC GDK twins or local stand-ins; no network | `xuser.rs`, `xstore.rs`, `xgamesave.rs`, `net.rs` | works |
| Console CPU instructions the PC lacks (CLZERO, MONITORX, INSERTQ ...) | Trapped and emulated | `victory-loader` | works |

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

## Rules

- Prefer an explicit API point over detecting side effects in memory.
- One behaviour, one code path. No `VICTORY_*` switch for experiments; a
  switch exists only for a user-facing choice (paths, logging, console
  profile, capture for testing, the few listed in the inventories).
- Linux runs exactly the Windows build under stock Proton, with app-local
  DLLs only: a current vkd3d-proton, `d3dcompiler_47`, DXC.
- Every change passes the gate before it lands.

## Gate

`/mnt/data/victory/autorun.sh <name> 600 "$(cat /mnt/data/victory/keys-start.txt)"`
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
| World frame time (Proton, 8 GB card) | 0.5–5 s | 390–470 ms (2.1–2.5 fps), Series S, world lit, card 6.8 GB of 8 |

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
| 17 | Console CPU instructions: ~21,000 SIGILL traps/s (CLZERO, MONITORX/MWAITX, INSERTQ) emulated per trap; patch each site once | subagent | running |
| 18 | Per-draw recording cost (~0.28 ms CPU per draw: shadow::find under a contended mutex, bind, descriptor mirroring, SipHash) | subagent | running |
| 19 | Memory: texture levels on demand by default, render targets on shared pages keyed by title address, open only committed runs (not whole regions) | subagent | running |
| 20 | Later: `globallycoherent` UAVs for the BVH builder (XCL found the builder needs it) | — | after fps/memory |
| 16 | Try XCL (github.com/xcl-dev/xcl) for ideas: builds and starts under stock Proton (`/mnt/data/xcl/run.sh`, own prefix); backs title memory with sparse VRAM windows + 512 MB coherent window | coordinator | stalled before its first frame under Proton (title graphics memory 4.5 GB > its 4 GB coherent window); idea study stopped |
| 12 | Texture quality setting (`VICTORY_TEXTURE_QUALITY`, medium under 10 GB) and lighter menus | subagent | merged (be13db8) |
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
- 2026-11-25: dropped the shader control-flow reshaping pass (not needed
  with a current vkd3d-proton).
