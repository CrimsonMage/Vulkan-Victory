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
| Unified memory: the CPU writes, the GPU reads the same pages | One CPU-visible D3D12 heap (ReBAR video memory, spilling to system memory); resources placed in it at the title's offsets; sync at fences, submits and DirectStorage completion | `victory-gpu` (new `unified.rs`), `victory-hle/xmem.rs` | **designing**: replaces write-watch, guards, snapshots and copies (`shadow`, `watch`, `tiles`, `inplace`, `copystat`) |
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
| Rust lines | 128,292 | 128,292 |
| `VICTORY_*` switches | 224 | 224 |
| `diagnostics`-only blocks | 283 | 283 |
| World frame time (Proton, 8 GB card) | 0.5–5 s | — |

## Workstreams

| # | Work | Owner | State |
|---|---|---|---|
| 1 | Prune `victory-gpu` switches and statistics | subagent | running |
| 2 | Prune `victory-shader` switches; equivalence sample of 1,000 shaders | subagent | running |
| 3 | Prune HLE, loader, formats, app | subagent | running |
| 4 | Unified memory design and probe (`VICTORY_UNIFIED`) | subagent | running |
| 5 | HLSL + DXC shader backend (`VICTORY_SHADER_BACKEND=hlsl`) | subagent | running |
| 6 | Merge 1–3, gate run | coordinator | waiting |
| 7 | Unified memory replaces mirroring; delete `shadow`/`watch`/`tiles`/`inplace` | — | after 4 and 6 |
| 8 | HLSL backend replaces the DXIL encoder | — | after 5 and 6 |

## Decisions log

- 2026-11-25: stock Proton install; a current vkd3d-proton beside the exe
  (the bundled converter fails on ~18% of shaders). No patched wineserver.
- 2026-11-25: memory ceilings plus an external GPU-memory watchdog (the
  NVIDIA driver corrupted itself and hung the kernel twice at ~7.7 of 8 GB).
- 2026-11-25: reverted reading DirectStorage page lists at enqueue (loading
  failed 4,922 times with it; 0 without).
- 2026-11-25: dropped the shader control-flow reshaping pass (not needed
  with a current vkd3d-proton).
