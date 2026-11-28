# Vulkan-Victory (work in progress)

A work-in-progress fork of [VIctory](https://github.com/Marvisak/VIctory), a
compatibility layer that runs an Xbox Series title natively on a PC from
files the user supplies. This fork runs the title's Direct3D 12.X on
**Vulkan directly**: its precompiled console GPU programs are translated to
SPIR-V ahead of time, and the layer is kept thin. It is developed on Linux
under stock Proton with an NVIDIA card.

**It is meant to work with a dump of any Xbox Series title**, not one game.
The loader, the reimplemented console services, the Direct3D 12.X surface
and the shader translator follow what the console does, and what a title
needs (its shaders, its pipelines, its memory layout) is taken from the dump
the user supplies at run time. That is the goal, not yet the state: it has
been developed and tested against one title only, and a few pieces still
key on that title (the stand-in for indirect light and some diagnostics
find their place by the names of its programs; per-build hooks in the
loader). Other titles are expected to need work where they use console
features this one does not.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's
developer or publisher.


## Credits

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors is the base of everything here: the loader, the runtime
  services, the Direct3D 12 back end and the shader recompiler. Licensed
  GPL-3.0-or-later; so is this fork.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)** is the
  fork this one grew out of; its Direct3D 12 path, which renders the same
  title lit on NVIDIA cards, is the reference the Vulkan translator was audited
  against (`docs/origin-findings.md`, `docs/opcode-audit.md`), and its
  instruction and wave tests are ported here as offline tests.
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)**:
  its instruction tests (`valu_tests.rs`) were ported first
  (`crates/victory-shader/src/vk/oracle/`).
- **[trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)**:
  the idea of estimating indirect light from the frame before while the
  title's ray tracing is off comes from its `ambient.rs`; the pass here was
  written anew for Vulkan.
- **[XCL](https://github.com/ivall/xcl)** by ivall and its contributors
  (GPL-2.0-only) was used as a reference for how the same title behaves under
  another layer. Fixes inspired by it were written anew in VIctory's code.
- The guard over every view of a memory page is code by **myokex**
  ([victory-amd](https://github.com/myokex/victory-amd), commit `c6dc4d8`), which
  adapts pull request #115 here by Spexivus, its correction by joroyt and a part of
  VIctory #1122 by tightfall. The read-beside fix for kernel reads into title
  memory is by **Joshua Odom** (commit `10ea38e` of the same fork family). Both
  are carried here unchanged apart from one merge conflict.
- This fork's changes were developed with [Claude Code](https://claude.com/claude-code).

## Where it stands (2026-11-28)

**Not playable: about 9 frames a second.** Tested with one title, on one
machine: Linux (stock Proton), an Intel CPU, NVIDIA RTX 5060 Ti 8 GB, 31 GB
RAM; the Series S profile.

| | |
|---|---|
| Boot, menus, opening | reached by scripted input in every test run |
| The world | the opening scene matches the console's picture: geometry, characters with their materials and lettering, hair, the sky, direct sun, fog; surface colours are the same every run |
| Indirect light | an estimate from the frame before stands in for it: the title's ray tracing does not run (its structure builder is left out), so there is no ray-traced bounce light, reflection or shadow |
| Known to be wrong | over-exposed sky; flat interior light; lettering and hair less sharp than the console's; a few triangles of a character's clothing go dark for a frame now and then; the title sometimes crashes during the load |
| Frame rate | about 9 frames a second in the world (the aim is 30): the card needs about 100 ms a frame, spread over thousands of small dispatches and draws, and the CPU's side of a frame about 65 ms |
| Memory | about 17.3 GB of RAM and 7.2 GB on the card at peak (the aim is a Series S's amounts) |
| Shaders | 25,694 of 25,889 console programs translate (the rest are unused geometry programs); the used set is compiled ahead of time by `victory bake` |

`docs/kiss.md` is the working record: the principle, the Xbox → PC map,
the rules, the measures and a dated decisions log of every finding (what
was wrong, how it was found, what fixed it). `docs/vulkan-host.md` and
`docs/vulkan-abi.md` describe the Vulkan host and the contract between
translated programs and the host; `docs/origin-findings.md` and
`docs/opcode-audit.md` compare the translator with the Direct3D 12 path it
was audited against.

The indices the tools write from a dump (`docs/pipeline-index.md`,
`docs/translation-index.md`, `docs/shader-golden.txt`) are generated on your
machine and are not part of this repository.

## What was removed from VIctory, and why (KISS)

The fork this grew out of had become hard to reason about: about 128,000
lines of Rust and over 200 `VICTORY_*` switches, with the same behaviour
reachable through several paths. The rule since 2026-11-25 is *keep it
simple*: **each Xbox-specific behaviour is turned back into the plain PC
API it was ported from, at the moment the title asks for it; nothing is
watched, guessed or mirrored when an API call already says what happens;
one behaviour has one code path.**

| Removed | Why | What stands in its place |
|---|---|---|
| The Direct3D 12 back end (device, host, draw binding, descriptor host heaps) and with it vkd3d-proton, DXC and `d3dcompiler` beside the exe on Linux | Two GPU back ends meant every fix twice; on Linux D3D12 was itself a translation to Vulkan, with translated shaders compiled twice | The title's D3D12.X recorded straight into Vulkan (`crates/victory-gpu/src/vulkan`) |
| The DXIL shader translator (`emit/` with its bitcode encoder, HLSL helpers, resource and structure analysis: about 40,000 lines) | Its programs carried a page-directory lookup on every memory access and took 0.5 to 10 s each to compile | A translator to SPIR-V in which title addresses are device addresses through a small memory map (`crates/victory-shader/src/vk`, contract in `docs/vulkan-abi.md`) |
| Memory mirroring: the 64 KiB page directory and card page cache, unified/shadow/gather/indirect views and windows, write watches, guards and canary pages | The title's memory was copied and re-checked every frame; this was the main cost in time and memory | Title memory imported into Vulkan by host pointer (zero copy); the few copies the card needs are refreshed when a write is seen, not by polling |
| The D3D12 pipeline catalog, library and translated-shader cache | They existed to hide slow compiles at run time | Stages translated once and kept on disk; `victory bake` compiles the used set before playing |
| D3D12 FSR, compose and capture paths; the old frame-time and process-memory lines | Tied to the removed back end | One Vulkan composition pass; one log line every 10 s |
| About 180 `VICTORY_*` switches (215 → 38) | Experiments had been left in as switches; nobody could say which combination was "the" behaviour | A switch exists only for a user-facing choice or a named diagnostic (`docs/switches-gpu.md`) |
| The ray-tracing structure builder on the DXIL path | It went with the D3D12 back end | Not yet rebuilt on the SPIR-V path: this is why indirect light is an estimate today |

Measured against the point where this work split off (`0b0074e`):
Rust went from 127,160 lines to 100,007, of which about 10,000 are tests and
tools that did not exist before (an interpreter of translated programs,
instruction tests ported from the Direct3D 12 path, a per-program behaviour
check, pass captures and their replay). `victory-gpu` went from 54,037 to
25,887 lines and `victory-shader` from 42,540 to 32,366.

What was kept on purpose: the loader, the reimplemented console services
(`victory-hle`), the audio and storage paths and the format code, which
already followed the rule.

## Getting started

Requirements: Windows 10/11 x64 (or Linux under stock Proton), a CPU with
AVX2, a Vulkan 1.3 GPU with buffer device address and descriptor buffers
(and mesh shaders for the world; `victory vkprobe` checks), Rust (pinned in
`rust-toolchain.toml`).

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p victory -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p victory
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Input: XInput pads. With none connected the keyboard is a pad while the
title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--list] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
victory vkprobe
victory bake [--jobs N] [--index <md>]
```

`victory bake` compiles the shader stages the title has used so far into the
driver's cache before playing (the first bake of a dump takes about an hour
on the test machine; later ones compile only what changed).

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

The translation index, `docs/translation-index.md` (what the shader
translator does with every program of a dump: opcode coverage, refusals,
approximations), is regenerated natively on Linux or Windows with
`cargo xtask translation-index --pack <shaders file or directory> [--out <md>] [--jobs N]`.

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_VK_DUMP`, `VICTORY_VK_STAGES` and the other `VICTORY_VK_*` | Vulkan backend diagnostics (`docs/vulkan-host.md`) |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
