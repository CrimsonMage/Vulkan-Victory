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

## Status (2026-11-27)

**Not playable.** Tested with one title, on one machine: Linux (stock Proton), an Intel CPU,
NVIDIA RTX 5060 Ti 8 GB, 31 GB RAM; the Series S profile.

| | |
|---|---|
| Boot, menus, opening | reached by scripted input in every test run |
| The world | the opening scene is on screen and matches the console in layout: geometry, characters, sky, direct sun, an estimate in place of indirect light |
| Known to be wrong | surface colours differ from run to run (streamed textures are dropped and their descriptor slots reused); no ray-traced light or shadows (the title's structure builder is left out); over-exposed sky; lettering on clothing missing |
| Frame rate | about 2 to 3 frames a second in the world (the card needs 320 to 420 ms a frame) |
| Memory | about 16.5 GB of RAM and 6.7 GB on the card at peak |
| Shaders | 25,694 of 25,889 console programs translate (the rest are unused geometry programs); the used set is compiled ahead of time by `victory bake` |

How it works, what was found and what is open: `docs/kiss.md` (the
decisions log), `docs/vulkan-host.md`, `docs/vulkan-abi.md`.

The indices the tools write from a dump (`docs/pipeline-index.md`,
`docs/translation-index.md`, `docs/shader-golden.txt`) are generated on your
machine and are not part of this repository.

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
