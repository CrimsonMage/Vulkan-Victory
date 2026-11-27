# victory-shader switches

Every `VICTORY_*` environment variable `crates/victory-shader` read before
the simplification of 2026-11-25, its default, and what became of it. The
rule: the default behaviour is kept, the switch and its other paths go.

The stage cache key (`victory-gpu`'s `vulkan/stage_cache.rs`) is
`victory_shader::translation_fingerprint()`: the hash of this crate's `src`
(made by `build.rs`) and the value of every `VICTORY_*` name quoted in
`src` (also found by `build.rs`).

## Kept

| Switch | Where | Default | Other value |
|---|---|---|---|
| `VICTORY_BVH_BUILDER` | `vk/lift_image.rs` `bvh_builder` (the host reads the same name in `victory-gpu` `vulkan/record.rs`) | unset or `0`: the title's structure builder does not run, and a ray/node test answers "no hit" and reads nothing (a box enters no child, anything else is missed; origin/main `emit/intersect.rs` `no_hit`) | `1`: the builder's dispatches run and node tests read the nodes (`emit_bvh.rs`) |

Its value is part of the stage cache key, so both forms are cached. The
tables below are the history of the removed DXIL translator
(`VICTORY_BVH_MISS` went with `emit/`).

## Removed (default kept)

| Switch | Where | Default (kept) | Removed path |
|---|---|---|---|
| `VICTORY_SHADER_OPT` | `emit/mod.rs`, `shader.rs`, `bitcode/opt.rs`, `lower.rs` | every clean-up pass | pass subsets and `0`. Measured on 1,023 programs (every 25th of the pack): 6.57 M instructions without the passes, 3.78 M with them (-42.5 %), DXIL bytes -33 %. Always on. |
| `VICTORY_SHADER_LOOP_LIMIT` | `emit/mod.rs` (diagnostics feature only) | per-stage limits: 128 pixel, 256 vertex/mesh, 4,096 compute | one named limit for every loop |
| `VICTORY_GROUP_SYNC` | `emit/mod.rs`, `lower.rs` | `2`: every barrier waits for the group | `1` (fence only where not every lane arrives) and `0` (no barrier waits) |
| `VICTORY_SHADER_CENSUS` | `emit/bitcode/census.rs` | off | instruction counts per kind, read by `victory translate`; the census module is gone |
| `VICTORY_WAVE_WHY` | `emit/mod.rs`, `lower.rs`, `own_bits.rs` | off | stderr reasons why a wave64 program keeps 64 lanes or how it was retried |
| `VICTORY_DESC_WHY` | `lower.rs` | off | stderr line per descriptor read only as data |
| `VICTORY_WAVE_LOCAL` | `emit/mod.rs` | on (approximate forms included) | `0` (never each host wave on its own) and `idiom` (exact forms only) |
| `VICTORY_WAVE_WIDE` | `emit/mod.rs` | wave64 retry: off; lane-mask retry: on | `=1` wave-wide branches in the wave64 retry (put barriers in branches; lost the card in runs 121-124); `=0` for the lane-mask retry |
| `VICTORY_WAVE_DEEP_SKIPS` | `emit/mod.rs` | on | `0` |
| `VICTORY_WAVE_OWN_BITS` | `lower.rs` | on | `0` (a wave OR where not every lane runs refused) |
| `VICTORY_FIRST_HALF` | `lower.rs` | on | `0` (first halves of interpolations never paired) |
| `VICTORY_FIRSTLANE_OWN` | `lower.rs` | off | `=1`: a first-lane read takes the host wave's own first lane |
| `VICTORY_F16_NEAREST` | `lower.rs` | off: the host's `f32tof16` | round-to-nearest emulation of 16-bit float results and its test |
| `VICTORY_STRICT_JOINS` | `lower.rs` | off | joins read through other joins marked as read |
| `VICTORY_MESH_OUTPUT_STRICT` | `lower.rs` | off | mesh shaders whose outputs pass 32 KiB refused |
| `VICTORY_DXIL` | `emit/validator.rs` | the newest Windows Kits `dxil.dll` | an explicit validator path (and the Linux `libdxil.so` loader with the unused `victory_native_d3d12` cfg) |
| `VICTORY_WARP` | `emit/host.rs` (tests, `victory translate`) | the default adapter | WARP by environment; `Host::new(warp)` keeps the argument |
| `VICTORY_D3D12_DEBUG` | `emit/host.rs` | off | the D3D12 debug layer in the test host |
| `VICTORY_REF_PS`, `VICTORY_REF_VS` | `emit/mod.rs` test | the built-in shaders | DXIL files read in place of them |
| `VICTORY_CONSOLE_DXC`, `VICTORY_PC_DXC` | `emit/diff_tests.rs` | unset: the tests skip | the differential tests (console compiler vs PC compiler on the GPU), which never ran without them; the file is gone |
| `VICTORY_DIFF_ONLY`, `VICTORY_EMIT_DUMP` | `emit/diff_tests.rs` | every case, no dumps | with the differential tests |

## Not in this crate

`VICTORY_VIEW_*`, `VICTORY_ROOT_*`, `VICTORY_SHADER_SAVE` and
`VICTORY_SHADER_CACHE_KEY` are read by `victory-gpu` (and `apps/victory`),
not here. No `VICTORY_SHADER_PATCH` or `VICTORY_SHADER_SWAP` exists in the
tree any more.
