# Shader translation pipeline (`crates/victory-shader`)

A console shader is a DXBC container whose `DXIL` part also carries the
precompiled RDNA2 (GFX10) program. We translate the GFX10 code, not the
bitcode (it uses console-only intrinsics). Entry point:
`emit::translate(container) -> HostShader` (DXIL container plus what the
host binds). `victory-gpu` calls it at pipeline creation and caches the
result under `emit::translation_fingerprint()`.

## Stages

| Stage | File | Output |
|---|---|---|
| Container | `dxbc.rs`, `dxil.rs` | DXBC parts; the `DXIL` part's header and Xbox `XDAT` blobs |
| Native program | `native.rs` | `NativeProgram`: stage, wave32/64, code, root layout, used resources, stage I/O tables |
| Decoder | `gfx10/` (`mod.rs` boundaries, `opcodes.rs`, `fields.rs` operands, `sp3.rs` text) | `Instruction`s, found by following control flow |
| IR | `ir.rs` (+ `spill.rs`) | `Program`: SSA, one `Inst` per machine instruction, every register dword a value, EXEC/SCC/M0 explicit. `spill.rs` turns a spilled user-data table into entry SGPRs. |
| Structuring | `structure.rs` | `Structured`: the program as one lane runs it; lane masks become per-lane booleans, wave-wide branches become per-lane ones, unprovable EXEC stays as per-instruction guards; reducible CFG with loops and exits |
| Resources | `resource.rs` | where every descriptor comes from (root table, root descriptor, heap index computed by the code), traced through SSA |
| I/O | `io.rs` | inputs, interpolation (`v_interp_*`) and exports as signature elements |
| Lowering | `emit/lower.rs` (11k lines, one machine instruction at a time) | a DXIL module (`emit/shader.rs` builds it: dx.op calls, signatures, metadata) |
| Clean-up | `emit/bitcode/opt.rs` | exact passes (dead code, folding, reuse, branch facts, loads); always on |
| Bitcode | `emit/bitcode.rs` | LLVM 3.7 bitcode, unabbreviated records |
| Container | `emit/container.rs` | SFI0/ISG1/OSG1/PSV0/DXIL parts and the D3D12 hash |

`emit/mod.rs` drives the stages and the retries (below). Helpers:
`emit/wave.rs` (whole-wave sections), `emit/own_bits.rs` (scalarised
loops), `emit/threading.rs` (jump threading retry), `emit/intersect.rs` +
`bvh.rs` (ray/node test and its CPU reference), `emit/sampler.rs`
(immediate samplers to static samplers), `emit/rect.rs` (the geometry
shader for rectangle lists). `emit/host.rs` and `emit/validator.rs` are
Windows-only test tools (D3D12 device, `dxil.dll`).

## Where the console semantics live

- **Scalar vs vector registers.** SGPRs are one value per wave, VGPRs one
  per lane; both are plain `i32` values in the IR. EXEC is an explicit
  operand of every vector instruction (`ir.rs`). `structure.rs` removes
  EXEC: lane masks become booleans, and a vector write a lane may not run
  keeps a guard (lowered as a select against the old value).
- **Waves of 64 on 32-lane GPUs.** A wave64 compute program whose code
  reads other lanes (ballots, `v_readlane`, `v_mbcnt`, DPP, LDS between
  halves) runs as two host waves of 32 that exchange through two LDS dwords
  and group barriers (`lower.rs`: `exchange`, `halves`, `split_lds`,
  `wave_base`). Wave reductions and prefix sums are recognised by
  simulating the whole-wave section (`wave.rs`) and become DXIL wave ops.
  When that cannot be made exact, `emit/mod.rs` retries other forms of the
  program (early-outs left out, skips left out, each host wave as its own
  wave, approximate per-half forms; `threading.rs` last). A program that
  still needs 64 lanes declares `WaveSize(64)`. `set_host_max_wave` gives the
  host's widest wave (32 on NVIDIA).
- **Descriptors read as data.** Console code loads descriptors from heap
  memory with scalar loads and may read their dwords as numbers (a buffer
  address used as a pointer, a stride). `resource.rs` traces them; the
  lowering takes such dwords from user data the host fills
  (`USER_VIEW_ADDRESS`, `USER_VIEW_BIAS`), the per-slot stride table
  (`SLOT_STRIDES`) or title memory by directory (`MEMORY_RUN_POOL`: the
  directory, the pool table and the copies in a run of view-heap slots);
  see the constants in `emit/shader.rs` and the module
  notes of `emit/mod.rs` for the binding contract.
- **Loops** get an iteration limit (128 pixel, 256 vertex/mesh, 4,096
  compute; `Lower::count_iteration`) so data the host lacks cannot hang
  the GPU.

## Switches

Only `VICTORY_BVH_MISS` remains (`docs/switches-shader.md`). `build.rs`
hashes `src/` and lists the `VICTORY_*` names quoted there; both feed the
cache key, so adding a file under `src/` changes the key automatically.
