# Opcode audit: our lift against origin/main's lowering

Every GFX10 opcode the title's programs use (the 390 of `docs/translation-index.md`, with its
program counts), our lift (`crates/victory-shader/src/vk/lift*.rs`, `emit*.rs`) read against
origin/main's lowering (`/mnt/data/victory/scratch-agent/forks/origin-main`, fxceless/victory-fork
`6742906`: `emit/lower.rs`, `emit/wave.rs`, `emit/sampler.rs`, `resource.rs`, `io.rs`; the decoder
`gfx10/` and `ir.rs` effects are the same files on both sides). Made 2026-11-26/27 by reading (four
read-only passes by family, the differences then checked and tested by hand); nothing was run on a
card. "theirs" = origin/main, "ours" = this tree **before** the fix named in the last column.

Verdicts: SAME; DIFFERENT (what); FIXED (was DIFFERENT, now theirs; commit subject and oracle test);
OPEN (DIFFERENT, not changed, why); theirs missing / ours missing. Tests are in
`vk/oracle/` (`origin1.rs`, `sections.rs`, `intersect.rs`, `valu1.rs`, `valu2.rs`; `nvidia.rs`:
cases of trustno1-coder/VIctory-NVIDIA `0f0ada1`, a Direct3D 12 fork of the same upstream).

## Summary

| | opcodes |
|---|---:|
| Used by the pack | 390 |
| Compared and SAME (values a program observes) | about 300 |
| DIFFERENT when audited | about 70 (most in three cross-cutting rules: bounds, pixel kill, waves) |
| … FIXED here (17 commits) | the multi-dword loads and stores (10), `exp` (pixel kill), position w, `v_cvt_u32/i32_f32`, `image_sample_c*` with immediate samplers, wave sections (DPP forms, `v_permlanex16_b32`, `v_readlane_b32`), `v_readfirstlane_b32`, `v_mad_legacy_f32`, `image_bvh*_intersect_ray` (builder off), unwritten registers |
| … OPEN | see "Open differences" (quad reads and `s_wqm`, gathers at a named mip, typed formats, zero signs, unfused mad, …) |
| Not compared in depth | `s_sendmsg`; the card's transport of the halves' exchange (`emit_xchg.rs`: barriers, counters). The exchanges themselves, `s_barrier` across waves and the far jump (`s_setpc_b64`) run in `vk/oracle/wave64.rs` (origin's `emit/wave64_tests.rs`, `wave64_after_loop.rs`) as one wave and as two halves; waterfalls in `vk/oracle/waterfall.rs` (origin's `emit/waterfall_tests.rs`) |

## Differences found, by effect

| opcode(s) | programs | verdict | what | test |
|---|---:|---|---|---|
| `exp` (pixel `done`) | 23,012 | **FIXED** | Theirs: `discard(!guard)` at the `done` export and at an end reached without one. Ours: no kill at all (0 of 1,599 used fragment modules demoted): the end's kill was gated on `s_endpgm` reading EXEC, which it did not. Now lanes outside EXEC at a `done` export, and at an end block without one, are demoted. | `a_pixel_outside_exec_at_the_last_export_is_killed`, `a_pixel_that_ends_outside_exec_without_an_export_is_killed` |
| `s_buffer_load_dwordx2/x4/x8/x16`, `buffer_load_dwordx2/x3/x4`, `buffer_store_dwordx2/x3/x4` | 20,207 / 21,499 / 14,198 / 14,389; 12,974 / 5,281 / 11,720; 117 / 301 / 226 | **FIXED** | The console bounds each dword (theirs: `raw_dwords`); ours bounded the whole access and read all zeros when the last dword passed the V#'s end. | `a_scalar_load_past_a_views_end_reads_the_dwords_inside`, `a_wide_…`, `a_vector_load_…`, `a_vector_store_…` |
| pixel input position w | 107 of 414 used pixel programs | **FIXED** | The console's w is w (D3D `SV_Position.w`); ours gave `FragCoord.w` = 1/w. | `a_pixels_position_w_is_w_not_its_reciprocal` |
| `image_sample_c_lz` (and other `_c`) with an immediate sampler | 3,685 (311 used programs list such a sampler) | **FIXED** | Theirs marks the static sampler a comparison one; ours: the host built immediates with compare off, the sample returned the stored depth. Now listed with `sampler::COMPARES`. | `an_immediate_sampler_of_a_comparison_sample_compares` |
| DPP forms of `v_or_b32`, `v_min_u32`, `v_max_u32`, `v_add_*`…, `v_permlanex16_b32`, `v_readlane_b32` in whole-wave sections | 2,285 / 3,042 | **FIXED** (not in a wave64 whose halves exchange: exact there) | Theirs: `wave.rs` + `shape` → wave reductions over active lanes. Ours: lane by lane through shuffles, wrong when the subgroup lacks lanes. Now subgroup reductions and scans. | `sections.rs` (2 tests, 252 cases each) |
| `v_cvt_u32_f32`, `v_cvt_i32_f32`, (`v_cvt_flr/rpi_i32_f32`), cube face | 17,284 / 7,162 | **FIXED** | Console and D3D saturate, NaN → 0; ours bare `OpConvertFToU/S` (undefined). The interpreter now poisons an unclamped conversion. | `float_to_integer_conversions_saturate_and_take_nan_to_zero` |
| `v_readfirstlane_b32` | 16,392 | **FIXED** | EXEC empty: ours read lane 31; console lane 0, theirs the first lane there. Pixel helper lanes take their own value (theirs `helper_reads_own`). | `a_first_lane_read_with_no_lane_in_exec_reads_the_first_lane` |
| `v_mad_legacy_f32` | 6,040 | **FIXED** | RDNA 2: fused, sources flushed, zero-factor rule; ours legacy product then add. | `legacy_mad_fuses_its_product_and_flushes_its_sources` (theirs) |
| `image_bvh64_intersect_ray`, `image_bvh_intersect_ray` | 24 / 40 | **FIXED** | Builder off: theirs answers no hit and reads nothing; ours read the node. | `intersect.rs` |
| every register read before a write | all | **FIXED** | Theirs 0; ours `OpUndef`. | `a_register_nothing_wrote_reads_zero` |
| `DB_SHADER_CONTROL` DEPTH_BEFORE_SHADER (bit 12 of a pixel program's control word) | 3,009 (223 of 414 used pixel programs, all 130 `TL_PS_*`) | **FIXED** | The console tests depth and stencil (and writes them) before the program. Ours parsed the word and never used it: the host tested after the program (every pixel module demotes, most can store), so hidden pixels were shaded and their stores landed. Now `EarlyFragmentTests`, unless the program gives the tests their depth or stencil (none of the 3,009 does). Theirs (trustno1-coder/VIctory-NVIDIA 41b4a05): the same rule behind a switch. | `the_consoles_early_depth_test_is_declared` (`oracle/nvidia.rs`) |
| V# with NUM_RECORDS −1 (root views; all MUBUF/MTBUF/`s_buffer_load_*`) | 165 of 313 used vertex/mesh programs | OPEN, fix on a side branch | Theirs: a root view has no bounds. Ours: zero block unless the map range has 4 GiB − 1 left. Branch `…-vsharp-range`: limit = min(V# bytes, bytes left in the range). | `a_view_of_all_memory_reads_to_the_end_of_its_range` (on that branch) |
| `ds_swizzle_b32`, DPP `quad_perm`, `s_wqm_b32/b64` | 6,270; 1,574 / 8,717 | OPEN | Theirs: quad operations (see helper lanes); `s_wqm` = every lane. Ours: shuffles and a ballot by `SubgroupLocalInvocationId & !3`, an inactive source reads 0. Equal while helpers are present and counted; undefined where the host leaves helper invocations out of shuffles/ballots. Needs `OpGroupNonUniformQuadSwap/Broadcast` (and a quad-aware `s_wqm`). | none |
| `image_gather4_l`, `image_gather4_l_o` | 16 / 16 | **FIXED** (2D) | Theirs: four texel loads at the rounded level, offsets in that level's texels. Ours dropped the level: `OpImageGather` read the base level (meshlet culling against the depth pyramid). Now four loads at the level. | `a_gather_at_a_named_level_reads_four_texels_of_that_level` |
| `image_sample_b` outside pixel programs | 906 (non-pixel share unknown) | OPEN | Theirs: the bias is the level; ours level 0. Both are stand-ins (the console takes lane derivatives). | none |
| `buffer_load_format_x/xy/xyz/xyzw` through the V#'s FORMAT | 10,199 / 1,663 / 1,935 / 534 | OPEN (edges) | (a) FORMAT codes we do not decode read one 32-bit float: 10_11_11 (30-36), 11_11_10 (37-43), 10_10_10_2 (44-49), 2_10_10_10 other than 50, all scaled kinds; the host writes code 36 for R11G11B10_FLOAT buffer views. (b) DST_SEL is not applied (BGRA views swap x and z). (c) the last element(s) of a view of 1- or 2-byte elements read 0 (bounds on max(stride, 4) bytes). | none |
| `buffer_load_format_d16_x/xyzw`, `buffer_store_format_d16_xyzw` | 38 / 37 / 65 | OPEN (edge) | A 16-bit float FORMAT goes through f32 and back: a signalling-NaN pattern (an integer kept in such a buffer) may be quieted. Theirs moves the halves for flagged views. | `valu1.rs` d16 tests (pass for the common cases) |
| `v_max_f32`, `v_min_f32`, `v_max3_f32`, `v_min3_f32`, `v_max/min_f16`, `v_pk_max/min_f16` | 18,008 / 9,071 / 8,364 / 3,434 | OPEN (known) | Theirs orders −0 below +0; ours leaves the zero's sign to the host's NMin/NMax. | 2 tests ignored in `valu1.rs` |
| `v_mul_legacy_f32` with a finite non-zero constant factor | 23,682 | OPEN (edge) | Ours returns the plain product: const × ±0 can be −0, the console's is +0. | none |
| `v_mad_f32`, `v_mac_f32`, `v_madak_f32` | 7,259 / 5,415 / 3,661 | OPEN (last place) | RDNA 2 and theirs: multiply then add; ours GLSL Fma (the host may fuse). | none |
| `v_sin_f32`, `v_cos_f32` | 11,833 / 12,097 | OPEN (≈2e-7) | Reduction by fract (ours) or x − rndne(x) (theirs). | `valu` tests pass |
| f16 arithmetic (`v_add/sub/mul_f16`, `v_fma_f16`, `v_fmac_f16`, `v_rcp/rsq/sqrt/exp/sin/cos/trunc_f16`, `v_pk_*_f16`, narrowing of `v_cvt_f16_f32`, `v_fma_mixlo/hi_f16`) | 41 … 5,971 | DIFFERENT, ours nearer | Theirs computes in f32 and narrows with the host's conversion (truncating on NVIDIA); ours native f16 (round to nearest, as the console). | `valu` f16 tests |
| `image_load`, `image_load_mip` with d16; on a STORAGE view with a mip | 9,049 / 257 | OPEN (edge) | d16: ours truncates the 32-bit pattern, theirs converts every channel to a half; the console converts by format. `OpImageRead` has no level. | none |
| `image_sample_d`, `image_sample_d_cl` on cubes | 2,880 / 29 | both approximate | Ours: the faces as a 2D array with the gradients (no filtering across edges); theirs: by direction at a level from the longer gradient. | none |
| `image_get_lod` outside pixel, on cubes | 645 | corners | Ours (0, 0) / wrong direction; theirs refuses. | none |
| unnormalized coordinates (`unrm` on samples) | not counted | OPEN | Theirs divides by the level-0 size; ours ignores the field. | none |
| an S# chosen between two immediate samplers by a branch | not counted | OPEN | Theirs samples with both and selects; ours takes it for a heap sampler (slot from word 3). | none |
| `v_writelane_b32` left after unspilling, outside compute | 1,171 (≈10 used pixel programs) | OPEN (edge) | Ours writes only if that lane exists; theirs refuses. | none |
| `v_interp_mov_f32` of a 16-bit row | 3,652 (none seen) | OPEN (latent) | Theirs returns the dword of two halves; ours one f32. | none |
| branches (`s_cbranch_*`) | 17,315 … 20,282 | DIFFERENT by design | Ours branches the subgroup as the console's wave; theirs each lane on its own bit. | — |
| `s_min/max_*` SCC, DPP `row_ror` (n ≠ 8), source op_sel of `v_lshrrev_b16`/`v_add_nc_u16`/u16 compares, `v_mov_b32` with abs/neg, SDWA sign-extended destinations, f64 literal operands, LDS out of range | — | latent | No instance in the used set. | none |
| vertex id | — | ours right | Ours includes the base vertex (as the hardware); theirs `SV_VertexID`. | — |
| `exp param` per primitive (mesh) | — | ours missing (refused) | | — |

## Compared and SAME

- **Images**: `image_sample` (9,610), `image_sample_lz` (10,201), `image_sample_l` (5,273),
  `image_gather4_lz` (4,072; component from dmask, four registers, host texel order),
  `image_gather4_lz_o` (6), `image_sample_o` (4; 1D/2D), `image_store` (2,443; dmask 0xF),
  `image_get_resinfo` (407; sampled views), `image_atomic_add/or/umax/umin/cmpswap`,
  `v_cubeid/cubema/cubesc/cubetc_f32`. Address order (offset, bias, reference, dx, dy, coordinates,
  level, clamp), dmask packing, cube face table: the same.
- **Float**: `v_add/sub/subrev/mul_f32`, `v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_med3_f32/f16`,
  `v_rcp_f32`, `v_rcp_iflag_f32`, `v_rsq_f32`, `v_sqrt_f32`, `v_exp_f32`, `v_log_f32` (base 2),
  `v_floor/ceil/trunc/rndne_f32`, `v_fract_f32`, `v_cvt_f32_i32/u32`, `v_cvt_f32_ubyte0-3`,
  `v_cvt_pk_u8_f32`, `v_cvt_pkrtz_f16_f32`, `v_cvt_f32_f16`, `v_pack_b32_f16`, `v_fma_mix_f32`,
  every `v_cmp_*_f32/f16` and `v_cmpx_*` (ordered and the `n` forms; NaN answers the same, on the
  host's honouring of unordered compares), `v_cmp_class_f32`, `v_cndmask_b32`, `s_setreg*` (both
  ignore), omod/clamp/abs/neg.
- **Integer, scalar, LDS**: every integer and bit VOP (adds, shifts with masked counts, `v_bfe/bfi/bfm`,
  bit scans with −1 for none, 24-bit multiplies, `v_perm_b32`, carries), every integer compare and
  `v_cmpx`, every `s_*` arithmetic, logic, shift, bit-field, compare and select, the `saveexec` and
  `wrexec` forms, `v_mbcnt_lo/hi`, `ds_bpermute_b32`, every `ds_*` read, write and atomic in range.
- **Memory and I/O**: `buffer_load_dword`, `buffer_store_dword`, the buffer atomics, `tbuffer_load_format_*`
  (every format of the used set), `global_load_*`, `s_load_*`, `exp mrt/mrtz/pos/param/prim` values,
  `v_interp_p1/p2_f32`, `v_interp_p1ll/p2_f16`, barycentrics, front face, instance id, compute ids.
