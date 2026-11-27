# victory-gpu switches

Every `VICTORY_*` environment switch victory-gpu read at `integrate`
(0b0074e + local fixes), its default as the code had it (not as the comments
said), and what became of it. "Removed" means the switch and every
non-default path are gone and the default behaviour is the only one.

Note: several comments said "off by default" for things the code had on.
The relay (`VICTORY_RELAY`) was ON by default (stage 2), and with it
`VICTORY_LATE_WAIT` stage 3. That is the path that was kept.

## Read now (2026-11-26, Vulkan only)

The D3D12 back end and every switch it read are gone (`docs/kiss.md`,
decisions log). What victory-gpu still reads:

| Switch | Default | What it does |
|---|---|---|
| `VICTORY_CAPTURE` | unset | folder for frame PNGs (diagnostics builds) |
| `VICTORY_CAPTURE_FRAMES` | unset | extra frames to save |
| `VICTORY_TEXTURE_QUALITY` | `medium` under 10 GB of card memory, else `high` | `high`, `medium`, `low`: card images of textures the title fills leave out 0, 1 or 2 of their largest levels (`quality.rs`) |
| `VICTORY_LIST_WORKERS` | fast physical cores, 2-16 | workers that record the list calls the title's threads write down (`list/thin.rs`); `0` records each call at once on the title's thread |
| `VICTORY_DISPLAY_MODE` | the PC's screen | display mode the title is told (`dxgi.rs`) |
| `VICTORY_VK_DUMP`, `VICTORY_VK_STAGES`, `VICTORY_VK_DRAW_TEST`, `VICTORY_VK_COMPOSE_TEST`, `VICTORY_DIAG_NO_INDIRECT` | off | Vulkan backend diagnostics (`docs/vulkan-host.md`) |
| `VICTORY_VK_TRACE_DRAWS`, `VICTORY_VK_CEILING_MB`, `VICTORY_VK_ROOM_MB` | off | diagnostics builds: every draw of the programs named (parts of names, comma separated) in the first dumped frame as a log line; the card-memory rule's ceiling and room in MiB (`vulkan/budget.rs`), to see a run with no image refused and no pool page evicted |

## Removed (before the D3D12 back end went)

| Switch | Default | What it did | Decision |
|---|---|---|---|
| `VICTORY_LEVELS_ON_DEMAND` | off | host copies of textures the title fills take card memory level by level (reserved textures, `UpdateTileMappings`) | the only path on a card with tiled resources tier 2; combined with texture quality |
| `VICTORY_LEVELS_MIN_KB` | 1024 | size floor for the above | constant 1 MiB |
| `VICTORY_RELAY` | on, stage 2 | flights landed by a thread of their own; exact waits | relay is the only path |
| `VICTORY_LATE_WAIT` | 3 (with relay) | submissions not waited for; present as a flight | stage 3 is the only path |
| `VICTORY_LATE_WAIT_LISTS` | 256 | lists in the air before waiting | constant 256 |
| `VICTORY_LATE_WAIT_WRITES` | on | bottom-of-pipe writes may fly | kept |
| `VICTORY_LATE_WAITS` | off | frame waits sleep to the next tick | removed |
| `VICTORY_QUEUE2`, `_LANDING_MB`, `_REFETCH` | off | copy queues for fetches/uploads | queue2.rs deleted |
| `VICTORY_CARD_FETCH_INFLIGHT`, `_MB` | off | fetch flagged pieces inside flights | removed |
| `VICTORY_CARD_REHASH_RUNS` | off | join hash passes | removed |
| `VICTORY_WIDE_MERGE` | off | merge card pieces on threads | removed |
| `VICTORY_WIDE_REFRESH` | off | refreshes on threads | removed |
| `VICTORY_WIDE_BACK` | on | snapshots/compares on threads | kept |
| `VICTORY_REFRESH_ONCE` | off | skip re-refresh in a submission | removed |
| `VICTORY_JOIN_COPIES` | off | join adjacent title copies | removed |
| `VICTORY_MERGE_RUNS` | 512 | cap of changed runs | constant |
| `VICTORY_MERGE_GAP` | 0 | join runs over gaps | removed |
| `VICTORY_MERGE_RACE` | off | merge race logging | removed |
| `VICTORY_FILL_WAIT` | off (fills read ahead) | fills wait for flights | removed, reading ahead kept |
| `VICTORY_KEEP_SNAPSHOTS`, `_MB` | off | snapshots kept across submissions | removed |
| `VICTORY_CARD_BUFFERS` | on | buffer copies in card memory | kept |
| `VICTORY_CARD_BUFFERS_MB`, `_MIN_MB` | budget/8 (256..1024), 1 | limits | constants |
| `VICTORY_CARD_ROOM_FOR_BUFFERS` | on | texture queue keeps room for them | kept |
| `VICTORY_CARD_WINDOWS_MB` | 0 | windows in card memory | removed |
| `VICTORY_WINDOW_KB` | 4096 | window grid | constant 4 MiB |
| `VICTORY_WINDOW_TAILS` | off | tail windows | removed (find_any = find) |
| `VICTORY_REFRESH_CHECK` | off | stale-copy check | removed |
| `VICTORY_INPLACE`, `_MB`, `_PROBE`, `_PROBE_MB` | off | title memory as host heaps | inplace.rs deleted (page views moved to views.rs) |
| `VICTORY_BACK_OWN_VIEW` | off | copy back through own view | removed |
| `VICTORY_COPY_PLAIN` | off | in-place guarded copies | plaincopy.rs deleted |
| `VICTORY_COPY_ASK` | off | VirtualQuery before copies | removed |
| `VICTORY_DIRECT_WRITES` | on | NtWriteVirtualMemory | kept |
| `VICTORY_COPY_MODE` | guarded | copystat copy modes | removed with copystat.rs |
| `VICTORY_BARRIER_TEST` | off | barrier experiments | removed |
| `VICTORY_MEMORY_MAP` | on (diag) | memory map line | removed with copystat.rs |
| `VICTORY_CARD_METER` | on in diag | GPU list timing | cardmeter.rs deleted |
| `VICTORY_FRAME_LOG` | on in diag | per-frame log | framelog.rs deleted |
| `VICTORY_FAST_CORES` | on | submit/relay threads on fast cores | kept (cores.rs) |
| `VICTORY_PROFILE*` (15 switches) | off | sampling profiler | profile.rs deleted |
| `VICTORY_GPU_TIMING` | off | per-list timing | removed |
| `VICTORY_D3D12_GBV` | off | GPU-based validation | removed |
| `VICTORY_D3D12_DRED` | off | device-removed data, object names | removed |
| `VICTORY_D3D12_WARP` | off | software adapter | removed (tests keep `warp_adapter`) |
| `VICTORY_D3D12_BACKGROUND` | left on | background shader work | removed |
| `VICTORY_SWEEP` | on | idle sweep | kept |
| `VICTORY_TRAIL` | on | removal trail | kept |
| `VICTORY_LAZY_PRELUDE` | on | preludes on demand | kept |
| `VICTORY_INDIRECT_CARD_BUDGET` | on | budget in card memory | kept |
| `VICTORY_SCRATCH_POOL` | on | shared scratch chunks | kept |
| `VICTORY_WALLS_TRACE` | off | walls trace logging | removed |
| `VICTORY_ASYNC_SUBMIT` | off | submit thread | removed |
| `VICTORY_ASYNC_PRESENT` | off | present behind queue | removed |
| `VICTORY_QUEUE_WAIT`, `VICTORY_SPLIT_WAITS`, `VICTORY_CUT_WAITS` | on | queue waits, cut at value waits | cut path is the only path |
| `VICTORY_WAIT_FORCE` | on | present forces held work | kept |
| `VICTORY_CUT_WRITES` | off | cut lists at value writes | removed |
| `VICTORY_WAIT_ASK` | off | VirtualQuery before value reads | removed |
| `VICTORY_LIST_POOL` | on | host list reuse | kept |
| `VICTORY_READ_BACK` | on | texture read-backs | kept |
| `VICTORY_INDIRECT_WINDOW`, `_WINDOW_CHECK` | off | indirect draw windows | removed (with `bound_ahead`, shader "ahead" bit) |
| `VICTORY_DRIVER_WATCH` | on | driver event-log watch | kept (log only) |
| `VICTORY_SCISSOR_TEST`, `VICTORY_MESH_TEST` | off | experiments | removed |
| `VICTORY_WRITE_SKIP` | on | declare writes only for storing shaders | kept |
| `VICTORY_EARLY_TEXTURES_MS`, `VICTORY_BAKE_TEXTURES_MS` | 12, 1500 | early texture pass budgets | constants |
| `VICTORY_FULL_CARD_PASS`, `VICTORY_REUPLOAD_WHEN_FULL`, `VICTORY_HEAP_RECHECK` | on | texture queue on a full card | kept |
| `VICTORY_HEAP_TEXTURES` | on | pass over textures picked by index | kept |
| `VICTORY_TEXTURE_STAGE_MB`, `_STAGE_MS`, `VICTORY_TEXTURE_CATCHUP` | 192, 250, on | pass budgets | constants |
| `VICTORY_CARD_ACCOUNTING` | off | accounting line | removed |
| `VICTORY_ROOT_VIEW_MB`, `VICTORY_ROOT_UAV_MB` | unlimited | root view limits | removed |
| `VICTORY_UNWATCHED_FRAMES` | 1 | refresh interval | every frame |
| `VICTORY_ROOT_SRV_DIRECT` | off (tables) | root views as descriptors | removed |
| `VICTORY_BVH_QUARANTINE`, `VICTORY_BVH_HOLD` | off | builder quarantine, hold walks | removed |
| `VICTORY_SKIP_PIPELINES` | none | skip pipelines (diag) | removed |
| `VICTORY_REFUSED_SAVE`, `VICTORY_SHADER_SAVE` | unset | dump shaders | removed |
| `VICTORY_PIPELINE_WAIT_MS`, `_WAIT_SHARED` | 500, on | pipeline wait budget | constant, shared |
| `VICTORY_LIBRARY_MAP`, `_HANDLE`, `_TRIM` | on | library mapping/handle/trim | kept |
| `VICTORY_LIBRARY_LOCK` | off | one library call at a time | removed |
| `VICTORY_MEMCAP` | on | ceilings and buffer pool | always on |
| `VICTORY_CARD_RESERVE_MB`, `VICTORY_BUFFER_POOL_MB` | 1024, 512 | memcap constants | constants |
| `VICTORY_HEAP_SYNC`, `_SYNC_PAGE` | on | mirror raw heap copies, by page | kept |
| `VICTORY_HEAP_SYNC_TABLES` | off | also mirror table slots | removed |
| `VICTORY_VIEW_INLINE` | on | views named whole in records | kept |
| `VICTORY_STALE_TEXTURES` | start rule | stale-view rule | start rule only |
| `VICTORY_TABLED_TEXTURES` | out of queue | tabled slots leave queue | kept |
| `VICTORY_VIEW_EXTENT`, `VICTORY_VIEW_OWN_COPY`, `VICTORY_KEEP_VIEWS` | on | buffer view copies, kept views | kept |
| `VICTORY_DEPTH_TWINS` | on, all | depth twins (`stencil`/`nodepth`/`0` variants) | full twins only |
| `VICTORY_ALIAS_COPY` | on, whole | alias copies | whole only |
| `VICTORY_LAZY_TEXTURES` | on | host textures on first use | kept |
| `VICTORY_TEXTURE_BUDGET_MB`, `VICTORY_TEXTURE_ROOM_PERCENT`, `VICTORY_CARD_ROOM_FOR_TARGETS_MB` | unset, 90, 3072/0.3 budget | texture room | constants |
| `VICTORY_TEXTURE_SWEEP` | on | idle texture sweep | kept |
| `VICTORY_PLACED_FILLER` | off | filler words in placed key | removed |
| `VICTORY_STAGING_POOL` | on | staging buffer pool | kept |
| `VICTORY_TEXTURE_NO_WAIT`, `VICTORY_TEXTURE_COPY` | off | refresh variants | removed |
| `VICTORY_TEXTURE_GUARD` | on | guarded view-memory textures | kept |
| `VICTORY_COMMITTED_QUERY`, `VICTORY_MAPPED_BY_PAGE` | off | VirtualQuery / per-page | removed |
| `VICTORY_CAPTURE_WORLD`, `_WORLD_ALSO`, `_WORLD_EVERY`, `VICTORY_DUMP_TARGETS` | off | capture extras | removed |

## Frame statistics

The diagnostics feature now covers only frame capture and the debug layer.
All per-frame counters are gone; one INFO line every five seconds says
`<fps> fps, <ms> ms/frame, <n> draws/frame; card memory <MB> MB in use of a
budget of <MB> MB; PC memory mapped for the card <MB> MB`, in every build.
