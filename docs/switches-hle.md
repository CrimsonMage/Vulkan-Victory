# Environment switches outside the GPU and shader crates

Every `VICTORY_*` variable read by victory-hle, victory-loader,
victory-formats, victory-core, victory-pe, victory-game, victory-pipeline,
apps/victory and xtask, as of the simplification pass (2026-11). The
"default" column is what happens with the variable unset; that behaviour is
what remains after a switch is removed.

38 switches were read before the pass; 7 remain.

## Kept

| Switch | Where | What it does |
|---|---|---|
| `VICTORY_LOG` | core `logging.rs` | tracing filter (falls back to `RUST_LOG`); debug and trace exist only in `diagnostics` builds |
| `VICTORY_CONFIG` | core `config.rs` | config file to read (default `./victory.toml`) |
| `VICTORY_GAME_DIR` | core `config.rs` | the title dump (else `[paths].game_dir`) |
| `VICTORY_CACHE_DIR` | core `config.rs` | derived data (else `[paths].cache_dir`, else `%LOCALAPPDATA%\victory\cache`) |
| `VICTORY_DATA_DIR` | core `config.rs` | saves and title storage (else `[paths].data_dir`, else `%LOCALAPPDATA%\victory\data`) |
| `VICTORY_KEYS` | hle `gameinput.rs` (`diagnostics` builds) | scripted key presses `<seconds>:<key>,...`, used by the test harness |
| `VICTORY_RELAUNCHED_FROM` | hle `lifecycle.rs` | not a user switch: the inherited handle a relaunched `victory` waits on when the title relaunches itself (`XLaunchNewGame`); part of the default path |

The `--console series-x|series-s` option of `victory run` is unchanged.

## Converted to command-line options

| Switch | Where | Default | Now |
|---|---|---|---|
| `VICTORY_HOST_WAVE` | apps `main.rs` (`translate`) | unset: translator's own default | `translate --host-wave <n>` |
| `VICTORY_HOST_MAX_WAVE` | apps `translate.rs` (`translate`) | unset; duplicate of the above | `translate --host-wave <n>` |
| `VICTORY_VIEW_LIMIT` | apps `main.rs` (`translate`) | unset: no view-heap cap | `translate --view-limit <n>` |

## Removed (default behaviour kept)

| Switch | Where | Default (kept) | Removed path |
|---|---|---|---|
| `VICTORY_EFFICIENCY_MODE` | hle `runtime.rs` | process opts out of execution-speed throttling | `=1` left Windows its efficiency mode |
| `VICTORY_WALLS_TRACE` | hle `dstorage.rs` | off | `1`/`2`: a log line per failed / every read request, with a table of opened files |
| `VICTORY_DISK_CORES` | hle `dstorage.rs` | workers run on any core | `=slow` pinned DirectStorage workers to slow cores |
| `VICTORY_DSTORAGE_RETRIES` | hle `dstorage.rs` | 12 retries of a failed page-destination write | other counts (0 = no retry) |
| `VICTORY_XMA_DUMP` | hle `audio.rs` (`diagnostics`) | off | saved the first 32 XMA voices' context and input |
| `VICTORY_AUDIO_CAPTURE` | hle `audio/capture.rs` (`diagnostics`) | off | WASAPI process-loopback recorder in a child copy of victory.exe (needed the `windows`/`windows-core` crates in hle and a self-relaunch); record from the host instead (PipeWire `pw-record`, OBS) |
| `VICTORY_AUDIO_CAPTURE_OF` | hle `audio/capture.rs` | internal to the above | |
| `VICTORY_SAVE_STOP`, `VICTORY_SAVE_CHILD`, `VICTORY_SAVE_OP` | hle `xgamesave.rs` (tests) | n/a | kill-at-every-step child-process save test and its hooks in the commit path |
| `VICTORY_STACKS` | loader `stacks.rs` (`diagnostics`) | off | scheduled dumps of every thread's stack |
| `VICTORY_CLZERO_ASK` | loader `console_insn.rs` | memory answers remembered per thread | `=1` asked `VirtualQuery` at every emulated instruction |
| `VICTORY_EXIT_CHILD` | loader `threads.rs` (test) | n/a | child-process test of `end_other_threads` |
| `VICTORY_TRAP_LOG` | loader `thunks.rs` (test) | n/a | child-process test of the trap crash log |
| `VICTORY_XMA_STREAMS` | formats `xma.rs` (ignored test) | n/a | decode saved streams from a directory |
| `VICTORY_XMA_REFERENCE` | formats `xma.rs` (ignored test) | n/a | compare with reference decodes |
| `VICTORY_BCPACK_TABLES`, `VICTORY_BCPACK_CASES` | formats `bcpack.rs` (ignored test) | n/a | compare with reference chunks |
| `VICTORY_TEST_TITLES` | core `config.rs` | `[paths].test_titles` from the config file | env override of that list (the config key stays) |
| `VICTORY_CRASH_SAVE` | apps `run.rs` (`diagnostics`) | off | saved title memory ranges on a crash |
| `VICTORY_WATCH` | apps `run.rs` (`diagnostics`) | off | polled and logged title u32 values |
| `VICTORY_WARM_SECONDS` | apps `translate.rs` | no time limit for `--warm` | stop after n seconds |
| `VICTORY_PAIRS` | apps `translate.rs` | off | build named vs+ps pipelines after the walk |
| `VICTORY_TRANSLATE_ONLY` | apps `translate.rs` | every container | only containers with listed hash prefixes |
| `VICTORY_DUMP_MATCH` | apps `translate.rs` | off | wrote matching containers to a directory |
| `VICTORY_FAIL_DUMP` | apps `translate.rs` | off | wrote the first containers of each failure reason |
| `VICTORY_SHADER_CENSUS` | apps `translate.rs` | off | per-program instruction census file |
| `VICTORY_EMIT_DUMP` | apps `translate.rs` | off | wrote emitted / failing DXIL to a directory |

Mentioned in these crates but read elsewhere (not counted): `VICTORY_DXIL`
(victory-shader validator path) and `VICTORY_PAGE_WRITES` (victory-gpu).
