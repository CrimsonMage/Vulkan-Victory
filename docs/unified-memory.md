# Unified memory

The console has one memory. The title writes buffers and textures with the
CPU and the GPU reads the same bytes; the GPU writes and the CPU reads them
after a fence. VIctory today keeps title memory apart from host GPU memory
and has to find out what changed: write-watch queries, read-only guards on
every alias of a physical page, snapshots and compares of what shaders may
have written, tile sweeps, pools and statistics (`shadow.rs`, `watch.rs`,
`tiles.rs`, `inplace.rs`, `copystat.rs`, most of `host.rs`). In the run of
2026-11-25 07:38 (Series X profile) a frame checked 8.2 GB for writes,
refreshed 11.8 GB of copies, compared 0.38 GB on the CPU and copied back
64 MB: 1.4 frames a second, 126 ms preparing and 104 ms copying back per
frame.

This document replaces all of that with one rule:

> **Every byte of title memory has one home, and the GPU reads and writes it
> there.** Title buffers have no host copy. Title textures have a host copy
> only because the layouts differ, and it is refreshed at points the title
> names itself (DirectStorage completion, page mapping, its own GPU writes),
> never by watching memory.

The home is **system memory opened as host heaps**, not a GPU-upload heap.
Section 1 says why; the rest is the design, the deletion list and a staged
plan with a test per stage. The probe that checks the facts on the PC is in
`crates/victory-gpu/src/unified_probe.rs` (`VICTORY_UNIFIED=2`). Stages 1
to 3 are implemented and on by default; section 12 says what was built,
what the probe measured, and how to read the log. Sections 1 to 11 are the
design as written before the probe ran (its level numbers are the old
ones).

## 1. What the host allows

Facts from the sources and the driver, before any design.

**GPU-upload heaps (ReBAR) exist here, for buffers only.** vkd3d-proton
master supports `D3D12_HEAP_TYPE_GPU_UPLOAD` when the device has one
device-local heap (`vkd3d_memory_info_decide_hvv_usage`, `resource.c`),
which this card has; the heap is `DEVICE_LOCAL | HOST_VISIBLE |
HOST_COHERENT` memory (`memory.c` `vkd3d_select_memory_flags`; a custom heap
with write-combine in L1 is normalised to the same type). On this driver
(620.73) that memory type, number 4, is not usable for optimally tiled
images (`vulkaninfo`: "IMAGE_TILING_OPTIMAL: None"). **No texture the GPU
samples efficiently can live in a GPU-upload heap on this card.**

**A heap the host maps lies where the host puts it, once.** D3D12 has no call
that maps a heap at a chosen address or twice. Under Proton the mapping is
the driver's `mmap` on the Unix side: winevulkan uses
`VK_EXT_map_memory_placed` only for WoW64 processes (`win32u/vulkan.c`,
`zero_bits`), so a 64-bit process gets an address near `0x7f..`, far above
1 TiB, where the copystat memory map already finds every host allocation.
Title memory must lie below 1 TiB (40-bit texture descriptors, the
ray-tracing builder; `xmem` `address_range`, decision 0001), and a physical
page must show at several addresses (pool view, tiled heap). A GPU-upload
heap can do neither. The probe logs the address it gets.

**The other direction works: a heap opened on memory the title already
has.** vkd3d-proton implements `OpenExistingHeapFromAddress1` and
`OpenExistingHeapFromFileMapping` with `VK_EXT_external_memory_host`
(`device.c`). The heap is a custom heap in the PC's memory (write-back, L0),
shared and cross-adapter, `ALLOW_ONLY_BUFFERS`. For a section it maps a
view, imports it and unmaps it again; the driver pins the pages. Wine makes
anonymous sections with `memfd_create` (`server/mapping.c`), so the pinned
pages are the pages every view of the section shows: **the GPU sees the same
bytes as every alias, with no copy and no detection.** The driver supports
the import (`minImportedHostPointerAlignment` 4 KiB, `maxBufferSize`
1 TiB, no allocation size limit).

Three catches, all handled below:

- **Silent fallback.** If the import fails, vkd3d-proton allocates memory of
  its own and returns success (`vkd3d_import_host_memory`). That heap is
  not the title's memory. Every opened heap is checked once with a canary.
- **Pinning.** Opened memory is resident from the moment it is opened, used
  or not.
- **Bandwidth.** The card reaches the PC's memory over the bus. This card's
  link is PCIe Gen4 x8 (`nvidia-smi`: GPU max gen 5, host max gen 4, width 8),
  about 13 GB/s in practice. The card's own memory is about 30 times faster.
  The probe measures both.

**Capacity.** The card has 8 GB. vkd3d's budget with the desktop running is
6.78 GiB. Title memory in the Series X run: 5.7 GB of heaps
(`XMemVirtualAlloc`), a 6.2 GB physical-page pool, and 4.1 GB of views that
are aliases of pool pages, not more memory. The Series S profile tells the
title it has 7.8 GB (`XMemGetWorkingSetStatistics`, latest log). Title
memory does not fit on the card beside the render targets (3.3 GB in the
07:38 run) and the detiled textures, whichever way it is mapped. The PC has
31 GB.

So title memory stays in the PC's memory, which is how it has to be anyway,
and becomes the GPU's memory by being opened, not copied. The card's memory
holds what only the host GPU touches: render targets and detiled textures.

## 2. Memory classes

| Class | What | Where | CPU | GPU |
|---|---|---|---|---|
| **A. Title memory** | `XMemVirtualAlloc` regions, physical-page sections, D3D12.X heaps, committed resources (from the title's allocator, so inside its heaps) | PC memory, at the title's addresses below 1 TiB, opened as heaps (L0, write-back, coherent) | title, DirectStorage, VIctory reads | reads and writes in place through one buffer spanning each heap |
| **B. Host textures** | detiled copies of title textures; render targets, depth and UAV textures the host owns (as today) | card memory (`DEFAULT`) | never | yes |
| **C. Upload ring** | bytes VIctory writes and never reads: detiled texels on their way into class B, its own constants | GPU-upload heap (ReBAR), buffers only, a fixed 128–256 MB | write-combined writes only | copy source |
| **D. VIctory's own** | indirect scratch, descriptor heaps, read-backs, query heaps | as today | as today | as today |

Rule for CPU reads (question 3): **the title's CPU reads only class A**,
which is write-back and coherent, so read-backs, resolved query results,
buffers shaders wrote (culling output, exposure, counters) and DirectStorage
destinations read at full CPU speed after the title's fence. Class C is
write-combined: CPU reads from it are uncached, and the probe prints how
slow. Nothing the title can read lives there. VIctory's own read-backs stay
in `READBACK` heaps.

## 3. Heap layout and address translation

Each class A unit becomes one host heap and one spanning buffer:

| Title memory | Made by | Opened as |
|---|---|---|
| `XMemVirtualAlloc` region (address-constrained, no base) | `VirtualAlloc2` as today, **without** `MEM_WRITE_WATCH` | `OpenExistingHeapFromAddress1(base, committed extent)` at the first GPU use of an address in it |
| `XMemAllocatePhysicalPages` | a section, **always `SEC_COMMIT`** (it already is under Wine) | `OpenExistingHeapFromFileMapping(section)` at the first GPU use of any of its pages. If the 6.2 GB pool does not open whole (probe `VICTORY_UNIFIED_PROBE_BIG_MB=6400`), xmem backs one request with sections of at most 1 GiB (`Slice` already records a section per page) |
| D3D12.X `CreateHeap` | today VIctory's own allocator, at an address above 1 TiB (`resource.rs` `Memory::new`) | an `xmem`-style constrained `VirtualAlloc2`, then opened like a region |

The spanning buffer is placed at offset 0 of the heap over all of it, with
`ALLOW_UNORDERED_ACCESS | ALLOW_CROSS_ADAPTER`, in the common state. Every
use of a title GPU address names it plus an offset: buffer views, root
views, vertex and index views, indirect arguments and counts, copy sources
and destinations, query resolves, and the raw-address directory of
`memory.rs` (the BVH walkers). Its entries become heap slots instead of
copy numbers, and change only when heaps open or pages are remapped.
D3D12.X buffer resources get no host resource at all. Their GPU address is
the title address, translated. If a driver limit ever shows, one placed
buffer per title buffer at the matching heap offset is the fallback. D3D12.X
textures cannot be placed in the heap: the layout differs (section 6), and
neither L0 imported heaps nor GPU-upload heaps hold optimal textures here.

Translation is `unified_map::Map`: by title address, a span's length, heap
and heap offset; `resolve(at, len)` answers a home, `Scattered` or
`Unknown`. It holds hundreds of entries, not millions, and replaces
`shadow::find`/`find_any`/`exact`/`extent`, the seam that
`descriptor/host.rs` and `draw.rs` call at about ten sites each today.

Every heap is checked when it is opened: the CPU writes 16 bytes at a
committed address of it, the GPU copies them out, and they must match.
A mismatch is the silent fallback. The heap is dropped, the unit is counted
as refused and logged, and its addresses resolve `Unknown`. Until stage 4,
`Unknown` means the old copy path. After stage 4 it is an error line.

## 4. Aliases (question 1)

**On the CPU, nothing changes.** `xmem` keeps placeholders and
`MapViewOfFile3` views of sections. That is already the console's page
table, and a view can be mapped, unmapped and remapped under an opened heap,
because the heap is the section, not a view (native result in
`inplace.rs`'s notes, and the vkd3d implementation unmaps its own view
right after import).

**On the GPU, a contiguous run is a span.** `XMemMapPhysicalPages`,
`...Scatter` and `map_alias` (the queue's `CopyPageMappingsBatchX`) call
`Map::insert(address, len, section heap, page offset)` per run of
consecutive pages and `Map::remove` for unmaps. Two addresses showing one
page resolve to the same heap offset, the same bytes. There is nothing to
keep in step.

**Scattered pages.** After `CopyPageMappingsBatchX` the tiled heap is page
by page from anywhere in the pool. A range over it resolves `Scattered`.

- *Textures* there are detiled by the CPU through the title's own addresses
  (section 6), where the scattering is the kernel's business. Most of the
  tiled heap is textures.
- *Buffers and raw windows* there need a GPU page table. D3D12 has one: a
  **reserved buffer over the tiled heap's reservation, whose 64 KiB tiles
  are mapped (`UpdateTileMappings`) into the section heaps** at the moments
  xmem maps pages, on the host queue in submission order. This is the
  console's GPU page table in D3D12 terms. It depends on the driver taking
  tile mappings into an opened heap. `inplace.rs` records that the software
  adapter refuses them and removes the device. `VICTORY_UNIFIED=2` asks this
  driver.
- Without tiles, a scattered buffer range keeps today's copy (the shadow
  path, restricted to scattered view memory) and is counted. The in-place
  census (`VICTORY_INPLACE_PROBE`) already sorts title buffers into "one run
  of a section" and "scattered pages". If scattered buffers turn out to be
  rare, the fallback is a few hundred lines kept, not a design.

## 5. Synchronisation

The title already orders everything. Its CPU writes come before the
`ExecuteCommandLists` that reads them, and its CPU reads come after a fence
wait on work that wrote. Coherent memory needs nothing more:

| Point | Today | Unified |
|---|---|---|
| `ExecuteCommandLists` | bring changed pages of every copy up to date; snapshot ranges shaders may write | detile the stale textures the lists sample (section 6), then submit. No buffer work |
| after a submission | compare snapshots, write differences back, settle flights | nothing; mark textures over the submission's declared write ranges stale |
| `Signal` / fence wait / `PresentX` | settle and copy back before the CPU may look | host fence only |
| DirectStorage completion (`dstorage.rs` `into_destination`, already calls `title_memory_written` / `pages_written`) | guard bookkeeping | `Stale::written(range)`, and for page destinations at every address the pages show (`xmem::shown_at`) |
| page mapping (`XMemMapPhysicalPages`, `map_alias`) | remap notifications, guards reset | `Map::insert/remove` + `Stale::written`; with tiles, `UpdateTileMappings` |
| title cache flush or invalidate packets, barriers | partly a trigger for copies | a UAV or global barrier; no CPU work |
| `VirtualFree` of an opened region, `XMemFreePhysicalPages` of a whole section | — | retire the heap after the GPU is idle on it (`host::retire`). Natively the heap must go before the free (error 87, `inplace::title_memory_freeing`). The title was not seen freeing its heaps |

Host GPU caches: the memory is host-coherent. Vulkan makes host writes
before `vkQueueSubmit` visible, and the title's fence waits map to host
fence waits after the GPU's writes are made available. A write the title
makes while the GPU reads the same bytes is the title's own race, as on the
console.

## 6. Textures (question 2)

They keep a host copy. The console's tiled layouts are AMD swizzles; the
card's are its own and not exposed; D3D12's standard swizzle is neither
supported here nor the console's; and on this card a texture cannot be put
in CPU-visible card memory at all. So title memory stays the truth and the
host texture is derived from it:

- **Made stale** by: creation (never detiled); DirectStorage completion over
  its memory; a page mapping over its memory; a title GPU write over its
  memory that is not to the texture object itself (copies into raw memory,
  UAV writes over raw memory: their ranges are known at record time),
  marked when the submission completes. Writes to the texture object itself
  go to the host texture directly, as today.
- **Refreshed** before the first submission that samples it while stale:
  the CPU detiles from the title's address (cached PC memory, coherent with
  the GPU's writes after the fence) straight into the upload ring (class C,
  write-combined streaming writes), and the GPU copies ring → host texture
  card to card. A later step can detile on the GPU from the spanning buffer.
- **The one case without an event: textures the title's CPU fills itself**
  (movie frames, glyph atlases, CPU-made textures), which no DirectStorage
  request or GPU command wrote. Those keep today's content fingerprint at
  sampling (`texture.rs` `fingerprint`). A texture is in that class while
  nothing announced has ever written it. This is the only residual check, a
  hash of the few textures in it (21 MB a frame in the 07:38 run). No write
  watching and no guards.
- **Eviction is free.** Title memory is the truth, so a host texture let go
  for room (the existing `let_go_idle` / `room`) is detiled again on its
  next use. There is never anything to copy back.

`VICTORY_UNIFIED_CHECK=1` (stage 3) fingerprints every sampled texture and
logs those that changed with no event. It should report only the CPU-filled
class. Anything else is a writer that has to be found.

## 7. Capacity and spill (question 4)

- **Class A never goes to the card** in stages 1–4. It is pinned PC memory:
  about 12 GB on the Series X profile (heaps plus pool; views add nothing),
  about 8 GB on Series S. The 07:38 run's process had 18.4 GB in memory
  (its memory line), mostly copies, and they go. Opening lazily, at the first GPU use
  of each unit, pins only units the GPU ever touches. The Series S profile
  is the right default for an 8 GB card.
- **Card memory, in order of claim:** render targets, depth and UAV
  textures (must be there, the GPU writes them every frame); the upload
  ring, fixed at 128–256 MB; VIctory's own buffers; then detiled title
  textures, least recently sampled let go first when
  `QueryVideoMemoryInfo` leaves less than 512 MB. If that is not enough,
  vkd3d places new `DEFAULT` allocations in PC memory: memory type 0 takes
  optimal images. That is the driver's spill, slow but correct.
- **No promotion of title buffers to the card** unless stage 2 shows the
  bus is the limit. Stage 5 says how, and it is optional.

The bus budget is the one number to watch. At 13 GB/s, a 30 fps frame can
read about 430 MB of buffers from PC memory before the bus alone costs the
frame. `nvidia-smi dmon -s t` prints the bus traffic per second during
play.

## 8. What goes, what stays (question 5)

Line counts from `integrate` at 37164d6. "Goes" for whole files is exact;
for shared files it is an estimate by section.

| File | Lines | Goes | Why |
|---|---:|---:|---|
| `victory-gpu/src/shadow.rs` | 4121 | 4121 | host copies of buffers and windows, flights, snapshots, compares, card copies |
| `victory-gpu/src/watch.rs` | 1653 | 1653 | write watch, guards on every alias, the fault handler |
| `victory-gpu/src/inplace.rs` | 1379 | 1379 | superseded: `unified.rs` opens the memory; its probe and census are replaced |
| `victory-gpu/src/tiles.rs` | 1587 | 1587 | levels on demand (off by default); eviction is free once title memory is the truth. Bring it back only if texture room is still the limit |
| `victory-gpu/src/plaincopy.rs` | 318 | 318 | copies in place of reads of title memory |
| `victory-gpu/src/copystat.rs` | 1358 | ~1100 | statistics of copies; the memory map and timing lines stay |
| `victory-gpu/src/memory.rs` | 319 | ~200 | the directory names heaps, not copies; faults go |
| `victory-gpu/src/host.rs` | 8824 | ~4500 | card work, in-flight fetch, relay, late wait and flights (1406–3695); card and relay methods (3838–5075); deferred refresh batch (6399–7075); `copy_mapped`/`read_*`/`write_mapped` (7075–7275); copy tests (8513–8824) |
| `victory-gpu/src/draw.rs` | 3489 | ~500 | root-view readiness, heap-buffer passes, refresh before draws |
| `victory-gpu/src/descriptor/host.rs` | 3644 | ~600 | choosing a copy for a buffer view (`buffer_copy`, `own_bytes*`, raw windows, exact copies) |
| `victory-gpu/src/texture.rs` | 1618 | ~450 | change detection by watch and query (`readable`, `committed_by_query`, `mapped_bytes`, waits); detiling and staging stay |
| `victory-gpu/src/resource.rs` | 4047 | ~300 | shadow registration and the buffer-copy accounting |
| `victory-gpu/src/list.rs`, `queue.rs`, `indirect.rs` | 6313 | ~300 | reads of title memory through copies |
| `victory-hle/src/modules/xmem.rs` | 1455 | ~120 | `MEM_WRITE_WATCH`, `shown`-for-guards, remapping hooks; ~60 lines added (sections committed, map and heap notifications) |
| `victory-hle/src/modules/dstorage.rs` | 1797 | ~20 | `kernel_will_write` and guard comments |
| **Total** | | **~17,100** | 32 `VICTORY_*` switches live in the five deleted files alone |

Added: `unified.rs` runtime (opening, canary, ring, stale-texture driving)
about 600 lines; `unified_map.rs` 230 (with tests); the probe, 970 lines,
can go once its answers are in this document. **Stays:** `xmem`'s placeholders
and sections (the CPU page table), detiling (`victory-formats`), host
textures, twins and render targets, descriptors, pipelines, shader
translation, `memcap.rs`, the frame and present path.

## 9. Staged migration

(As planned before the probe; the levels now are those of section 12.)

One switch, `VICTORY_UNIFIED`, with levels; each level includes the ones
below. The coordinator runs each test in-game. Default 0 changes nothing.

**Stage 0: probe (done, `VICTORY_UNIFIED=1`).** What to look for in the log
(`grep "unified probe"`):

- `aliases: ... the GPU reads what the CPU wrote through both true ... by a
  copy true, by a shader true` is the design's foundation. If it says `THE
  HEAP IS NOT THE MEMORY`, the import fell back and this design does not
  work as is.
- `private memory ... the GPU reads what the CPU wrote: true`.
- `GPU-upload heap ... the CPU sees it at 0x...: ABOVE 1 TiB` is expected
  under Proton and confirms section 1. `texture heap in GPU-upload: REFUSED`
  or `false` is expected on this card.
- `the card reads 256 MB at: X GB/s from a heap opened on a section, Y from
  a GPU-upload heap, Z from its own memory` gives the bus cost of section 7.
- the `VERDICT` line sums it up.

Then, separately: `VICTORY_UNIFIED=2` (tiles into an opened heap; may
remove the device, so run it alone and read the last "tiles:" line), and
`VICTORY_UNIFIED=1 VICTORY_UNIFIED_PROBE_BIG_MB=6400` (one pool-sized
section: open time, pinned memory, whether a 6.25 GB buffer places). Watch
free RAM.

**Stage 1: open and map, change nothing (`VICTORY_UNIFIED=3`).** xmem
notifies regions, sections and views (3 call sites already exist:
`title_memory_section`, `title_memory_view`, `title_memory_remapped`, plus
one for regions). `unified.rs` opens heaps lazily, canaries them and fills
the `Map`. `shadow::find` callers also ask the map, without using the
answer. Log every 10 s: heaps open and MB, refused and canary failures,
lookups resolved, scattered and unknown. **Test:** 5 minutes from boot into
the world. Pass: no canary failure, ≥ 99% of buffer lookups resolve or are
scattered, RAM within budget, frame rate unchanged.

**Stage 2: buffers in place (`VICTORY_UNIFIED=4`).** For a resolved range,
`shadow::find` returns an in-place shadow over the spanning buffer (the
`place: Some(..)` path that `inplace.rs` already feeds, so no refresh, no
snapshot and no copy-back are taken for it); scattered ranges keep copies.
**Test:** the 07:38 scene (`autorun.sh`, same keys). Pass: copystat "N
refreshes ... MB" and "shader writes looked for" near 0; "copying back" 0 ms;
captures match the previous run's frames; fps up. Record `nvidia-smi dmon
-s t` and the GPU ms in the fps line: that is the bus cost.

**Stage 3: textures by events (`VICTORY_UNIFIED=5`).** xmem stops
requesting `MEM_WRITE_WATCH`; guards and watch queries are off; `Stale` is fed
by DirectStorage, page mappings and declared GPU writes; CPU-filled textures
keep the fingerprint. **Test:** load into the world, walk, open the menu,
play a movie. Pass: textures stream in (no black or old textures), UI
text and movies update, `VICTORY_UNIFIED_CHECK=1` reports no texture changed
without an event outside the CPU-filled class, texture ms per frame down.

**Stage 4: delete.** Unified becomes the only path and the switch goes;
section 8's list is removed; `Unknown` ranges become an error line.
**Test:** stage 2 and 3 tests again on the clean build; build time and
binary size down; no new error lines over 15 minutes.

**Stage 5 (optional): what the measurements ask for.** If the probe said
tiles work and stage 1 counted scattered buffers, map the tiled heap with a
reserved buffer (section 4) and drop the last copies. If stage 2 shows the
bus is the limit (bus traffic near 13 GB/s, GPU ms high), promote
DirectStorage-filled buffers that nothing else writes into card memory at
DirectStorage completion, demoting on any announced write or on the first
CPU write (one read-only protection per promoted range, set once, no
re-arming). **Test:** bus traffic and GPU ms before and after.

## 10. Risks

1. **Silent import fallback** (vkd3d-proton). Mitigated by the canary per
   heap; a failure is loud and leaves the old path for that unit until
   stage 4.
2. **Pinned memory.** 8–12 GB resident. Mitigated by lazy opening, the
   Series S profile, smaller sections. A 16 GB PC would struggle.
3. **Bus bandwidth.** Buffer reads cross PCIe Gen4 x8. The design removes
   CPU work measured in hundreds of ms a frame and adds GPU-side bus time.
   Measure it (probe, stage 2); stage 5 is the answer if needed.
4. **D3D12 view rules on console offsets.** Table CBVs want 256-byte
   alignment, structured views an offset that is a multiple of the stride.
   Copies today start at the buffer's own offset, so the rules are no worse
   than now. Root views under vkd3d are raw addresses with no such rule.
5. **One buffer in many states.** The spanning buffer is read, written,
   copied and used as indirect arguments in one list. Buffers have no
   layout under enhanced barriers and vkd3d ignores buffer states; keep it
   in common with UAV/global barriers, as copies are used today.
6. **Native Windows.** `OpenExistingHeapFromFileMapping` refuses
   `SEC_RESERVE`, so the pool must be committed (the commit limit, error
   1450, is why it is reserved today), and an opened address range cannot be
   freed or protected (error 87). Proton first; on Windows, smaller sections
   and a pagefile note.
7. **CPU-filled textures misclassified.** A texture written by both
   DirectStorage and later the CPU would be missed. The check mode finds it;
   the fix is to keep it in the fingerprint class.
8. **Freed and reused memory.** A heap must be retired before its pages
   are reused, or the GPU reads the old pinned pages. The title was not seen
   freeing GPU memory. Hook its frees when it does.

## 11. Recommendation

Build the system-memory home (classes A–D), not "title memory is a
GPU-upload heap". On this stack the second cannot hold textures, cannot be
placed below 1 TiB, cannot alias, and would not fit. The first is what
vkd3d-proton already implements, keeps every title address and alias as it
is, and removes the whole detection machinery with it. GPU-upload memory
has one job: the write-only upload ring that carries detiled texels into
card memory.

Order: run the probe (`VICTORY_UNIFIED=1`, then `=2` alone, then the big
section). If the alias answers are true, stages 1 and 2 are about 600 lines
on top of the existing in-place plumbing and give the decisive measurement:
copy cost gone, bus cost in its place. Stage 3 removes write watching,
stage 4 deletes about 17,000 lines.

## 12. As built (stages 1–3, on by default)

### What the probe measured

RTX 5060 Ti 8 GB, PCIe 4 x8, Proton with vkd3d-proton master
(`VICTORY_UNIFIED=1` of the first probe build):

- A committed section opened with `OpenExistingHeapFromFileMapping`: two
  aliases below 1 TiB agree, the GPU reads through both and writes back
  through both, by copy and by shader. A texture placed in it is refused
  (`E_INVALIDARG`). Private memory opened with
  `OpenExistingHeapFromAddress1`: the GPU reads it.
- GPU-upload heap: buffers *and* textures (placed and committed) work,
  against what `vulkaninfo` suggested, but the mapping lies at
  `0x7fe7d0000000`, above 1 TiB. The custom L1 write-combine heap is the
  same.
- The card reads 12.6 GB/s from an opened section, 193 GB/s from a
  GPU-upload heap, 199 GB/s from its own memory. The CPU writes a
  GPU-upload mapping at 6.6 GB/s.

The address settles it: title memory cannot be a GPU-upload heap under
Proton. It stays in the PC's memory and is opened, as section 1 planned.

### The switch

`VICTORY_UNIFIED`: unset or `1`, on; `0` is taken as `1` since the copies
of title buffers went (2026-11-25: there is nothing to go back to); `2`,
on plus the probe at the first sweep; `3`, also the probe's tile-mapping test. The probe lives in
`unified_probe.rs`; `VICTORY_UNIFIED_PROBE_MB` and
`VICTORY_UNIFIED_PROBE_BIG_MB` as before.

### What happens

- **xmem** (Proton only, unified on). An `XMemVirtualAlloc` region at an
  address of our choosing is reserved 64 KiB longer: the extra page is a
  canary page of ours, committed, and the region is announced
  (`victory_gpu::title_region`). `XMemAllocatePhysicalPages` makes
  sections of at most 512 MiB (8192 pages), each with a canary page after
  the title's, and announces each (`title_section`, `title_section_gone`
  when its handle closes). Opening a 512 MiB section pins only that much.
- **Opening** (`unified.rs` `place`, at the first GPU use of an address in
  a unit), whole, so that no title buffer lies across two heaps. A region:
  its reserved pages are committed (counted in the opening line; run p7
  found none), every page is written once in place with an atomic add of 0
  (so pages a write watch keeps read-only can be pinned), then
  `OpenExistingHeapFromAddress1` over the region and its canary page. A
  section: `OpenExistingHeapFromFileMapping`. Then the canary: the CPU
  writes a 16-byte pattern on the canary page, the GPU reads it back
  through the heap (the probe's own queue), and they must match. A mismatch
  is vkd3d-proton's silent fallback; the unit is refused, logged, and its
  buffers become buffers of their own (below). (Opening 256 MiB pieces instead, tried in run p7,
  pinned 6,565 of 6,713 MB all the same and split buffers at piece edges.)
- **Buffers outside the units** (`unified.rs` `loose`): a title buffer of
  any size in memory no unit holds (a committed resource's memory from the
  title's allocator: the 174 MB GPU heap at 0xb9c0000 in run p7; buffers
  over views at 0xc34530000 in run p10) is opened as a heap of its own over
  its 64 KiB pages (its 4 KiB pages if the next 64 KiB is not committed or
  is another heap's), checked by a word the GPU must read as the CPU sees
  it, and let go when that memory is freed. A committed buffer's memory is
  64 KiB pages of its own, so no two share a page. Any committed protection
  that reads and writes counts, whatever the caching modifiers; view pages
  victory's guards keep read-only are opened first; memory over views is
  let go when the title maps anything else there. A refusal logs the first
  pages that refused (state, protection, type) and is asked again 2 s
  later; pages that lie partly in another heap are refused and said
  (`its pages lie partly in another heap`).
- **Buffers in place** (`shadow.rs`). A title buffer of any size, or a
  4 MiB window of title memory outside buffers, whose memory lies in one
  opened region, one run of one section or one opened buffer, becomes a
  buffer placed over the title's own bytes (on the 64 KiB line below them;
  `Shadow::offset` counts the bias). A list naming it only keeps it alive;
  nothing is copied, compared or merged, no write watch is asked. Shader
  writes are declared, per submission joined, so that CPU steps of a later
  submission that read there wait for it (`host.rs` `late_blocked`) and
  textures over it detile again once it has finished
  (`unified::gpu_wrote`, `settle`). A view that the title maps anew marks
  its in-place buffers stale and the next ask places new ones.
- **Not in place** (since 2026-11-25 the only other kind; card copies,
  PC copies with snapshots, compares, merges and fetch-backs are gone):
  a buffer view whose bytes do not start on an element of its in-place
  buffer (`shadow::exact`: placed buffers start on 64 KiB lines), and
  memory that does not open, get a buffer of their own in the PC's memory
  that each submission naming it fills from title memory
  (`HostList::defer_upload`) and that is never copied back. Shader writes
  into one stay there until the next fill (`shaders write a buffer that is
  not in place`, counted in the unified line). The unified line counts
  them: `not in place (buffers of their own, ...) N made (M MB)`; the frame
  line's `not in place (filled at submission) Y MB` is what lists name.
- **Title copies** (`list.rs` `copy_on_card`). A `CopyBufferRegion` or
  buffer `CopyResource` whose destination is in place runs on the card,
  in the list where the title put it, between UAV barriers, as on the
  console. The source is whatever host buffer holds its bytes. Other title
  copies stay on the CPU at submission; one whose destination lies in a
  unit first waits for the submissions in flight (they may be reading it).
- **Textures by events** (`texture.rs` `refresh_now`, `unified_map.rs`).
  Announced writes are kept as ranges with the number of their newest
  write: DirectStorage completion (plain and page destinations), file
  reads (`ReadFile`, `fread`), page mappings, title copies done on the CPU,
  and shader writes through in-place buffers once their submission has
  finished (`unified::settle`, at each submission and texture check). A
  texture detiles again only if a newer announced write lies over it, plus
  one hash every 64th frame by turns. A texture no announced write ever
  touched is filled by the title's CPU: its content hash decides, every
  frame while it changes, every 8th frame once it holds still. No write
  watch is asked and no guard is set for texture memory.

### Expected effect

- **Frame time:** the per-frame refresh of buffer copies, the snapshots and
  the CPU compares of what shaders wrote go for every in-place buffer:
  those were the 315 ms of copies and the 1.7 s of copy-back per frame of
  the first world frames in the last runs. Title copies into in-place
  buffers move from the submitting thread to the card. In exchange, every
  shader read of an in-place buffer crosses the bus at about 12.6 GB/s:
  watch the GPU time per frame.
- **Card memory:** down by the copies of title buffers and of
  the windows (they no longer exist), the big heap's included (about 350 MB
  with its "before" twin).
- **PC memory:** every opened unit is pinned whole: the regions the GPU
  touches (up to about 6 GB on the Series X profile) and the 512 MiB
  sections it touches. The copies in PC memory go.

### Log lines

- `unified: title memory opened as a heap what=region|section at=… mb=…
  ms=…`: once per unit; `ms` is the cost of pinning it.
- `unified: title memory NOT opened; its buffers keep copies … why=…`:
  must not appear. A `why` with "silent fallback" means the import
  failed.
- Every 10 s: `unified: N heaps opened (… MB of title memory), R refused;
  buffers in place: … alive (… MB) …; title copies on the GPU …, on the
  CPU after waiting for the card …; textures checked for an announced
  write …, by their hash …`. Good signs: R = 0, in-place buffers in the
  hundreds or thousands, "on the CPU after waiting" small next to "on the
  GPU".
- Every 10 s: `views: in 10 s mapped N (pages, scatter, alias, remnant,
  section, canary) showing P pages (A of pages shown elsewhere too),
  unmapped U (…), already as asked K (…), filler stand-ins S (…);
  placeholder splits, merges, maps refused F; title writes into stand-ins
  W, title threads that waited for a change T; views alive V (of one page
  …), stand-in pages …`. Under Proton each map and unmap is a wineserver
  request that walks every view of the process, so (N + U) × V is the
  wineserver's load. `xmem` maps a run of consecutive pages as one view,
  leaves a view that already shows what the title asks, writes
  DirectStorage page destinations through one view per section kept for
  good, and shows a *filler page* (a page of zeros the title maps at many
  places, as in the unused parts of tiled resources) by stand-ins: zeroed
  read-only memory of its own, no view. A filler page DirectStorage writes,
  or found written at the 10 s check, has its stand-ins turned into views
  (a warning says so). W should be 0 (a title write into a stand-in does
  not reach the page's other places). T counts title threads that touched
  a page while its view was away for a change and waited instead of
  faulting. F must be 0. The first filler found logs `xmem: a page of
  zeros the title shows at many places`.
- The gate line (`… fps, … ms/frame …; card memory …`) gives the result.
- Every 5 s, `frame time: … ms/frame over N frames: GPU busy B ms, idle
  I; S submissions/frame (U untimed); title threads recording into our
  lists R ms (summed; by call ms/n: state, draw, mesh, dispatch, indirect,
  reset, own | besides: barrier, copy, clear, value); in submissions P ms
  (… of them; by part: enter, descriptors, upkeep, steps, execute;
  textures brought up to date T ms), waiting for the card W ms [why ms/n,
  …]; flights landed in L ms; title fence events set n/frame, D ms after
  the ask on average; buffers named by lists: in place (read over the bus)
  X MB, not in place (filled at submission) Y MB; top pipelines (ms of a
  sampled frame, of A in all): name ms; …` (`frametime.rs`). All per
  frame. B and I come from GPU timestamps around each submission (B + I ≈
  the frame when the card is the limit only if I is small). The waits by
  why: `lock` (submission lock), `settle` (a waited submission, e.g. a
  saved frame), `reads` (a CPU step reads what flights write: a title copy
  into it, a texture over it), `step` (`CopyMemoryToMemoryX` on the CPU),
  `copy-in` (a CPU copy into title memory), `lists` (256 lists in the
  air), `fence` (the title blocks without an event: only the flights ahead
  of the fence's signal, all of them if no signal is in the line), `own`
  (the card running a waited submission). The pipelines are sampled every
  30th frame with a timestamp at each pipeline change of the title's
  lists. R is the sum of the kinds before the bar (the calls that pass
  through a list's draw state), each with its ms and calls per frame;
  barriers, copies, clears and memory-value waits and writes are timed
  beside it, not in R. The parts of a submission: `enter` (the submission
  lock and the flights in the air, waits included), `descriptors` (records
  the title copied into heaps as bytes), `upkeep` (the sweep, slots waiting
  for a texture, finished shader writes announced), `steps` (the lists'
  CPU steps: fills, copies, textures; the timestamps), `execute`
  (`ExecuteCommandLists` of a submission sent ahead; a waited one counts as
  the `own` wait).
- **Fences pass through.** Everything of ours runs on one host queue, in
  order. A title queue's `Wait` for a fence whose signal is already in the
  line of flights (`host::late_signalled`), and a list's wait for a memory
  value that a flight's bottom-of-pipe write or a fence signal in the line
  meets (`host::late_met`, a fence's value lives at its GPU address), are
  passed at once: what follows goes to the card behind it, and the card
  does the waiting. Only waits nothing in the line meets hold the title
  queue's work on the CPU until flights land.

### Risks of this build

1. **Bus bandwidth.** In-place buffers are read over PCIe 4 x8. If the GPU
   time per frame rises, look at which buffers are read most (stage 5 of
   section 9: copy big read-mostly buffers to the card at DirectStorage
   completion, with the GPU-upload heap as the staging ring; not built).
2. **Pinning.** Opening a 2 GB region pins and touches 2 GB at once, under
   the units' lock: a stall of seconds the first time, and that much PC
   memory for good.
3. **Write ordering the copies used to hide.** In-flight submissions now
   read title memory live. The title's own CPU writes are ordered by its
   fences (signalled only after landing). VIctory's CPU-side writes are
   ordered by `late_blocked` and the copy wait; a bottom-of-pipe value
   written at landing (`Late::Write`) is as late as before. Fills of upload
   memory that read memory a flight "may write" do not wait (`host.rs`
   `blocked_by`); with copies gone nothing checks them any more (run 208
   found the bytes the same 44 times of 44).
4. **CPU-filled textures in memory DirectStorage once filled** are found
   only by the 64th-frame hash (about two seconds late).
5. **Remapped views under live descriptor slots** keep the old buffer until
   the slot is written again or the draw-time check sees another buffer.
6. **Regions freed and reused** by the title would keep the old heap (the
   title was not seen freeing them; xmem has no free hook yet).
7. Native Windows is not covered: opening is Proton-only; on Windows the
   copies stay.

## 13. Memory follows the title

Content is streamed in once and stays until the title unloads it; what
victory holds for title memory follows it.

- **Units open whole** (section 12, Opening). Line every 10 s: `unified
  opening: title memory open now … MB (regions …, sections …, big buffers
  outside them …); pages committed for the import … MB; let go because the
  title freed the memory … MB`.
- **Levels on demand** (`tiles.rs`) is the only path on a card with tiled
  resources tier 2, with texture quality applied to the reserved copy. Line:
  `levels on demand (on): … textures made level by level, their levels … MB,
  tiles given … MB (kept off the card … MB), pool … MB in … heaps …; host
  copies with memory for all levels holding levels the title does not: …`.
  The gate line's "textures the title fills" now includes the pool's heaps.
- **Release at the API point** (`release.rs`): `VirtualFree` (hooked for the
  title, `xmem::memory`), `XMemFreeDefault`, `XMemMapPhysicalPages` with no
  pages and page-mapping copies to nothing call
  `victory_gpu::title_memory_released`: host copies of textures the title
  fills whose memory starts there go with their staging (at once if no list
  and no slot holds them, else when none does; running submissions keep
  them until they land), and opened pieces there go (a heap lives until its
  last placed buffer). Released resources free their host copy at once, as
  before. Eviction is for room only: while the card's memory in use is at
  90% of the layer's budget or more, idle copies go, the least recently
  used first, down to 85%, never one used in the last 5 s (and 4 frames);
  idle copies used to go after 5 s whatever the room (run p7: 43,095 let
  go, 18,790 made again). Line: `memory release
  since the start: because the title freed or unmapped memory … MB …,
  because the title released resources … MB …; by idle eviction … MB …;
  made again after either …; host copies alive …`. "Made again" growing
  every interval means churn.
- **Render targets** are measured, not changed: `render target memory: …
  host copies of textures the card writes take … MB of the card; their
  title memory … MB, … MB with overlaps counted once; … lie over title
  memory another one covers; sharing card pages by title address would
  save about … MB`. The large targets of the 13:53 log (15 of 16 MB or
  more) do not overlap.
- **Process memory** (`release.rs`, every 30 s, Linux view through `Z:`,
  of the process under `/proc` that maps victory's executable over its own
  code; `/proc/self` named another process in run p8):
  `process memory (Linux): resident … MB in … mappings; title addresses:
  anonymous …, shared …; host addresses: anonymous …, shared …; NVIDIA
  driver mappings …; files …; of ours in the PC's memory: …; largest …`.
  Run p7 had 21.6 GB resident against about 14 GB before; this line says
  where.
