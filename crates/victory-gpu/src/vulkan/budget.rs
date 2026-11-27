//! Card memory: one rule, and what tells where it goes.
//!
//! **The rule** (KISS, Series S-like): everything of ours on the card
//! stays under [`CEILING`], and never grows into the last [`ROOM`] of the
//! card's budget (`VK_EXT_memory_budget`: what the driver, the spills of
//! shaders and other programs leave us).
//!
//! - Images come first: a new image block takes the pool's GPU view's
//!   blocks if it must ([`fits`]). One that would still cross the limit is
//!   refused (counted as `images_refused`).
//! - The pool's GPU view takes what the images leave: it grows a block
//!   while ours stays under the limit, and past it evicts its least
//!   recently touched page (a map never fails). Every frame, a view over
//!   what the images leave gives blocks back ([`PoolView::shrink`]: the
//!   coldest blocks), down to [`LEAST`].
//!
//! **What tells** (diagnostics builds): every card allocation of 64 MiB or
//! more as it happens (what, size, ours after it), and the card-memory
//! line whenever ours or the heap's use moves by 256 MiB, besides every
//! 10 s.
//!
//! [`PoolView::shrink`]: victory_vk::pool::PoolView::shrink

use std::sync::atomic::{AtomicU64, Ordering};

use victory_vk::account::{self, Kind};
use victory_vk::ash::vk;
use victory_vk::host::Host;
use victory_vk::pool::LEAST;

use super::{Backend, images, memory};

/// Ours on the card at most. Raised from 4,608 MiB on 2026-11-27 (the
/// user's decision, to be lowered again as room is found): at 4,608 the
/// pool's GPU view got ~0.4 of the ~1.05 GiB the title shows through it,
/// and the character's head, the window mesh and the far buildings came
/// and went with eviction. The room to win back is in the ~2.2 GiB of
/// card-written images (among them seven 3840x2160 interface targets).
pub(super) const CEILING: u64 = 5504 << 20;
/// The card's budget we never grow into.
const ROOM: u64 = 1 << 30;
/// A move of this much prints the card-memory line.
const STEP: u64 = 256 << 20;

use account::card_heap as heap;

/// Bytes of card memory ours may take now: the ceiling, less what the
/// budget leaves past the others' use and the room.
pub(super) fn limit(host: &Host) -> u64 {
    let (ceiling, room) = rule();
    let Some((budget, usage)) = heap(host) else {
        return ceiling;
    };
    // The driver's own use for us (spills, internal heaps) counts against
    // the ceiling too: the process's whole use stays under it (vk40: ours
    // 5,318 MiB, the process 5,609).
    let others = usage.saturating_sub(account::card_total());
    ceiling.saturating_sub(others).min(budget.saturating_sub(others).saturating_sub(room))
}

/// [`CEILING`] and [`ROOM`]; diagnostics builds take
/// `VICTORY_VK_CEILING_MB` and `VICTORY_VK_ROOM_MB` instead (to see what a
/// run shows when no image is refused and no pool page evicted).
fn rule() -> (u64, u64) {
    static R: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let asked = |name: &str| cfg!(feature = "diagnostics").then(|| std::env::var(name).ok()?.trim().parse::<u64>().ok()).flatten().map(|mb| mb << 20);
        let (ceiling, room) = (asked("VICTORY_VK_CEILING_MB"), asked("VICTORY_VK_ROOM_MB"));
        if ceiling.is_some() || room.is_some() {
            tracing::warn!(ceiling_mb = ceiling.unwrap_or(CEILING) >> 20, room_mb = room.unwrap_or(ROOM) >> 20, "Vulkan: the card-memory rule is the environment's (diagnostics)");
        }
        (ceiling.unwrap_or(CEILING), room.unwrap_or(ROOM))
    })
}

/// The memory lock, waited for a few milliseconds at most: the caller
/// holds the images' lock, which a thread that holds this one may want.
fn memory_soon(b: &Backend) -> Option<std::sync::MutexGuard<'_, super::Memory>> {
    let start = std::time::Instant::now();
    loop {
        if let Ok(m) = b.memory.try_lock() {
            return Some(m);
        }
        if start.elapsed() > std::time::Duration::from_millis(5) {
            return None;
        }
        std::thread::yield_now();
    }
}

/// Whether `bytes` more of ours (an image block) may be allocated: never
/// past the limit for longer than the card takes to let go of what the
/// pool's views give back. Images come first: the views give blocks back
/// for them now (down to [`LEAST`]), and the blocks on their way out
/// count as gone. Before, they counted until the card had freed them and
/// the lock was only tried: the image that asked was refused, its
/// descriptor read the null image, and a texture programs pick by index
/// is not named again (run tx1: six images in the world's load).
pub(super) fn fits(b: &Backend, bytes: u64) -> bool {
    let host = &b.host;
    let over = |leaving: u64| (account::card_total() + bytes).saturating_sub(limit(host) + leaving);
    let mut leaving = 0;
    if over(0) > 0
        && let Some(mut m) = memory_soon(b)
    {
        let mut need = over(0);
        for p in &mut m.pools {
            let has = p.card_bytes();
            let target = has.saturating_sub(need).max(LEAST);
            if target < has && p.shrink(host, &b.timeline, target).is_ok() {
                need = need.saturating_sub(has - p.card_bytes());
            }
            leaving += p.retiring();
        }
    }
    let fits = over(leaving) == 0;
    if !fits {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::warn!(
                mb = bytes >> 20,
                ours_mb = account::card_total() >> 20,
                limit_mb = limit(host) >> 20,
                "Vulkan: card memory for images refused (the card-memory ceiling)"
            );
        }
    }
    fits
}

/// Every frame: each pool view's ceiling is the limit; ours over it
/// gives pool blocks back now, down to [`LEAST`] a view.
pub(super) fn follow_budget(b: &Backend) {
    let limit = limit(&b.host);
    let mut m = memory(b);
    let mut over = account::card_total().saturating_sub(limit);
    for p in &mut m.pools {
        p.ceiling = limit;
        let has = p.card_bytes();
        let target = has.saturating_sub(over).max(LEAST);
        over = over.saturating_sub(has - target.min(has));
        if has > target
            && let Err(e) = p.shrink(&b.host, &b.timeline, target)
        {
            tracing::warn!(error = %format!("{e:#}"), "Vulkan: the pool view could not give card memory back");
        }
    }
    super::paging::settle(b, &mut m);
}

/// All the card memory we allocated, by kind, against the driver's own
/// count of the card heaps' use; the difference is the driver's (shader
/// local memory for spills, internal heaps). Host-heap allocations and
/// imports are listed apart: they are system memory.
pub(super) fn card_memory(b: &Backend) {
    let (card, host) = account::totals_mb();
    let (heap_budget, heap_used) = heap(&b.host).map_or((0, 0), |(t, u)| (t >> 20, u >> 20));
    let ours: u64 = card.iter().sum();
    // The whole process on the card, as nvidia-smi counts it, and the part
    // no Vulkan heap's use shows: the driver's own objects.
    let (process, outside) = account::process(&b.host).map_or((0, 0), |(p, o)| (p >> 20, o >> 20));
    // Every heap: size, use and budget in MiB ("d" marks card heaps).
    let heaps: Vec<String> = b
        .host
        .heaps()
        .iter()
        .map(|(h, bu)| {
            let (budget, used) = bu.unwrap_or((0, 0));
            let d = if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) { "d" } else { "" };
            format!("{}{d}/{}/{}", h.size >> 20, used >> 20, budget >> 20)
        })
        .collect();
    tracing::info!(
        process_mb = process,
        outside_mb = outside,
        images_mb = card[0],
        images_bound_mb = images::used_mb(),
        pool_mb = card[1],
        buffers_mb = card[3],
        scratch_mb = card[5],
        other_mb = card[4],
        ours_mb = ours,
        limit_mb = limit(&b.host) >> 20,
        heap_used_mb = heap_used,
        heap_budget_mb = heap_budget,
        driver_mb = heap_used.saturating_sub(ours),
        host_buffers_mb = host[3],
        imports_mb = host[2],
        heaps = %heaps.join(" "),
        traced = %account::traced_line(),
        "Vulkan: card memory"
    );
}

/// Diagnostics: a card allocation of 64 MiB or more, as it happens.
fn big(kind: Kind, bytes: u64, total: u64) {
    tracing::info!(
        ?kind,
        mb = bytes >> 20,
        ours_mb = total >> 20,
        "Vulkan: a big card allocation"
    );
}

/// Diagnostics: a card allocation of 64 MiB or more freed, as it happens.
fn big_freed(kind: Kind, bytes: u64, total: u64) {
    tracing::info!(?kind, mb = bytes >> 20, ours_mb = total >> 20, "Vulkan: a big card allocation freed");
}

/// Diagnostics: the watchers of big allocations and frees are set.
pub(super) fn watch() {
    if cfg!(feature = "diagnostics") {
        account::watch(big);
        account::watch_frees(big_freed);
    }
}

/// Diagnostics, polled often: the card-memory line when ours or the
/// heap's use moved by [`STEP`] since it was last printed.
pub(super) fn steps(b: &Backend) {
    static LAST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let now = [account::card_total(), heap(&b.host).map_or(0, |(_, u)| u)];
    let moved = now
        .iter()
        .zip(&LAST)
        .any(|(&n, l)| n.abs_diff(l.load(Ordering::Relaxed)) >= STEP);
    if moved {
        for (n, l) in now.iter().zip(&LAST) {
            l.store(*n, Ordering::Relaxed);
        }
        card_memory(b);
    }
}

/// `victory vkprobe` with `VICTORY_VK_PROBE_SPV=<folder>`: every module of
/// the folder (by name: `*.spv`, a compute module; `*.vks`, a translation
/// of the stage cache) is compiled as the game compiles its stages, and
/// what the card lost over each compile is printed (when it is 1 MiB or
/// more): MiB outside every heap's use and MiB of heap use that is no
/// allocation of ours; then the sums by kind of stage. How the shape of a
/// module is tested against the driver (`docs/vulkan-host.md`, "Card
/// memory the driver takes").
pub(super) fn probe_spv(b: &Backend, dir: &std::path::Path) -> anyhow::Result<()> {
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "spv" || e == "vks"))
        .collect();
    files.sort();
    let d = &b.host.device;
    let now = || heap(&b.host).unwrap_or((0, 0));
    let start = now();
    let mut made = Vec::new();
    // By kind of stage: modules, SPIR-V words, bytes outside, bytes of heap use.
    let mut sums: std::collections::BTreeMap<String, [i64; 4]> = std::collections::BTreeMap::new();
    for file in &files {
        let (stage, words, samplers, tables) = if file.extension().is_some_and(|e| e == "vks") {
            let vs = super::stage_cache::read_file(file).ok_or_else(|| anyhow::anyhow!("{}: not a translation", file.display()))?;
            (vs.stage, vs.spirv, vs.samplers, vs.tables)
        } else {
            let bytes = std::fs::read(file)?;
            let words = bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
            (victory_shader::HwStage::Cs, words, Vec::new(), Vec::new())
        };
        let before = now();
        let t = std::time::Instant::now();
        let fitted = victory_vk::limits::fit_float_controls(&words, b.host.caps.signed_zero_inf_nan);
        let code = fitted.as_deref().unwrap_or(&words);
        // SAFETY: a SPIR-V module from a file the user names.
        let module = unsafe { d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None) }?;
        let p = super::stages::compile(b, stage, module, &samplers, &tables, None);
        let after = now();
        let (outside, heap) = (before.0 as i64 - after.0 as i64, after.1 as i64 - before.1 as i64);
        let s = sums.entry(format!("{stage:?}")).or_default();
        *s = [s[0] + 1, s[1] + words.len() as i64, s[2] + outside, s[3] + heap];
        if outside.abs().max(heap.abs()) >= 1 << 20 || p.is_err() {
            println!(
                "{}: {stage:?} words {} ms {} outside {} MiB heap {} MiB{}",
                file.file_name().unwrap_or_default().to_string_lossy(),
                words.len(),
                t.elapsed().as_millis(),
                outside >> 20,
                heap >> 20,
                p.as_ref().err().map(|e| format!(" FAILED {e:#}")).unwrap_or_default(),
            );
        }
        made.push((module, p.ok()));
    }
    for (kind, s) in &sums {
        println!("{kind}: modules {} kwords {} outside {} MiB heap {} MiB", s[0], s[1] / 1000, s[2] >> 20, s[3] >> 20);
    }
    for (module, p) in made {
        // SAFETY: ours, never used.
        unsafe {
            if let Some(p) = p {
                d.destroy_pipeline(p, None);
            }
            d.destroy_shader_module(module, None);
        }
    }
    let end = now();
    println!(
        "after destroying them: outside {} MiB heap {} MiB since the start",
        (start.0 as i64 - end.0 as i64) >> 20,
        (end.1 as i64 - start.1 as i64) >> 20
    );
    Ok(())
}
