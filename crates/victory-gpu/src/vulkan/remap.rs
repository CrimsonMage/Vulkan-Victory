//! Loose imports over pages the title maps anew.
//!
//! An import in place pins the pages it was made over. When `xmem` takes
//! away what is mapped somewhere inside a loose import (a view of physical
//! pages remapped, a filler stand-in replaced by a view) and maps other
//! pages there, the CPU writes the new pages while the card would go on
//! reading the pinned old ones: stale data, or zeros, for as long as the
//! import lived. Before, only regions wholly inside the unmapped range were
//! dropped.
//!
//! Now a loose region the unmapped (or freed) range overlaps leaves the
//! map at once, and is imported again, over the pages mapped there now, as
//! soon as every page of it is committed read-write memory (at the next
//! view `xmem` maps over it, or a later frame).
//!
//! **Imports out of use are buried** ([`Memory::bury`]), not freed: a
//! command buffer recorded before (a copy, a fill, a query resolve, a map
//! that names the import) may not be submitted yet. Each import is freed
//! once every command buffer begun before it left use is submitted and
//! the card has run it (`Timeline::mark`, `Timeline::passed`); vk39's copy
//! engine faulted on an unmapped device address after an import was freed
//! once the card had run only what was given so far.

use victory_vk::memory::Imported;

use super::{Backend, Memory, Region, memory};

/// Frames a range waits to be imported again before it is forgotten (its
/// pages are named again at the next recording that reads them).
const PATIENCE: u64 = 600;

impl Memory {
    /// Loose regions `[at, end)` overlaps without holding them whole leave
    /// the map (their imports to die later); whether any did.
    pub(super) fn drop_stale_loose(&mut self, b: &Backend, at: usize, end: usize) -> bool {
        let over: Vec<usize> = self
            .regions
            .range(..end)
            .filter(|&(&a, r)| r.loose && a + r.len > at && !(a >= at && a + r.len <= end))
            .map(|(&a, _)| a)
            .collect();
        for a in &over {
            if let Some(r) = self.regions.remove(a) {
                if let Some(i) = r.import {
                    self.bury(b, i);
                }
                self.rejoin.push((*a, r.len, self.tick));
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                    tracing::info!(
                        region = format_args!("{a:#x}"),
                        mb = r.len >> 20,
                        unmapped = format_args!("{at:#x}+{:#x}", end - at),
                        "Vulkan: pages inside a loose import were mapped anew; it is imported again over them"
                    );
                }
            }
        }
        !over.is_empty()
    }

    /// `i` is out of use: freed once nothing that may name it can still
    /// run ([`each_frame`]).
    pub(super) fn bury(&mut self, b: &Backend, i: Imported) {
        self.dying.push((b.timeline.mark(), i));
    }

    /// The ranges waiting to be imported again that overlap `[at, at +
    /// len)` (or all, for `None`) come back as wanted regions once every
    /// page of them is usable.
    pub(super) fn rejoin_over(&mut self, at: usize, len: usize) {
        self.rejoin_where(|base, bytes| base < at + len && at < base + bytes);
    }

    fn rejoin_where(&mut self, near: impl Fn(usize, usize) -> bool) {
        if self.rejoin.is_empty() {
            return;
        }
        let tick = self.tick;
        let mut left = Vec::new();
        for (base, bytes, since) in std::mem::take(&mut self.rejoin) {
            let free = self.regions.range(..base + bytes).all(|(&a, r)| a + r.len <= base);
            if near(base, bytes) && free && super::loose::all_usable(base, base + bytes) {
                self.regions.insert(base, Region { len: bytes, import: None, refused: false, wanted: true, loose: true });
            } else if tick.saturating_sub(since) < PATIENCE {
                left.push((base, bytes, since));
            }
        }
        self.rejoin = left;
    }
}

/// Every frame: buried imports nothing can name any more are freed; ranges
/// waiting to be imported again are tried.
pub(super) fn each_frame(b: &Backend) {
    let mut m = memory(b);
    m.rejoin_where(|_, _| true);
    if m.dying.is_empty() {
        return;
    }
    let mut ripe = Vec::new();
    let mut keep = Vec::new();
    for (mut mark, i) in std::mem::take(&mut m.dying) {
        if b.timeline.passed(&b.host, &mut mark) {
            ripe.push(i);
        } else {
            keep.push((mark, i));
        }
    }
    m.dying = keep;
    drop(m);
    for i in ripe {
        if cfg!(feature = "diagnostics") {
            tracing::info!(
                title = format_args!("{:#x}", i.title_base),
                device = format_args!("{:#x}..{:#x}", i.device_base, i.device_base + i.len),
                "Vulkan: an import freed (device addresses)"
            );
        }
        // SAFETY: out of every map, every command buffer begun while it was
        // in use is submitted, and the card has run them.
        unsafe { i.destroy(&b.host) };
    }
}
