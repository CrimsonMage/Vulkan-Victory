//! Victory's own heap, counted: the global allocator of the `diagnostics`
//! build ([`Counting`], installed by the `victory` binary) keeps the live
//! bytes of every Rust allocation in the process, split by owner, for the
//! process memory line (`victory_gpu`'s `procmem`).
//!
//! An allocation of [`LARGE`] bytes or more carries its owner in a header
//! in front of it: the owner the allocating thread named last
//! ([`enter`]), so it is counted against that owner until it is freed,
//! whichever thread frees it. Smaller ones are counted as "small" with no
//! header. The counters are per thread (a slot each, written only by its
//! thread, no atomic read-modify-write), summed when the line is made: an
//! allocation costs a thread-local read and a plain store.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering::Relaxed};

/// Who an allocation is for, as named by the code that makes it
/// ([`enter`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Owner {
    /// Nobody named: the rest of victory.
    Other = 0,
    /// Texture uploads: detiling and first-upload workers.
    Textures = 1,
    /// DirectStorage: read and decompression buffers.
    Streaming = 2,
    /// Shader translation and translated shaders kept for pipelines.
    Shaders = 3,
    /// Recording: list calls written down and translated into host lists.
    Recording = 4,
    /// Descriptor heap mirrors.
    Descriptors = 5,
    /// The pool directory's mirror and the card page cache's bookkeeping.
    Directory = 6,
    /// Blocks of the title's own `XMemAllocDefault` (title memory served
    /// from victory's heap).
    Title = 7,
}

/// The owners, in [`Owner`] order.
pub const OWNERS: [Owner; 8] = [
    Owner::Other,
    Owner::Textures,
    Owner::Streaming,
    Owner::Shaders,
    Owner::Recording,
    Owner::Descriptors,
    Owner::Directory,
    Owner::Title,
];

impl Owner {
    pub fn name(self) -> &'static str {
        match self {
            Owner::Other => "other",
            Owner::Textures => "textures",
            Owner::Streaming => "streaming",
            Owner::Shaders => "shaders",
            Owner::Recording => "recording",
            Owner::Descriptors => "descriptors",
            Owner::Directory => "directory",
            Owner::Title => "title blocks",
        }
    }
}

/// Allocations of at least this many bytes carry their owner.
pub const LARGE: usize = 4096;
/// The counter column of small allocations, after the owners'.
const SMALL: usize = OWNERS.len();
const COLUMNS: usize = OWNERS.len() + 1;
/// Threads with a counter slot of their own; later threads share one.
const SLOTS: usize = 512;

#[repr(align(128))]
struct Slot([AtomicIsize; COLUMNS]);

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: Slot = Slot([const { AtomicIsize::new(0) }; COLUMNS]);
static TABLE: [Slot; SLOTS] = [ZERO; SLOTS];
/// Shared by threads past [`SLOTS`], counted atomically.
static SHARED: Slot = ZERO;
static NEXT: AtomicUsize = AtomicUsize::new(0);
/// Whether [`Counting`] is the global allocator (it has counted once).
static ON: AtomicBool = AtomicBool::new(false);
const UNSET: u32 = u32::MAX;
const SHARED_SLOT: u32 = u32::MAX - 1;

thread_local! {
    static OWNER: Cell<u8> = const { Cell::new(0) };
    static MINE: Cell<u32> = const { Cell::new(UNSET) };
}

#[inline]
fn add(column: usize, delta: isize) {
    let slot = MINE
        .try_with(|m| {
            let mut i = m.get();
            if i == UNSET {
                let n = NEXT.fetch_add(1, Relaxed);
                i = if n < SLOTS { n as u32 } else { SHARED_SLOT };
                m.set(i);
            }
            i
        })
        .unwrap_or(SHARED_SLOT);
    if slot == SHARED_SLOT {
        SHARED.0[column].fetch_add(delta, Relaxed);
    } else {
        // Only this thread writes its slot.
        let c = &TABLE[slot as usize].0[column];
        c.store(c.load(Relaxed) + delta, Relaxed);
    }
}

#[inline]
fn current() -> u8 {
    OWNER.try_with(Cell::get).unwrap_or(0)
}

/// Names the owner of this thread's allocations until the scope drops (the
/// one before comes back then). Cheap; nothing is counted without
/// [`Counting`].
#[must_use]
pub struct Scope(u8);

pub fn enter(owner: Owner) -> Scope {
    Scope(OWNER.try_with(|o| o.replace(owner as u8)).unwrap_or(0))
}

impl Drop for Scope {
    fn drop(&mut self) {
        let _ = OWNER.try_with(|o| o.set(self.0));
    }
}

/// Names `owner` for the rest of this thread's life (worker threads that
/// serve one owner only).
pub fn enter_for_thread(owner: Owner) {
    let _ = OWNER.try_with(|o| o.set(owner as u8));
}

/// Live bytes per owner ([`OWNERS`] order) and then of small allocations;
/// `None` when [`Counting`] is not the global allocator.
pub fn live() -> Option<[u64; COLUMNS]> {
    if !ON.load(Relaxed) {
        return None;
    }
    let mut sum = [0isize; COLUMNS];
    for slot in TABLE.iter().chain(std::iter::once(&SHARED)) {
        for (s, c) in sum.iter_mut().zip(&slot.0) {
            *s += c.load(Relaxed);
        }
    }
    Some(sum.map(|s| s.max(0) as u64))
}

/// The bytes in front of a large allocation of alignment `align`: the
/// owner's byte, padded to keep the alignment.
#[inline]
fn header(align: usize) -> usize {
    align.max(16)
}

#[inline]
fn large(layout: Layout) -> Option<(Layout, usize)> {
    let h = header(layout.align());
    Some((Layout::from_size_align(layout.size().checked_add(h)?, h).ok()?, h))
}

/// The counting global allocator: the system's, with counters (see the
/// module comment).
pub struct Counting;

impl Counting {
    #[inline]
    unsafe fn alloc_with(&self, layout: Layout, zeroed: bool) -> *mut u8 {
        // SAFETY (all blocks): the caller's layout, or the same grown by
        // a header that only this allocator reads.
        unsafe {
            if layout.size() < LARGE {
                let p = if zeroed { System.alloc_zeroed(layout) } else { System.alloc(layout) };
                if !p.is_null() {
                    add(SMALL, layout.size() as isize);
                }
                return p;
            }
            let Some((outer, h)) = large(layout) else { return std::ptr::null_mut() };
            let base = if zeroed { System.alloc_zeroed(outer) } else { System.alloc(outer) };
            if base.is_null() {
                return base;
            }
            ON.store(true, Relaxed);
            let owner = current();
            base.write(owner);
            add(usize::from(owner), layout.size() as isize);
            base.add(h)
        }
    }
}

// SAFETY: every block goes back to `System` with the layout it was made
// with (the caller's, or the caller's grown by the header); the header is
// only in front of blocks of `LARGE` bytes or more, which `dealloc` and
// `realloc` tell from the layout they are given.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as the trait requires of the caller.
        unsafe { self.alloc_with(layout, false) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as the trait requires of the caller.
        unsafe { self.alloc_with(layout, true) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc_with` with this layout.
        unsafe {
            if layout.size() < LARGE {
                System.dealloc(ptr, layout);
                add(SMALL, -(layout.size() as isize));
                return;
            }
            let h = header(layout.align());
            let base = ptr.sub(h);
            add(usize::from(base.read()), -(layout.size() as isize));
            System.dealloc(base, Layout::from_size_align_unchecked(layout.size() + h, h));
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` came from `alloc_with` with `layout`; the system's
        // realloc keeps the header's byte at the front of the block.
        unsafe {
            let old = layout.size();
            match (old >= LARGE, new_size >= LARGE) {
                (false, false) => {
                    let p = System.realloc(ptr, layout, new_size);
                    if !p.is_null() {
                        add(SMALL, new_size as isize - old as isize);
                    }
                    p
                }
                (true, true) => {
                    let h = header(layout.align());
                    let base = ptr.sub(h);
                    let Some(grown) = new_size.checked_add(h) else { return std::ptr::null_mut() };
                    let p = System.realloc(base, Layout::from_size_align_unchecked(old + h, h), grown);
                    if p.is_null() {
                        return p;
                    }
                    add(usize::from(p.read()), new_size as isize - old as isize);
                    p.add(h)
                }
                _ => {
                    let new = Layout::from_size_align_unchecked(new_size, layout.align());
                    let p = self.alloc_with(new, false);
                    if !p.is_null() {
                        std::ptr::copy_nonoverlapping(ptr, p, old.min(new_size));
                        self.dealloc(ptr, layout);
                    }
                    p
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_blocks_keep_their_owner_and_alignment() {
        let a = Counting;
        let before = TABLE.iter().map(|s| s.0[Owner::Shaders as usize].load(Relaxed)).sum::<isize>();
        let layout = Layout::from_size_align(10_000, 64).unwrap();
        let p = {
            let _s = enter(Owner::Shaders);
            // SAFETY: a valid layout; freed below with the layout it has
            // then.
            unsafe { a.alloc(layout) }
        };
        assert!(!p.is_null());
        assert_eq!(p as usize % 64, 0);
        let now = TABLE.iter().map(|s| s.0[Owner::Shaders as usize].load(Relaxed)).sum::<isize>();
        assert_eq!(now - before, 10_000);
        // SAFETY: as above; grown, then shrunk below the header's size.
        unsafe {
            p.write_bytes(7, 10_000);
            let q = a.realloc(p, layout, 20_000);
            assert_eq!(q.add(9_999).read(), 7);
            let r = a.realloc(q, Layout::from_size_align(20_000, 64).unwrap(), 100);
            assert_eq!(r.read(), 7);
            a.dealloc(r, Layout::from_size_align(100, 64).unwrap());
        }
        let after = TABLE.iter().map(|s| s.0[Owner::Shaders as usize].load(Relaxed)).sum::<isize>();
        assert_eq!(after, before);
        assert_eq!(current(), Owner::Other as u8);
    }
}
