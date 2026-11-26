//! Changes of what title memory shows against reads of it in place: the
//! runtime takes [`title_memory_remapping`] around every change of its page
//! table, and a read of title memory in place [`title_memory_reading`].

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    let guard = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    HELD.with(|h| h.set(h.get() + 1));
    Remapping(guard)
}

impl Drop for Remapping {
    fn drop(&mut self) {
        HELD.with(|h| h.set(h.get() - 1));
    }
}

impl Drop for Reading {
    fn drop(&mut self) {
        HELD.with(|h| h.set(h.get() - 1));
    }
}

thread_local! {
    /// How many holds of [`MAPPING`] this thread has.
    static HELD: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Whether this thread holds the lock of mapping changes, either way: it
/// must not wait for a change to end ([`remapping_over`]).
pub fn mapping_lock_held() -> bool {
    HELD.with(|h| h.get() > 0)
}

/// Waits, at most `ms` milliseconds, until no change of what title memory
/// shows is under way; false if one still is. For a title thread that
/// touched a page in the moment the runtime had taken its view away to map
/// it again (the console changes its page table in one step): it may then
/// carry on. Not for a thread that holds the lock ([`mapping_lock_held`]).
pub fn remapping_over(ms: u64) -> bool {
    let start = std::time::Instant::now();
    loop {
        if let Ok(guard) = MAPPING.try_read() {
            drop(guard);
            return true;
        }
        if start.elapsed().as_millis() as u64 >= ms {
            return false;
        }
        std::thread::yield_now();
    }
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    let guard = MAPPING.read().unwrap_or_else(|e| e.into_inner());
    HELD.with(|h| h.set(h.get() + 1));
    Reading(guard)
}
