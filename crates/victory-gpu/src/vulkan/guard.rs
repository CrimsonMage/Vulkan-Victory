//! The loop guard's buffer (ABI v3, `docs/vulkan-abi.md` section 2.1).
//!
//! Every translated loop gives up after a fixed number of iterations; the
//! invocation then adds 1 to dword 0 of this buffer and stores its
//! program's guard id in dword 1. One 8-byte buffer in host-visible memory
//! serves every line: command buffers are recorded before the queue that
//! runs them is known, so the address pushed (push dwords 2–3) cannot be
//! a line's. After each landing the host reads it; a non-zero count is
//! logged once per guard id, with the program's name and hash, and zeroed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::{Backend, lock};

fn buffer(b: &Backend) -> Option<&'static Buffer> {
    static GUARD: OnceLock<Option<Buffer>> = OnceLock::new();
    GUARD
        .get_or_init(|| {
            let made = Buffer::new(&b.host, 8, vk::BufferUsageFlags::STORAGE_BUFFER, Place::Upload);
            match made {
                Ok(buf) => {
                    // SAFETY: a mapped buffer of 8 bytes, ours.
                    unsafe { buf.mapped.cast::<[u32; 2]>().write([0; 2]) };
                    tracing::info!(
                        address = format_args!("{:#x}", buf.address),
                        mapped = format_args!("{:p}", buf.mapped),
                        "Vulkan: the loop guard buffer (host-visible, coherent, device address; pushed in dwords 2-3)"
                    );
                    Some(buf)
                }
                Err(e) => {
                    tracing::error!(error = %format!("{e:#}"), "Vulkan: no loop guard buffer; loops that give up are not reported");
                    None
                }
            }
        })
        .as_ref()
}

/// The guard buffer's device address for push dwords 2–3 (0: none).
pub(super) fn address(b: &Backend) -> u64 {
    buffer(b).map_or(0, |buf| buf.address)
}

/// The programs by guard id: name and hash, for the report.
static PROGRAMS: Mutex<Option<HashMap<u32, String>>> = Mutex::new(None);

/// A stage was made from the program `name` (`hash`) with `guard_id`.
pub(super) fn register(guard_id: u32, name: &str, hash: &str) {
    lock(&PROGRAMS)
        .get_or_insert_with(HashMap::new)
        .entry(guard_id)
        .or_insert_with(|| format!("{name} {hash}"));
}

/// After a landing: a loop that gave up since the last check is logged
/// (once per guard id) and the count is zeroed. Whether one gave up.
/// At device loss: what the guard buffer holds (loops that gave up in
/// work that never landed: the hang may be theirs).
pub(super) fn at_loss(b: &Backend) {
    let Some(buf) = buffer(b) else { return };
    // SAFETY: as in `check`.
    let words = unsafe { &*buf.mapped.cast::<[AtomicU32; 2]>() };
    let (count, id) = (words[0].load(Ordering::Acquire), words[1].load(Ordering::Acquire));
    let program = lock(&PROGRAMS).as_ref().and_then(|p| p.get(&id).cloned()).unwrap_or_else(|| "unknown".into());
    tracing::error!(count, guard_id = format_args!("{id:#010x}"), program = %program, "Vulkan: device lost: the loop guard buffer (invocations that gave up in work not landed, and the last program that did)");
}

pub(super) fn check(b: &Backend) -> bool {
    let Some(buf) = buffer(b) else { return false };
    // SAFETY: the buffer's coherent mapping, 4-byte aligned; the card
    // writes it with device-scope atomics.
    let words = unsafe { &*buf.mapped.cast::<[AtomicU32; 2]>() };
    if words[0].load(Ordering::Acquire) == 0 {
        return false;
    }
    let count = words[0].swap(0, Ordering::AcqRel);
    let id = words[1].load(Ordering::Acquire);
    static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    let mut seen = lock(&SEEN);
    if seen.contains(&id) {
        return true;
    }
    seen.push(id);
    let program = lock(&PROGRAMS)
        .as_ref()
        .and_then(|p| p.get(&id).cloned())
        .unwrap_or_else(|| "unknown".into());
    tracing::warn!(
        count,
        guard_id = format_args!("{id:#010x}"),
        program = %program,
        "Vulkan: a shader loop hit its guard (it gave up; the program has a loop to fix)"
    );
    true
}
