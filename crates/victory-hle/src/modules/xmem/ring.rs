//! Diagnostics: the last mapping operations xmem made (a ring of
//! [`MOST`]), for the crash report. For an address a crashed thread's
//! registers point at, the describer lists the last operations that
//! touched its page: what (a view mapped, a view taken away, a filler
//! stand-in shown or taken away, a remnant shown again), which thread, how
//! long before the report, the range and the section page shown.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Operations kept.
const MOST: usize = 8192;
/// Operations listed per address.
const LISTED: usize = 8;

struct Op {
    at_ms: u64,
    thread: u32,
    what: &'static str,
    base: usize,
    len: usize,
    section: usize,
    page: usize,
}

static RING: Mutex<VecDeque<Op>> = Mutex::new(VecDeque::new());

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Notes an operation on `[base, base + len)` showing section page
/// (`section`, `page`) at `base` (0, 0 for none).
pub(super) fn note(what: &'static str, base: usize, len: usize, section: usize, page: usize) {
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let at_ms = start().elapsed().as_millis() as u64;
    // SAFETY: a plain query of our own thread.
    let thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let mut ring = RING.lock().unwrap_or_else(|e| e.into_inner());
    if ring.len() == MOST {
        ring.pop_front();
    }
    ring.push_back(Op { at_ms, thread, what, base, len, section, page });
}

/// The last operations that touched the page of `addr`, newest first
/// (never waits: the crashed thread may hold the ring).
pub(super) fn touching(addr: usize) -> String {
    if !cfg!(feature = "diagnostics") {
        return String::new();
    }
    let Ok(ring) = RING.try_lock() else {
        return "; recent mapping operations: the ring is held".into();
    };
    let page = addr & !(super::PAGE - 1);
    let now = start().elapsed().as_millis() as u64;
    let hits: Vec<String> = ring
        .iter()
        .rev()
        .filter(|o| o.base <= page && page < o.base + o.len.max(1))
        .take(LISTED)
        .map(|o| {
            format!(
                "{} {} ms before by thread {:#x}: {:#x}+{:#x} section {} page {}",
                o.what,
                now.saturating_sub(o.at_ms),
                o.thread,
                o.base,
                o.len,
                o.section,
                o.page + (page - o.base) / super::PAGE
            )
        })
        .collect();
    if hits.is_empty() {
        format!("; recent mapping operations: none of the last {} touched this page", ring.len())
    } else {
        format!("; recent mapping operations (newest first): {}", hits.join(" | "))
    }
}
