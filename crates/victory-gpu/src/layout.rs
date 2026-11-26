//! Which layout of the console's queue and list tables the title's D3D12.X
//! runtime has, found at run time.
//!
//! Two are known, with the same SDK version (XCL `real.rs`, "command queue
//! layouts"): the older (2020-era titles) and a newer one with two more
//! methods in the queue before `Signal` and two more in the list's console
//! part, so that every queue slot from 14 and every list slot from 59 on is
//! two later (`queue::newer_slot`, `list::newer_slot`). Nothing the title
//! passes at device creation says which, but its calls do:
//!
//! - a hint first: the device created as `ID3D12Device11` (the title of
//!   2026 does, Minecraft of 2022 does not) makes new objects newer;
//! - then shapes no other method has: a fence passed to the older queue's
//!   slot 16 or 17 (`Signal`/`Wait` of the newer, where the older has
//!   `GetTimestampFrequency(&u64)` and `GetClockCalibration`), a fence to
//!   the newer queue's 14 or 15 (the older's `Signal`/`Wait`), a command
//!   signature passed to the older list's slot 61 (`ExecuteIndirect` of the
//!   newer, `FillMemoryWith32BitValueX` of the older), a copy location of a
//!   resource to the older list's 85 (`CopyTextureRegionX` of the newer,
//!   `SetPredicationBufferX` of the older).
//!
//! The first shape decides for the run. Every live queue and list then gets
//! the decided table (tables come in older/newer pairs, [`born`]); calls
//! made through the wrong one before cannot be taken back.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crate::object::This;

const UNKNOWN: u8 = 0;
const HINT_NEWER: u8 = 1;
const OLDER: u8 = 2;
const NEWER: u8 = 3;

static STATE: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Whether objects made now take the newer tables.
pub(crate) fn newer() -> bool {
    matches!(STATE.load(Ordering::Relaxed), HINT_NEWER | NEWER)
}

/// Whether a call's shape has decided the layout.
pub(crate) fn decided() -> bool {
    matches!(STATE.load(Ordering::Relaxed), OLDER | NEWER)
}

/// The device was created as `ID3D12Device11`: newer, until a call shows
/// otherwise.
pub(crate) fn hint_newer() {
    if STATE.compare_exchange(UNKNOWN, HINT_NEWER, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        tracing::info!("D3D12.X layouts: newer, as the device was created as ID3D12Device11 (until a call shows otherwise)");
    }
}

/// A call showed the layout (`why`): decided for the run, and the live
/// objects follow.
pub(crate) fn shown(newer: bool, why: &str) {
    let want = if newer { NEWER } else { OLDER };
    let was = STATE.load(Ordering::Relaxed);
    if matches!(was, OLDER | NEWER) || STATE.compare_exchange(was, want, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        if was != want && matches!(was, OLDER | NEWER) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(why, "D3D12.X layouts: a call contradicts the layout decided before; kept"));
        }
        return;
    }
    tracing::info!(newer, why, "D3D12.X layouts: decided by a call");
    if (was == HINT_NEWER) != newer {
        let n = with_live(|live| {
            for (&at, tables) in live.iter() {
                // SAFETY: a live object of ours; its first word is its table,
                // which the title's threads read as one word.
                unsafe { (*(at as *const AtomicUsize)).store(tables[usize::from(newer)], Ordering::Release) };
            }
            live.len()
        });
        tracing::info!(objects = n, "D3D12.X layouts: live queues and lists take the decided tables");
    }
}

/// Live queues and lists: their older and newer tables.
static LIVE: Mutex<Option<HashMap<usize, [usize; 2]>>> = Mutex::new(None);

fn with_live<R>(f: impl FnOnce(&mut HashMap<usize, [usize; 2]>) -> R) -> R {
    f(LIVE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new))
}

/// A queue or list at `at` was made with one of the pair `older`/`newer`.
pub(crate) fn born(at: usize, older: *const *const (), newer: *const *const ()) {
    with_live(|l| l.insert(at, [older as usize, newer as usize]));
}

/// The object at `at` goes.
pub(crate) fn gone(at: usize) {
    with_live(|l| l.remove(&at));
}

/// Whether `p` points at readable memory whose object `is` one of ours
/// (checked without reading memory that is not there).
#[cfg(windows)]
pub(crate) fn names(p: u64, is: impl FnOnce(This) -> bool) -> bool {
    p != 0 && p % 8 == 0 && crate::host::read_into(p as *const u8, &mut [0u8; 16]) && is(p as This)
}

#[cfg(windows)]
fn fence(p: usize) -> bool {
    names(p as u64, crate::fence::is_fence)
}

/// The older queue's slot 16: `GetTimestampFrequency(&u64)`, or the newer's
/// `Signal(fence, value)`.
#[cfg(windows)]
pub(crate) extern "system" fn queue_16(this: This, a: usize, b: u64) -> crate::object::HRESULT {
    if fence(a) {
        shown(true, "a fence passed to queue slot 16");
        return crate::queue::signal(this, a as This, b);
    }
    crate::queue::timestamp_frequency(this, a as *mut u64)
}

/// The older queue's slot 17: `GetClockCalibration(&gpu, &cpu)`, or the
/// newer's `Wait(fence, value)`.
#[cfg(windows)]
pub(crate) extern "system" fn queue_17(this: This, a: usize, b: u64) -> crate::object::HRESULT {
    if fence(a) {
        shown(true, "a fence passed to queue slot 17");
        return crate::queue::wait(this, a as This, b);
    }
    if a == 0 || b == 0 {
        return crate::object::E_INVALIDARG;
    }
    crate::queue::Written::GpuTimestamp.write(a as *mut u8, 8);
    crate::queue::Written::SystemTimestamp.write(b as *mut u8, 8);
    crate::object::S_OK
}

/// The newer queue's slots 14 and 15 (methods of the newer runtime nobody
/// has seen called), or the older's `Signal`/`Wait(fence, value)`.
#[cfg(windows)]
pub(crate) extern "system" fn queue_14(this: This, a: usize, b: u64) -> crate::object::HRESULT {
    if fence(a) {
        shown(false, "a fence passed to queue slot 14");
        return crate::queue::signal(this, a as This, b);
    }
    crate::object::E_NOTIMPL
}
#[cfg(windows)]
pub(crate) extern "system" fn queue_15(this: This, a: usize, b: u64) -> crate::object::HRESULT {
    if fence(a) {
        shown(false, "a fence passed to queue slot 15");
        return crate::queue::wait(this, a as This, b);
    }
    crate::object::E_NOTIMPL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decision_switches_live_tables() {
        static OLD: [usize; 2] = [1, 2];
        static NEW: [usize; 2] = [3, 4];
        let mut object = [OLD.as_ptr() as usize, 0];
        let at = object.as_mut_ptr() as usize;
        born(at, OLD.as_ptr().cast(), NEW.as_ptr().cast());
        STATE.store(UNKNOWN, Ordering::Relaxed);
        shown(true, "test");
        assert!(newer() && decided());
        // SAFETY: the object above.
        assert_eq!(unsafe { *(at as *const usize) }, NEW.as_ptr() as usize);
        gone(at);
        shown(false, "contradiction kept");
        assert!(newer());
        STATE.store(UNKNOWN, Ordering::Relaxed);
        std::hint::black_box(&mut object);
    }
}
