//! `XMemAlloc` and `XMemFree`: the title's general allocator, for
//! middleware too, through the title's hooks if it has any.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{XMemAllocDefault, XMemFreeDefault};

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The routines `XMemSetAllocationHooks` set, (0, 0) for none.
static SET: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

/// Replaces the routines behind `XMemAlloc` and `XMemFree`; a null
/// allocation routine goes back to the defaults.
pub extern "system" fn XMemSetAllocationHooks(alloc: usize, free: usize) {
    let free = if alloc == 0 { 0 } else { free };
    tracing::info!(set = alloc != 0, "XMemSetAllocationHooks");
    SET[1].store(free, Ordering::Release);
    SET[0].store(alloc, Ordering::Release);
}

/// The routines set by `XMemSetAllocationHooks`, else those the title's exe
/// exports as `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    let set = SET.each_ref().map(|hook| hook.load(Ordering::Acquire));
    if set[0] != 0 {
        return set;
    }
    static AT: OnceLock<usize> = OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator: its hook if it has one, else
/// [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}
