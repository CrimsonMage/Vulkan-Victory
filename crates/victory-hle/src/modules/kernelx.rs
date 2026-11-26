//! `kernelx.dll`'s own exports: the console kernel's functions with no PC
//! function of the same name, each a thin mapping onto what victory already
//! has. The rest of kernelx is the PC's kernel32, kernelbase and advapi32,
//! our overrides first (`super::twins`).
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;

use windows_sys::Win32::Foundation::{HANDLE, LocalFree, SetLastError};
use windows_sys::Win32::System::Threading::{GetThreadDescription, SetThreadDescription};
use windows_sys::core::{BOOL, PCWSTR, PWSTR};

use super::xmem::{self, WorkingSet};
use crate::{HleModule, hle_export};

const ERROR_INVALID_PARAMETER: u32 = 87;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

fn fail(error: u32) -> BOOL {
    // SAFETY: a plain call.
    unsafe { SetLastError(error) };
    0
}

/// `TITLEMEMORYSTATUS`; the caller sets `length`.
#[repr(C)]
#[derive(Default)]
pub struct MemoryStatus {
    length: u32,
    reserved: u32,
    total: u64,
    available: u64,
    legacy_used: u64,
    legacy_peak: u64,
    legacy_available: u64,
    title_used: u64,
    title_available: u64,
    legacy_page_tables: u64,
    title_page_tables: u64,
}

/// The title's memory as `XMemGetWorkingSetStatistics` tells it: the
/// console's budget, and what the process has committed as used (up to the
/// budget). No legacy (Xbox One era) memory.
pub extern "system" fn TitleMemoryStatus(out: *mut MemoryStatus) -> BOOL {
    let size = size_of::<MemoryStatus>() as u32;
    // SAFETY: the title passes its structure, which starts with its size.
    if out.is_null() || unsafe { (*out).length } < size {
        return fail(ERROR_INVALID_PARAMETER);
    }
    let mut set = WorkingSet::default();
    xmem::XMemGetWorkingSetStatistics(0, &mut set);
    let total = set.title_limit as u64;
    let used = (set.title_used as u64).min(total);
    let status = MemoryStatus {
        length: size,
        total,
        available: total - used,
        title_used: used,
        title_available: total - used,
        ..Default::default()
    };
    // SAFETY: checked above: room for the structure.
    unsafe { out.write(status) };
    1
}

/// The same for the title's job, which holds only the title.
pub extern "system" fn JobTitleMemoryStatus(out: *mut MemoryStatus) -> BOOL {
    TitleMemoryStatus(out)
}

/// `TOOLINGMEMORYSTATUS`: a size, then four figures.
#[repr(C)]
pub struct ToolingStatus {
    length: u32,
    reserved: u32,
    figures: [u64; 4],
}

/// A retail console has no tools memory: all zero.
pub extern "system" fn ToolingMemoryStatus(out: *mut ToolingStatus) -> BOOL {
    // SAFETY: the title passes its structure, which starts with its size.
    if out.is_null() || unsafe { (*out).length } < size_of::<ToolingStatus>() as u32 {
        return fail(ERROR_INVALID_PARAMETER);
    }
    // SAFETY: checked above: room for the structure.
    unsafe { (*out).figures = [0; 4] };
    1
}

/// Physical pages are XMem's (64 KiB, numbers opaque to the title).
pub extern "system" fn AllocateTitlePhysicalPages(
    _process: HANDLE,
    flags: u32,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    xmem::XMemAllocatePhysicalPages(flags.into(), count, pages)
}

pub extern "system" fn FreeTitlePhysicalPages(
    _process: HANDLE,
    count: usize,
    pages: *const usize,
) -> BOOL {
    xmem::XMemFreePhysicalPages(count, pages)
}

/// Memory type and protection are those of the reservation (all of it is
/// reached by both processors here).
pub extern "system" fn MapTitlePhysicalPages(
    address: *mut c_void,
    count: usize,
    _kind: u32,
    _protect: u32,
    pages: *const usize,
) -> *mut c_void {
    xmem::XMemMapPhysicalPages(address, count, pages)
}

/// The PC's thread description.
pub extern "system" fn SetThreadName(thread: HANDLE, name: PCWSTR) -> BOOL {
    // SAFETY: the title's own arguments.
    let hr = unsafe { SetThreadDescription(thread, name) };
    if hr < 0 { fail(hr as u32 & 0xffff) } else { 1 }
}

/// The PC's thread description, copied with its NUL into `size`
/// characters; `used` gets its length without the NUL.
pub extern "system" fn GetThreadName(
    thread: HANDLE,
    out: *mut u16,
    size: usize,
    used: *mut usize,
) -> BOOL {
    let mut name: PWSTR = std::ptr::null_mut();
    // SAFETY: the title's thread handle; the host allocates the text.
    let hr = unsafe { GetThreadDescription(thread, &mut name) };
    if hr < 0 || name.is_null() {
        return fail(ERROR_INVALID_PARAMETER);
    }
    let text = super::xgameruntime::wide_text(name).unwrap_or_default();
    // SAFETY: the host's allocation, ours to free.
    unsafe { LocalFree(name.cast()) };
    let text: Vec<u16> = text.encode_utf16().collect();
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(text.len()) };
    }
    if out.is_null() || size <= text.len() {
        return fail(ERROR_INSUFFICIENT_BUFFER);
    }
    // SAFETY: the title's buffer holds `size` characters, more than the text.
    unsafe {
        out.copy_from_nonoverlapping(text.as_ptr(), text.len());
        out.add(text.len()).write(0);
    }
    1
}

/// Thread pools run where the PC schedules them.
pub extern "system" fn SetThreadpoolAffinityMask(_pool: *mut c_void, _mask: usize) -> BOOL {
    1
}

/// Running, idle and global time: not counted here (zeros).
pub extern "system" fn QueryProcessorSchedulingStatistics(out: *mut [u64; 3]) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for three counts.
        unsafe { out.write([0; 3]) };
    }
}

/// The console model the user chose (`VICTORY_CONSOLE`).
// shortcut: the Series values (5 Series S, 6 Series X, after the Xbox One
// family's 1-4) are not published; correct them if a title tells.
pub extern "system" fn GetConsoleType() -> u32 {
    match victory_gpu::console::current() {
        victory_gpu::console::Console::SeriesS => 5,
        victory_gpu::console::Console::SeriesX => 6,
    }
}

/// `SYSTEMOSVERSIONINFO`.
#[repr(C)]
pub struct OsVersion {
    major: u8,
    minor: u8,
    build: u16,
    revision: u16,
}

/// The console OS version XSystem's analytics report.
pub extern "system" fn GetSystemOSVersion(out: *mut OsVersion) {
    let os = super::xsystem::OS;
    let version = OsVersion {
        major: os.major as u8,
        minor: os.minor as u8,
        build: os.build,
        revision: os.revision,
    };
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for the structure.
        unsafe { out.write(version) };
    }
}

/// A DLL of the title's package, by its path in the package: the package
/// root is the title's folder, where `LoadLibraryW`'s title DLL loader looks.
pub extern "system" fn LoadPackagedLibrary(name: PCWSTR, _reserved: u32) -> *mut c_void {
    super::kernel::LoadLibraryW(name)
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "kernelx",
        dlls: &["kernelx.dll"],
        exports: vec![
            hle_export!(TitleMemoryStatus),
            hle_export!(JobTitleMemoryStatus),
            hle_export!(ToolingMemoryStatus),
            hle_export!(AllocateTitlePhysicalPages),
            hle_export!(FreeTitlePhysicalPages),
            hle_export!(MapTitlePhysicalPages),
            hle_export!(SetThreadName),
            hle_export!(GetThreadName),
            hle_export!(SetThreadpoolAffinityMask),
            hle_export!(QueryProcessorSchedulingStatistics),
            hle_export!(GetConsoleType),
            hle_export!(GetSystemOSVersion),
            hle_export!(LoadPackagedLibrary),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_status_fits_the_budget() {
        let mut status = MemoryStatus {
            length: size_of::<MemoryStatus>() as u32,
            ..Default::default()
        };
        assert_eq!(TitleMemoryStatus(&mut status), 1);
        assert!(status.total > 0);
        assert_eq!(status.available + status.title_used, status.total);
        let mut short = MemoryStatus::default();
        assert_eq!(TitleMemoryStatus(&mut short), 0);
    }

    #[test]
    fn thread_names_go_through_the_pc() {
        // SAFETY: the current thread's pseudo handle.
        let me = unsafe { windows_sys::Win32::System::Threading::GetCurrentThread() };
        let name: Vec<u16> = "worker".encode_utf16().chain([0]).collect();
        assert_eq!(SetThreadName(me, name.as_ptr()), 1);
        let (mut out, mut used) = ([0u16; 16], 0);
        assert_eq!(GetThreadName(me, out.as_mut_ptr(), out.len(), &mut used), 1);
        assert_eq!(&out[..used + 1], &name[..]);
        assert_eq!(GetThreadName(me, out.as_mut_ptr(), 3, &mut used), 0);
    }
}
