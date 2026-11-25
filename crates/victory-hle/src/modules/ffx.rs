//! `amd_fidelityfx_loader_xbox.dll`, the title's AMD FidelityFX loader
//! (Series S), stays the title's: mapped and bound as its other DLLs. Only
//! its `ffxDispatch` is ours, for the upscale: that runs the PC way, as a
//! linear scale (`victory_gpu::upscale`), because the upscaler's programs
//! ship only as ACO-compiled console code that does not translate. Every
//! other dispatch, and an upscale the scale cannot take, goes to the
//! title's own `ffxDispatch`.

use std::ffi::c_void;

use crate::HleExport;

/// The DLL whose export this replaces.
pub(crate) const DLL: &str = "amd_fidelityfx_loader_xbox.dll";

/// `FFX_API_RETURN_OK`, `FFX_API_RETURN_ERROR_RUNTIME_ERROR`.
const OK: u32 = 0;
#[cfg_attr(not(windows), allow(dead_code))]
const RUNTIME_ERROR: u32 = 3;

#[cfg(windows)]
type Dispatch = extern "system" fn(*mut c_void, *const c_void) -> u32;

/// `ffxDispatch(context, desc)`; `desc` starts with its type (`u64`) and
/// the next description in its chain.
#[allow(non_snake_case)]
extern "system" fn ffxDispatch(context: *mut c_void, desc: *const c_void) -> u32 {
    #[cfg(windows)]
    {
        // SAFETY: the title passes a description, which starts with its type.
        let kind = unsafe { desc.cast::<u64>().as_ref() }.copied();
        // SAFETY: an upscale description, as the title passes it.
        if kind == Some(victory_gpu::upscale::UPSCALE)
            && unsafe { victory_gpu::upscale::dispatch(desc.cast()) }
        {
            return OK;
        }
        match title_dispatch() {
            Some(dispatch) => dispatch(context, desc),
            None => RUNTIME_ERROR,
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (context, desc);
        OK
    }
}

/// The title's own `ffxDispatch`, in its loader DLL.
#[cfg(windows)]
fn title_dispatch() -> Option<Dispatch> {
    use std::sync::OnceLock;
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    static FOUND: OnceLock<Option<usize>> = OnceLock::new();
    let found = *FOUND.get_or_init(|| {
        let base = crate::runtime::image_named(DLL)?;
        // SAFETY: a title image we mapped is a host module; the name is
        // NUL-terminated.
        let f = unsafe { GetProcAddress(base as _, c"ffxDispatch".as_ptr().cast()) };
        if f.is_none() {
            tracing::warn!("the title's FidelityFX loader has no ffxDispatch");
        }
        f.map(|f| f as usize)
    });
    // SAFETY: the export's address, of this signature.
    found.map(|f| unsafe { std::mem::transmute::<usize, Dispatch>(f) })
}

/// The exports of title DLLs this replaces: (DLL, export).
pub(crate) fn overrides() -> Vec<(&'static str, HleExport)> {
    vec![(
        DLL,
        HleExport {
            name: "ffxDispatch",
            address: ffxDispatch as *const () as usize,
        },
    )]
}
