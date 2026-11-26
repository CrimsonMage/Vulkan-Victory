//! `amd_fidelityfx_loader_xbox.dll`, the title's AMD FidelityFX loader
//! (Series S), stays the title's: mapped and bound as its other DLLs. Only
//! its `ffxDispatch` is ours, for the upscale: that runs the PC way, as a
//! linear scale (`victory_gpu::upscale`, recorded into Vulkan), because
//! the upscaler's programs ship only as ACO-compiled console code that
//! does not translate. Every other dispatch, and an upscale the scale
//! cannot take, goes to the title's own `ffxDispatch`.
//!
//! What it rests on is the FidelityFX API, not one title: the dispatch
//! type `FFX_API_DISPATCH_DESC_TYPE_UPSCALE` and `ffxDispatchDescUpscale`
//! as the API's header lays it out (`victory_gpu::upscale`). That layout is
//! the same in every upscaler the FidelityFX API carries (FSR 3.1 on, the
//! API's start, through FSR 4), so the gate is the upscaler DLL beside the
//! loader and its version (`VS_FIXEDFILEINFO`): another major version, or
//! no upscaler DLL, leaves the upscale to the title, loudly.

use std::ffi::c_void;

use crate::HleExport;

/// The DLL whose export this replaces.
pub(crate) const DLL: &str = "amd_fidelityfx_loader_xbox.dll";
/// The upscaler the loader loads, beside it.
#[cfg_attr(not(windows), allow(dead_code))]
const UPSCALER: &str = "amd_fidelityfx_upscaler_xbox.dll";

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
            && upscaler_known()
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

/// A version as `VS_FIXEDFILEINFO` gives it: major, minor, patch.
type Version = (u16, u16, u16);

/// Whether `ffxDispatchDescUpscale` of upscaler `v` has the layout
/// `victory_gpu::upscale` reads: FSR 3.1 (the FidelityFX API's first) to
/// FSR 4.
#[cfg_attr(not(windows), allow(dead_code))]
fn layout_known(v: Version) -> bool {
    matches!(v, (3, 1.., _) | (4, _, _))
}

/// The file version in a PE file's version resource (`VS_FIXEDFILEINFO`,
/// found by its signature).
#[cfg_attr(not(windows), allow(dead_code))]
fn file_version(bytes: &[u8]) -> Option<Version> {
    const SIGNATURE: [u8; 4] = 0xFEEF_04BDu32.to_le_bytes();
    let at = bytes.windows(4).position(|w| w == SIGNATURE)?;
    let word = |k: usize| -> Option<u32> {
        let b = bytes.get(at + k * 4..at + k * 4 + 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    // Signature, structure version, then the file version (MS, LS).
    let (ms, ls) = (word(2)?, word(3)?);
    Some(((ms >> 16) as u16, ms as u16, (ls >> 16) as u16))
}

/// Whether the title's upscaler is one whose upscale description has the
/// layout read (decided once, logged once).
#[cfg(windows)]
fn upscaler_known() -> bool {
    use std::sync::OnceLock;
    static KNOWN: OnceLock<bool> = OnceLock::new();
    *KNOWN.get_or_init(|| {
        let path = crate::runtime::image_named(DLL)
            .and_then(crate::runtime::image_path)
            .and_then(|p| p.parent().map(|d| d.join(UPSCALER)));
        let version = path.as_ref().and_then(|p| std::fs::read(p).ok()).as_deref().and_then(file_version);
        match version {
            Some(v) if layout_known(v) => {
                tracing::info!(
                    version = format_args!("{}.{}.{}", v.0, v.1, v.2),
                    "FidelityFX upscaler: its upscales run as a linear scale (ffxDispatchDescUpscale as the FidelityFX API lays it out)"
                );
                true
            }
            other => {
                tracing::error!(
                    upscaler = ?path,
                    version = ?other,
                    "FidelityFX upscaler of no known API version (FSR 3.1 to 4 are): its upscales are left to it, and its console-only programs do not run (the upscaled image stays black)"
                );
                false
            }
        }
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_resource_is_read() {
        let mut file = vec![0u8; 64];
        file.extend_from_slice(&0xFEEF_04BDu32.to_le_bytes());
        file.extend_from_slice(&0x0001_0000u32.to_le_bytes());
        file.extend_from_slice(&(3u32 << 16 | 1).to_le_bytes());
        file.extend_from_slice(&(5u32 << 16).to_le_bytes());
        assert_eq!(file_version(&file), Some((3, 1, 5)));
        assert_eq!(file_version(&file[..70]), None);
    }

    #[test]
    fn only_the_apis_upscalers_are_known() {
        assert!(layout_known((3, 1, 5)));
        assert!(layout_known((4, 0, 2)));
        assert!(!layout_known((3, 0, 3)));
        assert!(!layout_known((5, 0, 0)));
        assert!(!layout_known((1, 0, 2)));
    }
}
