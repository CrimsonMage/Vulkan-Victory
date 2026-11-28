//! `amd_fidelityfx_loader_xbox.dll`, the title's AMD FidelityFX loader
//! (Series S), stays the title's: mapped and bound as its other DLLs. Only
//! its `ffxDispatch` is ours, for the upscale. The upscaler's passes are
//! console compute programs of another compiler (`ACCC` containers in its
//! DLL). The upscale runs the PC way as a linear scale
//! (`victory_gpu::upscale`, recorded into Vulkan). With `VICTORY_FFX_OWN=1`
//! and every pass translating (and none failed on the card,
//! `victory_gpu::upscale::pass_failed`), the upscale is the title's own
//! instead: its `ffxDispatch` runs and the passes go the way of every
//! compute stage. That is not the default while they are too slow and come
//! out black (see `decide`). Every other
//! dispatch, and an upscale the scale cannot take, goes to the title's own
//! `ffxDispatch`.
//!
//! What the stand-in rests on is the FidelityFX API, not one title: the
//! dispatch type `FFX_API_DISPATCH_DESC_TYPE_UPSCALE` and
//! `ffxDispatchDescUpscale` as the API's header lays it out
//! (`victory_gpu::upscale`). That layout is the same in every upscaler the
//! FidelityFX API carries (FSR 3.1 on, the API's start, through FSR 4), so
//! its gate is the upscaler DLL beside the loader and its version
//! (`VS_FIXEDFILEINFO`): another major version, or no upscaler DLL, leaves
//! the upscale to the title, loudly.

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
            && stands_in(upscaler(), victory_gpu::upscale::pass_failed())
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

/// Who runs the title's upscales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(windows), allow(dead_code))]
enum Upscaler {
    /// The title's own upscaler: every pass of it translates.
    Own,
    /// The linear scale: a pass does not translate, and the upscale
    /// description has the layout read.
    Scale,
    /// The title, for want of anything else: an upscaler of no known API
    /// version (or none).
    Unknown,
}

/// Whether the linear scale takes an upscale: the upscaler's passes do not
/// all translate, or one of them failed on the card (`pass_failed`).
#[cfg_attr(not(windows), allow(dead_code))]
fn stands_in(upscaler: Upscaler, pass_failed: bool) -> bool {
    match upscaler {
        Upscaler::Own => pass_failed,
        Upscaler::Scale => true,
        Upscaler::Unknown => false,
    }
}

/// Who runs the upscales of the upscaler whose file holds `bytes`.
#[cfg_attr(not(windows), allow(dead_code))]
///
/// `own` is the switch `VICTORY_FFX_OWN`: without it the linear scale runs
/// even when every pass translates. Measured 2026-11-27 (FSR 4, 29 passes,
/// RTX 5060 Ti): the translated passes take 3 to 9 s a frame, one of them
/// runs into its loop guard, and the picture comes out black.
#[cfg_attr(not(windows), allow(dead_code))]
fn decide(bytes: Option<&[u8]>, own: bool) -> (Upscaler, String) {
    let Some(v) = bytes.and_then(file_version).filter(|&v| layout_known(v)) else {
        return (Upscaler::Unknown, format!("version {:?}", bytes.and_then(file_version)));
    };
    let version = format!("{}.{}.{}", v.0, v.1, v.2);
    match victory_gpu::upscale::passes_translate(bytes.unwrap_or_default()) {
        Ok(passes) if own => (Upscaler::Own, format!("version {version}, {passes} passes")),
        Ok(passes) => (
            Upscaler::Scale,
            format!("version {version}; its {passes} passes translate but are not run (VICTORY_FFX_OWN=1 runs them)"),
        ),
        Err(why) => (Upscaler::Scale, format!("version {version}; {why}")),
    }
}

/// Who runs the title's upscales (decided once, logged once).
#[cfg(windows)]
fn upscaler() -> Upscaler {
    use std::sync::OnceLock;
    static KNOWN: OnceLock<Upscaler> = OnceLock::new();
    *KNOWN.get_or_init(|| {
        let path = crate::runtime::image_named(DLL)
            .and_then(crate::runtime::image_path)
            .and_then(|p| p.parent().map(|d| d.join(UPSCALER)));
        let bytes = path.as_ref().and_then(|p| std::fs::read(p).ok());
        let (who, what) = decide(bytes.as_deref(), std::env::var_os("VICTORY_FFX_OWN").is_some_and(|v| v == "1"));
        match who {
            Upscaler::Own => tracing::info!(what, "FidelityFX upscaler: every pass translates, its upscales are its own"),
            Upscaler::Scale => tracing::warn!(
                what,
                "FidelityFX upscaler: its upscales run as a linear scale (ffxDispatchDescUpscale as the FidelityFX API lays it out)"
            ),
            Upscaler::Unknown => tracing::error!(
                upscaler = ?path,
                what,
                "FidelityFX upscaler of no known API version (FSR 3.1 to 4 are): its upscales are left to it"
            ),
        }
        who
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

    /// The scale stands in while a pass has no translation or failed on
    /// the card, never for an upscaler whose description is not known.
    #[test]
    fn the_scale_stands_in_only_without_the_upscalers_own_passes() {
        assert!(!stands_in(Upscaler::Own, false));
        assert!(stands_in(Upscaler::Own, true));
        assert!(stands_in(Upscaler::Scale, false));
        assert!(!stands_in(Upscaler::Unknown, true));
        // no file, a file of no version, a file of a known version without passes
        assert_eq!(decide(None, true).0, Upscaler::Unknown);
        assert_eq!(decide(Some(&[0; 64]), true).0, Upscaler::Unknown);
        let mut file = vec![0u8; 64];
        file.extend_from_slice(&0xFEEF_04BDu32.to_le_bytes());
        file.extend_from_slice(&0x0001_0000u32.to_le_bytes());
        file.extend_from_slice(&(4u32 << 16).to_le_bytes());
        file.extend_from_slice(&(2u32 << 16).to_le_bytes());
        let (who, what) = decide(Some(&file), true);
        assert_eq!(who, Upscaler::Scale);
        assert!(what.contains("no pass found"), "{what}");
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
