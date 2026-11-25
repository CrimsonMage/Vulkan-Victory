//! Implementations, one module per area of the console API.
//!
//! Adding an API: write an `extern "system"` function named exactly like
//! the import, then list it with `hle_export!` in the module's `module()`.
//! Run `victory census` over your titles to see what is still missing.

pub mod apu;
pub mod audio;
pub mod d3d12x;
pub mod display;
pub mod dsp;
pub mod dstorage;
pub mod ffx;
pub mod files;
pub mod gameinput;
pub mod kernel;
pub mod lifecycle;
pub mod media;
pub mod net;
pub mod speech;
pub mod storage;
pub mod xasync;
pub mod xg;
pub mod xgame;
pub mod xgameruntime;
pub mod xgamesave;
pub mod xgameui;
pub mod xmem;
pub mod xnetworking;
pub mod xpackage;
pub mod xstore;
pub mod xsystem;
pub mod xuser;

use crate::HleModule;

pub(crate) fn all() -> Vec<HleModule> {
    vec![
        kernel::module(),
        xgameruntime::module(),
        gameinput::module(),
        d3d12x::module(),
        audio::module(),
        dsp::module(),
        xg::module(),
        apu::module(),
        media::module(),
        dstorage::module(),
        files::module(),
        lifecycle::module(),
        net::module(),
        xmem::module(),
        xsystem::module(),
    ]
}

/// Single exports of title DLLs we replace, the DLL itself staying the
/// title's (mapped and bound like its other DLLs): (DLL, export).
pub(crate) fn title_overrides() -> Vec<(&'static str, crate::HleExport)> {
    ffx::overrides()
}
