//! Xbox Series shaders to SPIR-V.
//!
//! An Xbox Series shader is a standard DXBC container ([`dxbc`]). Its `DXIL`
//! part ([`dxil`]) may carry Xbox additions around the LLVM bitcode: an
//! `XDAT` table before it indexing blobs after it, among them `xbop`
//! (compiler options) and the zlib-compressed GFX10 code. The bitcode may use
//! console-only intrinsics, so the GFX10 code is translated in general:
//! decoded ([`gfx10`], [`native`]), lifted into SSA ([`ir`], [`spill`],
//! [`io`]) and emitted as SPIR-V for the Vulkan host ([`vk`]). A container
//! of another compiler has its GFX10 code in an `ACCC` part instead (AMD's
//! FidelityFX upscaler; `native/fspx.rs`) and goes the same way.

pub mod dxbc;
pub mod dxil;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
pub mod sampler;
pub mod spill;
pub mod vk;
/// Whole-wave sections (the compiler's wave reductions and prefix sums),
/// found by simulation; `vk/lift_wave.rs` lowers what they leave.
#[allow(dead_code)]
pub mod wave;

pub use dxbc::{Container, FourCC, Part, containers};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs,
};

/// Everything a translation depends on besides the container: the
/// translator's own source (hashed at build time) and the value of every
/// `VICTORY_*` switch it reads (build.rs lists them). The stage cache keys
/// on this instead of on the whole executable, so a rebuild that leaves the
/// translator alone keeps its translations.
pub fn translation_fingerprint() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(env!("VICTORY_SHADER_SOURCE_HASH").as_bytes());
    out.push(u8::from(cfg!(debug_assertions)));
    for name in env!("VICTORY_SHADER_SWITCHES").split(',').filter(|n| !n.is_empty()) {
        out.extend_from_slice(name.as_bytes());
        out.push(b'=');
        if let Some(value) = std::env::var_os(name) {
            out.extend_from_slice(value.as_encoded_bytes());
        }
        out.push(0);
    }
    out
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShaderError {
    #[error("not a DXBC container")]
    NotDxbc,
    #[error("malformed shader: {0}")]
    Malformed(&'static str),
    #[error("GFX10 code at {offset:#x}: {reason}")]
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    #[error("GFX10 code at {offset:#x}: {what} is not supported")]
    Unsupported { offset: usize, what: &'static str },
}

#[cfg(test)]
mod fingerprint_tests {
    include!("../fingerprint.rs");

    /// Every file or folder the fingerprint leaves out exists (a rename
    /// would silently put it back in).
    #[test]
    fn fingerprint_exclusions_exist() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for x in FINGERPRINT_EXCLUDE {
            assert!(src.join(x).exists(), "{x} is in fingerprint.rs but not in src");
        }
        for (file, line) in FINGERPRINT_EXCLUDE_LINES {
            let text = std::fs::read_to_string(src.join(file)).unwrap_or_default();
            assert!(text.lines().any(|l| l == *line), "{file} has no line `{line}` (fingerprint.rs)");
        }
    }
}
