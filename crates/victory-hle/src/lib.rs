//! Our implementation of the Xbox system API surface.
//!
//! A title imports from three kinds of DLLs, and each gets its own policy:
//!
//! | kind | examples | handled by |
//! |------|----------|------------|
//! | Windows API sets the console shares with PC | `api-ms-win-core-synch-*`, `ws2_32.dll` | host pass-through, overridden here where console behaviour differs (paths, memory) |
//! | console-only system APIs | `xgameruntime.dll`, `xmem.dll`, `d3d12_xs.dll` | implemented here, in [`modules`] |
//! | console DLLs that are PC DLLs under another name | `kernelx.dll` | looked up as its PC twins (`modules::twins`) |
//! | anything else | | an unimplemented-import stub from the loader |
//!
//! Resolution order for every import: an implementation in [`modules`],
//! then the host if [`HostPolicy`] allows that DLL, then `Missing`.

mod host;
pub mod modules;
mod registry;
pub mod runtime;

pub use host::HostPolicy;
pub use registry::{HleExport, HleModule, registry};
pub use victory_gpu::console;

use victory_loader::{ImportResolver, Provenance, Resolution};
use victory_pe::ImportSymbol;

/// Resolves a title's imports against our runtime and the host.
#[derive(Debug, Default)]
pub struct HleResolver {
    pub host: HostPolicy,
}

impl HleResolver {
    pub fn new(host: HostPolicy) -> Self {
        HleResolver { host }
    }

    /// Where an import would come from, without binding anything.
    pub fn classify(&self, dll: &str, symbol: &ImportSymbol) -> Option<Provenance> {
        match self.resolve(dll, symbol) {
            Resolution::Found { provenance, .. } => Some(provenance),
            Resolution::Missing => None,
        }
    }
}

impl ImportResolver for HleResolver {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        // Ordinal imports look up `#n`. A title DLL's export we replace
        // comes before the DLL's own (`WithTitleDlls` asks us first).
        let name = symbol.to_string();
        // A console DLL that is PC DLLs under another name (`kernelx.dll`)
        // is looked up under those too; by name only, as ordinals differ.
        let twins = match symbol {
            ImportSymbol::Name { .. } => modules::twins(dll),
            ImportSymbol::Ordinal(_) => &[],
        };
        let dlls = || std::iter::once(dll).chain(twins.iter().copied());
        if let Some(address) = dlls()
            .find_map(|d| registry::lookup(d, &name))
            .or_else(|| registry::title_override(dll, &name))
        {
            return Resolution::Found {
                address,
                provenance: Provenance::Hle,
            };
        }
        match dlls().find_map(|d| self.host.resolve(d, symbol)) {
            Some(address) => Resolution::Found {
                address,
                provenance: Provenance::Host,
            },
            None => Resolution::Missing,
        }
    }
}

// The registry is code-only, so the "Missing" outcome can be checked on any
// host; host pass-through itself needs Windows.
#[cfg(test)]
mod tests {
    use super::*;

    fn name(n: &str) -> ImportSymbol {
        ImportSymbol::Name {
            hint: 0,
            name: n.into(),
        }
    }

    #[test]
    fn console_api_resolves_to_hle() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("XGameRuntime.dll", &name("XGameRuntimeInitialize")),
            Some(Provenance::Hle)
        );
    }

    #[test]
    fn strict_policy_reports_missing() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("api-ms-win-core-synch-l1-1-0.dll", &name("Sleep")),
            None
        );
    }

    #[test]
    fn shared_api_sets_pass_through_to_host() {
        let r = HleResolver::default();
        assert_eq!(
            r.classify("api-ms-win-core-synch-l1-2-0.dll", &name("Sleep")),
            Some(Provenance::Host)
        );
        // Console-only DLLs never pass through.
        assert_eq!(
            r.classify("xmem.dll", &name("XMemSetAllocationHooks")),
            None
        );
    }

    #[test]
    fn kernelx_imports_go_to_our_modules_then_the_pc_kernel() {
        let r = HleResolver::default();
        let kernelx = |n: &str| r.classify("kernelx.dll", &name(n));
        assert_eq!(kernelx("GetModuleHandleW"), Some(Provenance::Hle));
        assert_eq!(kernelx("CreateFileW"), Some(Provenance::Hle));
        assert_eq!(kernelx("XMemAlloc"), Some(Provenance::Hle));
        assert_eq!(
            kernelx("RegisterAppStateChangeNotification"),
            Some(Provenance::Hle)
        );
        assert_eq!(kernelx("Sleep"), Some(Provenance::Host));
        assert_eq!(kernelx("RegOpenKeyExW"), Some(Provenance::Hle));
        assert_eq!(kernelx("NotAnExport"), None);
        // No pass-through, no host twin.
        let strict = HleResolver::new(HostPolicy::none());
        assert_eq!(strict.classify("kernelx.dll", &name("Sleep")), None);
    }

    #[test]
    fn overrides_win_over_host() {
        let r = HleResolver::default();
        assert_eq!(
            r.classify(
                "api-ms-win-core-libraryloader-l1-2-0.dll",
                &name("GetModuleHandleW")
            ),
            Some(Provenance::Hle)
        );
    }
}
