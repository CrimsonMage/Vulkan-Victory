//! Microsoft's DXC (`dxcompiler.dll`, `libdxcompiler.so`) loaded at run time,
//! `IDxcCompiler3::Compile` through its COM interface.
//!
//! Found at `VICTORY_DXC` (a file) if set, else beside the executable, else
//! wherever the loader finds it. DXC signs its containers with `dxil.dll`
//! (`libdxil.so`) from the same place; an unsigned container is refused, as
//! Windows' D3D12 would refuse it.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

type Hresult = i32;

#[repr(C)]
struct Guid(u32, u16, u16, [u8; 8]);

const CLSID_DXC_COMPILER: Guid = Guid(0x73e2_2d93, 0xe6ce, 0x47f3, [0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0]);
const IID_IDXC_COMPILER3: Guid = Guid(0x228b_4687, 0x5a6a, 0x4730, [0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54]);
const IID_IDXC_RESULT: Guid = Guid(0x5834_6cda, 0xdde7, 0x4497, [0x94, 0x61, 0x6f, 0x87, 0xaf, 0x5e, 0x06, 0x59]);
const IID_IDXC_VERSION2: Guid = Guid(0xfb69_04c4, 0x42f0, 0x4b62, [0x9c, 0x46, 0x98, 0x3a, 0xf7, 0xda, 0x7c, 0x83]);

/// LPCWSTR: UTF-16 on Windows, `wchar_t` (32 bits) in DXC's Linux build.
#[cfg(windows)]
type WChar = u16;
#[cfg(not(windows))]
type WChar = u32;

fn wide(s: &str) -> Vec<WChar> {
    #[cfg(windows)]
    return s.encode_utf16().chain([0]).collect();
    #[cfg(not(windows))]
    return s.chars().map(|c| c as u32).chain([0]).collect();
}

#[cfg(windows)]
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
#[cfg(not(windows))]
unsafe extern "C" {
    fn dlopen(name: *const std::ffi::c_char, flags: i32) -> *mut c_void;
    fn dlsym(module: *mut c_void, name: *const std::ffi::c_char) -> *mut c_void;
}

type CreateInstance = unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> Hresult;
type Object = *mut *const usize;

unsafe fn method<F: Copy>(obj: Object, slot: usize) -> F {
    // SAFETY: the caller names a slot of the object's interface.
    unsafe { std::mem::transmute_copy(&*(*obj).add(slot)) }
}

unsafe fn release(obj: Object) {
    if !obj.is_null() {
        // SAFETY: slot 2 of every COM interface is Release.
        unsafe { method::<unsafe extern "system" fn(Object) -> u32>(obj, 2)(obj) };
    }
}

/// The bytes of an `IDxcBlob` (GetBufferPointer 3, GetBufferSize 4).
unsafe fn blob_bytes(blob: Object) -> Vec<u8> {
    // SAFETY: `blob` is a live IDxcBlob.
    unsafe {
        let ptr: unsafe extern "system" fn(Object) -> *const u8 = method(blob, 3);
        let len: unsafe extern "system" fn(Object) -> usize = method(blob, 4);
        let (p, n) = (ptr(blob), len(blob));
        if p.is_null() { Vec::new() } else { std::slice::from_raw_parts(p, n).to_vec() }
    }
}

#[repr(C)]
struct DxcBuffer {
    ptr: *const c_void,
    size: usize,
    encoding: u32,
}

/// A loaded DXC: its `DxcCreateInstance`.
pub struct Dxc {
    create: CreateInstance,
    version: String,
}

// SAFETY: `DxcCreateInstance` is free-threaded; compilers are made per thread.
unsafe impl Send for Dxc {}
unsafe impl Sync for Dxc {}

static OK: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static MICROS: AtomicU64 = AtomicU64::new(0);

/// Counts a run-time translation through this backend.
pub(super) fn note(ok: bool, took: std::time::Duration) {
    if ok { &OK } else { &FAILED }.fetch_add(1, Ordering::Relaxed);
    MICROS.fetch_add(took.as_micros() as u64, Ordering::Relaxed);
}

/// Shaders this backend translated at run time, those it handed to the
/// bitcode backend, and the time spent in both (translation and DXC), in
/// microseconds.
pub fn stats() -> (u64, u64, u64) {
    (OK.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), MICROS.load(Ordering::Relaxed))
}

/// The DXC the run uses, loaded once (`None` when there is none: every
/// shader goes to the bitcode backend).
pub fn shared() -> Option<&'static Dxc> {
    static DXC: std::sync::OnceLock<Option<Dxc>> = std::sync::OnceLock::new();
    DXC.get_or_init(|| match Dxc::load(default_path().as_deref()) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("HLSL backend: no DXC ({e}); the bitcode backend translates every shader");
            None
        }
    })
    .as_ref()
}

/// Where DXC is looked for: `VICTORY_DXC`, else beside the executable.
pub fn default_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VICTORY_DXC") {
        return Some(p.into());
    }
    let name = if cfg!(windows) { "dxcompiler.dll" } else { "libdxcompiler.so" };
    let beside = std::env::current_exe().ok()?.parent()?.join(name);
    beside.exists().then_some(beside)
}

impl Dxc {
    /// Loads `path` (or the default library name from the loader's path).
    pub fn load(path: Option<&std::path::Path>) -> Result<Self, String> {
        #[cfg(not(windows))]
        let create = {
            let name = path.map_or_else(|| "libdxcompiler.so".into(), |p| p.to_string_lossy().into_owned());
            let c = std::ffi::CString::new(name.clone()).map_err(|e| e.to_string())?;
            // SAFETY: dlopen/dlsym with NUL-terminated names (2 = RTLD_NOW).
            unsafe {
                let module = dlopen(c.as_ptr(), 2);
                if module.is_null() {
                    return Err(format!("cannot load {name}"));
                }
                let f = dlsym(module, c"DxcCreateInstance".as_ptr());
                if f.is_null() {
                    return Err(format!("{name} lacks DxcCreateInstance"));
                }
                f
            }
        };
        #[cfg(windows)]
        let create = {
            let name: Vec<u16> = path
                .map_or_else(|| "dxcompiler.dll".into(), |p| p.as_os_str().to_owned())
                .to_string_lossy()
                .encode_utf16()
                .chain([0])
                .collect();
            // SAFETY: plain Win32 calls with a NUL-terminated name.
            unsafe {
                let module = LoadLibraryW(name.as_ptr());
                if module.is_null() {
                    return Err(format!("cannot load {}", String::from_utf16_lossy(&name)));
                }
                GetProcAddress(module, c"DxcCreateInstance".as_ptr().cast()).ok_or("dxcompiler.dll lacks DxcCreateInstance")?
                    as *mut c_void
            }
        };
        // SAFETY: the export has this signature.
        let create: CreateInstance = unsafe { std::mem::transmute(create) };
        let mut d = Dxc { create, version: String::new() };
        d.version = d.query_version();
        Ok(d)
    }

    /// `major.minor commits-hash` of the compiler.
    pub fn version(&self) -> &str {
        &self.version
    }

    fn compiler(&self) -> Result<Object, String> {
        let mut object = std::ptr::null_mut();
        // SAFETY: valid GUIDs and out pointer.
        let hr = unsafe { (self.create)(&CLSID_DXC_COMPILER, &IID_IDXC_COMPILER3, &mut object) };
        if hr < 0 || object.is_null() {
            return Err(format!("DxcCreateInstance(compiler) failed: {hr:#x}"));
        }
        Ok(object.cast())
    }

    fn query_version(&self) -> String {
        let Ok(c) = self.compiler() else { return String::new() };
        // SAFETY: QueryInterface (0) for IDxcVersionInfo2; GetVersion (3),
        // GetCommitInfo (5); the commit hash is CoTaskMem memory we leak (once).
        unsafe {
            let mut v: Object = std::ptr::null_mut();
            let qi: unsafe extern "system" fn(Object, *const Guid, *mut Object) -> Hresult = method(c, 0);
            let mut out = String::new();
            if qi(c, &IID_IDXC_VERSION2, &mut v) >= 0 && !v.is_null() {
                let (mut major, mut minor, mut commits) = (0u32, 0u32, 0u32);
                let get: unsafe extern "system" fn(Object, *mut u32, *mut u32) -> Hresult = method(v, 3);
                get(v, &mut major, &mut minor);
                let mut hash: *const std::ffi::c_char = std::ptr::null();
                let commit: unsafe extern "system" fn(Object, *mut u32, *mut *const std::ffi::c_char) -> Hresult =
                    method(v, 5);
                commit(v, &mut commits, &mut hash);
                out = format!("{major}.{minor} {commits}");
                if !hash.is_null() {
                    out.push('-');
                    out.push_str(&std::ffi::CStr::from_ptr(hash).to_string_lossy());
                }
                release(v);
            }
            release(c);
            out
        }
    }

    /// Compiles `source` (entry `main`) for `profile`: the signed DXIL
    /// container, or DXC's messages.
    pub fn compile(&self, source: &str, profile: &str) -> Result<Vec<u8>, String> {
        thread_local! {
            static COMPILER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        }
        let c = COMPILER.with(|slot| -> Result<Object, String> {
            if slot.get() == 0 {
                slot.set(self.compiler()? as usize);
            }
            Ok(slot.get() as Object)
        })?;
        let args: Vec<Vec<WChar>> = compile_args(profile).iter().map(|a| wide(a)).collect();
        let ptrs: Vec<*const WChar> = args.iter().map(|a| a.as_ptr()).collect();
        let buffer = DxcBuffer {
            ptr: source.as_ptr().cast(),
            size: source.len(),
            encoding: 65001, // DXC_CP_UTF8
        };
        // SAFETY: IDxcCompiler3::Compile (3) with live buffers; IDxcResult
        // GetStatus (3), GetResult (4), GetErrorBuffer (5).
        unsafe {
            let compile: unsafe extern "system" fn(
                Object,
                *const DxcBuffer,
                *const *const WChar,
                u32,
                *mut c_void,
                *const Guid,
                *mut Object,
            ) -> Hresult = method(c, 3);
            let mut result: Object = std::ptr::null_mut();
            let hr = compile(c, &buffer, ptrs.as_ptr(), ptrs.len() as u32, std::ptr::null_mut(), &IID_IDXC_RESULT, &mut result);
            if hr < 0 || result.is_null() {
                return Err(format!("Compile failed: {hr:#x}"));
            }
            let mut status = 0;
            let get_status: unsafe extern "system" fn(Object, *mut Hresult) -> Hresult = method(result, 3);
            get_status(result, &mut status);
            let out = if status >= 0 {
                let mut blob: Object = std::ptr::null_mut();
                let get: unsafe extern "system" fn(Object, *mut Object) -> Hresult = method(result, 4);
                get(result, &mut blob);
                let bytes = if blob.is_null() { Vec::new() } else { blob_bytes(blob) };
                release(blob);
                if bytes.len() < 20 || bytes[4..20].iter().all(|&b| b == 0) {
                    Err("DXC left the container unsigned (no dxil.dll beside it?)".to_owned())
                } else {
                    Ok(bytes)
                }
            } else {
                let mut errors: Object = std::ptr::null_mut();
                let get: unsafe extern "system" fn(Object, *mut Object) -> Hresult = method(result, 5);
                get(result, &mut errors);
                let text = if errors.is_null() {
                    format!("rejected ({status:#x})")
                } else {
                    String::from_utf8_lossy(&blob_bytes(errors)).trim_end_matches('\0').trim().to_owned()
                };
                release(errors);
                Err(text)
            };
            release(result);
            out
        }
    }
}

/// DXC's arguments for a translated program (the command-line tool takes
/// the same): IEEE-strict floats (no fast-math folding of the console's
/// exact operations), full optimization, no debug or reflection data.
pub fn compile_args(profile: &str) -> Vec<String> {
    ["-T", profile, "-E", "main", "-HV", "2021", "-Gis", "-O3", "-Qstrip_debug", "-Qstrip_reflect"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}
