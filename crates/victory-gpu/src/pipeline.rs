//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are not made at every creation: the title creates some
//! 60,000 pipelines in its first 20 seconds and draws with a few thousand.
//! The Vulkan backend translates and compiles each shader stage once
//! (`vulkan/stages.rs`): at the title's creation of a pipeline whose stages
//! an earlier run translated (kept on disk, `vulkan/stage_cache.rs`), and
//! otherwise at its first draw. Shader containers are shared by content
//! hash.
//!
//! The 32-byte packet after the header stays zero: titles only copy it
//! inline, which our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    /// Its whole BLAKE3: what the stage cache knows it by.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) full: [u8; 32],
    /// The program's name, if its container tells (`CS_SkyMain`,
    /// `CS_TransmissionVolumetrics_Tiled`; see
    /// [`program_name`]); empty otherwise. Logged beside the hash so a log
    /// says which pass a failure is without the title's shader archive.
    pub(crate) name: String,
    /// A pass of the title's upscaler ([`is_pass`]).
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) pass: bool,
    container: Vec<u8>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// The program's name its container carries, as the engine's shader
/// archive names the record (`<entry>_<8 hex>_Wrapped`; the archive's
/// record adds `<family>:` before it, the container does not): the
/// printable run that ends at `_Wrapped`, without the 8-digit suffix and
/// `_Wrapped`. A container of another compiler's code (a pass of the
/// title's upscaler: an `ACCC` part) has no such string and names its
/// entry function in its `PSV0` part. Empty if the container has neither.
pub(crate) fn program_name(container: &[u8]) -> String {
    const END: &[u8] = b"_Wrapped";
    let Some(at) = container.windows(END.len()).position(|w| w == END) else {
        let entry = || victory_shader::Container::parse(container).ok()?.entry_name().map(str::to_owned);
        return if is_pass(container) { entry().unwrap_or_default() } else { String::new() };
    };
    let begin = container[..at]
        .iter()
        .rposition(|b| !(b.is_ascii_alphanumeric() || *b == b'_' || *b == b':'))
        .map_or(0, |p| p + 1);
    let name = String::from_utf8_lossy(&container[begin..at]).into_owned();
    // `_9e6def81`: the variant's hash, the same for a whole family.
    match name.rsplit_once('_') {
        Some((head, tail)) if tail.len() == 8 && tail.bytes().all(|b| b.is_ascii_hexdigit()) => head.to_owned(),
        _ => name,
    }
}

/// Whether `container` is a pass of the title's upscaler: console code of
/// another compiler, in an `ACCC` part (`victory_shader::native`).
pub(crate) fn is_pass(container: &[u8]) -> bool {
    use victory_shader::{Container, FourCC};
    Container::parse(container).is_ok_and(|c| c.part(FourCC::ACCC).is_some())
}

/// The names of `shaders`, for logs (`?` for a nameless one).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn names(shaders: &[Arc<Shader>]) -> String {
    shaders
        .iter()
        .map(|s| if s.name.is_empty() { "?" } else { s.name.as_str() })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// Takes the lock of the containers.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = lock(&SHADERS);
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            Arc::new(Shader {
                stage,
                hash: full.to_hex()[..16].to_owned(),
                full: *full.as_bytes(),
                name: program_name(bytes),
                pass: is_pass(bytes),
                container: bytes.to_vec(),
            })
        })
        .clone()
}

impl Shader {
    /// The container's bytes.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) fn container(&self) -> &[u8] {
        &self.container
    }
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// A stream's graphics state (none for compute and the plain graphics
    /// description).
    #[cfg(windows)]
    graphics: Option<Box<state::GraphicsState>>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, name = %shader.name, "first bind");
    }
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<state::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    // Stages an earlier run translated are made now, ahead of the draws.
    #[cfg(windows)]
    if shaders.iter().any(|s| crate::vulkan::stage_known(&s.full)) {
        crate::vulkan::pipeline_created(&shaders);
    }
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        graphics,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = state::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod state;

/// Where the caches of the stage translations go: `dir`'s siblings
/// (`vulkan-stages`, the pipeline caches).
static CACHE_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

/// Keeps the Vulkan backend's caches beside `dir` from now on.
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    let _ = CACHE_DIR.set(dir);
}


/// The folder of the pipeline caches, if one is set.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn cache_dir() -> Option<std::path::PathBuf> {
    CACHE_DIR.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// The stream layout the title builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }
}

#[cfg(test)]
mod name_tests {
    use super::{is_pass, program_name};

    #[test]
    fn names_come_from_the_wrapped_entry() {
        let c = b"DXBC\x00\x01junk\x1eCS_Visibility_7226f52c_Wrapped\x00more";
        assert_eq!(program_name(c), "CS_Visibility");
        assert_eq!(program_name(b"\x00PS_FogColor_Wrapped\x00"), "PS_FogColor");
        assert_eq!(program_name(b"DXBC no name here"), "");
    }

    /// A pass of the upscaler (an `ACCC` part) is named by its `PSV0`
    /// entry; a container of the title's compiler without `_Wrapped` (the
    /// ray-tracing structure builder's) stays nameless, PSV0 or not: the
    /// recorder knows the builder's programs by that.
    #[test]
    fn a_pass_of_the_upscaler_is_named_by_its_entry() {
        // PSV0: 52 bytes of runtime information (the name at 1 of the
        // string table), no resources, the table
        let mut psv = 52u32.to_le_bytes().to_vec();
        psv.resize(4 + 48, 0);
        psv.extend_from_slice(&1u32.to_le_bytes());
        psv.extend_from_slice(&0u32.to_le_bytes());
        psv.extend_from_slice(&12u32.to_le_bytes());
        psv.extend_from_slice(b"\0some_pass\0\0");
        let container = |code: &[u8; 4]| {
            let parts: [(&[u8; 4], &[u8]); 2] = [(b"PSV0", &psv), (code, b"FSPX")];
            let table = 0x20 + 4 * parts.len();
            let total = table + parts.iter().map(|p| 8 + p.1.len()).sum::<usize>();
            let mut c = b"DXBC".to_vec();
            c.extend_from_slice(&[0; 16]);
            c.extend_from_slice(&[1, 0, 0, 0]);
            c.extend_from_slice(&(total as u32).to_le_bytes());
            c.extend_from_slice(&(parts.len() as u32).to_le_bytes());
            let mut at = table;
            for p in &parts {
                c.extend_from_slice(&(at as u32).to_le_bytes());
                at += 8 + p.1.len();
            }
            for p in &parts {
                c.extend_from_slice(p.0);
                c.extend_from_slice(&(p.1.len() as u32).to_le_bytes());
                c.extend_from_slice(p.1);
            }
            c
        };
        assert_eq!(program_name(&container(b"ACCC")), "some_pass");
        assert_eq!(program_name(&container(b"DXIL")), "");
        assert!(is_pass(&container(b"ACCC")) && !is_pass(&container(b"DXIL")) && !is_pass(b"DXBC"));
    }
}

/// What the Vulkan backend links a title pipeline from: its stages'
/// containers and, for graphics, the state the title gave.
#[cfg(windows)]
pub(crate) fn parts<'a>(this: This) -> Option<(&'a [Arc<Shader>], Option<&'a state::GraphicsState>)> {
    if this.is_null() {
        return None;
    }
    let p = pipeline(this);
    Some((&p.shaders, p.graphics.as_deref()))
}
