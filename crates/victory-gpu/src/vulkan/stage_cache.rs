//! Translated stages kept on disk (one file per container), so that a warm
//! launch translates nothing: the stage workers spent most of a run's CPU
//! re-translating the catalog's ~2,000 programs at 10–15 ms each.
//!
//! A file is named by the container's BLAKE3 and a key of the translator
//! (its source fingerprint and ABI version) and this file's encoding; a
//! translator change makes new names, so stale files are never read. Plain
//! file reads and writes, nothing held in memory.

use std::path::PathBuf;
use std::sync::OnceLock;

use victory_shader::emit::UserData;
use victory_shader::vk::{Interface, VkShader};
use victory_shader::{HwStage, RootType};

const MAGIC: &[u8; 8] = b"vkstage2";

fn dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let Some(dir) = crate::pipeline::cache_dir().map(|d| d.with_file_name("vulkan-stages")) else {
            tracing::warn!("Vulkan: no cache folder: translations are not kept");
            return None;
        };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(dir = %dir.display(), error = %e, "Vulkan: translations are not kept");
            return None;
        }
        tracing::info!(dir = %dir.display(), key = key(), "Vulkan: translations kept on disk");
        load_registry(&dir);
        Some(dir)
    })
    .as_ref()
}

/// The translator's key (hex), the same for every file of this build's
/// translator.
fn key() -> &'static str {
    static KEY: OnceLock<String> = OnceLock::new();
    KEY.get_or_init(|| {
        let mut h = blake3::Hasher::new();
        h.update(MAGIC);
        h.update(&victory_shader::emit::translation_fingerprint());
        h.update(&victory_shader::vk::ABI_VERSION.to_le_bytes());
        h.update(&victory_shader::vk::semantics::SEMANTICS_VERSION.to_le_bytes());
        h.update(include_bytes!("stage_cache.rs"));
        h.finalize().to_hex()[..16].to_string()
    })
}

fn path(full: &[u8; 32]) -> Option<PathBuf> {
    let name: String = full[..16].iter().map(|b| format!("{b:02x}")).collect();
    Some(dir()?.join(format!("{name}-{}.vks", key())))
}

/// The cached translation of the container `full`, if there is one.
pub(crate) fn read(full: &[u8; 32]) -> Option<VkShader> {
    decode(&std::fs::read(path(full)?).ok()?)
}

/// The title's semantic registry (ABI v2 section 6) beside the
/// translations: its locations are in them, so the two are kept or dropped
/// together. With no registry file (or one that does not load), the
/// translations of this key are dropped.
fn registry_path(dir: &std::path::Path) -> PathBuf {
    dir.join(format!("semantics-{}.txt", key()))
}

static SAVED: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn load_registry(dir: &std::path::Path) {
    let p = registry_path(dir);
    let semantics = victory_shader::vk::Semantics::global();
    let loaded = std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|text| semantics.load(&text));
    match loaded {
        Ok(()) => *SAVED.lock().unwrap_or_else(|e| e.into_inner()) = semantics.save(),
        Err(e) => {
            tracing::info!(registry = %p.display(), why = %e, "Vulkan: no semantic registry; the translations kept with it are dropped");
            let _ = std::fs::remove_file(&p);
            let tail = format!("-{}.vks", key());
            for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                if entry.file_name().to_string_lossy().ends_with(&tail) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}

/// Writes the registry if it grew (before a translation that may use the
/// new entries is kept).
fn save_registry(dir: &std::path::Path) {
    let text = victory_shader::vk::Semantics::global().save();
    let mut saved = SAVED.lock().unwrap_or_else(|e| e.into_inner());
    if *saved == text {
        return;
    }
    let p = registry_path(dir);
    let tmp = p.with_extension("tmp");
    if std::fs::write(&tmp, &text).is_ok() && std::fs::rename(&tmp, &p).is_ok() {
        *saved = text;
    }
}

/// Keeps `vs`, the translation of the container `full`.
pub(crate) fn write(full: &[u8; 32], vs: &VkShader) {
    let Some(p) = path(full) else { return };
    if let Some(dir) = dir() {
        save_registry(dir);
    }
    let tmp = p.with_extension("tmp");
    if std::fs::write(&tmp, encode(vs)).is_ok() {
        let _ = std::fs::rename(&tmp, &p);
    }
}

fn stage_code(s: HwStage) -> u32 {
    match s {
        HwStage::Vs => 0,
        HwStage::Ps => 1,
        HwStage::Cs => 2,
        HwStage::Ms => 3,
        _ => 9,
    }
}

fn stage_of(c: u32) -> Option<HwStage> {
    Some(match c {
        0 => HwStage::Vs,
        1 => HwStage::Ps,
        2 => HwStage::Cs,
        3 => HwStage::Ms,
        _ => return None,
    })
}

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 3,
        RootType::ImmediateSampler => 4,
        RootType::ResourceHeap => 5,
        RootType::SamplerHeap => 6,
        RootType::Other(n) => 0x100 | n,
    }
}

fn root_of(c: u32) -> RootType {
    match c {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        3 => RootType::Table,
        4 => RootType::ImmediateSampler,
        5 => RootType::ResourceHeap,
        6 => RootType::SamplerHeap,
        n => RootType::Other(n & 0xff),
    }
}

fn encode(vs: &VkShader) -> Vec<u8> {
    let mut w: Vec<u32> = vec![stage_code(vs.stage)];
    w.extend(vs.local_size);
    w.push(vs.guard_id);
    w.push(vs.user_data.len() as u32);
    for u in &vs.user_data {
        w.extend([u.index, u32::from(u.slot), root_code(u.root), u.dword]);
    }
    w.push(vs.samplers.len() as u32);
    for s in &vs.samplers {
        w.extend(s);
    }
    w.push(vs.interface.len() as u32);
    for i in &vs.interface {
        let name = i.semantic.as_bytes();
        w.extend([i.index, u32::from(i.location), u32::from(i.input), name.len() as u32]);
        w.extend(name.chunks(4).map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        }));
    }
    w.push(vs.spirv.len() as u32);
    w.extend(&vs.spirv);
    let mut out = MAGIC.to_vec();
    out.extend(w.iter().flat_map(|x| x.to_le_bytes()));
    out
}

/// Reads words in order.
struct Words<'a>(&'a [u32], usize);

impl Words<'_> {
    fn next(&mut self) -> Option<u32> {
        let v = *self.0.get(self.1)?;
        self.1 += 1;
        Some(v)
    }

    fn take(&mut self, n: usize) -> Option<&[u32]> {
        let s = self.0.get(self.1..self.1 + n)?;
        self.1 += n;
        Some(s)
    }
}

fn decode(bytes: &[u8]) -> Option<VkShader> {
    let body = bytes.strip_prefix(MAGIC.as_slice())?;
    let words: Vec<u32> = body.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut r = Words(&words, 0);
    let stage = stage_of(r.next()?)?;
    let local_size = [r.next()?, r.next()?, r.next()?];
    let guard_id = r.next()?;
    let n = r.next()? as usize;
    let mut user_data = Vec::with_capacity(n);
    for _ in 0..n {
        let f = r.take(4)?;
        user_data.push(UserData { index: f[0], slot: f[1] as u8, root: root_of(f[2]), dword: f[3] });
    }
    let n = r.next()? as usize;
    let mut samplers = Vec::with_capacity(n);
    for _ in 0..n {
        let f = r.take(4)?;
        samplers.push([f[0], f[1], f[2], f[3]]);
    }
    let n = r.next()? as usize;
    let mut interface = Vec::with_capacity(n);
    for _ in 0..n {
        let f = r.take(4)?;
        let (index, location, input, len) = (f[0], f[1] as u8, f[2] != 0, f[3] as usize);
        let name: Vec<u8> = r.take(len.div_ceil(4))?.iter().flat_map(|x| x.to_le_bytes()).take(len).collect();
        interface.push(Interface { semantic: String::from_utf8(name).ok()?, index, location, input, word: None });
    }
    let n = r.next()? as usize;
    let spirv = r.take(n)?.to_vec();
    Some(VkShader { stage, spirv, user_data, samplers, interface, local_size, approximate: Vec::new(), ir_insts: 0, guard_id })
}
