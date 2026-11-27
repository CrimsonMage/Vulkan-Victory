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

use victory_shader::vk::UserData;
use victory_shader::vk::{Interface, VkShader};
use victory_shader::{HwStage, RootType};

/// The file encoding's version: bump it when `encode`/`decode` change.
/// (The key used to hash this whole file, so any edit here, even one that
/// left the encoding alone, threw every kept translation away.)
const MAGIC: &[u8; 8] = b"vkstage5";

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
        h.update(&victory_shader::translation_fingerprint());
        h.update(&victory_shader::vk::ABI_VERSION.to_le_bytes());
        h.update(&victory_shader::vk::semantics::SEMANTICS_VERSION.to_le_bytes());
        h.finalize().to_hex()[..16].to_string()
    })
}

fn file_name(full: &[u8; 32]) -> String {
    let name: String = full[..16].iter().map(|b| format!("{b:02x}")).collect();
    format!("{name}-{}.vks", key())
}

fn path(full: &[u8; 32]) -> Option<PathBuf> {
    Some(dir()?.join(file_name(full)))
}

/// The translations kept, by file name: the folder listed once, then each
/// one written.
fn kept() -> std::sync::MutexGuard<'static, std::collections::HashSet<String>> {
    static KEPT: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let kept = KEPT.get_or_init(|| {
        let names = dir()
            .and_then(|d| std::fs::read_dir(d).ok())
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".vks"))
            .collect();
        std::sync::Mutex::new(names)
    });
    kept.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether an earlier run (or this one) kept a translation of the
/// container `full`: its pipelines were used before, so the title's
/// creation of one makes its stages ahead of the draws.
pub(crate) fn known(full: &[u8; 32]) -> bool {
    dir().is_some() && kept().contains(&file_name(full))
}

/// Every translation of this key: (the container's hash, 32 hex digits,
/// and its file), sorted by hash.
pub(crate) fn all() -> Vec<(String, PathBuf)> {
    let Some(d) = dir() else { return Vec::new() };
    let tail = format!("-{}.vks", key());
    let mut out: Vec<(String, PathBuf)> = std::fs::read_dir(d)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.strip_suffix(&tail).map(|h| (h.to_string(), e.path()))
        })
        .collect();
    out.sort();
    out
}

/// The translation in `file`.
pub(crate) fn read_file(file: &std::path::Path) -> Option<VkShader> {
    decode(&std::fs::read(file).ok()?)
}

/// A line file of the cache folder as a set of its lines.
fn lines_of(name: &str) -> std::collections::BTreeSet<String> {
    dir()
        .and_then(|d| std::fs::read_to_string(d.join(name)).ok())
        .map(|t| t.lines().map(str::to_string).filter(|l| !l.is_empty()).collect())
        .unwrap_or_default()
}

/// Appends `line` to a line file of the cache folder.
fn append_line(name: &str, line: &str) {
    use std::io::Write;
    let Some(d) = dir() else { return };
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(d.join(name)) {
        let _ = writeln!(f, "{line}");
    }
}

/// The program names of the containers (hash, 32 hex digits -> name), kept
/// as the game translates or reads them (`stage-names.txt`).
pub(crate) fn names() -> std::collections::HashMap<String, String> {
    lines_of("stage-names.txt")
        .into_iter()
        .filter_map(|l| l.split_once(' ').map(|(h, n)| (h.to_string(), n.to_string())))
        .collect()
}

/// Notes the program name of container `full` (once per run and hash).
pub(crate) fn note_name(full: &[u8; 32], name: &str) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    if name.is_empty() {
        return;
    }
    let hash: String = full[..16].iter().map(|b| format!("{b:02x}")).collect();
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(|| names().into_keys().collect());
    if seen.insert(hash.clone()) {
        append_line("stage-names.txt", &format!("{hash} {name}"));
    }
}

/// Every translation file of any translator key: (hash, key, file). The
/// cache folder is also the record of which containers the title uses.
fn every_key() -> Vec<(String, String, PathBuf)> {
    let Some(d) = dir() else { return Vec::new() };
    std::fs::read_dir(d)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let (hash, key) = name.strip_suffix(".vks")?.split_once('-')?;
            (hash.len() == 32 && key.len() == 16).then(|| (hash.to_string(), key.to_string(), e.path()))
        })
        .collect()
}

/// The containers the title used (a translation under any key, or a name
/// noted) with no translation under this key.
pub(crate) fn wanted() -> std::collections::BTreeSet<String> {
    let here: std::collections::BTreeSet<String> = all().into_iter().map(|(h, _)| h).collect();
    every_key()
        .into_iter()
        .map(|(h, _, _)| h)
        .chain(names().into_keys())
        .filter(|h| !here.contains(h))
        .collect()
}

/// Translations of other keys whose container has one under this key are
/// deleted (each translator change would add ~1,000 files); how many.
pub(crate) fn prune_old_keys() -> usize {
    let here: std::collections::BTreeSet<String> = all().into_iter().map(|(h, _)| h).collect();
    every_key()
        .into_iter()
        .filter(|(h, k, _)| k != key() && here.contains(h))
        .filter(|(_, _, p)| std::fs::remove_file(p).is_ok())
        .count()
}

/// The file listing what `victory bake` (or the game's background bake)
/// did with each stage with this translator key on the device of `uuid`:
/// one line a stage, `hash<TAB>outcome` (compiled, refused or failed with
/// its reason, slow lane).
fn baked_file(uuid: &str) -> String {
    format!("baked-{}-{uuid}.txt", key())
}

/// What earlier bakes did, by hash (the last line of a hash wins).
pub(crate) fn baked(uuid: &str) -> std::collections::BTreeMap<String, String> {
    let text = dir().and_then(|d| std::fs::read_to_string(d.join(baked_file(uuid))).ok()).unwrap_or_default();
    text.lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .map(|l| match l.split_once('\t') {
            Some((h, o)) => (h.to_string(), o.to_string()),
            None => (l, "compiled".to_string()),
        })
        .collect()
}

pub(crate) fn mark_baked(uuid: &str, hash: &str, outcome: &str) {
    append_line(&baked_file(uuid), &format!("{hash}\t{}", outcome.replace(['\t', '\n'], " ")));
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

/// The immediate samplers' slots of earlier runs (`stages::immediate_slot`):
/// one sampler a line, its four words in hex, in slot order. Kept apart
/// from the translations (slots are the host's, not the translator's).
fn immediates_path() -> Option<PathBuf> {
    Some(dir()?.join("immediate-samplers.txt"))
}

pub(crate) fn load_immediates() -> Vec<[u32; 4]> {
    let Some(text) = immediates_path().and_then(|p| std::fs::read_to_string(p).ok()) else {
        return Vec::new();
    };
    let parse = |line: &str| -> Option<[u32; 4]> {
        let w: Vec<u32> = line.split_whitespace().map(|x| u32::from_str_radix(x, 16).ok()).collect::<Option<_>>()?;
        w.try_into().ok()
    };
    let list: Option<Vec<[u32; 4]>> = text.lines().filter(|l| !l.trim().is_empty()).map(parse).collect();
    let list = list.unwrap_or_default();
    tracing::info!(samplers = list.len(), "Vulkan: immediate sampler slots of earlier runs");
    list
}

pub(crate) fn save_immediates(list: &[[u32; 4]]) {
    let Some(p) = immediates_path() else { return };
    let text: String = list.iter().map(|w| format!("{:08x} {:08x} {:08x} {:08x}\n", w[0], w[1], w[2], w[3])).collect();
    let tmp = p.with_extension("tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, &p);
    }
}

/// The places of programs' code tables in the tables buffer that earlier
/// runs gave (`tables.rs`, `victory_vk::tables`): a line a place, in the
/// order given. Kept apart from the translations, as the immediate
/// samplers' slots: the places are the host's, and what the driver's cache
/// knows a pipeline by.
pub(crate) fn code_table_lines() -> Vec<String> {
    let text = dir().and_then(|d| std::fs::read_to_string(d.join("code-tables.txt")).ok()).unwrap_or_default();
    text.lines().map(str::to_string).collect()
}

/// Keeps a new place of code tables.
pub(crate) fn note_code_table(line: &str) {
    append_line("code-tables.txt", line);
}

/// Drops the places kept (they are given anew).
pub(crate) fn forget_code_tables() {
    if let Some(d) = dir() {
        let _ = std::fs::remove_file(d.join("code-tables.txt"));
    }
}

/// Keeps `vs`, the translation of the container `full`.
pub(crate) fn write(full: &[u8; 32], vs: &VkShader) {
    let Some(p) = path(full) else { return };
    if let Some(dir) = dir() {
        save_registry(dir);
    }
    let tmp = p.with_extension("tmp");
    if std::fs::write(&tmp, encode(vs)).is_ok() && std::fs::rename(&tmp, &p).is_ok() {
        kept().insert(file_name(full));
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
    w.push(vs.mesh_parts);
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
    w.push(vs.tables.len() as u32);
    w.extend(&vs.tables);
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
    let mesh_parts = r.next()?;
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
    let n = r.next()? as usize;
    let tables = r.take(n)?.to_vec();
    Some(VkShader { stage, spirv, user_data, samplers, interface, local_size, approximate: Vec::new(), ir_insts: 0, guard_id, mesh_parts, tables })
}
