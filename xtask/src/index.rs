//! `cargo xtask translation-index`: every shader container in a title's
//! dump through the SPIR-V translator (`victory_shader::vk::report`),
//! written up by index_md.rs as a markdown index.
//!
//! Containers are found the way `victory translate` finds them: the
//! records of SGD5 files (the engine's shader archive) and every DXBC
//! container stored as is. Each distinct container (by blake3) is
//! translated once, on a few threads that each hold one container and its
//! translation; the dump's files are read one at a time. With `spirv-val`
//! on PATH every module is validated as the host's ABI requires.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use victory_game::sgd5;
use victory_shader::vk::{self, Approx};
use victory_shader::{Container, HwStage};

/// Files above this size are skipped (read whole).
const MAX_FILE: u64 = 2 << 30;
/// The translator recurses over deep programs; the default 2 MiB is too
/// little for some (only what is touched is committed).
const WORKER_STACK: usize = 64 << 20;

/// One distinct container and what the translator made of it.
pub struct Program {
    /// First 16 hex digits of the container's blake3 (what a run's log names).
    pub hash: String,
    /// The record's name, else `file@offset`.
    pub name: String,
    pub stage: Option<HwStage>,
    pub opcodes: Vec<&'static str>,
    pub outcome: Outcome,
}

pub enum Outcome {
    /// `invalid`: spirv-val's first complaint.
    Translated {
        approx: Vec<Approx>,
        invalid: Option<String>,
    },
    /// `at`: the opcode the refusal names.
    Refused {
        reason: String,
        at: Option<&'static str>,
    },
}

/// Where the index came from (its header).
pub struct Origin {
    pub commit: String,
    pub translator: String,
    pub pack: String,
    pub bytes: u64,
    pub pack_hash: String,
    pub validated: bool,
    /// Packed records that would not unpack.
    pub corrupt: usize,
}

/// A container in a file: as is, or packed in an SGD5 record.
enum Source<'a> {
    Raw { offset: usize, data: &'a [u8] },
    Packed(sgd5::Record<'a>),
}

pub fn run(root: &Path, pack: &Path, out: &Path, jobs: usize) -> Result<()> {
    let validated = spirv_val_present();
    let files = files(pack)?;
    let seen = Mutex::new(HashSet::new());
    let programs = Mutex::new(Vec::new());
    let corrupt = AtomicUsize::new(0);
    let (mut bytes, mut hasher) = (0u64, blake3::Hasher::new());
    for path in &files {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        bytes += data.len() as u64;
        hasher.update(blake3::hash(&data).as_bytes());
        let label = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let ctx = Ctx {
            label: &label,
            validated,
            seen: &seen,
            corrupt: &corrupt,
        };
        let found = translate_file(&data, &ctx, jobs);
        lock(&programs).extend(found);
        eprintln!(
            "{}: {} programs so far",
            path.display(),
            lock(&programs).len()
        );
    }
    let mut programs = programs.into_inner().unwrap_or_else(|e| e.into_inner());
    programs.sort_by(|a, b| a.hash.cmp(&b.hash));
    let origin = Origin {
        commit: commit(root),
        translator: blake3::hash(&victory_shader::translation_fingerprint()).to_hex()[..16]
            .to_owned(),
        pack: pack.file_name().map_or_else(
            || pack.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        ),
        bytes,
        pack_hash: hasher.finalize().to_hex()[..16].to_owned(),
        validated,
        corrupt: corrupt.into_inner(),
    };
    let text = crate::index_md::render(&origin, &programs, &victory_shader::gfx10::known_names());
    std::fs::write(out, text).with_context(|| format!("writing {}", out.display()))?;
    println!("{}: {} programs", out.display(), programs.len());
    Ok(())
}

/// The pack, or every file under it in name order.
fn files(pack: &Path) -> Result<Vec<PathBuf>> {
    anyhow::ensure!(pack.exists(), "{} does not exist", pack.display());
    if pack.is_file() {
        return Ok(vec![pack.to_owned()]);
    }
    Ok(walkdir::WalkDir::new(pack)
        .sort_by_file_name()
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file() && e.metadata().is_ok_and(|m| m.len() <= MAX_FILE))
        .map(|e| e.into_path())
        .collect())
}

/// What every worker shares.
struct Ctx<'a> {
    label: &'a str,
    validated: bool,
    seen: &'a Mutex<HashSet<blake3::Hash>>,
    corrupt: &'a AtomicUsize,
}

fn translate_file(data: &[u8], ctx: &Ctx<'_>, jobs: usize) -> Vec<Program> {
    let sources = sources(data);
    let next = AtomicUsize::new(0);
    let found = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs.min(sources.len()) {
            let worker = || {
                let mut mine = Vec::new();
                while let Some(source) = sources.get(next.fetch_add(1, Ordering::Relaxed)) {
                    mine.extend(translate(source, ctx));
                }
                lock(&found).extend(mine);
            };
            let spawned = std::thread::Builder::new()
                .stack_size(WORKER_STACK)
                .spawn_scoped(s, worker);
            spawned.expect("starting a worker thread");
        }
    });
    found.into_inner().unwrap_or_else(|e| e.into_inner())
}

/// The packed records of SGD5 data, and every container stored as is.
fn sources(data: &[u8]) -> Vec<Source<'_>> {
    let mut out: Vec<Source<'_>> = Vec::new();
    if sgd5::is_sgd5(data) {
        out.extend(sgd5::records(data).map(Source::Packed));
    }
    for offset in memchr::memmem::find_iter(data, b"DXBC") {
        let total = data
            .get(offset + 0x18..offset + 0x1C)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let Some(c) = data.get(offset..offset + total as usize) else {
            continue;
        };
        if Container::parse(c).is_ok() {
            out.push(Source::Raw { offset, data: c });
        }
    }
    out
}

fn translate(source: &Source<'_>, ctx: &Ctx<'_>) -> Option<Program> {
    let (bytes, name) = match source {
        Source::Raw { offset, data } => (Some(data.to_vec()), format!("{}@{offset:#x}", ctx.label)),
        Source::Packed(r) => (
            r.unpack(),
            r.name
                .map_or_else(|| format!("{}@{:#x}", ctx.label, r.offset), str::to_owned),
        ),
    };
    let Some(bytes) = bytes else {
        ctx.corrupt.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let hash = blake3::hash(&bytes);
    if !lock(ctx.seen).insert(hash) {
        return None;
    }
    let hash = hash.to_hex()[..16].to_owned();
    let Ok(r) = std::panic::catch_unwind(|| vk::report(&bytes)) else {
        let outcome = Outcome::Refused {
            reason: "the translator panicked".into(),
            at: None,
        };
        return Some(Program {
            hash,
            name,
            stage: vk::report::stage(&bytes),
            opcodes: Vec::new(),
            outcome,
        });
    };
    let outcome = match r.result {
        Ok(s) => {
            let invalid = if ctx.validated {
                validate(&s.spirv)
            } else {
                None
            };
            Outcome::Translated {
                approx: s.approximate,
                invalid,
            }
        }
        Err(e) => Outcome::Refused {
            reason: vk::report::reason(&e),
            at: r.refused_at,
        },
    };
    Some(Program {
        hash,
        name,
        stage: r.stage,
        opcodes: r.opcodes,
        outcome,
    })
}

fn spirv_val_present() -> bool {
    Command::new("spirv-val")
        .arg("--version")
        .stdout(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// spirv-val as `docs/vulkan-abi.md` section 1 requires; its first
/// complaint, or `None` when the module is valid.
fn validate(words: &[u32]) -> Option<String> {
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "--scalar-block-layout", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let Ok(mut child) = child else {
        return Some("spirv-val did not start".into());
    };
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(&bytes);
    }
    let out = child.wait_with_output().ok()?;
    if out.status.success() {
        return None;
    }
    let text =
        String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    Some(
        text.lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("invalid")
            .trim()
            .to_owned(),
    )
}

/// HEAD, marked `-dirty` when the translator or this tool has changes
/// outside it.
fn commit(root: &Path) -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let Some(head) = git(&["rev-parse", "--short=12", "HEAD"]) else {
        return "unknown".into();
    };
    let dirty = git(&[
        "status",
        "--porcelain",
        "--",
        "crates/victory-shader",
        "crates/victory-game",
        "xtask",
    ])
    .is_some_and(|s| !s.is_empty());
    if dirty { format!("{head}-dirty") } else { head }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
