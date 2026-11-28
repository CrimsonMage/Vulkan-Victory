//! Repository maintenance. Run with `cargo xtask <command>`.

mod complexity;
#[cfg(feature = "index")]
mod golden;
#[cfg(feature = "index")]
mod index;
#[cfg(feature = "index")]
mod index_md;
mod policy;
mod quality;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enforce the asset policy: the repository holds only small text files
    /// we wrote. Checks tracked and untracked-but-not-ignored files.
    Policy {
        /// Check only what is staged for commit (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
    },
    /// Enforce the code-shape limits (docs/kiss.md): lines per file and
    /// cognitive / cyclomatic complexity per function, ratcheted by
    /// xtask/quality-baseline.toml.
    Quality {
        /// Check only the .rs files staged for commit; print errors only
        /// (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
        /// Measure a git revision instead of the working tree (read-only),
        /// against the working tree's baseline.
        #[arg(long, value_name = "REV")]
        rev: Option<String>,
        /// Also write a markdown report: findings, top functions, per-crate totals.
        #[arg(long, value_name = "PATH")]
        report: Option<PathBuf>,
        /// Rewrite the baseline: lowers recorded values and drops entries that
        /// are now under the mandatory limits (seeds it if missing).
        #[arg(long)]
        update_baseline: bool,
    },
    /// Point git at .githooks/ so every commit runs the policy and code-shape checks.
    InstallHooks,
    /// Translate every shader container of a title's dump (a file such as
    /// shaders.rpf, or a directory) and write the translation index:
    /// summary, opcode coverage, refusals, approximations. Runs from a
    /// release build with the translator (feature `index`).
    TranslationIndex {
        /// The title's shader pack, or a directory of its files.
        #[arg(long)]
        pack: PathBuf,
        /// The markdown file to write (default: docs/translation-index.md
        /// in the workspace).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Worker threads (each holds one container and its translation).
        #[arg(long, default_value_t = 4)]
        jobs: usize,
    },
    /// Run every program of the shader golden file in the interpreter with
    /// seeded inputs and print those whose behaviour changed (xtask/src/golden.rs).
    ShaderGolden {
        /// The title's shader pack, or a directory of its files.
        #[arg(long)]
        pack: PathBuf,
        /// The golden file (default: docs/shader-golden.txt).
        #[arg(long)]
        golden: Option<PathBuf>,
        /// The programs (a file with a hash per line) instead of the golden
        /// file's, for a first --bless.
        #[arg(long)]
        list: Option<PathBuf>,
        /// Write the golden file from this run.
        #[arg(long)]
        bless: bool,
        /// Instead of the golden comparison: each wave64 compute or mesh
        /// program's group run as console waves and as the host subgroups
        /// the card runs (its halves exchanging), and those that differ.
        #[arg(long)]
        halves: bool,
        /// Worker threads.
        #[arg(long, default_value_t = 3)]
        jobs: usize,
    },
    /// Print how each compute program of the shader golden file reaches
    /// title memory: the buffer views it stores and loads through
    /// (`victory_shader::vk::views`, what the host stages stores by).
    ShaderViews {
        /// The title's shader pack, or a directory of its files.
        #[arg(long)]
        pack: PathBuf,
        /// The golden file (default: docs/shader-golden.txt), or any file
        /// with a program's hash first on each line.
        #[arg(long)]
        golden: Option<PathBuf>,
        /// Also write the lines (hash, the views' line) to this file.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Worker threads.
        #[arg(long, default_value_t = 3)]
        jobs: usize,
    },
    /// Write one container of the pack (by hash) to a file, for
    /// `examples/replay`.
    ShaderExtract {
        #[arg(long)]
        pack: PathBuf,
        /// The first 16 hex digits of its blake3 (as the logs name it).
        #[arg(long)]
        hash: String,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::Quality {
            staged,
            rev,
            report,
            update_baseline,
        } => quality::run(
            &root,
            &quality::Options {
                staged,
                rev,
                report,
                update_baseline,
            },
        ),
        Cmd::TranslationIndex { pack, out, jobs } => {
            let out = out.unwrap_or_else(|| root.join("docs/translation-index.md"));
            translation_index(&root, &pack, &out, jobs)
        }
        Cmd::ShaderGolden { pack, golden, list, bless, halves, jobs } => {
            let golden = golden.unwrap_or_else(|| root.join("docs/shader-golden.txt"));
            shader_golden(&pack, &golden, list.as_deref(), bless, halves, jobs)
        }
        Cmd::ShaderViews { pack, golden, out, jobs } => shader_views(&pack, &golden.unwrap_or_else(|| root.join("docs/shader-golden.txt")), out.as_deref(), jobs),
        Cmd::ShaderExtract { pack, hash, out } => shader_extract(&pack, &hash, &out),
        Cmd::InstallHooks => {
            let status = Command::new("git")
                .args(["config", "core.hooksPath", ".githooks"])
                .current_dir(&root)
                .status()
                .context("running git")?;
            anyhow::ensure!(
                status.success(),
                "git config failed (is this a git repository?)"
            );
            println!(
                "hooks installed: commits now run `cargo xtask policy --staged` and `cargo xtask quality --staged`"
            );
            Ok(())
        }
    }
}

#[cfg(feature = "index")]
fn translation_index(root: &Path, pack: &Path, out: &Path, jobs: usize) -> Result<()> {
    index::run(root, pack, out, jobs.max(1))
}

#[cfg(feature = "index")]
fn shader_golden(pack: &Path, golden: &Path, list: Option<&Path>, bless: bool, halves: bool, jobs: usize) -> Result<()> {
    golden::run(pack, golden, list, bless, halves, jobs)
}

#[cfg(feature = "index")]
fn shader_extract(pack: &Path, hash: &str, out: &Path) -> Result<()> {
    golden::extract(pack, hash, out)
}

#[cfg(feature = "index")]
fn shader_views(pack: &Path, golden: &Path, out: Option<&Path>, jobs: usize) -> Result<()> {
    golden::views(pack, golden, out, jobs)
}

#[cfg(not(feature = "index"))]
fn shader_views(_pack: &Path, _golden: &Path, _out: Option<&Path>, _jobs: usize) -> Result<()> {
    with_translator()
}

#[cfg(not(feature = "index"))]
fn shader_extract(_pack: &Path, _hash: &str, _out: &Path) -> Result<()> {
    with_translator()
}

#[cfg(not(feature = "index"))]
fn shader_golden(_pack: &Path, _golden: &Path, _list: Option<&Path>, _bless: bool, _halves: bool, _jobs: usize) -> Result<()> {
    with_translator()
}

/// Without the translator: the same command again, from a release build
/// that has it (the index should show what the shipped translator does).
#[cfg(not(feature = "index"))]
fn translation_index(_root: &Path, _pack: &Path, _out: &Path, _jobs: usize) -> Result<()> {
    with_translator()
}

/// The same command again from a release build with the translator.
#[cfg(not(feature = "index"))]
fn with_translator() -> Result<()> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .args([
            "run",
            "--quiet",
            "--release",
            "--package",
            "xtask",
            "--features",
            "index",
            "--",
        ])
        .args(std::env::args_os().skip(1))
        .status()
        .context("running cargo")?;
    anyhow::ensure!(status.success(), "the command with the translator failed");
    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
