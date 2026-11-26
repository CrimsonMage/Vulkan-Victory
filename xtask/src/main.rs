//! Repository maintenance. Run with `cargo xtask <command>`.

mod complexity;
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

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
