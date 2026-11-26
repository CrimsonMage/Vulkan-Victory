//! Code-shape limits (docs/kiss.md): lines per file, and cognitive and
//! cyclomatic complexity per function (see complexity.rs).
//!
//! Over the plan limit: a warning, split or simplify when next there.
//! Over the mandatory limit: an error, except for the offenders recorded in
//! xtask/quality-baseline.toml, which may not grow past their recorded value.
//! `--update-baseline` only lowers recorded values and drops entries; it
//! seeds the file only when there is none.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::complexity::{self, FnScore};

#[derive(Clone, Copy)]
pub struct Limit {
    pub plan: u32,
    pub mandatory: u32,
}

pub const FILE_LINES: Limit = Limit {
    plan: 1000,
    mandatory: 1500,
};
pub const COGNITIVE: Limit = Limit {
    plan: 15,
    mandatory: 25,
};
/// Cyclomatic complexity only warns (owner's decision, 2026-11-26): flat
/// lookup tables score high on it while being the simplest code; cognitive
/// complexity is the mandatory gate. `mandatory` is kept for the report.
pub const CYCLOMATIC: Limit = Limit {
    plan: 10,
    mandatory: 20,
};

const BASELINE: &str = "xtask/quality-baseline.toml";

pub struct Options {
    /// Only files staged for commit, read from the index; errors only.
    pub staged: bool,
    /// Measure this revision instead of the working tree (read-only).
    pub rev: Option<String>,
    pub report: Option<PathBuf>,
    pub update_baseline: bool,
}

/// Where files are read from.
enum Source {
    WorkTree,
    Index,
    Rev(String),
}

/// One measured file.
pub struct FileStats {
    pub path: String,
    pub lines: u32,
    pub fns: Vec<FnScore>,
    pub parse_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Scores {
    pub cognitive: u32,
    pub cyclomatic: u32,
}

#[derive(Debug, Default, Deserialize)]
pub struct Baseline {
    #[serde(default)]
    pub files: BTreeMap<String, u32>,
    #[serde(default)]
    pub functions: BTreeMap<String, BTreeMap<String, Scores>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Plan,
    /// Over the mandatory limit but within its baseline allowance.
    Baselined,
    Error,
}

/// One file or function over a limit.
pub struct Finding {
    pub level: Level,
    pub path: String,
    /// None for a file finding.
    pub function: Option<FnScore>,
    pub lines: u32,
    pub note: String,
}

fn source(opts: &Options) -> Result<Source> {
    let source = match (&opts.rev, opts.staged) {
        (Some(_), true) => bail!("--rev and --staged exclude each other"),
        (Some(rev), false) => Source::Rev(rev.clone()),
        (None, true) => Source::Index,
        (None, false) => Source::WorkTree,
    };
    if opts.update_baseline && !matches!(source, Source::WorkTree) {
        bail!("--update-baseline measures the whole working tree; drop --staged/--rev");
    }
    Ok(source)
}

pub fn run(root: &Path, opts: &Options) -> Result<()> {
    let source = source(opts)?;
    let files = measure(root, &source)?;
    let baseline = load_baseline(root, &source)?;
    if opts.update_baseline {
        return update_baseline(root, baseline, &files);
    }

    let findings = check(&files, &baseline.unwrap_or_default());
    print_findings(&files, &findings, matches!(source, Source::Index));
    if let Some(path) = &opts.report {
        std::fs::write(path, markdown(&files, &findings))
            .with_context(|| format!("writing {}", path.display()))?;
        println!("report written to {}", path.display());
    }
    let errors = findings.iter().filter(|f| f.level == Level::Error).count();
    if errors > 0 {
        bail!(
            "{errors} code-shape error(s): split or simplify (limits in docs/kiss.md, \
             baseline in {BASELINE})"
        );
    }
    Ok(())
}

fn update_baseline(root: &Path, old: Option<Baseline>, files: &[FileStats]) -> Result<()> {
    let (verb, new) = match &old {
        Some(old) => ("updated", tighten(old, files)),
        None => ("seeded", seed_baseline(files)),
    };
    std::fs::write(root.join(BASELINE), render_baseline(&new))
        .with_context(|| format!("writing {BASELINE}"))?;
    println!(
        "{verb} {BASELINE}: {} files, {} functions",
        new.files.len(),
        new.functions.values().map(BTreeMap::len).sum::<usize>()
    );
    Ok(())
}

fn git(root: &Path, args: &[&str]) -> Result<Option<Vec<u8>>> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .context("running git")?;
    Ok(out.status.success().then_some(out.stdout))
}

fn list_paths(root: &Path, source: &Source) -> Result<Vec<String>> {
    let Source::Rev(rev) = source else {
        return crate::policy::list_files(root, matches!(source, Source::Index));
    };
    Ok(git(root, &["ls-tree", "-r", "-z", "--name-only", rev])?
        .with_context(|| format!("no revision {rev}"))?
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect())
}

fn read(root: &Path, source: &Source, path: &str) -> Result<Option<Vec<u8>>> {
    match source {
        Source::WorkTree => Ok(std::fs::read(root.join(path)).ok()),
        Source::Index => git(root, &["show", &format!(":{path}")]),
        Source::Rev(rev) => git(root, &["show", &format!("{rev}:{path}")]),
    }
}

fn measure(root: &Path, source: &Source) -> Result<Vec<FileStats>> {
    let paths = list_paths(root, source)?;
    let mut files = Vec::new();
    for path in paths.into_iter().filter(|p| p.ends_with(".rs")) {
        let Some(content) = read(root, source, &path)? else {
            continue; // deleted
        };
        let text = String::from_utf8_lossy(&content);
        let (fns, parse_error) = match complexity::analyze(&text) {
            Ok(fns) => (fns, None),
            Err(e) => (Vec::new(), Some(e.to_string())),
        };
        files.push(FileStats {
            lines: u32::try_from(text.lines().count()).unwrap_or(u32::MAX),
            path,
            fns,
            parse_error,
        });
    }
    Ok(files)
}

/// The baseline from the index when checking staged changes, else from disk.
fn load_baseline(root: &Path, source: &Source) -> Result<Option<Baseline>> {
    let text = match source {
        Source::Index => match git(root, &["show", &format!(":{BASELINE}")])? {
            Some(bytes) => Some(String::from_utf8_lossy(&bytes).into_owned()),
            None => std::fs::read_to_string(root.join(BASELINE)).ok(),
        },
        _ => std::fs::read_to_string(root.join(BASELINE)).ok(),
    };
    text.map(|t| toml::from_str(&t).with_context(|| format!("parsing {BASELINE}")))
        .transpose()
}

/// The crate (or top-level folder) a path belongs to.
pub fn crate_of(path: &str) -> &str {
    let mut parts = path.splitn(3, '/');
    match (parts.next(), parts.next()) {
        (Some("crates" | "apps"), Some(name)) => name,
        (Some(first), _) => first,
        _ => path,
    }
}

impl Baseline {
    /// What a function may score: its own entry, or else (when it moved to
    /// another file in the same crate) the entry of that name elsewhere.
    fn allowance(&self, path: &str, name: &str) -> Option<Scores> {
        if let Some(s) = self.functions.get(path).and_then(|m| m.get(name)) {
            return Some(*s);
        }
        self.functions
            .iter()
            .filter(|(p, _)| crate_of(p) == crate_of(path))
            .filter_map(|(_, m)| m.get(name))
            .copied()
            .reduce(|a, b| Scores {
                cognitive: a.cognitive.max(b.cognitive),
                cyclomatic: a.cyclomatic.max(b.cyclomatic),
            })
    }
}

/// How one value fares against a limit and its baseline allowance.
fn grade(value: u32, limit: Limit, allowance: Option<u32>) -> Option<Level> {
    if value > limit.mandatory {
        Some(match allowance {
            Some(a) if value <= a => Level::Baselined,
            _ => Level::Error,
        })
    } else {
        (value > limit.plan).then_some(Level::Plan)
    }
}

fn describe(what: &str, value: u32, limit: Limit, allowance: Option<u32>) -> String {
    match grade(value, limit, allowance) {
        Some(Level::Error) => match allowance {
            Some(a) => format!("{what} {value} grew past its baseline {a}"),
            None => format!("{what} {value} > {} (mandatory)", limit.mandatory),
        },
        Some(Level::Baselined) => format!("{what} {value} > {} (baselined)", limit.mandatory),
        Some(Level::Plan) => format!("{what} {value} > {} (plan)", limit.plan),
        None => String::new(),
    }
}

/// Every file and function over a limit. Pure, for testing.
pub fn check(files: &[FileStats], baseline: &Baseline) -> Vec<Finding> {
    let mut out = Vec::new();
    for file in files {
        let allowance = baseline.files.get(&file.path).copied();
        if let Some(level) = grade(file.lines, FILE_LINES, allowance) {
            out.push(Finding {
                level,
                path: file.path.clone(),
                function: None,
                lines: file.lines,
                note: describe("lines", file.lines, FILE_LINES, allowance),
            });
        }
        for f in &file.fns {
            let allowance = baseline.allowance(&file.path, &f.name);
            let cog = allowance.map(|a| a.cognitive);
            let cyc = allowance.map(|a| a.cyclomatic);
            let levels = [
                grade(f.cognitive, COGNITIVE, cog),
                grade(f.cyclomatic, CYCLOMATIC, cyc).map(|_| Level::Plan),
            ];
            let Some(level) = levels.into_iter().flatten().max() else {
                continue;
            };
            let notes: Vec<_> = [
                describe("cognitive", f.cognitive, COGNITIVE, cog),
                warn_only("cyclomatic", f.cyclomatic, CYCLOMATIC),
            ]
            .into_iter()
            .filter(|n| !n.is_empty())
            .collect();
            out.push(Finding {
                level,
                path: file.path.clone(),
                function: Some(f.clone()),
                lines: file.lines,
                note: notes.join(", "),
            });
        }
    }
    out
}

fn is_mandatory(f: &FnScore) -> bool {
    f.cognitive > COGNITIVE.mandatory
}

/// A score that only ever warns.
fn warn_only(what: &str, value: u32, limit: Limit) -> String {
    if value > limit.plan {
        format!("{what} {value} > {} (warning)", limit.plan)
    } else {
        String::new()
    }
}

/// A first baseline: every current offender over a mandatory limit.
fn seed_baseline(files: &[FileStats]) -> Baseline {
    let mut b = Baseline::default();
    for file in files {
        if file.lines > FILE_LINES.mandatory {
            b.files.insert(file.path.clone(), file.lines);
        }
        for f in file.fns.iter().filter(|f| is_mandatory(f)) {
            b.functions.entry(file.path.clone()).or_default().insert(
                f.name.clone(),
                Scores {
                    cognitive: f.cognitive,
                    cyclomatic: f.cyclomatic,
                },
            );
        }
    }
    b
}

/// The baseline lowered to what is measured now: values only go down, and
/// entries no longer over a mandatory limit are dropped. Nothing is added.
fn tighten(old: &Baseline, files: &[FileStats]) -> Baseline {
    let mut b = Baseline::default();
    for file in files {
        if file.parse_error.is_some() {
            // Not measured: keep what was recorded.
            if let Some(m) = old.functions.get(&file.path) {
                b.functions.insert(file.path.clone(), m.clone());
            }
        }
        if file.lines > FILE_LINES.mandatory
            && let Some(&a) = old.files.get(&file.path)
        {
            b.files.insert(file.path.clone(), file.lines.min(a));
        }
        for f in file.fns.iter().filter(|f| is_mandatory(f)) {
            if let Some(a) = old.allowance(&file.path, &f.name) {
                b.functions.entry(file.path.clone()).or_default().insert(
                    f.name.clone(),
                    Scores {
                        cognitive: f.cognitive.min(a.cognitive),
                        cyclomatic: f.cyclomatic.min(a.cyclomatic),
                    },
                );
            }
        }
    }
    b
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn render_baseline(b: &Baseline) -> String {
    let mut out = String::from(
        "# Code-shape ratchet: the offenders over a mandatory limit when the limits\n\
         # came in (docs/kiss.md, `cargo xtask quality`). They may not grow; new code\n\
         # stays under the limits. `cargo xtask quality --update-baseline` lowers\n\
         # values and drops entries as code gets split. Do not add entries.\n\n[files]\n",
    );
    for (path, lines) in &b.files {
        let _ = writeln!(out, "{} = {lines}", quote(path));
    }
    for (path, fns) in &b.functions {
        let _ = writeln!(out, "\n[functions.{}]", quote(path));
        for (name, s) in fns {
            let _ = writeln!(
                out,
                "{} = {{ cognitive = {}, cyclomatic = {} }}",
                quote(name),
                s.cognitive,
                s.cyclomatic
            );
        }
    }
    out
}

fn sorted(findings: &[Finding]) -> (Vec<&Finding>, Vec<&Finding>) {
    let (mut files, mut fns): (Vec<&Finding>, Vec<&Finding>) =
        findings.iter().partition(|f| f.function.is_none());
    files.sort_by_key(|f| std::cmp::Reverse(f.lines));
    fns.sort_by_key(|f| {
        let s = f.function.as_ref().expect("function finding");
        std::cmp::Reverse((s.cognitive, s.cyclomatic))
    });
    (files, fns)
}

fn tag(level: Level) -> &'static str {
    match level {
        Level::Error => "ERROR",
        Level::Baselined => "split",
        Level::Plan => "plan ",
    }
}

fn print_findings(files: &[FileStats], findings: &[Finding], errors_only: bool) {
    let (file_findings, fn_findings) = sorted(findings);
    let shown = |f: &&&Finding| !errors_only || f.level == Level::Error;
    for f in file_findings.iter().filter(shown) {
        println!("{} {}: {}", tag(f.level), f.path, f.note);
    }
    for f in fn_findings.iter().filter(shown) {
        let s = f.function.as_ref().expect("function finding");
        println!(
            "{} {}:{} {}: {}",
            tag(f.level),
            f.path,
            s.line,
            s.name,
            f.note
        );
    }
    for file in files {
        if let Some(e) = &file.parse_error {
            println!(
                "warning: {} not parsed, functions unchecked: {e}",
                file.path
            );
        }
    }
    let count = |l: Level| findings.iter().filter(|f| f.level == l).count();
    println!(
        "code shape: {} files, {} functions; {} errors, {} baselined over mandatory, {} over plan",
        files.len(),
        files.iter().map(|f| f.fns.len()).sum::<usize>(),
        count(Level::Error),
        count(Level::Baselined),
        count(Level::Plan)
    );
}

/// The markdown report: limits, findings, top lists and per-crate totals.
fn markdown(files: &[FileStats], findings: &[Finding]) -> String {
    let mut md = String::from("# Code shape\n\n");
    let _ = writeln!(
        md,
        "Limits (plan / mandatory): lines per file {} / {}, cognitive {} / {}, \
         cyclomatic {} / {}. Generated by `cargo xtask quality --report`.\n",
        FILE_LINES.plan,
        FILE_LINES.mandatory,
        COGNITIVE.plan,
        COGNITIVE.mandatory,
        CYCLOMATIC.plan,
        CYCLOMATIC.mandatory
    );
    let (file_findings, fn_findings) = sorted(findings);
    let over = |limit: u32| files.iter().filter(|f| f.lines > limit).count();
    let _ = writeln!(
        md,
        "{} files, {} functions. Files over {}: {}, over {}: {}. Functions over a plan \
         limit: {}, over a mandatory limit: {}. Errors: {}.\n",
        files.len(),
        files.iter().map(|f| f.fns.len()).sum::<usize>(),
        FILE_LINES.plan,
        over(FILE_LINES.plan),
        FILE_LINES.mandatory,
        over(FILE_LINES.mandatory),
        fn_findings.len(),
        fn_findings
            .iter()
            .filter(|f| f.function.as_ref().is_some_and(is_mandatory))
            .count(),
        findings.iter().filter(|f| f.level == Level::Error).count()
    );

    md.push_str("## Files over the limits\n\n| Level | File | Lines |\n|---|---|---:|\n");
    for f in &file_findings {
        let _ = writeln!(
            md,
            "| {} | `{}` | {} |",
            tag(f.level).trim(),
            f.path,
            f.lines
        );
    }

    let all_fns: Vec<(&str, &FnScore)> = files
        .iter()
        .flat_map(|f| f.fns.iter().map(move |s| (f.path.as_str(), s)))
        .collect();
    for (title, key) in [
        (
            "cognitive",
            (|s: &FnScore| (s.cognitive, s.cyclomatic)) as fn(&FnScore) -> (u32, u32),
        ),
        ("cyclomatic", |s: &FnScore| (s.cyclomatic, s.cognitive)),
    ] {
        let mut top = all_fns.clone();
        top.sort_by_key(|(_, s)| std::cmp::Reverse(key(s)));
        let _ = writeln!(
            md,
            "\n## Top 30 functions by {title}\n\n| Function | Location | Cognitive | Cyclomatic |\n|---|---|---:|---:|"
        );
        for (path, s) in top.iter().take(30) {
            let _ = writeln!(
                md,
                "| `{}` | `{path}:{}` | {} | {} |",
                s.name, s.line, s.cognitive, s.cyclomatic
            );
        }
    }

    md.push_str(&crate_totals(files));

    md.push_str("\n## All functions over the limits\n\n| Level | Function | Location | Cognitive | Cyclomatic |\n|---|---|---|---:|---:|\n");
    for f in &fn_findings {
        let s = f.function.as_ref().expect("function finding");
        let _ = writeln!(
            md,
            "| {} | `{}` | `{}:{}` | {} | {} |",
            tag(f.level).trim(),
            s.name,
            f.path,
            s.line,
            s.cognitive,
            s.cyclomatic
        );
    }
    md
}

fn crate_totals(files: &[FileStats]) -> String {
    #[derive(Default)]
    struct Totals {
        files: usize,
        lines: u64,
        fns: usize,
        files_plan: usize,
        files_mandatory: usize,
        fns_plan: usize,
        fns_mandatory: usize,
        max_cognitive: u32,
        max_cyclomatic: u32,
    }
    let mut by_crate = BTreeMap::<&str, Totals>::new();
    for file in files {
        let t = by_crate.entry(crate_of(&file.path)).or_default();
        t.files += 1;
        t.lines += u64::from(file.lines);
        t.fns += file.fns.len();
        t.files_plan += usize::from(file.lines > FILE_LINES.plan);
        t.files_mandatory += usize::from(file.lines > FILE_LINES.mandatory);
        for f in &file.fns {
            t.fns_plan +=
                usize::from(f.cognitive > COGNITIVE.plan || f.cyclomatic > CYCLOMATIC.plan);
            t.fns_mandatory += usize::from(is_mandatory(f));
            t.max_cognitive = t.max_cognitive.max(f.cognitive);
            t.max_cyclomatic = t.max_cyclomatic.max(f.cyclomatic);
        }
    }
    let mut md = String::from(
        "\n## Per crate\n\n| Crate | Files | Lines | Functions | Files > 1000 | Files > 1500 \
         | Fns > plan | Fns > mandatory | Max cognitive | Max cyclomatic |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for (name, t) in &by_crate {
        let _ = writeln!(
            md,
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            t.files,
            t.lines,
            t.fns,
            t.files_plan,
            t.files_mandatory,
            t.fns_plan,
            t.fns_mandatory,
            t.max_cognitive,
            t.max_cyclomatic
        );
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, lines: u32, fns: &[(&str, u32, u32)]) -> FileStats {
        FileStats {
            path: path.to_owned(),
            lines,
            fns: fns
                .iter()
                .map(|&(name, cognitive, cyclomatic)| FnScore {
                    name: name.to_owned(),
                    line: 1,
                    cognitive,
                    cyclomatic,
                })
                .collect(),
            parse_error: None,
        }
    }

    fn levels(files: &[FileStats], baseline: &Baseline) -> Vec<Level> {
        check(files, baseline)
            .into_iter()
            .map(|f| f.level)
            .collect()
    }

    /// The ratchet holds wherever tests run, even in clones without the hook.
    #[test]
    fn repository_passes_quality() {
        let root = crate::workspace_root();
        let files = measure(&root, &Source::WorkTree).unwrap();
        let baseline = load_baseline(&root, &Source::WorkTree)
            .unwrap()
            .unwrap_or_default();
        let errors: Vec<_> = check(&files, &baseline)
            .into_iter()
            .filter(|f| f.level == Level::Error)
            .map(|f| format!("{} {:?}: {}", f.path, f.function.map(|s| s.name), f.note))
            .collect();
        assert!(errors.is_empty(), "{errors:#?}");
    }

    #[test]
    fn new_code_must_stay_under_mandatory() {
        let b = Baseline::default();
        assert_eq!(levels(&[file("a.rs", 1000, &[("f", 15, 10)])], &b), []);
        assert_eq!(
            levels(&[file("a.rs", 1001, &[("f", 16, 3)])], &b),
            [Level::Plan, Level::Plan]
        );
        assert_eq!(levels(&[file("a.rs", 1501, &[])], &b), [Level::Error]);
        assert_eq!(
            levels(&[file("a.rs", 10, &[("f", 3, 21)])], &b),
            [Level::Error]
        );
    }

    #[test]
    fn baselined_entries_may_shrink_but_not_grow() {
        let b = seed_baseline(&[file("crates/c/src/a.rs", 2000, &[("f", 40, 12)])]);
        assert_eq!(b.files["crates/c/src/a.rs"], 2000);
        let at =
            |lines, cog, cyc| levels(&[file("crates/c/src/a.rs", lines, &[("f", cog, cyc)])], &b);
        assert_eq!(at(2000, 40, 12), [Level::Baselined, Level::Baselined]);
        assert_eq!(at(1900, 30, 20), [Level::Baselined, Level::Baselined]);
        assert_eq!(at(2001, 41, 12), [Level::Error, Level::Error]);
        // The other metric may grow up to the mandatory limit, not past it.
        assert_eq!(at(1400, 40, 21), [Level::Plan, Level::Error]);
        // Moved to another file of the same crate: the allowance follows.
        let moved = levels(&[file("crates/c/src/b.rs", 100, &[("f", 40, 12)])], &b);
        assert_eq!(moved, [Level::Baselined]);
        let other_crate = levels(&[file("crates/d/src/b.rs", 100, &[("f", 40, 12)])], &b);
        assert_eq!(other_crate, [Level::Error]);
    }

    #[test]
    fn update_only_lowers_and_drops() {
        let old = seed_baseline(&[file(
            "crates/c/src/a.rs",
            2000,
            &[("f", 40, 30), ("g", 30, 3)],
        )]);
        let now = [file(
            "crates/c/src/a.rs",
            2100,
            &[("f", 35, 31), ("g", 20, 3), ("h", 50, 50)],
        )];
        let new = tighten(&old, &now);
        assert_eq!(new.files["crates/c/src/a.rs"], 2000);
        let fns = &new.functions["crates/c/src/a.rs"];
        assert_eq!(
            fns["f"],
            Scores {
                cognitive: 35,
                cyclomatic: 30
            }
        );
        assert!(!fns.contains_key("g") && !fns.contains_key("h"));
        let parsed: Baseline = toml::from_str(&render_baseline(&new)).unwrap();
        assert_eq!(parsed.functions, new.functions);
    }
}
