//! The offline batch: every shader found goes through the SPIR-V translator
//! (`victory_shader::vk`), and what it cannot handle yet is tallied.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::{Container, HwStage};

use crate::shader::{MAX_FILE, Source, sources};

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    /// Translated, by stage.
    translated: BTreeMap<String, usize>,
    words: usize,
    approximate: usize,
    /// Failures by reason: count and an example.
    failed: BTreeMap<String, (usize, String)>,
    /// One line per program (with `--list`).
    lines: Vec<String>,
    list: bool,
}

impl Tally {
    fn fail(&mut self, reason: String, example: impl Fn() -> String) {
        let e = self.failed.entry(reason.clone()).or_insert_with(|| (0, example()));
        e.0 += 1;
        if self.list {
            self.lines.push(format!("{}  {reason}", example()));
        }
    }

    fn merge(&mut self, o: Tally) {
        self.containers += o.containers;
        self.duplicates += o.duplicates;
        for (k, n) in o.translated {
            *self.translated.entry(k).or_default() += n;
        }
        self.words += o.words;
        self.approximate += o.approximate;
        for (k, (n, e)) in o.failed {
            self.failed.entry(k).or_insert((0, e)).0 += n;
        }
        self.lines.extend(o.lines);
    }
}

pub fn run(paths: Vec<PathBuf>, list: bool, jobs: Option<usize>) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let threads = jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::default());
    for root in &paths {
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    s.spawn(|| {
                        let mut t = Tally { list, ..Tally::default() };
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            translate(job, entry.path(), &seen, &mut t);
                        }
                        lock(&total).merge(t);
                    });
                }
            });
        }
    }
    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    let done: usize = t.translated.values().sum();
    let failed: usize = t.failed.values().map(|(n, _)| n).sum();
    println!("containers   {} ({} duplicates skipped)", t.containers, t.duplicates);
    println!(
        "SPIR-V       {done} translated ({:?}; {} words, {} with approximations), {failed} not",
        t.translated, t.words, t.approximate
    );
    let mut lines = t.lines;
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    let mut failed: Vec<_> = t.failed.into_iter().collect();
    failed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (reason, (n, example)) in failed {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
    Ok(())
}

fn translate(source: &Source, path: &Path, seen: &Mutex<HashSet<blake3::Hash>>, t: &mut Tally) {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail("corrupt record".into(), example);
    };
    // The hash a run's log names the shader by (`hash=` in "first bind").
    let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
    let example = || format!("{}{} #{hash}", path.display(), source.label());
    t.containers += 1;
    let Some(dxil) = Container::parse(&bytes).ok().and_then(|c| c.code_part().map(|p| p.data)) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    // Containers that differ only around the code are translated once.
    if !lock(seen).insert(blake3::hash(dxil)) {
        t.duplicates += 1;
        return;
    }
    match std::panic::catch_unwind(|| victory_shader::vk::translate(&bytes)) {
        Ok(Ok(vs)) => {
            *t.translated.entry(stage_name(vs.stage).to_owned()).or_default() += 1;
            t.words += vs.spirv.len();
            t.approximate += usize::from(!vs.approximate.is_empty());
            if t.list {
                t.lines.push(format!("{}  {} ok, {} words", example(), stage_name(vs.stage), vs.spirv.len()));
            }
        }
        Ok(Err(e)) => t.fail(e.to_string(), example),
        Err(_) => t.fail("the translator panicked".into(), example),
    }
}

fn stage_name(stage: HwStage) -> &'static str {
    match stage {
        HwStage::Ps => "pixel",
        HwStage::Cs => "compute",
        HwStage::Ms => "mesh",
        _ => "vertex",
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
