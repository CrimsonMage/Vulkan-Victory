//! `victory bake`, first step: the containers the title used (the stage
//! cache's record, under any translator key) that have no translation
//! under the current key are found in the title's packs, by the same scan
//! as `victory translate`, and translated as the game would translate
//! them (`victory_gpu::bake_translate`: the title's semantic registry, the
//! stage cache, program names). The compile step then finds them all.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use crate::shader::{MAX_FILE, sources};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Translates every wanted container found in the configured game paths.
pub(crate) fn translate_wanted(jobs: Option<usize>) -> Result<()> {
    let wanted = Mutex::new(victory_gpu::bake_wanted());
    let total = lock(&wanted).len();
    if total == 0 {
        return Ok(());
    }
    println!("bake: {total} containers the title used have no translation under this translator; translating them from the packs");
    let threads = jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() / 2)).max(1);
    let (done, failed) = (AtomicUsize::new(0), Mutex::new(Vec::new()));
    for root in crate::default_paths(Vec::new())? {
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if lock(&wanted).is_empty() {
                break;
            }
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else { continue };
            scan(&data, threads, &wanted, &done, &failed);
        }
    }
    let failed = failed.into_inner().unwrap_or_else(|e| e.into_inner());
    let missing = wanted.into_inner().unwrap_or_else(|e| e.into_inner());
    println!(
        "bake: {} translated, {} did not translate, {} not found in the packs",
        done.load(Ordering::Relaxed),
        failed.len(),
        missing.len()
    );
    for (hash, why) in failed {
        println!("bake:   {hash}: {why}");
    }
    Ok(())
}

/// The wanted containers of one file, translated on `threads` threads.
fn scan(data: &[u8], threads: usize, wanted: &Mutex<BTreeSet<String>>, done: &AtomicUsize, failed: &Mutex<Vec<(String, String)>>) {
    let jobs = sources(data);
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(jobs.len()) {
            s.spawn(|| {
                while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let Some(bytes) = job.bytes() else { continue };
                    let hash: String = blake3::hash(&bytes).as_bytes()[..16].iter().map(|b| format!("{b:02x}")).collect();
                    if !lock(wanted).remove(&hash) {
                        continue;
                    }
                    match victory_gpu::bake_translate(&bytes) {
                        Ok(()) => {
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(why) => lock(failed).push((hash, why)),
                    }
                }
            });
        }
    });
}
