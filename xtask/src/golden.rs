//! The shader golden file: what each used program does in the interpreter
//! (`victory_shader::vk::interp`), as one hash per program, so that a
//! translator change is reviewed as "these programs changed behaviour"
//! instead of by a bake and a game run.
//!
//! Each program runs eight times, one console wave each, with seeded
//! inputs: random title memory (`Memory::noise`), random push-constant
//! dwords, a random texel for every image read (different per lane),
//! a work group by the seed. Its behaviour is the hash of how each run
//! stopped, the outputs it wrote, the image stores and the memory stores.
//! The golden file (`docs/shader-golden.txt`) holds `hash behaviour name`
//! per program; `--bless` rewrites it.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use victory_shader::vk::{interp, thread_ir};

use crate::index::{Source, files, sources};

/// Runs per program.
const SEEDS: u64 = 8;
/// Block visits a run may take (a run past it stops as `Limit`, which is
/// part of the behaviour).
const LIMIT: u64 = 4_000;

/// splitmix64.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The behaviour hash of a container (or why it has none).
pub fn behaviour(container: &[u8]) -> String {
    let (f, _, native) = match thread_ir(container) {
        Ok(t) => t,
        Err(e) => return format!("refused-{}", &blake3::hash(e.to_string().as_bytes()).to_hex()[..8]),
    };
    let mut h = blake3::Hasher::new();
    for seed in 1..=SEEDS {
        let mem = interp::Memory { range: (0, u64::MAX), noise: Some(seed), ..Default::default() };
        let lanes = if native.wave64 { 64 } else { 32 };
        let mut m = interp::Machine::new(&f, lanes, mem);
        for (k, p) in m.push.iter_mut().enumerate() {
            *p = mix(seed << 8 | k as u64) as u32;
        }
        m.group = [(seed % 3) as u32, (seed % 2) as u32, 0];
        m.limit = LIMIT;
        m.image_size = [64, 64, 1, 1];
        m.image_texel = [0, 1, 2, 3].map(|c| u64::from(((mix(seed * 4 + c) >> 40) as f32 / (1u64 << 24) as f32).to_bits()));
        m.image_vary = true;
        let stop = m.run();
        let mut s = format!("{stop:?}\n");
        for (k, v) in &m.outputs {
            let _ = writeln!(s, "o {k:?} {v:x}");
        }
        for (l, args) in &m.image_stores {
            let _ = writeln!(s, "i {l} {args:x?}");
        }
        for (a, v) in &m.mem.stored {
            let _ = writeln!(s, "m {a:x} {v:x}");
        }
        h.update(s.as_bytes());
    }
    h.finalize().to_hex()[..16].to_owned()
}

/// Program hash -> (behaviour, name).
type Found = BTreeMap<String, (String, String)>;

/// The golden file.
fn read_golden(path: &Path) -> Found {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            Some((w.next()?.to_owned(), (w.next()?.to_owned(), w.next().unwrap_or("").to_owned())))
        })
        .collect()
}

/// The behaviour of each wanted program of the pack, and the slowest.
fn collect(pack: &Path, wanted: &HashSet<String>, jobs: usize) -> Result<(Found, Vec<(u128, String)>)> {
    let found = Mutex::new(BTreeMap::new());
    let slow = Mutex::new(Vec::new());
    for path in files(pack)? {
        let data = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let srcs = sources(&data);
        let next = AtomicUsize::new(0);
        let worker = || {
            while let Some(src) = srcs.get(next.fetch_add(1, Ordering::Relaxed)) {
                let (bytes, name) = match src {
                    Source::Raw { data, .. } => (Some(data.to_vec()), String::new()),
                    Source::Packed(r) => (r.unpack(), r.name.unwrap_or("").to_owned()),
                };
                let Some(bytes) = bytes else { continue };
                let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
                if !wanted.contains(&hash) || found.lock().unwrap().contains_key(&hash) {
                    continue;
                }
                let t = std::time::Instant::now();
                let b = std::panic::catch_unwind(|| behaviour(&bytes)).unwrap_or_else(|_| "panic".into());
                slow.lock().unwrap().push((t.elapsed().as_millis(), hash.clone()));
                found.lock().unwrap().insert(hash, (b, name));
            }
        };
        std::thread::scope(|s| {
            for _ in 0..jobs.max(1) {
                s.spawn(worker);
            }
        });
    }
    let mut slow = slow.into_inner().unwrap_or_else(|e| e.into_inner());
    slow.sort_unstable_by(|a, b| b.cmp(a));
    slow.truncate(8);
    Ok((found.into_inner().unwrap_or_else(|e| e.into_inner()), slow))
}

/// Prints the programs whose behaviour differs from the golden file's.
fn compare(old: &Found, found: &Found, wanted: &HashSet<String>) {
    let mut changed = 0;
    for (hash, (b, name)) in found {
        match old.get(hash) {
            Some((was, _)) if was == b => {}
            Some((was, _)) => {
                changed += 1;
                println!("changed {hash} {name} ({was} -> {b})");
            }
            None => println!("new     {hash} {name} ({b})"),
        }
    }
    let missing: Vec<&String> = wanted.iter().filter(|h| !found.contains_key(*h)).collect();
    for h in &missing {
        println!("absent  {h} (not in the pack)");
    }
    println!("{} programs, {changed} changed behaviour, {} absent", found.len(), missing.len());
}

pub fn run(pack: &Path, golden: &Path, list: Option<&Path>, bless: bool, jobs: usize) -> Result<()> {
    let old = read_golden(golden);
    let wanted: HashSet<String> = match list {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("reading {}", p.display()))?
            .lines()
            .filter_map(|l| l.split_whitespace().next().map(str::to_owned))
            .collect(),
        None => old.keys().cloned().collect(),
    };
    anyhow::ensure!(!wanted.is_empty(), "no programs: give --list for a first --bless");
    let (found, slow) = collect(pack, &wanted, jobs)?;
    eprintln!("slowest (ms): {slow:?}");
    compare(&old, &found, &wanted);
    if bless {
        let mut text = String::from("# cargo xtask shader-golden: hash, behaviour in the interpreter, name (xtask/src/golden.rs)\n");
        for (hash, (b, name)) in &found {
            let _ = writeln!(text, "{hash} {b} {name}");
        }
        std::fs::write(golden, text).with_context(|| format!("writing {}", golden.display()))?;
        println!("blessed {}", golden.display());
    }
    Ok(())
}

/// Writes the container whose hash (first 16 hex digits of its blake3)
/// is `hash` to `out` (for `examples/replay`).
pub fn extract(pack: &Path, hash: &str, out: &Path) -> Result<()> {
    for path in files(pack)? {
        let data = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        for src in sources(&data) {
            let bytes = match &src {
                Source::Raw { data, .. } => Some(data.to_vec()),
                Source::Packed(r) => r.unpack(),
            };
            if let Some(bytes) = bytes.filter(|b| blake3::hash(b).to_hex().starts_with(hash)) {
                std::fs::write(out, &bytes).with_context(|| format!("writing {}", out.display()))?;
                println!("{} bytes to {}", bytes.len(), out.display());
                return Ok(());
            }
        }
    }
    anyhow::bail!("no container {hash} in {}", pack.display())
}
