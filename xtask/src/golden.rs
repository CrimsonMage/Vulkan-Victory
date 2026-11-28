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
use victory_shader::vk::interp_group::{Group, Mode};
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

/// A stop's text without the value numbers an error names (`%716`): they
/// shift with any change of the lowering, the behaviour does not.
fn unnumbered(stop: &str) -> String {
    let mut out = String::with_capacity(stop.len());
    let mut after_percent = false;
    for c in stop.chars() {
        if !(after_percent && c.is_ascii_digit()) {
            out.push(c);
            after_percent = c == '%';
        }
    }
    out
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
        let mut s = format!("{}\n", unnumbered(&format!("{stop:?}")));
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

/// `what` of each wanted program of the pack (its behaviour), and the slowest.
fn collect(pack: &Path, wanted: &HashSet<String>, jobs: usize, what: impl Fn(&[u8]) -> String + Sync + std::panic::RefUnwindSafe) -> Result<(Found, Vec<(u128, String)>)> {
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
                let b = std::panic::catch_unwind(|| what(&bytes)).unwrap_or_else(|_| "panic".into());
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

pub fn run(pack: &Path, golden: &Path, list: Option<&Path>, bless: bool, halves: bool, jobs: usize) -> Result<()> {
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
    if halves {
        return run_halves(pack, &wanted, jobs);
    }
    let (found, slow) = collect(pack, &wanted, jobs, behaviour)?;
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

/// How a container's compute program reaches title memory: the
/// container's hash as the stage cache names it (32 hex digits) and its
/// views' line (`victory_shader::vk::views`); empty for another stage or
/// a refused program.
fn views_text(container: &[u8]) -> String {
    let Ok((f, _, native)) = thread_ir(container) else { return String::new() };
    if native.stage != victory_shader::HwStage::Cs {
        return String::new();
    }
    format!("{} {}", &blake3::hash(container).to_hex()[..32], victory_shader::vk::views::views(&f).line())
}

/// Prints, for each compute program of the golden file, the views it
/// stores and loads through (`victory_shader::vk::views`: hash, the
/// views' line, name), and how many programs the host can stage. With
/// `out`: the lines as the host keeps them beside its translations
/// (`views-<key>.txt`; the game's log names the folder and the key).
pub fn views(pack: &Path, golden: &Path, out: Option<&Path>, jobs: usize) -> Result<()> {
    let list = std::fs::read_to_string(golden).with_context(|| format!("reading {}", golden.display()))?;
    let hashes = list.lines().filter(|l| !l.starts_with('#')).filter_map(|l| l.split_whitespace().next());
    let wanted: HashSet<String> = hashes.map(|h| h.chars().take(16).collect()).collect();
    let (found, _) = collect(pack, &wanted, jobs, views_text)?;
    let compute: Vec<(&str, &str)> = found.values().filter(|(text, _)| !text.is_empty()).map(|(text, name)| (text.as_str(), name.as_str())).collect();
    for (text, name) in &compute {
        println!("{text} {name}");
    }
    let parsed: Vec<_> = compute.iter().filter_map(|(text, _)| victory_shader::vk::views::Views::parse(text.split_once(' ')?.1)).collect();
    let storing = parsed.iter().filter(|v| v.other_stores || v.stores().next().is_some()).count();
    let staged = parsed.iter().filter(|v| v.resolved() && v.stores().next().is_some()).count();
    println!("{} compute programs, {storing} store into title memory, {staged} of them through views found whole", compute.len());
    if let Some(out) = out {
        let text: String = compute.iter().map(|(text, _)| format!("{text}\n")).collect();
        std::fs::write(out, text).with_context(|| format!("writing {}", out.display()))?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

/// What a whole group of a program does with the inputs of `seed`, run as
/// console waves or as the host subgroups the card runs (`upper_first`: the
/// upper half of each wave64 before the lower between their meetings):
/// how it stopped, and its outputs, image stores and memory stores as text.
fn group_run(f: &victory_shader::vk::tir::Func, native: &victory_shader::NativeProgram, seed: u64, mode: Mode, upper_first: bool) -> (interp::Stop, String, [u64; 2]) {
    let mem = interp::Memory { range: (0, u64::MAX), noise: Some(seed), ..Default::default() };
    let setup = |m: &mut interp::Machine| {
        for (k, p) in m.push.iter_mut().enumerate() {
            *p = mix(seed << 8 | k as u64) as u32;
        }
        m.group = [(seed % 3) as u32, (seed % 2) as u32, 0];
        m.limit = LIMIT;
        m.image_size = [64, 64, 1, 1];
        m.image_texel = [0, 1, 2, 3].map(|c| u64::from(((mix(seed * 4 + c) >> 40) as f32 / (1u64 << 24) as f32).to_bits()));
    };
    let mut g = Group::new(f, native.wave64, victory_shader::vk::host_threads(native), mode, mem, setup);
    g.upper_first = upper_first;
    let stop = g.run();
    let mut lines = Vec::new();
    for (u, m) in g.units.iter().enumerate() {
        // (an invocation by its place in the group, whatever the units are)
        let base = u * m.lanes();
        lines.extend(m.outputs.iter().map(|(k, v)| format!("o {} {} {} {} {v:x}", k.0, k.1, k.2, base + k.3)));
        lines.extend(m.image_stores.iter().map(|(l, args)| format!("i {} {args:x?}", base + l)));
    }
    lines.sort();
    lines.extend(g.mem.stored.iter().map(|(a, v)| format!("m {a:x} {v:x}")));
    (stop, lines.join("\n"), [g.races, g.lds_races])
}

/// How a wave64 compute or mesh program differs when its group runs as the
/// card's host subgroups, the halves exchanging where the translation says
/// so, from the same group run as console waves: "same", "limit" (a run
/// went past the block limit: nothing to compare), "same-but-raced" (in
/// some run the halves stored to a dword the other half accessed between
/// two meetings, as random descriptors make them; the other runs agree), or
/// the first difference. Such a dword of workgroup memory is a difference
/// too: the lift's meetings before LDS accesses (`vk::meetings`) leave none.
/// None for the programs with nothing to tell apart (no wave64 group).
pub fn halves(container: &[u8]) -> Option<String> {
    let (f, _, native) = thread_ir(container).ok()?;
    let grouped = matches!(native.stage, victory_shader::HwStage::Cs | victory_shader::HwStage::Ms);
    if !(native.wave64 && grouped && victory_shader::vk::host_threads(&native) > 32) {
        return None;
    }
    let (mut limit, mut raced) = (false, false);
    for seed in 1..=SEEDS {
        let (stop, wave, _) = group_run(&f, &native, seed, Mode::Waves, false);
        for upper_first in [false, true] {
            let (hstop, halves, [races, lds_races]) = group_run(&f, &native, seed, Mode::Halves, upper_first);
            if stop == interp::Stop::Limit || hstop == interp::Stop::Limit {
                limit = true;
                continue;
            }
            if lds_races > 0 {
                return Some(format!("seed {seed}, upper first {upper_first}: {lds_races} LDS dwords the halves accessed with no meeting between"));
            }
            // random memory: the halves stored to one dword between two
            // meetings (descriptors that collide), in an order the card
            // does not keep: nothing to compare in this run
            if races > 0 {
                raced = true;
                continue;
            }
            if let Some(d) = difference((&stop, &wave), (&hstop, &halves)) {
                return Some(format!("seed {seed}, upper first {upper_first}: {d}"));
            }
        }
    }
    Some(if raced { "same-but-raced" } else if limit { "limit" } else { "same" }.to_owned())
}

/// How a group's run as halves (how it stopped, its effects) differs from
/// its run as waves, if it does.
fn difference(wave: (&interp::Stop, &str), halves: (&interp::Stop, &str)) -> Option<String> {
    if wave.0 != halves.0 {
        return Some(format!("the wave {}, the halves {}", unnumbered(&format!("{:?}", wave.0)), unnumbered(&format!("{:?}", halves.0))));
    }
    if let Some((a, b)) = wave.1.lines().zip(halves.1.lines()).find(|(a, b)| a != b) {
        return Some(format!("the wave `{a}`, the halves `{b}`"));
    }
    let (n, m) = (wave.1.lines().count(), halves.1.lines().count());
    (n != m).then(|| format!("{n} effects as a wave, {m} as halves"))
}

/// Every wanted wave64 compute or mesh program's group as console waves and
/// as the card's halves ([`halves`]): the programs that differ.
pub fn run_halves(pack: &Path, wanted: &HashSet<String>, jobs: usize) -> Result<()> {
    let (found, slow) = collect(pack, wanted, jobs, |b| halves(b).unwrap_or_default())?;
    eprintln!("slowest (ms): {slow:?}");
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (hash, (b, name)) in &found {
        let kind = match b.as_str() {
            "" => "no wave64 group",
            "same" => "same",
            "limit" | "panic" | "same-but-raced" => {
                println!("{b} {hash} {name}");
                b.as_str()
            }
            _ => {
                println!("differs {hash} {name}: {b}");
                "differs"
            }
        };
        *counts.entry(kind).or_default() += 1;
    }
    println!("{} programs: {counts:?}", found.len());
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
