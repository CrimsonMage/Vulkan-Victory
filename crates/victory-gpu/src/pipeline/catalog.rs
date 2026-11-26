//! The pipeline catalog: every title pipeline a draw or dispatch has used,
//! with what it takes to make its host pipeline again without the title
//! (its shaders by hash, its graphics state), kept beside the host's
//! pipeline library. The bake (`super::bake`) makes these before the title
//! asks for them, in the order an earlier run first used them (intro,
//! menus, world start, later), and the title's creation of one puts it
//! ahead of the rest (`super::make::created`).
//!
//! Only pipelines a run drew or dispatched with are listed: the title
//! creates tens of thousands of pipelines while it loads and uses a few
//! thousand. One unused for [`FORGET_AFTER`] runs is dropped. The shaders
//! themselves are not kept here: their translations are in the
//! translated-shader cache under the container's hash
//! (`crate::shader_cache::by_hash`), the title's containers in its files.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// What a host pipeline is made from, by description: each stage's
/// shader (stage, container hash, program name for logs) and the encoded
/// graphics state (`host::GraphicsState::encode`; none for compute).
pub(crate) struct Desc {
    pub(crate) shaders: Vec<(&'static str, [u8; 32], String)>,
    pub(crate) graphics: Option<Vec<u8>>,
}

struct Entry {
    /// The title frame of its first use (the smallest seen over runs).
    first_frame: u64,
    /// The last run (game launch) that used it.
    last_run: u32,
    /// What the bake or the title last made it with ([`super::bake::stamp`]);
    /// zeros: never made with a catalog of this kind.
    made_with: [u8; 32],
    /// `None` for keys carried over from the old early list, until used.
    desc: Option<Arc<Desc>>,
}

struct Catalog {
    /// Game launches so far (the bake alone does not count).
    run: u32,
    entries: HashMap<String, Entry>,
    changed: bool,
}

const FILE: &str = "pipeline-catalog.bin";
/// The list of pipeline keys builds before the catalog kept: read once,
/// for the keys alone (made at the title's creation, described when used).
const OLD_EARLY_FILE: &str = "early-pipelines.txt";
const MAGIC: &[u8; 4] = b"VPC1";
/// Pipelines unused for this many launches are dropped.
const FORGET_AFTER: u32 = 5;
/// Stage names as stored (index).
const STAGES: [&str; 8] = [
    "vertex",
    "pixel",
    "domain",
    "hull",
    "geometry",
    "compute",
    "amplification",
    "mesh",
];

static CATALOG: Mutex<Option<Catalog>> = Mutex::new(None);

fn catalog() -> MutexGuard<'static, Option<Catalog>> {
    let mut held = super::lock(&CATALOG);
    if held.is_none() {
        let dir = super::library::dir();
        let loaded = dir
            .as_ref()
            .and_then(|d| std::fs::read(d.join(FILE)).ok())
            .and_then(|b| decode(&b));
        let catalog = match loaded {
            Some(c) => c,
            None => {
                // Keys only, from the list builds before kept.
                let old = dir
                    .and_then(|d| std::fs::read_to_string(d.join(OLD_EARLY_FILE)).ok())
                    .unwrap_or_default();
                let entries = old
                    .lines()
                    .filter(|l| !l.is_empty())
                    .map(|k| {
                        (
                            k.to_owned(),
                            Entry {
                                first_frame: u64::MAX,
                                last_run: 0,
                                made_with: [0; 32],
                                desc: None,
                            },
                        )
                    })
                    .collect();
                Catalog {
                    run: 0,
                    entries,
                    changed: true,
                }
            }
        };
        let described = catalog
            .entries
            .values()
            .filter(|e| e.desc.is_some())
            .count();
        tracing::info!(
            pipelines = catalog.entries.len(),
            described,
            run = catalog.run,
            "pipeline catalog"
        );
        *held = Some(catalog);
    }
    held
}

/// A launch of the title begins (counts for [`FORGET_AFTER`]).
pub(crate) fn begin_run() {
    if let Some(c) = catalog().as_mut() {
        c.run += 1;
        c.changed = true;
    }
}

/// Whether a run has used the title pipeline with `key`.
pub(crate) fn known(key: &str) -> bool {
    catalog()
        .as_ref()
        .is_some_and(|c| c.entries.contains_key(key))
}

/// A draw or dispatch used the title pipeline with `key` for the first
/// time this run, in title frame `frame`; `desc` describes it.
pub(crate) fn note_used(key: &str, frame: u64, desc: impl FnOnce() -> Option<Desc>) {
    let mut held = catalog();
    let Some(c) = held.as_mut() else { return };
    let run = c.run;
    let entry = c.entries.entry(key.to_owned()).or_insert_with(|| Entry {
        first_frame: frame,
        last_run: run,
        made_with: [0; 32],
        desc: None,
    });
    entry.first_frame = entry.first_frame.min(frame);
    entry.last_run = run;
    if entry.desc.is_none() {
        entry.desc = desc().map(Arc::new);
    }
    c.changed = true;
}

/// The title pipeline with `key` got a host pipeline made with `stamp`
/// (by the bake or for the title).
pub(crate) fn note_made(key: &str, stamp: &[u8; 32]) {
    let mut held = catalog();
    if let Some(e) = held.as_mut().and_then(|c| c.entries.get_mut(key))
        && e.made_with != *stamp
    {
        e.made_with = *stamp;
        if let Some(c) = held.as_mut() {
            c.changed = true;
        }
    }
}

/// Whether the title pipeline with `key` was made with `stamp` already:
/// the host pipeline is then in the caches.
pub(crate) fn made_with(key: &str, stamp: &[u8; 32]) -> bool {
    catalog()
        .as_ref()
        .and_then(|c| c.entries.get(key))
        .is_some_and(|e| e.made_with == *stamp)
}

/// The described pipelines, in the order of first use, and how many are
/// listed in all: those not yet made with `stamp` to bake, and the count
/// of those that were (skipped).
pub(crate) fn to_bake(stamp: &[u8; 32]) -> (Vec<(String, Arc<Desc>)>, usize) {
    let held = catalog();
    let Some(c) = held.as_ref() else {
        return (Vec::new(), 0);
    };
    let mut skipped = 0;
    let mut work: Vec<(u64, String, Arc<Desc>)> = Vec::new();
    for (key, e) in &c.entries {
        let Some(desc) = &e.desc else { continue };
        if e.made_with == *stamp {
            skipped += 1;
        } else {
            work.push((e.first_frame, key.clone(), desc.clone()));
        }
    }
    work.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    (work.into_iter().map(|(_, k, d)| (k, d)).collect(), skipped)
}

/// The container hashes of the described pipelines' shaders.
pub(crate) fn shaders() -> Vec<[u8; 32]> {
    let held = catalog();
    let Some(c) = held.as_ref() else {
        return Vec::new();
    };
    let mut all: Vec<[u8; 32]> = c
        .entries
        .values()
        .filter_map(|e| e.desc.as_ref())
        .flat_map(|d| d.shaders.iter().map(|s| s.1))
        .collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// Writes the catalog to `dir` if it changed, without the pipelines no
/// run used for [`FORGET_AFTER`] launches.
pub(crate) fn save(dir: &std::path::Path) -> std::io::Result<usize> {
    let bytes = {
        let mut held = catalog();
        let Some(c) = held.as_mut() else { return Ok(0) };
        if !c.changed {
            return Ok(c.entries.len());
        }
        let run = c.run;
        c.entries
            .retain(|_, e| run.saturating_sub(e.last_run) < FORGET_AFTER);
        c.changed = false;
        encode(c)
    };
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap_or_default()) as usize;
    super::library::write_atomically(&dir.join(FILE), &bytes)?;
    Ok(n)
}

fn encode(c: &Catalog) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend(c.run.to_le_bytes());
    out.extend((c.entries.len() as u32).to_le_bytes());
    let mut keys: Vec<&String> = c.entries.keys().collect();
    keys.sort();
    for key in keys {
        let e = &c.entries[key];
        out.push(key.len() as u8);
        out.extend(key.as_bytes());
        out.extend(e.first_frame.to_le_bytes());
        out.extend(e.last_run.to_le_bytes());
        out.extend(e.made_with);
        match &e.desc {
            None => out.push(0),
            Some(d) => {
                out.push(1);
                out.push(d.shaders.len() as u8);
                for (stage, full, name) in &d.shaders {
                    out.push(STAGES.iter().position(|s| s == stage).unwrap_or(0) as u8);
                    out.extend(full);
                    let name = &name.as_bytes()[..name.len().min(255)];
                    out.push(name.len() as u8);
                    out.extend(name);
                }
                let g = d.graphics.as_deref().unwrap_or_default();
                out.push(u8::from(d.graphics.is_some()));
                out.extend((g.len() as u32).to_le_bytes());
                out.extend(g);
            }
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<Catalog> {
    let mut r = bytes;
    let mut take = |n: usize| -> Option<&[u8]> {
        let (head, rest) = r.split_at_checked(n)?;
        r = rest;
        Some(head)
    };
    if take(4)? != MAGIC {
        return None;
    }
    let word = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap_or_default());
    let run = word(take(4)?);
    let count = word(take(4)?) as usize;
    let mut entries = HashMap::with_capacity(count.min(1 << 16));
    for _ in 0..count {
        let len = take(1)?[0] as usize;
        let key = String::from_utf8(take(len)?.to_vec()).ok()?;
        let first_frame = u64::from_le_bytes(take(8)?.try_into().ok()?);
        let last_run = word(take(4)?);
        let made_with: [u8; 32] = take(32)?.try_into().ok()?;
        let desc = match take(1)?[0] {
            0 => None,
            _ => {
                let n = take(1)?[0] as usize;
                let mut shaders = Vec::with_capacity(n);
                for _ in 0..n {
                    let stage = *STAGES.get(take(1)?[0] as usize)?;
                    let full: [u8; 32] = take(32)?.try_into().ok()?;
                    let len = take(1)?[0] as usize;
                    let name = String::from_utf8_lossy(take(len)?).into_owned();
                    shaders.push((stage, full, name));
                }
                let has = take(1)?[0] != 0;
                let len = word(take(4)?) as usize;
                let g = take(len)?.to_vec();
                Some(Arc::new(Desc {
                    shaders,
                    graphics: has.then_some(g),
                }))
            }
        };
        entries.insert(
            key,
            Entry {
                first_frame,
                last_run,
                made_with,
                desc,
            },
        );
    }
    r.is_empty().then_some(Catalog {
        run,
        entries,
        changed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalog_round_trips() {
        let mut entries = HashMap::new();
        entries.insert(
            "k1".to_owned(),
            Entry {
                first_frame: 7,
                last_run: 3,
                made_with: [9; 32],
                desc: Some(Arc::new(Desc {
                    shaders: vec![
                        ("vertex", [1; 32], "VS_A".into()),
                        ("pixel", [2; 32], String::new()),
                    ],
                    graphics: Some(vec![1, 2, 3]),
                })),
            },
        );
        entries.insert(
            "k2".to_owned(),
            Entry {
                first_frame: u64::MAX,
                last_run: 0,
                made_with: [0; 32],
                desc: None,
            },
        );
        entries.insert(
            "k3".to_owned(),
            Entry {
                first_frame: 1,
                last_run: 4,
                made_with: [0; 32],
                desc: Some(Arc::new(Desc {
                    shaders: vec![("compute", [3; 32], "CS".into())],
                    graphics: None,
                })),
            },
        );
        let c = Catalog {
            run: 4,
            entries,
            changed: true,
        };
        let bytes = encode(&c);
        let back = decode(&bytes).expect("decodes");
        assert_eq!(back.run, 4);
        assert_eq!(back.entries.len(), 3);
        let k1 = &back.entries["k1"];
        assert_eq!((k1.first_frame, k1.last_run, k1.made_with), (7, 3, [9; 32]));
        let d = k1.desc.as_ref().unwrap();
        assert_eq!(d.shaders[0], ("vertex", [1; 32], "VS_A".to_owned()));
        assert_eq!(d.graphics.as_deref(), Some(&[1u8, 2, 3][..]));
        assert!(back.entries["k2"].desc.is_none());
        assert!(back.entries["k3"].desc.as_ref().unwrap().graphics.is_none());
        assert!(decode(&bytes[..bytes.len() - 1]).is_none());
    }
}
