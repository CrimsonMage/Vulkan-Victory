//! Fingerprint of the translator's own source, for the stage cache key
//! (`victory-gpu`'s `vulkan/stage_cache.rs`): a rebuilt executable whose translator did
//! not change keeps its cached translations. Also lists every `VICTORY_*`
//! switch the translator reads, whose values join the key at run time.
//! Reads only this crate's `src` folder; writes nothing but cargo directives.
//!
//! Only what can change an emitted module or its interface counts: the
//! files of `fingerprint.rs` (the interpreter, the oracle tests) and every
//! `#[cfg(test)]` item are left out, so test and offline-tool edits keep
//! the cache. (Examples and xtask are outside `src` anyway.)

use std::path::{Path, PathBuf};

include!("fingerprint.rs");

/// `text` without its `#[cfg(test)]` items (the item after each such
/// attribute line, to its `;` (a `mod x;`) or its matching closing brace)
/// and without blank lines.
fn without_tests(text: &str) -> String {
    let mut out = String::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.trim().is_empty() {
            continue;
        }
        if line.trim() != "#[cfg(test)]" {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let mut depth = 0i64;
        for item in lines.by_ref() {
            depth += item.matches('{').count() as i64 - item.matches('}').count() as i64;
            if depth <= 0 && (item.contains('}') || item.trim_end().ends_with(';')) {
                break;
            }
        }
    }
    out
}

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// FNV-1a, 64 bits, two lanes with different offsets: a mismatch only costs
/// a cache miss, so this need not be cryptographic.
fn fnv(state: &mut [u64; 2], bytes: &[u8]) {
    for &b in bytes {
        for (i, s) in state.iter_mut().enumerate() {
            *s ^= u64::from(b) ^ (i as u64 * 0x9e);
            *s = s.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let mut list = Vec::new();
    files(&src, &mut list);
    list.sort();
    let mut state = [0xcbf2_9ce4_8422_2325u64, 0x8422_2325_cbf2_9ce4u64];
    let mut switches = std::collections::BTreeSet::new();
    println!("cargo:rerun-if-changed=fingerprint.rs");
    for p in &list {
        let rel = p.strip_prefix(&src).unwrap_or(p).to_string_lossy().replace('\\', "/");
        if FINGERPRINT_EXCLUDE.iter().any(|x| rel == *x || rel.starts_with(&format!("{x}/"))) {
            continue;
        }
        let text = without_tests(&String::from_utf8_lossy(&std::fs::read(p).unwrap_or_default())).into_bytes();
        fnv(&mut state, rel.as_bytes());
        fnv(&mut state, &(text.len() as u64).to_le_bytes());
        fnv(&mut state, &text);
        let s = String::from_utf8_lossy(&text);
        let mut rest = &s[..];
        while let Some(at) = rest.find("\"VICTORY_") {
            let tail = &rest[at + 1..];
            let end = tail
                .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                .unwrap_or(tail.len());
            if tail[end..].starts_with('"') {
                switches.insert(tail[..end].to_string());
            }
            rest = &tail[end..];
        }
    }
    println!("cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={:016x}{:016x}", state[0], state[1]);
    let joined: Vec<_> = switches.into_iter().collect();
    println!("cargo:rustc-env=VICTORY_SHADER_SWITCHES={}", joined.join(","));
}
