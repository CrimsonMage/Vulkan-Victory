//! The semantic registry: the location of each interface semantic (name,
//! index), the same in every program of a title so that any vertex or mesh
//! program links with any pixel program without rewriting either.
//!
//! A semantic gets its location the first time a program uses it: a
//! location no semantic holds yet, else (more than 31 semantics in the
//! title) the lowest one no other semantic of that program holds. The
//! registry only grows, so a module translated earlier keeps its
//! locations; the host saves it next to the stage cache ([`Semantics::save`])
//! and loads it before translating ([`Semantics::load`]). A program two of
//! whose semantics share a location is refused.

use std::sync::{Mutex, OnceLock};

/// Version of the saved form (part of the stage cache key).
pub const SEMANTICS_VERSION: u32 = 1;

/// User locations: 0..=30 (31 stays free).
const LOCATIONS: u8 = 31;

/// A seed that fits the first title's 37 semantics in 27 locations
/// (coloured offline so that no two semantics one program uses share one).
const SEED: &[(&str, u32, u8)] = &[
    ("TEXCOORD", 0, 0),
    ("TEXCOORD", 1, 1),
    ("TEXCOORD", 3, 2),
    ("COLOR", 0, 3),
    ("TEXCOORD", 4, 4),
    ("COLOR", 1, 5),
    ("TEXCOORD", 2, 6),
    ("TEXCOORD", 9, 7),
    ("TEXCOORD", 6, 8),
    ("TEXCOORD", 5, 9),
    ("TEXCOORD", 10, 10),
    ("TEXCOORD", 7, 11),
    ("TEXCOORD", 8, 12),
    ("COLOR", 2, 13),
    ("TEXCOORD", 15, 14),
    ("TEXCOORD", 11, 15),
    ("TEXCOORD", 20, 16),
    ("TEXCOORD", 21, 17),
    ("TEXCOORD", 13, 18),
    ("TEXCOORD", 17, 19),
    ("TEXCOORD", 14, 20),
    ("COLOR", 3, 21),
    ("TEXCOORD", 19, 22),
    ("TEXCOORD", 22, 23),
    ("TEXCOORD", 12, 24),
    ("TEXCOORD", 16, 25),
    ("TEXCOORD", 23, 26),
    ("TEXCOORD", 18, 7),
    ("NORMAL", 2, 18),
    ("NORMAL", 0, 3),
    ("TANGENT", 0, 10),
    ("TEXCOORD", 24, 15),
    ("NORMAL", 1, 5),
    ("POSITION", 1, 7),
    ("INSTANCEID", 0, 6),
    ("TANGENT", 1, 3),
    ("POSITION", 0, 1),
];

/// (NAME, index, location), names upper-cased.
pub struct Semantics {
    table: Mutex<Vec<(String, u32, u8)>>,
}

impl Semantics {
    /// The seeded registry.
    pub fn new() -> Self {
        let table = SEED.iter().map(|&(n, i, l)| (n.to_owned(), i, l)).collect();
        Semantics { table: Mutex::new(table) }
    }

    /// The process-wide registry [`super::translate`] uses.
    pub fn global() -> &'static Semantics {
        static G: OnceLock<Semantics> = OnceLock::new();
        G.get_or_init(Semantics::new)
    }

    /// Adds the saved lines (`NAME index location`); an entry that names a
    /// semantic the registry has at another location is an error (the
    /// saved registry and the cache it goes with no longer agree).
    pub fn load(&self, text: &str) -> Result<(), String> {
        let mut t = self.table.lock().unwrap();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let f: Vec<&str> = line.split_whitespace().collect();
            let parsed = match f[..] {
                [n, i, l] => i.parse::<u32>().ok().zip(l.parse::<u8>().ok().filter(|&l| l < LOCATIONS)).map(|(i, l)| (n, i, l)),
                _ => None,
            };
            let (n, i, l) = parsed.ok_or_else(|| format!("semantic registry: a bad line {line:?}"))?;
            let n = n.to_ascii_uppercase();
            match t.iter().find(|e| e.0 == n && e.1 == i) {
                Some(e) if e.2 != l => return Err(format!("semantic registry: {n}{i} at {l}, here at {}", e.2)),
                Some(_) => {}
                None => t.push((n, i, l)),
            }
        }
        Ok(())
    }

    /// The saved form: one line per semantic.
    pub fn save(&self) -> String {
        let t = self.table.lock().unwrap();
        let mut out = format!("# victory semantic registry v{SEMANTICS_VERSION}\n");
        for (n, i, l) in t.iter() {
            out.push_str(&format!("{n} {i} {l}\n"));
        }
        out
    }

    /// The locations of one program's semantics, assigning the new ones.
    pub fn assign(&self, used: &[(&str, u32)]) -> Result<Vec<u8>, &'static str> {
        let mut t = self.table.lock().unwrap();
        let find = |t: &[(String, u32, u8)], n: &str, i: u32| t.iter().find(|e| e.0.eq_ignore_ascii_case(n) && e.1 == i).map(|e| e.2);
        let mut out = Vec::new();
        for &(n, i) in used {
            let l = match find(&t, n, i) {
                Some(l) => l,
                None => {
                    let taken = |l: u8| out.contains(&l) || used.iter().any(|&(m, j)| find(&t, m, j) == Some(l));
                    let fresh = (0..LOCATIONS).find(|&l| !t.iter().any(|e| e.2 == l));
                    let l = fresh.or_else(|| (0..LOCATIONS).find(|&l| !taken(l))).ok_or("more than 31 user semantics")?;
                    t.push((n.to_ascii_uppercase(), i, l));
                    l
                }
            };
            out.push(l);
        }
        // the same semantic listed twice shares its location with itself
        let clash = (0..used.len()).any(|a| {
            (0..a).any(|b| out[a] == out[b] && !(used[a].0.eq_ignore_ascii_case(used[b].0) && used[a].1 == used[b].1))
        });
        if clash {
            return Err("two semantics of the program share a location in the registry");
        }
        Ok(out)
    }
}

impl Default for Semantics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_fresh_then_stable() {
        let s = Semantics::new();
        let a = s.assign(&[("TEXCOORD", 0), ("FOO", 0), ("FOO", 1)]).unwrap();
        assert_eq!(a[0], 0);
        assert!(a[1] >= 27 && a[2] >= 27 && a[1] != a[2]);
        let b = s.assign(&[("foo", 1)]).unwrap();
        assert_eq!(b[0], a[2]);
        let t = Semantics::new();
        t.load(&s.save()).unwrap();
        assert_eq!(t.assign(&[("FOO", 0)]).unwrap()[0], a[1]);
    }

    #[test]
    fn refuses_a_shared_location() {
        let s = Semantics::new();
        // the seed puts TEXCOORD9 and TEXCOORD18 at 7
        assert!(s.assign(&[("TEXCOORD", 9), ("TEXCOORD", 18)]).is_err());
    }
}
