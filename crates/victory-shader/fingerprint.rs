// What the translation fingerprint (build.rs) leaves out of `src`: files
// that cannot change an emitted module or its interface, so that editing
// them keeps the stage cache. Paths relative to `src`; a directory leaves
// out everything under it. `#[cfg(test)]` items of the other files are
// left out too (build.rs `without_tests`). Included by build.rs and by
// lib.rs's test that every entry exists.

/// The interpreter (offline checks: tests, examples, xtask), the oracle
/// tests, and the views a program reaches memory through (read from the
/// thread program for the host; no module depends on it).
pub const FINGERPRINT_EXCLUDE: &[&str] = &["vk/interp.rs", "vk/oracle", "vk/views.rs"];

/// Lines of fingerprinted files left out with a left-out file: its
/// declaration (file, line), so that adding the file keeps the cache.
/// (`pub mod interp;` is older than this list and stays in the
/// fingerprint: taking it out would change every key once.)
pub const FINGERPRINT_EXCLUDE_LINES: &[(&str, &str)] = &[("vk/mod.rs", "pub mod views;")];
