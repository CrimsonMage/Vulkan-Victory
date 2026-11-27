// What the translation fingerprint (build.rs) leaves out of `src`: files
// that cannot change an emitted module or its interface, so that editing
// them keeps the stage cache. Paths relative to `src`; a directory leaves
// out everything under it. `#[cfg(test)]` items of the other files are
// left out too (build.rs `without_tests`). Included by build.rs and by
// lib.rs's test that every entry exists.

/// The interpreter (offline checks: tests, examples, xtask) and the oracle
/// tests.
pub const FINGERPRINT_EXCLUDE: &[&str] = &["vk/interp.rs", "vk/oracle"];
