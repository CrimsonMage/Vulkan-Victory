//! `victory bake`: the title's pipelines compiled before it runs, the way
//! a PC game compiles its shaders at first start. The work itself is
//! `victory_gpu::pipeline::bake`, the same the game does in the background
//! as the title starts, into the same caches.

use std::path::PathBuf;

use anyhow::Result;

#[cfg(windows)]
pub fn run(title: Option<PathBuf>, jobs: Option<usize>) -> Result<()> {
    use std::collections::HashSet;
    use victory_hle::modules::d3d12x;
    let cfg = victory_core::Config::load()?;
    d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    // Shaders of the catalog the translated-shader cache lacks (a new
    // translator): translated from the title's files first.
    let missing: HashSet<[u8; 32]> = d3d12x::bake_missing_shaders().into_iter().collect();
    if !missing.is_empty() {
        let root = match title.or_else(|| cfg.game_dir.clone()) {
            Some(p) if p.is_file() => p.parent().map(PathBuf::from),
            other => other,
        };
        match root {
            Some(root) => {
                println!(
                    "bake         {} shaders to translate first, from {}",
                    missing.len(),
                    root.display()
                );
                let pick = |hash: &blake3::Hash| missing.contains(hash.as_bytes());
                let threads = jobs.unwrap_or_else(|| {
                    std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1))
                });
                // The shader archive first (one file read at a time, each
                // whole); the rest of the title only for what is still
                // missing.
                let archive = root.join("shaders.rpf");
                if archive.is_file() {
                    crate::translate::warm_matching(
                        &[archive],
                        &pick,
                        &missing.len().to_string(),
                        threads,
                    )?;
                }
                let still: HashSet<[u8; 32]> = d3d12x::bake_missing_shaders().into_iter().collect();
                if !still.is_empty() {
                    let pick = |hash: &blake3::Hash| still.contains(hash.as_bytes());
                    crate::translate::warm_matching(
                        &[root],
                        &pick,
                        &still.len().to_string(),
                        threads,
                    )?;
                }
            }
            None => println!(
                "bake         {} shaders have no translation and no title folder is known (--title): their pipelines are made when the game asks",
                missing.len()
            ),
        }
    }
    let baked = d3d12x::bake(jobs).map_err(anyhow::Error::msg)?;
    println!(
        "bake         done in {:.1} s: {} pipelines made, {} refused by the host, {} without a translation, {} made before",
        baked.seconds, baked.made, baked.refused, baked.untranslated, baked.cached_before
    );
    Ok(())
}

#[cfg(not(windows))]
pub fn run(_: Option<PathBuf>, _: Option<usize>) -> Result<()> {
    anyhow::bail!("the bake needs the Windows build (a D3D12 device)")
}
