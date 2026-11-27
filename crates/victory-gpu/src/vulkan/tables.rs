//! The tables buffer (`docs/vulkan-abi.md` section 3.1): the constant
//! tables console programs keep after their own code
//! (`VkShader::tables`), as memory the translated programs read by
//! address.
//!
//! As variables of the modules they were kept on the card by the driver
//! for 32,768 threads, outside every heap's count and for good: one
//! program's nine tables, 47.7 KiB, took 1.7 GiB. Here they are their own
//! bytes, once: one buffer in host-visible memory for every program, its
//! address in the memory map (`map_block`), a program's place in it in
//! specialization constant 1 (`stages::compile`). The places are kept
//! beside the translations (`victory_vk::tables`, `code-tables.txt`), so a
//! program is specialized the same in every run and by `victory bake`, and
//! the driver's cache holds its pipeline.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use anyhow::Context;
use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};
use victory_vk::tables::Places;

use super::{Backend, lock};

/// Bytes of the buffer. The 662 programs of the title's pack that have
/// tables (of 25,889) hold 300 KiB of them, 139 KiB as the 217 distinct
/// sets the buffer keeps once each.
const BYTES: u32 = 2 << 20;

fn buffer(b: &Backend) -> Option<&'static Buffer> {
    static BUFFER: OnceLock<Option<Buffer>> = OnceLock::new();
    BUFFER
        .get_or_init(|| match Buffer::new(&b.host, u64::from(BYTES), vk::BufferUsageFlags::STORAGE_BUFFER, Place::Upload) {
            Ok(buf) => {
                tracing::info!(address = format_args!("{:#x}", buf.address), kib = BYTES >> 10, "Vulkan: the tables buffer (programs' code tables, read by address)");
                Some(buf)
            }
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "Vulkan: no tables buffer; programs with code tables do not compile");
                None
            }
        })
        .as_ref()
}

/// The buffer's device address, for the memory map (0: no buffer).
pub(super) fn address(b: &Backend) -> u64 {
    buffer(b).map_or(0, |buf| buf.address)
}

/// The place of a program's tables in the buffer (bytes from its start),
/// with the tables there: before the stage that reads them is compiled.
pub(super) fn place(b: &Backend, words: &[u32]) -> anyhow::Result<u32> {
    static PLACES: OnceLock<Mutex<(Places, HashSet<u32>)>> = OnceLock::new();
    let buf = buffer(b).context("no tables buffer")?;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let key = &blake3::hash(&bytes).to_hex()[..32];
    let places = PLACES.get_or_init(|| {
        let lines = super::stage_cache::code_table_lines();
        let mut places = Places::load(BYTES, lines.iter().map(String::as_str));
        tracing::info!(kib = places.used() >> 10, "Vulkan: places of code tables given by earlier runs");
        // Places are never taken back (tables of translators long gone
        // keep theirs): half full, the list starts again, which costs a
        // compile of each program with tables and leaves room for all.
        if places.used() > BYTES / 2 {
            tracing::info!("Vulkan: the places of code tables start again (the list had filled half the buffer)");
            super::stage_cache::forget_code_tables();
            places = Places::load(BYTES, []);
        }
        Mutex::new((places, HashSet::new()))
    });
    let mut p = lock(places);
    let (at, line) = p.0.place(key, bytes.len() as u32).context("the tables buffer is full")?;
    if let Some(line) = line {
        super::stage_cache::note_code_table(&line);
    }
    if p.1.insert(at) {
        // SAFETY: `at + len` lies inside the buffer's mapping (`Places`
        // gives no other place); no program reads these bytes yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.mapped.add(at as usize), bytes.len()) };
    }
    Ok(at)
}
