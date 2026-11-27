//! Diagnostics (`VICTORY_VK_CENSUS=<file>`, diagnostics builds): what the
//! title says of its textures, one line an event, for reading offline how
//! it streams them:
//!
//! - `L`: a texture shape seen first: its console layout (bytes, the block
//!   of its small levels, the first level in that block, where each
//!   larger level starts).
//! - `V`: a texture view written into a title slot: the frame, the slot,
//!   the texture's address and shape, the view's words (the plain ten and
//!   the console's four), and which 64 KiB pages under its layout the
//!   title has mapped (a bit a page, the first page lowest; `r`: memory
//!   that is no pool page).
//! - `W`: an announced write (a storage read's destination, a title copy).
//! - `M`: a page-mapping call: where, pages mapped, pages unmapped.
//! - `D`: an image dropped, and why.

use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

use crate::resource::ResourceDesc;

type Out = Mutex<(std::io::BufWriter<std::fs::File>, std::collections::HashSet<[u32; 6]>)>;

fn out() -> Option<&'static Out> {
    static OUT: OnceLock<Option<Out>> = OnceLock::new();
    OUT.get_or_init(|| {
        let path = cfg!(feature = "diagnostics").then(|| std::env::var_os("VICTORY_VK_CENSUS")).flatten()?;
        let file = std::fs::File::create(path).ok()?;
        Some(Mutex::new((std::io::BufWriter::with_capacity(1 << 20, file), Default::default())))
    })
    .as_ref()
}

pub(crate) fn on() -> bool {
    out().is_some()
}

fn frame() -> u64 {
    super::super::present::FRAME.load(Ordering::Relaxed)
}

fn line(text: std::fmt::Arguments) {
    if let Some(o) = out() {
        let mut o = o.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(o.0, "{text}");
    }
}

/// The pages of `[at, at + len)` the title has mapped, as hex bits.
fn mapped(b: &super::super::Backend, at: u64, len: u64) -> String {
    let m = super::super::memory(b);
    let Some(pool) = m.pools.iter().find(|p| p.contains(at)) else { return "r".into() };
    let pages = (at >> 16..=(at + len.max(1) - 1) >> 16).take(256);
    let mut bits = [0u64; 4];
    for (i, page) in pages.enumerate() {
        if pool.page_at(page << 16).is_some() {
            bits[i / 64] |= 1 << (i % 64);
        }
    }
    format!("{:x}.{:x}.{:x}.{:x}", bits[0], bits[1], bits[2], bits[3])
}

/// A texture view was written into the title's slot `handle`.
pub(crate) fn view(b: &super::super::Backend, handle: u64, address: u64, r: &ResourceDesc, view: &[u32], unordered: bool) {
    let Some(o) = out() else { return };
    let Some(t) = crate::resource::texture_layout(r) else { return };
    let l = &t.planes[0].layout;
    let shape = [r.format, r.width as u32, r.height, u32::from(r.depth_or_array_size) << 16 | u32::from(r.mip_levels), r.layout, r.flags];
    let pages = mapped(b, address, l.size);
    let mut o = o.lock().unwrap_or_else(|e| e.into_inner());
    if o.1.insert(shape) {
        let starts: Vec<String> = l.mips.iter().map(|m| format!("{:x}", m.offset)).collect();
        let _ = writeln!(o.0, "L {shape:x?} size={:x} block={:x} first_tail={} slices={} starts={}", l.size, l.alignment, l.first_tail_mip, l.slices, starts.join(","));
    }
    let _ = writeln!(
        o.0,
        "V {} {handle:x} {address:x} {shape:x?} {} {view:x?} {pages}",
        frame(),
        u8::from(unordered)
    );
}

pub(crate) fn written(at: u64, len: u64) {
    line(format_args!("W {} {at:x} {len:x}", frame()));
}

pub(crate) fn mapping(to: u64, shown: usize, taken: usize) {
    line(format_args!("M {} {to:x} {shown} {taken}", frame()));
}

pub(crate) fn dropped(why: &str, at: u64, bytes: u64, by: u64) {
    line(format_args!("D {} {why} {at:x} {bytes:x} {by:x}", frame()));
}
