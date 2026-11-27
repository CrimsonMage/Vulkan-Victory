//! Diagnostics (`VICTORY_VK_WATCH=<address>,<x>,<y>[,c][;...]`, the
//! address in hex, up to four, with `VICTORY_VK_DUMP_TARGETS` on): who
//! writes a texel. In the first dumped frame the texel (x, y) of level 0 of the
//! image over that title address is read before and after every draw that
//! has the image attached and every dispatch; the ones that changed it are
//! logged (programs with their hashes, what was drawn, the bytes before and
//! after), and the next execution of the last few writers of the texels
//! marked `c` (`VICTORY_VK_WATCH_CAPTURE=<n>`, 3 unless told) is captured as
//! `VICTORY_VK_DUMP_AT` captures a program (`pass-<frame>-<name>-w<n>`):
//! a draw is known again by its pixel program and its fourth root
//! parameter (a ped's draws share the program and differ there).
//!
//! The readings of one frame are in the order of recording, not of
//! execution (lists are recorded side by side): the order of the writes
//! is told by the values (one's `after` is the next one's `before`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::images::{self, Image};
use super::{Backend, lock};

/// Readings a frame holds at most, and the bytes of one.
const MOST: u64 = 65536;
const EACH: u64 = 16;

struct Spot {
    address: u64,
    x: u32,
    y: u32,
    /// Its writers are captured.
    capture: bool,
}

fn spots() -> &'static [Spot] {
    static SPOTS: OnceLock<Vec<Spot>> = OnceLock::new();
    SPOTS.get_or_init(|| {
        if !cfg!(feature = "diagnostics") {
            return Vec::new();
        }
        let set = std::env::var("VICTORY_VK_WATCH").unwrap_or_default();
        set.split(';')
            .filter_map(|s| {
                let mut p = s.split(',').map(str::trim);
                let address = u64::from_str_radix(p.next()?.trim_start_matches("0x"), 16).ok()?;
                Some(Spot { address, x: p.next()?.parse().ok()?, y: p.next()?.parse().ok()?, capture: p.next() == Some("c") })
            })
            .take(4)
            .collect()
    })
}

/// A draw or dispatch as it is known again in a later frame: the hash of
/// its last program and its fourth root parameter (none: a program run
/// once a frame).
type Key = (String, Vec<u32>);

struct Entry {
    programs: String,
    name: String,
    what: String,
    key: Key,
    /// Spot, the reading before and the reading after.
    readings: Vec<(usize, u64, u64)>,
}

#[derive(Default)]
struct State {
    held: Option<Buffer>,
    mapped: usize,
    next: u64,
    born: u64,
    entries: Vec<Entry>,
    /// The image found for each spot by a dispatch.
    found: Vec<Option<Arc<Image>>>,
    chosen: Vec<(Key, String)>,
    reported: bool,
}

// SAFETY: the mapped pointer is read only at the report, under the lock, with the queue idle.
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static ARMED: AtomicBool = AtomicBool::new(false);

/// Whether texels are read in the frame being recorded.
pub(super) fn on() -> bool {
    !spots().is_empty() && super::targets::from() == Some(super::present::FRAME.load(Ordering::Relaxed))
}

/// Whether a draw on these attachments can write a watched texel.
pub(super) fn touches(attached: &[Arc<Image>]) -> bool {
    spots().iter().any(|s| attached.iter().any(|i| i.key.address == s.address))
}

/// The watched texels as they are at this point of the list: a reading
/// (spot, place) for each one found (among `attached`, or any image the
/// card writes over the address).
///
/// # Safety
/// Recording into `cmd`, outside rendering.
pub(super) unsafe fn read(b: &Backend, cmd: vk::CommandBuffer, attached: Option<&[Arc<Image>]>) -> Vec<(usize, u64)> {
    let mut guard = lock(&STATE);
    let st = guard.get_or_insert_with(State::default);
    if st.held.is_none() {
        let Ok(buf) = Buffer::new(&b.host, MOST * EACH, vk::BufferUsageFlags::TRANSFER_DST, Place::Readback) else { return Vec::new() };
        st.mapped = buf.mapped as usize;
        st.held = Some(buf);
        st.born = super::present::FRAME.load(Ordering::Relaxed);
        st.found = vec![None; spots().len()];
    }
    let Some(buffer) = st.held.as_ref().map(|h| h.buffer) else { return Vec::new() };
    let mut out = Vec::new();
    for (k, s) in spots().iter().enumerate() {
        let img = match attached {
            Some(att) => att.iter().find(|i| i.key.address == s.address).cloned(),
            None => {
                if st.found[k].is_none() {
                    st.found[k] = images::card_written_images().into_iter().find(|i| i.key.address == s.address && i.samples == 1);
                }
                st.found[k].clone()
            }
        };
        let Some(img) = img else { continue };
        let each = if img.depth() { 4 } else { u64::from(images::format_bytes(img.format)) };
        if img.samples > 1 || s.x >= img.extent.width || s.y >= img.extent.height || each == 0 || each > EACH || st.next >= MOST {
            continue;
        }
        let aspect = if img.depth() { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
        let region = vk::BufferImageCopy::default()
            .buffer_offset(st.next * EACH)
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_offset(vk::Offset3D { x: s.x as i32, y: s.y as i32, z: 0 })
            .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 });
        // SAFETY: as the caller says; the place lies inside the buffer.
        unsafe {
            images::full_barrier(&b.host.device, cmd);
            b.host.device.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, buffer, &[region]);
            images::full_barrier(&b.host.device, cmd);
        }
        out.push((k, st.next));
        st.next += 1;
    }
    out
}

/// The key of a draw or dispatch of `shaders` with these root parameters.
fn key(shaders: &[Arc<crate::pipeline::Shader>], roots: &[Vec<u32>]) -> Key {
    (shaders.last().map(|s| s.hash.clone()).unwrap_or_default(), roots.get(3).cloned().unwrap_or_default())
}

/// A draw or dispatch was recorded between two readings.
pub(super) fn entry(before: &[(usize, u64)], after: &[(usize, u64)], shaders: &[Arc<crate::pipeline::Shader>], roots: &[Vec<u32>], what: &str) {
    let readings: Vec<(usize, u64, u64)> = before.iter().filter_map(|&(k, b)| after.iter().find(|a| a.0 == k).map(|a| (k, b, a.1))).collect();
    if readings.is_empty() {
        return;
    }
    if let Some(st) = lock(&STATE).as_mut() {
        st.entries.push(Entry {
            programs: shaders.iter().map(|s| format!("{} {}", s.name, s.hash)).collect::<Vec<_>>().join(" + "),
            name: shaders.last().map(|s| s.name.clone()).unwrap_or_default(),
            what: what.to_owned(),
            key: key(shaders, roots),
            readings,
        });
    }
}

/// Whether the readings of an earlier frame wait to be told.
pub(super) fn pending(frame: u64) -> bool {
    !spots().is_empty() && lock(&STATE).as_ref().is_some_and(|st| !st.reported && st.held.is_some() && st.born + super::passes::SETTLE < frame)
}

/// At a present with the queue idle: the writes of the watched texels are
/// logged and the last of them chosen for a capture.
pub(super) fn report(frame: u64) {
    let mut guard = lock(&STATE);
    let Some(st) = guard.as_mut() else { return };
    if st.reported || st.born + super::passes::SETTLE >= frame {
        return;
    }
    st.reported = true;
    let bytes = |place: u64| {
        // SAFETY: the mapped readback buffer; the copies into it have finished.
        let src = unsafe { std::slice::from_raw_parts((st.mapped as *const u8).add((place * EACH) as usize), EACH as usize) };
        src.to_vec()
    };
    let hex = |v: &[u8]| v.chunks(4).map(|c| c.iter().rev().map(|x| format!("{x:02x}")).collect::<String>()).collect::<Vec<_>>().join(" ");
    let mut writers: Vec<(Key, String)> = Vec::new();
    let mut told = 0u32;
    for (n, e) in st.entries.iter().enumerate() {
        for &(k, before, after) in &e.readings {
            let (was, is) = (bytes(before), bytes(after));
            if was == is {
                continue;
            }
            told += 1;
            let s = &spots()[k];
            tracing::info!(
                n,
                spot = format_args!("{:#x} {} {}", s.address, s.x, s.y),
                programs = %e.programs,
                what = %e.what,
                root3 = %e.key.1.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(":"),
                before = %hex(&was),
                after = %hex(&is),
                "Vulkan: a watched texel was written"
            );
            if s.capture {
                writers.retain(|w| w.0 != e.key);
                writers.push((e.key.clone(), format!("{}-w{n}", e.name)));
            }
        }
    }
    let most = std::env::var("VICTORY_VK_WATCH_CAPTURE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(3).min(6);
    let from = writers.len().saturating_sub(most);
    st.chosen = writers.split_off(from);
    tracing::info!(
        frame = st.born,
        draws_and_dispatches = st.entries.len(),
        readings = st.next,
        full = st.next >= MOST,
        writes = told,
        to_capture = %st.chosen.iter().map(|c| c.1.clone()).collect::<Vec<_>>().join(" "),
        "Vulkan: the watched texels of a frame"
    );
    st.entries = Vec::new();
    ARMED.store(!st.chosen.is_empty(), Ordering::Relaxed);
    if let Some(held) = st.held.take() {
        super::passes::bury(held, frame);
    }
}

/// Before a draw or dispatch of `shaders`: the name its capture takes, if
/// it is one of the writers chosen (each is captured once).
pub(super) fn capture(shaders: &[Arc<crate::pipeline::Shader>], roots: &[Vec<u32>]) -> Option<String> {
    if !ARMED.load(Ordering::Relaxed) {
        return None;
    }
    let mine = key(shaders, roots);
    let mut guard = lock(&STATE);
    let st = guard.as_mut()?;
    let at = st.chosen.iter().position(|(k, _)| k.0 == mine.0 && (k.1.is_empty() || k.1 == mine.1))?;
    let (_, name) = st.chosen.remove(at);
    if st.chosen.is_empty() {
        ARMED.store(false, Ordering::Relaxed);
    }
    Some(name)
}
