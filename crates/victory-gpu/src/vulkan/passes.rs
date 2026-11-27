//! Diagnostics (`VICTORY_VK_DUMP_AT`, with dumps on): what the first
//! execution of a named program reads, written as a folder the offline
//! interpreter replays (`examples/tir_run`), so a pass is examined without
//! the title or the card: `pass-<frame>-<program>/meta.txt` (one record a
//! line, hex without 0x), `mem/<address>.bin` (title memory: the root
//! tables and views, the tables and V# ranges they point at) and
//! `img/<image id>.bin` (level 0 of each image a T# of those tables names,
//! as the card holds it just before the program runs: rows tightly packed,
//! slices then layers one after another).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use victory_vk::ash::vk;
use victory_vk::memory::{Buffer, Place};

use super::images::{self, Image};
use super::{Backend, lock, read_title};

/// Bytes read where a pointer leads (a table: 128 descriptors).
const TABLE: u64 = 4096;
/// Bytes read at a root view, and the most at a V#.
const VIEW: u64 = 64 << 10;
const RANGE: u64 = 4 << 20;
/// The most a capture holds: memory blocks, memory bytes, image bytes.
const BLOCKS: usize = 16384;
const MEMORY: u64 = 768 << 20;
const IMAGES: u64 = 1024 << 20;

struct Shot {
    dir: PathBuf,
    held: Buffer,
    mapped: usize,
    /// Image, aspect, offset in the buffer, bytes, file name.
    items: Vec<(Arc<Image>, vk::ImageAspectFlags, u64, u64, String)>,
    born: u64,
}

static PENDING: Mutex<Vec<Shot>> = Mutex::new(Vec::new());
// The capture later records of its program go to (push constants, the
// draw or dispatch): the recording thread's own, lists are recorded side
// by side.
thread_local! {
    static OPEN: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

fn append(dir: &std::path::Path, lines: &[String]) {
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("meta.txt")) {
        for l in lines {
            let _ = writeln!(f, "{l}");
        }
    }
}

/// A later record of the program being captured.
pub(super) fn note(line: impl FnOnce() -> String) {
    OPEN.with(|o| {
        if let Some(dir) = o.borrow().as_ref() {
            append(dir, &[line()]);
        }
    });
}

/// The program's draw or dispatch is recorded: its capture takes no more.
pub(super) fn close() {
    OPEN.with(|o| *o.borrow_mut() = None);
}

/// Title memory reached from the roots.
struct Walk<'a> {
    b: &'a Backend,
    dir: &'a std::path::Path,
    have: BTreeMap<u64, u64>,
    total: u64,
    slots: BTreeSet<u32>,
    todo: VecDeque<(u64, u64, u32)>,
    meta: Vec<String>,
}

impl Walk<'_> {
    fn take(&mut self, at: u64, want: u64) -> Option<Vec<u8>> {
        let inside = self.have.range(..=at).next_back().is_some_and(|(&a, &l)| at + want.min(super::title_left(self.b, at)) <= a + l);
        if inside || self.total >= MEMORY || self.have.len() >= BLOCKS {
            return None;
        }
        // to the end of its allocation at most (a constant buffer of a few
        // hundred bytes is a whole allocation's tail)
        let left = super::title_left(self.b, at);
        let bytes = [want, VIEW.min(want), TABLE.min(want)].into_iter().map(|len| len.min(left)).filter(|&len| len != 0).find_map(|len| read_title(self.b, at, len).ok())?;
        if std::fs::write(self.dir.join(format!("mem/{at:x}.bin")), &bytes).is_err() {
            return None;
        }
        self.total += bytes.len() as u64;
        self.have.insert(at, bytes.len() as u64);
        Some(bytes)
    }

    /// The descriptors and pointers in a block's first bytes.
    fn scan(&mut self, bytes: &[u8], depth: u32) {
        let w: Vec<u32> = bytes[..bytes.len().min(TABLE as usize)].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut k = 0;
        while k + 4 <= w.len() {
            if k % 8 == 0 && k + 8 <= w.len() && w[k + 6] ^ w[k + 7] == 0x7A3C_5E91 {
                self.slots.insert(w[k + 7] & 0xff_ffff);
                k += 8;
                continue;
            }
            // a V#: base, stride, records
            let base = u64::from(w[k]) | u64::from(w[k + 1] & 0xffff) << 32;
            let (stride, records) = (u64::from((w[k + 1] >> 16) & 0x3fff).max(1), u64::from(w[k + 2]));
            // (title memory starts low: constant buffers sit at 32-bit
            // addresses; `take` drops what is not title memory)
            if base >= 0x1_0000 && records != 0 {
                self.todo.push_back((base, (stride * records).min(RANGE), depth + 1));
            }
            // plain pointers (a table named by a constant)
            for p in [k, k + 2] {
                let at = u64::from(w[p]) | u64::from(w[p + 1]) << 32;
                if w[p + 1] < 0x40 && at >= 0x1_0000 && at % 4 == 0 {
                    self.todo.push_back((at, TABLE, depth + 1));
                }
            }
            k += 4;
        }
    }

    fn one(&mut self, at: u64, want: u64, depth: u32) {
        let Some(bytes) = self.take(at, want) else { return };
        self.meta.push(format!("mem {at:x} {:x}", bytes.len()));
        if depth < 3 {
            self.scan(&bytes, depth);
        }
    }

    /// Small blocks first (tables, constant buffers), so the large ranges
    /// cannot use the capture's room up before them.
    fn run(&mut self) {
        let mut large = VecDeque::new();
        loop {
            while let Some((at, want, depth)) = self.todo.pop_front() {
                if want > VIEW {
                    large.push_back((at, want, depth));
                } else {
                    self.one(at, want, depth);
                }
            }
            let Some((at, want, depth)) = large.pop_front() else { break };
            self.one(at, want, depth);
        }
    }
}

/// Bytes of level 0 of one aspect of an image, all its slices and layers.
fn level0(img: &Image, aspect: vk::ImageAspectFlags) -> u64 {
    let (w, h) = (u64::from(img.extent.width), u64::from(img.extent.height));
    let texels = if format!("{:?}", img.format).ends_with("_BLOCK") { w.div_ceil(4) * h.div_ceil(4) } else { w * h };
    let each = match (aspect, img.format) {
        (vk::ImageAspectFlags::STENCIL, _) => 1,
        (_, vk::Format::D16_UNORM) => 2,
        _ if img.depth() => 4,
        (_, f) => u64::from(images::format_bytes(f)),
    };
    texels * each * u64::from(img.extent.depth.max(1)) * u64::from(img.layers.max(1))
}

/// Before the first draw or dispatch of a program a dump was asked before
/// (outside rendering): its capture is begun. `roots` are the root
/// parameters as the title set them.
///
/// # Safety
/// Recording into `cmd`, outside rendering.
pub(super) unsafe fn before(b: &Backend, cmd: vk::CommandBuffer, programs: &[(String, String, &str)], name: &str, roots: &[Vec<u32>]) {
    let Some(folder) = crate::capture::folder() else { return };
    let born = super::present::FRAME.load(std::sync::atomic::Ordering::Relaxed);
    let dir = folder.join(format!("pass-{born}-{name}"));
    if std::fs::create_dir_all(dir.join("mem")).is_err() || std::fs::create_dir_all(dir.join("img")).is_err() {
        return;
    }
    let mut meta: Vec<String> = programs.iter().map(|(n, hash, stage)| format!("program {n} {hash} {stage}")).collect();
    // the memory map the card translates title addresses through (what
    // lies outside every range reads the zero block there)
    for r in super::memory(b).map.ranges() {
        meta.push(format!("maprange {:x} {:x} {:x}", r.title_start, r.title_size, r.device_base));
    }
    let mut walk = Walk { b, dir: &dir, have: BTreeMap::new(), total: 0, slots: BTreeSet::new(), todo: VecDeque::new(), meta: Vec::new() };
    for (param, r) in roots.iter().enumerate() {
        meta.push(format!("root {param} {}", r.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")));
        for p in r.chunks_exact(2) {
            walk.todo.push_back((u64::from(p[0]) | u64::from(p[1]) << 32, VIEW, 0));
        }
    }
    walk.run();
    meta.append(&mut walk.meta);
    // the images its T#s name
    let (mut items, mut at, mut seen) = (Vec::new(), 0u64, BTreeSet::new());
    for &slot in &walk.slots {
        let Some((img, key)) = super::descriptors::slot_image(slot) else {
            meta.push(format!("image {slot:x} none"));
            continue;
        };
        meta.push(format!(
            "image {slot:x} file {} address {:x} format {} {:?} {}x{}x{} layers {} mips {} dropped {} samples {} view {} {:?} {:?} mips {}+{} layers {}+{} swizzle {:?} storage {}",
            img.id,
            img.key.address,
            img.format.as_raw(),
            img.format,
            img.extent.width,
            img.extent.height,
            img.extent.depth,
            img.layers,
            img.mips,
            img.dropped,
            img.samples,
            key.format.as_raw(),
            key.format,
            key.kind,
            key.base_mip,
            key.mips,
            key.base_layer,
            key.layers,
            key.components.map(|c| c.as_raw()),
            key.storage
        ));
        if img.samples > 1 || !seen.insert(img.id) {
            continue;
        }
        let planes: &[(vk::ImageAspectFlags, &str)] = match (img.depth(), img.aspect().contains(vk::ImageAspectFlags::STENCIL)) {
            (true, true) => &[(vk::ImageAspectFlags::DEPTH, ""), (vk::ImageAspectFlags::STENCIL, ".stencil")],
            (true, false) => &[(vk::ImageAspectFlags::DEPTH, "")],
            _ => &[(vk::ImageAspectFlags::COLOR, "")],
        };
        for &(aspect, suffix) in planes {
            let bytes = level0(&img, aspect);
            if bytes == 0 || at + bytes > IMAGES {
                meta.push(format!("left out {} ({bytes:x} bytes)", img.id));
                continue;
            }
            items.push((img.clone(), aspect, at, bytes, format!("img/{}{suffix}.bin", img.id)));
            at += bytes.next_multiple_of(16);
        }
    }
    append(&dir, &meta);
    OPEN.with(|o| *o.borrow_mut() = Some(dir.clone()));
    let Ok(buf) = Buffer::new(&b.host, at.max(16), vk::BufferUsageFlags::TRANSFER_DST, Place::Readback) else { return };
    // SAFETY: as the caller says.
    unsafe {
        images::full_barrier(&b.host.device, cmd);
        for (img, aspect, at, _, _) in &items {
            let region = vk::BufferImageCopy::default()
                .buffer_offset(*at)
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: *aspect, mip_level: 0, base_array_layer: 0, layer_count: img.layers.max(1) })
                .image_extent(vk::Extent3D { width: img.extent.width, height: img.extent.height, depth: img.extent.depth.max(1) });
            b.host.device.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, buf.buffer, &[region]);
        }
        images::full_barrier(&b.host.device, cmd);
    }
    tracing::info!(program = name, dir = %dir.display(), memory_blocks = walk.have.len(), memory_bytes = walk.total, images = items.len(), image_bytes = at, "Vulkan: a pass's inputs are captured for replay");
    lock(&PENDING).push(Shot { dir, mapped: buf.mapped as usize, held: buf, items, born });
}

/// Frames a readback waits before it is read: its copies are recorded in
/// a list of the title's, which may be executed a frame or two after it
/// was recorded (read at the next present, the buffer was freed before the
/// card wrote it: a copy-engine fault, runs vk82 and vk87).
pub(super) const SETTLE: u64 = 4;
/// Frames a readback buffer is kept after its read.
const KEPT: u64 = 600;

/// Readback buffers read, with the frame of the read.
static GRAVE: Mutex<Vec<(u64, Buffer)>> = Mutex::new(Vec::new());

/// A readback buffer was read at `frame`: it is destroyed much later.
pub(super) fn bury(buffer: Buffer, frame: u64) {
    lock(&GRAVE).push((frame, buffer));
}

/// Whether a capture of an earlier frame waits to be saved, or a buffer
/// to be destroyed.
pub(super) fn pending(frame: u64) -> bool {
    lock(&PENDING).iter().any(|p| p.born + SETTLE < frame) || lock(&GRAVE).iter().any(|(at, _)| at + KEPT < frame)
}

/// At a present with the queue idle: the images of earlier frames'
/// captures are written.
pub(super) fn save_pending(b: &Backend, frame: u64) {
    let (old, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut *lock(&GRAVE)).into_iter().partition(|(at, _)| at + KEPT < frame);
    lock(&GRAVE).extend(kept);
    for (_, buffer) in old {
        // SAFETY: the queue is idle and its copies ran hundreds of frames ago.
        unsafe { buffer.destroy(&b.host) };
    }
    let (ready, later): (Vec<Shot>, Vec<Shot>) = std::mem::take(&mut *lock(&PENDING)).into_iter().partition(|p| p.born + SETTLE < frame);
    lock(&PENDING).extend(later);
    for p in ready {
        for (_, _, at, bytes, file) in &p.items {
            // SAFETY: the mapped readback buffer; the copies into it have finished.
            let src = unsafe { std::slice::from_raw_parts((p.mapped as *const u8).add(*at as usize), *bytes as usize) };
            let _ = std::fs::write(p.dir.join(file), src);
        }
        append(&p.dir, &["images written".to_string()]);
        bury(p.held, frame);
    }
}
