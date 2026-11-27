//! replay <capture dir> <container> [out dir]: runs a captured pass in the
//! interpreter (`vk::interp`): a compute program over its dispatch, a pixel
//! program over its viewport (from FragCoord: no vertex program, no
//! rasteriser, so it suits full-screen passes), with title memory from
//! `mem/` and images from `img/` (`crates/victory-gpu/src/vulkan/passes.rs`
//! writes the capture; `cargo xtask shader-extract` the container).
//!
//! Writes `out/` (default `<capture>/out`): each image the program stores
//! to and each render target as `<name>.f32` (RGBA f32, rows tight) and a
//! `.png` (clamped to 0..1), `census.txt` (min, max and non-finite count a
//! channel), `missing.txt` (memory read that the capture lacks, as
//! ranges) and `notes.txt` (what replay assumed).
//!
//! Assumed: every sampler filters linearly and clamps to the edge (no
//! sampler state is captured); level 0 only (an explicit level, bias or
//! gradient is ignored); depth comparison is less-or-equal; block-compressed
//! formats read 0. Environment: `STRIDE=n` runs every n-th group (compute)
//! or block (pixel), `LIMIT=n` block visits a wave.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use victory_shader::HwStage;
use victory_shader::native::StageIo;
use victory_shader::vk::interp::{Images, Machine, Memory, Stop};
use victory_shader::vk::tir::{ImageClass, ImageKind, Sampler, img};
use victory_shader::vk::thread_ir;

type W = [u64; 4];

/// One `image` line and its data.
struct Slot {
    name: String,
    w: u32,
    h: u32,
    d: u32,
    layers: u32,
    base_layer: u32,
    view_format: String,
    swizzle: [i32; 4],
    data: Rc<Vec<u8>>,
    missing_mip: bool,
}

#[derive(Default)]
struct Capture {
    program: Vec<(String, String, String)>,
    push: Vec<u32>,
    dispatch: Option<[u32; 3]>,
    viewport: Option<[f32; 4]>,
    mem: Vec<(u64, PathBuf)>,
    slots: HashMap<u32, Slot>,
    lines: Vec<String>,
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0)
}

/// The words after `key` in a line.
fn after<'a>(w: &'a [&'a str], key: &str) -> Option<&'a str> {
    w.iter().position(|x| *x == key).and_then(|i| w.get(i + 1)).copied()
}

fn parse_image(dir: &Path, w: &[&str], files: &mut HashMap<String, Rc<Vec<u8>>>) -> Option<(u32, Slot)> {
    let slot = hex(w.get(1)?) as u32;
    let file = after(w, "file")?;
    let fmt_at = w.iter().position(|x| *x == "format")?;
    let dims: Vec<u32> = w.get(fmt_at + 3)?.split('x').filter_map(|v| v.parse().ok()).collect();
    let view_at = w.iter().position(|x| *x == "view")?;
    let ranges: Vec<(u32, u32)> = w
        .iter()
        .enumerate()
        .filter(|(i, x)| *i > view_at && (**x == "mips" || **x == "layers"))
        .filter_map(|(i, _)| {
            let (a, b) = w.get(i + 1)?.split_once('+')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .collect();
    let sw_at = w.iter().position(|x| *x == "swizzle")?;
    let swizzle: Vec<i32> = w[sw_at + 1..].iter().take(4).map(|x| x.trim_matches(['[', ']', ',']).parse().unwrap_or(0)).collect();
    let data = files.entry(file.to_owned()).or_insert_with(|| Rc::new(std::fs::read(dir.join(format!("img/{file}.bin"))).unwrap_or_default())).clone();
    Some((
        slot,
        Slot {
            name: format!("slot{slot:x}-{}", w.get(fmt_at + 2)?),
            w: dims.first().copied().unwrap_or(1),
            h: dims.get(1).copied().unwrap_or(1),
            d: dims.get(2).copied().unwrap_or(1),
            layers: after(w, "layers")?.parse().unwrap_or(1),
            base_layer: ranges.get(1).map_or(0, |r| r.0),
            view_format: w.get(view_at + 2)?.to_string(),
            swizzle: [0, 1, 2, 3].map(|i| swizzle.get(i).copied().unwrap_or(0)),
            data,
            missing_mip: ranges.first().is_some_and(|r| r.0 > 0),
        },
    ))
}

fn viewport(line: &str) -> Option<[f32; 4]> {
    let s = line.split("viewport Some((").nth(1)?;
    let v: Vec<f32> = s.split([',', ')']).filter_map(|x| x.trim().parse().ok()).collect();
    Some([v[0], v[1], *v.get(2)?, *v.get(3)?])
}

fn read_capture(dir: &Path) -> Capture {
    let mut c = Capture::default();
    let mut files = HashMap::new();
    let text = std::fs::read_to_string(dir.join("meta.txt")).expect("meta.txt");
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.first().copied() {
            Some("program") if w.len() >= 4 => c.program.push((w[1].into(), w[2].into(), w[3].into())),
            Some("push") if c.dispatch.is_none() && c.viewport.is_none() => c.push = w[1..].iter().map(|x| hex(x) as u32).collect(),
            Some("mem") => c.mem.push((hex(w[1]), dir.join(format!("mem/{}.bin", w[1])))),
            Some("dispatch") => c.dispatch = Some([1, 2, 3].map(|i| w.get(i).and_then(|x| x.parse().ok()).unwrap_or(1))),
            Some("did") => c.viewport = viewport(line),
            Some("image") if w.get(2) != Some(&"none") => {
                if let Some((slot, s)) = parse_image(dir, &w, &mut files) {
                    c.slots.insert(slot, s);
                }
            }
            _ => {}
        }
        c.lines.push(line.to_owned());
    }
    c
}

// ---- texels ----

fn f16(h: u16) -> f32 {
    victory_shader::vk::interp::f16_to_f32(h)
}

/// An unsigned float of `m` mantissa bits and a 5-bit exponent.
fn small_float(bits: u32, m: u32) -> f32 {
    let (e, f) = (bits >> m, bits & ((1 << m) - 1));
    match e {
        0 => f as f32 / (1 << m) as f32 * 2f32.powi(-14),
        31 => if f == 0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + f as f32 / (1 << m) as f32) * 2f32.powi(e as i32 - 15),
    }
}

fn srgb(x: f32) -> f32 {
    if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

/// Bytes a texel of the format takes, and whether replay decodes it.
fn texel_bytes(name: &str) -> (usize, bool) {
    let n = name;
    let b = if n.contains("BLOCK") {
        0
    } else if n.starts_with("R32G32B32A32") {
        16
    } else if n.starts_with("R32G32B32") {
        12
    } else if n.starts_with("R32G32") || n.starts_with("R16G16B16A16") || n.starts_with("R64") {
        8
    } else if n.starts_with("R32") || n.starts_with("D32") || n.starts_with("R16G16") || n.contains("PACK32") || n.starts_with("R8G8B8A8") || n.starts_with("B8G8R8A8") || n.starts_with("D24") || n.starts_with("X8") {
        4
    } else if n.starts_with("R16") || n.starts_with("D16") || n.starts_with("R8G8") {
        2
    } else if n.starts_with("R8") || n.starts_with("S8") {
        1
    } else {
        0
    };
    (b, b > 0)
}

/// A texel as four channels: floats (or integers for `uint` reads) as bits.
/// Channels a format holds.
fn channels(name: &str) -> usize {
    if name.starts_with("R8G8B8A8") || name.starts_with("B8G8R8A8") || name.contains("A16") || name.contains("A32") {
        4
    } else if name.starts_with("R32G32B32") {
        3
    } else if name.starts_with("R8G8") || name.starts_with("R16G16") || name.starts_with("R32G32") {
        2
    } else {
        1
    }
}

/// Channel `c` of a plain 8-, 16- or 32-bit format: (float, integer).
fn channel(name: &str, t: &[u8], c: usize) -> (f32, u32) {
    let snorm = name.contains("SNORM");
    if name.starts_with("R8") || name.starts_with("B8") {
        let b = t[c];
        return (if snorm { (b as i8 as f32 / 127.0).max(-1.0) } else { b as f32 / 255.0 }, u32::from(b));
    }
    if name.starts_with("R16") {
        let h = u16::from_le_bytes([t[2 * c], t[2 * c + 1]]);
        let f = if name.contains("SFLOAT") {
            f16(h)
        } else if snorm {
            (h as i16 as f32 / 32767.0).max(-1.0)
        } else {
            h as f32 / 65535.0
        };
        return (f, u32::from(h));
    }
    let v = u32::from_le_bytes([t[4 * c], t[4 * c + 1], t[4 * c + 2], t[4 * c + 3]]);
    (if name.contains("SFLOAT") { f32::from_bits(v) } else { v as f32 }, v)
}

/// The packed and depth formats: (floats, integers).
fn packed(name: &str, t: &[u8]) -> Option<([f32; 4], [u32; 4])> {
    let v = u32::from_le_bytes([t[0], t[1], t.get(2).copied().unwrap_or(0), t.get(3).copied().unwrap_or(0)]);
    Some(match name {
        n if n.starts_with("B10G11R11") => ([small_float(v & 0x7FF, 6), small_float(v >> 11 & 0x7FF, 6), small_float(v >> 22, 5), 1.0], [0, 0, 0, 1]),
        n if n.starts_with("A2B10G10R10") => {
            let i = [v & 0x3FF, v >> 10 & 0x3FF, v >> 20 & 0x3FF, v >> 30];
            ([i[0] as f32 / 1023.0, i[1] as f32 / 1023.0, i[2] as f32 / 1023.0, i[3] as f32 / 3.0], i)
        }
        n if n.starts_with("D32") || n.starts_with("D24") => ([f32::from_bits(v), 0.0, 0.0, 1.0], [v, 0, 0, 1]),
        n if n.starts_with("D16") => ([(v & 0xFFFF) as f32 / 65535.0, 0.0, 0.0, 1.0], [v & 0xFFFF, 0, 0, 1]),
        _ => return None,
    })
}

/// A texel as four channels: floats (or integers for `uint` reads) as bits.
fn decode(name: &str, t: &[u8], uint: bool) -> W {
    let (x, ints) = packed(name, t).unwrap_or_else(|| {
        let ch = channels(name);
        let (mut x, mut ints) = ([0.0, 0.0, 0.0, 1.0], [0, 0, 0, 1]);
        for c in 0..ch {
            (x[c], ints[c]) = channel(name, t, c);
        }
        if name.starts_with("B8G8R8A8") {
            x.swap(0, 2);
            ints.swap(0, 2);
        }
        if name.contains("SRGB") {
            for v in x.iter_mut().take(3) {
                *v = srgb(*v);
            }
        }
        (x, ints)
    });
    if uint { ints.map(u64::from) } else { x.map(|f| u64::from(f.to_bits())) }
}

/// The view's swizzle (VkComponentSwizzle: 0 identity, 1 zero, 2 one, 3..6 r..a).
fn swizzled(s: &[i32; 4], x: W, uint: bool) -> W {
    let one = if uint { 1 } else { u64::from(1f32.to_bits()) };
    [0, 1, 2, 3].map(|c| match s[c] {
        0 => x[c],
        1 => 0,
        2 => one,
        k @ 3..=6 => x[(k - 3) as usize],
        _ => x[c],
    })
}

// ---- images ----

struct State {
    slots: HashMap<u32, Slot>,
    /// Stores: slot -> texel index -> the channels (bits).
    written: BTreeMap<u32, BTreeMap<usize, W>>,
    notes: BTreeSet<String>,
}

#[derive(Clone)]
struct Replay(Rc<RefCell<State>>);

/// An image instruction's operands (as `vk::emit_image::ImgArgs`).
struct Args<'a> {
    coord: W,
    dref: Option<W>,
    offset: Option<W>,
    data: Option<W>,
    _all: &'a [W],
}

fn args_of(sampler: Sampler, flags: u16, a: &[W]) -> Args<'_> {
    let mut i = 1 + usize::from(sampler == Sampler::Dynamic);
    let mut next = |n: usize| {
        i += n;
        a.get(i - n).copied().unwrap_or_default()
    };
    let coord = next(1);
    let mut opt = |bit: u16, n: usize| (flags & bit != 0).then(|| next(n));
    let dref = opt(img::DREF, 1);
    let _ = (opt(img::BIAS, 1), opt(img::LOD, 1), opt(img::GRAD, 2));
    let offset = opt(img::OFFSET, 1);
    let _ = (opt(img::MIN_LOD, 1), opt(img::SAMPLE, 1));
    let data = opt(img::DATA, 1);
    Args { coord, dref, offset, data, _all: a }
}

impl State {
    /// Texel (x, y, z) of a slot (clamped), or its stored value.
    fn texel(&mut self, slot: u32, x: i64, y: i64, z: i64, uint: bool) -> W {
        let Some(s) = self.slots.get(&slot) else {
            self.notes.insert(format!("slot {slot:x}: no image (reads 0)"));
            return [0; 4];
        };
        let depth = (s.d.max(1) * s.layers.max(1)) as i64;
        let (x, y, z) = (x.clamp(0, s.w as i64 - 1), y.clamp(0, s.h as i64 - 1), z.clamp(0, depth - 1));
        let idx = ((z * s.h as i64 + y) * s.w as i64 + x) as usize;
        if let Some(v) = self.written.get(&slot).and_then(|m| m.get(&idx)) {
            return *v;
        }
        let (bytes, known) = texel_bytes(&s.view_format);
        if !known {
            self.notes.insert(format!("slot {slot:x}: format {} not decoded (reads 0)", s.view_format));
            return [0; 4];
        }
        if s.missing_mip {
            self.notes.insert(format!("slot {slot:x}: the view starts past level 0 (level 0 read)"));
        }
        let at = idx * bytes;
        let Some(t) = s.data.get(at..at + bytes) else {
            self.notes.insert(format!("slot {slot:x}: no data at texel {idx}"));
            return [0; 4];
        };
        let v = decode(&s.view_format, t, uint);
        swizzled(&s.swizzle, v, uint)
    }

    fn dims(&self, slot: u32) -> (f32, f32, f32, u32) {
        self.slots.get(&slot).map_or((1.0, 1.0, 1.0, 0), |s| (s.w as f32, s.h as f32, s.d.max(1) as f32, s.base_layer))
    }

    /// A bilinear sample (trilinear for volumes) at normalized coordinates.
    fn sample(&mut self, slot: u32, class: ImageClass, c: [f32; 4], off: [i64; 2]) -> [f32; 4] {
        let (w, h, d, base) = self.dims(slot);
        let (u, v) = (c[0] * w - 0.5 + off[0] as f32, c[1] * h - 0.5 + off[1] as f32);
        let layer = if class.arrayed { c[if class.dim == 2 { 3 } else { 2 }].round() as i64 + i64::from(base) } else { i64::from(base) };
        let zs: Vec<(i64, f32)> = if class.dim == 2 {
            let z = c[2] * d - 0.5;
            let z0 = z.floor();
            vec![(z0 as i64, 1.0 - (z - z0)), (z0 as i64 + 1, z - z0)]
        } else {
            vec![(layer, 1.0)]
        };
        let (x0, y0) = (u.floor(), v.floor());
        let (fx, fy) = (u - x0, v - y0);
        let mut out = [0f32; 4];
        for (z, wz) in zs {
            for (dx, dy, wt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
                let t = self.texel(slot, x0 as i64 + dx, y0 as i64 + dy, z, false);
                for k in 0..4 {
                    let x = f32::from_bits(t[k] as u32);
                    if wt * wz != 0.0 {
                        out[k] += x * wt * wz;
                    }
                }
            }
        }
        out
    }

    fn gather(&mut self, slot: u32, comp: u8, c: [f32; 4], off: [i64; 2]) -> W {
        let (w, h, _, base) = self.dims(slot);
        let (u, v) = (c[0] * w - 0.5 + off[0] as f32, c[1] * h - 0.5 + off[1] as f32);
        let (x0, y0) = (u.floor() as i64, v.floor() as i64);
        let k = usize::from(comp.min(3));
        [(0, 1), (1, 1), (1, 0), (0, 0)].map(|(dx, dy)| self.texel(slot, x0 + dx, y0 + dy, i64::from(base), false)[k])
    }
}

/// A cube direction as (face u, face v, face) for the six-layer array.
fn cube(c: [f32; 4]) -> [f32; 4] {
    let (x, y, z) = (c[0], c[1], c[2]);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    let (face, sc, tc, ma) = if az >= ax && az >= ay {
        if z >= 0.0 { (4.0, x, -y, az) } else { (5.0, -x, -y, az) }
    } else if ay >= ax {
        if y >= 0.0 { (2.0, x, z, ay) } else { (3.0, x, -z, ay) }
    } else if x >= 0.0 {
        (0.0, -z, -y, ax)
    } else {
        (1.0, z, -y, ax)
    };
    [(sc / ma + 1.0) / 2.0, (tc / ma + 1.0) / 2.0, face + 6.0 * c[3].round(), 0.0]
}

impl Images for Replay {
    fn image(&mut self, kind: ImageKind, class: ImageClass, sampler: Sampler, flags: u16, a: &[W], _lane: usize) -> W {
        let st = &mut *self.0.borrow_mut();
        let slot = (a.first().map_or(0, |w| w[0]) & 0xFF_FFFF) as u32;
        let x = args_of(sampler, flags, a);
        let f = x.coord.map(|v| f32::from_bits(v as u32));
        let off = x.offset.map_or([0, 0], |o| [o[0] as i32 as i64, o[1] as i32 as i64]);
        let i = x.coord.map(|v| v as i32 as i64);
        let layered = |i: [i64; 4]| if class.dim == 2 || class.arrayed { i[2] } else { 0 };
        match kind {
            ImageKind::Sample { .. } => {
                let c = if class.dim == 3 && !class.cube_layers { cube(f) } else { f };
                let cl = if class.dim == 3 { ImageClass { dim: 1, arrayed: true, ..class } } else { class };
                let v = st.sample(slot, cl, c, off);
                match x.dref {
                    Some(r) => {
                        st.notes.insert("depth comparison taken as less-or-equal".into());
                        let r = f32::from_bits(r[0] as u32);
                        [u64::from(if r <= v[0] { 1f32 } else { 0.0 }.to_bits()), 0, 0, 0]
                    }
                    None => v.map(|f| u64::from(f.to_bits())),
                }
            }
            ImageKind::Gather { comp, .. } => st.gather(slot, comp, f, off),
            ImageKind::Load { .. } => st.texel(slot, i[0], i[1], layered(i), class.uint),
            ImageKind::Store { .. } => {
                let (w, h, _, _) = st.dims(slot);
                let idx = ((layered(i).max(0) * h as i64 + i[1].max(0)) * w as i64 + i[0].max(0)) as usize;
                st.written.entry(slot).or_default().insert(idx, x.data.unwrap_or_default());
                [0; 4]
            }
            ImageKind::Size => {
                let s = st.slots.get(&slot);
                s.map_or([1, 1, 1, 0], |s| [u64::from(s.w), u64::from(s.h), u64::from(if class.arrayed { s.layers } else { s.d }), 0])
            }
            ImageKind::Levels => [1, 0, 0, 0],
            _ => {
                st.notes.insert(format!("{kind:?} reads 0"));
                [0; 4]
            }
        }
    }
}

// ---- output ----

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

/// An RGBA8 PNG (stored, uncompressed deflate blocks).
fn png(path: &Path, w: u32, h: u32, rgba: &[u8]) {
    let mut raw = Vec::new();
    for row in rgba.chunks(w as usize * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (k, b) in blocks.iter().enumerate() {
        z.push(u8::from(k + 1 == blocks.len()));
        z.extend_from_slice(&(b.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(b.len() as u16)).to_le_bytes());
        z.extend_from_slice(b);
    }
    let (mut a, mut bb) = (1u32, 0u32);
    for &x in &raw {
        a = (a + u32::from(x)) % 65521;
        bb = (bb + a) % 65521;
    }
    z.extend_from_slice(&(bb << 16 | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 13, 10, 26, 10];
    let mut chunk = |kind: &[u8], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut c = kind.to_vec();
        c.extend_from_slice(data);
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    let _ = std::fs::write(path, out);
}

/// Writes an RGBA f32 image (`.f32`, `.png`) and its census line.
fn write_image(out: &Path, name: &str, w: u32, h: u32, px: &[[f32; 4]], census: &mut String) {
    let raw: Vec<u8> = px.iter().flat_map(|p| p.iter().flat_map(|f| f.to_le_bytes())).collect();
    let _ = std::fs::write(out.join(format!("{name}.f32")), raw);
    let rgba: Vec<u8> = px.iter().flat_map(|p| p.map(|f| (f.clamp(0.0, 1.0) * 255.0) as u8)).collect();
    png(&out.join(format!("{name}.png")), w, h, &rgba);
    let _ = write!(census, "{name} {w}x{h}:");
    for c in 0..4 {
        let vals: Vec<f32> = px.iter().map(|p| p[c]).collect();
        let bad = vals.iter().filter(|v| !v.is_finite()).count();
        let fin = vals.iter().copied().filter(|v| v.is_finite());
        let (lo, hi) = fin.fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        let _ = write!(census, " [{}] min {lo} max {hi} non-finite {bad};", "RGBA".as_bytes()[c] as char);
    }
    census.push('\n');
}

// ---- running ----

fn memory(c: &Capture) -> Memory {
    let mut mem = Memory { range: (0, u64::MAX), ..Default::default() };
    for (at, path) in &c.mem {
        let bytes = std::fs::read(path).unwrap_or_default();
        let words: Vec<u32> = bytes.chunks(4).map(|b| b.iter().rev().fold(0u32, |w, &x| w << 8 | u32::from(x))).collect();
        mem.put(*at, &words);
    }
    mem
}

fn env(k: &str, default: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Missing memory as ranges (address, bytes, reads).
fn missing_ranges(missing: &BTreeMap<u64, u32>) -> String {
    let mut out = String::new();
    let mut run: Option<(u64, u64, u32)> = None;
    for (&a, &n) in missing {
        run = match run {
            Some((s, e, k)) if a <= e + 64 => Some((s, a + 4, k + n)),
            Some((s, e, k)) => {
                let _ = writeln!(out, "missing {s:x} {:x} bytes read {k}x", e - s);
                Some((a, a + 4, n))
            }
            None => Some((a, a + 4, n)),
        };
    }
    if let Some((s, e, k)) = run {
        let _ = writeln!(out, "missing {s:x} {:x} bytes read {k}x", e - s);
    }
    out
}

struct Run<'a> {
    f: &'a victory_shader::vk::tir::Func,
    cap: &'a Capture,
    mem: Memory,
    images: Replay,
    lanes: usize,
    missing: BTreeMap<u64, u32>,
    stops: BTreeMap<String, u32>,
}

impl<'a> Run<'a> {
    fn wave(&mut self, setup: impl FnOnce(&mut Machine<'a>)) -> Machine<'a> {
        let mut m = Machine::new(self.f, self.lanes, std::mem::take(&mut self.mem));
        m.push = self.cap.push.clone();
        m.push.resize(64, 0);
        m.limit = env("LIMIT", 1 << 22);
        m.images = Some(Box::new(self.images.clone()));
        setup(&mut m);
        let stop = m.run();
        *self.stops.entry(match stop {
            Stop::Returned => "returned".into(),
            s => format!("{s:?}"),
        }).or_default() += 1;
        for (a, (n, _)) in std::mem::take(&mut m.mem.missing) {
            *self.missing.entry(a).or_default() += n;
        }
        m.mem.stored.clear();
        self.mem = std::mem::take(&mut m.mem);
        m
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(&args[1]);
    let container = std::fs::read(&args[2]).expect("container");
    let out = args.get(3).map_or_else(|| dir.join("out"), PathBuf::from);
    std::fs::create_dir_all(&out).expect("out dir");
    let mut cap = read_capture(&dir);
    let (f, _, native) = thread_ir(&container).expect("translation");
    let state = State { slots: std::mem::take(&mut cap.slots), written: BTreeMap::new(), notes: BTreeSet::new() };
    let images = Replay(Rc::new(RefCell::new(state)));
    let lanes = if native.wave64 { 64 } else { 32 };
    let mut run = Run { f: &f, cap: &cap, mem: memory(&cap), images: images.clone(), lanes, missing: BTreeMap::new(), stops: BTreeMap::new() };
    let stride = env("STRIDE", 1).max(1) as usize;
    let mut census = String::new();
    match (&native.io, native.stage) {
        (StageIo::Cs(io), _) => compute(&mut run, io.threads, stride),
        (_, HwStage::Ps) => pixels(&mut run, stride, &out, &mut census),
        _ => panic!("replay runs compute and pixel programs"),
    }
    let st = images.0.borrow();
    for (slot, texels) in &st.written {
        let (w, h, d, _) = st.dims(*slot);
        let (w, h) = (w as u32, (h * d) as u32);
        let mut px = vec![[0f32; 4]; (w * h) as usize];
        for (&i, v) in texels {
            if let Some(p) = px.get_mut(i) {
                *p = v.map(|x| f32::from_bits(x as u32));
            }
        }
        let name = st.slots.get(slot).map_or_else(|| format!("slot{slot:x}"), |s| s.name.clone());
        write_image(&out, &format!("stored-{name}"), w, h, &px, &mut census);
    }
    let mut notes: Vec<String> = st.notes.iter().cloned().collect();
    notes.push(format!("stops: {:?}", run.stops));
    notes.push("samplers: linear, clamp to edge (no sampler state captured); level 0 only".into());
    let _ = std::fs::write(out.join("census.txt"), &census);
    let _ = std::fs::write(out.join("missing.txt"), missing_ranges(&run.missing));
    let _ = std::fs::write(out.join("notes.txt"), notes.join("\n") + "\n");
    print!("{census}");
    println!("{} missing ranges; notes: {}", missing_ranges(&run.missing).lines().count(), notes.join("; "));
}

fn compute(run: &mut Run<'_>, threads: [u32; 3], stride: usize) {
    let [gx, gy, gz] = run.cap.dispatch.unwrap_or([1, 1, 1]);
    let per = (threads.iter().product::<u32>() as usize).div_ceil(run.lanes);
    let groups: Vec<[u32; 3]> = (0..gz).flat_map(|z| (0..gy).flat_map(move |y| (0..gx).map(move |x| [x, y, z]))).collect();
    for g in groups.iter().step_by(stride) {
        for wave in 0..per {
            run.wave(|m| {
                m.group = *g;
                m.wave = wave as u32;
                m.local_size = threads;
            });
        }
    }
}

fn pixels(run: &mut Run<'_>, stride: usize, out: &Path, census: &mut String) {
    let [vx, vy, vw, vh] = run.cap.viewport.unwrap_or([0.0, 0.0, 1280.0, 720.0]);
    let (w, h) = (vw as u32, vh as u32);
    let mut targets: BTreeMap<u16, Vec<[f32; 4]>> = BTreeMap::new();
    let blocks: Vec<[u32; 2]> = (0..h.div_ceil(4)).flat_map(|by| (0..w.div_ceil(8)).map(move |bx| [bx * 8, by * 4])).collect();
    for b in blocks.iter().step_by(stride) {
        let m = run.wave(|m| m.frag_block = Some([vx as u32 + b[0], vy as u32 + b[1]]));
        for (&(slot, comp, _, lane), &v) in &m.outputs {
            let q = lane as u32 / 4;
            let (x, y) = (b[0] + q % 4 * 2 + (lane as u32 & 1), b[1] + q / 4 * 2 + (lane as u32 >> 1 & 1));
            if x < w && y < h {
                let t = targets.entry(slot).or_insert_with(|| vec![[0f32; 4]; (w * h) as usize]);
                t[(y * w + x) as usize][usize::from(comp.min(3))] = f32::from_bits(v as u32);
            }
        }
    }
    for (slot, px) in &targets {
        write_image(out, &format!("target{slot}"), w, h, px, census);
    }
}
