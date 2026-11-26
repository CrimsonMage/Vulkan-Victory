//! The console's own list methods (`super::super::console`) and the
//! queries, written down on the title's threads like the others and
//! recorded later: into Vulkan command buffers (`crate::vulkan::record`,
//! its `console.rs` and `query.rs`), or on the D3D12 path through the list
//! methods.

use super::*;
use super::super::console as methods;
use crate::vulkan::Backend;
use crate::vulkan::record::{Rec, ValueWrite};

const CLEAR_UAV: u64 = after(19);
const FILL: u64 = after(20);
const DRAW_INDEXED_X: u64 = after(21);
const PREDICATION: u64 = after(22);
const WRITE: u64 = after(23);
const KICKOFF: u64 = after(24);
const DEPTH_BIAS: u64 = after(25);
const BEGIN_QUERY: u64 = after(26);
const END_QUERY: u64 = after(27);

/// The methods by their slot in the older layout, as written down.
pub(super) const METHODS: [(usize, *const ()); 14] = [
    (28, begin_query as *const ()),
    (29, end_query as *const ()),
    (53, clear_uav_uint as *const ()),
    (54, clear_uav_float as *const ()),
    (61, fill32_or_indirect as *const ()),
    (62, fill64 as *const ()),
    (63, fill128 as *const ()),
    (81, draw_indexed_x as *const ()),
    (85, predication_or_copy as *const ()),
    (88, write32_top as *const ()),
    (89, write64_top as *const ()),
    (90, write32_bottom_of_shader as *const ()),
    (93, kickoff as *const ()),
    (94, depth_bias as *const ()),
];

extern "system" fn begin_query(this: This, heap: This, kind: u32, index: u32) {
    front(this).put(BEGIN_QUERY, &[heap as u64, u64::from(kind), u64::from(index)]);
}
extern "system" fn end_query(this: This, heap: This, kind: u32, index: u32) {
    front(this).put(END_QUERY, &[heap as u64, u64::from(kind), u64::from(index)]);
}

/// The descriptor at a UAV handle as it is now (the console's own words:
/// a T# or a V#), so that its slot may be written again.
fn descriptor(gpu: u64, cpu: u64) -> [u32; 8] {
    let slot = if cpu != 0 { cpu } else { gpu };
    if slot == 0 {
        return [0; 8];
    }
    // SAFETY: a descriptor slot the title passes (32 bytes or more).
    unsafe { (slot as *const [u32; 8]).read_unaligned() }
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
fn clear_uav(this: This, float: bool, gpu: u64, cpu: u64, resource: This, values: *const [u32; 4], count: u32, rects: *const Rect) {
    if values.is_null() {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let words = descriptor(gpu, cpu);
    front(this).put_with(
        CLEAR_UAV,
        &[u64::from(float), resource as u64, u64::from(count)],
        &[bytes_of(&words, 1), bytes_of(values, 1), bytes_of(rects, count as usize)],
    );
}
extern "system" fn clear_uav_uint(this: This, gpu: u64, cpu: u64, resource: This, values: *const [u32; 4], count: u32, rects: *const Rect) {
    clear_uav(this, false, gpu, cpu, resource, values, count, rects);
}
extern "system" fn clear_uav_float(this: This, gpu: u64, cpu: u64, resource: This, values: *const [f32; 4], count: u32, rects: *const Rect) {
    clear_uav(this, true, gpu, cpu, resource, values.cast(), count, rects);
}

fn fill(this: This, address: u64, size: u64, pattern: [u8; 16], width: u64) {
    let p: [u64; 2] = [0, 8].map(|k| u64::from_le_bytes(std::array::from_fn(|i| pattern[k + i])));
    front(this).put(FILL, &[address, size, width, p[0], p[1]]);
}
extern "system" fn fill32(this: This, address: u64, size: u64, value: u32) {
    fill(this, address, size, methods::pattern(&value.to_le_bytes()), 4);
}
extern "system" fn fill64(this: This, address: u64, size: u64, value: u64) {
    fill(this, address, size, methods::pattern(&value.to_le_bytes()), 8);
}
extern "system" fn fill128(this: This, address: u64, size: u64, value: *const [u8; 16]) {
    // SAFETY: the title passes 16 bytes (null is tolerated).
    if let Some(&value) = unsafe { value.as_ref() } {
        fill(this, address, size, value, 16);
    }
}

/// The older layout's slot 61, or the newer's `ExecuteIndirect(signature,
/// max, arguments, offset, count, count offset)`: a command signature first
/// tells (`crate::layout`).
#[allow(clippy::too_many_arguments)] // the newer method's arguments
extern "system" fn fill32_or_indirect(this: This, a: u64, b: u64, c: u64, d: u64, e: u64, f: u64) {
    if !crate::layout::decided() && crate::layout::names(a, crate::signature::is_signature) {
        crate::layout::shown(true, "a command signature passed to list slot 61");
        return super::execute_indirect(this, a as This, b as u32, c as This, d, e as This, f);
    }
    fill32(this, a, b, c as u32);
}

/// The older layout's slot 85, or the newer's `CopyTextureRegionX(dst, x,
/// y, z, src, box, flags)`: a copy location of a resource first tells.
#[allow(clippy::too_many_arguments)] // the newer method's arguments
extern "system" fn predication_or_copy(this: This, a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64) {
    let location = |p: This| {
        // SAFETY: readable (checked by `names`), at least 16 bytes.
        let resource = unsafe { p.cast::<u64>().read_unaligned() };
        crate::layout::names(resource, crate::resource::is_resource)
    };
    if !crate::layout::decided() && crate::layout::names(a, location) {
        crate::layout::shown(true, "a copy location passed to list slot 85");
        let (dst, src) = (a as *const CopyLocation, e as *const CopyLocation);
        return super::copy_texture_region_x(this, dst, b as u32, c as u32, d as u32, src, f as *const [u32; 6], g as u32);
    }
    set_predication(this, a as *const u64);
}

extern "system" fn draw_indexed_x(this: This, address: u64, format: u32, count: u32, _flags: u32) {
    front(this).put(DRAW_INDEXED_X, &[address, u64::from(format), u64::from(count)]);
}

/// `{address, operation}` copied at the call (null: off).
extern "system" fn set_predication(this: This, desc: *const u64) {
    if desc.is_null() {
        front(this).put(PREDICATION, &[0, 0, 0]);
        return;
    }
    // SAFETY: the title passes the console's 12-byte description.
    let (address, op) = unsafe { (desc.read_unaligned(), desc.add(1).cast::<u32>().read_unaligned()) };
    front(this).put(PREDICATION, &[1, address, u64::from(op)]);
}

extern "system" fn write32_top(this: This, address: u64, value: u32, _flags: u32) {
    front(this).put(WRITE, &[address, u64::from(value), 4, ValueWrite::Top as u64]);
}
extern "system" fn write64_top(this: This, address: u64, value: u64, flags: u32) {
    let kind = if flags & methods::TOP_GPU_TIMESTAMP != 0 {
        ValueWrite::TopTimestamp
    } else if flags & methods::TOP_SYSTEM_TIMESTAMP != 0 {
        ValueWrite::TopSystemTimestamp
    } else {
        ValueWrite::Top
    };
    front(this).put(WRITE, &[address, value, 8, kind as u64]);
}
extern "system" fn write32_bottom_of_shader(this: This, address: u64, value: u32, _stage: u32) {
    front(this).put(WRITE, &[address, u64::from(value), 4, ValueWrite::BottomOfShader as u64]);
}

extern "system" fn kickoff(this: This) {
    front(this).put(KICKOFF, &[]);
}

extern "system" fn depth_bias(this: This, bias: f32, clamp: f32, slope: f32) {
    front(this).put(DEPTH_BIAS, &[bias.word(), clamp.word(), slope.word()]);
}

/// One of these calls on the D3D12 path, through the list methods; whether
/// `op` was one.
pub(super) fn run(op: u64, list: This, w: &[u64]) -> bool {
    let u = |k: usize| w.get(k).copied().unwrap_or(0);
    match op {
        CLEAR_UAV => methods::clear_uav(list),
        FILL => {
            let mut pattern = [0u8; 16];
            pattern[..8].copy_from_slice(&u(3).to_le_bytes());
            pattern[8..].copy_from_slice(&u(4).to_le_bytes());
            methods::fill(list, u(0), u(1), pattern);
        }
        DRAW_INDEXED_X => methods::draw_indexed_x(list),
        PREDICATION => methods::set_predication(list),
        WRITE => match u(3) {
            x if x == ValueWrite::Top as u64 && u(2) == 4 => methods::write32_top(list, u(0), u(1) as u32, 0),
            x if x == ValueWrite::BottomOfShader as u64 => methods::write32_bottom_of_shader(list, u(0), u(1) as u32, 0),
            x if x == ValueWrite::TopTimestamp as u64 => methods::write64_top(list, u(0), 0, methods::TOP_GPU_TIMESTAMP),
            x if x == ValueWrite::TopSystemTimestamp as u64 => methods::write64_top(list, u(0), 0, methods::TOP_SYSTEM_TIMESTAMP),
            _ => methods::write64_top(list, u(0), u(1), 0),
        },
        DEPTH_BIAS => methods::depth_bias(list),
        KICKOFF | BEGIN_QUERY | END_QUERY => {}
        _ => return false,
    }
    true
}

/// One of these calls recorded into the list's Vulkan command buffer;
/// whether `op` was one.
pub(super) fn record(rec: &mut Rec, b: &'static Backend, list: This, op: u64, w: &[u64]) -> bool {
    let u = |k: usize| w.get(k).copied().unwrap_or(0);
    match op {
        CLEAR_UAV => {
            let n = u(2) as usize;
            // SAFETY: written whole by `clear_uav`.
            let (words, values) = unsafe { (at_or::<[u32; 8]>(w, 3).read_unaligned(), at_or::<[u32; 4]>(w, 7).read_unaligned()) };
            let rects = at::<Rect>(w, 9, n);
            // SAFETY: `n` rectangles follow (none: null).
            let rects = if rects.is_null() { &[][..] } else { unsafe { std::slice::from_raw_parts(rects, n) } };
            rec.clear_unordered(b, u(1) as This, words, values, u(0) != 0, rects);
        }
        FILL => {
            let mut pattern = [0u8; 16];
            pattern[..8].copy_from_slice(&u(3).to_le_bytes());
            pattern[8..].copy_from_slice(&u(4).to_le_bytes());
            rec.fill(b, u(0), u(1), pattern, u(2) as usize);
        }
        DRAW_INDEXED_X => rec.draw_indexed_x(b, u(0), u(1) as u32, u(2) as u32),
        PREDICATION => rec.predication(b, (u(0) != 0).then(|| (u(1), u(2) as u32))),
        WRITE => rec.write_value(b, list, u(0), u(1), u(2) as usize, ValueWrite::from_word(u(3))),
        KICKOFF => rec.kickoff(list, b),
        DEPTH_BIAS => rec.depth_bias([u(0), u(1), u(2)].map(f32::unword)),
        BEGIN_QUERY => rec.begin_query(b, u(0) as This, u(1) as u32, u(2) as u32),
        END_QUERY => rec.end_query(b, u(0) as This, u(1) as u32, u(2) as u32),
        _ => return false,
    }
    true
}
