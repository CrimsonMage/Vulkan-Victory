//! [`super::translate`] under `VICTORY_GPU=vulkan`: a stream's calls are
//! recorded into Vulkan command buffers (`crate::vulkan::record`). Waits,
//! writes and resets still go through the list's methods, which hand them
//! to the Vulkan backend's queue lines.

use super::*;
use crate::vulkan::Backend;
use crate::vulkan::record::{self, Rec};

/// One call of a stream: its op and words.
struct Call<'a> {
    op: u64,
    w: &'a [u64],
}

impl Call<'_> {
    fn u(&self, k: usize) -> u64 {
        self.w.get(k).copied().unwrap_or(0)
    }

    /// The first word as a count.
    fn n(&self) -> usize {
        self.u(0) as usize
    }

    /// `n` things of type `T` from word `from` (none if absent).
    fn items<T>(&self, from: usize, n: usize) -> &[T] {
        let p = at::<T>(self.w, from, n);
        if p.is_null() {
            &[]
        } else {
            // SAFETY: the stream holds `n` of them from word `from`.
            unsafe { std::slice::from_raw_parts(p, n) }
        }
    }

    /// One thing of type `T` copied from word `from`.
    fn read<T>(&self, from: usize) -> T {
        // SAFETY: the thing was copied whole into the stream there.
        unsafe { at_or::<T>(self.w, from).read_unaligned() }
    }
}

pub(super) fn translate(list: This, stream: &[u64]) -> (u64, bool) {
    let Some(b) = crate::vulkan::backend_for_recording() else {
        return (0, false);
    };
    let started = std::time::Instant::now();
    let mut rec = record::take(list);
    let (mut i, mut calls, mut reset) = (0, 0, false);
    while i < stream.len() {
        let (op, len) = (stream[i] & 0xffff, (stream[i] >> 16) as usize);
        let call = Call { op, w: &stream[i + 1..i + 1 + len] };
        i += 1 + len;
        calls += 1;
        reset |= one(&mut rec, b, list, &call);
    }
    record::put(list, rec);
    crate::vulkan::split(crate::vulkan::Split::Record, started);
    (calls, reset)
}

/// Records one call; whether it was a reset.
fn one(rec: &mut Rec, b: &'static Backend, list: This, c: &Call) -> bool {
    if plain_state(rec, c) || plain_work(rec, b, list, c) || state(rec, c) {
        return false;
    }
    work(rec, b, list, c)
}

/// The plain calls that set state.
fn plain_state(rec: &mut Rec, c: &Call) -> bool {
    let op = c.op;
    let u = |k| c.u(k);
    match op {
        x if x == Plain::TOPOLOGY as u64 => rec.topology(u(0) as u32),
        x if x == Plain::STENCIL_REF as u64 => rec.stencil_ref(u(0) as u32),
        x if x == Plain::DEPTH_BOUNDS as u64 => rec.depth_bounds(f32::unword(u(0)), f32::unword(u(1))),
        x if x == Plain::COMPUTE_TABLE as u64 => rec.table(true, u(0) as u32, u(1)),
        x if x == Plain::GRAPHICS_TABLE as u64 => rec.table(false, u(0) as u32, u(1)),
        x if x == Plain::COMPUTE_CONSTANT as u64 => rec.constants(true, u(0) as u32, u(2) as u32, &[u(1) as u32]),
        x if x == Plain::GRAPHICS_CONSTANT as u64 => rec.constants(false, u(0) as u32, u(2) as u32, &[u(1) as u32]),
        x if x == Plain::COMPUTE_VIEW as u64 => rec.view(true, u(0) as u32, u(1)),
        x if x == Plain::GRAPHICS_VIEW as u64 => rec.view(false, u(0) as u32, u(1)),
        x if x == Plain::SET_PIPELINE as u64 => rec.set_pipeline(u(0) as This),
        _ => return false,
    }
    true
}

/// The plain calls that do work (and the ones left to the list methods).
fn plain_work(rec: &mut Rec, b: &'static Backend, list: This, c: &Call) -> bool {
    let op = c.op;
    let u = |k| c.u(k);
    let is = |p: Plain| op == p as u64;
    if op >= Plain::END as u64 {
        return false;
    }
    if is(Plain::DRAW) {
        rec.draw(b, u(0) as u32, u(1) as u32, u(2) as u32, u(3) as u32);
    } else if is(Plain::DRAW_INDEXED) {
        rec.draw_indexed(b, u(0) as u32, u(1) as u32, u(2) as u32, u(3) as i32, u(4) as u32);
    } else if is(Plain::DISPATCH) || is(Plain::DISPATCH_X) {
        rec.dispatch(b, u(0) as u32, u(1) as u32, u(2) as u32);
    } else if is(Plain::DISPATCH_MESH) {
        rec.draw_mesh(b, u(0) as u32, u(1) as u32, u(2) as u32);
    } else if is(Plain::DISPATCH_MESH_X) {
        rec.draw_mesh(b, u(0) as u32, 1, 1);
    } else if is(Plain::COPY_BUFFER) {
        rec.copy_buffer_region(b, u(0) as This, u(1), u(2) as This, u(3), u(4));
    } else if is(Plain::COPY_RESOURCE) || is(Plain::COPY_RESOURCE_X) {
        rec.copy_resource(b, u(0) as This, u(1) as This);
    } else if is(Plain::COPY_MEMORY) {
        rec.copy_memory(b, u(0), u(1), u(2));
    } else if is(Plain::RESOLVE) {
        rec.resolve(b, u(0) as This, u(1) as u32, u(2) as This, u(3) as u32);
    } else if is(Plain::EXECUTE_INDIRECT) || is(Plain::RESOLVE_QUERIES) {
        rec.not_yet();
    } else {
        if is(Plain::WAIT_VALUE) || is(Plain::WAIT_VALUE64) {
            // What was recorded so far goes before the wait.
            rec.cut(list, b);
        }
        // Waits and bottom-of-pipe writes: the list's methods.
        run_plain(op, list, c.w);
    }
    true
}

/// The other calls that set state.
fn state(rec: &mut Rec, c: &Call) -> bool {
    let n = c.n();
    match c.op {
        VIEWPORTS => rec.viewports(c.items::<D3D12_VIEWPORT>(1, n)),
        SCISSORS => rec.scissors(c.items::<RECT>(1, n)),
        BLEND_FACTOR => rec.blend_factor(c.items::<[f32; 4]>(1, n).first().copied()),
        HEAPS => rec.heaps(c.items::<This>(1, n)),
        COMPUTE_CONSTANTS | GRAPHICS_CONSTANTS => {
            rec.constants(c.op == COMPUTE_CONSTANTS, c.u(0) as u32, (c.u(0) >> 32) as u32, c.items::<u32>(2, c.u(1) as usize))
        }
        INDEX_BUFFER => rec.index_buffer((n != 0).then(|| c.read::<IndexView>(1))),
        VERTEX_BUFFERS => vertex_buffers(rec, c),
        TARGETS => targets(rec, c),
        _ => return false,
    }
    true
}

fn vertex_buffers(rec: &mut Rec, c: &Call) {
    let (start, count) = (c.u(0) as u32, (c.u(0) >> 32) as usize);
    if c.u(1) != 0 && !at::<VertexView>(c.w, 2, count).is_null() {
        let views: Vec<VertexView> = (0..count).map(|k| c.read::<VertexView>(2 + k * size_of::<VertexView>() / 8)).collect();
        rec.vertex_buffers(start, &views);
    }
}

fn targets(rec: &mut Rec, c: &Call) {
    let n = c.n();
    let colors: Vec<Option<RtvRecord>> = (0..n).map(|k| (c.u(1) >> k & 1 != 0).then(|| c.read::<RtvRecord>(3 + k * RECORD))).collect();
    let depth = (c.u(2) != 0).then(|| c.read::<RtvRecord>(3 + n * RECORD));
    rec.targets(&colors, depth);
}

/// The other calls that do work; whether it was a reset.
fn work(rec: &mut Rec, b: &'static Backend, list: This, c: &Call) -> bool {
    let n = c.n();
    match c.op {
        RESET => {
            rec.reset(b, c.u(0) as This);
            let allocator: This = std::ptr::NonNull::<Object>::dangling().as_ptr().cast_const();
            let _ = super::super::reset(list, allocator, c.u(0) as This);
            return true;
        }
        CLOSE => rec.cut(list, b),
        UPLOAD => {
            // SAFETY: made by `allocate_upload`, named once.
            let block = *unsafe { Box::from_raw(c.u(0) as *mut Vec<u8>) };
            // SAFETY: one of our lists.
            let uploads = unsafe { &(*list.cast::<List>()).uploads };
            uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
        }
        // SAFETY: made by `own`, named once (nothing of ours records into
        // Vulkan lists this way yet).
        OWN => drop(*unsafe { Box::from_raw(c.u(0) as *mut Own) }),
        BARRIER | BARRIER_GROUPS => rec.barrier(b),
        CLEAR_RTV => {
            record::rects_ignored(c.items::<Rect>(1 + RECORD + 2, n));
            rec.clear_color(b, &c.read::<RtvRecord>(1), c.read::<[f32; 4]>(1 + RECORD));
        }
        CLEAR_DSV => rec.clear_depth(b, &c.read::<RtvRecord>(3), c.u(1) as u32, f32::unword(c.u(2)), (c.u(1) >> 32) as u8),
        COPY_TEXTURE => {
            let step = size_of::<CopyLocation>().div_ceil(8);
            let boxed = ((c.u(1) >> 32) != 0).then(|| c.read::<[u32; 6]>(2 + 2 * step));
            let (dst, src) = (c.read::<CopyLocation>(2), c.read::<CopyLocation>(2 + step));
            rec.copy_texture(b, &dst, c.u(0) as u32, (c.u(0) >> 32) as u32, c.u(1) as u32, &src, boxed);
        }
        _ => {}
    }
    false
}
