//! List calls that are only written down on the title's threads; workers of
//! ours translate them into the host lists.
//!
//! Ported from VIctory-7900XTX (github.com/avatarmanz/VIctory-7900XTX,
//! GPL-3.0-or-later, `crates/victory-gpu/src/list/thin.rs`, its
//! `VICTORY_THIN`), adapted to this tree: the stream goes into the title's
//! own list (no second list of ours behind it), a submission waits for its
//! lists to be translated instead of going into a line of submissions, and
//! it is the only way lists record.
//!
//! Why: the title's recording threads spent ~600 ms (summed) a world frame
//! inside our list methods, the whole frame long. On the console such a
//! call writes a few words into the list's memory and the card reads them
//! when the list runs. Here a call appends its op and arguments to a stream
//! of words ([`Front`]), copying what its pointers point at, and returns.
//! Every [`CHUNK`] words the stream is handed to a pool of workers ([`start`]),
//! which run it, call by call and in order, through the list methods there
//! were before, on the same title list: one worker per list at a time, many
//! lists at once. `ExecuteCommandLists` waits until its lists are translated
//! ([`join`]) and goes on as before.
//!
//! What may be read later than at the call, and why that is all right: what
//! the console's card reads when the list runs (descriptor heaps, constants,
//! upload memory, indirect arguments, title memory) must stay as it is until
//! then, and the translation is done before the list is submitted. What
//! D3D12 reads at the call is copied at the call: the arrays and structures
//! passed, and the render target and depth views, whose descriptor slots may
//! be written again right after (their records are copied; the copy stands
//! for the handle). Objects named (pipelines, resources) live until the list
//! has run, as on the console.
//!
//! Who touches what: the title's thread only [`Front`]; the worker that runs
//! a stream everything else of the list (host list, draw state, parts,
//! upload blocks), and nobody else until the stream is idle, which
//! [`join`] and the list's release wait for. Workers translate different
//! lists at once, as the title's threads did before; what lists share was
//! made for that then.

mod vulkan;

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use super::*;
use crate::descriptor::RtvRecord;
use crate::draw::{IndexView, VertexView};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

/// Words after which a stream is handed to the workers.
const CHUNK: usize = 1024;

/// The workers translating streams (0: each call is translated at once on
/// the title's thread, through the same stream; `VICTORY_LIST_WORKERS`).
/// By default the fast cores (physical, not logical: their second threads
/// are the title's), 2 to 16; 8 on an i7-14700 (8 P-cores, 12 E-cores).
pub(crate) fn workers() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let asked = std::env::var("VICTORY_LIST_WORKERS").ok().and_then(|v| v.trim().parse::<usize>().ok());
        let n = asked.unwrap_or_else(|| crate::cores::fast_cores().clamp(2, 16)).min(32);
        if n > 0 {
            start(n);
        } else {
            tracing::info!("list calls are translated on the title's threads (VICTORY_LIST_WORKERS=0)");
        }
        n
    })
}

/// What a title list keeps on the title's side: its stream and the words
/// not handed over yet.
pub(super) struct Front {
    stream: Arc<Stream>,
    words: Vec<u64>,
}

impl Front {
    /// The front of the title list at `list` (its address, which stays).
    pub(super) fn new(list: usize) -> Front {
        Front { stream: Arc::new(Stream { list, inner: Mutex::default(), idle: Condvar::new() }), words: Vec::new() }
    }

    #[inline]
    fn put(&mut self, op: u64, args: &[u64]) {
        let began = crate::fast::ticks();
        self.words.push(op | (args.len() as u64) << 16);
        self.words.extend_from_slice(args);
        self.after(began);
    }

    /// [`Self::put`] of some words and then the bytes of some things.
    fn put_with(&mut self, op: u64, args: &[u64], blobs: &[&[u8]]) {
        let began = crate::fast::ticks();
        let at = self.words.len();
        self.words.push(0);
        self.words.extend_from_slice(args);
        for blob in blobs {
            let from = self.words.len();
            self.words.resize(from + blob.len().div_ceil(8), 0);
            // SAFETY: the words just made hold the bytes; plain data.
            unsafe {
                std::ptr::copy_nonoverlapping(blob.as_ptr(), self.words.as_mut_ptr().add(from).cast(), blob.len())
            };
        }
        let len = (self.words.len() - at - 1) as u64;
        self.words[at] = op | len << 16;
        self.after(began);
    }

    #[inline]
    fn after(&mut self, began: u64) {
        if self.words.len() >= CHUNK || workers() == 0 {
            self.hand_over();
        }
        crate::frametime::written_down(crate::fast::ticks().wrapping_sub(began));
    }

    /// Hands the words written so far to the workers (with none, translates
    /// them here).
    fn hand_over(&mut self) {
        if self.words.is_empty() {
            return;
        }
        // (Translating each call at once, a chunk is a call: the recording
        // keeps them all until its reset, so they are small.)
        let fresh = if workers() == 0 {
            Vec::new()
        } else {
            held(&WORDS).pop().unwrap_or_else(|| {
                let _owner = victory_core::heap::enter(victory_core::heap::Owner::Recording);
                Vec::with_capacity(CHUNK + 256)
            })
        };
        let words = std::mem::replace(&mut self.words, fresh);
        let schedule = {
            let mut inner = held(&self.stream.inner);
            inner.chunks.push_back(words);
            !std::mem::replace(&mut inner.scheduled, true)
        };
        if !schedule {
            return;
        }
        if workers() == 0 {
            self.stream.run();
        } else {
            held(&READY).push_back(self.stream.clone());
            WORK.notify_one();
        }
    }

    /// Hands over what is left and waits until the stream is translated.
    fn join(&mut self) {
        self.hand_over();
        let mut inner = held(&self.stream.inner);
        while inner.scheduled {
            inner = self.stream.idle.wait(inner).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Streams' memory to use again.
static WORDS: Mutex<Vec<Vec<u64>>> = Mutex::new(Vec::new());
/// Streams with words to translate, each once.
static READY: Mutex<VecDeque<Arc<Stream>>> = Mutex::new(VecDeque::new());
static WORK: Condvar = Condvar::new();

fn held<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The stream of one title list.
struct Stream {
    /// The title list (its address).
    list: usize,
    inner: Mutex<Inner>,
    /// Told when the stream has nothing left to translate.
    idle: Condvar,
}

#[derive(Default)]
struct Inner {
    chunks: VecDeque<Vec<u64>>,
    /// Chunks of the current recording that were translated: the draw state
    /// keeps handles into them (render target records) until the next reset.
    kept: Vec<Vec<u64>>,
    /// It has words handed over and not translated: among [`READY`] or being
    /// translated.
    scheduled: bool,
}

impl Stream {
    /// Translates what there is.
    fn run(&self) {
        let list = self.list as This;
        loop {
            let chunk = {
                let mut inner = held(&self.inner);
                let Some(chunk) = inner.chunks.pop_front() else {
                    inner.scheduled = false;
                    self.idle.notify_all();
                    return;
                };
                chunk
            };
            let began = crate::fast::ticks();
            let (calls, reset) = translate(list, &chunk);
            crate::frametime::translated(crate::fast::ticks().wrapping_sub(began), calls);
            let mut inner = held(&self.inner);
            if reset {
                give_back(std::mem::take(&mut inner.kept));
            }
            inner.kept.push(chunk);
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        give_back(std::mem::take(&mut inner.kept));
    }
}

fn give_back(streams: Vec<Vec<u64>>) {
    let mut words = held(&WORDS);
    for mut s in streams {
        if words.len() < 1024 && s.capacity() <= 16 * CHUNK {
            s.clear();
            words.push(s);
        }
    }
}

fn start(threads: usize) {
    for i in 0..threads {
        let worker = move || {
            crate::cores::keep_on_fast_cores("victory-list");
            victory_core::heap::enter_for_thread(victory_core::heap::Owner::Recording);
            loop {
                let stream = {
                    let mut ready = held(&READY);
                    loop {
                        match ready.pop_front() {
                            Some(s) => break s,
                            None => ready = WORK.wait(ready).unwrap_or_else(|e| e.into_inner()),
                        }
                    }
                };
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stream.run())).is_err() {
                    crate::host::fail(format_args!("a worker translating list calls panicked"));
                }
            }
        };
        if let Err(e) = std::thread::Builder::new().name(format!("victory-list-{i}")).spawn(worker) {
            tracing::warn!(error = %e, "a worker less to translate list calls");
        }
    }
    tracing::info!(threads, "list calls are written down on the title's threads and translated by workers of ours");
}

/// Whether `this` is a list whose calls are written down.
pub(super) fn is_thin(this: This) -> bool {
    object::has_table(this, &TABLE) || object::has_table(this, &NEWER_TABLE)
}

/// Waits until the submitted `lists` are translated (each to its close).
/// Before anything of a submission looks at its lists.
pub(crate) fn join(lists: &[This]) {
    let began = crate::fast::ticks();
    for &l in lists {
        if is_thin(l) {
            front(l).join();
        }
    }
    crate::frametime::joined(crate::fast::ticks().wrapping_sub(began));
}

/// A list of ours goes: its stream is translated first (the workers hold
/// the list until then).
pub(super) fn release(this: This) {
    // Only the last release lets it go.
    let last = !this.is_null() && object::references(this) <= 1;
    if last && is_thin(this) {
        front(this).join();
    }
}

#[inline]
fn front<'a>(this: This) -> &'a mut Front {
    // SAFETY: the title passes one of our lists; a list is recorded (or
    // submitted) by one title thread at a time, and only title threads touch
    // its front.
    unsafe { &mut *std::ptr::addr_of_mut!((*this.cast_mut().cast::<List>()).thin) }
}

/// An argument as a word.
trait Word: Copy {
    fn word(self) -> u64;
    fn unword(w: u64) -> Self;
}
macro_rules! words {
    ($($t:ty),*) => { $(impl Word for $t {
        #[inline]
        fn word(self) -> u64 { self as u64 }
        #[inline]
        fn unword(w: u64) -> Self { w as $t }
    })* };
}
words!(u8, u32, i32, u64, This);
impl Word for f32 {
    #[inline]
    fn word(self) -> u64 {
        u64::from(self.to_bits())
    }
    #[inline]
    fn unword(w: u64) -> Self {
        f32::from_bits(w as u32)
    }
}

/// The methods whose arguments are all plain values: the one the title calls,
/// and the call made of it later.
macro_rules! plain {
    ($($op:ident $name:ident = $was:path, ($($a:ident: $t:ty),*);)*) => {
        #[allow(clippy::upper_case_acronyms, non_camel_case_types)]
        #[repr(u64)]
        enum Plain { $($op),* , END }
        $(extern "system" fn $name(this: This $(, $a: $t)*) {
            front(this).put(Plain::$op as u64, &[$(Word::word($a)),*]);
        })*
        /// Whether `op` was one of them; done on `list`.
        #[inline]
        fn run_plain(op: u64, list: This, w: &[u64]) -> bool {
            match op {
                $(n if n == Plain::$op as u64 => {
                    let mut words = w.iter();
                    $(let $a: $t = Word::unword(words.next().copied().unwrap_or(0));)*
                    let _ = words;
                    $was(list $(, $a)*);
                })*
                _ => return false,
            }
            true
        }
    };
}

plain! {
    TOPOLOGY topology = state::topology, (v: u32);
    STENCIL_REF stencil_ref = state::stencil_ref, (v: u32);
    DEPTH_BOUNDS depth_bounds = state::depth_bounds, (min: f32, max: f32);
    COMPUTE_TABLE compute_table = state::compute_table, (slot: u32, handle: u64);
    GRAPHICS_TABLE graphics_table = state::graphics_table, (slot: u32, handle: u64);
    COMPUTE_CONSTANT compute_constant = state::compute_constant, (slot: u32, value: u32, offset: u32);
    GRAPHICS_CONSTANT graphics_constant = state::graphics_constant, (slot: u32, value: u32, offset: u32);
    COMPUTE_VIEW compute_view = state::compute_view, (slot: u32, address: u64);
    GRAPHICS_VIEW graphics_view = state::graphics_view, (slot: u32, address: u64);
    SET_PIPELINE set_pipeline = super::set_pipeline, (pipeline: This);
    DRAW draw = state::draw, (vertices: u32, instances: u32, first: u32, first_instance: u32);
    DRAW_INDEXED draw_indexed = state::draw_indexed, (indices: u32, instances: u32, first: u32, base: i32, first_instance: u32);
    DISPATCH dispatch = state::dispatch, (x: u32, y: u32, z: u32);
    DISPATCH_X dispatch_x = super::dispatch_x, (x: u32, y: u32, z: u32, flags: u32);
    DISPATCH_MESH dispatch_mesh = state::dispatch_mesh, (x: u32, y: u32, z: u32);
    DISPATCH_MESH_X dispatch_mesh_x = state::dispatch_mesh_x, (x: u32, flags: u64, indirect: u64, payload: u64);
    COPY_BUFFER copy_buffer_region = super::copy_buffer_region, (dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64);
    COPY_RESOURCE copy_resource = super::copy_resource, (dst: This, src: This);
    COPY_RESOURCE_X copy_resource_x = super::copy_resource_x, (dst: This, src: This, flags: u32);
    COPY_MEMORY copy_memory = super::copy_memory, (dst: u64, src: u64, len: u64, flags: u32);
    RESOLVE resolve = super::resolve, (dst: This, dst_sub: u32, src: This, src_sub: u32, format: u32);
    RESOLVE_QUERIES resolve_queries = super::resolve_queries, (heap: This, kind: u32, first: u32, count: u32, buffer: This, offset: u64);
    EXECUTE_INDIRECT execute_indirect = super::execute_indirect, (signature: This, max: u32, arguments: This, offset: u64, count: This, count_offset: u64);
    WRITE32 write32 = super::write32, (address: u64, value: u32, flush: u32, flags: u32);
    WRITE64 write64 = super::write64, (address: u64, value: u64, flush: u32, flags: u32);
    WAIT_VALUE wait_value = super::wait_value, (address: u64, comparison: u32, reference: u64);
    WAIT_VALUE64 wait_value64 = super::wait_value64, (address: u64, comparison: u32, reference: u64);
}

const fn after(n: u64) -> u64 {
    Plain::END as u64 + n
}
const RESET: u64 = after(0);
const CLOSE: u64 = after(1);
const UPLOAD: u64 = after(2);
const BARRIER: u64 = after(3);
const BARRIER_GROUPS: u64 = after(4);
const VIEWPORTS: u64 = after(5);
const SCISSORS: u64 = after(6);
const BLEND_FACTOR: u64 = after(7);
const HEAPS: u64 = after(8);
const COMPUTE_CONSTANTS: u64 = after(9);
const GRAPHICS_CONSTANTS: u64 = after(10);
const INDEX_BUFFER: u64 = after(11);
const VERTEX_BUFFERS: u64 = after(12);
const TARGETS: u64 = after(13);
const CLEAR_RTV: u64 = after(14);
const CLEAR_DSV: u64 = after(15);
const COPY_TEXTURE: u64 = after(16);
const OWN: u64 = after(17);

/// The bytes of `count` things at `items` (none if null).
fn bytes_of<'a, T>(items: *const T, count: usize) -> &'a [u8] {
    if items.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: the title passes `count` items.
    unsafe { std::slice::from_raw_parts(items.cast(), count * size_of::<T>()) }
}

/// Where the things are in a stream (null for none).
fn at<T>(w: &[u64], from: usize, count: usize) -> *const T {
    if count == 0 || from >= w.len() { std::ptr::null() } else { w[from..].as_ptr().cast() }
}

/// [`at`] where an empty array is not null.
fn at_or<T>(w: &[u64], from: usize) -> *const T {
    // SAFETY: at most one past the end.
    unsafe { w.as_ptr().add(from.min(w.len())).cast() }
}

/// A view's record as it is now (zeros for no view), so that its slot may be
/// written again: the copy stands for the handle.
fn record(handle: u64) -> [u64; RECORD] {
    let mut words = [0u64; RECORD];
    if handle != 0 {
        // SAFETY: a descriptor slot the title passes (they are larger).
        unsafe {
            std::ptr::copy_nonoverlapping(handle as *const u8, words.as_mut_ptr().cast(), size_of::<RtvRecord>())
        };
    }
    words
}
const RECORD: usize = size_of::<RtvRecord>().div_ceil(8);

extern "system" fn list_close(this: This) -> HRESULT {
    let front = front(this);
    front.put(CLOSE, &[]);
    // Its tail is translated from now on.
    front.hand_over();
    S_OK
}
extern "system" fn list_reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    front(this).put(RESET, &[pipeline as u64]);
    S_OK
}
extern "system" fn list_reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    list_reset(this, allocator, pipeline)
}
/// The memory is handed out now; the list holds it from where its stream
/// stands ([`super::allocate_upload`]).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let mut block = vec![0u8; len];
    let start = block.as_mut_ptr();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let place = unsafe { start.add(start.align_offset(align)) };
    // (Moving the vector keeps its heap block where it is.)
    front(this).put(UPLOAD, &[Box::into_raw(Box::new(block)) as u64]);
    place
}
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    if !barriers.is_null() {
        front(this).put_with(BARRIER, &[u64::from(count)], &[bytes_of(barriers, count as usize)]);
    }
    0
}
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    front(this).put(BARRIER_GROUPS, &[u64::from(count)]);
}
extern "system" fn viewports(this: This, count: u32, items: *const D3D12_VIEWPORT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(VIEWPORTS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn scissors(this: This, count: u32, items: *const RECT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(SCISSORS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
    let given = !factor.is_null();
    front(this).put_with(BLEND_FACTOR, &[u64::from(given)], &[bytes_of(factor, usize::from(given))]);
}
extern "system" fn heaps(this: This, count: u32, items: *const This) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(HEAPS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
fn constants(this: This, op: u64, slot: u32, count: u32, values: *const u32, offset: u32) {
    let count = if values.is_null() { 0 } else { count };
    front(this).put_with(
        op,
        &[u64::from(slot) | u64::from(offset) << 32, u64::from(count)],
        &[bytes_of(values, count as usize)],
    );
}
extern "system" fn compute_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, COMPUTE_CONSTANTS, slot, count, values, offset);
}
extern "system" fn graphics_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, GRAPHICS_CONSTANTS, slot, count, values, offset);
}
extern "system" fn index_buffer(this: This, view: *const IndexView) {
    let given = !view.is_null();
    front(this).put_with(INDEX_BUFFER, &[u64::from(given)], &[bytes_of(view, usize::from(given))]);
}
extern "system" fn vertex_buffers(this: This, start: u32, count: u32, views: *const VertexView) {
    let given = !views.is_null();
    front(this).put_with(
        VERTEX_BUFFERS,
        &[u64::from(start) | u64::from(count) << 32, u64::from(given)],
        &[bytes_of(views, count as usize)],
    );
}
extern "system" fn targets(this: This, count: u32, handles: *const u64, one_range: i32, depth: *const u64) {
    let count = if handles.is_null() { 0 } else { (count as usize).min(8) };
    // SAFETY: the title passes a handle or null.
    let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
    let mut records = [0u64; RECORD * 9];
    let mut present = 0u64;
    for i in 0..count {
        // SAFETY: the title passes `count` handles, or the first of a range
        // of views of 128 bytes.
        let handle = unsafe { if one_range != 0 { *handles + 128 * i as u64 } else { *handles.add(i) } };
        present |= u64::from(handle != 0) << i;
        records[i * RECORD..(i + 1) * RECORD].copy_from_slice(&record(handle));
    }
    records[count * RECORD..(count + 1) * RECORD].copy_from_slice(&record(depth));
    let records = &records[..(count + 1) * RECORD];
    front(this).put_with(
        TARGETS,
        &[count as u64, present, u64::from(depth != 0)],
        &[bytes_of(records.as_ptr(), records.len())],
    );
}
extern "system" fn clear_rtv(this: This, handle: u64, color: *const [f32; 4], count: u32, rects: *const Rect) {
    if handle == 0 || color.is_null() {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_RTV,
        &[u64::from(count)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(color, 1), bytes_of(rects, count as usize)],
    );
}
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    if handle == 0 {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_DSV,
        &[u64::from(count), u64::from(flags) | u64::from(stencil) << 32, Word::word(depth)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(rects, count as usize)],
    );
}
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    if dst.is_null() || src.is_null() {
        return;
    }
    let boxed = !source_box.is_null();
    front(this).put_with(
        COPY_TEXTURE,
        &[u64::from(x) | u64::from(y) << 32, u64::from(z) | u64::from(boxed) << 32],
        &[bytes_of(dst, 1), bytes_of(src, 1), bytes_of(source_box, usize::from(boxed))],
    );
}
#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

/// Work of ours recorded into a title list where its recording stands
/// ([`super::record_own`]), run by the worker in its place in the stream.
pub(super) type Own = Box<dyn FnOnce(&crate::host::HostList) -> bool>;

/// [`super::record_own`] of a list whose calls are written down.
pub(super) fn own(this: This, f: Own) {
    front(this).put(OWN, &[Box::into_raw(Box::new(f)) as u64]);
}

/// Makes the calls of a stream on `list`: how many, and whether one of them
/// was a reset (what the draw state kept of the stream before is let go).
fn translate(list: This, stream: &[u64]) -> (u64, bool) {
    if crate::vulkan::on() {
        return vulkan::translate(list, stream);
    }
    let (mut i, mut calls, mut reset) = (0, 0, false);
    while i < stream.len() {
        let (op, len) = (stream[i] & 0xffff, (stream[i] >> 16) as usize);
        let w = &stream[i + 1..i + 1 + len];
        i += 1 + len;
        calls += 1;
        if run_plain(op, list, w) {
            continue;
        }
        let n = w.first().map_or(0, |&n| n as usize);
        match op {
            RESET => {
                // (Only whether there is an allocator is looked at.)
                let allocator: This = std::ptr::NonNull::<Object>::dangling().as_ptr().cast_const();
                let _ = super::reset(list, allocator, w[0] as This);
                reset = true;
            }
            CLOSE => {
                let _ = super::close(list);
            }
            UPLOAD => {
                // SAFETY: made by `allocate_upload`, named once.
                let block = *unsafe { Box::from_raw(w[0] as *mut Vec<u8>) };
                // SAFETY: one of our lists.
                let uploads = unsafe { &(*list.cast::<List>()).uploads };
                uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
            }
            OWN => {
                // SAFETY: made by `own`, named once.
                let f = *unsafe { Box::from_raw(w[0] as *mut Own) };
                super::record_own_here(list, f);
            }
            BARRIER => {
                super::barrier(list, n as u32, at_or(w, 1));
            }
            BARRIER_GROUPS => super::barrier_groups(list, n as u32, std::ptr::null()),
            VIEWPORTS => state::viewports(list, n as u32, at(w, 1, n)),
            SCISSORS => state::scissors(list, n as u32, at(w, 1, n)),
            BLEND_FACTOR => state::blend_factor(list, at(w, 1, n)),
            HEAPS => state::heaps(list, n as u32, at(w, 1, n)),
            COMPUTE_CONSTANTS => {
                state::compute_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            GRAPHICS_CONSTANTS => {
                state::graphics_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            INDEX_BUFFER => state::index_buffer(list, at(w, 1, n)),
            VERTEX_BUFFERS => {
                let views = if w[1] != 0 { at_or(w, 2) } else { std::ptr::null() };
                state::vertex_buffers(list, w[0] as u32, (w[0] >> 32) as u32, views);
            }
            TARGETS => {
                let mut handles = [0u64; 8];
                let place = |k: usize| w[3 + k * RECORD..].as_ptr() as u64;
                for (k, h) in handles.iter_mut().enumerate().take(n) {
                    if w[1] >> k & 1 != 0 {
                        *h = place(k);
                    }
                }
                let depth = if w[2] != 0 { place(n) } else { 0 };
                state::targets(list, n as u32, handles.as_ptr(), 0, &depth);
            }
            CLEAR_RTV => {
                super::clear_rtv(list, w[1..].as_ptr() as u64, at_or(w, 1 + RECORD), n as u32, at(w, 1 + RECORD + 2, n));
            }
            CLEAR_DSV => {
                super::clear_dsv(
                    list,
                    w[3..].as_ptr() as u64,
                    w[1] as u32,
                    Word::unword(w[2]),
                    (w[1] >> 32) as u8,
                    n as u32,
                    at(w, 3 + RECORD, n),
                );
            }
            COPY_TEXTURE => {
                let step = size_of::<CopyLocation>().div_ceil(8);
                let boxed = (w[1] >> 32) as usize;
                super::copy_texture_region(
                    list,
                    at_or(w, 2),
                    w[0] as u32,
                    (w[0] >> 32) as u32,
                    w[1] as u32,
                    at_or(w, 2 + step),
                    at(w, 2 + 2 * step, boxed),
                );
            }
            _ => {}
        }
    }
    (calls, reset)
}

/// The table of a list whose calls are written down: that of the others with
/// the methods that record replaced (slots as in [`super::METHODS`]).
const fn table(newer: bool) -> [*const (); 160] {
    let mut f = super::list_table(newer);
    let mut i = 0;
    while i < MINE.len() {
        let (slot, method) = MINE[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    if newer {
        f[146] = barrier_groups as *const ();
        f[148] = allocate_upload as *const ();
    }
    f
}

pub(super) static TABLE: Table<160> = Table(table(false));
pub(super) static NEWER_TABLE: Table<160> = Table(table(true));

const MINE: [(usize, *const ()); 48] = [
    (9, list_close as *const ()),
    (10, list_reset as *const ()),
    (12, draw as *const ()),
    (13, draw_indexed as *const ()),
    (14, dispatch as *const ()),
    (15, copy_buffer_region as *const ()),
    (16, copy_texture_region as *const ()),
    (17, copy_resource as *const ()),
    (19, resolve as *const ()),
    (20, topology as *const ()),
    (21, viewports as *const ()),
    (22, scissors as *const ()),
    (23, blend_factor as *const ()),
    (24, stencil_ref as *const ()),
    (25, set_pipeline as *const ()),
    (26, barrier as *const ()),
    (30, resolve_queries as *const ()),
    (32, heaps as *const ()),
    (35, compute_table as *const ()),
    (36, graphics_table as *const ()),
    (37, compute_constant as *const ()),
    (38, graphics_constant as *const ()),
    (39, compute_constants as *const ()),
    (40, graphics_constants as *const ()),
    (41, compute_view as *const ()),
    (42, graphics_view as *const ()),
    (43, compute_view as *const ()),
    (44, graphics_view as *const ()),
    (45, compute_view as *const ()),
    (46, graphics_view as *const ()),
    (47, index_buffer as *const ()),
    (48, vertex_buffers as *const ()),
    (50, targets as *const ()),
    (51, clear_dsv as *const ()),
    (52, clear_rtv as *const ()),
    (59, execute_indirect as *const ()),
    (60, copy_memory as *const ()),
    (68, depth_bounds as *const ()),
    (69, write32 as *const ()),
    (70, write64 as *const ()),
    (71, wait_value as *const ()),
    (72, wait_value64 as *const ()),
    (78, dispatch_x as *const ()),
    (83, copy_texture_region_x as *const ()),
    (84, copy_resource_x as *const ()),
    (98, dispatch_mesh_x as *const ()),
    (127, dispatch_mesh as *const ()),
    (137, list_reset_x as *const ()),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every method of the tables that records is written down: what is left
    /// in both tables alike records nothing.
    #[test]
    fn every_recording_method_is_written_down() {
        // IUnknown, object, device child, type; queries; root signatures,
        // depth order, driver hints, shading rate; syncs.
        const NOTHING: [usize; 18] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 28, 29, 33, 34, 73, 99, 124, 129, 130];
        let mine: Vec<usize> = MINE.iter().map(|m| m.0).collect();
        for (slot, _) in METHODS.iter().chain(MORE_METHODS.iter()) {
            assert!(mine.contains(slot) != NOTHING.contains(slot), "slot {slot}");
        }
        for &slot in &mine {
            assert!(
                METHODS.iter().chain(MORE_METHODS.iter()).any(|m| m.0 == slot),
                "slot {slot} is written down and not there"
            );
        }
    }

    /// Arguments come out of a stream as they went in.
    #[test]
    fn a_stream_gives_back_what_was_written() {
        let mut f = Front::new(0);
        // (Kept below a chunk: nothing is handed over.)
        f.words.reserve(64);
        let put = |f: &mut Front, op: u64, args: &[u64], blobs: &[&[u8]]| {
            let at = f.words.len();
            f.words.push(0);
            f.words.extend_from_slice(args);
            for blob in blobs {
                let from = f.words.len();
                f.words.resize(from + blob.len().div_ceil(8), 0);
                // SAFETY: as in `put_with`.
                unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr(), f.words.as_mut_ptr().add(from).cast(), blob.len()) };
            }
            let len = (f.words.len() - at - 1) as u64;
            f.words[at] = op | len << 16;
        };
        put(&mut f, Plain::DRAW_INDEXED as u64, &[3, 1, 0, Word::word(-7i32), 5], &[]);
        let viewport = [1.0f32, 2.0, 3.0, 4.0, 0.0, 1.0];
        put(&mut f, VIEWPORTS, &[1], &[bytes_of(viewport.as_ptr(), 6)]);
        put(&mut f, COMPUTE_CONSTANTS, &[4 | 2 << 32, 3], &[bytes_of([7u32, 8, 9].as_ptr(), 3)]);
        let s = std::mem::take(&mut f.words);
        assert_eq!((s[0] & 0xffff, s[0] >> 16), (Plain::DRAW_INDEXED as u64, 5));
        assert_eq!(<i32 as Word>::unword(s[4]), -7);
        let w = &s[6..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16, w[1]), (VIEWPORTS, 4, 1));
        // SAFETY: what was written just now.
        assert_eq!(unsafe { *at::<[f32; 6]>(&w[1..], 1, 1) }, viewport);
        let w = &w[5..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16), (COMPUTE_CONSTANTS, 4));
        // SAFETY: as above.
        assert_eq!(unsafe { *at::<[u32; 3]>(&w[1..], 2, 3) }, [7, 8, 9]);
        assert!(at::<u32>(&w[1..], 2, 0).is_null());
    }

    /// A view's record is the one at the call, also if its slot is written
    /// again right after.
    #[test]
    fn a_target_is_the_one_at_the_call() {
        let mut slot = [0x5256_5476u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let kept = record(slot.as_ptr() as u64);
        slot[1] = 99;
        std::hint::black_box(&slot);
        assert_eq!(kept[1], 2);
        assert_eq!(record(0), [0; RECORD]);
    }
}
