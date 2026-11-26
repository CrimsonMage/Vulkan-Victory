//! The console's DirectStorage (`XCRDAPI.dll`): a factory that opens files
//! and creates request queues and status arrays the title polls. Its
//! interfaces have grown since 2020 (new interface ids, new slots); the
//! slots here are the ones the title calls, read from its call
//! sites. Slots nobody has needed yet log their offset.
//!
//! A file the title opens here (its `.rpf` and `.ttpf` archives) is one
//! handle the workers share; each request reads exactly its source range
//! with one positioned read, when it runs, and nothing of the file is
//! mapped or kept. An uncompressed request is read straight into its
//! destination (an address, or physical pages); a compressed one is read
//! into the worker's reusable buffer, then zlib data is inflated and/or
//! BCPack data unpacked from there into the destination.
//!
//! Submitted requests of every queue go, in submission order, into one
//! line taken by a few workers ([`workers`]), which run them in parallel
//! and sleep on a condition variable while the line is empty. Completion
//! is what stays in order: a status entry is written once every request
//! enqueued before it on its queue has finished, with the first failure
//! (in queue order) since the previous status entry, or success. A status
//! entry closes a segment of its queue's requests; the segment counts what
//! it still waits for (its requests, the segment before it, its own
//! submission) and the last of those writes the status and lets the next
//! segment go on. Per request, only the line's lock is taken (to put it in
//! and to take it out).
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_void};
use std::os::windows::fs::FileExt;
use std::os::windows::io::FromRawHandle;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Once, OnceLock, PoisonError};
use std::thread::LocalKey;
use std::time::{Duration, Instant};

use victory_formats::bcpack::{self, Tables};
use windows_sys::Win32::Foundation::E_NOINTERFACE;

use super::xasync::{E_INVALIDARG, E_NOTIMPL, E_PENDING};
use super::xgameruntime::{Guid, HRESULT, Methods, Object, S_OK, add_ref, missing, put};
use crate::{HleModule, hle_export};

/// The factory interface the title asks for.
const FACTORY: Guid = Guid::new(
    0x6295_9d11,
    0xd45f,
    0x4884,
    [0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae],
);
/// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the source ends early.
const END_OF_FILE: HRESULT = 0x8007_0026_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: a bad compressed stream.
const INVALID_DATA: HRESULT = 0x8007_000D_u32 as i32;
/// `HRESULT_FROM_WIN32` of `ERROR_INVALID_USER_BUFFER` and
/// `ERROR_NOACCESS`: the host refused a read into this destination.
const REFUSED: [HRESULT; 2] = [0x8007_06F8_u32 as i32, 0x8007_03E6_u32 as i32];

type Out = *mut *mut c_void;

static FACTORY_OBJECT: Object = Object::new("dstorage factory", &FACTORY_VTABLE);

pub extern "system" fn DStorageGetFactory(iid: *const Guid, out: Out) -> HRESULT {
    // SAFETY: the title passes a GUID (null tolerated).
    match unsafe { iid.as_ref() } {
        Some(&iid) if iid == FACTORY => {
            put(out, std::ptr::from_ref(&FACTORY_OBJECT).cast_mut().cast())
        }
        iid => {
            tracing::warn!(iid = ?iid.map(Guid::to_string), "DStorageGetFactory: unknown interface");
            E_NOINTERFACE
        }
    }
}

/// One of our objects with state: the runtime's object header (method
/// table first), a reference count, then `T`.
#[repr(C)]
struct Instance<T> {
    base: Object,
    refs: AtomicU32,
    state: T,
}

impl<T> Instance<T> {
    fn create(base: Object, state: T) -> *mut c_void {
        Box::into_raw(Box::new(Instance {
            base,
            refs: AtomicU32::new(1),
            state,
        }))
        .cast()
    }
}

extern "system" fn add_ref_instance<T>(this: *const Instance<T>) -> u32 {
    // SAFETY: an instance we created and the title still holds.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

extern "system" fn release<T>(this: *mut Instance<T>) -> u32 {
    // SAFETY: as above.
    let left = unsafe { &*this }.refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference; created by `Instance::create`.
        drop(unsafe { Box::from_raw(this) });
    }
    left
}

fn iid_text(iid: *const Guid) -> String {
    // SAFETY: the title passes a GUID (null tolerated).
    unsafe { iid.as_ref() }.map_or_else(|| "null".into(), Guid::to_string)
}

/// `HRESULT_FROM_WIN32` of a host error.
fn from_win32(error: &std::io::Error) -> HRESULT {
    error
        .raw_os_error()
        .map_or(E_INVALIDARG, |e| (0x8007_0000 | (e as u32 & 0xffff)) as i32)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// The factory.

type Factory = *const Object;

/// Opens `path` for reading; the file object owns the handle.
extern "system" fn open_file(_: Factory, path: *const u16, iid: *const Guid, out: Out) -> HRESULT {
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 1;
    const OPEN_EXISTING: u32 = 3;
    if path.is_null() {
        return E_INVALIDARG;
    }
    // Goes through `files`, so paths on our virtual drives work too.
    let handle = super::files::CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        std::ptr::null(),
        OPEN_EXISTING,
        0,
        std::ptr::null_mut(),
    );
    let name = super::xgameruntime::wide_text(path).unwrap_or_default();
    if handle as isize == -1 {
        let error = std::io::Error::last_os_error();
        tracing::warn!(path = name, %error, "dstorage: opening a file failed");
        return from_win32(&error);
    }
    tracing::info!(path = name, iid = iid_text(iid), "dstorage: file opened");
    // SAFETY: a handle we just opened and nobody else owns.
    let file = unsafe { std::fs::File::from_raw_handle(handle) };
    let file = Instance::create(Object::new("dstorage file", &FILE_VTABLE), File { file });
    put(out, file)
}

/// `(capacity, name, iid, out)`.
extern "system" fn create_status_array(
    _: Factory,
    capacity: u32,
    name: *const c_char,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    tracing::info!(
        capacity,
        name = text(name),
        iid = iid_text(iid),
        "dstorage: status array"
    );
    let entries = (0..capacity).map(|_| AtomicI32::new(S_OK)).collect();
    let array = Instance::create(
        Object::new("dstorage status array", &STATUS_VTABLE),
        StatusArray { entries },
    );
    put(out, array)
}

/// The queue's description: as in 2020 (source type, capacity, priority,
/// name), plus a field the title sets to 1.
#[repr(C)]
struct QueueDesc {
    source_type: u64,
    capacity: u16,
    priority: i8,
    name: *const c_char,
    unknown: u64,
}

/// Every queue's requests run in submission order, whatever its priority.
// shortcut: the title makes all its queues (17) at one priority (real
// time), so order across queues is all a priority could change; serve
// higher priorities first if a title mixes them.
extern "system" fn create_queue(
    _: Factory,
    desc: *const QueueDesc,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    // SAFETY: the title passes its description (null tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let name = text(desc.name);
    tracing::info!(
        source = desc.source_type,
        capacity = desc.capacity,
        priority = desc.priority,
        name,
        unknown = desc.unknown,
        iid = iid_text(iid),
        "dstorage: queue"
    );
    let lane = Arc::new(Lane::new(desc.capacity));
    put(
        out,
        Instance::create(Object::new("dstorage queue", &QUEUE_VTABLE), Queue { lane }),
    )
}

/// The title asks for 32 MiB and stops if this fails. Nothing is staged
/// here: sources are slices of the mapped files or of title memory.
extern "system" fn set_staging_buffer_size(_: Factory, size: u32) -> HRESULT {
    tracing::info!(size, "dstorage: staging buffer size");
    S_OK
}

fn text(name: *const c_char) -> String {
    if name.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated name from the title.
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

static FACTORY_VTABLE: Methods<16> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),            // 0x18 (2020: queue creation)
    open_file as *const (),               // 0x20
    create_status_array as *const (),     // 0x28
    missing::<6> as *const (),            // 0x30 (2020: CPU affinity)
    missing::<7> as *const (),            // 0x38 (2020: debug flags)
    set_staging_buffer_size as *const (), // 0x40
    missing::<9> as *const (),
    create_queue as *const (), // 0x50
    missing::<11> as *const (),
    missing::<12> as *const (),
    missing::<13> as *const (),
    missing::<14> as *const (),
    missing::<15> as *const (),
]);

// Files.

/// An opened file: one handle, which every worker reads at its own
/// offsets.
struct File {
    file: std::fs::File,
}

static FILE_VTABLE: Methods<8> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<File> as *const (),
    release::<File> as *const (),
    missing::<3> as *const (),
    missing::<4> as *const (),
    missing::<5> as *const (),
    missing::<6> as *const (),
    missing::<7> as *const (),
]);

// Status arrays.

struct StatusArray {
    entries: Box<[AtomicI32]>,
}

type Array = *mut Instance<StatusArray>;

/// The result of the requests before entry `index`: pending, the first
/// failure, or success.
extern "system" fn status_result(this: Array, index: u32) -> HRESULT {
    // SAFETY: an array we created and the title still holds.
    let entries = &unsafe { &*this }.state.entries;
    entries
        .get(index as usize)
        .map_or(E_INVALIDARG, |e| e.load(Ordering::Acquire))
}

/// The title polls this (entry 0, a `Sleep(1)` loop) and never
/// reads the result itself.
extern "system" fn is_complete(this: Array, index: u32) -> bool {
    status_result(this, index) != E_PENDING
}

static STATUS_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<StatusArray> as *const (),
    release::<StatusArray> as *const (),
    is_complete as *const (),   // 0x18
    status_result as *const (), // 0x20
]);

// Queues.

/// A queue object: its lane, which its requests on their way also hold.
struct Queue {
    lane: Arc<Lane>,
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.lane.close();
    }
}

type QueueRef = *const Instance<Queue>;

fn lane<'a>(this: QueueRef) -> &'a Arc<Lane> {
    // SAFETY: a queue we created and the title still holds while it calls.
    &unsafe { &*this }.state.lane
}

/// A queue's state. Only the title's threads take `open` (to enqueue,
/// submit and close); workers touch atomics alone.
struct Lane {
    capacity: usize,
    open: Mutex<Open>,
    /// Requests and status entries held: enqueued, submitted, running.
    held: AtomicUsize,
    /// Requests running now.
    running: AtomicUsize,
    closed: AtomicBool,
}

/// What the title has enqueued and not yet submitted, and the segment its
/// next request joins.
struct Open {
    commands: Vec<Command>,
    segment: Arc<Segment>,
    /// Requests in `segment` so far: the next one's place.
    index: u32,
}

enum Command {
    Read(Job),
    /// A status entry: the segment it closes, let go at submission.
    Status(Arc<Segment>),
}

/// The requests between two status entries of a queue.
struct Segment {
    /// What it waits for: its requests not yet finished, plus one while
    /// the segment before it has not completed, plus one until its status
    /// entry is submitted.
    pending: AtomicUsize,
    /// The first failure in queue order: place << 32 | HRESULT.
    failure: AtomicU64,
    /// Set when the status entry closes it, with `next` (before its
    /// submission can complete it).
    status: OnceLock<Status>,
    next: OnceLock<Arc<Segment>>,
}

/// A status entry to write, holding a reference on its array.
struct Status {
    array: Array,
    index: u32,
}

// SAFETY: the array's reference count and entries are atomic, and it stays
// alive while this holds its reference.
unsafe impl Send for Status {}
// SAFETY: as above.
unsafe impl Sync for Status {}

impl Drop for Status {
    fn drop(&mut self) {
        release(self.array);
    }
}

impl Status {
    fn entry(&self) -> &AtomicI32 {
        // SAFETY: we hold a reference; the index was checked when enqueued.
        &unsafe { &*self.array }.state.entries[self.index as usize]
    }
}

/// A request on its way to a worker.
struct Job {
    request: Request,
    lane: Arc<Lane>,
    segment: Arc<Segment>,
    /// Its place in the segment.
    index: u32,
}

// SAFETY: requests point into title memory and our own objects, which stay
// valid until the title has seen them complete.
unsafe impl Send for Job {}

impl Segment {
    fn new(after_another: bool) -> Arc<Segment> {
        Arc::new(Segment {
            pending: AtomicUsize::new(1 + usize::from(after_another)),
            failure: AtomicU64::new(u64::MAX),
            status: OnceLock::new(),
            next: OnceLock::new(),
        })
    }

    /// One thing it waited for is done; at the last, its status is written
    /// and the next segment no longer waits for it. The caller holds this
    /// segment, and each segment holds the next.
    fn release(&self, lane: &Lane) {
        let mut segment = self;
        while segment.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Complete, so closed: `status` and `next` are set.
            let (Some(status), Some(next)) = (segment.status.get(), segment.next.get()) else {
                return;
            };
            let failure = segment.failure.load(Ordering::Relaxed);
            let result = if failure == u64::MAX { S_OK } else { failure as u32 as HRESULT };
            status.entry().store(result, Ordering::Release);
            lane.held.fetch_sub(1, Ordering::SeqCst);
            segment = next;
        }
    }
}

impl Lane {
    fn new(capacity: u16) -> Lane {
        Lane {
            capacity: usize::from(capacity),
            open: Mutex::new(Open {
                commands: Vec::new(),
                segment: Segment::new(false),
                index: 0,
            }),
            held: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        }
    }

    /// Takes a slot for one more command, waiting while the queue is full
    /// (one slot stays free, as the console's count of empty slots says);
    /// none once it is closed. It waits as the title waits for its status
    /// entries, a millisecond at a time.
    fn slot(&self) -> Option<MutexGuard<'_, Open>> {
        loop {
            let open = lock(&self.open);
            if self.closed.load(Ordering::SeqCst) {
                return None;
            }
            if self.held.load(Ordering::SeqCst) + 1 < self.capacity.max(2) {
                self.held.fetch_add(1, Ordering::SeqCst);
                return Some(open);
            }
            drop(open);
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A request joining the open segment.
    fn read(self: &Arc<Self>, open: &mut Open, request: Request) -> Job {
        open.segment.pending.fetch_add(1, Ordering::Relaxed);
        let index = open.index;
        open.index = open.index.wrapping_add(1);
        Job {
            request,
            lane: Arc::clone(self),
            segment: Arc::clone(&open.segment),
            index,
        }
    }

    /// Closes the open segment with a status entry and opens the next.
    fn close_segment(open: &mut Open, status: Status) -> Arc<Segment> {
        let next = Segment::new(true);
        let closed = std::mem::replace(&mut open.segment, Arc::clone(&next));
        // Both are unset: the segment was open.
        let _ = closed.status.set(status);
        let _ = closed.next.set(next);
        open.index = 0;
        closed
    }

    /// Adds a command; submits on its own once half the queue is
    /// enqueued, as the console does.
    fn push(&self, open: &mut Open, command: Command) {
        open.commands.push(command);
        if open.commands.len() >= (self.capacity / 2).max(1) {
            self.submit(open);
        }
    }

    /// Hands the enqueued requests to the workers, in order.
    fn submit(&self, open: &mut Open) {
        start_workers();
        for command in open.commands.drain(..) {
            match command {
                Command::Read(job) => {
                    let depth = STATS.depth.fetch_add(1, Ordering::Relaxed) + 1;
                    STATS.peak.fetch_max(depth, Ordering::Relaxed);
                    lock(&LINE.0).push_back(job);
                    LINE.1.notify_one();
                }
                Command::Status(segment) => segment.release(self),
            }
        }
    }

    /// Ends the queue: what was not submitted is dropped, what was and has
    /// not started never runs (a worker drops it), and this returns once
    /// the requests already running have finished (they write title
    /// memory). Their status entries are never written.
    fn close(&self) {
        {
            let mut open = lock(&self.open);
            self.closed.store(true, Ordering::SeqCst);
            open.commands.clear(); // releases their arrays
        }
        while self.running.load(Ordering::SeqCst) != 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Job {
    /// Runs the request unless its queue was closed.
    fn run(self) {
        let lane = &self.lane;
        // Counted before `closed` is read, as `close` sets `closed` before
        // it reads the count: either the close waits for this request, or
        // this sees the close and drops the request.
        lane.running.fetch_add(1, Ordering::SeqCst);
        if !lane.closed.load(Ordering::SeqCst) {
            let result = execute(&self.request);
            STATS.requests.fetch_add(1, Ordering::Relaxed);
            self.finish(result);
        }
        STATS.depth.fetch_sub(1, Ordering::Relaxed);
        lane.running.fetch_sub(1, Ordering::SeqCst);
    }

    /// Counts the request done in its segment.
    fn finish(&self, result: Result<(), HRESULT>) {
        if let Err(hr) = result {
            let failure = (u64::from(self.index) << 32) | u64::from(hr as u32);
            self.segment.failure.fetch_min(failure, Ordering::Relaxed);
        }
        self.segment.release(&self.lane);
        self.lane.held.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Workers of [`LINE`]: half the cores. A request is one positioned read
/// (waiting on the disk) then an inflate and/or BCPack unpack; loading
/// bursts take ~25 s of worker time per 10 s (run p31), and four workers
/// doubled the load (run p33). Idle, they sleep on the condition variable,
/// so more of them cost nothing.
fn workers() -> usize {
    std::thread::available_parallelism().map_or(8, |n| (n.get() / 2).clamp(4, 16))
}

/// The submitted requests of every queue, in submission order, and the
/// condition variable signalled once per request put in.
static LINE: (Mutex<VecDeque<Job>>, Condvar) = (Mutex::new(VecDeque::new()), Condvar::new());

/// Starts the workers and the streaming line, once.
fn start_workers() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let spawn = |name: String, f: fn()| std::thread::Builder::new().name(name).spawn(f).is_ok();
        let started = (0..workers()).filter(|i| spawn(format!("dstorage {i}"), work)).count();
        spawn("dstorage stats".into(), report);
        tracing::info!(workers = started, "dstorage: workers");
    });
}

/// A worker: the line's requests in order, asleep while there are none.
fn work() {
    victory_core::heap::enter_for_thread(victory_core::heap::Owner::Streaming);
    let (line, ready) = &LINE;
    loop {
        let next = ready.wait_while(lock(line), |jobs| jobs.is_empty()).unwrap_or_else(PoisonError::into_inner).pop_front();
        if let Some(job) = next {
            job.run();
        }
    }
}

/// What the workers did since the last streaming line.
struct Stats {
    requests: AtomicU64,
    /// Bytes read from files, and of those read straight into
    /// destinations.
    read: AtomicU64,
    direct: AtomicU64,
    /// Destination bytes of compressed requests.
    decompressed: AtomicU64,
    /// Summed over workers.
    read_ns: AtomicU64,
    decompress_ns: AtomicU64,
    copy_ns: AtomicU64,
    /// Reads into a destination the host refused, read again beside.
    bounced: AtomicU64,
    /// Submitted, not yet finished; the most since the last line.
    depth: AtomicUsize,
    peak: AtomicUsize,
}

static STATS: Stats = Stats {
    requests: AtomicU64::new(0),
    read: AtomicU64::new(0),
    direct: AtomicU64::new(0),
    decompressed: AtomicU64::new(0),
    read_ns: AtomicU64::new(0),
    decompress_ns: AtomicU64::new(0),
    copy_ns: AtomicU64::new(0),
    bounced: AtomicU64::new(0),
    depth: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};

fn nanos_since(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// One line every 10 s while requests come.
fn report() {
    loop {
        std::thread::sleep(Duration::from_secs(10));
        let take = |n: &AtomicU64| n.swap(0, Ordering::Relaxed);
        let requests = take(&STATS.requests);
        let depth = STATS.depth.load(Ordering::Relaxed);
        let peak = STATS.peak.swap(depth, Ordering::Relaxed);
        if requests == 0 && peak == 0 {
            continue;
        }
        let mb = |n: &AtomicU64| take(n) as f64 / f64::from(1 << 20);
        let ms = |n: &AtomicU64| take(n) / 1_000_000;
        tracing::info!(
            "dstorage streaming (10 s): {requests} requests, {:.1} MB read ({:.1} MB straight into \
             destinations), {:.1} MB decompressed, {} ms reading, {} ms decompressing, {} ms copying, \
             {} reads refused and read beside, queue depth {depth} (peak {peak})",
            mb(&STATS.read),
            mb(&STATS.direct),
            mb(&STATS.decompressed),
            ms(&STATS.read_ns),
            ms(&STATS.decompress_ns),
            ms(&STATS.copy_ns),
            take(&STATS.bounced),
        );
    }
}

/// A read request as the title builds it, 0x48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Request {
    options: u64,
    /// An address, or with `PAGE_DESTINATION` an array of 64 KiB page
    /// numbers.
    destination: *mut c_void,
    /// With `PAGE_DESTINATION`, the low 16 bits are where in the first page
    /// the destination starts.
    page_offset: u64,
    destination_size: u32,
    /// Our file object, or with `MEMORY_SOURCE` an address.
    source: *const c_void,
    offset: u64,
    source_size: u32,
    /// Compressed size between two decompression stages: the BCPack stream
    /// a zlib stream inflates to.
    intermediate_size: u32,
    /// Matched by cancellation; the title leaves it 0.
    tag: u64,
    name: *const c_char,
}

// Request options.
/// Inflate zlib (RFC 1950) data.
const ZLIB: u64 = 1;
/// Bits 1-4: decode BCPack data of this BC format number (0: none).
const BCPACK: u64 = 0xf << 1;
const PAGE_DESTINATION: u64 = 1 << 21;
const MEMORY_SOURCE: u64 = 1 << 22;
// Bits 5-20 select a further transform, bit 23 a source in physical pages;
// The title's packages need neither, so requests with them fail.

const PAGE: usize = 64 * 1024;

extern "system" fn enqueue_request(this: QueueRef, request: *const Request) {
    // SAFETY: the title passes its request (null tolerated).
    let Some(&request) = (unsafe { request.as_ref() }) else {
        return;
    };
    let lane = lane(this);
    let Some(mut open) = lane.slot() else {
        return; // closed: dropped
    };
    let job = lane.read(&mut open, request);
    lane.push(&mut open, Command::Read(job));
}

extern "system" fn enqueue_status(this: QueueRef, array: Array, index: u32) {
    // SAFETY: the title passes an array we created (null tolerated).
    let Some(entries) = (unsafe { array.as_ref() }).map(|a| &a.state.entries) else {
        return;
    };
    let Some(entry) = entries.get(index as usize) else {
        tracing::warn!(index, "dstorage: status entry out of range");
        return;
    };
    entry.store(E_PENDING, Ordering::Release);
    let lane = lane(this);
    let Some(mut open) = lane.slot() else {
        return; // closed: never completes
    };
    add_ref_instance(array);
    let segment = Lane::close_segment(&mut open, Status { array, index });
    lane.push(&mut open, Command::Status(segment));
}

extern "system" fn submit(this: QueueRef) {
    let lane = lane(this);
    let mut open = lock(&lane.open);
    tracing::debug!(commands = open.commands.len(), "dstorage: submit");
    lane.submit(&mut open);
}

/// Ends the queue whatever its reference count: what was not started never
/// completes, and this returns once the reads already running have
/// finished. The title calls this, not `Release`.
extern "system" fn close(this: QueueRef) {
    lane(this).close();
}

/// Inflates and/or unpacks, or copies, one request into its destination.
fn execute(request: &Request) -> Result<(), HRESULT> {
    let options = request.options;
    let fail = |hr: HRESULT, why: &str| {
        tracing::warn!(
            options = format_args!("{options:#x}"),
            offset = request.offset,
            source_size = request.source_size,
            destination_size = request.destination_size,
            intermediate_size = request.intermediate_size,
            tag = request.tag,
            name = text(request.name),
            hr = format_args!("{:#010x}", hr as u32),
            "dstorage: request failed: {why}"
        );
        hr
    };
    if options & !(ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE) != 0 {
        return Err(fail(E_NOTIMPL, "unsupported options"));
    }
    if request.source.is_null() || request.destination.is_null() {
        return Err(fail(E_INVALIDARG, "no source or destination"));
    }
    let bc = ((options & BCPACK) >> 1) as u8;
    let memory = (options & MEMORY_SOURCE != 0).then(|| {
        // SAFETY: the title's source holds `source_size` bytes until
        // completion.
        unsafe { std::slice::from_raw_parts(request.source.cast::<u8>(), request.source_size as usize) }
    });
    if options & (ZLIB | BCPACK) == 0 {
        if request.source_size > request.destination_size {
            return Err(fail(E_INVALIDARG, "source larger than destination"));
        }
        let size = request.source_size as usize;
        let copy = |from: &[u8], out: &mut [u8]| {
            let start = Instant::now();
            out.copy_from_slice(from);
            STATS.copy_ns.fetch_add(nanos_since(start), Ordering::Relaxed);
            Ok(size)
        };
        return into_destination(request, size, |out| match memory {
            Some(source) => copy(source, out),
            // Straight into the destination. A kernel read into a page that
            // is not there for a moment (its view away while xmem maps it
            // again) is refused instead of faulting; that one is read
            // beside and copied in from user mode, where it faults like the
            // title's own writes (`xmem::on_fault` waits).
            None => match read_at(request, out) {
                Err(hr) if REFUSED.contains(&hr) => {
                    STATS.bounced.fetch_add(1, Ordering::Relaxed);
                    scratch(&SOURCE, size, |bytes| {
                        read_at(request, bytes)?;
                        copy(bytes, out)
                    })
                }
                read => read.map(|()| {
                    STATS.direct.fetch_add(size as u64, Ordering::Relaxed);
                    size
                }),
            },
        })
        .map_err(|hr| fail(hr, "reading into the destination"));
    }
    let tables = if bc != 0 { Some(bcpack_tables()?) } else { None };
    let size = request.destination_size as usize;
    let decode = |source: &[u8]| -> Result<(), HRESULT> {
        let start = Instant::now();
        let unpack = |stream: &[u8]| {
            into_destination(request, size, |out| match tables {
                Some(tables) => bcpack::decode(bc, stream, out, tables)
                    .map(|()| out.len())
                    .map_err(|e| fail(INVALID_DATA, &e.to_string())),
                None => inflate(stream, out).map_err(|hr| fail(hr, "zlib")),
            })
        };
        // The console's order: zlib first, then BCPack.
        if bc != 0 && options & ZLIB != 0 {
            scratch(&BETWEEN, request.intermediate_size as usize, |packed| {
                let n = inflate(source, packed).map_err(|hr| fail(hr, "zlib"))?;
                unpack(&packed[..n])
            })?;
        } else {
            unpack(source)?;
        }
        STATS.decompress_ns.fetch_add(nanos_since(start), Ordering::Relaxed);
        STATS.decompressed.fetch_add(size as u64, Ordering::Relaxed);
        Ok(())
    };
    match memory {
        Some(source) => decode(source)?,
        None => scratch(&SOURCE, request.source_size as usize, |bytes| {
            read_at(request, bytes).map_err(|hr| fail(hr, "reading the source"))?;
            decode(bytes)
        })?,
    }
    tracing::trace!(
        options = format_args!("{options:#x}"),
        offset = request.offset,
        size,
        "dstorage: request done"
    );
    Ok(())
}

thread_local! {
    /// A worker's compressed source, read from the file.
    static SOURCE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// A worker's BCPack stream between zlib and BCPack.
    static BETWEEN: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// A worker's destination bytes for physical pages that are not one
    /// window.
    static SPREAD: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` on `len` bytes of a worker's buffer, which is kept for the
/// next request up to 1 MiB (larger ones give their memory back). Requests
/// average 20 to 100 KiB; when there was a worker per core (28 here),
/// keeping up to 8 MiB in each of three buffers held up to 670 MB of the
/// PC's memory after a burst, never given back.
fn scratch<R>(buffer: &'static LocalKey<RefCell<Vec<u8>>>, len: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    const KEPT: usize = 1 << 20;
    buffer.with_borrow_mut(|bytes| {
        if bytes.len() < len {
            bytes.resize(len, 0);
        }
        let result = f(&mut bytes[..len]);
        if bytes.len() > KEPT {
            *bytes = Vec::new();
        }
        result
    })
}

/// Inflates a zlib stream into `out`, returning the bytes written. A stream
/// that ends short of `out` passes: the title's chunks fill it exactly, and
/// nothing says that is required.
fn inflate(source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
    thread_local! {
        static INFLATE: std::cell::RefCell<zlib_rs::Inflate> =
            std::cell::RefCell::new(zlib_rs::Inflate::new(true, 15));
    }
    INFLATE.with_borrow_mut(|state| {
        state.reset(true);
        // `Inflate::decompress` cannot distinguish a valid empty stream from
        // an insufficient buffer when `avail_out == 0`, so use a 1-byte scratch
        // destination and verify `written <= out_len` afterward.
        let out_len = out.len();
        let mut scratch = [0u8; 1];
        let dest = if out_len == 0 { &mut scratch[..] } else { out };
        let (mut in_pos, mut out_pos) = (0usize, 0usize);
        let status = loop {
            match state.decompress(
                &source[in_pos..],
                &mut dest[out_pos..],
                zlib_rs::InflateFlush::NoFlush,
            ) {
                Ok(zlib_rs::Status::StreamEnd) => break Ok(zlib_rs::Status::StreamEnd),
                Ok(zlib_rs::Status::Ok) => {
                    let next_in = state.total_in() as usize;
                    let next_out = state.total_out() as usize;
                    if next_in == in_pos && next_out == out_pos {
                        break Ok(zlib_rs::Status::BufError);
                    }
                    in_pos = next_in;
                    out_pos = next_out;
                }
                other => break other,
            }
        };
        let written = state.total_out() as usize;
        match status {
            Ok(zlib_rs::Status::StreamEnd) if written <= out_len => Ok(written),
            other => {
                tracing::debug!(?other, "dstorage: zlib stream");
                Err(INVALID_DATA)
            }
        }
    })
}

/// The BCPack format's constant tables, parsed once for the process. If
/// they do not parse every BCPack request fails: one warning, then debug.
fn bcpack_tables() -> Result<&'static Tables, HRESULT> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    static WARNED: AtomicBool = AtomicBool::new(false);
    match TABLES.get_or_init(load_bcpack_tables) {
        Ok(tables) => Ok(tables),
        Err(why) if !WARNED.swap(true, Ordering::Relaxed) => {
            tracing::warn!("dstorage: BCPack requests fail: {why} (more at debug)");
            Err(E_NOTIMPL)
        }
        Err(_) => {
            tracing::debug!("dstorage: BCPack request failed: no tables");
            Err(E_NOTIMPL)
        }
    }
}

/// The tables VIctory ships with.
#[cfg(not(test))]
fn load_bcpack_tables() -> Result<Tables, String> {
    Tables::builtin().map_err(|e| format!("the built-in BCPack tables: {e}"))
}

/// Tests decode streams made against tables of their own, from this folder.
#[cfg(test)]
static BCPACK_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

#[cfg(test)]
fn use_bcpack_tables(dir: std::path::PathBuf) {
    let _ = BCPACK_DIR.set(dir);
}

#[cfg(test)]
fn load_bcpack_tables() -> Result<Tables, String> {
    let dir = BCPACK_DIR.get().ok_or("no folder set for the test tables")?;
    let bytes = std::fs::read(dir.join("tables.bin")).map_err(|e| e.to_string())?;
    Tables::parse(&bytes).map_err(|e| e.to_string())
}

/// Fills `out` from the request's file at the request's offset: one
/// positioned read (more only if the host returns less).
fn read_at(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    // SAFETY: a file object we created; the title keeps it open.
    let file = &unsafe { &*request.source.cast::<Instance<File>>() }.state.file;
    let start = Instant::now();
    let mut done = 0;
    let result = loop {
        if done == out.len() {
            break Ok(());
        }
        match file.seek_read(&mut out[done..], request.offset + done as u64) {
            Ok(0) => break Err(END_OF_FILE),
            Ok(n) => done += n,
            Err(error) => break Err(from_win32(&error)),
        }
    };
    STATS.read_ns.fetch_add(nanos_since(start), Ordering::Relaxed);
    STATS.read.fetch_add(done as u64, Ordering::Relaxed);
    result
}

/// Runs `fill` on the destination's first `len` bytes; `fill` returns how
/// many it produced. GPU-visible destinations are host memory like the
/// rest (GPU address = CPU address), so an address is written in place;
/// physical pages are too when they are consecutive pages of one
/// allocation, and otherwise go through a buffer.
fn into_destination(
    request: &Request,
    len: usize,
    fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>,
) -> Result<(), HRESULT> {
    if request.options & PAGE_DESTINATION == 0 {
        let at = request.destination as usize;
        // The physical pages shown there now, if xmem shows any: none is
        // taken for a filler page of zeros while the bytes come, and one
        // that was stops being one first (its stand-ins elsewhere would
        // never show them).
        let _target = super::xmem::target_memory(at, len);
        // SAFETY: the title's destination holds `destination_size` bytes, at
        // least `len`, and nothing else writes it until completion.
        let out = unsafe { std::slice::from_raw_parts_mut(request.destination.cast::<u8>(), len) };
        let filled = fill(out).map(drop);
        // The bytes are there: the GPU side's copies of them follow.
        victory_gpu::title_memory_written(at, len);
        // Every other place that shows those pages reads the same bytes
        // (said once per place where not).
        #[cfg(feature = "diagnostics")]
        super::xmem::check_aliases(at, len);
        return filled;
    }
    let offset = (request.page_offset & 0xffff) as usize;
    // SAFETY: the title passes a page number per 64 KiB of destination,
    // which holds at least `len` bytes from `offset`.
    let pages = unsafe {
        std::slice::from_raw_parts(
            request.destination.cast::<usize>(),
            (offset + len).div_ceil(PAGE),
        )
    };
    // As for an address: no page under the request is a filler page.
    let _target = super::xmem::target_pages(pages, offset, len);
    // Consecutive pages of one allocation (the usual case): written in
    // place through one view.
    if let Some(mut window) = super::xmem::page_window(pages, offset, len) {
        let filled = fill(window.bytes()).map(drop);
        drop(window);
        // The title maps pages where the card reads them and has them
        // filled afterwards: the GPU side's copies of those addresses
        // learn of the bytes only from this (upstream c75fdba, fork PR
        // #571). After a decode that failed part way too: what it wrote
        // is there.
        super::xmem::pages_written(pages, offset, len);
        return filled;
    }
    scratch(&SPREAD, len, |data| {
        let n = fill(data)?;
        // A page the title has just made, or a view the host refuses for a
        // moment, comes right a little later: the write is tried again
        // before the request fails (12 tries, about 0.9 s in all). Runs
        // 175-178 lost 12 to 446 reads a launch here, all in one second of
        // the loading; the one build that tried again (run 165) lost none
        // of 13.
        const RETRIES: u32 = 12;
        let mut written = super::xmem::write_physical(pages, offset, &data[..n]);
        let mut tries = 0u32;
        while !written && tries < RETRIES {
            std::thread::sleep(Duration::from_millis((2u64 << tries.min(6)).min(128)));
            tries += 1;
            written = super::xmem::write_physical(pages, offset, &data[..n]);
        }
        if written {
            // As above, after the write that succeeded.
            super::xmem::pages_written(pages, offset, n);
            Ok(())
        } else {
            // Said aloud: the title never reads a request's result
            // (it only polls for completion), so a destination left unwritten
            // is a model or texture that silently holds whatever its pages
            // held before.
            tracing::warn!(
                offset,
                len,
                pages = pages.len(),
                known = super::xmem::pages_known(pages, offset, len),
                tries,
                "dstorage: a destination's physical pages could not be written (unknown page or a host failure); its request fails"
            );
            Err(E_INVALIDARG)
        }
    })
}

/// Slots by the 2020 order, which the argument shapes at
/// the title's call sites match (0x18, 0x20, 0x30 and 0x40 are called).
static QUEUE_VTABLE: Methods<12> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<Queue> as *const (),
    release::<Queue> as *const (),
    enqueue_request as *const (), // 0x18
    enqueue_status as *const (),  // 0x20
    missing::<5> as *const (),    // 0x28 (2020: fence signal)
    submit as *const (),          // 0x30
    missing::<7> as *const (),    // 0x38 (2020: cancel by tag)
    close as *const (),           // 0x40
    missing::<9> as *const (),    // 0x48 (2020: error event)
    missing::<10> as *const (),   // 0x50 (2020: error record)
    missing::<11> as *const (),   // 0x58 (2020: queue info)
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dstorage",
        dlls: &["xcrdapi.dll"],
        exports: vec![hle_export!(DStorageGetFactory)],
    }
}

#[cfg(test)]
mod tests;
