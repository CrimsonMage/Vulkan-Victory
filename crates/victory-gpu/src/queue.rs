//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given after a wait for a fence that had not been
    /// reached, in order; it runs when the fence gets there.
    #[cfg(windows)]
    held: std::sync::Mutex<Vec<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence (its address) to reach the value.
    Wait(usize, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), and the order they run in.
    Run(Vec<crate::list::Detached>, Vec<usize>),
    /// A signal of the queue's, due after what it holds before it.
    Signal(usize, u64),
}

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A submission to a queue. Its lists are cut into host lists at their
/// waits for memory values (`crate::list::cut`), and the queue runs it part
/// by part: it stands at each wait until the value is in memory, with
/// everything recorded before the wait run and its bottom-of-pipe writes
/// done, as on the console. (The title's lists are chains of jobs, each
/// behind a wait for a value another queue's list writes in the middle of
/// its own chain.) The lists before the first one that waits run at once if
/// the queue holds nothing; the rest goes behind what the queue holds, part
/// by part (`crate::list::detach_parts`): the waits of a part, then its host
/// lists. [`release_held`] runs the queue's line as far as the values are
/// there, now and whenever a submission has run.
#[cfg(windows)]
fn submit_cut(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let first = if lock(&queue(this).held).is_empty() {
        lists
            .iter()
            .position(|&l| crate::list::cut_list(l))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {first} run at once, the rest part by part)",
            Some(queue(this).kind),
            first < lists.len()
        )
    });
    if first > 0 {
        // Sent to the card and not waited for: their host sides are taken
        // from the title's lists (which it may reset before they have run)
        // and come back, with their tables' slots, when they have landed.
        let (taken, order) = crate::list::detach(&lists[..first]);
        host.execute_late(taken, order);
    }
    if first < lists.len() {
        let parts = crate::list::detach_parts(&lists[first..]);
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = first,
                parts = parts.len(),
                waits = parts.iter().map(|p| p.0.len()).sum::<usize>(),
                "a submission is given to its queue part by part (its lists wait for memory values)"
            );
        }
        {
            let mut held = lock(&queue(this).held);
            for (waits, taken, order) in parts {
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Run(taken, order));
            }
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just
    // given to the queue may be free to run already.
    release_held(false);
}

/// Adds `what()` to what the queue holds, if it holds anything (it waits
/// for a fence): whether it was added.
#[cfg(windows)]
fn hold(this: This, what: impl FnOnce() -> Held) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        return false;
    }
    held.push(what());
    true
}

/// Runs what waiting queues hold as far as their fences have been reached
/// (`all`: all of it), each queue's in the order it was given. Called
/// where a fence may have moved: after a signal and after a submission (a
/// list's bottom-of-pipe write may be into a fence), never from inside
/// one.
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    // One caller at a time, so a queue's submissions keep their order.
    static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let _one = lock(&RELEASING);
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        for q in queues {
            // SAFETY: a live queue: one released is taken off the list.
            let line = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(line);
            loop {
                match held.first() {
                    None => break,
                    Some(&Held::Wait(fence, value)) if !all && !fence_met(fence, value) => break,
                    Some(Held::Value(wait)) if !all && !value_met(wait) => break,
                    Some(_) => {}
                }
                match held.remove(0) {
                    Held::Wait(..) => {}
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if all && !value_met(&wait) {
                            static SEEN: AtomicU64 = AtomicU64::new(0);
                            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                                tracing::warn!(
                                    queue = format_args!("{q:#x}"),
                                    ?wait,
                                    "a wait for a memory value was not met at a present: the queue runs on"
                                );
                            }
                        }
                    }
                    Held::Run(lists, order) => {
                        #[cfg(feature = "diagnostics")]
                        crate::capture::op(q, || {
                            format!(
                                "RUN held submission of {} lists (forced {all})",
                                order.len()
                            )
                        });
                        // Not waited for: the host keeps the lists until
                        // they have landed.
                        if let Some(host) = crate::host::host() {
                            host.execute_late(lists, order);
                        }
                        if all {
                            static ONCE: AtomicBool = AtomicBool::new(false);
                            if !ONCE.swap(true, Ordering::Relaxed) {
                                tracing::warn!(
                                    "a queue still held a submission at a present: its fence was not reached"
                                );
                            }
                        }
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if !crate::host::late_signal(fence, value) {
                            crate::fence::set(fence as This, value);
                        }
                    }
                }
                moved = true;
            }
            if held.is_empty() {
                lock(&WAITING).retain(|&w| w != q);
            }
        }
        if !moved {
            break;
        }
    }
}

/// Whether a queue may pass a wait for `fence` to reach `value`: it has,
/// or a signal that gets it there is in the line of submissions sent to
/// the card (`crate::host::late_signalled`). Then what follows goes to the
/// card behind that signal's submissions, on the one host queue, which runs
/// them in order: the card waits, not the title's work on the CPU.
#[cfg(windows)]
fn fence_met(fence: usize, value: u64) -> bool {
    crate::fence::completed(fence as This) >= value || crate::host::late_signalled(fence, value)
}

/// Whether a queue may pass `wait`: memory holds the value, or it will
/// once the submissions in the air have run.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    wait.met() || crate::host::late_met(wait)
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    let left = object::release_boxed::<Queue>(this);
    #[cfg(windows)]
    if left == 0 {
        lock(&WAITING).retain(|&q| q != this as usize);
    }
    left
}

/// Runs the lists' host lists and waits for them (decision 0007). A queue
/// that waits for a fence keeps them until the fence is reached.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if crate::vulkan::on() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        crate::vulkan::execute(this, lists);
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        // What the title's threads wrote down into them is translated first.
        crate::list::join(lists);
        if !this.is_null() {
            submit_cut(this, lists);
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        crate::list::tables_done(lists);
        // What the lists wrote at the bottom of the pipe may be a fence
        // another queue waits for.
        release_held(false);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// A present plane, 72 bytes, as the title fills them (title+0x248232d,
/// presented at title+0x24824c1: up to two, 0x48 apart): frame token, the
/// plane's resources, per resource the rectangle of it shown (left, top,
/// right, bottom texels), where the plane goes on the screen (left, top,
/// right, bottom as fractions of it; the title passes 0, 0, 1, 1), then a
/// word from the title's plane record and words it leaves zero.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
    source: *const [i32; 4],
    placement: *const [f32; 4],
    color_space: u32,
    rest: [u32; 7],
}
const _: () = assert!(size_of::<Plane>() == 0x48);
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::vulkan::on() {
        crate::vulkan::signal(this, fence, value);
        return S_OK;
    }
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(fence as usize, value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late_signal(fence as usize, value) {
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`: how the console orders work between queues (the title's
/// compute queue waits for the depth the graphics queue draws). Our
/// submissions run when they are made, so a wait for a value already
/// reached is nothing; otherwise what the queue is given next is held
/// ([`Held`]) until a signal or a list's bottom-of-pipe write gets the
/// fence there.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::vulkan::on() {
        crate::vulkan::wait(this, fence, value);
        return S_OK;
    }
    #[cfg(windows)]
    if !this.is_null() {
        let now = crate::fence::completed(fence);
        // Whether the title uses this at all (the first calls).
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(kind = queue(this).kind, now, value, "a queue is told to wait for a fence");
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || format!("WAIT queue kind {} for fence {fence:?} value {value} (now {now})", queue(this).kind));
        let behind = {
            let mut held = lock(&queue(this).held);
            let behind = !held.is_empty() || !fence_met(fence as usize, value);
            if behind {
                held.push(Held::Wait(fence as usize, value));
            }
            behind
        };
        if behind {
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    now,
                    value,
                    "a queue waits for a fence not reached yet"
                );
            }
        }
        if behind {
            {
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
            // The fence may have got there meanwhile.
            release_held(false);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the title's planes in our window: each plane's first resource,
/// its rectangle scaled to the plane's place on the screen, later planes
/// over earlier ones (`Host::present`). With one plane the console's
/// display shows the title's final image; with two, its scaler puts the
/// first (the scene, in a smaller size) under the second (the interface
/// at the display's size), as the title does on Series S.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    let _ = this;
    // The present is sent ahead too: what waits for a value a flight will
    // write runs behind that flight, and nothing is waited for here. Then,
    // shortcut: what a queue still holds at a present runs now, whatever
    // its fence: a wait is not kept across frames.
    #[cfg(windows)]
    {
        release_held(false);
        release_held(true);
    }
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(windows)]
    if crate::vulkan::on() {
        let given: &[Plane] = if plane_parameters.is_null() {
            &[]
        } else {
            // SAFETY: the title passes `planes` plane descriptions.
            unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(4) as usize) }
        };
        let size = given
            .iter()
            .filter(|p| p.count > 0 && !p.resources.is_null())
            // SAFETY: the plane lists `count` resources.
            .filter_map(|p| crate::resource::texture_memory(unsafe { *p.resources }))
            .map(|(_, desc)| desc.extent())
            .max_by_key(|&(w, h)| u64::from(w) * u64::from(h));
        let shown: Vec<crate::vulkan::Shown> = given
            .iter()
            .filter(|p| p.count > 0 && !p.resources.is_null())
            .map(|p| crate::vulkan::Shown {
                // SAFETY: the plane lists `count` resources; its rectangles
                // are given or null.
                resource: unsafe { *p.resources },
                source: unsafe { p.source.as_ref() }.copied(),
                placement: unsafe { p.placement.as_ref() }.copied(),
            })
            .collect();
        crate::vulkan::present(frame, size, &shown);
        return S_OK;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // A state the host texture is not left in: what it was in comes
        // back, and goes back after.
        const COPY_SOURCE: u32 = 0x800;
        let given: &[Plane] = if plane_parameters.is_null() {
            &[]
        } else {
            // SAFETY: the title passes `planes` plane descriptions.
            unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(4) as usize) }
        };
        log_planes(frame, given);
        let mut shown: Vec<(This, crate::host::Shown)> = Vec::new();
        for plane in given {
            if plane.count == 0 || plane.resources.is_null() {
                continue;
            }
            // SAFETY: the plane lists `count` resources.
            let resource = unsafe { *plane.resources };
            if shown.iter().any(|(r, _)| *r == resource) {
                continue;
            }
            let Some(texture) = crate::resource::host_resource(resource) else {
                continue;
            };
            let state = crate::resource::host_transition(resource, COPY_SOURCE)
                .map_or(COPY_SOURCE, |(_, before)| before);
            // SAFETY: given, a rectangle and a placement per resource.
            let (source, placement) = unsafe { (plane.source.as_ref(), plane.placement.as_ref()) };
            let plane = crate::host::Shown {
                texture: texture.clone(),
                state,
                source: source.copied(),
                placement: placement.copied(),
            };
            shown.push((resource, plane));
        }
        let planes: Vec<_> = shown.iter().map(|(_, plane)| plane).collect();
        host.present(&planes, frame);
        for (resource, plane) in &shown {
            crate::resource::host_transition(*resource, plane.state);
        }
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Logs the planes the title presents, at the first present and whenever
/// their number, sizes or placement change (a few times at most).
#[cfg(windows)]
fn log_planes(frame: u64, given: &[Plane]) {
    use std::fmt::Write;
    let mut text = String::new();
    for (i, plane) in given.iter().enumerate() {
        let _ = write!(text, "plane {i}: {} resources", plane.count);
        if plane.count > 0 && !plane.resources.is_null() {
            // SAFETY: the plane lists `count` resources.
            let resource = unsafe { *plane.resources };
            if let Some(texture) = crate::resource::host_resource(resource) {
                // SAFETY: a live resource.
                let desc = unsafe { texture.GetDesc() };
                let _ = write!(text, ", the first {}x{} format {}", desc.Width, desc.Height, desc.Format.0);
            }
        }
        // SAFETY: given, a rectangle and a placement per resource.
        let (source, placement) = unsafe { (plane.source.as_ref(), plane.placement.as_ref()) };
        let _ = write!(
            text,
            ", shown {source:?} at {placement:?}, word {:#x}, then {:x?}; ",
            plane.color_space, plane.rest
        );
    }
    static LAST: std::sync::Mutex<(String, u32)> = std::sync::Mutex::new((String::new(), 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.0 != text && last.1 < 16 {
        last.1 += 1;
        tracing::info!(frame, planes = given.len(), "the title presents {text}");
        last.0 = text;
    }
}
/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. The title streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        // Ranges that continue each other (pool pages that follow on, or
        // unmapped pages) are one request, `(destination, source or 0,
        // bytes)`: the runtime maps a run of consecutive pages as one view.
        let mut pending: Option<(u64, u64, u64)> = None;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            let next = match r.kind {
                0 if inside(r.count) => Some(source),
                2 => Some(0),
                _ => None,
            };
            if let (Some((to, from, bytes)), Some(next)) = (pending.as_mut(), next)
                && *to + *bytes == at
                && (if next == 0 { *from == 0 } else { *from != 0 && *from + *bytes == next })
            {
                *bytes += len;
                at += len;
                continue;
            }
            if let Some((to, from, bytes)) = pending.take() {
                ok &= map(to, from, bytes, false);
            }
            match next {
                Some(from) => pending = Some((at, from, len)),
                None => {
                    ok &= match r.kind {
                        1 if inside(1) => map(at, source, len, true),
                        3 => true,
                        _ => false,
                    };
                }
            }
            at += len;
        }
        if let Some((to, from, bytes)) = pending {
            ok &= map(to, from, bytes, false);
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from the title's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
