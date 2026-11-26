//! Queues and fences (`docs/vulkan-host.md`, "Queues and fences"): the one
//! held line of the Vulkan backend.
//!
//! Each title queue keeps a line of items: waits for fences, submitted
//! lists (cut at their waits for memory values into parts), signals. Each
//! title queue has a line of the host's own ([`victory_vk::timeline`]): a
//! timeline semaphore on the queue its kind runs on.
//!
//! - A **part** goes to its queue's line when its waits are met: memory
//!   holds the value, or a write in the air will write it. A write in the
//!   air on another line becomes a semaphore wait of the submission.
//! - A **fence wait** passes when the fence is reached, or a signal of it
//!   is in the air. A signal on another line becomes a semaphore wait of
//!   the queue's next submission.
//! - A **fence signal** rides on the queue's newest submission in the air
//!   and is set when that lands (at once if none is in the air).
//! - **Bottom-of-pipe writes** are done by the card at the end of their
//!   part, into the imported title memory (`writes.rs`). Writes the card
//!   cannot make (the pool's GPU view, CPU timestamps) are done by the CPU
//!   when the part lands.
//! - Memory-value waits nothing in the air meets are held on the CPU, as
//!   ever: Vulkan has no wait on a memory value.
//!
//! A landing thread follows every line: as a submission's value is
//! reached, its CPU writes are done and its fence signals are set, then
//! every queue's line runs on.

use std::collections::{HashMap, VecDeque};
use std::sync::MutexGuard;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use victory_vk::timeline::{HOST, Kind, Line, Recording};

use super::{Backend, Split, backend, images, lock, memory, record, split, title_memory_written};
use crate::list::ValueWait;
use crate::object::This;
use crate::queue::Written;

/// A bottom-of-pipe write: title address, what, bytes (4 or 8).
pub(super) type Write = (u64, Written, usize);

#[derive(Default)]
struct Part {
    waits: Vec<ValueWait>,
    writes: Vec<Write>,
    /// The part's recorded command buffers.
    cmds: Vec<Recording>,
}

enum Item {
    Wait(usize, u64),
    Run(VecDeque<Part>),
    Signal(usize, u64),
}

/// A submission in the air.
struct Flight {
    line: Line,
    value: u64,
    /// Every write it makes (what later waits may count on).
    writes: Vec<Write>,
    /// The writes the CPU makes when it lands.
    cpu: Vec<Write>,
    signals: Vec<(usize, u64)>,
    /// Diagnostics: when it was given, and what its command buffers record.
    given: Instant,
    trail: Vec<String>,
    /// Beside each note of `trail`: its descriptor table addresses.
    tables: Vec<Vec<u32>>,
}

/// A title queue: its line of the host's, its items, and the semaphore
/// waits its next submission makes.
struct TitleQueue {
    line: Line,
    items: VecDeque<Item>,
    waits: Vec<(Line, u64)>,
}

/// How a wait can be passed.
enum Reach {
    /// It is met now.
    Now,
    /// A submission in the air (line, value) meets it.
    Behind(Line, u64),
    /// Nothing given meets it.
    No,
}

#[derive(Default)]
pub(super) struct Engine {
    queues: HashMap<usize, TitleQueue>,
    flights: VecDeque<Flight>,
    /// What each title list recorded since its reset.
    lists: HashMap<usize, Vec<Part>>,
    /// Diagnostics: each line's last [`RECENT`] compute notes (dispatches,
    /// indirect executions, bounding passes), for a device loss.
    recent: HashMap<Line, VecDeque<String>>,
}

/// Compute notes kept per line.
const RECENT: usize = 8;

/// Waits passed at a present without being met.
pub(super) static FORCED: AtomicU64 = AtomicU64::new(0);

impl Engine {
    /// Title queue `q`, its line made at first use.
    fn queue(&mut self, b: &Backend, q: usize) -> &mut TitleQueue {
        self.queues.entry(q).or_insert_with(|| {
            // A null queue (the title passed none) runs as a direct one.
            let kind = match if q == 0 { 0 } else { crate::queue::kind(q as This) } {
                2 => Kind::Compute,
                3 => Kind::Copy,
                _ => Kind::Direct,
            };
            let line = b.timeline.line(&b.host, kind).unwrap_or_else(|e| {
                tracing::error!(error = %format!("{e:#}"), "Vulkan: no line of its own for a title queue; it runs on the host line");
                HOST
            });
            tracing::info!(queue = format_args!("{q:#x}"), ?kind, line, "Vulkan: a title queue runs on a line of its own");
            TitleQueue { line, items: VecDeque::new(), waits: Vec::new() }
        })
    }

    fn push(&mut self, b: &Backend, q: usize, item: Item) {
        self.queue(b, q).items.push_back(item);
    }

    /// Items the queues hold.
    fn held(&self) -> usize {
        self.queues.values().map(|q| q.items.len()).sum()
    }

    fn signalled(&self, fence: usize, value: u64) -> Reach {
        if crate::fence::completed(fence as This) >= value {
            return Reach::Now;
        }
        self.flights
            .iter()
            .find(|f| f.signals.iter().any(|&(x, v)| x == fence && v >= value))
            .map_or(Reach::No, |f| Reach::Behind(f.line, f.value))
    }

    fn met(&self, wait: &ValueWait) -> Reach {
        if wait.met() {
            return Reach::Now;
        }
        let (at, _) = wait.place();
        // The last write in the air at that address decides.
        let last = self.flights.iter().rev().find_map(|f| {
            f.writes
                .iter()
                .rev()
                .find(|(a, ..)| *a as usize == at)
                .map(|w| (f, w.1))
        });
        match last {
            Some((f, Written::Value(v))) if wait.holds_for(v) => Reach::Behind(f.line, f.value),
            _ => Reach::No,
        }
    }

    /// Passes a wait of queue `q` that reaches as `reach`: one behind a
    /// submission on another line becomes a wait of the queue's next
    /// submission. Whether it passed (`force`: always, counted).
    fn pass(&mut self, q: usize, reach: Reach, force: bool) -> bool {
        match reach {
            Reach::Now => {}
            Reach::Behind(line, value) => {
                let tq = self.queues.get_mut(&q).expect("queue");
                if tq.line != line {
                    tq.waits.push((line, value));
                }
            }
            Reach::No if force => {
                FORCED.fetch_add(1, Ordering::Relaxed);
            }
            Reach::No => return false,
        }
        true
    }

    /// Runs every queue's line as far as it can; `force` passes waits that
    /// are not met (at a present, as the D3D12 path does).
    pub(super) fn pump(&mut self, b: &Backend, force: bool) {
        loop {
            let keys: Vec<usize> = self.queues.keys().copied().collect();
            let mut moved = false;
            for q in keys {
                moved |= self.run_line(b, q, force);
            }
            if !moved {
                break;
            }
        }
    }

    /// Runs queue `q`'s line as far as it can; whether anything moved.
    fn run_line(&mut self, b: &Backend, q: usize, force: bool) -> bool {
        let mut moved = false;
        while let Some(front) = self.queues.get_mut(&q).and_then(|l| l.items.pop_front()) {
            if let Some(held) = self.run_item(b, q, front, force) {
                self.queues
                    .get_mut(&q)
                    .expect("queue")
                    .items
                    .push_front(held);
                break;
            }
            moved = true;
        }
        moved
    }

    /// Runs one item of a line; what is left of it if it must wait.
    fn run_item(&mut self, b: &Backend, q: usize, item: Item, force: bool) -> Option<Item> {
        match item {
            Item::Wait(fence, value) => {
                let reach = self.signalled(fence, value);
                if !self.pass(q, reach, force) {
                    return Some(Item::Wait(fence, value));
                }
            }
            Item::Signal(fence, value) => self.signal(b, q, fence, value),
            Item::Run(parts) => return self.run_parts(b, q, parts, force),
        }
        None
    }

    /// Submits parts while their waits can be passed; what is left.
    fn run_parts(
        &mut self,
        b: &Backend,
        q: usize,
        mut parts: VecDeque<Part>,
        force: bool,
    ) -> Option<Item> {
        while let Some(part) = parts.pop_front() {
            let reaches: Vec<Reach> = part.waits.iter().map(|w| self.met(w)).collect();
            if !force && reaches.iter().any(|r| matches!(r, Reach::No)) {
                parts.push_front(part);
                return Some(Item::Run(parts));
            }
            for r in reaches {
                self.pass(q, r, true);
            }
            // A part with neither commands nor writes has nothing to run;
            // its waits go to the next submission.
            if !part.writes.is_empty() || !part.cmds.is_empty() {
                self.submit(b, q, part);
            }
        }
        None
    }

    /// A fence signal of queue `q`: behind its newest submission in the air
    /// (after a marker that waits, if the queue passed waits since).
    fn signal(&mut self, b: &Backend, q: usize, fence: usize, value: u64) {
        if !self.queues[&q].waits.is_empty() {
            self.submit(b, q, Part::default());
        }
        let line = self.queues[&q].line;
        match self.flights.iter_mut().rev().find(|f| f.line == line) {
            Some(f) => f.signals.push((fence, value)),
            None => crate::fence::set(fence as This, value),
        }
    }

    /// Submits a part's commands, with the card's bottom-of-pipe writes at
    /// their end, on queue `q`'s line.
    fn submit(&mut self, b: &Backend, q: usize, part: Part) {
        let tq = self.queues.get_mut(&q).expect("queue");
        let (line, waits) = (tq.line, std::mem::take(&mut tq.waits));
        let mut cmds = part.cmds;
        if !cmds.is_empty() {
            images::flush(b);
        }
        let cpu = super::writes::record(b, &mut cmds, &part.writes);
        let (trail, tables) = take_trails(&mut cmds);
        match b.timeline.submit_on(&b.host, line, cmds, &waits) {
            Ok(value) => {
                self.remember(line, value, &trail);
                self.flights.push_back(Flight {
                line,
                value,
                writes: part.writes,
                cpu,
                signals: Vec::new(),
                given: Instant::now(),
                trail,
                tables,
                });
            }
            Err(e) => {
                if b.timeline.lost() {
                    self.report_lost(b);
                    lost_once();
                } else {
                    tracing::error!(error = %format!("{e:#}"), "Vulkan: a submission failed; its writes are done now");
                }
                for (at, what, bytes) in part.writes {
                    write_now(b, at, what, bytes);
                }
            }
        }
    }

    /// Diagnostics: keeps the compute notes of `line`'s submission `value`.
    fn remember(&mut self, line: Line, value: u64, trail: &[String]) {
        let compute = trail
            .iter()
            .filter(|n| n.starts_with("dispatch") || n.starts_with("indirect") || n.starts_with("bounding"));
        let recent = self.recent.entry(line).or_default();
        for n in compute {
            if recent.len() == RECENT {
                recent.pop_front();
            }
            recent.push_back(format!("v{value}: {n}"));
        }
    }

    /// Diagnostics: a loop gave up in a submission that just landed on one
    /// of `lines`: each such line's last 4 compute notes (at most 16
    /// reports in a run).
    fn report_guard_hit(&self, b: &Backend, lines: &[Line]) {
        static REPORTS: AtomicU64 = AtomicU64::new(0);
        if !cfg!(feature = "diagnostics") {
            return;
        }
        let mut seen: Vec<Line> = Vec::new();
        for &line in lines {
            if seen.contains(&line) || REPORTS.fetch_add(1, Ordering::Relaxed) >= 16 {
                continue;
            }
            seen.push(line);
            let Some(notes) = self.recent.get(&line) else { continue };
            let last: Vec<&str> = notes.iter().rev().take(4).rev().map(String::as_str).collect();
            tracing::warn!(
                line = %b.timeline.line_given(line).1,
                dispatches = %last.join(" | "),
                "Vulkan: a loop gave up in a submission of this line: its last compute work"
            );
        }
    }

    /// Diagnostics, once, when the device is found lost: every submission
    /// still in the air (the faulting one is among them) with the last 16
    /// notes of its trail, and each line's last compute notes.
    pub(super) fn report_lost(&self, b: &Backend) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        if !cfg!(feature = "diagnostics") {
            return;
        }
        ONCE.call_once(|| {
            for f in &self.flights {
                let (given, label) = b.timeline.line_given(f.line);
                let from = f.trail.len().saturating_sub(16);
                let tail = &f.trail[from..];
                // The last 4 draws or dispatches: the T#s their tables hold.
                for (note, tables) in f.trail[from..].iter().zip(&f.tables[from.min(f.tables.len())..]).rev().filter(|(_, t)| !t.is_empty()).take(4) {
                    tracing::error!(
                        line = %label,
                        value = f.value,
                        note = %note,
                        tsharps = %tsharps(b, tables),
                        "Vulkan: device lost: a draw's descriptor tables"
                    );
                }
                tracing::error!(
                    line = %label,
                    value = f.value,
                    given_last = given,
                    age_ms = f.given.elapsed().as_millis() as u64,
                    notes = f.trail.len(),
                    trail = %tail.join(" | "),
                    "Vulkan: device lost: a submission in the air"
                );
            }
            for (&line, notes) in &self.recent {
                let (given, label) = b.timeline.line_given(line);
                tracing::error!(
                    line = %label,
                    given_last = given,
                    dispatches = %notes.iter().cloned().collect::<Vec<_>>().join(" | "),
                    "Vulkan: device lost: the line's last compute work"
                );
            }
        });
    }

    /// The first submission in the air of each line (line, value).
    fn fronts(&self) -> Vec<(Line, u64)> {
        let mut out: Vec<(Line, u64)> = Vec::new();
        for f in &self.flights {
            if !out.iter().any(|&(l, _)| l == f.line) {
                out.push((f.line, f.value));
            }
        }
        out
    }

    /// Lands every submission that has run (all if the device was lost):
    /// its CPU writes, then its signals.
    fn land(&mut self, b: &Backend, lost: bool) {
        let reached: Vec<(Line, u64)> = self
            .fronts()
            .into_iter()
            .map(|(l, _)| {
                (
                    l,
                    if lost {
                        u64::MAX
                    } else {
                        b.timeline.line_completed(&b.host, l)
                    },
                )
            })
            .collect();
        let ran = |f: &Flight| reached.iter().any(|&(l, v)| l == f.line && f.value <= v);
        let (done, left): (VecDeque<Flight>, VecDeque<Flight>) =
            std::mem::take(&mut self.flights).into_iter().partition(ran);
        self.flights = left;
        let lines: Vec<Line> = done.iter().map(|f| f.line).collect();
        for f in done {
            for (at, what, bytes) in f.cpu {
                write_now(b, at, what, bytes);
            }
            for (fence, v) in f.signals {
                crate::fence::set(fence as This, v);
            }
        }
        if super::guard::check(b) {
            self.report_guard_hit(b, &lines);
            super::probe::describe_last(b);
        }
        super::probe::check(b);
    }
}

/// A bottom-of-pipe write done by the CPU when its part has run: into the
/// title's memory, or for the pool's GPU view into the page's home and
/// then the page's card copy.
fn write_now(b: &Backend, at: u64, what: Written, bytes: usize) {
    let target = memory(b).cpu_of(at);
    match target {
        Some((cpu, pool)) => {
            what.write(cpu as *mut u8, bytes);
            if pool.is_some() {
                title_memory_written(at as usize, bytes);
            }
        }
        None => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    at = format_args!("{at:#x}"),
                    "Vulkan: a write into a page of the pool's GPU view nothing shows was dropped"
                )
            });
        }
    }
}

pub(super) fn engine(b: &Backend) -> MutexGuard<'_, Engine> {
    lock(&b.engine)
}

/// The current part of `list`'s recording (made if there is none).
fn current(e: &mut Engine, list: This) -> &mut Part {
    let parts = e.lists.entry(list as usize).or_default();
    if parts.is_empty() {
        parts.push(Part::default());
    }
    parts.last_mut().expect("part")
}

/// A list's wait for a memory value: what it records next is behind it.
pub(crate) fn list_wait(list: This, wait: ValueWait) {
    let Some(b) = backend() else { return };
    let mut e = engine(b);
    let parts = e.lists.entry(list as usize).or_default();
    match parts.last_mut() {
        Some(p) if p.writes.is_empty() && p.cmds.is_empty() => p.waits.push(wait),
        _ => parts.push(Part {
            waits: vec![wait],
            ..Part::default()
        }),
    }
}

/// A list goes: what it recorded and did not submit goes with it.
pub(crate) fn list_released(list: This) {
    list_reset(list);
    record::forget(list);
}

/// A list's recorded command buffer, into its current part (at a cut or
/// its close).
pub(crate) fn part_commands(list: This, rec: Recording) {
    let Some(b) = backend() else { return };
    current(&mut engine(b), list).cmds.push(rec);
}

/// A list's bottom-of-pipe write.
pub(crate) fn list_write(list: This, at: u64, what: Written, bytes: usize) {
    let Some(b) = backend() else { return };
    current(&mut engine(b), list).writes.push((at, what, bytes));
}

/// A list's reset (or release): what it recorded and did not submit goes.
pub(crate) fn list_reset(list: This) {
    if let Some(b) = backend() {
        let parts = engine(b).lists.remove(&(list as usize));
        for part in parts.into_iter().flatten() {
            for rec in part.cmds {
                b.timeline.abandon(&b.host, rec);
            }
        }
    }
}

pub(crate) fn execute(queue: This, lists: &[This]) {
    let Some(b) = backend() else { return };
    // What the title's threads wrote down into them is recorded first.
    let t = Instant::now();
    crate::list::join(lists);
    split(Split::Join, t);
    let mut e = engine(b);
    let mut parts = VecDeque::new();
    for &l in lists {
        match e.lists.remove(&(l as usize)) {
            Some(p) => parts.extend(p),
            // A list with nothing to wait for or write still runs, in order.
            None => parts.push_back(Part::default()),
        }
    }
    e.push(b, queue as usize, Item::Run(parts));
    let t = Instant::now();
    images::flush(b);
    split(Split::Flush, t);
    let t = Instant::now();
    e.pump(b, false);
    split(Split::Submit, t);
    b.changed.notify_all();
}

pub(crate) fn signal(queue: This, fence: This, value: u64) {
    let Some(b) = backend() else {
        crate::fence::set(fence, value);
        return;
    };
    let mut e = engine(b);
    e.push(b, queue as usize, Item::Signal(fence as usize, value));
    e.pump(b, false);
    b.changed.notify_all();
}

pub(crate) fn wait(queue: This, fence: This, value: u64) {
    let Some(b) = backend() else { return };
    let mut e = engine(b);
    e.push(b, queue as usize, Item::Wait(fence as usize, value));
    e.pump(b, false);
}

/// A fence was signalled by the CPU: lines waiting for it run on.
pub(crate) fn fence_signalled() {
    let Some(b) = backend() else { return };
    engine(b).pump(b, false);
    b.changed.notify_all();
}

/// A CPU wait for a fence with no event: blocks until it is reached, or
/// until nothing given can reach it (then `false`).
pub(crate) fn wait_for_fence(fence: This, value: u64) -> bool {
    let Some(b) = backend() else { return false };
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut e = engine(b);
    loop {
        if crate::fence::completed(fence) >= value {
            return true;
        }
        if e.flights.is_empty() && e.held() == 0 {
            return false;
        }
        if Instant::now() > deadline {
            tracing::warn!(value, "Vulkan: a fence wait gave up after 10 s");
            return false;
        }
        e = b
            .changed
            .wait_timeout(e, Duration::from_millis(50))
            .unwrap_or_else(|p| p.into_inner())
            .0;
    }
}

pub(super) fn start_landing() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-vk-landing".into())
            .spawn(landing);
    });
}

/// Diagnostics: the notes of a submission's command buffers.
fn take_trails(cmds: &mut [Recording]) -> (Vec<String>, Vec<Vec<u32>>) {
    if !cfg!(feature = "diagnostics") {
        return (Vec::new(), Vec::new());
    }
    let (mut out, mut tables) = (Vec::new(), Vec::new());
    for (k, r) in cmds.iter_mut().enumerate() {
        let t = std::mem::take(&mut r.trail);
        if t.notes.is_empty() {
            continue;
        }
        out.push(format!("cmd {k} ({} notes before)", t.earlier));
        tables.push(Vec::new());
        out.extend(t.notes);
        tables.extend(t.tables);
    }
    (out, tables)
}

/// Diagnostics at a device loss: the T#s in the first 16 descriptors (32
/// bytes each) at each table address, decoded, with the view the host
/// wrote at each one's slot.
fn tsharps(b: &Backend, tables: &[u32]) -> String {
    let mut out = Vec::new();
    for &t in tables.iter().take(3) {
        let Ok(bytes) = super::read_title(b, u64::from(t), 16 * 32) else {
            out.push(format!("table {t:#x}: unreadable"));
            continue;
        };
        for (k, d) in bytes.chunks_exact(32).enumerate() {
            let w: Vec<u32> = d.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            if w[6] ^ w[7] != 0x7A3C_5E91 {
                continue;
            }
            let slot = w[7] & 0xff_ffff;
            out.push(format!(
                "table {t:#x}[{k}]: TYPE={} fmt={:#x} slot={slot} int={} storage={} host={}",
                w[3] >> 28,
                (w[1] >> 20) & 0x1ff,
                w[7] >> 24 & 1,
                w[7] >> 25 & 1,
                super::descriptors::slot_view(slot)
            ));
        }
    }
    out.join("; ")
}

/// A submission the card has not finished this long is reported.
const STALL: Duration = Duration::from_millis(1000);

/// Diagnostics, polled every 250 ms: when a line's oldest submission in the
/// air has not run within [`STALL`], the oldest unfinished submission of
/// every line still running is reported once, with what its command
/// buffers record and the lines' values (a hang: the shader that runs too
/// long is in the first such submission of the line that did not move;
/// the others may wait for it).
pub(super) fn report_stalls(b: &Backend) {
    static REPORTED: std::sync::Mutex<Vec<(Line, u64)>> = std::sync::Mutex::new(Vec::new());
    if !cfg!(feature = "diagnostics") || b.timeline.lost() {
        return;
    }
    let e = engine(b);
    let mut fronts: Vec<(&Flight, u64)> = Vec::new();
    for f in &e.flights {
        if fronts.iter().any(|(g, _)| g.line == f.line) {
            continue;
        }
        let reached = b.timeline.line_completed(&b.host, f.line);
        if reached < f.value {
            fronts.push((f, reached));
        }
    }
    if !fronts.iter().any(|(f, _)| f.given.elapsed() >= STALL) {
        return;
    }
    for (f, reached) in fronts {
        if lock(&REPORTED).contains(&(f.line, f.value)) {
            continue;
        }
        lock(&REPORTED).push((f.line, f.value));
        let (given, label) = b.timeline.line_given(f.line);
        let (host_given, _) = b.timeline.line_given(HOST);
        tracing::error!(
            line = %label,
            value = f.value,
            reached,
            given_last = given,
            host_reached = b.timeline.line_completed(&b.host, HOST),
            host_given,
            waited_ms = f.given.elapsed().as_millis() as u64,
            notes = f.trail.len(),
            trail = %f.trail.join(" | "),
            "Vulkan: a submission has not finished (a hang?)"
        );
    }
}

/// Says once, loudly, that the device was lost.
pub(super) fn lost_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::error!(
            "Vulkan: the device was lost; the run goes on without the card (bottom-of-pipe writes and fence signals on the CPU, no presents)"
        )
    });
}

/// Follows every line: lands submissions as they run and runs the queues'
/// lines on.
fn landing() {
    let Some(b) = backend() else { return };
    loop {
        let fronts = {
            let e = engine(b);
            let fronts = e.fronts();
            if fronts.is_empty() {
                let _ = b.changed.wait_timeout(e, Duration::from_millis(100));
                continue;
            }
            fronts
        };
        // A lost device reaches nothing: what is in the air lands now, so
        // that the title's fences still move (its writes and signals on
        // the CPU, as without a card).
        let lost = b.timeline.lost();
        if lost {
            engine(b).report_lost(b);
            lost_once();
        } else if !b.timeline.wait_any(&b.host, &fronts, 100_000_000) {
            continue;
        }
        let mut e = engine(b);
        e.land(b, lost);
        e.pump(b, false);
        drop(e);
        b.changed.notify_all();
    }
}

/// The queues' part of the 10 s line: submissions in the air, items held,
/// and each line's submissions and GPU busy time per frame (and as a share
/// of the time since the last line).
pub(super) fn report_text(b: &Backend, frames: u64) -> (usize, usize, String) {
    static LAST: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
    let wall_ms = {
        let mut last = lock(&LAST);
        let ms = last.map_or(10_000.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        *last = Some(Instant::now());
        ms.max(1.0)
    };
    let (flights, held) = {
        let e = engine(b);
        (e.flights.len(), e.held())
    };
    let frames = frames.max(1) as f64;
    let lines: Vec<String> = b
        .timeline
        .take_stats(&b.host)
        .iter()
        .filter(|l| l.submissions > 0)
        .map(|l| {
            format!(
                "{} {:.1} ms/frame {:.0}% n={:.1}/frame",
                l.label,
                l.busy_ms / frames,
                100.0 * l.busy_ms / wall_ms,
                l.submissions as f64 / frames
            )
        })
        .collect();
    (flights, held, lines.join("; "))
}
