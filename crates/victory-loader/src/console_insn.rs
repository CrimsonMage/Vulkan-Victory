//! Instructions the console's CPU (Zen 2) has and an Intel PC's lacks: the title's code raises
//! STATUS_ILLEGAL_INSTRUCTION on them (it zeroes memory with `clzero` loops all through a
//! frame, spin-waits with MONITORX / MWAITX, packs fields with INSERTQ), and the vectored
//! handler emulates them here. Each such site is then rewritten, once, as the plain x86 code
//! it stands for (`console_insn_stub.rs`), so it does not trap again: under Wine a trap costs
//! a signal and an exception dispatch, tens of microseconds, and the title met ~21,000 a
//! second in its world. What has no rewrite stays emulated at every trap.
//!
//! Ported from XCL (github.com/ivall/xcl, GPL-2.0-only): `insn.rs` (decoder, as
//! `console_insn_decode.rs`) and `eh.rs` `emulate_console_only`.

#[path = "console_insn_decode.rs"]
mod decode;
#[path = "console_insn_stub.rs"]
mod stub;

use decode::ConsoleOnly;
use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering::Relaxed};
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

const CONTINUE_EXECUTION: i32 = -1;

/// What Windows said about memory is remembered per thread (the regions it found writable,
/// the code bytes it found readable) and not asked again at every emulated instruction. A
/// query walks the pages of the region it answers for: it took 2-9 ms in the title's heaps of
/// 0.6-2.2 GB and 89 ms in its 6.2 GB page pool (run 30), and the title zeroes blocks of them
/// with `clzero` loops all through a frame.
///
/// Answers a remembered region gives before Windows is asked about it again: one freed or
/// protected meanwhile is forgotten then.
const ANSWERS: u32 = 4096;

thread_local! {
    /// The regions Windows last said are committed and writable (start, end), newest first,
    /// and the answers they gave since they were emptied.
    static WRITABLE: Cell<([(usize, usize); 4], u32)> = const { Cell::new(([(0, 0); 4], 0)) };
    /// Faulting addresses and the code bytes found readable there, newest first.
    static CODE: Cell<[(usize, usize); 8]> = const { Cell::new([(0, 0); 8]) };
}

fn readable(va: usize, bytes: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    found && m.State == MEM_COMMIT && m.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

fn writable(va: usize, bytes: usize) -> bool {
    writable_to(va, va + bytes) == Some(va + bytes)
}

/// Where the committed, writable memory that `va` lies in ends, at most `end`; None if `va`
/// is not in such memory. A region found is remembered ([`remembers`]).
/// shortcut: a remembered region that was freed or protected since is written all the same
/// for up to [`ANSWERS`] answers: the write then faults inside the handler, which the guard
/// on view memory opens as it does any write, and which is a crash otherwise (as the
/// unchecked single-line emulation behind this one already is).
fn writable_to(va: usize, end: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualQuery};
    let (regions, answers) = WRITABLE.with(Cell::get);
    if let Some(&(_, to)) = regions.iter().find(|&&(from, to)| from <= va && va < to) {
        let next = if answers >= ANSWERS { ([(0, 0); 4], 0) } else { (regions, answers + 1) };
        WRITABLE.with(|w| w.set(next));
        return Some(to.min(end));
    }
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    if !found || m.State != MEM_COMMIT || m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 {
        return None;
    }
    let (from, to) = (m.BaseAddress as usize, m.BaseAddress as usize + m.RegionSize);
    WRITABLE.with(|w| {
        let (mut regions, answers) = w.get();
        regions.rotate_right(1);
        regions[0] = (from, to);
        w.set((regions, answers));
    });
    Some(to.min(end))
}

/// An illegal instruction at `ctx.Rip`: a console-only one is emulated and stepped over
/// (`CONTINUE_EXECUTION`); anything else returns 0 (the crash report and the title's handlers
/// see it).
///
/// # Safety
/// `ctx` is the faulting thread's context, from the exception being dispatched.
pub(crate) unsafe fn emulate(ctx: &mut CONTEXT) -> i32 {
    let rip = ctx.Rip as usize;
    // The same few instructions fault again and again: their bytes are asked about once.
    let known = CODE.with(Cell::get);
    let n = match known.iter().find(|&&(at, n)| at == rip && n > 0) {
        Some(&(_, n)) => n,
        None => {
            if !readable(rip, 1) {
                return 0;
            }
            let n = (1..=16).rev().find(|&n| readable(rip, n)).unwrap_or(0);
            let mut known = known;
            known.rotate_right(1);
            known[0] = (rip, n);
            CODE.with(|c| c.set(known));
            n
        }
    };
    // A copy: the rewrite below changes the bytes at `rip`.
    let mut copy = [0u8; 16];
    // SAFETY: `n` (at most 16) bytes at `rip` are readable.
    copy[..n].copy_from_slice(unsafe { std::slice::from_raw_parts(rip as *const u8, n) });
    let code = &copy[..n];
    let Some((what, len)) = decode::decode(code) else {
        // Rewritten by another thread after this one fetched the old bytes: run it again.
        return match find(rip) {
            Some(site) if site.state.load(Relaxed) == REWRITTEN => {
                count(site);
                CONTINUE_EXECUTION
            }
            _ => 0,
        };
    };
    if what != ConsoleOnly::Ud2
        && let Some(site) = site(rip, &what)
    {
        count(site);
        rewrite(site, rip, &what, &code[..len]);
    }
    static SEEN: std::sync::Mutex<Vec<std::mem::Discriminant<ConsoleOnly>>> = std::sync::Mutex::new(Vec::new());
    if let Ok(mut seen) = SEEN.lock()
        && !seen.contains(&std::mem::discriminant(&what))
    {
        seen.push(std::mem::discriminant(&what));
        if what != ConsoleOnly::Ud2 {
            tracing::warn!(instruction = ?what, at = format_args!("{rip:#x}"), "a console CPU instruction this CPU lacks: emulated (logged once)");
        }
    }
    match what {
        ConsoleOnly::Ud2 => return 0,
        // nothing to write back on the PC (rewritten as a NOP above)
        ConsoleOnly::Clwb => {}
        ConsoleOnly::Monitorx => {}
        ConsoleOnly::Mwaitx => std::hint::spin_loop(),
        ConsoleOnly::Clzero => {
            // with an address-size prefix (0x67) the line is at EAX, zero-extended
            let a32 = code[..len - 3].contains(&0x67);
            let line = (if a32 { ctx.Rax & 0xFFFF_FFFF } else { ctx.Rax } as usize) & !63;
            if !writable(line, 64) {
                return 0; // the real instruction faults there too
            }
            // SAFETY: the 64-byte line is committed and writable.
            unsafe { std::ptr::write_bytes(line as *mut u8, 0, 64) };
        }
        ConsoleOnly::ClzeroLoop { end_reg } => {
            // the whole `clzero; add rax,40h; cmp rax,end; jb` loop at once
            let regs: [u64; 16] = [
                ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.Rbx, ctx.Rsp, ctx.Rbp, ctx.Rsi, ctx.Rdi, ctx.R8, ctx.R9, ctx.R10, ctx.R11, ctx.R12, ctx.R13, ctx.R14, ctx.R15,
            ];
            let end = regs[end_reg as usize & 15] as usize;
            let start = ctx.Rax as usize;
            if end > start && end - start < 2 << 30 {
                // Region by region: a block may lie across several (the title's pages are
                // mapped one view each), and one answer for all of it was a refusal, after
                // which every 64-byte line took an exception of its own.
                let mut at = start;
                while at < end {
                    let Some(to) = writable_to(at, end) else { break };
                    // SAFETY: committed and writable from `at` to `to`.
                    unsafe { std::ptr::write_bytes(at as *mut u8, 0, to - at) };
                    at = to;
                }
                if at == start {
                    return 0;
                }
                if at < end {
                    // What follows is not writable: the loop goes on there (the same
                    // instruction again), where the real one faults too.
                    ctx.Rax = at as u64;
                    return CONTINUE_EXECUTION;
                }
            }
            ctx.Rax = end as u64;
        }
        ConsoleOnly::Rdpru => {
            let mut v = 0i64;
            // SAFETY: a valid out pointer.
            unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut v) };
            let v = if ctx.Rcx as u32 <= 1 { v as u64 } else { 0 };
            ctx.Rax = v & 0xFFFF_FFFF;
            ctx.Rdx = v >> 32;
        }
        ConsoleOnly::Rdpid(r) => {
            // SAFETY: no preconditions.
            let cpu = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as u64;
            let regs: [&mut u64; 16] = [
                &mut ctx.Rax, &mut ctx.Rcx, &mut ctx.Rdx, &mut ctx.Rbx, &mut ctx.Rsp, &mut ctx.Rbp, &mut ctx.Rsi, &mut ctx.Rdi, &mut ctx.R8, &mut ctx.R9, &mut ctx.R10, &mut ctx.R11, &mut ctx.R12, &mut ctx.R13,
                &mut ctx.R14, &mut ctx.R15,
            ];
            *regs[r as usize & 15] = cpu;
        }
        // the low quadword changes; the high one is zeroed, as a Zen 5 does
        ConsoleOnly::Extrq { dst, src, len, idx } => {
            // SAFETY: the context's XMM registers (CONTEXT_FLOATING_POINT is captured for exceptions).
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let (len, idx) = match src {
                Some(s) => {
                    let c = x[s as usize & 15].Low;
                    (c as u8, (c >> 8) as u8)
                }
                None => (len, idx),
            };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::extrq(d.Low, len, idx);
            d.High = 0;
        }
        ConsoleOnly::Insertq { dst, src, reg, len, idx } => {
            // SAFETY: as above.
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let s = x[src as usize & 15];
            let (len, idx) = if reg { (s.High as u8, (s.High >> 8) as u8) } else { (len, idx) };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::insertq(d.Low, s.Low, len, idx);
            d.High = 0;
        }
    }
    ctx.Rip += len as u64;
    CONTINUE_EXECUTION
}

/// What traps, by kind of instruction (indexes of [`KINDS`]).
const KINDS: [&str; 9] = ["clzero", "clzero loop", "monitorx", "mwaitx", "insertq", "extrq", "rdpru", "rdpid", "clwb"];

fn kind(what: &ConsoleOnly) -> u8 {
    match what {
        ConsoleOnly::Clzero | ConsoleOnly::Ud2 => 0,
        ConsoleOnly::ClzeroLoop { .. } => 1,
        ConsoleOnly::Monitorx => 2,
        ConsoleOnly::Mwaitx => 3,
        ConsoleOnly::Insertq { .. } => 4,
        ConsoleOnly::Extrq { .. } => 5,
        ConsoleOnly::Rdpru => 6,
        ConsoleOnly::Rdpid(_) => 7,
        ConsoleOnly::Clwb => 8,
    }
}

/// Traps since the start, by kind.
static TRAPS: [AtomicU64; KINDS.len()] = [const { AtomicU64::new(0) }; KINDS.len()];

/// A trapping instruction: where, how often, what it is, and whether it was rewritten.
struct Site {
    at: AtomicUsize,
    traps: AtomicU64,
    kind: AtomicU8,
    state: AtomicU8,
}

/// [`Site::state`]: not looked at yet, rewritten, or left to the emulation.
const UNTRIED: u8 = 0;
const REWRITTEN: u8 = 1;
const EMULATED: u8 = 2;

/// Sites by address (open addressing); one past the last free slot is emulated, uncounted.
const SLOTS: usize = 1024;
static SITES: [Site; SLOTS] = [const {
    Site {
        at: AtomicUsize::new(0),
        traps: AtomicU64::new(0),
        kind: AtomicU8::new(0),
        state: AtomicU8::new(UNTRIED),
    }
}; SLOTS];

fn slot(at: usize) -> usize {
    ((at as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - SLOTS.trailing_zeros())) as usize
}

/// The site at `at`, if it trapped before.
fn find(at: usize) -> Option<&'static Site> {
    (0..SLOTS)
        .map(|k| &SITES[(slot(at) + k) % SLOTS])
        .take_while(|s| s.at.load(Relaxed) != 0)
        .find(|s| s.at.load(Relaxed) == at)
}

/// The site at `at`, taken on its first trap.
fn site(at: usize, what: &ConsoleOnly) -> Option<&'static Site> {
    for k in 0..SLOTS {
        let s = &SITES[(slot(at) + k) % SLOTS];
        match s.at.compare_exchange(0, at, Relaxed, Relaxed) {
            Ok(_) => {
                s.kind.store(kind(what), Relaxed);
                return Some(s);
            }
            Err(taken) if taken == at => return Some(s),
            Err(_) => {}
        }
    }
    None
}

fn count(site: &Site) {
    site.traps.fetch_add(1, Relaxed);
    TRAPS[site.kind.load(Relaxed) as usize % KINDS.len()].fetch_add(1, Relaxed);
}

/// Pages of stubs: (address, bytes used).
struct Stubs([(usize, usize); 16]);

const STUB_PAGE: usize = 4096;

impl Stubs {
    /// Room for `len` bytes of stub that `site` jumps to and that jumps back to `site`.
    fn room(&mut self, site: usize, len: usize) -> Option<&mut (usize, usize)> {
        use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
        let fits = |&(page, used): &(usize, usize)| {
            page != 0 && used + len <= STUB_PAGE && stub::jmp(site, page + used).is_some() && stub::jmp(page + used + len, site).is_some()
        };
        if let Some(i) = self.0.iter().position(fits) {
            return Some(&mut self.0[i]);
        }
        let free = self.0.iter().position(|&(page, _)| page == 0)?;
        let page = crate::tls::near(site & !0xFFFF, 0x1_0000).ok()? as usize;
        let mut old = 0;
        // SAFETY: the fresh page of ours.
        if unsafe { VirtualProtect(page as *const _, STUB_PAGE, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            return None;
        }
        self.0[free] = (page, 0);
        fits(&self.0[free]).then_some(&mut self.0[free])
    }
}

/// Rewrites the instruction at `at` (its bytes `code`, decoded to `what`) on its first trap,
/// so it traps no more. Other threads may be running the code around it: the bytes that
/// change are written in one atomic store (none if they cross an aligned 16 bytes), so a
/// thread fetches either the old instruction, which traps and is emulated (or, already
/// rewritten, run again), or the new one. A stub replaces a whole instruction, or a whole
/// `clzero` loop, which the emulation steps over at once, so no thread is ever inside it.
fn rewrite(site: &Site, at: usize, what: &ConsoleOnly, code: &[u8]) {
    static STUBS: std::sync::Mutex<Stubs> = std::sync::Mutex::new(Stubs([(0, 0); 16]));
    if site.state.load(Relaxed) != UNTRIED {
        return;
    }
    if !image_code(at) {
        // not an image's code (a test's buffer): left as it is
        site.state.store(EMULATED, Relaxed);
        return;
    }
    // Another thread is rewriting: this site waits for its next trap.
    let Ok(mut stubs) = STUBS.try_lock() else { return };
    if site.state.load(Relaxed) != UNTRIED {
        return;
    }
    let how = match stub::rewrite(what, code) {
        Some(stub::Rewrite::InPlace(bytes)) => write_code(at, &bytes).then_some("in place"),
        Some(stub::Rewrite::Stub) => (|| {
            let room = stubs.room(at, 128)?;
            let to = room.0 + room.1;
            let bytes = stub::stub(what, code, at, to)?;
            let jump = stub::jmp(at, to)?;
            // SAFETY: free bytes of a stub page of ours, which nothing runs before the jump.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, bytes.len()) };
            room.1 += bytes.len().next_multiple_of(16);
            write_code(at, &jump).then_some("jump to a stub")
        })(),
        None => None,
    };
    site.state.store(if how.is_some() { REWRITTEN } else { EMULATED }, Relaxed);
    let title = crate::threads::title_address(at).map(|a| format!(" (title {a:#x})")).unwrap_or_default();
    match how {
        Some(how) => tracing::info!(instruction = ?what, how, "console CPU instruction at {at:#x}{title} rewritten as plain x86 code"),
        None => tracing::info!(instruction = ?what, "console CPU instruction at {at:#x}{title} has no rewrite: emulated at each trap"),
    }
}

/// Whether `at` is executable code of a mapped image.
fn image_code(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_IMAGE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } != 0;
    found && m.Type == MEM_IMAGE && m.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0
}

/// Writes `bytes` at `at` in one atomic store, of the aligned 8 or 16 bytes of code that hold
/// them. False if they cross an aligned 16 bytes, or the code changed meanwhile.
fn write_code(at: usize, bytes: &[u8]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let n = bytes.len();
    let width = if at % 8 + n <= 8 {
        8
    } else if at % 16 + n <= 16 {
        16
    } else {
        return false;
    };
    let word = at & !(width - 1);
    let off = at - word;
    let mut old = 0;
    // SAFETY: the code's own page, made writable for the store and restored after.
    if unsafe { VirtualProtect(word as *const _, width, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        return false;
    }
    let done = if width == 8 {
        // SAFETY: aligned, and writable now.
        let word = unsafe { AtomicU64::from_ptr(word as *mut u64) };
        let was = word.load(Relaxed);
        let mut new = was.to_le_bytes();
        new[off..off + n].copy_from_slice(bytes);
        word.compare_exchange(was, u64::from_le_bytes(new), Relaxed, Relaxed).is_ok()
    } else {
        // SAFETY: aligned, and writable now; a torn read fails the exchange.
        let was = unsafe { (word as *const u128).read_volatile() };
        let mut new = was.to_le_bytes();
        new[off..off + n].copy_from_slice(bytes);
        // SAFETY: as above.
        unsafe { exchange16(word as *mut u128, was, u128::from_le_bytes(new)) }
    };
    // SAFETY: restores the protection found; the range is the code just written.
    unsafe {
        VirtualProtect(word as *const _, width, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at as *const _, n);
    }
    done
}

/// `lock cmpxchg16b`: `*p = new` if it holds `old`.
///
/// # Safety
/// `p` is 16-byte aligned and writable.
unsafe fn exchange16(p: *mut u128, old: u128, new: u128) -> bool {
    let done: u8;
    // SAFETY: as the caller says; rbx (which the compiler reserves) is swapped in and back.
    unsafe {
        std::arch::asm!(
            "xchg {low}, rbx",
            "lock cmpxchg16b xmmword ptr [{p}]",
            "sete {done}",
            "mov rbx, {low}",
            p = in(reg) p,
            low = inout(reg) new as u64 => _,
            done = out(reg_byte) done,
            in("rcx") (new >> 64) as u64,
            inout("rax") old as u64 => _,
            inout("rdx") (old >> 64) as u64 => _,
        );
    }
    done != 0
}

/// Exceptions dispatched to handlers since the start: access violations on reads (and
/// executes), on writes, illegal instructions, others. Under Wine a fault it settles itself
/// (a write to a write-watched page) is a signal that never gets here.
static DISPATCHED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

unsafe extern "system" fn count_exception(info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> i32 {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, EXCEPTION_ILLEGAL_INSTRUCTION};
    // SAFETY: the system passes the exception being dispatched.
    let r = unsafe { &*(*info).ExceptionRecord };
    let i = match r.ExceptionCode {
        EXCEPTION_ACCESS_VIOLATION if r.NumberParameters >= 1 && r.ExceptionInformation[0] == 1 => 1,
        EXCEPTION_ACCESS_VIOLATION => 0,
        EXCEPTION_ILLEGAL_INSTRUCTION => 2,
        _ => 3,
    };
    DISPATCHED[i].fetch_add(1, Relaxed);
    0 // EXCEPTION_CONTINUE_SEARCH
}

/// Starts the line every 10 s on console CPU instructions: traps by kind, the sites and how
/// many were rewritten, the busiest (at their address in the title's file), and the
/// exceptions dispatched (which tells the faults the runtime's handlers take from those Wine
/// settles itself). Silent until the first trap.
pub(crate) fn start_counting() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("console-insn-stats".into()).spawn(|| {
            use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, RemoveVectoredExceptionHandler};
            let mut before = ([0u64; KINDS.len()], [0u64; 4]);
            let mut counter: *mut std::ffi::c_void = std::ptr::null_mut();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(10));
                // First in line again (handlers added since go before it), to see them all.
                // SAFETY: a handler that only counts.
                unsafe {
                    if !counter.is_null() {
                        RemoveVectoredExceptionHandler(counter);
                    }
                    counter = AddVectoredExceptionHandler(1, Some(count_exception));
                }
                let traps: [u64; KINDS.len()] = std::array::from_fn(|i| TRAPS[i].load(Relaxed));
                let dispatched: [u64; 4] = std::array::from_fn(|i| DISPATCHED[i].load(Relaxed));
                if traps.iter().sum::<u64>() == 0 {
                    continue;
                }
                tracing::info!("{}", stats_line(&traps, &before.0, &dispatched, &before.1));
                before = (traps, dispatched);
            }
        });
    });
}

fn stats_line(traps: &[u64; KINDS.len()], before: &[u64; KINDS.len()], dispatched: &[u64; 4], dispatched_before: &[u64; 4]) -> String {
    let d: Vec<u64> = traps.iter().zip(before).map(|(n, b)| n - b).collect();
    let by_kind: Vec<String> = (0..KINDS.len()).filter(|&i| d[i] > 0).map(|i| format!("{} {}", KINDS[i], d[i])).collect();
    let mut sites: Vec<(u64, usize, u8, u8)> = SITES
        .iter()
        .filter(|s| s.at.load(Relaxed) != 0)
        .map(|s| (s.traps.load(Relaxed), s.at.load(Relaxed), s.kind.load(Relaxed), s.state.load(Relaxed)))
        .collect();
    let rewritten = sites.iter().filter(|s| s.3 == REWRITTEN).count();
    sites.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
    let busiest: Vec<String> = sites
        .iter()
        .take(5)
        .map(|&(n, at, kind, state)| {
            let place = crate::threads::title_address(at).map_or_else(|| crate::crash::locate(at), |a| format!("{a:#x}"));
            let state = ["", " (rewritten)", " (emulated)"][state as usize % 3];
            format!("{place} {} {n}{state}", KINDS[kind as usize % KINDS.len()])
        })
        .collect();
    let e: Vec<u64> = dispatched.iter().zip(dispatched_before).map(|(n, b)| n - b).collect();
    format!(
        "console CPU instructions: {} traps in 10 s ({}), {} since the start at {} sites ({} rewritten); busiest: {}; \
         exceptions dispatched in 10 s: access violations {} ({} on writes), illegal instructions {}, other {}",
        d.iter().sum::<u64>(),
        if by_kind.is_empty() { "none".into() } else { by_kind.join(", ") },
        traps.iter().sum::<u64>(),
        sites.len(),
        rewritten,
        busiest.join(", "),
        e[0] + e[1],
        e[1],
        e[2],
        e[3],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Diagnostics::Debug::M128A;

    fn reg(low: u64, high: u64) -> M128A {
        M128A {
            Low: low,
            High: high as i64,
        }
    }

    /// Gives `code` to the emulation as if it faulted at its first byte,
    /// with `rax` and `xmm` as the saved registers. The emulation does not
    /// execute `code`, so this works on any CPU. Returns the number of bytes
    /// that the emulation skipped, or `None` if it did not emulate the
    /// instruction. `xmm` gets the saved registers after.
    fn run(code: &[u8], rax: u64, xmm: &mut [M128A; 16]) -> Option<u64> {
        // `int3` after the instruction, so that a decoder that reads too
        // far sees no valid bytes.
        let mut bytes = [0xcc_u8; 16];
        bytes[..code.len()].copy_from_slice(code);
        let start = bytes.as_ptr() as u64;
        // SAFETY: a zeroed context is valid; the emulation reads only
        // `bytes` and writes only `context` (and memory at `rax` for
        // clzero, which the caller owns).
        unsafe {
            let mut context: CONTEXT = std::mem::zeroed();
            context.Rip = start;
            context.Rax = rax;
            context.Anonymous.FltSave.XmmRegisters = *xmm;
            let emulated = emulate(&mut context) == CONTINUE_EXECUTION;
            *xmm = context.Anonymous.FltSave.XmmRegisters;
            emulated.then(|| context.Rip - start)
        }
    }

    fn low(xmm: &[M128A; 16], i: usize) -> u64 {
        xmm[i].Low
    }

    #[test]
    fn emulated_clzero_zeroes_its_line() {
        #[repr(align(64))]
        struct Line([u8; 128]);
        let mut line = Line([0xff; 128]);
        let at = line.0.as_mut_ptr().wrapping_add(5) as u64; // inside the first line
        let mut xmm = [reg(0, 0); 16];
        assert_eq!(run(&[0x0f, 0x01, 0xfc], at, &mut xmm), Some(3));
        assert!(line.0[..64].iter().all(|&b| b == 0));
        assert!(line.0[64..].iter().all(|&b| b == 0xff));
    }

    #[test]
    fn emulated_monitorx_and_mwaitx_are_skipped() {
        let mut xmm = [reg(1, 2); 16];
        assert_eq!(run(&[0x0f, 0x01, 0xfa], 0, &mut xmm), Some(3));
        assert_eq!(run(&[0x0f, 0x01, 0xfb], 0, &mut xmm), Some(3));
        assert!(xmm.iter().all(|r| r.Low == 1 && r.High == 2));
    }

    #[test]
    fn emulated_insertq_and_extrq_with_immediates() {
        let mut xmm = [reg(0, 0); 16];
        xmm[0] = reg(0x1111_2222_3333_4444, 0x5555);
        xmm[1] = reg(0xabcd, 0x6666);
        // insertq xmm0, xmm1, length 16, index 16
        let insertq = [0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10];
        assert_eq!(run(&insertq, 0, &mut xmm), Some(6));
        assert_eq!(low(&xmm, 0), 0x1111_2222_abcd_4444);
        assert_eq!((low(&xmm, 1), xmm[1].High), (0xabcd, 0x6666));
        // extrq xmm0, length 8, index 20
        let extrq = [0x66, 0x0f, 0x78, 0xc0, 0x08, 0x14];
        assert_eq!(run(&extrq, 0, &mut xmm), Some(6));
        assert_eq!(low(&xmm, 0), 0xbc);
    }

    #[test]
    fn emulated_insertq_and_extrq_with_fields_in_a_register() {
        let mut xmm = [reg(0, 0); 16];
        // insertq xmm0, xmm1: length and index in xmm1's high half.
        xmm[0] = reg(0x1111_2222_3333_4444, 0);
        xmm[1] = reg(0xabcd, 0x10_10);
        assert_eq!(run(&[0xf2, 0x0f, 0x79, 0xc1], 0, &mut xmm), Some(4));
        assert_eq!(low(&xmm, 0), 0x1111_2222_abcd_4444);
        // extrq xmm0, xmm1: length and index in xmm1's low half.
        xmm[1] = reg(0x14_08, 0);
        assert_eq!(run(&[0x66, 0x0f, 0x79, 0xc1], 0, &mut xmm), Some(4));
        assert_eq!(low(&xmm, 0), 0xbc);
    }

    #[test]
    fn emulated_sse4a_reaches_xmm8_to_xmm15() {
        let mut xmm = [reg(0, 0); 16];
        xmm[8] = reg(0x1111_2222_3333_4444, 0);
        xmm[9] = reg(0xabcd, 0);
        // insertq xmm8, xmm9, length 16, index 16 (REX.R and REX.B)
        let insertq = [0xf2, 0x45, 0x0f, 0x78, 0xc1, 0x10, 0x10];
        assert_eq!(run(&insertq, 0, &mut xmm), Some(7));
        assert_eq!(low(&xmm, 8), 0x1111_2222_abcd_4444);
        assert_eq!((low(&xmm, 0), low(&xmm, 1)), (0, 0));
        // extrq xmm9, length 8, index 4 (REX.B)
        let extrq = [0x66, 0x41, 0x0f, 0x78, 0xc1, 0x08, 0x04];
        assert_eq!(run(&extrq, 0, &mut xmm), Some(7));
        assert_eq!(low(&xmm, 9), 0xbc);
        assert_eq!(low(&xmm, 1), 0);
    }

    #[test]
    fn emulated_sse4a_length_0_is_64_bits() {
        let mut xmm = [reg(0, 0); 16];
        xmm[0] = reg(0x1122_3344_5566_7788, 0);
        // extrq xmm0, length 0 (64), index 0. A non-zero index with
        // length 64 goes past bit 63, and AMD leaves that result undefined.
        assert_eq!(
            run(&[0x66, 0x0f, 0x78, 0xc0, 0x00, 0x00], 0, &mut xmm),
            Some(6)
        );
        assert_eq!(low(&xmm, 0), 0x1122_3344_5566_7788);
        // insertq xmm0, xmm1, length 0 (64), index 0
        xmm[1] = reg(0xdead_beef_cafe_f00d, 0);
        assert_eq!(
            run(&[0xf2, 0x0f, 0x78, 0xc1, 0x00, 0x00], 0, &mut xmm),
            Some(6)
        );
        assert_eq!(low(&xmm, 0), 0xdead_beef_cafe_f00d);
    }

    #[test]
    fn other_instructions_are_not_emulated() {
        let mut xmm = [reg(0, 0); 16];
        assert_eq!(run(&[0x0f, 0x0b], 0, &mut xmm), None); // ud2
        // extrq with a memory operand, which does not exist
        assert_eq!(run(&[0x66, 0x0f, 0x79, 0x00], 0, &mut xmm), None);
        // extrq with immediates needs ModRM.reg 0
        let bad_extrq = [0x66, 0x0f, 0x78, 0xc8, 0x08, 0x14];
        assert_eq!(run(&bad_extrq, 0, &mut xmm), None);
    }
}
