//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use victory_gpu::mapstat::Site;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // Unified memory (`victory_gpu::unified`, Proton): a region of our
    // choosing may become a host heap the GPU reads in place, checked by a
    // canary page of ours right after the title's bytes.
    let unified = constrained && kind & MEM_RESERVE_PLACEHOLDER == 0 && on_wine() && victory_gpu::unified_on();
    let title_bytes = size.next_multiple_of(PAGE);
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            if unified { title_bytes + PAGE } else { size },
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    if constrained && !at.is_null() {
        RESERVED.write().unwrap_or_else(|e| e.into_inner()).insert(at as usize, size);
        install_fault_handler();
    }
    // A reservation physical pages are mapped into, page by page: the GPU
    // reaches it by directory (`victory_gpu::memory`).
    if placeholder && !at.is_null() && victory_gpu::unified_on() {
        victory_gpu::title_page_reservation(at as usize, size.next_multiple_of(PAGE));
    }
    if unified && !at.is_null() {
        let canary = (at as usize + title_bytes) as *const c_void;
        // SAFETY: the last page of the reservation just made, ours.
        if !unsafe { VirtualAlloc(canary, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
            victory_gpu::title_region(at as usize, title_bytes);
        }
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            memory::free(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

struct Section {
    handle: HANDLE,
    /// The title's bytes in it (a canary page of unified memory may follow).
    bytes: usize,
    /// One view of all the title's bytes, for victory's own writes into
    /// pages (DirectStorage): mapped at the first such write and kept
    /// until the section goes, so that a write costs no map and no unmap
    /// (each a wineserver request that walks every view of the process).
    /// 0 until then.
    whole: AtomicUsize,
}

/// Serialises the mapping of [`Section::whole`].
static WHOLE: Mutex<()> = Mutex::new(());

// SAFETY: a section handle may be used from any thread; the view is plain
// memory.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    fn new(handle: HANDLE, bytes: usize) -> Section {
        Section { handle, bytes, whole: AtomicUsize::new(0) }
    }

    /// The address of the view of all the title's bytes, mapping it the
    /// first time.
    fn whole(&self) -> Option<usize> {
        let at = self.whole.load(Ordering::Acquire);
        if at != 0 {
            return Some(at);
        }
        let _one = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let at = self.whole.load(Ordering::Acquire);
        if at != 0 {
            return Some(at);
        }
        // Above every address a title may be given (46 bits at most,
        // `address_range`), so that the view takes none of its room.
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 46) as *mut c_void,
            HighestEndingAddress: std::ptr::null_mut(),
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: our section, mapped whole; the parameter points at
        // `requirements`, which outlives the call. The view is ours until
        // the section goes.
        let mut at = unsafe {
            MapViewOfFile3(self.handle, GetCurrentProcess(), std::ptr::null(), 0, self.bytes, 0, PAGE_READWRITE, &mut parameter, 1)
        }
        .Value as usize;
        if at == 0 {
            use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
            // SAFETY: as above, anywhere.
            at = unsafe { MapViewOfFile(self.handle, FILE_MAP_WRITE, 0, 0, self.bytes) }.Value as usize;
        }
        if at == 0 {
            tracing::warn!(bytes = self.bytes, error = %std::io::Error::last_os_error(), "no view of a physical-page section");
            return None;
        }
        victory_gpu::mapstat::mapped(Site::Section, self.bytes);
        self.whole.store(at, Ordering::Release);
        Some(at)
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        let at = *self.whole.get_mut();
        if at != 0 {
            use windows_sys::Win32::System::Memory::UnmapViewOfFile;
            // SAFETY: our view; no window on it outlives the section (each
            // holds the section).
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void }) };
            victory_gpu::mapstat::unmapped(Site::Section, self.bytes);
        }
        victory_gpu::title_section_gone(self.handle as usize);
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// What is mapped, by address: views, and stand-ins for filler pages.
    views: BTreeMap<usize, Mapped>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
    /// Filler pages by their key ([`Slice::key`]): pages of zeros the title
    /// shows at many places (in the unused parts of its tiled resources),
    /// shown there by stand-ins with no view ([`fill`]).
    fillers: HashMap<(usize, usize), Slice>,
    /// Pages found holding data when a request showed them at several
    /// places: views, as any page.
    not_fillers: std::collections::HashSet<(usize, usize)>,
    /// The pages of GPU views of the pool, by address: the page shown
    /// there ([`Slice::key`]); no view shows them ([`map_alias`]).
    aliased: HashMap<usize, (usize, usize)>,
}

/// What is mapped at an address: `len` bytes showing the slices from
/// `slice` on, in a view; or, for a stand-in (`fill`), the filler page
/// `slice` at every page, as zeroed memory of ours.
#[derive(Clone)]
struct Mapped {
    len: usize,
    slice: Slice,
    fill: bool,
}

impl Mapped {
    /// The index of the slice shown at `addr`, for what is mapped at `base`.
    fn index_at(&self, base: usize, addr: usize) -> usize {
        if self.fill { self.slice.index } else { self.slice.index + (addr - base) / PAGE }
    }
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    if n.checked_mul(PAGE).is_none_or(|s| s == 0) {
        return 0;
    }
    // Unified memory (`victory_gpu::unified`, Proton): each section may
    // become a host heap the GPU reads in place. Sections of at most
    // `UNIFIED_SECTION` pages, so that opening one pins only that much of
    // the PC's memory, each with a canary page of ours after the title's.
    let unified = on_wine() && victory_gpu::unified_on();
    let chunk = if unified { UNIFIED_SECTION } else { n };
    let mut sections = Vec::new();
    let mut done = 0;
    while done < n {
        let pages_here = chunk.min(n - done);
        let bytes = pages_here * PAGE + if unified { PAGE } else { 0 };
        // A pagefile-backed section, its pages committed when first mapped
        // or written (`commit`): the title asks for 6.1 GiB at once, which
        // a busy host often cannot commit (error 1450).
        // SAFETY: plain section creation.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                // Under Wine a committed section is a sparse file, free until
                // touched, and committing a reserved one page by page costs a
                // wineserver request each that scans every range committed so
                // far: commit it whole there.
                PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
                (bytes as u64 >> 32) as u32,
                bytes as u32,
                std::ptr::null(),
            )
        };
        if handle.is_null() {
            let error = std::io::Error::last_os_error();
            tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
            // The sections made so far close with `sections`.
            // SAFETY: checked non-null.
            unsafe { count.write(0) };
            return 0;
        }
        if unified {
            victory_gpu::title_section(handle as usize, (pages_here * PAGE) as u64);
        }
        sections.push((Arc::new(Section::new(handle, pages_here * PAGE)), pages_here));
        done += pages_here;
    }
    tracing::debug!(
        n,
        sections = sections.len(),
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    let mut k = 0;
    for (section, pages_here) in &sections {
        for index in 0..*pages_here {
            let slice = Slice {
                section: section.clone(),
                index,
            };
            let number = first + k * FRAMES;
            physical.pages.insert(number, slice);
            // SAFETY: the title passes room for `n` page numbers.
            unsafe { pages.add(k).write(number) };
            k += 1;
        }
    }
    1
}

/// Pages per section with unified memory: 512 MiB.
const UNIFIED_SECTION: usize = 8192;

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        physical.pages.remove(&unsafe { *pages.add(i) });
    }
    1
}

/// Consecutive slices of one section, committed and writable, in the view
/// of the whole section that victory keeps ([`Section::whole`]): no view
/// is mapped or unmapped for it.
struct View {
    /// Keeps the section, and so its view, alive.
    _section: Arc<Section>,
    at: usize,
    len: usize,
}

impl View {
    /// `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        let len = n * PAGE;
        if (slice.index + n) * PAGE > slice.section.bytes {
            return None;
        }
        let at = slice.section.whole()? + slice.index * PAGE;
        commit(at, len).then(|| View { _section: slice.section.clone(), at, len })
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the section's view holds these `len` committed read-write
        // bytes while the section lives, which `self` ensures; other views
        // of the slices only alias them, as the console's do.
        unsafe { std::slice::from_raw_parts_mut(self.at as *mut u8, self.len) }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Through the view of each section that victory keeps ([`View`]), with
/// no map or unmap. False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    unfill_written(&runs);
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    unfill_written(&runs);
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is the GPU side's to say). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

/// What a range of title memory is to show, a run at a time: from `at`,
/// `bytes` showing the slices from `slice` on, or `slice` at every page
/// (`repeat`: one page shown at many places, as titles show a filler page
/// in the unused parts of their tiled resources).
#[derive(Clone)]
struct Run {
    at: usize,
    bytes: usize,
    slice: Slice,
    repeat: bool,
}

impl Run {
    /// The index of the slice shown at `addr`.
    fn index_at(&self, addr: usize) -> usize {
        if self.repeat { self.slice.index } else { self.slice.index + (addr - self.at) / PAGE }
    }
}

/// Adds the page at `at`, showing `slice`, to `runs`: to the last run if
/// the address follows on and the slice follows on (or is the same page
/// again), else as a run of its own.
fn push_page(runs: &mut Vec<Run>, at: usize, slice: Slice) {
    if let Some(last) = runs.last_mut()
        && last.at + last.bytes == at
        && Arc::ptr_eq(&last.slice.section, &slice.section)
    {
        let single = last.bytes == PAGE;
        if !last.repeat && last.slice.index + last.bytes / PAGE == slice.index {
            last.bytes += PAGE;
            return;
        }
        if (last.repeat || single) && last.slice.index == slice.index {
            last.repeat = true;
            last.bytes += PAGE;
            return;
        }
    }
    runs.push(Run { at, bytes: PAGE, slice, repeat: false });
}

/// Places at which a page shows, from which it may be a filler page.
const FILLER_PLACES: usize = 8;

/// Pages a request shows at more than one place, or that show at
/// [`FILLER_PLACES`] places with this request, are filler candidates: one
/// that holds only zeros becomes a filler page, shown by stand-ins
/// ([`fill`]) rather than a view per place.
fn note_fillers(physical: &mut Physical, slices: &[Slice]) {
    let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
    for slice in slices {
        *seen.entry(slice.key(0)).or_default() += 1;
    }
    for slice in slices {
        let key = slice.key(0);
        let Some(&here) = seen.get(&key) else {
            continue;
        };
        let places = here + physical.shown.get(&key).map_or(0, Vec::len);
        if (here < 2 && places < FILLER_PLACES) || physical.fillers.contains_key(&key) || physical.not_fillers.contains(&key) {
            continue;
        }
        seen.remove(&key);
        if zero(slice) {
            tracing::info!(page = slice.index, "xmem: a page of zeros the title shows at many places: shown there by stand-ins, not views");
            physical.fillers.insert(key, slice.clone());
        } else {
            physical.not_fillers.insert(key);
        }
    }
}

/// Whether the page holds only zeros.
fn zero(slice: &Slice) -> bool {
    View::map(slice, 1).is_some_and(|mut v| v.bytes().iter().all(|&b| b == 0))
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    map_pages(address, n, pages, Site::Pages)
}

fn map_pages(address: *mut c_void, n: usize, pages: *const usize, site: Site) -> *mut c_void {
    let base = address as usize;
    let mut physical = physical();
    let mut runs = Vec::new();
    let mut known = true;
    if !pages.is_null() {
        // SAFETY: the title passes `n` page numbers.
        let pages = unsafe { std::slice::from_raw_parts(pages, n) };
        let mut slices = Vec::with_capacity(n);
        for page in pages {
            let Some(slice) = physical.pages.get(page).cloned() else {
                tracing::warn!(page, "XMemMapPhysicalPages: unknown page");
                known = false;
                break;
            };
            slices.push(slice);
        }
        note_fillers(&mut physical, &slices);
        for (k, slice) in slices.into_iter().enumerate() {
            push_page(&mut runs, base + k * PAGE, slice);
        }
    }
    // Until the pages are mapped again: `show` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`); a title thread
    // that does waits for the change to end (`on_fault`).
    let changing = victory_gpu::title_memory_remapping();
    unfill_dirty(&mut physical);
    let shown = show(&mut physical, base, n * PAGE, &runs, site);
    drop((changing, physical));
    victory_gpu::mapstat::tick();
    // Unmapped: what victory holds for that memory goes.
    if pages.is_null() && shown {
        victory_gpu::title_memory_released(base, n * PAGE);
    }
    if shown && known { address } else { std::ptr::null_mut() }
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear). Addresses that follow on
/// from each other are one request, so that consecutive pages there are
/// one view.
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
    let addresses = unsafe { std::slice::from_raw_parts(addresses, n) };
    // SAFETY: as above.
    let page = |i: usize| if pages.is_null() { 0 } else { unsafe { *pages.add(i) } };
    // A page this call shows at several addresses may be a filler page.
    {
        let mut physical = physical();
        let slices: Vec<Slice> = (0..n).filter_map(|i| physical.pages.get(&page(i)).cloned()).collect();
        note_fillers(&mut physical, &slices);
    }
    let mut i = 0;
    while i < n {
        let unmapping = page(i) == 0;
        let mut j = i + 1;
        while j < n
            && addresses[j] as usize == (addresses[j - 1] as usize).wrapping_add(PAGE)
            && (page(j) == 0) == unmapping
        {
            j += 1;
        }
        // SAFETY: pages `i..j` exist when `pages` is not null.
        let these = if unmapping { std::ptr::null() } else { unsafe { pages.add(i) } };
        if map_pages(addresses[i], j - i, these, Site::Scatter).is_null() {
            return 0;
        }
        i = j;
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`): the title maps
/// its pool whole into a pool view once, then has the GPU map pool pages
/// one by one into the GPU view of its pool, which its CPU never touches.
/// So no host view is made there (they were ~62k single-page views, half
/// the mapping churn): the GPU side keeps the pages in its directory, and
/// reads them for the CPU through each section's whole view
/// ([`Section::whole`], `victory_gpu::title_pages_aliased`). With
/// `repeat`, the one page at `from` at every page of `to`. False if a
/// source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64, repeat: bool) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let n = len / PAGE;
    let mut found = true;
    let mut pages: Vec<Option<victory_gpu::GpuPage>> = vec![None; n];
    {
        let mut physical = physical();
        let mut slices: Vec<Option<Slice>> = vec![None; n];
        if from != 0 {
            for (k, slot) in slices.iter_mut().enumerate() {
                let at = if repeat { from } else { from + k * PAGE };
                let Some(slice) = physical
                    .views
                    .range(..=at)
                    .next_back()
                    .filter(|(base, m)| at < *base + m.len)
                    .map(|(base, m)| Slice { index: m.index_at(*base, at), section: m.slice.section.clone() })
                else {
                    found = false;
                    break;
                };
                *slot = Some(slice);
            }
        }
        // Views an older path mapped there go (nothing else maps there).
        if physical.views.range(..to + len).next_back().is_some_and(|(base, m)| base + m.len > to) {
            let changing = victory_gpu::title_memory_remapping();
            unfill_dirty(&mut physical);
            show(&mut physical, to, len, &[], Site::Alias);
            drop(changing);
        }
        for (k, slice) in slices.iter().enumerate() {
            let at = to + k * PAGE;
            // Where each page shows, for writes through our own views
            // (`shown_at`, DirectStorage).
            if let Some(old) = physical.aliased.remove(&at)
                && let Some(addresses) = physical.shown.get_mut(&old)
            {
                addresses.retain(|&a| a != at);
                if addresses.is_empty() {
                    physical.shown.remove(&old);
                }
            }
            let Some(slice) = slice else {
                continue;
            };
            let Some(whole) = slice.section.whole() else {
                found = false;
                continue;
            };
            physical.aliased.insert(at, slice.key(0));
            physical.shown.entry(slice.key(0)).or_default().push(at);
            pages[k] = Some(victory_gpu::GpuPage {
                section: slice.section.handle as usize,
                offset: (slice.index * PAGE) as u64,
                cpu: whole + slice.index * PAGE,
            });
        }
    }
    victory_gpu::title_pages_aliased(to, &pages);
    found
}

/// Makes `[base, base + len)` show `runs` (which lie in it, in address
/// order) and nothing elsewhere, with as few host calls as that takes: a
/// view that already shows there what is asked stays as it is (the title
/// maps the same pages at the same place again, which costs the console
/// nothing), and only what differs is unmapped and mapped. Under Proton
/// each map and unmap is a wineserver request that walks every view of
/// the process. A filler page is shown by stand-ins, with no view. False
/// if the host refuses a map.
fn show(physical: &mut Physical, base: usize, len: usize, runs: &[Run], site: Site) -> bool {
    // No bytes: nothing to change. The search below would still find the
    // view holding `base`, and its range of views starting inside would run
    // backwards (which `BTreeMap` panics on).
    if len == 0 {
        return true;
    }
    install_fault_handler();
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside.
    let first = physical.views.range(..=base).next_back().filter(|(at, m)| *at + m.len > base).map(|(at, _)| *at);
    let hit: Vec<usize> = first.into_iter().chain(physical.views.range(base + 1..end).map(|(at, _)| *at)).collect();
    // Bytes of the range already showing what is asked, in address order.
    let mut kept: Vec<(usize, usize)> = Vec::new();
    let mut changed = false;
    for at in hit {
        let Some(m) = physical.views.get(&at).cloned() else {
            continue;
        };
        let (lo, hi) = (at.max(base), (at + m.len).min(end));
        let same = runs.iter().any(|r| {
            r.at <= lo
                && hi <= r.at + r.bytes
                && Arc::ptr_eq(&r.slice.section, &m.slice.section)
                && r.index_at(lo) == m.index_at(at, lo)
                && (hi - lo == PAGE || r.repeat == m.fill)
        });
        if same {
            kept.push((lo, hi));
        } else {
            unmap_view(physical, at, base, end, site);
            changed = true;
        }
    }
    for r in runs {
        let (mut at, stop) = (r.at, r.at + r.bytes);
        let mut gaps = Vec::new();
        for &(lo, hi) in &kept {
            if hi <= at || lo >= stop {
                continue;
            }
            if lo > at {
                gaps.push((at, lo));
            }
            at = at.max(hi);
        }
        if at < stop {
            gaps.push((at, stop));
        }
        for (lo, hi) in gaps {
            let slice = Slice { section: r.slice.section.clone(), index: r.index_at(lo) };
            let one_page = r.repeat || r.bytes == PAGE;
            let ok = if !one_page {
                map(physical, lo, hi - lo, slice, site)
            } else if physical.fillers.contains_key(&slice.key(0)) && fill(physical, lo, hi - lo, slice.clone(), site) {
                true
            } else {
                (lo..hi).step_by(PAGE).all(|p| map(physical, p, PAGE, slice.clone(), site))
            };
            if !ok {
                return false;
            }
            changed = true;
        }
    }
    // A page mapping is a write the title announces (textures over it
    // detile again, `victory_gpu::unified`), also where it maps what was
    // there already; nothing else changed there.
    for (lo, hi) in kept {
        victory_gpu::title_memory_written(lo, hi - lo, true);
    }
    if !changed {
        victory_gpu::mapstat::kept(site);
    }
    true
}

/// Takes away what is mapped at `at` (a view, back to a placeholder, or a
/// stand-in), and shows again what of it lies outside `[base, end)`.
fn unmap_view(physical: &mut Physical, at: usize, base: usize, end: usize, site: Site) {
    let Some(m) = physical.views.remove(&at) else {
        return;
    };
    let size = m.len;
    if m.fill {
        victory_gpu::title_memory_remapped(at, size);
        fill_ranges().remove(&at);
        // SAFETY: a stand-in of ours; it becomes a placeholder again.
        unsafe { VirtualFree(at as *mut c_void, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
        victory_gpu::mapstat::unfilled(size);
    } else {
        for k in 0..size.div_ceil(PAGE) {
            let key = m.slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void }, MEM_PRESERVE_PLACEHOLDER)
        };
        victory_gpu::mapstat::unmapped(site, size);
    }
    let again = |physical: &mut Physical, lo: usize, hi: usize| {
        let slice = Slice { section: m.slice.section.clone(), index: m.index_at(at, lo) };
        if !(m.fill && fill(physical, lo, hi - lo, slice.clone(), Site::Remnant)) {
            if m.fill {
                (lo..hi).step_by(PAGE).all(|p| map(physical, p, PAGE, slice.clone(), Site::Remnant));
            } else {
                map(physical, lo, hi - lo, slice, Site::Remnant);
            }
        }
    };
    if at < base {
        again(physical, at, base);
    }
    if at + size > end {
        again(physical, end, at + size);
    }
}

/// The host's flag that joins adjacent placeholders (not in windows-sys).
const MEM_COALESCE_PLACEHOLDERS: u32 = 1;

/// Makes `[base, base + len)` exactly one placeholder: the placeholders it
/// lies in are joined if it spans more than one (views unmapped one by one
/// leave one each), and what lies outside it is split off. Placeholders
/// change in the process alone (no wineserver request under Proton).
fn placeholder(base: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_PRIVATE, VirtualQuery};
    let query = |at: usize| {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        let n = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        // A free placeholder: reserved and private (a view is mapped memory).
        (n != 0 && info.State == MEM_RESERVE && info.Type == MEM_PRIVATE).then_some(info)
    };
    let end = base + len;
    let Some(info) = query(base) else {
        return false;
    };
    let start = info.AllocationBase as usize;
    let mut stop = info.BaseAddress as usize + info.RegionSize;
    let mut pieces = 1;
    while stop < end {
        match query(stop) {
            Some(next) if next.AllocationBase as usize == stop => {
                stop = next.BaseAddress as usize + next.RegionSize;
                pieces += 1;
            }
            _ => return false,
        }
    }
    if pieces > 1 {
        // SAFETY: adjacent free placeholders, joined into one.
        if unsafe { VirtualFree(start as *mut c_void, stop - start, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) } == 0 {
            return false;
        }
        victory_gpu::mapstat::merged();
    }
    if start != base || stop != end {
        // SAFETY: splits the placeholder around the range.
        if unsafe { VirtualFree(base as *mut c_void, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) } == 0 {
            return false;
        }
        victory_gpu::mapstat::split();
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must be free placeholders.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice, site: Site) -> bool {
    let view = if placeholder(base, len) {
        // SAFETY: maps the section in place of the placeholder just made.
        unsafe {
            MapViewOfFile3(
                slice.section.handle,
                GetCurrentProcess(),
                base as *const c_void,
                (slice.index * PAGE) as u64,
                len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        }
    } else {
        MEMORY_MAPPED_VIEW_ADDRESS { Value: std::ptr::null_mut() }
    };
    if view.Value as usize != base {
        victory_gpu::mapstat::failed();
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    victory_gpu::mapstat::mapped(site, len);
    if !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        victory_gpu::mapstat::unmapped(site, len);
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(base, len, slice.section.handle as usize, (slice.index * PAGE) as u64);
    if (0..len.div_ceil(PAGE)).any(|k| physical.shown.contains_key(&slice.key(k))) {
        victory_gpu::mapstat::aliased();
    }
    for k in 0..len.div_ceil(PAGE) {
        physical.shown.entry(slice.key(k)).or_default().push(base + k * PAGE);
    }
    physical.views.insert(base, Mapped { len, slice, fill: false });
    true
}

/// Shows the filler page `slice` at every page of `[base, base + len)`,
/// which must be free placeholders, by a stand-in: zeroed memory of ours,
/// read-only, with no view (the title reads its filler pages and never
/// writes them; a write is let through and counted, `on_fault`). False if
/// the host refuses.
fn fill(physical: &mut Physical, base: usize, len: usize, slice: Slice, site: Site) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    // SAFETY: private memory in place of the placeholder just made.
    let ok = placeholder(base, len)
        && unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                base as *const c_void,
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READONLY,
                std::ptr::null_mut(),
                0,
            )
        } as usize
            == base;
    if !ok {
        tracing::warn!(address = format_args!("{base:#x}"), len, error = %std::io::Error::last_os_error(), "a stand-in for a filler page failed; views instead");
        return false;
    }
    victory_gpu::mapstat::filled(site, len);
    victory_gpu::title_memory_remapped(base, len);
    fill_ranges().insert(base, len);
    physical.views.insert(base, Mapped { len, slice, fill: true });
    true
}

/// Filler pages found written (`check_fillers`), to be shown by views from
/// the next mapping on.
static DIRTY: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Every 10 s (the views line): a filler page that no longer holds only
/// zeros was written after all, and its stand-ins show what it held
/// before. Said aloud, and its stand-ins become views at the next mapping
/// change. Never waits for the page table (the line comes from wherever).
fn check_fillers() {
    let fillers: Vec<Slice> = match PHYSICAL.try_lock() {
        Ok(physical) => physical.fillers.values().cloned().collect(),
        Err(_) => return,
    };
    for slice in fillers {
        if !zero(&slice) {
            tracing::warn!(page = slice.index, "xmem: a filler page was written; its stand-ins become views");
            DIRTY.lock().unwrap_or_else(|e| e.into_inner()).push(slice.key(0));
        }
    }
}

/// Stand-ins of filler pages found written become views. With the change
/// lock held.
fn unfill_dirty(physical: &mut Physical) {
    let dirty = std::mem::take(&mut *DIRTY.lock().unwrap_or_else(|e| e.into_inner()));
    for key in dirty {
        unfill(physical, key);
    }
}

/// The filler page `key` is one no longer: its stand-ins become views.
/// With the change lock held.
fn unfill(physical: &mut Physical, key: (usize, usize)) {
    if physical.fillers.remove(&key).is_none() {
        return;
    }
    physical.not_fillers.insert(key);
    let stand_ins: Vec<usize> = physical.views.iter().filter(|(_, m)| m.fill && m.slice.key(0) == key).map(|(&at, _)| at).collect();
    for at in stand_ins {
        let Some(m) = physical.views.get(&at).cloned() else {
            continue;
        };
        unmap_view(physical, at, at, at + m.len, Site::Remnant);
        for p in (at..at + m.len).step_by(PAGE) {
            map(physical, p, PAGE, m.slice.clone(), Site::Remnant);
        }
    }
}

/// Before victory writes into pages (DirectStorage): a filler page among
/// them is one no longer, and its stand-ins become views first.
fn unfill_written(runs: &[(Slice, usize)]) {
    let mut physical = physical();
    let keys: Vec<(usize, usize)> = runs
        .iter()
        .flat_map(|(slice, n)| (0..*n).map(|k| slice.key(k)))
        .filter(|key| physical.fillers.contains_key(key))
        .collect();
    if keys.is_empty() {
        return;
    }
    let _changing = victory_gpu::title_memory_remapping();
    for key in keys {
        tracing::warn!(page = key.1, "xmem: DirectStorage writes a filler page; its stand-ins become views");
        unfill(&mut physical, key);
    }
}

/// Stand-ins by address, and their bytes, for the fault handler (which
/// must not wait for the page table).
fn fill_ranges() -> std::sync::RwLockWriteGuard<'static, BTreeMap<usize, usize>> {
    FILL_RANGES.write().unwrap_or_else(|e| e.into_inner())
}

static FILL_RANGES: std::sync::RwLock<BTreeMap<usize, usize>> = std::sync::RwLock::new(BTreeMap::new());

/// Installs [`on_fault`] once, ahead of the crash reporter's handler.
fn install_fault_handler() {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        victory_gpu::mapstat::on_tick(check_fillers);
        // SAFETY: the handler touches only its own state and protections.
        unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
    });
}

/// Access violations in title memory that the runtime caused:
/// - a title thread touched a page in the moment its view was away to be
///   mapped again (the runtime cannot change a view in one step, and an
///   unmap of part of a view takes all of it away): it waits for the change
///   to end and carries on, as on the console, which changes its page
///   table at once. In memory xmem reserved for the title, a page not
///   there is waited for up to a second whatever the lock says (the
///   bookkeeping decides, not a flag), and found there, the thread goes on;
/// - the title wrote into a stand-in for a filler page: the page is made
///   writable and the write counted (it does not reach the filler page's
///   other places).
unsafe extern "system" fn on_fault(info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> i32 {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH};
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS, VirtualProtect, VirtualQuery};
    // SAFETY: the system passes the exception being dispatched.
    let r = unsafe { &*(*info).ExceptionRecord };
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (kind, at) = (r.ExceptionInformation[0], r.ExceptionInformation[1]);
    const READ: usize = 0;
    const WRITE: usize = 1;
    if kind != READ && kind != WRITE {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // A change under way ends first (stand-ins change only during one).
    let holder = victory_gpu::mapping_lock_held();
    if !holder && !victory_gpu::remapping_over(2000) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if kind == WRITE {
        let in_fill = FILL_RANGES
            .read()
            .is_ok_and(|f| f.range(..=at).next_back().is_some_and(|(&base, &len)| at < base + len));
        let mut old = 0;
        // SAFETY: a page of a stand-in of ours.
        if in_fill && unsafe { VirtualProtect((at & !(PAGE - 1)) as *const c_void, PAGE, PAGE_READWRITE, &mut old) } != 0 {
            victory_gpu::mapstat::filler_written();
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if holder {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Is the page there now? In memory xmem reserved for the title (its
    // reservations for physical pages, its regions), a page not there may
    // be one a change is about to show, whichever thread makes it: wait for
    // it a while. Elsewhere, once.
    let fine = |at: usize| {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        let n = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        let protect = info.Protect & 0xff;
        n != 0
            && info.State == MEM_COMMIT
            && info.Protect & PAGE_GUARD == 0
            && protect != PAGE_NOACCESS
            && (kind == READ || protect == PAGE_READWRITE || protect == PAGE_EXECUTE_READWRITE)
    };
    // Also any reserved page of title memory (below 1 TiB): a wait there
    // only delays a fault that is the title's own by a second.
    let reserved_title_page = at < 1 << 40 && {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query into our structure.
        (unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) }) != 0
            && info.State == MEM_RESERVE
    };
    let patience = if reserved_title_page || reserved_by_xmem(at) { 1000 } else { 0 };
    let start = std::time::Instant::now();
    loop {
        if fine(at) {
            victory_gpu::mapstat::waited(at, start.elapsed().as_millis() as u64);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if start.elapsed().as_millis() >= patience {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
        // A change that began meanwhile ends first.
        if !victory_gpu::remapping_over(1000) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
    }
}

/// What xmem reserved for the title, by address: `XMemVirtualAlloc`
/// regions and reservations for physical pages, and their bytes. Kept for
/// the fault handler, which must not wait for the page table.
static RESERVED: std::sync::RwLock<BTreeMap<usize, usize>> = std::sync::RwLock::new(BTreeMap::new());

/// Whether `at` lies in memory xmem reserved for the title.
fn reserved_by_xmem(at: usize) -> bool {
    RESERVED.read().is_ok_and(|r| r.range(..=at).next_back().is_some_and(|(&base, &len)| at < base + len))
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is all the process has committed, the layer's own
    // memory and the card's included (27 GB in run 60's gameplay against a
    // limit of 13.5 GB). A title that budgets its streaming from these two
    // figures believes it is far over its limit.
    // shortcut: the figure is not yet the title's own memory alone (its
    // heaps and the pages it mapped); count those if this line shows the
    // title asking.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

/// The title's `VirtualFree` (it imports it from
/// `api-ms-win-core-memory-l1-1-0.dll` and frees its `XMemVirtualAlloc`
/// regions with it): the host's, then what victory holds for that memory
/// goes (`victory_gpu::title_memory_released`).
pub mod memory {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    use windows_sys::core::BOOL;

    use crate::{HleModule, hle_export};

    const MEM_DECOMMIT: u32 = 0x4000;
    const MEM_RELEASE: u32 = 0x8000;
    /// `MEM_COALESCE_PLACEHOLDERS`, `MEM_PRESERVE_PLACEHOLDER`: placeholder
    /// work, not a free.
    const PLACEHOLDER: u32 = 0x1 | 0x2;

    #[allow(non_snake_case)]
    pub extern "system" fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> BOOL {
        let span = freed(address as usize, size, kind);
        // SAFETY: the title's own arguments.
        let ok = unsafe { windows_sys::Win32::System::Memory::VirtualFree(address, size, kind) };
        if ok != 0 && size == 0 && kind & MEM_RELEASE != 0 {
            super::RESERVED.write().unwrap_or_else(|e| e.into_inner()).remove(&(address as usize));
        }
        if ok != 0
            && let Some((at, len)) = span
        {
            // SAFETY: plain calls; the title sees the error the host left.
            let error = unsafe { GetLastError() };
            victory_gpu::title_memory_released(at, len);
            unsafe { SetLastError(error) };
        }
        ok
    }

    /// Victory's frees of title memory (`XMemFreeDefault`) go the same way.
    pub(crate) fn free(address: *mut c_void, size: usize, kind: u32) -> BOOL {
        VirtualFree(address, size, kind)
    }

    /// The memory a `VirtualFree` of the title's frees or decommits: below
    /// 1 TiB (title memory), and for a size of 0 the whole allocation that
    /// starts at `at`, asked before it goes.
    fn freed(at: usize, size: usize, kind: u32) -> Option<(usize, usize)> {
        if at == 0 || at as u64 >= 1 << 40 || kind & (MEM_DECOMMIT | MEM_RELEASE) == 0 || kind & PLACEHOLDER != 0 {
            return None;
        }
        if size != 0 {
            return Some((at, size.next_multiple_of(4096)));
        }
        let mut end = at;
        loop {
            // SAFETY: plain data, filled by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a query into our structure.
            if unsafe { VirtualQuery(end as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0
                || info.AllocationBase as usize != at
                || info.State == MEM_FREE
                || info.RegionSize == 0
            {
                break;
            }
            end = info.BaseAddress as usize + info.RegionSize;
        }
        (end > at).then(|| (at, end - at))
    }

    pub(crate) fn module() -> HleModule {
        HleModule {
            name: "memory",
            dlls: &["kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-*"],
            exports: vec![hle_export!(VirtualFree)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`,
        // with no host view there (the GPU side keeps the page); writes
        // into the page are told at `b` too.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64, false));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(shown_at(&pages[2..], 0, 1).contains(&(b as usize, 1)));
        assert!(map_alias(b as u64, 0, PAGE as u64, false));
        assert!(!shown_at(&pages[2..], 0, 1).contains(&(b as usize, 1)));
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64, false));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0, false));
            assert!(map_alias(at as u64, 0, 0, false));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.len);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// The views at `[at, at + n pages)`: their lengths in pages, in order.
    fn views_at(at: *mut c_void, n: usize) -> Vec<usize> {
        let at = at as usize;
        physical().views.range(at..at + n * PAGE).map(|(_, m)| m.len / PAGE).collect()
    }

    /// Few views, and no change where nothing changes: under Proton every
    /// map and unmap is a wineserver request that walks all the views.
    #[test]
    fn mappings_take_few_views() {
        let pages = allocate(4);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        let (a, b, c) = (reserve(4), reserve(4), reserve(4));
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        assert_eq!(views_at(a, 4), vec![4]);
        // SAFETY: mapped read-write.
        unsafe { at(a, 3).cast::<u8>().write(3) };
        // The same pages again, whole or in part: the view stays.
        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(at(a, 1), 2, pages[1..].as_ptr()), at(a, 1));
        assert_eq!(views_at(a, 4), vec![4]);
        // A pool range of consecutive pages, aliased: one view.
        assert!(map_alias(b as u64, a as u64, 4 * PAGE as u64, false));
        assert_eq!(views_at(b, 4), vec![4]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 3).cast::<u8>().read() }, 3);
        // One page at every page: a view each, cut once from the view
        // there; the same again changes nothing.
        assert!(map_alias(b as u64, at(a, 3) as u64, 3 * PAGE as u64, true));
        assert_eq!(views_at(b, 4), vec![1, 1, 1, 1]);
        assert!(map_alias(b as u64, at(a, 3) as u64, 3 * PAGE as u64, true));
        assert_eq!(views_at(b, 4), vec![1, 1, 1, 1]);
        for page in 0..4 {
            // SAFETY: mapped read-write.
            assert_eq!(unsafe { at(b, page).cast::<u8>().read() }, 3);
        }
        // A run over what were four views maps as one: the placeholders
        // they leave are joined.
        assert_eq!(XMemMapPhysicalPages(b, 4, std::ptr::null()), b);
        assert_eq!(query(at(b, 2)).host.State, MEM_RESERVE);
        assert_eq!(XMemMapPhysicalPages(b, 4, pages.as_ptr()), b);
        assert_eq!(views_at(b, 4), vec![4]);
        // Scattered pages at addresses that follow on: one view.
        let addresses = [c, at(c, 1), at(c, 2)];
        assert_eq!(XMemMapPhysicalPagesScatter(addresses.as_ptr(), 3, pages.as_ptr()), 1);
        assert_eq!(views_at(c, 4), vec![3]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(c, 2).cast::<u8>().read() }, 0);
        for p in [a, b, c] {
            assert_eq!(XMemMapPhysicalPages(p, 4, std::ptr::null()), p);
        }
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// A page of zeros shown at many places is shown by one stand-in, no
    /// view per place; once written, its stand-ins become views of it.
    #[test]
    fn filler_pages_are_shown_by_stand_ins() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 8 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!a.is_null());
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        // The filler page at six places, a real page after them.
        let mut list = [pages[0]; 7];
        list[6] = pages[1];
        assert_eq!(XMemMapPhysicalPages(a, 7, list.as_ptr()), a);
        let stand_in = physical().views.get(&(a as usize)).map(|m| (m.len, m.fill));
        assert_eq!(stand_in, Some((6 * PAGE, true)));
        assert_eq!(views_at(a, 8), vec![6, 1]);
        // SAFETY: shown, readable.
        assert_eq!(unsafe { at(5).cast::<u8>().read() }, 0);
        // The same again changes nothing; unmapping part of it keeps the rest.
        assert_eq!(XMemMapPhysicalPages(a, 7, list.as_ptr()), a);
        assert_eq!(views_at(a, 8), vec![6, 1]);
        assert_eq!(XMemMapPhysicalPages(at(2), 1, std::ptr::null()), at(2));
        assert_eq!(views_at(a, 8), vec![2, 3, 1]);
        assert_eq!(query(at(2)).host.State, MEM_RESERVE);
        // DirectStorage writes the filler page: the stand-ins become views
        // that show what it wrote.
        assert!(write_physical(&pages[..1], 0, &[4]));
        assert_eq!(views_at(a, 8), vec![1, 1, 1, 1, 1, 1]);
        for page in [0, 1, 3, 4, 5] {
            // SAFETY: mapped.
            assert_eq!(unsafe { at(page).cast::<u8>().read() }, 4);
        }
        assert_eq!(XMemMapPhysicalPages(a, 7, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }
}
