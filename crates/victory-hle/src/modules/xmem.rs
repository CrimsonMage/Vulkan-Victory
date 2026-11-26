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

mod fillers;
mod hooks;
mod places;
use fillers::*;
pub use hooks::{XMemAlloc, XMemFree, XMemSetAllocationHooks};
#[cfg(feature = "diagnostics")]
pub(crate) use places::check_aliases;
use places::describe;
pub(crate) use places::{target_memory, target_pages};

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
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder && kind & MEM_REPLACE_PLACEHOLDER != 0 {
        // Part of a placeholder the title split: it stays one.
        return base;
    }
    let protect = match protect & HOST_PROTECTION {
        p if p != 0 && !placeholder => p,
        _ => PAGE_NOACCESS,
    };
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M { 2 * MIB } else { 0 },
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    let kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    let kind = if placeholder { kind | MEM_RESERVE | MEM_RESERVE_PLACEHOLDER } else { kind };
    // Address requirements apply to reservations at an address of our
    // choice; those not of placeholders are regions the GPU side reaches.
    let constrained = base.is_null();
    let region = constrained && kind & MEM_RESERVE_PLACEHOLDER == 0;
    let (kind, bytes) = reservation(kind, size, constrained, region);
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            bytes,
            kind,
            protect,
            if constrained { &mut parameter } else { std::ptr::null_mut() },
            constrained.into(),
        )
    };
    if constrained {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
        if !at.is_null() {
            reserved(at as usize, size, placeholder, region);
        }
    }
    at
}

/// How memory is reserved, as (kind, bytes); always at an address of our
/// choice. A region of ours is the title's 64 KiB pages.
fn reservation(kind: u32, size: usize, constrained: bool, region: bool) -> (u32, usize) {
    let kind = if constrained { kind | MEM_RESERVE } else { kind };
    if !region {
        return (kind, size);
    }
    (kind, size.next_multiple_of(PAGE))
}

/// A reservation made at an address of our choice: noted for the fault
/// handler, and for the GPU side a reservation physical pages are mapped
/// into page by page (the pool's GPU view), or a region.
fn reserved(at: usize, size: usize, placeholder: bool, region: bool) {
    RESERVED.write().unwrap_or_else(|e| e.into_inner()).insert(at, size);
    install_fault_handler();
    let title_bytes = size.next_multiple_of(PAGE);
    if placeholder {
        victory_gpu::title_page_reservation(at, title_bytes);
    }
    if region {
        victory_gpu::title_region(at, title_bytes);
    }
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
    let _owner = victory_core::heap::enter(victory_core::heap::Owner::Title);
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
    /// Never reused, unlike the section's address: the keys of its pages
    /// ([`Slice::key`]) name it alone, so that no entry of a section gone
    /// applies to a new one at the same address.
    id: usize,
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
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        Section { handle, id: NEXT.fetch_add(1, Ordering::Relaxed), bytes, whole: AtomicUsize::new(0) }
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
    /// The key in `Physical::shown` of the page `k` pages after this one:
    /// its section's id and its index there.
    fn key(&self, k: usize) -> (usize, usize) {
        (self.section.id, self.index + k)
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
    /// shows over and over (in the unused parts of its tiled resources),
    /// shown there by stand-ins with no view ([`fill`]). Forgotten when the
    /// page is freed or shown nowhere any more.
    fillers: HashMap<(usize, usize), Slice>,
    /// Pages found holding data when a request repeated them: views, as
    /// any page. Forgotten as fillers are.
    not_fillers: std::collections::HashSet<(usize, usize)>,
    /// Pages a DirectStorage request is writing now, and how many requests
    /// ([`Target`]): never made filler pages meanwhile (fresh pages are
    /// zeros until the request lands).
    targeted: HashMap<(usize, usize), usize>,
    /// Pages a change under way took away from somewhere
    /// ([`forget_unshown`]).
    unshown: Vec<(usize, usize)>,
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
    let Some(sections) = new_sections(n) else {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    };
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

/// Sections for `n` pages, with how many pages each holds; none if the
/// host refuses one. The GPU side reads sections in place: the Vulkan
/// backend imports them by host pointer. Sections of at most
/// `UNIFIED_SECTION` pages, so that importing one pins only that much of
/// the PC's memory.
fn new_sections(n: usize) -> Option<Vec<(Arc<Section>, usize)>> {
    let mut sections = Vec::new();
    let mut done = 0;
    while done < n {
        let pages_here = UNIFIED_SECTION.min(n - done);
        let bytes = pages_here * PAGE;
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
            return None;
        }
        victory_gpu::title_section(handle as usize, (pages_here * PAGE) as u64);
        sections.push((Arc::new(Section::new(handle, pages_here * PAGE)), pages_here));
        done += pages_here;
    }
    Some(sections)
}

/// Pages per section: 512 MiB.
const UNIFIED_SECTION: usize = 8192;

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
/// What xmem noted about a freed page goes with it: whether it is a filler
/// page, and where it shows (no write through victory's views can reach
/// it any more, so none is told anywhere).
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        let Some(slice) = physical.pages.remove(&unsafe { *pages.add(i) }) else {
            continue;
        };
        let key = slice.key(0);
        physical.fillers.remove(&key);
        physical.not_fillers.remove(&key);
        physical.shown.remove(&key);
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
        victory_gpu::title_memory_written(at, bytes);
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
        for (k, slice) in slices.into_iter().enumerate() {
            push_page(&mut runs, base + k * PAGE, slice);
        }
        note_fillers(&mut physical, &runs);
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
                forget_unshown(physical);
                return false;
            }
            changed = true;
        }
    }
    if !changed {
        victory_gpu::mapstat::kept(site);
    }
    forget_unshown(physical);
    true
}

/// Takes away what is mapped at `at` (a view, back to a placeholder, or a
/// stand-in), and shows again what of it lies outside `[base, end)`.
fn unmap_view(physical: &mut Physical, at: usize, base: usize, end: usize, site: Site) {
    let Some(m) = physical.views.remove(&at) else {
        return;
    };
    let size = m.len;
    // Out of the GPU side's map; what stays is shown again below.
    victory_gpu::title_memory_unmapped(at, size);
    if m.fill {
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
    // Looked at once the change is done ([`forget_unshown`]): the same
    // page may be shown elsewhere by it.
    let pages = if m.fill { 1 } else { size.div_ceil(PAGE) };
    physical.unshown.extend((0..pages).map(|k| m.slice.key(k)));
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

/// Installs [`on_fault`] once, ahead of the crash reporter's handler.
fn install_fault_handler() {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        victory_gpu::mapstat::on_tick(check_fillers);
        victory_loader::describe_title_memory_with(describe);
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
            hle_export!(XMemSetAllocationHooks),
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

    /// The console's `VirtualAlloc` flags of graphics and title memory
    /// (`MEM_GRAPHICS`, `MEM_TITLE`), which the host refuses.
    const CONSOLE_KINDS: u32 = 0x1000_0000 | 0x4000_0000;
    /// The console's GPU access bits of a protection (`PAGE_GRAPHICS_*`).
    const GRAPHICS_PROTECTION: u32 = 0x7_f800;

    /// The title's `VirtualAlloc`: with console flags it goes the XMem way
    /// (`XMemVirtualAlloc` strips them and places the memory as title
    /// memory); anything else is the host's, unchanged.
    #[allow(non_snake_case)]
    pub extern "system" fn VirtualAlloc(at: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void {
        if kind & CONSOLE_KINDS == 0 && protect & GRAPHICS_PROTECTION == 0 {
            // SAFETY: the title's own arguments.
            return unsafe { windows_sys::Win32::System::Memory::VirtualAlloc(at, size, kind, protect) };
        }
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| tracing::info!(kind, protect, "VirtualAlloc with console flags: the XMem way"));
        super::XMemVirtualAlloc(at, size, kind & !CONSOLE_KINDS, 0, protect)
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
            exports: vec![hle_export!(VirtualAlloc), hle_export!(VirtualFree)],
        }
    }
}

#[cfg(test)]
mod tests;
