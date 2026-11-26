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

/// Whether xmem holds the page as a filler page.
fn is_filler(page: usize) -> bool {
    let physical = physical();
    let key = physical.pages.get(&page).expect("known page").key(0);
    physical.fillers.contains_key(&key)
}

/// A fresh page shown at many places, one at a time or twice in one
/// scatter, is zeros because nothing has filled it yet: it stays a view
/// at each place, so that what comes into it later shows everywhere
/// (runs vk17 and vk35 read zeros through stand-ins of such pages).
#[test]
fn pages_shown_at_many_places_stay_views() {
    let pages = allocate(1);
    let a = XMemVirtualAlloc(std::ptr::null_mut(), 20 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
    assert!(!a.is_null());
    let at = |page: usize| a.wrapping_byte_add(page * PAGE);
    for k in (0..18).step_by(2) {
        assert_eq!(XMemMapPhysicalPages(at(k), 1, pages.as_ptr()), at(k));
    }
    let addresses = [at(1), at(5)];
    assert_eq!(XMemMapPhysicalPagesScatter(addresses.as_ptr(), 2, [pages[0]; 2].as_ptr()), 1);
    assert!(!is_filler(pages[0]));
    assert!(physical().views.range(a as usize..at(20) as usize).all(|(_, m)| !m.fill));
    // Written through one place, read through all the others.
    // SAFETY: mapped read-write.
    unsafe { at(16).cast::<u8>().write(6) };
    for k in [0, 1, 5, 14] {
        // SAFETY: mapped.
        assert_eq!(unsafe { at(k).cast::<u8>().read() }, 6);
    }
    assert_eq!(XMemMapPhysicalPages(a, 20, std::ptr::null()), a);
    assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
}

/// A filler page is forgotten once nothing shows it, or once freed; and
/// while a DirectStorage request writes a page, it is never taken for one.
#[test]
fn fillers_are_forgotten_and_never_targeted() {
    let pages = allocate(1);
    let a = XMemVirtualAlloc(std::ptr::null_mut(), 4 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
    assert!(!a.is_null());
    let list = [pages[0]; 4];
    assert_eq!(XMemMapPhysicalPages(a, 4, list.as_ptr()), a);
    assert!(is_filler(pages[0]));
    let said = describe(a as usize);
    assert!(said.as_deref().is_some_and(|d| d.contains("stand-in for a filler page")), "{said:?}");
    assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
    assert!(!is_filler(pages[0]), "shown nowhere");
    // Targeted by a request: views, though zeros and repeated.
    let target = target_pages(&pages, 0, 16);
    assert_eq!(XMemMapPhysicalPages(a, 4, list.as_ptr()), a);
    assert!(!is_filler(pages[0]));
    assert_eq!(views_at(a, 4), vec![1, 1, 1, 1]);
    drop(target);
    assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
    // A filler page under a memory destination becomes views before the
    // bytes come.
    assert_eq!(XMemMapPhysicalPages(a, 4, list.as_ptr()), a);
    assert!(is_filler(pages[0]));
    let target = target_memory(a as usize + 8, 8);
    assert!(!is_filler(pages[0]));
    assert_eq!(views_at(a, 4), vec![1, 1, 1, 1]);
    drop(target);
    assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
    // Freed: forgotten.
    assert_eq!(XMemMapPhysicalPages(a, 4, list.as_ptr()), a);
    let key = physical().pages.get(&pages[0]).expect("known page").key(0);
    assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    assert!(!physical().fillers.contains_key(&key));
    assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
}
