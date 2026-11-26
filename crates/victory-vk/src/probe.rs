//! `victory vkprobe`: the Vulkan backend's self-test. Each step prints one
//! line, `[ok]`, `[FAIL]`, `[no]` (a step that tries something the plan
//! may use; its failure fails nothing) or `[info]`. Nothing of a title is
//! touched: the "title memory" is the probe's own, made the way `xmem`
//! makes the title's (a private region below 1 TiB; a committed section
//! mapped twice).

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use ash::vk;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
    MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile,
    VirtualAlloc, VirtualFree,
};

use crate::compute::{self, Push};
use crate::descriptors::{self, Fallback};
use crate::graphics;
use crate::host::{Host, Need, version};
use crate::memory::{Buffer, Imported, Place, TITLE_USAGE};
use crate::present;

const MIB: usize = 1 << 20;
const PAGE: u64 = 64 << 10;
/// Words each compute run reads (1 MiB).
const COUNT: u32 = 1 << 18;

/// Runs every step; `Ok(true)` when every required step passed.
pub fn run() -> Result<bool> {
    let mut p = Report::default();
    println!("vkprobe: Vulkan host self-test (docs/vulkan-host.md)");
    let host = match Host::new() {
        Ok(h) => h,
        Err(e) => {
            p.fail("device", format!("{e:#}"));
            return Ok(p.end());
        }
    };
    describe(&host, &mut p);
    // SAFETY: every object below is made and destroyed on this thread, and
    // the device is idle before anything a submission used is destroyed.
    let result = unsafe { steps(&host, &mut p) };
    if let Err(e) = result {
        p.fail("probe stopped", format!("{e:#}"));
    }
    Ok(p.end())
}

#[derive(Default)]
struct Report {
    passed: u32,
    failed: u32,
}

impl Report {
    fn ok(&mut self, step: &str, what: impl AsRef<str>) {
        self.passed += 1;
        println!("[ok]   {step}: {}", what.as_ref());
    }
    fn fail(&mut self, step: &str, what: impl AsRef<str>) {
        self.failed += 1;
        println!("[FAIL] {step}: {}", what.as_ref());
    }
    fn no(&mut self, step: &str, what: impl AsRef<str>) {
        println!("[no]   {step}: {}", what.as_ref());
    }
    fn info(&mut self, step: &str, what: impl AsRef<str>) {
        println!("[info] {step}: {}", what.as_ref());
    }
    /// A required step.
    fn step(&mut self, step: &str, r: Result<String>) -> bool {
        match r {
            Ok(s) => {
                self.ok(step, s);
                true
            }
            Err(e) => {
                self.fail(step, format!("{e:#}"));
                false
            }
        }
    }
    /// A step that tries something; its failure is a finding, not a failure.
    fn try_step(&mut self, step: &str, r: Result<String>) {
        match r {
            Ok(s) => println!("[ok]   {step}: {s}"),
            Err(e) => self.no(step, format!("{e:#}")),
        }
    }
    fn end(&self) -> bool {
        println!(
            "vkprobe: {} of {} required steps passed",
            self.passed,
            self.passed + self.failed
        );
        self.failed == 0
    }
}

fn describe(host: &Host, p: &mut Report) {
    let c = &host.caps;
    p.ok(
        "instance",
        format!(
            "loader {}; {}",
            version(host.instance_version),
            host.instance_extensions.join(", ")
        ),
    );
    let q = &host.queues;
    let queue = |q: Option<crate::host::Queue>| {
        q.map_or("none".to_owned(), |q| {
            format!(
                "family {} (x{}{})",
                q.family,
                q.family_count,
                if q.sparse { ", sparse" } else { "" }
            )
        })
    };
    p.ok(
        "device",
        format!(
            "{} ({:?}), {}, Vulkan {}; queues: graphics {}, compute {}, transfer {}",
            c.device_name,
            c.device_type,
            c.driver,
            version(c.api_version),
            queue(Some(q.graphics)),
            queue(q.compute),
            queue(q.transfer)
        ),
    );
    p.info(
        "limits",
        format!(
            "import alignment {:#x}, max buffer {} GiB, max allocation {} GiB, push constants {} B, waves {}-{}, timestamp {} ns",
            c.min_imported_host_pointer_alignment,
            c.max_buffer_size >> 30,
            c.max_memory_allocation_size >> 30,
            c.max_push_constants_size,
            c.min_subgroup_size,
            c.max_subgroup_size,
            c.timestamp_period
        ),
    );
    for (i, (heap, budget)) in host.heaps().iter().enumerate() {
        let local = heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        let budget = budget.map_or(String::new(), |(b, u)| {
            format!(", budget {} MiB, used {} MiB", b >> 20, u >> 20)
        });
        p.info(
            "memory heap",
            format!(
                "{i}: {} MiB {}{budget}",
                heap.size >> 20,
                if local { "card" } else { "PC" }
            ),
        );
    }
    for t in &host.memory.memory_types[..host.memory.memory_type_count as usize] {
        p.info(
            "memory type",
            format!("heap {}: {:?}", t.heap_index, t.property_flags),
        );
    }
    let present: Vec<String> = c
        .extensions
        .iter()
        .filter(|(_, has)| *has)
        .map(|(e, _)| e.name.to_string_lossy().into_owned())
        .collect();
    p.info("extensions present", present.join(", "));
    for (need, name, purpose) in c.missing() {
        let tag = match need {
            Need::Required => "required",
            Need::Wanted => "wanted",
            Need::Report => "later",
        };
        p.info("missing", format!("{name} ({tag}): {purpose}"));
    }
}

/// The probe's stand-ins for title memory.
struct TitleMemory {
    /// A private region, as `XMemVirtualAlloc` gives (64 MiB).
    region: *mut u8,
    region_len: usize,
    /// A second small region: a descriptor heap of the title's.
    heap: *mut u8,
    /// A committed section and two views of it, as physical pages and
    /// their aliases.
    section: HANDLE,
    view_a: *mut u8,
    view_b: *mut u8,
    section_len: usize,
}

impl TitleMemory {
    unsafe fn new() -> Result<TitleMemory> {
        unsafe {
            let region_len = 64 * MIB;
            // Below 1 TiB, as the title's memory must be (40-bit texture
            // descriptors); anywhere if that address is taken.
            let mut region = VirtualAlloc(
                0x40_0000_0000usize as *const _,
                region_len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
            if region.is_null() {
                region = VirtualAlloc(
                    std::ptr::null(),
                    region_len,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                );
            }
            ensure!(!region.is_null(), "VirtualAlloc of 64 MiB failed");
            let heap = VirtualAlloc(
                std::ptr::null(),
                MIB,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
            ensure!(!heap.is_null(), "VirtualAlloc of 1 MiB failed");
            let section_len = 32 * MIB;
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                section_len as u32,
                std::ptr::null(),
            );
            ensure!(!section.is_null(), "CreateFileMappingW of 32 MiB failed");
            let view_a = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, section_len).Value;
            let view_b = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, section_len).Value;
            ensure!(
                !view_a.is_null() && !view_b.is_null(),
                "MapViewOfFile failed"
            );
            Ok(TitleMemory {
                region: region.cast(),
                region_len,
                heap: heap.cast(),
                section,
                view_a: view_a.cast(),
                view_b: view_b.cast(),
                section_len,
            })
        }
    }

    fn words(at: *mut u8, offset: usize, count: usize) -> &'static mut [u32] {
        // SAFETY: callers pass committed ranges of the probe's memory that
        // outlive the probe's use of the slice.
        unsafe { std::slice::from_raw_parts_mut(at.add(offset).cast(), count) }
    }
}

impl Drop for TitleMemory {
    fn drop(&mut self) {
        // SAFETY: the probe's own mappings, unused by now.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.view_a.cast(),
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.view_b.cast(),
            });
            CloseHandle(self.section);
            VirtualFree(self.region.cast(), 0, MEM_RELEASE);
            VirtualFree(self.heap.cast(), 0, MEM_RELEASE);
        }
    }
}

fn pattern(i: usize, seed: u32) -> u32 {
    (i as u32).wrapping_mul(0x9e37_79b9) ^ seed
}

/// Where the result buffer's descriptor lives.
enum Binding {
    /// A descriptor buffer of the backend's own: its address and usage.
    InMemory {
        address: u64,
        usage: vk::BufferUsageFlags,
        offset: u64,
    },
    /// The fallback set.
    Set(vk::DescriptorSet),
}

/// Records and runs the probe program.
struct Runner<'a> {
    host: &'a Host,
    probe: compute::Probe,
    result: Buffer,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    timeline: vk::Semaphore,
    value: u64,
}

impl<'a> Runner<'a> {
    unsafe fn new(host: &'a Host, probe: compute::Probe, result: Buffer) -> Result<Runner<'a>> {
        let device = &host.device;
        unsafe {
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(host.queues.graphics.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let timeline = device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )?;
            Ok(Runner {
                host,
                probe,
                result,
                pool,
                cmd,
                timeline,
                value: 0,
            })
        }
    }

    /// Runs the program over `count` words; `before` records work ahead
    /// of the dispatch. Returns the milliseconds from submit to the
    /// timeline value.
    unsafe fn run(
        &mut self,
        binding: &Binding,
        push: Push,
        before: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<f64> {
        let device = &self.host.device;
        let cmd = self.cmd;
        unsafe {
            std::ptr::write_bytes(self.result.mapped, 0, self.result.size as usize);
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            before(device, cmd);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.probe.pipeline);
            match binding {
                Binding::InMemory {
                    address,
                    usage,
                    offset,
                } => {
                    let db = self
                        .host
                        .descriptor_buffer
                        .as_ref()
                        .context("descriptor buffer")?;
                    db.cmd_bind_descriptor_buffers(
                        cmd,
                        &[vk::DescriptorBufferBindingInfoEXT::default()
                            .address(*address)
                            .usage(*usage)],
                    );
                    db.cmd_set_descriptor_buffer_offsets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.probe.layout,
                        0,
                        &[0],
                        &[*offset],
                    );
                }
                Binding::Set(set) => device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.probe.layout,
                    0,
                    &[*set],
                    &[],
                ),
            }
            let bytes =
                std::slice::from_raw_parts((&raw const push).cast::<u8>(), size_of::<Push>());
            device.cmd_push_constants(
                cmd,
                self.probe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
            device.cmd_dispatch(cmd, push.count.div_ceil(compute::GROUP), 1, 1);
            let to_host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
            device.end_command_buffer(cmd)?;

            self.value += 1;
            let signals = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(self.value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = vk::SubmitInfo2::default()
                .command_buffer_infos(&cmds)
                .signal_semaphore_infos(&signals);
            let start = Instant::now();
            device.queue_submit2(
                self.host.queues.graphics.queue,
                &[submit],
                vk::Fence::null(),
            )?;
            let sems = [self.timeline];
            let values = [self.value];
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&sems)
                        .values(&values),
                    5_000_000_000,
                )
                .context("waiting for the timeline value")?;
            Ok(start.elapsed().as_secs_f64() * 1000.0)
        }
    }

    /// Checks the result buffer against `expect` (the words the program
    /// read) and the written-back words.
    fn check(&self, expect: impl Fn(usize) -> u32, written: &[u32], salt: u32) -> Result<()> {
        let out = TitleMemory::words(self.result.mapped, 0, COUNT as usize + 1);
        let mut sum = 0u32;
        for i in 0..COUNT as usize {
            let v = expect(i);
            sum = sum.wrapping_add(v);
            ensure!(
                out[1 + i] == v.wrapping_add(1),
                "word {i}: the program read {:#x} (through the descriptor: {:#x}), expected {v:#x}",
                out[1 + i].wrapping_sub(1),
                out[1 + i]
            );
            ensure!(
                written[i] == v ^ salt,
                "word {i}: written back {:#x}, expected {:#x}",
                written[i],
                v ^ salt
            );
        }
        ensure!(out[0] == sum, "sum {:#x}, expected {sum:#x}", out[0]);
        Ok(())
    }

    unsafe fn destroy(&self) {
        unsafe {
            let device = &self.host.device;
            device.destroy_semaphore(self.timeline, None);
            device.destroy_command_pool(self.pool, None);
            self.probe.destroy(self.host);
            self.result.destroy(self.host);
        }
    }
}

unsafe fn steps(host: &Host, p: &mut Report) -> Result<()> {
    // Presentation first: the steps after it try things a driver may not take.
    let res = present::Window::new("victory vkprobe", 960, 540).and_then(|w| {
        let r = present::clear_frames(host, &w, 120)?;
        Ok(format!(
            "{} frames of {}x{} {:?}, {:?}, {} images, {:.1} ms/frame{}",
            r.frames,
            r.extent.width,
            r.extent.height,
            r.format,
            r.mode,
            r.images,
            r.ms_per_frame,
            if r.suboptimal > 0 {
                format!(", {} suboptimal", r.suboptimal)
            } else {
                String::new()
            }
        ))
    });
    p.step("present", res);

    let memory = unsafe { TitleMemory::new() }?;

    // Title memory: a private region.
    let region_import = unsafe {
        Imported::new(
            host,
            memory.region.cast(),
            memory.region_len as u64,
            vk::BufferUsageFlags::empty(),
        )
    };
    let region_import = match region_import {
        Ok(i) => {
            let probe = i.title_base + 0x1234;
            p.ok(
                "title memory, private region",
                format!(
                    "{} MiB at title {:#x} imported (memory type {}); device address {:#x}; title {probe:#x} -> device {:#x}",
                    i.len >> 20,
                    i.title_base,
                    i.memory_type,
                    i.device_base,
                    i.device_address(probe).unwrap_or(0)
                ),
            );
            Some(i)
        }
        Err(e) => {
            p.fail("title memory, private region", format!("{e:#}"));
            None
        }
    };

    // Title memory: a section seen through two views; view A imported.
    let section_import = unsafe {
        Imported::new(
            host,
            memory.view_a.cast(),
            memory.section_len as u64,
            vk::BufferUsageFlags::empty(),
        )
    };
    let section_import = match section_import {
        Ok(i) => {
            p.ok(
                "title memory, section",
                format!(
                    "{} MiB section, views at {:#x} (imported) and {:#x}; device address {:#x}",
                    i.len >> 20,
                    memory.view_a as u64,
                    memory.view_b as u64,
                    i.device_base
                ),
            );
            Some(i)
        }
        Err(e) => {
            p.fail("title memory, section", format!("{e:#}"));
            None
        }
    };

    // Descriptors.
    let in_memory = host.caps.descriptor_buffer;
    let result_size = 4 + 4 * u64::from(COUNT);
    let result = Buffer::new(
        host,
        result_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        Place::Readback,
    )?;
    let set_layout = descriptors::storage_layout(host, in_memory)?;
    let props = descriptors::properties(host);
    let mut own_heap = None;
    let mut fallback = None;
    let mut binding = None;
    if let Some(db) = &props {
        let r = (|| -> Result<String> {
            let (size, offset) =
                descriptors::layout_size(host, set_layout).context("layout size")?;
            let heap = Buffer::new(
                host,
                PAGE,
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
                Place::Upload,
            )?;
            let n = db.storage_buffer_descriptor_size;
            let mut bytes = vec![0u8; n];
            descriptors::write_storage_buffer(host, result.address, result_size, &mut bytes);
            // SAFETY: the heap is mapped and holds the layout.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), heap.mapped.add(offset as usize), n)
            };
            binding = Some(Binding::InMemory {
                address: heap.address,
                usage: vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                offset: 0,
            });
            own_heap = Some(heap);
            Ok(format!(
                "layout {size} B (binding 0 at {offset}); descriptor sizes: sampler {}, sampled image {}, storage image {}, uniform texel {}, storage texel {}, uniform buffer {}, storage buffer {}, acceleration structure {}; offset alignment {}, resource range {} MiB, resource address space {} MiB, sampler address space {} MiB, push descriptors without a buffer {}",
                db.sampler_descriptor_size,
                db.sampled_image_descriptor_size,
                db.storage_image_descriptor_size,
                db.uniform_texel_buffer_descriptor_size,
                db.storage_texel_buffer_descriptor_size,
                db.uniform_buffer_descriptor_size,
                db.storage_buffer_descriptor_size,
                db.acceleration_structure_descriptor_size,
                db.descriptor_buffer_offset_alignment,
                db.max_resource_descriptor_buffer_range >> 20,
                db.resource_descriptor_buffer_address_space_size >> 20,
                db.sampler_descriptor_buffer_address_space_size >> 20,
                db.bufferless_push_descriptors == vk::TRUE,
            ))
        })();
        p.step("descriptor buffer", r);
        let fits = db.sampled_image_descriptor_size <= 32
            && db.storage_image_descriptor_size <= 32
            && db.storage_buffer_descriptor_size <= 16
            && db.sampler_descriptor_size <= 16;
        p.info(
            "descriptor sizes against the console's",
            format!(
                "views fit in a 32-byte T# slot and buffers/samplers in 16 bytes: {fits} (host image {} B, buffer {} B, sampler {} B)",
                db.sampled_image_descriptor_size.max(db.storage_image_descriptor_size),
                db.storage_buffer_descriptor_size,
                db.sampler_descriptor_size
            ),
        );
    } else {
        let r = Fallback::new(host, set_layout, result.buffer, result_size).map(|f| {
            binding = Some(Binding::Set(f.set));
            fallback = Some(f);
            "no VK_EXT_descriptor_buffer: one descriptor set stands in (the backend's fallback is one descriptor-indexing set)".to_owned()
        });
        p.step("descriptor set (fallback)", r);
    }

    // The pipeline.
    let probe = compute::Probe::new(host, set_layout, in_memory);
    let probe = match probe {
        Ok(pr) => {
            p.ok(
                "compute pipeline",
                format!(
                    "SPIR-V {} words, made in {:.2} ms",
                    compute::SPIRV.len(),
                    pr.ms
                ),
            );
            pr
        }
        Err(e) => {
            p.fail("compute pipeline", format!("{e:#}"));
            bail!("no pipeline, no compute steps");
        }
    };
    let cache_uuid = host.caps.pipeline_cache_uuid;
    let mut runner = unsafe { Runner::new(host, probe, result) }?;
    let binding = binding.context("no descriptor binding")?;

    // Compute: a private region read and written by address.
    if let Some(r) = &region_import {
        let seed = 0x5eed_0001;
        let src = TitleMemory::words(memory.region, 0, COUNT as usize);
        for (i, w) in src.iter_mut().enumerate() {
            *w = pattern(i, seed);
        }
        let dst_off = 32 * MIB;
        let salt = 0xa5a5_0001;
        let push = Push {
            src: r.device_base,
            dst: r.device_base + dst_off as u64,
            count: COUNT,
            salt,
        };
        let ms = unsafe { runner.run(&binding, push, |_, _| {}) };
        let res = ms.and_then(|ms| {
            runner.check(|i| pattern(i, seed), TitleMemory::words(memory.region, dst_off, COUNT as usize), salt)?;
            Ok(format!(
                "{COUNT} words read at device {:#x} and written back at {:#x}; the CPU sees them in place; {ms:.2} ms submit to timeline value",
                push.src, push.dst
            ))
        });
        p.step("compute, region by address", res);
    }

    // Compute: a section's alias. The CPU writes through view B, the card
    // reads through view A's import and writes back; the CPU reads view B.
    if let Some(s) = &section_import {
        let seed = 0x5eed_0002;
        for (i, w) in TitleMemory::words(memory.view_b, 0, COUNT as usize)
            .iter_mut()
            .enumerate()
        {
            *w = pattern(i, seed);
        }
        let dst_off = 16 * MIB;
        let salt = 0xa5a5_0002;
        let push = Push {
            src: s.device_base,
            dst: s.device_base + dst_off as u64,
            count: COUNT,
            salt,
        };
        let res = unsafe { runner.run(&binding, push, |_, _| {}) }.and_then(|ms| {
            runner.check(|i| pattern(i, seed), TitleMemory::words(memory.view_b, dst_off, COUNT as usize), salt)?;
            Ok(format!(
                "the card read through view A what the CPU wrote through view B, and the CPU read through view B what the card wrote through view A; {ms:.2} ms"
            ))
        });
        p.step("compute, section alias", res);
    }

    // The streamed-once class: a copy on the card, read by address.
    if let Some(r) = &region_import {
        let res = (|| -> Result<String> {
            let card = Buffer::new(
                host,
                MIB as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                Place::Card,
            )?;
            let seed = 0x5eed_0001; // the region's words from the first run
            let dst_off = 40 * MIB;
            let salt = 0xa5a5_0003;
            let push = Push {
                src: card.address,
                dst: r.device_base + dst_off as u64,
                count: COUNT,
                salt,
            };
            let ms = unsafe {
                runner.run(&binding, push, |device, cmd| {
                    device.cmd_copy_buffer(
                        cmd,
                        r.buffer,
                        card.buffer,
                        &[vk::BufferCopy::default().size(MIB as u64)],
                    );
                    let b = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COPY)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&b),
                    );
                })
            };
            let checked = ms.and_then(|ms| {
                runner.check(
                    |i| pattern(i, seed),
                    TitleMemory::words(memory.region, dst_off, COUNT as usize),
                    salt,
                )?;
                Ok(ms)
            });
            unsafe { card.destroy(host) };
            Ok(format!(
                "1 MiB copied from title memory into card memory and read there by address ({:.2} ms)",
                checked?
            ))
        })();
        p.step("card copy (streamed once)", res);
    }

    // A descriptor heap in title memory, used as the descriptor buffer.
    if let Some(db) = &props {
        let res = (|| -> Result<String> {
            let (_, offset) = descriptors::layout_size(host, set_layout).context("layout size")?;
            let usage = vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT;
            let heap = unsafe { Imported::new(host, memory.heap.cast(), MIB as u64, usage) }?;
            let n = db.storage_buffer_descriptor_size;
            let mut bytes = vec![0u8; n];
            descriptors::write_storage_buffer(host, runner.result.address, result_size, &mut bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), memory.heap.add(offset as usize), n)
            };
            let bind = Binding::InMemory {
                address: heap.device_base,
                usage: TITLE_USAGE | usage,
                offset: 0,
            };
            let seed = 0x5eed_0001;
            let dst_off = 44 * MIB;
            let salt = 0xa5a5_0004;
            let push = Push {
                src: region_import.as_ref().context("no region")?.device_base,
                dst: region_import.as_ref().context("no region")?.device_base + dst_off as u64,
                count: COUNT,
                salt,
            };
            let r = unsafe { runner.run(&bind, push, |_, _| {}) }.and_then(|ms| {
                runner.check(
                    |i| pattern(i, seed),
                    TitleMemory::words(memory.region, dst_off, COUNT as usize),
                    salt,
                )?;
                Ok(ms)
            });
            unsafe { heap.destroy(host) };
            Ok(format!(
                "descriptor written by the CPU into imported title memory at device {:#x} and read from there ({:.2} ms)",
                heap.device_base, r?
            ))
        })();
        p.try_step("descriptor heap in title memory", res);
    }

    // The pool's GPU view as one sparse buffer over imported section pages.
    if let (Some(s), Some(r)) = (&section_import, &region_import) {
        let res = unsafe { sparse_alias(host, &mut runner, &binding, &memory, s, r) };
        p.try_step("title window (sparse buffer over imports)", res);
    }

    // Graphics pipelines: stages compiled once, linked per draw.
    let c = &host.caps;
    p.info(
        "pipeline model",
        format!(
            "graphics pipeline libraries {}, fast linking {}, independent interpolation {}; shader objects {}; blend state dynamic {}, raster state dynamic {}",
            c.pipeline_library,
            c.fast_linking,
            c.independent_interpolation,
            c.shader_object,
            c.dynamic_blend,
            c.dynamic_raster
        ),
    );
    match graphics::Target::new(host) {
        Ok(target) => {
            unsafe { graphics_steps(host, p, &target, runner.probe.cache) };
            unsafe { target.destroy(host) };
        }
        Err(e) => p.no("graphics target", format!("{e:#}")),
    }

    // Images over title memory (a query; no image is made).
    p.info("images over title memory", image_support(host));

    let bytes = runner.probe.cache_bytes(host);
    p.ok(
        "pipeline cache",
        format!(
            "{bytes} bytes after one pipeline; cache UUID {}",
            cache_uuid
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ),
    );

    unsafe {
        host.device.device_wait_idle()?;
        runner.destroy();
        if let Some(h) = own_heap {
            h.destroy(host);
        }
        if let Some(f) = fallback {
            f.destroy(host);
        }
        host.device.destroy_descriptor_set_layout(set_layout, None);
        if let Some(i) = region_import {
            i.destroy(host);
        }
        if let Some(i) = section_import {
            i.destroy(host);
        }
    }
    drop(memory);

    Ok(())
}

/// The title window (`docs/vulkan-host.md`, "One window over all title
/// memory"): one sparse buffer whose pages are bound to imported title
/// memory at the offsets of the title's addresses, so that every title
/// address is `window + address`. Binds 16 pages of the imported section
/// 48 GiB into a 64 GiB window, out of order and one page at two places
/// (the console's GPU page table: `CopyPageMappingsBatchX` as
/// `vkQueueBindSparse`), then reads them by address.
unsafe fn sparse_alias(
    host: &Host,
    runner: &mut Runner<'_>,
    binding: &Binding,
    memory: &TitleMemory,
    section: &Imported,
    region: &Imported,
) -> Result<String> {
    let c = &host.caps;
    ensure!(
        c.sparse_binding && c.sparse_residency_buffer,
        "no sparse binding or sparse buffer residency"
    );
    let queue = host.queues.graphics;
    ensure!(
        queue.sparse,
        "the graphics queue does not bind sparse memory"
    );
    let sparse = vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY;
    let host_allocation = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external_props = vk::ExternalBufferProperties::default();
    unsafe {
        host.instance
            .get_physical_device_external_buffer_properties(
                host.physical,
                &vk::PhysicalDeviceExternalBufferInfo::default()
                    .flags(sparse)
                    .usage(TITLE_USAGE)
                    .handle_type(host_allocation),
                &mut external_props,
            )
    };
    let m = external_props.external_memory_properties;
    ensure!(
        m.external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            && m.compatible_handle_types.contains(host_allocation),
        "the driver does not import host memory into sparse buffers (features {:?}, handle types {:?})",
        m.external_memory_features,
        m.compatible_handle_types
    );
    let device = &host.device;
    let pages = (u64::from(COUNT) * 4).div_ceil(PAGE);
    const WINDOW: u64 = 64 << 30;
    let (size, at) = if c.sparse_address_space_size >= WINDOW && c.max_buffer_size >= WINDOW {
        (WINDOW, WINDOW - (16 << 30))
    } else {
        (pages * PAGE, 0)
    };
    let map = |page: u64| [3u64, 1, 3, 0][(page % 4) as usize]; // GPU page -> section page
    unsafe {
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .flags(sparse)
                .size(size)
                .usage(TITLE_USAGE)
                .push_next(&mut external),
            None,
        )?;
        let result = (|| -> Result<String> {
            let needs = device.get_buffer_memory_requirements(buffer);
            ensure!(
                needs.memory_type_bits & (1 << section.memory_type) != 0,
                "the sparse buffer takes memory types {:#x}, the import is type {}",
                needs.memory_type_bits,
                section.memory_type
            );
            ensure!(
                PAGE.is_multiple_of(needs.alignment),
                "sparse page size {:#x}",
                needs.alignment
            );
            let binds: Vec<vk::SparseMemoryBind> = (0..pages)
                .map(|g| {
                    vk::SparseMemoryBind::default()
                        .resource_offset(at + g * PAGE)
                        .size(PAGE)
                        .memory(section.memory)
                        .memory_offset(map(g) * PAGE)
                })
                .collect();
            let buffer_binds = [vk::SparseBufferMemoryBindInfo::default()
                .buffer(buffer)
                .binds(&binds)];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let bound = device.queue_bind_sparse(
                queue.queue,
                &[vk::BindSparseInfo::default().buffer_binds(&buffer_binds)],
                fence,
            );
            let waited = bound.and_then(|_| device.wait_for_fences(&[fence], true, 5_000_000_000));
            device.destroy_fence(fence, None);
            waited.context("vkQueueBindSparse")?;
            let address = device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

            let seed = 0x5eed_0005;
            let words_per_page = (PAGE / 4) as usize;
            let section_words = TitleMemory::words(memory.view_b, 0, 4 * words_per_page);
            for (i, w) in section_words.iter_mut().enumerate() {
                *w = pattern(i, seed);
            }
            let expect = |i: usize| {
                let g = (i / words_per_page) as u64;
                pattern(map(g) as usize * words_per_page + i % words_per_page, seed)
            };
            let dst_off = 48 * MIB;
            let salt = 0xa5a5_0005;
            let push = Push {
                src: address + at,
                dst: region.device_base + dst_off as u64,
                count: COUNT,
                salt,
            };
            let ms = runner.run(binding, push, |_, _| {})?;
            runner.check(
                expect,
                TitleMemory::words(memory.region, dst_off, COUNT as usize),
                salt,
            )?;
            Ok(format!(
                "a {} GiB sparse window at device {address:#x}: {pages} pages bound {} GiB in show section pages 3,1,3,0,... (one page at two places); read right ({ms:.2} ms); sparse address space {} GiB, unbound pages read zeros: {}",
                size >> 30,
                at >> 30,
                c.sparse_address_space_size >> 30,
                c.non_resident_strict
            ))
        })();
        device.device_wait_idle()?;
        device.destroy_buffer(buffer, None);
        result
    }
}

fn image_support(host: &Host) -> String {
    let mut out = Vec::new();
    for (tiling, name) in [
        (vk::ImageTiling::LINEAR, "linear"),
        (vk::ImageTiling::OPTIMAL, "optimal"),
    ] {
        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(tiling)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .push_next(&mut external);
        let mut ext_props = vk::ExternalImageFormatProperties::default();
        let mut props = vk::ImageFormatProperties2::default().push_next(&mut ext_props);
        // SAFETY: a plain query on a live physical device.
        let r = unsafe {
            host.instance.get_physical_device_image_format_properties2(
                host.physical,
                &info,
                &mut props,
            )
        };
        let importable = ext_props
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
        out.push(match r {
            Ok(()) if importable => format!("{name} RGBA8 sampled images importable"),
            Ok(()) => format!("{name} RGBA8: format ok, not importable"),
            Err(e) => format!("{name} RGBA8: {e}"),
        });
    }
    out.join("; ")
}

/// Pipeline libraries and shader objects: compile times, link times, and a
/// draw with each.
unsafe fn graphics_steps(
    host: &Host,
    p: &mut Report,
    target: &graphics::Target,
    cache: vk::PipelineCache,
) {
    let device = &host.device;
    let layout = match unsafe {
        device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
    } {
        Ok(l) => l,
        Err(e) => return p.no("pipeline libraries", format!("vkCreatePipelineLayout: {e}")),
    };
    let res = graphics::Libraries::new(host, layout, cache).and_then(|libs| {
        let draws = (|| -> Result<String> {
            let fast = target.draw(host, |cmd| unsafe { graphics::draw_pipeline(host, cmd, libs.fast) })?;
            let optimised = target.draw(host, |cmd| unsafe { graphics::draw_pipeline(host, cmd, libs.optimised) })?;
            ensure!(graphics::right(fast), "the fast-linked pipeline drew {fast:?}, expected {:?}", graphics::EXPECT);
            ensure!(graphics::right(optimised), "the optimised pipeline drew {optimised:?}, expected {:?}", graphics::EXPECT);
            let t = &libs.times;
            Ok(format!(
                "parts compiled in {:.2} (vertex input) + {:.2} (pre-rasterisation) + {:.2} (fragment) + {:.2} (output) ms; fast link {:.3} ms, optimised link {:.2} ms, the same pipeline whole {:.2} ms; both linked pipelines drew right",
                t.vertex_input, t.pre_raster, t.fragment, t.output, t.fast_link, t.optimised_link, t.monolithic
            ))
        })();
        unsafe { libs.destroy(host) };
        draws
    });
    p.try_step("pipeline libraries", res);

    let res = graphics::Objects::new(host).and_then(|objects| {
        let draw = target
            .draw(host, |cmd| unsafe { objects.draw(host, cmd) })
            .and_then(|px| {
                ensure!(
                    graphics::right(px),
                    "shader objects drew {px:?}, expected {:?}",
                    graphics::EXPECT
                );
                Ok(format!(
                    "both stages compiled in {:.2} ms (no link), binaries {} bytes; drew right",
                    objects.ms, objects.binary_bytes
                ))
            });
        unsafe { objects.destroy(host) };
        draw
    });
    p.try_step("shader objects", res);
    unsafe { device.destroy_pipeline_layout(layout, None) };
}
