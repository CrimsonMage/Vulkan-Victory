//! The Vulkan instance and device (`docs/vulkan-host.md`, "Device").
//!
//! One instance, one physical device (the first discrete GPU, else the
//! first one), one device with one queue per kind the title's D3D12.X
//! queues need: graphics (direct lists, presentation), compute (async
//! compute lists), transfer (copy lists, first texture uploads). Every
//! feature and extension the backend uses is asked for if the driver has
//! it; what is missing is logged with what it is needed for, and only the
//! required ones stop device creation.
//!
//! Features are enabled as the driver reports them, minus the few that
//! cost speed or need more set-up than they give (robust buffer access,
//! capture/replay, protected memory): the backend's paths pick by
//! [`Caps`], not by what it asked for.

use std::ffi::{CStr, c_char};

use anyhow::{Context, Result, bail};
use ash::{ext, khr, vk};

/// How much the backend depends on a feature or extension.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Need {
    /// Device creation fails without it.
    Required,
    /// A path of the backend uses it; without it that path takes its
    /// fallback (named in the purpose).
    Wanted,
    /// Not enabled in this milestone; reported for the plan.
    Report,
}

/// One device extension the backend knows, with why.
pub struct Extension {
    pub name: &'static CStr,
    pub need: Need,
    pub purpose: &'static str,
}

const fn ext(name: &'static CStr, need: Need, purpose: &'static str) -> Extension {
    Extension {
        name,
        need,
        purpose,
    }
}

/// The device extensions, in the order they are reported.
pub const EXTENSIONS: &[Extension] = &[
    ext(
        khr::swapchain::NAME,
        Need::Required,
        "presentation on the title's window",
    ),
    ext(
        ext::external_memory_host::NAME,
        Need::Required,
        "title memory imported in place (regions and sections), read by device address",
    ),
    ext(
        ext::descriptor_buffer::NAME,
        Need::Wanted,
        "descriptors in memory as on the console (fallback: one descriptor-indexing set)",
    ),
    ext(
        ext::mesh_shader::NAME,
        Need::Wanted,
        "the title's mesh and amplification programs",
    ),
    ext(
        ext::robustness2::NAME,
        Need::Wanted,
        "null descriptors for empty slots",
    ),
    ext(
        khr::fragment_shader_barycentric::NAME,
        Need::Wanted,
        "pixel programs that read per-vertex attributes (v_interp_mov; without it those programs are refused)",
    ),
    ext(
        ext::shader_stencil_export::NAME,
        Need::Wanted,
        "pixel programs that export stencil (SPV_EXT_shader_stencil_export; without it those programs are refused)",
    ),
    ext(
        ext::extended_dynamic_state3::NAME,
        Need::Wanted,
        "blend, raster and depth state set per draw (fewer pipelines)",
    ),
    ext(
        khr::pipeline_library::NAME,
        Need::Wanted,
        "pipeline libraries (with graphics_pipeline_library)",
    ),
    ext(
        ext::graphics_pipeline_library::NAME,
        Need::Wanted,
        "stages compiled once, linked in a millisecond, optimised later",
    ),
    ext(
        ext::shader_object::NAME,
        Need::Wanted,
        "stages compiled once, no link, all state dynamic (alternative to pipeline libraries)",
    ),
    ext(
        ext::mutable_descriptor_type::NAME,
        Need::Wanted,
        "one heap-like set of mutable descriptors (descriptor-buffer fallback)",
    ),
    ext(
        khr::maintenance5::NAME,
        Need::Wanted,
        "64-bit pipeline and buffer usage flags, index buffer ranges",
    ),
    ext(
        ext::custom_border_color::NAME,
        Need::Wanted,
        "the console samplers' border colours",
    ),
    ext(
        ext::depth_clip_enable::NAME,
        Need::Wanted,
        "depth clip off with depth clamp, as the title's raster state asks",
    ),
    ext(
        ext::conservative_rasterization::NAME,
        Need::Wanted,
        "conservative raster state",
    ),
    ext(
        ext::memory_budget::NAME,
        Need::Wanted,
        "the card's budget for the memory ceiling (memcap)",
    ),
    ext(
        khr::acceleration_structure::NAME,
        Need::Report,
        "ray tracing (with ray_query; the title's BVH builder is its own)",
    ),
    ext(
        khr::ray_query::NAME,
        Need::Report,
        "ray queries in translated programs",
    ),
    ext(
        khr::fragment_shading_rate::NAME,
        Need::Report,
        "variable-rate shading",
    ),
    ext(
        ext::conditional_rendering::NAME,
        Need::Report,
        "predication",
    ),
    ext(
        ext::memory_priority::NAME,
        Need::Report,
        "eviction priority of card memory",
    ),
    ext(
        ext::pageable_device_local_memory::NAME,
        Need::Report,
        "driver paging of card memory under pressure",
    ),
    ext(
        khr::present_wait::NAME,
        Need::Report,
        "frame pacing on present (with present_id)",
    ),
    ext(
        c"VK_EXT_device_generated_commands",
        Need::Report,
        "ExecuteIndirect with state changes on the card",
    ),
    ext(
        c"VK_EXT_descriptor_heap",
        Need::Report,
        "D3D12-style descriptor heaps and push data (successor of descriptor_buffer)",
    ),
    ext(
        c"VK_KHR_shader_untyped_pointers",
        Need::Report,
        "untyped pointers in SPIR-V (raw title memory as bytes)",
    ),
    ext(
        c"VK_EXT_shader_64bit_indexing",
        Need::Report,
        "64-bit indexing of buffers over 4 GiB",
    ),
    ext(
        c"VK_KHR_shader_maximal_reconvergence",
        Need::Report,
        "wave semantics the console's code assumes",
    ),
    ext(
        c"VK_KHR_shader_quad_control",
        Need::Report,
        "quad semantics the console's code assumes",
    ),
    ext(
        c"VK_EXT_map_memory_placed",
        Need::Report,
        "card memory mapped at a chosen address (Wine keeps it internal)",
    ),
    ext(
        c"VK_KHR_external_memory_win32",
        Need::Report,
        "Win32 handles (Wine: its own shared resources only, not sections)",
    ),
];

/// One feature bit the backend checks, with why.
pub struct Feature {
    pub name: &'static str,
    pub need: Need,
    pub purpose: &'static str,
    pub present: bool,
}

/// What the device turned out to have.
pub struct Caps {
    pub device_name: String,
    pub driver: String,
    pub api_version: u32,
    pub device_type: vk::PhysicalDeviceType,
    /// Each known extension and whether the device has it.
    pub extensions: Vec<(&'static Extension, bool)>,
    pub features: Vec<Feature>,
    /// The core features as enabled.
    pub core: vk::PhysicalDeviceFeatures,
    pub descriptor_buffer: bool,
    pub host_memory: bool,
    pub mesh_shader: bool,
    /// Graphics pipeline libraries (with their pair extension).
    pub pipeline_library: bool,
    /// The driver's fast link is fast (property `graphicsPipelineLibraryFastLinking`).
    pub fast_linking: bool,
    /// Fragment libraries need not repeat the pre-rasterisation stage's
    /// interpolation decorations.
    pub independent_interpolation: bool,
    pub shader_object: bool,
    /// Blend enable, equation and write mask as dynamic state.
    pub dynamic_blend: bool,
    /// Polygon mode, depth clamp, rasterisation samples, sample mask and
    /// alpha to coverage as dynamic state (what shader objects need).
    pub dynamic_raster: bool,
    pub memory_budget: bool,
    pub win32_surface: bool,
    pub min_imported_host_pointer_alignment: u64,
    pub max_buffer_size: u64,
    pub max_memory_allocation_size: u64,
    pub max_push_constants_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub pipeline_cache_uuid: [u8; vk::UUID_SIZE],
    pub sparse_binding: bool,
    pub sparse_residency_buffer: bool,
    /// Bytes of device address space sparse resources may take in all.
    pub sparse_address_space_size: u64,
    /// Unbound sparse pages read zeros (and drop writes).
    pub non_resident_strict: bool,
    pub timestamp_period: f32,
}

impl Caps {
    pub fn has(&self, name: &CStr) -> bool {
        self.extensions
            .iter()
            .any(|(e, present)| *present && e.name == name)
    }

    /// Missing extensions and features, required first.
    pub fn missing(&self) -> Vec<(Need, String, &'static str)> {
        let mut out: Vec<_> = self
            .extensions
            .iter()
            .filter(|(_, present)| !present)
            .map(|(e, _)| (e.need, e.name.to_string_lossy().into_owned(), e.purpose))
            .chain(
                self.features
                    .iter()
                    .filter(|f| !f.present)
                    .map(|f| (f.need, f.name.to_owned(), f.purpose)),
            )
            .collect();
        out.sort_by_key(|(need, ..)| *need as u8);
        out
    }
}

/// A queue and its family.
#[derive(Clone, Copy, Debug)]
pub struct Queue {
    pub family: u32,
    pub queue: vk::Queue,
    /// Queues the family offers (the title's queues of one kind share one).
    pub family_count: u32,
    pub sparse: bool,
}

pub struct Queues {
    /// Graphics + compute + transfer; presents.
    pub graphics: Queue,
    /// A compute-only family, if the device has one.
    pub compute: Option<Queue>,
    /// A transfer-only family, if the device has one.
    pub transfer: Option<Queue>,
}

/// The instance and device with the extension tables the backend uses.
pub struct Host {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub caps: Caps,
    pub queues: Queues,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub surface: Option<khr::surface::Instance>,
    pub win32_surface: Option<khr::win32_surface::Instance>,
    pub swapchain: khr::swapchain::Device,
    pub host_memory: ext::external_memory_host::Device,
    pub shader_objects: Option<ext::shader_object::Device>,
    pub descriptor_buffer: Option<ext::descriptor_buffer::Device>,
    /// The instance API version the loader reports.
    pub instance_version: u32,
    pub instance_extensions: Vec<String>,
}

/// `f2` with `s` chained when `on`.
fn chain_features<'a, T: vk::ExtendsPhysicalDeviceFeatures2 + ?Sized>(
    f2: vk::PhysicalDeviceFeatures2<'a>,
    on: bool,
    s: &'a mut T,
) -> vk::PhysicalDeviceFeatures2<'a> {
    if on { f2.push_next(s) } else { f2 }
}

/// `info` with `s` chained when `on`.
fn chain_device<'a, T: vk::ExtendsDeviceCreateInfo + ?Sized>(info: vk::DeviceCreateInfo<'a>, on: bool, s: &'a mut T) -> vk::DeviceCreateInfo<'a> {
    if on { info.push_next(s) } else { info }
}

pub fn version(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

fn name_of(raw: &[c_char]) -> String {
    // SAFETY: Vulkan fills fixed arrays with NUL-terminated strings.
    unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl Host {
    /// Loads the Vulkan loader (`vulkan-1.dll`: winevulkan under Proton)
    /// and makes the instance and device.
    pub fn new() -> Result<Host> {
        // SAFETY: loading the system's Vulkan loader; every call below
        // follows the Vulkan specification's valid usage.
        unsafe { Self::create() }
    }

    unsafe fn create() -> Result<Host> {
        let entry = unsafe { ash::Entry::load() }.context("no Vulkan loader (vulkan-1.dll)")?;
        let instance_version =
            unsafe { entry.try_enumerate_instance_version() }?.unwrap_or(vk::API_VERSION_1_0);
        if instance_version < vk::API_VERSION_1_3 {
            bail!(
                "the Vulkan loader offers {}; the backend needs 1.3",
                version(instance_version)
            );
        }

        // Instance: surfaces (Win32), debug names when present.
        let available: Vec<String> =
            unsafe { entry.enumerate_instance_extension_properties(None) }?
                .iter()
                .map(|e| name_of(&e.extension_name))
                .collect();
        let has_instance = |n: &CStr| available.iter().any(|a| a.as_bytes() == n.to_bytes());
        let mut instance_exts: Vec<*const c_char> = Vec::new();
        let mut instance_extensions = Vec::new();
        for n in [
            khr::surface::NAME,
            khr::win32_surface::NAME,
            ext::debug_utils::NAME,
        ] {
            if has_instance(n) {
                instance_exts.push(n.as_ptr());
                instance_extensions.push(n.to_string_lossy().into_owned());
            }
        }
        let app = vk::ApplicationInfo::default()
            .application_name(c"victory")
            .engine_name(c"victory")
            .api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&instance_exts);
        let instance = unsafe { entry.create_instance(&info, None) }.context("vkCreateInstance")?;

        // Physical device: the first discrete GPU, else the first.
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let physical = devices
            .iter()
            .copied()
            .find(|&d| {
                unsafe { instance.get_physical_device_properties(d) }.device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .or_else(|| devices.first().copied())
            .context("no Vulkan device")?;

        let device_exts: Vec<String> =
            unsafe { instance.enumerate_device_extension_properties(physical) }?
                .iter()
                .map(|e| name_of(&e.extension_name))
                .collect();
        let has = |n: &CStr| device_exts.iter().any(|a| a.as_bytes() == n.to_bytes());

        // Properties.
        let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
        let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut m4_props = vk::PhysicalDeviceMaintenance4Properties::default();
        let mut v11_props = vk::PhysicalDeviceVulkan11Properties::default();
        let mut v13_props = vk::PhysicalDeviceVulkan13Properties::default();
        let mut gpl_props = vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
        let props = {
            let mut p2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut driver_props)
                .push_next(&mut m4_props)
                .push_next(&mut v11_props)
                .push_next(&mut v13_props);
            if has(ext::external_memory_host::NAME) {
                p2 = p2.push_next(&mut host_props);
            }
            if has(ext::graphics_pipeline_library::NAME) {
                p2 = p2.push_next(&mut gpl_props);
            }
            unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
            p2.properties
        };
        if props.api_version < vk::API_VERSION_1_3 {
            bail!(
                "{} offers Vulkan {}; the backend needs 1.3",
                name_of(&props.device_name),
                version(props.api_version)
            );
        }

        // Features: query what the driver has, then enable that.
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut db = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
        let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut rob2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut eds3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default();
        let mut gpl = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default();
        let mut mutable = vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default();
        let mut m5 = vk::PhysicalDeviceMaintenance5FeaturesKHR::default();
        let mut border = vk::PhysicalDeviceCustomBorderColorFeaturesEXT::default();
        let mut clip = vk::PhysicalDeviceDepthClipEnableFeaturesEXT::default();
        let mut objects = vk::PhysicalDeviceShaderObjectFeaturesEXT::default();
        let mut bary = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default();
        let on = |n: &CStr| has(n);
        let mut core = {
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut v11)
                .push_next(&mut v12)
                .push_next(&mut v13);
            if on(ext::descriptor_buffer::NAME) {
                f2 = f2.push_next(&mut db);
            }
            if on(ext::mesh_shader::NAME) {
                f2 = f2.push_next(&mut mesh);
            }
            if on(ext::robustness2::NAME) {
                f2 = f2.push_next(&mut rob2);
            }
            if on(ext::extended_dynamic_state3::NAME) {
                f2 = f2.push_next(&mut eds3);
            }
            if on(ext::graphics_pipeline_library::NAME) && on(khr::pipeline_library::NAME) {
                f2 = f2.push_next(&mut gpl);
            }
            if on(ext::mutable_descriptor_type::NAME) {
                f2 = f2.push_next(&mut mutable);
            }
            if on(khr::maintenance5::NAME) {
                f2 = f2.push_next(&mut m5);
            }
            if on(ext::custom_border_color::NAME) {
                f2 = f2.push_next(&mut border);
            }
            if on(ext::depth_clip_enable::NAME) {
                f2 = f2.push_next(&mut clip);
            }
            if on(ext::shader_object::NAME) {
                f2 = f2.push_next(&mut objects);
            }
            f2 = chain_features(f2, on(khr::fragment_shader_barycentric::NAME), &mut bary);
            unsafe { instance.get_physical_device_features2(physical, &mut f2) };
            f2.features
        };
        let null = std::ptr::null_mut();
        v11.p_next = null;
        v12.p_next = null;
        v13.p_next = null;
        db.p_next = null;
        mesh.p_next = null;
        rob2.p_next = null;
        eds3.p_next = null;
        gpl.p_next = null;
        mutable.p_next = null;
        m5.p_next = null;
        border.p_next = null;
        clip.p_next = null;
        objects.p_next = null;
        bary.p_next = null;

        let features = feature_table(&core, &v11, &v12, &v13);
        let descriptor_buffer = db.descriptor_buffer == vk::TRUE;
        let mesh_shader = mesh.mesh_shader == vk::TRUE;
        let pipeline_library = gpl.graphics_pipeline_library == vk::TRUE;
        let shader_object = objects.shader_object == vk::TRUE;
        let dynamic_blend = eds3.extended_dynamic_state3_color_blend_enable == vk::TRUE
            && eds3.extended_dynamic_state3_color_blend_equation == vk::TRUE
            && eds3.extended_dynamic_state3_color_write_mask == vk::TRUE;
        let dynamic_raster = eds3.extended_dynamic_state3_polygon_mode == vk::TRUE
            && eds3.extended_dynamic_state3_depth_clamp_enable == vk::TRUE
            && eds3.extended_dynamic_state3_rasterization_samples == vk::TRUE
            && eds3.extended_dynamic_state3_sample_mask == vk::TRUE
            && eds3.extended_dynamic_state3_alpha_to_coverage_enable == vk::TRUE;

        // What costs speed or needs set-up the backend does not do.
        core.robust_buffer_access = vk::FALSE;
        v11.protected_memory = vk::FALSE;
        v12.buffer_device_address_capture_replay = vk::FALSE;
        v12.buffer_device_address_multi_device = vk::FALSE;
        v13.robust_image_access = vk::FALSE;
        db.descriptor_buffer_capture_replay = vk::FALSE;
        mesh.multiview_mesh_shader &= v11.multiview;
        mesh.primitive_fragment_shading_rate_mesh_shader = vk::FALSE;
        mesh.mesh_shader_queries = vk::FALSE;
        rob2.robust_buffer_access2 = vk::FALSE;
        rob2.robust_image_access2 = vk::FALSE;

        let extensions: Vec<(&'static Extension, bool)> =
            EXTENSIONS.iter().map(|e| (e, has(e.name))).collect();
        let missing_required: Vec<String> = extensions
            .iter()
            .filter(|(e, p)| e.need == Need::Required && !p)
            .map(|(e, _)| e.name.to_string_lossy().into_owned())
            .chain(
                features
                    .iter()
                    .filter(|f| f.need == Need::Required && !f.present)
                    .map(|f| f.name.to_owned()),
            )
            .collect();
        if !missing_required.is_empty() {
            bail!(
                "{} lacks what the backend requires: {}",
                name_of(&props.device_name),
                missing_required.join(", ")
            );
        }
        // Enable every known extension the device has, except the reported
        // ones (not used in this milestone) and libraries without their pair.
        let mut enable: Vec<*const c_char> = extensions
            .iter()
            .filter(|(e, p)| *p && e.need != Need::Report)
            .map(|(e, _)| e.name.as_ptr())
            .collect();
        if !(has(ext::graphics_pipeline_library::NAME) && has(khr::pipeline_library::NAME)) {
            enable.retain(|&n| {
                let n = unsafe { CStr::from_ptr(n) };
                n != ext::graphics_pipeline_library::NAME && n != khr::pipeline_library::NAME
            });
        }

        // Queues: one graphics, one compute-only, one transfer-only.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let find = |want: vk::QueueFlags, not: vk::QueueFlags| {
            families
                .iter()
                .position(|f| f.queue_flags.contains(want) && !f.queue_flags.intersects(not))
                .map(|i| i as u32)
        };
        let graphics = find(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            vk::QueueFlags::empty(),
        )
        .context("no graphics queue")?;
        let compute = find(vk::QueueFlags::COMPUTE, vk::QueueFlags::GRAPHICS);
        let transfer = find(
            vk::QueueFlags::TRANSFER,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
        );
        let priority = [1.0f32];
        let queue_infos: Vec<vk::DeviceQueueCreateInfo> = [Some(graphics), compute, transfer]
            .into_iter()
            .flatten()
            .map(|family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priority)
            })
            .collect();

        let device = {
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&enable)
                .enabled_features(&core)
                .push_next(&mut v11)
                .push_next(&mut v12)
                .push_next(&mut v13);
            if descriptor_buffer {
                info = info.push_next(&mut db);
            }
            if mesh_shader {
                info = info.push_next(&mut mesh);
            }
            if has(ext::robustness2::NAME) {
                info = info.push_next(&mut rob2);
            }
            if has(ext::extended_dynamic_state3::NAME) {
                info = info.push_next(&mut eds3);
            }
            if has(ext::graphics_pipeline_library::NAME) && has(khr::pipeline_library::NAME) {
                info = info.push_next(&mut gpl);
            }
            if has(ext::mutable_descriptor_type::NAME) {
                info = info.push_next(&mut mutable);
            }
            if has(khr::maintenance5::NAME) {
                info = info.push_next(&mut m5);
            }
            if has(ext::custom_border_color::NAME) {
                info = info.push_next(&mut border);
            }
            if has(ext::depth_clip_enable::NAME) {
                info = info.push_next(&mut clip);
            }
            if shader_object {
                info = info.push_next(&mut objects);
            }
            info = chain_device(info, has(khr::fragment_shader_barycentric::NAME), &mut bary);
            unsafe { instance.create_device(physical, &info, None) }.context("vkCreateDevice")?
        };

        let queue = |family: u32| Queue {
            family,
            queue: unsafe { device.get_device_queue(family, 0) },
            family_count: families[family as usize].queue_count,
            sparse: families[family as usize]
                .queue_flags
                .contains(vk::QueueFlags::SPARSE_BINDING),
        };
        let queues = Queues {
            graphics: queue(graphics),
            compute: compute.map(queue),
            transfer: transfer.map(queue),
        };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let win32_surface = instance_extensions
            .iter()
            .any(|e| e.as_bytes() == khr::win32_surface::NAME.to_bytes());

        let caps = Caps {
            device_name: name_of(&props.device_name),
            driver: format!(
                "{} {}",
                name_of(&driver_props.driver_name),
                name_of(&driver_props.driver_info)
            ),
            api_version: props.api_version,
            device_type: props.device_type,
            descriptor_buffer,
            host_memory: true,
            mesh_shader,
            pipeline_library,
            fast_linking: gpl_props.graphics_pipeline_library_fast_linking == vk::TRUE,
            independent_interpolation: gpl_props
                .graphics_pipeline_library_independent_interpolation_decoration
                == vk::TRUE,
            shader_object,
            dynamic_blend,
            dynamic_raster,
            memory_budget: has(ext::memory_budget::NAME),
            win32_surface,
            min_imported_host_pointer_alignment: host_props.min_imported_host_pointer_alignment,
            max_buffer_size: m4_props.max_buffer_size,
            max_memory_allocation_size: v11_props.max_memory_allocation_size,
            max_push_constants_size: props.limits.max_push_constants_size,
            min_subgroup_size: v13_props.min_subgroup_size,
            max_subgroup_size: v13_props.max_subgroup_size,
            pipeline_cache_uuid: props.pipeline_cache_uuid,
            sparse_binding: core.sparse_binding == vk::TRUE,
            sparse_residency_buffer: core.sparse_residency_buffer == vk::TRUE,
            sparse_address_space_size: props.limits.sparse_address_space_size,
            non_resident_strict: props.sparse_properties.residency_non_resident_strict == vk::TRUE,
            timestamp_period: props.limits.timestamp_period,
            extensions,
            features,
            core,
        };

        let surface = win32_surface.then(|| khr::surface::Instance::new(&entry, &instance));
        let win32 = win32_surface.then(|| khr::win32_surface::Instance::new(&entry, &instance));
        let swapchain = khr::swapchain::Device::new(&instance, &device);
        let host_memory = ext::external_memory_host::Device::new(&instance, &device);
        let shader_objects =
            shader_object.then(|| ext::shader_object::Device::new(&instance, &device));
        let descriptor_buffer_fn =
            descriptor_buffer.then(|| ext::descriptor_buffer::Device::new(&instance, &device));

        Ok(Host {
            entry,
            instance,
            physical,
            device,
            caps,
            queues,
            memory,
            surface,
            win32_surface: win32,
            swapchain,
            host_memory,
            shader_objects,
            descriptor_buffer: descriptor_buffer_fn,
            instance_version,
            instance_extensions,
        })
    }

    /// A memory type among `bits` with all of `want`, preferring one
    /// without `avoid`.
    pub fn memory_type(
        &self,
        bits: u32,
        want: vk::MemoryPropertyFlags,
        avoid: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        let fits =
            |i: usize, t: &vk::MemoryType| bits & (1 << i) != 0 && t.property_flags.contains(want);
        types
            .iter()
            .enumerate()
            .find(|(i, t)| fits(*i, t) && !t.property_flags.intersects(avoid))
            .or_else(|| types.iter().enumerate().find(|(i, t)| fits(*i, t)))
            .map(|(i, _)| i as u32)
    }

    /// Heap sizes and, with VK_EXT_memory_budget, budgets and use (bytes).
    pub fn heaps(&self) -> Vec<(vk::MemoryHeap, Option<(u64, u64)>)> {
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default();
        if self.caps.memory_budget {
            p2 = p2.push_next(&mut budget);
        }
        // SAFETY: a plain query on a live physical device.
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical, &mut p2)
        };
        let count = p2.memory_properties.memory_heap_count as usize;
        let heaps = p2.memory_properties.memory_heaps;
        (0..count)
            .map(|i| {
                (
                    heaps[i],
                    self.caps
                        .memory_budget
                        .then(|| (budget.heap_budget[i], budget.heap_usage[i])),
                )
            })
            .collect()
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: the owner destroyed every child object before dropping.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn feature_table(
    core: &vk::PhysicalDeviceFeatures,
    v11: &vk::PhysicalDeviceVulkan11Features,
    v12: &vk::PhysicalDeviceVulkan12Features,
    v13: &vk::PhysicalDeviceVulkan13Features,
) -> Vec<Feature> {
    use Need::{Required as R, Wanted as W};
    let f = |name, need, purpose, b: vk::Bool32| Feature {
        name,
        need,
        purpose,
        present: b == vk::TRUE,
    };
    vec![
        f(
            "bufferDeviceAddress",
            R,
            "title memory by device address (base + offset)",
            v12.buffer_device_address,
        ),
        f(
            "shaderInt64",
            R,
            "64-bit title addresses in shaders",
            core.shader_int64,
        ),
        f(
            "timelineSemaphore",
            R,
            "title fences as timeline values",
            v12.timeline_semaphore,
        ),
        f(
            "synchronization2",
            R,
            "barriers and submissions",
            v13.synchronization2,
        ),
        f(
            "dynamicRendering",
            R,
            "render passes without render pass objects",
            v13.dynamic_rendering,
        ),
        f(
            "scalarBlockLayout",
            W,
            "console data layouts read as they are",
            v12.scalar_block_layout,
        ),
        f(
            "vulkanMemoryModel",
            W,
            "device-scope coherence (the BVH builder's sync)",
            v12.vulkan_memory_model,
        ),
        f(
            "vulkanMemoryModelDeviceScope",
            W,
            "device-scope coherence",
            v12.vulkan_memory_model_device_scope,
        ),
        f(
            "runtimeDescriptorArray",
            W,
            "heap-sized descriptor arrays",
            v12.runtime_descriptor_array,
        ),
        f(
            "descriptorBindingPartiallyBound",
            W,
            "sparse heaps",
            v12.descriptor_binding_partially_bound,
        ),
        f(
            "descriptorBindingVariableDescriptorCount",
            W,
            "heap-sized sets",
            v12.descriptor_binding_variable_descriptor_count,
        ),
        f(
            "descriptorBindingSampledImageUpdateAfterBind",
            W,
            "descriptor-indexing fallback",
            v12.descriptor_binding_sampled_image_update_after_bind,
        ),
        f(
            "shaderSampledImageArrayNonUniformIndexing",
            W,
            "slots picked per lane",
            v12.shader_sampled_image_array_non_uniform_indexing,
        ),
        f(
            "shaderStorageBufferArrayNonUniformIndexing",
            W,
            "slots picked per lane",
            v12.shader_storage_buffer_array_non_uniform_indexing,
        ),
        f(
            "shaderFloat16",
            W,
            "16-bit maths in translated programs",
            v12.shader_float16,
        ),
        f(
            "shaderInt16",
            W,
            "16-bit maths in translated programs",
            core.shader_int16,
        ),
        f(
            "shaderInt8",
            W,
            "byte maths in translated programs",
            v12.shader_int8,
        ),
        f(
            "storageBuffer8BitAccess",
            W,
            "byte loads and stores of title memory",
            v12.storage_buffer8_bit_access,
        ),
        f(
            "storageBuffer16BitAccess",
            W,
            "16-bit loads and stores of title memory",
            v11.storage_buffer16_bit_access,
        ),
        f(
            "subgroupSizeControl",
            W,
            "wave size per pipeline",
            v13.subgroup_size_control,
        ),
        f(
            "computeFullSubgroups",
            W,
            "full waves for wave-wide code",
            v13.compute_full_subgroups,
        ),
        f(
            "maintenance4",
            W,
            "buffer size limits over 4 GiB",
            v13.maintenance4,
        ),
        f(
            "shaderDemoteToHelperInvocation",
            W,
            "discard as the console does it",
            v13.shader_demote_to_helper_invocation,
        ),
        f(
            "drawIndirectCount",
            W,
            "ExecuteIndirect with a count buffer",
            v12.draw_indirect_count,
        ),
        f(
            "multiDrawIndirect",
            W,
            "ExecuteIndirect with many records",
            core.multi_draw_indirect,
        ),
        f(
            "drawIndirectFirstInstance",
            W,
            "indirect records with a first instance",
            core.draw_indirect_first_instance,
        ),
        f(
            "samplerFilterMinmax",
            W,
            "min/max reduction samplers",
            v12.sampler_filter_minmax,
        ),
        f(
            "hostQueryReset",
            W,
            "query resets from the CPU",
            v12.host_query_reset,
        ),
        f(
            "shaderOutputLayer",
            W,
            "layer output from vertex programs",
            v12.shader_output_layer,
        ),
        f(
            "shaderOutputViewportIndex",
            W,
            "viewport output from vertex programs",
            v12.shader_output_viewport_index,
        ),
        f(
            "samplerAnisotropy",
            W,
            "anisotropic samplers",
            core.sampler_anisotropy,
        ),
        f(
            "textureCompressionBC",
            W,
            "BC textures",
            core.texture_compression_bc,
        ),
        f("imageCubeArray", W, "cube arrays", core.image_cube_array),
        f(
            "independentBlend",
            W,
            "blend per target",
            core.independent_blend,
        ),
        f(
            "dualSrcBlend",
            W,
            "dual-source blending",
            core.dual_src_blend,
        ),
        f("depthClamp", W, "depth clamp", core.depth_clamp),
        f("depthBounds", W, "depth bounds test", core.depth_bounds),
        f(
            "fillModeNonSolid",
            W,
            "wireframe state",
            core.fill_mode_non_solid,
        ),
        f("multiViewport", W, "several viewports", core.multi_viewport),
        f(
            "sampleRateShading",
            W,
            "per-sample shading",
            core.sample_rate_shading,
        ),
        f(
            "geometryShader",
            W,
            "rectangle lists' fourth corner",
            core.geometry_shader,
        ),
        f(
            "fragmentStoresAndAtomics",
            W,
            "pixel programs writing memory",
            core.fragment_stores_and_atomics,
        ),
        f(
            "vertexPipelineStoresAndAtomics",
            W,
            "vertex programs writing memory",
            core.vertex_pipeline_stores_and_atomics,
        ),
        f(
            "shaderImageGatherExtended",
            W,
            "gathers with offsets",
            core.shader_image_gather_extended,
        ),
        f(
            "shaderStorageImageReadWithoutFormat",
            W,
            "typed loads without a format",
            core.shader_storage_image_read_without_format,
        ),
        f(
            "shaderStorageImageWriteWithoutFormat",
            W,
            "typed stores without a format",
            core.shader_storage_image_write_without_format,
        ),
        f(
            "shaderClipDistance",
            W,
            "clip distances",
            core.shader_clip_distance,
        ),
        f(
            "shaderCullDistance",
            W,
            "cull distances",
            core.shader_cull_distance,
        ),
        f(
            "occlusionQueryPrecise",
            W,
            "occlusion queries",
            core.occlusion_query_precise,
        ),
        f(
            "pipelineStatisticsQuery",
            W,
            "pipeline statistics",
            core.pipeline_statistics_query,
        ),
        f(
            "sparseBinding",
            W,
            "the pool's GPU view as one sparse buffer (to try)",
            core.sparse_binding,
        ),
        f(
            "sparseResidencyBuffer",
            W,
            "the pool's GPU view as one sparse buffer (to try)",
            core.sparse_residency_buffer,
        ),
    ]
}
