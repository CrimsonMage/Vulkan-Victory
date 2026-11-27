//! M2b step (c): the title's list calls recorded into Vulkan command
//! buffers (`docs/vulkan-host.md`, "Command lists").
//!
//! The title's threads write their list calls down (`list/thin.rs`); the
//! list workers replay each stream here, into one command buffer per part
//! of the list (parts are cut at memory-value waits, `vulkan.rs`).
//!
//! - **Pipelines.** A draw links its stages' library parts (`stages`) with a
//!   vertex-input part and a fragment-output part (the bound targets'
//!   formats) by a fast link, cached per tuple. A draw or dispatch whose
//!   stage is being compiled waits for it, up to 2 s of waiting a frame
//!   (`stages::lookup_waiting`); past that it is skipped and counted.
//! - **User data** goes into push constants (ABI section 2) at every draw.
//! - **Descriptor buffers** and the memory map are bound once per command
//!   buffer.
//! - **Dynamic rendering** begins at the first draw after the targets
//!   change and ends at anything that is not a draw. Every image lives in
//!   `GENERAL`; a title barrier is a full memory barrier.
//! - **Console methods** past the PC's (`console.rs`) and **queries**
//!   (`query.rs`); the FidelityFX upscale as a scale (`upscale.rs`).
//! - **Indirect execution** (`record/indirect.rs`): records bounded by a
//!   pass of ours into the command buffer's scratch, then drawn from it.
//! - **Not yet** (counted in the 10 s line): MSAA draws, tessellation and
//!   geometry stages, line loops and quads; indirect signatures of more
//!   than one argument.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use victory_vk::ash::vk;
use victory_vk::timeline::Recording;
use windows::Win32::Graphics::Direct3D12::*;

use super::images::{self, Image, ViewKey};
use super::stages::{self, Stage};
use super::{Backend, lock};
use crate::descriptor::RtvRecord;
use crate::draw::{IndexView, VertexView};
use crate::object::This;
use victory_shader::{HwStage, RootType};

mod clear;
mod console;
mod copy;
mod indirect;
mod link;
mod query;
mod state;
mod trail;
pub(crate) mod upscale;

pub(crate) use console::ValueWrite;
pub(crate) use query::timestamp_frequency;

use state::d3d_flip;

use link::{DrawKey, Linked, draw_test, layout_text, linked, stage_of};

/// Counters for the 10 s line: draws recorded, dispatches, skipped (stage
/// not ready), skipped (stage failed or no pipeline), skipped (other:
/// indirect, MSAA, unknown topology), command buffers, links, maps bound
/// again inside a command buffer, draws whose first table the bound map
/// does not reach (diagnostics).
pub(crate) static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

#[derive(Default, Clone)]
struct Root {
    /// Per root parameter: its dwords (constants), or an address (lo, hi).
    slots: Vec<Vec<u32>>,
}

impl Root {
    fn slot(&mut self, slot: u32) -> &mut Vec<u32> {
        let i = slot as usize;
        if self.slots.len() <= i {
            self.slots.resize(i + 1, Vec::new());
        }
        &mut self.slots[i]
    }

    fn set_address(&mut self, slot: u32, address: u64) {
        let s = self.slot(slot);
        s.clear();
        s.extend([address as u32, (address >> 32) as u32]);
    }

    fn set_constants(&mut self, slot: u32, offset: u32, values: &[u32]) {
        let s = self.slot(slot);
        let end = offset as usize + values.len();
        if s.len() < end {
            s.resize(end, 0);
        }
        s[offset as usize..end].copy_from_slice(values);
    }

    fn word(&self, slot: u8, dword: u32) -> u32 {
        self.slots
            .get(slot as usize)
            .and_then(|s| s.get(dword as usize))
            .copied()
            .unwrap_or(0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PassKey {
    colors: [vk::ImageView; 8],
    count: u32,
    depth: vk::ImageView,
    extent: (u32, u32),
}

/// What a list records with, between its calls.
pub(crate) struct Rec {
    cmd: Option<Recording>,
    /// The map generation the command buffer has bound, and whether an
    /// address was named since (the map may need a newer block).
    map_gen: u64,
    touched: bool,
    /// The statistics query of the command buffer (diagnostics).
    query: Option<u32>,
    pipeline: This,
    graphics: Root,
    compute: Root,
    heaps: [u64; 2],
    viewports: Vec<vk::Viewport>,
    /// The first viewport's depth range as the title gave it (diagnostics).
    raw_depth: (f32, f32),
    scissors: Vec<vk::Rect2D>,
    topology: u32,
    stencil_ref: u32,
    blend: [f32; 4],
    depth_bounds: (f32, f32),
    index: Option<IndexView>,
    vertex: [Option<VertexView>; 16],
    colors: [Option<RtvRecord>; 8],
    color_count: u32,
    depth: Option<RtvRecord>,
    /// Rendering is open with these attachments, and their formats.
    pass: Option<(PassKey, [vk::Format; 8], vk::Format)>,
    targets_changed: bool,
    /// Depth images whose stored planes were copied in since the last
    /// dispatch or copy of this list (`copy_planes`).
    planes_synced: Vec<u64>,
    /// A draw of the open pass writes depth or stencil.
    pass_wrote_depth: bool,
    bound: vk::Pipeline,
    bound_compute: vk::Pipeline,
    /// Images the recording names (kept alive until the list is reset).
    keep: Vec<Arc<Image>>,
    /// Draw pipelines found this recording, by pipeline and pass formats
    /// (the stage lookups, attributes and link done once each).
    links: HashMap<DrawKey, Arc<Linked>>,
    /// The dynamic state set in the command buffer (a digest of what sets
    /// it), and whether `last_push` is pushed there: set again only on change.
    state_set: Option<u64>,
    pushed: bool,
    /// The open pass's attachment images (draws are counted on them).
    attached: Vec<Arc<Image>>,
    /// Bounded indirect records in the command buffer's scratch.
    indirect: indirect::Windows,
    /// Diagnostics: the last push constants, and the last draw's vertex
    /// attributes and buffers found.
    last_push: [u32; 64],
    last_inputs: (usize, usize),
    last_user: String,
    last_layout: String,
    /// Diagnostics: the title's records of the indirect execution being
    /// recorded (their address, the count's).
    last_indirect: Option<(u64, Option<u64>)>,
    /// What the console's own methods keep (`console.rs`, `query.rs`).
    console: console::State,
    /// Diagnostics: the pipeline the trail last named (`trail.rs`), and
    /// the guard id of the compute stage last bound for a dispatch.
    noted: This,
    guard_id: u32,
    /// Diagnostics: the descriptor table addresses of the last push.
    last_tables: Vec<u32>,
}

// SAFETY: a list is recorded by one worker at a time; the records hold
// title handles (plain addresses) only.
unsafe impl Send for Rec {}

impl Default for Rec {
    fn default() -> Rec {
        Rec {
            cmd: None,
            map_gen: 0,
            touched: false,
            query: None,
            pipeline: std::ptr::null(),
            graphics: Root::default(),
            compute: Root::default(),
            heaps: [0; 2],
            viewports: Vec::new(),
            raw_depth: (0.0, 1.0),
            scissors: Vec::new(),
            topology: 4,
            stencil_ref: 0,
            blend: [0.0; 4],
            depth_bounds: (0.0, 1.0),
            index: None,
            vertex: [None; 16],
            colors: [None; 8],
            color_count: 0,
            depth: None,
            pass: None,
            targets_changed: true,
            planes_synced: Vec::new(),
            pass_wrote_depth: false,
            bound: vk::Pipeline::null(),
            bound_compute: vk::Pipeline::null(),
            keep: Vec::new(),
            links: HashMap::new(),
            state_set: None,
            pushed: false,
            attached: Vec::new(),
            indirect: indirect::Windows::default(),
            last_push: [0; 64],
            last_inputs: (0, 0),
            last_user: String::new(),
            last_layout: String::new(),
            last_indirect: None,
            noted: std::ptr::null(),
            guard_id: 0,
            last_tables: Vec::new(),
            console: console::State::default(),
        }
    }
}

static RECS: Mutex<Option<HashMap<usize, Rec>>> = Mutex::new(None);

/// The list's recording state, taken out while a worker replays a chunk.
pub(crate) fn take(list: This) -> Rec {
    lock(&RECS)
        .get_or_insert_with(HashMap::new)
        .remove(&(list as usize))
        .unwrap_or_default()
}

pub(crate) fn put(list: This, rec: Rec) {
    lock(&RECS).get_or_insert_with(HashMap::new).insert(list as usize, rec);
}

/// The list is released.
pub(crate) fn forget(list: This) {
    let rec = lock(&RECS).as_mut().and_then(|m| m.remove(&(list as usize)));
    if let (Some(mut rec), Some(b)) = (rec, super::backend()) {
        rec.abandon(b);
    }
}

/// A view of a target the record names, as an attachment.
fn target_view(b: &Backend, r: &RtvRecord, depth: bool) -> Option<(Arc<Image>, vk::ImageView, vk::Format, (u32, u32))> {
    let image = images::of_resource(b, r.resource)?;
    let words = r.words();
    let format_word = words.map_or(0, |w| w[0]);
    let dim = words.map_or(if depth { 3 } else { 4 }, |w| w[1]);
    // RTV: format, dimension, then mip, (first, count) or plane; DSV:
    // format, dimension, flags, then mip, (first, count).
    let w = words.unwrap_or([0; 6]);
    let (mip, layer) = if depth {
        match dim {
            1 | 3 => (w[3], 0),
            2 | 4 => (w[3], w[4]),
            6 => (0, w[3]),
            _ => (0, 0),
        }
    } else {
        match dim {
            2 | 4 => (w[2], 0),
            3 | 5 => (w[2], w[3]),
            7 => (0, w[2]),
            8 => (w[2], w[3]),
            _ => (0, 0),
        }
    };
    let format = if image.depth() {
        image.format
    } else {
        let f = if format_word == 0 { None } else { images::color_format(format_word, image.card_written()) };
        f.filter(|f| images::format_bytes(*f) == images::format_bytes(image.format)).unwrap_or(image.format)
    };
    let mip = mip.min(image.mips - 1);
    let view = image.view(
        b,
        ViewKey {
            format,
            kind: vk::ImageViewType::TYPE_2D,
            aspect: image.aspect(),
            base_mip: mip,
            mips: 1,
            base_layer: layer,
            layers: 1,
            components: [vk::ComponentSwizzle::IDENTITY; 4],
            storage: false,
            attachment: true,
        },
    )?;
    let extent = ((image.extent.width >> mip).max(1), (image.extent.height >> mip).max(1));
    Some((image, view, format, extent))
}

/// Counts a skipped draw or dispatch: not ready (`true`) or not to be.
fn skipped(waiting: bool) {
    STATS[if waiting { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
}

impl Rec {
    fn cmd(&mut self, b: &'static Backend) -> Option<vk::CommandBuffer> {
        if self.cmd.is_none() {
            let rec = b.timeline.begin(&b.host).ok()?;
            let heaps = super::descriptors::heaps(b)?;
            heaps.immediates(b);
            let (generation, block) = super::map_block(b, heaps.zero_sink());
            let map = heaps.map_offset(b, generation, &block);
            // SAFETY: recording into our own command buffer.
            unsafe { heaps.bind(b, rec.cmd, map) };
            self.map_gen = generation;
            self.touched = false;
            // SAFETY: outside rendering in our new command buffer.
            self.query = unsafe { super::gpu_stats::begin(b, rec.cmd) };
            STATS[5].fetch_add(1, Ordering::Relaxed);
            self.cmd = Some(rec);
            self.bound = vk::Pipeline::null();
            self.bound_compute = vk::Pipeline::null();
            self.pass = None;
            self.state_set = None;
            self.pushed = false;
            self.indirect.clear();
            self.planes_synced.clear();
        }
        self.cmd.as_ref().map(|r| r.cmd)
    }

    /// Binds a newer memory map when an address named since the command
    /// buffer began brought memory in (heaps, tables, views, buffers).
    fn refresh_map(&mut self, b: &'static Backend, cmd: vk::CommandBuffer) {
        if !std::mem::take(&mut self.touched) || super::map_generation(b) == self.map_gen {
            return;
        }
        let Some(heaps) = super::descriptors::heaps(b) else { return };
        let (generation, block) = super::map_block(b, heaps.zero_sink());
        let map = heaps.map_offset(b, generation, &block);
        // SAFETY: recording into our own command buffer (binding descriptor
        // buffers is allowed inside rendering).
        unsafe { heaps.bind(b, cmd, map) };
        self.map_gen = generation;
        STATS[7].fetch_add(1, Ordering::Relaxed);
    }

    fn end_pass(&mut self, b: &Backend) {
        if self.pass.take().is_some()
            && let Some(cmd) = self.cmd.as_ref().map(|r| r.cmd)
        {
            // SAFETY: rendering is open in our command buffer.
            unsafe { b.host.device.cmd_end_rendering(cmd) };
            // What its draws wrote to depth or stencil goes to the planes
            // compute reads and stores.
            if std::mem::take(&mut self.pass_wrote_depth)
                && let Some(depth) = self.attached.iter().find(|i| i.depth()).cloned()
            {
                // SAFETY: outside rendering now, in our command buffer.
                unsafe { self.copy_planes(b, cmd, &depth, false) };
            }
        }
    }

    /// Hands the command buffer to the list's current part (at a cut and
    /// at the close).
    pub(crate) fn cut(&mut self, list: This, b: &Backend) {
        self.end_pass(b);
        if let Some(rec) = self.cmd.take() {
            self.console_cut(b, rec.cmd);
            if let Some(slot) = self.query.take() {
                // SAFETY: outside rendering, before the buffer ends.
                unsafe { super::gpu_stats::end(b, rec.cmd, slot, true) };
            }
            super::part_commands(list, rec);
        }
    }

    pub(crate) fn abandon(&mut self, b: &Backend) {
        self.end_pass(b);
        if let Some(rec) = self.cmd.take() {
            self.console_cut(b, rec.cmd);
            if let Some(slot) = self.query.take() {
                // SAFETY: outside rendering, before the buffer ends.
                unsafe { super::gpu_stats::end(b, rec.cmd, slot, false) };
            }
            b.timeline.abandon(&b.host, rec);
        }
    }

    pub(crate) fn reset(&mut self, b: &Backend, pipeline: This) {
        self.abandon(b);
        *self = Rec {
            pipeline,
            ..Rec::default()
        };
    }

    // ---- state ----

    pub(crate) fn set_pipeline(&mut self, p: This) {
        self.pipeline = p;
        crate::pipeline::bound(p);
    }
    pub(crate) fn topology(&mut self, t: u32) {
        self.topology = t;
    }
    pub(crate) fn stencil_ref(&mut self, v: u32) {
        self.stencil_ref = v;
    }
    pub(crate) fn depth_bounds(&mut self, min: f32, max: f32) {
        self.depth_bounds = (min, max);
    }
    pub(crate) fn blend_factor(&mut self, f: Option<[f32; 4]>) {
        self.blend = f.unwrap_or([1.0; 4]);
    }
    pub(crate) fn viewports(&mut self, v: &[D3D12_VIEWPORT]) {
        self.viewports = v.iter().map(d3d_flip).collect();
        self.raw_depth = v.first().map_or((0.0, 1.0), |v| (v.MinDepth, v.MaxDepth));
    }
    pub(crate) fn scissors(&mut self, r: &[windows::Win32::Foundation::RECT]) {
        self.scissors = r
            .iter()
            .map(|r| vk::Rect2D {
                offset: vk::Offset2D { x: r.left.max(0), y: r.top.max(0) },
                extent: vk::Extent2D {
                    width: (r.right - r.left.max(0)).max(0) as u32,
                    height: (r.bottom - r.top.max(0)).max(0) as u32,
                },
            })
            .collect();
    }
    pub(crate) fn heaps(&mut self, heaps: &[This]) {
        self.touched = true;
        for &h in heaps {
            if h.is_null() {
                continue;
            }
            let heap = crate::descriptor::heap(h);
            self.heaps[usize::from(heap.samplers())] = heap.start;
            super::touch(heap.start);
        }
    }
    pub(crate) fn table(&mut self, compute: bool, slot: u32, handle: u64) {
        self.touched = true;
        super::touch(handle);
        self.root(compute).set_address(slot, handle);
    }
    pub(crate) fn view(&mut self, compute: bool, slot: u32, address: u64) {
        self.touched = true;
        super::touch(address);
        self.root(compute).set_address(slot, address);
    }
    pub(crate) fn constants(&mut self, compute: bool, slot: u32, offset: u32, values: &[u32]) {
        self.root(compute).set_constants(slot, offset, values);
    }
    fn root(&mut self, compute: bool) -> &mut Root {
        if compute { &mut self.compute } else { &mut self.graphics }
    }
    pub(crate) fn index_buffer(&mut self, v: Option<IndexView>) {
        self.touched = true;
        self.index = v;
    }
    pub(crate) fn vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        self.touched = true;
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = Some(*v);
            }
        }
    }
    pub(crate) fn targets(&mut self, colors: &[Option<RtvRecord>], depth: Option<RtvRecord>) {
        self.colors = [None; 8];
        for (i, c) in colors.iter().take(8).enumerate() {
            self.colors[i] = *c;
        }
        self.color_count = colors.len().min(8) as u32;
        self.depth = depth;
        self.targets_changed = true;
    }

    // ---- work ----

    pub(crate) fn barrier(&mut self, b: &'static Backend) {
        self.end_pass(b);
        if let Some(cmd) = self.cmd(b) {
            // SAFETY: outside rendering, in our command buffer.
            unsafe { images::full_barrier(&b.host.device, cmd) };
        }
    }

    /// Opens rendering on the bound targets if it is not open on them.
    fn begin_pass(&mut self, b: &'static Backend, cmd: vk::CommandBuffer) -> Option<([vk::Format; 8], u32, vk::Format)> {
        if !self.targets_changed
            && let Some((key, formats, depth)) = &self.pass
        {
            return Some((*formats, key.count, *depth));
        }
        let mut key = PassKey {
            colors: [vk::ImageView::null(); 8],
            count: self.color_count,
            depth: vk::ImageView::null(),
            extent: (u32::MAX, u32::MAX),
        };
        let mut formats = [vk::Format::UNDEFINED; 8];
        let mut att: Vec<Arc<Image>> = Vec::new();
        let mut depth_format = vk::Format::UNDEFINED;
        for i in 0..self.color_count as usize {
            if let Some(r) = self.colors[i]
                && let Some((image, view, format, extent)) = target_view(b, &r, false)
            {
                key.colors[i] = view;
                formats[i] = format;
                key.extent = (key.extent.0.min(extent.0), key.extent.1.min(extent.1));
                att.push(image);
            }
        }
        if let Some(r) = self.depth
            && let Some((image, view, format, extent)) = target_view(b, &r, true)
        {
            key.depth = view;
            depth_format = format;
            key.extent = (key.extent.0.min(extent.0), key.extent.1.min(extent.1));
            att.push(image);
        }
        if key.extent.0 == u32::MAX {
            // A draw with no targets at all (its pixel program writes
            // storage views only: fog volume and light proxies) is drawn
            // over its scissor; one whose targets have no image is not.
            let none_asked = self.depth.is_none() && self.colors[..self.color_count as usize].iter().all(Option::is_none);
            let area = self.scissors.first().map(|r| (r.offset.x.max(0) as u32 + r.extent.width, r.offset.y.max(0) as u32 + r.extent.height));
            match area.filter(|a| none_asked && a.0 != 0 && a.1 != 0) {
                Some(a) => key.extent = (a.0.min(16384), a.1.min(16384)),
                None => return None,
            }
        }
        self.keep.extend(att.iter().cloned());
        self.attached = att;
        self.targets_changed = false;
        if let Some((open, ..)) = &self.pass
            && *open == key
        {
            return Some((formats, key.count, depth_format));
        }
        self.end_pass(b);
        // What compute stored into the depth buffer's memory since goes
        // into its planes first.
        if let Some(depth) = self.attached.iter().find(|i| i.depth()).cloned()
            && !self.planes_synced.contains(&depth.id)
        {
            self.planes_synced.push(depth.id);
            // SAFETY: outside rendering, in our command buffer.
            unsafe { self.copy_planes(b, cmd, &depth, true) };
        }
        let attachment = |view: vk::ImageView| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        };
        let colors: Vec<vk::RenderingAttachmentInfo> = (0..key.count as usize)
            .map(|i| if key.colors[i] == vk::ImageView::null() { vk::RenderingAttachmentInfo::default() } else { attachment(key.colors[i]) })
            .collect();
        let depth = attachment(key.depth);
        let stencil = matches!(depth_format, vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT);
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D { width: key.extent.0, height: key.extent.1 },
            })
            .layer_count(1)
            .color_attachments(&colors);
        if key.depth != vk::ImageView::null() {
            info = info.depth_attachment(&depth);
            if stencil {
                info = info.stencil_attachment(&depth);
            }
        }
        // SAFETY: outside rendering in our command buffer; views of live images.
        unsafe { b.host.device.cmd_begin_rendering(cmd, &info) };
        self.pass = Some((key, formats, depth_format));
        Some((formats, key.count, depth_format))
    }

    /// The push constants of the stages (ABI section 2).
    fn push(&mut self, b: &Backend, cmd: vk::CommandBuffer, stages: &[&Stage], compute: bool) {
        let Some(layout) = stages::layout(b) else { return };
        let mut words = [0u32; 64];
        // ABI v3: the loop guard buffer's address in dwords 2–3.
        let guard = super::guard::address(b);
        words[2] = guard as u32;
        words[3] = (guard >> 32) as u32;
        if cfg!(feature = "diagnostics") && compute && guard == 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::error!("Vulkan: a compute dispatch is pushed no loop guard address"));
        }
        let root = if compute { &self.compute } else { &self.graphics };
        for st in stages {
            for ud in &st.user_data {
                let i = ud.index as usize;
                if i >= 64 {
                    continue;
                }
                words[i] = match ud.root {
                    RootType::Constants | RootType::Descriptor | RootType::Table => root.word(ud.slot, ud.dword),
                    RootType::ResourceHeap => self.heaps[0] as u32,
                    RootType::SamplerHeap => self.heaps[1] as u32,
                    _ => 0,
                };
            }
        }
        super::passes::note(|| format!("push {}", words.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")));
        if cfg!(feature = "diagnostics") {
            count_root_views(b, stages, root);
        }
        if self.pushed && words == self.last_push {
            return;
        }
        self.last_push = words;
        self.pushed = true;
        if cfg!(feature = "diagnostics") {
            self.last_tables = stages
                .iter()
                .flat_map(|s| s.user_data.iter())
                .filter(|u| matches!(u.root, RootType::Table) && (u.index as usize) < 64)
                .map(|u| words[u.index as usize])
                .filter(|&w| w != 0)
                .collect();
            self.last_tables.sort_unstable();
            self.last_tables.dedup();
            self.last_user = stages
                .iter()
                .flat_map(|s| s.user_data.iter().map(|u| format!("{}<-{}.{}{:?}", u.index, u.slot, u.dword, u.root)))
                .collect::<Vec<_>>()
                .join(" ");
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        // SAFETY: recording into our command buffer; 256 bytes of the layout's range.
        unsafe { b.host.device.cmd_push_constants(cmd, layout.pipeline, vk::ShaderStageFlags::ALL, 0, &bytes) };
    }

    /// Everything a draw needs bound; `None` (counted) if it is skipped.
    fn prepare_draw(&mut self, b: &'static Backend, mesh: bool) -> Option<vk::CommandBuffer> {
        let Some((shaders, state)) = crate::pipeline::parts(self.pipeline) else {
            skipped(false);
            return None;
        };
        let Some(gs) = state else {
            skipped(false);
            return None;
        };
        let (pre, pixel) = draw_stages(b, shaders, mesh)?;
        if gs.samples.Count > 1 {
            not_yet(NotYet::Msaa);
            return None;
        }
        let Some(topology) = topology(self.topology, mesh) else {
            not_yet(NotYet::Topology);
            return None;
        };
        let cmd = self.cmd(b)?;
        let mut told = None;
        for s in shaders {
            if super::targets::wants(&s.name) {
                self.end_pass(b);
                // SAFETY: outside rendering, in our command buffer.
                unsafe { super::targets::before(b, cmd, &s.name) };
                super::targets::tables(b, &s.name, &self.graphics.slots);
                let programs: Vec<_> = shaders.iter().map(|p| (p.name.clone(), p.hash.clone(), p.stage)).collect();
                // SAFETY: as above.
                unsafe { super::passes::before(b, cmd, &programs, &s.name, &self.graphics.slots) };
                told = Some(s.name.as_str());
            }
        }
        if let Some(h) = super::descriptors::heaps(b) {
            h.immediates(b);
        }
        self.refresh_map(b, cmd);
        self.last_inputs = (0, 0);
        let Some((formats, count, depth)) = self.begin_pass(b, cmd) else {
            self.no_pass(b, shaders);
            return None;
        };
        if let Some(name) = told {
            self.tell_draw(name, gs);
            // the capture takes the colour targets as they are before the
            // draw (and `drawn` after it): rendering is closed for the
            // copies and opened again
            self.end_pass(b);
            // SAFETY: outside rendering, in our command buffer.
            unsafe { super::passes::targets(b, cmd, &self.attached, "before") };
            self.begin_pass(b, cmd)?;
        }
        let key = DrawKey { pipeline: self.pipeline as usize, formats, count, depth, rect: self.topology == RECTANGLES, mesh };
        let made = match self.links.get(&key) {
            Some(l) => l.clone(),
            None => {
                let Some(l) = linked(b, &pre, pixel.as_deref(), gs, key) else {
                    skipped(false);
                    return None;
                };
                if cfg!(feature = "diagnostics") {
                    self.last_layout = layout_text(&pre, gs, &l.inputs);
                }
                let l = Arc::new(l);
                self.links.insert(key, l.clone());
                l
            }
        };
        let d = &b.host.device;
        // SAFETY: recording into our command buffer, rendering open.
        unsafe {
            if self.bound != made.pipeline {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, made.pipeline);
                self.bound = made.pipeline;
                super::gputime::mark(b, cmd, || shaders.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join("+"));
            }
            let mut list: Vec<&Stage> = vec![&pre];
            if let Some(p) = &pixel {
                list.push(p);
            }
            self.push(b, cmd, &list, false);
            let z = &gs.depth;
            self.pass_wrote_depth |= (z.DepthEnable.as_bool() && z.DepthWriteMask.0 == 1) || (z.StencilEnable.as_bool() && z.StencilWriteMask != 0);
            let state = self.state_key(gs, topology, count);
            if self.state_set != Some(state) {
                self.dynamic(b, cmd, gs, topology, count);
                self.state_set = Some(state);
            }
            self.bind_vertices(b, cmd, &made);
        }
        Some(cmd)
    }

    /// Binds the vertex buffers the draw's attributes name.
    ///
    /// # Safety
    /// Recording into `cmd`.
    unsafe fn bind_vertices(&mut self, b: &'static Backend, cmd: vk::CommandBuffer, made: &Linked) {
        let mut found = 0;
        for &slot in &made.bindings {
            let v = self.vertex.get(slot as usize).copied().flatten();
            if let Some(v) = v
                && let Some((buf, off)) = super::buffer_over(b, v.address, u64::from(v.size))
            {
                // SAFETY: as the caller says; a title buffer.
                unsafe { b.host.device.cmd_bind_vertex_buffers2(cmd, slot, &[buf], &[off], Some(&[u64::from(v.size)]), Some(&[u64::from(v.stride)])) };
                found += 1;
            }
        }
        self.last_inputs = (made.inputs.len() + 1000 * made.bindings.len(), found);
    }

    /// Diagnostics: the descriptor the first table of the vertex program
    /// points at (push dword 12), the memory its V# names as the CPU sees
    /// it, and whether the memory map reaches that memory.
    fn table_text(&self) -> String {
        let table = u64::from(self.last_push[12]);
        let Some(b) = super::backend() else { return String::new() };
        if table == 0 {
            return String::new();
        }
        let Ok(v) = super::read_title(b, table, 16) else { return format!("table {table:#x} unreadable") };
        let w: Vec<u32> = v.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let base = u64::from(w[0]) | (u64::from(w[1] & 0xffff) << 32);
        let mapped = super::map_reaches(b, table);
        let base_mapped = super::map_reaches(b, base);
        let bytes = super::read_title(b, base, 64)
            .map(|v| v.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect::<Vec<_>>().join(" "))
            .unwrap_or_else(|e| format!("{e:#}"));
        format!("table {table:#x} (mapped {mapped}) V# {w:08x?} base {base:#x} (mapped {base_mapped}) bytes {bytes}")
    }

    /// Diagnostics: the first 48 bytes of vertex view 0.
    fn vertex_bytes(&self) -> String {
        let Some(v) = self.vertex[0] else { return String::new() };
        let Some(b) = super::backend() else { return String::new() };
        match super::read_title(b, v.address, 48) {
            Ok(bytes) => bytes.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect::<Vec<_>>().join(" "),
            Err(e) => format!("{e:#}"),
        }
    }

    pub(crate) fn draw(&mut self, b: &'static Backend, vertices: u32, instances: u32, first: u32, first_instance: u32) {
        if let Some(cmd) = self.prepare_draw(b, false) {
            // SAFETY: rendering open, pipeline bound.
            unsafe { b.host.device.cmd_draw(cmd, vertices, instances, first, first_instance) };
            self.drawn(&format!("draw {vertices}x{instances}"));
        }
    }

    /// Diagnostics (`VICTORY_VK_TRACE_DRAWS=<part of a program's name>,...`):
    /// every draw of such a program in the first dumped frame, as a line:
    /// its programs, what was drawn, the title's indirect records as the
    /// CPU reads them, scissor, depth and blend state, targets and roots.
    fn trace_draw(&mut self, what: &str) {
        if !cfg!(feature = "diagnostics") {
            return;
        }
        let indirect = self.last_indirect.take();
        let Some((shaders, state)) = crate::pipeline::parts(self.pipeline) else { return };
        if !super::targets::traced(|| shaders.iter().map(|s| s.name.clone()).collect()) {
            return;
        }
        let Some(b) = super::backend() else { return };
        let words = |at: u64, n: u64| {
            super::read_title(b, at, 4 * n).map_or_else(
                |_| "unreadable".into(),
                |bytes| format!("{:?}", bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>()),
            )
        };
        let records = indirect.map_or_else(String::new, |(src, count)| {
            format!("records at {src:#x} = {} count {}", words(src, 6), count.map_or_else(|| "none".into(), |c| format!("at {c:#x} = {}", words(c, 1))))
        });
        let depth = state.map_or_else(String::new, |gs| {
            let z = &gs.depth;
            format!("test {} write {} func {} stencil {} blend {}", z.DepthEnable.as_bool(), z.DepthWriteMask.0, z.DepthFunc.0, z.StencilEnable.as_bool(), gs.blend.RenderTarget[0].BlendEnable.as_bool())
        });
        tracing::info!(
            programs = %shaders.iter().map(|s| format!("{} {}", s.name, s.hash)).collect::<Vec<_>>().join(" + "),
            what,
            records,
            scissor = ?self.scissors.first().map(|r| (r.offset.x, r.offset.y, r.extent.width, r.extent.height)),
            state = depth,
            targets = %self.attached.iter().map(|i| format!("{:#x}", i.key.address)).collect::<Vec<_>>().join(" "),
            roots = %self.graphics.slots.iter().take(6).map(|r| r.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(":")).collect::<Vec<_>>().join(" "),
            "Vulkan: a traced draw"
        );
    }

    /// Counts a recorded draw, on the attachments too; diagnostics builds
    /// describe the first ones.
    fn drawn(&mut self, what: &str) {
        super::passes::note(|| format!("did {what} index {:x?} viewport {:?} scissor {:?}", self.index.map(|i| (i.address, i.size, i.format)), self.viewports.first().map(|v| (v.x, v.y, v.width, v.height, v.min_depth, v.max_depth)), self.scissors.first().map(|r| (r.offset.x, r.offset.y, r.extent.width, r.extent.height))));
        if super::passes::open()
            && let Some(b) = super::backend()
        {
            // what the captured draw left in its colour targets (the next
            // draw opens rendering again)
            self.end_pass(b);
            if let Some(cmd) = self.cmd.as_ref().map(|r| r.cmd) {
                // SAFETY: outside rendering, in our command buffer.
                unsafe { super::passes::targets(b, cmd, &self.attached, "after") };
            }
        }
        super::passes::close();
        self.note(|| what.to_owned());
        self.trace_draw(what);
        let n = STATS[0].fetch_add(1, Ordering::Relaxed);
        if cfg!(feature = "diagnostics")
            && self.last_push[12] != 0
            && let Some(b) = super::backend()
            && !super::map_reaches(b, u64::from(self.last_push[12]))
        {
            STATS[8].fetch_add(1, Ordering::Relaxed);
        }
        for i in &self.attached {
            i.written[0].fetch_add(1, Ordering::Relaxed);
        }
        // Diagnostics: the first two draws of each of the first 64 pipelines.
        static SEEN: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
        let describe = cfg!(feature = "diagnostics") && {
            let mut seen = lock(&SEEN);
            let m = seen.get_or_insert_with(HashMap::new);
            let known = m.len();
            let c = m.entry(self.pipeline as usize).or_insert(0);
            *c += 1;
            *c <= 2 && known < 64
        };
        let _ = n;
        if describe {
            let names = crate::pipeline::parts(self.pipeline)
                .map(|(s, _)| s.iter().map(|s| format!("{}:{}", s.stage, s.name)).collect::<Vec<_>>().join(" "))
                .unwrap_or_default();
            let targets: Vec<String> = self.attached.iter().map(|i| format!("{}@{:#x} {:?} {}x{}", i.id, i.key.address, i.format, i.extent.width, i.extent.height)).collect();
            let push: Vec<String> = self.last_push.iter().enumerate().filter(|(_, w)| **w != 0).map(|(k, w)| format!("{k}:{w:#x}")).collect();
            tracing::info!(
                what,
                push = ?push,
                user_data = %self.last_user,
                layout = %self.last_layout,
                table0 = %self.table_text(),
                vertex_bytes = %self.vertex_bytes(),
                attributes = self.last_inputs.0 % 1000,
                bindings = self.last_inputs.0 / 1000,

                bindings_found = self.last_inputs.1,
                vertex_views = ?self.vertex.iter().take(4).map(|v| v.map(|v| (format!("{:#x}", v.address), v.size, v.stride))).collect::<Vec<_>>(),
                index = ?self.index.map(|i| (format!("{:#x}", i.address), i.size, i.format)),
                pipeline = %names,
                viewports = ?self.viewports.iter().map(|v| (v.x, v.y, v.width, v.height)).collect::<Vec<_>>(),
                scissors = ?self.scissors.iter().map(|s| (s.offset.x, s.offset.y, s.extent.width, s.extent.height)).collect::<Vec<_>>(),
                topology = self.topology,
                sample_mask = format_args!("{:#x}", crate::pipeline::parts(self.pipeline).and_then(|(_, g)| g).map_or(0, |g| g.sample_mask)),
                targets = ?targets,
                "Vulkan: a draw recorded"
            );
        }
    }

    pub(crate) fn draw_indexed(&mut self, b: &'static Backend, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
        let Some(index) = self.index else {
            skipped(false);
            return;
        };
        let Some((buffer, offset)) = super::buffer_over(b, index.address, u64::from(index.size)) else {
            skipped(false);
            return;
        };
        if let Some(cmd) = self.prepare_draw(b, false) {
            let ty = if index.format == 57 { vk::IndexType::UINT16 } else { vk::IndexType::UINT32 };
            // SAFETY: rendering open, pipeline bound; a title buffer.
            unsafe {
                b.host.device.cmd_bind_index_buffer(cmd, buffer, offset, ty);
                b.host.device.cmd_draw_indexed(cmd, indices, instances, first, base, first_instance);
            }
            self.drawn(&format!("indexed {indices}x{instances}"));
        }
    }

    pub(crate) fn draw_mesh(&mut self, b: &'static Backend, x: u32, y: u32, z: u32) {
        let Some(mesh) = super::mesh(b) else {
            skipped(false);
            return;
        };
        // A dispatch of no groups draws nothing (it ran one group before);
        // a split program takes `parts` host groups in y per console group;
        // counts stay within the device's mesh limits.
        let Some(parts) = self.mesh_parts(b) else { return };
        let l = super::stages::limits(b);
        let y_most = l.mesh_groups[1] / parts * parts;
        let [x, y, z] = [x.min(l.mesh_groups[0]), y.saturating_mul(parts).min(y_most), z.min(l.mesh_groups[2])];
        if x == 0 || y == 0 || z == 0 || u64::from(x) * u64::from(y) * u64::from(z) > u64::from(l.mesh_groups_total) {
            skipped(false);
            return;
        }
        if let Some(cmd) = self.prepare_draw(b, true) {
            // SAFETY: rendering open, a mesh pipeline bound.
            unsafe { mesh.cmd_draw_mesh_tasks(cmd, x, y, z) };
            self.drawn(&format!("mesh {x}x{y}x{z}"));
        }
    }

    pub(crate) fn dispatch(&mut self, b: &'static Backend, x: u32, y: u32, z: u32) {
        if let Some(cmd) = self.prepare_dispatch(b) {
            let mut lap = super::Lap::start(57);
            self.watch_cull(b, Some([x, y, z]), None);
            lap.next();
            // SAFETY: outside rendering, the pipeline bound.
            unsafe { b.host.device.cmd_dispatch(cmd, x, y, z) };
            self.ambient(b, cmd);
            lap.next();
            STATS[1].fetch_add(1, Ordering::Relaxed);
            let ud = [self.last_push[9], self.last_push[10], self.last_push[11], self.last_push[12]];
            self.note_as(|| format!("dispatch {x}x{y}x{z} ud5-8={:08x} {:08x} {:08x} {:08x}", ud[0], ud[1], ud[2], ud[3]), true);
            super::passes::note(|| format!("dispatch {x} {y} {z}"));
            super::passes::close();
        }
    }

    /// After a dispatch of a program the stand-in for indirect light
    /// follows (`ambient.rs`): its own dispatches, after which the title's
    /// pipeline and push constants are set again.
    fn ambient(&mut self, b: &'static Backend, cmd: vk::CommandBuffer) {
        if super::ambient::on()
            && let Some((shaders, _)) = crate::pipeline::parts(self.pipeline)
            && let Some(role) = shaders.iter().find_map(|s| super::ambient::role(&s.name))
            // SAFETY: outside rendering, in our command buffer, after `prepare_dispatch`.
            && unsafe { super::ambient::after(b, cmd, role, &self.compute.slots) }
        {
            self.bound_compute = vk::Pipeline::null();
            self.pushed = false;
        }
    }

    /// Everything a dispatch needs bound; `None` (counted) if it is
    /// skipped. What it writes may be indirect records: no window holds.
    fn prepare_dispatch(&mut self, b: &'static Backend) -> Option<vk::CommandBuffer> {
        let mut lap = super::Lap::start(48);
        let Some((shaders, _)) = crate::pipeline::parts(self.pipeline) else {
            skipped(false);
            return None;
        };
        // The ray-tracing structure builder (the GDK's XboxBvhBuilder.dll:
        // compute programs without a program name) is not run unless asked
        // for: partly working, it writes garbage structures into title
        // memory (wild light, and the title crashing on what it reads
        // back). Without structures every ray of the title misses.
        if !bvh_builder() && shaders.iter().all(|s| s.name.is_empty()) {
            BUILDER_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        lap.next();
        let stage = match stage_of(b, shaders, "compute") {
            Ok(Some(s)) => s,
            Ok(None) => {
                skipped(false);
                return None;
            }
            Err(waiting) => {
                skipped(waiting);
                return None;
            }
        };
        let Some(pipeline) = stage.part.filter(|_| stage.stage == HwStage::Cs) else {
            skipped(false);
            return None;
        };
        self.guard_id = stage.guard_id;
        lap.next();
        self.end_pass(b);
        let cmd = self.cmd(b)?;
        lap.next();
        for s in shaders {
            if super::targets::wants(&s.name) {
                // SAFETY: outside rendering, in our command buffer.
                unsafe { super::targets::before(b, cmd, &s.name) };
                super::targets::tables(b, &s.name, &self.compute.slots);
                let programs: Vec<_> = shaders.iter().map(|p| (p.name.clone(), p.hash.clone(), p.stage)).collect();
                // SAFETY: as above.
                unsafe { super::passes::before(b, cmd, &programs, &s.name, &self.compute.slots) };
            }
        }
        lap.next();
        self.indirect.clear();
        self.planes_synced.clear();
        if let Some(h) = super::descriptors::heaps(b) {
            h.immediates(b);
        }
        lap.next();
        self.refresh_map(b, cmd);
        lap.next();
        // SAFETY: outside rendering, in our command buffer.
        unsafe {
            if self.bound_compute != pipeline {
                b.host.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.bound_compute = pipeline;
                super::gputime::mark(b, cmd, || shaders.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join("+"));
            }
            lap.next();
            self.push(b, cmd, &[&stage], true);
            lap.next();
        }
        Some(cmd)
    }

}

/// `VICTORY_BVH_BUILDER=1`: the title's ray-tracing structure builder runs.
pub(super) fn bvh_builder() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v != "0"))
}

/// Dispatches of the structure builder left out (the 10 s line).
pub(crate) static BUILDER_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// What the recording does not do yet (counted by kind in the 10 s line).
#[derive(Clone, Copy)]
pub(crate) enum NotYet {
    Indirect,
    Msaa,
    /// Tessellation, geometry and amplification stages.
    Stages,
    Topology,
}

static NOT_YET: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

impl NotYet {
    fn text() -> String {
        let n = |i: usize| NOT_YET[i].load(Ordering::Relaxed);
        format!("indirect={} msaa={} stages={} topology={}", n(0), n(1), n(2), n(3))
    }
}

fn not_yet(why: NotYet) {
    STATS[4].fetch_add(1, Ordering::Relaxed);
    NOT_YET[why as usize].fetch_add(1, Ordering::Relaxed);
}

/// The console's rectangle list: three corners a rectangle.
const RECTANGLES: u32 = 17;

/// A D3D topology as Vulkan's (mesh programs draw triangles; rectangle
/// lists go as triangle lists to the rectangle geometry stage).
fn topology(t: u32, mesh: bool) -> Option<vk::PrimitiveTopology> {
    use vk::PrimitiveTopology as P;
    Some(match t {
        1 => P::POINT_LIST,
        2 => P::LINE_LIST,
        3 => P::LINE_STRIP,
        4 | RECTANGLES => P::TRIANGLE_LIST,
        5 => P::TRIANGLE_STRIP,
        6 => P::TRIANGLE_FAN,
        _ if mesh => P::TRIANGLE_LIST,
        _ => {
            once_topology(t);
            return None;
        }
    })
}

/// Diagnostics: each topology the recording does not draw, once.
fn once_topology(t: u32) {
    static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    let mut seen = lock(&SEEN);
    if !seen.contains(&t) {
        seen.push(t);
        tracing::info!(topology = t, "Vulkan: a topology not drawn yet (counted under not_yet topology)");
    }
}


/// The pre-rasterisation and pixel stages of a draw, ready; `None` (counted)
/// if the draw is skipped.
fn draw_stages(b: &'static Backend, shaders: &[Arc<crate::pipeline::Shader>], mesh: bool) -> Option<(Arc<Stage>, Option<Arc<Stage>>)> {
    if shaders.iter().any(|s| matches!(s.stage, "hull" | "domain" | "geometry" | "amplification")) {
        not_yet(NotYet::Stages);
        return None;
    }
    let pre = stage_of(b, shaders, if mesh { "mesh" } else { "vertex" }).map_err(skipped).ok()?;
    let Some(pre) = pre else {
        skipped(false);
        return None;
    };
    let pixel = stage_of(b, shaders, "pixel").map_err(skipped).ok()?;
    Some((pre, pixel))
}

/// The recording's part of the 10 s line.
/// Draws left out because no target of theirs has an image.
static NO_PASS: AtomicU64 = AtomicU64::new(0);
/// Copies of a depth buffer's stored plane into its depth plane.
static PLANE_COPIES: AtomicU64 = AtomicU64::new(0);

/// Diagnostics: programs (stage, guard id) pushed a root view whose range
/// is short -> times.
static SHORT_ROOTS: Mutex<std::collections::BTreeMap<String, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// Diagnostics: the root views a push carries, by how much of their
/// memory-map range is left.
fn count_root_views(b: &Backend, stages: &[&Stage], root: &Root) {
    for st in stages {
        for ud in st.user_data.iter().filter(|u| matches!(u.root, RootType::Descriptor) && u.dword == 0) {
            let at = u64::from(root.word(ud.slot, 0)) | u64::from(root.word(ud.slot, 1)) << 32;
            if at != 0 && super::root_view_short(b, at) {
                let mut names = lock(&SHORT_ROOTS);
                if names.len() < 4096 {
                    *names.entry(format!("{:?} {:#x}", st.stage, st.guard_id)).or_insert(0) += 1;
                }
            }
        }
    }
}

/// The ten programs most often pushed a short root view.
pub(crate) fn short_root_programs() -> String {
    let mut all: Vec<(String, u64)> = lock(&SHORT_ROOTS).iter().map(|(k, &v)| (k.clone(), v)).collect();
    all.sort_by_key(|p| std::cmp::Reverse(p.1));
    format!("draws without a pass {}; depth plane copies {}; {} programs; {}", NO_PASS.load(Ordering::Relaxed), PLANE_COPIES.load(Ordering::Relaxed), all.len(), all.iter().take(10).map(|(k, n)| format!("{k} x{n}")).collect::<Vec<_>>().join(", "))
}

pub(crate) fn report() {
    let s: Vec<u64> = STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let v: Vec<u64> = super::descriptors::STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let i: Vec<u64> = images::STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    tracing::info!(
        draws = s[0],
        dispatches = s[1],
        builder_dispatches_left_out = BUILDER_SKIPPED.load(Ordering::Relaxed),
        skipped_not_ready = s[2],
        skipped_no_pipeline = s[3],
        not_yet = s[4],
        not_yet_reasons = %NotYet::text(),
        indirect = %indirect::text(),
        command_buffers = s[5],
        links = s[6],
        maps_rebound = s[7],
        draws_table_unmapped = s[8],
        views = v[0],
        samplers = v[1],
        views_without_image = v[2],
        images = images::count(),
        images_made = i[0],
        fills = i[1],
        filled_mb = i[2] >> 20,
        images_refused = i[3],
        refills_by_fingerprint = i[4],
        gpu_vs = super::gpu_stats::SUMS[0].load(Ordering::Relaxed),
        gpu_clip_in = super::gpu_stats::SUMS[1].load(Ordering::Relaxed),
        gpu_clip_out = super::gpu_stats::SUMS[2].load(Ordering::Relaxed),
        gpu_fs = super::gpu_stats::SUMS[3].load(Ordering::Relaxed),
        gpu_counted_buffers = super::gpu_stats::SUMS[4].load(Ordering::Relaxed),
        "Vulkan: draws recorded"
    );
}
