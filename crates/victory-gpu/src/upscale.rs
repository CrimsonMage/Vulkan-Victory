//! The title's upscale (AMD FidelityFX, `ffxDispatch` with an upscale
//! description), the plain PC way: the render-resolution colour scaled
//! into the output with a linear filter, in the title's list where it asked
//! for the upscale.
//!
//! On Series S the title renders the world at 1280x720 and has the AMD
//! upscaler (FSR4, `amd_fidelityfx_upscaler_xbox.dll`) bring it to
//! 1920x1080; everything after (post-processing, tone mapping, the scene
//! plane) reads the upscaler's output. The upscaler's programs ship only
//! as ACO-compiled console code (an `ACCC` part, no DXIL part): its
//! `DeserializeComputePipelineStateX` path fails here, its fallback hands
//! the same containers to `CreateComputePipelineState`, none of its 29
//! passes translates, and its output was never written: the world showed
//! black, the tone-mapper's dither on zero (run s4, 2026-11-25). Series X
//! uses the title's own temporal pass instead, which runs.
//!
//! shortcut: no temporal accumulation and no sharpening; the scene keeps
//! its sub-pixel jitter. The PC twin (the FidelityFX DX12 runtime) is not
//! with the title.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, compile, host};
use crate::object::This;

/// `FFX_API_DISPATCH_DESC_TYPE_UPSCALE`.
pub const UPSCALE: u64 = 0x0001_0001;

/// `FfxApiResource` (FidelityFX API `ffx_api_types.h`): the resource, its
/// description (type, format, width, height, depth, mips, flags, usage)
/// and its state.
#[repr(C)]
pub struct ApiResource {
    pub resource: *mut std::ffi::c_void,
    pub description: [u32; 8],
    pub state: u32,
}

/// `ffxDispatchDescUpscale` as the FidelityFX API's `ffx_upscale.h` lays
/// it out (the same from FSR 3.1, the API's first upscaler, to FSR 4; the
/// caller checks the upscaler's version).
#[repr(C)]
pub struct DispatchDescUpscale {
    pub kind: u64,
    pub next: *const std::ffi::c_void,
    pub command_list: *mut std::ffi::c_void,
    pub color: ApiResource,
    pub depth: ApiResource,
    pub motion_vectors: ApiResource,
    pub exposure: ApiResource,
    pub reactive: ApiResource,
    pub transparency_and_composition: ApiResource,
    pub output: ApiResource,
    pub jitter_offset: [f32; 2],
    pub motion_vector_scale: [f32; 2],
    pub render_size: [u32; 2],
    pub upscale_size: [u32; 2],
    pub enable_sharpening: bool,
    pub sharpness: f32,
    pub frame_time_delta: f32,
    pub pre_exposure: f32,
    pub reset: bool,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub flags: u32,
}
const _: () = {
    use std::mem::offset_of;
    assert!(size_of::<ApiResource>() == 0x30);
    assert!(offset_of!(DispatchDescUpscale, color) == 0x18);
    assert!(offset_of!(DispatchDescUpscale, output) == 0x138);
    assert!(offset_of!(DispatchDescUpscale, render_size) == 0x178);
    assert!(offset_of!(DispatchDescUpscale, flags) == 0x1ac);
    assert!(size_of::<DispatchDescUpscale>() == 0x1b0);
};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants = 8, b0), \
    DescriptorTable(SRV(t0), UAV(u0)), \
    StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, \
        addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"
cbuffer Scale : register(b0) { float2 extent; float2 last; uint2 size; };
Texture2DArray<float4> color : register(t0);
RWTexture2DArray<float4> output : register(u0);
SamplerState linear_clamp : register(s0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    if (at.x >= size.x || at.y >= size.y) return;
    // The rendered rectangle only: not past the centre of its last texels.
    float2 uv = min((float2(at.xy) + 0.5) / float2(size) * extent, last);
    output[uint3(at.xy, 0)] = color.SampleLevel(linear_clamp, float3(uv, 0.0), 0.0);
}
"#;

/// View pairs, used in turn: far more than the upscales of the lists in
/// flight (one a frame).
const SLOTS: u32 = 256;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let code = compile(SOURCE, s!("main"), s!("cs_5_1"))?;
    // SAFETY: the root signature is inside the shader; plain creations.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        })?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS * 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Pass {
            root,
            pso,
            views,
            step: device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        make(&host()?.device)
            .inspect_err(|e| tracing::warn!(error = %e, "no pass for the title's upscale"))
            .ok()
    })
    .as_ref()
}

/// The typed format a view of `format` reads or writes with: its family's
/// float or unsigned-normalized one for a typeless format.
fn typed(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format.0 {
        1 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        9 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        23 => DXGI_FORMAT_R10G10B10A2_UNORM,
        27 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 => DXGI_FORMAT_B8G8R8A8_UNORM,
        _ => format,
    }
}

/// A one-sample 2D texture of one array slice: what the pass reads and
/// writes.
fn plain(desc: &D3D12_RESOURCE_DESC) -> bool {
    desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.SampleDesc.Count == 1
        && desc.DepthOrArraySize == 1
}

/// `width` x `height` if both are set and fit in `w` x `h`, else all of it.
pub(crate) fn within((width, height): (u32, u32), w: u32, h: u32) -> (u32, u32) {
    if width == 0 || height == 0 || width > w || height > h {
        (w, h)
    } else {
        (width, height)
    }
}

/// Runs the upscale `desc` describes as a linear scale. False, recording
/// nothing, if its list or textures are not ours or not of a shape the
/// pass takes: the title's upscaler then runs as before.
///
/// # Safety
///
/// `desc` is an `ffxDispatchDescUpscale` (its type checked by the caller),
/// as the title passes it to `ffxDispatch`, on the thread recording its
/// command list.
pub unsafe fn dispatch(desc: *const u8) -> bool {
    // SAFETY: as the caller says.
    let d = unsafe { &*desc.cast::<DispatchDescUpscale>() };
    let (list, color, output) = (
        d.command_list as This,
        d.color.resource as This,
        d.output.resource as This,
    );
    let (render, upscale) = ((d.render_size[0], d.render_size[1]), (d.upscale_size[0], d.upscale_size[1]));
    if !crate::resource::is_resource(color) || !crate::resource::is_resource(output) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::error!("the title's upscale is left to its upscaler: its colour or output is no resource of ours"));
        return false;
    }
    if crate::vulkan::on() {
        return crate::vulkan::record::upscale::dispatch(list, color, output, render, upscale);
    }
    let (Some(host), Some(pass)) = (host(), pass()) else {
        return false;
    };
    let (Some(from), Some(to)) = (
        crate::resource::host_resource(color),
        crate::resource::host_resource(output),
    ) else {
        return false;
    };
    // SAFETY: live resources.
    let (a, b) = unsafe { (from.GetDesc(), to.GetDesc()) };
    const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
    if !plain(&a) || !plain(&b) || b.Flags.0 & UNORDERED == 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                color = ?a,
                output = ?b,
                "the title's upscale is left to its upscaler: textures of another shape"
            );
        });
        return false;
    }
    let (cw, ch) = (a.Width as u32, a.Height);
    let (rw, rh) = within(render, cw, ch);
    let (ow, oh) = within(upscale, b.Width as u32, b.Height);
    // (Run later, by the worker that translates the list: `move`.)
    crate::list::record_own(list, move |list: &HostList| {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                from = format_args!("{rw}x{rh} of {cw}x{ch} f{}", a.Format.0),
                to = format_args!("{ow}x{oh} of {}x{} f{}", b.Width, b.Height, b.Format.0),
                "the title's upscale (AMD FidelityFX) runs as a linear scale"
            );
        });
        // As for a texture a shader reads and one it writes.
        if crate::texture::title_written(from) {
            crate::texture::upload(list, color, from);
        }
        list.twin_use(color, false, false);
        crate::list::ensure_state(list, color, 0x40 | 0x80);
        crate::list::ensure_state(list, output, 0x8);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            format!("upscale as a linear scale: {color:?} {rw}x{rh} into {output:?} {ow}x{oh}")
        });
        let slot = pass.next.fetch_add(1, Ordering::Relaxed) % SLOTS;
        let l = &list.list;
        // SAFETY: an open list; live textures in the states set above;
        // views in slots no list in flight reads; the pass's objects live
        // as long as the process.
        unsafe {
            let cpu = pass.views.GetCPUDescriptorHandleForHeapStart().ptr
                + (slot * 2 * pass.step) as usize;
            let gpu = pass.views.GetGPUDescriptorHandleForHeapStart().ptr
                + u64::from(slot * 2 * pass.step);
            host.device.CreateShaderResourceView(
                from,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: typed(a.Format),
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                            MipLevels: 1,
                            ArraySize: 1,
                            ..Default::default()
                        },
                    },
                }),
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu },
            );
            host.device.CreateUnorderedAccessView(
                to,
                None::<&ID3D12Resource>,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: typed(b.Format),
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                            ArraySize: 1,
                            ..Default::default()
                        },
                    },
                }),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu + pass.step as usize,
                },
            );
            l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
            l.SetComputeRootSignature(&pass.root);
            l.SetPipelineState(&pass.pso);
            let extent = [rw as f32 / cw as f32, rh as f32 / ch as f32];
            let last = [(rw as f32 - 0.5) / cw as f32, (rh as f32 - 0.5) / ch as f32];
            let constants: [u32; 8] = [
                extent[0].to_bits(),
                extent[1].to_bits(),
                last[0].to_bits(),
                last[1].to_bits(),
                ow,
                oh,
                0,
                0,
            ];
            l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
            l.SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu });
            l.Dispatch(ow.div_ceil(8), oh.div_ceil(8), 1);
            l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }]);
        }
        true
    })
}

#[cfg(test)]
mod tests {
    /// The pass compiles with the system's compiler and the host takes it.
    #[test]
    fn upscale_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        super::make(&host.device).expect("the upscale pass");
    }

    #[test]
    fn sizes_outside_the_texture_mean_all_of_it() {
        assert_eq!(super::within((0, 0), 1280, 720), (1280, 720));
        assert_eq!(super::within((1920, 1080), 1280, 720), (1280, 720));
        assert_eq!(super::within((1152, 648), 1280, 720), (1152, 648));
    }
}
