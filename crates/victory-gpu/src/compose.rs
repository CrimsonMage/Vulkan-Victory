//! The console's display planes, drawn into the swap chain's back buffer
//! the PC way: each plane's rectangle scaled (linear filter) to its place,
//! the first plane as it is, later ones over it with premultiplied alpha
//! (one, one minus source alpha: the blend the title uses itself to put
//! its interface over the scene when it composes on Series X).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::{Shown, compile, host};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants = 4, b0), \
    DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL), \
    StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, \
        addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, \
        visibility = SHADER_VISIBILITY_PIXEL)"
cbuffer Shown : register(b0) { float4 shown; }
Texture2DArray<float4> plane : register(t0);
SamplerState linear_clamp : register(s0);
struct Corner { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
Corner corner(uint i : SV_VertexID) {
    float2 t = float2(float(i & 1), float(i >> 1)) * 2.0;
    Corner c;
    c.position = float4(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0, 0.0, 1.0);
    c.uv = lerp(shown.xy, shown.zw, t);
    return c;
}
[RootSignature(RS)]
float4 texel(Corner c) : SV_Target {
    return plane.SampleLevel(linear_clamp, float3(c.uv, 0.0), 0.0);
}
"#;

/// Views of planes, in a shader-visible heap of ours used as a ring.
/// shortcut: a view must last until its present has run; two planes a
/// present and a few presents in the air are far from coming round.
const VIEWS: u32 = 256;

struct Pass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    /// Per back-buffer format: the root signature and the pipelines for
    /// the first plane and for planes over it (`None`: the host refused).
    made: Mutex<Vec<(i32, Option<Pipelines>)>>,
}

type Pipelines = (
    ID3D12RootSignature,
    ID3D12PipelineState,
    ID3D12PipelineState,
);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let device = &host()?.device;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(Pass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

impl Pass {
    fn pipelines(&self, format: DXGI_FORMAT) -> Option<Pipelines> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, pipelines)) = made.iter().find(|m| m.0 == format.0) {
            return pipelines.clone();
        }
        let pipelines = match make(format) {
            Ok(pipelines) => Some(pipelines),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    format = format.0,
                    "no pipeline to draw the title's planes: only the first plane is shown, as it is"
                );
                None
            }
        };
        made.push((format.0, pipelines.clone()));
        pipelines
    }
}

fn make(format: DXGI_FORMAT) -> windows::core::Result<Pipelines> {
    use windows::core::s;
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let vs = compile(SOURCE, s!("corner"), s!("vs_5_1"))?;
    let ps = compile(SOURCE, s!("texel"), s!("ps_5_1"))?;
    let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    };
    // SAFETY: the root signature is inside the shaders; the descriptions
    // point at live data and never release their pointers.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = format;
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: code(&vs[..]),
            PS: code(&ps[..]),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: formats,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
            D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let first: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
        desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let over: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
        Ok((root, first, over))
    }
}

/// The format a plane is read as: its family's plain unsigned-normalized
/// or float format (bytes as they are, as a copy shows them), `None` for
/// a format the display cannot show.
fn read_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        88 | 92 | 93 => DXGI_FORMAT_B8G8R8X8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        11 => DXGI_FORMAT_R16G16B16A16_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

/// A plane's rectangle as texture coordinates (left, top, right, bottom):
/// the one the title names if it lies in the texture, else all of it.
fn shown_rectangle(plane: &Shown, width: u64, height: u32) -> [f32; 4] {
    match plane.source {
        Some([l, t, r, b])
            if 0 <= l
                && l < r
                && i64::from(r) <= width as i64
                && 0 <= t
                && t < b
                && b as u32 <= height =>
        {
            [
                l as f32 / width as f32,
                t as f32 / height as f32,
                r as f32 / width as f32,
                b as f32 / height as f32,
            ]
        }
        _ => [0.0, 0.0, 1.0, 1.0],
    }
}

/// Where a plane goes (left, top, right, bottom as fractions of the
/// screen): the placement the title names if it is one, else all of it.
fn placement(plane: &Shown) -> [f32; 4] {
    match plane.placement {
        Some([l, t, r, b])
            if (0.0..1.0).contains(&l)
                && l < r
                && r <= 1.0
                && (0.0..1.0).contains(&t)
                && t < b
                && b <= 1.0 =>
        {
            [l, t, r, b]
        }
        _ => [0.0, 0.0, 1.0, 1.0],
    }
}

/// Whether `plane` is shown whole over the whole screen (copied as it is
/// when it is the only one).
pub(crate) fn whole(plane: &Shown) -> bool {
    // SAFETY: a live resource.
    let desc = unsafe { plane.texture.GetDesc() };
    shown_rectangle(plane, desc.Width, desc.Height) == [0.0, 0.0, 1.0, 1.0]
        && placement(plane) == [0.0, 0.0, 1.0, 1.0]
}

/// Whether the pass can draw into a back buffer of `format` (its
/// pipelines made now if they were not).
pub(crate) fn ready(format: DXGI_FORMAT) -> bool {
    pass().and_then(|pass| pass.pipelines(format)).is_some()
}

/// The state planes are read in by the pass.
pub(crate) const READ: u32 = 0x40 | 0x80;

/// Whether the pass can read `texture` (shader reads are allowed on it,
/// and its format is one the display shows).
pub(crate) fn readable(texture: &ID3D12Resource) -> bool {
    // SAFETY: a live resource.
    let desc = unsafe { texture.GetDesc() };
    desc.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
        && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.SampleDesc.Count == 1
        && read_format(desc.Format).is_some()
}

/// Records into `list` the planes drawn into `back` (in the render-target
/// state, `width` x `height` of `format`), over black: the planes in
/// [`READ`] and [`readable`]. Returns false, recording nothing, if the
/// host gave no pipeline. The pass sets the list's descriptor heap, root
/// signature, pipeline, target, viewport, scissor and topology.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    back: &ID3D12Resource,
    format: DXGI_FORMAT,
    (width, height): (u32, u32),
    planes: &[&Shown],
) -> bool {
    let (Some(host), Some(pass)) = (host(), pass()) else {
        return false;
    };
    let Some((root, first, over)) = pass.pipelines(format) else {
        return false;
    };
    let target = host.rtv(Some(back), None);
    // SAFETY: an open list; slots of our heaps; live resources and objects
    // (the pass's own live as long as the process).
    unsafe {
        list.ClearRenderTargetView(target, &[0.0, 0.0, 0.0, 1.0], None);
        list.OMSetRenderTargets(1, Some(&target), false, None);
        list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        list.SetGraphicsRootSignature(&root);
        list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(
            4,
        ));
        for (i, plane) in planes.iter().enumerate() {
            let desc = plane.texture.GetDesc();
            let Some(read) = read_format(desc.Format) else {
                continue;
            };
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: read,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        FirstArraySlice: 0,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            host.device.CreateShaderResourceView(
                &plane.texture,
                Some(&view),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: pass.cpu + (slot * pass.step) as usize,
                },
            );
            let [l, t, r, b] = placement(plane);
            let (w, h) = (width as f32, height as f32);
            let area = RECT {
                left: (l * w).round() as i32,
                top: (t * h).round() as i32,
                right: (r * w).round() as i32,
                bottom: (b * h).round() as i32,
            };
            list.SetPipelineState(if i == 0 { &first } else { &over });
            list.SetGraphicsRoot32BitConstants(
                0,
                4,
                shown_rectangle(plane, desc.Width, desc.Height)
                    .as_ptr()
                    .cast(),
                0,
            );
            list.SetGraphicsRootDescriptorTable(
                1,
                D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: pass.gpu + u64::from(slot * pass.step),
                },
            );
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: area.left as f32,
                TopLeftY: area.top as f32,
                Width: (area.right - area.left) as f32,
                Height: (area.bottom - area.top) as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[area]);
            list.DrawInstanced(3, 1, 0, 0);
        }
    }
    true
}
