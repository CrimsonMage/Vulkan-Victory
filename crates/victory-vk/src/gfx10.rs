//! The console's descriptors as its shaders read them (`docs/vulkan-abi.md`
//! sections 4 and 5): T# (image, 8 dwords), S# (sampler, 4 dwords) and V#
//! (buffer, 4 dwords), written by the host at `Create*View` and
//! `CreateSampler` into the title's heap slots, with the host's words where
//! the ABI puts them (T# WORD7: slot, integer, storage; S# WORD3: slot).
//!
//! Ported from Leonida (GPL-3.0-or-later, `runtime/src/gfx10_descriptors.h`),
//! which takes field positions and values from Mesa's register database
//! (`src/amd/registers/gfx10-rsrc.json`, gfx103 variants), `gfx6.json` and
//! `ac_descriptors.c` (MIT). Only what a Vulkan view or sampler needs is
//! encoded; other bits (tiling, compression, tuning) are zero.

use ash::vk;

/// DST_SEL: what one channel selects (SQ_SEL_XYZW01).
pub type Swizzle = [u8; 4];
pub const SEL_0: u8 = 0;
pub const SEL_1: u8 = 1;
pub const SEL_X: u8 = 4;
pub const SEL_Y: u8 = 5;
pub const SEL_Z: u8 = 6;
pub const SEL_W: u8 = 7;
pub const IDENTITY: Swizzle = [SEL_X, SEL_Y, SEL_Z, SEL_W];

const fn field(value: u32, low: u32, width: u32) -> u32 {
    (value & ((1u32 << width) - 1)) << low
}

fn unsigned_fixed(v: f32, lo: f32, hi: f32) -> u32 {
    let c = if v > lo {
        if v < hi { v } else { hi }
    } else {
        lo
    };
    (c * 256.0) as u32
}

fn signed_fixed(v: f32, lo: f32, hi: f32) -> u32 {
    let c = if v > lo {
        if v < hi { v } else { hi }
    } else {
        lo
    };
    (c * 256.0) as i32 as u32
}

// ---- T# ----

/// SQ_RSRC_IMG_TYPE (WORD3.TYPE); 0..7 mark a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
pub enum ImageType {
    D1 = 8,
    D2 = 9,
    D3 = 10,
    Cube = 11,
    D1Array = 12,
    D2Array = 13,
    D2Msaa = 14,
    D2MsaaArray = 15,
}

/// What a T# says.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Image {
    /// The title's address of the image (kept in 256-byte units).
    pub address: u64,
    /// GFX10_FORMAT code.
    pub format: u32,
    pub kind: ImageType,
    pub width: u32,
    pub height: u32,
    /// Of mip 0, for 3D images.
    pub depth: u32,
    pub first_layer: u32,
    pub layer_count: u32,
    pub first_mip: u32,
    pub mip_count: u32,
    pub resource_mips: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
    pub min_lod: f32,
}

/// The host's word of a T# (WORD7, ABI v1 section 5.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageSlot {
    /// Index into set 0 bindings 0 and 1 (24 bits).
    pub slot: u32,
    /// The view is the format's UINT twin: the shader samples as uint.
    pub integer: bool,
    /// Binding 1 holds a storage view at the slot (a UAV).
    pub storage: bool,
}

impl Image {
    /// The eight dwords, as `ac_build_gfx10_texture_descriptor` fills them,
    /// WORD6 zero and WORD7 the host's.
    pub fn encode(&self, host: ImageSlot) -> [u32; 8] {
        let msaa = matches!(self.kind, ImageType::D2Msaa | ImageType::D2MsaaArray);
        let sample_log2 = 31 - self.samples.max(1).leading_zeros();
        let base_level = if msaa { 0 } else { self.first_mip };
        let last_level = if msaa {
            sample_log2
        } else {
            self.first_mip + self.mip_count.max(1) - 1
        };
        let max_mip = if msaa {
            sample_log2
        } else {
            self.resource_mips.max(1) - 1
        };
        let last = if self.kind == ImageType::D3 {
            self.depth.max(1) - 1
        } else {
            self.first_layer + self.layer_count.max(1) - 1
        };
        let units = self.address >> 8;
        let (w, h) = (self.width.max(1) - 1, self.height.max(1) - 1);
        let mut out = [0u32; 8];
        out[0] = units as u32;
        out[1] = field((units >> 32) as u32, 0, 8)
            | field(unsigned_fixed(self.min_lod, 0.0, 15.0), 8, 12)
            | field(self.format, 20, 9)
            | field(w, 30, 2);
        out[2] = field(w >> 2, 0, 12) | field(h, 14, 14);
        out[3] = field(u32::from(self.swizzle[0]), 0, 3)
            | field(u32::from(self.swizzle[1]), 3, 3)
            | field(u32::from(self.swizzle[2]), 6, 3)
            | field(u32::from(self.swizzle[3]), 9, 3)
            | field(base_level, 12, 4)
            | field(last_level, 16, 4)
            | field(self.kind as u32, 28, 4);
        out[4] = field(last, 0, 13) | field(self.first_layer, 16, 13);
        out[5] = field(max_mip, 4, 4);
        out[7] = field(host.slot, 0, 24)
            | field(u32::from(host.integer), 24, 1)
            | field(u32::from(host.storage), 25, 1);
        out
    }
}

// ---- S# ----

/// What an S# says (SQ_TEX_* values).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    /// SQ_TEX_CLAMP per axis: 0 wrap, 1 mirror, 2 clamp, 3 mirror once,
    /// 6 border, 7 mirror once border.
    pub address: [u32; 3],
    /// SQ_TEX_XY_FILTER: 0 point, 1 bilinear, 2 aniso point, 3 aniso bilinear.
    pub mag: u32,
    pub min: u32,
    /// SQ_TEX_MIP_FILTER: 0 none, 1 point, 2 linear.
    pub mip: u32,
    pub max_anisotropy: u32,
    /// SQ_TEX_DEPTH_COMPARE, the values of VkCompareOp; 0 with no compare.
    pub compare: u32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub lod_bias: f32,
    /// SQ_TEX_BORDER_COLOR: 0 transparent black, 1 opaque black, 2 opaque
    /// white, 3 table.
    pub border: u32,
    /// SQ_IMG_FILTER_TYPE: 0 blend, 1 minimum, 2 maximum.
    pub filter_mode: u32,
}

impl Sampler {
    /// The four dwords as `ac_build_sampler_descriptor` fills them, with
    /// the host's slot in BORDER_COLOR_PTR (ABI v1 section 5.3).
    pub fn encode(&self, slot: u32) -> [u32; 4] {
        let aniso_log2 = (31 - self.max_anisotropy.max(1).leading_zeros()).min(4);
        [
            field(self.address[0], 0, 3)
                | field(self.address[1], 3, 3)
                | field(self.address[2], 6, 3)
                | field(aniso_log2, 9, 3)
                | field(self.compare, 12, 3)
                | field(self.filter_mode, 29, 2),
            field(unsigned_fixed(self.min_lod, 0.0, 15.0), 0, 12)
                | field(unsigned_fixed(self.max_lod, 0.0, 15.0), 12, 12),
            field(signed_fixed(self.lod_bias, -32.0, 31.0), 0, 14)
                | field(self.mag, 20, 2)
                | field(self.min, 22, 2)
                | field(self.mip, 26, 2),
            field(slot, 0, 12) | field(self.border, 30, 2),
        ]
    }
}

// ---- V# ----

/// SQ_BUF_RSRC_WORD3.OOB_SELECT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Oob {
    StructuredWithOffset = 0,
    Structured = 1,
    Disabled = 2,
    Raw = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    /// The title's address of the first byte (48 bits).
    pub address: u64,
    /// Bytes a record (14 bits); 0 for a raw buffer.
    pub stride: u32,
    /// Records, or bytes with stride 0.
    pub records: u32,
    /// A buffer format code (below 128) for typed access, else 0.
    pub format: u32,
    pub swizzle: Swizzle,
    pub oob: Oob,
}

impl Buffer {
    /// A raw view of `bytes` at `address`.
    pub fn raw(address: u64, bytes: u32) -> Buffer {
        Buffer {
            address,
            stride: 0,
            records: bytes,
            format: 0,
            swizzle: IDENTITY,
            oob: Oob::Raw,
        }
    }

    /// As `ac_build_buffer_descriptor` fills it (TYPE 0, a buffer).
    pub fn encode(&self) -> [u32; 4] {
        [
            self.address as u32,
            field((self.address >> 32) as u32, 0, 16) | field(self.stride, 16, 14),
            self.records,
            field(u32::from(self.swizzle[0]), 0, 3)
                | field(u32::from(self.swizzle[1]), 3, 3)
                | field(u32::from(self.swizzle[2]), 6, 3)
                | field(u32::from(self.swizzle[3]), 9, 3)
                | field(self.format, 12, 7)
                | field(self.oob as u32, 28, 2),
        ]
    }
}

// ---- formats ----

const R: Swizzle = [SEL_X, SEL_0, SEL_0, SEL_1];
const RG: Swizzle = [SEL_X, SEL_Y, SEL_0, SEL_1];
const RGB: Swizzle = [SEL_X, SEL_Y, SEL_Z, SEL_1];
const RGBA: Swizzle = IDENTITY;
const BGR: Swizzle = [SEL_Z, SEL_Y, SEL_X, SEL_1];
const BGRA: Swizzle = [SEL_Z, SEL_Y, SEL_X, SEL_W];
const ABGR: Swizzle = [SEL_W, SEL_Z, SEL_Y, SEL_X];

/// GFX10_FORMAT code, Vulkan format, and the DST_SEL that reads the
/// format's channels as R, G, B, A. The first row of a code is the format
/// in the hardware's channel order.
const FORMATS: &[(u32, vk::Format, Swizzle)] = {
    use vk::Format as F;
    &[
        (1, F::R8_UNORM, R),
        (2, F::R8_SNORM, R),
        (3, F::R8_USCALED, R),
        (4, F::R8_SSCALED, R),
        (5, F::R8_UINT, R),
        (5, F::S8_UINT, R),
        (6, F::R8_SINT, R),
        (7, F::R16_UNORM, R),
        (7, F::D16_UNORM, R),
        (8, F::R16_SNORM, R),
        (9, F::R16_USCALED, R),
        (10, F::R16_SSCALED, R),
        (11, F::R16_UINT, R),
        (12, F::R16_SINT, R),
        (13, F::R16_SFLOAT, R),
        (14, F::R8G8_UNORM, RG),
        (15, F::R8G8_SNORM, RG),
        (16, F::R8G8_USCALED, RG),
        (17, F::R8G8_SSCALED, RG),
        (18, F::R8G8_UINT, RG),
        (19, F::R8G8_SINT, RG),
        (20, F::R32_UINT, R),
        (21, F::R32_SINT, R),
        (22, F::R32_SFLOAT, R),
        (22, F::D32_SFLOAT, R),
        (23, F::R16G16_UNORM, RG),
        (24, F::R16G16_SNORM, RG),
        (25, F::R16G16_USCALED, RG),
        (26, F::R16G16_SSCALED, RG),
        (27, F::R16G16_UINT, RG),
        (28, F::R16G16_SINT, RG),
        (29, F::R16G16_SFLOAT, RG),
        (36, F::B10G11R11_UFLOAT_PACK32, RGB),
        (50, F::A2B10G10R10_UNORM_PACK32, RGBA),
        (50, F::A2R10G10B10_UNORM_PACK32, BGRA),
        (51, F::A2B10G10R10_SNORM_PACK32, RGBA),
        (52, F::A2B10G10R10_USCALED_PACK32, RGBA),
        (53, F::A2B10G10R10_SSCALED_PACK32, RGBA),
        (54, F::A2B10G10R10_UINT_PACK32, RGBA),
        (54, F::A2R10G10B10_UINT_PACK32, BGRA),
        (55, F::A2B10G10R10_SINT_PACK32, RGBA),
        (56, F::R8G8B8A8_UNORM, RGBA),
        (56, F::B8G8R8A8_UNORM, BGRA),
        (57, F::R8G8B8A8_SNORM, RGBA),
        (58, F::R8G8B8A8_USCALED, RGBA),
        (59, F::R8G8B8A8_SSCALED, RGBA),
        (60, F::R8G8B8A8_UINT, RGBA),
        (61, F::R8G8B8A8_SINT, RGBA),
        (62, F::R32G32_UINT, RG),
        (63, F::R32G32_SINT, RG),
        (64, F::R32G32_SFLOAT, RG),
        (65, F::R16G16B16A16_UNORM, RGBA),
        (66, F::R16G16B16A16_SNORM, RGBA),
        (67, F::R16G16B16A16_USCALED, RGBA),
        (68, F::R16G16B16A16_SSCALED, RGBA),
        (69, F::R16G16B16A16_UINT, RGBA),
        (70, F::R16G16B16A16_SINT, RGBA),
        (71, F::R16G16B16A16_SFLOAT, RGBA),
        (72, F::R32G32B32_UINT, RGB),
        (73, F::R32G32B32_SINT, RGB),
        (74, F::R32G32B32_SFLOAT, RGB),
        (75, F::R32G32B32A32_UINT, RGBA),
        (76, F::R32G32B32A32_SINT, RGBA),
        (77, F::R32G32B32A32_SFLOAT, RGBA),
        (128, F::R8_SRGB, R),
        (129, F::R8G8_SRGB, RG),
        (130, F::R8G8B8A8_SRGB, RGBA),
        (130, F::B8G8R8A8_SRGB, BGRA),
        (132, F::E5B9G9R9_UFLOAT_PACK32, RGB),
        (133, F::B5G6R5_UNORM_PACK16, RGB),
        (133, F::R5G6B5_UNORM_PACK16, BGR),
        (134, F::A1R5G5B5_UNORM_PACK16, BGRA),
        (136, F::A4B4G4R4_UNORM_PACK16, RGBA),
        (136, F::A4R4G4B4_UNORM_PACK16, BGRA),
        (136, F::R4G4B4A4_UNORM_PACK16, ABGR),
        (141, F::D24_UNORM_S8_UINT, R),
        (141, F::X8_D24_UNORM_PACK32, R),
        (146, F::D32_SFLOAT_S8_UINT, R),
        (169, F::BC1_RGBA_UNORM_BLOCK, RGBA),
        (169, F::BC1_RGB_UNORM_BLOCK, RGB),
        (170, F::BC1_RGBA_SRGB_BLOCK, RGBA),
        (170, F::BC1_RGB_SRGB_BLOCK, RGB),
        (171, F::BC2_UNORM_BLOCK, RGBA),
        (172, F::BC2_SRGB_BLOCK, RGBA),
        (173, F::BC3_UNORM_BLOCK, RGBA),
        (174, F::BC3_SRGB_BLOCK, RGBA),
        (175, F::BC4_UNORM_BLOCK, R),
        (176, F::BC4_SNORM_BLOCK, R),
        (177, F::BC5_UNORM_BLOCK, RG),
        (178, F::BC5_SNORM_BLOCK, RG),
        (179, F::BC6H_UFLOAT_BLOCK, RGB),
        (180, F::BC6H_SFLOAT_BLOCK, RGB),
        (181, F::BC7_UNORM_BLOCK, RGBA),
        (182, F::BC7_SRGB_BLOCK, RGBA),
    ]
};

/// The GFX10 code of a Vulkan format and the DST_SEL that reads its
/// channels as R, G, B, A.
pub fn format_of(format: vk::Format) -> Option<(u32, Swizzle)> {
    FORMATS
        .iter()
        .find(|(_, f, _)| *f == format)
        .map(|&(c, _, s)| (c, s))
}

/// The Vulkan format of a GFX10 code, in the hardware's channel order.
pub fn vulkan_format(code: u32) -> Option<vk::Format> {
    FORMATS
        .iter()
        .find(|(c, ..)| *c == code)
        .map(|&(_, f, _)| f)
}

/// Whether views of the format sample as integers (T# WORD7 INTEGER).
pub fn is_integer(format: vk::Format) -> bool {
    use vk::Format as F;
    matches!(
        format,
        F::R8_UINT
            | F::R8_SINT
            | F::S8_UINT
            | F::R16_UINT
            | F::R16_SINT
            | F::R8G8_UINT
            | F::R8G8_SINT
            | F::R32_UINT
            | F::R32_SINT
            | F::R16G16_UINT
            | F::R16G16_SINT
            | F::A2B10G10R10_UINT_PACK32
            | F::A2R10G10B10_UINT_PACK32
            | F::A2B10G10R10_SINT_PACK32
            | F::R8G8B8A8_UINT
            | F::R8G8B8A8_SINT
            | F::R32G32_UINT
            | F::R32G32_SINT
            | F::R16G16B16A16_UINT
            | F::R16G16B16A16_SINT
            | F::R32G32B32_UINT
            | F::R32G32B32_SINT
            | F::R32G32B32A32_UINT
            | F::R32G32B32A32_SINT
    )
}

/// The Vulkan format of a DXGI format (the title's view and resource
/// descriptions); typeless formats take their usual view.
pub fn from_dxgi(dxgi: u32) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match dxgi {
        1..=2 => F::R32G32B32A32_SFLOAT,
        3 => F::R32G32B32A32_UINT,
        4 => F::R32G32B32A32_SINT,
        5..=6 => F::R32G32B32_SFLOAT,
        7 => F::R32G32B32_UINT,
        8 => F::R32G32B32_SINT,
        9..=10 => F::R16G16B16A16_SFLOAT,
        11 => F::R16G16B16A16_UNORM,
        12 => F::R16G16B16A16_UINT,
        13 => F::R16G16B16A16_SNORM,
        14 => F::R16G16B16A16_SINT,
        15..=16 => F::R32G32_SFLOAT,
        17 => F::R32G32_UINT,
        18 => F::R32G32_SINT,
        19..=21 => F::D32_SFLOAT_S8_UINT,
        22 => F::R32G32_UINT,
        23..=24 => F::A2B10G10R10_UNORM_PACK32,
        25 => F::A2B10G10R10_UINT_PACK32,
        26 => F::B10G11R11_UFLOAT_PACK32,
        27..=28 => F::R8G8B8A8_UNORM,
        29 => F::R8G8B8A8_SRGB,
        30 => F::R8G8B8A8_UINT,
        31 => F::R8G8B8A8_SNORM,
        32 => F::R8G8B8A8_SINT,
        33..=34 => F::R16G16_SFLOAT,
        35 => F::R16G16_UNORM,
        36 => F::R16G16_UINT,
        37 => F::R16G16_SNORM,
        38 => F::R16G16_SINT,
        39 | 41 => F::R32_SFLOAT,
        40 => F::D32_SFLOAT,
        42 => F::R32_UINT,
        43 => F::R32_SINT,
        44..=47 => F::D24_UNORM_S8_UINT,
        48..=49 => F::R8G8_UNORM,
        50 => F::R8G8_UINT,
        51 => F::R8G8_SNORM,
        52 => F::R8G8_SINT,
        53 | 56 => F::R16_UNORM,
        54 => F::R16_SFLOAT,
        55 => F::D16_UNORM,
        57 => F::R16_UINT,
        58 => F::R16_SNORM,
        59 => F::R16_SINT,
        60..=61 | 65 => F::R8_UNORM,
        62 => F::R8_UINT,
        63 => F::R8_SNORM,
        64 => F::R8_SINT,
        67 => F::E5B9G9R9_UFLOAT_PACK32,
        70..=71 => F::BC1_RGBA_UNORM_BLOCK,
        72 => F::BC1_RGBA_SRGB_BLOCK,
        73..=74 => F::BC2_UNORM_BLOCK,
        75 => F::BC2_SRGB_BLOCK,
        76..=77 => F::BC3_UNORM_BLOCK,
        78 => F::BC3_SRGB_BLOCK,
        79..=80 => F::BC4_UNORM_BLOCK,
        81 => F::BC4_SNORM_BLOCK,
        82..=83 => F::BC5_UNORM_BLOCK,
        84 => F::BC5_SNORM_BLOCK,
        85 => F::R5G6B5_UNORM_PACK16,
        86 => F::A1R5G5B5_UNORM_PACK16,
        87..=88 | 90 | 92 => F::B8G8R8A8_UNORM,
        91 | 93 => F::B8G8R8A8_SRGB,
        94..=95 => F::BC6H_UFLOAT_BLOCK,
        96 => F::BC6H_SFLOAT_BLOCK,
        97..=98 => F::BC7_UNORM_BLOCK,
        99 => F::BC7_SRGB_BLOCK,
        115 => F::A4R4G4B4_UNORM_PACK16,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_fields() {
        let img = Image {
            address: 0x12_3456_7800,
            format: 56,
            kind: ImageType::D2,
            width: 1920,
            height: 1080,
            depth: 1,
            first_layer: 0,
            layer_count: 1,
            first_mip: 1,
            mip_count: 3,
            resource_mips: 11,
            samples: 1,
            swizzle: BGRA,
            min_lod: 0.0,
        };
        let w = img.encode(ImageSlot {
            slot: 0xabcde,
            integer: true,
            storage: false,
        });
        assert_eq!(w[0], (0x12_3456_7800u64 >> 8) as u32);
        assert_eq!((w[1] >> 20) & 0x1ff, 56, "FORMAT at word 1 bits 28:20");
        let width = ((w[1] >> 30) | ((w[2] & 0xfff) << 2)) + 1;
        assert_eq!(width, 1920);
        assert_eq!(((w[2] >> 14) & 0x3fff) + 1, 1080);
        assert_eq!(w[3] >> 28, 9, "2D");
        assert_eq!((w[3] >> 12) & 0xf, 1, "base level");
        assert_eq!((w[3] >> 16) & 0xf, 3, "last level");
        assert_eq!((w[5] >> 4) & 0xf, 10, "max mip");
        assert_eq!(w[6], 0);
        assert_eq!(w[7] & 0xff_ffff, 0xabcde);
        assert_eq!(w[7] >> 24, 1);
    }

    #[test]
    fn buffer_and_sampler_fields() {
        let b = Buffer {
            address: 0x0001_2345_6789_abcd,
            stride: 48,
            records: 1000,
            format: 0,
            swizzle: IDENTITY,
            oob: Oob::Structured,
        };
        let w = b.encode();
        assert_eq!(
            u64::from(w[0]) | u64::from(w[1] & 0xffff) << 32,
            0x2345_6789_abcd
        );
        assert_eq!((w[1] >> 16) & 0x3fff, 48);
        assert_eq!(w[2], 1000);
        assert_eq!(w[3] >> 30, 0, "TYPE buffer");
        let s = Sampler {
            address: [0, 2, 6],
            mag: 1,
            min: 1,
            mip: 2,
            max_anisotropy: 16,
            compare: 0,
            min_lod: 0.0,
            max_lod: 15.0,
            lod_bias: -1.0,
            border: 2,
            filter_mode: 0,
        }
        .encode(77);
        assert_eq!(s[0] & 7, 0);
        assert_eq!((s[0] >> 3) & 7, 2);
        assert_eq!((s[0] >> 6) & 7, 6);
        assert_eq!((s[0] >> 9) & 7, 4);
        assert_eq!(s[3] & 0xfff, 77);
        assert_eq!(s[3] >> 30, 2);
    }

    #[test]
    fn formats_round_trip() {
        assert_eq!(format_of(vk::Format::B8G8R8A8_UNORM), Some((56, BGRA)));
        assert_eq!(vulkan_format(56), Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(from_dxgi(87), Some(vk::Format::B8G8R8A8_UNORM));
        assert_eq!(from_dxgi(71), Some(vk::Format::BC1_RGBA_UNORM_BLOCK));
        assert!(is_integer(vk::Format::R32_UINT));
        for dxgi in 1..=115 {
            if let Some(f) = from_dxgi(dxgi) {
                assert!(
                    format_of(f).is_some(),
                    "DXGI {dxgi} -> {f:?} has no GFX10 code"
                );
            }
        }
    }
}
