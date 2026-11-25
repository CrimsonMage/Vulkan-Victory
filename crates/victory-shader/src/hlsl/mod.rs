//! The HLSL backend (prototype): a console program as HLSL text, compiled by
//! Microsoft's DXC instead of our own DXIL writer.
//!
//! It reuses every stage before emission: the SSA form (`ir`, `spill`), the
//! per-lane structure (`structure`: lane masks as per-lane booleans,
//! guards), descriptor tracking (`resource`) and I/O raising (`io`). Each
//! machine instruction becomes a line or two of HLSL over `uint` variables
//! (one per SSA value; floats travel as bits), the block graph becomes
//! structured control flow ([`cfg`]), and DXC does the rest: SSA
//! construction, structurization for DXIL, optimization, the container,
//! signatures and the validator's hash.
//!
//! Bindings are the bitcode backend's (`emit` module notes): user data at
//! `b0, space0`, descriptors through `ResourceDescriptorHeap[]` at the
//! console's index, immediate samplers `s<n>, space0`, root views in
//! `space1`. A [`translate`] result is a [`HostShader`] like `emit`'s, so
//! the run-time path is the same; signatures are checked against the
//! console's (DXC packs them itself) and a mismatch is a refusal.
//!
//! What it covers is listed by what it refuses ([`ShaderError::Unsupported`]
//! with the construct). Stages: pixel and compute programs, vertex programs
//! without 16-bit outputs. A console wave of 64 run as host waves of 32:
//! lanes stay independent except for cross-lane operations, which act on
//! the host wave ("approximate", counted in [`Hlsl::approximate`]).

mod cfg;
pub mod dxc;
mod lower;

use crate::dxbc::{Container, FourCC};
use crate::emit::{HostShader, StaticSampler};
use crate::native::{HwStage, NativeProgram};
use crate::{ShaderError, io, ir, resource, structure};

/// `VICTORY_SHADER_BACKEND=hlsl`: translate through this backend at run time
/// (each shader falls back to the bitcode backend when it fails).
pub fn selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SHADER_BACKEND").is_ok_and(|v| v.trim() == "hlsl"))
}

/// A program as HLSL, with what the host binds for it.
#[derive(Debug, Clone)]
pub struct Hlsl {
    pub stage: HwStage,
    pub source: String,
    /// DXC target profile (`ps_6_6`...).
    pub profile: &'static str,
    /// Compiled with `-enable-16bit-types` (16-bit signature elements).
    pub enable_16bit: bool,
    pub shader: HostShader,
    /// Cross-lane operations taken per host wave where the console's wave
    /// is wider (a console wave64 compute program on 32-lane hosts): what
    /// they were. Empty: exact.
    pub approximate: Vec<&'static str>,
}

/// One row of a console signature part (`ISG1`, `OSG1`, `PSG1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Sig {
    pub name: String,
    pub index: u32,
    /// D3D_NAME (0 a user semantic, 1 position, 64 target...).
    pub sv: u32,
    /// D3D_REGISTER_COMPONENT_TYPE: 1 uint, 2 int, 3 float.
    pub comp: u32,
    pub reg: Option<u32>,
    pub mask: u8,
    /// Minimum precision (16-bit elements on the console are declared
    /// 32-bit on the host, as the bitcode backend does).
    pub half: bool,
    /// A native 16-bit type (`float16_t`...) on the console.
    pub native16: bool,
    /// DXIL interpolation mode from `PSV0` (0 if unknown).
    pub interp: u8,
    /// Position in the console's declaration order (`PSV0` lists elements
    /// as declared, the signature part by register).
    pub order: usize,
}

impl Sig {
    pub fn col(&self) -> u8 {
        self.mask.trailing_zeros().min(3) as u8
    }
    pub fn cols(&self) -> u8 {
        self.mask.count_ones().max(1) as u8
    }
}

pub(crate) fn signature(part: &[u8]) -> Result<Vec<Sig>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let raw = u32_at(part, at + 16)?;
        let comp = match raw {
            1..=3 => raw,
            4..=6 => raw - 3,
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a 64-bit or untyped signature element",
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(Sig {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            sv: u32_at(part, at + 12)?,
            comp,
            reg: (register != u32::MAX).then_some(register),
            mask: *part.get(at + 24).ok_or(ShaderError::Malformed("truncated"))?,
            half: (4..=6).contains(&raw) || u32_at(part, at + 28)? != 0,
            native16: (4..=6).contains(&raw),
            interp: 0,
            order: i,
        });
    }
    Ok(out)
}

/// The signature elements `PSV0` lists (runtime info version 1 on), as
/// declared: per part (inputs, outputs, primitive outputs) each element's
/// first register, first column and interpolation mode. Empty when the part
/// does not say.
fn psv_elements(psv: &[u8]) -> [Vec<(u8, u8, u8)>; 3] {
    let none = || [Vec::new(), Vec::new(), Vec::new()];
    let r = |o: usize| crate::dxbc::u32_at(psv, o).ok().map(|x| x as usize);
    let Some(info) = r(0) else { return none() };
    if info < 36 {
        return none();
    }
    let counts = [28, 29, 30].map(|k| psv.get(4 + k).copied().unwrap_or(0) as usize);
    let mut at = 4 + info;
    let Some(resources) = r(at) else { return none() };
    at += 4;
    if resources > 0 {
        let Some(size) = r(at) else { return none() };
        at += 4 + size * resources;
    }
    let Some(strings) = r(at) else { return none() };
    at += 4 + strings;
    let Some(indices) = r(at) else { return none() };
    at += 4 + 4 * indices;
    let Some(size) = r(at) else { return none() };
    at += 4;
    let mut out = none();
    let mut k = 0;
    for (part, &n) in counts.iter().enumerate() {
        for _ in 0..n {
            let e = at + k * size;
            k += 1;
            let (Some(&row), Some(&cols), Some(&interp)) = (psv.get(e + 9), psv.get(e + 10), psv.get(e + 13)) else {
                return none();
            };
            out[part].push((row, cols >> 4 & 3, interp));
        }
    }
    out
}

/// The `PSV0` elements' interpolation modes and declaration order, matched
/// to the signature part's elements by register and column.
fn apply_psv(sig: &mut [Sig], psv: &[(u8, u8, u8)]) {
    if psv.len() != sig.len() {
        return;
    }
    for (order, &(row, col, interp)) in psv.iter().enumerate() {
        if let Some(e) = sig.iter_mut().find(|e| e.reg == Some(u32::from(row)) && e.col() == col) {
            e.interp = interp;
            e.order = order;
        }
    }
}

/// The console program of `container` as HLSL, its signature elements
/// declared in the console's order.
pub fn emit(container: &[u8]) -> Result<Hlsl, ShaderError> {
    emit_with(container, 0)
}

/// [`emit`] in another form: `form` bit 0 declares signature elements in
/// register order instead (DXC then packs some signatures as the console
/// did that it packs otherwise from the console's order), bit 1 declares
/// the console's 16-bit elements 16-bit (DXC packs them apart from 32-bit
/// ones, as the console did; the host signature then has 16-bit types).
/// [`translate`] tries the forms in order: [`FORMS`].
pub fn emit_with(container: &[u8], form: u8) -> Result<Hlsl, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c.part(FourCC::DXIL).ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms))
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    let p = crate::spill::build(&native)?;
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit_program(&c, &native, &p, &st, &io, &res, form)
}

/// [`emit`] with the earlier stages already run.
pub fn emit_program(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    form: u8,
) -> Result<Hlsl, ShaderError> {
    let sig = |fourcc| -> Result<Vec<Sig>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    let psv = c.part(FourCC::PSV0).map_or_else(|| [Vec::new(), Vec::new(), Vec::new()], |p| psv_elements(p.data));
    apply_psv(&mut inputs, &psv[0]);
    apply_psv(&mut outputs, &psv[1]);
    if native.stage != HwStage::Ps {
        // (only pixel inputs interpolate)
        for e in inputs.iter_mut().chain(outputs.iter_mut()) {
            e.interp = 0;
        }
    }
    let mesh = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let mut prims = sig(FourCC::PSG1)?;
            apply_psv(&mut prims, &psv[2]);
            // The console never exports a primitive id or cull flag.
            prims.retain(|e| !matches!(e.sv, 7 | 25));
            // The group id arrives multiplied by max(threads, vertices,
            // primitives); the thread count is in PSV0 from runtime info
            // version 2 on (as `emit`).
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            let stride = threads.map_or(ms.threads, |t| t.max(ms.max_vertices).max(ms.max_primitives));
            Some(lower::Mesh {
                prims,
                stride,
                threads: ms.threads,
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
            })
        }
        _ => None,
    };
    lower::lower(native, p, st, io, res, inputs, outputs, mesh, form)
}

/// The forms [`translate`] tries ([`emit_with`]): the console's order, then
/// register order, each with 32-bit then 16-bit elements.
pub const FORMS: [u8; 4] = [0, 1, 2, 3];

/// Translates through HLSL and DXC: a [`HostShader`] like
/// [`crate::emit::translate`]'s. `compiler` is the loaded DXC.
pub fn translate(container: &[u8], compiler: &dxc::Dxc) -> Result<(HostShader, Hlsl), String> {
    let mut first = None;
    for form in FORMS {
        let mut h = emit_with(container, form).map_err(|e| e.to_string())?;
        if form & 2 != 0 && !h.enable_16bit {
            continue;
        }
        let dxil = compiler.compile(&h.source, h.profile, h.enable_16bit)?;
        match check_signatures(container, &dxil) {
            Ok(()) => {
                h.shader.dxil = dxil;
                return Ok((h.shader.clone(), h));
            }
            Err(e) => {
                first.get_or_insert(e);
            }
        }
    }
    Err(first.unwrap_or_default())
}

/// The registers and masks DXC gave the signature elements must be the
/// console's: the previous and next stages (and the input layout) are
/// linked by them.
pub fn check_signatures(console: &[u8], host: &[u8]) -> Result<(), String> {
    let a = Container::parse(console).map_err(|e| e.to_string())?;
    let b = Container::parse(host).map_err(|e| e.to_string())?;
    for fourcc in [FourCC::ISG1, FourCC::OSG1, FourCC::PSG1] {
        let rows = |c: &Container| -> Result<Vec<(String, u32, Option<u32>, u8)>, String> {
            let Some(part) = c.part(fourcc) else { return Ok(Vec::new()) };
            let mut v: Vec<_> = signature(part.data)
                .map_err(|e| e.to_string())?
                .into_iter()
                // System values the hardware supplies have no register to match.
                .filter(|s| s.reg.is_some())
                .map(|s| (s.name.to_ascii_uppercase(), s.index, s.reg, s.mask))
                .collect();
            v.sort();
            Ok(v)
        };
        let (x, y) = (rows(&a)?, rows(&b)?);
        // Every console element the host declares sits where the console's
        // does (a system value the console's signature leaves out, the
        // vertex and instance ids, links by nothing).
        for e in &y {
            let console = x.iter().any(|f| f.0 == e.0 && f.1 == e.1);
            if !console && e.0.starts_with("SV_") {
                continue;
            }
            if !x.iter().any(|f| f.0 == e.0 && f.1 == e.1 && f.2 == e.2 && f.3 == e.3) {
                return Err(format!("signature element {}{} at {:?} mask {:#x} is not the console's", e.0, e.1, e.2, e.3));
            }
        }
    }
    Ok(())
}

pub(crate) fn static_samplers(words: &[[u32; 4]], cmp: u64) -> Vec<StaticSampler> {
    words
        .iter()
        .enumerate()
        .map(|(n, &w)| {
            let mut s = StaticSampler::from_words(w);
            if cmp >> n & 1 != 0 {
                s.filter |= 0x80;
            }
            s
        })
        .collect()
}

/// What joins the shader cache key when this backend is selected: the DXC
/// that compiles (its version), so a new `dxcompiler.dll` translates again.
pub fn fingerprint() -> Vec<u8> {
    if !selected() {
        return Vec::new();
    }
    let mut out = b"hlsl\0".to_vec();
    if let Some(d) = dxc::shared() {
        out.extend_from_slice(d.version().as_bytes());
    }
    out
}

/// The run-time hook (`emit::translate`): this backend's translation when
/// it is selected and succeeds, `None` otherwise (the bitcode backend then
/// translates).
pub fn try_translate(container: &[u8]) -> Option<HostShader> {
    if !selected() {
        return None;
    }
    let d = dxc::shared()?;
    let t = std::time::Instant::now();
    match translate(container, d) {
        Ok((shader, _)) => {
            dxc::note(true, t.elapsed());
            Some(shader)
        }
        Err(e) => {
            dxc::note(false, t.elapsed());
            if std::env::var_os("VICTORY_HLSL_WHY").is_some() {
                eprintln!("HLSL backend: {e}; the bitcode backend translates");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::StageIo;

    /// An `OSG1` part of one element: `SV_Target0`, float4.
    fn target_signature() -> Vec<u8> {
        let mut d = Vec::new();
        let w = |d: &mut Vec<u8>, x: u32| d.extend_from_slice(&x.to_le_bytes());
        w(&mut d, 1);
        w(&mut d, 8);
        // stream, name offset, index, system value, component type, register
        for x in [0, 8 + 32, 0, 64, 3, 0] {
            w(&mut d, x);
        }
        d.extend_from_slice(&[0xF, 0, 0, 0]);
        w(&mut d, 0);
        d.extend_from_slice(b"SV_Target\0\0\0");
        d
    }

    /// A pixel program writing 1.0 to every channel of render target 0:
    /// `v_mov_b32 v0, 1.0; exp mrt0 v0, v0, v0, v0 done vm; s_endpgm`.
    #[test]
    fn a_constant_colour_becomes_an_hlsl_pixel_shader() {
        let words = [0x7E00_02F2u32, 0xF800_180F, 0x0000_0000, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: true,
            vgprs: 1,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps { inputs: vec![], input_addr: 0, db_shader_control: 0, patches: vec![] },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let osg1 = target_signature();
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let h = emit_program(&c, &native, &p, &st, &io, &res, 0).unwrap();
        assert_eq!(h.profile, "ps_6_6");
        assert!(h.source.contains("float4 o0 : SV_Target0;"), "{}", h.source);
        assert!(h.source.contains("o.o0.w = asfloat(0x3f800000u);"), "{}", h.source);
        assert!(h.approximate.is_empty());
        // With DXC at hand (`VICTORY_DXC`), it compiles and keeps the console's
        // signature.
        if std::env::var_os("VICTORY_DXC").is_some() {
            let d = dxc::Dxc::load(dxc::default_path().as_deref()).unwrap();
            let dxil = d.compile(&h.source, h.profile, h.enable_16bit).unwrap();
            check_signatures(&bytes, &dxil).unwrap();
        }
    }
}
