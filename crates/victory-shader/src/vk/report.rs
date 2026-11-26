//! The translation report, an opt-in for tools (`cargo xtask
//! translation-index`): which opcodes a program uses, the translation, and
//! the opcode a refusal stops at. [`super::translate`] pays nothing for it
//! beyond naming the opcode of each approximation it takes.

use super::VkShader;
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram};
use crate::{ShaderError, gfx10};

/// One container through the translator.
#[derive(Debug)]
pub struct Report {
    /// The program's stage ([`stage`]).
    pub stage: Option<HwStage>,
    /// Every opcode the program's code uses (decoder names, `exp` for
    /// exports, `?` for opcodes outside the decoder's table), sorted, once each.
    pub opcodes: Vec<&'static str>,
    pub result: Result<VkShader, ShaderError>,
    /// The opcode of the instruction a refusal names, when it names one.
    pub refused_at: Option<&'static str>,
}

/// Translates `container` with the process-wide semantic registry and
/// reports on it (decodes the code a second time for the opcode list).
pub fn report(container: &[u8]) -> Report {
    let program = program(container);
    let stage = program.as_ref().map(|p| p.stage);
    let insts = program.and_then(|p| gfx10::decode(&p.code).ok()).unwrap_or_default();
    let mut opcodes: Vec<&'static str> = insts.iter().map(|i| i.name().unwrap_or("?")).collect();
    opcodes.sort_unstable();
    opcodes.dedup();
    let result = super::translate(container);
    let refused_at = result.as_ref().err().and_then(|e| refusal_opcode(e, &insts));
    Report { stage, opcodes, result, refused_at }
}

/// The stage of [`program`].
pub fn stage(container: &[u8]) -> Option<HwStage> {
    program(container).map(|p| p.stage)
}

/// The program the backend takes, else the container's first program (a
/// geometry or hull stage); `None` without one.
fn program(container: &[u8]) -> Option<NativeProgram> {
    if let Ok((_, n)) = super::native(container) {
        return Some(n);
    }
    let c = Container::parse(container).ok()?;
    crate::native_programs(c.part(FourCC::DXIL)?.data).ok()?.into_iter().next()
}

/// The opcode at a refusal's offset. Offset 0 is also where refusals of
/// whole-program constructs point, so there it counts only when the
/// refusal names the opcode itself.
fn refusal_opcode(e: &ShaderError, insts: &[gfx10::Instruction]) -> Option<&'static str> {
    let (offset, what) = match *e {
        ShaderError::Unsupported { offset, what } => (offset, what),
        ShaderError::Isa { offset, reason } => (offset, reason),
        _ => return None,
    };
    let name = insts.iter().find(|i| i.offset == offset)?.name()?;
    (offset != 0 || what == name).then_some(name)
}

/// A refusal without the offset, so that equal reasons group together.
pub fn reason(e: &ShaderError) -> String {
    match e {
        ShaderError::Unsupported { what, .. } => format!("{what} is not supported"),
        ShaderError::Isa { reason, .. } => format!("GFX10 code: {reason}"),
        _ => e.to_string(),
    }
}
