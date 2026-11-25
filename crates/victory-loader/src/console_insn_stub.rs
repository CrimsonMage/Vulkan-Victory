//! The plain x86 code a console-only instruction becomes once it has trapped (pure: bytes
//! only; `console_insn` writes them). A wait hint becomes a pause or a NOP in place; a
//! `clzero` loop and SSE4a's INSERTQ / EXTRQ with immediates become a jump to a stub that
//! does the same with ordinary stores and BMI2, then jumps back.
//!
//! Stubs keep every register and flag the original keeps: their scratch registers are
//! pushed and popped, and they use only instructions that leave the flags alone (`mov`,
//! `movq`, `pext`, `pdep`, `not`, `lea`, `push`, `pop`), but for the `clzero` loop, which
//! ends on the loop's own `cmp`, as the original does.

use super::decode::{ConsoleOnly, field_mask};

/// A rewrite of the instruction at a site: bytes that take the place of its first bytes.
#[derive(Debug, PartialEq, Eq)]
pub enum Rewrite {
    /// The whole instruction, replaced by one of the same length.
    InPlace(Vec<u8>),
    /// A 5-byte `jmp` to a stub (made by [`stub`]); the bytes after it are never run.
    Stub,
}

/// What the instruction `what` (its bytes `code`) becomes, if it has a rewrite.
pub fn rewrite(what: &ConsoleOnly, code: &[u8]) -> Option<Rewrite> {
    let len = code.len();
    match *what {
        // A 3-byte NOP: the monitor is only armed for the MWAITX that follows.
        ConsoleOnly::Monitorx if len == 3 => Some(Rewrite::InPlace(vec![0x0F, 0x1F, 0x00])),
        // PAUSE, NOP: a spin-wait hint, as MWAITX is one (it may return at any time).
        ConsoleOnly::Mwaitx if len == 3 => Some(Rewrite::InPlace(vec![0xF3, 0x90, 0x90])),
        // `clwb m8` (66 [REX] 0F AE /6 m) becomes `nop m16` (66 [REX] 0F 1F /0 m) with the
        // same operand bytes, one instruction of the same length.
        ConsoleOnly::Clwb => {
            let at = code.iter().position(|&b| b == 0x0F)?;
            if code.get(at + 1) != Some(&0xAE) {
                return None;
            }
            let mut nop = code.to_vec();
            nop[at + 1] = 0x1F;
            nop[at + 2] &= !0x38;
            Some(Rewrite::InPlace(nop))
        }
        ConsoleOnly::ClzeroLoop { end_reg } if clzero_loop_parts(code, end_reg).is_some() => {
            Some(Rewrite::Stub)
        }
        ConsoleOnly::Insertq { reg: false, .. } | ConsoleOnly::Extrq { src: None, .. }
            if len >= 5 =>
        {
            Some(Rewrite::Stub)
        }
        _ => None,
    }
}

/// `jmp rel32` at `from` to `to`, if `to` is in reach.
pub fn jmp(from: usize, to: usize) -> Option<[u8; 5]> {
    let rel = i32::try_from((to as i64).wrapping_sub(from as i64 + 5)).ok()?;
    let r = rel.to_le_bytes();
    Some([0xE9, r[0], r[1], r[2], r[3]])
}

/// `clzero; add rax, 0x40; cmp rax, <end>; jb` as decoded: the `add` and `cmp` bytes, when
/// the `cmp` really is `cmp rax, <end>` (ModRM register-direct, rm = rax) and `<end>` is not
/// rsp (the stub pushes).
fn clzero_loop_parts(code: &[u8], end_reg: u8) -> Option<(&[u8], &[u8])> {
    if code.len() != 12 {
        return None;
    }
    let (add, cmp) = (&code[3..7], &code[7..10]);
    (cmp[2] & 0xC7 == 0xC0 && end_reg != 4).then_some((add, cmp))
}

/// The stub for a site rewritten with [`Rewrite::Stub`]: the instruction at `site` (`code`,
/// as decoded to `what`), for a stub placed at `at`. It ends with a jump back to the
/// instruction after the one it replaces (for a `clzero` loop, after the loop). None if a
/// jump does not reach.
pub fn stub(what: &ConsoleOnly, code: &[u8], site: usize, at: usize) -> Option<Vec<u8>> {
    let mut s = Vec::with_capacity(96);
    let back = site + code.len();
    match *what {
        ConsoleOnly::ClzeroLoop { end_reg } => {
            // The line of rax is zeroed through a scratch register, as CLZERO zeroes the
            // aligned 64 bytes holding rAX; rax steps as in the loop, the loop's own
            // `add` and `cmp` bytes deciding when it ends.
            let (add, cmp) = clzero_loop_parts(code, end_reg)?;
            let r: u8 = if end_reg == 1 { 2 } else { 1 }; // rcx, or rdx when the end is in rcx
            s.push(0x50 + r); // push r
            s.extend([0x48, 0x89, 0xC0 | r]); // mov r, rax
            s.extend([0x48, 0x83, 0xE0 | r, 0xC0]); // and r, -64
            let top = s.len();
            for k in 0..8u8 {
                // mov qword [r + 8k], 0
                if k == 0 {
                    s.extend([0x48, 0xC7, r]);
                } else {
                    s.extend([0x48, 0xC7, 0x40 | r, 8 * k]);
                }
                s.extend([0, 0, 0, 0]);
            }
            s.extend([0x48, 0x83, 0xC0 | r, 0x40]); // add r, 0x40
            s.extend(add); // add rax, 0x40
            s.extend(cmp); // cmp rax, <end>
            let rel = i8::try_from(top as isize - (s.len() as isize + 2)).ok()?;
            s.extend([0x72, rel as u8]); // jb top
            s.push(0x58 + r); // pop r (the flags stay the cmp's)
        }
        ConsoleOnly::Insertq {
            dst,
            src,
            reg: false,
            len,
            idx,
        } => {
            // dst.low = (dst.low & !mask) | pdep(src.low, mask); dst.high = 0, as the
            // emulation does (the console leaves it undefined).
            let mask = field_mask(len, idx);
            s.extend([0x50, 0x51, 0x52]); // push rax, rcx, rdx
            movq_from_xmm(&mut s, dst, 0); // movq rax, xmm(dst)
            movq_from_xmm(&mut s, src, 1); // movq rcx, xmm(src)
            s.extend([0x48, 0xBA]); // mov rdx, !mask
            s.extend((!mask).to_le_bytes());
            s.extend([0xC4, 0xE2, 0xFA, 0xF5, 0xC2]); // pext rax, rax, rdx
            s.extend([0xC4, 0xE2, 0xFB, 0xF5, 0xC2]); // pdep rax, rax, rdx: dst & !mask
            s.extend([0x48, 0xF7, 0xD2]); // not rdx
            s.extend([0xC4, 0xE2, 0xF3, 0xF5, 0xCA]); // pdep rcx, rcx, rdx: the field
            s.extend([0x48, 0x8D, 0x04, 0x08]); // lea rax, [rax + rcx]: no bits in common
            movq_to_xmm(&mut s, dst, 0); // movq xmm(dst), rax
            s.extend([0x5A, 0x59, 0x58]); // pop rdx, rcx, rax
        }
        ConsoleOnly::Extrq {
            dst,
            src: None,
            len,
            idx,
        } => {
            // dst.low = pext(dst.low, mask); dst.high = 0
            s.extend([0x50, 0x52]); // push rax, rdx
            movq_from_xmm(&mut s, dst, 0); // movq rax, xmm(dst)
            s.extend([0x48, 0xBA]); // mov rdx, mask
            s.extend(field_mask(len, idx).to_le_bytes());
            s.extend([0xC4, 0xE2, 0xFA, 0xF5, 0xC2]); // pext rax, rax, rdx
            movq_to_xmm(&mut s, dst, 0); // movq xmm(dst), rax
            s.extend([0x5A, 0x58]); // pop rdx, rax
        }
        _ => return None,
    }
    let from = at + s.len();
    s.extend(jmp(from, back)?);
    Some(s)
}

/// `movq r64(gpr), xmm(x)` (66 REX.W 0F 7E /r), gpr below 8.
fn movq_from_xmm(s: &mut Vec<u8>, x: u8, gpr: u8) {
    s.extend([
        0x66,
        0x48 | (x & 8) >> 1,
        0x0F,
        0x7E,
        0xC0 | (x & 7) << 3 | gpr,
    ]);
}

/// `movq xmm(x), r64(gpr)` (66 REX.W 0F 6E /r): the high quadword zeroed.
fn movq_to_xmm(s: &mut Vec<u8>, x: u8, gpr: u8) {
    s.extend([
        0x66,
        0x48 | (x & 8) >> 1,
        0x0F,
        0x6E,
        0xC0 | (x & 7) << 3 | gpr,
    ]);
}

#[cfg(test)]
mod tests {
    use super::super::decode::decode;
    use super::*;

    #[test]
    fn rewrites_have_the_instruction_length() {
        for code in [
            &[0x0F, 0x01, 0xFA][..],
            &[0x0F, 0x01, 0xFB],
            &[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08],
            &[0x66, 0x41, 0x0F, 0xAE, 0x30],
        ] {
            let (what, len) = decode(code).unwrap();
            let Some(Rewrite::InPlace(new)) = rewrite(&what, &code[..len]) else {
                panic!("{code:02x?}")
            };
            assert_eq!(new.len(), len);
            assert_eq!(decode(&new), None, "not console-only any more: {new:02x?}");
        }
        // clwb [rsp+8] becomes nop word [rsp+8]
        let (what, len) = decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]).unwrap();
        assert_eq!(
            rewrite(&what, &[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08][..len]),
            Some(Rewrite::InPlace(vec![0x66, 0x0F, 0x1F, 0x44, 0x24, 0x08]))
        );
    }

    #[test]
    fn stubs_for_the_title_sites() {
        // clzero loop against r10, as the title zeroes its blocks
        let code = [
            0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4,
        ];
        let (what, len) = decode(&code).unwrap();
        assert_eq!(rewrite(&what, &code[..len]), Some(Rewrite::Stub));
        let s = stub(&what, &code, 0x1000, 0x2000).unwrap();
        // ends with jmp back to after the loop
        assert_eq!(
            &s[s.len() - 5..],
            &jmp(0x2000 + s.len() - 5, 0x100C).unwrap()
        );
        // a `cmp` that is not against rax, and rsp as the end, are left to the emulation
        let mut other = code;
        other[9] = 0xD1; // cmp rcx, r10
        assert_eq!(rewrite(&decode(&other).unwrap().0, &other), None);
        // insertq xmm0, xmm1, 16, 16 and with REX (xmm11, xmm12)
        for code in [
            &[0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10][..],
            &[0xF2, 0x45, 0x0F, 0x78, 0xDC, 0x10, 0x10],
        ] {
            let (what, len) = decode(code).unwrap();
            assert_eq!(rewrite(&what, &code[..len]), Some(Rewrite::Stub));
            assert!(stub(&what, code, 0x1000, 0x2000).is_some());
        }
        // a register-form INSERTQ (4 bytes) is not rewritten
        let (what, _) = decode(&[0xF2, 0x0F, 0x79, 0xC1]).unwrap();
        assert_eq!(rewrite(&what, &[0xF2, 0x0F, 0x79, 0xC1]), None);
        // out of reach
        assert_eq!(jmp(0, 1 << 32), None);
    }
}
