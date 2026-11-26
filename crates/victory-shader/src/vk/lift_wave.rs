//! Whole-wave values of a console wave64 run as two host subgroups, for
//! thread groups of one console wave (33 to 64 threads): the halves
//! exchange one word through workgroup memory between barriers (the group
//! is the wave, so every invocation reaches every exchange as long as the
//! scalar values agree, which the exchanges make them do). Masks read as
//! numbers, `any` of a mask, `v_readlane` and `v_readfirstlane` are then
//! exact. Larger groups keep the per-half forms.

use super::*;

impl Lift<'_> {
    /// Whether the halves exchange (set by `Lift::new`).
    pub(super) fn one_wave(native: &NativeProgram) -> bool {
        let StageIo::Cs(cs) = &native.io else { return false };
        let threads = cs.threads.iter().product::<u32>();
        native.wave64 && threads > 32 && threads <= 64
    }

    /// Both halves' value of a word uniform within each half: (lower
    /// half's, upper half's).
    pub(super) fn exchange(&mut self, own: V) -> (V, V) {
        let pair = self.emit(Op::Xchg, Ty::vec(S::U32, 2), vec![own]);
        (self.extract(pair, 0), self.extract(pair, 1))
    }

    /// The two dwords of a wave64 mask: exchanged once per block (masks
    /// read inside a guard keep the per-half value: a guard is not reached
    /// by every invocation).
    pub(super) fn mask_words(&mut self, b: V) -> Option<(V, V)> {
        if !self.xwave {
            return None;
        }
        if let Some(&w) = self.words.get(&b) {
            return Some(w);
        }
        if self.guarded {
            return None;
        }
        let own = self.ballot(b);
        let w = self.exchange(own);
        self.words.insert(b, w);
        Some(w)
    }

    /// Exchanges the masks a VALU instruction reads as numbers before its
    /// guard opens (lane selects and carries read their own bit).
    pub(super) fn prefetch_masks(&mut self, inst: &Inst, name: &str, v: &Valu) {
        if !self.xwave {
            return;
        }
        // a mask the open guard defines is read after the guard
        self.need(inst);
        let lane_bit = name.starts_with("v_cndmask") || name.contains("_co_ci") || name.contains("addc") || name.contains("subb");
        for (i, o) in v.src.iter().enumerate() {
            if lane_bit && i == 2 {
                continue;
            }
            if let Operand::Scalar { code, .. } = *o
                && inst.read(Reg::S(code)).is_some()
                && let Ok(R::M(b, _)) = self.val(inst, Reg::S(code))
            {
                self.mask_words(b);
            }
        }
    }

    /// A dword of a lane mask read as a number: exchanged in a wave of
    /// exchanging halves; else this half's ballot (the high dword 0).
    pub(super) fn mask_word(&mut self, b: V, half: Half) -> V {
        // a constant mask: every lane or none (the upper half only exists in
        // a wave of exchanging halves)
        if let Some(c) = self.f.as_const(b)
            && (half == Half::Lo || self.xwave)
        {
            return self.k32(if c != 0 { u32::MAX } else { 0 });
        }
        if let Some((lo, hi)) = self.mask_words(b) {
            return if half == Half::Lo { lo } else { hi };
        }
        if self.xwave {
            self.approx("a wave64 mask read as a number inside a guard (per half)");
        }
        match half {
            Half::Lo => self.ballot(b),
            Half::Hi => {
                if self.wave64_halves() {
                    self.approx("a wave64 mask's high half read as a number");
                }
                self.k32(0)
            }
        }
    }

    /// Any lane of the console wave: both halves' ballots.
    pub(super) fn wave_any(&mut self, b: V) -> Option<V> {
        let (lo, hi) = self.mask_words(b)?;
        let or = self.spv(op::BITWISE_OR, Ty::U32, &[lo, hi]);
        let z = self.k32(0);
        Some(self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[or, z]))
    }

    /// `v_readlane` of console lane `l` (uniform): the half that owns it
    /// gives its value.
    pub(super) fn wave_readlane(&mut self, x: V, l: V) -> Option<V> {
        if !self.xwave || self.guarded {
            return None;
        }
        let k31 = self.k32(31);
        let li = self.spv(op::BITWISE_AND, Ty::U32, &[l, k31]);
        let own = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![x, li]);
        let (lo, hi) = self.exchange(own);
        let k32 = self.k32(32);
        let upper = self.spv(op::U_GREATER_THAN_EQUAL, Ty::BOOL, &[l, k32]);
        Some(self.spv(op::SELECT, Ty::U32, &[upper, hi, lo]))
    }

    /// The first set bit of a 64-bit value (-1 for none).
    pub(super) fn ff1_64(&mut self, lo: V, hi: V) -> V {
        let a = self.gl(glsl::FIND_I_LSB, Ty::U32, &[lo]);
        let b = self.gl(glsl::FIND_I_LSB, Ty::U32, &[hi]);
        let k32 = self.k32(32);
        let b32 = self.spv(op::I_ADD, Ty::U32, &[b, k32]);
        let neg = self.k32(u32::MAX);
        let z = self.k32(0);
        let hi_set = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[hi, z]);
        let b = self.spv(op::SELECT, Ty::U32, &[hi_set, b32, neg]);
        let lo_set = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[lo, z]);
        self.spv(op::SELECT, Ty::U32, &[lo_set, a, b])
    }

    /// The host subgroup's live invocations: a console wave has all its
    /// lanes, a host subgroup may be partly filled (pixels), and reading a
    /// lane that is not there gives an undefined value.
    fn live(&mut self) -> V {
        let t = self.kb(true);
        let v4 = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_BALLOT, u32::MAX), Ty::vec(S::U32, 4), vec![t]);
        self.extract(v4, 0)
    }

    /// Whether lane `l` (0-31) of the host subgroup is live.
    pub(super) fn lane_live(&mut self, l: V) -> V {
        let m = self.live();
        let b = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[m, l]);
        let one = self.k32(1);
        let b = self.spv(op::BITWISE_AND, Ty::U32, &[b, one]);
        let z = self.k32(0);
        self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[b, z])
    }

    /// `x` of lane `from`, or this lane's own `x` when that lane is not
    /// there (the console's lanes all exist; for the reductions that read
    /// other lanes the own value is neutral).
    pub(super) fn shuffle_live(&mut self, x: V, from: V) -> V {
        let r = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![x, from]);
        let ok = self.lane_live(from);
        self.spv(op::SELECT, Ty::U32, &[ok, r, x])
    }

    /// `v_readlane` of lane `l` (0-31): that lane, or the first live one
    /// when it is not there (a value made uniform by a reduction is the same
    /// in every lane).
    pub(super) fn readlane_live(&mut self, x: V, l: V) -> V {
        let ok = self.lane_live(l);
        let m = self.live();
        let first = self.gl(glsl::FIND_I_LSB, Ty::U32, &[m]);
        let k31 = self.k32(31);
        let first = self.spv(op::BITWISE_AND, Ty::U32, &[first, k31]);
        let idx = self.spv(op::SELECT, Ty::U32, &[ok, l, first]);
        self.emit(Op::Wave(op::GROUP_NON_UNIFORM_SHUFFLE, u32::MAX), Ty::U32, vec![x, idx])
    }
}
