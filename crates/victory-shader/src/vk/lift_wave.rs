//! Whole-wave values of a console wave64 run as two host subgroups
//! (compute groups, and mesh groups of more than 32 threads), and the
//! console's thread numbering on the host: the halves exchange
//! one word (emit_xchg.rs: group barriers in a group of one console wave,
//! counters per pair in larger groups). Every invocation reaches every
//! exchange as long as the scalar values agree, which the exchanges make
//! them do. Masks read as numbers, `any` of a mask, `v_readlane` and
//! `v_readfirstlane` are then exact.

use super::*;

impl Lift<'_> {
    /// Whether the halves exchange (set by `Lift::new`): a wave64 compute
    /// or mesh group with an upper half.
    pub(super) fn paired_halves(native: &NativeProgram) -> bool {
        native.wave64 && Self::host_threads(native) > 32
    }

    /// Both halves' value of a word uniform within each half: (lower
    /// half's, upper half's).
    pub(super) fn exchange(&mut self, own: V) -> (V, V) {
        let pair = self.emit(Op::Xchg, Ty::vec(S::U32, 2), vec![own]);
        self.lds_dirty = false;
        (self.extract(pair, 0), self.extract(pair, 1))
    }

    /// Before an LDS access: the halves meet when either may have made
    /// another since they last did. The console's wave runs its LDS
    /// accesses in program order across all 64 lanes (a lane initialises a
    /// counter, the wave adds to it, a lane reads the total); two host
    /// subgroups are not ordered without it.
    pub(super) fn lds_meet(&mut self) {
        if !self.xwave || !self.lds_dirty {
            return;
        }
        self.close();
        let z = self.k32(0);
        self.exchange(z);
    }

    /// At the console's memory barrier (`s_waitcnt_vscnt`): the halves meet
    /// (the meeting orders buffer and image memory too).
    pub(super) fn memory_meet(&mut self) {
        if !self.xwave {
            return;
        }
        self.close();
        let z = self.k32(0);
        self.exchange(z);
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

    /// Lane-mask registers `code`, `code + 1` as this invocation's bit. In
    /// a wave64 whose dwords were written apart (`s_mov_b32 exec_lo, 0`,
    /// `s_mov_b32 exec_hi, -1` in a prefix sum) each half takes its own
    /// dword.
    pub(super) fn mask_reg(&mut self, inst: &Inst, code: u8) -> Res<V> {
        let lo = self.val(inst, Reg::S(code))?;
        let hi = match inst.read(Reg::S(code + 1)) {
            Some(h) if self.wave64 => self.value(h, inst.machine.offset)?,
            _ => return Ok(self.mask_of(lo)),
        };
        Ok(self.mask_pair(lo, hi))
    }

    /// A 64-bit lane mask from its two dwords: one mask, or the dword of
    /// this invocation's half.
    pub(super) fn mask_pair(&mut self, lo: R, hi: R) -> V {
        if let (R::M(a, _), R::M(b, _)) = (lo, hi)
            && a == b
        {
            return a;
        }
        let l = self.mask_of(lo);
        let Some(up) = self.upper_half() else { return l };
        let h = self.mask_of(hi);
        if l == h {
            return l;
        }
        self.spv(op::SELECT, Ty::BOOL, &[up, h, l])
    }

    /// Whether this invocation is in the upper half of its console wave64
    /// (compute and mesh, whose waves are pairs of host subgroups); none
    /// where a host subgroup stands for a whole wave.
    pub(super) fn upper_half(&mut self) -> Option<V> {
        if !self.wave64 || !matches!(self.native.stage, HwStage::Cs | HwStage::Ms) {
            return None;
        }
        let h = self.half_index();
        let z = self.k32(0);
        Some(self.global_bin(op::I_NOT_EQUAL, Ty::BOOL, h, z))
    }

    /// Threads in a compute or mesh group as the program declares it (0 for
    /// other stages).
    pub(super) fn group_threads(native: &NativeProgram) -> u32 {
        match &native.io {
            StageIo::Cs(cs) => cs.threads.iter().product(),
            StageIo::Ms(ms) => ms.threads,
            _ => 0,
        }
    }

    /// Invocations of the host's group: a compute group is whole console
    /// waves, as the console's hardware runs it (the lanes past the group
    /// exist, inactive), so whole-wave code (`s_or_saveexec -1` and DPP
    /// steps) finds its identity in them instead of a missing lane, and a
    /// wave64 always has its upper half. One dimension: thread ids come
    /// from the console thread index.
    pub(crate) fn host_threads(native: &NativeProgram) -> u32 {
        let n = Self::group_threads(native);
        match &native.io {
            StageIo::Cs(_) => n.max(1).next_multiple_of(if native.wave64 { 64 } else { 32 }),
            _ => n,
        }
    }

    /// This invocation's thread index in its group as the console counts
    /// it: the host subgroups in order are the console waves (or their
    /// halves), so a thread's id, its lane and its wave agree whichever
    /// invocations the host puts in a subgroup. Groups of whole subgroups;
    /// others keep `LocalInvocationIndex`.
    pub(super) fn console_thread(&mut self) -> V {
        let n = Self::host_threads(self.native);
        if n == 0 || !n.is_multiple_of(32) {
            return self.f.global(Op::Builtin(Builtin::LocalInvocationIndex), Ty::U32);
        }
        let sg = self.f.global(Op::Builtin(Builtin::SubgroupId), Ty::U32);
        let k5 = self.k32(5);
        let base = self.global_bin(op::SHIFT_LEFT_LOGICAL, Ty::U32, sg, k5);
        let lane = self.lane();
        self.global_bin(op::BITWISE_OR, Ty::U32, base, lane)
    }

    /// Whether this invocation is a thread of the console's group (not a
    /// lane that fills its last wave): EXEC at the start. None when every
    /// invocation is.
    pub(super) fn in_group(&mut self) -> Option<V> {
        let n = Self::group_threads(self.native);
        if n == Self::host_threads(self.native) {
            return None;
        }
        let t = self.console_thread();
        let k = self.k32(n);
        Some(self.global_bin(op::U_LESS_THAN, Ty::BOOL, t, k))
    }

    /// Component `c` of the thread id in the group, from the console's
    /// thread index.
    pub(super) fn thread_id(&mut self, c: u8) -> V {
        let StageIo::Cs(cs) = &self.native.io else {
            return self.f.global(Op::Builtin(Builtin::LocalInvocationId(c)), Ty::U32);
        };
        let [x, y, z] = cs.threads.map(|d| d.max(1));
        let (div, modulo) = match c {
            0 => (1, x),
            1 => (x, y),
            _ => (x * y, z),
        };
        if modulo == 1 {
            return self.k32(0);
        }
        let t = self.console_thread();
        let q = if div == 1 {
            t
        } else {
            let k = self.k32(div);
            self.global_bin(op::U_DIV, Ty::U32, t, k)
        };
        let k = self.k32(modulo);
        self.global_bin(op::U_MOD, Ty::U32, q, k)
    }

    /// 0 for the lower half of a console wave64, 1 for the upper.
    pub(super) fn half_index(&mut self) -> V {
        let sg = self.f.global(Op::Builtin(Builtin::SubgroupId), Ty::U32);
        let one = self.k32(1);
        self.global_bin(op::BITWISE_AND, Ty::U32, sg, one)
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
