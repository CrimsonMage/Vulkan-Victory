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
        self.meet();
    }

    /// At the console's memory barrier (`s_waitcnt_vscnt`): the halves meet
    /// (the meeting orders buffer and image memory too).
    pub(super) fn memory_meet(&mut self) {
        if !self.xwave {
            return;
        }
        self.meet();
    }

    /// The halves meet, nothing exchanged (`uniform::MEET`: it goes in a
    /// program with nothing shared to order).
    fn meet(&mut self) {
        self.close();
        let z = self.k32(0);
        self.emit_imm(Op::Xchg, Ty::vec(S::U32, 2), vec![z], vec![crate::vk::uniform::MEET]);
        self.lds_dirty = false;
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
        // the bits of the lanes that hold no work (lift_hi.rs)
        let idle = self.idle_of.get(&(b, half)).copied().flatten();
        match half {
            Half::Lo => {
                let own = self.ballot(b);
                if self.native.stage != HwStage::Ps {
                    return own;
                }
                // a fragment subgroup may lack lanes: the console's mask has
                // their bits
                match idle {
                    Some(false) => own,
                    Some(true) => {
                        let live = self.live();
                        let rest = self.spv(op::NOT, Ty::U32, &[live]);
                        self.spv(op::BITWISE_OR, Ty::U32, &[own, rest])
                    }
                    None => {
                        self.approx("a lane mask read as a number: the bits of the lanes a pixel subgroup lacks are not known");
                        own
                    }
                }
            }
            Half::Hi => {
                // one subgroup for the wave: its upper half holds no work
                if let Some(ones) = idle {
                    return self.k32(if ones { u32::MAX } else { 0 });
                }
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

    /// Whether lane number `l` of a wave64 is in the lower half, or is the
    /// -1 of a search that found no lane (the console then reads lane 63,
    /// which holds no work where one subgroup stands for the wave: nobody's
    /// value): a constant, the first set bit of one dword (a mask's lower
    /// dword, its upper one known to be 0: lift_hi.rs), or a select of
    /// such.
    pub(super) fn lower_lane(&self, l: V) -> bool {
        let n = self.f.node(l);
        match n.op {
            Op::Const(c) => c < 32 || c == u64::from(u32::MAX),
            Op::Glsl(glsl::FIND_I_LSB) => true,
            Op::Spv(op::SELECT) => self.lower_lane(n.args[1]) && self.lower_lane(n.args[2]),
            _ => false,
        }
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

    /// The whole-wave sections of every block (`crate::wave::sections`:
    /// the code between `s_or_saveexec -1` and the restore of EXEC, which
    /// the console runs in every lane of the wave, the inactive ones with
    /// an identity: the compiler's wave reductions and prefix sums), by the
    /// index of the restore: the values written inside that turned out to
    /// be a reduction or a prefix over the lanes. None in a wave64 whose
    /// halves exchange (the lanes are all there, and the tree's reads
    /// across the halves are exchanged).
    pub(super) fn find_sections(&self) -> Vec<HashMap<usize, Vec<(Value, crate::wave::Shape)>>> {
        use crate::wave::Shape;
        let p = self.p;
        (0..p.blocks.len())
            .map(|b| {
                if self.xwave {
                    return HashMap::new();
                }
                let mut at_end = HashMap::new();
                for (start, sec) in crate::wave::sections(p, b) {
                    let mut vals = Vec::new();
                    for inst in &p.blocks[b].insts[start + 1..sec.end] {
                        for &(_, w) in &inst.writes {
                            if let Some(&sh @ (Shape::Active(..) | Shape::Prefix(..))) = sec.shapes.get(&w) {
                                vals.push((w, sh));
                            }
                        }
                    }
                    at_end.insert(sec.end, vals);
                }
                at_end
            })
            .collect()
    }

    /// At the end of a whole-wave section (before instruction `end`, the
    /// restore of EXEC): each value the section's simulation found to be a
    /// reduction or a prefix sum over the wave's lanes of `mask ? src :
    /// identity` becomes that subgroup operation over the invocations that
    /// are there (origin/main `emit/lower.rs` `shape`, `section_inst`).
    /// The console's lanes all exist and run the section's tree of DPP,
    /// `v_permlanex16` and `v_readlane` steps; a host subgroup may be
    /// partly filled (pixels), and the tree run lane by lane through
    /// shuffles loses whatever passed through a lane that is not there
    /// (lanes 0-11: lane 0 ends with lanes 0-7 only). The steps themselves
    /// stay as lifted for the values with no such form.
    pub(super) fn section_values(&mut self, b: usize, end: usize) -> Res<()> {
        let Some(vals) = self.sections[b].get(&end).cloned() else { return Ok(()) };
        self.close();
        for (w, shape) in vals {
            if let Some(x) = self.shape(shape) {
                self.vals.insert(w, R::D(x));
            }
        }
        Ok(())
    }

    /// A reduction (`Shape::Active`) or a prefix sum (`Shape::Prefix`) as a
    /// subgroup operation; none for a form SPIR-V has no operation for (a
    /// 16-bit float sum, a prefix of anything but a sum).
    fn shape(&mut self, shape: crate::wave::Shape) -> Option<V> {
        use crate::wave::{Op as W, Shape};
        let (o, src, mask, ident, scan) = match shape {
            Shape::Active(o, src, mask, ident) => (o, src, mask, ident, None),
            Shape::Prefix(o @ (W::AddU | W::AddF), src, mask, ident, inclusive) => (o, src, mask, ident, Some(inclusive)),
            _ => return None,
        };
        let (opc, float) = match o {
            W::AddU => (349, false),
            W::AddF => (350, true),
            W::MinI => (353, false),
            W::MinU => (354, false),
            W::MinF => (355, true),
            W::MaxI => (356, false),
            W::MaxU => (357, false),
            W::MaxF => (358, true),
            W::And => (359, false),
            W::Or => (360, false),
            W::Xor => (361, false),
            W::AddH => return None,
        };
        let m = self.value(mask, 0).ok()?;
        let on = self.mask_of(m);
        let x = self.value(src, 0).ok()?;
        let x = self.data(x);
        let x = self.u32v(x);
        let id = self.k32(ident);
        let x = self.spv(op::SELECT, Ty::U32, &[on, x, id]);
        let x = if float { self.f32v(x) } else { x };
        let ty = if float { Ty::F32 } else { Ty::U32 };
        // group operation: 0 reduce, 1 inclusive scan, 2 exclusive scan
        let group = match scan {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        };
        let r = self.emit(Op::Wave(opc, group), ty, vec![x]);
        Some(self.u32v(r))
    }

    /// The host subgroup's live invocations: a console wave has all its
    /// lanes, a host subgroup may be partly filled (pixels), and reading a
    /// lane that is not there gives an undefined value.
    pub(super) fn live(&mut self) -> V {
        let t = self.kb(true);
        let v4 = self.emit(Op::Wave(op::GROUP_NON_UNIFORM_BALLOT, u32::MAX), Ty::vec(S::U32, 4), vec![t]);
        self.extract(v4, 0)
    }

    /// The first lane of the ballot `b`, or with none in it the first
    /// lane the subgroup has.
    pub(super) fn first_lane_of(&mut self, b: V) -> V {
        let live = self.live();
        let z = self.k32(0);
        let any = self.spv(op::I_NOT_EQUAL, Ty::BOOL, &[b, z]);
        let m = self.spv(op::SELECT, Ty::U32, &[any, b, live]);
        let l = self.gl(glsl::FIND_I_LSB, Ty::U32, &[m]);
        let k31 = self.k32(31);
        self.spv(op::BITWISE_AND, Ty::U32, &[l, k31])
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
