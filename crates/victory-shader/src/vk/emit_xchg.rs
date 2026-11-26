//! The exchange between the two host subgroups ("halves") of a console
//! wave64 (`Op::Xchg`, made by lift_wave.rs): each half gives a word that
//! is uniform in it, both get (lower half's word, upper half's word).
//!
//! **Barrier form.** Each half stores its word, a group barrier, both read
//! both words, a group barrier. Used where every invocation of the group
//! reaches the exchange the same number of times: in a group of one
//! console wave (at most 64 threads: the group is the wave, and both halves
//! make the same exchanges), and in larger groups at an exchange made
//! exactly once by every invocation (its block is in no loop and on every
//! path to a return), unless the program's console barriers are counted
//! (emit_bar.rs).
//!
//! **Counter form.** Elsewhere in a larger group a barrier would also wait
//! for the other console waves, whose control flow may differ, so each
//! pair meets through workgroup counters (as `docs/wave64-on-wave32.md` of
//! xcl): one invocation of each half stores its word, counts its arrival
//! and waits until the partner's count reaches this exchange; slots are
//! double buffered by the exchange's parity (the partner is at most one
//! exchange ahead). The wait is bounded ([`SPIN_LIMIT`] polls): a half that
//! runs out, or finds its partner gone (a return writes `GONE` into the
//! half's counter), stops exchanging for the rest of the invocation and
//! writes `GONE` into its own counter, so its partner stops too; its halves
//! then read the partner's word as 0. So a pair waits out the bound at most
//! once.

use super::*;

/// Polls of the partner's counter before a half stops exchanging (at most
/// some microseconds: the partner runs the same instructions).
const SPIN_LIMIT: u32 = 1 << 10;

/// A counter value: the subgroup returned, or stopped meeting.
pub(super) const GONE: u32 = u32::MAX;

/// Variables of the exchange (0: not made yet).
#[derive(Default)]
pub(super) struct XchgVars {
    /// Workgroup `u32[6 * waves]`: per console wave, parity and half (the
    /// counter form), then per subgroup (the barrier form).
    slots: Id,
    /// Workgroup `u32[2 * waves]`: exchanges each half has arrived at.
    pub(super) count: Id,
    /// Private: this invocation's exchanges so far.
    done: Id,
    /// Private: the pair stopped exchanging.
    dead: Id,
    /// The counter form, as a function `uvec2 (u32)`.
    func: Id,
    /// Exchanges in the barrier form in a larger group.
    pub(super) once: HashSet<V>,
}

impl Emit<'_> {
    pub(super) fn group_threads(&self) -> u32 {
        self.st.local_size.iter().map(|&x| x.max(1)).product()
    }

    /// Whether pairs meet through counters (several console waves per group).
    fn counted(&self) -> bool {
        self.group_threads() > 64
    }

    pub(super) fn xchg(&mut self, own: Id, v: V) -> Id {
        self.xchg_vars();
        let u = self.t.u32_;
        let v2 = self.m.t_vec(u, 2);
        if self.counted() && !self.xv.once.contains(&v) {
            if self.xv.func == 0 {
                self.xv.func = self.m.id();
            }
            return self.m.op(57, v2, &[self.xv.func, own]); // OpFunctionCall
        }
        // each half stores its word in its slot; a group barrier; both
        // slots; a group barrier (the next exchange writes them again)
        let sg = self.builtin(Builtin::SubgroupId);
        let one = self.c32(1);
        let base = self.c32(4 * self.group_threads().div_ceil(64));
        let mine = self.m.op(op::I_ADD, u, &[base, sg]);
        let p = self.m.t_ptr(en::SC_WORKGROUP, u);
        let at = self.m.op(op::ACCESS_CHAIN, p, &[self.xv.slots, mine]);
        self.m.op_void(op::STORE, &[at, own]);
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        self.m.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
        let not1 = self.c32(!1);
        let lo_i = self.m.op(op::BITWISE_AND, u, &[mine, not1]);
        let hi_i = self.m.op(op::BITWISE_OR, u, &[mine, one]);
        let p0 = self.m.op(op::ACCESS_CHAIN, p, &[self.xv.slots, lo_i]);
        let p1 = self.m.op(op::ACCESS_CHAIN, p, &[self.xv.slots, hi_i]);
        let lo = self.m.op(op::LOAD, u, &[p0]);
        let hi = self.m.op(op::LOAD, u, &[p1]);
        self.m.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
        self.m.op(op::COMPOSITE_CONSTRUCT, v2, &[lo, hi])
    }

    fn xchg_vars(&mut self) {
        if self.xv.slots != 0 {
            return;
        }
        let (u, b) = (self.t.u32_, self.t.bool_);
        let waves = self.group_threads().div_ceil(64);
        let arr = self.m.t_array_plain(u, 6 * waves);
        let p = self.m.t_ptr(en::SC_WORKGROUP, arr);
        self.xv.slots = self.m.variable(p, en::SC_WORKGROUP);
        self.interface.push(self.xv.slots);
        if !self.counted() {
            return;
        }
        let arr = self.m.t_array_plain(u, 2 * waves);
        let p = self.m.t_ptr(en::SC_WORKGROUP, arr);
        self.xv.count = self.m.variable(p, en::SC_WORKGROUP);
        let (pu, pb) = (self.m.t_ptr(en::SC_PRIVATE, u), self.m.t_ptr(en::SC_PRIVATE, b));
        let (z, f) = (self.c32(0), self.m.c_bool(false));
        self.xv.done = self.m.variable_init(pu, en::SC_PRIVATE, z);
        self.xv.dead = self.m.variable_init(pb, en::SC_PRIVATE, f);
        self.interface.extend_from_slice(&[self.xv.count, self.xv.done, self.xv.dead]);
    }

    /// Blocks every invocation runs exactly once: in no loop, and on every
    /// path to a return (they dominate every returning block).
    fn once_blocks(&self) -> Vec<bool> {
        let returns: Vec<usize> = (0..self.f.blocks.len())
            .filter(|&b| self.g.rpo[b] != usize::MAX && matches!(self.f.blocks[b].term, Term::Return))
            .collect();
        (0..self.f.blocks.len())
            .map(|b| self.g.rpo[b] != usize::MAX && !self.g.in_loop[b] && returns.iter().all(|&r| self.g.dominates(b, r)))
            .collect()
    }

    /// The nodes of `op` at the top level of their blocks (not inside a
    /// lane guard): (in a block run once, node).
    fn top_level(&self, once: &[bool], want: &Op) -> Vec<(bool, V)> {
        let mut out = Vec::new();
        for (b, block) in self.f.blocks.iter().enumerate() {
            for it in &block.body {
                if let Item::N(v) = it
                    && self.f.node(*v).op == *want
                {
                    out.push((once[b], *v));
                }
            }
        }
        out
    }

    /// Whether some console barrier of a larger group is inside a lane
    /// guard (never: guards hold no barriers) or in a block that not every
    /// invocation runs exactly once.
    fn uneven_barriers(&self, once: &[bool]) -> bool {
        let top = self.top_level(once, &Op::Barrier);
        let all = self.f.nodes.iter().filter(|n| n.op == Op::Barrier).count();
        top.len() != all || top.iter().any(|&(o, _)| !o)
    }

    /// At the program's start: how the waves meet (the barrier form for the
    /// exchanges run once, counted console barriers for a mesh program
    /// whose barriers are uneven); then, when anything counts, every
    /// counter 0 and a group barrier (all invocations are there).
    pub(super) fn xchg_init(&mut self) {
        if self.group_threads() <= 32 {
            return;
        }
        let once = self.once_blocks();
        self.bar_counted = self.st.stage == HwStage::Ms && self.uneven_barriers(&once);
        let has_xchg = self.f.nodes.iter().any(|n| n.op == Op::Xchg);
        if self.counted() && !self.bar_counted {
            self.xv.once = self.top_level(&once, &Op::Xchg).into_iter().filter(|&(o, _)| o).map(|(_, v)| v).collect();
        }
        if !(self.counted() && has_xchg) && !self.bar_counted {
            return;
        }
        let u = self.t.u32_;
        let me = self.builtin(Builtin::LocalInvocationIndex);
        let z = self.c32(0);
        let p = self.m.t_ptr(en::SC_WORKGROUP, u);
        let mut counters = Vec::new();
        if self.counted() && has_xchg {
            self.xchg_vars();
            counters.push(self.xv.count);
        }
        if self.bar_counted {
            counters.push(self.bar_vars());
        }
        let last = self.c32(self.group_threads().div_ceil(32) - 1);
        let i = self.m.ext(glsl::U_MIN, u, &[me, last]);
        for c in counters {
            let ptr = self.m.op(op::ACCESS_CHAIN, p, &[c, i]);
            self.m.op_void(op::STORE, &[ptr, z]);
        }
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(en::SEM_ACQUIRE_RELEASE | en::SEM_WORKGROUP_MEMORY);
        self.m.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
    }

    /// Before a return: the subgroup's counters `GONE`, so no partner or
    /// other wave waits for it.
    pub(super) fn leave(&mut self) {
        let counters: Vec<Id> = [self.xv.count, self.bar.count].into_iter().filter(|&c| c != 0).collect();
        if counters.is_empty() {
            return;
        }
        let u = self.t.u32_;
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let sg = self.builtin(Builtin::SubgroupId);
        let (z, gone) = (self.c32(0), self.c32(GONE));
        let p = self.m.t_ptr(en::SC_WORKGROUP, u);
        for c in counters {
            let ptr = self.m.op(op::ACCESS_CHAIN, p, &[c, sg]);
            self.m.op_void(op::ATOMIC_STORE, &[ptr, wg, z, gone]);
        }
    }

    /// The counter form (written once, after main): returns (lower half's
    /// word, upper half's word).
    pub(super) fn write_xchg_fn(&mut self) {
        if self.xv.func == 0 {
            return;
        }
        self.m.capability(en::CAP_GROUP_NON_UNIFORM);
        self.m.capability(en::CAP_GROUP_NON_UNIFORM_BALLOT);
        let (u, b) = (self.t.u32_, self.t.bool_);
        let v2 = self.m.t_vec(u, 2);
        let fty = self.m.global_fn_type(v2, &[u]);
        self.m.op_to(op::FUNCTION, v2, self.xv.func, &[0, fty]);
        let own = self.m.id();
        self.m.op_to(55, u, own, &[]); // OpFunctionParameter
        let entry = self.m.id();
        self.m.label(entry);
        let (z, one) = (self.c32(0), self.c32(1));
        // every invocation's workgroup memory accesses before the meeting
        // happen before the partner's after it
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let sg = self.builtin(Builtin::SubgroupId);
        let half = self.m.op(op::BITWISE_AND, u, &[sg, one]);
        // this exchange's number, counted per invocation
        let k = self.m.op(op::LOAD, u, &[self.xv.done]);
        let k1 = self.m.op(op::I_ADD, u, &[k, one]);
        self.m.op_void(op::STORE, &[self.xv.done, k1]);
        // a half whose partner is not in the group exchanges with nobody
        let mate = self.m.op(op::BITWISE_XOR, u, &[sg, one]);
        let halves = self.c32(self.group_threads().div_ceil(32));
        let alone = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[mate, halves]);
        let dead = self.m.op(op::LOAD, b, &[self.xv.dead]);
        let skip = self.m.op(op::LOGICAL_OR, b, &[dead, alone]);
        let (l_ex, m_ex) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[m_ex, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[skip, m_ex, l_ex]);
        self.m.label(l_ex);
        let sub = self.c32(en::SCOPE_SUBGROUP);
        let first = self.m.op(op::GROUP_NON_UNIFORM_ELECT, b, &[sub]);
        let (l_el, m_el) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[m_el, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[first, l_el, m_el]);
        self.m.label(l_el);
        let (word, arrived, l_end) = self.xchg_meet(own, sg, k, k1);
        self.m.op_void(op::BRANCH, &[m_el]);
        // the whole half takes that invocation's result
        self.m.label(m_el);
        let (uu, ub) = (self.m.undef(u), self.m.undef(b));
        let wph = self.m.op(op::PHI, u, &[word, l_end, uu, l_ex]);
        let aph = self.m.op(op::PHI, b, &[arrived, l_end, ub, l_ex]);
        let got = self.m.op(op::GROUP_NON_UNIFORM_BROADCAST_FIRST, u, &[sub, wph]);
        let met = self.m.op(op::GROUP_NON_UNIFORM_BROADCAST_FIRST, b, &[sub, aph]);
        let gone = self.m.op(op::LOGICAL_NOT, b, &[met]);
        self.m.op_void(op::STORE, &[self.xv.dead, gone]);
        self.m.op_void(op::BRANCH, &[m_ex]);
        self.m.label(m_ex);
        let partner = self.m.op(op::PHI, u, &[z, entry, got, m_el]);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let lower = self.m.op(op::I_EQUAL, b, &[half, z]);
        let lo = self.m.op(op::SELECT, u, &[lower, own, partner]);
        let hi = self.m.op(op::SELECT, u, &[lower, partner, own]);
        let r = self.m.op(op::COMPOSITE_CONSTRUCT, v2, &[lo, hi]);
        self.m.op_void(254, &[r]); // OpReturnValue
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    /// One invocation of the half: store the word, count the arrival, wait
    /// for the partner's; a half that does not meet it writes `GONE`.
    /// Returns (partner's word, whether it met, the block it ends in).
    fn xchg_meet(&mut self, own: Id, sg: Id, k: Id, k1: Id) -> (Id, Id, Id) {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let (z, one, two) = (self.c32(0), self.c32(1), self.c32(2));
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        let l_el = self.m.id();
        self.m.op_void(op::BRANCH, &[l_el]);
        self.m.label(l_el);
        let half = self.m.op(op::BITWISE_AND, u, &[sg, one]);
        let wave = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[sg, one]);
        let parity = self.m.op(op::BITWISE_AND, u, &[k, one]);
        let w2 = self.m.op(op::I_MUL, u, &[wave, two]);
        let wp = self.m.op(op::I_ADD, u, &[w2, parity]);
        let base = self.m.op(op::I_MUL, u, &[wp, two]);
        let other = self.m.op(op::BITWISE_XOR, u, &[half, one]);
        let (si, pi) = (self.m.op(op::I_ADD, u, &[base, half]), self.m.op(op::I_ADD, u, &[base, other]));
        let mi = self.m.op(op::I_ADD, u, &[w2, other]);
        let pw = self.m.t_ptr(en::SC_WORKGROUP, u);
        let sp = self.m.op(op::ACCESS_CHAIN, pw, &[self.xv.slots, si]);
        self.m.op_void(op::STORE, &[sp, own]);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let cp = self.m.op(op::ACCESS_CHAIN, pw, &[self.xv.count, sg]);
        self.m.op(op::ATOMIC_I_ADD, u, &[cp, wg, z, one]);
        let mp = self.m.op(op::ACCESS_CHAIN, pw, &[self.xv.count, mi]);
        let (l_hdr, l_body, l_cont, l_after) = (self.m.id(), self.m.id(), self.m.id(), self.m.id());
        self.m.op_void(op::BRANCH, &[l_hdr]);
        self.m.label(l_hdr);
        let polls = self.m.id();
        let i = self.m.op(op::PHI, u, &[z, l_el, polls, l_cont]);
        self.m.op_void(op::LOOP_MERGE, &[l_after, l_cont, 0]);
        self.m.op_void(op::BRANCH, &[l_body]);
        self.m.label(l_body);
        let c = self.m.op(op::ATOMIC_LOAD, u, &[mp, wg, z]);
        let reached = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[c, k1]);
        self.m.op_to(op::I_ADD, u, polls, &[i, one]);
        let lim = self.c32(SPIN_LIMIT);
        let out = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[polls, lim]);
        let stop = self.m.op(op::LOGICAL_OR, b, &[reached, out]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[stop, l_after, l_cont]);
        self.m.label(l_cont);
        self.m.op_void(op::BRANCH, &[l_hdr]);
        self.m.label(l_after);
        let gone = self.c32(GONE);
        let alive = self.m.op(op::I_NOT_EQUAL, b, &[c, gone]);
        let arrived = self.m.op(op::LOGICAL_AND, b, &[reached, alive]);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let pp = self.m.op(op::ACCESS_CHAIN, pw, &[self.xv.slots, pi]);
        let word = self.m.op(op::LOAD, u, &[pp]);
        let word = self.m.op(op::SELECT, u, &[arrived, word, z]);
        // not met: this half stops, and says so to its partner
        let mine = self.m.op(op::ATOMIC_LOAD, u, &[cp, wg, z]);
        let mark = self.m.op(op::SELECT, u, &[arrived, mine, gone]);
        self.m.op_void(op::ATOMIC_STORE, &[cp, wg, z, mark]);
        (word, arrived, l_after)
    }
}
