//! The exchange between the two host subgroups ("halves") of a console
//! wave64 (`Op::Xchg`, made by lift_wave.rs): each half gives a word that
//! is uniform in it, both get (lower half's word, upper half's word).
//!
//! A group of one console wave (at most 64 threads) meets at group
//! barriers: the group is the wave, and both halves make the same
//! exchanges. In a larger group a barrier would also wait for the other
//! console waves, whose control flow may differ, so each pair meets through
//! workgroup counters instead (as `docs/wave64-on-wave32.md` of xcl): one
//! invocation of each half stores its word, counts its arrival and waits
//! until the partner's count reaches this exchange; slots are double
//! buffered by the exchange's parity (the partner is at most one exchange
//! ahead). The wait is bounded: a pair that runs out stops exchanging
//! (its halves then read the partner's word as 0) instead of hanging.

use super::*;

/// Polls of the partner's counter before a half stops exchanging.
const SPIN_LIMIT: u32 = 1 << 20;

/// Variables of the exchange (0: not made yet).
#[derive(Default)]
pub(super) struct XchgVars {
    /// Workgroup `u32[4 * waves]`: per console wave, parity and half.
    slots: Id,
    /// Workgroup `u32[2 * waves]`: exchanges each half has arrived at.
    count: Id,
    /// Private: this invocation's exchanges so far.
    done: Id,
    /// Private: the pair stopped exchanging.
    dead: Id,
    /// The counter form, as a function `uvec2 (u32)`.
    func: Id,
}

impl Emit<'_> {
    fn group_threads(&self) -> u32 {
        self.st.local_size.iter().map(|&x| x.max(1)).product()
    }

    /// Whether pairs meet through counters (several console waves per group).
    fn counted(&self) -> bool {
        self.group_threads() > 64
    }

    pub(super) fn xchg(&mut self, own: Id) -> Id {
        self.xchg_vars();
        let u = self.t.u32_;
        let v2 = self.m.t_vec(u, 2);
        if self.counted() {
            if self.xv.func == 0 {
                self.xv.func = self.m.id();
            }
            return self.m.op(57, v2, &[self.xv.func, own]); // OpFunctionCall
        }
        // each half stores its word in its slot; a group barrier; both
        // slots; a group barrier (the next exchange writes them again)
        let half = {
            let sg = self.builtin(Builtin::SubgroupId);
            let one = self.c32(1);
            self.m.op(op::BITWISE_AND, u, &[sg, one])
        };
        let p = self.m.t_ptr(en::SC_WORKGROUP, u);
        let mine = self.m.op(op::ACCESS_CHAIN, p, &[self.xv.slots, half]);
        self.m.op_void(op::STORE, &[mine, own]);
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        self.m.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
        let (k0, k1) = (self.c32(0), self.c32(1));
        let p0 = self.m.op(op::ACCESS_CHAIN, p, &[self.xv.slots, k0]);
        let p1 = self.m.op(op::ACCESS_CHAIN, p, &[self.xv.slots, k1]);
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
        let arr = self.m.t_array_plain(u, 4 * waves);
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

    /// At the program's start, when pairs meet through counters: every
    /// counter 0, then a group barrier (all invocations are there).
    pub(super) fn xchg_init(&mut self) {
        if !self.counted() || !self.f.nodes.iter().any(|n| n.op == Op::Xchg) {
            return;
        }
        self.xchg_vars();
        let u = self.t.u32_;
        let me = self.builtin(Builtin::LocalInvocationIndex);
        let last = self.c32(2 * self.group_threads().div_ceil(64) - 1);
        let i = self.m.ext(glsl::U_MIN, u, &[me, last]);
        let p = self.m.t_ptr(en::SC_WORKGROUP, u);
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[self.xv.count, i]);
        let z = self.c32(0);
        self.m.op_void(op::STORE, &[ptr, z]);
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(en::SEM_ACQUIRE_RELEASE | en::SEM_WORKGROUP_MEMORY);
        self.m.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
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
        let (z, one, two) = (self.c32(0), self.c32(1), self.c32(2));
        // every invocation's workgroup memory accesses before the meeting
        // happen before the partner's after it
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let sg = self.builtin(Builtin::SubgroupId);
        let half = self.m.op(op::BITWISE_AND, u, &[sg, one]);
        let wave = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[sg, one]);
        // this exchange's number, counted per invocation
        let k = self.m.op(op::LOAD, u, &[self.xv.done]);
        let k1 = self.m.op(op::I_ADD, u, &[k, one]);
        self.m.op_void(op::STORE, &[self.xv.done, k1]);
        let parity = self.m.op(op::BITWISE_AND, u, &[k, one]);
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
        // one invocation: store, count the arrival, wait for the partner's
        self.m.label(l_el);
        let w2 = self.m.op(op::I_MUL, u, &[wave, two]);
        let wp = self.m.op(op::I_ADD, u, &[w2, parity]);
        let base = self.m.op(op::I_MUL, u, &[wp, two]);
        let other = self.m.op(op::BITWISE_XOR, u, &[half, one]);
        let (si, pi) = (self.m.op(op::I_ADD, u, &[base, half]), self.m.op(op::I_ADD, u, &[base, other]));
        let (ci, mi) = (self.m.op(op::I_ADD, u, &[w2, half]), self.m.op(op::I_ADD, u, &[w2, other]));
        let pw = self.m.t_ptr(en::SC_WORKGROUP, u);
        let sp = self.m.op(op::ACCESS_CHAIN, pw, &[self.xv.slots, si]);
        self.m.op_void(op::STORE, &[sp, own]);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let cp = self.m.op(op::ACCESS_CHAIN, pw, &[self.xv.count, ci]);
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
        let arrived = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[c, k1]);
        self.m.op_to(op::I_ADD, u, polls, &[i, one]);
        let lim = self.c32(SPIN_LIMIT);
        let out = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[polls, lim]);
        let stop = self.m.op(op::LOGICAL_OR, b, &[arrived, out]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[stop, l_after, l_cont]);
        self.m.label(l_cont);
        self.m.op_void(op::BRANCH, &[l_hdr]);
        self.m.label(l_after);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let pp = self.m.op(op::ACCESS_CHAIN, pw, &[self.xv.slots, pi]);
        let word = self.m.op(op::LOAD, u, &[pp]);
        let word = self.m.op(op::SELECT, u, &[arrived, word, z]);
        self.m.op_void(op::BRANCH, &[m_el]);
        // the whole half takes that invocation's result
        self.m.label(m_el);
        let (uu, ub) = (self.m.undef(u), self.m.undef(b));
        let wph = self.m.op(op::PHI, u, &[word, l_after, uu, l_ex]);
        let aph = self.m.op(op::PHI, b, &[arrived, l_after, ub, l_ex]);
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
}
