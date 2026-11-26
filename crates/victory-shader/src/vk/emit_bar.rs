//! Console barriers (`s_barrier`) counted per subgroup, for a mesh program
//! of several waves whose barriers not every invocation reaches the same
//! number of times.
//!
//! The console's barrier counts arrivals of waves, at whatever barrier
//! instruction, and waves that have ended no longer count: the NGG code for
//! a group that culls everything has its first wave alone take one more
//! barrier before its dummy export. A group barrier (`OpControlBarrier`)
//! must be reached by every invocation in uniform control flow, so such a
//! program waits on counters instead: each subgroup counts its barriers,
//! and at its k-th waits until every other subgroup has passed k or has
//! returned (a return writes `GONE`, emit_xchg.rs). The wait is bounded
//! ([`BAR_SPIN_LIMIT`] polls; console waves are not in step, so it is
//! longer than an exchange's): a subgroup that runs out stops waiting at
//! barriers for the rest of the invocation.

use super::*;

/// Polls of the other subgroups' counters at one barrier before a subgroup
/// stops waiting at barriers.
const BAR_SPIN_LIMIT: u32 = 1 << 14;

/// Variables of the counted barrier (0: not made yet).
#[derive(Default)]
pub(super) struct BarVars {
    /// Workgroup `u32[subgroups]`: barriers each subgroup has passed.
    pub(super) count: Id,
    /// Private: this invocation's barriers so far.
    done: Id,
    /// Private: the subgroup stopped waiting.
    dead: Id,
    /// The barrier, as a function `void ()`.
    func: Id,
}

impl Emit<'_> {
    /// The counters (made once); returns the workgroup array.
    pub(super) fn bar_vars(&mut self) -> Id {
        if self.bar.count != 0 {
            return self.bar.count;
        }
        let (u, b) = (self.t.u32_, self.t.bool_);
        let arr = self.m.t_array_plain(u, self.group_threads().div_ceil(32));
        let p = self.m.t_ptr(en::SC_WORKGROUP, arr);
        self.bar.count = self.m.variable(p, en::SC_WORKGROUP);
        let (pu, pb) = (self.m.t_ptr(en::SC_PRIVATE, u), self.m.t_ptr(en::SC_PRIVATE, b));
        let (z, f) = (self.c32(0), self.m.c_bool(false));
        self.bar.done = self.m.variable_init(pu, en::SC_PRIVATE, z);
        self.bar.dead = self.m.variable_init(pb, en::SC_PRIVATE, f);
        self.interface.extend_from_slice(&[self.bar.count, self.bar.done, self.bar.dead]);
        self.bar.count
    }

    /// `s_barrier`: a group barrier, or the counted one.
    pub(super) fn barrier(&mut self) {
        if self.bar_counted {
            if self.bar.func == 0 {
                self.bar.func = self.m.id();
            }
            let void = self.t.void;
            self.m.op(57, void, &[self.bar.func]); // OpFunctionCall
            return;
        }
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        self.m.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
    }

    /// The counted barrier (written once, after main).
    pub(super) fn write_bar_fn(&mut self) {
        if self.bar.func == 0 {
            return;
        }
        self.m.capability(en::CAP_GROUP_NON_UNIFORM);
        let (u, b, void) = (self.t.u32_, self.t.bool_, self.t.void);
        let fty = self.m.t_func(void);
        self.m.op_to(op::FUNCTION, void, self.bar.func, &[0, fty]);
        let entry = self.m.id();
        self.m.label(entry);
        let one = self.c32(1);
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let sem = self.c32(super::SEM_GROUP);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        let sg = self.builtin(Builtin::SubgroupId);
        let k = self.m.op(op::LOAD, u, &[self.bar.done]);
        let k1 = self.m.op(op::I_ADD, u, &[k, one]);
        self.m.op_void(op::STORE, &[self.bar.done, k1]);
        let dead = self.m.op(op::LOAD, b, &[self.bar.dead]);
        let (l_go, m_go) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[m_go, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[dead, m_go, l_go]);
        self.m.label(l_go);
        let sub = self.c32(en::SCOPE_SUBGROUP);
        let first = self.m.op(op::GROUP_NON_UNIFORM_ELECT, b, &[sub]);
        let (l_el, m_el) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[m_el, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[first, l_el, m_el]);
        self.m.label(l_el);
        let (out, l_end) = self.bar_wait(sg, k1);
        self.m.op_void(op::BRANCH, &[m_el]);
        self.m.label(m_el);
        let ub = self.m.undef(b);
        let oph = self.m.op(op::PHI, b, &[out, l_end, ub, l_go]);
        let stopped = self.m.op(op::GROUP_NON_UNIFORM_BROADCAST_FIRST, b, &[sub, oph]);
        self.m.op_void(op::STORE, &[self.bar.dead, stopped]);
        self.m.op_void(op::BRANCH, &[m_go]);
        self.m.label(m_go);
        self.m.op_void(op::MEMORY_BARRIER, &[wg, sem]);
        self.m.op_void(op::RETURN, &[]);
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    /// One invocation of the subgroup: its count to `k1`, then a wait over
    /// every subgroup's count in turn. Returns (whether it ran out, the
    /// block it ends in).
    fn bar_wait(&mut self, sg: Id, k1: Id) -> (Id, Id) {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let (z, one) = (self.c32(0), self.c32(1));
        let wg = self.c32(en::SCOPE_WORKGROUP);
        let start = self.m.id();
        self.m.op_void(op::BRANCH, &[start]);
        self.m.label(start);
        let pw = self.m.t_ptr(en::SC_WORKGROUP, u);
        let cp = self.m.op(op::ACCESS_CHAIN, pw, &[self.bar.count, sg]);
        self.m.op_void(op::ATOMIC_STORE, &[cp, wg, z, k1]);
        let (l_hdr, l_body, l_cont, l_after) = (self.m.id(), self.m.id(), self.m.id(), self.m.id());
        self.m.op_void(op::BRANCH, &[l_hdr]);
        self.m.label(l_hdr);
        let (s_next, p_next) = (self.m.id(), self.m.id());
        let s = self.m.op(op::PHI, u, &[z, start, s_next, l_cont]);
        let polls = self.m.op(op::PHI, u, &[z, start, p_next, l_cont]);
        self.m.op_void(op::LOOP_MERGE, &[l_after, l_cont, 0]);
        self.m.op_void(op::BRANCH, &[l_body]);
        self.m.label(l_body);
        let op_ = self.m.op(op::ACCESS_CHAIN, pw, &[self.bar.count, s]);
        let c = self.m.op(op::ATOMIC_LOAD, u, &[op_, wg, z]);
        let passed = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[c, k1]);
        let s1 = self.m.op(op::I_ADD, u, &[s, one]);
        self.m.op_to(op::SELECT, u, s_next, &[passed, s1, s]);
        let p1 = self.m.op(op::I_ADD, u, &[polls, one]);
        self.m.op_to(op::SELECT, u, p_next, &[passed, polls, p1]);
        let subgroups = self.c32(self.group_threads().div_ceil(32));
        let all = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[s_next, subgroups]);
        let lim = self.c32(BAR_SPIN_LIMIT);
        let out = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[p_next, lim]);
        let stop = self.m.op(op::LOGICAL_OR, b, &[all, out]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[stop, l_after, l_cont]);
        self.m.label(l_cont);
        self.m.op_void(op::BRANCH, &[l_hdr]);
        self.m.label(l_after);
        // ran out: no one waits for this subgroup at barriers any more
        let gone = self.c32(super::xchg::GONE);
        let mark = self.m.op(op::SELECT, u, &[all, k1, gone]);
        self.m.op_void(op::ATOMIC_STORE, &[cp, wg, z, mark]);
        let ran_out = self.m.op(op::LOGICAL_NOT, b, &[all]);
        (ran_out, l_after)
    }
}
