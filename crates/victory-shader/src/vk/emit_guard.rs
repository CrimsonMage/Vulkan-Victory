//! The loop guard: every invocation counts the iterations of all its loops
//! and gives up after `Stage::loop_limit` of them, so a loop that never
//! exits (a mistranslation, or memory that reads zeros), or nested loops
//! that multiply, cannot hang the GPU. A loop that gives up leaves through
//! its merge and returns; it first counts itself in the host's guard buffer
//! (push dwords 2-3: its device address, 0 for none): dword 0 += 1, dword 1
//! = the program's guard id.
//!
//! The count goes up by a value the driver cannot see through (1 + the top
//! bit of push dword 3, the high half of a device address: always 1). With
//! a constant step NVIDIA's compiler analyses the shared counter across
//! nested loops for a very long time: CS_DummyResolve (30 loops, 28,672
//! words) compiled in 1,376 s, and in 0.5 s with this step.

use super::*;

impl Emit<'_> {
    /// The invocation's iteration count so far: a private `u32`, 0 at start.
    fn spent(&mut self) -> Id {
        if self.spent_var == 0 {
            let u = self.t.u32_;
            let p = self.m.t_ptr(en::SC_PRIVATE, u);
            let z = self.c32(0);
            self.spent_var = self.m.variable_init(p, en::SC_PRIVATE, z);
            self.interface.push(self.spent_var);
        }
        self.spent_var
    }

    /// At the continue target: one more iteration; past the limit the loop
    /// leaves through its merge.
    pub(super) fn guard_back(&mut self, cont: Id, header: Id, merge: Id) {
        let var = self.spent();
        let (u, b) = (self.t.u32_, self.t.bool_);
        let (hi, k31, one) = (self.push_load(3), self.c32(31), self.c32(1));
        let top = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[hi, k31]);
        let step = self.m.op(op::I_ADD, u, &[top, one]);
        let count = self.m.op(op::LOAD, u, &[var]);
        let next = self.m.op(op::I_ADD, u, &[count, step]);
        self.m.op_void(op::STORE, &[var, next]);
        let lim = self.c32(self.st.loop_limit);
        let over = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[next, lim]);
        self.merge_preds.entry(merge).or_default().push(cont);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[over, merge, header]);
    }

    /// At the loop's merge: arrived from the continue target, the guard
    /// fired: count it, then return.
    pub(super) fn guard_exit(&mut self, cont: Id, preds: &[Id]) {
        let b = self.t.bool_;
        let (t, f) = (self.m.c_bool(true), self.m.c_bool(false));
        let over = self.phis_at(&[b], preds, &|p, _| Some(if p == cont { t } else { f }))[0];
        let (lr, lc) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[lc, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[over, lr, lc]);
        self.m.label(lr);
        self.block = lr;
        self.guard_write();
        self.leave();
        self.m.op_void(op::RETURN, &[]);
        self.m.label(lc);
        self.block = lc;
    }

    /// A call of the guard function (written once, after main).
    fn guard_write(&mut self) {
        if self.guard_fn == 0 {
            self.guard_fn = self.m.id();
        }
        let void = self.t.void;
        self.m.op(57, void, &[self.guard_fn]); // OpFunctionCall
    }

    /// The guard function: dword 0 += 1, dword 1 = the guard id, when the
    /// host gave a buffer.
    pub(super) fn write_guard_fn(&mut self) {
        if self.guard_fn == 0 {
            return;
        }
        let (u, q, b, void) = (self.t.u32_, self.t.u64_, self.t.bool_, self.t.void);
        let fty = self.m.t_func(void);
        self.m.op_to(op::FUNCTION, void, self.guard_fn, &[0, fty]);
        let entry = self.m.id();
        self.m.label(entry);
        let (lo, hi) = (self.push_load(2), self.push_load(3));
        let (lo, hi) = (self.m.op(op::U_CONVERT, q, &[lo]), self.m.op(op::U_CONVERT, q, &[hi]));
        let k32 = self.c32(32);
        let hi = self.m.op(op::SHIFT_LEFT_LOGICAL, q, &[hi, k32]);
        let addr = self.m.op(op::BITWISE_OR, q, &[hi, lo]);
        let z = self.m.c_u64(0);
        let some = self.m.op(op::I_NOT_EQUAL, b, &[addr, z]);
        let (lw, lm) = (self.m.id(), self.m.id());
        self.m.op_void(op::SELECTION_MERGE, &[lm, 0]);
        self.m.op_void(op::BRANCH_CONDITIONAL, &[some, lw, lm]);
        self.m.label(lw);
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
        let count = self.m.op(op::CONVERT_U_TO_PTR, p, &[addr]);
        let (scope, sem, one) = (self.c32(en::SCOPE_DEVICE), self.c32(0), self.c32(1));
        self.m.op(op::ATOMIC_I_ADD, u, &[count, scope, sem, one]);
        let four = self.m.c_u64(4);
        let a1 = self.m.op(op::I_ADD, q, &[addr, four]);
        let idp = self.m.op(op::CONVERT_U_TO_PTR, p, &[a1]);
        let id = self.c32(self.st.guard_id);
        self.m.op_void(op::STORE, &[idp, id, en::MEM_ALIGNED, 4]);
        self.m.op_void(op::BRANCH, &[lm]);
        self.m.label(lm);
        self.m.op_void(op::RETURN, &[]);
        self.m.op_void(op::FUNCTION_END, &[]);
    }
}
