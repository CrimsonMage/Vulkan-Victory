//! Memory, LDS, wave, output and pick instructions.

use super::*;
use crate::vk::tir::Node;

impl Emit<'_> {
    /// Push constant dword `k`.
    pub(super) fn push_load(&mut self, k: u32) -> Id {
        let var = self.push_var();
        let u = self.t.u32_;
        let p = self.m.t_ptr(en::SC_PUSH_CONSTANT, u);
        let (z, kk) = (self.c32(0), self.c32(k));
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, z, kk]);
        self.m.op(op::LOAD, u, &[ptr])
    }

    /// Title memory through a physical pointer (stores and atomics of the
    /// lanes outside a flat guard go to the sink).
    pub(super) fn memory_op(&mut self, n: &Node, args: &[Id], tid: Id) -> Id {
        match n.op {
            Op::Load { align } => {
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, tid);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[args[0]]);
                self.m.op(op::LOAD, tid, &[ptr, en::MEM_ALIGNED, align])
            }
            Op::Store { align } => {
                let vt = self.f.ty(n.args[1]);
                let vtid = self.ty(vt);
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, vtid);
                let a = self.predicated_addr(args[0]);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[a]);
                self.m.op_void(op::STORE, &[ptr, args[1], en::MEM_ALIGNED, align]);
                0
            }
            Op::Atomic(o) => {
                let u = self.t.u32_;
                let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
                let a = self.predicated_addr(args[0]);
                let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[a]);
                self.atomic(o, ptr, en::SCOPE_DEVICE, args)
            }
            _ => unreachable!(),
        }
    }

    /// An atomic on a pointer: args[1] the value (args[2] the comparator).
    fn atomic(&mut self, o: u16, ptr: Id, scope: u32, args: &[Id]) -> Id {
        let u = self.t.u32_;
        let (scope, sem) = (self.c32(scope), self.c32(0));
        if o == op::ATOMIC_F_MIN_EXT || o == op::ATOMIC_F_MAX_EXT {
            return self.atomic_float(o, ptr, scope, sem, args[1]);
        }
        if o == op::ATOMIC_COMPARE_EXCHANGE {
            self.m.op(o, u, &[ptr, scope, sem, sem, args[1], args[2]])
        } else {
            self.m.op(o, u, &[ptr, scope, sem, args[1]])
        }
    }

    /// A float min/max atomic on a u32 pointer, as compare-exchanges (the
    /// float min/max atomics are an extension NVIDIA lacks): four attempts
    /// in a row, each computing the wanted value from what the last one
    /// read. An attempt after a successful one stores the same value again,
    /// so no attempt needs a branch; a value that lost four races in a row
    /// is dropped. Returns the first value read.
    fn atomic_float(&mut self, o: u16, ptr: Id, scope: Id, sem: Id, x: Id) -> Id {
        let (u, f) = (self.t.u32_, self.t.f32_);
        let pick = if o == op::ATOMIC_F_MIN_EXT { glsl::N_MIN } else { glsl::N_MAX };
        let fx = self.m.op(op::BITCAST, f, &[x]);
        let first = self.m.op(op::ATOMIC_LOAD, u, &[ptr, scope, sem]);
        let mut seen = first;
        for _ in 0..4 {
            let was = self.m.op(op::BITCAST, f, &[seen]);
            let want = self.m.ext(pick, f, &[was, fx]);
            let want = self.m.op(op::BITCAST, u, &[want]);
            seen = self.m.op(op::ATOMIC_COMPARE_EXCHANGE, u, &[ptr, scope, sem, sem, want, seen]);
        }
        first
    }

    /// LDS dwords (the index clamped; the lanes outside a flat guard write
    /// the spare dword).
    pub(super) fn lds_op(&mut self, n: &Node, args: &[Id]) -> Id {
        let lds = self.lds_var();
        let u = self.t.u32_;
        let p = self.m.t_ptr(en::SC_WORKGROUP, u);
        let max = self.c32(self.iface.lds_dwords.max(1) - 1);
        let mut idx = self.m.ext(glsl::U_MIN, u, &[args[0], max]);
        if let Some(c) = self.pred
            && n.op != Op::LdsLoad
        {
            let spare = self.c32(self.iface.lds_dwords.max(1));
            idx = self.m.op(op::SELECT, u, &[c, idx, spare]);
        }
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[lds, idx]);
        match n.op {
            Op::LdsLoad => self.m.op(op::LOAD, u, &[ptr]),
            Op::LdsStore => {
                self.m.op_void(op::STORE, &[ptr, args[1]]);
                0
            }
            Op::LdsAtomic(o) => self.atomic(o, ptr, en::SCOPE_WORKGROUP, args),
            _ => unreachable!(),
        }
    }

    /// A subgroup operation (`g`: its group operation, or none) and the
    /// capability it needs.
    pub(super) fn wave_op(&mut self, o: u16, g: u32, args: &[Id], tid: Id) -> Id {
        self.m.capability(en::CAP_GROUP_NON_UNIFORM);
        let cap = match o {
            op::GROUP_NON_UNIFORM_BALLOT | op::GROUP_NON_UNIFORM_BROADCAST_FIRST | op::GROUP_NON_UNIFORM_BROADCAST => {
                en::CAP_GROUP_NON_UNIFORM_BALLOT
            }
            op::GROUP_NON_UNIFORM_ANY | op::GROUP_NON_UNIFORM_ALL => en::CAP_GROUP_NON_UNIFORM_VOTE,
            op::GROUP_NON_UNIFORM_SHUFFLE | op::GROUP_NON_UNIFORM_SHUFFLE_XOR => en::CAP_GROUP_NON_UNIFORM_SHUFFLE,
            _ => en::CAP_GROUP_NON_UNIFORM_ARITHMETIC,
        };
        self.m.capability(cap);
        let scope = self.c32(en::SCOPE_SUBGROUP);
        let mut ops = vec![scope];
        if g != u32::MAX {
            ops.push(g);
        }
        ops.extend_from_slice(args);
        self.m.op(o, tid, &ops)
    }

    /// An output component: user outputs of vertex and mesh programs at
    /// their semantics' locations (none: not linked, dropped).
    pub(super) fn output_op(&mut self, n: &Node, slot: u16, comp: u8, args: &[Id]) -> Id {
        let (slot, comp) = if slot < 0x100 && self.st.stage != HwStage::Ps {
            match self.st.locs.get(&(slot as u8, comp)) {
                Some(&(l, c)) => (u16::from(l), c),
                None => return 0,
            }
        } else {
            (slot, comp)
        };
        let (var, elem) = self.output(slot);
        if self.st.stage == HwStage::Ms {
            self.mesh_output_op(n, slot, comp, args, var, elem);
            return 0;
        }
        let at = self.f.ty(n.args[0]);
        let at = self.ty(at);
        let x = if at != elem { self.m.op(op::BITCAST, elem, &[args[0]]) } else { args[0] };
        let ptr = match slot {
            OUT_DEPTH | OUT_STENCIL | OUT_POINT_SIZE | OUT_LAYER | OUT_VIEWPORT => var,
            _ => {
                let p = self.m.t_ptr(en::SC_OUTPUT, elem);
                let k = self.c32(u32::from(comp));
                self.m.op(op::ACCESS_CHAIN, p, &[var, k])
            }
        };
        self.m.op_void(op::STORE, &[ptr, x]);
        0
    }

    /// args[0] picks one of the rest (clamped) through a function array.
    pub(super) fn pick(&mut self, v: V, args: &[Id]) -> Id {
        let var = self.picks[&v];
        let u = self.t.u32_;
        let p = self.m.t_ptr(en::SC_FUNCTION, u);
        for (i, &a) in args[1..].iter().enumerate() {
            let k = self.c32(i as u32);
            let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, k]);
            self.m.op_void(op::STORE, &[ptr, a]);
        }
        let max = self.c32((args.len() - 2) as u32);
        let idx = self.m.ext(glsl::U_MIN, u, &[args[0], max]);
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, idx]);
        self.m.op(op::LOAD, u, &[ptr])
    }
}
