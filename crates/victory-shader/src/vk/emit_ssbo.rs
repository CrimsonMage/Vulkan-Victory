//! EXPERIMENT (variant e): title memory through storage-buffer descriptors
//! instead of device addresses. Each memory-map range is one raw storage
//! buffer of set 3 binding 1 (`uint data[]` and its byte and short twins,
//! aliasing); a translated address is (range id << 32 | byte offset), and
//! an access indexes the range's buffer, where robustBufferAccess2 makes
//! reads past it 0 and drops writes.

use super::*;
use crate::vk::tir::Node;

/// The offset an unmapped address gets: past any range, so the access is
/// out of bounds.
pub(super) const UNMAPPED: u64 = 0xFFFF_FFF0;

/// Whether this build of a module takes variant e.
pub(super) fn ssbo_mode() -> bool {
    std::env::var("VICTORY_VK_EXP").as_deref() == Ok("e")
}

impl Emit<'_> {
    /// The binding's variable viewed with elements of `s` (U8, U16, U32).
    fn ssbo_var(&mut self, s: S) -> Id {
        if let Some(&v) = self.ssbo_vars.get(&s) {
            return v;
        }
        let et = self.scalar(s);
        let bytes = match s {
            S::U8 => 1,
            S::U16 => 2,
            _ => 4,
        };
        let rta = self.m.t_runtime_array_plain(et);
        self.m.decorate(rta, en::DEC_ARRAY_STRIDE, &[bytes]);
        let st = self.m.t_struct(&[rta], 100 + bytes);
        self.m.decorate(st, en::DEC_BLOCK, &[]);
        self.m.member_decorate(st, 0, en::DEC_OFFSET, &[0]);
        let arr = self.m.t_runtime_array_plain(st);
        let p = self.m.t_ptr(en::SC_STORAGE_BUFFER, arr);
        let v = self.m.variable(p, en::SC_STORAGE_BUFFER);
        self.m.decorate(v, en::DEC_DESCRIPTOR_SET, &[3]);
        self.m.decorate(v, en::DEC_BINDING, &[1]);
        self.m.capability(en::CAP_RUNTIME_DESCRIPTOR_ARRAY);
        self.m.capability(5301); // ShaderNonUniform
        self.m.capability(5308); // StorageBufferArrayNonUniformIndexing
        self.interface.push(v);
        self.ssbo_vars.insert(s, v);
        v
    }

    /// A pointer to element `k` (counted in `s`) past the byte offset of
    /// the translated address `enc`.
    fn ssbo_ptr(&mut self, enc: Id, s: S, k: u32) -> Id {
        let (u, q) = (self.t.u32_, self.t.u64_);
        let var = self.ssbo_var(s);
        let k32 = self.c32(32);
        let rid = self.m.op(op::SHIFT_RIGHT_LOGICAL, q, &[enc, k32]);
        let rid = self.m.op(op::U_CONVERT, u, &[rid]);
        let off = self.m.op(op::U_CONVERT, u, &[enc]);
        let shift = match s {
            S::U8 => 0,
            S::U16 => 1,
            _ => 2,
        };
        let ks = self.c32(shift);
        let idx = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[off, ks]);
        let kk = self.c32(k);
        let idx = self.m.op(op::I_ADD, u, &[idx, kk]);
        let et = self.scalar(s);
        let p = self.m.t_ptr(en::SC_STORAGE_BUFFER, et);
        let z = self.c32(0);
        let ptr = self.m.op(op::ACCESS_CHAIN, p, &[var, rid, z, idx]);
        self.m.decorate(ptr, en::DEC_NON_UNIFORM, &[]);
        ptr
    }

    /// One u32 at a translated address.
    pub(super) fn ssbo_load_u32(&mut self, enc: Id) -> Id {
        let u = self.t.u32_;
        let ptr = self.ssbo_ptr(enc, S::U32, 0);
        let x = self.m.op(op::LOAD, u, &[ptr]);
        self.m.decorate(x, en::DEC_NON_UNIFORM, &[]);
        x
    }

    /// The element type an access of `s` goes through, and whether the
    /// value is bitcast from it.
    fn ssbo_elem(s: S) -> (S, bool) {
        match s {
            S::U8 => (S::U8, false),
            S::U16 => (S::U16, false),
            S::F16 => (S::U16, true),
            S::F32 => (S::U32, true),
            _ => (S::U32, false),
        }
    }

    /// `Op::Load`, `Op::Store`, `Op::Atomic` through the range buffers
    /// (vectors one component at a time).
    pub(super) fn ssbo_memory_op(&mut self, n: &Node, args: &[Id], tid: Id) -> Id {
        match n.op {
            Op::Load { .. } => {
                let t = n.ty;
                let (es, cast) = Self::ssbo_elem(t.s);
                let (et, st) = (self.scalar(es), self.scalar(t.s));
                let parts: Vec<Id> = (0..u32::from(t.n))
                    .map(|k| {
                        let ptr = self.ssbo_ptr(args[0], es, k);
                        let x = self.m.op(op::LOAD, et, &[ptr]);
                        if cast { self.m.op(op::BITCAST, st, &[x]) } else { x }
                    })
                    .collect();
                if parts.len() == 1 { parts[0] } else { self.m.op(op::COMPOSITE_CONSTRUCT, tid, &parts) }
            }
            Op::Store { .. } => {
                let vt = self.f.ty(n.args[1]);
                let a = self.predicated_addr(args[0]);
                let (es, cast) = Self::ssbo_elem(vt.s);
                let (et, st) = (self.scalar(es), self.scalar(vt.s));
                for k in 0..u32::from(vt.n) {
                    let x = if vt.n == 1 { args[1] } else { self.m.op(op::COMPOSITE_EXTRACT, st, &[args[1], k]) };
                    let x = if cast { self.m.op(op::BITCAST, et, &[x]) } else { x };
                    let ptr = self.ssbo_ptr(a, es, k);
                    self.m.op_void(op::STORE, &[ptr, x]);
                }
                0
            }
            Op::Atomic(o) => {
                let a = self.predicated_addr(args[0]);
                let ptr = self.ssbo_ptr(a, S::U32, 0);
                self.atomic(o, ptr, en::SCOPE_DEVICE, args)
            }
            _ => unreachable!(),
        }
    }
}
