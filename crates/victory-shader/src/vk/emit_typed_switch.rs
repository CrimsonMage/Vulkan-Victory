//! EXPERIMENT: the earlier typed-buffer decoder (a switch over the FORMAT
//! codes), chosen by `VICTORY_VK_TYPED=switch` to time against the
//! branchless one.

use super::*;

/// Whether this build of a module takes the switch decoder.
pub(super) fn switch_mode() -> bool {
    std::env::var("VICTORY_VK_TYPED").as_deref() == Ok("switch")
}

impl Emit<'_> {
    /// The typed-buffer decoder: a switch over the GFX10 FORMAT codes the
    /// title's views use; others read one 32-bit channel.
    /// A 2_10_10_10 unorm element (x in the low bits) as four float bits.
    fn typed_packed(&mut self, addr: Id) -> Vec<Id> {
        let (u, f) = (self.t.u32_, self.t.f32_);
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
        let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[addr]);
        let w = self.m.op(op::LOAD, u, &[ptr, en::MEM_ALIGNED, 4]);
        let mut at = 0;
        let mut ch = Vec::new();
        for width in [10u32, 10, 10, 2] {
            let (o, wv) = (self.c32(at), self.c32(width));
            at += width;
            let r = self.m.op(op::BIT_FIELD_U_EXTRACT, u, &[w, o, wv]);
            let fl = self.m.op(op::CONVERT_U_TO_F, f, &[r]);
            let k = self.m.c_f32((1.0 / ((1u64 << width) - 1) as f32).to_bits());
            let m = self.m.op(op::F_MUL, f, &[fl, k]);
            ch.push(self.m.op(op::BITCAST, u, &[m]));
        }
        ch
    }

    /// The channels of an element of `comps` x `bits` (num: 0 float, 1
    /// unorm, 2 snorm, 3 uint, 4 sint), missing ones 0 and alpha 1.
    fn typed_channels(&mut self, addr: Id, comps: u32, bits: u32, num: u32) -> Vec<Id> {
        let st = match bits {
            32 => S::U32,
            16 => S::U16,
            _ => S::U8,
        };
        let et = self.scalar(st);
        let lt = self.m.t_vec(et, comps);
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, lt);
        let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[addr]);
        let v = self.m.op(op::LOAD, lt, &[ptr, en::MEM_ALIGNED, (bits / 8).max(1)]);
        let mut ch = Vec::new();
        for c in 0..comps {
            let e = if comps == 1 { v } else { self.m.op(op::COMPOSITE_EXTRACT, et, &[v, c]) };
            ch.push(self.typed_channel(e, bits, num));
        }
        let (z, one_i, one_f) = (self.c32(0), self.c32(1), self.m.c_u32(1.0f32.to_bits()));
        while ch.len() < 4 {
            ch.push(match (ch.len(), num >= 3) {
                (3, true) => one_i,
                (3, false) => one_f,
                _ => z,
            });
        }
        ch
    }

    /// One channel's value as 32 bits.
    fn typed_channel(&mut self, e: Id, bits: u32, num: u32) -> Id {
        let (u, f) = (self.t.u32_, self.t.f32_);
        match (bits, num) {
            (32, _) => e,
            (16, 0) => {
                let h = self.scalar(S::F16);
                let hf = self.m.op(op::BITCAST, h, &[e]);
                let fl = self.m.op(op::F_CONVERT, f, &[hf]);
                self.m.op(op::BITCAST, u, &[fl])
            }
            (_, 1) => {
                let i = self.m.op(op::U_CONVERT, u, &[e]);
                let fl = self.m.op(op::CONVERT_U_TO_F, f, &[i]);
                let k = self.m.c_f32((1.0 / ((1u64 << bits) - 1) as f32).to_bits());
                let m = self.m.op(op::F_MUL, f, &[fl, k]);
                self.m.op(op::BITCAST, u, &[m])
            }
            (_, 2) => {
                let i = self.m.op(op::S_CONVERT, u, &[e]);
                let fl = self.m.op(op::CONVERT_S_TO_F, f, &[i]);
                let k = self.m.c_f32((1.0 / ((1u64 << (bits - 1)) - 1) as f32).to_bits());
                let m = self.m.op(op::F_MUL, f, &[fl, k]);
                let lo = self.m.c_f32((-1.0f32).to_bits());
                let c = self.m.ext(glsl::N_MAX, f, &[m, lo]);
                self.m.op(op::BITCAST, u, &[c])
            }
            (_, 3) => self.m.op(op::U_CONVERT, u, &[e]),
            _ => self.m.op(op::S_CONVERT, u, &[e]),
        }
    }

    pub(super) fn write_typed_switch(&mut self, fid: Id) {
        let (u, q) = (self.t.u32_, self.t.u64_);
        let v4 = self.m.t_vec(u, 4);
        let fty = self.m.global_fn_type(v4, &[q, u]);
        self.m.op_to(op::FUNCTION, v4, fid, &[0, fty]);
        let (addr, fmt) = (self.m.id(), self.m.id());
        self.m.op_to(55, q, addr, &[]);
        self.m.op_to(55, u, fmt, &[]);
        let entry = self.m.id();
        self.m.label(entry);
        // (codes, components, bits, num: 0 float 1 unorm 2 snorm 3 uint 4 sint)
        let table: &[(&[u32], u32, u32, u32)] = &[
            (&[22], 1, 32, 0),
            (&[20, 21], 1, 32, 3),
            (&[64], 2, 32, 0),
            (&[62, 63], 2, 32, 3),
            (&[74], 3, 32, 0),
            (&[72, 73], 3, 32, 3),
            (&[77], 4, 32, 0),
            (&[75, 76], 4, 32, 3),
            (&[13], 1, 16, 0),
            (&[29], 2, 16, 0),
            (&[71], 4, 16, 0),
            (&[7], 1, 16, 1),
            (&[23], 2, 16, 1),
            (&[65], 4, 16, 1),
            (&[8], 1, 16, 2),
            (&[24], 2, 16, 2),
            (&[66], 4, 16, 2),
            (&[11], 1, 16, 3),
            (&[27], 2, 16, 3),
            (&[69], 4, 16, 3),
            (&[12], 1, 16, 4),
            (&[28], 2, 16, 4),
            (&[70], 4, 16, 4),
            (&[1], 1, 8, 1),
            (&[14], 2, 8, 1),
            (&[56], 4, 8, 1),
            (&[2], 1, 8, 2),
            (&[15], 2, 8, 2),
            (&[57], 4, 8, 2),
            (&[5], 1, 8, 3),
            (&[18], 2, 8, 3),
            (&[60], 4, 8, 3),
            (&[6], 1, 8, 4),
            (&[19], 2, 8, 4),
            (&[61], 4, 8, 4),
            (&[50], 4, 0, 1),
        ];
        let merge = self.m.id();
        let default = self.m.id();
        let labels: Vec<Id> = table.iter().map(|_| self.m.id()).collect();
        let mut sw = vec![fmt, default];
        for (i, e) in table.iter().enumerate() {
            for &c in e.0 {
                sw.push(c);
                sw.push(labels[i]);
            }
        }
        self.m.op_void(op::SELECTION_MERGE, &[merge, 0]);
        self.m.op_void(251, &sw); // OpSwitch
        let one_f = self.m.c_u32(1.0f32.to_bits());
        let z = self.c32(0);
        let mut incoming = Vec::new();
        for (i, &(_, comps, bits, num)) in table.iter().enumerate() {
            self.m.label(labels[i]);
            let ch = if bits == 0 { self.typed_packed(addr) } else { self.typed_channels(addr, comps, bits, num) };
            let r = self.m.op(op::COMPOSITE_CONSTRUCT, v4, &ch);
            incoming.push((r, labels[i]));
            self.m.op_void(op::BRANCH, &[merge]);
        }
        self.m.label(default);
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
        let ptr = self.m.op(op::CONVERT_U_TO_PTR, p, &[addr]);
        let w = self.m.op(op::LOAD, u, &[ptr, en::MEM_ALIGNED, 4]);
        let r = self.m.op(op::COMPOSITE_CONSTRUCT, v4, &[w, z, z, one_f]);
        incoming.push((r, default));
        self.m.op_void(op::BRANCH, &[merge]);
        self.m.label(merge);
        let mut ops = Vec::new();
        for (r, l) in incoming {
            ops.push(r);
            ops.push(l);
        }
        let res = self.m.op(op::PHI, v4, &ops);
        self.m.op_void(254, &[res]);
        self.m.op_void(op::FUNCTION_END, &[]);
    }
}
