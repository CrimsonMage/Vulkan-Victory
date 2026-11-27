//! The typed-buffer decoder (`Helper::Typed`: device address, the format's
//! shape -> uvec4, floats as their bits, missing channels 0 and alpha 1),
//! without branches: the shape (components, bits, number type; `entry`)
//! is worked out once per V# from its FORMAT (lift_mem.rs) and each channel
//! is a bit field of the element, converted by its number type. A switch
//! over the formats, inlined at every typed load, made NVIDIA's compiler
//! take minutes on a program of 3,000 of them
//! (CS_ScreenProbes_ActivateProbes).

use super::*;

/// Number types.
const FLOAT: u32 = 0;
const UNORM: u32 = 1;
const SNORM: u32 = 2;
const UINT: u32 = 3;
const SINT: u32 = 4;

/// GFX10 buffer FORMAT codes the title's views use: (codes, components,
/// bits per channel, number type); 10 bits: 10_10_10_2 (the fourth
/// channel 2 bits wide). Any other code reads one 32-bit channel.
const FORMATS: &[(&[u32], u32, u32, u32)] = &[
    (&[22], 1, 32, FLOAT),
    (&[20, 21], 1, 32, UINT),
    (&[64], 2, 32, FLOAT),
    (&[62, 63], 2, 32, UINT),
    (&[74], 3, 32, FLOAT),
    (&[72, 73], 3, 32, UINT),
    (&[77], 4, 32, FLOAT),
    (&[75, 76], 4, 32, UINT),
    (&[13], 1, 16, FLOAT),
    (&[29], 2, 16, FLOAT),
    (&[71], 4, 16, FLOAT),
    (&[7], 1, 16, UNORM),
    (&[23], 2, 16, UNORM),
    (&[65], 4, 16, UNORM),
    (&[8], 1, 16, SNORM),
    (&[24], 2, 16, SNORM),
    (&[66], 4, 16, SNORM),
    (&[11], 1, 16, UINT),
    (&[27], 2, 16, UINT),
    (&[69], 4, 16, UINT),
    (&[12], 1, 16, SINT),
    (&[28], 2, 16, SINT),
    (&[70], 4, 16, SINT),
    (&[1], 1, 8, UNORM),
    (&[14], 2, 8, UNORM),
    (&[56], 4, 8, UNORM),
    (&[2], 1, 8, SNORM),
    (&[15], 2, 8, SNORM),
    (&[57], 4, 8, SNORM),
    (&[5], 1, 8, UINT),
    (&[18], 2, 8, UINT),
    (&[60], 4, 8, UINT),
    (&[6], 1, 8, SINT),
    (&[19], 2, 8, SINT),
    (&[61], 4, 8, SINT),
    (&[50], 4, 10, UNORM),
];

/// A shape: components in bits 0-2, bits per channel in 8-15, number type
/// in 16-18.
fn entry(comps: u32, bits: u32, num: u32) -> u32 {
    comps | bits << 8 | num << 16
}

/// The table over all 128 FORMAT codes.
pub(crate) fn format_table() -> [u32; 128] {
    let mut t = [entry(1, 32, FLOAT); 128];
    for &(codes, comps, bits, num) in FORMATS {
        for &c in codes {
            t[c as usize] = entry(comps, bits, num);
        }
    }
    t
}

impl Emit<'_> {
    /// `CodeTable(t)`: the program's code from the table's start, a private
    /// array with its constant, at the dword index (0 past the end).
    pub(super) fn write_code_table(&mut self, fid: Id, t: usize) {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let table = self.f.tables[t].clone();
        let len = table.len().max(1) as u32;
        let arr = self.m.t_array_plain(u, len);
        let words: Vec<Id> = if table.is_empty() { vec![self.c32(0)] } else { table.iter().map(|&w| self.c32(w)).collect() };
        let init = self.m.c_composite(arr, &words);
        let pa = self.m.t_ptr(en::SC_PRIVATE, arr);
        let var = self.m.variable_init(pa, en::SC_PRIVATE, init);
        self.interface.push(var);
        let fty = self.m.global_fn_type(u, &[u]);
        self.m.op_to(op::FUNCTION, u, fid, &[0, fty]);
        let i = self.m.id();
        self.m.op_to(55, u, i, &[]);
        let entry = self.m.id();
        self.m.label(entry);
        let last = self.c32(len - 1);
        let at = self.m.ext(glsl::U_MIN, u, &[i, last]);
        let pu = self.m.t_ptr(en::SC_PRIVATE, u);
        let ep = self.m.op(op::ACCESS_CHAIN, pu, &[var, at]);
        let e = self.m.op(op::LOAD, u, &[ep]);
        let past = self.m.op(op::U_GREATER_THAN, b, &[i, last]);
        let past = if table.is_empty() { self.m.c_bool(true) } else { past };
        let z = self.c32(0);
        let r = self.m.op(op::SELECT, u, &[past, z, e]);
        self.m.op_void(254, &[r]); // OpReturnValue
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    /// `TypedShape`: the table, a private array with its constant, at the
    /// FORMAT code.
    pub(super) fn write_typed_shape(&mut self, fid: Id) {
        let u = self.t.u32_;
        let arr = self.m.t_array_plain(u, 128);
        let words: Vec<Id> = format_table().iter().map(|&w| self.c32(w)).collect();
        let init = self.m.c_composite(arr, &words);
        let pa = self.m.t_ptr(en::SC_PRIVATE, arr);
        let table = self.m.variable_init(pa, en::SC_PRIVATE, init);
        self.interface.push(table);
        let fty = self.m.global_fn_type(u, &[u]);
        self.m.op_to(op::FUNCTION, u, fid, &[0, fty]);
        let fmt = self.m.id();
        self.m.op_to(55, u, fmt, &[]);
        let entry = self.m.id();
        self.m.label(entry);
        let k127 = self.c32(127);
        let i = self.m.op(op::BITWISE_AND, u, &[fmt, k127]);
        let pu = self.m.t_ptr(en::SC_PRIVATE, u);
        let ep = self.m.op(op::ACCESS_CHAIN, pu, &[table, i]);
        let e = self.m.op(op::LOAD, u, &[ep]);
        self.m.op_void(254, &[e]); // OpReturnValue
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    pub(super) fn write_typed(&mut self, fid: Id) {
        let (u, q, b) = (self.t.u32_, self.t.u64_, self.t.bool_);
        let v4 = self.m.t_vec(u, 4);
        let fty = self.m.global_fn_type(v4, &[q, u]);
        self.m.op_to(op::FUNCTION, v4, fid, &[0, fty]);
        let (addr, e) = (self.m.id(), self.m.id());
        self.m.op_to(55, q, addr, &[]);
        self.m.op_to(55, u, e, &[]);
        let entry = self.m.id();
        self.m.label(entry);
        let field = |s: &mut Self, at: u32, n: u32| {
            let (a, w) = (s.c32(at), s.c32(n));
            s.m.op(op::BIT_FIELD_U_EXTRACT, u, &[e, a, w])
        };
        let comps = field(self, 0, 3);
        let bits = field(self, 8, 8);
        let num = field(self, 16, 3);
        // the element's first dword and its bit there
        let k3 = self.m.c_u64(3);
        let nk3 = self.m.c_u64(!3);
        let a4 = self.m.op(op::BITWISE_AND, q, &[addr, nk3]);
        let lo2 = self.m.op(op::BITWISE_AND, q, &[addr, k3]);
        let lo2 = self.m.op(op::U_CONVERT, u, &[lo2]);
        let k3u = self.c32(3);
        let first = self.m.op(op::SHIFT_LEFT_LOGICAL, u, &[lo2, k3u]);
        let is = |s: &mut Self, x: Id, k: u32| {
            let kk = s.c32(k);
            s.m.op(op::I_EQUAL, b, &[x, kk])
        };
        let float = is(self, num, FLOAT);
        let unorm = is(self, num, UNORM);
        let snorm = is(self, num, SNORM);
        let sint = is(self, num, SINT);
        let ints = {
            let k = self.c32(UINT);
            self.m.op(op::U_GREATER_THAN_EQUAL, b, &[num, k])
        };
        let (z, one) = (self.c32(0), self.c32(1));
        let one_f = self.c32(1.0f32.to_bits());
        let alpha = self.m.op(op::SELECT, u, &[ints, one, one_f]);
        let mut ch = Vec::new();
        for c in 0..4u32 {
            let kc = self.c32(c);
            // the channel's width: 10_10_10_2 has a 2-bit fourth
            let w = if c == 3 {
                let ten = is(self, bits, 10);
                let two = self.c32(2);
                self.m.op(op::SELECT, u, &[ten, two, bits])
            } else {
                bits
            };
            // a channel past the format's reads the first one's dword (no
            // read past the element)
            let have = self.m.op(op::U_LESS_THAN, b, &[kc, comps]);
            let at = self.m.op(op::I_MUL, u, &[kc, bits]);
            let at = self.m.op(op::SELECT, u, &[have, at, z]);
            let g = self.m.op(op::I_ADD, u, &[first, at]);
            let k5 = self.c32(5);
            let dw = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[g, k5]);
            let k31 = self.c32(31);
            let sh = self.m.op(op::BITWISE_AND, u, &[g, k31]);
            let raw = self.typed_field(a4, dw, sh, w);
            let x = self.typed_convert(raw, w, [float, unorm, snorm, sint]);
            // channels past the format's: 0, alpha 1
            let none = if c == 3 { alpha } else { z };
            ch.push(self.m.op(op::SELECT, u, &[have, x, none]));
        }
        let r = self.m.op(op::COMPOSITE_CONSTRUCT, v4, &ch);
        self.m.op_void(254, &[r]); // OpReturnValue
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    /// Bits `sh..sh + w` of the dwords at `a4 + 4 dw` (and the next, read
    /// only when the field reaches into it: the address stays inside the
    /// element).
    fn typed_field(&mut self, a4: Id, dw: Id, sh: Id, w: Id) -> Id {
        let (u, q, b) = (self.t.u32_, self.t.u64_, self.t.bool_);
        let k2 = self.c32(2);
        let off = self.m.op(op::SHIFT_LEFT_LOGICAL, u, &[dw, k2]);
        let off = self.m.op(op::U_CONVERT, q, &[off]);
        let a0 = self.m.op(op::I_ADD, q, &[a4, off]);
        let end = self.m.op(op::I_ADD, u, &[sh, w]);
        let k32 = self.c32(32);
        let spill = self.m.op(op::U_GREATER_THAN, b, &[end, k32]);
        let (k4, k0) = (self.m.c_u64(4), self.m.c_u64(0));
        let step = self.m.op(op::SELECT, q, &[spill, k4, k0]);
        let a1 = self.m.op(op::I_ADD, q, &[a0, step]);
        let p = self.m.t_ptr(en::SC_PHYSICAL_STORAGE_BUFFER, u);
        let p0 = self.m.op(op::CONVERT_U_TO_PTR, p, &[a0]);
        let p1 = self.m.op(op::CONVERT_U_TO_PTR, p, &[a1]);
        let d0 = self.m.op(op::LOAD, u, &[p0, en::MEM_ALIGNED, 4]);
        let d1 = self.m.op(op::LOAD, u, &[p1, en::MEM_ALIGNED, 4]);
        let lo = self.m.op(op::U_CONVERT, q, &[d0]);
        let hi = self.m.op(op::U_CONVERT, q, &[d1]);
        let k32q = self.c32(32);
        let hi = self.m.op(op::SHIFT_LEFT_LOGICAL, q, &[hi, k32q]);
        let both = self.m.op(op::BITWISE_OR, q, &[hi, lo]);
        let x = self.m.op(op::SHIFT_RIGHT_LOGICAL, q, &[both, sh]);
        let x = self.m.op(op::U_CONVERT, u, &[x]);
        // the low w bits (w = 32: all)
        let z = self.c32(0);
        self.m.op(op::BIT_FIELD_U_EXTRACT, u, &[x, z, w])
    }

    /// A channel of `w` bits as 32: float (16-bit halves widened), unorm,
    /// snorm, uint, sint; `kinds` = (float, unorm, snorm, sint) flags.
    fn typed_convert(&mut self, raw: Id, w: Id, kinds: [Id; 4]) -> Id {
        let (u, f) = (self.t.u32_, self.t.f32_);
        let [float, unorm, snorm, sint] = kinds;
        let z = self.c32(0);
        let signed = self.m.op(op::BIT_FIELD_S_EXTRACT, u, &[raw, z, w]);
        // 16-bit float: the low half widened
        let v2 = self.m.t_vec(f, 2);
        let h = self.m.ext(glsl::UNPACK_HALF_2X16, v2, &[raw]);
        let h = self.m.op(op::COMPOSITE_EXTRACT, f, &[h, 0]);
        let h = self.m.op(op::BITCAST, u, &[h]);
        let k32 = self.c32(32);
        let full = self.m.op(op::I_EQUAL, self.t.bool_, &[w, k32]);
        let fl = self.m.op(op::SELECT, u, &[full, raw, h]);
        // unorm: raw / (2^w - 1); snorm: max(signed / (2^(w-1) - 1), -1),
        // each the float nearest the exact quotient (`Lift::norm`; the
        // table's widths: unorm 8, 16, 10 and 2, snorm 8 and 16)
        let bool_ = self.t.bool_;
        let w16 = {
            let k = self.c32(16);
            self.m.op(op::I_EQUAL, bool_, &[w, k])
        };
        let w8 = {
            let k = self.c32(8);
            self.m.op(op::I_EQUAL, bool_, &[w, k])
        };
        let w10 = {
            let k = self.c32(10);
            self.m.op(op::I_EQUAL, bool_, &[w, k])
        };
        let un = {
            let pick = |s: &mut Self, c: Id, a: u64, b: Id| {
                let a = s.m.c_u64(a);
                s.m.op(op::SELECT, s.t.u64_, &[c, a, b])
            };
            let k = self.m.c_u64(norm_k(2, 40));
            let k = pick(self, w10, norm_k(10, 40), k);
            let k = pick(self, w8, norm_k(8, 40), k);
            let k = pick(self, w16, norm_k(16, 48), k);
            let (s40, s48) = (self.c32(scale(40)), self.c32(scale(48)));
            let sc = self.m.op(op::SELECT, u, &[w16, s48, s40]);
            self.norm_quotient(raw, k, sc)
        };
        let sn = {
            let mag = self.m.ext(glsl::S_ABS, u, &[signed]);
            let (d8, d16) = (self.c32(127), self.c32(32767));
            let d = self.m.op(op::SELECT, u, &[w16, d16, d8]);
            let mag = self.m.ext(glsl::U_MIN, u, &[mag, d]);
            let (k8, k16) = (self.m.c_u64(norm_k(7, 42)), self.m.c_u64(norm_k(15, 45)));
            let k = self.m.op(op::SELECT, self.t.u64_, &[w16, k16, k8]);
            let (s42, s45) = (self.c32(scale(42)), self.c32(scale(45)));
            let sc = self.m.op(op::SELECT, u, &[w16, s45, s42]);
            let x = self.norm_quotient(mag, k, sc);
            let xf = self.m.op(op::BITCAST, f, &[x]);
            let nx = self.m.op(op::F_NEGATE, f, &[xf]);
            let nx = self.m.op(op::BITCAST, u, &[nx]);
            let neg = self.m.op(op::S_LESS_THAN, bool_, &[signed, z]);
            self.m.op(op::SELECT, u, &[neg, nx, x])
        };
        let x = self.m.op(op::SELECT, u, &[sint, signed, raw]);
        let x = self.m.op(op::SELECT, u, &[snorm, sn, x]);
        let x = self.m.op(op::SELECT, u, &[unorm, un, x]);
        self.m.op(op::SELECT, u, &[float, fl, x])
    }
}

/// `(2^sh - 1) / (2^v - 1)` (`sh` a multiple of `v`).
fn norm_k(v: u32, sh: u32) -> u64 {
    ((1u64 << sh) - 1) / ((1u64 << v) - 1)
}

/// The bits of 2^-sh.
fn scale(sh: u32) -> u32 {
    (127 - sh) << 23
}

impl Emit<'_> {
    /// The float nearest `a / d` for a code `0 <= a <= d` (bits), `k` =
    /// [`norm_k`] of `d` and `sc` the matching [`scale`]: `(a k | 1)`
    /// converted once, scaled; 0 for 0.
    fn norm_quotient(&mut self, a: Id, k: Id, sc: Id) -> Id {
        let (u, q, f, b) = (self.t.u32_, self.t.u64_, self.t.f32_, self.t.bool_);
        let a64 = self.m.op(op::U_CONVERT, q, &[a]);
        let p = self.m.op(op::I_MUL, q, &[a64, k]);
        let one = self.m.c_u64(1);
        let p = self.m.op(op::BITWISE_OR, q, &[p, one]);
        let x = self.m.op(op::CONVERT_U_TO_F, f, &[p]);
        let scf = self.m.op(op::BITCAST, f, &[sc]);
        let x = self.m.op(op::F_MUL, f, &[x, scf]);
        let x = self.m.op(op::BITCAST, u, &[x]);
        let z = self.c32(0);
        let zero = self.m.op(op::I_EQUAL, b, &[a, z]);
        self.m.op(op::SELECT, u, &[zero, z, x])
    }

    /// `TypedPack`: the encoder, the decoder's inverse. (shape, the four
    /// channel registers, d16: 1 when they are 16-bit values) -> the
    /// element's dwords (bits past it 0). Floats round to nearest even,
    /// unorm and snorm clamp (NaN to 0) and round to nearest even, integers
    /// clamp to the channel's range.
    pub(super) fn write_typed_pack(&mut self, fid: Id) {
        let (u, b) = (self.t.u32_, self.t.bool_);
        let v4 = self.m.t_vec(u, 4);
        let fty = self.m.global_fn_type(v4, &[u, v4, u]);
        self.m.op_to(op::FUNCTION, v4, fid, &[0, fty]);
        let (e, vals, d16) = (self.m.id(), self.m.id(), self.m.id());
        self.m.op_to(55, u, e, &[]);
        self.m.op_to(55, v4, vals, &[]);
        self.m.op_to(55, u, d16, &[]);
        let entry = self.m.id();
        self.m.label(entry);
        let field = |s: &mut Self, at: u32, n: u32| {
            let (a, w) = (s.c32(at), s.c32(n));
            s.m.op(op::BIT_FIELD_U_EXTRACT, u, &[e, a, w])
        };
        let comps = field(self, 0, 3);
        let bits = field(self, 8, 8);
        let num = field(self, 16, 3);
        let is = |s: &mut Self, x: Id, k: u32| {
            let kk = s.c32(k);
            s.m.op(op::I_EQUAL, b, &[x, kk])
        };
        let kinds = [is(self, num, FLOAT), is(self, num, UNORM), is(self, num, SNORM), is(self, num, SINT)];
        let ints = {
            let k = self.c32(UINT);
            self.m.op(op::U_GREATER_THAN_EQUAL, b, &[num, k])
        };
        let z = self.c32(0);
        let zb = self.m.op(op::I_NOT_EQUAL, b, &[d16, z]);
        let ten = is(self, bits, 10);
        let mut words = [z; 4];
        for c in 0..4u32 {
            let kc = self.c32(c);
            let w = if c == 3 {
                let two = self.c32(2);
                self.m.op(op::SELECT, u, &[ten, two, bits])
            } else {
                bits
            };
            let raw = self.m.op(op::COMPOSITE_EXTRACT, u, &[vals, c]);
            let raw = self.typed_widen(raw, zb, ints, kinds[3]);
            let r = self.typed_encode(raw, w, kinds);
            // the channel's dword and bit (every format keeps a channel in one dword)
            let at = self.m.op(op::I_MUL, u, &[kc, bits]);
            let k5 = self.c32(5);
            let dw = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[at, k5]);
            let k31 = self.c32(31);
            let sh = self.m.op(op::BITWISE_AND, u, &[at, k31]);
            let r = self.m.op(op::SHIFT_LEFT_LOGICAL, u, &[r, sh]);
            let have = self.m.op(op::U_LESS_THAN, b, &[kc, comps]);
            for (j, word) in words.iter_mut().enumerate() {
                let kj = self.c32(j as u32);
                let here = self.m.op(op::I_EQUAL, b, &[dw, kj]);
                let here = self.m.op(op::LOGICAL_AND, b, &[here, have]);
                let x = self.m.op(op::SELECT, u, &[here, r, z]);
                *word = self.m.op(op::BITWISE_OR, u, &[*word, x]);
            }
        }
        let r = self.m.op(op::COMPOSITE_CONSTRUCT, v4, &words);
        self.m.op_void(254, &[r]); // OpReturnValue
        self.m.op_void(op::FUNCTION_END, &[]);
    }

    /// A d16 channel as 32 bits: a half widened to a float, a 16-bit
    /// integer zero- or sign-extended; `d16` false: as it is.
    fn typed_widen(&mut self, raw: Id, d16: Id, ints: Id, sint: Id) -> Id {
        let (u, f) = (self.t.u32_, self.t.f32_);
        let (z, k16) = (self.c32(0), self.c32(16));
        let v2 = self.m.t_vec(f, 2);
        let h = self.m.ext(glsl::UNPACK_HALF_2X16, v2, &[raw]);
        let h = self.m.op(op::COMPOSITE_EXTRACT, f, &[h, 0]);
        let h = self.m.op(op::BITCAST, u, &[h]);
        let lo = self.m.op(op::BIT_FIELD_U_EXTRACT, u, &[raw, z, k16]);
        let slo = self.m.op(op::BIT_FIELD_S_EXTRACT, u, &[raw, z, k16]);
        let i = self.m.op(op::SELECT, u, &[sint, slo, lo]);
        let x = self.m.op(op::SELECT, u, &[ints, i, h]);
        self.m.op(op::SELECT, u, &[d16, x, raw])
    }

    /// A 32-bit channel value as a `w`-bit field (low bits) of its number
    /// type; `kinds` = (float, unorm, snorm, sint) flags.
    fn typed_encode(&mut self, raw: Id, w: Id, kinds: [Id; 4]) -> Id {
        let (u, f, b) = (self.t.u32_, self.t.f32_, self.t.bool_);
        let [float, unorm, snorm, sint] = kinds;
        let one = self.c32(1);
        // the field's mask (all ones for 32 bits), the signed range
        let k32 = self.c32(32);
        let full = self.m.op(op::U_GREATER_THAN_EQUAL, b, &[w, k32]);
        let k31 = self.c32(31);
        let wc = self.m.ext(glsl::U_MIN, u, &[w, k31]);
        let top = self.m.op(op::SHIFT_LEFT_LOGICAL, u, &[one, wc]);
        let m = self.m.op(op::I_SUB, u, &[top, one]);
        let ones = self.c32(u32::MAX);
        let mask = self.m.op(op::SELECT, u, &[full, ones, m]);
        let smax = self.m.op(op::SHIFT_RIGHT_LOGICAL, u, &[mask, one]);
        let smin = self.m.op(op::NOT, u, &[smax]);
        let x = self.m.op(op::BITCAST, f, &[raw]);
        // float: 32 bits as they are, 16 a half
        let v2 = self.m.t_vec(f, 2);
        let fz = self.m.c_f32(0);
        let pair = self.m.op(op::COMPOSITE_CONSTRUCT, v2, &[x, fz]);
        let half = self.m.ext(glsl::PACK_HALF_2X16, u, &[pair]);
        let fl = self.m.op(op::SELECT, u, &[full, raw, half]);
        // unorm, snorm: clamped (NaN to the low end), scaled, rounded
        let (f0, f1, fm1) = (self.m.c_f32(0), self.m.c_f32(1.0f32.to_bits()), self.m.c_f32((-1.0f32).to_bits()));
        let cu = self.m.ext(glsl::N_CLAMP, f, &[x, f0, f1]);
        let cs = self.m.ext(glsl::N_CLAMP, f, &[x, fm1, f1]);
        let nan = self.m.op(op::IS_NAN, b, &[x]);
        let cs = self.m.op(op::SELECT, f, &[nan, f0, cs]);
        let mf = self.m.op(op::CONVERT_U_TO_F, f, &[mask]);
        let sf = self.m.op(op::CONVERT_U_TO_F, f, &[smax]);
        let un = self.m.op(op::F_MUL, f, &[cu, mf]);
        let un = self.m.ext(glsl::ROUND_EVEN, f, &[un]);
        let un = self.m.op(op::CONVERT_F_TO_U, u, &[un]);
        let sn = self.m.op(op::F_MUL, f, &[cs, sf]);
        let sn = self.m.ext(glsl::ROUND_EVEN, f, &[sn]);
        let sn = self.m.op(op::CONVERT_F_TO_S, u, &[sn]);
        // integers: clamped to the field's range
        let ui = self.m.ext(glsl::U_MIN, u, &[raw, mask]);
        let si = self.m.ext(glsl::S_MAX, u, &[raw, smin]);
        let si = self.m.ext(glsl::S_MIN, u, &[si, smax]);
        let r = self.m.op(op::SELECT, u, &[sint, si, ui]);
        let r = self.m.op(op::SELECT, u, &[snorm, sn, r]);
        let r = self.m.op(op::SELECT, u, &[unorm, un, r]);
        let r = self.m.op(op::SELECT, u, &[float, fl, r]);
        self.m.op(op::BITWISE_AND, u, &[r, mask])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_shapes() {
        let t = format_table();
        assert_eq!(t[22], entry(1, 32, FLOAT));
        assert_eq!(t[50], entry(4, 10, UNORM));
        assert_eq!(t[61], entry(4, 8, SINT));
        // unknown codes: one 32-bit channel
        assert_eq!(t[0], entry(1, 32, FLOAT));
        assert_eq!(t[127], entry(1, 32, FLOAT));
    }

    /// `(a k | 1)` converted once and scaled is the float nearest `a / d`
    /// for every code of the widths the decoders use.
    #[test]
    fn norm_quotients_round_to_nearest() {
        for (v, sh) in [(8, 40), (16, 48), (10, 40), (2, 40), (7, 42), (15, 45), (9, 45)] {
            let d = (1u64 << v) - 1;
            let k = norm_k(v, sh);
            for a in 1..=d {
                let x = ((a * k) | 1) as f32 * f32::from_bits(scale(sh));
                assert_eq!(x, a as f32 / d as f32, "{a} / {d}");
            }
        }
    }
}
