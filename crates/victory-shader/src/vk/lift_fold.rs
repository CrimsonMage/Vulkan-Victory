//! Constant folding of the common pure forms.

use super::*;

impl Lift<'_> {
    /// Constant folding of the common pure forms.
    pub(super) fn fold(&mut self, n: &Node) -> Option<V> {
        let Op::Spv(o) = n.op else { return None };
        let c: Vec<Option<u64>> = n.args.iter().map(|&a| self.f.as_const(a)).collect();
        let a = c.first().copied().flatten();
        let b = c.get(1).copied().flatten();
        let ty = n.ty;
        let r = match o {
            _ if ty == Ty::U32 && n.args.len() == 2 => self.fold_u32(o, n, a, b),
            op::I_ADD if ty == Ty::U64 && n.args.len() == 2 => self.fold_add64(n, a, b),
            _ if ty == Ty::BOOL => self.fold_bool(o, n, a, b),
            _ => None,
        };
        if r.is_some() {
            return r;
        }
        match o {
            op::U_CONVERT | op::S_CONVERT => a.map(|x| self.fold_convert(o, n, x)),
            op::SELECT if a.is_some() => Some(if a != Some(0) { n.args[1] } else { n.args[2] }),
            op::SELECT if n.args[1] == n.args[2] => Some(n.args[1]),
            op::BITCAST if n.args.len() == 1 => self.fold_bitcast(n),
            _ => None,
        }
    }

    /// 32-bit integer operations of constants, and identities.
    pub(super) fn fold_u32(&mut self, o: u16, n: &Node, a: Option<u64>, b: Option<u64>) -> Option<V> {
        if let (Some(x), Some(y)) = (a, b) {
            let r = fold_u32_op(o, x as u32, y as u32)?;
            return Some(self.f.konst(Ty::U32, u64::from(r)));
        }
        let (l, r) = (n.args[0], n.args[1]);
        let zero = |s: &mut Self| Some(s.f.konst(Ty::U32, 0));
        match (o, a.map(|x| x as u32), b.map(|x| x as u32)) {
            (op::I_ADD | op::BITWISE_OR | op::BITWISE_XOR, Some(0), _) => Some(r),
            (op::I_ADD | op::BITWISE_OR | op::BITWISE_XOR | op::I_SUB, _, Some(0)) => Some(l),
            (op::SHIFT_LEFT_LOGICAL | op::SHIFT_RIGHT_LOGICAL | op::SHIFT_RIGHT_ARITHMETIC, _, Some(0)) => Some(l),
            (op::BITWISE_AND, Some(0), _) | (op::BITWISE_AND, _, Some(0)) => zero(self),
            (op::BITWISE_AND, Some(u32::MAX), _) => Some(r),
            (op::BITWISE_AND, _, Some(u32::MAX)) => Some(l),
            (op::I_MUL, Some(1), _) => Some(r),
            (op::I_MUL, _, Some(1)) => Some(l),
            (op::I_MUL, Some(0), _) | (op::I_MUL, _, Some(0)) => zero(self),
            _ => None,
        }
    }

    pub(super) fn fold_add64(&mut self, n: &Node, a: Option<u64>, b: Option<u64>) -> Option<V> {
        match (a, b) {
            (Some(x), Some(y)) => Some(self.f.konst(Ty::U64, x.wrapping_add(y))),
            (_, Some(0)) => Some(n.args[0]),
            (Some(0), _) => Some(n.args[1]),
            _ => None,
        }
    }

    /// Logic and compares of constants; and/or with a constant or twice
    /// the same value; double negation.
    pub(super) fn fold_bool(&mut self, o: u16, n: &Node, a: Option<u64>, b: Option<u64>) -> Option<V> {
        let same = n.args.len() == 2 && n.args[0] == n.args[1];
        match o {
            op::LOGICAL_AND | op::LOGICAL_OR => {
                // the absorbing constant: 0 for and, 1 for or
                let absorb = u64::from(o == op::LOGICAL_OR);
                match (a, b) {
                    (Some(x), _) | (_, Some(x)) if x == absorb => Some(self.kb(absorb != 0)),
                    (Some(_), _) => Some(n.args[1]),
                    (_, Some(_)) => Some(n.args[0]),
                    _ if same => Some(n.args[0]),
                    _ => None,
                }
            }
            op::LOGICAL_NOT => match (a, &self.f.node(n.args[0]).op) {
                (Some(x), _) => Some(self.kb(x == 0)),
                (_, Op::Spv(op::LOGICAL_NOT)) => Some(self.f.node(n.args[0]).args[0]),
                _ => None,
            },
            op::LOGICAL_NOT_EQUAL => match (a, b) {
                (Some(x), Some(y)) => Some(self.kb(x != y)),
                (Some(0), _) => Some(n.args[1]),
                (_, Some(0)) => Some(n.args[0]),
                _ => None,
            },
            op::I_EQUAL | op::I_NOT_EQUAL | op::U_LESS_THAN | op::U_GREATER_THAN_EQUAL => {
                let (x, y) = (a?, b?);
                let r = match o {
                    op::I_EQUAL => x == y,
                    op::I_NOT_EQUAL => x != y,
                    op::U_LESS_THAN => x < y,
                    _ => x >= y,
                };
                Some(self.kb(r))
            }
            _ => None,
        }
    }

    /// A width conversion of a constant.
    pub(super) fn fold_convert(&mut self, o: u16, n: &Node, x: u64) -> V {
        let bits = |s: S| match s {
            S::U8 => 8,
            S::U16 => 16,
            S::U32 => 32,
            _ => 64,
        };
        let from = bits(self.f.ty(n.args[0]).s);
        let x = if o == op::S_CONVERT && from < 64 && (x >> (from - 1)) & 1 != 0 { x | (!0u64 << from) } else { x };
        let to = bits(n.ty.s);
        let mask = if to == 64 { u64::MAX } else { (1u64 << to) - 1 };
        self.f.konst(n.ty, x & mask)
    }

    /// A bitcast of a value of the same type, of a scalar constant, or of a
    /// bitcast back.
    pub(super) fn fold_bitcast(&mut self, n: &Node) -> Option<V> {
        let ty = n.ty;
        let src = self.f.node(n.args[0]).clone();
        if self.f.ty(n.args[0]) == ty {
            return Some(n.args[0]);
        }
        if let Op::Const(bits) = src.op
            && ty.n == 1
            && self.f.ty(n.args[0]).n == 1
        {
            return Some(self.f.konst(ty, bits));
        }
        if src.op == Op::Spv(op::BITCAST) && self.f.ty(src.args[0]) == ty {
            return Some(src.args[0]);
        }
        None
    }
}
