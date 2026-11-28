//! The buffer views a program reaches title memory through, read from its
//! thread IR: which V#s it loads and stores through and where each V#
//! lies (a descriptor table's address in the push constants plus a
//! constant), and whether it reaches memory in any other way.
//!
//! For the host's store staging (`docs/vulkan-host.md`, "Store staging"):
//! a dispatch's store views get card memory for the dispatch, the memory
//! map pointing there. A V# must lie whole in one range of the map
//! (`docs/vulkan-abi.md` section 4), so the host has to know every view
//! of the program, not only the ones it stores through, and that it has
//! no view it cannot find.
//!
//! Nothing here changes a module: the file is left out of the translation
//! fingerprint (`fingerprint.rs`), and the views are kept beside the kept
//! translations, a line a program ([`Views::line`]), not in them.

use std::collections::HashMap;

use super::spirv::op;
use super::tir::{Func, Helper, Item, Op, S, V};
use crate::ShaderError;

/// A place in title memory: `offset` bytes past the address the push
/// constants hold at dword `push` (a descriptor table's, 32 bits) or at
/// dwords `push` and `push + 1` (`wide`: a root address), plus the push
/// dword `plus` if there is one (a table's place in its heap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Place {
    pub push: u32,
    pub wide: bool,
    pub plus: Option<u32>,
    pub offset: u64,
}

impl Place {
    /// The title address, with the push constants `push` of a dispatch.
    pub fn address(&self, push: &[u32]) -> u64 {
        let dword = |k: u32| u64::from(push.get(k as usize).copied().unwrap_or(0));
        let high = if self.wide { dword(self.push + 1) << 32 } else { 0 };
        (dword(self.push) | high).wrapping_add(self.plus.map_or(0, dword)).wrapping_add(self.offset)
    }
}

/// A buffer view: the four dwords of its V# lie at `at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    pub at: Place,
    pub loads: bool,
    pub stores: bool,
    /// A store through it lands where data says: its offset comes from
    /// what an invocation read (memory at a place of its own, an image),
    /// not from the invocation's place in the dispatch or its turn at a
    /// shared counter alone. Such stores are scattered over the view; the
    /// others fill it in the dispatch's order or append to it.
    pub scattered: bool,
}

/// How a program reaches title memory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Views {
    /// The views whose V# lies at a constant place, by place.
    pub views: Vec<View>,
    /// It stores through something else: a view whose V# is found at run
    /// time (an index from memory, a V# inside a buffer), an address it
    /// computes.
    pub other_stores: bool,
    /// It loads through a view whose V# is found at run time.
    pub other_views: bool,
    /// It loads at addresses it computes: pointers from memory, a
    /// ray-tracing structure's nodes. (Each such access is translated on
    /// its own, so no range of the map has to hold more than the access;
    /// loads at constant places, the V#s themselves among them, are the
    /// same and not noted.)
    pub computed_loads: bool,
}

impl Views {
    /// Whether every store goes through a view of [`Views::views`] and
    /// every view is one of them: what the host needs to give the store
    /// views another place.
    pub fn resolved(&self) -> bool {
        !self.other_stores && !self.other_views
    }

    pub fn stores(&self) -> impl Iterator<Item = &View> {
        self.views.iter().filter(|v| v.stores)
    }

    /// The views as a line of text: the flags (`s` other stores, `v`
    /// other views, `c` computed loads, `-` none), then a word a view:
    /// `L`, `S` or `B` (loads, stores, both), `!` if its stores are
    /// scattered, the push dword, `w` if wide, `+` and the added push
    /// dword if any, `@` and the offset in hexadecimal.
    pub fn line(&self) -> String {
        let flags: String = [(self.other_stores, 's'), (self.other_views, 'v'), (self.computed_loads, 'c')].iter().filter(|f| f.0).map(|f| f.1).collect();
        let mut out = if flags.is_empty() { "-".to_owned() } else { flags };
        for v in &self.views {
            let kind = if v.loads && v.stores { 'B' } else if v.stores { 'S' } else { 'L' };
            let plus = v.at.plus.map_or(String::new(), |k| format!("+{k}"));
            let scattered = if v.scattered { "!" } else { "" };
            out.push_str(&format!(" {kind}{scattered}{}{}{plus}@{:x}", v.at.push, if v.at.wide { "w" } else { "" }, v.at.offset));
        }
        out
    }

    /// The views of a [`Views::line`].
    pub fn parse(line: &str) -> Option<Views> {
        let mut words = line.split_whitespace();
        let flags = words.next()?;
        if !flags.chars().all(|c| "svc-".contains(c)) {
            return None;
        }
        let mut out = Views { other_stores: flags.contains('s'), other_views: flags.contains('v'), computed_loads: flags.contains('c'), ..Views::default() };
        for word in words {
            let (kind, rest) = word.split_at_checked(1)?;
            let scattered = rest.starts_with('!');
            let (place, offset) = rest.trim_start_matches('!').split_once('@')?;
            let (push, plus) = match place.split_once('+') {
                Some((push, plus)) => (push, Some(plus.parse().ok()?)),
                None => (place, None),
            };
            let wide = push.ends_with('w');
            let at = Place { push: push.trim_end_matches('w').parse().ok()?, wide, plus, offset: u64::from_str_radix(offset, 16).ok()? };
            let (loads, stores) = match kind {
                "L" => (true, false),
                "S" => (false, true),
                "B" => (true, true),
                _ => return None,
            };
            out.views.push(View { at, loads, stores, scattered });
        }
        Some(out)
    }
}

/// What a value is, as far as the views need it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sym {
    Const(u64),
    /// Push-constant dword k.
    Push(u32),
    /// A title address.
    Addr(Place),
    /// The dword of title memory at a place.
    Word(Place),
    /// The base address of the V# at a place (its dwords 0 and 1).
    Base(Place),
    Unknown,
}

/// How a value differs between the invocations of a dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Varies {
    No,
    /// By the invocation's place alone (its ids).
    ByPlace,
    /// By its turn at something the invocations share: a counter all of
    /// them count up (the next free record of a list), the work group's
    /// memory. What is stored by it lies in the order of arrival.
    ByTurn,
    /// By what it read where others read something else.
    ByData,
}

struct Walk<'a> {
    f: &'a Func,
    syms: HashMap<V, Sym>,
    varies: HashMap<V, Varies>,
    out: Views,
}

impl Walk<'_> {
    fn varies(&mut self, v: V) -> Varies {
        if let Some(&known) = self.varies.get(&v) {
            return known;
        }
        // a value met again while it is worked out (a loop's phi) adds nothing
        self.varies.insert(v, Varies::No);
        let n = self.f.node(v).clone();
        let of_args = |w: &mut Self| n.args.iter().map(|&a| w.varies(a)).max().unwrap_or(Varies::No);
        let known = match n.op {
            Op::Builtin(_) | Op::Input { .. } | Op::PerVertex { .. } => Varies::ByPlace,
            // memory every invocation reads at the same place is the same for all
            Op::Load { .. } | Op::Call(Helper::Typed | Helper::BvhRay) => match of_args(self) {
                Varies::No => Varies::No,
                _ => Varies::ByData,
            },
            // a counter found by data is data; one all count up is a turn
            Op::Atomic(_) => match self.varies(n.args[0]) {
                Varies::ByData => Varies::ByData,
                _ => Varies::ByTurn,
            },
            Op::LdsLoad | Op::LdsAtomic(_) | Op::Xchg => Varies::ByTurn,
            Op::Image { .. } => Varies::ByData,
            _ => of_args(self),
        };
        self.varies.insert(v, known);
        known
    }

    fn sym(&mut self, v: V) -> Sym {
        if let Some(&s) = self.syms.get(&v) {
            return s;
        }
        // a value met again while it is worked out (a loop's phi) is unknown
        self.syms.insert(v, Sym::Unknown);
        let s = self.sym_of(v);
        self.syms.insert(v, s);
        s
    }

    /// `v` without a mask of its low 16 bits (a V#'s BASE_ADDRESS_HI).
    fn unmasked(&self, v: V) -> V {
        let n = self.f.node(v);
        match n.op {
            Op::Spv(op::BITWISE_AND) if self.f.as_const(n.args[1]) == Some(0xFFFF) => n.args[0],
            _ => v,
        }
    }

    fn sym_of(&mut self, v: V) -> Sym {
        let n = self.f.node(v).clone();
        let wide = n.ty.s == S::U64;
        match n.op {
            Op::Const(c) => Sym::Const(c),
            Op::Push(k) => Sym::Push(k),
            Op::Spv(op::U_CONVERT) if wide => match self.sym(n.args[0]) {
                Sym::Push(push) => Sym::Addr(Place { push, wide: false, plus: None, offset: 0 }),
                Sym::Const(c) => Sym::Const(c),
                _ => Sym::Unknown,
            },
            Op::Spv(op::BITCAST) if wide => self.pair(n.args[0]),
            // a dword's bits read as another type
            Op::Spv(op::BITCAST) if n.ty.n == 1 => match self.sym(n.args[0]) {
                s @ Sym::Word(_) => s,
                _ => Sym::Unknown,
            },
            Op::Spv(op::I_ADD) => {
                let (x, y) = (self.sym(n.args[0]), self.sym(n.args[1]));
                sum(x, y, wide)
            }
            Op::Load { .. } if n.ty.n == 1 => self.loaded(n.args[0], 0),
            Op::Spv(op::COMPOSITE_EXTRACT) if matches!(self.f.node(n.args[0]).op, Op::Load { .. }) => {
                let load = self.f.node(n.args[0]).args[0];
                self.loaded(load, u64::from(n.imm[0]))
            }
            // another lane's value of something the whole wave has alike
            Op::Wave(op::GROUP_NON_UNIFORM_BROADCAST | op::GROUP_NON_UNIFORM_BROADCAST_FIRST | op::GROUP_NON_UNIFORM_SHUFFLE, _) => self.sym(n.args[0]),
            Op::Phi | Op::Spv(op::SELECT) => {
                let values = if n.op == Op::Phi { &n.args[..] } else { &n.args[1..] };
                let first = values.first().map_or(Sym::Unknown, |&a| self.sym(a));
                if values.iter().all(|&a| self.sym(a) == first) { first } else { Sym::Unknown }
            }
            _ => Sym::Unknown,
        }
    }

    /// A 64-bit value made of the two dwords `pair`: a root address (two
    /// push dwords), or a V#'s base (its dwords 0 and 1, the high one
    /// masked).
    fn pair(&mut self, pair: V) -> Sym {
        let pair = self.f.node(pair).clone();
        if pair.op != Op::Spv(op::COMPOSITE_CONSTRUCT) || pair.args.len() != 2 {
            return Sym::Unknown;
        }
        let hi = self.unmasked(pair.args[1]);
        let masked = hi != pair.args[1];
        match (self.sym(pair.args[0]), self.sym(hi)) {
            (Sym::Push(lo), Sym::Push(hi)) if hi == lo + 1 => Sym::Addr(Place { push: lo, wide: true, plus: None, offset: 0 }),
            (Sym::Word(lo), Sym::Word(hi)) if masked && hi == (Place { offset: lo.offset + 4, ..lo }) => Sym::Base(lo),
            _ => Sym::Unknown,
        }
    }

    /// Dword `k` of a load at the device address `dev`: a dword of title
    /// memory when the address is a constant place's translation.
    fn loaded(&mut self, dev: V, k: u64) -> Sym {
        let d = self.f.node(dev).clone();
        if d.op != Op::Call(Helper::Translate) {
            return Sym::Unknown;
        }
        match self.sym(d.args[0]) {
            Sym::Addr(p) => Sym::Word(Place { offset: p.offset + 4 * k, ..p }),
            _ => Sym::Unknown,
        }
    }

    /// Whether `bytes` (a V#'s byte count as `lift_mem.rs` `vsharp_of`
    /// makes it) is made of the dwords 1 and 2 of the V# at `at` and of
    /// no other memory.
    fn bytes_of(&mut self, bytes: V, at: Place) -> bool {
        let mut stack = vec![(bytes, 0)];
        let (mut count, mut stride) = (false, false);
        while let Some((v, depth)) = stack.pop() {
            match self.sym(v) {
                Sym::Word(p) if p == (Place { offset: at.offset + 8, ..at }) => count = true,
                Sym::Word(p) if p == (Place { offset: at.offset + 4, ..at }) => stride = true,
                Sym::Const(_) => {}
                Sym::Unknown if depth < 12 && !self.f.node(v).args.is_empty() && !matches!(self.f.node(v).op, Op::Load { .. } | Op::Call(_)) => {
                    stack.extend(self.f.node(v).args.iter().map(|&a| (a, depth + 1)));
                }
                _ => return false,
            }
        }
        count && stride
    }

    /// The translations a device address comes from (the map's zero and
    /// sink blocks left out); `None` among them: something else.
    fn roots(&self, dev: V, out: &mut Vec<Option<V>>, seen: &mut Vec<V>) {
        if seen.contains(&dev) {
            return;
        }
        seen.push(dev);
        let n = self.f.node(dev);
        match n.op {
            Op::Call(Helper::Translate | Helper::TranslateRange) => out.push(Some(dev)),
            Op::MapField(_) | Op::Spv(op::U_CONVERT) | Op::Const(_) => {}
            Op::Spv(op::I_ADD | op::SELECT) | Op::Phi | Op::Pick => {
                for &a in &n.args {
                    if self.f.ty(a).s == S::U64 {
                        self.roots(a, out, seen);
                    }
                }
            }
            _ => out.push(None),
        }
    }

    fn view(&mut self, at: Place) -> &mut View {
        let k = match self.out.views.iter().position(|v| v.at == at) {
            Some(k) => k,
            None => {
                self.out.views.push(View { at, loads: false, stores: false, scattered: false });
                self.out.views.len() - 1
            }
        };
        &mut self.out.views[k]
    }

    /// A load or store at the device address `dev`.
    fn access(&mut self, dev: V, store: bool) {
        let mut roots = Vec::new();
        self.roots(dev, &mut roots, &mut Vec::new());
        for root in roots {
            let place = root.map(|r| {
                let call = self.f.node(r).clone();
                (call.op == Op::Call(Helper::TranslateRange), self.sym(call.args[0]), call.args.get(1).copied())
            });
            match place {
                Some((true, Sym::Base(at), Some(count))) if self.bytes_of(count, at) => {
                    let scattered = store && self.varies(dev) == Varies::ByData;
                    let v = self.view(at);
                    v.loads |= !store;
                    v.stores |= store;
                    v.scattered |= scattered;
                }
                Some((false, Sym::Addr(_), _)) if !store => {}
                _ if store => self.out.other_stores = true,
                Some((true, ..)) => self.out.other_views = true,
                _ => self.out.computed_loads = true,
            }
        }
    }

    fn items(&mut self, list: &[Item]) {
        for it in list {
            match it {
                Item::N(v) => {
                    let n = self.f.node(*v);
                    match n.op {
                        // a typed element and a ray-tracing node are loads too
                        Op::Load { .. } | Op::Call(Helper::Typed | Helper::BvhRay) => self.access(n.args[0], false),
                        Op::Store { .. } | Op::Atomic(_) => self.access(n.args[0], true),
                        _ => {}
                    }
                }
                Item::Guard { body, .. } => self.items(body),
            }
        }
    }
}

/// The sum of two values (`wide`: 64 bits): an address moved by a
/// constant, a table's place in its heap (both from the push constants),
/// a constant.
fn sum(x: Sym, y: Sym, wide: bool) -> Sym {
    match (x, y) {
        (Sym::Addr(p), Sym::Const(c)) | (Sym::Const(c), Sym::Addr(p)) if wide => Sym::Addr(Place { offset: p.offset.wrapping_add(c), ..p }),
        (Sym::Addr(p), Sym::Addr(q)) if wide && p.plus.is_none() && q.plus.is_none() && !(p.wide && q.wide) => {
            let (base, index) = if q.wide || (!p.wide && q.push < p.push) { (q, p) } else { (p, q) };
            Sym::Addr(Place { plus: Some(index.push), offset: base.offset.wrapping_add(index.offset), ..base })
        }
        (Sym::Const(a), Sym::Const(b)) => Sym::Const(if wide { a.wrapping_add(b) } else { u64::from((a as u32).wrapping_add(b as u32)) }),
        _ => Sym::Unknown,
    }
}

/// The views of the thread program `f`.
pub fn views(f: &Func) -> Views {
    let mut w = Walk { f, syms: HashMap::new(), varies: HashMap::new(), out: Views::default() };
    for b in &f.blocks {
        w.items(&b.body);
    }
    w.out.views.sort_by_key(|v| v.at);
    w.out
}

/// The views of a container's program, as its translation reaches memory.
pub fn of(container: &[u8]) -> Result<Views, ShaderError> {
    let (f, _, _) = super::thread_ir(container)?;
    Ok(views(&f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::tir::{Block, Node, Term, Ty};

    struct Build(Func, Vec<Item>);

    impl Build {
        fn node(&mut self, op: Op, ty: Ty, args: &[V], imm: &[u32]) -> V {
            let v = self.0.push(Node { op, ty, args: args.to_vec(), imm: imm.to_vec() });
            self.1.push(Item::N(v));
            v
        }

        fn spv(&mut self, o: u16, ty: Ty, args: &[V]) -> V {
            self.node(Op::Spv(o), ty, args, &[])
        }

        /// The device base and byte count of the V# at `offset` of the
        /// table whose address push dword 5 holds, as `vsharp_of` makes
        /// them.
        fn vsharp(&mut self, offset: u64) -> V {
            let table = self.0.global(Op::Push(5), Ty::U32);
            let table = self.spv(op::U_CONVERT, Ty::U64, &[table]);
            let k = self.0.konst(Ty::U64, offset);
            let at = self.spv(op::I_ADD, Ty::U64, &[table, k]);
            let dev = self.node(Op::Call(Helper::Translate), Ty::U64, &[at], &[]);
            let words = self.node(Op::Load { align: 4 }, Ty::vec(S::U32, 4), &[dev], &[]);
            let w: Vec<V> = (0..3).map(|i| self.node(Op::Spv(op::COMPOSITE_EXTRACT), Ty::U32, &[words], &[i])).collect();
            let mask = self.0.konst(Ty::U32, 0xFFFF);
            let hi = self.spv(op::BITWISE_AND, Ty::U32, &[w[1], mask]);
            let pair = self.spv(op::COMPOSITE_CONSTRUCT, Ty::vec(S::U32, 2), &[w[0], hi]);
            let base = self.spv(op::BITCAST, Ty::U64, &[pair]);
            let k16 = self.0.konst(Ty::U32, 16);
            let stride = self.spv(op::SHIFT_RIGHT_LOGICAL, Ty::U32, &[w[1], k16]);
            let bytes = self.spv(op::I_MUL, Ty::U32, &[w[2], stride]);
            self.node(Op::Call(Helper::TranslateRange), Ty::U64, &[base, bytes], &[])
        }

        fn done(mut self) -> Views {
            self.0.blocks.push(Block { phis: Vec::new(), preds: Vec::new(), body: std::mem::take(&mut self.1), term: Term::Return });
            views(&self.0)
        }
    }

    #[test]
    fn a_table_view_stored_through_and_one_read() {
        let mut b = Build(Func::default(), Vec::new());
        let read = b.vsharp(0x20);
        let x = b.node(Op::Load { align: 4 }, Ty::U32, &[read], &[]);
        let written = b.vsharp(0x40);
        let sink = b.0.global(Op::MapField(1), Ty::U64);
        let ok = b.0.konst(Ty::BOOL, 1);
        let a = b.spv(op::SELECT, Ty::U64, &[ok, written, sink]);
        b.node(Op::Store { align: 4 }, Ty::VOID, &[a, x], &[]);
        let v = b.done();
        let place = |offset| Place { push: 5, wide: false, plus: None, offset };
        assert_eq!(v.views, [View { at: place(0x20), loads: true, stores: false, scattered: false }, View { at: place(0x40), loads: false, stores: true, scattered: false }]);
        assert!(v.resolved() && !v.computed_loads);
        assert_eq!(v.line(), "- L5@20 S5@40");
    }

    #[test]
    fn a_line_and_back() {
        let at = Place { push: 14, wide: true, plus: Some(9), offset: 0x1c0 };
        let low = Place { push: 8, wide: false, plus: None, offset: 0 };
        let v = Views { views: vec![View { at, loads: true, stores: true, scattered: true }, View { at: low, loads: true, stores: false, scattered: false }], other_stores: false, other_views: true, computed_loads: true };
        assert_eq!(v.line(), "vc B!14w+9@1c0 L8@0");
        assert_eq!(Views::parse(&v.line()), Some(v));
        assert_eq!(Views::parse("-"), Some(Views::default()));
        assert_eq!(Views::parse("x S1@0"), None);
        assert_eq!(Views::parse("- S1"), None);
    }

    #[test]
    fn a_store_where_a_load_of_the_invocations_own_says_is_scattered() {
        let mut b = Build(Func::default(), Vec::new());
        let list = b.vsharp(0x20);
        let thread = b.0.global(Op::Builtin(crate::vk::tir::Builtin::LocalInvocationIndex), Ty::U32);
        let thread = b.spv(op::U_CONVERT, Ty::U64, &[thread]);
        // by the invocation's place: the view is filled in order
        let ordered = b.vsharp(0x40);
        let a = b.spv(op::I_ADD, Ty::U64, &[ordered, thread]);
        let x = b.0.konst(Ty::U32, 7);
        b.node(Op::Store { align: 4 }, Ty::VOID, &[a, x], &[]);
        // by an index each invocation reads from its own place
        let own = b.spv(op::I_ADD, Ty::U64, &[list, thread]);
        let index = b.node(Op::Load { align: 4 }, Ty::U32, &[own], &[]);
        let index = b.spv(op::U_CONVERT, Ty::U64, &[index]);
        let picked = b.vsharp(0x60);
        let a = b.spv(op::I_ADD, Ty::U64, &[picked, index]);
        b.node(Op::Store { align: 4 }, Ty::VOID, &[a, x], &[]);
        assert_eq!(b.done().line(), "- L5@20 S5@40 S!5@60");
    }

    #[test]
    fn a_store_at_a_computed_address_is_not_resolved() {
        let mut b = Build(Func::default(), Vec::new());
        let view = b.vsharp(0);
        let pointer = b.node(Op::Load { align: 8 }, Ty::U64, &[view], &[]);
        let dev = b.node(Op::Call(Helper::Translate), Ty::U64, &[pointer], &[]);
        let x = b.0.konst(Ty::U32, 7);
        b.node(Op::Store { align: 4 }, Ty::VOID, &[dev, x], &[]);
        let v = b.done();
        assert!(v.other_stores && !v.resolved());
        assert_eq!(v.views.len(), 1);
    }
}
