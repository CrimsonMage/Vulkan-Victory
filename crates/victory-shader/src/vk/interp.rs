//! A CPU interpreter of the thread IR ([`super::tir`]): one console wave
//! (one or two host subgroups of 32 invocations) stepped in lockstep over a
//! memory image, for checking a translation against the console code with
//! real data (offline; nothing in the translator uses it).
//!
//! It follows what the backend writes: every block is uniform (a branch
//! whose condition differs between invocations is reported), a guard's body
//! runs for the invocations whose condition holds (a guard the backend
//! writes flat runs for all of them, its stores predicated), subgroup
//! operations see the invocations the SPIR-V would have active. Title
//! addresses are device addresses (the memory map is the identity over the
//! mapped range); reads of bytes the image does not hold read zeros and are
//! listed, so a run says which memory to dump next.

use std::collections::{BTreeMap, HashMap};

use super::spirv::{glsl, op};
use super::tir::{Builtin, Func, Helper, ImageKind, Item, Node, Op, S, Term, Ty, V};

/// A value of one invocation: up to four components, as bits.
type W = [u64; 4];

/// The zero and sink blocks of the memory map.
pub const ZERO: u64 = 0xF000_0000_0000;
pub const SINK: u64 = 0xF100_0000_0000;

/// Title memory: dwords by address; reads of missing dwords give 0 and are
/// counted.
#[derive(Debug, Default, Clone)]
pub struct Memory {
    pub dwords: HashMap<u64, u32>,
    /// The mapped range (title start, bytes): addresses outside read the
    /// zero block.
    pub range: (u64, u64),
    /// Missing dwords read: address -> (reads, first reading node).
    pub missing: BTreeMap<u64, (u32, V)>,
    /// Stores: address -> value (also applied to `dwords`).
    pub stored: BTreeMap<u64, u32>,
    /// What a missing dword reads.
    pub fill: u32,
}

impl Memory {
    /// Puts `words` at `addr`.
    pub fn put(&mut self, addr: u64, words: &[u32]) {
        for (i, &w) in words.iter().enumerate() {
            self.dwords.insert(addr + 4 * i as u64, w);
        }
    }

    fn byte(&mut self, a: u64, quiet: bool, who: V) -> u8 {
        if a >= ZERO {
            return 0;
        }
        let d = a & !3;
        match self.dwords.get(&d) {
            Some(&w) => (w >> (8 * (a & 3))) as u8,
            None => {
                if !quiet {
                    self.missing.entry(d).or_insert((0, who)).0 += 1;
                }
                (self.fill >> (8 * (a & 3))) as u8
            }
        }
    }

    fn read(&mut self, a: u64, bytes: u32, quiet: bool, who: V) -> u64 {
        (0..u64::from(bytes)).fold(0, |acc, i| acc | u64::from(self.byte(a + i, quiet, who)) << (8 * i))
    }

    fn write(&mut self, a: u64, bytes: u32, x: u64) {
        if a >= SINK {
            return;
        }
        for i in 0..u64::from(bytes) {
            let b = a + i;
            let d = b & !3;
            let sh = 8 * (b & 3);
            let w = self.dwords.get(&d).copied().unwrap_or(0);
            let w = (w & !(0xFF << sh)) | (((x >> (8 * i)) as u32 & 0xFF) << sh);
            self.dwords.insert(d, w);
            self.stored.insert(d, w);
        }
    }

    fn translate(&self, a: u64, bytes: u64) -> u64 {
        let d = a.wrapping_sub(self.range.0);
        if d < self.range.1 && self.range.1 - d >= bytes { a } else { ZERO }
    }
}

/// Why a run stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    Returned,
    /// Block visits ran past the limit (a loop that does not leave).
    Limit,
    /// A branch condition differed between invocations, or an operation
    /// the interpreter lacks.
    Error(String),
}

/// One console wave of a thread program.
pub struct Machine<'a> {
    f: &'a Func,
    lanes: usize,
    vals: Vec<Vec<W>>,
    pub mem: Memory,
    /// Push-constant dwords (user data k at 4 + k).
    pub push: Vec<u32>,
    pub group: [u32; 3],
    /// The console wave's index in its group (host subgroups 2w, 2w+1 for
    /// wave64).
    pub wave: u32,
    pub lds: Vec<u32>,
    /// Block visits after which the run stops.
    pub limit: u64,
    pub visits: Vec<u64>,
    /// Blocks in the order entered (the last `trail_len`).
    pub trail: std::collections::VecDeque<usize>,
    pub trail_len: usize,
    /// Notes (undefined shifts and the like), deduplicated by node.
    pub notes: BTreeMap<V, String>,
    /// Values to print each time they are computed.
    pub watch: Vec<V>,
    pub log: Vec<String>,
    /// What every image reads: its size, levels and texels (no images in
    /// the image).
    pub image_size: W,
    pub image_levels: u32,
    pub image_texel: W,
    /// What an undefined value reads (a poison to find uses of them).
    pub undef: u64,
}

fn width(s: S) -> u32 {
    match s {
        S::Bool => 1,
        S::U8 => 8,
        S::U16 | S::F16 => 16,
        S::U32 | S::F32 => 32,
        S::U64 | S::F64 => 64,
        S::Void => 32,
    }
}

fn mask(s: S) -> u64 {
    match width(s) {
        64 => u64::MAX,
        w => (1u64 << w) - 1,
    }
}

fn sext(x: u64, w: u32) -> i64 {
    if w >= 64 { x as i64 } else { ((x << (64 - w)) as i64) >> (64 - w) }
}

/// f16 bits to f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let s = u32::from(h >> 15) << 31;
    let e = u32::from((h >> 10) & 0x1F);
    let m = u32::from(h & 0x3FF);
    let bits = match (e, m) {
        (0, 0) => s,
        (0, _) => {
            // subnormal
            let f = m as f32 / 1024.0 * 2f32.powi(-14);
            return if s != 0 { -f } else { f };
        }
        (31, 0) => s | 0x7F80_0000,
        (31, _) => s | 0x7FC0_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to f16 bits (round to nearest even).
pub fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x7F_FFFF;
    if e == 255 {
        return s | 0x7C00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7C00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let r = if rem > half || (rem == half && r & 1 != 0) { r + 1 } else { r };
        return s | r as u16;
    }
    let r = (e2 as u32) << 10 | (m >> 13);
    let rem = m & 0x1FFF;
    let r = if rem > 0x1000 || (rem == 0x1000 && r & 1 != 0) { r + 1 } else { r };
    s | r as u16
}

fn to_f(s: S, x: u64) -> f64 {
    match s {
        S::F16 => f64::from(f16_to_f32(x as u16)),
        S::F32 => f64::from(f32::from_bits(x as u32)),
        S::F64 => f64::from_bits(x),
        _ => x as f64,
    }
}

fn from_f(s: S, x: f64) -> u64 {
    match s {
        S::F16 => u64::from(f32_to_f16(x as f32)),
        S::F64 => x.to_bits(),
        _ => u64::from((x as f32).to_bits()),
    }
}

/// Packs components of `width` bits into one integer and back (bitcasts).
fn repack(src: &W, sw: u32, sn: u8, dw: u32, dn: u8) -> W {
    let mut all: u128 = 0;
    for i in (0..usize::from(sn)).rev() {
        all = (all << sw) | u128::from(src[i] & if sw == 64 { u64::MAX } else { (1 << sw) - 1 });
    }
    let mut out = [0; 4];
    for o in out.iter_mut().take(usize::from(dn)) {
        *o = (all & if dw == 64 { u128::from(u64::MAX) } else { (1 << dw) - 1 }) as u64;
        all >>= dw;
    }
    out
}

impl<'a> Machine<'a> {
    /// A machine for one console wave of `lanes` (32 or 64) invocations.
    pub fn new(f: &'a Func, lanes: usize, mem: Memory) -> Self {
        Machine {
            f,
            lanes,
            vals: vec![Vec::new(); f.nodes.len()],
            mem,
            push: vec![0; 64],
            group: [0; 3],
            wave: 0,
            lds: vec![0; 16384],
            limit: 1 << 22,
            visits: vec![0; f.blocks.len()],
            trail: Default::default(),
            trail_len: 64,
            notes: BTreeMap::new(),
            watch: Vec::new(),
            log: Vec::new(),
            image_size: [1, 1, 1, 1],
            image_levels: 1,
            image_texel: [0; 4],
            undef: 0,
        }
    }

    /// The value of `v` in invocation `l`.
    pub fn value(&self, v: V, l: usize) -> W {
        match self.f.node(v).op {
            Op::Const(c) => [c, 0, 0, 0],
            Op::Undef => [self.undef; 4],
            _ => self.vals[v as usize].get(l).copied().unwrap_or([0; 4]),
        }
    }

    fn note(&mut self, v: V, what: impl FnOnce() -> String) {
        self.notes.entry(v).or_insert_with(what);
    }

    /// Runs the wave from the entry block.
    pub fn run(&mut self) -> Stop {
        let all = vec![true; self.lanes];
        for &v in &self.f.prelude.clone() {
            if let Err(e) = self.eval(v, &all, &all) {
                return Stop::Error(e);
            }
        }
        let (mut b, mut prev) = (0usize, usize::MAX);
        let mut total = 0u64;
        loop {
            self.visits[b] += 1;
            total += 1;
            self.trail.push_back(b);
            if self.trail.len() > self.trail_len {
                self.trail.pop_front();
            }
            if total > self.limit {
                return Stop::Limit;
            }
            match self.enter(b, prev).and_then(|()| self.block(b, &all)) {
                Ok(Some(t)) => (prev, b) = (b, t),
                Ok(None) => return Stop::Returned,
                Err(e) => return Stop::Error(format!("block {b}: {e}")),
            }
        }
    }

    /// The phis of block `b` entered from `prev` (all at once).
    fn enter(&mut self, b: usize, prev: usize) -> Result<(), String> {
        let blk = &self.f.blocks[b];
        if prev == usize::MAX {
            return Ok(());
        }
        let k = blk.preds.iter().position(|&p| p == prev).ok_or_else(|| format!("entered from {prev}, not a predecessor"))?;
        let new: Vec<(V, Vec<W>)> = blk
            .phis
            .iter()
            .map(|&p| {
                let a = self.f.node(p).args[k];
                (p, (0..self.lanes).map(|l| self.value(a, l)).collect())
            })
            .collect();
        for (p, x) in new {
            self.vals[p as usize] = x;
            self.watched(p);
        }
        Ok(())
    }

    /// Runs block `b`'s body; the block it goes to (none: it returned).
    fn block(&mut self, b: usize, all: &[bool]) -> Result<Option<usize>, String> {
        let blk = &self.f.blocks[b];
        self.items(&blk.body, all, all)?;
        match blk.term {
            Term::Return => Ok(None),
            Term::Jump(t) => Ok(Some(t)),
            Term::Branch(c, t, e) => {
                let x0 = self.value(c, 0)[0];
                if (1..self.lanes).any(|l| self.value(c, l)[0] != x0) {
                    return Err(format!("the branch on %{c} differs between invocations"));
                }
                Ok(Some(if x0 != 0 { t } else { e }))
            }
        }
    }

    fn watched(&mut self, v: V) {
        if self.watch.contains(&v) {
            let x: Vec<String> = (0..self.lanes.min(4)).map(|l| format!("{:x?}", &self.value(v, l)[..usize::from(self.f.ty(v).n)])).collect();
            self.log.push(format!("%{v} = {}", x.join(" ")));
        }
    }

    fn items(&mut self, list: &[Item], pred: &[bool], spv: &[bool]) -> Result<(), String> {
        for it in list {
            match it {
                Item::N(v) => self.eval(*v, pred, spv)?,
                Item::Guard { cond, body, phis } => {
                    let c: Vec<bool> = (0..self.lanes).map(|l| self.value(*cond, l)[0] != 0).collect();
                    let flat = self.predicable(body);
                    let ip: Vec<bool> = (0..self.lanes).map(|l| pred[l] && c[l]).collect();
                    let is: Vec<bool> = if flat { spv.to_vec() } else { (0..self.lanes).map(|l| spv[l] && c[l]).collect() };
                    self.items(body, &ip, &is)?;
                    for &p in phis {
                        let n = self.f.node(p);
                        let (i, o) = (n.args[0], n.args[1]);
                        let x = (0..self.lanes).map(|l| if c[l] { self.value(i, l) } else { self.value(o, l) }).collect();
                        self.vals[p as usize] = x;
                        self.watched(p);
                    }
                }
            }
        }
        Ok(())
    }

    fn predicable(&self, body: &[Item]) -> bool {
        body.iter().all(|it| {
            matches!(it, Item::N(v) if {
                let op = &self.f.node(*v).op;
                (op.removable() && *op != Op::Phi)
                    || matches!(op, Op::Store { .. } | Op::Atomic(_) | Op::LdsStore | Op::LdsAtomic(_))
            })
        })
    }

    /// The subgroup (32 invocations) of invocation `l`.
    fn sub(l: usize) -> std::ops::Range<usize> {
        let s = l / 32 * 32;
        s..s + 32
    }

    fn eval(&mut self, v: V, pred: &[bool], spv: &[bool]) -> Result<(), String> {
        let n = self.f.node(v).clone();
        let mut out = vec![[0u64; 4]; self.lanes];
        for l in 0..self.lanes {
            if !spv[l] {
                continue;
            }
            out[l] = self.one(v, &n, l, pred, spv)?;
        }
        self.vals[v as usize] = out;
        self.watched(v);
        Ok(())
    }

    fn arg(&self, n: &Node, k: usize, l: usize) -> W {
        self.value(n.args[k], l)
    }

    fn one(&mut self, v: V, n: &Node, l: usize, pred: &[bool], spv: &[bool]) -> Result<W, String> {
        let s = |x: u64| [x, 0, 0, 0];
        Ok(match &n.op {
            Op::Const(c) => s(*c),
            Op::Undef => [self.undef; 4],
            Op::Push(k) => s(u64::from(self.push.get(*k as usize).copied().unwrap_or(0))),
            Op::MapField(k) => s(if *k == 0 { ZERO } else { SINK }),
            Op::Builtin(b) => s(u64::from(self.builtin(*b, l)?)),
            Op::Spv(o) => self.spv_op(v, *o, n, l)?,
            Op::Glsl(g) => self.glsl_op(*g, n, l)?,
            Op::Call(h) => self.call(*h, v, n, l, pred[l])?,
            Op::Load { .. } | Op::Store { .. } | Op::Atomic(_) => self.mem_op(v, n, l, pred[l])?,
            Op::LdsLoad | Op::LdsStore | Op::LdsAtomic(_) => self.lds_op(n, l, pred[l])?,
            Op::Wave(o, _) => self.wave_op(*o, n, l, spv)?,
            Op::Xchg if self.lanes == 64 => {
                // the word is uniform in each half: the first active invocation's
                let first = |h: usize| (h * 32..h * 32 + 32).find(|&i| spv[i]).map_or(0, |i| self.arg(n, 0, i)[0]);
                [first(0), first(1), 0, 0]
            }
            Op::Xchg => return Err("an exchange in a wave32 run".into()),
            Op::Barrier | Op::Output { .. } | Op::Demote => [0; 4],
            Op::Pick => {
                let k = (self.arg(n, 0, l)[0] as usize).min(n.args.len() - 2);
                self.arg(n, 1 + k, l)
            }
            Op::Phi => return Err("a phi outside its block".into()),
            Op::Image { kind, .. } => match kind {
                ImageKind::Size => self.image_size,
                ImageKind::Levels => s(u64::from(self.image_levels)),
                ImageKind::Store { .. } => [0; 4],
                _ => self.image_texel,
            },
            Op::Input { loc, comp, .. } | Op::PerVertex { loc, comp, .. } => s(input(*loc, *comp, l, n.ty)),
        })
    }

    fn builtin(&self, b: Builtin, l: usize) -> Result<u32, String> {
        let sg = self.wave * (self.lanes as u32 / 32) + (l / 32) as u32;
        let lane = (l % 32) as u32;
        Ok(match b {
            Builtin::WorkgroupId(c) => self.group[usize::from(c)],
            Builtin::SubgroupId => sg,
            Builtin::SubgroupLocalId => lane,
            Builtin::LocalInvocationIndex | Builtin::LocalInvocationId(0) => sg * 32 + lane,
            Builtin::LocalInvocationId(_) => 0,
            // pixels: lane l at (l % 8, l / 8) of an 8x4 tile, no helpers
            Builtin::FragCoord(0) => (0.5f32 + (lane % 8) as f32).to_bits(),
            Builtin::FragCoord(1) => (0.5f32 + (lane / 8) as f32).to_bits(),
            Builtin::FragCoord(2) => 0.5f32.to_bits(),
            Builtin::FragCoord(_) => 1.0f32.to_bits(),
            Builtin::FrontFacing | Builtin::SampleMaskIn => 1,
            Builtin::Helper | Builtin::SampleId | Builtin::PrimitiveId => 0,
            Builtin::Bary { comp, .. } => [0.25f32, 0.25, 0.5][usize::from(comp).min(2)].to_bits(),
            _ => return Err(format!("built-in {b:?}")),
        })
    }

    fn call(&mut self, h: Helper, v: V, n: &Node, l: usize, on: bool) -> Result<W, String> {
        let a = self.arg(n, 0, l)[0];
        Ok(match h {
            Helper::Translate => [self.mem.translate(a, 1), 0, 0, 0],
            Helper::TranslateRange => [self.mem.translate(a, self.arg(n, 1, l)[0]), 0, 0, 0],
            Helper::RangeLeft => {
                let d = a.wrapping_sub(self.mem.range.0);
                [if d < self.mem.range.1 { self.mem.range.1 - d } else { 0 }, 0, 0, 0]
            }
            Helper::TypedShape => [u64::from(super::emit::format_table()[(a & 127) as usize]), 0, 0, 0],
            Helper::Typed => {
                let e = self.arg(n, 1, l)[0] as u32;
                let bytes: Vec<u8> = (0..16).map(|i| self.mem.read(a + i, 1, !on, v) as u8).collect();
                typed_decode(e, &bytes)
            }
            Helper::TypedPack => {
                let x = self.arg(n, 1, l);
                typed_pack(a as u32, [x[0] as u32, x[1] as u32, x[2] as u32, x[3] as u32], self.arg(n, 2, l)[0] != 0)
            }
            Helper::BvhRay => return Err(format!("helper {h:?}")),
        })
    }

    /// A load, store or atomic on title memory (`on`: the invocation is
    /// inside its guard; stores and atomics of the others go nowhere).
    fn mem_op(&mut self, v: V, n: &Node, l: usize, on: bool) -> Result<W, String> {
        let a = self.arg(n, 0, l)[0];
        match n.op {
            Op::Load { .. } => {
                let w = (width(n.ty.s) / 8).max(1);
                let mut r = [0; 4];
                for (i, x) in r.iter_mut().enumerate().take(usize::from(n.ty.n)) {
                    *x = self.mem.read(a + u64::from(w) * i as u64, w, !on, v);
                }
                Ok(r)
            }
            Op::Store { .. } => {
                let t = self.f.ty(n.args[1]);
                let x = self.arg(n, 1, l);
                let w = width(t.s) / 8;
                for (i, &c) in x.iter().enumerate().take(usize::from(t.n)).filter(|_| on) {
                    self.mem.write(a + u64::from(w) * i as u64, w, c);
                }
                Ok([0; 4])
            }
            Op::Atomic(o) => {
                let old = self.mem.read(a, 4, !on, v) as u32;
                if on {
                    let new = atomic(o, old, self.arg(n, 1, l)[0] as u32, n.args.get(2).map(|_| self.arg(n, 2, l)[0] as u32))?;
                    self.mem.write(a, 4, u64::from(new));
                }
                Ok([u64::from(old), 0, 0, 0])
            }
            _ => Err("not a memory operation".into()),
        }
    }

    fn lds_op(&mut self, n: &Node, l: usize, on: bool) -> Result<W, String> {
        let i = (self.arg(n, 0, l)[0] as usize).min(self.lds.len() - 1);
        let old = self.lds[i];
        match n.op {
            Op::LdsStore if on => self.lds[i] = self.arg(n, 1, l)[0] as u32,
            Op::LdsAtomic(o) if on => {
                self.lds[i] = atomic(o, old, self.arg(n, 1, l)[0] as u32, n.args.get(2).map(|_| self.arg(n, 2, l)[0] as u32))?;
            }
            _ => {}
        }
        Ok([u64::from(old), 0, 0, 0])
    }

    fn wave_op(&mut self, o: u16, n: &Node, l: usize, spv: &[bool]) -> Result<W, String> {
        let r = Self::sub(l);
        Ok(match o {
            op::GROUP_NON_UNIFORM_BALLOT => {
                let m = r.clone().filter(|&i| spv[i] && self.arg(n, 0, i)[0] != 0).fold(0u64, |m, i| m | 1 << (i - r.start));
                [m, 0, 0, 0]
            }
            op::GROUP_NON_UNIFORM_ANY => [u64::from(r.clone().any(|i| spv[i] && self.arg(n, 0, i)[0] != 0)), 0, 0, 0],
            op::GROUP_NON_UNIFORM_ALL => [u64::from(r.clone().all(|i| !spv[i] || self.arg(n, 0, i)[0] != 0)), 0, 0, 0],
            op::GROUP_NON_UNIFORM_BROADCAST_FIRST => {
                let f = r.clone().find(|&i| spv[i]).unwrap_or(l);
                self.arg(n, 0, f)
            }
            op::GROUP_NON_UNIFORM_SHUFFLE => {
                let id = self.arg(n, 1, l)[0] as usize;
                if id >= 32 {
                    return Err(format!("shuffle from invocation {id}"));
                }
                let src = r.start + id;
                if !spv[src] {
                    self.note(n.args[0], || "a shuffle reads an inactive invocation".into());
                }
                self.arg(n, 0, src)
            }
            _ => return Err(format!("subgroup op {o}")),
        })
    }

    fn spv_op(&mut self, v: V, o: u16, n: &Node, l: usize) -> Result<W, String> {
        let a: Vec<W> = (0..n.args.len()).map(|k| self.arg(n, k, l)).collect();
        let mut p = Pure { f: self.f, notes: Vec::new() };
        let r = p.spv_op(v, o, n, &a);
        self.take_notes(p.notes);
        r
    }

    fn glsl_op(&mut self, g: u32, n: &Node, l: usize) -> Result<W, String> {
        let a: Vec<W> = (0..n.args.len()).map(|k| self.arg(n, k, l)).collect();
        let mut p = Pure { f: self.f, notes: Vec::new() };
        let r = p.glsl_op(g, n, &a);
        self.take_notes(p.notes);
        r
    }

    fn take_notes(&mut self, notes: Vec<(V, String)>) {
        for (v, s) in notes {
            self.notes.entry(v).or_insert(s);
        }
    }
}

/// The pure operations of the IR (SPIR-V core and GLSL.std.450) on values.
struct Pure<'a> {
    f: &'a Func,
    notes: Vec<(V, String)>,
}

impl Pure<'_> {
    fn spv_op(&mut self, v: V, o: u16, n: &Node, a: &[W]) -> Result<W, String> {
        let ty = n.ty;
        let a0t = n.args.first().map(|&a| self.f.ty(a)).unwrap_or(Ty::VOID);
        let mut r = [0u64; 4];
        match o {
            op::COMPOSITE_EXTRACT => r[0] = a[0][n.imm[0] as usize],
            op::COMPOSITE_CONSTRUCT => {
                let parts = n.args.iter().zip(a).flat_map(|(&x, w)| w[..usize::from(self.f.ty(x).n)].iter().copied());
                for (d, c) in r.iter_mut().zip(parts) {
                    *d = c;
                }
            }
            op::VECTOR_SHUFFLE => {
                let both: Vec<u64> = a[0][..usize::from(a0t.n)].iter().chain(&a[1]).copied().collect();
                for (d, &k) in r.iter_mut().zip(&n.imm) {
                    *d = both[k as usize];
                }
            }
            op::BITCAST => r = repack(&a[0], width(a0t.s), a0t.n, width(ty.s), ty.n),
            op::SELECT => {
                let vector = self.f.ty(n.args[0]).n > 1;
                for (i, d) in r.iter_mut().enumerate().take(usize::from(ty.n)) {
                    *d = if a[0][if vector { i } else { 0 }] != 0 { a[1][i] } else { a[2][i] };
                }
            }
            op::U_MUL_EXTENDED | op::S_MUL_EXTENDED => {
                let (x, y) = (a[0][0], a[1][0]);
                let p = if o == op::U_MUL_EXTENDED { x.wrapping_mul(y) } else { sext(x, 32).wrapping_mul(sext(y, 32)) as u64 };
                r = [p & 0xFFFF_FFFF, p >> 32, 0, 0];
            }
            _ => {
                for i in 0..usize::from(ty.n.max(1)) {
                    let xs: Vec<u64> = a.iter().map(|w| w[i]).collect();
                    r[i] = self.scalar_op(v, o, ty, a0t, &xs)?;
                }
            }
        }
        Ok(r)
    }

    fn scalar_op(&mut self, v: V, o: u16, ty: Ty, at: Ty, x: &[u64]) -> Result<u64, String> {
        if let Some(r) = compare(o, at, x).or_else(|| arith(o, ty, at, x)) {
            return Ok(r);
        }
        let m = mask(ty.s);
        let w = width(at.s);
        if matches!(o, op::U_DIV | op::U_MOD | op::S_DIV | op::S_REM) {
            if x[1] == 0 {
                self.notes.push((v, "division by zero".into()));
                return Ok(m);
            }
            let (a, b) = (sext(x[0], w), sext(x[1], w));
            return Ok(match o {
                op::U_DIV => x[0] / x[1],
                op::U_MOD => x[0] % x[1],
                op::S_DIV => (a.wrapping_div(b) as u64) & m,
                _ => (a.wrapping_rem(b) as u64) & m,
            });
        }
        self.bits_op(v, o, ty, at, x)
    }

    /// Shifts and bit fields (noting the amounts SPIR-V leaves undefined).
    fn bits_op(&mut self, v: V, o: u16, ty: Ty, at: Ty, x: &[u64]) -> Result<u64, String> {
        let m = mask(ty.s);
        let w = u64::from(width(at.s));
        let (amount, span) = match o {
            op::SHIFT_RIGHT_LOGICAL | op::SHIFT_RIGHT_ARITHMETIC | op::SHIFT_LEFT_LOGICAL => (x[1], 0),
            op::BIT_FIELD_INSERT => (x[2], x[3]),
            op::BIT_FIELD_U_EXTRACT | op::BIT_FIELD_S_EXTRACT => (x[1], x[2]),
            _ => return Err(format!("SPIR-V op {o}")),
        };
        if amount + span.max(1) > w {
            self.notes.push((v, format!("op {o}: bits {amount}+{span} of a {w}-bit value (undefined in SPIR-V)")));
        }
        let field = |cnt: u64| if cnt >= 64 { u64::MAX } else { (1u64 << cnt) - 1 };
        let s = amount.min(63) as u32;
        Ok(match o {
            op::SHIFT_RIGHT_LOGICAL => x[0].checked_shr(amount.min(w) as u32).unwrap_or(0),
            op::SHIFT_LEFT_LOGICAL => x[0].checked_shl(amount.min(w) as u32).unwrap_or(0) & m,
            op::SHIFT_RIGHT_ARITHMETIC => (sext(x[0], w as u32) >> s) as u64 & m,
            op::BIT_FIELD_INSERT => {
                let fm = field(span) << s;
                ((x[0] & !fm) | ((x[1] << s) & fm)) & m
            }
            _ if span == 0 => 0,
            op::BIT_FIELD_U_EXTRACT => (x[0] >> s) & field(span),
            _ => (sext((x[0] >> s) & field(span), span as u32) as u64) & m,
        })
    }

    fn glsl_op(&mut self, g: u32, n: &Node, a: &[W]) -> Result<W, String> {
        let at = self.f.ty(n.args[0]);
        if let Some(r) = glsl_whole(g, at, a[0]) {
            return Ok(r);
        }
        let mut r = [0u64; 4];
        for i in 0..usize::from(n.ty.n.max(1)) {
            let x: Vec<u64> = a.iter().map(|w| w[i]).collect();
            r[i] = glsl_one(g, n.ty, at, &x).ok_or_else(|| format!("GLSL op {g}"))?;
        }
        Ok(r)
    }
}

/// GLSL.std.450 operations on a whole value (structs, packing).
fn glsl_whole(g: u32, at: Ty, x: W) -> Option<W> {
    let h = |b: u64| u64::from(f16_to_f32(b as u16).to_bits());
    let p = |b: u64| u64::from(f32_to_f16(f32::from_bits(b as u32)));
    Some(match g {
        glsl::FREXP_STRUCT => {
            let (mant, exp) = frexp(to_f(at.s, x[0]));
            [from_f(at.s, mant), exp as i64 as u64 & 0xFFFF_FFFF, 0, 0]
        }
        glsl::PACK_HALF_2X16 => [p(x[0]) | p(x[1]) << 16, 0, 0, 0],
        glsl::UNPACK_HALF_2X16 => [h(x[0]), h(x[0] >> 16), 0, 0],
        _ => return None,
    })
}

/// A GLSL.std.450 operation on one component.
fn glsl_one(g: u32, ty: Ty, at: Ty, x: &[u64]) -> Option<u64> {
    let f = |k: usize| to_f(at.s, x[k]);
    let rf = |y: f64| from_f(ty.s, y);
    let s = |k: usize| sext(x[k], width(at.s));
    let m = mask(ty.s);
    // the lowest or highest set bit, -1 for none
    let bit = |y: u64, low: bool| match (y, low) {
        (0, _) => 0xFFFF_FFFF,
        (_, true) => u64::from(y.trailing_zeros()),
        _ => u64::from(63 - y.leading_zeros()),
    };
    let pick = |less: bool| if f(1).is_nan() || less { f(0) } else { f(1) };
    Some(match g {
        glsl::ROUND_EVEN => rf(f(0).round_ties_even()),
        glsl::TRUNC => rf(f(0).trunc()),
        glsl::F_ABS => rf(f(0).abs()),
        glsl::S_ABS => s(0).unsigned_abs() & m,
        glsl::FLOOR => rf(f(0).floor()),
        glsl::CEIL => rf(f(0).ceil()),
        glsl::FRACT => rf(f(0) - f(0).floor()),
        glsl::SIN => rf(f(0).sin()),
        glsl::COS => rf(f(0).cos()),
        glsl::EXP2 => rf(f(0).exp2()),
        glsl::LOG2 => rf(f(0).log2()),
        glsl::SQRT => rf(f(0).sqrt()),
        glsl::INVERSE_SQRT => rf(1.0 / f(0).sqrt()),
        glsl::U_MIN => x[0].min(x[1]),
        glsl::U_MAX => x[0].max(x[1]),
        glsl::S_MIN => (s(0).min(s(1)) as u64) & m,
        glsl::S_MAX => (s(0).max(s(1)) as u64) & m,
        glsl::F_MIN | glsl::N_MIN => rf(pick(f(0) < f(1))),
        glsl::F_MAX | glsl::N_MAX => rf(pick(f(0) > f(1))),
        glsl::F_CLAMP | glsl::N_CLAMP => rf(f(0).max(f(1)).min(f(2))),
        glsl::FMA => rf(f(0).mul_add(f(1), f(2))),
        glsl::LDEXP => rf(f(0) * 2f64.powi(sext(x[1], 32) as i32)),
        glsl::FIND_I_LSB => bit(x[0], true),
        glsl::FIND_U_MSB => bit(x[0], false),
        glsl::FIND_S_MSB => bit(if s(0) < 0 { !s(0) as u64 } else { s(0) as u64 }, false),
        _ => return None,
    })
}

/// Comparisons and logic of one component (`at`: the operands' type).
fn compare(o: u16, at: Ty, x: &[u64]) -> Option<u64> {
    let w = width(at.s);
    let s = |k: usize| sext(x[k], w);
    let f = |k: usize| to_f(at.s, x[k]);
    let unord = || f(0).is_nan() || f(1).is_nan();
    let r = match o {
        op::LOGICAL_EQUAL | op::I_EQUAL => x[0] == x[1],
        op::LOGICAL_NOT_EQUAL | op::I_NOT_EQUAL => x[0] != x[1],
        op::LOGICAL_OR => x[0] != 0 || x[1] != 0,
        op::LOGICAL_AND => x[0] != 0 && x[1] != 0,
        op::LOGICAL_NOT => x[0] == 0,
        op::U_GREATER_THAN => x[0] > x[1],
        op::S_GREATER_THAN => s(0) > s(1),
        op::U_GREATER_THAN_EQUAL => x[0] >= x[1],
        op::S_GREATER_THAN_EQUAL => s(0) >= s(1),
        op::U_LESS_THAN => x[0] < x[1],
        op::S_LESS_THAN => s(0) < s(1),
        op::U_LESS_THAN_EQUAL => x[0] <= x[1],
        op::S_LESS_THAN_EQUAL => s(0) <= s(1),
        op::IS_NAN => f(0).is_nan(),
        op::IS_INF => f(0).is_infinite(),
        op::F_ORD_EQUAL => f(0) == f(1),
        op::F_UNORD_EQUAL => unord() || f(0) == f(1),
        op::F_ORD_NOT_EQUAL => !unord() && f(0) != f(1),
        op::F_UNORD_NOT_EQUAL => f(0) != f(1),
        op::F_ORD_LESS_THAN => f(0) < f(1),
        op::F_UNORD_LESS_THAN => unord() || f(0) < f(1),
        op::F_ORD_GREATER_THAN => f(0) > f(1),
        op::F_UNORD_GREATER_THAN => unord() || f(0) > f(1),
        op::F_ORD_LESS_THAN_EQUAL => f(0) <= f(1),
        op::F_UNORD_LESS_THAN_EQUAL => unord() || f(0) <= f(1),
        op::F_ORD_GREATER_THAN_EQUAL => f(0) >= f(1),
        op::F_UNORD_GREATER_THAN_EQUAL => unord() || f(0) >= f(1),
        _ => return None,
    };
    Some(u64::from(r))
}

/// Conversions and arithmetic of one component (division aside).
fn arith(o: u16, ty: Ty, at: Ty, x: &[u64]) -> Option<u64> {
    let m = mask(ty.s);
    let s = |k: usize| sext(x[k], width(at.s));
    let f = |k: usize| to_f(at.s, x[k]);
    let rf = |y: f64| from_f(ty.s, y);
    let half = 2f64.powi(width(ty.s) as i32 - 1);
    Some(match o {
        op::U_CONVERT => x[0] & m,
        op::S_CONVERT => (s(0) as u64) & m,
        op::F_CONVERT => rf(f(0)),
        op::CONVERT_F_TO_U if f(0).is_nan() => 0,
        op::CONVERT_F_TO_U => f(0).clamp(0.0, m as f64) as u64 & m,
        op::CONVERT_F_TO_S if f(0).is_nan() => 0,
        op::CONVERT_F_TO_S => (f(0).clamp(-half, half - 1.0) as i64 as u64) & m,
        op::CONVERT_S_TO_F => rf(s(0) as f64),
        op::CONVERT_U_TO_F => rf(x[0] as f64),
        op::S_NEGATE => (s(0).wrapping_neg() as u64) & m,
        op::F_NEGATE => x[0] ^ (1 << (width(ty.s) - 1)),
        op::I_ADD => x[0].wrapping_add(x[1]) & m,
        op::I_SUB => x[0].wrapping_sub(x[1]) & m,
        op::I_MUL => x[0].wrapping_mul(x[1]) & m,
        op::F_ADD => rf(f32_op(f(0), f(1), |a, b| a + b)),
        op::F_SUB => rf(f32_op(f(0), f(1), |a, b| a - b)),
        op::F_MUL => rf(f32_op(f(0), f(1), |a, b| a * b)),
        op::F_DIV => rf(f32_op(f(0), f(1), |a, b| a / b)),
        op::F_REM => rf(f(0) % f(1)),
        op::BITWISE_OR => x[0] | x[1],
        op::BITWISE_XOR => x[0] ^ x[1],
        op::BITWISE_AND => x[0] & x[1],
        op::NOT => !x[0] & m,
        op::BIT_REVERSE => (x[0].reverse_bits() >> (64 - width(at.s))) & m,
        op::BIT_COUNT => u64::from(x[0].count_ones()),
        _ => return None,
    })
}

/// A typed element's channels (emit_typed.rs `Typed`): shape `e`, the
/// element's bytes (16 read; those past it unused).
pub fn typed_decode(e: u32, bytes: &[u8]) -> W {
    let (comps, bits, num) = (e & 7, e >> 8 & 0xFF, e >> 16 & 7);
    let all = u128::from_le_bytes(bytes[..16].try_into().unwrap());
    let mut r = [0u64; 4];
    for c in 0..4u32 {
        let w = if c == 3 && bits == 10 { 2 } else { bits };
        if c >= comps {
            r[c as usize] = if c == 3 { u64::from(if num >= 3 { 1 } else { 1.0f32.to_bits() }) } else { 0 };
            continue;
        }
        let raw = ((all >> (c * bits)) as u64 & ((1u64 << w) - 1)) as u32;
        let max = |b: u32| ((1u64 << b) - 1) as f32;
        let sx = (sext(u64::from(raw), w) as i32) as f32;
        r[c as usize] = u64::from(match num {
            0 if w == 32 => raw,
            0 => f16_to_f32(raw as u16).to_bits(),
            1 => (raw as f32 / max(w)).to_bits(),
            2 => (sx / max(w - 1)).max(-1.0).to_bits(),
            4 => sext(u64::from(raw), w) as u32,
            _ => raw,
        });
    }
    r
}

/// The encoder's reference (emit_typed.rs `TypedPack`): shape `e`, the
/// channel registers, d16 -> the element's dwords.
pub fn typed_pack(e: u32, x: [u32; 4], d16: bool) -> W {
    let (comps, bits, num) = (e & 7, e >> 8 & 0xFF, e >> 16 & 7);
    let mut out = [0u64; 4];
    for c in 0..comps.min(4) {
        let w = if c == 3 && bits == 10 { 2 } else { bits };
        let raw = x[c as usize];
        let raw = match (d16, num) {
            (false, _) => raw,
            (true, 3) => raw & 0xFFFF,
            (true, 4) => sext(u64::from(raw & 0xFFFF), 16) as u32,
            (true, _) => f16_to_f32(raw as u16).to_bits(),
        };
        let mask = if w >= 32 { u32::MAX } else { (1 << w) - 1 };
        let smax = (mask >> 1) as i32;
        let f = f32::from_bits(raw);
        let clamp = |lo: f32| if f.is_nan() { 0.0 } else { f.clamp(lo, 1.0) };
        let v = match num {
            0 if w == 32 => raw,
            0 => u32::from(f32_to_f16(f)),
            1 => (clamp(0.0) * mask as f32).round_ties_even() as u32,
            2 => (clamp(-1.0) * smax as f32).round_ties_even() as i32 as u32,
            4 => (raw as i32).clamp(!smax, smax) as u32,
            _ => raw.min(mask),
        } & mask;
        let at = c * bits;
        out[(at / 32) as usize] |= u64::from(v) << (at % 32);
    }
    out
}

/// A pixel input: a value in [0, 1) per location, component and lane.
fn input(loc: u8, comp: u8, l: usize, ty: Ty) -> u64 {
    let h = (u32::from(loc) * 4 + u32::from(comp)).wrapping_mul(0x9E37_79B9) ^ (l as u32).wrapping_mul(0x85EB_CA6B);
    let x = (h >> 8) as f32 / (1u32 << 24) as f32;
    u64::from(if ty.s == S::F32 { x.to_bits() } else { h & 0xFF })
}

/// An f32 operation rounded as f32 (not through f64 twice).
fn f32_op(a: f64, b: f64, o: impl Fn(f32, f32) -> f32) -> f64 {
    f64::from(o(a as f32, b as f32))
}

fn frexp(y: f64) -> (f64, i32) {
    if y == 0.0 || !y.is_finite() {
        return (y, 0);
    }
    let e = y.abs().log2().floor() as i32 + 1;
    let m = y / 2f64.powi(e);
    if m.abs() >= 1.0 { (m / 2.0, e + 1) } else if m.abs() < 0.5 { (m * 2.0, e - 1) } else { (m, e) }
}

fn atomic(o: u16, old: u32, x: u32, cmp: Option<u32>) -> Result<u32, String> {
    Ok(match o {
        op::ATOMIC_I_ADD => old.wrapping_add(x),
        op::ATOMIC_I_SUB => old.wrapping_sub(x),
        op::ATOMIC_AND => old & x,
        op::ATOMIC_OR => old | x,
        op::ATOMIC_XOR => old ^ x,
        op::ATOMIC_U_MIN => old.min(x),
        op::ATOMIC_U_MAX => old.max(x),
        op::ATOMIC_S_MIN => (old as i32).min(x as i32) as u32,
        op::ATOMIC_S_MAX => (old as i32).max(x as i32) as u32,
        op::ATOMIC_EXCHANGE => x,
        op::ATOMIC_COMPARE_EXCHANGE => {
            if Some(old) == cmp { x } else { old }
        }
        _ => return Err(format!("atomic {o}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::tir::Block;

    /// `for (i = 0; i + 1 != n; i++) store(a, i)` with n from user data 0.
    fn counted() -> Func {
        let mut f = Func::default();
        let zero = f.konst(Ty::U32, 0);
        let one = f.konst(Ty::U32, 1);
        let at = f.konst(Ty::U64, 0x1000);
        let n = f.global(Op::Push(4), Ty::U32);
        let i = f.push(Node { op: Op::Phi, ty: Ty::U32, args: vec![zero, 0], imm: vec![] });
        let next = f.push(Node { op: Op::Spv(op::I_ADD), ty: Ty::U32, args: vec![i, one], imm: vec![] });
        f.nodes[i as usize].args[1] = next;
        let done = f.push(Node { op: Op::Spv(op::I_EQUAL), ty: Ty::BOOL, args: vec![next, n], imm: vec![] });
        let store = f.push(Node { op: Op::Store { align: 4 }, ty: Ty::VOID, args: vec![at, i], imm: vec![] });
        f.blocks.push(Block { phis: vec![], preds: vec![], body: vec![], term: Term::Jump(1) });
        f.blocks.push(Block {
            phis: vec![i],
            preds: vec![0, 1],
            body: vec![Item::N(next), Item::N(done), Item::N(store)],
            term: Term::Branch(done, 2, 1),
        });
        f.blocks.push(Block { phis: vec![], preds: vec![1], body: vec![], term: Term::Return });
        f
    }

    #[test]
    fn a_counted_loop_runs_its_count_and_one_past_zero_hits_the_limit() {
        let f = counted();
        let mut m = Machine::new(&f, 32, Memory { range: (0, 1 << 32), ..Default::default() });
        m.push[4] = 5;
        assert_eq!(m.run(), Stop::Returned);
        assert_eq!(m.visits[1], 5);
        assert_eq!(m.mem.stored.get(&0x1000), Some(&4));
        // n = 0: the exit test (equality after the increment) never holds
        let mut m = Machine::new(&f, 32, Memory::default());
        m.limit = 1000;
        assert_eq!(m.run(), Stop::Limit);
    }

    #[test]
    fn typed_elements_encode_and_decode() {
        let t = crate::vk::emit::format_table();
        let bytes = |w: W| -> Vec<u8> { w.iter().flat_map(|&d| (d as u32).to_le_bytes()).collect() };
        let f = |x: f32| x.to_bits();
        // unorm8 x4 (56): 0.5 rounds to 128 (ties to even), clamps, NaN 0
        let w = typed_pack(t[56], [f(0.5), f(1.5), f(-1.0), 0x7FC0_0000], false);
        assert_eq!(w[0], 0x0000_FF80);
        // snorm16 x4 (66): -1, 1, 0.5 -> 16384, NaN 0
        let w = typed_pack(t[66], [f(-1.0), f(1.0), f(0.5), 0x7FC0_0000], false);
        assert_eq!((w[0], w[1]), (0x7FFF_8001, 0x0000_4000));
        // sint8 x4 (61) clamps; uint16 x1 (11) clamps
        assert_eq!(typed_pack(t[61], [300, (-300i32) as u32, 5, 0xFFFF_FFFF], false)[0], 0xFF05_807F);
        assert_eq!(typed_pack(t[11], [70000, 0, 0, 0], false)[0], 0xFFFF);
        // 10_10_10_2 unorm (50)
        assert_eq!(typed_pack(t[50], [f(1.0), f(0.0), f(1.0), f(1.0)], false)[0], 0xFFF0_03FF);
        // float16 x4 (71), d16 data passes halves through
        let w = typed_pack(t[71], [0x3C00, 0xC000, 0, 0x7BFF], true);
        assert_eq!((w[0], w[1]), (0xC000_3C00, 0x7BFF_0000));
        // decode inverts what fits
        let d = typed_decode(t[65], &bytes(typed_pack(t[65], [f(0.25), f(1.0), f(0.0), f(0.75)], false)));
        assert_eq!(f32::from_bits(d[1] as u32), 1.0);
        assert!((f32::from_bits(d[0] as u32) - 0.25).abs() < 1e-4);
    }

    #[test]
    fn halves_convert_both_ways() {
        for h in [0u16, 0x3C00, 0xC000, 0x7BFF, 0x0001, 0x7C00] {
            assert_eq!(f32_to_f16(f16_to_f32(h)), h);
        }
    }
}
