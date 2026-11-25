//! Machine instructions as HLSL statements (see the module notes).
//!
//! Every SSA value is a function-scope variable `v<n>` (`uint`, or `bool`
//! for SCC), declared at the top: blocks are placed by the structurizer,
//! so a definition's scope need not contain its uses. A phi is assigned on
//! each incoming edge. Lane masks are per-lane booleans (`structure`): a
//! compare's result is `bool l<n>`, a mask join `bool m<n>`. An instruction
//! the backend cannot express leaves its results missing; reading one fails
//! the reader, and a missing side effect fails the program.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write;

use super::cfg::{self, End};
use super::{Hlsl, Sig};
use crate::ShaderError;
use crate::emit::{HostShader, UserData};
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Inst, M0, Program, Reg, VCC, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, UsedResource};
use crate::resource::{Address, CODE, MEMORY, Resources, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

#[derive(Debug, Clone)]
enum Val {
    /// A `uint` expression (a variable or a literal).
    U(String),
    /// A `bool` variable (SCC).
    B(String),
    Missing(Why),
}

/// Loop limits as the bitcode backend sets them (`emit::lower::count_iteration`).
fn loop_limit(stage: HwStage) -> u32 {
    match stage {
        HwStage::Ps => 1 << 7,
        HwStage::Cs => 1 << 12,
        _ => 1 << 8,
    }
}

fn lit(x: u32) -> String {
    if x < 10 { format!("{x}u") } else { format!("0x{x:x}u") }
}

fn flt(e: &str) -> String {
    format!("asfloat({e})")
}

fn inline(c: u8) -> Option<u32> {
    crate::resource::inline(c)
}

fn inline_f16(c: u8) -> u32 {
    [0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118][usize::from(c - 240)]
}

struct L<'a> {
    native: &'a NativeProgram,
    p: &'a Program,
    st: &'a Structured,
    io: &'a Io,
    res: &'a Resources,
    inputs: Vec<Sig>,
    outputs: Vec<Sig>,
    bad: &'a HashSet<Value>,
    stage: HwStage,
    vals: HashMap<Value, Val>,
    /// Variables to declare: name -> type.
    decls: BTreeMap<String, &'static str>,
    lane_bits: HashMap<Value, String>,
    lane_fail: HashMap<Value, Why>,
    body: String,
    cur: usize,
    bodies: Vec<String>,
    ends: Vec<End>,
    edges: HashMap<(usize, usize), String>,
    /// Booleans built in the current block.
    bmemo: HashMap<Bool, String>,
    params: HashMap<Value, (usize, u32)>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    cmp_samplers: u64,
    root_cbvs: BTreeSet<u8>,
    root_srvs: BTreeSet<u8>,
    root_uavs: BTreeSet<u8>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    exported: Vec<bool>,
    rpo: Vec<usize>,
    phi_at: HashMap<Value, (usize, usize)>,
    read_phis: HashSet<Value>,
    /// Pixel inputs read: element -> channels; the mode they were read with.
    input_use: Vec<(u8, Option<u8>)>,
    output_use: Vec<u8>,
    /// Declared resources at global scope (root views, statics).
    globals: BTreeSet<String>,
    lds_dwords: u32,
    writes_uavs: bool,
    approximate: BTreeSet<&'static str>,
    tmp: usize,
    headers: Vec<usize>,
    /// Pixel position, front face, ids: (field, HLSL declaration).
    system: BTreeMap<&'static str, String>,
    /// Constant table in the code (dwords from byte `first`).
    code_table: Option<u32>,
    /// Lane-mask joins read, and those an edge could not give a value.
    mask_used: HashSet<Value>,
    mask_fail: HashSet<Value>,
    mesh: Option<Mesh>,
    prim_use: Vec<u8>,
    /// The program needs host waves of the console's 32 lanes.
    wave32: bool,
    /// Values some instruction other than a memory access reads, or a join
    /// takes (made on first use).
    numbers: Option<HashSet<Value>>,
    /// Ballots of lane masks made in the current block.
    ballots: HashMap<Value, (String, String)>,
    /// Heap indices of descriptors chosen by control flow, by join block.
    desc_phis: HashMap<(usize, bool), String>,
    /// Signature elements declared in register order ([`sig_order`]).
    sorted: bool,
    /// The console's 16-bit signature elements declared 16-bit
    /// (`float16_t`, DXC's `-enable-16bit-types`) rather than 32-bit: DXC then
    /// packs them apart from 32-bit ones, as the console's compiler did.
    half_io: bool,
}

/// Lowers the program; retries with phis whose incoming values have no HLSL
/// form treated as missing.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower(
    native: &NativeProgram,
    p: &Program,
    st: &Structured,
    io: &Io,
    res: &Resources,
    inputs: Vec<Sig>,
    outputs: Vec<Sig>,
    mesh: Option<Mesh>,
    form: u8,
) -> Result<Hlsl, ShaderError> {
    let mut bad = HashSet::new();
    for _ in 0..64 {
        let mut l = L::new(native, p, st, io, res, inputs.clone(), outputs.clone(), &bad);
        l.mesh = mesh.clone();
        l.sorted = form & 1 != 0;
        l.half_io = form & 2 != 0;
        l.prim_use = vec![0; mesh.as_ref().map_or(0, |m| m.prims.len())];
        match l.run() {
            Ok(()) => {}
            Err(Stop::Fail(e)) => return Err(e),
        }
        // Phis read by an instruction with an incoming value that has no HLSL
        // form. As in the bitcode backend (`fill_phis`): a join that only
        // other joins read takes 0 for such a value, and a join of numbers
        // with dead registers (a descriptor's or a lane mask's dwords: no
        // path that brings one reads the result) takes 0 for those.
        let mut missing: HashSet<Value> = HashSet::new();
        for b in &p.blocks {
            for phi in &b.phis {
                if bad.contains(&phi.value) || matches!(l.vals.get(&phi.value), Some(Val::Missing(_)) | None) {
                    continue;
                }
                let args: Vec<&Value> =
                    phi.args.iter().zip(&b.preds).filter(|&(_, &q)| l.rpo[q] != usize::MAX).map(|(a, _)| a).collect();
                let numbers = phi.reg != Reg::Scc && args.iter().any(|a| matches!(l.vals.get(a), Some(Val::U(_))));
                let gone = args.iter().any(|a| match l.vals.get(a) {
                    Some(Val::Missing(why)) => !(numbers && DEAD.contains(why)),
                    None => true,
                    _ => false,
                });
                if gone {
                    if l.read_phis.contains(&phi.value) && std::env::var_os("VICTORY_HLSL_WHY").is_some() {
                        let why: Vec<String> = args.iter().map(|a| format!("{:?}", l.vals.get(a))).collect();
                        eprintln!("join {:?} ({:?}) at block {:#x}: {}", phi.value, phi.reg, b.offset, why.join(", "));
                    }
                    missing.insert(phi.value);
                }
            }
        }
        let mut read: Vec<Value> = missing.iter().copied().filter(|v| l.read_phis.contains(v)).collect();
        read.extend(l.mask_fail.iter().copied().filter(|v| l.mask_used.contains(v)));
        if !read.is_empty() {
            bad.extend(read);
            continue;
        }
        return l.finish(&missing);
    }
    Err(ShaderError::Unsupported { offset: 0, what: "a phi that does not settle" })
}

enum Stop {
    Fail(ShaderError),
}

/// A mesh program's outputs and limits.
#[derive(Debug, Clone)]
pub(super) struct Mesh {
    /// Per-primitive outputs (`PSG1`, without the primitive id and cull flag).
    pub prims: Vec<Sig>,
    /// What the group id arrives multiplied by.
    pub stride: u32,
    pub threads: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
}

/// Registers no program reads as numbers (see `lower`).
const DEAD: [Why; 4] =
    ["a descriptor's dwords read as data", "a lane mask read as data", "a lane mask", "barycentrics read as data"];

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

impl<'a> L<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        native: &'a NativeProgram,
        p: &'a Program,
        st: &'a Structured,
        io: &'a Io,
        res: &'a Resources,
        inputs: Vec<Sig>,
        outputs: Vec<Sig>,
        bad: &'a HashSet<Value>,
    ) -> Self {
        let n = p.blocks.len();
        let mut phi_at = HashMap::new();
        for (b, blk) in p.blocks.iter().enumerate() {
            for (i, phi) in blk.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let n_in = inputs.len();
        let n_out = outputs.len();
        L {
            native,
            p,
            st,
            io,
            res,
            inputs,
            outputs,
            bad,
            stage: native.stage,
            vals: HashMap::new(),
            decls: BTreeMap::new(),
            lane_bits: HashMap::new(),
            lane_fail: HashMap::new(),
            body: String::new(),
            cur: 0,
            bodies: vec![String::new(); n],
            ends: vec![End::Leave(String::new()); n],
            edges: HashMap::new(),
            bmemo: HashMap::new(),
            params: res.params.iter().map(|&(v, param, dword)| (v, (param, dword))).collect(),
            user_reads: BTreeMap::new(),
            samplers: Vec::new(),
            cmp_samplers: 0,
            root_cbvs: BTreeSet::new(),
            root_srvs: BTreeSet::new(),
            root_uavs: BTreeSet::new(),
            descriptors: res.descriptors.iter().map(|d| ((d.site, d.role == crate::resource::Role::Sampler), &d.source)).collect(),
            scalar: res.scalar.iter().copied().collect(),
            exports: io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            exported: vec![false; n],
            rpo: vec![usize::MAX; n],
            phi_at,
            read_phis: HashSet::new(),
            input_use: vec![(0, None); n_in],
            output_use: vec![0; n_out],
            globals: BTreeSet::new(),
            lds_dwords: 0,
            writes_uavs: false,
            approximate: BTreeSet::new(),
            tmp: 0,
            headers: Vec::new(),
            system: BTreeMap::new(),
            code_table: None,
            mask_used: HashSet::new(),
            mask_fail: HashSet::new(),
            mesh: None,
            prim_use: Vec::new(),
            wave32: false,
            numbers: None,
            ballots: HashMap::new(),
            desc_phis: HashMap::new(),
            sorted: false,
            half_io: false,
        }
    }

    fn pixel(&self) -> bool {
        self.stage == HwStage::Ps
    }

    /// A console wave wider than the host's may be: cross-lane operations
    /// then act on the host wave.
    fn wide_waves(&self) -> bool {
        self.native.wave64 && self.stage == HwStage::Cs
    }

    /// Host lanes are the console's: a wave32 compute program, run with
    /// 32-lane waves (`[WaveSize(32)]` once it reads lane masks as numbers).
    fn console_lanes(&self) -> bool {
        self.stage == HwStage::Cs && !self.native.wave64
    }

    fn cross_lane(&mut self, what: Why) {
        if self.console_lanes() {
            self.wave32 = true;
        }
        if self.wide_waves() {
            self.approximate.insert(what);
        }
    }

    /// A temporary of the current block, declared where it is made (a
    /// block's statements share one scope).
    fn fresh(&mut self, prefix: &str, ty: &'static str) -> String {
        self.tmp += 1;
        let name = format!("{prefix}{}", self.tmp);
        if !ty.is_empty() {
            self.line(&format!("{ty} {name};"));
        }
        name
    }

    /// A variable several blocks set or read, declared at the top.
    fn fresh_global(&mut self, prefix: &str, ty: &'static str) -> String {
        self.tmp += 1;
        let name = format!("{prefix}{}", self.tmp);
        self.decls.insert(name.clone(), ty);
        name
    }

    fn line(&mut self, s: &str) {
        self.body.push_str(s);
        self.body.push('\n');
    }

    // ---- values ----

    fn name(v: Value) -> String {
        format!("v{}", v.0)
    }

    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        if let Some(x) = self.vals.get(&v) {
            return match x {
                Val::Missing(w) => Err(w),
                x => Ok(x.clone()),
            };
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let x = self.user(param, dword)?;
            self.vals.insert(v, Val::U(x.clone()));
            return Ok(Val::U(x));
        }
        if self.io.inputs.iter().any(|&(w, i)| w == v && matches!(i, Input::Barycentric { .. })) {
            return Err("barycentrics read as data");
        }
        Err("a value defined after its use")
    }

    fn u(&mut self, v: Value) -> Result<String, Why> {
        Ok(match self.get(v)? {
            Val::U(x) => x,
            Val::B(b) => format!("({b} ? 1u : 0u)"),
            Val::Missing(w) => return Err(w),
        })
    }

    /// Defines `v` as `expr` (a uint).
    fn set_u(&mut self, v: Option<Value>, expr: String) {
        let Some(v) = v else { return };
        let n = Self::name(v);
        // Literals and plain variables need no variable of their own.
        let simple = expr.ends_with('u') && expr.bytes().all(|b| b.is_ascii_hexdigit() || b == b'x' || b == b'u')
            || expr.starts_with('v') && expr[1..].bytes().all(|b| b.is_ascii_digit())
            || (expr.starts_with("ud[") || expr.starts_with("gtid.") || expr.starts_with("gid.") || expr.starts_with("i.")) && !expr.contains(' ');
        if simple {
            self.vals.insert(v, Val::U(expr));
            return;
        }
        // Declared here (`finish`: unless it is a join's).
        self.line(&format!("\u{1}{n} = {expr};"));
        self.vals.insert(v, Val::U(n));
    }

    fn set_b(&mut self, v: Option<Value>, expr: String) {
        let Some(v) = v else { return };
        let n = Self::name(v);
        self.line(&format!("\u{2}{n} = {expr};"));
        self.vals.insert(v, Val::B(n));
    }

    /// User-data dword of a root parameter.
    fn user(&mut self, param: usize, dword: u32) -> Result<String, Why> {
        let rp = self.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Ok("0u".into()),
            RootType::Constants | RootType::Table => {}
            _ => return Err("a root descriptor"),
        }
        let first = if crate::spill::is_spilled(self.native) { 16 } else { 0 };
        let index = u32::from(rp.user_data) + dword - first;
        self.user_reads.insert(index, UserData { index, slot: rp.slot, root: rp.root, dword });
        Ok(Self::ud(index))
    }

    fn ud(index: u32) -> String {
        format!("ud[{}].{}", index / 4, ["x", "y", "z", "w"][(index % 4) as usize])
    }

    /// A user-data dword the host computes (`emit::shader` `USER_*` codes).
    fn host_word(&mut self, root: RootType, slot: u8, dword: u32) -> String {
        if let Some(u) = self.user_reads.values().find(|u| u.root == root && u.slot == slot && u.dword == dword) {
            return Self::ud(u.index);
        }
        let first = if crate::spill::is_spilled(self.native) { 16 } else { 0 };
        let mut index = self
            .native
            .layout
            .iter()
            .map(|e| u32::from(e.user_data) + e.sgprs)
            .max()
            .unwrap_or(0)
            .saturating_sub(first);
        while self.user_reads.contains_key(&index) {
            index += 1;
        }
        self.user_reads.insert(index, UserData { index, slot, root, dword });
        Self::ud(index)
    }

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<String, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok("0u".into()),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                // A lane mask read as a number where host lanes are the
                // console's: its ballot.
                if self.console_lanes() && !matches!(self.vals.get(&v), Some(Val::U(_) | Val::B(_))) {
                    if let Some(&low) = self.st.highs.get(&v) {
                        return Ok(self.mask_data(low)?.1);
                    }
                    if self.st.masks.contains_key(&v) {
                        return Ok(self.mask_data(v)?.0);
                    }
                }
                self.u(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.u(v)
            }
            Operand::Const(c) => inline(c).map(lit).ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(lit(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<String, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => Ok(lit(inline_f16(c))),
            _ => self.operand(inst, op),
        }
    }

    fn vf(inst: &Inst) -> &crate::gfx10::Valu {
        match &inst.fields {
            Fields::Valu(v) => v,
            _ => unreachable!("a VALU instruction"),
        }
    }

    fn select_bits(x: &str, sel: u8, sext: bool) -> String {
        let (width, at) = if sel < 4 { (8, 8 * sel as u32) } else { (16, 16 * (sel as u32 - 4)) };
        let left = 32 - width - at;
        if sext {
            format!("uint(int({x} << {left}u) >> {}u)", 32 - width)
        } else {
            format!("(({x} >> {at}u) & {})", lit((1u32 << width) - 1))
        }
    }

    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let s = Self::vf(inst).sdwa?;
        let sel = *s.src_sel.get(k)?;
        (sel < 6).then_some((sel, s.sext >> k & 1 != 0))
    }

    /// Integer source `k` (SDWA applied; DPP src0 given by `src0`).
    fn src(&mut self, inst: &Inst, k: usize, src0: &Option<String>) -> Result<String, Why> {
        let v = Self::vf(inst);
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        if k == 0
            && let Some(x) = src0
        {
            return Ok(x.clone());
        }
        let x = self.operand(inst, &op)?;
        Ok(match Self::sdwa_src(inst, k) {
            Some((sel, sext)) => Self::select_bits(&x, sel, sext),
            None => x,
        })
    }

    /// Float source `k` with abs/neg.
    fn fsrc(&mut self, inst: &Inst, k: usize, src0: &Option<String>) -> Result<String, Why> {
        let v = Self::vf(inst);
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let x = match (k, src0) {
            (0, Some(x)) => x.clone(),
            _ => {
                let x = self.operand(inst, &op)?;
                match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => Self::select_bits(&x, sel, sext),
                    None => x,
                }
            }
        };
        let mut f = match x.strip_prefix("asuint(").and_then(|s| s.strip_suffix(')')) {
            Some(inner) if !inner.contains('(') => inner.to_owned(),
            _ => flt(&x),
        };
        if v.abs >> k & 1 != 0 {
            f = format!("abs({f})");
        }
        if v.neg >> k & 1 != 0 {
            f = format!("(-{f})");
        }
        Ok(f)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool) -> Result<String, Why> {
        if b == Bool::TRUE {
            return Ok("true".into());
        }
        if b == Bool::FALSE {
            return Ok("false".into());
        }
        if let Some(x) = self.bmemo.get(&b) {
            return Ok(x.clone());
        }
        let (atom, lo, hi) = self.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom)?;
        let e = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => format!("!{a}"),
            (Bool::FALSE, h) => format!("({a} && {})", self.boolean(h)?),
            (l, Bool::TRUE) => format!("({a} || {})", self.boolean(l)?),
            (l, h) => {
                let (h, l) = (self.boolean(h)?, self.boolean(l)?);
                format!("({a} ? {h} : {l})")
            }
        };
        let n = self.fresh("b", "bool");
        self.line(&format!("{n} = {e};"));
        self.bmemo.insert(b, n.clone());
        Ok(n)
    }

    fn atom(&mut self, atom: Atom) -> Result<String, Why> {
        match atom {
            Atom::Lane(v) => self
                .lane_bits
                .get(&v)
                .cloned()
                .ok_or_else(|| self.lane_fail.get(&v).copied().unwrap_or("a lane mask from a carry")),
            Atom::Uniform(v) => match self.get(v)? {
                Val::B(x) => Ok(x),
                Val::U(x) => Ok(format!("({x} != 0u)")),
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if self.bad.contains(&id) {
                    return Err("a lane-mask join without a value");
                }
                self.mask_used.insert(id);
                let n = format!("m{}", id.0);
                self.decls.insert(n.clone(), "bool");
                Ok(n)
            }
            // As the bitcode backend: lanes it adds only feed derivatives.
            Atom::Quad(_) => Ok("true".into()),
            // A mask held as a number, where host lanes are the console's: the
            // lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes() => {
                let x = match self.vals.get(&v) {
                    Some(Val::U(x)) => x.clone(),
                    _ => return Err("an opaque lane mask"),
                };
                self.wave32 = true;
                Ok(format!("((({x}) >> WaveGetLaneIndex()) & 1u) != 0u"))
            }
            Atom::Constant(k) if self.console_lanes() => {
                self.wave32 = true;
                Ok(format!("(({}u >> WaveGetLaneIndex()) & 1u) != 0u", k as u32))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<String>, Why> {
        match self.st.guards[b][i] {
            None => Ok(None),
            Some(g) => self.boolean(g).map(Some),
        }
    }

    /// The lane mask whose low dword is `v` as data (a ballot): (low, high).
    fn mask_data(&mut self, v: Value) -> Result<(String, String), Why> {
        if let Some(x) = self.ballots.get(&v) {
            return Ok(x.clone());
        }
        let &b = self.st.masks.get(&v).ok_or("a lane mask read as data")?;
        let e = self.boolean(b)?;
        self.cross_lane("a lane mask read as data");
        if self.console_lanes() {
            self.wave32 = true;
        }
        let n = self.fresh("w", "uint4");
        self.line(&format!("{n} = WaveActiveBallot({e});"));
        let r = (format!("{n}.x"), if self.native.wave64 { format!("{n}.y") } else { "0u".into() });
        self.ballots.insert(v, r.clone());
        Ok(r)
    }

    // ---- writes ----

    /// Writes VGPR `reg` (SDWA destination applied), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: String, guard: &Option<String>) -> Result<(), Why> {
        let Some(new) = inst.write(reg) else { return Ok(()) };
        let mut x = x;
        if let (Fields::Valu(v), Reg::V(_)) = (&inst.fields, reg)
            && let Some(s) = v.sdwa
            && s.dst_sel < 6
        {
            let (width, at) = if s.dst_sel < 4 { (8u32, 8 * s.dst_sel as u32) } else { (16, 16 * (s.dst_sel as u32 - 4)) };
            let left = 32 - width;
            let right = 32 - width - at;
            x = match s.dst_unused {
                1 => format!("uint(int(({x}) << {left}u) >> {right}u)"),
                2 => {
                    let old = match inst.read(reg) {
                        Some(o) => self.u(o)?,
                        None => "0u".into(),
                    };
                    let keep = !((((1u64 << width) - 1) as u32) << at);
                    format!("(({old} & {}) | ((({x}) << {left}u) >> {right}u))", lit(keep))
                }
                _ => format!("((({x}) << {left}u) >> {right}u)"),
            };
        }
        if let (Some(g), Reg::V(_)) = (guard, reg) {
            let old = match inst.read(reg).map(|o| self.u(o)) {
                Some(Ok(o)) => o,
                _ => "0u".into(),
            };
            x = format!("({g} ? {x} : {old})");
        }
        self.set_u(Some(new), x);
        Ok(())
    }

    fn write_f(&mut self, inst: &Inst, f: String, guard: &Option<String>) -> Result<(), Why> {
        let v = Self::vf(inst);
        let mut f = f;
        match v.omod {
            0 => {}
            1 => f = format!("({f} * 2.0f)"),
            2 => f = format!("({f} * 4.0f)"),
            _ => f = format!("({f} * 0.5f)"),
        }
        if v.clamp {
            f = format!("saturate({f})");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), format!("asuint({f})"), guard)
    }

    fn write_i(&mut self, inst: &Inst, x: String, guard: &Option<String>) -> Result<(), Why> {
        let v = Self::vf(inst);
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, &None),
            _ => Err("a result without a register"),
        }
    }

    // ---- program ----

    fn run(&mut self) -> Result<(), Stop> {
        let p = self.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for &b in &order {
            if p.blocks[b].preds.iter().any(|&q| self.rpo[q] != usize::MAX && self.rpo[b] <= self.rpo[q]) {
                self.headers.push(b);
            }
        }
        if let Some(m) = &self.mesh {
            let StageIo::Ms(ms) = &self.native.io else { return Err(unsupported(0, "a mesh program without stage data")) };
            if ms.topology != 2 {
                return Err(unsupported(0, "a mesh shader not drawing triangles"));
            }
            if m.threads > 128 || m.max_vertices > 256 || m.max_primitives > 256 {
                return Err(unsupported(0, "a mesh shader beyond the host's limits"));
            }
        }
        self.inputs()?;
        // Every join's variable first: an edge's copies are made where the
        // edge leaves, before the join's block is lowered. The joins of one
        // register share a variable (`r<register>`): a join's value is dead
        // once the register joins again, as in the machine code (DXC turns
        // fewer variables into SSA much faster; giant programs have
        // thousands of joins).
        for &b in &order {
            for phi in &p.blocks[b].phis {
                let n = match phi.reg {
                    Reg::S(c) => format!("rs{c}"),
                    Reg::V(c) => format!("rv{c}"),
                    Reg::Scc => "rscc".to_owned(),
                };
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains(&phi.value) => Val::Missing("a join of a value without HLSL form"),
                    Reg::Scc => {
                        self.decls.insert(n.clone(), "bool");
                        Val::B(n)
                    }
                    _ => {
                        self.decls.insert(n.clone(), "uint");
                        Val::U(n)
                    }
                };
                self.vals.insert(phi.value, val);
            }
        }
        for &b in &order {
            self.cur = b;
            self.bmemo.clear();
            self.ballots.clear();
            self.body = std::mem::take(&mut self.bodies[b]);
            let preds: Vec<usize> = p.blocks[b].preds.iter().copied().filter(|&q| self.rpo[q] != usize::MAX).collect();
            self.exported[b] = !preds.is_empty()
                && preds.iter().all(|&q| self.exported[q] || self.has_done_export(q));
            for i in 0..p.blocks[b].insts.len() {
                self.inst(b, i)?;
            }
            self.exit(b)?;
            self.bodies[b] = std::mem::take(&mut self.body);
        }
        Ok(())
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.p.blocks[b].insts.len()).any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue,
                // Read only by interpolation, which names them by meaning; as
                // numbers they have no host value (a join of them is dead).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    Val::Missing("barycentrics read as data")
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    None if n <= 105 && self.io.initialized_sgprs.is_some_and(|k| n >= k) => Val::U("0u".into()),
                    None if n >= crate::ir::PARKED => Val::U("0u".into()),
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    None => Val::U("0u".into()),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        match input {
            Input::ThreadIdInGroup(c) if c < 3 => {
                self.system.insert("gtid", "uint3 gtid : SV_GroupThreadID".into());
                Val::U(format!("gtid.{}", ["x", "y", "z"][usize::from(c)]))
            }
            Input::GroupId(c) if c < 3 => {
                self.system.insert("gid", "uint3 gid : SV_GroupID".into());
                Val::U(format!("gid.{}", ["x", "y", "z"][usize::from(c)]))
            }
            Input::Position(c) if c < 4 => match self.inputs.iter().position(|e| e.sv == 1) {
                Some(id) => {
                    self.input_use[id].0 |= 1 << c;
                    Val::U(format!("asuint(i.a{id}.{})", ["x", "y", "z", "w"][usize::from(c)]))
                }
                None => Val::Missing("a pixel system input the signature lacks"),
            },
            Input::FrontFace => match self.inputs.iter().position(|e| e.sv == 9) {
                Some(id) => {
                    self.input_use[id].0 |= 1;
                    Val::U(format!("(i.a{id} ? 1u : 0u)"))
                }
                None => Val::Missing("a pixel system input the signature lacks"),
            },
            Input::VertexId => self.vs_system(6, "vid", "uint vid : SV_VertexID"),
            Input::MeshGroup => {
                let stride = self.mesh.as_ref().map_or(1, |m| m.stride);
                self.system.insert("gid", "uint3 gid : SV_GroupID".into());
                Val::U(format!("(gid.x * {stride}u)"))
            }
            Input::MergedWaveInfo => {
                // The wave index in bits 24-27 (the console's lane counts are read by no program).
                let shift = if self.native.wave64 { 6 } else { 5 };
                self.system.insert("gi", "uint gi : SV_GroupIndex".into());
                Val::U(format!("((gi >> {shift}u) << 24)"))
            }
            Input::InstanceId => self.vs_system(8, "iid", "uint iid : SV_InstanceID"),
            Input::Attribute { row, channel } => {
                match self.inputs.iter().position(|e| e.reg == Some(u32::from(row)) && channel >= e.col() && channel < e.col() + e.cols()) {
                    Some(id) if self.inputs[id].half => Val::Missing("a 16-bit input"),
                    Some(id) => {
                        let c = channel - self.inputs[id].col();
                        self.input_use[id].0 |= 1 << c;
                        let comp = self.comp_expr(id, c, true);
                        Val::U(comp)
                    }
                    None => Val::Missing("a vertex input the signature lacks"),
                }
            }
            _ => Val::Missing("a system input the HLSL prototype lacks"),
        }
    }

    /// A vertex or instance id: the signature's element, or one of ours.
    fn vs_system(&mut self, sv: u32, key: &'static str, decl: &str) -> Val {
        match self.inputs.iter().position(|e| e.sv == sv) {
            Some(id) => {
                self.input_use[id].0 |= 1;
                Val::U(format!("i.a{id}"))
            }
            None => {
                self.system.insert(key, decl.into());
                Val::U(format!("i.{key}"))
            }
        }
    }

    /// Channel `c` of input element `id` as uint bits.
    fn comp_expr(&self, id: usize, c: u8, input: bool) -> String {
        let e = if input { &self.inputs[id] } else { &self.outputs[id] };
        let field = if e.cols() == 1 {
            format!("i.a{id}")
        } else {
            format!("i.a{id}.{}", ["x", "y", "z", "w"][usize::from(c)])
        };
        match e.comp {
            3 if e.native16 && self.half_io => format!("asuint(float({field}))"),
            3 => format!("asuint({field})"),
            _ => format!("uint({field})"),
        }
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read") || name.starts_with("ds_swizzle") || name.contains("permute"));
        let mark = self.body.len();
        let memo = self.bmemo.clone();
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" => {
                let all = self.writes_uavs_any();
                self.line(if all { "AllMemoryBarrierWithGroupSync();" } else { "GroupMemoryBarrierWithGroupSync();" });
                Ok(())
            }
            Fields::Sopp { .. } if name == "s_sendmsg" && self.stage != HwStage::Ms => Ok(()),
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } if name.starts_with("image_bvh") => Err("ray-tracing node tests"),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                // Nothing of a half-lowered instruction stays.
                self.body.truncate(mark);
                self.bmemo = memo;
                for &(_, w) in &inst.writes {
                    self.vals.insert(w, Val::Missing(why));
                    self.lane_bits.remove(&w);
                }
                Ok(())
            }
        }
    }

    fn writes_uavs_any(&mut self) -> bool {
        // UAV writes anywhere in the program make barriers fence them too.
        let any = self.p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            let n = i.name();
            (n.starts_with("buffer_") || n.starts_with("image_") || n.starts_with("global_"))
                && (n.contains("store") || n.contains("atomic"))
        });
        self.writes_uavs |= any;
        any
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.p.blocks[b].insts.last().map_or(self.p.blocks[b].offset, |i| i.machine.offset);
        let end = match self.st.exits[b] {
            LaneExit::Jump(t) => End::Jump(t),
            LaneExit::Branch { when, taken, not_taken } => match self.st.bools.node(when) {
                Some((atom, Bool::TRUE, Bool::FALSE)) => {
                    let a = self.atom(atom).map_err(|w| unsupported(offset, w))?;
                    End::Branch(a, not_taken, taken)
                }
                _ => {
                    let c = self.boolean(when).map_err(|w| unsupported(offset, w))?;
                    End::Branch(c, taken, not_taken)
                }
            },
            LaneExit::End => {
                let mut s = String::new();
                if self.pixel() && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    s.push_str("discard;\n");
                }
                s.push_str(if matches!(self.stage, HwStage::Cs | HwStage::Ms) { "return;" } else { "return o;" });
                End::Leave(s)
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        // Phi copies on each outgoing edge, computed here (the values they
        // read are defined at or before this block).
        let succs = match &end {
            End::Jump(t) => vec![*t],
            End::Branch(_, t, f) => vec![*t, *f],
            End::Leave(_) => vec![],
        };
        for s in succs {
            let copies = self.copies(b, s).map_err(|w| unsupported(offset, w))?;
            self.edges.insert((b, s), copies);
        }
        self.ends[b] = end;
        Ok(())
    }

    /// The phi assignments of the edge `q -> b` (parallel: through
    /// temporaries when one reads another phi of `b`).
    fn copies(&mut self, q: usize, b: usize) -> Result<String, Why> {
        let blk = &self.p.blocks[b];
        let Some(k) = blk.preds.iter().position(|&x| x == q) else { return Ok(String::new()) };
        let mut pairs: Vec<(String, String)> = Vec::new();
        for phi in &blk.phis {
            let target = match self.vals.get(&phi.value) {
                Some(Val::U(n) | Val::B(n)) => n.clone(),
                _ => Self::name(phi.value),
            };
            let arg = phi.args[k];
            // Lane-mask joins: the incoming mask's lane bit.
            if let Some(edges) = self.st.phi_masks.get(&phi.value) {
                let mb = format!("m{}", phi.value.0);
                if let Some(&e) = edges.get(k) {
                    match self.boolean(e) {
                        Ok(e) => {
                            self.decls.insert(mb.clone(), "bool");
                            pairs.push((mb, e));
                        }
                        Err(_) => {
                            self.mask_fail.insert(phi.value);
                        }
                    }
                }
            }
            if matches!(self.vals.get(&phi.value), Some(Val::Missing(_)) | None) {
                continue;
            }
            // A root parameter nothing read before the join.
            if !self.vals.contains_key(&arg)
                && let Some(&(param, dword)) = self.params.get(&arg)
                && let Ok(x) = self.user(param, dword)
            {
                self.vals.insert(arg, Val::U(x));
            }
            let x = match self.vals.get(&arg) {
                Some(Val::U(x)) => {
                    if matches!(self.vals.get(&phi.value), Some(Val::B(_))) {
                        format!("({x} != 0u)")
                    } else {
                        x.clone()
                    }
                }
                Some(Val::B(x)) => {
                    if matches!(self.vals.get(&phi.value), Some(Val::B(_))) {
                        x.clone()
                    } else {
                        format!("({x} ? 1u : 0u)")
                    }
                }
                // No value on this edge: 0 (`lower` refuses where that is read).
                _ if matches!(self.vals.get(&phi.value), Some(Val::B(_))) => "false".into(),
                _ => "0u".into(),
            };
            if x != target {
                pairs.push((target, x));
            }
        }
        // Booleans made for the copies sit in the block body (before its exit).
        let reads_phi = pairs.iter().any(|(_, x)| pairs.iter().any(|(t, _)| x.contains(t.as_str())));
        let mut s = String::new();
        if reads_phi && pairs.len() > 1 {
            for (n, (t, x)) in pairs.iter().enumerate() {
                let ty = self.decls.get(t).copied().unwrap_or("uint");
                let _ = writeln!(s, "{ty} t{n}_{q}_{b} = {x};");
            }
            for (n, (t, _)) in pairs.iter().enumerate() {
                let _ = writeln!(s, "{t} = t{n}_{q}_{b};");
            }
            s = format!("{{\n{s}}}\n");
        } else {
            for (t, x) in &pairs {
                let _ = writeln!(s, "{t} = {x};");
            }
        }
        Ok(s)
    }

    // ---- scalar ALU ----

    fn scc_in(&mut self, inst: &Inst) -> Result<String, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::B(x) => Ok(x),
            Val::U(x) => Ok(format!("({x} != 0u)")),
            Val::Missing(w) => Err(w),
        }
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else { unreachable!() };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || lo.is_some_and(|v| self.st.masks.contains_key(&v)) && !self.console_lanes() {
            // Lane masks live as booleans; their dwords are no numbers here.
            return Err("a lane mask read as data");
        }
        if matches!(name, "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version") {
            return Err("a register the emitter does not model");
        }
        if name == "s_getpc_b64" {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            self.set_u(lo, lit(inst.machine.offset as u32 + 4));
            self.set_u(hi, "0u".into());
            return Ok(());
        }
        let srcs: Vec<Operand> = src.clone();
        let s32 = |l: &mut Self, k: usize| -> Result<String, Why> {
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = srcs.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = lit(i32::from(imm.unwrap_or(0) as i16) as u32);
        let half = |l: &mut Self, k: usize, h: u8| -> Result<String, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    if l.st.masks.contains_key(&v) || l.st.highs.contains_key(&v) {
                        // A mask read as a number: its ballot.
                        let lowv = if h == 0 { v } else { *l.st.highs.get(&v).ok_or("a lane mask read as data")? };
                        let (a, b) = l.mask_data(lowv)?;
                        return Ok(if h == 0 { a } else { b });
                    }
                    l.u(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = inline(*k).ok_or("a special operand")?;
                    Ok(if (x as i32) < 0 { lit(u32::MAX) } else { "0u".into() })
                }
                _ => Ok("0u".into()),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            let bit = s32(self, 1)?;
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                format!("((({bit}) & 32u) != 0u ? {b} : {a})")
            } else {
                s32(self, 0)?
            };
            let want = u32::from(op.starts_with('1'));
            self.set_b(scc, format!("((({word} >> (({bit}) & 31u)) & 1u) == {want}u)"));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            let (a0, b0, a1, b1) = (half(self, 0, 0)?, half(self, 1, 0)?, half(self, 0, 1)?, half(self, 1, 1)?);
            let eq = format!("({a0} == {b0} && {a1} == {b1})");
            self.set_b(scc, if op.starts_with("lg") { format!("!{eq}") } else { eq });
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") { sext16.clone() } else { lit(u32::from(imm.unwrap_or(0))) };
                (s32(self, 0)?, b)
            } else {
                (s32(self, 0)?, s32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let (a, b) = if signed { (format!("int({a})"), format!("int({b})")) } else { (a, b) };
            let o = match &op[..2] {
                "eq" => "==",
                "lg" => "!=",
                "gt" => ">",
                "ge" => ">=",
                "lt" => "<",
                "le" => "<=",
                _ => return Err("a scalar compare"),
            };
            self.set_b(scc, format!("({a} {o} {b})"));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            match name {
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = self.fresh("t", "uint");
                    self.line(&format!("{n} = countbits({a}) + countbits({b});"));
                    self.set_u(lo, n.clone());
                    self.set_b(scc, format!("({n} != 0u)"));
                }
                "s_ff1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set_u(lo, format!("({a} != 0u ? firstbitlow({a}) : ({b} != 0u ? firstbitlow({b}) + 32u : 0xffffffffu))"));
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64" | "s_nand_b64" | "s_nor_b64"
                | "s_xnor_b64" | "s_not_b64" => {
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            format!("~{a}")
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = format!("~{b}");
                            }
                            let o = match &name[2..5] {
                                "and" | "nan" => "&",
                                "or_" | "orn" | "nor" => "|",
                                _ => "^",
                            };
                            let x = format!("({a} {o} {b})");
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") { format!("~{x}") } else { x }
                        };
                        r.push(x);
                    }
                    let (n0, n1) = (self.fresh("t", "uint"), self.fresh("t", "uint"));
                    self.line(&format!("{n0} = {};", r[0]));
                    self.line(&format!("{n1} = {};", r[1]));
                    self.set_u(lo, n0.clone());
                    self.set_u(hi, n1.clone());
                    self.set_b(scc, format!("(({n0} | {n1}) != 0u)"));
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = s32(self, 1)?;
                    let t = self.fresh("q", "uint64_t");
                    let op = match name {
                        "s_lshl_b64" => format!("(uint64_t({a}) | (uint64_t({b}) << 32)) << (({n}) & 63u)"),
                        "s_lshr_b64" => format!("(uint64_t({a}) | (uint64_t({b}) << 32)) >> (({n}) & 63u)"),
                        _ => format!("uint64_t(int64_t(uint64_t({a}) | (uint64_t({b}) << 32)) >> (({n}) & 63u))"),
                    };
                    self.line(&format!("{t} = {op};"));
                    self.set_u(lo, format!("uint({t})"));
                    self.set_u(hi, format!("uint({t} >> 32)"));
                    self.set_b(scc, format!("({t} != 0)"));
                }
                "s_mov_b64" => {
                    for (h, v) in [(0, lo), (1, hi)] {
                        match half(self, 0, h) {
                            Ok(a) => self.set_u(v, a),
                            Err(why) => {
                                if let Some(v) = v {
                                    self.vals.insert(v, Val::Missing(why));
                                }
                            }
                        }
                    }
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        self.set_u(v, format!("({cond} ? {a} : {b})"));
                    }
                }
                _ => return Err(name),
            }
            return Ok(());
        }
        let (r, carry): (String, Option<String>) = match name {
            "s_mov_b32" => (s32(self, 0)?, None),
            "s_movk_i32" => (sext16.clone(), None),
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (s32(self, 0)?, s32(self, 1)?);
                (format!("({cond} ? {a} : {b})"), None)
            }
            "s_brev_b32" => (format!("reversebits({})", s32(self, 0)?), None),
            "s_not_b32" => (format!("~{}", s32(self, 0)?), None),
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = s32(self, 0)?;
                let b = if name == "s_addk_i32" { sext16.clone() } else { s32(self, 1)? };
                if name == "s_add_u32" {
                    let t = self.fresh("t", "uint");
                    self.line(&format!("{t} = {a} + {b};"));
                    (t.clone(), Some(format!("({t} < {a})")))
                } else {
                    (format!("({a} + {b})"), None)
                }
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (s32(self, 0)?, s32(self, 1)?);
                let carry = (name == "s_sub_u32").then(|| format!("({a} < {b})"));
                (format!("({a} - {b})"), carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (s32(self, 0)?, s32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, c) = self.add_sub_carry(name == "s_addc_u32", &a, &b, Some(&cin));
                (r, Some(c))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = s32(self, 0)?;
                let b = if name == "s_mulk_i32" { sext16.clone() } else { s32(self, 1)? };
                (format!("({a} * {b})"), None)
            }
            "s_mul_hi_u32" => (format!("hw_mul_hi_u({}, {})", s32(self, 0)?, s32(self, 1)?), None),
            "s_mul_hi_i32" => (format!("hw_mul_hi_i({}, {})", s32(self, 0)?, s32(self, 1)?), None),
            "s_flbit_i32_b32" => {
                let a = s32(self, 0)?;
                (format!("({a} == 0u ? 0xffffffffu : 31u - firstbithigh({a}))"), None)
            }
            "s_ff1_i32_b32" => (format!("firstbitlow({})", s32(self, 0)?), None),
            "s_bcnt1_i32_b32" => {
                let src0 = src.first().copied();
                let a = match src0 {
                    Some(Operand::Scalar { code, .. }) if code != NULL => {
                        let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                        if self.st.masks.contains_key(&v) { self.mask_data(v)?.0 } else { self.u(v)? }
                    }
                    _ => s32(self, 0)?,
                };
                (format!("countbits({a})"), None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                let (a, b) = (s32(self, 0)?, s32(self, 1)?);
                let f = if name == "s_bfe_i32" { "hw_bfe_i" } else { "hw_bfe_u" };
                (format!("{f}({a}, {b}, ({b} >> 16) & 0x7fu)"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32" | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (s32(self, 0)?, s32(self, 1)?);
                if name.contains("n2") {
                    b = format!("~{b}");
                }
                let o = match &name[2..5] {
                    "and" | "nan" => "&",
                    "or_" | "orn" | "nor" => "|",
                    _ => "^",
                };
                let mut r = format!("({a} {o} {b})");
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = format!("~{r}");
                }
                (r, None)
            }
            "s_lshl_b32" => (format!("({} << ({} & 31u))", s32(self, 0)?, s32(self, 1)?), None),
            "s_lshr_b32" => (format!("({} >> ({} & 31u))", s32(self, 0)?, s32(self, 1)?), None),
            "s_ashr_i32" => (format!("uint(int({}) >> ({} & 31u))", s32(self, 0)?, s32(self, 1)?), None),
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let k = name.as_bytes()[6] - b'0';
                (format!("(({} << {k}u) + {})", s32(self, 0)?, s32(self, 1)?), None)
            }
            "s_min_u32" => (format!("min({}, {})", s32(self, 0)?, s32(self, 1)?), None),
            "s_max_u32" => (format!("max({}, {})", s32(self, 0)?, s32(self, 1)?), None),
            "s_min_i32" => (format!("uint(min(int({}), int({})))", s32(self, 0)?, s32(self, 1)?), None),
            "s_max_i32" => (format!("uint(max(int({}), int({})))", s32(self, 0)?, s32(self, 1)?), None),
            "s_abs_i32" => (format!("uint(abs(int({})))", s32(self, 0)?), None),
            "s_bitset0_b32" | "s_bitset1_b32" => {
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = s32(self, 0)?;
                if name == "s_bitset1_b32" {
                    (format!("({old} | (1u << ({a} & 31u)))"), None)
                } else {
                    (format!("({old} & ~(1u << ({a} & 31u)))"), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (s32(self, 0)?, s32(self, 1)?);
                (format!("(((1u << ({a} & 31u)) - 1u) << ({b} & 31u))"), None)
            }
            _ => return Err(name),
        };
        let needs_scc = scc.is_some()
            && !matches!(name, "s_mov_b32" | "s_movk_i32" | "s_cselect_b32" | "s_brev_b32" | "s_mul_i32" | "s_mulk_i32");
        if needs_scc && carry.is_none() && !(name.starts_with("s_add") || name.starts_with("s_sub")) {
            // SCC is "the result is not 0": name the result once.
            let t = self.fresh("t", "uint");
            self.line(&format!("{t} = {r};"));
            self.set_u(lo, t.clone());
            self.set_b(scc, format!("({t} != 0u)"));
            return Ok(());
        }
        self.set_u(lo, r);
        if needs_scc {
            match carry {
                Some(c) => self.set_b(scc, c),
                None => {
                    if let Some(s) = scc {
                        self.vals.insert(s, Val::Missing("the carry of a scalar operation"));
                    }
                }
            }
        }
        Ok(())
    }

    fn add_sub_carry(&mut self, add: bool, a: &str, b: &str, cin: Option<&str>) -> (String, String) {
        let op = if add { "+" } else { "-" };
        let t = self.fresh("t", "uint");
        self.line(&format!("{t} = {a} {op} {b};"));
        let mut carry = if add { format!("({t} < {a})") } else { format!("({a} < {b})") };
        let mut r = t.clone();
        if let Some(cin) = cin {
            let r2 = self.fresh("t", "uint");
            self.line(&format!("{r2} = {t} {op} ({cin} ? 1u : 0u);"));
            let edge = if add { "0xffffffffu" } else { "0u" };
            carry = format!("({carry} || ({cin} && {t} == {edge}))");
            r = r2;
        }
        let c = self.fresh("k", "bool");
        self.line(&format!("{c} = {carry};"));
        (r, c)
    }

    // ---- scalar memory ----

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem { data, offset, soffset, .. } = &inst.fields else { unreachable!() };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if self.scalar.get(&site).is_some_and(|a| a.param == MEMORY) {
            return Err("title memory by address");
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count: u32 = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else { return Err("a scalar buffer load") };
        let src = self.descriptors.get(&(site, false)).copied().ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        let k = (*offset as u32).wrapping_add(base);
        let mut addr = lit(k);
        let mut constant = true;
        if let Some(x) = extra {
            addr = format!("({addr} + {x})");
            constant = false;
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = format!("({addr} + {so})");
            constant = false;
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            return Err("a descriptor built around a table in the code");
        }
        let picked = match kind {
            None => self.heap_buffer(src)?,
            Some(_) => false,
        };
        if picked {
            // A buffer the program picks from the view heap by an index it
            // computes: the heap holds a buffer view there or a constant
            // buffer; the host's table of strides by slot tells which (as
            // `emit::lower::smem`).
            let stride = self.slot_stride(src)?.ok_or("a buffer picked from the view heap")?;
            let a = self.fresh("a", "uint");
            self.line(&format!("{a} = {addr};"));
            let ts: Vec<String> = (0..count).map(|_| self.fresh("c", "uint")).collect();
            self.line(&format!("if ({stride} != 0u) {{"));
            let h = self.handle(site, src, "ByteAddressBuffer", false)?;
            for (j, t) in ts.iter().enumerate() {
                self.line(&format!("{t} = {h}.Load({a} + {}u);", 4 * j));
            }
            self.line("} else {");
            let h = self.handle(site, src, "CB", false)?;
            for (j, t) in ts.iter().enumerate() {
                self.line(&format!("{t} = {h}.d[({a} + {}u) >> 4][(({a} + {}u) >> 2) & 3u];", 4 * j, 4 * j));
            }
            self.line("}");
            for (j, t) in ts.into_iter().enumerate() {
                self.write(inst, Reg::S(code + j as u8), t, &None)?;
            }
            return Ok(());
        }
        if matches!(kind, Some(u) if u.kind == ResourceType::Srv) {
            let h = self.handle(site, src, "ByteAddressBuffer", false)?;
            let a = self.fresh("a", "uint");
            self.line(&format!("{a} = {addr};"));
            for j in 0..count {
                let x = format!("{h}.Load({a} + {}u)", 4 * j);
                self.write(inst, Reg::S(code + j as u8), x, &None)?;
            }
            return Ok(());
        }
        let h = self.handle(site, src, "CB", false)?;
        if constant {
            for j in 0..count {
                let byte = k + 4 * j;
                let x = format!("{h}.d[{}].{}", byte >> 4, ["x", "y", "z", "w"][((byte >> 2) & 3) as usize]);
                let t = self.fresh("c", "uint");
                self.line(&format!("{t} = {x};"));
                self.write(inst, Reg::S(code + j as u8), t, &None)?;
            }
        } else {
            let a = self.fresh("a", "uint");
            self.line(&format!("{a} = {addr};"));
            for j in 0..count {
                let t = self.fresh("c", "uint");
                self.line(&format!("{t} = {h}.d[({a} + {}u) >> 4][(({a} + {}u) >> 2) & 3u];", 4 * j, 4 * j));
                self.write(inst, Reg::S(code + j as u8), t, &None)?;
            }
        }
        Ok(())
    }

    /// A descriptor load. Its dwords are no numbers here (the resources it
    /// names are reached by its address), except the sizes the console
    /// compiler reads out of them instead of querying (an image's width,
    /// height, depth and levels, a buffer's record count): those are rebuilt
    /// from the host view's dimensions in the GFX10 layout, as
    /// `emit::lower::descriptor_data` does. The others stay missing.
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const DEAD_WORD: Why = "a descriptor's dwords read as data";
        let Fields::Smem { data: Some(Operand::Scalar { code, .. }), .. } = inst.fields else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // Is any dword read as a number at all? (Most descriptor loads only
        // name resources.)
        if self.numbers.is_none() {
            let mut n = HashSet::new();
            for b in &self.p.blocks {
                for i in &b.insts {
                    if !matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. } | Fields::Smem { .. }) {
                        n.extend(i.reads.iter().map(|r| r.1));
                    }
                }
                for ph in &b.phis {
                    n.extend(ph.args.iter().copied());
                }
            }
            self.numbers = Some(n);
        }
        let numbers = self.numbers.as_ref().expect("made");
        let read = inst.writes.iter().any(|w| numbers.contains(&w.1));
        // What uses the descriptor tells the view's kind (an image or buffer
        // instruction rather than a scalar buffer load).
        let uses: Vec<(Site, u8)> = self
            .res
            .descriptors
            .iter()
            .filter(|d| d.role == crate::resource::Role::Resource)
            .filter_map(|d| {
                d.source.leaves().into_iter().find_map(|l| match *l {
                    Source::Loaded { load, first } if load == site => Some((d.site, first)),
                    _ => None,
                })
            })
            .collect();
        let by_smem = |u: &(Site, u8)| matches!(self.p.blocks[u.0.0].insts[u.0.1].fields, Fields::Smem { .. });
        let use_at = uses.iter().find(|u| !by_smem(u)).or(uses.first()).copied();
        let mut words: Vec<(u32, String)> = Vec::new();
        let mut first = 0u8;
        if read && let Some((at, f)) = use_at {
            first = f;
            let src = Source::Loaded { load: site, first: f };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store") || un.contains("atomic") || used.is_some_and(|u| u.kind == ResourceType::Uav);
            match &ui.fields {
                Fields::Image { dim, .. } if !un.starts_with("image_bvh") => {
                    let (tex, ty, comps) = match dim {
                        0 => ("Texture1D", 8, 1),
                        1 => ("Texture2D", 9, 2),
                        2 => ("Texture3D", 10, 3),
                        4 => ("Texture1DArray", 12, 2),
                        5 => ("Texture2DArray", 13, 3),
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    let t = if uav { format!("RW{tex}<float4>") } else { format!("{tex}<float4>") };
                    let h = self.handle(site, &src, &t, false)?;
                    let d = self.fresh("d", "uint4");
                    let outs = match comps {
                        1 => format!("{d}.x"),
                        2 => format!("{d}.x, {d}.y"),
                        _ => format!("{d}.x, {d}.y, {d}.z"),
                    };
                    if uav {
                        self.line(&format!("{h}.GetDimensions({outs}); {d}.w = 1u;"));
                    } else {
                        self.line(&format!("{h}.GetDimensions(0u, {outs}, {d}.w);"));
                    }
                    let (height, depth) = match dim {
                        0 => ("1u".to_owned(), "1u".to_owned()),
                        1 => (format!("{d}.y"), "1u".into()),
                        4 => ("1u".into(), format!("{d}.y")),
                        _ => (format!("{d}.y"), format!("{d}.z")),
                    };
                    words.push((1, format!("(({d}.x - 1u) << 30)")));
                    words.push((2, format!("((({d}.x - 1u) >> 2) & 0x3fffu) | (({height} - 1u) << 14)")));
                    words.push((3, format!("({} | (({d}.w - 1u) << 16))", lit(0xFAC | ty << 28))));
                    words.push((4, format!("({depth} - 1u)")));
                }
                Fields::Buffer { .. } | Fields::Smem { .. } if used.is_none_or(|u| u.kind != ResourceType::Cbv) => {
                    if matches!(ui.fields, Fields::Smem { .. }) && used.is_none_or(|u| u.kind != ResourceType::Srv) {
                        // A constant buffer: nothing of its descriptor is a number.
                    } else {
                        let ty = if uav { "RWByteAddressBuffer" } else { "ByteAddressBuffer" };
                        let h = self.handle(site, &src, ty, false)?;
                        let n = self.fresh("d", "uint");
                        self.line(&format!("{h}.GetDimensions({n});"));
                        let stride = used.map_or(0, |u| u.stride);
                        // A raw view counts bytes; the record count is in elements.
                        words.push((2, if stride > 1 { format!("({n} / {stride}u)") } else { n }));
                    }
                }
                _ => {}
            }
        }
        for k in 0..count {
            let reg = Reg::S(code + k as u8);
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some((_, x)) => {
                    let t = self.fresh("t", "uint");
                    self.line(&format!("{t} = {x};"));
                    self.write(inst, reg, t, &None)?;
                }
                None => {
                    if let Some(v) = inst.write(reg) {
                        self.vals.insert(v, Val::Missing(DEAD_WORD));
                    }
                }
            }
        }
        Ok(())
    }

    /// Reads from the program's constant table (bytes after the code).
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem { data: Some(Operand::Scalar { code, .. }), .. } = inst.fields else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let words = self.native.code.len() as u32 / 4;
        match a.dynamic {
            None => {
                for j in 0..count {
                    let byte = a.offset + 4 * j;
                    let w = self
                        .native
                        .code
                        .get(byte as usize..byte as usize + 4)
                        .map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
                    self.write(inst, Reg::S(code + j as u8), lit(w), &None)?;
                }
            }
            Some(v) => {
                let d = self.u(v)?;
                self.code_table = Some(words);
                let at = self.fresh("a", "uint");
                self.line(&format!("{at} = ({} + {d}) >> 2;", lit(a.offset)));
                for j in 0..count {
                    let x = format!("code_table[min({at} + {j}u, {}u)]", words.saturating_sub(1));
                    self.write(inst, Reg::S(code + j as u8), x, &None)?;
                }
            }
        }
        Ok(())
    }

    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } = src {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        let native = self.native;
        if let Source::Built { address, .. } = src {
            let rp = native.layout.get(address.param)?;
            return native.used.iter().find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else { return None };
        let a = self.scalar.get(load)?;
        let rp = native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        native
            .used
            .iter()
            .filter(|u| u.root == RootType::Table && u.user_data == rp.user_data && u.table_offset_dwords <= dword)
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// Whether `src` is picked from the view heap at a computed index (root
    /// type `ResourceHeap`).
    fn heap_buffer(&self, src: &Source) -> Result<bool, Why> {
        let Source::Loaded { load, .. } = src else { return Ok(false) };
        Ok(self
            .scalar
            .get(load)
            .and_then(|a| self.native.layout.get(a.param))
            .is_some_and(|p| p.root == RootType::ResourceHeap))
    }

    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<String>), Why> {
        match src {
            Source::Built { address, .. } => {
                let mut x = match address.dynamic {
                    Some(v) => Some(self.u(v)?),
                    None => None,
                };
                if let Some(bias) = self.root_bias(address.param) {
                    x = Some(match x {
                        Some(d) => format!("({d} + {bias})"),
                        None => bias,
                    });
                }
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    fn root_bias(&mut self, param: usize) -> Option<String> {
        let rp = *self.native.layout.get(param)?;
        if rp.root != RootType::Descriptor {
            return None;
        }
        let kind = self.native.used.iter().find(|u| u.user_data == rp.user_data).map(|u| u.kind);
        let biased = match kind {
            Some(ResourceType::Uav) => true,
            Some(ResourceType::Srv) => crate::spill::is_spilled(self.native),
            _ => false,
        };
        biased.then(|| self.host_word(RootType::Other(crate::emit::USER_VIEW_BIAS), rp.slot, 0))
    }

    // ---- descriptors ----

    /// The bytes per element of the buffer view `src` names when the program
    /// picks it from the view heap at a computed index: read from the host's
    /// table by heap slot (`emit::SLOT_STRIDES`).
    fn slot_stride(&mut self, src: &Source) -> Result<Option<String>, Why> {
        let Source::Loaded { load, first } = src else { return Ok(None) };
        if !self.heap_buffer(src)? {
            return Ok(None);
        }
        let slot = self.heap_index(*load, *first, false)?;
        let k = crate::emit::SLOT_STRIDES;
        self.root_srvs.insert(k);
        self.globals.insert(format!("ByteAddressBuffer slot_strides : register(t{k}, space1);"));
        let t = self.fresh("s", "uint");
        self.line(&format!("{t} = slot_strides.Load(min({slot}, {}u) << 2);", crate::emit::SLOT_STRIDES_COUNT - 1));
        Ok(Some(t))
    }

    /// The heap index of the descriptor at dword `first` of the scalar load
    /// at `load`.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<String, Why> {
        let a = *self.scalar.get(&load).ok_or("an untracked descriptor load")?;
        match self.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.p.blocks[load.0].insts[load.1];
        let Fields::Smem { base, offset, soffset, .. } = &li.fields else { return Err("a descriptor load") };
        let Operand::Scalar { code, .. } = *base else { return Err("a descriptor load") };
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.u(base)?;
        let k = (*offset as u32).wrapping_add(4 * u32::from(first));
        if k != 0 {
            addr = format!("({addr} + {})", lit(k));
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = format!("({addr} + {so})");
        }
        Ok(format!("({addr} >> {}u)", if sampler { 4 } else { 5 }))
    }

    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<String, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else { return Err("a descriptor chosen per lane") };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self.st.masks.get(mask).ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m)?
                } else {
                    match self.get(*mask)? {
                        Val::B(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let (x, y) = (self.lane_heap_index(a, sampler)?, self.lane_heap_index(b, sampler)?);
                Ok(format!("({cond} ? {x} : {y})"))
            }
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            _ => Err("a per-lane descriptor from a join"),
        }
    }

    /// The heap index a descriptor chosen by control flow picks: each
    /// incoming edge sets it (a predecessor's own choice, itself a join, gives
    /// that join's variable, at most `depth` joins deep).
    fn descriptor_phi(&mut self, block: usize, args: &[Source], sampler: bool, depth: u32) -> Result<String, Why> {
        const WHY: Why = "a descriptor chosen by control flow";
        if let Some(n) = self.desc_phis.get(&(block, sampler)) {
            return Ok(n.clone());
        }
        let n = self.fresh_global("d", "uint");
        let preds = self.p.blocks[block].preds.clone();
        for (&q, arg) in preds.iter().zip(args) {
            if self.rpo[q] == usize::MAX {
                continue;
            }
            // (an edge not lowered yet, a loop's back edge: no value there)
            if self.rpo[q] >= self.rpo[block] || !self.edges.contains_key(&(q, block)) {
                return Err(WHY);
            }
            let index = match arg {
                Source::Loaded { load, first } => self.heap_index(*load, *first, sampler)?,
                Source::Phi { block: inner, args: inner_args } if depth > 0 => {
                    self.descriptor_phi(*inner, inner_args, sampler, depth - 1)?
                }
                _ => return Err(WHY),
            };
            let e = self.edges.get_mut(&(q, block)).expect("checked");
            e.push_str(&format!("{n} = {index};\n"));
        }
        self.desc_phis.insert((block, sampler), n.clone());
        Ok(n)
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address over the descriptor
    /// size (as `emit::lower::fetched_index`).
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<String, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self.scalar.get(&load).ok_or("an untracked descriptor load")?;
        match self.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.p.blocks[load.0].insts[load.1];
        let Fields::Flat { addr: Some(Operand::Vector { reg, .. }), saddr, offset, .. } = li.fields else {
            return Err(WHY);
        };
        let mut at = self.u(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.u(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = format!("({base} + {at})");
        }
        let k = (offset as i32 as u32).wrapping_add(4 * u32::from(first));
        Ok(format!("(({at} + {}) >> {}u)", lit(k), if sampler { 4 } else { 5 }))
    }

    /// A local variable of type `ty` for the descriptor `src` read at
    /// `site` (resources: `CB` is a constant buffer of 4,096 rows).
    fn handle(&mut self, site: Site, src: &Source, ty: &str, sampler: bool) -> Result<String, Why> {
        let _ = site;
        let (index, nonuniform) = match src {
            Source::Loaded { load, first } => {
                (self.heap_index(*load, *first, sampler)?, self.res.waterfall.loads.contains(load))
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words.get(..4).and_then(|w| w.try_into().ok()).ok_or("a sampler constant")?;
                let n = match self.samplers.iter().position(|w| *w == words) {
                    Some(n) => n,
                    None => {
                        self.samplers.push(words);
                        self.samplers.len() - 1
                    }
                };
                if ty == "SamplerComparisonState" {
                    self.cmp_samplers |= 1 << n;
                    self.globals.insert(format!("SamplerComparisonState s{n}c : register(s{n}, space0);"));
                    return Ok(format!("s{n}c"));
                }
                self.globals.insert(format!("SamplerState s{n} : register(s{n}, space0);"));
                return Ok(format!("s{n}"));
            }
            Source::Constant(_) => return Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self.native.layout.get(address.param).ok_or("a descriptor built around a table in the code")?;
                let kind = self.native.used.iter().find(|u| u.user_data == rp.user_data).map(|u| u.kind);
                let slot = rp.slot;
                return match (rp.root, kind, ty) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, "CB") => {
                        self.root_cbvs.insert(slot);
                        self.globals.insert(format!("ConstantBuffer<CB> rcb{slot} : register(b{slot}, space1);"));
                        Ok(format!("rcb{slot}"))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), "ByteAddressBuffer") => {
                        self.root_srvs.insert(slot);
                        self.globals.insert(format!("ByteAddressBuffer rsrv{slot} : register(t{slot}, space1);"));
                        Ok(format!("rsrv{slot}"))
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav), "ByteAddressBuffer" | "RWByteAddressBuffer") => {
                        self.root_uavs.insert(slot);
                        self.globals.insert(format!("RWByteAddressBuffer ruav{slot} : register(u{slot}, space1);"));
                        Ok(format!("ruav{slot}"))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), "RWByteAddressBuffer") => Err("a write through a root SRV"),
                    _ => Err("a typed root view"),
                };
            }
            Source::Select { scc, args } => {
                let [Source::Loaded { load: a, first: fa }, Source::Loaded { load: b, first: fb }] = args[..] else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::B(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let (x, y) = (self.heap_index(a, fa, sampler)?, self.heap_index(b, fb, sampler)?);
                (format!("({cond} ? {x} : {y})"), false)
            }
            Source::Phi { block, args } => (self.descriptor_phi(*block, args, sampler, 8)?, false),
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => (self.lane_heap_index(src, sampler)?, true),
        };
        let limit = crate::emit::view_heap_limit();
        let index = if sampler {
            format!("min({index}, 2047u)")
        } else if limit != 0 {
            format!("min({index}, {}u)", limit)
        } else {
            index
        };
        let index = if nonuniform { format!("NonUniformResourceIndex({index})") } else { index };
        let heap = if sampler { "SamplerDescriptorHeap" } else { "ResourceDescriptorHeap" };
        let ty = if ty == "CB" { "ConstantBuffer<CB>" } else { ty };
        let n = self.fresh("r", "");
        self.decls.remove(&n);
        self.line(&format!("{ty} {n} = {heap}[{index}];"));
        Ok(n)
    }

    // ---- vector ALU ----

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let v = Self::vf(inst);
        let name = inst.name();
        if self.res.waterfall.compares.contains(&(b, i)) {
            self.lane_bits_for(inst, "true".into());
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() { Err("a compare with DPP") } else { self.compare(inst, rest) };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        let mut src0 = None;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(format!("({g} && {e})")),
                (g, e) => g.or(e),
            };
            src0 = Some(x);
        }
        self.valu_op(inst, &guard, &src0)
    }

    fn lane_index(&mut self) -> String {
        "WaveGetLaneIndex()".into()
    }

    /// DPP: src0 as read from another lane, and which lanes write.
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(String, Option<String>), Why> {
        let v = Self::vf(inst);
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else { return Err("a DPP operand") };
        let x = self.u(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        let t = self.fresh("t", "uint");
        self.line(&format!("{t} = {x};"));
        match dpp {
            Dpp::Row { ctrl: ctrl @ 0..=0xFF, row_mask: 0xF, bank_mask: 0xF, .. } => {
                // Quad permutation: inside the quad, which pixel layouts keep.
                let picks: Vec<u32> = (0..4).map(|k| u32::from(ctrl) >> (2 * k) & 3).collect();
                let q = self.fresh("q", "uint");
                self.line(&format!("{q} = WaveGetLaneIndex() & 3u;"));
                let r = self.fresh("t", "uint");
                self.line(&format!(
                    "{r} = {q} == 0u ? QuadReadLaneAt({t}, {}u) : {q} == 1u ? QuadReadLaneAt({t}, {}u) : {q} == 2u ? QuadReadLaneAt({t}, {}u) : QuadReadLaneAt({t}, {}u);",
                    picks[0], picks[1], picks[2], picks[3]
                ));
                Ok((r, None))
            }
            Dpp::Row { ctrl, row_mask: 0xF, bank_mask: 0xF, bound_ctrl, .. } => {
                self.cross_lane("a DPP row operation");
                let n = u32::from(ctrl & 15);
                let lane = self.lane_index();
                let (src, inside) = match ctrl {
                    0x101..=0x10F => (format!("({lane} + {n}u)"), Some(format!("(({lane} & 15u) + {n}u < 16u)"))),
                    0x111..=0x11F => (format!("({lane} - {n}u)"), Some(format!("(({lane} & 15u) >= {n}u)"))),
                    0x121..=0x12F => (format!("(({lane} & ~15u) | ((({lane} & 15u) - {n}u) & 15u))"), None),
                    0x140 => (format!("({lane} ^ 15u)"), None),
                    0x141 => (format!("({lane} ^ 7u)"), None),
                    0x150..=0x15F => (format!("(({lane} & ~15u) | {n}u)"), None),
                    0x160..=0x16F => (format!("({lane} ^ {n}u)"), None),
                    _ => return Err("a DPP lane pattern"),
                };
                let r = self.fresh("t", "uint");
                self.line(&format!("{r} = WaveReadLaneAt({t}, {src});"));
                match inside {
                    Some(c) if bound_ctrl => Ok((format!("({c} ? {r} : 0u)"), None)),
                    Some(c) => Ok((r, Some(c))),
                    None => Ok((r, None)),
                }
            }
            Dpp::Lanes { sel, .. } => {
                self.cross_lane("DPP8");
                let lane = self.lane_index();
                let r = self.fresh("t", "uint");
                self.line(&format!(
                    "{r} = WaveReadLaneAt({t}, ({lane} & ~7u) | (({}u >> (({lane} & 7u) * 3u)) & 7u));",
                    sel
                ));
                Ok((r, None))
            }
            _ => Err("DPP row or bank masks"),
        }
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: String) {
        let v = Self::vf(inst);
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            let n = format!("l{}", lo.0);
            self.decls.insert(n.clone(), "bool");
            self.line(&format!("{n} = {r};"));
            self.lane_bits.insert(lo, n);
        }
        for &(_, w) in &inst.writes {
            self.vals.entry(w).or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn carry_out(&mut self, inst: &Inst, carry: String) {
        let v = Self::vf(inst);
        if let Some(Operand::Scalar { code, .. }) = v.carry
            && let Some(w) = inst.write(Reg::S(code))
        {
            let n = format!("l{}", w.0);
            self.decls.insert(n.clone(), "bool");
            self.line(&format!("{n} = {carry};"));
            self.lane_bits.insert(w, n);
        }
        for &(_, w) in &inst.writes {
            self.vals.entry(w).or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let none = None;
        let r = match ty {
            "f32" if op == "class" => {
                let x = self.fsrc(inst, 0, &none)?;
                let m = self.src(inst, 1, &none)?;
                format!("hw_class(asuint({x}), {m})")
            }
            "f32" => {
                let (a, b) = (self.fsrc(inst, 0, &none)?, self.fsrc(inst, 1, &none)?);
                match op {
                    "f" => "false".into(),
                    "tru" => "true".into(),
                    "lt" => format!("({a} < {b})"),
                    "eq" => format!("({a} == {b})"),
                    "le" => format!("({a} <= {b})"),
                    "gt" => format!("({a} > {b})"),
                    "ge" => format!("({a} >= {b})"),
                    "lg" => format!("({a} < {b} || {a} > {b})"),
                    "o" => format!("(!hw_isnan({a}) && !hw_isnan({b}))"),
                    "u" => format!("(hw_isnan({a}) || hw_isnan({b}))"),
                    "nge" => format!("!({a} >= {b})"),
                    "nlg" => format!("!({a} < {b} || {a} > {b})"),
                    "ngt" => format!("!({a} > {b})"),
                    "nle" => format!("!({a} <= {b})"),
                    "neq" => format!("!({a} == {b})"),
                    "nlt" => format!("!({a} < {b})"),
                    _ => return Err(inst.name()),
                }
            }
            "u32" | "i32" => {
                if op == "f" || op == "t" {
                    self.lane_bits_for(inst, (op == "t").to_string());
                    return Ok(());
                }
                let (a, b) = (self.src(inst, 0, &none)?, self.src(inst, 1, &none)?);
                let (a, b) = if ty == "i32" { (format!("int({a})"), format!("int({b})")) } else { (a, b) };
                let o = match op {
                    "eq" => "==",
                    "ne" | "lg" => "!=",
                    "lt" => "<",
                    "le" => "<=",
                    "gt" => ">",
                    "ge" => ">=",
                    _ => return Err("an integer compare"),
                };
                format!("({a} {o} {b})")
            }
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let (a, b) = if ty == "i64" { (format!("int64_t({a})"), format!("int64_t({b})")) } else { (a, b) };
                let o = match op {
                    "eq" => "==",
                    "ne" | "lg" => "!=",
                    "lt" => "<",
                    "le" => "<=",
                    "gt" => ">",
                    "ge" => ">=",
                    _ => return Err("a 64-bit compare"),
                };
                format!("({a} {o} {b})")
            }
            "f16" | "u16" | "i16" => {
                let (a, b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                if ty == "f16" {
                    let v = Self::vf(inst);
                    let f = |x: &str, k: u8| {
                        let mut f = format!("f16tof32({x})");
                        if v.abs >> k & 1 != 0 {
                            f = format!("abs({f})");
                        }
                        if v.neg >> k & 1 != 0 {
                            f = format!("(-{f})");
                        }
                        f
                    };
                    let (a, b) = (f(&a, 0), f(&b, 1));
                    match op {
                        "lt" => format!("({a} < {b})"),
                        "eq" => format!("({a} == {b})"),
                        "le" => format!("({a} <= {b})"),
                        "gt" => format!("({a} > {b})"),
                        "ge" => format!("({a} >= {b})"),
                        "lg" => format!("({a} < {b} || {a} > {b})"),
                        "neq" => format!("!({a} == {b})"),
                        "nlt" => format!("!({a} < {b})"),
                        "nge" => format!("!({a} >= {b})"),
                        "ngt" => format!("!({a} > {b})"),
                        "nle" => format!("!({a} <= {b})"),
                        _ => return Err("a 16-bit float compare"),
                    }
                } else {
                    let (a, b) = if ty == "i16" {
                        (format!("int({a} << 16) >> 16"), format!("int({b} << 16) >> 16"))
                    } else {
                        (a, b)
                    };
                    let o = match op {
                        "eq" => "==",
                        "ne" | "lg" => "!=",
                        "lt" => "<",
                        "le" => "<=",
                        "gt" => ">",
                        "ge" => ">=",
                        _ => return Err("a 16-bit compare"),
                    };
                    format!("(({a}) {o} ({b}))")
                }
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    /// The low (op_sel: high) half of source `k`, zero-extended.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<String, Why> {
        let v = Self::vf(inst);
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let x = self.f16_operand(inst, &op)?;
        Ok(if v.opsel >> k & 1 != 0 { format!("({x} >> 16)") } else { format!("({x} & 0xffffu)") })
    }

    fn src64(&mut self, inst: &Inst, k: usize) -> Result<String, Why> {
        let v = Self::vf(inst);
        match v.src.get(k) {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let lo = self.u(inst.read(Reg::S(code)).ok_or("an unread operand")?)?;
                let hi = self.u(inst.read(Reg::S(code + 1)).ok_or("an unread operand")?)?;
                Ok(format!("(uint64_t({lo}) | (uint64_t({hi}) << 32))"))
            }
            Some(&Operand::Vector { reg, .. }) => {
                let lo = self.u(inst.read(Reg::V(reg)).ok_or("an unread operand")?)?;
                let hi = self.u(inst.read(Reg::V(reg + 1)).ok_or("an unread operand")?)?;
                Ok(format!("(uint64_t({lo}) | (uint64_t({hi}) << 32))"))
            }
            Some(Operand::Const(c)) => {
                let x = inline(*c).ok_or("a special operand")?;
                Ok(format!("uint64_t(int64_t(int({})))", lit(x)))
            }
            Some(Operand::Literal(x)) => Ok(format!("uint64_t({})", lit(*x))),
            _ => Err("a 64-bit operand"),
        }
    }

    fn mask_bool(&mut self, inst: &Inst, k: usize) -> Result<String, Why> {
        let v = Self::vf(inst);
        let m = match v.src.get(k) {
            Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
            _ => None,
        };
        let b = m.and_then(|m| self.st.masks.get(&m).copied()).ok_or("a select on an untracked mask")?;
        self.boolean(b)
    }

    fn valu_op(&mut self, inst: &Inst, guard: &Option<String>, src0: &Option<String>) -> Result<(), Why> {
        let v = Self::vf(inst);
        let name = inst.name();
        let f = |l: &mut Self, k| l.fsrc(inst, k, src0);
        let s = |l: &mut Self, k| l.src(inst, k, src0);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        let fl = |l: &mut Self, r: String| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: String| l.write_i(inst, r, guard);
        match name {
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return fl(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                fl(self, format!("({a} + {b})"))
            }
            "v_sub_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                fl(self, format!("({a} - {b})"))
            }
            "v_subrev_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                fl(self, format!("({b} - {a})"))
            }
            "v_mul_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                fl(self, format!("({a} * {b})"))
            }
            "v_mul_legacy_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                fl(self, format!("hw_mul_legacy({a}, {b})"))
            }
            "v_max_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                fl(self, format!("hw_max({a}, {b})"))
            }
            "v_min_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                fl(self, format!("hw_min({a}, {b})"))
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                fl(self, format!("mad({a}, {b}, {c})"))
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" => {
                let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                fl(self, format!("({a} * {b} + {c})"))
            }
            "v_mad_legacy_f32" => {
                let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                fl(self, format!("(hw_mul_legacy({a}, {b}) + {c})"))
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("an accumulator") };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = flt(&self.u(acc)?);
                let r = match name {
                    "v_fmac_f32" => format!("mad({a}, {b}, {acc})"),
                    "v_mac_f32" => format!("({a} * {b} + {acc})"),
                    _ => format!("(hw_mul_legacy({a}, {b}) + {acc})"),
                };
                fl(self, r)
            }
            "v_max3_f32" => {
                let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                fl(self, format!("hw_max(hw_max({a}, {b}), {c})"))
            }
            "v_min3_f32" => {
                let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                fl(self, format!("hw_min(hw_min({a}, {b}), {c})"))
            }
            "v_med3_f32" => {
                let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                fl(self, format!("hw_med3({a}, {b}, {c})"))
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("(1.0f / {a})"))
            }
            "v_rsq_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("rsqrt({a})"))
            }
            "v_sqrt_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("sqrt({a})"))
            }
            "v_log_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("log2({a})"))
            }
            "v_exp_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("exp2({a})"))
            }
            "v_fract_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("hw_fract({a})"))
            }
            "v_trunc_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("trunc({a})"))
            }
            "v_floor_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("floor({a})"))
            }
            "v_ceil_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("ceil({a})"))
            }
            "v_rndne_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("round({a})"))
            }
            "v_sin_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("hw_sin({a})"))
            }
            "v_cos_f32" => {
                let a = f(self, 0)?;
                fl(self, format!("hw_cos({a})"))
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                fl(self, format!("hw_{}({a}, {b}, {c})", &name[2..8]))
            }
            "v_cvt_i32_f32" => {
                let a = f(self, 0)?;
                int(self, format!("hw_cvt_i32({a})"))
            }
            "v_cvt_u32_f32" => {
                let a = f(self, 0)?;
                int(self, format!("hw_cvt_u32({a})"))
            }
            "v_cvt_flr_i32_f32" => {
                let a = f(self, 0)?;
                int(self, format!("hw_cvt_i32(floor({a}))"))
            }
            "v_cvt_rpi_i32_f32" => {
                let a = f(self, 0)?;
                int(self, format!("hw_cvt_i32(floor({a} + 0.5f))"))
            }
            "v_cvt_f32_i32" => {
                let a = s(self, 0)?;
                fl(self, format!("float(int({a}))"))
            }
            "v_cvt_f32_u32" => {
                let a = s(self, 0)?;
                fl(self, format!("float({a})"))
            }
            "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1" | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let a = s(self, 0)?;
                let n = 8 * u32::from(name.as_bytes()[15] - b'0');
                fl(self, format!("float(({a} >> {n}u) & 0xffu)"))
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                fl(self, format!("f16tof32({a})"))
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                let a = f(self, 0)?;
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a 16-bit result outside a VGPR") };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.u(o)?,
                    None => "0u".into(),
                };
                int(self, format!("(({old} & 0xffff0000u) | (f32tof16({a}) & 0xffffu))"))
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                int(self, format!("(hw_f16_rtz({a}) | (hw_f16_rtz({b}) << 16))"))
            }
            "v_cvt_pk_u8_f32" => {
                let a = f(self, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                int(
                    self,
                    format!(
                        "(({d} & ~(0xffu << (({b} & 3u) * 8u))) | (uint(min(max(round({a}), 0.0f), 255.0f)) << (({b} & 3u) * 8u)))"
                    ),
                )
            }
            "v_pack_b32_f16" => {
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = *v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.f16_operand(inst, &op)?;
                    if v.opsel >> k & 1 != 0 {
                        x = format!("({x} >> 16)");
                    }
                    x = format!("({x} & {})", if v.abs >> k & 1 != 0 { "0x7fffu" } else { "0xffffu" });
                    if v.neg >> k & 1 != 0 {
                        x = format!("({x} ^ 0x8000u)");
                    }
                    halves.push(x);
                }
                int(self, format!("({} | ({} << 16))", halves[0], halves[1]))
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                let mut xs = Vec::new();
                for k in 0..3 {
                    let op = *v.src.get(k).ok_or("a missing operand")?;
                    let mut x = if v.opsel_hi >> k & 1 != 0 {
                        let x = self.f16_operand(inst, &op)?;
                        let h = if v.opsel >> k & 1 != 0 { format!("({x} >> 16)") } else { x };
                        format!("f16tof32({h})")
                    } else {
                        flt(&self.operand(inst, &op)?)
                    };
                    if v.neg_hi >> k & 1 != 0 {
                        x = format!("abs({x})");
                    }
                    if v.neg >> k & 1 != 0 {
                        x = format!("(-{x})");
                    }
                    xs.push(x);
                }
                let r = format!("mad({}, {}, {})", xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return fl(self, r);
                }
                let r = if v.clamp { format!("saturate({r})") } else { r };
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a mixed-precision result") };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.u(o)?,
                    None => "0u".into(),
                };
                let x = if name == "v_fma_mixhi_f16" {
                    format!("(({old} & 0xffffu) | (f32tof16({r}) << 16))")
                } else {
                    format!("(({old} & 0xffff0000u) | (f32tof16({r}) & 0xffffu))")
                };
                self.write(inst, Reg::V(reg), x, guard)
            }
            "v_cndmask_b32" => {
                let cond = self.mask_bool(inst, 2)?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, b) = (f(self, 0)?, f(self, 1)?);
                    return fl(self, format!("({cond} ? {b} : {a})"));
                }
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                int(self, format!("({cond} ? {b} : {a})"))
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                let (mut a, mut b) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut b);
                }
                let cin = if name.ends_with("_ci_u32") { Some(self.mask_bool(inst, 2)?) } else { None };
                let (r, carry) = self.add_sub_carry(name.starts_with("v_add"), &a, &b, cin.as_deref());
                int(self, r)?;
                self.carry_out(inst, carry);
                Ok(())
            }
            "v_mad_u64_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.src64(inst, 2)?;
                let t = self.fresh("q", "uint64_t");
                self.line(&format!("{t} = uint64_t({a}) * uint64_t({b}) + {x};"));
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a 64-bit result") };
                self.write(inst, Reg::V(reg), format!("uint({t})"), guard)?;
                self.write(inst, Reg::V(reg + 1), format!("uint({t} >> 32)"), guard)?;
                self.carry_out(inst, format!("({t} < {x})"));
                Ok(())
            }
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let t = self.fresh("q", "uint64_t");
                let e = match name {
                    "v_lshlrev_b64" => format!("{x} << ({n} & 63u)"),
                    "v_lshrrev_b64" => format!("{x} >> ({n} & 63u)"),
                    _ => format!("uint64_t(int64_t({x}) >> ({n} & 63u))"),
                };
                self.line(&format!("{t} = {e};"));
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a 64-bit result") };
                self.write(inst, Reg::V(reg), format!("uint({t})"), guard)?;
                self.write(inst, Reg::V(reg + 1), format!("uint({t} >> 32)"), guard)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                if inst.writes.first().is_some_and(|w| self.res.waterfall.reads.contains(&w.1)) {
                    // A waterfall loop's read of a descriptor dword: the lane's own.
                    return int(self, a);
                }
                let first = match guard {
                    Some(g) => format!("WaveReadLaneAt({a}, firstbitlow(WaveActiveBallot({g}).x))"),
                    None => format!("WaveReadLaneFirst({a})"),
                };
                self.cross_lane("a first-lane read");
                let r = if self.pixel() { format!("(IsHelperLane() ? {a} : {first})") } else { first };
                let t = self.fresh("t", "uint");
                self.line(&format!("{t} = {r};"));
                int(self, t)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the VGPRs the
                // instruction names (a small array kept in VGPRs).
                let m0 = self.u(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (Some(&Operand::Vector { reg: sreg, .. }), Some(Operand::Vector { reg: dreg, .. })) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let value = if name != "v_movreld_b32" {
                    let t = self.fresh("t", "uint");
                    self.line(&format!("{t} = 0u;"));
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else { break };
                        let x = self.u(x)?;
                        self.line(&format!("if ({m0} == {k}u) {t} = {x};"));
                    }
                    t
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    if inst.write(Reg::V(dreg + k)).is_none() {
                        break;
                    }
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.u(o)?,
                        None => "0u".into(),
                    };
                    self.write(inst, Reg::V(dreg + k), format!("({m0} == {k}u ? {value} : {old})"), guard)?;
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (Some(&Operand::Vector { reg: sreg, .. }), Some(Operand::Vector { reg: dreg, .. })) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.u(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.u(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                let (ta, tb) = (self.fresh("t", "uint"), self.fresh("t", "uint"));
                self.line(&format!("{ta} = {a}; {tb} = {b};"));
                self.write(inst, Reg::V(dreg), tb, guard)?;
                if dreg != sreg {
                    self.write(inst, Reg::V(sreg), ta, guard)?;
                }
                Ok(())
            }
            "v_readlane_b32" => {
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                self.cross_lane("a lane read");
                let t = self.fresh("t", "uint");
                self.line(&format!("{t} = WaveReadLaneAt({a}, {lane});"));
                int(self, t)
            }
            "v_writelane_b32" => {
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a lane write") };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.u(o)?,
                    None => "0u".into(),
                };
                self.cross_lane("a lane write");
                int(self, format!("(WaveGetLaneIndex() == {lane} ? {a} : {old})"))
            }
            "v_permlane16_b32" | "v_permlanex16_b32" => {
                let x = s(self, 0)?;
                let (s1, s2) = (s(self, 1)?, s(self, 2)?);
                let lane = self.lane_index();
                let row = if name == "v_permlanex16_b32" {
                    format!("(({lane} & ~31u) | (({lane} & 16u) ^ 16u))")
                } else {
                    format!("({lane} & ~15u)")
                };
                let pick = format!("(((({lane} & 15u) >= 8u ? {s2} : {s1}) >> (({lane} & 7u) * 4u)) & 15u)");
                let t = self.fresh("t", "uint");
                self.line(&format!("{t} = WaveReadLaneAt({x}, {row} | {pick});"));
                int(self, t)
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus
                // src1: the lane's host index with every lane.
                let op0 = *v.src.first().ok_or("a missing operand")?;
                let m = match op0 {
                    Operand::Const(193) => "0xffffffffu".to_owned(),
                    Operand::Scalar { code, .. } if code != NULL => {
                        let mv = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                        if let Some(&low) = self.st.highs.get(&mv) {
                            self.mask_data(low)?.1
                        } else if self.st.masks.contains_key(&mv) {
                            self.mask_data(mv)?.0
                        } else {
                            return Err("v_mbcnt of a mask that is no ballot");
                        }
                    }
                    _ => return Err("v_mbcnt of a mask that is no ballot"),
                };
                let b = s(self, 1)?;
                let lane = if self.mesh.is_some() {
                    // Mesh lanes are the console's: the thread index in the group.
                    if m != "0xffffffffu" {
                        return Err("v_mbcnt of a lane mask in a mesh program");
                    }
                    self.system.insert("gi", "uint gi : SV_GroupIndex".into());
                    format!("(gi & {}u)", if self.native.wave64 { 63 } else { 31 })
                } else {
                    self.cross_lane("v_mbcnt");
                    "WaveGetLaneIndex()".to_owned()
                };
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    lane
                } else {
                    format!("({lane} < 32u ? 0u : {lane} - 32u)")
                };
                int(self, format!("(countbits({m} & ({below} >= 32u ? 0xffffffffu : ((1u << {below}) - 1u))) + {b})"))
            }
            _ => {
                let r = self.integer(inst, name, src0)?;
                int(self, r)
            }
        }
    }

    fn integer(&mut self, inst: &Inst, name: &'static str, src0: &Option<String>) -> Result<String, Why> {
        let s = |l: &mut Self, k| l.src(inst, k, src0);
        Ok(match name {
            "v_and_b32" => format!("({} & {})", s(self, 0)?, s(self, 1)?),
            "v_or_b32" => format!("({} | {})", s(self, 0)?, s(self, 1)?),
            "v_xor_b32" => format!("({} ^ {})", s(self, 0)?, s(self, 1)?),
            "v_xnor_b32" => format!("~({} ^ {})", s(self, 0)?, s(self, 1)?),
            "v_not_b32" => format!("~{}", s(self, 0)?),
            "v_add_nc_u32" => format!("({} + {})", s(self, 0)?, s(self, 1)?),
            "v_sub_nc_u32" => format!("({} - {})", s(self, 0)?, s(self, 1)?),
            "v_subrev_nc_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                format!("({b} - {a})")
            }
            "v_mul_lo_u32" => format!("({} * {})", s(self, 0)?, s(self, 1)?),
            "v_lshlrev_b32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                format!("({a} << ({sh} & 31u))")
            }
            "v_lshrrev_b32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                format!("({a} >> ({sh} & 31u))")
            }
            "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                format!("uint(int({a}) >> ({sh} & 31u))")
            }
            "v_lshl_add_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                format!("(({a} << ({b} & 31u)) + {d})")
            }
            "v_lshl_or_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                format!("(({a} << ({b} & 31u)) | {d})")
            }
            "v_add_lshl_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                format!("(({a} + {b}) << ({d} & 31u))")
            }
            "v_and_or_b32" => format!("(({} & {}) | {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_or3_b32" => format!("({} | {} | {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_xor3_b32" => format!("({} ^ {} ^ {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_xad_u32" => format!("(({} ^ {}) + {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_add3_u32" => format!("({} + {} + {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_mul_u32_u24" => format!("(({} & 0xffffffu) * ({} & 0xffffffu))", s(self, 0)?, s(self, 1)?),
            "v_mul_i32_i24" => format!("uint((int({} << 8) >> 8) * (int({} << 8) >> 8))", s(self, 0)?, s(self, 1)?),
            "v_mad_u32_u24" => format!("(({} & 0xffffffu) * ({} & 0xffffffu) + {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_mad_i32_i24" => {
                format!("(uint((int({} << 8) >> 8) * (int({} << 8) >> 8)) + {})", s(self, 0)?, s(self, 1)?, s(self, 2)?)
            }
            "v_bfe_u32" => format!("hw_bfe_u({}, {}, {} & 31u)", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_bfe_i32" => format!("hw_bfe_i({}, {}, {} & 31u)", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                format!("(({m} & {a}) | (~{m} & {b}))")
            }
            "v_max_i32" => format!("uint(max(int({}), int({})))", s(self, 0)?, s(self, 1)?),
            "v_min_i32" => format!("uint(min(int({}), int({})))", s(self, 0)?, s(self, 1)?),
            "v_max_u32" => format!("max({}, {})", s(self, 0)?, s(self, 1)?),
            "v_min_u32" => format!("min({}, {})", s(self, 0)?, s(self, 1)?),
            "v_mul_hi_u32" => format!("hw_mul_hi_u({}, {})", s(self, 0)?, s(self, 1)?),
            "v_mul_hi_i32" => format!("hw_mul_hi_i({}, {})", s(self, 0)?, s(self, 1)?),
            "v_alignbit_b32" => format!("hw_alignbit({}, {}, {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_alignbyte_b32" => format!("hw_alignbit({}, {}, ({} & 3u) * 8u)", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_bfrev_b32" => format!("reversebits({})", s(self, 0)?),
            "v_bcnt_u32_b32" => format!("(countbits({}) + {})", s(self, 0)?, s(self, 1)?),
            "v_ffbl_b32" => format!("firstbitlow({})", s(self, 0)?),
            "v_ffbh_u32" => {
                let a = s(self, 0)?;
                format!("({a} == 0u ? 0xffffffffu : 31u - firstbithigh({a}))")
            }
            "v_ffbh_i32" => {
                let a = s(self, 0)?;
                format!("(firstbithigh(int({a})) == 0xffffffffu ? 0xffffffffu : 31u - firstbithigh(int({a})))")
            }
            "v_min3_i32" => format!("uint(min(min(int({}), int({})), int({})))", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_max3_i32" => format!("uint(max(max(int({}), int({})), int({})))", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_min3_u32" => format!("min(min({}, {}), {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_max3_u32" => format!("max(max({}, {}), {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_med3_i32" => {
                let (a, b, c) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                format!("uint(max(min(int({a}), int({b})), min(max(int({a}), int({b})), int({c}))))")
            }
            "v_med3_u32" => {
                let (a, b, c) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                format!("max(min({a}, {b}), min(max({a}, {b}), {c}))")
            }
            "v_perm_b32" => format!("hw_perm({}, {}, {})", s(self, 0)?, s(self, 1)?, s(self, 2)?),
            "v_bfm_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                format!("(((1u << ({a} & 31u)) - 1u) << ({b} & 31u))")
            }
            _ => return Err(name),
        })
    }

    /// 16-bit VALU operations (results in the low half; the high half kept).
    /// 16-bit integer operations: on the low halves, the low 16 bits kept in
    /// the destination's half (op_sel bit 3: the high one).
    fn int16(&mut self, inst: &Inst, name: &str, guard: &Option<String>) -> Option<Result<(), Why>> {
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
        ];
        if !INT.contains(&name) {
            return None;
        }
        Some((|| {
            let v = Self::vf(inst);
            if v.sdwa.is_some() || v.dpp.is_some() || v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let (a, b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
            let sx = |x: &str| format!("(int({x} << 16) >> 16)");
            let r = match name {
                "v_add_nc_u16" => format!("({a} + {b})"),
                "v_sub_nc_u16" => format!("({a} - {b})"),
                "v_subrev_nc_u16" => format!("({b} - {a})"),
                "v_mul_lo_u16" => format!("({a} * {b})"),
                "v_lshlrev_b16" => format!("({b} << ({a} & 15u))"),
                "v_lshrrev_b16" => format!("({b} >> ({a} & 15u))"),
                "v_ashrrev_i16" => format!("uint({} >> ({a} & 15u))", sx(&b)),
                "v_max_u16" => format!("max({a}, {b})"),
                "v_min_u16" => format!("min({a}, {b})"),
                "v_max_i16" => format!("uint(max({}, {}))", sx(&a), sx(&b)),
                _ => format!("uint(min({}, {}))", sx(&a), sx(&b)),
            };
            let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a 16-bit result outside a VGPR") };
            let old = match inst.read(Reg::V(reg)) {
                Some(o) => self.u(o)?,
                None => "0u".into(),
            };
            let x = if v.opsel & 8 != 0 {
                format!("(({old} & 0xffffu) | (({r}) << 16))")
            } else {
                format!("(({old} & 0xffff0000u) | (({r}) & 0xffffu))")
            };
            self.write(inst, Reg::V(reg), x, guard)
        })())
    }

    fn valu16(&mut self, inst: &Inst, name: &str, guard: &Option<String>) -> Option<Result<(), Why>> {
        if let Some(r) = self.int16(inst, name, guard) {
            return Some(r);
        }
        let op = name.strip_suffix("_f16").filter(|_| name.starts_with("v_") && !name.starts_with("v_cmp") && !name.starts_with("v_interp"))?;
        let op = &op[2..];
        if matches!(op, "cvt_f32" | "fma_mixlo" | "fma_mixhi" | "cvt_pkrtz" | "pack_b32") || op.starts_with("cvt_") && op != "cvt_f16" {
            return None;
        }
        Some((|| {
            let v = Self::vf(inst);
            if v.sdwa.is_some() || v.dpp.is_some() {
                return Err("a 16-bit operation with SDWA or DPP");
            }
            let mut fs = Vec::new();
            for k in 0..v.src.len().min(3) {
                let h = self.half_src(inst, k)?;
                let mut f = format!("f16tof32({h})");
                if v.abs >> k & 1 != 0 {
                    f = format!("abs({f})");
                }
                if v.neg >> k & 1 != 0 {
                    f = format!("(-{f})");
                }
                fs.push(f);
            }
            let a = |k: usize| fs.get(k).cloned().ok_or("a missing operand");
            let r = match op {
                "add" => format!("({} + {})", a(0)?, a(1)?),
                "sub" => format!("({} - {})", a(0)?, a(1)?),
                "subrev" => format!("({} - {})", a(1)?, a(0)?),
                "mul" => format!("({} * {})", a(0)?, a(1)?),
                "max" => format!("hw_max({}, {})", a(0)?, a(1)?),
                "min" => format!("hw_min({}, {})", a(0)?, a(1)?),
                "fma" | "mad" => format!("mad({}, {}, {})", a(0)?, a(1)?, a(2)?),
                "fmac" | "mac" => {
                    let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("an accumulator") };
                    let acc = self.u(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                    format!("mad({}, {}, f16tof32({acc} & 0xffffu))", a(0)?, a(1)?)
                }
                "rcp" => format!("(1.0f / {})", a(0)?),
                "rsq" => format!("rsqrt({})", a(0)?),
                "sqrt" => format!("sqrt({})", a(0)?),
                "floor" => format!("floor({})", a(0)?),
                "ceil" => format!("ceil({})", a(0)?),
                "trunc" => format!("trunc({})", a(0)?),
                "rndne" => format!("round({})", a(0)?),
                "fract" => format!("hw_fract({})", a(0)?),
                "med3" => format!("hw_med3({}, {}, {})", a(0)?, a(1)?, a(2)?),
                "max3" => format!("hw_max(hw_max({}, {}), {})", a(0)?, a(1)?, a(2)?),
                "min3" => format!("hw_min(hw_min({}, {}), {})", a(0)?, a(1)?, a(2)?),
                "exp" => format!("exp2({})", a(0)?),
                "log" => format!("log2({})", a(0)?),
                "cvt_f16" => return Err("v_cvt_f16"),
                _ => return Err("a 16-bit operation"),
            };
            let r = if v.clamp { format!("saturate({r})") } else { r };
            let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a 16-bit result outside a VGPR") };
            let old = match inst.read(Reg::V(reg)) {
                Some(o) => self.u(o)?,
                None => "0u".into(),
            };
            let x = if v.opsel & 8 != 0 {
                format!("(({old} & 0xffffu) | (f32tof16({r}) << 16))")
            } else {
                format!("(({old} & 0xffff0000u) | (f32tof16({r}) & 0xffffu))")
            };
            self.write(inst, Reg::V(reg), x, guard)
        })())
    }

    // ---- interpolation ----

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            let name = inst.name();
            if name == "v_interp_p1_f32" || name == "v_interp_p1ll_f16" {
                let dst = match &inst.fields {
                    Fields::Vintrp { dst, .. } => Some(*dst),
                    Fields::Valu(v) => v.dst,
                    _ => None,
                };
                if let Some(Operand::Vector { reg, .. }) = dst {
                    return self.write(inst, Reg::V(reg), "0u".into(), &None);
                }
            }
            return Err("the first half of an interpolation");
        };
        let name = inst.name();
        let dst = match &inst.fields {
            Fields::Vintrp { dst, .. } => *dst,
            Fields::Valu(v) => {
                if v.abs != 0 || v.neg != 0 {
                    return Err("interpolation operand modifiers");
                }
                v.dst.ok_or("an interpolation")?
            }
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else { return Err("an interpolation") };
        let (attribute, mode) = match *interp {
            Interp::Interpolated { attribute, barycentric, .. } => {
                let mode = match barycentric {
                    Barycentric::PerspectiveCenter => 2,
                    Barycentric::PerspectiveCentroid => 3,
                    Barycentric::LinearCenter => 4,
                    Barycentric::LinearCentroid => 5,
                    Barycentric::PerspectiveSample => 6,
                    Barycentric::LinearSample => 7,
                    Barycentric::PerspectivePullModel => return Err("pull-model interpolation"),
                };
                (attribute, mode)
            }
            Interp::Vertex { attribute, vertex: 2 } => (attribute, 1),
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            Interp::Weighted { .. } => return Err("interpolation with computed barycentrics"),
        };
        let packed = match &self.native.io {
            StageIo::Ps { inputs, .. } => inputs.iter().find(|i| i.row == attribute.row && i.fp16).map(|i| i.pack2),
            _ => None,
        };
        let f16 = name.ends_with("_f16");
        let read = |l: &mut Self, ch: u8| -> Result<String, Why> {
            let id = l
                .inputs
                .iter()
                .position(|e| e.reg == Some(u32::from(attribute.row)) && ch >= e.col() && ch < e.col() + e.cols())
                .ok_or("an input the signature lacks")?;
            let c = ch - l.inputs[id].col();
            l.use_input(id, c, mode)
        };
        let x = match packed {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => read(self, attribute.channel)?,
            Some(pack2) => {
                // 16-bit values: one in the low half of each dword, or two per
                // dword (`packed_half`); the host's are 32-bit floats.
                let packs = if pack2 { input_packs(&self.native.io, &self.inputs) } else { Vec::new() };
                let half = |l: &mut Self, h: u8| -> Result<Option<String>, Why> {
                    let src = if pack2 {
                        packed_half(&l.inputs, &packs, attribute.row, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    match src {
                        Some((row, ch)) if row == attribute.row || pack2 => {
                            let id = l
                                .inputs
                                .iter()
                                .position(|e| e.reg == Some(u32::from(row)) && ch >= e.col() && ch < e.col() + e.cols())
                                .ok_or("an input the signature lacks")?;
                            let c = ch - l.inputs[id].col();
                            l.use_input(id, c, mode).map(Some)
                        }
                        _ => Ok(None),
                    }
                };
                let h = 2 * attribute.channel + u8::from(attribute.high);
                if name == "v_interp_p1ll_f16" {
                    half(self, h)?.ok_or("a packed input half no element holds")?
                } else if f16 {
                    let x = half(self, h)?.ok_or("a packed input half no element holds")?;
                    let old = match inst.read(Reg::V(reg)) {
                        Some(o) => self.u(o)?,
                        None => "0u".into(),
                    };
                    format!("(({old} & 0xffff0000u) | (f32tof16(asfloat({x})) & 0xffffu))")
                } else if mode == 1 {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let lo = half(self, h)?.map_or("0u".into(), |x| format!("(f32tof16(asfloat({x})) & 0xffffu)"));
                    let hi = half(self, h + 1)?.map_or("0u".into(), |x| format!("(f32tof16(asfloat({x})) << 16)"));
                    format!("({lo} | {hi})")
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        self.write(inst, Reg::V(reg), x, &None)
    }

    /// Channel `c` of pixel input `id` read with interpolation `mode`. A
    /// constant element reads its provoking vertex whatever the instruction
    /// (the hardware shades it flat); otherwise the element is declared with
    /// the mode its reads use (a different packing shows in the signature
    /// check).
    fn use_input(&mut self, id: usize, c: u8, mode: u8) -> Result<String, Why> {
        let declared = self.inputs[id].interp;
        let mode = if declared == 1 { 1 } else { mode };
        // Read at another place of the pixel than declared (center against
        // centroid, same perspective): evaluated there, the declaration
        // (and with it the signature's packing) kept.
        let perspective = |m: u8| matches!(m, 2 | 3 | 6);
        if declared > 1 && mode > 1 && mode != declared && perspective(mode) == perspective(declared) && self.inputs[id].comp == 3 {
            let e = &self.inputs[id];
            let field = format!("i.a{id}");
            let at = match mode {
                2 | 4 => format!("EvaluateAttributeSnapped({field}, int2(0, 0))"),
                3 | 5 => format!("EvaluateAttributeCentroid({field})"),
                _ => return Err("an input evaluated at a sample"),
            };
            let x = if e.cols() == 1 { at } else { format!("{at}.{}", ["x", "y", "z", "w"][usize::from(c)]) };
            let (used, m) = self.input_use[id];
            self.input_use[id] = (used | 1 << c, m.or(Some(declared)));
            return Ok(if e.native16 && self.half_io { format!("asuint(float({x}))") } else { format!("asuint({x})") });
        }
        let (used, m) = self.input_use[id];
        if m.is_some_and(|m| m != mode) {
            return Err("one input interpolated two ways");
        }
        self.input_use[id] = (used | 1 << c, Some(mode));
        Ok(self.comp_expr(id, c, true))
    }

    // ---- exports ----

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self.exports.get(&(b, i)).ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.mesh.clone();
        if let Some(g) = &guard
            && mesh.is_none()
        {
            if !(e.done && self.pixel()) {
                return Err("a conditional export");
            }
            self.line(&format!("if (!{g}) discard;"));
        }
        let primitive = matches!(e.target, Target::Primitive | Target::PrimitiveParam { .. });
        let mut body = String::new();
        if e.target == Target::Primitive {
            let m = mesh.as_ref().ok_or("a primitive export")?;
            let v = e.values[0].ok_or("a primitive export without its indices")?;
            let x = self.u(v)?;
            let written = m.max_vertices.min(m.threads);
            // Bit 31 culls; so does a vertex no lane writes.
            let _ = writeln!(
                body,
                "{{ uint p = {x}; uint3 t = uint3(p & 0x3ffu, (p >> 10) & 0x3ffu, (p >> 20) & 0x3ffu); bool cull = int(p) < 0 || any(t >= {written}u); tris[gi] = cull ? uint3(0u, 0u, 0u) : t; }}"
            );
        } else {
            for (id, col, x) in self.export_stores(e)? {
                let els = if primitive { &mesh.as_ref().ok_or("a primitive export")?.prims } else { &self.outputs };
                let o = &els[id];
                let x = match o.comp {
                    3 => flt(&x),
                    2 => format!("int({x})"),
                    _ => x,
                };
                let field = if o.cols() == 1 { String::new() } else { format!(".{}", ["x", "y", "z", "w"][usize::from(col)]) };
                let target = match (&mesh, primitive) {
                    (None, _) => format!("o.o{id}"),
                    (Some(_), false) => format!("verts[gi].o{id}"),
                    (Some(_), true) => format!("prims[gi].p{id}"),
                };
                if primitive {
                    self.prim_use[id] |= 1 << col;
                } else {
                    self.output_use[id] |= 1 << col;
                }
                let _ = writeln!(body, "{target}{field} = {x};");
            }
        }
        match &mesh {
            None => self.line(&body),
            Some(m) => {
                // A mesh lane writes its own vertex or primitive, where it exports.
                self.system.insert("gi", "uint gi : SV_GroupIndex".into());
                let limit = if primitive { m.max_primitives } else { m.max_vertices };
                let mut conds: Vec<String> = guard.into_iter().collect();
                if m.threads > limit {
                    conds.push(format!("gi < {limit}u"));
                }
                if conds.is_empty() {
                    self.line(&body);
                } else {
                    self.line(&format!("if ({}) {{\n{body}}}", conds.join(" && ")));
                }
            }
        }
        Ok(())
    }

    /// The output element, channel and value (uint bits) of each exported
    /// channel.
    fn export_stores(&mut self, e: &Export) -> Result<Vec<(usize, u8, String)>, Why> {
        let mesh = self.mesh.clone();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive { mesh.as_ref().map(|m| m.prims.clone()).unwrap_or_default() } else { self.outputs.clone() };
        let by_sv = |sv: u32| outputs.iter().position(|o| o.sv == sv);
        let param_row = |l: &Self, usage, row| match usage {
            crate::native::Usage::Other(0) if mesh.is_some() && !primitive => l.unnamed_row(),
            _ => Some(row),
        };
        let mut stores = Vec::new();
        // 16-bit vertex outputs: the halves each dword carries.
        if let Target::Param { usage, row } = e.target
            && let Some(row) = param_row(self, usage, row)
            && let Some(&(pack2, _)) = half_outputs(&self.native.io, &outputs).get(&row)
        {
            let halves = half_outputs(&self.native.io, &outputs);
            let packs: Vec<(u8, Option<u8>)> = halves.iter().filter(|(_, h)| h.0).map(|(&r, h)| (r, h.1)).collect();
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else { continue };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 { packed_half(&outputs, &packs, row, h) } else { (!high).then_some((row, ch)) };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| o.reg == Some(u32::from(r)) && c >= o.col() && c < o.col() + o.cols())
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col();
                        let x = self.u(value)?;
                        let x = if high { format!("({x} >> 16)") } else { x };
                        stores.push((id, col, format!("asuint(f16tof32({x}))")));
                    }
                }
            }
            return Ok(stores);
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else { continue };
            let found = match e.target {
                Target::Color(n) => {
                    outputs.iter().position(|o| o.sv == 64 && o.index == u32::from(n) && ch < o.cols()).map(|id| (id, ch))
                }
                Target::Position(0) => by_sv(1).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_sv(65).or(by_sv(67)).or(by_sv(68)).map(|id| (id, 0)),
                    1 => by_sv(69).map(|id| (id, 0)),
                    2 => by_sv(66).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(self, usage, row);
                    outputs
                        .iter()
                        .position(|o| row.is_some_and(|r| o.reg == Some(u32::from(r))) && ch >= o.col() && ch < o.col() + o.cols())
                        .map(|id| (id, ch - outputs[id].col()))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // Without the misc vector, pos1 and pos2 carry the clip then
                    // cull distances, in semantic order (as `emit::lower::export`).
                    if outputs.iter().any(|o| matches!(o.sv, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len()).filter(|&i| matches!(outputs[i].sv, 2 | 3)).collect();
                    dists.sort_by_key(|&i| (outputs[i].sv, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists.iter().flat_map(|&i| (0..outputs[i].cols()).map(move |c| (i, c))).collect();
                    flat.get(usize::from(n - 1) * 4 + usize::from(ch)).copied()
                }
                Target::Position(_) => return Err("point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                if outputs[id].half {
                    return Err("a 16-bit output the stage data does not mark");
                }
                stores.push((id, col, self.u(value)?));
            }
        }
        Ok(stores)
    }

    /// The per-vertex row no mesh output entry names but one (the 2020
    /// compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.native.io else { return None };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != crate::native::Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: BTreeSet<u32> = self.outputs.iter().filter(|o| o.sv == 0).filter_map(|o| o.reg).filter(|&r| !named(r)).collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    // ---- global memory ----

    /// `global_load_*` at an address traced to a root view or the code (as
    /// `emit::lower::flat`); a load of descriptors from a heap leaves its
    /// dwords without numbers (the accesses that use them find them by
    /// address). Title memory by address is refused.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else { unreachable!() };
        let name = inst.name();
        let a = *self.scalar.get(&(b, i)).ok_or("an untracked global address")?;
        if a.param == MEMORY {
            return Err("title memory by address");
        }
        let rest = name.strip_prefix("global_load_").ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1u32, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8u32, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let Some(Operand::Vector { reg, .. }) = *dst else { return Err("a global load result") };
        if a.param != CODE
            && matches!(
                self.native.layout.get(a.param).map(|p| p.root),
                Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap)
            )
        {
            for k in 0..n {
                if let Some(v) = inst.write(Reg::V(reg + k as u8)) {
                    self.vals.insert(v, Val::Missing("a descriptor's dwords read as data"));
                }
            }
            return Ok(());
        }
        let guard = self.guard(b, i)?;
        let mut off = lit(a.offset);
        if let Some(d) = a.dynamic {
            off = format!("({off} + {})", self.u(d)?);
        }
        let h = if a.param == CODE {
            None
        } else {
            if let Some(bias) = self.root_bias(a.param) {
                off = format!("({off} + {bias})");
            }
            let src = Source::Built { address: a, bits: 0, words: [0, 0] };
            Some(self.handle((b, i), &src, "ByteAddressBuffer", false)?)
        };
        let o = self.fresh("a", "uint");
        self.line(&format!("{o} = {off};"));
        let words = self.native.code.len() as u32 / 4;
        let dword = |l: &mut Self, at: String| -> String {
            match &h {
                Some(h) => format!("{h}.Load({at})"),
                None => {
                    l.code_table = Some(words);
                    format!("code_table[min(({at}) >> 2, {}u)]", words.saturating_sub(1))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => dword(self, format!("{o} + {}u", 4 * k)),
                Some((width, signed)) => {
                    let w = dword(self, format!("{o} & ~3u"));
                    let x = format!("(({w} >> (({o} & 3u) * 8u)) << {}u)", 32 - width);
                    if signed { format!("uint(int({x}) >> {}u)", 32 - width) } else { format!("({x} >> {}u)", 32 - width) }
                }
            };
            let t = self.fresh("t", "uint");
            self.line(&format!("{t} = {x};"));
            self.write(inst, Reg::V(reg + k as u8), t, &guard)?;
        }
        Ok(())
    }

    // ---- LDS ----

    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds { dst, addr, data, offset0, offset1, gds } = &inst.fields else { unreachable!() };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        if name == "ds_swizzle_b32" {
            return self.swizzle(inst, &guard);
        }
        if self.lds_dwords == 0 {
            let bytes = match &self.native.io {
                StageIo::Cs(cs) => cs.lds_bytes(),
                StageIo::Ms(ms) => ms.lds_bytes,
                _ => 0,
            };
            if bytes == 0 {
                return Err("LDS outside a compute program");
            }
            self.lds_dwords = bytes / 4;
        }
        if self.wide_waves() {
            self.approximate.insert("LDS shared between the halves of a console wave");
        }
        let a = match addr {
            Some(Operand::Vector { reg, .. }) => self.u(inst.read(Reg::V(*reg)).ok_or("an LDS address")?)?,
            _ => return Err("an LDS address"),
        };
        let off16 = u32::from(*offset0) | u32::from(*offset1) << 8;
        let last = self.lds_dwords - 1;
        let at = |x: String| format!("lds[min(({x}) >> 2, {last}u)]");
        let data_v = |l: &mut Self, k: usize, n: u8| -> Result<Vec<String>, Why> {
            let Some(Operand::Vector { reg, .. }) = data.get(k) else { return Err("LDS data") };
            (0..n).map(|j| l.u(inst.read(Reg::V(reg + j)).ok_or("LDS data")?)).collect()
        };
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        let ab = self.fresh("a", "uint");
        self.line(&format!("{ab} = {a};"));
        let begin = |l: &mut Self| {
            if let Some(g) = &guard {
                l.line(&format!("if ({g}) {{"));
            }
        };
        let end = |l: &mut Self| {
            if guard.is_some() {
                l.line("}");
            }
        };
        match name {
            "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" => {
                let n = match name { "ds_read_b32" => 1, "ds_read_b64" => 2, "ds_read_b96" => 3, _ => 4 };
                let reg = dst_reg.ok_or("an LDS result")?;
                for j in 0..n {
                    let x = at(format!("{ab} + {}u", off16 + 4 * j));
                    let t = self.fresh("t", "uint");
                    self.line(&format!("{t} = {x};"));
                    self.write(inst, Reg::V(reg + j as u8), t, &guard)?;
                }
            }
            "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
                let wide = name.ends_with("b64");
                let unit = match name { "ds_read2_b32" => 4, "ds_read2st64_b32" => 256, "ds_read2_b64" => 8, _ => 512 };
                let reg = dst_reg.ok_or("an LDS result")?;
                let per = if wide { 2 } else { 1 };
                for (k, o) in [u32::from(*offset0), u32::from(*offset1)].into_iter().enumerate() {
                    for j in 0..per {
                        let x = at(format!("{ab} + {}u", o * unit + 4 * j));
                        let t = self.fresh("t", "uint");
                        self.line(&format!("{t} = {x};"));
                        self.write(inst, Reg::V(reg + (k as u32 * per + j) as u8), t, &guard)?;
                    }
                }
            }
            "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" => {
                let n = match name { "ds_write_b32" => 1, "ds_write_b64" => 2, "ds_write_b96" => 3, _ => 4 };
                let xs = data_v(self, 0, n)?;
                begin(self);
                for (j, x) in xs.iter().enumerate() {
                    let p = at(format!("{ab} + {}u", off16 + 4 * j as u32));
                    self.line(&format!("{p} = {x};"));
                }
                end(self);
            }
            "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
                let wide = name.ends_with("b64");
                let unit = match name { "ds_write2_b32" => 4, "ds_write2st64_b32" => 256, "ds_write2_b64" => 8, _ => 512 };
                let per = if wide { 2 } else { 1 };
                let x0 = data_v(self, 0, per)?;
                let x1 = data_v(self, 1, per)?;
                begin(self);
                for (o, xs) in [(u32::from(*offset0), x0), (u32::from(*offset1), x1)] {
                    for (j, x) in xs.iter().enumerate() {
                        let p = at(format!("{ab} + {}u", o * unit + 4 * j as u32));
                        self.line(&format!("{p} = {x};"));
                    }
                }
                end(self);
            }
            "ds_add_u32" | "ds_sub_u32" | "ds_and_b32" | "ds_or_b32" | "ds_xor_b32" | "ds_min_u32" | "ds_max_u32" | "ds_min_i32"
            | "ds_max_i32" | "ds_add_rtn_u32" | "ds_sub_rtn_u32" | "ds_and_rtn_b32" | "ds_or_rtn_b32" | "ds_xor_rtn_b32"
            | "ds_min_rtn_u32" | "ds_max_rtn_u32" | "ds_wrxchg_rtn_b32" | "ds_min_rtn_i32" | "ds_max_rtn_i32" => {
                let x = data_v(self, 0, 1)?.remove(0);
                let base = name.replace("_rtn", "");
                let (f, x) = match &base[3..] {
                    "add_u32" => ("InterlockedAdd", x),
                    "sub_u32" => ("InterlockedAdd", format!("(0u - {x})")),
                    "and_b32" => ("InterlockedAnd", x),
                    "or_b32" => ("InterlockedOr", x),
                    "xor_b32" => ("InterlockedXor", x),
                    "min_u32" => ("InterlockedMin", x),
                    "max_u32" => ("InterlockedMax", x),
                    "min_i32" => ("InterlockedMin", format!("int({x})")),
                    "max_i32" => ("InterlockedMax", format!("int({x})")),
                    "wrxchg_b32" => ("InterlockedExchange", x),
                    _ => return Err(name),
                };
                let p = at(format!("{ab} + {off16}u"));
                let p = if f == "InterlockedMin" || f == "InterlockedMax" {
                    if x.starts_with("int(") { return Err("a signed LDS min or max") } else { p }
                } else {
                    p
                };
                let old = self.fresh("t", "uint");
                let keep = match dst_reg.and_then(|r| inst.read(Reg::V(r))) {
                    Some(o) if name.contains("rtn") => self.u(o)?,
                    _ => "0u".into(),
                };
                self.line(&format!("{old} = {keep};"));
                begin(self);
                self.line(&format!("{f}({p}, {x}, {old});"));
                end(self);
                if name.contains("rtn") {
                    let reg = dst_reg.ok_or("an LDS result")?;
                    self.write(inst, Reg::V(reg), old, &None)?;
                }
            }
            _ => return Err(name),
        }
        Ok(())
    }

    /// `ds_swizzle_b32`: quad permutations (offset bit 15) and bit-mask
    /// swizzles within 32 lanes.
    fn swizzle(&mut self, inst: &Inst, guard: &Option<String>) -> Result<(), Why> {
        let Fields::Ds { dst, addr, offset0, offset1, .. } = &inst.fields else { unreachable!() };
        let off = u32::from(*offset0) | u32::from(*offset1) << 8;
        let Some(Operand::Vector { reg: dreg, .. }) = dst else { return Err("a swizzle result") };
        // (the source is in the address field)
        let Some(Operand::Vector { reg, .. }) = addr else { return Err("swizzle data") };
        let x = self.u(inst.read(Reg::V(*reg)).ok_or("swizzle data")?)?;
        let t = self.fresh("t", "uint");
        self.line(&format!("{t} = {x};"));
        let r = self.fresh("t", "uint");
        if off & 0x8000 != 0 {
            let picks: Vec<u32> = (0..4).map(|k| off >> (2 * k) & 3).collect();
            let q = self.fresh("q", "uint");
            self.line(&format!("{q} = WaveGetLaneIndex() & 3u;"));
            self.line(&format!(
                "{r} = {q} == 0u ? QuadReadLaneAt({t}, {}u) : {q} == 1u ? QuadReadLaneAt({t}, {}u) : {q} == 2u ? QuadReadLaneAt({t}, {}u) : QuadReadLaneAt({t}, {}u);",
                picks[0], picks[1], picks[2], picks[3]
            ));
        } else {
            let (and, or, xor) = (off & 31, off >> 5 & 31, off >> 10 & 31);
            self.cross_lane("a lane swizzle");
            self.line(&format!(
                "{r} = WaveReadLaneAt({t}, (WaveGetLaneIndex() & ~31u) | ((((WaveGetLaneIndex() & 31u) & {and}u) | {or}u) ^ {xor}u));"
            ));
        }
        self.write(inst, Reg::V(*dreg), r, guard)
    }

    // ---- images ----

    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image { data, addr, dmask, dim, unorm, a16, d16, tfe, lwe, cache, .. } = &inst.fields else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        let dmask = if *dmask == 0 { 1 } else { *dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let sampled = name.starts_with("image_sample") || name.starts_with("image_gather") || name == "image_get_lod";
        let (tex, coords, cube, array) = match dim {
            0 => ("Texture1D", 1, false, false),
            1 => ("Texture2D", 2, false, false),
            2 => ("Texture3D", 3, false, false),
            3 if sampled || name == "image_get_resinfo" => ("TextureCube", 3, true, false),
            4 => ("Texture1DArray", 2, false, true),
            5 => ("Texture2DArray", 3, false, true),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        let addr_u = |l: &mut Self, k: usize| -> Result<String, Why> {
            let r = *regs.get(k).ok_or("an image address")?;
            l.u(inst.read(Reg::V(r)).ok_or("an image address")?)
        };
        // Coordinate `k` from address VGPR `base` on: with a16 two 16-bit
        // values per VGPR, low half first.
        let a16 = *a16;
        let at16 = |l: &mut Self, base: usize, k: usize| -> Result<String, Why> {
            if a16 {
                let x = addr_u(l, base + k / 2)?;
                Ok(if k.is_multiple_of(2) { format!("({x} & 0xffffu)") } else { format!("({x} >> 16)") })
            } else {
                addr_u(l, base + k)
            }
        };
        let cf = |x: String| if a16 { format!("f16tof32({x})") } else { flt(&x) };
        let src = self.descriptors.get(&(site, false)).copied().ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let atomic = name.strip_prefix("image_atomic_");
        let uav = store || atomic.is_some() || self.used_kind(src).is_some_and(|u| u.kind == ResourceType::Uav);
        let rw = |t: &str| format!("RW{t}<float4>");
        let comps = ["x", "y", "z", "w"];
        let vec_of = |xs: &[String]| -> String {
            match xs.len() {
                1 => xs[0].clone(),
                n => format!("uint{n}({})", xs.join(", ")),
            }
        };
        if let Some(op) = atomic {
            if cube || *d16 {
                return Err("a cube or 16-bit image atomic");
            }
            let f = match op {
                "add" => "InterlockedAdd",
                "and" => "InterlockedAnd",
                "or" => "InterlockedOr",
                "xor" => "InterlockedXor",
                "umin" => "InterlockedMin",
                "umax" => "InterlockedMax",
                "swap" => "InterlockedExchange",
                _ => return Err("an image atomic"),
            };
            // One 32-bit channel, or a uint64_t in two (an R32G32_UINT view).
            let wide = match dmask {
                1 => false,
                3 => true,
                _ => return Err("an image atomic data mask"),
            };
            let cs: Vec<String> = (0..coords).map(|k| at16(self, 0, k)).collect::<Result<_, _>>()?;
            let Operand::Vector { reg, .. } = *data else { return Err("image atomic data") };
            let lo = self.u(inst.read(Reg::V(reg)).ok_or("image atomic data")?)?;
            let (x, ty, ety) = if wide {
                let hi = self.u(inst.read(Reg::V(reg + 1)).ok_or("image atomic data")?)?;
                (format!("(uint64_t({lo}) | (uint64_t({hi}) << 32))"), "uint64_t", "uint64_t")
            } else {
                (lo.clone(), "uint", "uint")
            };
            let h = self.handle(site, src, &format!("RW{tex}<{ty}>"), false)?;
            let old = self.fresh("t", ety);
            let keep = if cache.glc { x.clone() } else { "0".into() };
            self.line(&format!("{old} = {keep};"));
            if let Some(g) = &guard {
                self.line(&format!("if ({g}) {{"));
            }
            self.line(&format!("{f}({h}[{}], {x}, {old});", vec_of(&cs)));
            if guard.is_some() {
                self.line("}");
            }
            self.writes_uavs = true;
            if cache.glc {
                if wide {
                    self.write(inst, Reg::V(reg), format!("uint({old})"), &None)?;
                    self.write(inst, Reg::V(reg + 1), format!("uint({old} >> 32)"), &None)?;
                } else {
                    self.write(inst, Reg::V(reg), old, &None)?;
                }
            }
            return Ok(());
        }
        if store {
            if name != "image_store" || cube {
                return Err("an image store to a mip level or a cube");
            }
            if dmask != 0xF {
                return Err("an image store of some channels");
            }
            let cs: Vec<String> = (0..coords).map(|k| at16(self, 0, k)).collect::<Result<_, _>>()?;
            let Operand::Vector { reg, .. } = *data else { return Err("image store data") };
            let mut vals = Vec::new();
            for k in 0..4u8 {
                let r = if *d16 { k / 2 } else { k };
                let x = self.u(inst.read(Reg::V(reg + r)).ok_or("image store data")?)?;
                vals.push(if *d16 {
                    if k % 2 == 1 { format!("f16tof32({x} >> 16)") } else { format!("f16tof32({x})") }
                } else {
                    flt(&x)
                });
            }
            let h = self.handle(site, src, &rw(tex), false)?;
            if let Some(g) = &guard {
                self.line(&format!("if ({g}) {{"));
            }
            self.line(&format!("{h}[{}] = float4({});", vec_of(&cs), vals.join(", ")));
            if guard.is_some() {
                self.line("}");
            }
            self.writes_uavs = true;
            return Ok(());
        }
        let Operand::Vector { reg, .. } = *data else { return Err("an image result") };
        let mut results: Vec<String> = Vec::new();
        if name == "image_get_resinfo" {
            let h = self.handle(site, src, &if uav { rw(tex) } else { format!("{tex}<float4>") }, false)?;
            let mip = if uav { None } else { Some(at16(self, 0, 0)?) };
            let dims = self.fresh("d", "uint4");
            let (w, hgt, dep, lv) = (format!("{dims}.x"), format!("{dims}.y"), format!("{dims}.z"), format!("{dims}.w"));
            let call = match (coords, cube, array, &mip) {
                (1, _, _, Some(m)) => format!("{h}.GetDimensions({m}, {w}, {lv}); {hgt} = 1u; {dep} = 1u;"),
                (1, _, _, None) => format!("{h}.GetDimensions({w}); {hgt} = 1u; {dep} = 1u; {lv} = 1u;"),
                (2, false, false, Some(m)) | (3, true, _, Some(m)) => format!("{h}.GetDimensions({m}, {w}, {hgt}, {lv}); {dep} = 1u;"),
                (2, false, false, None) => format!("{h}.GetDimensions({w}, {hgt}); {dep} = 1u; {lv} = 1u;"),
                (2, false, true, Some(m)) => format!("{h}.GetDimensions({m}, {w}, {hgt}, {lv}); {dep} = {hgt}; {hgt} = 1u;"),
                (_, _, _, Some(m)) => format!("{h}.GetDimensions({m}, {w}, {hgt}, {dep}, {lv});"),
                (_, _, _, None) => format!("{h}.GetDimensions({w}, {hgt}, {dep}); {lv} = 1u;"),
            };
            self.line(&call);
            for ch in 0..4 {
                if dmask >> ch & 1 != 0 {
                    results.push(format!("{dims}.{}", comps[ch]));
                }
            }
        } else if !sampled {
            // Loads.
            if cube {
                return Err("a cube load");
            }
            let cs: Vec<String> = (0..coords).map(|k| at16(self, 0, k)).collect::<Result<_, _>>()?;
            let r = self.fresh("x", "float4");
            if uav {
                if name != "image_load" {
                    return Err("a UAV load at a mip level");
                }
                let h = self.handle(site, src, &rw(tex), false)?;
                self.line(&format!("{r} = {h}[{}];", vec_of(&cs)));
            } else {
                let mip = if name == "image_load_mip" { at16(self, 0, coords)? } else { "0u".into() };
                let h = self.handle(site, src, &format!("{tex}<float4>"), false)?;
                let mut all = cs.clone();
                all.push(mip);
                self.line(&format!("{r} = {h}.Load(int{}({}));", all.len(), all.join(", ")));
            }
            for ch in 0..4 {
                if dmask >> ch & 1 != 0 {
                    results.push(format!("{r}.{}", comps[ch]));
                }
            }
        } else {
            let sname = name;
            let cmp = matches!(sname, "image_sample_c_lz" | "image_sample_c" | "image_sample_c_cl" | "image_sample_c_l");
            if cmp && dmask.count_ones() != 1 {
                return Err("a comparison sample of several channels");
            }
            if uav {
                return Err("sampling a UAV");
            }
            let ss = self.descriptors.get(&(site, true)).copied().ok_or("an untracked sampler")?;
            let s = self.handle(site, ss, if cmp { "SamplerComparisonState" } else { "SamplerState" }, true)?;
            let offset_word = sname.ends_with("_o");
            let first = usize::from(offset_word) + usize::from(sname.starts_with("image_sample_b")) + usize::from(cmp);
            let spatial = coords - usize::from(array);
            let grads = if cube { 2 } else { spatial };
            let deriv = if sname.starts_with("image_sample_d") { if a16 { grads.div_ceil(2) } else { grads } } else { 0 };
            let base = first + 2 * deriv;
            let mut cs: Vec<String> = Vec::new();
            for k in 0..coords {
                cs.push(cf(at16(self, base, k)?));
            }
            let h = self.handle(site, src, &format!("{tex}<float4>"), false)?;
            let coord = if cube {
                let d = self.fresh("c", "float3");
                self.line(&format!("{d} = hw_cube_dir({}, {}, {});", cs[0], cs[1], cs[2]));
                d
            } else {
                let mut cs = cs.clone();
                if *unorm {
                    let dims = self.fresh("d", "uint4");
                    let call = match (spatial, array) {
                        (1, false) => format!("{h}.GetDimensions(0u, {dims}.x, {dims}.w);"),
                        (1, true) => format!("{h}.GetDimensions(0u, {dims}.x, {dims}.z, {dims}.w);"),
                        (2, false) => format!("{h}.GetDimensions(0u, {dims}.x, {dims}.y, {dims}.w);"),
                        _ => format!("{h}.GetDimensions(0u, {dims}.x, {dims}.y, {dims}.z, {dims}.w);"),
                    };
                    self.line(&call);
                    for (k, c) in cs.iter_mut().enumerate().take(spatial) {
                        *c = format!("({c} / float({dims}.{}))", comps[k]);
                    }
                }
                match cs.len() {
                    1 => cs[0].clone(),
                    n => format!("float{n}({})", cs.join(", ")),
                }
            };
            let offs = if offset_word && !cube {
                let w = addr_u(self, 0)?;
                let parts: Vec<String> =
                    (0..spatial.min(3)).map(|k| format!("(int({w} << {}u) >> 26)", 26 - 8 * k)).collect();
                Some(match parts.len() {
                    1 => parts[0].clone(),
                    n => format!("int{n}({})", parts.join(", ")),
                })
            } else {
                None
            };
            let off_arg = |o: &Option<String>| o.as_ref().map_or(String::new(), |o| format!(", {o}"));
            let r = self.fresh("x", "float4");
            let lod_at = |l: &mut Self| -> Result<String, Why> { Ok(cf(at16(l, base, coords)?)) };
            let call = match sname {
                "image_sample" | "image_sample_o" if self.pixel() => format!("{h}.Sample({s}, {coord}{})", off_arg(&offs)),
                "image_sample" | "image_sample_o" => format!("{h}.SampleLevel({s}, {coord}, 0.0f{})", off_arg(&offs)),
                "image_sample_cl" if self.pixel() && cube => {
                    let cl = lod_at(self)?;
                    format!("{h}.Sample({s}, {coord}, {cl})")
                }
                "image_sample_cl" if self.pixel() => {
                    let cl = lod_at(self)?;
                    format!("{h}.Sample({s}, {coord}, {}, {cl})", offs.clone().unwrap_or(zero_off(spatial)))
                }
                "image_sample_lz" | "image_sample_lz_o" => format!("{h}.SampleLevel({s}, {coord}, 0.0f{})", off_arg(&offs)),
                "image_sample_l" | "image_sample_l_o" => {
                    let lod = lod_at(self)?;
                    format!("{h}.SampleLevel({s}, {coord}, {lod}{})", off_arg(&offs))
                }
                "image_sample_b" | "image_sample_b_o" => {
                    let bias = addr_u(self, usize::from(offset_word))?;
                    let bias = if a16 { format!("f16tof32({bias} & 0xffffu)") } else { flt(&bias) };
                    if self.pixel() {
                        format!("{h}.SampleBias({s}, {coord}, {bias}{})", off_arg(&offs))
                    } else {
                        format!("{h}.SampleLevel({s}, {coord}, {bias}{})", off_arg(&offs))
                    }
                }
                "image_sample_d" | "image_sample_d_o" if !cube => {
                    let gx: Vec<String> = (0..spatial).map(|k| at16(self, first, k).map(cf)).collect::<Result<_, _>>()?;
                    let gy: Vec<String> =
                        (0..spatial).map(|k| at16(self, first + deriv, k).map(cf)).collect::<Result<_, _>>()?;
                    let v = |g: &[String]| if g.len() == 1 { g[0].clone() } else { format!("float{}({})", g.len(), g.join(", ")) };
                    format!("{h}.SampleGrad({s}, {coord}, {}, {}{})", v(&gx), v(&gy), off_arg(&offs))
                }
                "image_sample_c_lz" | "image_sample_c" | "image_sample_c_l" => {
                    let reference = flt(&addr_u(self, first - 1)?);
                    if sname == "image_sample_c" && self.pixel() {
                        format!("{h}.SampleCmp({s}, {coord}, {reference}{}).xxxx", off_arg(&offs))
                    } else if sname == "image_sample_c_lz" {
                        format!("{h}.SampleCmpLevelZero({s}, {coord}, {reference}{}).xxxx", off_arg(&offs))
                    } else {
                        return Err("a comparison sample at a level");
                    }
                }
                "image_gather4_lz" | "image_gather4_lz_o" => {
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    let f = ["GatherRed", "GatherGreen", "GatherBlue", "GatherAlpha"][dmask.trailing_zeros() as usize];
                    format!("{h}.{f}({s}, {coord}{})", off_arg(&offs))
                }
                "image_get_lod" if self.pixel() && !cube => {
                    format!("float4({h}.CalculateLevelOfDetail({s}, {coord}), {h}.CalculateLevelOfDetailUnclamped({s}, {coord}), 0.0f, 0.0f)")
                }
                _ => return Err(sname),
            };
            self.line(&format!("{r} = {call};"));
            if sname.starts_with("image_gather") {
                results = (0..4).map(|ch| format!("{r}.{}", comps[ch])).collect();
            } else {
                for ch in 0..4 {
                    if dmask >> ch & 1 != 0 {
                        results.push(format!("{r}.{}", comps[ch]));
                    }
                }
            }
        }
        let xs: Vec<String> = if name == "image_get_resinfo" {
            results
        } else if *d16 {
            results
                .chunks(2)
                .map(|p| match p {
                    [lo, hi] => format!("(f32tof16({lo}) | (f32tof16({hi}) << 16))"),
                    [lo] => format!("f32tof16({lo})"),
                    _ => unreachable!(),
                })
                .collect()
        } else {
            results.into_iter().map(|x| format!("asuint({x})")).collect()
        };
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, &guard)?;
        }
        Ok(())
    }

    // ---- buffers ----

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer { data, addr, soffset, offset, offen, idxen, format, lds, tfe, cache, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let store = name.starts_with("buffer_store_");
        let atomic = name.strip_prefix("buffer_atomic_");
        if name.contains("byte") || name.contains("short") || name.contains("d16") {
            return Err("a byte, 16-bit or d16 buffer access");
        }
        let (typed, count) = match name
            .strip_prefix("buffer_load_")
            .or(name.strip_prefix("buffer_store_"))
            .or(name.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1u32),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else { return Err("a buffer address") };
        let vaddr = |l: &mut Self, k: u8| -> Result<String, Why> { l.u(inst.read(Reg::V(areg + k)).ok_or("a buffer address")?) };
        let mut index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = lit(u32::from(*offset));
        if *offen {
            off = format!("({off} + {})", vaddr(self, u8::from(*idxen))?);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. } | Operand::Const(128)) {
            off = format!("({off} + {})", self.operand(inst, soffset)?);
        }
        let src = self.descriptors.get(&(site, false)).copied().ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            off = format!("({off} + {})", lit(base));
        }
        if let Some(x) = extra {
            off = format!("({off} + {x})");
        }
        if self.heap_buffer(src)? && used.is_none() {
            // Picked from the view heap at a computed index: the host's
            // table gives the element stride (a raw access at its byte).
            let stride = self.slot_stride(src)?.ok_or("a buffer picked from the view heap")?;
            if let Some(element) = &index
                && !typed
            {
                off = format!("({element} * {stride} + {off})");
                index = None;
            }
        }
        if let (Some(element), Some(u)) = (&index, used)
            && !typed
            && u.kind != ResourceType::Cbv
            && u.stride >= 4
            && u.stride.is_multiple_of(4)
        {
            off = format!("({element} * {}u + {off})", u.stride);
            index = None;
        }
        if let (Some(element), Some(u)) = (&index, used)
            && !typed
            && u.stride == 1
        {
            off = format!("({element} + {off})");
            index = None;
        }
        let o = self.fresh("a", "uint");
        self.line(&format!("{o} = {off};"));
        let begin = |l: &mut Self| {
            if let Some(g) = &guard {
                l.line(&format!("if ({g}) {{"));
            }
        };
        let end = |l: &mut Self| {
            if guard.is_some() {
                l.line("}");
            }
        };
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            let h = self.handle(site, src, "CB", false)?;
            let byte = match &index {
                Some(i) => format!("({i} * 16u + {o})"),
                None => o.clone(),
            };
            let Some(Operand::Vector { reg, .. }) = *data else { return Err("a buffer result") };
            for k in 0..count {
                let t = self.fresh("t", "uint");
                self.line(&format!("{t} = {h}.d[({byte} + {}u) >> 4][(({byte} + {}u) >> 2) & 3u];", 4 * k, 4 * k));
                self.write(inst, Reg::V(reg + k as u8), t, &guard)?;
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        if let Some(op) = atomic {
            if index.is_some() {
                return Err("a structured buffer of unknown stride");
            }
            let f = match op {
                "add" => "InterlockedAdd",
                "and" => "InterlockedAnd",
                "or" => "InterlockedOr",
                "xor" => "InterlockedXor",
                "umin" => "InterlockedMin",
                "umax" => "InterlockedMax",
                "swap" => "InterlockedExchange",
                "cmpswap" => "InterlockedCompareExchange",
                _ => return Err(name),
            };
            let Some(Operand::Vector { reg, .. }) = *data else { return Err("buffer atomic data") };
            let x = self.u(inst.read(Reg::V(reg)).ok_or("buffer atomic data")?)?;
            let h = self.handle(site, src, "RWByteAddressBuffer", false)?;
            let old = self.fresh("t", "uint");
            let keep = if cache.glc { x.clone() } else { "0u".into() };
            self.line(&format!("{old} = {keep};"));
            begin(self);
            if op == "cmpswap" {
                let cmp = self.u(inst.read(Reg::V(reg + 1)).ok_or("buffer atomic data")?)?;
                self.line(&format!("{h}.{f}({o}, {cmp}, {x}, {old});"));
            } else {
                self.line(&format!("{h}.{f}({o}, {x}, {old});"));
            }
            end(self);
            self.writes_uavs = true;
            if cache.glc {
                self.write(inst, Reg::V(reg), old, &None)?;
            }
            return Ok(());
        }
        if store {
            let Some(Operand::Vector { reg, .. }) = *data else { return Err("buffer store data") };
            let vals: Vec<String> = (0..count).map(|k| self.u(inst.read(Reg::V(reg + k as u8)).ok_or("buffer store data")?)).collect::<Result<_, _>>()?;
            self.writes_uavs = true;
            if typed {
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                let h = self.handle(site, src, "RWBuffer<float4>", false)?;
                let i = index.unwrap_or("0u".into());
                begin(self);
                let fs: Vec<String> = vals.iter().map(|x| flt(x)).collect();
                self.line(&format!("{h}[{i}] = float4({});", fs.join(", ")));
                end(self);
                return Ok(());
            }
            if index.is_some() {
                return Err("a structured buffer of unknown stride");
            }
            let h = self.handle(site, src, "RWByteAddressBuffer", false)?;
            begin(self);
            let call = match count {
                1 => format!("{h}.Store({o}, {});", vals[0]),
                n => format!("{h}.Store{n}({o}, uint{n}({}));", vals.join(", ")),
            };
            self.line(&call);
            end(self);
            return Ok(());
        }
        let Some(Operand::Vector { reg, .. }) = *data else { return Err("a buffer result") };
        let ty = if uav { "RWByteAddressBuffer" } else { "ByteAddressBuffer" };
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let h = self.handle(site, src, ty, false)?;
            let xs = self.format_load(&h, &o, f, count)?;
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, &guard)?;
            }
            return Ok(());
        }
        if typed {
            if *offen || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            let h = self.handle(site, src, if uav { "RWBuffer<float4>" } else { "Buffer<float4>" }, false)?;
            let r = self.fresh("x", "float4");
            self.line(&format!("{r} = {h}[{}];", index.unwrap_or("0u".into())));
            for k in 0..count {
                self.write(inst, Reg::V(reg + k as u8), format!("asuint({r}.{})", ["x", "y", "z", "w"][k as usize]), &guard)?;
            }
            return Ok(());
        }
        if index.is_some() {
            return Err("a structured buffer of unknown stride");
        }
        let h = self.handle(site, src, ty, false)?;
        let r = self.fresh("x", ["uint", "uint2", "uint3", "uint4"][count as usize - 1]);
        let call = match count {
            1 => format!("{h}.Load({o})"),
            n => format!("{h}.Load{n}({o})"),
        };
        self.line(&format!("{r} = {call};"));
        for k in 0..count {
            let x = if count == 1 { r.clone() } else { format!("{r}.{}", ["x", "y", "z", "w"][k as usize]) };
            self.write(inst, Reg::V(reg + k as u8), x, &guard)?;
        }
        Ok(())
    }

    /// Typed-format loads converted from raw bytes as the hardware does.
    fn format_load(&mut self, h: &str, off: &str, format: u8, channels: u32) -> Result<Vec<String>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let total: u32 = bits.iter().sum();
        let words: Vec<String> = if total < 32 {
            let w = self.fresh("t", "uint");
            self.line(&format!("{w} = {h}.Load({off} & ~3u) >> (({off} & 3u) * 8u);"));
            vec![w]
        } else {
            let n = total / 32;
            let w = self.fresh("x", ["uint", "uint2", "uint3", "uint4"][n as usize - 1]);
            self.line(&format!("{w} = {h}.Load{}({off});", if n == 1 { String::new() } else { n.to_string() }));
            if n == 1 { vec![w] } else { (0..n).map(|k| format!("{w}.{}", ["x", "y", "z", "w"][k as usize])).collect() }
        };
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = &words[(at / 32) as usize];
            let field = if w == 32 { word.clone() } else { format!("(({word} >> {}u) & {})", at % 32, lit((1u32 << w) - 1)) };
            at += w;
            out.push(match numeric {
                Numeric::Unorm => format!("hw_unorm({field}, {w}u)"),
                Numeric::Snorm => format!("hw_snorm({field}, {w}u)"),
                Numeric::Uscaled => format!("asuint(float({field}))"),
                Numeric::Sscaled => format!("asuint(float(int(hw_sext({field}, {w}u))))"),
                Numeric::Uint => field,
                Numeric::Sint => format!("hw_sext({field}, {w}u)"),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let sh = if w == 16 { 0 } else { 15 - w };
                    format!("asuint(f16tof32({field} << {sh}u))")
                }
            });
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => "1u".to_owned(),
            _ => lit(1.0f32.to_bits()),
        };
        for k in out.len() as u32..channels {
            out.push(if k == 3 { one.clone() } else { "0u".into() });
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    // ---- assembly ----

    fn finish(self, missing: &HashSet<Value>) -> Result<Hlsl, ShaderError> {
        let _ = missing;
        let p = self.p;
        struct Bl<'b> {
            l: &'b L<'b>,
        }
        impl cfg::Blocks for Bl<'_> {
            fn count(&self) -> usize {
                self.l.p.blocks.len()
            }
            fn body(&self, b: usize) -> &str {
                &self.l.bodies[b]
            }
            fn end(&self, b: usize) -> &End {
                &self.l.ends[b]
            }
            fn edge(&self, from: usize, to: usize) -> String {
                self.l.edges.get(&(from, to)).cloned().unwrap_or_default()
            }
            fn back_edge(&self, h: usize) -> String {
                let abort = match self.l.stage {
                    HwStage::Ps => "discard; return o;",
                    HwStage::Cs | HwStage::Ms => "return;",
                    _ => "return o;",
                };
                format!("if (++n{h} >= {}u) {{ {abort} }}", loop_limit(self.l.stage))
            }
            fn loop_entry(&self, h: usize) -> String {
                format!("n{h} = 0u;")
            }
        }
        let _ = p;
        let body = cfg::structure(&Bl { l: &self }).map_err(|w| ShaderError::Unsupported { offset: 0, what: w })?;
        // A value is declared where it is defined: its block's statements
        // come before everything the block dominates (`cfg`), which is where
        // SSA puts its uses. Joins and lane bits are declared at the top.
        let decls = self.decls.clone();
        let mut resolved = String::with_capacity(body.len() + body.len() / 8);
        for line in body.split_inclusive('\n') {
            let t = line.trim_start();
            let indent = &line[..line.len() - t.len()];
            match t.chars().next() {
                Some(c @ ('\u{1}' | '\u{2}')) => {
                    let rest = &t[1..];
                    let name = rest.split(' ').next().unwrap_or("");
                    resolved.push_str(indent);
                    if !decls.contains_key(name) {
                        resolved.push_str(if c == '\u{1}' { "uint " } else { "bool " });
                    }
                    resolved.push_str(rest);
                }
                _ => resolved.push_str(line),
            }
        }
        let body = resolved;
        let mut src = String::new();
        src.push_str("// Translated by victory-shader (HLSL backend).\n");
        src.push_str(include_str!("prelude.hlsl"));
        src.push('\n');
        // User data.
        let ud_dwords = self.user_reads.keys().map(|k| k + 1).max().unwrap_or(0).next_multiple_of(4);
        if ud_dwords > 0 {
            let _ = writeln!(src, "cbuffer UserData : register(b0, space0) {{ uint4 ud[{}]; }};", ud_dwords / 4);
        }
        src.push_str("struct CB { uint4 d[4096]; };\n");
        for g in &self.globals {
            src.push_str(g);
            src.push('\n');
        }
        if self.lds_dwords > 0 {
            let _ = writeln!(src, "groupshared uint lds[{}];", self.lds_dwords);
        }
        if let Some(words) = self.code_table {
            let code: Vec<String> = self
                .native
                .code
                .chunks(4)
                .take(words as usize)
                .map(|c| lit(u32::from_le_bytes([c[0], c.get(1).copied().unwrap_or(0), c.get(2).copied().unwrap_or(0), c.get(3).copied().unwrap_or(0)])))
                .collect();
            let _ = writeln!(src, "static const uint code_table[{}] = {{ {} }};", code.len(), code.join(", "));
        }
        // Entry.
        let (profile, head) = match self.stage {
            HwStage::Ps => {
                let mut s = String::from("struct PSIn {\n");
                for (id, e) in sig_order(&self.inputs, self.sorted) {
                    let ty = sig_type(e, self.half_io);
                    let mode = match (e.sv, self.input_use[id].1, e.interp) {
                        (0, Some(m), _) | (0, None, m) => interp_keyword(m, e.comp),
                        _ => "",
                    };
                    let _ = writeln!(s, "  {mode}{ty} a{id} : {}{};", sem_name(e), sem_index(e));
                }
                s.push_str("};\nstruct PSOut {\n");
                for (id, e) in sig_order(&self.outputs, self.sorted) {
                    let ty = sig_type(e, self.half_io);
                    let _ = writeln!(s, "  {ty} o{id} : {}{};", sem_name(e), sem_index(e));
                }
                s.push_str("};\n");
                let attrs = "";
                (
                    "ps_6_6",
                    format!("{s}{attrs}PSOut main(PSIn i) {{\n  PSOut o = (PSOut)0;\n"),
                )
            }
            HwStage::Cs => {
                let StageIo::Cs(cs) = &self.native.io else {
                    return Err(ShaderError::Malformed("a compute program without dispatch settings"));
                };
                let t = cs.threads;
                let params: Vec<&String> = self.system.values().collect();
                let params: Vec<String> = params.into_iter().cloned().collect();
                (
                    "cs_6_6",
                    format!(
                        "{}[numthreads({}, {}, {})]\nvoid main({}) {{\n",
                        if self.wave32 { "[WaveSize(32)]\n" } else { "" },
                        t[0],
                        t[1],
                        t[2],
                        params.join(", ")
                    ),
                )
            }
            HwStage::Vs => {
                let mut s = String::from("struct VSIn {\n");
                for (id, e) in sig_order(&self.inputs, self.sorted) {
                    let ty = sig_type(e, self.half_io);
                    let _ = writeln!(s, "  {ty} a{id} : {}{};", sem_name(e), sem_index(e));
                }
                for (k, v) in &self.system {
                    let _ = writeln!(s, "  {v};");
                    let _ = k;
                }
                s.push_str("};\nstruct VSOut {\n");
                for (id, e) in sig_order(&self.outputs, self.sorted) {
                    let ty = sig_type(e, self.half_io);
                    let _ = writeln!(s, "  {ty} o{id} : {}{};", sem_name(e), sem_index(e));
                }
                s.push_str("};\n");
                ("vs_6_6", format!("{s}VSOut main(VSIn i) {{\n  VSOut o = (VSOut)0;\n"))
            }
            HwStage::Ms => {
                let m = self.mesh.clone().ok_or(ShaderError::Malformed("a mesh program without its limits"))?;
                let mut s = String::from("struct MSVert {\n");
                for (id, e) in sig_order(&self.outputs, self.sorted) {
                    let _ = writeln!(s, "  {} o{id} : {}{};", sig_type(e, self.half_io), sem_name(e), sem_index(e));
                }
                s.push_str("};\n");
                if !m.prims.is_empty() {
                    s.push_str("struct MSPrim {\n");
                    for (id, e) in sig_order(&m.prims, self.sorted) {
                        let _ = writeln!(s, "  {} p{id} : {}{};", sig_type(e, self.half_io), sem_name(e), sem_index(e));
                    }
                    s.push_str("};\n");
                }
                // D3D12 caps the declared outputs at 32 KiB; lane n writes only
                // vertex n and primitive n, so maxima past the lanes come down
                // to the lane count where the outputs do not fit.
                let scalars = |els: &[Sig]| els.iter().map(|e| u64::from(e.cols())).sum::<u64>();
                let bytes = |v: u32, p: u32| {
                    4 * (scalars(&self.outputs) * u64::from(v).next_multiple_of(32) + scalars(&m.prims) * u64::from(p).next_multiple_of(32))
                };
                let (mut maxv, mut maxp) = (m.max_vertices, m.max_primitives);
                if bytes(maxv, maxp) > 32 * 1024 {
                    maxv = maxv.min(m.threads);
                    maxp = maxp.min(m.threads);
                }
                let mut params = vec!["uint gi : SV_GroupIndex".to_owned()];
                if self.system.contains_key("gid") {
                    params.push("uint3 gid : SV_GroupID".into());
                }
                if self.system.contains_key("gtid") {
                    params.push("uint3 gtid : SV_GroupThreadID".into());
                }
                params.push(format!("out vertices MSVert verts[{maxv}]"));
                params.push(format!("out indices uint3 tris[{maxp}]"));
                if !m.prims.is_empty() {
                    params.push(format!("out primitives MSPrim prims[{maxp}]"));
                }
                // Every primitive starts culled (also those past the lanes): the
                // host draws all it was promised.
                let head = format!(
                    "{s}[outputtopology(\"triangle\")]\n[numthreads({}, 1, 1)]\nvoid main({}) {{\n  SetMeshOutputCounts({maxv}u, {maxp}u);\n  for (uint k = gi; k < {maxp}u; k += {}u) tris[k] = uint3(0u, 0u, 0u);\n",
                    m.threads,
                    params.join(", "),
                    m.threads
                );
                ("ms_6_6", head)
            }
            _ => return Err(ShaderError::Unsupported { offset: 0, what: "a stage the HLSL prototype lacks" }),
        };
        src.push_str(&head);
        for (n, ty) in &decls {
            if ty.is_empty() {
                continue;
            }
            let init = match *ty {
                "bool" => "false",
                "uint" => "0u",
                "uint4" => "uint4(0u, 0u, 0u, 0u)",
                "float4" => "float4(0.0f, 0.0f, 0.0f, 0.0f)",
                "float3" => "float3(0.0f, 0.0f, 0.0f)",
                "uint64_t" => "0",
                _ => "0",
            };
            let _ = writeln!(src, "  {ty} {n} = {init};");
        }
        for h in &self.headers {
            let _ = writeln!(src, "  uint n{h} = 0u;");
        }
        src.push_str(&body);
        // (every path returns already; the compiler does not see it)
        src.push_str(if matches!(self.stage, HwStage::Cs | HwStage::Ms) { "  return;\n}\n" } else { "  return o;\n}\n" });
        let user_data: Vec<UserData> = self.user_reads.values().copied().collect();
        let shader = HostShader {
            stage: self.stage,
            dxil: Vec::new(),
            user_data,
            static_samplers: super::static_samplers(&self.samplers, self.cmp_samplers),
            samplers: self.samplers.clone(),
            root_cbvs: self.root_cbvs.iter().copied().collect(),
            root_srvs: self.root_srvs.iter().copied().collect(),
            root_uavs: self.root_uavs.iter().copied().collect(),
            root_indexed_views: self.res.root_indexed_views.clone(),
        };
        let enable_16bit =
            self.half_io && self.inputs.iter().chain(&self.outputs).chain(self.mesh.iter().flat_map(|m| &m.prims)).any(|e| e.native16);
        Ok(Hlsl {
            stage: self.stage,
            source: src,
            profile,
            enable_16bit,
            shader,
            approximate: self.approximate.into_iter().collect(),
        })
    }
}

/// Parameters packing 16-bit rows two per dword, and where each lays its
/// elements out: (first row, [(dword, register, column, columns)]). The
/// console's layout, as `emit::lower::packed_layout` documents it: a
/// parameter leads with the first element of its first row, then that of
/// its second row; an element sharing a register behind another follows in
/// the first parameter, in row order, with whole dwords left.
type PackedLayout = Vec<(u8, Vec<(u8, u8, u8, u8)>)>;

fn packed_layout(elements: &[Sig], packs: &[(u8, Option<u8>)]) -> PackedLayout {
    let of_row = |r: u8| {
        let mut of: Vec<&Sig> = elements.iter().filter(|e| e.reg == Some(u32::from(r))).collect();
        of.sort_by_key(|e| e.col());
        of
    };
    let mut packs = packs.to_vec();
    packs.sort_unstable();
    packs.dedup_by_key(|p| p.0);
    let mut layout: Vec<(u8, u8, Vec<(u8, u8, u8, u8)>)> = Vec::new();
    let mut behind = Vec::new();
    for &(a, b) in &packs {
        let mut placed = Vec::new();
        let mut at = 0u8;
        if !of_row(a).is_empty() {
            for r in [Some(a), b].into_iter().flatten() {
                let of = of_row(r);
                let Some((lead, more)) = of.split_first() else { continue };
                placed.push((at, r, lead.col(), lead.cols()));
                at += lead.cols().div_ceil(2);
                behind.extend(more.iter().map(|e| (r, e.col(), e.cols())));
            }
        }
        layout.push((a, at, placed));
    }
    behind.sort_unstable();
    for (r, col, cols) in behind {
        let need = cols.div_ceil(2);
        if let Some(p) = layout.iter_mut().find(|p| p.1 + need <= 4) {
            p.2.push((p.1, r, col, cols));
            p.1 += need;
        }
    }
    layout.into_iter().map(|(a, _, of)| (a, of)).collect()
}

/// Where half `h` of the parameter whose first 16-bit row is `a` comes from.
fn packed_half(elements: &[Sig], packs: &[(u8, Option<u8>)], a: u8, h: u8) -> Option<(u8, u8)> {
    let layout = packed_layout(elements, packs);
    let (_, of) = layout.iter().find(|p| p.0 == a)?;
    of.iter().find_map(|&(dword, row, col, cols)| {
        let k = h.checked_sub(2 * dword)?;
        (k < cols).then_some((row, col + k))
    })
}

/// Vertex or mesh outputs of 16-bit values: row -> (two per dword, the row
/// packed after it: the next 16-bit row of the signature no output entry
/// names, in row order), as `emit::lower::half_outputs`.
fn half_outputs(io: &StageIo, outputs: &[Sig]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.sv == 0)
        .filter_map(|e| u8::try_from(e.reg?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o.pack2.then(|| free.iter().position(|&r| r > o.row)).flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// Pixel inputs packing two 16-bit rows per dword: (first row, second row).
fn input_packs(io: &StageIo, inputs: &[Sig]) -> Vec<(u8, Option<u8>)> {
    let StageIo::Ps { inputs: entries, .. } = io else { return Vec::new() };
    let packed = || entries.iter().filter(|i| i.fp16 && i.pack2);
    // A second row the metadata leaves out (`row2` = `row`): the next 16-bit
    // row of the signature no input entry names, in row order.
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.sv == 0)
        .filter_map(|e| u8::try_from(e.reg?).ok())
        .filter(|&r| !entries.iter().any(|i| i.row == r))
        .filter(|&r| !packed().any(|i| i.row2 != i.row && i.row2 == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut missing: Vec<_> = packed().filter(|i| i.row2 == i.row).collect();
    missing.sort_by_key(|i| i.row);
    let mut partners = HashMap::new();
    for i in missing {
        if let Some(k) = free.iter().position(|&r| r > i.row) {
            partners.insert(i.row, free.remove(k));
        }
    }
    packed()
        .map(|i| (i.row, if i.row2 != i.row { Some(i.row2) } else { partners.get(&i.row).copied() }))
        .collect()
}

/// Signature elements to declare: in the console's order, or in register
/// order (then column). DXC packs them in declaration order; the console's
/// own order gives its rows for most programs, register order for others
/// (`super::translate` tries both).
fn sig_order(sig: &[Sig], sorted: bool) -> Vec<(usize, &Sig)> {
    let mut v: Vec<(usize, &Sig)> = sig.iter().enumerate().collect();
    if sorted {
        v.sort_by_key(|(id, e)| (e.reg.unwrap_or(u32::MAX), e.col(), *id));
    } else {
        v.sort_by_key(|(id, e)| (e.order, *id));
    }
    v
}

fn zero_off(n: usize) -> String {
    match n {
        1 => "0".into(),
        n => format!("int{n}({})", vec!["0"; n].join(", ")),
    }
}

fn sem_name(e: &Sig) -> String {
    match e.sv {
        0 => e.name.clone(),
        1 => "SV_Position".into(),
        2 => "SV_ClipDistance".into(),
        3 => "SV_CullDistance".into(),
        4 => "SV_RenderTargetArrayIndex".into(),
        5 => "SV_ViewportArrayIndex".into(),
        6 => "SV_VertexID".into(),
        7 => "SV_PrimitiveID".into(),
        8 => "SV_InstanceID".into(),
        9 => "SV_IsFrontFace".into(),
        10 => "SV_SampleIndex".into(),
        64 => "SV_Target".into(),
        65 => "SV_Depth".into(),
        66 => "SV_Coverage".into(),
        67 => "SV_DepthGreaterEqual".into(),
        68 => "SV_DepthLessEqual".into(),
        69 => "SV_StencilRef".into(),
        23 => "SV_Barycentrics".into(),
        _ => e.name.clone(),
    }
}

fn sem_index(e: &Sig) -> String {
    match e.sv {
        0 | 2 | 3 | 64 | 23 => e.index.to_string(),
        _ => String::new(),
    }
}

fn sig_type(e: &Sig, half_io: bool) -> String {
    if e.sv == 9 {
        return "bool".into();
    }
    let base = match (e.comp, e.native16 && half_io) {
        (1, false) => "uint",
        (2, false) => "int",
        (_, false) => "float",
        (1, true) => "uint16_t",
        (2, true) => "int16_t",
        (_, true) => "float16_t",
    };
    let n = e.cols();
    if n == 1 { base.into() } else { format!("{base}{n}") }
}

fn interp_keyword(mode: u8, comp: u32) -> &'static str {
    if comp != 3 {
        return "nointerpolation ";
    }
    match mode {
        1 => "nointerpolation ",
        3 => "centroid ",
        4 => "noperspective ",
        5 => "noperspective centroid ",
        6 => "sample ",
        7 => "noperspective sample ",
        _ => "",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// GFX10 buffer formats (as `emit::lower::buffer_format`).
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),
        (&[10, 11, 11], ALL),
        (&[2, 10, 10, 10], NORM),
        (&[10, 10, 10, 2], NORM),
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

#[allow(dead_code)]
const _: (u8, u8) = (VCC, M0);
