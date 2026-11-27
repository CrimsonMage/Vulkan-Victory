//! Control: blocks lifted in dominator order, their phis, edges and terminators; EXEC guards around vector code.

use super::*;

impl Lift<'_> {
    pub(super) fn close(&mut self) {
        let Some(mut o) = self.open.take() else { return };
        let mut phis = Vec::new();
        let mut masks: HashMap<V, V> = HashMap::new();
        for (_, new, old) in o.defs {
            if let Some(&R::M(inner, half)) = self.vals.get(&new) {
                let _ = old;
                let phi = match masks.get(&inner) {
                    Some(&p) => p,
                    None => {
                        let f = self.kb(false);
                        let p = self.f.push(Node { op: Op::Phi, ty: Ty::BOOL, args: vec![inner, f], imm: Vec::new() });
                        phis.push(p);
                        masks.insert(inner, p);
                        p
                    }
                };
                self.vals.insert(new, R::M(phi, half));
                continue;
            }
            let mut inner = match self.vals.get(&new) {
                Some(R::D(v)) => *v,
                _ => continue,
            };
            // the type its readers want
            let want = match self.floaty.get(&new).copied().unwrap_or(0) {
                x if x > 0 => Ty::F32,
                x if x < 0 => Ty::U32,
                _ => self.f.ty(inner),
            };
            if want != self.f.ty(inner) && matches!(self.f.ty(inner), Ty::F32 | Ty::U32) {
                inner = match self.f.as_const(inner) {
                    Some(c) => self.f.konst(want, c),
                    None => {
                        let c = self.f.push(Node { op: Op::Spv(op::BITCAST), ty: want, args: vec![inner], imm: Vec::new() });
                        o.items.push(Item::N(c));
                        c
                    }
                };
            }
            let ty = self.f.ty(inner);
            let outer = match self.value(old, 0) {
                Ok(r) => {
                    let d = self.data(r);
                    match ty {
                        Ty::F32 => self.f32v(d),
                        _ => self.u32v(d),
                    }
                }
                Err(_) => self.f.global(Op::Undef, ty),
            };
            let phi = self.f.push(Node { op: Op::Phi, ty, args: vec![inner, outer], imm: Vec::new() });
            self.born.insert(phi, self.cur_block);
            phis.push(phi);
            self.vals.insert(new, R::D(phi));
        }
        self.cur.push(Item::Guard { cond: o.cond, body: o.items, phis });
    }

    /// Before an unguarded instruction: closes the open guard if it reads a
    /// value written there.
    pub(super) fn need(&mut self, inst: &Inst) {
        if let Some(o) = &self.open
            && inst.reads.iter().any(|(_, v)| o.defined.contains(v))
        {
            self.close();
        }
    }

    /// Opens (or keeps) a guard for `inst`; `false` when every lane runs.
    pub(super) fn guard(&mut self, inst: &Inst) -> Res<bool> {
        let a = self.active(inst)?;
        if self.f.as_const(a) == Some(1) {
            self.need(inst);
            return Ok(false);
        }
        if let Some(o) = &self.open
            && o.cond != a
        {
            self.close();
        }
        if self.open.is_none() {
            self.open =
                Some(Open {
                cond: a,
                items: Vec::new(),
                defs: Vec::new(),
                defined: HashSet::new(),
                cse: HashMap::new(),
                inner: HashSet::new(),
                effects: false,
            });
        }
        Ok(true)
    }

    pub fn run(mut self) -> Res<(Func, Interface)> {
        let p = self.p;
        // Blocks in dominator-tree preorder: a block's dominators first, so
        // pure values are shared down the tree (one CSE scope per block).
        let (pre, idom) = dom_preorder(p);
        for &b in pre.iter().skip(1) {
            self.depth[b] = self.depth[idom[b]] + 1;
        }
        self.done = vec![None; p.blocks.len()];
        let mut scopes: Vec<(usize, usize)> = Vec::new();
        for &b in &pre {
            while let Some(&(top, mark)) = scopes.last() {
                if top == idom[b] {
                    break;
                }
                self.rollback(mark);
                scopes.pop();
            }
            scopes.push((b, self.undo.len()));
            self.lift_block(b)?;
        }
        self.fill_unreached();
        // Unreachable blocks (none from reverse postorder) become returns.
        self.f.blocks = std::mem::take(&mut self.done)
            .into_iter()
            .map(|b| b.unwrap_or(tir::Block { phis: Vec::new(), preds: Vec::new(), body: Vec::new(), term: Term::Return }))
            .collect();
        Ok((self.f, self.iface))
    }

    pub(super) fn lift_block(&mut self, b: usize) -> Res<()> {
        let p = self.p;
        self.cur = Vec::new();
        self.cur_block = b;
        self.words.clear();
        // loads are not shared across blocks
        self.epoch += 1;
        self.block_phis(b);
        self.made[b] = true;
        // Edges from blocks lifted before: their values, converted at
        // the end of each source.
        for (k, &q) in p.blocks[b].preds.iter().enumerate() {
            if self.done[q].is_some() {
                let saved_cur = std::mem::take(&mut self.cur);
                self.cur = std::mem::take(&mut self.done[q].as_mut().unwrap().body);
                self.no_cse = true;
                self.cur_block = q;
                let r = self.edge_k(q, b, k);
                self.cur_block = b;
                self.no_cse = false;
                self.done[q].as_mut().unwrap().body = std::mem::replace(&mut self.cur, saved_cur);
                r?;
            }
        }
        // values exchanged at the end of a source stay there; the halves may
        // arrive from different LDS accesses
        self.words.clear();
        self.lds_dirty = b != 0;
        let n = p.blocks[b].insts.len();
        let branch = matches!(p.blocks[b].exit, Exit::Branch { .. });
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            self.site = (b, i);
            if i + 1 == n && branch {
                break;
            }
            self.inst(inst)?;
        }
        self.close();
        let term = self.terminator(b)?;
        let body = std::mem::take(&mut self.cur);
        let mut phis = self.all_phis[b].clone();
        let mut seen = HashSet::new();
        phis.retain(|x| seen.insert(*x));
        self.done[b] = Some(tir::Block { phis, preds: p.blocks[b].preds.clone(), body, term });
        Ok(())
    }

    /// Phis: lane masks and SCC as booleans, data as f32 when every value
    /// known so far is f32 (back edges convert), else u32. The two dwords of
    /// a wave64 lane mask share one phi (each half's bit).
    pub(super) fn block_phis(&mut self, b: usize) {
        let p = self.p;
        let kinds: Vec<(bool, Half)> = p.blocks[b].phis.iter().map(|phi| self.phi_kind(phi)).collect();
        let mut pairs = Vec::new();
        for (j, phi) in p.blocks[b].phis.iter().enumerate() {
            if let (Reg::S(c), (true, Half::Hi)) = (phi.reg, kinds[j])
                && self.wave64
                && c > 0
                && let Some(i) = p.blocks[b].phis.iter().position(|q| q.reg == Reg::S(c - 1))
                && kinds[i] == (true, Half::Lo)
            {
                pairs.push((i, j));
            }
        }
        let mut nodes = vec![u32::MAX; kinds.len()];
        for (j, phi) in p.blocks[b].phis.iter().enumerate() {
            if pairs.iter().any(|&(_, h)| h == j) {
                continue;
            }
            let (mask, half) = kinds[j];
            let ty = if mask {
                Ty::BOOL
            } else if self.floaty.get(&phi.value).copied().unwrap_or(0) > 0 {
                Ty::F32
            } else {
                Ty::U32
            };
            let x = self.f.push(Node { op: Op::Phi, ty, args: vec![u32::MAX; phi.args.len()], imm: Vec::new() });
            self.born.insert(x, b);
            nodes[j] = x;
            let r = match (phi.reg == Reg::Scc, mask) {
                (true, _) => R::B(x),
                (false, true) => R::M(x, half),
                _ => R::D(x),
            };
            self.vals.insert(phi.value, r);
        }
        for &(i, j) in &pairs {
            nodes[j] = nodes[i];
            self.vals.insert(p.blocks[b].phis[j].value, R::M(nodes[i], Half::Hi));
        }
        self.all_phis[b] = nodes;
        self.mask_pairs.insert(b, pairs);
    }

    /// Whether a phi is a lane mask (or SCC), and which dword of a mask.
    fn phi_kind(&self, phi: &ir::Phi) -> (bool, Half) {
        let wide = |c: u8| c == ir::EXEC || c == ir::EXEC + 1 || c == ir::VCC || c == ir::VCC + 1;
        let known: Vec<R> = phi.args.iter().filter_map(|a| self.vals.get(a).copied()).collect();
        let mask = matches!(phi.reg, Reg::S(c) if wide(c))
            || phi.reg == Reg::Scc
            || (!known.is_empty() && known.iter().all(|r| matches!(r, R::M(..))));
        let hi = match phi.reg {
            Reg::S(c) if self.wave64 && (c == ir::VCC + 1 || c == ir::EXEC + 1) => true,
            Reg::S(_) => self.wave64 && !known.is_empty() && known.iter().all(|r| matches!(r, R::M(_, Half::Hi))),
            _ => false,
        };
        (mask, if hi { Half::Hi } else { Half::Lo })
    }

    /// A block's terminator, after the values its edges give lifted targets.
    pub(super) fn terminator(&mut self, b: usize) -> Res<Term> {
        let p = self.p;
        Ok(match p.blocks[b].exit {
            Exit::Jump(t) => {
                self.edge(b, t)?;
                Term::Jump(t)
            }
            Exit::Branch { taken, not_taken } => {
                let inst = p.blocks[b].insts.last().unwrap();
                self.site = (b, p.blocks[b].insts.len() - 1);
                let c = self.branch_cond(inst)?;
                self.edge(b, taken)?;
                self.edge(b, not_taken)?;
                Term::Branch(c, taken, not_taken)
            }
            Exit::End => {
                // pixels: a lane outside EXEC at the end, in a block
                // without the last export (which kills its own, `inst`),
                // left without exporting: killed (origin/main
                // `emit/lower.rs`, `LaneExit::End`: "A lane leaving without
                // its final export was killed")
                let last = p.blocks[b].insts.last();
                let exported = p.blocks[b].insts.iter().any(|i| matches!(i.fields, Fields::Exp { done: true, .. }));
                if self.native.stage == HwStage::Ps
                    && !exported
                    && let Some(inst) = last.filter(|i| i.read(Reg::S(ir::EXEC)).is_some())
                {
                    let a = self.active(inst)?;
                    self.kill_inactive(a);
                }
                Term::Return
            }
            Exit::Indirect => return Err(unsupported(p.blocks[b].offset, "an indirect jump (s_setpc_b64)")),
        })
    }

    /// Phi args of edges from blocks never reached: undefined.
    pub(super) fn fill_unreached(&mut self) {
        for b in 0..self.p.blocks.len() {
            for j in 0..self.all_phis[b].len() {
                let node = self.all_phis[b][j];
                let ty = self.f.ty(node);
                let undef = self.f.global(Op::Undef, ty);
                for a in &mut self.f.nodes[node as usize].args {
                    if *a == u32::MAX {
                        *a = undef;
                    }
                }
            }
        }
    }

    pub(super) fn kill_inactive(&mut self, a: V) {
        if self.f.as_const(a) == Some(1) {
            return;
        }
        let na = self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]);
        let d = self.f.push(Node { op: Op::Demote, ty: Ty::VOID, args: vec![], imm: vec![] });
        self.cur.push(Item::Guard { cond: na, body: vec![Item::N(d)], phis: vec![] });
        self.iface.demotes = true;
    }

    /// At the end of `from`: the values the edge gives the phis of a block
    /// lifted already (a back edge); other targets take theirs when they
    /// are lifted.
    pub(super) fn edge(&mut self, from: usize, to: usize) -> Res<()> {
        let p = self.p;
        if !self.made[to] {
            return Ok(());
        }
        for (k, &q) in p.blocks[to].preds.iter().enumerate() {
            if q == from {
                self.edge_k(from, to, k)?;
            }
        }
        Ok(())
    }

    /// The values edge `k` (from `from`) gives `to`'s phis, converted to
    /// their types where the current list is.
    pub(super) fn edge_k(&mut self, from: usize, to: usize, k: usize) -> Res<()> {
        let p = self.p;
        let pairs = self.mask_pairs.get(&to).cloned().unwrap_or_default();
        for (j, phi) in p.blocks[to].phis.iter().enumerate() {
            if pairs.iter().any(|&(_, h)| h == j) {
                continue;
            }
            let node = self.all_phis[to][j];
            let ty = self.f.ty(node);
            let off = p.blocks[from].offset;
            let hi = pairs.iter().find(|&&(l, _)| l == j).map(|&(_, h)| self.value(p.blocks[to].phis[h].args[k], off));
            let x = match self.value(phi.args[k], off) {
                Ok(lo) if hi.is_some() => match hi {
                    Some(Ok(hi)) => self.mask_pair(lo, hi),
                    _ => self.mask_of(lo),
                },
                Ok(r) => match ty {
                    Ty::BOOL if phi.reg == Reg::Scc => self.scc_of(r),
                    Ty::BOOL => self.mask_of(r),
                    Ty::F32 => {
                        let d = self.data(r);
                        self.f32v(d)
                    }
                    _ => {
                        let d = self.data(r);
                        self.u32v(d)
                    }
                },
                Err(_) => self.f.global(Op::Undef, ty),
            };
            self.f.nodes[node as usize].args[k] = x;
        }
        Ok(())
    }

    pub(super) fn branch_cond(&mut self, inst: &Inst) -> Res<V> {
        let name = inst.name();
        Ok(match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let r = self.val(inst, Reg::Scc)?;
                let b = self.scc_of(r);
                if name == "s_cbranch_scc0" { self.spv(op::LOGICAL_NOT, Ty::BOOL, &[b]) } else { b }
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") { ir::VCC } else { ir::EXEC };
                let m = self.mask_reg(inst, reg)?;
                let a = self.any(m);
                if name.ends_with('z') && !name.ends_with("nz") { self.spv(op::LOGICAL_NOT, Ty::BOOL, &[a]) } else { a }
            }
            _ => return Err(unsupported(inst.machine.offset, "a branch")),
        })
    }

    pub(super) fn inst(&mut self, inst: &Inst) -> Res<()> {
        let name = inst.name();
        let off = inst.machine.offset;
        match &inst.fields {
            Fields::Sop { dst, src, imm } => {
                let (dst, src, imm) = (*dst, src.clone(), *imm);
                self.need(inst);
                self.sop(inst, name, &dst, &src, imm)
            }
            Fields::Sopp { .. } => match name {
                "s_barrier" => {
                    self.close();
                    self.lds_dirty = false;
                    self.emit(Op::Barrier, Ty::VOID, vec![]);
                    Ok(())
                }
                // Mesh output counts: the module declares the maxima at its
                // start (primitives culled until exported).
                "s_sendmsg" | "s_sendmsghalt" => Ok(()),
                "s_endpgm" | "s_branch" | "s_icache_inv" | "s_round_mode" | "s_denorm_mode" | "s_wakeup" => Ok(()),
                _ => Err(unsupported(off, name)),
            },
            Fields::Smem { data, base, offset, soffset, .. } => {
                let (data, base, offset, soffset) = (*data, *base, *offset, *soffset);
                self.need(inst);
                self.smem(inst, name, data, base, offset, soffset)
            }
            Fields::Valu(v) => {
                let v = v.clone();
                self.valu(inst, name, &v)
            }
            Fields::Vintrp { dst, .. } => {
                let dst = *dst;
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.interp(inst, dst);
                self.guarded = false;
                r
            }
            Fields::Buffer { .. } => {
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.buffer(inst, name);
                self.guarded = false;
                r
            }
            Fields::Ds { .. } => {
                if name == "ds_swizzle_b32" || name == "ds_bpermute_b32" || name == "ds_permute_b32" {
                    self.need(inst);
                    return self.ds_cross(inst, name);
                }
                self.lds_meet();
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.ds(inst, name);
                self.guarded = false;
                self.lds_dirty = self.xwave;
                r
            }
            Fields::Image { .. } => {
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.image(inst, name);
                self.guarded = false;
                r
            }
            Fields::Flat { .. } => {
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.flat(inst, name);
                self.guarded = false;
                r
            }
            Fields::Exp { done, .. } => {
                let done = *done;
                let g = self.guard(inst)?;
                self.guarded = g;
                let r = self.export(inst);
                self.guarded = false;
                r?;
                // A pixel's last export hands EXEC to the pipeline as the
                // valid mask: a lane outside it is killed (no colour, depth
                // or stencil is written for it), which is how the console's
                // `discard` and `clip` end (origin/main `emit/lower.rs`
                // `export`: `discard(!guard)` at the `done` export).
                if done && self.native.stage == HwStage::Ps {
                    self.close();
                    let a = self.active(inst)?;
                    self.kill_inactive(a);
                }
                Ok(())
            }
        }
    }
}
