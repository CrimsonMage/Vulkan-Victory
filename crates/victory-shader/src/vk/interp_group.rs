//! A whole compute or mesh group in the interpreter, as the console runs
//! it and as the card does.
//!
//! [`Mode::Waves`] (the reference): one [`Machine`] per console wave, a
//! wave64's 64 invocations stepped together; its two host subgroups keep
//! their own subgroup operations, but an exchange ([`Op::Xchg`]) is taken
//! at once and every branch must agree across the whole wave.
//!
//! [`Mode::Halves`] (the card): one machine per host subgroup of 32. A
//! wave64's two halves run on their own, each with its own control flow,
//! and see each other only through title memory, workgroup memory and the
//! exchanges the lift put in (lift_wave.rs), which are evaluated here
//! between the halves as emit_xchg.rs makes the card do it: a half waits
//! at its k-th exchange until its partner is at its k-th, each gives the
//! word its invocations hold (one word: it must be the same in all of
//! them) and both get (lower half's, upper half's). What the card's forms
//! of the exchange add (group barriers, or arrival counters with a bounded
//! wait) is not modelled: a half whose partner is not at the same exchange
//! (another node, a barrier, or returned) is an error here, where the card
//! would exchange the wrong word, hang at a barrier or wait out its bound.
//!
//! Between two meetings nothing orders the halves; `upper_first` runs the
//! upper half of every wave before the lower, so a result that depends on
//! that order shows as a difference between the two schedules.
//!
//! In both modes a console barrier (`s_barrier`) is passed when every unit
//! that still runs is at one (the console drops waves that ended), and all
//! units share title memory and workgroup memory.

use super::interp::{Machine, Memory, Stop};
use super::tir::{Func, Op, V};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Console waves, each stepped as one (the reference).
    Waves,
    /// Host subgroups of 32, the halves of a wave64 on their own.
    Halves,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum At {
    Ready,
    Xchg(V, u64),
    Barrier(V),
    Done,
}

/// A group's units and the memory they share.
pub struct Group<'a> {
    f: &'a Func,
    pub units: Vec<Machine<'a>>,
    at: Vec<At>,
    /// The halves of a wave64 are units of their own.
    halves: bool,
    /// Run the upper half of each wave before the lower.
    pub upper_first: bool,
    pub mem: Memory,
    pub lds: Vec<u32>,
    /// Exchanges the halves made (pairs).
    pub exchanges: u64,
}

impl<'a> Group<'a> {
    /// A group of `threads` host invocations (`Lift::host_threads`) of a
    /// program for waves of 64 (`wave64`) or 32. `setup` prepares each unit
    /// (push constants, limits); its wave index and memory are set here.
    pub fn new(f: &'a Func, wave64: bool, threads: u32, mode: Mode, mem: Memory, setup: impl Fn(&mut Machine<'a>)) -> Self {
        let paired = wave64 && threads > 32;
        let halves = paired && mode == Mode::Halves;
        let lanes = if paired && !halves { 64 } else { 32 };
        let units: Vec<Machine<'a>> = (0..threads.max(1).div_ceil(lanes))
            .map(|u| {
                let mut m = Machine::new(f, lanes as usize, Memory::default());
                setup(&mut m);
                // (a unit of 32 is a host subgroup, one of 64 a console wave)
                m.wave = u;
                let last = threads.max(1) - u * lanes;
                m.live = (0..lanes).map(|l| l < last).collect();
                m
            })
            .collect();
        let lds = vec![0; units[0].lds.len()];
        Group { f, at: vec![At::Ready; units.len()], units, halves, upper_first: false, mem, lds, exchanges: 0 }
    }

    /// Runs the group to its end.
    pub fn run(&mut self) -> Stop {
        loop {
            for u in self.order() {
                if self.at[u] == At::Ready
                    && let Err(stop) = self.advance(u)
                {
                    return stop;
                }
            }
            if self.at.iter().all(|a| *a == At::Done) {
                return Stop::Returned;
            }
            match self.meet() {
                Ok(true) => {}
                Ok(false) => return Stop::Error(format!("the group cannot go on: its units are at {:?}", self.at)),
                Err(e) => return Stop::Error(e),
            }
        }
    }

    /// The units in the order they run between meetings.
    fn order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.units.len()).collect();
        if self.halves && self.upper_first {
            for pair in order.chunks_mut(2) {
                pair.reverse();
            }
        }
        order
    }

    /// Runs unit `u` to its next exchange, barrier or end, on the shared
    /// memory.
    fn advance(&mut self, u: usize) -> Result<(), Stop> {
        let m = &mut self.units[u];
        std::mem::swap(&mut m.mem, &mut self.mem);
        std::mem::swap(&mut m.lds, &mut self.lds);
        let (node, stop) = m.step(self.halves, true);
        std::mem::swap(&mut m.mem, &mut self.mem);
        std::mem::swap(&mut m.lds, &mut self.lds);
        self.at[u] = match (node, stop) {
            (Some(v), _) if self.f.node(v).op == Op::Xchg => At::Xchg(v, m.own_word(v).map_err(|e| Stop::Error(format!("unit {u}: {e}")))?),
            (Some(v), _) => At::Barrier(v),
            (None, Stop::Returned) => At::Done,
            (None, Stop::Error(e)) => return Err(Stop::Error(format!("unit {u}: {e}"))),
            (None, stop) => return Err(stop),
        };
        Ok(())
    }

    /// What the waiting units can do together: the pairs at an exchange
    /// exchange, and when every unit still running is at a barrier they
    /// pass it. Whether anything went on.
    fn meet(&mut self) -> Result<bool, String> {
        let mut progress = false;
        for w in (0..self.units.len()).step_by(2).filter(|_| self.halves) {
            let lo = self.at[w];
            // a half whose partner is not in the group exchanges with nobody
            let Some(&hi) = self.at.get(w + 1) else {
                if let At::Xchg(a, x) = lo {
                    self.units[w].settle(a, [x, 0]);
                    self.at[w] = At::Ready;
                    progress = true;
                }
                continue;
            };
            match (lo, hi) {
                (At::Xchg(a, x), At::Xchg(b, y)) if a == b => {
                    self.units[w].settle(a, [x, y]);
                    self.units[w + 1].settle(b, [x, y]);
                    (self.at[w], self.at[w + 1]) = (At::Ready, At::Ready);
                    self.exchanges += 1;
                    progress = true;
                }
                (At::Xchg(..), _) | (_, At::Xchg(..)) => {
                    return Err(format!("the halves of wave {} do not meet: the lower is at {lo:?}, the upper at {hi:?}", w / 2));
                }
                _ => {}
            }
        }
        let waiting = |a: &At| matches!(a, At::Barrier(_));
        if self.at.iter().any(waiting) && self.at.iter().all(|a| waiting(a) || *a == At::Done) {
            for u in 0..self.units.len() {
                if let At::Barrier(v) = self.at[u] {
                    self.units[u].settle(v, [0, 0]);
                    self.at[u] = At::Ready;
                }
            }
            progress = true;
        }
        Ok(progress)
    }
}
