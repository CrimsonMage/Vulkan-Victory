//! Diagnostics: does the CPU change pool pages after the card copied them,
//! without an announcement?
//!
//! A page of the pool's GPU view is a card copy of its physical page, made
//! when the title maps it and at announced writes (DirectStorage, file
//! reads, card copies into title memory). A plain CPU store through the
//! page's CPU view afterwards is not seen: the card would read stale bytes
//! (on the console both views show the same memory). This counts how often
//! that happens, on a sample: one page in [`ONE_IN`] keeps the fingerprint
//! of its home at its last copy, and the 10 s line compares.

use std::collections::BTreeMap;
use std::sync::Mutex;

use victory_vk::pool::PAGE;

use super::{Backend, lock, read_title};

const ONE_IN: u64 = 8;
const MOST: usize = 4096;

/// Sampled page (its address in the GPU view) -> its home's fingerprint
/// when the card last copied it.
static SAMPLES: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());

fn fingerprint(b: &Backend, page: u64) -> Option<u64> {
    let bytes = read_title(b, page, PAGE).ok()?;
    let mut first = [0u8; 8];
    first.copy_from_slice(&blake3::hash(&bytes).as_bytes()[..8]);
    Some(u64::from_le_bytes(first))
}

/// The card copied (or lost) the pages of `[at, at + len)`: the sampled
/// ones among them take their fingerprint again.
pub(super) fn copied(b: &Backend, at: u64, len: u64) {
    if !cfg!(feature = "diagnostics") {
        return;
    }
    let first = at / PAGE;
    let last = (at + len.max(1) - 1) / PAGE;
    for n in (first..=last).filter(|n| n % ONE_IN == 0) {
        let page = n * PAGE;
        // read before the lock: `read_title` takes the memory lock
        let h = fingerprint(b, page);
        let mut s = lock(&SAMPLES);
        match h {
            Some(h) if s.len() < MOST || s.contains_key(&page) => {
                s.insert(page, h);
            }
            _ => {
                s.remove(&page);
            }
        }
    }
}

/// The 10 s line: sampled pages whose home no longer holds what the card
/// copied.
pub(super) fn report(b: &Backend) {
    if cfg!(feature = "diagnostics") {
        check(b);
    }
}

fn check(b: &Backend) {
    let samples: Vec<(u64, u64)> = lock(&SAMPLES).iter().map(|(&p, &h)| (p, h)).collect();
    let mut stale = Vec::new();
    for &(page, was) in &samples {
        match fingerprint(b, page) {
            Some(now) if now != was => stale.push(page),
            Some(_) => {}
            None => {
                lock(&SAMPLES).remove(&page);
            }
        }
    }
    let first: Vec<String> = stale.iter().take(6).map(|p| format!("{p:#x}")).collect();
    let (pages, runs) = super::memory(b).pools.iter().map(|p| p.runs()).fold((0, 0), |a, r| (a.0 + r.0, a.1 + r.1));
    tracing::info!(pages, runs, "Vulkan: the pool's GPU view: pages shown and the runs they make");
    tracing::info!(
        sampled = samples.len(),
        stale = stale.len(),
        first = %first.join(" "),
        "Vulkan: pool pages the CPU changed after the card copied them, unannounced (a sample: one page in 8)"
    );
}
