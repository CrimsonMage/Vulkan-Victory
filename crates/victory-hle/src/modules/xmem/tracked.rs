//! What write tracking (`victory_gpu::track`) needs of the page table:
//! every address that shows the memory shown at a page, and word of views
//! that come and go.

use victory_gpu::track::{self, Shown};

use super::*;

/// The page table answers write tracking from now on.
pub(super) fn start() {
    track::views_with(shown);
}

/// What each page address shows, handed to `then` under the page table's
/// lock: nothing is mapped or unmapped while it protects the places.
fn shown(pages: &[usize], then: &mut dyn FnMut(&[Shown])) {
    let physical = physical();
    let all: Vec<Shown> = pages.iter().map(|&page| Shown { page, places: places(&physical, page) }).collect();
    then(&all);
}

/// Every address that shows the memory shown at `page`, with whether a CPU
/// view is there: for a physical page its views and the places of the
/// pool's GPU view that show it; for a page of a region, itself. None for
/// memory that is not `xmem`'s, and for a stand-in for a filler page (its
/// writes have their own count, `on_fault`).
fn places(physical: &Physical, page: usize) -> Vec<(usize, bool)> {
    let view = physical.views.range(..=page).next_back().filter(|(base, m)| page < **base + m.len);
    let key = match (physical.aliased.get(&page), view) {
        (Some(&key), _) => key,
        (None, Some((_, m))) if m.fill => return Vec::new(),
        (None, Some((base, m))) => (m.slice.section.id, m.index_at(*base, page)),
        (None, None) if reserved_by_xmem(page) => return vec![(page, true)],
        (None, None) => return Vec::new(),
    };
    let places = physical.shown.get(&key).map(Vec::as_slice).unwrap_or_default();
    places.iter().map(|&at| (at, !physical.aliased.contains_key(&at))).collect()
}

/// The title sets the protection of `[at, at + len)` itself (a commit at
/// an address of its choice, `VirtualProtect`, a free): no page there stays
/// write-protected by us or counted as such; the next flush protects
/// again what the card still has a copy of.
pub(super) fn title_protects(at: *const c_void, len: usize) {
    if !at.is_null() {
        track::released(at as usize, len);
    }
}

/// The view page at `at` is being unmapped (its place already out of
/// `others`, the places that still show the page): stores into it not yet
/// flushed are told through another place.
pub(super) fn unmapped(at: usize, others: Option<&Vec<usize>>) {
    if track::forget(at)
        && let Some(&other) = others.and_then(|o| o.first())
    {
        track::written(other);
    }
}

/// A view page was mapped at `at` over a page shown elsewhere too
/// (`others`): it is open, so if the card holds a copy the next flush
/// protects it.
pub(super) fn mapped(at: usize, others: &[usize]) {
    if !others.is_empty() {
        track::written(at);
    }
}
