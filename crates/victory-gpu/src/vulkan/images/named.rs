//! Which textures the title's descriptor slots name.
//!
//! A texture the title places in memory of its own has no resource: the
//! title knows it by the view it writes into a slot of its descriptor
//! heap, and stops knowing it when it writes that slot again (the
//! texture moved, was unloaded, or the slot went to another). So a slot
//! names the texture of the last view call that wrote it (a copy call
//! carries the name along), and a texture no slot names for [`GRACE`]
//! frames is one the title has let go: its image goes (`fill::flush`).
//!
//! Before, an image went when a newer texture's whole console layout
//! reached over it. The title packs streamed textures by the levels they
//! hold, far closer than their layouts (`levels.rs`), so textures took
//! their neighbours' images out as they came: at the end of the world's
//! load 20,000 of the 48,000 textures the title's slots named had an
//! image, a different set in every run (run tx0, `docs/kiss.md`).
//!
//! shortcut: descriptor bytes the title copies itself are not seen. A
//! copy outlives its source's image only if the title rewrites the source
//! slot and keeps using the copy, which on the console reads memory the
//! title has given to another texture.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use super::Key;

/// Frames a texture stays after its last slot was written over: the
/// title often names a texture in one slot, writes that slot over and
/// names the texture in its own slot a few frames later, and frames in
/// flight may still read the old slot's copies.
const GRACE: u64 = 4;
/// Bytes of a view's slot (`crate::descriptor`).
const SLOT: u64 = 32;

#[derive(Default)]
struct Named {
    /// Title slot -> the texture its last view names.
    slots: HashMap<u64, Key>,
    /// Texture -> the slots naming it.
    holders: HashMap<Key, u32>,
    /// Textures that lost their last slot, and the frame.
    lost: VecDeque<(u64, Key)>,
}

static NAMED: Mutex<Option<Named>> = Mutex::new(None);

fn frame() -> u64 {
    super::super::present::FRAME.load(std::sync::atomic::Ordering::Relaxed)
}

impl Named {
    /// Slot `handle` names `key` now (`None`: no texture).
    fn set(&mut self, handle: u64, key: Option<Key>) {
        let old = match key {
            Some(k) => self.slots.insert(handle, k),
            None => self.slots.remove(&handle),
        };
        if old == key {
            return;
        }
        if let Some(k) = key {
            *self.holders.entry(k).or_insert(0) += 1;
        }
        let Some(old) = old else { return };
        let left = self.holders.get_mut(&old).map_or(0, |n| {
            *n = n.saturating_sub(1);
            *n
        });
        if left == 0 {
            self.holders.remove(&old);
            self.lost.push_back((frame(), old));
        }
    }
}

/// A view call wrote the title's slot `handle`: it names the texture
/// `key` (`None`: a buffer, or no texture).
pub(crate) fn shows(handle: u64, key: Option<Key>) {
    let mut guard = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    let named = guard.get_or_insert_with(Named::default);
    if key.is_some() || named.slots.contains_key(&handle) {
        named.set(handle, key);
    }
}

/// A copy call copied `count` slots from `src` to `dst`.
pub(crate) fn copied(dst: u64, src: u64, count: u32) {
    let mut guard = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    let Some(named) = guard.as_mut() else { return };
    if dst == src {
        return;
    }
    for i in 0..u64::from(count) {
        let key = named.slots.get(&(src + i * SLOT)).copied();
        if key.is_some() || named.slots.contains_key(&(dst + i * SLOT)) {
            named.set(dst + i * SLOT, key);
        }
    }
}

/// The textures that lost their last slot [`GRACE`] frames ago or more
/// and that no slot names now.
pub(super) fn let_go() -> Vec<Key> {
    let mut guard = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    let Some(named) = guard.as_mut() else { return Vec::new() };
    let now = frame();
    let mut gone = Vec::new();
    while let Some(&(since, key)) = named.lost.front() {
        if since + GRACE > now {
            break;
        }
        named.lost.pop_front();
        if !named.holders.contains_key(&key) {
            gone.push(key);
        }
    }
    gone
}

/// Diagnostics: the textures slots name for which `has_image` says no.
pub(super) fn without(has_image: impl Fn(&Key) -> bool) -> usize {
    let guard = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map_or(0, |n| n.holders.keys().filter(|k| !has_image(k)).count())
}

/// For the 10 s line: title slots that name a texture, textures named.
pub(super) fn counts() -> String {
    let guard = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map_or_else(String::new, |n| format!("{} textures in {} slots", n.holders.len(), n.slots.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(address: u64) -> Key {
        Key { address, dimension: 3, width: 4, height: 4, depth_or_array: 1, mips: 1, element: 4, samples: 1, depth: false }
    }

    #[test]
    fn a_texture_is_named_until_its_last_slot_is_written_over() {
        let mut n = Named::default();
        n.set(0x20, Some(key(1)));
        n.set(0x40, Some(key(1)));
        n.set(0x20, Some(key(2)));
        assert!(n.holders.contains_key(&key(1)) && n.lost.is_empty());
        // the same view again changes nothing
        n.set(0x40, Some(key(1)));
        assert_eq!(n.holders[&key(1)], 1);
        n.set(0x40, None);
        assert!(!n.holders.contains_key(&key(1)));
        assert_eq!(n.lost.len(), 1);
        assert_eq!(n.holders[&key(2)], 1);
    }
}
