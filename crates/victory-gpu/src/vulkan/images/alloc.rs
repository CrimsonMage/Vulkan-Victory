//! Card memory for images: 256 MiB blocks with first-fit free lists;
//! images over 64 MiB get memory of their own. Images over the same title
//! memory share one piece (`PLACES`, `USERS`).

use std::collections::BTreeMap;
use std::sync::Mutex;

use victory_vk::ash::vk;

use super::super::{Backend, lock};


const BLOCK: u64 = 256 << 20;

pub(super) struct Block {
    memory: vk::DeviceMemory,
    type_index: u32,
    free: Vec<(u64, u64)>,
}

pub(super) fn allocate(
    b: &Backend,
    blocks: &mut Vec<Block>,
    needs: vk::MemoryRequirements,
) -> Option<(vk::DeviceMemory, u64)> {
    let type_index = b.host.memory_type(
        needs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    if needs.size > BLOCK / 4 {
        // SAFETY: valid allocate info.
        let m = unsafe {
            b.host.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(type_index),
                None,
            )
        }
        .ok()?;
        victory_vk::account::allocated(&b.host, m, type_index, needs.size, victory_vk::account::Kind::Images);
        return Some((m, u64::MAX));
    }
    let take = |blk: &mut Block| -> Option<u64> {
        let i = blk.free.iter().position(|&(at, len)| {
            let start = at.next_multiple_of(needs.alignment.max(1));
            start + needs.size <= at + len
        })?;
        let (at, len) = blk.free[i];
        let start = at.next_multiple_of(needs.alignment.max(1));
        let end = start + needs.size;
        blk.free.remove(i);
        if start > at {
            blk.free.push((at, start - at));
        }
        if end < at + len {
            blk.free.push((end, at + len - end));
        }
        Some(start)
    };
    for blk in blocks.iter_mut().filter(|k| k.type_index == type_index) {
        if let Some(at) = take(blk) {
            return Some((blk.memory, at));
        }
    }
    // SAFETY: valid allocate info.
    let memory = unsafe {
        b.host.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(BLOCK)
                .memory_type_index(type_index),
            None,
        )
    }
    .ok()?;
    victory_vk::account::allocated(&b.host, memory, type_index, BLOCK, victory_vk::account::Kind::Images);
    let mut blk = Block {
        memory,
        type_index,
        free: vec![(0, BLOCK)],
    };
    let at = take(&mut blk)?;
    blocks.push(blk);
    Some((memory, at))
}

pub(super) fn release(b: &Backend, blocks: &mut Vec<Block>, memory: vk::DeviceMemory, at: u64, size: u64) {
    if at == u64::MAX {
        victory_vk::account::freed(memory);
        // SAFETY: a dedicated allocation nothing uses any more.
        unsafe { b.host.device.free_memory(memory, None) };
        return;
    }
    if let Some(blk) = blocks.iter_mut().find(|k| k.memory == memory) {
        blk.free.push((at, size));
        // Merge neighbours.
        blk.free.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(blk.free.len());
        for &(a, l) in &blk.free {
            match merged.last_mut() {
                Some((pa, pl)) if *pa + *pl == a => *pl += l,
                _ => merged.push((a, l)),
            }
        }
        blk.free = merged;
    }
    // A block nothing uses any more goes back to the driver.
    if let Some(i) = blocks.iter().position(|k| k.memory == memory && k.free == [(0, BLOCK)]) {
        let blk = blocks.swap_remove(i);
        victory_vk::account::freed(blk.memory);
        // SAFETY: no image is bound to any of it.
        unsafe { b.host.device.free_memory(blk.memory, None) };
    }
}

/// MiB of card memory live images are bound to (aliases counted once).
pub(crate) fn used_mb() -> u64 {
    let mut seen = std::collections::HashSet::new();
    lock(&PLACES).values().filter(|p| seen.insert((p.0, p.1))).map(|p| p.2).sum::<u64>() >> 20
}

/// Where an image's memory is: (memory, offset or MAX for dedicated, size,
/// memory type).
pub(super) static PLACES: Mutex<BTreeMap<u64, (vk::DeviceMemory, u64, u64, u32)>> = Mutex::new(BTreeMap::new());
/// Images bound to each piece of card memory (aliases share one).
pub(super) static USERS: Mutex<BTreeMap<(vk::DeviceMemory, u64), u32>> = Mutex::new(BTreeMap::new());

/// The memory type `allocate` picks for `needs`.
pub(super) fn memory_type_of(b: &Backend, needs: vk::MemoryRequirements) -> u32 {
    b.host
        .memory_type(needs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::HOST_VISIBLE)
        .unwrap_or(0)
}

