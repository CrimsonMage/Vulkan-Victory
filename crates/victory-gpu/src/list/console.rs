//! The console's own list methods past the PC's (D3D12.X), and the
//! queries, as the list methods behind the written-down calls (`thin`)
//! run them on the D3D12 path. Under `VICTORY_GPU=vulkan` the same calls
//! are recorded into Vulkan instead (`thin/console.rs`,
//! `vulkan/record/console.rs` and `vulkan/record/query.rs`).
//!
//! Slots are the older layout's (the newer one has them two later from
//! 59 on, as the others), as XCL found them (`hle/d3d12x/list_x.rs`):
//! ClearUnorderedAccessViewUint/Float #53/#54, FillMemoryWith{32,64,128}
//! BitValueX #61-#63, DrawIndexedX #81, SetPredicationBufferX #85,
//! Write{32,64}BitValueTopOfPipeX #88/#89, Write32BitValueBottomOfShaderX
//! #90, KickoffX #93, RSSetDepthBiasX #94.

use super::*;

/// The methods by their slot in the older layout.
pub(super) const METHODS: [(usize, *const ()); 12] = [
    (53, clear_uav as *const ()),          // 0x1a8 uint
    (54, clear_uav as *const ()),          // 0x1b0 float
    (61, fill32 as *const ()),             // 0x1e8
    (62, fill64 as *const ()),             // 0x1f0
    (63, fill128 as *const ()),            // 0x1f8
    (81, draw_indexed_x as *const ()),     // 0x288
    (85, set_predication as *const ()),    // 0x2a8
    (88, write32_top as *const ()),        // 0x2c0
    (89, write64_top as *const ()),        // 0x2c8
    (90, write32_bottom_of_shader as *const ()), // 0x2d0
    (93, kickoff as *const ()),            // 0x2e8
    (94, depth_bias as *const ()),         // 0x2f0
];

/// `Write64BitValue{Bottom,Top}OfPipeX` flags that write a timestamp in
/// place of the value: GPU (1 bottom, 4 top) or system (2 bottom, 8 top).
pub(crate) const TOP_GPU_TIMESTAMP: u32 = 4;
pub(crate) const TOP_SYSTEM_TIMESTAMP: u32 = 8;

/// A method the D3D12 path does not do (it is going, `docs/vulkan-host.md`):
/// logged once by name.
fn not_done(what: &'static str) {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&what) {
        seen.push(what);
        tracing::info!(method = what, "a console list method the D3D12 path does not do (the Vulkan backend does)");
    }
}

/// `ClearUnorderedAccessView{Uint,Float}(gpu handle, cpu handle, resource,
/// values, count, rects)`.
pub(super) extern "system" fn clear_uav(_: This) {
    not_done("ClearUnorderedAccessView");
}

/// `FillMemoryWith{32,64,128}BitValueX(address, size, value)`: on the D3D12
/// path the fill is done on the CPU once the list has run.
pub(super) extern "system" fn fill32(this: This, address: u64, size: u64, value: u32) {
    fill(this, address, size, pattern(&value.to_le_bytes()));
}
pub(super) extern "system" fn fill64(this: This, address: u64, size: u64, value: u64) {
    fill(this, address, size, pattern(&value.to_le_bytes()));
}
pub(super) extern "system" fn fill128(this: This, address: u64, size: u64, value: *const [u8; 16]) {
    // SAFETY: the title passes 16 bytes (null is tolerated).
    if let Some(value) = unsafe { value.as_ref() } {
        fill(this, address, size, *value);
    }
}

/// `bytes` repeated into 16.
pub(crate) fn pattern(bytes: &[u8]) -> [u8; 16] {
    std::array::from_fn(|i| bytes[i % bytes.len()])
}

pub(super) fn fill(this: This, address: u64, size: u64, pattern: [u8; 16]) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && address != 0
    {
        let len = size as usize;
        list.defer(move || {
            if crate::texture::committed(address as *const u8, len) {
                // SAFETY: committed title memory.
                let to = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, len) };
                for (k, b) in to.iter_mut().enumerate() {
                    *b = pattern[k % 16];
                }
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, address, size, pattern);
}

/// `DrawIndexedX(index address, index format, index count, flags)`.
pub(super) extern "system" fn draw_indexed_x(_: This) {
    not_done("DrawIndexedX");
}

/// `SetPredicationBufferX(&{address, operation})` (null: off).
pub(super) extern "system" fn set_predication(_: This) {
    not_done("SetPredicationBufferX");
}

/// `Write32BitValueTopOfPipeX(address, value, flags)`.
pub(super) extern "system" fn write32_top(this: This, address: u64, value: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}

/// `Write64BitValueTopOfPipeX(address, value, flags)`.
pub(super) extern "system" fn write64_top(this: This, address: u64, value: u64, flags: u32) {
    let what = if flags & TOP_GPU_TIMESTAMP != 0 {
        Written::GpuTimestamp
    } else if flags & TOP_SYSTEM_TIMESTAMP != 0 {
        Written::SystemTimestamp
    } else {
        Written::Value(value)
    };
    defer_write(this, address, what, 8);
}

/// `Write32BitValueBottomOfShaderX(address, value, stage)`.
pub(super) extern "system" fn write32_bottom_of_shader(this: This, address: u64, value: u32, _stage: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}

/// `KickoffX()`: the console's card starts on what the list holds so far.
/// Here the list runs whole at its submission.
pub(super) extern "system" fn kickoff(_: This) {}

/// `RSSetDepthBiasX(bias, clamp, slope)`.
pub(super) extern "system" fn depth_bias(_: This) {
    not_done("RSSetDepthBiasX");
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: on the D3D12 path queries
/// are answered when resolved.
pub(super) extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)` on the
/// D3D12 path, and for query types the Vulkan backend does not measure:
/// writes each query's result into the buffer once the list has run.
/// Timestamps (type 2) are the GPU clock then; occlusion counts (0) say a
/// million samples passed and binary occlusion (1) visible, so nothing is
/// culled by a query.
/// shortcut: plausible values, not measured ones (time spans come out as
/// the submission's length; pipeline statistics as zeros).
pub(crate) extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    timed!(Copy);
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    if offset + u64::from(count) * bytes > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_repeat_their_value() {
        assert_eq!(pattern(&0x0403_0201u32.to_le_bytes())[4..8], [1, 2, 3, 4]);
        assert_eq!(pattern(&[9; 16]), [9; 16]);
    }

    #[test]
    fn console_slots_are_free_in_both_layouts() {
        let others: Vec<usize> = METHODS_ALL.iter().map(|m| m.0).collect();
        for (slot, _) in METHODS {
            assert!(!others.contains(&slot), "slot {slot} taken twice");
        }
        let newer = list_table(true);
        assert_eq!(newer[newer_slot(61)], fill32 as *const ());
        assert_eq!(list_table(false)[53], clear_uav as *const ());
    }

    const METHODS_ALL: [(usize, *const ()); super::super::METHODS.len() + super::super::MORE_METHODS.len()] = {
        let mut all = [(0, std::ptr::null()); super::super::METHODS.len() + super::super::MORE_METHODS.len()];
        let mut i = 0;
        while i < all.len() {
            all[i] = if i < super::super::METHODS.len() {
                super::super::METHODS[i]
            } else {
                super::super::MORE_METHODS[i - super::super::METHODS.len()]
            };
            i += 1;
        }
        all
    };
}
