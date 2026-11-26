//! 16-bit interface values packed two per dword (the console's fp16
//! parameters), mapped to signature rows and columns.
//!
//! A packed parameter carries the halves of its first 16-bit row, then of a
//! second row, then elements that share a register behind another one
//! (`packed_layout`): half `h` of a parameter (dword `h / 2`, the high half
//! when odd) is one column of one signature row. The Vulkan module keeps
//! every element at its semantic's location as 32-bit floats, so both
//! stages only need the row and column of each half.

use std::collections::HashMap;

use crate::native::{StageIo, Usage};

/// What the packing needs of a signature row.
#[derive(Debug, Clone, Copy)]
pub struct Elem {
    pub row: u8,
    pub col: u8,
    pub cols: u8,
    /// A 16-bit (minimum precision or native 16-bit) element.
    pub half: bool,
    /// A user semantic (no system value).
    pub user: bool,
}

type Layout = Vec<(u8, Vec<(u8, u8, u8, u8)>)>;

/// Per packed parameter (first row): (dword, row, first column, columns)
/// of the elements it carries, in order.
fn packed_layout(elements: &[Elem], packs: &[(u8, Option<u8>)]) -> Layout {
    let of_row = |r: u8| {
        let mut of: Vec<&Elem> = elements.iter().filter(|e| e.row == r).collect();
        of.sort_by_key(|e| e.col);
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
                let Some((lead, more)) = of.split_first() else {
                    continue;
                };
                placed.push((at, r, lead.col, lead.cols));
                at += lead.cols.div_ceil(2);
                behind.extend(more.iter().map(|e| (r, e.col, e.cols)));
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

/// Where half `h` of the parameter whose first row is `a` comes from.
fn packed_half(layout: &Layout, a: u8, h: u8) -> Option<(u8, u8)> {
    let (_, of) = layout.iter().find(|p| p.0 == a)?;
    of.iter().find_map(|&(dword, row, col, cols)| {
        let k = h.checked_sub(2 * dword)?;
        (k < cols).then_some((row, col + k))
    })
}

/// (row, half) -> (row, column) for the 16-bit rows of a stage's
/// interface: vertex/mesh outputs or pixel inputs. Rows absent are 32-bit.
pub fn halves(io: &StageIo, elements: &[Elem]) -> HashMap<(u8, u8), (u8, u8)> {
    let rows = match io {
        StageIo::Vs { outputs, .. } | StageIo::Ms(crate::native::MeshIo { outputs, .. }) => output_rows(outputs, elements),
        StageIo::Ps { inputs, .. } => input_rows(inputs, elements),
        _ => Vec::new(),
    };
    let packs: Vec<(u8, Option<u8>)> = rows.iter().filter(|r| r.1).map(|r| (r.0, r.2)).collect();
    let layout = packed_layout(elements, &packs);
    let mut out = HashMap::new();
    for &(row, pack2, _) in &rows {
        for h in 0..8u8 {
            let src = if pack2 { packed_half(&layout, row, h) } else { h.is_multiple_of(2).then_some((row, h / 2)) };
            if let Some(s) = src {
                out.insert((row, h), s);
            }
        }
    }
    out
}

/// The 16-bit user rows of `elements` that `taken` leaves free, in order.
fn free_rows(elements: &[Elem], taken: &dyn Fn(u8) -> bool) -> Vec<u8> {
    let mut free: Vec<u8> = elements.iter().filter(|e| e.half && e.user).map(|e| e.row).filter(|&r| !taken(r)).collect();
    free.sort_unstable();
    free.dedup();
    free
}

/// Vertex/mesh 16-bit rows: (row, two per dword, second row). A packed row's
/// partner is the next free 16-bit row.
fn output_rows(outputs: &[crate::native::VsOutput], elements: &[Elem]) -> Vec<(u8, bool, Option<u8>)> {
    let per_vertex: Vec<_> = outputs.iter().filter(|o| o.param & 0x80 == 0).collect();
    let mut free = free_rows(elements, &|r| per_vertex.iter().any(|o| o.row == r));
    let mut fp16: Vec<_> = per_vertex.iter().filter(|o| o.fp16).collect();
    fp16.sort_by_key(|o| o.row);
    let mut rows = Vec::new();
    for o in fp16 {
        let b = o.pack2.then(|| free.iter().position(|&r| r > o.row)).flatten().map(|i| free.remove(i));
        rows.push((o.row, o.pack2, b));
    }
    rows
}

/// Pixel 16-bit rows; a packed row the metadata names no partner for takes
/// the next free one.
fn input_rows(inputs: &[crate::native::PsInput], elements: &[Elem]) -> Vec<(u8, bool, Option<u8>)> {
    let packed: Vec<_> = inputs.iter().filter(|i| i.fp16 && i.pack2).collect();
    let mut free = free_rows(elements, &|r| {
        inputs.iter().any(|i| i.row == r) || packed.iter().any(|i| i.row2 != i.row && i.row2 == r)
    });
    let mut missing: Vec<_> = packed.iter().filter(|i| i.row2 == i.row).collect();
    missing.sort_by_key(|i| i.row);
    let mut partners = HashMap::new();
    for i in missing {
        if let Some(k) = free.iter().position(|&r| r > i.row) {
            partners.insert(i.row, free.remove(k));
        }
    }
    inputs
        .iter()
        .filter(|i| i.fp16)
        .map(|i| {
            let b = if i.row2 != i.row { Some(i.row2) } else { partners.get(&i.row).copied() };
            (i.row, i.pack2, if i.pack2 { b } else { None })
        })
        .collect()
}

/// The row a mesh parameter entry names: the 2020 compiler leaves a mesh
/// program's first entry zeroed (usage "position"); its row is then the
/// one no other entry names.
pub fn unnamed_row(io: &StageIo, elements: &[Elem]) -> Option<u8> {
    let StageIo::Ms(ms) = io else { return None };
    let named = |r: u8| ms.outputs.iter().any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && o.row == r);
    let mut free: Vec<u8> = elements.iter().filter(|e| e.user && !named(e.row)).map(|e| e.row).collect();
    free.sort_unstable();
    free.dedup();
    match free[..] {
        [r] => Some(r),
        _ => None,
    }
}
