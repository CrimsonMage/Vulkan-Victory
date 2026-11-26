//! What a SPIR-V module asks of the device, read from its words: the one
//! place the mesh output memory is counted, for the translator (which keeps
//! its modules under the budget) and the host (which checks them against
//! the device before compiling).
//!
//! Mesh output memory follows the Vulkan specification ("Mesh Shader
//! Output"): every output except the primitive indices counts, each user
//! output as whole locations (4 scalars per location, whatever components
//! it uses), each built-in as its scalars:
//!
//! `per_vertex × 4 × align(OutputVertices, g_v) + per_primitive × 4 ×
//! align(OutputPrimitivesEXT, g_p)`, scalars per vertex and per primitive.

use std::collections::{HashMap, HashSet};

/// SPIR-V execution models.
pub const MODEL_VERTEX: u32 = 0;
pub const MODEL_FRAGMENT: u32 = 4;
pub const MODEL_COMPUTE: u32 = 5;
pub const MODEL_MESH: u32 = 5365;

/// The mesh output budget the translator keeps every module under: the
/// smallest `maxMeshOutputMemorySize` of the cards in use (NVIDIA's 32 KiB)
/// at their output granularities (32 vertices, 32 primitives).
pub const MESH_OUTPUT_BUDGET: Budget = Budget { bytes: 32768, vertex_granularity: 32, primitive_granularity: 32 };

/// A device's mesh output memory and its granularities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub bytes: u32,
    pub vertex_granularity: u32,
    pub primitive_granularity: u32,
}

/// Mesh output memory in bytes: `per_vertex` and `per_primitive` scalars,
/// the declared maxima, at a device's granularities.
pub fn mesh_output_bytes(per_vertex: u32, per_primitive: u32, vertices: u32, primitives: u32, b: Budget) -> u64 {
    let align = |v: u32, to: u32| u64::from(v.div_ceil(to.max(1)) * to.max(1));
    u64::from(per_vertex) * 4 * align(vertices, b.vertex_granularity)
        + u64::from(per_primitive) * 4 * align(primitives, b.primitive_granularity)
}

/// What one module uses (from its words).
#[derive(Clone, Debug, Default)]
pub struct Usage {
    /// Execution model of the first entry point.
    pub model: u32,
    pub local_size: [u32; 3],
    /// Mesh: `OutputVertices`, `OutputPrimitivesEXT`.
    pub out_vertices: u32,
    pub out_primitives: u32,
    /// Bytes of `Workgroup` variables.
    pub workgroup_bytes: u64,
    /// Mesh: output scalars per vertex and per primitive, as the
    /// specification counts them (primitive indices left out).
    pub per_vertex_scalars: u32,
    pub per_primitive_scalars: u32,
    /// Output locations used (vertex, mesh).
    pub out_locations: u32,
    pub capabilities: Vec<u32>,
}

impl Usage {
    /// The mesh output memory at a device's granularities.
    pub fn mesh_output_bytes(&self, b: Budget) -> u64 {
        mesh_output_bytes(self.per_vertex_scalars, self.per_primitive_scalars, self.out_vertices, self.out_primitives, b)
    }
}

/// Reads `words` (a module the translator made). Unknown instructions are
/// skipped; a malformed module gives what was read up to the fault.
pub fn usage(words: &[u32]) -> Usage {
    let mut p = Parse::default();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            break;
        }
        p.instruction(words[at] & 0xffff, &words[at + 1..at + count]);
        at += count;
    }
    p.finish()
}

const BI_PRIMITIVE_ID: u32 = 7;
const BI_LAYER: u32 = 9;
const BI_VIEWPORT_INDEX: u32 = 10;
const BI_PRIMITIVE_POINT_INDICES: u32 = 5294;
const BI_PRIMITIVE_LINE_INDICES: u32 = 5295;
const BI_PRIMITIVE_TRIANGLE_INDICES: u32 = 5296;
const BI_CULL_PRIMITIVE: u32 = 5299;

/// What the parse keeps of a module: type sizes, array shapes, constants,
/// pointers, decorations and variables.
#[derive(Default)]
struct Parse {
    u: Usage,
    entry: bool,
    /// Type -> bytes (booleans count 4: one scalar).
    size: HashMap<u32, u64>,
    /// Array -> (element, length).
    array_len: HashMap<u32, (u32, u64)>,
    constant: HashMap<u32, u64>,
    /// Pointer type -> (storage class, pointee).
    pointer: HashMap<u32, (u32, u32)>,
    builtin: HashMap<u32, u32>,
    per_primitive: HashSet<u32>,
    location: HashSet<u32>,
    /// (id, pointer type, storage class).
    variables: Vec<(u32, u32, u32)>,
}

impl Parse {
    fn size_of(&self, id: u32) -> u64 {
        self.size.get(&id).copied().unwrap_or(4)
    }

    fn instruction(&mut self, op: u32, w: &[u32]) {
        let arg = |i: usize| w.get(i).copied().unwrap_or(0);
        match op {
            17 => self.u.capabilities.push(arg(0)),
            15 if !self.entry => {
                self.entry = true;
                self.u.model = arg(0);
            }
            16 => self.execution_mode(arg(1), w),
            71 => self.decorate(arg(0), arg(1), arg(2)),
            20..=32 => self.type_of(op, w),
            43 => {
                self.constant.insert(arg(1), u64::from(arg(2)));
            }
            59 => self.variables.push((arg(1), arg(0), arg(2))),
            _ => {}
        }
    }

    fn execution_mode(&mut self, mode: u32, w: &[u32]) {
        let arg = |i: usize| w.get(i).copied().unwrap_or(0);
        match mode {
            17 => self.u.local_size = [arg(2), arg(3), arg(4)],
            26 => self.u.out_vertices = arg(2),
            5270 => self.u.out_primitives = arg(2),
            _ => {}
        }
    }

    fn decorate(&mut self, id: u32, decoration: u32, value: u32) {
        match decoration {
            11 => {
                self.builtin.insert(id, value);
            }
            30 => {
                self.location.insert(id);
            }
            5271 => {
                self.per_primitive.insert(id);
            }
            _ => {}
        }
    }

    fn type_of(&mut self, op: u32, w: &[u32]) {
        let arg = |i: usize| w.get(i).copied().unwrap_or(0);
        let bytes = match op {
            20 => 4,
            21 | 22 => u64::from(arg(1) / 8),
            23 | 24 => self.size_of(arg(1)) * u64::from(arg(2)),
            28 => {
                let len = self.constant.get(&arg(2)).copied().unwrap_or(1);
                self.array_len.insert(arg(0), (arg(1), len));
                self.size_of(arg(1)) * len
            }
            30 => w[1..].iter().map(|&m| self.size_of(m)).sum(),
            32 => {
                self.pointer.insert(arg(0), (arg(1), arg(2)));
                return;
            }
            _ => return,
        };
        self.size.insert(arg(0), bytes);
    }

    fn finish(mut self) -> Usage {
        for (id, ptr, storage) in std::mem::take(&mut self.variables) {
            let Some(&(_, pointee)) = self.pointer.get(&ptr) else { continue };
            match storage {
                4 => self.u.workgroup_bytes += self.size.get(&pointee).copied().unwrap_or(0),
                3 => self.output(id, pointee),
                _ => {}
            }
        }
        self.u
    }

    /// An output variable: counted as a location; for mesh programs, its
    /// scalars per vertex or per primitive (one element of the array).
    fn output(&mut self, id: u32, pointee: u32) {
        let user = self.location.contains(&id);
        self.u.out_locations += u32::from(user);
        if self.u.model != MODEL_MESH {
            return;
        }
        let element = self.array_len.get(&pointee).map_or(pointee, |&(e, _)| e);
        let builtin = self.builtin.get(&id).copied();
        if matches!(builtin, Some(BI_PRIMITIVE_POINT_INDICES | BI_PRIMITIVE_LINE_INDICES | BI_PRIMITIVE_TRIANGLE_INDICES)) {
            return;
        }
        let bytes = self.size.get(&element).copied().unwrap_or(0);
        // user outputs: whole locations (a 64-bit vec3/vec4 takes two)
        let scalars = if user { bytes.div_ceil(16).max(1) * 4 } else { bytes.div_ceil(4) };
        let scalars = u32::try_from(scalars).unwrap_or(u32::MAX);
        let prim = self.per_primitive.contains(&id)
            || matches!(builtin, Some(BI_CULL_PRIMITIVE | BI_PRIMITIVE_ID | BI_LAYER | BI_VIEWPORT_INDEX));
        if prim {
            self.u.per_primitive_scalars += scalars;
        } else {
            self.u.per_vertex_scalars += scalars;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_like_the_specification() {
        // 256 vertices of position + 7 locations, 256 primitives of a cull
        // flag: (4 + 28) × 4 × 256 + 1 × 4 × 256
        assert_eq!(mesh_output_bytes(32, 1, 256, 256, MESH_OUTPUT_BUDGET), 32768 + 1024);
        // counts rounded up to the granularity
        assert_eq!(mesh_output_bytes(4, 1, 33, 1, MESH_OUTPUT_BUDGET), 16 * 64 + 4 * 32);
    }
}
