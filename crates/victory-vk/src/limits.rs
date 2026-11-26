//! What a SPIR-V module asks of the device, checked against the device's
//! limits before the module is compiled.
//!
//! A module over a limit (a console mesh program with 256 threads where
//! NVIDIA allows 128, 64 KiB of LDS where it allows 48) is invalid usage:
//! the driver may refuse it, or fault inside the compile, which under
//! winevulkan ends the process silently (`ExitProcess(3)` after an
//! exception in a Unix call). So the host reads the few numbers that
//! matter from the words and refuses such a module with a reason.

use ash::vk;

/// What one module uses (from its words).
#[derive(Clone, Debug, Default)]
pub struct Usage {
    /// Execution model of the first entry point (SPIR-V numbering).
    pub model: u32,
    pub local_size: [u32; 3],
    /// Mesh: `OutputVertices`, `OutputPrimitivesEXT`.
    pub out_vertices: u32,
    pub out_primitives: u32,
    /// Bytes of `Workgroup` variables.
    pub workgroup_bytes: u64,
    /// Mesh: bytes of output per vertex and per primitive (each array's
    /// element; user outputs count 16 bytes per location).
    pub per_vertex_bytes: u64,
    pub per_primitive_bytes: u64,
    /// Output locations used (vertex, mesh).
    pub out_locations: u32,
    pub capabilities: Vec<u32>,
}

/// SPIR-V execution models and capabilities the check knows.
pub const MODEL_VERTEX: u32 = 0;
pub const MODEL_FRAGMENT: u32 = 4;
pub const MODEL_COMPUTE: u32 = 5;
pub const MODEL_MESH: u32 = 5365;
const CAP_STENCIL_EXPORT: u32 = 5013;
const CAP_FRAGMENT_BARYCENTRIC: u32 = 5284;
const CAP_MESH_SHADING: u32 = 5283;
const CAP_STORAGE_IMAGE_MULTISAMPLE: u32 = 27;

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

/// What the parse keeps of a module: type sizes, array shapes, constants,
/// pointers, decorations and variables.
#[derive(Default)]
struct Parse {
    u: Usage,
    entry: bool,
    size: HashMap<u32, u64>,
    /// Array -> (element, length).
    array_len: HashMap<u32, (u32, u64)>,
    constant: HashMap<u32, u64>,
    /// Pointer type -> (storage class, pointee).
    pointer: HashMap<u32, (u32, u32)>,
    builtin: HashMap<u32, u32>,
    per_primitive: HashSet<u32>,
    location: HashMap<u32, u32>,
    /// (id, pointer type, storage class).
    variables: Vec<(u32, u32, u32)>,
}

use std::collections::{HashMap, HashSet};

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
                self.location.insert(id, value);
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
    /// bytes per vertex or per primitive.
    fn output(&mut self, id: u32, pointee: u32) {
        let user = self.location.contains_key(&id);
        self.u.out_locations += u32::from(user);
        if self.u.model != MODEL_MESH {
            return;
        }
        // Per-vertex or per-primitive arrays: one element each.
        let element = self.array_len.get(&pointee).map_or(pointee, |&(e, _)| e);
        let bytes = if user { 16 } else { self.size.get(&element).copied().unwrap_or(0) };
        let prim = self.per_primitive.contains(&id)
            || matches!(self.builtin.get(&id), Some(5296 | 5299 | 7 | 9 | 10))
            || self.per_primitive.contains(&element);
        if prim {
            self.u.per_primitive_bytes += bytes;
        } else {
            self.u.per_vertex_bytes += bytes;
        }
    }
}

/// The device limits the check compares with.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub compute_invocations: u32,
    pub compute_size: [u32; 3],
    pub compute_shared: u32,
    pub mesh_invocations: u32,
    pub mesh_size: [u32; 3],
    pub mesh_shared: u32,
    pub mesh_output_memory: u32,
    pub mesh_vertices: u32,
    pub mesh_primitives: u32,
    pub mesh_components: u32,
    pub mesh_vertex_granularity: u32,
    pub mesh_primitive_granularity: u32,
    pub vertex_components: u32,
    pub mesh: bool,
    pub stencil_export: bool,
    pub barycentric: bool,
    pub storage_image_multisample: bool,
}

impl Limits {
    pub fn of(host: &crate::host::Host) -> Limits {
        let mut mesh = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default();
        if host.caps.mesh_shader {
            p2 = p2.push_next(&mut mesh);
        }
        // SAFETY: a valid chain for the device the host made.
        unsafe {
            host.instance
                .get_physical_device_properties2(host.physical, &mut p2)
        };
        let l = p2.properties.limits;
        Limits {
            compute_invocations: l.max_compute_work_group_invocations,
            compute_size: l.max_compute_work_group_size,
            compute_shared: l.max_compute_shared_memory_size,
            mesh_invocations: mesh.max_mesh_work_group_invocations,
            mesh_size: mesh.max_mesh_work_group_size,
            mesh_shared: mesh.max_mesh_shared_memory_size,
            mesh_output_memory: mesh.max_mesh_output_memory_size,
            mesh_vertices: mesh.max_mesh_output_vertices,
            mesh_primitives: mesh.max_mesh_output_primitives,
            mesh_components: mesh.max_mesh_output_components,
            mesh_vertex_granularity: mesh.mesh_output_per_vertex_granularity.max(1),
            mesh_primitive_granularity: mesh.mesh_output_per_primitive_granularity.max(1),
            vertex_components: l.max_vertex_output_components,
            mesh: host.caps.mesh_shader,
            stencil_export: host.caps.has(ash::ext::shader_stencil_export::NAME),
            barycentric: host.caps.has(ash::khr::fragment_shader_barycentric::NAME),
            storage_image_multisample: host.caps.core.shader_storage_image_multisample
                == vk::TRUE,
        }
    }
}

fn align(v: u64, to: u32) -> u64 {
    v.div_ceil(u64::from(to)) * u64::from(to)
}

impl Usage {
    /// The mesh output memory as the spec counts it
    /// (`maxMeshOutputMemorySize`: per-vertex and per-primitive sizes times
    /// the counts rounded up to the granularities).
    pub fn mesh_output_bytes(&self, l: &Limits) -> u64 {
        align(u64::from(self.out_vertices), l.mesh_vertex_granularity) * self.per_vertex_bytes
            + align(u64::from(self.out_primitives), l.mesh_primitive_granularity)
                * self.per_primitive_bytes
    }

    /// Why the device cannot take this module, if it cannot.
    pub fn refused(&self, l: &Limits) -> Option<String> {
        self.capability_refused(l).or_else(|| match self.model {
            MODEL_COMPUTE => self.compute_refused(l),
            MODEL_MESH => self.mesh_refused(l),
            MODEL_VERTEX if self.out_locations * 4 > l.vertex_components => Some(format!(
                "vertex output components {} over the device's {}",
                self.out_locations * 4,
                l.vertex_components
            )),
            _ => None,
        })
    }

    fn capability_refused(&self, l: &Limits) -> Option<String> {
        let needs = [
            (CAP_STENCIL_EXPORT, l.stencil_export, "stencil export without VK_EXT_shader_stencil_export"),
            (CAP_FRAGMENT_BARYCENTRIC, l.barycentric, "per-vertex attributes without VK_KHR_fragment_shader_barycentric"),
            (CAP_MESH_SHADING, l.mesh, "a mesh program without mesh shaders"),
            (CAP_STORAGE_IMAGE_MULTISAMPLE, l.storage_image_multisample, "multisampled storage images without the feature"),
        ];
        needs
            .iter()
            .find(|(cap, has, _)| !has && self.capabilities.contains(cap))
            .map(|(_, _, why)| (*why).to_owned())
    }

    fn invocations(&self) -> u64 {
        self.local_size.iter().map(|&v| u64::from(v.max(1))).product()
    }

    fn group_over(&self, size: [u32; 3]) -> bool {
        (0..3).any(|i| self.local_size[i] > size[i])
    }

    fn compute_refused(&self, l: &Limits) -> Option<String> {
        if self.invocations() > u64::from(l.compute_invocations) || self.group_over(l.compute_size) {
            return Some(format!("compute group {:?} over the device's {} invocations", self.local_size, l.compute_invocations));
        }
        (self.workgroup_bytes > u64::from(l.compute_shared))
            .then(|| format!("compute shared memory {} bytes over the device's {}", self.workgroup_bytes, l.compute_shared))
    }

    fn mesh_refused(&self, l: &Limits) -> Option<String> {
        if self.invocations() > u64::from(l.mesh_invocations) || self.group_over(l.mesh_size) {
            return Some(format!("mesh group {:?} over the device's {} invocations", self.local_size, l.mesh_invocations));
        }
        if self.out_vertices > l.mesh_vertices || self.out_primitives > l.mesh_primitives {
            return Some(format!(
                "mesh outputs {} vertices / {} primitives over the device's {} / {}",
                self.out_vertices, self.out_primitives, l.mesh_vertices, l.mesh_primitives
            ));
        }
        if self.workgroup_bytes > u64::from(l.mesh_shared) {
            return Some(format!("mesh shared memory {} bytes over the device's {}", self.workgroup_bytes, l.mesh_shared));
        }
        let out = self.mesh_output_bytes(l);
        if out > u64::from(l.mesh_output_memory) {
            return Some(format!("mesh output memory {out} bytes over the device's {}", l.mesh_output_memory));
        }
        (self.out_locations * 4 > l.mesh_components)
            .then(|| format!("mesh output components {} over the device's {}", self.out_locations * 4, l.mesh_components))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_probe_program() {
        let u = usage(crate::compute::SPIRV);
        assert_eq!(u.model, MODEL_COMPUTE);
        assert!(u.local_size[0] >= 1);
    }

    #[test]
    fn refuses_a_big_mesh_group() {
        let u = Usage {
            model: MODEL_MESH,
            local_size: [256, 1, 1],
            ..Default::default()
        };
        let l = Limits {
            mesh: true,
            mesh_invocations: 128,
            mesh_size: [128, 128, 128],
            mesh_vertices: 256,
            mesh_primitives: 256,
            mesh_shared: 28672,
            mesh_output_memory: 32768,
            mesh_components: 128,
            mesh_vertex_granularity: 32,
            mesh_primitive_granularity: 32,
            ..Default::default()
        };
        assert!(u.refused(&l).unwrap().contains("mesh group"));
    }
}
