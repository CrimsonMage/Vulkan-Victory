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
    use std::collections::HashMap;
    let mut u = Usage::default();
    // Type sizes (bytes), constants, pointers (storage class, pointee).
    let mut size: HashMap<u32, u64> = HashMap::new();
    let mut array_len: HashMap<u32, (u32, u64)> = HashMap::new(); // array -> (element, length)
    let mut constant: HashMap<u32, u64> = HashMap::new();
    let mut pointer: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut builtin: HashMap<u32, u32> = HashMap::new();
    let mut per_primitive: std::collections::HashSet<u32> = Default::default();
    let mut location: HashMap<u32, u32> = HashMap::new();
    let mut variables: Vec<(u32, u32, u32)> = Vec::new(); // (id, pointer type, storage)
    let mut struct_builtin: std::collections::HashSet<u32> = Default::default();
    let mut entry = false;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = words[at] & 0xffff;
        if count == 0 || at + count > words.len() {
            break;
        }
        let w = &words[at + 1..at + count];
        let arg = |i: usize| w.get(i).copied().unwrap_or(0);
        match op {
            17 => u.capabilities.push(arg(0)),
            15 if !entry => {
                entry = true;
                u.model = arg(0);
            }
            16 => match arg(1) {
                17 => u.local_size = [arg(2), arg(3), arg(4)],
                26 => u.out_vertices = arg(2),
                5270 => u.out_primitives = arg(2),
                _ => {}
            },
            71 => match arg(1) {
                11 => {
                    builtin.insert(arg(0), arg(2));
                }
                30 => {
                    location.insert(arg(0), arg(2));
                }
                5271 => {
                    per_primitive.insert(arg(0));
                }
                _ => {}
            },
            72 if arg(2) == 11 => {
                struct_builtin.insert(arg(0));
            }
            20 => {
                size.insert(arg(0), 4);
            }
            21 | 22 => {
                size.insert(arg(0), u64::from(arg(1) / 8));
            }
            23 | 24 => {
                let s = size.get(&arg(1)).copied().unwrap_or(4) * u64::from(arg(2));
                size.insert(arg(0), s);
            }
            28 => {
                let len = constant.get(&arg(2)).copied().unwrap_or(1);
                let s = size.get(&arg(1)).copied().unwrap_or(4) * len;
                size.insert(arg(0), s);
                array_len.insert(arg(0), (arg(1), len));
            }
            30 => {
                let s = w[1..].iter().map(|m| size.get(m).copied().unwrap_or(4)).sum();
                size.insert(arg(0), s);
            }
            32 => {
                pointer.insert(arg(0), (arg(1), arg(2)));
            }
            43 => {
                constant.insert(arg(1), u64::from(arg(2)));
            }
            59 => variables.push((arg(1), arg(0), arg(2))),
            _ => {}
        }
        at += count;
    }
    let mut locations = 0u32;
    for (id, ptr, storage) in variables {
        let Some(&(_, pointee)) = pointer.get(&ptr) else {
            continue;
        };
        match storage {
            4 => u.workgroup_bytes += size.get(&pointee).copied().unwrap_or(0),
            3 => {
                let user = location.contains_key(&id);
                if user {
                    locations += 1;
                }
                if u.model != MODEL_MESH {
                    continue;
                }
                // Per-vertex or per-primitive arrays: one element each.
                let element = match array_len.get(&pointee) {
                    Some(&(e, _)) => e,
                    None => pointee,
                };
                let bytes = if user {
                    16
                } else {
                    size.get(&element).copied().unwrap_or(0)
                };
                let prim = per_primitive.contains(&id)
                    || matches!(builtin.get(&id), Some(5296 | 5299 | 7 | 9 | 10))
                    || per_primitive.contains(&element);
                if prim {
                    u.per_primitive_bytes += bytes;
                } else {
                    u.per_vertex_bytes += bytes;
                }
            }
            _ => {}
        }
    }
    u.out_locations = locations;
    u
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
        let has = |c: u32| self.capabilities.contains(&c);
        if has(CAP_STENCIL_EXPORT) && !l.stencil_export {
            return Some("stencil export without VK_EXT_shader_stencil_export".into());
        }
        if has(CAP_FRAGMENT_BARYCENTRIC) && !l.barycentric {
            return Some("per-vertex attributes without VK_KHR_fragment_shader_barycentric".into());
        }
        if has(CAP_MESH_SHADING) && !l.mesh {
            return Some("a mesh program without mesh shaders".into());
        }
        if has(CAP_STORAGE_IMAGE_MULTISAMPLE) && !l.storage_image_multisample {
            return Some("multisampled storage images without the feature".into());
        }
        let n: u64 = self.local_size.iter().map(|&v| u64::from(v.max(1))).product();
        let over = |size: [u32; 3]| (0..3).any(|i| self.local_size[i] > size[i]);
        match self.model {
            MODEL_COMPUTE => {
                if n > u64::from(l.compute_invocations) || over(l.compute_size) {
                    return Some(format!(
                        "compute group {:?} over the device's {} invocations",
                        self.local_size, l.compute_invocations
                    ));
                }
                if self.workgroup_bytes > u64::from(l.compute_shared) {
                    return Some(format!(
                        "compute shared memory {} bytes over the device's {}",
                        self.workgroup_bytes, l.compute_shared
                    ));
                }
            }
            MODEL_MESH => {
                if n > u64::from(l.mesh_invocations) || over(l.mesh_size) {
                    return Some(format!(
                        "mesh group {:?} over the device's {} invocations",
                        self.local_size, l.mesh_invocations
                    ));
                }
                if self.out_vertices > l.mesh_vertices || self.out_primitives > l.mesh_primitives
                {
                    return Some(format!(
                        "mesh outputs {} vertices / {} primitives over the device's {} / {}",
                        self.out_vertices, self.out_primitives, l.mesh_vertices, l.mesh_primitives
                    ));
                }
                if self.workgroup_bytes > u64::from(l.mesh_shared) {
                    return Some(format!(
                        "mesh shared memory {} bytes over the device's {}",
                        self.workgroup_bytes, l.mesh_shared
                    ));
                }
                let out = self.mesh_output_bytes(l);
                if out > u64::from(l.mesh_output_memory) {
                    return Some(format!(
                        "mesh output memory {out} bytes over the device's {}",
                        l.mesh_output_memory
                    ));
                }
                if self.out_locations * 4 > l.mesh_components {
                    return Some(format!(
                        "mesh output components {} over the device's {}",
                        self.out_locations * 4,
                        l.mesh_components
                    ));
                }
            }
            MODEL_VERTEX if self.out_locations * 4 > l.vertex_components => {
                return Some(format!(
                    "vertex output components {} over the device's {}",
                    self.out_locations * 4,
                    l.vertex_components
                ));
            }
            _ => {}
        }
        None
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
