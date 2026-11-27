//! What a SPIR-V module asks of the device, checked against the device's
//! limits before the module is compiled.
//!
//! A module over a limit (a console mesh program with 256 threads where
//! NVIDIA allows 128, 64 KiB of LDS where it allows 48) is invalid usage:
//! the driver may refuse it, or fault inside the compile, which under
//! winevulkan ends the process silently (`ExitProcess(3)` after an
//! exception in a Unix call). So the host reads the few numbers that
//! matter from the words with the translator's own reader
//! (`victory_shader::vk::usage`, which also counts the mesh output memory
//! the translator keeps its modules under) and refuses such a module with
//! a reason.

use ash::vk;
pub use victory_shader::vk::usage::{MODEL_COMPUTE, MODEL_FRAGMENT, MODEL_MESH, MODEL_VERTEX, Usage, usage};
use victory_shader::vk::usage::Budget;

/// SPIR-V capabilities the check knows.
const CAP_STENCIL_EXPORT: u32 = 5013;
const CAP_FRAGMENT_BARYCENTRIC: u32 = 5284;
const CAP_MESH_SHADING: u32 = 5283;
const CAP_STORAGE_IMAGE_MULTISAMPLE: u32 = 27;
const CAP_INT64_IMAGE: u32 = 5016;

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
    /// Mesh task groups a draw may take, per dimension and in all.
    pub mesh_groups: [u32; 3],
    pub mesh_groups_total: u32,
    pub vertex_components: u32,
    pub mesh: bool,
    pub stencil_export: bool,
    pub barycentric: bool,
    pub storage_image_multisample: bool,
    /// `shaderImageInt64Atomics` (ABI v5: 64-bit image atomics).
    pub image_int64_atomics: bool,
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
            mesh_groups: mesh.max_mesh_work_group_count,
            mesh_groups_total: mesh.max_mesh_work_group_total_count,
            vertex_components: l.max_vertex_output_components,
            mesh: host.caps.mesh_shader,
            stencil_export: host.caps.has(ash::ext::shader_stencil_export::NAME),
            barycentric: host.caps.has(ash::khr::fragment_shader_barycentric::NAME),
            storage_image_multisample: host.caps.core.shader_storage_image_multisample
                == vk::TRUE,
            image_int64_atomics: host.caps.image_int64_atomics,
        }
    }

    /// The device's mesh output memory and its granularities.
    pub fn mesh_budget(&self) -> Budget {
        Budget {
            bytes: self.mesh_output_memory,
            vertex_granularity: self.mesh_vertex_granularity,
            primitive_granularity: self.mesh_primitive_granularity,
        }
    }

    /// Why the device cannot take module `u`, if it cannot.
    pub fn refuses(&self, u: &Usage) -> Option<String> {
        self.capability_refused(u).or_else(|| match u.model {
            MODEL_COMPUTE => self.compute_refused(u),
            MODEL_MESH => self.mesh_refused(u),
            MODEL_VERTEX if u.out_locations * 4 > self.vertex_components => Some(format!(
                "vertex output components {} over the device's {}",
                u.out_locations * 4,
                self.vertex_components
            )),
            _ => None,
        })
    }

    fn capability_refused(&self, u: &Usage) -> Option<String> {
        let needs = [
            (CAP_STENCIL_EXPORT, self.stencil_export, "stencil export without VK_EXT_shader_stencil_export"),
            (CAP_FRAGMENT_BARYCENTRIC, self.barycentric, "per-vertex attributes without VK_KHR_fragment_shader_barycentric"),
            (CAP_MESH_SHADING, self.mesh, "a mesh program without mesh shaders"),
            (CAP_STORAGE_IMAGE_MULTISAMPLE, self.storage_image_multisample, "multisampled storage images without the feature"),
            (CAP_INT64_IMAGE, self.image_int64_atomics, "64-bit image atomics without shaderImageInt64Atomics"),
        ];
        needs
            .iter()
            .find(|(cap, has, _)| !has && u.capabilities.contains(cap))
            .map(|(_, _, why)| (*why).to_owned())
    }

    fn compute_refused(&self, u: &Usage) -> Option<String> {
        if invocations(u) > u64::from(self.compute_invocations) || group_over(u, self.compute_size) {
            return Some(format!("compute group {:?} over the device's {} invocations", u.local_size, self.compute_invocations));
        }
        (u.workgroup_bytes > u64::from(self.compute_shared))
            .then(|| format!("compute shared memory {} bytes over the device's {}", u.workgroup_bytes, self.compute_shared))
    }

    fn mesh_refused(&self, u: &Usage) -> Option<String> {
        if invocations(u) > u64::from(self.mesh_invocations) || group_over(u, self.mesh_size) {
            return Some(format!("mesh group {:?} over the device's {} invocations", u.local_size, self.mesh_invocations));
        }
        if u.out_vertices > self.mesh_vertices || u.out_primitives > self.mesh_primitives {
            return Some(format!(
                "mesh outputs {} vertices / {} primitives over the device's {} / {}",
                u.out_vertices, u.out_primitives, self.mesh_vertices, self.mesh_primitives
            ));
        }
        if u.workgroup_bytes > u64::from(self.mesh_shared) {
            return Some(format!("mesh shared memory {} bytes over the device's {}", u.workgroup_bytes, self.mesh_shared));
        }
        let out = u.mesh_output_bytes(self.mesh_budget());
        if out > u64::from(self.mesh_output_memory) {
            return Some(format!("mesh output memory {out} bytes over the device's {}", self.mesh_output_memory));
        }
        (u.out_locations * 4 > self.mesh_components)
            .then(|| format!("mesh output components {} over the device's {}", u.out_locations * 4, self.mesh_components))
    }
}

/// SignedZeroInfNanPreserve: the capability and the execution mode.
const CAP_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4466;
const EM_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4461;

/// The module the device takes: the translator declares
/// SignedZeroInfNanPreserve for each float width a module uses; the
/// execution modes of widths the device does not preserve (`preserve`:
/// 16, 32 and 64 bits) are left out, and the capability with the last of
/// them. `None`: the module as it is.
pub fn fit_float_controls(words: &[u32], preserve: [bool; 3]) -> Option<Vec<u32>> {
    let instructions = || {
        let mut i = 5;
        std::iter::from_fn(move || {
            let n = ((*words.get(i)? >> 16) as usize).max(1);
            let ins = words.get(i..i + n)?;
            i += n;
            Some(ins)
        })
    };
    // an execution mode of this kind: Some(whether the device keeps it)
    let mode = |ins: &[u32]| match (ins[0] & 0xFFFF, ins.get(2), ins.get(3)) {
        (16, Some(&EM_SIGNED_ZERO_INF_NAN_PRESERVE), Some(&w)) => Some(matches!((w, preserve), (16, [true, _, _]) | (32, [_, true, _]) | (64, [_, _, true]))),
        _ => None,
    };
    if words.len() < 5 || !instructions().any(|ins| mode(ins) == Some(false)) {
        return None;
    }
    let kept = instructions().any(|ins| mode(ins) == Some(true));
    let capability = |ins: &[u32]| ins[0] & 0xFFFF == 17 && ins.get(1) == Some(&CAP_SIGNED_ZERO_INF_NAN_PRESERVE);
    let mut out = words[..5].to_vec();
    for ins in instructions().filter(|&ins| mode(ins) != Some(false) && (kept || !capability(ins))) {
        out.extend_from_slice(ins);
    }
    Some(out)
}

/// A module's own variables may take this much for each thread.
pub const THREAD_BYTES: u64 = 16 << 10;

/// Bytes of a module's `Private` and `Function` variables: what each
/// thread of the program keeps for itself.
///
/// NVIDIA holds this ("local memory") on the card for 32,768 threads,
/// outside every heap's count, sized by the pipeline that asks most, and
/// never gives it back, not even when the pipeline is destroyed. Whether a
/// variable is ever written does not matter: 47.7 KiB of constant tables
/// in one compute program (`Private` arrays with initialisers) held
/// 1.7 GiB (`docs/vulkan-host.md`, "Card memory the driver takes").
pub fn thread_bytes(words: &[u32]) -> u64 {
    use std::collections::HashMap;
    let (mut size, mut value, mut pointee) = (HashMap::new(), HashMap::new(), HashMap::new());
    let mut total = 0u64;
    let mut i = 5;
    while let Some(&head) = words.get(i) {
        let n = ((head >> 16) as usize).max(1);
        let Some(a) = words.get(i + 1..i + n) else { break };
        let of = |id: Option<&u32>| id.and_then(|id| size.get(id)).copied().unwrap_or(0u64);
        match (head & 0xFFFF, a) {
            (20, [id]) => drop(size.insert(*id, 4)),
            (21 | 22, [id, width, ..]) => drop(size.insert(*id, u64::from(*width / 8))),
            (23 | 24, [id, part, count]) => drop(size.insert(*id, of(Some(part)) * u64::from(*count))),
            (28, [id, part, length]) => drop(size.insert(*id, of(Some(part)) * value.get(length).copied().unwrap_or(1))),
            (30, [id, members @ ..]) => drop(size.insert(*id, members.iter().map(|m| of(Some(m))).sum())),
            (32, [id, _, to]) => drop(pointee.insert(*id, *to)),
            (43, [_, id, low, ..]) => drop(value.insert(*id, u64::from(*low))),
            (59, [ty, _, 6 | 7, ..]) => total += of(pointee.get(ty)),
            _ => {}
        }
        i += n;
    }
    total
}

/// Why the card cannot take a module of `words` for its threads' own
/// memory, if it cannot ([`thread_bytes`] over [`THREAD_BYTES`]).
pub fn thread_memory_refused(words: &[u32]) -> Option<String> {
    let bytes = thread_bytes(words);
    (bytes > THREAD_BYTES).then(|| {
        format!("{bytes} bytes of variables for each thread over {THREAD_BYTES} (the driver keeps them on the card for 32,768 threads)")
    })
}

fn invocations(u: &Usage) -> u64 {
    u.local_size.iter().map(|&v| u64::from(v.max(1))).product()
}

fn group_over(u: &Usage, size: [u32; 3]) -> bool {
    (0..3).any(|i| u.local_size[i] > size[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_controls_fit_the_device() {
        let cap = [2 << 16 | 17, CAP_SIGNED_ZERO_INF_NAN_PRESERVE];
        let mode = |w| [4 << 16 | 16, 1, EM_SIGNED_ZERO_INF_NAN_PRESERVE, w];
        let mut m = vec![0x0723_0203, 0x0001_0600, 0, 2, 0];
        m.extend(cap);
        m.extend(mode(16));
        m.extend(mode(32));
        assert_eq!(fit_float_controls(&m, [true, true, true]), None);
        let mut only32 = m[..7].to_vec();
        only32.extend(mode(32));
        assert_eq!(fit_float_controls(&m, [false, true, false]), Some(only32));
        assert_eq!(fit_float_controls(&m, [false, false, true]), Some(m[..5].to_vec()));
    }

    fn nvidia() -> Limits {
        Limits {
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
        }
    }

    #[test]
    fn reads_the_probe_program() {
        let u = usage(crate::compute::SPIRV);
        assert_eq!(u.model, MODEL_COMPUTE);
        assert!(u.local_size[0] >= 1);
    }

    #[test]
    fn counts_what_each_thread_keeps() {
        // uint, uint 600, uint[600], pointers to it (Private, Function),
        // one variable of each, and a Uniform one that does not count
        let m = [
            0x0723_0203, 0x0001_0600, 0, 9, 0,
            4 << 16 | 21, 1, 32, 0,
            4 << 16 | 43, 1, 2, 600,
            4 << 16 | 28, 3, 1, 2,
            4 << 16 | 32, 4, 6, 3,
            4 << 16 | 32, 5, 7, 3,
            4 << 16 | 32, 6, 2, 3,
            4 << 16 | 59, 4, 7, 6,
            4 << 16 | 59, 5, 8, 7,
            4 << 16 | 59, 6, 9, 2,
        ];
        assert_eq!(thread_bytes(&m), 4800);
        assert_eq!(thread_memory_refused(&m), None);
        let mut big = m.to_vec();
        big[12] = 6000;
        assert!(thread_memory_refused(&big).unwrap().starts_with("48000 bytes"));
        assert_eq!(thread_bytes(crate::compute::SPIRV), 0);
    }

    #[test]
    fn refuses_a_big_mesh_group() {
        let u = Usage { model: MODEL_MESH, local_size: [256, 1, 1], ..Default::default() };
        assert!(nvidia().refuses(&u).unwrap().contains("mesh group"));
    }

    #[test]
    fn counts_mesh_output_as_the_translator_does() {
        // position + 15 locations at 128 vertices: 64 × 4 × 128 bytes
        let mut u = Usage {
            model: MODEL_MESH,
            local_size: [128, 1, 1],
            out_vertices: 128,
            out_primitives: 128,
            per_vertex_scalars: 64,
            out_locations: 15,
            ..Default::default()
        };
        let l = nvidia();
        assert_eq!(l.mesh_budget(), victory_shader::vk::usage::MESH_OUTPUT_BUDGET);
        assert_eq!(l.refuses(&u), None);
        u.per_primitive_scalars = 1;
        assert!(l.refuses(&u).unwrap().contains("mesh output memory 33280"));
    }
}
