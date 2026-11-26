//! A small SPIR-V module writer: sections in the order the specification
//! requires, ids, and deduplicated types and constants. It knows only the
//! opcodes the Vulkan backend emits (`op` numbers from the SPIR-V 1.6
//! grammar).

use std::collections::HashMap;

pub type Id = u32;

/// Opcodes (SPIR-V 1.6 unified grammar).
#[allow(dead_code)]
pub mod op {
    pub const UNDEF: u16 = 1;
    pub const NAME: u16 = 5;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const EXTENSION: u16 = 10;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_IMAGE: u16 = 25;
    pub const TYPE_SAMPLER: u16 = 26;
    pub const TYPE_SAMPLED_IMAGE: u16 = 27;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const CONSTANT_NULL: u16 = 46;
    pub const SPEC_CONSTANT: u16 = 50;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const VECTOR_SHUFFLE: u16 = 79;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const COMPOSITE_INSERT: u16 = 82;
    pub const SAMPLED_IMAGE: u16 = 86;
    pub const IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
    pub const IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
    pub const IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u16 = 89;
    pub const IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
    pub const IMAGE_FETCH: u16 = 95;
    pub const IMAGE_GATHER: u16 = 96;
    pub const IMAGE_DREF_GATHER: u16 = 97;
    pub const IMAGE_READ: u16 = 98;
    pub const IMAGE_WRITE: u16 = 99;
    pub const IMAGE: u16 = 100;
    pub const IMAGE_QUERY_SIZE_LOD: u16 = 103;
    pub const IMAGE_QUERY_SIZE: u16 = 104;
    pub const IMAGE_QUERY_LOD: u16 = 105;
    pub const IMAGE_QUERY_LEVELS: u16 = 106;
    pub const IMAGE_QUERY_SAMPLES: u16 = 107;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const U_CONVERT: u16 = 113;
    pub const S_CONVERT: u16 = 114;
    pub const F_CONVERT: u16 = 115;
    pub const CONVERT_U_TO_PTR: u16 = 120;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const S_DIV: u16 = 135;
    pub const F_DIV: u16 = 136;
    pub const U_MOD: u16 = 137;
    pub const S_REM: u16 = 138;
    pub const F_REM: u16 = 140;
    pub const U_MUL_EXTENDED: u16 = 151;
    pub const S_MUL_EXTENDED: u16 = 152;
    pub const IS_NAN: u16 = 156;
    pub const IS_INF: u16 = 157;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_EQUAL: u16 = 181;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_UNORD_LESS_THAN: u16 = 185;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_UNORD_GREATER_THAN: u16 = 187;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_UNORD_LESS_THAN_EQUAL: u16 = 189;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const F_UNORD_GREATER_THAN_EQUAL: u16 = 191;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const BIT_FIELD_INSERT: u16 = 201;
    pub const BIT_FIELD_S_EXTRACT: u16 = 202;
    pub const BIT_FIELD_U_EXTRACT: u16 = 203;
    pub const BIT_REVERSE: u16 = 204;
    pub const BIT_COUNT: u16 = 205;
    pub const DPDX: u16 = 207;
    pub const DPDY: u16 = 208;
    pub const DPDX_FINE: u16 = 210;
    pub const DPDY_FINE: u16 = 211;
    pub const DPDX_COARSE: u16 = 213;
    pub const DPDY_COARSE: u16 = 214;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const MEMORY_BARRIER: u16 = 225;
    pub const ATOMIC_LOAD: u16 = 227;
    pub const ATOMIC_STORE: u16 = 228;
    pub const ATOMIC_EXCHANGE: u16 = 229;
    pub const ATOMIC_COMPARE_EXCHANGE: u16 = 230;
    pub const ATOMIC_I_INCREMENT: u16 = 232;
    pub const ATOMIC_I_DECREMENT: u16 = 233;
    pub const ATOMIC_I_ADD: u16 = 234;
    pub const ATOMIC_I_SUB: u16 = 235;
    pub const ATOMIC_S_MIN: u16 = 236;
    pub const ATOMIC_U_MIN: u16 = 237;
    pub const ATOMIC_S_MAX: u16 = 238;
    pub const ATOMIC_U_MAX: u16 = 239;
    pub const ATOMIC_AND: u16 = 240;
    pub const ATOMIC_OR: u16 = 241;
    pub const ATOMIC_XOR: u16 = 242;
    pub const ATOMIC_F_MIN_EXT: u16 = 5614;
    pub const ATOMIC_F_MAX_EXT: u16 = 5615;
    pub const PHI: u16 = 245;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const RETURN: u16 = 253;
    pub const UNREACHABLE: u16 = 255;
    pub const IMAGE_SPARSE_SAMPLE_IMPLICIT_LOD: u16 = 305;
    pub const GROUP_NON_UNIFORM_ELECT: u16 = 333;
    pub const GROUP_NON_UNIFORM_ALL: u16 = 334;
    pub const GROUP_NON_UNIFORM_ANY: u16 = 335;
    pub const GROUP_NON_UNIFORM_BROADCAST: u16 = 337;
    pub const GROUP_NON_UNIFORM_BROADCAST_FIRST: u16 = 338;
    pub const GROUP_NON_UNIFORM_BALLOT: u16 = 339;
    pub const GROUP_NON_UNIFORM_BALLOT_BIT_COUNT: u16 = 342;
    pub const GROUP_NON_UNIFORM_BALLOT_FIND_LSB: u16 = 343;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
    pub const GROUP_NON_UNIFORM_SHUFFLE_XOR: u16 = 346;
    pub const GROUP_NON_UNIFORM_I_ADD: u16 = 349;
    pub const GROUP_NON_UNIFORM_F_ADD: u16 = 350;
    pub const GROUP_NON_UNIFORM_QUAD_BROADCAST: u16 = 365;
    pub const GROUP_NON_UNIFORM_QUAD_SWAP: u16 = 366;
    pub const DEMOTE_TO_HELPER_INVOCATION: u16 = 5380;
    pub const IS_HELPER_INVOCATION: u16 = 5381;
    pub const SET_MESH_OUTPUTS: u16 = 5295;
}

/// Capabilities, storage classes, decorations, built-ins and the other
/// enumerants used.
#[allow(dead_code)]
pub mod en {
    pub const CAP_SHADER: u32 = 1;
    pub const CAP_FLOAT16: u32 = 9;
    pub const CAP_INT64: u32 = 11;
    pub const CAP_INT16: u32 = 22;
    pub const CAP_IMAGE_GATHER_EXTENDED: u32 = 25;
    pub const CAP_STORAGE_IMAGE_MULTISAMPLE: u32 = 27;
    pub const CAP_SAMPLED_1D: u32 = 43;
    pub const CAP_IMAGE_1D: u32 = 44;
    pub const CAP_SAMPLED_CUBE_ARRAY: u32 = 45;
    pub const CAP_IMAGE_QUERY: u32 = 50;
    pub const CAP_DERIVATIVE_CONTROL: u32 = 51;
    pub const CAP_STORAGE_IMAGE_READ_WITHOUT_FORMAT: u32 = 55;
    pub const CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
    pub const CAP_MIN_LOD: u32 = 42;
    pub const CAP_GROUP_NON_UNIFORM: u32 = 61;
    pub const CAP_GROUP_NON_UNIFORM_VOTE: u32 = 62;
    pub const CAP_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
    pub const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
    pub const CAP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
    pub const CAP_GROUP_NON_UNIFORM_QUAD: u32 = 68;
    pub const CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING: u32 = 5307;
    pub const CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING: u32 = 5309;
    pub const CAP_RUNTIME_DESCRIPTOR_ARRAY: u32 = 5302;
    pub const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;
    pub const CAP_DEMOTE_TO_HELPER_INVOCATION: u32 = 5379;
    pub const CAP_STORAGE_BUFFER_16BIT_ACCESS: u32 = 4433;
    pub const CAP_STORAGE_BUFFER_8BIT_ACCESS: u32 = 4448;
    pub const CAP_INT8: u32 = 39;
    pub const CAP_STENCIL_EXPORT: u32 = 5013;
    pub const CAP_ATOMIC_FLOAT32_MIN_MAX_EXT: u32 = 5612;
    pub const CAP_MESH_SHADING_EXT: u32 = 5283;
    pub const CAP_FRAGMENT_BARYCENTRIC_KHR: u32 = 5284;
    pub const CAP_INT64_ATOMICS: u32 = 12;
    pub const CAP_SAMPLE_RATE_SHADING: u32 = 35;

    pub const SC_UNIFORM_CONSTANT: u32 = 0;
    pub const SC_INPUT: u32 = 1;
    pub const SC_UNIFORM: u32 = 2;
    pub const SC_OUTPUT: u32 = 3;
    pub const SC_WORKGROUP: u32 = 4;
    pub const SC_PRIVATE: u32 = 6;
    pub const SC_FUNCTION: u32 = 7;
    pub const SC_PUSH_CONSTANT: u32 = 9;
    pub const SC_STORAGE_BUFFER: u32 = 12;
    pub const SC_PHYSICAL_STORAGE_BUFFER: u32 = 5349;

    pub const DEC_SPEC_ID: u32 = 1;
    pub const DEC_BLOCK: u32 = 2;
    pub const DEC_ARRAY_STRIDE: u32 = 6;
    pub const DEC_BUILTIN: u32 = 11;
    pub const DEC_NO_PERSPECTIVE: u32 = 13;
    pub const DEC_FLAT: u32 = 14;
    pub const DEC_CENTROID: u32 = 16;
    pub const DEC_SAMPLE: u32 = 17;
    pub const DEC_LOCATION: u32 = 30;
    pub const DEC_BINDING: u32 = 33;
    pub const DEC_DESCRIPTOR_SET: u32 = 34;
    pub const DEC_OFFSET: u32 = 35;
    pub const DEC_NON_UNIFORM: u32 = 5300;
    pub const DEC_PER_PRIMITIVE_EXT: u32 = 5271;
    pub const DEC_RESTRICT: u32 = 19;
    pub const DEC_ALIASED: u32 = 20;
    pub const DEC_COHERENT: u32 = 23;

    pub const BI_POSITION: u32 = 0;
    pub const BI_CLIP_DISTANCE: u32 = 3;
    pub const BI_PRIMITIVE_ID: u32 = 7;
    pub const BI_LAYER: u32 = 9;
    pub const BI_VIEWPORT_INDEX: u32 = 10;
    pub const BI_FRAG_COORD: u32 = 15;
    pub const BI_FRONT_FACING: u32 = 17;
    pub const BI_SAMPLE_ID: u32 = 18;
    pub const BI_SAMPLE_MASK: u32 = 20;
    pub const BI_FRAG_DEPTH: u32 = 22;
    pub const BI_HELPER_INVOCATION: u32 = 23;
    pub const BI_WORKGROUP_ID: u32 = 26;
    pub const BI_LOCAL_INVOCATION_ID: u32 = 27;
    pub const BI_LOCAL_INVOCATION_INDEX: u32 = 29;
    pub const BI_SUBGROUP_ID: u32 = 40;
    pub const BI_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
    pub const BI_VERTEX_INDEX: u32 = 42;
    pub const BI_INSTANCE_INDEX: u32 = 43;
    pub const BI_BASE_INSTANCE: u32 = 4425;
    pub const BI_FRAG_STENCIL_REF: u32 = 5014;
    pub const BI_PRIMITIVE_TRIANGLE_INDICES_EXT: u32 = 5296;
    pub const BI_CULL_PRIMITIVE_EXT: u32 = 5299;
    pub const BI_BARY_COORD_KHR: u32 = 5286;

    pub const EM_ORIGIN_UPPER_LEFT: u32 = 7;
    pub const EM_EARLY_FRAGMENT_TESTS: u32 = 9;
    pub const EM_DEPTH_REPLACING: u32 = 12;
    pub const EM_LOCAL_SIZE: u32 = 17;
    pub const EM_OUTPUT_VERTICES: u32 = 26;
    pub const EM_OUTPUT_PRIMITIVES_EXT: u32 = 5270;
    pub const EM_OUTPUT_TRIANGLES_EXT: u32 = 5298;
    pub const EM_STENCIL_REF_REPLACING: u32 = 5027;

    pub const EXEC_VERTEX: u32 = 0;
    pub const EXEC_FRAGMENT: u32 = 4;
    pub const EXEC_GL_COMPUTE: u32 = 5;
    pub const EXEC_MESH_EXT: u32 = 5365;

    pub const SCOPE_DEVICE: u32 = 1;
    pub const SCOPE_WORKGROUP: u32 = 2;
    pub const SCOPE_SUBGROUP: u32 = 3;
    pub const SCOPE_QUEUE_FAMILY: u32 = 5;

    /// Memory semantics.
    pub const SEM_ACQUIRE_RELEASE: u32 = 0x8;
    pub const SEM_UNIFORM_MEMORY: u32 = 0x40;
    pub const SEM_WORKGROUP_MEMORY: u32 = 0x100;
    pub const SEM_IMAGE_MEMORY: u32 = 0x800;

    /// Memory operands.
    pub const MEM_ALIGNED: u32 = 0x2;

    /// Image operands.
    pub const IMG_BIAS: u32 = 0x1;
    pub const IMG_LOD: u32 = 0x2;
    pub const IMG_GRAD: u32 = 0x4;
    pub const IMG_CONST_OFFSET: u32 = 0x8;
    pub const IMG_OFFSET: u32 = 0x10;
    pub const IMG_SAMPLE: u32 = 0x40;
    pub const IMG_MIN_LOD: u32 = 0x80;

    pub const GROUP_REDUCE: u32 = 0;
    pub const GROUP_INCLUSIVE_SCAN: u32 = 1;
    pub const GROUP_EXCLUSIVE_SCAN: u32 = 2;

    pub const DIM_1D: u32 = 0;
    pub const DIM_2D: u32 = 1;
    pub const DIM_3D: u32 = 2;
    pub const DIM_CUBE: u32 = 3;
}

/// GLSL.std.450 instructions.
#[allow(dead_code)]
pub mod glsl {
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const F_ABS: u32 = 4;
    pub const S_ABS: u32 = 5;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    pub const U_MIN: u32 = 38;
    pub const S_MIN: u32 = 39;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const F_CLAMP: u32 = 43;
    pub const FMA: u32 = 50;
    pub const FREXP_STRUCT: u32 = 52;
    pub const LDEXP: u32 = 53;
    pub const PACK_HALF_2X16: u32 = 58;
    pub const UNPACK_HALF_2X16: u32 = 62;
    pub const FIND_I_LSB: u32 = 73;
    pub const FIND_S_MSB: u32 = 74;
    pub const FIND_U_MSB: u32 = 75;
    pub const N_MIN: u32 = 79;
    pub const N_MAX: u32 = 80;
    pub const N_CLAMP: u32 = 81;
    pub const F_MIN: u32 = 37;
    pub const F_MAX: u32 = 40;
}

#[derive(Default)]
pub struct Module {
    next: Id,
    pub caps: Vec<u32>,
    pub exts: Vec<u32>,
    imports: Vec<u32>,
    model: Vec<u32>,
    entries: Vec<u32>,
    modes: Vec<u32>,
    pub annotations: Vec<u32>,
    pub globals: Vec<u32>,
    pub body: Vec<u32>,
    dedup: HashMap<(u16, Vec<u32>), Id>,
    cap_set: Vec<u32>,
    ext_set: Vec<String>,
    glsl: Id,
}

fn put(buf: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    buf.push(((operands.len() as u32 + 1) << 16) | u32::from(opcode));
    buf.extend_from_slice(operands);
}

/// A literal string, NUL-terminated and padded to words.
pub fn string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

impl Module {
    pub fn new() -> Self {
        let mut m = Module { next: 1, ..Default::default() };
        m.capability(en::CAP_SHADER);
        m.glsl = m.id();
        let mut ops = vec![m.glsl];
        ops.extend(string("GLSL.std.450"));
        put(&mut m.imports, op::EXT_INST_IMPORT, &ops);
        // PhysicalStorageBuffer64, GLSL450
        m.capability(en::CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES);
        put(&mut m.model, op::MEMORY_MODEL, &[5348, 1]);
        m
    }

    pub fn id(&mut self) -> Id {
        let id = self.next;
        self.next += 1;
        id
    }

    pub fn bound(&self) -> Id {
        self.next
    }

    pub fn capability(&mut self, cap: u32) {
        if !self.cap_set.contains(&cap) {
            self.cap_set.push(cap);
            put(&mut self.caps, op::CAPABILITY, &[cap]);
        }
    }

    pub fn extension(&mut self, name: &str) {
        if !self.ext_set.iter().any(|e| e == name) {
            self.ext_set.push(name.to_owned());
            put(&mut self.exts, op::EXTENSION, &string(name));
        }
    }

    /// A type or constant, deduplicated by opcode and operands.
    fn global(&mut self, opcode: u16, operands: &[u32]) -> Id {
        if let Some(&id) = self.dedup.get(&(opcode, operands.to_vec())) {
            return id;
        }
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(operands);
        put(&mut self.globals, opcode, &ops);
        self.dedup.insert((opcode, operands.to_vec()), id);
        id
    }

    /// A constant: result type first in `operands`, as the instruction has it.
    fn constant(&mut self, opcode: u16, ty: Id, operands: &[u32]) -> Id {
        let mut key = vec![ty];
        key.extend_from_slice(operands);
        if let Some(&id) = self.dedup.get(&(opcode, key.clone())) {
            return id;
        }
        let id = self.id();
        let mut ops = vec![ty, id];
        ops.extend_from_slice(operands);
        put(&mut self.globals, opcode, &ops);
        self.dedup.insert((opcode, key), id);
        id
    }

    pub fn t_void(&mut self) -> Id {
        self.global(op::TYPE_VOID, &[])
    }
    pub fn t_bool(&mut self) -> Id {
        self.global(op::TYPE_BOOL, &[])
    }
    pub fn t_int(&mut self, width: u32) -> Id {
        match width {
            64 => self.capability(en::CAP_INT64),
            16 => self.capability(en::CAP_INT16),
            8 => self.capability(en::CAP_INT8),
            _ => {}
        }
        self.global(op::TYPE_INT, &[width, 0])
    }
    pub fn t_float(&mut self, width: u32) -> Id {
        if width == 16 {
            self.capability(en::CAP_FLOAT16);
        }
        self.global(op::TYPE_FLOAT, &[width])
    }
    pub fn t_vec(&mut self, elem: Id, n: u32) -> Id {
        if n == 1 {
            return elem;
        }
        self.global(op::TYPE_VECTOR, &[elem, n])
    }
    pub fn t_ptr(&mut self, storage: u32, pointee: Id) -> Id {
        self.global(op::TYPE_POINTER, &[storage, pointee])
    }
    pub fn t_array(&mut self, elem: Id, n: u32) -> Id {
        let u = self.t_int(32);
        let len = self.c_u32_typed(u, n);
        self.global(op::TYPE_ARRAY, &[elem, len])
    }
    /// An array type of its own (no explicit layout from a buffer's array
    /// of the same shape): for workgroup variables.
    pub fn t_array_plain(&mut self, elem: Id, n: u32) -> Id {
        let u = self.t_int(32);
        let len = self.c_u32_typed(u, n);
        let id = self.id();
        put(&mut self.globals, op::TYPE_ARRAY, &[id, elem, len]);
        id
    }
    /// A struct type; `tag` keeps otherwise identical structs apart.
    pub fn t_struct(&mut self, members: &[Id], tag: u32) -> Id {
        let mut key = members.to_vec();
        key.push(u32::MAX - tag);
        if let Some(&id) = self.dedup.get(&(op::TYPE_STRUCT, key.clone())) {
            return id;
        }
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(members);
        put(&mut self.globals, op::TYPE_STRUCT, &ops);
        self.dedup.insert((op::TYPE_STRUCT, key), id);
        id
    }
    pub fn t_func(&mut self, ret: Id) -> Id {
        self.global(op::TYPE_FUNCTION, &[ret])
    }
    /// OpTypeImage with an explicit depth flag (comparison sampling).
    pub fn t_image_d(&mut self, sampled_ty: Id, dim: u32, depth: bool, arrayed: bool, ms: bool, sampled: u32) -> Id {
        self.global(
            op::TYPE_IMAGE,
            &[sampled_ty, dim, u32::from(depth), u32::from(arrayed), u32::from(ms), sampled, 0],
        )
    }
    /// OpTypeImage with a depth flag and an image format.
    #[allow(clippy::too_many_arguments)]
    pub fn t_image_df(&mut self, sampled_ty: Id, dim: u32, depth: bool, arrayed: bool, ms: bool, sampled: u32, format: u32) -> Id {
        self.global(
            op::TYPE_IMAGE,
            &[sampled_ty, dim, u32::from(depth), u32::from(arrayed), u32::from(ms), sampled, format],
        )
    }
    /// A function type with parameters.
    pub fn global_fn_type(&mut self, ret: Id, params: &[Id]) -> Id {
        let mut ops = vec![ret];
        ops.extend_from_slice(params);
        self.global(op::TYPE_FUNCTION, &ops)
    }
    pub fn c_typed(&mut self, ty: Id, v: u32) -> Id {
        self.constant(op::CONSTANT, ty, &[v])
    }
    pub fn t_sampler(&mut self) -> Id {
        self.global(op::TYPE_SAMPLER, &[])
    }
    pub fn t_sampled_image(&mut self, image: Id) -> Id {
        self.global(op::TYPE_SAMPLED_IMAGE, &[image])
    }

    fn c_u32_typed(&mut self, ty: Id, v: u32) -> Id {
        self.constant(op::CONSTANT, ty, &[v])
    }
    pub fn c_u32(&mut self, v: u32) -> Id {
        let t = self.t_int(32);
        self.constant(op::CONSTANT, t, &[v])
    }
    pub fn c_u64(&mut self, v: u64) -> Id {
        let t = self.t_int(64);
        self.constant(op::CONSTANT, t, &[v as u32, (v >> 32) as u32])
    }
    pub fn c_u16(&mut self, v: u16) -> Id {
        let t = self.t_int(16);
        self.constant(op::CONSTANT, t, &[u32::from(v)])
    }
    pub fn c_f32(&mut self, bits: u32) -> Id {
        let t = self.t_float(32);
        self.constant(op::CONSTANT, t, &[bits])
    }
    pub fn c_f16(&mut self, bits: u16) -> Id {
        let t = self.t_float(16);
        self.constant(op::CONSTANT, t, &[u32::from(bits)])
    }
    pub fn c_bool(&mut self, v: bool) -> Id {
        let t = self.t_bool();
        self.constant(if v { op::CONSTANT_TRUE } else { op::CONSTANT_FALSE }, t, &[])
    }
    /// OpSpecConstant of a 32-bit type (not deduplicated).
    pub fn spec_u32(&mut self, ty: Id, default: u32) -> Id {
        let id = self.id();
        put(&mut self.globals, op::SPEC_CONSTANT, &[ty, id, default]);
        id
    }
    /// OpSpecConstantTrue (not deduplicated: each gets its own SpecId).
    pub fn spec_true(&mut self, ty: Id) -> Id {
        let id = self.id();
        put(&mut self.globals, 48, &[ty, id]);
        id
    }
    pub fn c_null(&mut self, ty: Id) -> Id {
        self.constant(op::CONSTANT_NULL, ty, &[])
    }
    pub fn c_composite(&mut self, ty: Id, parts: &[Id]) -> Id {
        self.constant(op::CONSTANT_COMPOSITE, ty, parts)
    }
    pub fn undef(&mut self, ty: Id) -> Id {
        self.constant(op::UNDEF, ty, &[])
    }

    /// A global variable.
    pub fn variable(&mut self, ptr_ty: Id, storage: u32) -> Id {
        let id = self.id();
        put(&mut self.globals, op::VARIABLE, &[ptr_ty, id, storage]);
        id
    }

    /// A global variable with an initializer.
    pub fn variable_init(&mut self, ptr_ty: Id, storage: u32, init: Id) -> Id {
        let id = self.id();
        put(&mut self.globals, op::VARIABLE, &[ptr_ty, id, storage, init]);
        id
    }

    pub fn decorate(&mut self, target: Id, decoration: u32, args: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(args);
        put(&mut self.annotations, op::DECORATE, &ops);
    }

    /// Like [`decorate`](Self::decorate) with one literal; returns the word
    /// index of that literal in the finished module (for linking).
    pub fn decorate_at(&mut self, target: Id, decoration: u32, arg: u32) -> usize {
        self.decorate(target, decoration, &[arg]);
        self.annotations.len() - 1
    }

    pub fn member_decorate(&mut self, ty: Id, member: u32, decoration: u32, args: &[u32]) {
        let mut ops = vec![ty, member, decoration];
        ops.extend_from_slice(args);
        put(&mut self.annotations, op::MEMBER_DECORATE, &ops);
    }

    pub fn entry_point(&mut self, model: u32, func: Id, name: &str, interface: &[Id]) {
        let mut ops = vec![model, func];
        ops.extend(string(name));
        ops.extend_from_slice(interface);
        put(&mut self.entries, op::ENTRY_POINT, &ops);
    }

    pub fn execution_mode(&mut self, func: Id, mode: u32, args: &[u32]) {
        let mut ops = vec![func, mode];
        ops.extend_from_slice(args);
        put(&mut self.modes, op::EXECUTION_MODE, &ops);
    }

    /// An instruction with a result in the function body.
    pub fn op(&mut self, opcode: u16, ty: Id, operands: &[u32]) -> Id {
        let id = self.id();
        self.op_to(opcode, ty, id, operands);
        id
    }

    /// The same with a result id chosen before (phis, forward references).
    pub fn op_to(&mut self, opcode: u16, ty: Id, id: Id, operands: &[u32]) {
        self.body.push(((operands.len() as u32 + 3) << 16) | u32::from(opcode));
        self.body.push(ty);
        self.body.push(id);
        self.body.extend_from_slice(operands);
    }

    /// An instruction without a result.
    pub fn op_void(&mut self, opcode: u16, operands: &[u32]) {
        put(&mut self.body, opcode, operands);
    }

    pub fn ext(&mut self, inst: u32, ty: Id, args: &[Id]) -> Id {
        let mut ops = vec![self.glsl, inst];
        ops.extend_from_slice(args);
        self.op(op::EXT_INST, ty, &ops)
    }

    pub fn label(&mut self, id: Id) {
        put(&mut self.body, op::LABEL, &[id]);
    }

    /// The finished module (SPIR-V 1.6). Also returns where the annotation
    /// section starts, so word indices from [`decorate_at`](Self::decorate_at)
    /// can be made absolute.
    pub fn finish(self) -> (Vec<u32>, usize) {
        let mut out = vec![0x0723_0203, 0x0001_0600, 0, self.next, 0];
        out.extend(self.caps);
        out.extend(self.exts);
        out.extend(self.imports);
        out.extend(self.model);
        out.extend(self.entries);
        out.extend(self.modes);
        let annotations_at = out.len();
        out.extend(self.annotations);
        out.extend(self.globals);
        out.extend(self.body);
        (out, annotations_at)
    }
}
