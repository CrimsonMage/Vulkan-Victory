//! The vertex and index buffer views titles pass, laid out as on PC
//! (`D3D12_VERTEX_BUFFER_VIEW`, `D3D12_INDEX_BUFFER_VIEW`); the Vulkan
//! backend binds them (`vulkan/record.rs`).

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) format: u32,
}
