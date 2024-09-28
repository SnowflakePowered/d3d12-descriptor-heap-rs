# d3d12-descriptor-heap

A simple to use descriptor heap for Direct3D 12, using the `windows` crate.

## Usage
Declare ZST structs for each heap type, and implement `D3D12DescriptorHeapType` for it.

```rust 
#[derive(Clone)]
pub struct CpuStagingHeap;

impl D3D12DescriptorHeapType for CpuStagingHeap {
    fn create_desc(size: usize) -> D3D12_DESCRIPTOR_HEAP_DESC {
        D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: size as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        }
    }
}
```

Then create the heap and hand out ref-counted handles to slots in the heap. The slot will be freed
on drop.

```rust 
fn create_heap(device: &ID3D12Device) -> Result<(), D3D12DescriptorHeapError> {
    let heap = D3D12DescriptorHeap::<CpuStagingHeap>::new(device, 10);
    let slot = heap.allocate_descriptor()?;
}
```

Heap slots implement `AsRef<D3D12_CPU_DESCRIPTOR_HANDLE>` so they can be passed directly into Direct3D 12 APIs that
take `D3D12_CPU_DESCRIPTOR_HANDLE`. For shader-visible heaps, implement `D3D12ShaderVisibleDescriptorHeapType` for the heap type.

```rust 
#[derive(Clone)]
pub struct SamplerHeap;

impl D3D12DescriptorHeapType for SamplerHeap {
    fn create_desc(size: usize) -> D3D12_DESCRIPTOR_HEAP_DESC {
        D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: size as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        }
    }
}

unsafe impl D3D12ShaderVisibleDescriptorHeapType for SamplerHeap {}
```

Heap slots that are allocated for a heap type implementing `D3D12ShaderVisibleDescriptorHeapType` also implement
`AsRef<D3D12_GPU_DESCRIPTOR_HANDLE>`.

