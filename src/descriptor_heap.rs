use bitvec::bitvec;
use bitvec::boxed::BitBox;
use bitvec::order::Lsb0;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::{
    ID3D12DescriptorHeap, ID3D12Device, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE,
    D3D12_GPU_DESCRIPTOR_HANDLE,
};

/// Error type for user-space heap errors.
///
/// Creating heaps return [`windows::core::Result`] instead.
#[derive(Debug, thiserror::Error)]
pub enum D3D12DescriptorHeapError {
    /// The heap has no more available descriptors
    #[error("The descriptor heap of size {0} has no more available descriptors")]
    HeapOverflow(usize),
    /// The heap is too small to fit the number of requested reserved descriptors.
    #[error("The heap only has {available} descriptors free but {requested} reserved descriptors were requested.")]
    HeapUndersized {
        /// The number of requested reserved descriptors.
        requested: usize,
        /// The number of descriptors available.
        available: usize,
    },
    /// The partition scheme requested is not total, and there are remaining available descriptors
    #[error("The requested partitioning is incomplete with {0} remainder descriptors.")]
    IncompletePartioning(usize),
}

/// Marker trait for types of descriptor heaps.
pub trait D3D12DescriptorHeapType {
    /// Create a heap description for this heap type, for the given size of heap.
    fn create_desc(size: usize) -> D3D12_DESCRIPTOR_HEAP_DESC;
}

/// Marker trait for descriptor heaps that are visible to shaders.
///
/// ## Safety
/// This trait is unsafe to implement because the programmer must ensure that
/// heap types with this marker trait are GPU-visible.
pub unsafe trait D3D12ShaderVisibleDescriptorHeapType: D3D12DescriptorHeapType {}

/// An allocated slot on a descriptor heap.
///
/// The slot is deallocated from the heap when this struct is dropped.
#[derive(Clone)]
#[repr(transparent)]
pub struct D3D12DescriptorHeapSlot<T>(Arc<D3D12DescriptorHeapSlotInner<T>>);

struct D3D12DescriptorHeapSlotInner<T> {
    cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu_handle: Option<D3D12_GPU_DESCRIPTOR_HANDLE>,
    heap: Arc<D3D12DescriptorHeapInner>,
    slot: usize,
    _pd: PhantomData<T>,
}

impl<T> D3D12DescriptorHeapSlot<T> {
    /// Get the index of the resource within the heap.
    pub fn index(&self) -> usize {
        self.0.slot
    }

    /// Copy the source handle to this heap slot.
    ///
    /// ## Safety
    /// The type of the resource that the source descriptor handle is for must match
    /// the type of the heap that this heap slot is allocated for.
    pub unsafe fn copy_descriptor(&self, source: D3D12_CPU_DESCRIPTOR_HANDLE) {
        unsafe {
            let heap = self.0.heap.deref();

            heap.device
                .CopyDescriptorsSimple(1, self.0.cpu_handle, source, heap.ty)
        }
    }
}

impl<T> AsRef<D3D12_CPU_DESCRIPTOR_HANDLE> for D3D12DescriptorHeapSlot<T> {
    fn as_ref(&self) -> &D3D12_CPU_DESCRIPTOR_HANDLE {
        &self.0.cpu_handle
    }
}

impl<T: D3D12ShaderVisibleDescriptorHeapType> AsRef<D3D12_GPU_DESCRIPTOR_HANDLE>
    for D3D12DescriptorHeapSlotInner<T>
{
    fn as_ref(&self) -> &D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: D3D12ShaderVisibleHeapType must have a GPU handle, because it's
        // D3D12ShaderVisibleDescriptorHeapType.
        unsafe { self.gpu_handle.as_ref().unwrap_unchecked() }
    }
}

impl<T: D3D12DescriptorHeapType> From<&D3D12DescriptorHeap<T>> for ID3D12DescriptorHeap {
    fn from(value: &D3D12DescriptorHeap<T>) -> Self {
        value.0.heap.clone()
    }
}

#[derive(Debug)]
struct D3D12DescriptorHeapInner {
    device: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    ty: D3D12_DESCRIPTOR_HEAP_TYPE,
    cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu_start: Option<D3D12_GPU_DESCRIPTOR_HANDLE>,
    handle_size: usize,
    start: AtomicUsize,
    num_descriptors: usize,
    map: BitBox<AtomicUsize>,
}

/// An descriptor heap.
pub struct D3D12DescriptorHeap<T>(Arc<D3D12DescriptorHeapInner>, PhantomData<T>);

/// A descriptor heap partitioned into multiple parts.
pub struct D3D12PartitionedHeap<T> {
    /// The equally-sized partitioned portions of the heap,
    pub partitioned: Vec<D3D12DescriptorHeap<T>>,
    /// The reserved portions of the heap
    pub reserved: Option<D3D12DescriptorHeap<T>>,
    /// The root COM pointer to the heap.
    pub handle: ID3D12DescriptorHeap,
}
/// A descriptor heap that can be partitioned into a reserved chunk, and then
/// chunks of equal size.
#[repr(transparent)]
pub struct D3D12PartitionableHeap<T>(D3D12DescriptorHeap<T>);

impl<T: D3D12DescriptorHeapType> D3D12PartitionableHeap<T> {
    /// Create a new partitionable heap for the specified heap type
    pub unsafe fn new(
        device: &ID3D12Device,
        size: usize,
    ) -> windows::core::Result<D3D12PartitionableHeap<T>> {
        let desc = T::create_desc(size);
        unsafe { D3D12PartitionableHeap::new_with_desc(device, desc) }
    }
}

impl<T> D3D12PartitionableHeap<T> {
    /// Create a new heap with the specified descriptor heap description.
    pub unsafe fn new_with_desc(
        device: &ID3D12Device,
        desc: D3D12_DESCRIPTOR_HEAP_DESC,
    ) -> windows::core::Result<D3D12PartitionableHeap<T>> {
        unsafe {
            let heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&desc)?;
            let cpu_start = heap.GetCPUDescriptorHandleForHeapStart();

            let gpu_start = if (desc.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE).0 != 0 {
                Some(heap.GetGPUDescriptorHandleForHeapStart())
            } else {
                None
            };

            Ok(D3D12PartitionableHeap(D3D12DescriptorHeap(
                Arc::new(D3D12DescriptorHeapInner {
                    device: device.clone(),
                    heap,
                    ty: desc.Type,
                    cpu_start,
                    gpu_start,
                    handle_size: device.GetDescriptorHandleIncrementSize(desc.Type) as usize,
                    start: AtomicUsize::new(0),
                    num_descriptors: desc.NumDescriptors as usize,
                    map: bitvec![AtomicUsize, Lsb0; 0; desc.NumDescriptors as usize]
                        .into_boxed_bitslice(),
                }),
                PhantomData::default(),
            )))
        }
    }

    /// Partitions this heap into equally sized chunks, after a number of reserved
    /// descriptors.
    ///
    /// If there aren't enough descriptors, an error is returned.
    /// The partitioning must be **total**; that is the size of each partition must divide equally
    /// into the size of the heap, minus the number of reserved descriptors.
    ///
    /// Returns a tuple of the partitioned heap, the heap of reserved descriptors if not 0,
    pub fn into_partitioned(
        self,
        size: usize,
        reserved: usize,
    ) -> Result<D3D12PartitionedHeap<T>, D3D12DescriptorHeapError> {
        // has to be called right after creation.
        assert_eq!(
            Arc::strong_count(&self.0 .0),
            1,
            "A D3D12PartionableHeap can only be partitioned immediately after creation."
        );

        let Ok(inner) = Arc::try_unwrap(self.0 .0) else {
            unreachable!("A D3D12PartionableHeap should have no live descriptors.")
        };

        let num_descriptors = inner.num_descriptors - reserved;

        // number of suballocated heaps
        let num_heaps = num_descriptors / size;
        let remainder = num_descriptors % size;

        if remainder != 0 {
            return Err(D3D12DescriptorHeapError::IncompletePartioning(remainder));
        }

        let mut heaps = Vec::new();

        let mut start = 0;
        let root_cpu_ptr = inner.cpu_start.ptr;
        let root_gpu_ptr = inner.gpu_start.map(|p| p.ptr);

        for _ in 0..num_heaps {
            let new_cpu_start = root_cpu_ptr + (start * inner.handle_size);
            let new_gpu_start = root_gpu_ptr.map(|r| D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: r + (start as u64 * inner.handle_size as u64),
            });

            heaps.push(D3D12DescriptorHeapInner {
                device: inner.device.clone(),
                heap: inner.heap.clone(),
                ty: inner.ty,
                cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE { ptr: new_cpu_start },
                gpu_start: new_gpu_start,
                handle_size: inner.handle_size,
                start: AtomicUsize::new(0),
                num_descriptors: size,
                map: bitvec![AtomicUsize, Lsb0; 0; size].into_boxed_bitslice(),
            });

            start += size;
        }

        let mut reserved_heap = None;
        if reserved != 0 {
            if reserved != inner.num_descriptors - start {
                return Err(D3D12DescriptorHeapError::HeapUndersized {
                    requested: reserved,
                    available: inner.num_descriptors - start,
                });
            }

            let new_cpu_start = root_cpu_ptr + (start * inner.handle_size);
            let new_gpu_start = root_gpu_ptr.map(|r| D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: r + (start as u64 * inner.handle_size as u64),
            });

            reserved_heap = Some(D3D12DescriptorHeapInner {
                device: inner.device.clone(),
                heap: inner.heap.clone(),
                ty: inner.ty,
                cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE { ptr: new_cpu_start },
                gpu_start: new_gpu_start,
                handle_size: inner.handle_size,
                start: AtomicUsize::new(0),
                num_descriptors: reserved,
                map: bitvec![AtomicUsize, Lsb0; 0; reserved].into_boxed_bitslice(),
            });
        }

        Ok(D3D12PartitionedHeap {
            partitioned: heaps
                .into_iter()
                .map(|inner| D3D12DescriptorHeap(Arc::new(inner), PhantomData::default()))
                .collect(),
            reserved: reserved_heap
                .map(|inner| D3D12DescriptorHeap(Arc::new(inner), PhantomData::default())),
            handle: inner.heap,
        })
    }

    /// Return the entire heap, without partitioning.
    ///
    /// A descriptor heap can only be partitioned immediately after creation.
    /// Once the entire heap is claimed, it can never be partitioned again.
    pub fn into_heap(self) -> D3D12DescriptorHeap<T> {
        self.0
    }
}


impl<T: D3D12DescriptorHeapType> D3D12DescriptorHeap<T> {
    /// Create a new heap for the specified heap type
    pub unsafe fn new(
        device: &ID3D12Device,
        size: usize,
    ) -> windows::core::Result<D3D12DescriptorHeap<T>> {
        D3D12PartitionableHeap::new(device, size).map(D3D12PartitionableHeap::into_heap)
    }
}

impl<T> D3D12DescriptorHeap<T> {
    /// Create a new heap with the specified descriptor heap description.
    pub unsafe fn new_with_desc(
        device: &ID3D12Device,
        desc: D3D12_DESCRIPTOR_HEAP_DESC,
    ) -> windows::core::Result<D3D12DescriptorHeap<T>> {
        unsafe {
            D3D12PartitionableHeap::new_with_desc(device, desc)
                .map(D3D12PartitionableHeap::into_heap)
        }
    }

    /// Allocate a descriptor.
    ///
    /// If there are no more free descriptors, returns an error with the number of
    /// descriptors in this descriptor heap.
    pub fn allocate_descriptor(
        &mut self,
    ) -> Result<D3D12DescriptorHeapSlot<T>, D3D12DescriptorHeapError> {
        let mut handle = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 };

        let inner = &self.0;
        let start = inner.start.load(Ordering::Acquire);
        for i in start..inner.num_descriptors {
            if !inner.map[i] {
                inner.map.set_aliased(i, true);
                handle.ptr = inner.cpu_start.ptr + (i * inner.handle_size);
                inner.start.store(i + 1, Ordering::Release);

                let gpu_handle = inner
                    .gpu_start
                    .map(|gpu_start| D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: (handle.ptr as u64 - inner.cpu_start.ptr as u64) + gpu_start.ptr,
                    });

                return Ok(D3D12DescriptorHeapSlot(Arc::new(
                    D3D12DescriptorHeapSlotInner {
                        cpu_handle: handle,
                        slot: i,
                        heap: Arc::clone(&self.0),
                        gpu_handle,
                        _pd: Default::default(),
                    },
                )));
            }
        }

        // overflow
        Err(D3D12DescriptorHeapError::HeapOverflow(
            inner.num_descriptors,
        ))
    }

    /// Allocate a range of descriptors.
    pub fn allocate_descriptor_range<const NUM_DESC: usize>(
        &mut self,
    ) -> Result<[D3D12DescriptorHeapSlot<T>; NUM_DESC], D3D12DescriptorHeapError> {
        let dest = array_init::try_array_init(|_| self.allocate_descriptor())?;
        Ok(dest)
    }

    /// Gets a cloned handle to the inner heap.
    pub fn handle(&self) -> ID3D12DescriptorHeap {
        let inner = &self.0;
        inner.heap.clone()
    }
}

impl<T> Drop for D3D12DescriptorHeapSlotInner<T> {
    fn drop(&mut self) {
        let inner = &self.heap;
        inner.map.set_aliased(self.slot, false);
        // inner.start > self.slot => inner.start = self.slot
        inner.start.fetch_min(self.slot, Ordering::AcqRel);
    }
}
