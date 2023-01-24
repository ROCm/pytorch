#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>

namespace at {
namespace cuda {

//
// A caching allocator for UVM allocations.
//
// This provides a drop-in replacement for THCudaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device. We implement this for storages and tensors in
// copy_from_cpu_async_ and copy_to_cpu_async_.
//
// Note that this allocator does not split larger allocations into smaller
// blocks, unlike the caching device allocator.
//
namespace CachingManagedAllocator {

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from cudaMalloc().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via cudaFree)
  StatArray inactive_split;

  // SUM: bytes requested by client code
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;

  // COUNT: total number of failed calls to CUDA malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  bool allocated = false;
  bool active = false;
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  std::vector<BlockInfo> blocks;
};

TORCH_CUDA_CPP_API c10::Allocator* get(DeviceType device_type);

// Records an event in the specified stream. The allocation 'ptr' will not be
// re-used until the event has occurred.
TORCH_CUDA_CPP_API cudaError_t
recordEvent(void* ptr, c10::cuda::CUDAStream stream);

// Releases cached pinned memory allocations via cudaHostFree
TORCH_CUDA_CPP_API void emptyCache();
TORCH_CUDA_CPP_API DeviceStats getDeviceStats();
TORCH_CUDA_CPP_API void* raw_alloc(size_t nbytes);
TORCH_CUDA_CPP_API void* raw_alloc_with_stream(
    size_t nbytes,
    cudaStream_t stream);
TORCH_CUDA_CPP_API void raw_delete(void* ptr);

// inline TORCH_CUDA_CPP_API at::DataPtr ManagedAlloc(size_t size) {
//   return getCachingManagedAllocator()->allocate(size);
// }

} // namespace CachingManagedAllocator

} // namespace cuda
} // namespace at
