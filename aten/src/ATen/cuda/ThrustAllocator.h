#pragma once

#include <cstddef>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CachingManagedAllocator.h>

namespace at {
namespace cuda {

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class ThrustAllocator {
public:
  typedef char value_type;

  char* allocate(std::ptrdiff_t size) {
    if (at::globalContext().userEnabledUVM())
      return static_cast<char*>(at::cuda::CachingManagedAllocator::raw_alloc(size));
    else
      return static_cast<char*>(c10::cuda::CUDACachingAllocator::raw_alloc(size));
  }

  void deallocate(char* p, size_t size) {
    if (at::globalContext().userEnabledUVM())
      return at::cuda::CachingManagedAllocator::raw_delete(p);
    else
      return c10::cuda::CUDACachingAllocator::raw_delete(p);
  }
};

}
}
