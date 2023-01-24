#pragma once

#include <c10/core/Allocator.h>
#include <ATen/cuda/CachingManagedAllocator.h>

namespace at { namespace cuda {

inline TORCH_CUDA_CPP_API at::Allocator* getUnifiedDeviceAllocator() {
  return CachingManagedAllocator::get(DeviceType::CUDA);
}

inline TORCH_CUDA_CPP_API at::Allocator* getUnifiedDeviceAllocatorCpu() {
  return CachingManagedAllocator::get(DeviceType::CPU);
}

}} // namespace at::cuda
