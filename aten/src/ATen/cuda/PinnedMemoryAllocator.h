#pragma once

<<<<<<< HEAD
#include <c10/core/Allocator.h>
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <ATen/cuda/CachingHostAllocator.h>

namespace at::cuda {

<<<<<<< HEAD
inline TORCH_CUDA_CPP_API at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
=======
inline TORCH_CUDA_CPP_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kCUDA);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}
} // namespace at::cuda
