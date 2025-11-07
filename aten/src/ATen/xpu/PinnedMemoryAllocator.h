#pragma once

#include <ATen/xpu/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

namespace at::xpu {

<<<<<<< HEAD
inline TORCH_XPU_API at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
=======
inline TORCH_XPU_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kXPU);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}
} // namespace at::xpu
