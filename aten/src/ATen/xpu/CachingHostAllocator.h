#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/core/Allocator.h>
<<<<<<< HEAD
=======
#include <c10/util/Deprecated.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <c10/xpu/XPUStream.h>

namespace at::xpu {

<<<<<<< HEAD
TORCH_XPU_API c10::Allocator* getCachingHostAllocator();

TORCH_XPU_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream);

TORCH_XPU_API void CachingHostAllocator_emptyCache();

inline TORCH_XPU_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
=======
C10_DEPRECATED_MESSAGE(
    "at::xpu::getCachingHostAllocator() is deprecated. Please use at::getHostAllocator(at::kXPU) instead.")
inline TORCH_XPU_API at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kXPU);
}

C10_DEPRECATED_MESSAGE(
    "at::xpu::CachingHostAllocator_recordEvent(...) is deprecated. Please use at::getHostAllocator(at::kXPU)->record_event(...) instead.")
inline TORCH_XPU_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream) {
  return getHostAllocator(at::kXPU)->record_event(ptr, ctx, stream.unwrap());
}

C10_DEPRECATED_MESSAGE(
    "at::xpu::CachingHostAllocator_emptyCache() is deprecated. Please use at::getHostAllocator(at::kXPU)->empty_cache() instead.")
inline TORCH_XPU_API void CachingHostAllocator_emptyCache() {
  getHostAllocator(at::kXPU)->empty_cache();
}

C10_DEPRECATED_MESSAGE(
    "at::xpu::HostAlloc(...) is deprecated. Please use at::getHostAllocator(at::kXPU)->allocate(...) instead.")
inline TORCH_XPU_API at::DataPtr HostAlloc(size_t size) {
  return getHostAllocator(at::kXPU)->allocate(size);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

} // namespace at::xpu
