#pragma once

#include <cstdint>

namespace torch::nativert {

enum class OpKernelKind : uint8_t {
  kPrimKernel,
  kStaticDispatchKernel,
  kInterpreterFallbackKernel,
<<<<<<< HEAD
  // static dispatch kernels that don't reuse
  // out TensorImpl
  kNativeStaticDispatchKernel,
  kTritonKernel,
=======
  // static dispatch kernels that don't re-use
  // out TensorImpl
  kNativeStaticDispatchKernel,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
};

} // namespace torch::nativert
