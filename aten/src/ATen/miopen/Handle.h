#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

<<<<<<< HEAD
TORCH_CUDA_CPP_API miopenHandle_t getMiopenHandle();

=======
TORCH_HIP_CPP_API miopenHandle_t getMiopenHandle();
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace at::native
