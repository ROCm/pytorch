#pragma once

#include <ATen/Tensor.h>
#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

<<<<<<< HEAD
TORCH_CUDA_CPP_API miopenDataType_t getMiopenDataType(const at::Tensor& tensor);
=======
TORCH_HIP_CPP_API miopenDataType_t getMiopenDataType(const at::Tensor& tensor);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

int64_t miopen_version();

} // namespace at::native
