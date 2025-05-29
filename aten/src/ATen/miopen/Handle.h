#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

TORCH_CUDA_CPP_API miopenHandle_t getMiopenHandle();
<<<<<<< HEAD

} // namespace at::native