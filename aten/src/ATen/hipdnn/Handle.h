#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>
#include <hipdnn_frontend.hpp>

namespace at::native {

TORCH_HIP_CPP_API hipdnnHandle_t getHipdnnHandle();
} // namespace at::native
