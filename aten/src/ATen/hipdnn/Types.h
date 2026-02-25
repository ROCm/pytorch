#pragma once

#include <ATen/Tensor.h>
#include <ATen/hipdnn/hipdnn-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

TORCH_CUDA_CPP_API hipdnn_frontend::DataType getHipdnnDataType(const at::Tensor& tensor);

// int64_t miopen_version();

} // namespace at::native
