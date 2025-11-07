#pragma once
#include <ATen/core/Tensor.h>

namespace at::native {

<<<<<<< HEAD
Tensor& qembeddingbag_byte_prepack_out(
    Tensor& output,
    const Tensor& weight,
    const std::optional<Tensor>& rowwise_min_max_opt = std::nullopt);
=======
Tensor& qembeddingbag_byte_prepack_out(Tensor& output, const Tensor& weight);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

Tensor qembeddingbag_byte_prepack(const Tensor& weight);

Tensor qembeddingbag_byte_prepack_meta(const Tensor& weight);

} // namespace at::native
