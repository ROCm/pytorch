#pragma once

namespace at::native::mps {
<<<<<<< HEAD
void complex_mul_out(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
}
=======
void binary_op_kernel(
    const std::string func_name,
    const Tensor& input,
    const Tensor& other,
    const Tensor& output,
    const std::optional<Scalar> alpha = std::nullopt);
} // namespace at::native::mps
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
