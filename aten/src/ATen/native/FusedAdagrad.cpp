#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdagrad.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adagrad.h>
#include <ATen/ops/_fused_adagrad_native.h>
#endif

<<<<<<< HEAD

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
namespace at::native {

void _fused_adagrad_kernel_cpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  if (found_inf_ptr && *found_inf_ptr == 1.0) {
<<<<<<< HEAD
      return;
=======
    return;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  }
  size_t n_tensors = params.size();
  TORCH_CHECK(grads.size() == n_tensors);
  TORCH_CHECK(state_sums.size() == n_tensors);
  TORCH_CHECK(state_steps.size() == n_tensors);
<<<<<<< HEAD
  for (size_t i = 0; i < n_tensors; i++){
    fused_adagrad_stub(
      kCPU,
      params[i],
      grads[i],
      state_sums[i],
      state_steps[i],
      lr,
=======
  for (size_t i = 0; i < n_tensors; i++) {
    fused_adagrad_stub(
        kCPU,
        params[i],
        grads[i],
        state_sums[i],
        state_steps[i],
        lr,
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale_ptr);
  }
}

void _fused_adagrad_kernel_cpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  _fused_adagrad_kernel_cpu_(
      params,
      grads,
      state_sums,
      state_steps,
      lr.item<double>(),
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
      lr_decay,
      weight_decay,
      eps,
      maximize,
<<<<<<< HEAD
      grad_scale_ptr);
  }
=======
      grad_scale,
      found_inf);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

DEFINE_DISPATCH(fused_adagrad_stub);

<<<<<<< HEAD
}
=======
} // namespace at::native
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
