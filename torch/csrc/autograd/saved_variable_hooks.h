#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SafePyObject.h>

namespace torch::autograd {

struct TORCH_API SavedVariableHooks {
  virtual void call_pack_hook(const at::Tensor& tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual ~SavedVariableHooks() = default;
  virtual std::optional<std::pair<c10::SafePyObject, c10::SafePyObject>>
  retrieve_unpack_hook_data() const {
<<<<<<< HEAD
    TORCH_CHECK(
        false, "Compiled Autograd only supports python saved tensor hooks ");
=======
    throw std::runtime_error(
        "Compiled Autograd only supports python saved tensor hooks ");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  }
};

} // namespace torch::autograd
