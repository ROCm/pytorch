#include <ATen/core/op_registration/adaption.h>


namespace c10::impl {

void common_device_check_failure(Device common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
<<<<<<< HEAD
    "Expected all tensors to be on the same device, but "
    "found at least two devices, ", common_device, " and ", tensor.device(), "! "
    "(when checking argument for argument ", argName, " in method ", methodName, ")");
=======
    "Expected all tensors to be on the same device, but got ", argName, " is on ", tensor.device(),
    ", different from other tensors on ", common_device, " (when checking argument in method ", methodName, ")");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

} // namespace c10::impl
