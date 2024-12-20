#ifndef CK_LAYER_NORM_H
#define CK_LAYER_NORM_H

#include <ATen/core/Tensor.h>

namespace at { namespace native {
    
std::tuple<Tensor, Tensor, Tensor> ck_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps);

}} // namespace at::native

#endif
