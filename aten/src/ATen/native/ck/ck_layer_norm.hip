#include <ATen/native/ck/ck_layer_norm.h>
#include "layernorm2d_fwd.hpp"
#include <ATen/ops/empty.h>
#include <ATen/native/layer_norm.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>

namespace at { namespace native {

// common utility functions
#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", at::kFloat)             \
    F("fp16", at::kHalf)              \
    F("bf16", at::kBFloat16)          \
    F("int32", at::kInt)            \
    F("int8", at::kChar)

inline std::string torchDTypeToStr(ScalarType dtype)
{
#define TYPE_CASE(type, torch_type) \
    case torch_type:                \
    {                               \
        return type;                \
    }

    switch (dtype)
    {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        AT_ERROR("Unsupported datatype " + std::to_string((int8_t)(dtype)));
    }

#undef TYPE_CASE
}

std::tuple<Tensor, Tensor, Tensor> ck_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps)
{
    auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
                "ck_tile layernorm2d only supports fp16 and bf16 datatype.");
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;

    auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
    auto M = M_N.first;
    auto N = M_N.second;
    auto input_reshaped = input.reshape({M, N});
    auto weight_reshaped = weight.reshape({N});
    auto bias_reshaped = bias.reshape({N});

    auto output = at::empty(input_reshaped.sizes(), input.options());
    std::string dtype_str = torchDTypeToStr(dtype);
    int stride = input_reshaped.stride(0);
    int xr_stride = -1;
    int y_stride = output.stride(0);
    int yr_stride = -1;
    bool saveMeanVar = true;
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(); 
    
    auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
    auto mean = at::empty({M}, input.options().dtype(acc_type));
    auto rstd = at::empty({M}, input.options().dtype(acc_type));
    layernorm2d_fwd({
                        dtype_str, // input precision
                        dtype_str, // output precision
                        dtype_str, // x-scale, used for [1*N] input smooth quant
                        dtype_str, // y-scale, used for [M*1] output for next layer
                        saveMeanVar,
                        0, // fused_add
                        0  // fused_quant
                    },
                    {input_reshaped.data_ptr(),
                     nullptr, // p_x_residual
                     nullptr, // p_x_scale
                     weight_reshaped.data_ptr(), bias_reshaped.data_ptr(), output.data_ptr(),
                     nullptr, // p_y_residual
                     nullptr, // p_y_scale
                     mean.data_ptr(), // nullptr, // p_mean
                     rstd.data_ptr(), // nullptr, // p_invStd
                     static_cast<float>(eps), static_cast<int>(M), static_cast<int>(N), stride, xr_stride, y_stride, yr_stride},
                    {stream}); 

    auto output_reshaped = output.reshape(input.sizes());
    return std::make_tuple(output_reshaped, mean, rstd);  
}

}} //namespace at::native
