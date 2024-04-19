#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/miopen_batch_norm_native.h>
#include <ATen/ops/miopen_batch_norm_backward_native.h>
#endif

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#include <iostream>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt,
    bool training, double exponential_average_factor, double epsilon) {
  AT_ERROR("miopen_batch_norm: ATen not compiled with MIOpen support");
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_var_opt,
    double epsilon) {
  AT_ERROR("miopen_batch_norm_backward: ATen not compiled with MIOpen support");
}

}}  // namespace at::native

#else // AT_ROCM_ENABLED

#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <ATen/TensorUtils.h>


namespace at { namespace native {

namespace {

Tensor expandScale(const Tensor& t, int64_t dim) {
  std::vector<int64_t> size{ 1, t.numel() };
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  return t.view(size);
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input_t, const Tensor& weight_t, const c10::optional<Tensor>& bias_t_opt, const c10::optional<Tensor>& running_mean_t_opt, const c10::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  std::cout << "%%%%% ENTER miopen_batch_norm" << std::endl;
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(running_mean_t_opt, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(running_var_t_opt, [] {return Tensor();});

  auto memory_format = input_t.suggest_memory_format();

   // See #4500
  std::cout << "%%%%% weight_t.contiguous miopen_batch_norm" << std::endl;
  Tensor weight_contig = weight_t.contiguous(/*memory_format*/);
  std::cout << "%%%%% weight_contig.resize_ miopen_batch_norm" << std::endl;
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes() /*, memory_format*/);
  std::cout << "%%%%% input_t.contiguous(memory_format) miopen_batch_norm" << std::endl;
  Tensor input_contig = input_t.contiguous(memory_format);
  std::cout << "%%%%% input_contig.resize_ miopen_batch_norm" << std::endl;
  input_contig.resize_(input_contig.sizes(), memory_format);
  std::cout << "%%%%% output_t = at::empty miopen_batch_norm" << std::endl;
  auto output_t = at::empty(input_t.sizes(), input_t.options());
  std::cout << "%%%%% output_t.contiguous miopen_batch_norm" << std::endl;
  Tensor output_contig = output_t.contiguous(memory_format);
  
  std::cout << "%%%%% TensorArg miopen_batch_norm" << std::endl;
  TensorArg 
            input{ input_contig, "input", 1 },
            weight{ weight_contig, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 },
            output{ output_contig, "output", 0 };

  CheckedFrom c = "miopen_batch_norm";

  std::cout << "%%%%% checkAllDefined miopen_batch_norm" << std::endl;
  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    std::cout << "%%%%% training checkAllDefined miopen_batch_norm" << std::endl;
    checkAllDefined(c, {running_mean, running_var});
  }
  std::cout << "%%%%% checkAllSameGPU miopen_batch_norm" << std::endl;
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() != ScalarType::Half) {
    std::cout << "%%%%% checkAllSameType miopen_batch_norm" << std::endl;
    checkAllSameType(c, {input, weight});
  }
  std::cout << "%%%%% checkAllSameType miopen_batch_norm" << std::endl;
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  std::cout << "%%%%% checkAllContiguous miopen_batch_norm" << std::endl;
  // checkAllContiguous(c, {input, weight, bias, running_mean, running_var});
  std::cout << "%%%%% checkDimRange miopen_batch_norm" << std::endl;
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  // auto output_t = at::empty(input->sizes(), input->options());
  // TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);
  auto ex_weight = expandScale(*weight, input->dim()).contiguous(memory_format);
  ex_weight.resize_(ex_weight.sizes()); // NC11
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ex_weight , 4 };  // descriptor for weight, bias, running_mean, etc.
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;

  std::cout << "%%%%% WEIGHT dim=" << ex_weight.dim() 
            << " size=" << ex_weight.sizes()
            << " stride=" << ex_weight.strides()
            << " mf=" << ex_weight.suggest_memory_format()
            << std::endl;

  std::cout << "%%%%% INPUT dim=" << input->dim() 
            << " size=" << input->sizes()
            << " stride=" << input->strides()
            << " mf=" << input->suggest_memory_format()
            << std::endl;

  std::cout << "%%%%% OUTPUT dim=" << output->dim() 
            << " size=" << output->sizes()
            << " stride=" << output->strides()
            << " mf=" << output->suggest_memory_format()
            << std::endl;

  if (training) {
    std::cout << "%%%%% ENTER miopen_batch_norm training" << std::endl;
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
    MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
      handle, mode, &one, &zero,
      idesc.desc(), input->data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      ex_weight.data_ptr(),
      bias->data_ptr(),
      exponential_average_factor,
      at::maybe_data_ptr(running_mean),
      at::maybe_data_ptr(running_var),
      epsilon,
      save_mean.mutable_data_ptr(),
      save_var.mutable_data_ptr()));
  } else {
    std::cout << "%%%%% ENTER miopen_batch_norm infer" << std::endl;
    save_mean = at::empty({0}, weight_t.options());
    save_var = at::empty({0}, weight_t.options());
    MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
      handle, mode, &one, &zero,
      idesc.desc(), input->data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      ex_weight.data_ptr(),
      bias->data_ptr(),
      running_mean->data_ptr(),
      running_var->data_ptr(),
      epsilon));
  }

  std::cout << "%%%%% OUTPUT OUT dim=" << output->dim() 
            << " size=" << output->sizes()
            << " stride=" << output->strides()
            << " mf=" << output->suggest_memory_format()
            << std::endl;
  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  std::cout << "%%%%% EXIT miopen_batch_norm" << std::endl;
  return std::tuple<Tensor, Tensor, Tensor>{output_contig, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input_t,
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const optional<Tensor>& running_mean_opt,
    const optional<Tensor>& running_var_opt,
    const optional<Tensor>& save_mean_t_opt,
    const optional<Tensor>& save_var_t_opt,
    double epsilon) {
  std::cout << "%%%%% ENTER miopen_batch_norm_backward" << std::endl;
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return Tensor(); });
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });
  const Tensor& save_mean_t =
      c10::value_or_else(save_mean_t_opt, [] { return Tensor(); });
  const Tensor& save_var_t =
      c10::value_or_else(save_var_t_opt, [] { return Tensor(); });

  TensorArg input{ input_t, "input", 1 },
            grad_output{ grad_output_t, "grad_output", 2 },
            weight{ weight_t, "weight", 3 },
            save_mean{ save_mean_t, "save_mean", 4 },
            save_var{ save_var_t, "save_var", 5 };
  CheckedFrom c = "miopen_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  checkAllContiguous(c, {input, grad_output, save_mean, save_var});
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  auto grad_input_t  = at::empty(input->sizes(), input->options());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t   = at::empty(weight->sizes(), weight->options());

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);

  TensorDescriptor idesc{ *input, 4 };  // input, output, grad_output descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, save_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
    idesc.desc(), input->data_ptr(),
    idesc.desc(), grad_output->data_ptr(),
    idesc.desc(), grad_input_t.data_ptr(),
    wdesc.desc(), weight->data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->data_ptr(),
    save_var->data_ptr()));

  std::cout << "%%%%% EXIT miopen_batch_norm_backward" << std::endl;
  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif
