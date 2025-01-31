#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
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
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool training, double exponential_average_factor, double epsilon) {
  AT_ERROR("miopen_batch_norm: ATen not compiled with MIOpen support");
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
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

bool PYTORCH_MIOPEN_EXTRA_LOGGING = c10::utils::check_env("PYTORCH_MIOPEN_EXTRA_LOGGING").value_or(false);
bool PYTORCH_MIOPEN_USE_API_V2 = c10::utils::check_env("PYTORCH_MIOPEN_USE_API_V2").value_or(false);
bool PYTORCH_MIOPEN_BATCHNORM_ENABLE_CK = c10::utils::check_env("PYTORCH_MIOPEN_BATCHNORM_ENABLE_CK").value_or(false);

miopenBatchNormMode_t getMiopenBatchNormMode(const Tensor& t)
{
  return (t.dim() == 2) ? miopenBNPerActivation :  miopenBNSpatial;
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_train_forward_v2(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout
      << "$$$$$ miopen_batch_norm_train_forward V2"
      << " input_t=" << input_t.scalar_type()
      << " weight_t=" << weight_t.scalar_type()
      << " bias_t_opt=" << (bias_t_opt.has_value() ? bias_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_mean_t_opt=" << (running_mean_t_opt.has_value() ? running_mean_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_var_t_opt=" << (running_var_t_opt.has_value() ? running_var_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " training=" << training
      // << " exponential_average_factor=" << exponential_average_factor
      // << " epsilon=" << epsilon
      << std::endl;

  const bool use_CK = (input_t.scalar_type() == at::kBFloat16 || input_t.scalar_type() == at::kHalf);// && PYTORCH_MIOPEN_BATCHNORM_USE_CK;
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(running_mean_t_opt, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(running_var_t_opt, [] {return Tensor();});

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{  /*use_CK ? running_mean_t.to(at::kFloat) :*/ running_mean_t, "running_mean", 4 },
            running_var{  /*use_CK? running_var_t.to(at::kFloat) :*/ running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";
  checkAllDefined(c, {input, weight, bias});
  // if (!training) {
  //   checkAllDefined(c, {running_mean, running_var});
  // }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  // if (input->scalar_type() != ScalarType::Half || input->scalar_type() != ScalarType::BFloat16) {
  checkAllSameType(c, {input, weight});
  // }
  // checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  miopenBatchNormMode_t mode = getMiopenBatchNormMode(input_t);

  auto output_t = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
  TensorArg output{ output_t, "output", 0 };

  auto dataType = getMiopenDataType(*input);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;
  
  if (use_CK)
  {
    save_mean = at::empty(num_features, at::kFloat, weight_t.layout(), weight_t.device(), weight_t.is_pinned(), weight_t.suggest_memory_format());
    save_var = at::empty(num_features, at::kFloat, weight_t.layout(), weight_t.device(), weight_t.is_pinned(), weight_t.suggest_memory_format());
  }
  else 
  {
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
  }

  auto handle = getMiopenHandle();
 
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
   std::cout << "##### miopenBatchNormalizationForward V2 Training "
          << " use_CK=" << use_CK
          << " training=" << training
          << " mode=" << mode
          << " input=" << input->scalar_type()   // in
          << " output=" << output->scalar_type() // out
          << " weight=" << weight->scalar_type() // in
          << " bias=" << bias->scalar_type()     // in 
          // << " eaf=" << exponential_average_factor
          << " running_mean=" << running_mean->scalar_type() // out
          << " running_var=" << running_var->scalar_type()   // out
          // << " epsilon=" << epsilon
          << " save_mean=" << save_mean.scalar_type()        // out
          << " save_var=" << save_var.scalar_type()          // out
          << std::endl;
  // std::cout << "*** XXXXXXXXX INPUT miopenBatchNormalizationForward running_mean = " << running_mean->data() << std::endl;
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.
  TensorDescriptor sdesc{ expandScale(save_mean, input->dim()), 4};

  MIOPEN_CHECK(miopenBatchNormalizationForwardTraining_V2(
    handle, mode, &one, &zero,
    idesc.desc(), input->const_data_ptr(),
    idesc.desc(), output->data_ptr(),
    wdesc.desc(), // weight
    wdesc.desc(), // bias
    sdesc.desc(), // saved_mean
    sdesc.desc(), // saved_var
    // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
    // not outputs. However, unfortunately the function signature only takes
    // non-const pointers, presumably by accident
    const_cast<void*>(weight->const_data_ptr()),
    const_cast<void*>(bias->const_data_ptr()),
    exponential_average_factor,
    at::maybe_data_ptr(running_mean),
    at::maybe_data_ptr(running_var),
    epsilon,
    save_mean.mutable_data_ptr(),
    save_var.mutable_data_ptr()));
  // std::cout << "*** XXXXXXX OUTPUT miopenBatchNormalizationForward running_mean = " << running_mean->data() << std::endl;
  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout << "##### miopenBatchNormalizationForward V2 RETURN"
              << " training=" << training
              << " output=" << output->scalar_type()
              << " save_mean=" << save_mean.scalar_type()
              << " save_var=" << save_var.scalar_type()
              << std::endl;
  // if (use_CK)
  // {
  //     std::cout << "##### miopenBatchNormalizationForward RETURN convert to " << input->scalar_type() << std::endl;
  //   return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean.to(input->scalar_type()), save_var.to(input->scalar_type())};
  // }
  // else 
    return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_train_forward(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout
      << "$$$$$ miopen_batch_norm_train_forward"
      << " input_t=" << input_t.scalar_type()
      << " weight_t=" << weight_t.scalar_type()
      << " bias_t_opt=" << (bias_t_opt.has_value() ? bias_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_mean_t_opt=" << (running_mean_t_opt.has_value() ? running_mean_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_var_t_opt=" << (running_var_t_opt.has_value() ? running_var_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " training=" << training
      // << " exponential_average_factor=" << exponential_average_factor
      // << " epsilon=" << epsilon
      << std::endl;

  const bool use_CK = (input_t.scalar_type() == at::kBFloat16 || input_t.scalar_type() == at::kHalf);
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(running_mean_t_opt, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(running_var_t_opt, [] {return Tensor();});

  // if (input_t.scalar_type() != ScalarType::Half || input_t.scalar_type() != ScalarType::BFloat16)

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{  /*use_CK ? running_mean_t.to(at::kFloat) :*/ running_mean_t, "running_mean", 4 },
            running_var{  /*use_CK? running_var_t.to(at::kFloat) :*/ running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";
  checkAllDefined(c, {input, weight, bias});
  // if (!training) {
  //   checkAllDefined(c, {running_mean, running_var});
  // }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  // if (input->scalar_type() != ScalarType::Half || input->scalar_type() != ScalarType::BFloat16) {
  checkAllSameType(c, {input, weight});
  // }
  // checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  miopenBatchNormMode_t mode = getMiopenBatchNormMode(input_t);

  auto output_t = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;
  
  // int64_t num_features = input_t.size(1);
  
  if (use_CK)
  {
    save_mean = at::empty(num_features, at::kFloat, weight_t.layout(), weight_t.device(), weight_t.is_pinned(), weight_t.suggest_memory_format());
    save_var = at::empty(num_features, at::kFloat, weight_t.layout(), weight_t.device(), weight_t.is_pinned(), weight_t.suggest_memory_format());
  }
  else 
  {
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
  }
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
   std::cout << "##### miopenBatchNormalizationForward Training "
          << " use_CK=" << use_CK
          << " training=" << training
          << " mode=" << mode
          << " input=" << input->scalar_type()   // in
          << " output=" << output->scalar_type() // out
          << " weight=" << weight->scalar_type() // in
          << " bias=" << bias->scalar_type()     // in 
          // << " eaf=" << exponential_average_factor
          << " running_mean=" << running_mean->scalar_type() // out
          << " running_var=" << running_var->scalar_type()   // out
          // << " epsilon=" << epsilon
          << " save_mean=" << save_mean.scalar_type()        // out
          << " save_var=" << save_var.scalar_type()          // out
          << std::endl;
  // std::cout << "*** XXXXXXXXX INPUT miopenBatchNormalizationForward running_mean = " << running_mean->data() << std::endl;
  MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
    handle, mode, &one, &zero,
    idesc.desc(), input->const_data_ptr(),
    idesc.desc(), output->data_ptr(),
    wdesc.desc(),
    // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
    // not outputs. However, unfortunately the function signature only takes
    // non-const pointers, presumably by accident
    const_cast<void*>(weight->const_data_ptr()),
    const_cast<void*>(bias->const_data_ptr()),
    exponential_average_factor,
    at::maybe_data_ptr(running_mean),
    at::maybe_data_ptr(running_var),
    epsilon,
    save_mean.mutable_data_ptr(),
    save_var.mutable_data_ptr()));
  // std::cout << "*** XXXXXXX OUTPUT miopenBatchNormalizationForward running_mean = " << running_mean->data() << std::endl;
  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout << "##### miopenBatchNormalizationForward RETURN"
              << " training=" << training
              << " output=" << output->scalar_type()
              << " save_mean=" << save_mean.scalar_type()
              << " save_var=" << save_var.scalar_type()
              << std::endl;
  // if (use_CK)
  // {
  //     std::cout << "##### miopenBatchNormalizationForward RETURN convert to " << input->scalar_type() << std::endl;
  //   return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean.to(input->scalar_type()), save_var.to(input->scalar_type())};
  // }
  // else 
    return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_inference(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  if (PYTORCH_MIOPEN_EXTRA_LOGGING) 
    std::cout
      << "$$$$$ miopen_batch_norm_inference"
      << " input_t=" << input_t.scalar_type()
      << " weight_t=" << weight_t.scalar_type()
      << " bias_t_opt=" << (bias_t_opt.has_value() ? bias_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_mean_t_opt=" << (running_mean_t_opt.has_value() ? running_mean_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_var_t_opt=" << (running_var_t_opt.has_value() ? running_var_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " training=" << training
      // << " exponential_average_factor=" << exponential_average_factor
      // << " epsilon=" << epsilon
      << std::endl;

  const bool use_CK = (input_t.scalar_type() == at::kBFloat16 || input_t.scalar_type() == at::kHalf);
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(running_mean_t_opt, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(running_var_t_opt, [] {return Tensor();});

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ /*use_CK ? running_mean_t.to(at::kFloat):*/running_mean_t, "running_mean", 4 },
            running_var{ /*use_CK ? running_var_t.to(at::kFloat):*/running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";

  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout << "$$$$$XXXXX"
            << " training=" << training
            << " dim=" << input->dim()
            << " memory_format=" << input->suggest_memory_format()
            << "\ninput["
              << " dtype=" << input->scalar_type()
              << " sizes=" << input->sizes()
              << " strides=" << input->strides()
            << " ]\nweight["
              << " dtype=" << weight->scalar_type()
              << " sizes=" << weight->sizes()
              << " strides=" << weight->strides()
            << " ]\nbias["
              << " dtype=" << bias->scalar_type()
              << " sizes=" << bias->sizes()
              << " strides=" << bias->strides()
            << " ]\nrunning_mean["
              << " dtype=" << running_mean->scalar_type()
              << " sizes=" << running_mean->sizes()
              << " strides=" << running_mean->strides()
            << " ]\nrunning_var["
              << " dtype=" << running_var->scalar_type()
              << " sizes=" << running_var->sizes()
              << " strides=" << running_var->strides()
            
            << std::endl;
  checkAllDefined(c, {input, weight, bias});
  checkAllDefined(c, {running_mean, running_var});
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  // if (input->scalar_type() != ScalarType::Half || input->scalar_type() != ScalarType::BFloat16) {
  checkAllSameType(c, {input, weight});
  // }
  // checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  auto mode= getMiopenBatchNormMode(input_t);

  auto output_t = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;
  save_mean = at::empty({ num_features }, weight_t.options());
  save_var = at::empty({ num_features }, weight_t.options());
  if (use_CK) /* && input->suggest_memory_format() == MemoryFormat::ChannelsLast */
  {
    save_mean = save_mean.to(at::kFloat);
    save_var = save_var.to(at::kFloat);
  }

  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
  {
    std::cout << "##### INPUT miopenBatchNormalizationForward running_mean = " << (float*)running_mean->data_ptr() << std::endl;
    std::cout << "##### miopenBatchNormalizationForward Inference "
            << " use_CK=" << use_CK
            << " training=" << training
            << " mode=" << mode
            << " input=" << input->scalar_type()
            << " output=" << output->scalar_type()
            << " weight=" << weight->scalar_type()
            << " bias=" << bias->scalar_type()
            // << " eaf=" << exponential_average_factor
            << " running_mean=" << running_mean->scalar_type()
            << " running_var=" << running_var->scalar_type()
            // << " epsilon=" << epsilon
            << " save_mean=" << save_mean.scalar_type()
            << " save_var=" << save_var.scalar_type()
            << std::endl;
  }
  MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
    handle, mode, &one, &zero,
    idesc.desc(), input->const_data_ptr(), // in
    idesc.desc(), output->data_ptr(),      // out
    wdesc.desc(),
    // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
    // not outputs. However, unfortunately the function signature only takes
    // non-const pointers, presumably by accident
    const_cast<void*>(weight->const_data_ptr()), // in
    const_cast<void*>(bias->const_data_ptr()),   // in
    running_mean->data_ptr(),                    // in
    running_var->data_ptr(),                     // in
    epsilon));

  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
  {
    std::cout << "#####*** OUTPUT miopenBatchNormalizationForward running_mean = " << "AAA" /*(float*)running_mean->data_ptr()*/ << std::endl;
    std::cout << "#####XXXXX miopenBatchNormalizationForward RETURN"
              << " training=" << training
              << " output=" << output->scalar_type()
              << " save_mean=" << save_mean.scalar_type()
              << " save_var=" << save_var.scalar_type()
              << std::endl;
  }
  if (use_CK)
  {
    if (PYTORCH_MIOPEN_EXTRA_LOGGING)
      std::cout << "##### miopenBatchNormalizationForward Inference RETURN convert to " << input->scalar_type() << std::endl;
    return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean.to(input->scalar_type()), save_var.to(input->scalar_type())};
  }
  else 
    return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
  
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_inference_v2(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  if (PYTORCH_MIOPEN_EXTRA_LOGGING) 
    std::cout
      << "$$$$$ miopen_batch_norm_inference V2"
      << " input_t=" << input_t.scalar_type()
      << " weight_t=" << weight_t.scalar_type()
      << " bias_t_opt=" << (bias_t_opt.has_value() ? bias_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_mean_t_opt=" << (running_mean_t_opt.has_value() ? running_mean_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_var_t_opt=" << (running_var_t_opt.has_value() ? running_var_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " training=" << training
      // << " exponential_average_factor=" << exponential_average_factor
      // << " epsilon=" << epsilon
      << std::endl;

  const bool use_CK = (input_t.scalar_type() == at::kBFloat16 || input_t.scalar_type() == at::kHalf);
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(running_mean_t_opt, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(running_var_t_opt, [] {return Tensor();});

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ /*use_CK ? running_mean_t.to(at::kFloat):*/running_mean_t, "running_mean", 4 },
            running_var{ /*use_CK ? running_var_t.to(at::kFloat):*/running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";

  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout << "$$$$$XXXXX V2"
            << " training=" << training
            << " dim=" << input->dim()
            << " memory_format=" << input->suggest_memory_format()
            << "\ninput["
              << " dtype=" << input->scalar_type()
              << " sizes=" << input->sizes()
              << " strides=" << input->strides()
            << " ]\nweight["
              << " dtype=" << weight->scalar_type()
              << " sizes=" << weight->sizes()
              << " strides=" << weight->strides()
            << " ]\nbias["
              << " dtype=" << bias->scalar_type()
              << " sizes=" << bias->sizes()
              << " strides=" << bias->strides()
            << " ]\nrunning_mean["
              << " dtype=" << running_mean->scalar_type()
              << " sizes=" << running_mean->sizes()
              << " strides=" << running_mean->strides()
            << " ]\nrunning_var["
              << " dtype=" << running_var->scalar_type()
              << " sizes=" << running_var->sizes()
              << " strides=" << running_var->strides()
            
            << std::endl;
  checkAllDefined(c, {input, weight, bias});
  checkAllDefined(c, {running_mean, running_var});
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  // if (input->scalar_type() != ScalarType::Half || input->scalar_type() != ScalarType::BFloat16) {
  checkAllSameType(c, {input, weight});
  // }
  // checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  auto mode= getMiopenBatchNormMode(input_t);

  auto output_t = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.
  TensorDescriptor rdesc{ expandScale(*running_mean, input->dim()), 4 };

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;
  save_mean = at::empty({ num_features }, weight_t.options());
  save_var = at::empty({ num_features }, weight_t.options());
  if (use_CK) /* && input->suggest_memory_format() == MemoryFormat::ChannelsLast */
  {
    save_mean = save_mean.to(at::kFloat);
    save_var = save_var.to(at::kFloat);
  }

  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
  {
    std::cout << "##### INPUT miopenBatchNormalizationForward running_mean = " << (float*)running_mean->data_ptr() << std::endl;
    std::cout << "##### miopenBatchNormalizationForward Inference "
            << " use_CK=" << use_CK
            << " training=" << training
            << " mode=" << mode
            << " input=" << input->scalar_type()
            << " output=" << output->scalar_type()
            << " weight=" << weight->scalar_type()
            << " bias=" << bias->scalar_type()
            // << " eaf=" << exponential_average_factor
            << " running_mean=" << running_mean->scalar_type()
            << " running_var=" << running_var->scalar_type()
            // << " epsilon=" << epsilon
            << " save_mean=" << save_mean.scalar_type()
            << " save_var=" << save_var.scalar_type()
            << std::endl;
  }
  MIOPEN_CHECK(miopenBatchNormalizationForwardInference_V2(
    handle, mode, &one, &zero,
    idesc.desc(), input->const_data_ptr(), // in
    idesc.desc(), output->data_ptr(),      // out
    wdesc.desc(), // weight
    wdesc.desc(), // bias
    rdesc.desc(), // running_mean
    rdesc.desc(), // running_var
    // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
    // not outputs. However, unfortunately the function signature only takes
    // non-const pointers, presumably by accident
    const_cast<void*>(weight->const_data_ptr()), // in
    const_cast<void*>(bias->const_data_ptr()),   // in
    running_mean->data_ptr(),                    // in
    running_var->data_ptr(),                     // in
    epsilon));

  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
  {
    std::cout << "#####*** OUTPUT miopenBatchNormalizationForward running_mean = " << "AAA" /*(float*)running_mean->data_ptr()*/ << std::endl;
    std::cout << "#####XXXXX miopenBatchNormalizationForward RETURN"
              << " training=" << training
              << " output=" << output->scalar_type()
              << " save_mean=" << save_mean.scalar_type()
              << " save_var=" << save_var.scalar_type()
              << std::endl;
  }
  if (use_CK)
  {
    if (PYTORCH_MIOPEN_EXTRA_LOGGING)
      std::cout << "##### miopenBatchNormalizationForward Inference RETURN convert to " << input->scalar_type() << std::endl;
    return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean.to(input->scalar_type()), save_var.to(input->scalar_type())};
  }
  else 
    return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
  
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout
      << "$$$$$ miopen_batch_norm"
      << " input_t=" << input_t.scalar_type()
      << " weight_t=" << weight_t.scalar_type()
      << " bias_t_opt=" << (bias_t_opt.has_value() ? bias_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_mean_t_opt=" << (running_mean_t_opt.has_value() ? running_mean_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_var_t_opt=" << (running_var_t_opt.has_value() ? running_var_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " training=" << training
      // << " exponential_average_factor=" << exponential_average_factor
      // << " epsilon=" << epsilon
      << std::endl;

    if (training)
      return PYTORCH_MIOPEN_USE_API_V2?
            miopen_batch_norm_train_forward_v2(input_t, weight_t, bias_t_opt, running_mean_t_opt, running_var_t_opt,
                                            training, exponential_average_factor, epsilon)
            : miopen_batch_norm_train_forward(input_t, weight_t, bias_t_opt, running_mean_t_opt, running_var_t_opt,
                                            training, exponential_average_factor, epsilon);
    else 
      return PYTORCH_MIOPEN_USE_API_V2?
            miopen_batch_norm_inference_v2(input_t, weight_t, bias_t_opt, running_mean_t_opt, running_var_t_opt,
                                            training, exponential_average_factor, epsilon)
            : miopen_batch_norm_inference(input_t, weight_t, bias_t_opt, running_mean_t_opt, running_var_t_opt,
                                            training, exponential_average_factor, epsilon);

}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input_t,
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_t_opt,
    const std::optional<Tensor>& save_var_t_opt,
    double epsilon) {

    const bool use_CK = (input_t.scalar_type() == at::kBFloat16 || input_t.scalar_type() == at::kHalf);
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout
      << "$$$$$ miopen_batch_norm_backward"
      << " use_CK=" << use_CK
      << " input_t=" << input_t.scalar_type()
      << " grad_output_t=" << grad_output_t.scalar_type()
      << " weight_t=" << weight_t.scalar_type()
      << " running_mean_opt=" << (running_mean_opt.has_value() ? running_mean_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " running_var_opt=" << (running_var_opt.has_value() ? running_var_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " save_mean_t_opt=" << (save_mean_t_opt.has_value() ? save_mean_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      << " save_var_t_opt=" << (save_var_t_opt.has_value() ? save_var_t_opt.value().scalar_type() : at::ScalarType::Undefined)
      // << " epsilon=" << epsilon
      << std::endl;
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return Tensor(); });
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });
  const Tensor& save_mean_t =
      c10::value_or_else(save_mean_t_opt, [] { return Tensor(); });
  const Tensor& save_var_t =
      c10::value_or_else(save_var_t_opt, [] { return Tensor(); });

  // auto grad_output_contig =
  //     grad_output_t.contiguous(input_t.suggest_memory_format());

  at::Tensor grad_input_t, grad_weight_t, grad_bias_t, grad_output_contig;

  if (use_CK /* && input_t.suggest_memory_format() == MemoryFormat::ChannelsLast */)
  {
    if (PYTORCH_MIOPEN_EXTRA_LOGGING)
      std::cout << "##### miopen_batch_norm_backward (BF16/FP16 NHWC)"
                << " input_t=" << input_t.scalar_type() << " : "  // << (c10::MemoryFormat) input_t.suggest_memory_format()
                << " weight_t=" << weight_t.scalar_type() << " : " // << (c10::MemoryFormat) weight_t.suggest_memory_format()
                << std::endl;
    grad_input_t  = at::empty(input_t.sizes(), at::kFloat, input_t.layout(), input_t.device(), input_t.is_pinned(), input_t.suggest_memory_format());
    grad_weight_t = at::empty(weight_t.sizes(), at::kFloat, weight_t.layout(), weight_t.device(), weight_t.is_pinned(), MemoryFormat::Contiguous);
    grad_bias_t   = at::empty(weight_t.sizes(), at::kFloat, weight_t.layout(), weight_t.device(), weight_t.is_pinned(), MemoryFormat::Contiguous);
    grad_output_contig = grad_output_t.to(at::kFloat).contiguous(input_t.suggest_memory_format());
  }
  else
  {
    if (PYTORCH_MIOPEN_EXTRA_LOGGING)
      std::cout << "##### miopen_batch_norm_backward non (BF16/FP16 NHWC)"
              << " input_t=" << input_t.scalar_type() << " : " // << (c10::MemoryFormat) input_t.suggest_memory_format()
              << " weight_t=" << weight_t.scalar_type() << " : " // << (c10::MemoryFormat) weight_t.suggest_memory_format()
              << std::endl;
    grad_input_t = at::empty(input_t.sizes(), input_t.scalar_type(), input_t.layout(), input_t.device(), input_t.is_pinned(), input_t.suggest_memory_format());
    grad_weight_t = at::empty(weight_t.sizes(), weight_t.options());
    grad_bias_t = at::empty(weight_t.sizes(), weight_t.options());
    grad_output_contig = grad_output_t.contiguous(input_t.suggest_memory_format());
  }

  TensorArg input{ input_t, "input", 1 },
            grad_output{ grad_output_contig, "grad_output", 2 },
            weight{ weight_t, "weight", 3 },
            save_mean{ save_mean_t, "save_mean", 4 },
            save_var{ save_var_t, "save_var", 5 };
  CheckedFrom c = "miopen_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  // // if (input->scalar_type() == ScalarType::Half) {
  // //   checkScalarType(c, weight, ScalarType::Float);
  // // } else {
  checkAllSameType(c, {input, weight});
  // // }
  // checkAllSameType(c, {input, grad_output});
  // checkAllSameType(c, {weight, save_mean, save_var});
  checkAllContiguous(c, {save_mean, save_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  TORCH_CHECK(grad_output->is_contiguous(input->suggest_memory_format()));
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

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);

  
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout
      << "##### miopenBatchNormalizationBackward "
      << " mode=" << mode
      << " input=" << input->scalar_type()
      << " grad_output=" << grad_output->scalar_type()
      << " grad_input=" << grad_input_t.scalar_type()
      << " weight=" << weight->scalar_type()
      << " grad_weight=" << grad_weight_t.scalar_type()
      << " grad_bias=" << grad_bias_t.scalar_type()
      // << " epsilon=" << epsilon
      << " save_mean=" << save_mean->scalar_type()
      << " save_var=" << save_var->scalar_type()
      << std::endl;
  if (PYTORCH_MIOPEN_USE_API_V2)
  { 
    TensorDescriptor inputdesc{ *input, 4 };  // input, output, grad_output descriptor
    TensorDescriptor gradoutdesc{ *grad_output, 4 };  // grad_input descriptor
    TensorDescriptor gradinputdesc{ grad_input_t, 4 };  // grad_input descriptor
    TensorDescriptor weightdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, save_mean, etc.
    TensorDescriptor biasgraddesc { expandScale(grad_bias_t, input->dim()), 4 };
    TensorDescriptor sdesc { expandScale(*save_mean, input->dim()), 4 };

    MIOPEN_CHECK(miopenBatchNormalizationBackward_V2(
    handle, mode, &one, &zero, &one, &zero,
    inputdesc.desc(), input->const_data_ptr(),
    gradoutdesc.desc(), grad_output->const_data_ptr(),
    gradinputdesc.desc(), grad_input_t.data_ptr(),
    weightdesc.desc(), // weight
    biasgraddesc.desc(), // grad bias
    sdesc.desc(), // saved mean
    sdesc.desc(), // saved var    
    weight->const_data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->const_data_ptr(),
    save_var->const_data_ptr()));

  }
  else{
  TensorDescriptor inputdesc{ *input, 4 };  // input, output, grad_output descriptor
  TensorDescriptor gradoutdesc{ *grad_output, 4 };  // grad_input descriptor
  TensorDescriptor gradinputdesc{ grad_input_t, 4 };  // grad_input descriptor
  TensorDescriptor weightdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, save_mean, etc.
  // TensorDescriptor biasgraddesc { expandScale(grad_bias_t, input->dim()), 4 };
  // TensorDescriptor sdesc { expandScale(*save_mean, input->dim()), 4 };
  
  MIOPEN_CHECK(miopenBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
    inputdesc.desc(), input->const_data_ptr(),
    gradoutdesc.desc(), grad_output->const_data_ptr(),
    gradinputdesc.desc(), grad_input_t.data_ptr(),
    weightdesc.desc(), weight->const_data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->const_data_ptr(),
    save_var->const_data_ptr()));
  }
  if (PYTORCH_MIOPEN_EXTRA_LOGGING)
    std::cout << "##### miopenBatchNormalizationBackward RETURN"
              << " grad_input=" << grad_input_t.scalar_type()
              << " grad_weight=" << grad_weight_t.scalar_type()
              << " grad_bias=" << grad_bias_t.scalar_type()
              << std::endl;

  // if (use_CK)
  // {
  //   return std::tuple<Tensor,Tensor,Tensor>{grad_input_t.to(input_t.scalar_type()), grad_weight_t.to(input_t.scalar_type()), grad_bias_t.to(input_t.scalar_type())};
  // }
  // else
   return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif
