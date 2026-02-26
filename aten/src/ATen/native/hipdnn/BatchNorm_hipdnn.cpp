#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/hipdnn_batch_norm_native.h>
#include <ATen/ops/hipdnn_batch_norm_backward_native.h>
#endif

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at::native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool training, double exponential_average_factor, double epsilon) {
  TORCH_CHECK(false, "hipdnn_batch_norm: ATen not compiled with ROCM support");
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    double epsilon) {
  TORCH_CHECK(false, "hipdnn_batch_norm_backward: ATen not compiled with ROCM support");
}

}  // namespace at::native

#else // AT_ROCM_ENABLED

#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <hipdnn_frontend.hpp>
#include <ATen/hipdnn/Types.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/Exceptions.h>

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

inline std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
    createTensorAttributes(const Tensor& t)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    tensor->set_dim(t.sizes().vec()).set_data_type(getHipdnnDataType(t));
    tensor->set_stride(t.strides().vec());

    return tensor;
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  std::cout << ">>> hipdnn_batch_norm: " << std::endl;
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = running_mean_t_opt.value_or(Tensor());
  const Tensor& running_var_t = running_var_t_opt.value_or(Tensor());

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "hipdnn_batch_norm";

  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() == ScalarType::Half || input->scalar_type() == ScalarType::BFloat16) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
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

  auto output_t = at::empty_like(input_t, input_t.options(), input_t.suggest_memory_format());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;

  if (training) {
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
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
      {
    // auto bnAttributes = graph::BatchnormAttributes();
    // bnAttributes.set_name("bn_training_node");

    // auto inputType = getHipdnnDataType(*input);
    // auto intermediateType = getHipdnnDataType(*weight);
    // auto graph = std::make_shared<graph::Graph>();
    // graph->set_io_data_type(inputType)
    //     .set_intermediate_data_type(intermediateType)
    //     .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    // auto input_attr = createTensorAttributes(*input);
    // auto weight_attr = createTensorAttributes(*weight);
    // auto bias_attr = createTensorAttributes(*bias);

    // auto epsilon = std::make_shared<graph::TensorAttributes>();
    // epsilon->set_value(epsilon);
    // bnAttributes.set_epsilon(epsilon);

    // bool useRunningStats = running_mean->defined();

    // // double momentum = 1 - exponential_average_factor;
    // // std::shared_ptr<graph::TensorAttributes> running_mean_attr;
    // // std::shared_ptr<graph::TensorAttributes> running_var_attr;

    // if (useRunningStats) {`
    //   auto prevRunningMean = createTensorAttributes(*running_mean);
    //   auto prevRunningVar = createTensorAttributes(*running_var);
    //   auto momentum = std::make_shared<graph::TensorAttributes>();
    //   momentum->set_value(1 - exponential_average_factor);
    //   bnAttributes.set_previous_running_stats(prevRunningMean, prevRunningVar, momentum);
    // }
    // auto [y, savedMean, savedInvVariance, nextRunningMean, nextRunningVariance]
    //     = graph->batchnorm(x, scale, bias, bnAttributes);

    // y->set_output(true);
    // savedMean->set_output(true).set_data_type(intermediateType);
    // savedInvVariance->set_output(true).set_data_type(intermediateType);

    // if(useRunningStats)
    // {
    //     nextRunningMean->set_output(true).set_data_type(intermediateType);
    //     nextRunningVariance->set_output(true).set_data_type(intermediateType);
    // }
      }

  } else {

    save_mean = at::empty({0}, weight_t.options());
    save_var = at::empty({0}, weight_t.options());
    // MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
    //   handle, mode, &one, &zero,
    //  idesc.desc(), input->const_data_ptr(),
    //  idesc.desc(), output->data_ptr(),
    //  wdesc.desc(),
    //  // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
    //  // not outputs. However, unfortunately the function signature only takes
    //  // non-const pointers, presumably by accident
    //  const_cast<void*>(weight->const_data_ptr()),
    //  const_cast<void*>(bias->const_data_ptr()),
    //  running_mean->data_ptr(),
    //  running_var->data_ptr(),
    //  epsilon));

    std::cout << "+++++++ HIPDNN INFERENCE" << std::endl;
    auto handle = getHipdnnHandle();
    auto inputType = getHipdnnDataType(*input);
    auto intermediateType = getHipdnnDataType(*weight);
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    auto bnAttributes = hipdnn_frontend::graph::BatchnormInferenceAttributes();
    bnAttributes.set_name("bn_inference_node");

    auto input_attr = createTensorAttributes(*input);
    auto weight_attr = createTensorAttributes(expandScale(*weight, input->dim()));
    auto bias_attr = createTensorAttributes(expandScale(*bias, input->dim()));
    auto mean_attr = createTensorAttributes(expandScale(*running_mean, input->dim()));
    auto invVariance_attr = createTensorAttributes(expandScale(*running_var, input->dim()));

    auto output_attr = graph->batchnorm_inference(
      input_attr, mean_attr, invVariance_attr, weight_attr, bias_attr, bnAttributes);
    output_attr->set_output(true);

    std::cout << "+++++++ HIPDNN INFERENCE BUILD ~~~" << std::endl;
    HIPDNN_FE_CHECK(graph->build(handle));
    
    std::cout << "+++++++ HIPDNN INFERENCE variantPack" << std::endl;
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[input_attr->get_uid()] = input->data_ptr();
    variantPack[weight_attr->get_uid()] = weight->data_ptr();
    variantPack[bias_attr->get_uid()] = bias->data_ptr();
    variantPack[mean_attr->get_uid()] = running_mean->data_ptr();
    variantPack[invVariance_attr->get_uid()] = running_var->data_ptr();
    variantPack[output_attr->get_uid()] = output->data_ptr();

    std::cout << "+++++++ HIPDNN INFERENCE EXECUTE" << std::endl;
    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));
  }

  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_batch_norm_backward(
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

      std::cout << ">>> hipdnn_batch_norm_backward: " << std::endl;
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& save_mean_t = save_mean_t_opt.value_or(Tensor());
  const Tensor& save_var_t = save_var_t_opt.value_or(Tensor());

  auto grad_output_contig =
      grad_output_t.contiguous(input_t.suggest_memory_format());
  TensorArg input{input_t, "input", 1},
      grad_output{grad_output_contig, "grad_output", 2},
      weight{weight_t, "weight", 3}, save_mean{save_mean_t, "save_mean", 4},
      save_var{save_var_t, "save_var", 5};
  CheckedFrom c = "miopen_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half || input->scalar_type() == ScalarType::BFloat16) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  // TODO: is weight required to be contiguous?
  checkAllContiguous(c, {save_mean, save_var});
  // TODO: TensorArg check should start handle memory format
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

  auto grad_input_t  = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
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
    idesc.desc(), input->const_data_ptr(),
    idesc.desc(), grad_output->const_data_ptr(),
    idesc.desc(), grad_input_t.data_ptr(),
    wdesc.desc(), weight->const_data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->const_data_ptr(),
    save_var->const_data_ptr()));

    // std::cout << "+++++++ HIPDNN BACKWARD" << std::endl;
    // auto handle = getHipdnnHandle();
    // auto dataType = getHipdnnDataType(*input);
    // auto inputType = getHipdnnDataType(*input);
    // auto intermediateType = getHipdnnDataType(*weight);
    // auto savedMeanAttr = createTensorAttributes(*save_mean);
    // auto savedInvVarianceAttr = createTensorAttributes(*save_var);

    // auto graph = std::make_shared<graph::Graph>();
    // graph->set_io_data_type(inputType)
    //     .set_intermediate_data_type(intermediateType)
    //     .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    //     auto bnBwdAttributes = graph::BatchnormBackwardAttributes();
    // bnBwdAttributes.set_name("bn_backward_node");
    // bnBwdAttributes.set_saved_mean_and_inv_variance(savedMeanAttr, savedInvVarianceAttr);

    // auto [dx, dscale, dbias] = graph->batchnorm_backward(dy, x, scale, bnBwdAttributes);

  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif
