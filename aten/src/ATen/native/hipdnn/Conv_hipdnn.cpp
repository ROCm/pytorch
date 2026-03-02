#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/hipdnn_convolution_native.h>
#include <ATen/ops/hipdnn_convolution_transpose_native.h>
#endif

#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at::native {

Tensor hipdnn_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation,
    c10::SymInt groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "hipdnn_convolution: ATen not compiled with ROCm support");
}

Tensor hipdnn_convolution_transpose(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation,
    c10::SymInt groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "hipdnn_convolution_transpose: ATen not compiled with ROCm support");
}

} // namespace at::native

#elif !defined(USE_HIPDNN)

namespace at::native {

Tensor hipdnn_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation,
    c10::SymInt groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "hipdnn_convolution: not compiled with hipDNN support");
}

Tensor hipdnn_convolution_transpose(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation,
    c10::SymInt groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "hipdnn_convolution_transpose: not compiled with hipDNN support");
}

} // namespace at::native

#else // AT_ROCM_ENABLED && USE_HIPDNN

#include <hipdnn_frontend.hpp>
#include <ATen/hipdnn/Types.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/Exceptions.h>
#include <ATen/hipdnn/Utils.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <list>
#include <mutex>
#include <unordered_map>

namespace at { namespace native {

// ---------------------------------------------------------------------------
// Cache key: captures everything that determines graph topology
// ---------------------------------------------------------------------------
constexpr int hipdnn_max_dim = 3;

struct HipdnnConvParams {
  c10::DeviceIndex device_id;
  hipdnn_frontend::DataType dataType;
  int input_size[2 + hipdnn_max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int weight_size[2 + hipdnn_max_dim];
  int padding[hipdnn_max_dim];
  int stride[hipdnn_max_dim];
  int dilation[hipdnn_max_dim];
  int64_t groups;
  int operation; // 0=fprop, 1=dgrad, 2=wgrad
};

static void setHipdnnConvParams(
    HipdnnConvParams* params,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format,
    int operation) {
  memset(params, 0, sizeof(*params));
  params->device_id = input.device().index();
  params->dataType = getHipdnnDataType(input);
  params->input_dim = static_cast<uint8_t>(input.dim());
  params->memory_format = memory_format;
  params->groups = groups;
  params->operation = operation;
  for (int i = 0; i < input.dim(); i++) {
    params->input_size[i] = static_cast<int>(input.size(i));
  }
  for (int i = 0; i < weight.dim(); i++) {
    params->weight_size[i] = static_cast<int>(weight.size(i));
  }
  int spatial_dims = input.dim() - 2;
  for (int i = 0; i < spatial_dims; i++) {
    params->padding[i] = static_cast<int>(padding[i]);
    params->stride[i] = static_cast<int>(stride[i]);
    params->dilation[i] = static_cast<int>(dilation[i]);
  }
}

// ---------------------------------------------------------------------------
// Cached graph value
// ---------------------------------------------------------------------------
struct HipdnnConvCachedGraph {
  std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
  int64_t workspace_size;
  int64_t input_uid;
  int64_t weight_uid;
  int64_t output_uid;
};

// ---------------------------------------------------------------------------
// Thread-local LRU cache (same pattern as cuDNN v8 Conv_v8.cpp)
// ---------------------------------------------------------------------------
static int getHipdnnConvCacheLimit() {
  static int limit = []{
    auto val = c10::utils::check_env("TORCH_HIPDNN_CONV_LRU_CACHE_LIMIT");
    return val.has_value() ? (val.value() ? 10000 : -1) : 10000;
  }();
  return limit;
}

template <typename KeyType>
struct HipdnnGraphCache {
  using KeyWrapper = ParamsWrapper<KeyType>;
  std::list<KeyWrapper> cache_order;
  std::unordered_map<
      KeyWrapper,
      std::pair<HipdnnConvCachedGraph, typename std::list<KeyWrapper>::iterator>,
      ParamsWrapperHash<KeyWrapper>> cache;

  HipdnnConvCachedGraph* find(const KeyType& key) {
    int cache_limit = getHipdnnConvCacheLimit();
    if (cache_limit < 0) return nullptr;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end()) return nullptr;
    if (cache_limit) {
      cache_order.splice(cache_order.begin(), cache_order, it->second.second);
    }
    return &(it->second.first);
  }

  void update(const KeyType& key, HipdnnConvCachedGraph entry) {
    int cache_limit = getHipdnnConvCacheLimit();
    if (cache_limit < 0) return;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end()) {
      if (cache_limit && static_cast<long>(cache.size()) >= cache_limit) {
        cache.erase(cache_order.back());
        cache_order.pop_back();
      }
      cache_order.emplace_front(wrapped);
      cache.emplace(wrapped, std::make_pair(std::move(entry), cache_order.begin()));
    } else {
      it->second.first = std::move(entry);
      if (cache_limit) {
        cache_order.splice(cache_order.begin(), cache_order, it->second.second);
      }
    }
  }
};

static HipdnnGraphCache<HipdnnConvParams>* getHipdnnConvCache() {
  static thread_local auto* cache = new HipdnnGraphCache<HipdnnConvParams>();
  return cache;
}

// ---------------------------------------------------------------------------
// Deterministic UID assignment for graph tensors
// ---------------------------------------------------------------------------
enum HipdnnConvUid : int64_t {
  UID_INPUT = 1,
  UID_WEIGHT = 2,
  UID_OUTPUT = 3,
};

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------
static HipdnnConvCachedGraph buildConvFpropGraph(
    hipdnnHandle_t handle,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {

  auto inputType = getHipdnnDataType(input);
  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  graph->set_io_data_type(inputType)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

  auto x_attr = createTensorAttributes(input);
  x_attr->set_uid(UID_INPUT);
  auto w_attr = createTensorAttributes(weight);
  w_attr->set_uid(UID_WEIGHT);

  hipdnn_frontend::graph::ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding(std::vector<int64_t>(padding.begin(), padding.end()));
  conv_attrs.set_stride(std::vector<int64_t>(stride.begin(), stride.end()));
  conv_attrs.set_dilation(std::vector<int64_t>(dilation.begin(), dilation.end()));

  auto y_attr = graph->conv_fprop(x_attr, w_attr, conv_attrs);
  y_attr->set_output(true).set_uid(UID_OUTPUT);

  HIPDNN_FE_CHECK(graph->build(handle));

  int64_t ws = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(ws));

  return {std::move(graph), ws, UID_INPUT, UID_WEIGHT, UID_OUTPUT};
}

static HipdnnConvCachedGraph buildConvDgradGraph(
    hipdnnHandle_t handle,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef input_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {

  auto inputType = getHipdnnDataType(grad_output);
  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  graph->set_io_data_type(inputType)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

  auto dy_attr = createTensorAttributes(grad_output);
  dy_attr->set_uid(UID_INPUT);
  auto w_attr = createTensorAttributes(weight);
  w_attr->set_uid(UID_WEIGHT);

  hipdnn_frontend::graph::ConvDgradAttributes conv_attrs;
  conv_attrs.set_padding(std::vector<int64_t>(padding.begin(), padding.end()));
  conv_attrs.set_stride(std::vector<int64_t>(stride.begin(), stride.end()));
  conv_attrs.set_dilation(std::vector<int64_t>(dilation.begin(), dilation.end()));

  auto dx_attr = graph->conv_dgrad(dy_attr, w_attr, conv_attrs);
  dx_attr->set_output(true).set_uid(UID_OUTPUT);

  HIPDNN_FE_CHECK(graph->build(handle));

  int64_t ws = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(ws));

  return {std::move(graph), ws, UID_INPUT, UID_WEIGHT, UID_OUTPUT};
}

static HipdnnConvCachedGraph buildConvWgradGraph(
    hipdnnHandle_t handle,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {

  auto inputType = getHipdnnDataType(input);
  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  graph->set_io_data_type(inputType)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

  auto dy_attr = createTensorAttributes(grad_output);
  dy_attr->set_uid(UID_INPUT);
  auto x_attr = createTensorAttributes(input);
  x_attr->set_uid(UID_WEIGHT);

  hipdnn_frontend::graph::ConvWgradAttributes conv_attrs;
  conv_attrs.set_padding(std::vector<int64_t>(padding.begin(), padding.end()));
  conv_attrs.set_stride(std::vector<int64_t>(stride.begin(), stride.end()));
  conv_attrs.set_dilation(std::vector<int64_t>(dilation.begin(), dilation.end()));

  auto dw_attr = graph->conv_wgrad(dy_attr, x_attr, conv_attrs);
  dw_attr->set_output(true).set_uid(UID_OUTPUT);

  HIPDNN_FE_CHECK(graph->build(handle));

  int64_t ws = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(ws));

  return {std::move(graph), ws, UID_INPUT, UID_WEIGHT, UID_OUTPUT};
}

// ---------------------------------------------------------------------------
// Graph execution helpers (cache-check-then-build-and-execute)
// ---------------------------------------------------------------------------
static void runHipdnnConvFprop(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format) {

  auto handle = getHipdnnHandle();
  auto* cache = getHipdnnConvCache();

  HipdnnConvParams key;
  setHipdnnConvParams(&key, input, weight, padding, stride, dilation,
                      groups, memory_format, /*operation=*/0);

  auto* cached = cache->find(key);
  if (!cached) {
    auto entry = buildConvFpropGraph(handle, input, weight, padding, stride, dilation);
    cache->update(key, std::move(entry));
    cached = cache->find(key);
  }

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[cached->input_uid] = input.data_ptr();
  variantPack[cached->weight_uid] = weight.data_ptr();
  variantPack[cached->output_uid] = output.data_ptr();

  auto workspace = at::empty({cached->workspace_size}, input.options().dtype(at::kByte));
  HIPDNN_FE_CHECK(cached->graph->execute(handle, variantPack, workspace.data_ptr()));
}

static void runHipdnnConvDgrad(
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& grad_input,
    IntArrayRef input_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format) {

  auto handle = getHipdnnHandle();
  auto* cache = getHipdnnConvCache();

  HipdnnConvParams key;
  // For dgrad, use grad_output as the "input" for the cache key
  setHipdnnConvParams(&key, grad_output, weight, padding, stride, dilation,
                      groups, memory_format, /*operation=*/1);

  auto* cached = cache->find(key);
  if (!cached) {
    auto entry = buildConvDgradGraph(handle, grad_output, weight, input_size,
                                     padding, stride, dilation);
    cache->update(key, std::move(entry));
    cached = cache->find(key);
  }

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[cached->input_uid] = grad_output.data_ptr();
  variantPack[cached->weight_uid] = weight.data_ptr();
  variantPack[cached->output_uid] = grad_input.data_ptr();

  auto workspace = at::empty({cached->workspace_size}, grad_output.options().dtype(at::kByte));
  HIPDNN_FE_CHECK(cached->graph->execute(handle, variantPack, workspace.data_ptr()));
}

static void runHipdnnConvWgrad(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grad_weight,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format) {

  auto handle = getHipdnnHandle();
  auto* cache = getHipdnnConvCache();

  HipdnnConvParams key;
  // For wgrad, use grad_output+input shape as key
  setHipdnnConvParams(&key, grad_output, input, padding, stride, dilation,
                      groups, memory_format, /*operation=*/2);

  auto* cached = cache->find(key);
  if (!cached) {
    auto entry = buildConvWgradGraph(handle, grad_output, input, weight_size,
                                     padding, stride, dilation);
    cache->update(key, std::move(entry));
    cached = cache->find(key);
  }

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[cached->input_uid] = grad_output.data_ptr();
  variantPack[cached->weight_uid] = input.data_ptr();
  variantPack[cached->output_uid] = grad_weight.data_ptr();

  auto workspace = at::empty({cached->workspace_size}, grad_output.options().dtype(at::kByte));
  HIPDNN_FE_CHECK(cached->graph->execute(handle, variantPack, workspace.data_ptr()));
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
Tensor hipdnn_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    c10::SymIntArrayRef padding_,
    c10::SymIntArrayRef stride_,
    c10::SymIntArrayRef dilation_,
    c10::SymInt groups_,
    bool benchmark,
    bool deterministic) {

  auto padding = C10_AS_INTARRAYREF_SLOW(padding_);
  auto stride = C10_AS_INTARRAYREF_SLOW(stride_);
  auto dilation = C10_AS_INTARRAYREF_SLOW(dilation_);
  auto groups = groups_.expect_int();

  TensorArg input{input_t, "input", 1};
  TensorArg weight{weight_t, "weight", 2};
  CheckedFrom c = "hipdnn_convolution";
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = hipdnn_conv_suggest_memory_format(input_t, weight_t);
  auto input_c = input_t.contiguous(memory_format);
  auto weight_c = weight_t.contiguous(memory_format);

  auto output_size = conv_output_size(
      input_c.sizes(), weight_c.sizes(), padding, stride, dilation);
  auto output = at::empty(output_size, input_c.options(), memory_format);

  runHipdnnConvFprop(input_c, weight_c, output, padding, stride, dilation,
                     groups, memory_format);

  if (bias_opt.has_value() && bias_opt->defined()) {
    output.add_(reshape_bias(input_c.dim(), *bias_opt));
  }

  return output;
}

Tensor hipdnn_convolution_transpose(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    c10::SymIntArrayRef padding_,
    c10::SymIntArrayRef output_padding_,
    c10::SymIntArrayRef stride_,
    c10::SymIntArrayRef dilation_,
    c10::SymInt groups_,
    bool benchmark,
    bool deterministic) {

  auto padding = C10_AS_INTARRAYREF_SLOW(padding_);
  auto output_padding = C10_AS_INTARRAYREF_SLOW(output_padding_);
  auto stride = C10_AS_INTARRAYREF_SLOW(stride_);
  auto dilation = C10_AS_INTARRAYREF_SLOW(dilation_);
  auto groups = groups_.expect_int();

  TensorArg input{input_t, "input", 1};
  TensorArg weight{weight_t, "weight", 2};
  CheckedFrom c = "hipdnn_convolution_transpose";
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = hipdnn_conv_suggest_memory_format(input_t, weight_t);
  auto input_c = input_t.contiguous(memory_format);
  auto weight_c = weight_t.contiguous(memory_format);

  auto trans_output_size = conv_input_size(
      input_c.sizes(), weight_c.sizes(), padding, output_padding, stride, dilation, groups);
  auto output = at::empty(trans_output_size, input_c.options(), memory_format);

  // Transposed conv forward is dgrad
  runHipdnnConvDgrad(input_c, weight_c, output,
                     trans_output_size, padding, stride, dilation,
                     groups, memory_format);

  if (bias_opt.has_value() && bias_opt->defined()) {
    output.add_(reshape_bias(input_c.dim(), *bias_opt));
  }

  return output;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
std::tuple<Tensor, Tensor, Tensor> hipdnn_convolution_backward(
    const Tensor& input,
    const Tensor& grad_output_t,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask) {

  auto memory_format = hipdnn_conv_suggest_memory_format(input, weight);
  auto grad_output = grad_output_t.contiguous(memory_format);
  auto input_c = input.contiguous(memory_format);
  auto weight_c = weight.contiguous(memory_format);

  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty(input_c.sizes(), input_c.options(), memory_format);
    runHipdnnConvDgrad(grad_output, weight_c, grad_input, input_c.sizes(),
                       padding, stride, dilation, groups, memory_format);
  }

  if (output_mask[1]) {
    grad_weight = at::empty(weight_c.sizes(), weight_c.options(), memory_format);
    runHipdnnConvWgrad(grad_output, input_c, grad_weight, weight_c.sizes(),
                       padding, stride, dilation, groups, memory_format);
  }

  if (output_mask[2]) {
    // Sum over all dims except channel dim (dim 1)
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    for (int64_t i = 2; i < grad_output.dim(); i++) {
      reduce_dims.push_back(i);
    }
    grad_bias = grad_output.sum(reduce_dims);
  }

  return std::make_tuple(
      std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_convolution_transpose_backward(
    const Tensor& input,
    const Tensor& grad_output_t,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask) {

  auto memory_format = hipdnn_conv_suggest_memory_format(input, weight);
  auto grad_output = grad_output_t.contiguous(memory_format);
  auto input_c = input.contiguous(memory_format);
  auto weight_c = weight.contiguous(memory_format);

  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    // Transpose backward-input = fprop
    grad_input = at::empty(input_c.sizes(), input_c.options(), memory_format);
    runHipdnnConvFprop(grad_output, weight_c, grad_input,
                       padding, stride, dilation, groups, memory_format);
  }

  if (output_mask[1]) {
    // Transpose backward-weight = wgrad
    grad_weight = at::empty(weight_c.sizes(), weight_c.options(), memory_format);
    runHipdnnConvWgrad(input_c, grad_output, grad_weight, weight_c.sizes(),
                       padding, stride, dilation, groups, memory_format);
  }

  if (output_mask[2]) {
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    for (int64_t i = 2; i < grad_output.dim(); i++) {
      reduce_dims.push_back(i);
    }
    grad_bias = grad_output.sum(reduce_dims);
  }

  return std::make_tuple(
      std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

// ---------------------------------------------------------------------------
// Dispatch stub registration
// ---------------------------------------------------------------------------
REGISTER_CUDA_DISPATCH(hipdnn_convolution_backward_stub, &hipdnn_convolution_backward)
REGISTER_CUDA_DISPATCH(hipdnn_convolution_transpose_backward_stub, &hipdnn_convolution_transpose_backward)

}} // namespace at::native

#endif
