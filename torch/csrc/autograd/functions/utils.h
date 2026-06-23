#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/InferenceMode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/core/Tensor.h>

#include <functional>
#include <memory>
#include <vector>


namespace torch::autograd::stream_tag {
TORCH_API void push();
TORCH_API void pop();
TORCH_API bool active();
} // namespace

namespace torch::ddp_model2_stream {

struct HasStream12 {
  HasStream12() {}
  HasStream12(bool has1, bool has2) : has_stream1(has1), has_stream2(has2) {}
  bool has_stream1 = false;
  bool has_stream2 = false;
};

struct Registry {
  std::mutex mu;

  // model2 module identity (python object)
  PyObject* model2_module = nullptr;   // strong ref
  bool enabled = false;
  bool start_compute = false;

  // // streams
  // c10::Stream bwd_stream;
  // c10::Stream rccl_stream;

  int64_t bwd_stream_id = 0;
  int64_t bwd_device_index = 0;
  int64_t bwd_device_type = 0;

  int64_t rccl_stream_id = 0;
  int64_t rccl_device_index = 0;
  int64_t rccl_device_type = 0;

  int32_t rccl_cnt = 0;

  // params (for bucket classification)
  std::unordered_set<c10::TensorImpl*> model2_param_impls;

  std::unordered_map<c10::TensorImpl*, HasStream12> bucket_tensor_has_stream;
};

TORCH_API Registry& registry();

} // namespace torch::ddp_model2_stream

namespace torch::autograd {

using function_constructor = std::function<std::shared_ptr<Node>(edge_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
TORCH_API variable_list wrap_outputs(
    const variable_list& inputs,
    tensor_list&& outputs,
    const function_constructor& ctr);

///  Checks that inputs contains exactly `args` items and that the first
///  `required_args`
/// items are not nullptr. If not specified, `required_args` defaults to `args`.
TORCH_API void check_input_variables(
    const char* name,
    const variable_list& inputs,
    int args,
    int required_args = -1,
    bool allow_undefined = false);

struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  void operator()(const std::optional<at::Tensor>& tensor) {
    if (tensor.has_value()) {
      (*this)(*tensor);
    }
  }
  bool short_circuit() {
    return out;
  }
};

template <typename... Args>
inline bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}

inline void set_history(
    const at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  TORCH_CHECK(grad_fn != nullptr);

  if (torch::autograd::stream_tag::active() && !grad_fn->cca_tag()) {
    // CCADEBUG(std::fprintf(stderr, "cca_log set_history override_stream grad_fn->name %s GetTraceID %d\n", grad_fn->name().c_str(), GetTraceID(true)));
    grad_fn->set_cca_tag(true);

    auto& reg = torch::ddp_model2_stream::registry();
    std::lock_guard<std::mutex> g(reg.mu);
    auto bwd_stream = c10::Stream::unpack3(
      reg.bwd_stream_id,
      static_cast<c10::DeviceIndex>(reg.bwd_device_index),
      static_cast<c10::DeviceType>(reg.bwd_device_type));
    grad_fn->set_override_stream(bwd_stream);
  } else {
    // CCADEBUG(std::fprintf(stderr, "cca_log set_history not_override_stream grad_fn->name %s GetTraceID %d\n", grad_fn->name().c_str(), GetTraceID(true)));
  }

  if (variable.defined()) {
    // If the codegen triggers this, you most likely want to add your newly
    // added function to the DONT_REQUIRE_DERIVATIVE list in
    // tools/autograd/gen_variable_type.py
    TORCH_CHECK(
        isDifferentiableType(variable.scalar_type()),
        "Autograd not support dtype: ",
        variable.scalar_type());
    auto output_nr = grad_fn->add_input_metadata(variable);
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

inline void set_history(
    const std::vector<Variable>& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

inline bool isFwGradDefined(const std::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
}

inline bool isFwGradDefinedTensorList(const at::ITensorListRef& variables) {
  bool ret = false;
  for (auto& variable : variables) {
    ret |= isFwGradDefined(variable);
  }
  return ret;
}

inline bool isFwGradDefinedTensorList(
    const c10::List<std::optional<at::Tensor>>& li) {
  bool ret = false;
  for (auto i : c10::irange(li.size())) {
    auto t = li.get(i);
    ret |= isFwGradDefined(t);
  }
  return ret;
}

} // namespace torch::autograd
