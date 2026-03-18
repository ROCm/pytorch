#include <torch/csrc/profiler/standalone/roctx_observer.h>

#include <torch/csrc/profiler/util.h>

#ifdef USE_ROCM
#include <roctx.h>
#endif

namespace torch::profiler::impl {

#ifdef USE_ROCM

struct ROCTXThreadLocalState : ProfilerStateBase {
  explicit ROCTXThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~ROCTXThreadLocalState() override = default;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::ROCTX;
  }

  void reportMemoryUsage(
      void* /*ptr*/,
      int64_t /*alloc_size*/,
      size_t /*total_allocated*/,
      size_t /*total_reserved*/,
      c10::Device /*device*/) override {}

  static ROCTXThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::ROCTX);
    return static_cast<ROCTXThreadLocalState*>(tls);
  }
  std::pair<at::RecordFunctionHandle, int> getOpIdFromInput(
      const at::Tensor& tensor);

  void setProducerTensorMap(
      at::TensorImpl* tensor,
      at::RecordFunctionHandle op_id,
      int output_nr) {
    producer_tensor_map_[(void*)tensor] =
        std::pair<at::RecordFunctionHandle, int>{op_id, output_nr};
  }

 protected:
  std::unordered_map<void*, std::pair<at::RecordFunctionHandle, int>>
      producer_tensor_map_;
};

std::pair<at::RecordFunctionHandle, int> ROCTXThreadLocalState::getOpIdFromInput(
    const at::Tensor& tensor) {
  std::pair<at::RecordFunctionHandle, int> producer_op_pair(0, -1);
  if (tensor.defined()) {
    at::TensorImpl* ten_addr = tensor.unsafeGetTensorImpl();
    if (producer_tensor_map_.count((void*)ten_addr) > 0) {
      producer_op_pair = producer_tensor_map_[(void*)ten_addr];
    }
  }
  return producer_op_pair;
}

static std::list<std::pair<at::RecordFunctionHandle, int>> flattenOpIdListROCTX(
    const c10::List<c10::IValue>& list) {
  std::list<std::pair<at::RecordFunctionHandle, int>> input_op_id_list;
  auto state_ptr = ROCTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& input : list) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      auto producer_op_pair = state_ptr->getOpIdFromInput(tensor);
      input_op_id_list.push_back(producer_op_pair);
    }
  }
  return input_op_id_list;
}

static std::list<std::pair<at::RecordFunctionHandle, int>>
getInputTensorOpIdsROCTX(const at::RecordFunction& fn) {
  std::pair<at::RecordFunctionHandle, int> undefined_op_pair(0, -1);
  std::list<std::pair<at::RecordFunctionHandle, int>> input_producer_ops_;
  auto state_ptr = ROCTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& input_item : fn.inputs()) {
    if (input_item.isTensor()) {
      const at::Tensor& tensor = input_item.toTensor();
      auto producer_pair = state_ptr->getOpIdFromInput(tensor);
      input_producer_ops_.push_back(producer_pair);
    } else {
      if (input_item.isList()) {
        std::list<std::pair<at::RecordFunctionHandle, int>> tmp_op_ids =
            flattenOpIdListROCTX(input_item.toList());
        if (!tmp_op_ids.empty()) {
          input_producer_ops_.splice(input_producer_ops_.end(), tmp_op_ids);
        } else {
          input_producer_ops_.emplace_back(undefined_op_pair);
        }
      } else {
        input_producer_ops_.emplace_back(undefined_op_pair);
      }
    }
  }
  return input_producer_ops_;
}

static void updateOutputTensorTrackerROCTX(const at::RecordFunction& fn) {
  int output_nr = 0;
  auto state_ptr = ROCTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& s_tensor : fn.outputs()) {
    if (s_tensor.isTensor()) {
      const at::Tensor& tensor = s_tensor.toTensor();
      if (tensor.defined()) {
        auto ten_addr = tensor.unsafeGetTensorImpl();
        state_ptr->setProducerTensorMap(ten_addr, fn.handle(), output_nr);
      }
    }
    output_nr++;
  }
}

template <bool report_input_shapes>
static std::unique_ptr<at::ObserverContext> enterROCTX(
    const at::RecordFunction& fn) {
  if (ROCTXThreadLocalState::getTLS() != nullptr) {
    auto input_op_ids = getInputTensorOpIdsROCTX(fn);
    std::string name = torch::profiler::impl::getNvtxStr(
        fn.name(),
        fn.seqNr(),
        report_input_shapes ? torch::profiler::impl::inputSizes(fn, true)
                            : std::vector<std::vector<int64_t>>(),
        fn.handle(),
        report_input_shapes
            ? input_op_ids
            : std::list<std::pair<at::RecordFunctionHandle, int>>());
    roctxRangePushA(name.c_str());
  }
  return nullptr;
}

void pushROCTXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  // Marker visible in rocprof/rocprofv3 --marker-trace: confirms new (standalone) observer is active
  roctxMarkA("PyTorch_ROCTX_observer_v2_active");
  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<ROCTXThreadLocalState>(config));

  auto state_ptr = ROCTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          state_ptr->config().report_input_shapes
              ? &enterROCTX</*report_input_shapes=*/true>
              : &enterROCTX</*report_input_shapes=*/false>,
          [](const at::RecordFunction& fn, at::ObserverContext* ctx) {
    (void)ctx;
    roctxRangePop();
    updateOutputTensorTrackerROCTX(fn);
          })
          .needsInputs(config.report_input_shapes)
          .needsOutputs(config.report_input_shapes)
          .needsIds(true)
          .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

#else // !USE_ROCM

void pushROCTXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  (void)config;
  (void)scopes;
  TORCH_CHECK(
      false,
      "ROCTX profiler is only available in ROCm builds. "
      "Rebuild PyTorch with USE_ROCM=ON.");
}

#endif // USE_ROCM

} // namespace torch::profiler::impl
