#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include <torch/nativert/executor/SerialGraphExecutor.h>

namespace torch::nativert {

std::vector<c10::IValue> SerialGraphExecutor::execute(
    ExecutionFrame& executionFrame,
    std::vector<c10::IValue> inputs) {
  fillUserInputs(executionFrame, std::move(inputs));

  return executeWithPrefilledFrame(executionFrame);
}

std::vector<c10::IValue> SerialGraphExecutor::executeWithPrefilledFrame(
    ExecutionFrame& executionFrame) {
<<<<<<< HEAD
  executionFrame.withManagedMemory([&](const LayoutManager* layout_manager) {
    // Execute kernels for all nodes except prim.Input and prim.Output
    for (NodeIndex nodeIdx = 1; nodeIdx < nodeKernels_.size() - 1; ++nodeIdx) {
      nodeKernels_[nodeIdx]->compute(executionFrame);

#ifndef NDEBUG
      if (layout_manager != nullptr) {
        layout_manager->assert_no_overlapping_storages(nodeIdx);
      }
#endif

      // don't free intermediate values when static memory planning is enabled
      if (executorConfig_.tryFreeUnmanagedValuesAfterUse) {
        // Free the intermediate values that are no used anymore
        for (const auto& valueKey : execPlan_->valuesToFree[nodeIdx]) {
          executionFrame.releaseValueIfNeeded(valueKey);
        }
      }
    }
  });
=======
  // Execute kernels for all nodes except prim.Input and prim.Output
  for (NodeIndex nodeIdx = 1; nodeIdx < nodeKernels_.size() - 1; ++nodeIdx) {
    nodeKernels_[nodeIdx]->compute(executionFrame);

    // don't free intermediate values when static memory planning is enabled
    if (!executorConfig_.enableStaticMemoryPlanning) {
      // Free the intermediate values that are no used anymore
      for (const auto& valueKey : execPlan_->valuesToFree[nodeIdx]) {
        executionFrame.releaseValue(valueKey);
      }
    }
  }

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  return executionFrame.tryMoveUserOutputs();
}

} // namespace torch::nativert
