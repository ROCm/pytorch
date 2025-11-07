#pragma once

<<<<<<< HEAD
#include <torch/nativert/executor/memory/LayoutPlannerSettings.h>
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <cstdint>
#include <string>

namespace torch::nativert {

struct ExecutorConfig {
  bool validateInputs = false;
  bool debugNan = false;
<<<<<<< HEAD
  bool enableStaticCPUKernels = true;
  bool runConstFolding = false;
  bool doExecutionFrameCleanup = true;
  bool tryFreeUnmanagedValuesAfterUse = true;
=======
  bool enableStaticCPUKernels = false;
  bool enableStaticMemoryPlanning = false;
  bool runConstFolding = false;
  bool doExecutionFrameCleanup = true;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  // allows up to max number of concurrent threads.
  int64_t maxNumConcurrentThreads = 8;
  // allows up to max number of parallel ops.
  int64_t maxParallelOps = 1;
  int64_t minNumExecutionFrames = 1;
  int64_t executionFramePoolCleanupIntervalSec = 600;
<<<<<<< HEAD
  LayoutPlannerSettings layoutPlannerSettings;
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  std::string modelName = "unknown";
};

} // namespace torch::nativert
