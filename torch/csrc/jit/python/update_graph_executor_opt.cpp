<<<<<<< HEAD
=======
#include <torch/csrc/jit/jit_log.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <torch/csrc/jit/python/update_graph_executor_opt.h>

namespace torch::jit {

<<<<<<< HEAD
thread_local bool kOptimize = true;
void setGraphExecutorOptimize(bool o) {
  kOptimize = o;
=======
static thread_local bool kOptimize = true;
void setGraphExecutorOptimize(bool o) {
  kOptimize = o;
  GRAPH_DEBUG("GraphExecutorOptimize set to ", o);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}
bool getGraphExecutorOptimize() {
  return kOptimize;
}

} // namespace torch::jit
