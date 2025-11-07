#include <caffe2/utils/threadpool/thread_pool_guard.h>

namespace caffe2 {

<<<<<<< HEAD
thread_local bool _NoPThreadPoolGuard_enabled = false;
=======
static thread_local bool _NoPThreadPoolGuard_enabled = false;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

bool _NoPThreadPoolGuard::is_enabled() {
  return _NoPThreadPoolGuard_enabled;
}

void _NoPThreadPoolGuard::set_enabled(bool enabled) {
  _NoPThreadPoolGuard_enabled = enabled;
}

} // namespace at
