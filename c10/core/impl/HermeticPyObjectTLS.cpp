#include <c10/core/impl/HermeticPyObjectTLS.h>

namespace c10::impl {

<<<<<<< HEAD
thread_local std::atomic<bool> hermeticPyObjectState{false};
=======
thread_local static std::atomic<bool> hermeticPyObjectState{false};
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

std::atomic<bool> HermeticPyObjectTLS::haveState_{false};

void HermeticPyObjectTLS::set_state(bool state) {
  hermeticPyObjectState = state;
}

bool HermeticPyObjectTLS::get_tls_state() {
  return hermeticPyObjectState;
}

void HermeticPyObjectTLS::init_state() {
  haveState_ = true;
}

} // namespace c10::impl
