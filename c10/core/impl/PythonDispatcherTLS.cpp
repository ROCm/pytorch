#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>

namespace c10::impl {

<<<<<<< HEAD
thread_local PyInterpreter* pythonDispatcherState;
=======
thread_local static PyInterpreter* pythonDispatcherState;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

void PythonDispatcherTLS::set_state(PyInterpreter* state) {
  if (state) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, true);
  } else {
    PythonDispatcherTLS::reset_state();
  }
  pythonDispatcherState = state;
}

PyInterpreter* PythonDispatcherTLS::get_state() {
  return pythonDispatcherState;
}

void PythonDispatcherTLS::reset_state() {
  pythonDispatcherState = nullptr;
  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::PythonDispatcher, false);
}

} // namespace c10::impl
