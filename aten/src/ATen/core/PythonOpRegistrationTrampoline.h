#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

// TODO: this can probably live in c10


namespace at::impl {

class TORCH_API PythonOpRegistrationTrampoline final {
  static std::atomic<c10::impl::PyInterpreter*> interpreter_;

public:
  //  Returns true if you successfully registered yourself (that means
  //  you are in the hot seat for doing the operator registrations!)
<<<<<<< HEAD
  static bool registerInterpreter(c10::impl::PyInterpreter* /*interp*/);
=======
  static bool registerInterpreter(c10::impl::PyInterpreter*);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

  // Returns nullptr if no interpreter has been registered yet.
  static c10::impl::PyInterpreter* getInterpreter();
};

} // namespace at::impl
