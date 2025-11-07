#include <c10/util/UniqueVoidPtr.h>

namespace c10::detail {

<<<<<<< HEAD
void deleteNothing(void* /*unused*/) {}
=======
void deleteNothing(void*) {}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

} // namespace c10::detail
