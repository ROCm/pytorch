#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
<<<<<<< HEAD
#include <c10/core/Allocator.h>

#if AT_MKLDNN_ENABLED()

// needs to be included only once in library.
#include <ideep_pin_singletons.hpp>

using namespace ideep;

RegisterEngineAllocator cpu_alloc(
=======

#if AT_MKLDNN_ENABLED()

#include <c10/core/Allocator.h>
// needs to be included only once in library.
#include <ideep_pin_singletons.hpp>
#include <ATen/native/mkldnn/IDeepRegistration.h>

using namespace ideep;

static RegisterEngineAllocator cpu_alloc(
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  engine::cpu_engine(),
  [](size_t size) {
    return c10::GetAllocator(c10::DeviceType::CPU)->raw_allocate(size);
  },
  [](void* p) {
    c10::GetAllocator(c10::DeviceType::CPU)->raw_deallocate(p);
  }
);

namespace at::native::mkldnn{
<<<<<<< HEAD
void clear_computation_cache();

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
void clear_computation_cache() {
  // Reset computation_cache for forward convolutions
  // As it also caches max number of OpenMP workers
  ideep::convolution_forward::t_store().clear();
}

} // namespace  at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
