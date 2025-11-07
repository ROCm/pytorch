// Just a little test file to make sure that the CUDA library works

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/impl/CUDATest.h>

#include <cuda_runtime.h>

namespace c10::cuda::impl {

<<<<<<< HEAD
bool has_cuda_gpu() {
=======
static bool has_cuda_gpu() {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  int count = 0;
  C10_CUDA_IGNORE_ERROR(cudaGetDeviceCount(&count));

  return count != 0;
}

int c10_cuda_test() {
  int r = 0;
  if (has_cuda_gpu()) {
    C10_CUDA_CHECK(cudaGetDevice(&r));
  }
  return r;
}

<<<<<<< HEAD
// This function is not exported
int c10_cuda_private_test() {
  return 2;
}

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace c10::cuda::impl
