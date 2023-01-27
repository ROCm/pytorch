#include <ATen/ATen.h>
#ifdef USE_ROCM
#include <ATen/cuda/detail/BFloat16Utils.cuh>
#include <hip/hip_bfloat16.h>
#endif

namespace at {
namespace cuda {
namespace detail {

__global__ void out_of_place_fp32_to_bf16_kernel(float* in, uint16_t* out, int nElements)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  for(int i=id; i<nElements; i+=blockDim.x*gridDim.x)
  {
    uint32_t v = reinterpret_cast<uint32_t&>(in[i]);
    out[i] = v>>16;
  }
}
#ifdef USE_ROCM
void out_of_place_fp32_to_bf16(void* in, void* out, int nElements, hipStream_t stream)
{
  int blocks = std::min(1024, (nElements+255)/256);
  int threads = 256;
  hipLaunchKernelGGL(out_of_place_fp32_to_bf16_kernel, dim3(blocks, 1, 1), dim3(threads, 1, 1), 0, stream, (float*)in, (uint16_t*)out, nElements);
}
#endif
}
}
}

