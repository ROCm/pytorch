#include "../cub_common.cuh"

namespace at {
namespace cuda {
namespace cub {
namespace detail {

// BFloat16 Radix sort is supported from ROCm 4.5 onwards
#if !AT_ROCM_ENABLED() || (AT_ROCM_ENABLED() && ROCM_VERSION >= 40500)
AT_INSTANTIATE_SORT_PAIRS(c10::BFloat16, 8)
#endif
}


}}}  // namespace at::cuda::cub::detail
