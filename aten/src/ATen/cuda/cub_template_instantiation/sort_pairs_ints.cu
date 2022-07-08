#include "../cub_common.cuh"

namespace at {
namespace cuda {
namespace cub {
namespace detail {

AT_INSTANTIATE_SORT_PAIRS(int32_t, 1)
AT_INSTANTIATE_SORT_PAIRS(int32_t, 2)
AT_INSTANTIATE_SORT_PAIRS(int32_t, 4)
AT_INSTANTIATE_SORT_PAIRS(int64_t, 1)
AT_INSTANTIATE_SORT_PAIRS(int64_t, 2)
AT_INSTANTIATE_SORT_PAIRS(int64_t, 4)

}  // namespace detail

}}}  // namespace at::cuda::cub
