#include "../cub_common.cuh"

namespace at {
namespace cuda {
namespace cub {

template void inclusive_sum_truncating(const int32_t *input, int32_t *output, int64_t num_items);
template void inclusive_sum_truncating(const int64_t *input, int64_t *output, int64_t num_items);
template void inclusive_sum_truncating(const int32_t *input, int64_t *output, int64_t num_items);

}}}  // namespace at::cuda::cub::detail
