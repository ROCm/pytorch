#include "../cub_common.cuh"

namespace at {
namespace cuda {
namespace cub {

template void exclusive_sum_in_common_type(const int32_t *input, int32_t *output, int64_t num_items);
template void exclusive_sum_in_common_type(const int64_t *input, int64_t *output, int64_t num_items);
template void exclusive_sum_in_common_type(const bool *input, int64_t *output, int64_t num_items);
template void exclusive_sum_in_common_type(const uint8_t *input, int64_t *output, int64_t num_items);

}}}  // namespace at::cuda::cub::detail

