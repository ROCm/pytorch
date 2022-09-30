#include "../cub_common.cuh"

namespace at {
namespace cuda {
namespace cub {
namespace detail {

AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, AT_INSTANTIATE_SORT_PAIRS_8)

}  // namespace detail

}}}  // namespace at::cuda::cub::detail
