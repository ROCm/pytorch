#include "../cub_common.cuh"

namespace at {
namespace cuda {
namespace cub {

AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, AT_INSTATIATE_CUB_TEMPLATE_3)

}}}  // namespace at::cuda::cub::detail

