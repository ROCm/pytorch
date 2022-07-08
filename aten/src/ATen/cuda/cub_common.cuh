#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAConfig.h>

namespace at {
namespace cuda {
namespace cub {
namespace detail {

template<typename key_t, int value_size>
void radix_sort_pairs_impl(
    const key_t *keys_in, key_t *keys_out,
    const OpaqueType<value_size> *values_in, OpaqueType<value_size> *values_out,
    int64_t n, bool descending, int64_t begin_bit, int64_t end_bit) {
  TORCH_CHECK(n <= std::numeric_limits<int>::max(),
    "cub sort does not support sorting more than INT_MAX elements");
  using key_t_ = typename detail::cuda_type<key_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(n * sizeof(key_t));
    keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
  }

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in, values_out, n,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortPairs,
      keys_in_, keys_out_, values_in, values_out, n,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

#define AT_INSTANTIATE_SORT_PAIRS(key_t, value_size)                    \
  template void radix_sort_pairs_impl(                                  \
      const key_t *keys_in, key_t *keys_out,                            \
      const OpaqueType<value_size> *values_in,                          \
      OpaqueType<value_size> *values_out,                               \
      int64_t n, bool descending, int64_t begin_bit, int64_t end_bit);

#define AT_INSTANTIATE_SORT_PAIRS_8(scalar_t, ScalarType)   \
  AT_INSTANTIATE_SORT_PAIRS(scalar_t, 8)

}  // namespace detail

template<typename key_t>
void radix_sort_keys(
    const key_t *keys_in, key_t *keys_out,
    int64_t n, bool descending, int64_t begin_bit, int64_t end_bit) {
  TORCH_CHECK(n <= std::numeric_limits<int>::max(),
              "cub sort does not support sorting more than INT_MAX elements");
  using key_t_ = typename detail::cuda_type<key_t>::type;

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortKeysDescending,
                keys_in_, keys_out_, n,
                begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceRadixSort::SortKeys,
                keys_in_, keys_out_, n,
                begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

template<typename scalar_t>
void unique(const scalar_t *input, scalar_t *output, int64_t *num_selected_out, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub unique does not support more than INT_MAX elements");
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSelect::Unique,
              input, output, num_selected_out, num_items, at::cuda::getCurrentCUDAStream());
}

template <typename scalar_t>
void run_length_encode(const scalar_t *input, scalar_t *output, int64_t *counts_out,
                       int64_t *length_out, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub run_length_encode does not support more than INT_MAX elements");
  CUB_WRAPPER(
      NO_ROCM(at_cuda_detail)::cub::DeviceRunLengthEncode::Encode,
      input, output, counts_out, length_out, num_items,
      at::cuda::getCurrentCUDAStream());
}

#define AT_INSTATIATE_CUB_TEMPLATE_1(scalar_t, ScalarType)          \
  template void radix_sort_keys(                                    \
      const scalar_t *keys_in, scalar_t *keys_out, int64_t n,       \
      bool descending, int64_t begin_bit, int64_t end_bit);

#define AT_INSTATIATE_CUB_TEMPLATE_2(scalar_t, ScalarType)          \
  template void unique(                                             \
      const scalar_t *input, scalar_t *output,                      \
      int64_t *num_selected_out, int64_t num_items);

#define AT_INSTATIATE_CUB_TEMPLATE_3(scalar_t, ScalarType)          \
  template void run_length_encode(                                  \
      const scalar_t *input, scalar_t *output, int64_t *counts_out, \
      int64_t *length_out, int64_t n);

namespace {
template <typename scalar_t>
struct SumOp {
  __device__ scalar_t operator () (scalar_t a, scalar_t b) const {
    return a + b;
  }
};
}

template <typename input_t, typename output_t>
void inclusive_sum_truncating(const input_t *input, output_t *output, int64_t num_items) {
  using NO_ROCM(at_cuda_detail)::cub::Sum;
  inclusive_scan(input, output, Sum{}, num_items);
}

template <typename input_t, typename output_t>
void exclusive_sum_in_common_type(const input_t *input, output_t *output, int64_t num_items) {
  using scalar_t = std::common_type_t<input_t, output_t>;
  exclusive_scan(input, output, SumOp<scalar_t>{}, scalar_t(0), num_items);
}

}}}  // namespace at::cuda::cub
