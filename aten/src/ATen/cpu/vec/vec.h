#pragma once

#if defined(CPU_CAPABILITY_AVX512)
#include <ATen/cpu/vec/vec512/vec512.h>
#else
#include <ATen/cpu/vec/vec128/vec128.h>
#include <ATen/cpu/vec/vec256/vec256.h>
#endif

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

inline Vectorized<bool> convert_to_bool(Vectorized<int8_t> x) {
  __at_align__ bool buffer[x.size()];
  x.ne(Vectorized<int8_t>(0)).store(buffer);

  Vectorized<bool> ret;
  static_assert(x.size() == ret.size());
  std::memcpy(ret, buffer, ret.size() * sizeof(bool));
  return ret;
}

template <>
inline Vectorized<bool> Vectorized<bool>::loadu(const void* ptr) {
  // See NOTE [Loading boolean values]
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr));
}

template <>
<<<<<<< HEAD
inline Vectorized<bool> Vectorized<bool>::loadu(const void* ptr, int64_t count) {
=======
inline Vectorized<bool> Vectorized<bool>::loadu(
    const void* ptr,
    int64_t count) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  // See NOTE [Loading boolean values]
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr, count));
}

template <typename VT>
<<<<<<< HEAD
struct VecHoldType { using hold_type = typename VT::value_type; };

template <>
struct VecHoldType<Vectorized<BFloat16>> { using hold_type = BFloat16; };

template <>
struct VecHoldType<Vectorized<Half>> {using hold_type = Half; };
=======
struct VecHoldType {
  using hold_type = typename VT::value_type;
};

template <>
struct VecHoldType<Vectorized<BFloat16>> {
  using hold_type = BFloat16;
};

template <>
struct VecHoldType<Vectorized<Half>> {
  using hold_type = Half;
};
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

template <typename VT>
using vechold_type = typename VecHoldType<VT>::hold_type;

<<<<<<< HEAD
}} // namespace at::vec::CPU_CAPABILITY
=======
} // namespace CPU_CAPABILITY
} // namespace at::vec
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
