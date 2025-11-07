#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <ostream>

#if defined(__CUDACC__) && !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif

<<<<<<< HEAD
#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp> // for SYCL 1.2.1
#else
#include <sycl/sycl.hpp> // for SYCL 2020
#endif
#include <ext/oneapi/bfloat16.hpp>
#endif
=======
#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp> // for SYCL 1.2.1
#elif defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp> // for SYCL 2020
#endif
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

namespace c10 {

namespace detail {
inline C10_HOST_DEVICE float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;

<<<<<<< HEAD
#if defined(USE_ROCM)
=======
#if defined(USE_ROCM) && defined(__HIPCC__)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  float* tempRes;

  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  tempRes = reinterpret_cast<float*>(&tmp);
  res = *tempRes;
#else
  std::memcpy(&res, &tmp, sizeof(tmp));
#endif

  return res;
}

inline C10_HOST_DEVICE uint16_t bits_from_f32(float src) {
  uint32_t res = 0;

<<<<<<< HEAD
#if defined(USE_ROCM)
=======
#if defined(USE_ROCM) && defined(__HIPCC__)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  uint32_t* tempRes = reinterpret_cast<uint32_t*>(&src);
  res = *tempRes;
#else
  std::memcpy(&res, &src, sizeof(res));
#endif

  return res >> 16;
}

inline C10_HOST_DEVICE uint16_t round_to_nearest_even(float src) {
<<<<<<< HEAD
#if defined(USE_ROCM)
=======
#if defined(USE_ROCM) && defined(__HIPCC__)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  if (src != src) {
#elif defined(_MSC_VER)
  if (isnan(src)) {
#else
  if (std::isnan(src)) {
#endif
    return UINT16_C(0x7FC0);
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t U32; // NOLINT(facebook-hte-BadMemberName)
      float F32; // NOLINT(facebook-hte-BadMemberName)
    };

    F32 = src;
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}
} // namespace detail

struct alignas(2) BFloat16 {
  uint16_t x;

  // HIP wants __host__ __device__ tag, CUDA does not
<<<<<<< HEAD
#if defined(USE_ROCM)
=======
#if defined(USE_ROCM) && defined(__HIPCC__)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t)
      : x(bits) {}
  /* implicit */ inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;

#if defined(__CUDACC__) && !defined(USE_ROCM)
  inline C10_HOST_DEVICE BFloat16(const __nv_bfloat16& value);
  explicit inline C10_HOST_DEVICE operator __nv_bfloat16() const;
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
  inline C10_HOST_DEVICE BFloat16(const sycl::ext::oneapi::bfloat16& value);
  explicit inline C10_HOST_DEVICE operator sycl::ext::oneapi::bfloat16() const;
#endif
};

C10_API inline std::ostream& operator<<(
    std::ostream& out,
    const BFloat16& value) {
  out << (float)value;
  return out;
}

} // namespace c10

#include <c10/util/BFloat16-inl.h> // IWYU pragma: keep
