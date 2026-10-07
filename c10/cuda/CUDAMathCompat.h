#pragma once

/* This file defines math functions compatible across different gpu
 * platforms (currently CUDA and HIP).
 */
#if defined(__CUDACC__) || defined(__HIPCC__)

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>

#include <cmath>

#ifdef __HIPCC__
#define __MATH_FUNCTIONS_DECL__ inline C10_DEVICE
#else /* __HIPCC__ */
#ifdef __CUDACC_RTC__
#define __MATH_FUNCTIONS_DECL__ C10_HOST_DEVICE
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ inline C10_HOST_DEVICE
#endif /* __CUDACC_RTC__ */
#endif /* __HIPCC__ */

namespace c10::cuda::compat {

__MATH_FUNCTIONS_DECL__ float abs(float x) {
  return ::fabsf(x);
}
__MATH_FUNCTIONS_DECL__ double abs(double x) {
  return ::fabs(x);
}

__MATH_FUNCTIONS_DECL__ float exp(float x) {
  return ::expf(x);
}
__MATH_FUNCTIONS_DECL__ double exp(double x) {
  return ::exp(x);
}

// f32 exp for gfx1250. That arch's v_exp_bf16 instruction is exp2, not exp,
// and is not correctly rounded. If the compiler contracts an f32 exp whose
// result is stored as bf16 into v_exp_bf16, sigmoid disagrees with the CPU
// reference and with gfx942 (which has no v_exp_bf16). The volatile float
// forces the value to exist as f32 so that contraction cannot happen.
// Other architectures return std::exp(float), same as the previous call.
__MATH_FUNCTIONS_DECL__ float exp_f32(float x) {
#if defined(USE_ROCM) && defined(__HIP_DEVICE_COMPILE__)
  if (__builtin_amdgcn_processor_is("gfx1250")) {
    float y = ::expf(x);
    volatile float yv = y;
    return yv;
  }
#endif
  return std::exp(x);
}

// bf16 sigmoid backward. gfx942 and CUDA keep the existing bf16 arithmetic.
// gfx1250 evaluates it in f32 so true16 bf16 muls are not used, matching the
// fused CPU kernel, then rounds once.
__MATH_FUNCTIONS_DECL__ c10::BFloat16 sigmoid_backward_bf16(
    c10::BFloat16 a,
    c10::BFloat16 b) {
#if defined(USE_ROCM) && defined(__HIP_DEVICE_COMPILE__)
  if (__builtin_amdgcn_processor_is("gfx1250")) {
    float ao = static_cast<float>(a);
    float bo = static_cast<float>(b);
    float r = ao * (1.f - bo) * bo;
    volatile float rv = r;
    return static_cast<c10::BFloat16>(rv);
  }
#endif
  return a * (c10::BFloat16(1.f) - b) * b;
}

__MATH_FUNCTIONS_DECL__ float ceil(float x) {
  return ::ceilf(x);
}
__MATH_FUNCTIONS_DECL__ double ceil(double x) {
  return ::ceil(x);
}

__MATH_FUNCTIONS_DECL__ float copysign(float x, float y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysignf(x, y);
#else
  // std::copysign gets ICE/Segfaults with gcc 7.5/8 on arm64
  // (e.g. Jetson), see PyTorch PR #51834
  // This host function needs to be here for the compiler but is never used
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}
__MATH_FUNCTIONS_DECL__ double copysign(double x, double y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysign(x, y);
#else
  // see above
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}

__MATH_FUNCTIONS_DECL__ float floor(float x) {
  return ::floorf(x);
}
__MATH_FUNCTIONS_DECL__ double floor(double x) {
  return ::floor(x);
}

__MATH_FUNCTIONS_DECL__ float log(float x) {
  return ::logf(x);
}
__MATH_FUNCTIONS_DECL__ double log(double x) {
  return ::log(x);
}

__MATH_FUNCTIONS_DECL__ float log1p(float x) {
  return ::log1pf(x);
}

__MATH_FUNCTIONS_DECL__ double log1p(double x) {
  return ::log1p(x);
}

__MATH_FUNCTIONS_DECL__ float max(float x, float y) {
  return ::fmaxf(x, y);
}
__MATH_FUNCTIONS_DECL__ double max(double x, double y) {
  return ::fmax(x, y);
}

__MATH_FUNCTIONS_DECL__ float min(float x, float y) {
  return ::fminf(x, y);
}
__MATH_FUNCTIONS_DECL__ double min(double x, double y) {
  return ::fmin(x, y);
}

__MATH_FUNCTIONS_DECL__ float pow(float x, float y) {
  return ::powf(x, y);
}
__MATH_FUNCTIONS_DECL__ double pow(double x, double y) {
  return ::pow(x, y);
}

__MATH_FUNCTIONS_DECL__ void sincos(float x, float* sptr, float* cptr) {
  return ::sincosf(x, sptr, cptr);
}
__MATH_FUNCTIONS_DECL__ void sincos(double x, double* sptr, double* cptr) {
  return ::sincos(x, sptr, cptr);
}

__MATH_FUNCTIONS_DECL__ float sqrt(float x) {
  return ::sqrtf(x);
}
__MATH_FUNCTIONS_DECL__ double sqrt(double x) {
  return ::sqrt(x);
}

__MATH_FUNCTIONS_DECL__ float rsqrt(float x) {
  return ::rsqrtf(x);
}
__MATH_FUNCTIONS_DECL__ double rsqrt(double x) {
  return ::rsqrt(x);
}

__MATH_FUNCTIONS_DECL__ float tan(float x) {
  return ::tanf(x);
}
__MATH_FUNCTIONS_DECL__ double tan(double x) {
  return ::tan(x);
}

__MATH_FUNCTIONS_DECL__ float tanh(float x) {
  return ::tanhf(x);
}
__MATH_FUNCTIONS_DECL__ double tanh(double x) {
  return ::tanh(x);
}

__MATH_FUNCTIONS_DECL__ float normcdf(float x) {
  return ::normcdff(x);
}
__MATH_FUNCTIONS_DECL__ double normcdf(double x) {
  return ::normcdf(x);
}

} // namespace c10::cuda::compat

#endif
