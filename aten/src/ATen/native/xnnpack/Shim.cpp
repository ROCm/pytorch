#ifndef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/core/Tensor.h>

//
// This file is here so as to provide an implementation even in cases where
// PyTorch is compiled without XNNPACK support.  Under those scenarios, either
// all XNNPACK usage must be gated with #ifdefs at call-sites which would make
// for cluttered logic, or alternatively, all use can be routed to a central
// place, namely here, where available() calls return false preventing the
// XNNPACK related codepaths to be taken, and use of the actual operators
// trigger an error.
//

namespace at::native::xnnpack {
namespace internal {
namespace {

constexpr const char * const kError =
    "Not Implemented! Reason: PyTorch not built with XNNPACK support.";

} // namespace
} // namespace internal

bool available() {
    return false;
}

bool use_convolution2d(
<<<<<<< HEAD
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const at::OptionalIntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const int64_t /*unused*/,
    bool /*unused*/) {
=======
    const Tensor&,
    const Tensor&,
    const at::OptionalIntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const int64_t,
    bool) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  return false;
}

Tensor convolution2d(
<<<<<<< HEAD
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const int64_t /*unused*/) {
=======
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const int64_t) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  TORCH_CHECK(false, internal::kError);
}

bool use_linear(
<<<<<<< HEAD
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/) {
=======
    const Tensor&,
    const Tensor&,
    const Tensor&) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  return false;
}

Tensor linear(
<<<<<<< HEAD
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/) {
=======
    const Tensor&,
    const Tensor&,
    const Tensor&) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  TORCH_CHECK(false, internal::kError);
}

bool use_max_pool2d(
<<<<<<< HEAD
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const bool /*unused*/,
    const float /*unused*/,
    const float /*unused*/) {
=======
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    IntArrayRef,
    const IntArrayRef,
    const bool,
    const float,
    const float) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  return false;
}

Tensor max_pool2d(
<<<<<<< HEAD
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const bool /*unused*/,
    const float /*unused*/,
    const float /*unused*/) {
=======
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    IntArrayRef,
    const IntArrayRef,
    const bool,
    const float,
    const float) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  TORCH_CHECK(false, internal::kError);
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
