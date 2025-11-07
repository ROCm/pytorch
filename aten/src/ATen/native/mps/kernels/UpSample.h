#pragma once
<<<<<<< HEAD
#include <c10/metal/common.h>

template <unsigned N = 5>
struct UpsampleParams {
  ::c10::metal::array<uint64_t, N> input_strides;
  ::c10::metal::array<uint64_t, N> input_sizes;
  ::c10::metal::array<uint64_t, N> output_strides;
  ::c10::metal::array<uint64_t, N> output_sizes;
  ::c10::metal::array<float, N - 2> scales;
=======

#ifndef __METAL__
#include <array>
using ulong = unsigned long;
#define _ARRAY_NS std
#else
#include <metal_array>
#define _ARRAY_NS metal
#endif

template <unsigned N = 5>
struct UpsampleParams {
  _ARRAY_NS::array<ulong, N> input_strides;
  _ARRAY_NS::array<ulong, N> input_sizes;
  _ARRAY_NS::array<ulong, N> output_strides;
  _ARRAY_NS::array<ulong, N> output_sizes;
  _ARRAY_NS::array<float, N - 2> scales;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  bool align_corners;
};
