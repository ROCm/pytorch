#pragma once

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

namespace at { namespace detail {

template <typename T>
inline T load(const void* data, ScalarType src_type) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, src_type, "load", [&]() {
    return at::convert<T>(*(scalar_t*)data);
  });
}

template <typename T>
inline void store(T value, void* dst, ScalarType dst_type) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, dst_type, "store", [&]() {
    *(scalar_t*)dst = at::convert<scalar_t>(value);
  });
}

}} // namespace at::detail
