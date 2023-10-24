#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>
#include <random>

namespace at {
namespace native {

template <typename T>
inline std::vector<T> _expand_param_if_needed(
    ArrayRef<T> list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<T>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

inline std::vector<int64_t> expand_param_if_needed(
    IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  return _expand_param_if_needed(list_param, param_name, expected_dim);
}

inline std::vector<c10::SymInt> expand_param_if_needed(
    SymIntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  return _expand_param_if_needed(list_param, param_name, expected_dim);
}

static inline uint32_t random_seed()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
    return distribution(gen);
}



} // namespace native
} // namespace at
