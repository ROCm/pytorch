#pragma once

namespace torch::utils {

inline constexpr bool cuda_enabled() {
#ifdef USE_ROCM
  return true;
#else
  return false;
#endif
}

} // namespace torch::utils
