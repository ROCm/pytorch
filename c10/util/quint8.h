<<<<<<< HEAD
#include <torch/headeronly/util/quint8.h>
=======
#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * quint8 is for unsigned 8 bit quantized Tensors
 */
struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  quint8() = default;
  C10_HOST_DEVICE explicit quint8(uint8_t val) : val_(val) {}
};

} // namespace c10
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
