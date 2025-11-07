// No "#pragma once" because this is a raw definition that can be copied by jit codegen.
// Eager mode clients should not include this file directly, instead,
// they should #include <ATen/cuda/PhiloxCudaState.h>, which has a #pragma once.

// Stores RNG state values. Passed as a kernel argument.
// See Note [CUDA Graph-safe RNG states].
//
// The raw definition lives in its own file so jit codegen can easily copy it.
namespace at {

struct PhiloxCudaState {
  PhiloxCudaState() = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed,
                  uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(int64_t* seed,
                  int64_t* offset_extragraph,
<<<<<<< HEAD
                  uint64_t offset_intragraph) {
=======
                  uint32_t offset_intragraph) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_{};
  Payload offset_{};
<<<<<<< HEAD
  uint64_t offset_intragraph_ = 0;
=======
  uint32_t offset_intragraph_ = 0;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  bool captured_ = false;
};

} // namespace at
