#pragma once

<<<<<<< HEAD
#include <cstdint>

namespace c10d::symmetric_memory {

// Covers NVL72
constexpr int max_cuda_p2p_domain_size = 72;
// Maximum number of channels
constexpr int symm_max_nblocks = 32;

// Maximally, a rank will need to sync with all other ranks, over all
// channels. Each signal is 32 bits, which is the minimum unit for atomic cas.
constexpr size_t signal_pad_size =
    symm_max_nblocks * max_cuda_p2p_domain_size * sizeof(uint32_t);
=======
namespace c10d::symmetric_memory {

constexpr size_t signal_pad_size = 2048;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
using HandleType = CUmemGenericAllocationHandle;
#elif defined(USE_ROCM)
using HandleType = hipMemGenericAllocationHandle_t;
#else
using HandleType = void*;
#endif

} // namespace c10d::symmetric_memory
