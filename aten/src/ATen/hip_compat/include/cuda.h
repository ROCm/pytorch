#pragma once

// CUDA driver API header; on HIP, forward to hip_runtime which exposes the
// equivalent driver entry points used by c10/cuda/CUDAException.h.
#include <hip/hip_runtime.h>
