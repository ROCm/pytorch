#pragma once

// `cuda_runtime.h` is the catch-all CUDA SDK header; on HIP builds, forward
// to the equivalent. cuda_runtime_api.h carries the type/function aliases.
#include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>
