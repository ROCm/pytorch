#pragma once

#include <ATen/cuda/CUDAContextLight.h>

// Preserved for BC, as many files depend on these includes
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/Exceptions.h>

#if defined(USE_ROCM)
#define ROCBLAS_BETA_FEATURES_API
#define ROCBLAS_NO_DEPRECATED_WARNINGS
#endif
