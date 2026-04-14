#pragma once

// Shim of `<ATen/cuda/Exceptions.h>` for HIP builds. Defines
// AT_CUDNN_FRONTEND_CHECK in terms of hipDNN's check macro.

#include <ATen/hipdnn/Exceptions.h>

#define AT_CUDNN_FRONTEND_CHECK(e) HIPDNN_FE_CHECK(e)
