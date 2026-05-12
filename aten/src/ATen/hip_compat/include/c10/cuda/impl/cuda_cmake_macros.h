#pragma once

// CMake-generated cuda_cmake_macros.h doesn't exist on HIP builds; forward
// to its hip equivalent so c10/cuda/CUDAMacros.h transitive-includes work.
#include <c10/hip/impl/hip_cmake_macros.h>
