#pragma once

// Shim of `<ATen/cudnn/Handle.h>` for HIP builds. Forwards the cuDNN handle
// symbols to their hipDNN equivalents so non-hipified files compile against
// the cuDNN-named API.

#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/hipdnn-wrapper.h>

using cudnnHandle_t = hipdnnHandle_t;
inline cudnnHandle_t getCudnnHandle() {
  return at::native::getHipdnnHandle();
}
