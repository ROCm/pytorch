#pragma once

#include <string>

#include <torch/csrc/Export.h>

namespace torch::profiler::impl::rpd {

TORCH_API bool available();
TORCH_API void prepareTrace();
TORCH_API void startTrace();
TORCH_API void stopTrace();
TORCH_API std::string traceFilePath();

} // namespace torch::profiler::impl::rpd
