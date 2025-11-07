#pragma once

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/python_stub.h>

namespace torch::utils {

void initializeMemoryFormats();

// This methods returns a borrowed reference!
<<<<<<< HEAD
TORCH_PYTHON_API PyObject* getTHPMemoryFormat(
    c10::MemoryFormat /*memory_format*/);
=======
TORCH_PYTHON_API PyObject* getTHPMemoryFormat(c10::MemoryFormat);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

} // namespace torch::utils
