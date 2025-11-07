#ifndef THCP_GDSFILE_INC
#define THCP_GDSFILE_INC

#include <torch/csrc/python_headers.h>

<<<<<<< HEAD
void initGdsBindings(PyObject* module);
=======
namespace torch::cuda::shared {
void initGdsBindings(PyObject* module);
}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#endif // THCP_GDSFILE_INC
