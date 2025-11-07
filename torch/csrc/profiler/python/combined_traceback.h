#include <torch/csrc/profiler/combined_traceback.h>

<<<<<<< HEAD
=======
#include <nlohmann/json.hpp>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {

// symbolize combined traceback objects, converting them into lists of
// dictionaries that are easily consumed in python.

// returns std::vector because one use is to call it with a batch of
// tracebacks that come from a larger datastructure (e.g. a memory snapshot)
// and then have more c++ code to put those objects in the right place.
TORCH_API std::vector<pybind11::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize);

<<<<<<< HEAD
=======
// Return the callback in json format so that it can be used within cpp
TORCH_API std::vector<nlohmann::json> json_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize);

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
// requires GIL to be held, frees any pending free frames
TORCH_PYTHON_API void freeDeadCapturedTracebackFrames();

TORCH_PYTHON_API void installCapturedTracebackPython();

} // namespace torch
