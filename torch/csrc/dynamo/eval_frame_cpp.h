#pragma once
#include <Python.h>

#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#ifdef __cplusplus

extern "C" {

#endif

PyObject* dynamo__custom_eval_frame(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag,
    PyObject* callback);

PyObject* set_code_exec_strategy(PyObject* dummy, PyObject* obj);
<<<<<<< HEAD
void skip_code_recursive(PyCodeObject* code);
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

#ifdef __cplusplus

} // extern "C"

#endif
