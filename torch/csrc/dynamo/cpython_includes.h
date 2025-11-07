#pragma once

#include <torch/csrc/utils/python_compat.h>

// Problem in CPython includes when mixing core and non-core build
// The fix was not backported to 3.12 so this is needed here
// https://github.com/python/cpython/issues/105268
#if IS_PYTHON_3_12_PLUS
#undef _PyGC_FINALIZED
#endif

// see https://bugs.python.org/issue35886
<<<<<<< HEAD
=======
#if PY_VERSION_HEX >= 0x03080000
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#define Py_BUILD_CORE

#ifndef __cplusplus
// C-only headers
#include <internal/pycore_pystate.h>

#endif // __cplusplus

#if IS_PYTHON_3_11_PLUS
#include <internal/pycore_frame.h>
<<<<<<< HEAD

#if IS_PYTHON_3_14_PLUS && !defined(_WIN32)
#include <internal/pycore_code.h>
#include <internal/pycore_genobject.h>
#include <internal/pycore_interpframe.h>
#include <internal/pycore_stackref.h>
#elif IS_PYTHON_3_14_PLUS && defined(_WIN32)
#include <internal/pycore_interpframe_structs.h> // _PyInterpreterFrame
#endif

#endif

#undef Py_BUILD_CORE
=======
#endif

#undef Py_BUILD_CORE
#endif // PY_VERSION_HEX >= 0x03080000
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

#ifdef __cplusplus
extern "C" {
#endif

<<<<<<< HEAD
#if IS_PYTHON_3_14_PLUS && !defined(_WIN32)

#define F_CODE(x) ((PyCodeObject*)PyStackRef_AsPyObjectBorrow(x->f_executable))
#define PREV_INSTR(x) (x)->instr_ptr

#elif IS_PYTHON_3_14_PLUS && defined(_WIN32)

#define F_CODE(x) ((PyCodeObject*)((x)->f_executable.bits))
#define PREV_INSTR(x) (x)->instr_ptr

#else

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#if IS_PYTHON_3_13_PLUS
#define F_CODE(x) ((PyCodeObject*)(x)->f_executable)
#define PREV_INSTR(x) (x)->instr_ptr
#else
#define F_CODE(x) ((PyCodeObject*)(x)->f_code)
#define PREV_INSTR(x) (x)->prev_instr
#endif

<<<<<<< HEAD
#endif // IS_PYTHON_3_14_PLUS

#if IS_PYTHON_3_14_PLUS && !defined(_WIN32)
#define FUNC(x) ((PyFunctionObject*)PyStackRef_AsPyObjectBorrow((x)->f_funcobj))
#elif IS_PYTHON_3_14_PLUS && defined(_WIN32)
#define FUNC(x) ((PyFunctionObject*)((x)->f_funcobj.bits))
#elif IS_PYTHON_3_12_PLUS
#define FUNC(x) ((PyFunctionObject*)(x)->f_funcobj)
#else
#define FUNC(x) ((PyFunctionObject*)(x)->f_func)
=======
#if IS_PYTHON_3_12_PLUS
#define FUNC(x) ((x)->f_funcobj)
#else
#define FUNC(x) ((x)->f_func)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#endif

#ifdef __cplusplus
} // extern "C"
#endif
