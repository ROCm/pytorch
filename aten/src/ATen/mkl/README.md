All files living in this directory are written with the assumption that MKL is available,
which means that these code are not guarded by `#if AT_MKL_ENABLED()`. Therefore, whenever
you need to use definitions from here, please guard the `#include<ATen/mkl/*.h>` and
<<<<<<< HEAD
definition usages with `#if AT_MKL_ENABLED()` macro, e.g. [SpectralOps.cpp](native/mkl/SpectralOps.cpp).
=======
definition usages with `#if AT_MKL_ENABLED()` macro, e.g. [SpectralOps.cpp](../native/mkl/SpectralOps.cpp).
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
