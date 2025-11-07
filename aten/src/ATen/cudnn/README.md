All files living in this directory are written with the assumption that cuDNN is available,
which means that these code are not guarded by `#if AT_CUDNN_ENABLED()`. Therefore, whenever
you need to use definitions from here, please guard the `#include<ATen/cudnn/*.h>` and
<<<<<<< HEAD
definition usages with `#if AT_CUDNN_ENABLED()` macro, e.g. [native/cudnn/BatchNorm.cpp](native/cudnn/BatchNorm.cpp).
=======
definition usages with `#if AT_CUDNN_ENABLED()` macro, e.g. [native/cudnn/BatchNorm.cpp](../native/cudnn/BatchNorm.cpp).
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
