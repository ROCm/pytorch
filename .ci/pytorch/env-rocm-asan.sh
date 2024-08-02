#!/bin/bash

export PYTORCH_ROCM_ARCH="gfx942:xnack+"

# detect_leaks=0: Python is very leaky, so we need suppress it
# symbolize=1: Gives us much better errors when things go wrong
export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1:symbolize=1:detect_odr_violation=0
export CMAKE_PREFIX_PATH=/opt/conda

# otherwise any program run at build time will fail
# TODO: Remove hardcoded python version specific paths
export LD_LIBRARY_PATH="/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel/llvm/lib/clang/22/lib/linux:/opt/conda/envs/py_3.10/lib/python3.10/site-packages/_rocm_sdk_devel/llvm/lib/clang/22/lib/linux"
export HSA_XNACK=1

# TODO: Make the ASAN flags a centralized env var and unify with USE_ASAN option
# TODO: Keeping the CC and CXX to point to hipcc instead of clang/clang++
export CC="/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel/bin/hipcc"
export CXX="/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel/bin/hipcc"
export LDSHARED="/opt/rocm/llvm/bin/clang --shared -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld -fsanitize=address -shared-libasan -g"
export CFLAGS="-g -fsanitize=address -shared-libasan -Wno-cast-function-type-strict"
export CXXFLAGS="-g -fsanitize=address -shared-libasan -Wno-cast-function-type-strict"
export USE_ASAN=1
export USE_CUDA=0
export USE_ROCM=1
export USE_MKLDNN=0

# only add these env vars after build is completed
# TODO: Remove hardcoded python version specific paths
if test "x$BUILD_ONLY_ENV_VARS" = x
then
    export LD_PRELOAD="/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel/llvm/lib/clang/22/lib/linux/libclang_rt.asan-x86_64.so:/opt/conda/envs/py_3.10/lib/python3.10/site-packages/_rocm_sdk_devel/llvm/lib/clang/22/lib/linux/libclang_rt.asan-x86_64.so"
    export LD_LIBRARY_PATH="/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel/llvm/lib/clang/22/lib/linux:/opt/conda/envs/py_3.10/lib/python3.10/site-packages/_rocm_sdk_devel/llvm/lib/clang/22/lib/linux"
fi
