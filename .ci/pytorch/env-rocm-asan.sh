#!/bin/bash

export PYTORCH_ROCM_ARCH="gfx90a:xnack+;gfx942:xnack+"

# detect_leaks=0: Python is very leaky, so we need suppress it
# symbolize=1: Gives us much better errors when things go wrong
export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1:symbolize=1:detect_odr_violation=0
export CMAKE_PREFIX_PATH=/opt/conda

# otherwise any program run at build time will fail
export LD_LIBRARY_PATH=/opt/rocm/llvm/lib/clang/17/lib/linux
export HSA_XNACK=1

# TODO: Make the ASAN flags a centralized env var and unify with USE_ASAN option
export CC="/opt/rocm/llvm/bin/clang"
export CXX="/opt/rocm/llvm/bin/clang++"
export LDSHARED="/opt/rocm/llvm/bin/clang --shared -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld -fsanitize=address -shared-libasan -g"
export CFLAGS="-g -fsanitize=address -shared-libasan -Wno-cast-function-type-strict"
export CXXFLAGS="-g -fsanitize=address -shared-libasan -Wno-cast-function-type-strict"
export USE_ASAN=1
export USE_CUDA=0
export USE_ROCM=1
export USE_MKLDNN=0

# only add these env vars after build is completed
if test "x$BUILD_ONLY_ENV_VARS" = x
then
    export LD_PRELOAD="/opt/rocm/llvm/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so"
    export LD_LIBRARY_PATH="/opt/rocm/llvm/lib/clang/17/lib/linux:/opt/rocm/lib/asan"
fi
