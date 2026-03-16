#!/bin/bash

export PYTORCH_ROCM_ARCH="gfx90a:xnack+;gfx942:xnack+"

# detect_leaks=0: Python is very leaky, so we need suppress it
# symbolize=1: Gives us much better errors when things go wrong
export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1:symbolize=1
export CMAKE_PREFIX_PATH=/opt/conda

# otherwise any program run at build time will fail
export LD_LIBRARY_PATH=/opt/rocm/llvm/lib/clang/22/lib/linux
export HSA_XNACK=1

# TODO: Make the ASAN flags a centralized env var and unify with USE_ASAN option
export CC="/opt/rocm/llvm/bin/clang"
export CXX="/opt/rocm/llvm/bin/clang++"
export LDSHARED="/opt/rocm/llvm/bin/clang --shared -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld -fsanitize=address -shared-libasan -g"
export CFLAGS="-g -fsanitize=address -shared-libasan -Wno-cast-function-type-strict -fclang-abi-compat=17 -mllvm -asan-use-private-alias=1"
export CXXFLAGS="-g -fsanitize=address -shared-libasan -Wno-cast-function-type-strict -fclang-abi-compat=17 -mllvm -asan-use-private-alias=1"
export USE_ASAN=1
export USE_CUDA=0
export USE_ROCM=1
export USE_MKLDNN=0

# only add these env vars after build is completed
if test "x$BUILD_ONLY_ENV_VARS" = x
then
    # must preload libamdhip64, but should preload the ASAN version of the lib
    export LD_PRELOAD="/opt/rocm/llvm/lib/clang/22/lib/linux/libclang_rt.asan-x86_64.so /opt/rocm/lib/libamdhip64.so"
    # Include torch lib dir so dlopen("libcaffe2_nvrtc.so") succeeds
    # (RUNPATH on the caller doesn't propagate to dlopen lookups)
    TORCH_LIB_DIR=$(python -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null)
    export LD_LIBRARY_PATH="/opt/rocm/llvm/lib/clang/22/lib/linux:/opt/rocm/lib/asan${TORCH_LIB_DIR:+:$TORCH_LIB_DIR}"
    # disable caching allocator, otherwise OOB memory accesses can be masked
    # however, the caching allocator is required for proper hipGraph capture
    # setting the CUDAGRAPH env var can help skip some tests that use that feature
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    export PYTORCH_TEST_SKIP_CUDAGRAPH=1
fi
