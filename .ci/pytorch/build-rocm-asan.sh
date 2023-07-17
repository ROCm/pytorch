#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

echo "Clang version:"
/opt/rocm/llvm/bin/clang --version

# hipify sources
python tools/amd_build/build_amd.py

# sccache somehow forces gfx906 -x hip, remove it all
rm -rf /opt/cache

# patch fbgemm to work around build failure
pushd third_party/fbgemm
patch -p1 -i ../../.ci/pytorch/fbgemm.patch || true
popd

python tools/stats/export_test_times.py

# detect_leaks=0: Python is very leaky, so we need suppress it
# symbolize=1: Gives us much better errors when things go wrong
export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1:symbolize=1:detect_odr_violation=0
export CMAKE_PREFIX_PATH=/opt/conda

# otherwise any program run at build time will fail
export LD_LIBRARY_PATH=/opt/rocm/llvm/lib/clang/17.0.0/lib/linux
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
python setup.py bdist_wheel
python -mpip install "$(echo dist/*.whl)[opt-einsum]"
