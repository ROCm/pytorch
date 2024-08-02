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
pushd /opt/rocm/llvm/bin
if [[ -d original ]]; then
  sudo mv original/clang .
  sudo mv original/clang++ .
fi
sudo rm -rf original
popd
sudo rm -rf /opt/cache

# patch fbgemm to work around build failure
pushd third_party/fbgemm
patch -p1 -i ../../.ci/pytorch/fbgemm.patch || true
popd

# patch XNNPACK to work around build failure
pushd third_party/XNNPACK
patch -p1 -i ../../.ci/pytorch/XNNPACK.patch || true
popd

python tools/stats/export_test_times.py

# shellcheck source=./env-rocm-asan.sh
export BUILD_ONLY_ENV_VARS=1
source "$(dirname "${BASH_SOURCE[0]}")/env-rocm-asan.sh"

python setup.py bdist_wheel
pip_install_whl "$(echo dist/*.whl)"
