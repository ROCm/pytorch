#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

if [[ "$OS" != "windows-arm64" ]]; then
    export CUDA_VERSION="${DESIRED_CUDA/cu/}"
    export USE_SCCACHE=1
    export SCCACHE_BUCKET=ossci-compiler-cache
    export SCCACHE_IGNORE_SERVER_IO_ERROR=1
    export VC_YEAR=2022
fi

if [[ "$DESIRED_CUDA" == 'xpu' ]]; then
    export VC_YEAR=2022
    export USE_SCCACHE=0
<<<<<<< HEAD
    export XPU_VERSION=2025.2
=======
    export XPU_VERSION=2025.1
    export XPU_ENABLE_KINETO=1
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
fi

echo "Free space on filesystem before build:"
df -h

pushd "$PYTORCH_ROOT/.ci/pytorch/"
export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"

if [[ "$OS" == "windows-arm64" ]]; then
    if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
        ./windows/arm64/build_libtorch.bat
    elif [[ "$PACKAGE_TYPE" == 'wheel' ]]; then
        ./windows/arm64/build_pytorch.bat
    fi
else
    ./windows/internal/build_wheels.bat
fi

echo "Free space on filesystem after build:"
df -h
