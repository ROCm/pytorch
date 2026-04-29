# Dockerfile
#
# PyTorch + ROCm image built from TheRock portable wheels.
# Does NOT include kernel drivers — the host must provide compatible
# AMDGPU/ROCm kernel components and device access.
#
# Recommended docker run flags (mirrors TheRock CI container options):
#   docker run \
#     --shm-size=10g \
#     --cap-add=SYS_PTRACE \
#     --group-add video \
#     --device /dev/kfd \
#     --device /dev/dri \
#     <image>
#
# Supported base images (examples)
# - ubuntu:24.04
# - almalinux:8
# - mcr.microsoft.com/azurelinux/base/core:3.0
#
# Build arguments
# - BASE_IMAGE       : Base Docker image (default: ubuntu:24.04)
# - ROCM_VERSION     : Full ROCm version string. Supported formats:
#                      - Nightly: 7.13.0a20260413
#                      - Dev: 7.12.0.dev0+849eec43b2075459511b9a9ffe3bf1948490e9ee
# - AMDGPU_FAMILY    : AMD GPU family (e.g., gfx94X-dcgpu, gfx90X-dcgpu, gfx950-dcgpu)
# - PYTHON_VERSION   : Python version for PyTorch (default: 3.12)
# - INDEX_URL        : (Required) Base URL for PyTorch wheels index
# - PYTORCH_ROCM_ARCH: GPU arch target for source builds / local rebuilds
# - TORCH_VERSION      : Optional specific PyTorch version. If not set, installs latest.
# - TORCHAUDIO_VERSION : Optional specific torchaudio version. If not set, installs latest.
# - TORCHVISION_VERSION: Optional specific torchvision version. If not set, installs latest.
# - TRITON_VERSION     : Optional specific triton version. If not set, uses torch's dependency.
#
# Note: The PyTorch source is included at /tmp/pytorch (from the repo root).
#
# Build example (run from repo root):
#
#   docker build \
#     --build-arg BASE_IMAGE=ubuntu:24.04 \
#     --build-arg ROCM_VERSION=7.13.0a20260413 \
#     --build-arg AMDGPU_FAMILY=gfx94X-dcgpu \
#     --build-arg PYTHON_VERSION=3.12 \
#     --build-arg INDEX_URL=https://rocm.nightlies.amd.com/v2-staging \
#     -f dockerfiles/Dockerfile \
#     -t pytorch-rocm:ubuntu24.04-gfx94X-dcgpu-7.13.0a20260413 \
#     .
#

# Base image selection
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# ROCm configuration arguments
ARG ROCM_VERSION
ARG AMDGPU_FAMILY
ARG RELEASE_TYPE=nightly

# PyTorch configuration arguments
ARG PYTHON_VERSION=3.12
ARG INDEX_URL
ARG PYTORCH_ROCM_ARCH
ARG TORCH_VERSION
ARG TORCH_VERSION_PREFIX
ARG TORCHAUDIO_VERSION
ARG TORCHVISION_VERSION
ARG TRITON_VERSION

# Copy installation scripts
COPY .github/scripts/install_rocm_deps.sh /tmp/
COPY .github/scripts/install_pytorch_wheels.py /tmp/

# Copy PyTorch source from the repo root
COPY . /tmp/pytorch

# Install system dependencies
RUN chmod +x /tmp/install_rocm_deps.sh && \
    /tmp/install_rocm_deps.sh

# Install ccache for PyTorch source rebuilds in the image.
RUN if ! command -v ccache >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y ccache; \
    fi

# Install the requested Python version if not already available.
# Ubuntu 24.04 ships with 3.12; other versions come from deadsnakes PPA.
RUN if ! command -v python${PYTHON_VERSION} >/dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends software-properties-common && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python${PYTHON_VERSION} \
            python${PYTHON_VERSION}-dev \
            python${PYTHON_VERSION}-venv && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Create Python virtual environment and upgrade pip/setuptools
RUN python${PYTHON_VERSION} -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip && \
    /opt/venv/bin/python -m pip install --upgrade setuptools

ENV PATH="/opt/venv/bin:${PATH}"

# Install PyTorch wheels from the public nightlies index.
RUN /opt/venv/bin/python /tmp/install_pytorch_wheels.py \
        --no-break-system-packages \
        --skip-verify \
        --index-url "${INDEX_URL}" \
        --amdgpu-family "${AMDGPU_FAMILY}" \
        ${ROCM_VERSION:+--rocm-version "${ROCM_VERSION}"} \
        ${TORCH_VERSION:+--torch-version "${TORCH_VERSION}"} \
        ${TORCH_VERSION_PREFIX:+--torch-version-prefix "${TORCH_VERSION_PREFIX}"} \
        ${TORCHAUDIO_VERSION:+--torchaudio-version "${TORCHAUDIO_VERSION}"} \
        ${TORCHVISION_VERSION:+--torchvision-version "${TORCHVISION_VERSION}"} \
        ${TRITON_VERSION:+--triton-version "${TRITON_VERSION}"}

# Run rocm-sdk init to make rocm buildable
RUN rocm-sdk init

# ROCm environment variables (mirrors TheRock CI setup in
# test_pytorch_wheels_full.yml "Initialize ROCm SDK and configure environment").
# All paths derive from ROCM_HOME which is the rocm-sdk install location.
ENV ROCM_HOME="/opt/venv/lib/python${PYTHON_VERSION}/site-packages/_rocm_sdk_devel" \
    ROCM_PATH="/opt/venv/lib/python${PYTHON_VERSION}/site-packages/_rocm_sdk_devel" \
    ROCM_SOURCE_DIR="/opt/venv/lib/python${PYTHON_VERSION}/site-packages/_rocm_sdk_devel" \
    ROCM_BIN="/opt/venv/lib/python${PYTHON_VERSION}/site-packages/_rocm_sdk_devel/bin" \
    ROCM_CMAKE="/opt/venv/lib/python${PYTHON_VERSION}/site-packages/_rocm_sdk_devel/lib/cmake" \
    VIRTUAL_ENV=/opt/venv \
    USE_MSLK=0

ENV CMAKE_PREFIX_PATH="${ROCM_CMAKE}" \
    HIP_DEVICE_LIB_PATH="${ROCM_HOME}/lib/llvm/amdgcn/bitcode" \
    ROCM_DEVICE_LIB_PATH="${ROCM_HOME}/lib/llvm/amdgcn/bitcode" \
    ROCM_SYSDEPS_INCLUDE="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    CPLUS_INCLUDE_PATH="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    C_INCLUDE_PATH="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    PKG_CONFIG_PATH="${ROCM_HOME}/lib/rocm_sysdeps/lib/pkgconfig" \
    LD_LIBRARY_PATH="${ROCM_HOME}/lib/host-math/lib:${ROCM_HOME}/lib/rocm_sysdeps/lib" \
    LIBRARY_PATH="${ROCM_HOME}/lib/host-math/lib:${ROCM_HOME}/lib/rocm_sysdeps/lib" \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    CMAKE_HIP_COMPILER_LAUNCHER=ccache \
    PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
    PATH="${ROCM_BIN}:${PATH}"

# Verify PyTorch imports and environment
RUN python <<'PYEOF'
import os, torch
print('torch', torch.__version__)
print('ROCm/HIP', torch.version.hip)
print(f'ROCM_HOME={os.environ.get("ROCM_HOME", "NOT SET")}')
print(f'CC={os.environ.get("CC", "NOT SET")}')
print(f'CXX={os.environ.get("CXX", "NOT SET")}')
print(f'PYTORCH_ROCM_ARCH={os.environ.get("PYTORCH_ROCM_ARCH", "NOT SET")}')
for mod in ['torchaudio', 'torchvision', 'triton']:
    try:
        m = __import__(mod)
        print(f'{mod} {m.__version__}')
    except Exception as e:
        print(f'{mod}: skipped ({e})')
PYEOF

# Clean up installation scripts
RUN rm -f /tmp/install_rocm_deps.sh /tmp/install_pytorch_wheels.py

WORKDIR /tmp/pytorch
