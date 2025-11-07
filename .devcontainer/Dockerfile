<<<<<<< HEAD
FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

#  I am suprised this is needed
RUN conda init

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY .devcontainer/cuda/environment.yml .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && sudo rm -rf /tmp/conda-tmp

# Tools needed for llvm
RUN sudo apt-get -y update
RUN sudo apt install -y lsb-release wget software-properties-common gnupg
=======
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

# Tools needed for development
RUN apt-get -y update && \
    apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenblas-dev

# Tools needed for llvm
RUN apt-get install --no-install-recommends -y lsb-release wget software-properties-common gnupg && \
    sudo apt-get clean -y

# Create Python virtual environment
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install --upgrade pip
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

# Install CLANG if version is specified
ARG CLANG_VERSION
RUN if [ -n "$CLANG_VERSION" ]; then \
<<<<<<< HEAD
    sudo wget https://apt.llvm.org/llvm.sh; \
    chmod +x llvm.sh; \
    sudo ./llvm.sh "${CLANG_VERSION}"; \
    echo 'export CC=clang' >> ~/.bashrc; \
    echo 'export CXX=clang++' >> ~/.bashrc; \
    sudo apt update; \
    sudo apt install -y clang; \
    sudo apt install -y libomp-dev; \
    fi


# Install cuda if version is specified
ARG CUDA_VERSION
RUN if [ -n "$CUDA_VERSION" ]; then \
       conda install -y cuda -c "nvidia/label/cuda-${CUDA_VERSION}"; \
    fi
=======
    wget https://apt.llvm.org/llvm.sh; \
    chmod +x llvm.sh; \
    ./llvm.sh "${CLANG_VERSION}"; \
    echo 'export CC=clang' >> ~/.bashrc; \
    echo 'export CXX=clang++' >> ~/.bashrc; \
    apt-get install --no-install-recommends -y clang libomp-dev && \
    apt-get clean -y; \
    fi


# Install CUDA if version is specified
ARG CUDA_VERSION
RUN if [ -n "$CUDA_VERSION" ]; then \
    CUDA_REPO_VERSION=$(echo ${CUDA_VERSION} | sed 's/\./\-/g'); \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get install --no-install-recommends -y cuda-toolkit-${CUDA_VERSION} && \
    apt-get clean -y; \
    fi

# Set PATH for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PIP_BREAK_SYSTEM_PACKAGES=1
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
