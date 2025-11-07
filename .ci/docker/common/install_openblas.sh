#!/bin/bash
# Script used only in CD pipeline

set -ex

<<<<<<< HEAD
OPENBLAS_VERSION=${OPENBLAS_VERSION:-"v0.3.30"}

# Clone OpenBLAS
git clone https://github.com/OpenMathLib/OpenBLAS.git -b "${OPENBLAS_VERSION}" --depth 1 --shallow-submodules
=======
cd /
git clone https://github.com/OpenMathLib/OpenBLAS.git -b "${OPENBLAS_VERSION:-v0.3.30}" --depth 1 --shallow-submodules
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

OPENBLAS_CHECKOUT_DIR="OpenBLAS"
OPENBLAS_BUILD_FLAGS="
NUM_THREADS=128
USE_OPENMP=1
NO_SHARED=0
DYNAMIC_ARCH=1
TARGET=ARMV8
CFLAGS=-O3
BUILD_BFLOAT16=1
"

<<<<<<< HEAD
make -j8 ${OPENBLAS_BUILD_FLAGS} -C $OPENBLAS_CHECKOUT_DIR
sudo make install -C $OPENBLAS_CHECKOUT_DIR

rm -rf $OPENBLAS_CHECKOUT_DIR
=======
make -j8 ${OPENBLAS_BUILD_FLAGS} -C ${OPENBLAS_CHECKOUT_DIR}
make -j8 ${OPENBLAS_BUILD_FLAGS} install -C ${OPENBLAS_CHECKOUT_DIR}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
