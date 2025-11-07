#!/usr/bin/env bash
<<<<<<< HEAD
# Script that replaces the magma install from a conda package

set -eou pipefail

function do_install() {
    cuda_version_nodot=${1/./}
    anaconda_python_version=$2

    MAGMA_VERSION="2.6.1"
    magma_archive="magma-cuda${cuda_version_nodot}-${MAGMA_VERSION}-1.tar.bz2"

    anaconda_dir="/opt/conda/envs/py_${anaconda_python_version}"
    (
        set -x
        tmp_dir=$(mktemp -d)
        pushd ${tmp_dir}
        curl -OLs https://ossci-linux.s3.us-east-1.amazonaws.com/${magma_archive}
        tar -xvf "${magma_archive}"
        mv include/* "${anaconda_dir}/include/"
        mv lib/* "${anaconda_dir}/lib"
        popd
    )
}

do_install $1 $2
=======
# Script that installs magma from tarball inside conda environment.
# It replaces anaconda magma-cuda package which is no longer published.
# Execute it inside active conda environment.
# See issue: https://github.com/pytorch/pytorch/issues/138506

set -eou pipefail

cuda_version_nodot=${1/./}
anaconda_dir=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

MAGMA_VERSION="2.6.1"
magma_archive="magma-cuda${cuda_version_nodot}-${MAGMA_VERSION}-1.tar.bz2"
(
    set -x
    tmp_dir=$(mktemp -d)
    pushd ${tmp_dir}
    curl -OLs https://ossci-linux.s3.us-east-1.amazonaws.com/${magma_archive}
    tar -xvf "${magma_archive}"
    mv include/* "${anaconda_dir}/include/"
    mv lib/* "${anaconda_dir}/lib"
    popd
)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
