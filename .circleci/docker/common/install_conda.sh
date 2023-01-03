#!/bin/bash

set -ex

# Optionally install conda
if [ -n "$ANACONDA_PYTHON_VERSION" ]; then
  BASE_URL="https://repo.anaconda.com/miniconda"

  MAJOR_PYTHON_VERSION=$(echo "$ANACONDA_PYTHON_VERSION" | cut -d . -f 1)

  case "$MAJOR_PYTHON_VERSION" in
    2)
      CONDA_FILE="Miniconda2-latest-Linux-x86_64.sh"
    ;;
    3)
      CONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"
    ;;
    *)
      echo "Unsupported ANACONDA_PYTHON_VERSION: $ANACONDA_PYTHON_VERSION"
      exit 1
      ;;
  esac

  mkdir /opt/conda
  chown jenkins:jenkins /opt/conda

  # Work around bug where devtoolset replaces sudo and breaks it.
  if [ -n "$DEVTOOLSET_VERSION" ]; then
    SUDO=/bin/sudo
  else
    SUDO=sudo
  fi

  as_jenkins() {
    # NB: unsetting the environment variables works around a conda bug
    # https://github.com/conda/conda/issues/6576
    # NB: Pass on PATH and LD_LIBRARY_PATH to sudo invocation
    # NB: This must be run from a directory that jenkins has access to,
    # works around https://github.com/conda/conda-package-handling/pull/34
    $SUDO -H -u jenkins env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" $*
  }

  pushd /tmp
  wget -q "${BASE_URL}/${CONDA_FILE}"
  # NB: Manually invoke bash per https://github.com/conda/conda/issues/10431
  as_jenkins bash "${CONDA_FILE}" -b -f -p "/opt/conda"
  popd

  # NB: Don't do this, rely on the rpath to get it right
  #echo "/opt/conda/lib" > /etc/ld.so.conf.d/conda-python.conf
  #ldconfig
  sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
  export PATH="/opt/conda/bin:$PATH"

  # Ensure we run conda in a directory that jenkins has write access to
  pushd /opt/conda

  # Prevent conda from updating to 4.14.0, which causes docker build failures
  # See https://hud.pytorch.org/pytorch/pytorch/commit/754d7f05b6841e555cea5a4b2c505dd9e0baec1d
  # Uncomment the below when resolved to track the latest conda update,
  # but this is required for CentOS stream 9 builds
  ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
  OS_VERSION=$(grep -oP '(?<=^VERSION_ID=).+' /etc/os-release | tr -d '"')
  if [[ $ID == centos && $OS_VERSION == 9 ]]; then
    as_jenkins conda update -y -n base conda
  fi

  # Install correct Python version
  as_jenkins conda install -y python="$ANACONDA_PYTHON_VERSION"

  conda_install() {
    # Ensure that the install command don't upgrade/downgrade Python
    # This should be called as
    #   conda_install pkg1 pkg2 ... [-c channel]
    as_jenkins conda install -q -y python="$ANACONDA_PYTHON_VERSION" $*
  }

  # Install PyTorch conda deps, as per https://github.com/pytorch/pytorch README
  # DO NOT install cmake here as it would install a version newer than 3.10, but
  # we want to pin to version 3.10.
  SCIPY_VERSION=1.6.3
  CONDA_COMMON_DEPS="astunparse pyyaml mkl=2022.0.1 mkl-include=2022.0.1 setuptools cffi future six"
  if [ "$ANACONDA_PYTHON_VERSION" = "3.9" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    conda_install numpy=1.19.2 ${CONDA_COMMON_DEPS} llvmdev=8.0.0 -c conda-forge
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.8" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    conda_install numpy=1.18.5 ${CONDA_COMMON_DEPS} llvmdev=8.0.0
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.7" ]; then
    # DO NOT install dataclasses if installing python-3.7, since its part of python-3.7 core packages
    conda_install numpy=1.18.5 ${CONDA_COMMON_DEPS} typing_extensions
  else
    conda_install numpy=1.18.5 ${CONDA_COMMON_DEPS} dataclasses typing_extensions
  fi

  if [[ "$CUDA_VERSION" == 10.2* ]]; then
    conda_install magma-cuda102 -c pytorch
  elif [[ "$CUDA_VERSION" == 11.0* ]]; then
    conda_install magma-cuda110 -c pytorch
  elif [[ "$CUDA_VERSION" == 11.1* ]]; then
    conda_install magma-cuda111 -c pytorch
  elif [[ "$CUDA_VERSION" == 11.3* ]]; then
    conda_install magma-cuda113 -c pytorch
  fi

  # TODO: This isn't working atm
  conda_install nnpack -c killeent

  # Install some other packages, including those needed for Python test reporting
  # TODO: Why is scipy pinned
  # Pin MyPy version because new errors are likely to appear with each release
  # Pin hypothesis to avoid flakiness: https://github.com/pytorch/pytorch/issues/31136
  # Pin coverage so we can use COVERAGE_RCFILE
  as_jenkins pip install --progress-bar off pytest \
    scipy==$SCIPY_VERSION \
    scikit-image \
    psutil \
    unittest-xml-reporting \
    boto3==1.16.34 \
    coverage==5.5 \
    hypothesis==4.53.2 \
    expecttest==0.1.3 \
    mypy==0.812 \
    tb-nightly

  # Install numba only on python-3.8 or below
  # For numba issue see https://github.com/pytorch/pytorch/issues/51511
  if [[ $(python -c "import sys; print(int(sys.version_info < (3, 9)))") == "1" ]]; then
    as_jenkins pip install --progress-bar off numba "librosa>=0.6.2<0.9.0"
  else
    as_jenkins pip install --progress-bar off numba==0.49.0 "librosa>=0.6.2,<0.9.0"
  fi

  # Update scikit-learn to a python-3.8 compatible version
  if [[ $(python -c "import sys; print(int(sys.version_info >= (3, 8)))") == "1" ]]; then
    as_jenkins pip install --progress-bar off -U scikit-learn
  else
    # Pinned scikit-learn due to https://github.com/scikit-learn/scikit-learn/issues/14485 (affects gcc 5.5 only)
    as_jenkins pip install --progress-bar off scikit-learn==0.20.3
  fi

  popd
fi
