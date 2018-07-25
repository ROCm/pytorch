#!/bin/bash

set -ex

# Install ccache from source.
# Needs specific branch to work with nvcc (ccache/ccache#145)
# Also pulls in a commit that disables documentation generation,
# as this requires asciidoc to be installed (which pulls in a LOT of deps).
pushd /tmp
git clone https://github.com/pietern/ccache -b ccbin
pushd ccache
./autogen.sh
./configure --prefix=/usr/local
make "-j$(nproc)" install
popd
popd

# Install sccache from pre-compiled binary.
curl https://s3.amazonaws.com/ossci-linux/sccache -o /usr/local/bin/sccache
chmod a+x /usr/local/bin/sccache

# Setup SCCACHE
###############################################################################
mkdir -p ./sccache

SCCACHE="$(which sccache)"
if [ -z "${SCCACHE}" ]; then
  echo "Unable to find sccache..."
  exit 1
fi

# List of compilers to use sccache on.
declare -a compilers=("cc" "c++" "gcc" "g++" "x86_64-linux-gnu-gcc")

# If cuda build, add nvcc to sccache.
if [[ "${BUILD_ENVIRONMENT}" == *-cuda* ]]; then
  compilers+=("nvcc")
fi

# If rocm build, add hcc to sccache.
if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
  compilers+=("hcc")
fi

# Setup wrapper scripts
for compiler in "${compilers[@]}"; do
  (
    echo "#!/bin/sh"
    echo "exec $SCCACHE $(which $compiler) \"\$@\""
  ) > "./sccache/$compiler"
  chmod +x "./sccache/$compiler"
done

export CACHE_WRAPPER_DIR="$PWD/sccache"

# CMake must find these wrapper scripts
export PATH="$CACHE_WRAPPER_DIR:$PATH"
