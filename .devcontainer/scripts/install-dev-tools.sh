#!/usr/bin/env bash
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below
pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive

# This takes some time
make setup-lint

# Add CMAKE_PREFIX_PATH to bashrc
<<<<<<< HEAD
echo 'export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}' >> ~/.bashrc
# Add linker path so that cuda-related libraries can be found
echo 'export LDFLAGS="-L${CONDA_PREFIX}/lib/ $LDFLAGS"' >> ~/.bashrc
=======
echo 'export CMAKE_PREFIX_PATH=/usr/local' >> ~/.bashrc
# Add linker path so that cuda-related libraries can be found
echo 'export LDFLAGS="-L/usr/local/cuda/lib64/ $LDFLAGS"' >> ~/.bashrc
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
