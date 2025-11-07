#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname $(dirname "$script_path")))
tp2_dir="$top_dir/third_party"

pip install ninja

# Install onnx
pip install --no-use-pep517 -e "$tp2_dir/onnx"

# Install caffe2 and pytorch
pip install -r "$top_dir/caffe2/requirements.txt"
pip install -r "$top_dir/requirements.txt"
<<<<<<< HEAD
python -m pip install --no-build-isolation -v -e .
=======
python setup.py develop
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
