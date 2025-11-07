#!/bin/bash

set -euxo pipefail

# Download requirements
cd llm-target-determinator
pip install -q -r requirements.txt
cd ../codellama
<<<<<<< HEAD
pip install --no-build-isolation -v -e .
=======
pip install -e .
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
pip install numpy==1.26.0

# Run indexer
cd ../llm-target-determinator

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    indexer.py \
    --experiment-name indexer-files \
    --granularity FILE
