#!/bin/bash

<<<<<<< HEAD
base_url='https://github.com/AlnisM/autoheuristic-datasets/raw/main/'
=======
base_url='https://github.com/AlnisM/autoheuristic-datasets/raw/main/'  # @lint-ignore
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
a100_data='mixedmm_a100_data.zip'
h100_data='mixedmm_h100_data.zip'
datasets=("${a100_data}" "${h100_data}")
for dataset in "${datasets[@]}"; do
    rm -f ${dataset}
    url="${base_url}${dataset}"
    wget ${url}
    unzip -o ${dataset}
    rm ${dataset}
done
