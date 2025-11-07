#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Error: This script requires exactly one argument."
    echo "`bash generate_heuristic_mixedmm.sh collect` to run benchmark and collect training data."
    echo "`bash generate_heuristic_mixedmm.sh generate` to use the collected data to learn a heuristic."
    exit 1
fi

MODE=$1

# !!! SPECIFY THE GPUs THAT YOU WANT TO USE HERE !!!
GPU_DEVICE_IDS="4,5"

<<<<<<< HEAD
# !!! SPECIFY THE CONDA ENVIRONEMNT THAT YOU WANT TO BE ACTIVATED HERE !!!
=======
# !!! SPECIFY THE CONDA ENVIRONMENT THAT YOU WANT TO BE ACTIVATED HERE !!!
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
CONDA_ENV=heuristic-pr

NUM_SAMPLES=2000

# This is where AutoHeuristic will store autotuning results
OUTPUT_DIR="a100"

# !!! CHANGE THE NAME OF THE HEURISTIC IF YOU WANT TO LEARN A HEURISTIC FOR A GPU THAT IS NOT A100 !!!
HEURISTIC_NAME="MixedMMA100"

BENCHMARK_SCRIPT="gen_data_mixed_mm.py"

TRAIN_SCRIPT="train_decision_mixedmm.py"

bash ../generate_heuristic.sh ${MODE} ${GPU_DEVICE_IDS} ${CONDA_ENV} ${NUM_SAMPLES} ${OUTPUT_DIR} ${HEURISTIC_NAME} ${BENCHMARK_SCRIPT} ${TRAIN_SCRIPT}
