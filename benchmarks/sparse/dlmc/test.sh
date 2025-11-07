#!/bin/bash

DATASET_ROOT_DIR=$HOME/datasets/

<<<<<<< HEAD
# wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz -P $DATASET_ROOT_DIR
=======
# wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz -P $DATASET_ROOT_DIR  # @lint-ignore
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
# tar -xvf $DATASET_ROOT_DIR/dlmc.tar.gz

echo "!! SPARSE SPMS TIME BENCHMARK!! "

# cpu
python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@sparse
python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@sparse --backward-test

python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@dense
python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@dense --backward-test

python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@vector


# cuda
python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@sparse --with-cuda
python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@sparse --with-cuda --backward-test

python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@dense --with-cuda
python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@dense --with-cuda --backward-test

python -m dlmc.matmul_bench --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset magnitude_pruning --operation sparse@vector --with-cuda
