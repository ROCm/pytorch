#!/bin/bash -x

# Please set these environment variables
if [[ -z "${GITHUB_TOKEN}" ]] || [[ -z "${AWS_ACCESS_KEY_ID}" ]] || [[ -z "${AWS_SECRET_ACCESS_KEY}" ]]; then
  echo "ERROR: Please set these environment variables
        GITHUB_TOKEN
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
       "
  exit 1
fi

export GITHUB_TOKEN=${GITHUB_TOKEN}
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

if [[ -z $1 ]]; then
  echo "ERROR: Please provide pytorch commit SHA as first argument"
  echo "Usage: ./parity.sh <SHA> [OUT_FOLDER]"
  echo "  SHA        - PyTorch commit SHA"
  echo "  OUT_FOLDER - (optional) Output folder from download_testlogs. If not provided, auto-detects folder matching SHA."
  exit 1
fi

SHA=$1

# TZ='America/Los_Angeles' date
DATE=`TZ='America/Los_Angeles' date '+%Y%m%d'`

# Output folder - use second argument if provided, otherwise auto-detect
if [[ -n $2 ]]; then
  OUT_FOLDER=$2
else
  # Auto-detect folder matching the SHA
  OUT_FOLDER=$(ls -d *_${SHA} 2>/dev/null | head -1)
  if [[ -z $OUT_FOLDER ]]; then
    echo "ERROR: Could not find folder matching SHA ${SHA}"
    echo "       Please provide the output folder as second argument: ./parity.sh <SHA> <OUT_FOLDER>"
    exit 1
  fi
fi

echo "Using output folder: ${OUT_FOLDER}"

# Previous week's all tests status CSV - provides skip_reason, assignee, comments AND existed_last_week
PREV_WEEK_CSV=${DATE}_input_1.csv

if [[ ! -e $PREV_WEEK_CSV ]]; then
  echo "WARNING: Previous week CSV ${PREV_WEEK_CSV} doesn't exist"
  echo "         skip_reason, assignee, comments, and existed_last_week columns will be empty"
  PREV_WEEK_ARG=""
else
  echo "Using previous week's CSV: ${PREV_WEEK_CSV}"
  PREV_WEEK_ARG="--prev_week_csv $PREV_WEEK_CSV"
fi

echo "Processing CI results for https://github.com/pytorch/pytorch/commit/$SHA"

#******************* ROCm jobs *****************************
# Default (Single GPU) all 6 shards - rocm / linux-focal-rocm6.1-py3.8 / test (default, 1, 6, linux.rocm.gpu.2)
# Distributed (mGPU) all 2 shards - periodic / linux-focal-rocm6.1-py3.8 / test (distributed, 1, 2, linux.rocm.gpu)
# Inductor all 2 shards - inductor / rocm6.1-py3.8-inductor / test (inductor, 1, 2, linux.rocm.gpu.2)

#******************* CUDA jobs *****************************
# Default (Single GPU) all 5 shards - pull / linux-focal-cuda12.1-py3.10-gcc9-sm86 / test (default, 1, 5, linux.g5.4xlarge.nvidia.gpu)
# Distributed (mGPU) all 3 shards - pull / linux-focal-cuda11.8-py3.10-gcc9 / test (distributed, 1, 3, linux.8xlarge.nvidia.gpu)
# Inductor all 2 shards - inductor / cuda12.1-py3.10-gcc9-sm86 / test (inductor, 1, 2, linux.g5.4xlarge.nvidia.gpu)


# python3 -u download_testlogs --max_pages 50 --sha1 $SHA  --ignore_status 2>&1 |tee download_testlogs_${DATE}.log && exit 1
# python3 -u download_testlogs --max_pages 50 --sha1 $SHA --exclude_distributed --ignore_status 2>&1 |tee download_testlogs_${DATE}.log && exit 1

# Generate all tests status CSV (use this as input for next week)
python3 -u summarize_xml_testreports.py --set1 $OUT_FOLDER/rocm_xml_dir --set2 $OUT_FOLDER/cuda_xml_dir \
	$PREV_WEEK_ARG \
	--output_csv $OUT_FOLDER/${DATE}_all_tests_status.csv 2>&1 | tee $OUT_FOLDER/xml_processing_${DATE}_all.log

mv file_running_time_output.csv $OUT_FOLDER/${DATE}_running_time.csv

