#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  local version
  commit=$(get_pinned_commit huggingface)
  pip_install "git+https://github.com/huggingface/transformers@${commit}"
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)

  pip_install "git+https://github.com/huggingface/pytorch-image-models@${commit}"
  # Clean up
<<<<<<< HEAD
  conda_run pip uninstall -y cmake torch torchvision triton
=======
  conda_run pip uninstall -y torch torchvision triton
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

# Pango is needed for weasyprint which is needed for doctr
conda_install pango
install_huggingface
install_timm
