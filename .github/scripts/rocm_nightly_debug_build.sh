#!/usr/bin/env bash

set -euxo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-/debug-artifacts}"
WORKDIR=/tmp/pytorch
PATCH_SHA=519160d466782f5a62365be051fcb3ef90fa0b00
LOG_HELPER="${LOG_HELPER:-/workspace/rocm-nightly-workflow/.github/scripts/run_with_log_heartbeat.sh}"
PYTORCH_SOURCE_SHA="${PYTORCH_SOURCE_SHA:-8a6524408a49ab2293f694b43131d0fc17e45a32}"
TARGET_NINJA="${TARGET_NINJA:-auto}"

detect_failed_target() {
  local log_file=$1
  local failed_line
  local target
  local -a outputs

  failed_line=$(grep -E '^FAILED: ' "$log_file" | tail -n 1 || true)
  if [[ -z "$failed_line" ]]; then
    return 1
  fi

  failed_line=${failed_line#FAILED: }
  read -r -a outputs <<< "$failed_line"
  if [[ ${#outputs[@]} -eq 0 ]]; then
    return 1
  fi

  for target in "${outputs[@]}"; do
    if [[ $target == "$WORKDIR/build/"* ]]; then
      printf '%s\n' "${target#"$WORKDIR/build/"}"
      return 0
    fi
    if [[ $target != /* ]]; then
      printf '%s\n' "$target"
      return 0
    fi
  done

  printf '%s\n' "${outputs[0]}"
}

mkdir -p "$ARTIFACT_DIR"
if ! touch "$ARTIFACT_DIR/.write-test" 2>/dev/null; then
  echo "Artifact directory '$ARTIFACT_DIR' is not writable by uid $(id -u)." >&2
  exit 1
fi
rm -f "$ARTIFACT_DIR/.write-test"
rm -rf "$WORKDIR"

git clone https://github.com/pytorch/pytorch --recursive "$WORKDIR"
cd "$WORKDIR"
git checkout "$PYTORCH_SOURCE_SHA"
git submodule sync --recursive
git submodule update --init --recursive

pip install -r requirements.txt
git config --local user.name "AMD AMD"
git config --local user.email "amd@amd.com"
git remote add rocm https://github.com/ROCm/pytorch.git
git fetch rocm
git cherry-pick "$PATCH_SHA"

if bash "$LOG_HELPER" "$ARTIFACT_DIR/build.log" -- .ci/pytorch/build.sh; then
  if [[ -f build/.ninja_log ]]; then
    cp build/.ninja_log "$ARTIFACT_DIR"/
  fi
  exit 0
fi

if [[ -f build/.ninja_log ]]; then
  cp build/.ninja_log "$ARTIFACT_DIR"/
fi

if [[ ! -d build ]]; then
  echo "Expected build directory 'build' was not found after the failed build." | tee -a "$ARTIFACT_DIR/build.log"
  exit 1
fi

rerun_target=$TARGET_NINJA
if [[ $rerun_target == auto ]]; then
  rerun_target=$(detect_failed_target "$ARTIFACT_DIR/build.log" || true)
fi

if [[ -z "$rerun_target" ]]; then
  echo "Unable to determine the failed Ninja target from build.log. Set TARGET_NINJA to override auto detection." | tee -a "$ARTIFACT_DIR/build.log"
  exit 1
fi

target_log_name="${rerun_target//[^A-Za-z0-9_.-]/_}.log"
echo "PyTorch build failed at source SHA ${PYTORCH_SOURCE_SHA}. Re-running detected target ${rerun_target} with serial verbose Ninja output." | tee -a "$ARTIFACT_DIR/build.log"

ninja -C build -t clean "$rerun_target" || true

if ! bash "$LOG_HELPER" "$ARTIFACT_DIR/$target_log_name" -- \
  ninja -C build -j1 -v "$rerun_target"; then
  {
    echo "Focused rerun of ${rerun_target} failed. Last 200 lines from ${target_log_name}:"
    tail -n 200 "$ARTIFACT_DIR/$target_log_name" || true
  } | tee -a "$ARTIFACT_DIR/build.log"
fi

exit 1
