#!/usr/bin/env bash

set -euxo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-/debug-artifacts}"
WORKDIR=/tmp/pytorch
PATCH_SHA=519160d466782f5a62365be051fcb3ef90fa0b00
LOG_HELPER="${LOG_HELPER:-/workspace/rocm-nightly-workflow/.github/scripts/run_with_log_heartbeat.sh}"

mkdir -p "$ARTIFACT_DIR"
rm -rf "$WORKDIR"

git clone https://github.com/pytorch/pytorch --recursive "$WORKDIR"
cd "$WORKDIR"

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

echo "PyTorch build failed. Re-running gloo_hip wrappers with verbose output." | tee -a "$ARTIFACT_DIR/build.log"

GLOO_DIR=build/third_party/gloo/gloo/CMakeFiles/gloo_hip.dir
if [[ ! -d "$GLOO_DIR" ]]; then
  echo "Expected gloo_hip build directory '$GLOO_DIR' was not found." | tee -a "$ARTIFACT_DIR/gloo-debug.log"
  exit 1
fi

ninja -C build -t clean gloo_hip || true

find "$GLOO_DIR" -name 'gloo_hip_generated_*.cmake' | sort > "$ARTIFACT_DIR/gloo_wrappers.txt"
if [[ ! -s "$ARTIFACT_DIR/gloo_wrappers.txt" ]]; then
  echo "No gloo_hip wrapper scripts were found." | tee -a "$ARTIFACT_DIR/gloo-debug.log"
  exit 1
fi

status=0
wrapper_index=0
: > "$ARTIFACT_DIR/gloo_wrapper_logs.txt"
while IFS= read -r wrapper; do
  wrapper_index=$((wrapper_index + 1))
  generated_file="${wrapper%.cmake}"
  wrapper_log="$ARTIFACT_DIR/gloo-wrapper-$(printf '%03d' "$wrapper_index").log"
  {
    echo
    echo "===== Re-running $wrapper ====="
  } | tee -a "$ARTIFACT_DIR/gloo-debug.log"
  printf '%s\t%s\n' "$wrapper" "$(basename "$wrapper_log")" >> "$ARTIFACT_DIR/gloo_wrapper_logs.txt"

  if ! bash "$LOG_HELPER" "$wrapper_log" -- \
    cmake \
      -D verbose:BOOL=ON \
      -D build_configuration:STRING=RELEASE \
      -D generated_file:STRING="$generated_file" \
      -P "$wrapper"; then
    {
      echo "Wrapper failed. Last 200 lines from $(basename "$wrapper_log"):"
      tail -n 200 "$wrapper_log" || true
    } | tee -a "$ARTIFACT_DIR/gloo-debug.log"
    status=1
    break
  fi
done < "$ARTIFACT_DIR/gloo_wrappers.txt"

exit "$status"
