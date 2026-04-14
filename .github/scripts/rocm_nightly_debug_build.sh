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

# Capture the real error context from the original build.log. The main build
# runs with high parallelism, so the `FAILED:` line is typically buried before
# hundreds of lines of unrelated warnings from siblings that were compiling
# concurrently. Dump the window around it so the error is actually visible.
{
  echo "=== Error context around FAILED: line in build.log ==="
  awk '
    { buf[NR]=$0 }
    /^FAILED: / && !printing {
      start = NR-80; if (start<1) start=1
      for (i=start; i<NR; i++) if (i in buf) print buf[i]
      printing=1; lines=0
    }
    printing { print; lines++; if (lines>=120) exit }
  ' "$ARTIFACT_DIR/build.log" || true
  echo "=== End error context ==="
} | tee -a "$ARTIFACT_DIR/build.log"

echo "PyTorch build failed at source SHA ${PYTORCH_SOURCE_SHA}. Re-running detected target ${rerun_target} with serial verbose Ninja output." | tee -a "$ARTIFACT_DIR/build.log"

# Do NOT `ninja -t clean <target>` here: that is transitive and wipes every
# dependency of the target (often ~all of libtorch), forcing a multi-hour
# cold rebuild at -j1. The failing target's output does not exist because
# the build failed, so ninja will naturally re-run only the failing command.

# The .ci build epilogue stops sccache; restart it so the rerun can still
# hit whatever objects were cached during the main build.
if command -v sccache >/dev/null 2>&1; then
  sccache --start-server || true
fi

if ! bash "$LOG_HELPER" "$ARTIFACT_DIR/$target_log_name" -- \
  ninja -C build -j1 -v "$rerun_target"; then
  {
    echo "Focused rerun of ${rerun_target} failed. Last 200 lines from ${target_log_name}:"
    tail -n 200 "$ARTIFACT_DIR/$target_log_name" || true
  } | tee -a "$ARTIFACT_DIR/build.log"
fi

exit 1
