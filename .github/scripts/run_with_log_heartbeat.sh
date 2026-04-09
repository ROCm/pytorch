#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 LOG_FILE -- COMMAND [ARGS...]" >&2
  exit 2
}

if [[ $# -lt 3 ]]; then
  usage
fi

log_file=$1
shift

if [[ $1 != "--" ]]; then
  usage
fi
shift

heartbeat_seconds="${HEARTBEAT_SECONDS:-300}"
tail_lines="${TAIL_LINES:-200}"
check_interval=5

mkdir -p "$(dirname "$log_file")"
: >"$log_file"

"$@" >"$log_file" 2>&1 &
cmd_pid=$!

cleanup() {
  if kill -0 "$cmd_pid" 2>/dev/null; then
    kill "$cmd_pid" 2>/dev/null || true
    wait "$cmd_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

command_str=$(printf '%q ' "$@")
command_str=${command_str% }

next_heartbeat=0
while kill -0 "$cmd_pid" 2>/dev/null; do
  now=$(date +%s)
  if (( now >= next_heartbeat )); then
    echo "[$(date -u +%FT%TZ)] Command still running: ${command_str}"
    echo "[$(date -u +%FT%TZ)] Log file: ${log_file} ($(du -h "$log_file" | cut -f1))"
    next_heartbeat=$((now + heartbeat_seconds))
  fi
  sleep "$check_interval"
done

if wait "$cmd_pid"; then
  status=0
else
  status=$?
fi

trap - EXIT

if [[ $status -eq 0 ]]; then
  echo "Command completed successfully. Full log saved to ${log_file}"
  exit 0
fi

echo "Command failed with exit code ${status}. Last ${tail_lines} lines from ${log_file}:"
tail -n "$tail_lines" "$log_file" || true
exit "$status"
