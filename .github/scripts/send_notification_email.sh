#!/usr/bin/env bash

set -euo pipefail

required_vars=(
  ROCM_SMTP_URL
  ROCM_SMTP_USERNAME
  ROCM_SMTP_PASSWORD
  ROCM_EMAIL_FROM
  ROCM_EMAIL_TO
  EMAIL_SUBJECT
  EMAIL_BODY
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: ${var_name}" >&2
    exit 1
  fi
done

message_file="$(mktemp)"
trap 'rm -f "$message_file"' EXIT

{
  printf 'From: %s\n' "$ROCM_EMAIL_FROM"
  printf 'To: %s\n' "$ROCM_EMAIL_TO"
  printf 'Subject: %s\n' "$EMAIL_SUBJECT"
  printf 'MIME-Version: 1.0\n'
  printf 'Content-Type: text/plain; charset=UTF-8\n'
  printf '\n'
  printf '%s\n' "$EMAIL_BODY"
} > "$message_file"

IFS=',' read -r -a recipients <<< "$ROCM_EMAIL_TO"
curl_args=()
for recipient in "${recipients[@]}"; do
  recipient="${recipient#"${recipient%%[![:space:]]*}"}"
  recipient="${recipient%"${recipient##*[![:space:]]}"}"
  if [[ -n "$recipient" ]]; then
    curl_args+=(--mail-rcpt "$recipient")
  fi
done

if [[ "${#curl_args[@]}" -eq 0 ]]; then
  echo "ROCM_EMAIL_TO did not contain any recipients" >&2
  exit 1
fi

curl --silent --show-error --fail --ssl-reqd \
  --url "$ROCM_SMTP_URL" \
  --user "${ROCM_SMTP_USERNAME}:${ROCM_SMTP_PASSWORD}" \
  --mail-from "$ROCM_EMAIL_FROM" \
  "${curl_args[@]}" \
  --upload-file "$message_file"

echo "Notification email sent to $ROCM_EMAIL_TO"
