#!/usr/bin/env bash
# Obtain temporary AWS STS credentials and print export commands for shell.
# Usage: source scripts/get_aws_session.sh

set -euo pipefail

DURATION=${1:-3600}

echo "Requesting STS session token for ${DURATION}s..."
resp=$(aws sts get-session-token --duration-seconds "$DURATION" --output json) || {
  echo "Failed to get-session-token. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set or AWS CLI configured." >&2
  return 1
}

access_key_id=$(echo "$resp" | jq -r '.Credentials.AccessKeyId')
secret_access_key=$(echo "$resp" | jq -r '.Credentials.SecretAccessKey')
session_token=$(echo "$resp" | jq -r '.Credentials.SessionToken')
expiration=$(echo "$resp" | jq -r '.Credentials.Expiration')

cat <<EOF
export AWS_ACCESS_KEY_ID="$access_key_id"
export AWS_SECRET_ACCESS_KEY="$secret_access_key"
export AWS_SESSION_TOKEN="$session_token"
# Session expires at: $expiration
EOF

echo "Copy/paste the above lines or run: eval \$(scripts/get_aws_session.sh) to set them in your shell."
