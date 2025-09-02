#!/usr/bin/env bash
# Save temporary AWS STS session credentials to a local .env file safely.
# Usage:
#   scripts/save_session_to_env.sh        # interactively asks before overwrite
#   scripts/save_session_to_env.sh -f     # force overwrite without prompt
#
# This script calls scripts/get_aws_session.sh to obtain temp credentials,
# writes them to .env.local (not committed), makes a timestamped backup if one exists,
# and restricts file permissions to 600.

set -euo pipefail

FORCE=0
DURATION=3600
OUTFILE=".env.local"

while [[ ${1:-} != "" ]]; do
  case "$1" in
    -f|--force)
      FORCE=1; shift;;
    -d|--duration)
      DURATION=${2:-3600}; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found in PATH. Install and configure AWS CLI first." >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found. Install jq (apt, yum, or conda) and retry." >&2
  exit 2
fi

if [[ -f "$OUTFILE" && $FORCE -ne 1 ]]; then
  read -r -p "$OUTFILE already exists — backup and overwrite? [y/N] " resp
  case "$resp" in
    [yY][eE][sS]|[yY]) ;;
    *) echo "Aborted by user"; exit 0;;
  esac
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "Requesting STS session token for ${DURATION}s (this will not print secrets)..."
if ! scripts/get_aws_session.sh "$DURATION" > "$TMP"; then
  echo "Failed to obtain STS session token" >&2
  exit 1
fi

# Source the temp file in a subshell to capture variables without polluting current shell
AWS_ACCESS_KEY_ID_VAL=$(bash -c "source '$TMP' >/dev/null 2>&1; printf '%s' \"\$AWS_ACCESS_KEY_ID\"")
AWS_SECRET_ACCESS_KEY_VAL=$(bash -c "source '$TMP' >/dev/null 2>&1; printf '%s' \"\$AWS_SECRET_ACCESS_KEY\"")
AWS_SESSION_TOKEN_VAL=$(bash -c "source '$TMP' >/dev/null 2>&1; printf '%s' \"\$AWS_SESSION_TOKEN\"")

if [[ -z "$AWS_ACCESS_KEY_ID_VAL" || -z "$AWS_SECRET_ACCESS_KEY_VAL" || -z "$AWS_SESSION_TOKEN_VAL" ]]; then
  echo "Did not find expected AWS credentials in session output; aborting." >&2
  exit 1
fi

# backup existing outfile if present
if [[ -f "$OUTFILE" ]]; then
  bak="${OUTFILE}.$(date -u +%Y%m%dT%H%M%SZ).bak"
  echo "Backing up existing $OUTFILE -> $bak"
  cp -p "$OUTFILE" "$bak"
fi

cat > "$OUTFILE" <<EOF
# Local temporary AWS session credentials (auto-generated). Do NOT commit.
AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID_VAL"
AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY_VAL"
AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN_VAL"
AWS_REGION="${AWS_REGION:-${AWS_REGION:-eu-central-1}}"
# Optional: set your chosen Bedrock model id here
# BEDROCK_MODEL_ID="amazon.titan-text-express-v1:0:8k"
EOF

chmod 600 "$OUTFILE"
echo "Wrote temporary credentials to $OUTFILE (permissions set to 600)."
echo "Do NOT commit this file. To load credentials in your shell:"
echo "  source $OUTFILE"

exit 0
