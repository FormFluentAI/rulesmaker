#!/usr/bin/env bash
# List Bedrock foundation models or available models in account/region.
# Usage: scripts/list_bedrock_models.sh

set -euo pipefail

REGION=${1:-${AWS_REGION:-us-east-1}}

echo "Listing Bedrock models in region: $REGION"

# Prefer the newer bedrock commands if available
if aws bedrock list-foundation-models --region "$REGION" --output json >/dev/null 2>&1; then
  aws bedrock list-foundation-models --region "$REGION" --output json | jq -r '.models[] | [.modelId, .modelType] | @tsv' || true
else
  # Fallbacks for possible CLI variants
  if aws bedrock list-models --region "$REGION" --output json >/dev/null 2>&1; then
    aws bedrock list-models --region "$REGION" --output json | jq -r '.modelSummaries[] | [.modelId, .modelType] | @tsv' || true
  else
    echo "Could not find bedrock list-models command. Ensure AWS CLI v2 with Bedrock is installed or consult AWS console." >&2
    exit 2
  fi
fi
