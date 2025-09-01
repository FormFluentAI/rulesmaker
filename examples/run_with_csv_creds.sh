#!/usr/bin/env bash
set -euo pipefail
# Safely read AWS Access Key ID and Secret Access Key from a CSV placed at
# repo root: bedrock-long-term-api-key.csv
# Expected CSV headers (case-insensitive): something like
# "User name,Access key ID,Secret access key"

cd "$(dirname "$0")/.."

# parse CSV via Python to avoid shell quoting pitfalls; print AK and SK on two lines
read AK SK <<'PY'
$(python - <<'PYCODE'
import csv
import sys
p='bedrock-long-term-api-key.csv'
with open(p, newline='') as f:
    r = csv.reader(f)
    headers = next(r)
    row = next(r)

ak = None
sk = None
for i,h in enumerate(headers):
    lh = h.strip().lower()
    if 'access key' in lh and 'id' in lh:
        ak = row[i].strip()
    if 'secret' in lh:
        sk = row[i].strip()
# fallback to columns 2 and 3 if headers didn't match
if not ak:
    ak = row[1].strip()
if not sk:
    sk = row[2].strip()
print(ak)
print(sk)
PYCODE
)
PY

# Export into environment for the child process (do NOT echo these values)
export AWS_ACCESS_KEY_ID="$AK"
export AWS_SECRET_ACCESS_KEY="$SK"
export AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN:-}"
export BEDROCK_MODEL_ID="arn:aws:bedrock:eu-central-1:205930650303:inference-profile/eu.amazon.nova-lite-v1:0"
export AWS_REGION="eu-central-1"

# Run the runner (uses examples/run_nova_client_local.py which handles sys.path)
python examples/run_nova_client_local.py > test2.txt 2> test2.err || true

# Show diagnostics (not printing secrets)
echo "=== test2.err (first 200 lines) ==="
sed -n '1,200p' test2.err || true

echo "=== test2.txt (first 200 lines) ==="
sed -n '1,200p' test2.txt || true

# done
exit 0
