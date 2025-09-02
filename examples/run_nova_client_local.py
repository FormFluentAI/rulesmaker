"""Local runner to call examples.nova_client.run_once from the repository root.

Usage:
  BEDROCK_MODEL_ID="..." AWS_REGION=eu-central-1 \
  conda run -n rulescraper --no-capture-output python examples/run_nova_client_local.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.nova_client import run_once

prompt = (
    "Provide a step-by-step integration guide (with code snippets) showing how to call "
    "Nova Lite from this repository to validate and apply parsed rules. Include error "
    "handling and a suggestion for a retry/backoff strategy."
)

import sys
try:
  print(run_once(prompt, temperature=0.7, max_tokens=1024))
except Exception as e:
  # Print stack to stderr for diagnosis but keep stdout clean
  import traceback
  traceback.print_exc(file=sys.stderr)
  print(f"ERROR: {e}", file=sys.stderr)
  raise
