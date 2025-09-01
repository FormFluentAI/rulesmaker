#!/usr/bin/env python3
"""Diagnostic: print masked AWS env presence, AWS CLI identity, and boto3 session info.
This script intentionally does NOT print secret values.
"""
import os
import subprocess
import json
import boto3

print("== Environment presence (masked) ==")
keys = [k for k in os.environ if k.startswith("AWS") or k in ("BEDROCK_MODEL_ID", "AWS_REGION")]
for k in sorted(keys):
    v = os.environ.get(k)
    if not v:
        print(f"{k}: <not set>")
    else:
        print(f"{k}: <set> (len={len(v)})")

print("\n== AWS CLI identity ==")
try:
    out = subprocess.check_output(["aws", "sts", "get-caller-identity", "--output", "json"], stderr=subprocess.STDOUT)
    data = json.loads(out)
    print(json.dumps({"Account": data.get("Account"), "Arn": data.get("Arn")}, indent=2))
except subprocess.CalledProcessError as e:
    print("aws sts failed:")
    try:
        print(e.output.decode())
    except Exception:
        print(str(e))
except FileNotFoundError:
    print("aws CLI not found in PATH")

print("\n== boto3 session info ==")
s = boto3.Session()
print("profile_name:", s.profile_name)
creds = s.get_credentials()
print("has_creds:", bool(creds))
try:
    frozen = creds.get_frozen_credentials() if creds else None
    print("access_key_present:", bool(frozen.access_key) if frozen else False)
    print("token_present:", bool(frozen.token) if frozen else False)
except Exception as e:
    print("creds inspect error:", e)

print("default region name:", s.region_name)
print("boto3 version:", boto3.__version__)
try:
    import botocore
    print("botocore version:", botocore.__version__)
except Exception:
    pass
