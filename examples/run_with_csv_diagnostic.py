#!/usr/bin/env python3
"""Diagnostic: read AWS credentials from CSV and test STS identity and Bedrock access.
Does NOT print secret values.
"""
import csv
import os
import json
from botocore.exceptions import ClientError

CSV_PATH = 'bedrock-long-term-api-key.csv'
if not os.path.exists(CSV_PATH):
    raise SystemExit(f"CSV not found: {CSV_PATH}")

with open(CSV_PATH, newline='') as f:
    r = csv.reader(f)
    headers = next(r)
    row = next(r)

# find columns
ak = None
sk = None
user = None
for i,h in enumerate(headers):
    lh = h.strip().lower()
    if 'access key' in lh and 'id' in lh:
        ak = row[i].strip()
    if 'secret' in lh:
        sk = row[i].strip()
    if 'user' in lh or 'user name' in lh:
        user = row[i].strip()
if not ak:
    ak = row[1].strip()
if not sk:
    sk = row[2].strip()

print('CSV headers:', headers)
print('csv user field (if any):', user)
print('Using credentials from CSV to create a boto3 Session (secrets not printed)')

import boto3

sess = boto3.Session(aws_access_key_id=ak, aws_secret_access_key=sk)

# STS identity
try:
    sts = sess.client('sts')
    identity = sts.get_caller_identity()
    print('\n== STS identity from CSV creds ==')
    print(json.dumps({k: identity.get(k) for k in ('UserId','Account','Arn')}, indent=2))
except ClientError as e:
    print('STS ClientError:')
    print(json.dumps(e.response, indent=2, default=str))
    raise SystemExit(1)

# Try a tiny Bedrock converse
model = os.environ.get('BEDROCK_MODEL_ID','arn:aws:bedrock:eu-central-1:205930650303:inference-profile/eu.amazon.nova-lite-v1:0')
region = os.environ.get('AWS_REGION','eu-central-1')
print(f"\nAttempting bedrock-runtime.converse to model {model} in region {region}")
client = sess.client('bedrock-runtime', region_name=region)
try:
    resp = client.converse(modelId=model, messages=[{'role':'user','content':[{'text':'hello'}]}], inferenceConfig={'maxTokens':10,'temperature':0.0})
    print('\nConverse OK; output summary:')
    # show only summarized output keys
    out = resp.get('output', {}).get('message', {}).get('content', [])
    print('content_count:', len(out))
    if out:
        first = out[0]
        print('first_content_keys:', list(first.keys()))
        if isinstance(first, dict) and 'text' in first:
            print('first_text_len:', len(first.get('text','')))
except ClientError as e:
    print('\nBedrock ClientError:')
    print(json.dumps(e.response, indent=2, default=str))
    raise SystemExit(1)

print('\nDone')
