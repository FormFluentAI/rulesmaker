#!/usr/bin/env python3
"""Deep diagnostic for Bedrock client and credentials.
"""
import os
import json
import boto3
from botocore.exceptions import ClientError

model_id = os.environ.get('BEDROCK_MODEL_ID')
region = os.environ.get('AWS_REGION', 'eu-central-1')
print('model_id:', model_id)
print('region:', region)

s = boto3.Session()
creds = s.get_credentials()
print('boto3 session profile:', s.profile_name)
if creds:
    frozen = creds.get_frozen_credentials()
    print('access_key:', getattr(frozen,'access_key',None))
    print('token_present:', bool(getattr(frozen,'token',None)))
else:
    print('no creds from boto3')

print('\n== STS via boto3 ==')
try:
    sts = s.client('sts')
    print(json.dumps(sts.get_caller_identity(), indent=2, default=str))
except Exception as e:
    print('sts error:', repr(e))

print('\n== Bedrock client meta ==')
client = s.client('bedrock-runtime', region_name=region)
try:
    print('endpoint_url:', client.meta.endpoint_url)
except Exception as e:
    print('endpoint inspect error:', e)

print('\n== Attempt tiny converse ==')
try:
    resp = client.converse(modelId=model_id, messages=[{'role':'user','content':[{'text':'hello'}]}], inferenceConfig={'maxTokens':10,'temperature':0.0})
    print('converse ok:', json.dumps(resp)[:1000])
except ClientError as e:
    # print full botocore exception dict if available
    err = getattr(e, 'response', None)
    print('ClientError response:')
    print(json.dumps(err, indent=2, default=str))
except Exception as e:
    print('other error:', repr(e))
