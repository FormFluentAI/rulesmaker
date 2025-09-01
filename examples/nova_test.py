"""Simple Nova Lite Bedrock test script for rules-maker

Usage:
  - Copy `.env.example` to `.env` and fill AWS credentials and BEDROCK_MODEL_ID.
  - Activate your conda env (rulescraper) and install dependencies: `pip install -r requirements.txt`.
  - To run a dry-run (no API call): `python examples/nova_test.py`
  - To actually call Bedrock set env var `RUN_BEDROCK_TEST=1` and run the script.

This script is intentionally conservative to avoid accidental API calls.
"""
import os
import json

# Attempt to load environment variables from a local .env file if python-dotenv is available.
try:
    from dotenv import load_dotenv, find_dotenv

    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    else:
        # call load_dotenv() to load default .env if present in CWD
        load_dotenv()
        print("No explicit .env file found via find_dotenv(); attempted default load")
except Exception:
    # Not fatal — continue without .env loaded
    print("python-dotenv not available; continuing without loading .env")

RUN = os.environ.get("RUN_BEDROCK_TEST", "0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")

message = "Hello from rules-maker. Please respond briefly with 'ok' and the current model id.'"

print("Prepared Bedrock request:\n", json.dumps({"modelId": MODEL_ID, "messages": [{"role": "user", "content": [{"text": message}]}]}, indent=2))

if RUN != "1":
    print("\nDry run only. To make a real Bedrock API call set RUN_BEDROCK_TEST=1 in your environment.\n")
else:
    try:
        import boto3
        from botocore.config import Config

        # optional config for longer responses
        config = Config(read_timeout=900, retries={"max_attempts": 3})
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=config)

        print("Calling Bedrock (model=%s, region=%s)" % (MODEL_ID, AWS_REGION))
        response = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": message}]}],
            inferenceConfig={"maxTokens": 256, "temperature": 0.2}
        )

        out = response.get("output", {}).get("message", {}).get("content", [])
        if out:
            text = out[0].get("text")
            print("\nBedrock response:\n", text)
        else:
            print("No text content found in response:\n", response)

    except Exception as e:
        print("Bedrock call failed:", str(e))

