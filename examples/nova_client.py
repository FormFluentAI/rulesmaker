"""Small helper to call Bedrock Nova Lite and return assistant text.

Usage:
    from examples.nova_client import run_once
    text = run_once("Summarize ...", temperature=0.2, max_tokens=200)
"""
import os
import json
import time


def _get_client(region=None):
    # Only load a local .env file if no AWS credentials are already present in the environment.
    # This avoids accidentally overriding working CLI/credential-file credentials with stale values
    # from a repository .env file.
    try:
        aws_keys = (os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY"), os.environ.get("AWS_SESSION_TOKEN"))
        if not any(aws_keys):
            try:
                from dotenv import load_dotenv, find_dotenv
                env_file = find_dotenv()
                if env_file:
                    load_dotenv(env_file)
                else:
                    load_dotenv()
            except Exception:
                # python-dotenv not installed or failed to load; proceed with existing env/credential chain
                pass
    except Exception:
        # be defensive; if anything goes wrong keep going
        pass

    import boto3
    region = region or os.environ.get("AWS_REGION", "eu-central-1")
    return boto3.client("bedrock-runtime", region_name=region)


def _extract_text_from_response(resp):
    out = resp.get("output", {}).get("message", {}).get("content", [])
    if out and isinstance(out, list):
        # join text entries if multiple
        texts = [c.get("text", "") for c in out if isinstance(c, dict)]
        return "\n".join(texts).strip()
    return ""


def run_once(prompt: str, temperature: float = 0.2, max_tokens: int = 256, model_id: str = None, region: str = None):
    """Send a single message to the configured Bedrock model and return assistant text.

    Returns the assistant text (string) or raises exceptions from boto3.
    """
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
    if not model_id:
        raise ValueError("BEDROCK_MODEL_ID must be set in environment or passed to run_once")

    client = _get_client(region=region)

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    start = time.time()
    resp = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    duration = time.time() - start

    text = _extract_text_from_response(resp)
    # attach a small metadata header
    metadata = {
        "modelId": model_id,
        "region": region or os.environ.get("AWS_REGION", "eu-central-1"),
        "duration_seconds": duration,
    }
    return f"--- METADATA: {json.dumps(metadata)} ---\n\n{text}"
 
