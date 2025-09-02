"""
Phase 2 tests: pricing/alias, retry/backoff metrics, and credentials parsing/validation.

These tests avoid real network calls by stubbing boto3 Session/client.
"""

import asyncio
import base64
import json
from pathlib import Path

import pytest


def _mk_client_error(code: str = "ThrottlingException", status: int = 429):
    from botocore.exceptions import ClientError
    return ClientError(
        error_response={
            "Error": {"Code": code, "Message": f"Simulated {code}"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation_name="Converse",
    )


@pytest.mark.asyncio
async def test_pricing_and_alias_and_retry_metrics(monkeypatch):
    # Stub heavy optional dependency before importing extractor package tree
    import sys, types
    sys.modules.setdefault('sentence_transformers', types.SimpleNamespace(SentenceTransformer=object))

    from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider

    # Prepare a fake boto3 client to trigger throttling twice then succeed
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def converse(self, modelId, messages, inferenceConfig):
            self.calls += 1
            if self.calls < 3:
                raise _mk_client_error("ThrottlingException", 429)
            # Success response
            return {
                "output": {
                    "message": {
                        "content": [{"text": json.dumps({"ok": True})}]
                    }
                },
                "usage": {"inputTokens": 100, "outputTokens": 50},
            }

    class FakeSession:
        def client(self, name, region_name=None):
            assert name == "bedrock-runtime"
            return FakeClient()

    class FakeBoto3:
        Session = lambda self=None, **kwargs: FakeSession()  # noqa: E731

    # Monkeypatch boto3 and asyncio.sleep to avoid real waits
    monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("BEDROCK_RETRY_BASE_MS", "1")
    monkeypatch.setenv("BEDROCK_RETRY_MAX_MS", "2")

    sys.modules['boto3'] = FakeBoto3()

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    extractor = LLMContentExtractor(
        llm_config=LLMConfig(
            provider=LLMProvider.BEDROCK,
            model_name="amazon.nova-lite-v1:0",
            region="us-east-1",
        )
    )

    # Call Bedrock request and ensure it succeeds after retries
    result = await extractor._make_llm_request("hi", "sys")
    assert isinstance(result, dict) and result.get("ok") is True

    stats = extractor.get_usage_stats()
    # Usage recorded and cost estimated (map present for nova-lite)
    assert stats["input_tokens"] >= 100
    assert stats["output_tokens"] >= 50
    assert stats["estimated_cost_usd"] > 0.0
    # Alias present and equal
    assert pytest.approx(stats["estimated_cost"]) == stats["estimated_cost_usd"]
    # Limits present and show retries and at least one throttle event
    limits = stats.get("limits", {})
    assert limits.get("retries", 0) >= 1
    assert limits.get("throttle_events", 0) >= 1
    # Cleanup
    await extractor.close()


def test_pricing_lookup_with_arn_and_region_variant():
    # Stub heavy optional dependency before importing extractor package tree
    import sys, types
    sys.modules.setdefault('sentence_transformers', types.SimpleNamespace(SentenceTransformer=object))

    from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider

    extractor = LLMContentExtractor(
        llm_config=LLMConfig(provider=LLMProvider.BEDROCK, model_name="arn:aws:bedrock:eu-central-1:123:inference-profile/eu.amazon.nova-lite-v1:0")
    )
    # Simulate token recording; should match nova-lite pricing via substring
    extractor._record_usage('bedrock', extractor.config.model_name, input_tokens=1000, output_tokens=1000)
    stats = extractor.get_usage_stats()
    assert stats["estimated_cost_usd"] > 0.0


def test_credentials_parsing_csv_formats(tmp_path):
    from rules_maker.utils.credentials import CredentialManager

    # Case 1: composite name:base64(JSON)
    creds_json = {"AccessKeyId": "AKIA_TEST", "SecretAccessKey": "SECRET", "SessionToken": "TOKEN"}
    encoded_json = base64.b64encode(json.dumps(creds_json).encode()).decode()
    p1 = tmp_path / "creds1.csv"
    p1.write_text("API key name,API key\nkeyname,name:" + encoded_json + "\n")
    mgr = CredentialManager(project_root=str(tmp_path))
    c1 = mgr.load_bedrock_credentials_from_csv(str(p1))
    assert c1["aws_access_key_id"] == "AKIA_TEST"
    assert c1["aws_secret_access_key"] == "SECRET"
    assert c1["aws_session_token"] == "TOKEN"

    # Case 2: base64 of key-value pairs via composite prefix (to ensure decode path)
    kv = "AWS_ACCESS_KEY_ID=AKIA_KV\nAWS_SECRET_ACCESS_KEY=SEC\nAWS_SESSION_TOKEN=TOK\n"
    encoded_kv = base64.b64encode(kv.encode()).decode()
    p2 = tmp_path / "creds2.csv"
    p2.write_text("API key name,API key\nkeyname,name:" + encoded_kv + "\n")
    c2 = mgr.load_bedrock_credentials_from_csv(str(p2))
    assert c2["aws_access_key_id"] == "AKIA_KV"
    assert c2["aws_secret_access_key"] == "SEC"
    assert c2["aws_session_token"] == "TOK"

    # Case 3: access-key-only
    p3 = tmp_path / "creds3.csv"
    p3.write_text("API key name,API key\nkeyname,AKIA_ONLY\n")
    c3 = mgr.load_bedrock_credentials_from_csv(str(p3))
    assert c3["aws_access_key_id"] == "AKIA_ONLY"
    assert c3.get("source") in {"csv_access_key_only", "csv_raw"}


def test_validate_bedrock_access_stubbed(monkeypatch):
    from rules_maker.utils.credentials import CredentialManager

    class FakeClient:
        def converse(self, modelId, messages, inferenceConfig):
            return {
                "output": {"message": {"content": [{"text": "Hello"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 5},
            }
        class _Meta:
            endpoint_url = "https://bedrock-runtime.fake"
        meta = _Meta()

    class FakeSession:
        def client(self, name, region_name=None):
            assert name == "bedrock-runtime"
            return FakeClient()

    class FakeBoto3:
        Session = lambda self=None, **kwargs: FakeSession()  # noqa: E731

    import sys
    sys.modules['boto3'] = FakeBoto3()

    mgr = CredentialManager()
    res = mgr.validate_bedrock_access(model_id="amazon.nova-lite-v1:0", region="us-east-1")
    assert res["success"] is True
    assert res["model_id"] == "amazon.nova-lite-v1:0"
    assert res["region"] == "us-east-1"
