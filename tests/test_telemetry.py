import asyncio
import json
import types
import pytest

from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider


@pytest.mark.asyncio
async def test_structured_logs_and_redaction(monkeypatch):
    # Configure extractor with telemetry JSON + redaction
    cfg = LLMConfig(
        provider=LLMProvider.BEDROCK,
        model_name="amazon.nova-lite-v1:0",
        region="us-east-1",
        timeout=5,
    )
    extractor = LLMContentExtractor(
        llm_config=cfg,
        config={
            'telemetry': {'json': True, 'redact_prompts': True},
        },
    )

    # Stub bedrock request to succeed and increment usage
    async def ok_bedrock(self, prompt, system_prompt=None):
        # Simulate token usage so deltas are > 0
        self._record_usage('bedrock', self.config.model_name, input_tokens=12, output_tokens=34)
        return {"content": "ok"}

    monkeypatch.setattr(LLMContentExtractor, "_bedrock_request", ok_bedrock, raising=True)

    # Capture logs by monkeypatching _log_event to collect records
    events = []

    def capture(self, event, payload, redact=True):
        events.append({'event': event, **payload})

    monkeypatch.setattr(LLMContentExtractor, "_log_event", capture, raising=True)

    res = await extractor._make_llm_request("secret prompt contents")
    assert res.get('content') == 'ok'

    # There should be start and end events
    names = [e['event'] for e in events]
    assert 'llm_request_start' in names and 'llm_request_end' in names
    start = next(e for e in events if e['event'] == 'llm_request_start')
    end = next(e for e in events if e['event'] == 'llm_request_end')

    # Redaction: start should have prompt_hash and prompt_len but no raw prompt
    assert 'prompt_hash' in start and 'prompt_len' in start
    assert 'prompt' not in start

    # End event contains required metadata
    assert end['success'] is True
    assert isinstance(end['latency_ms'], int) and end['latency_ms'] >= 0
    assert 'usage_delta' in end and end['usage_delta']['input_tokens'] >= 12
    assert 'limits_delta' in end and 'retries' in end['limits_delta']

