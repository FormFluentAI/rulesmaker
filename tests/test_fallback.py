import asyncio
import types
import pytest

from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider


@pytest.mark.asyncio
async def test_bedrock_fallback_to_local_with_usage_split(monkeypatch):
    # Configure primary as Bedrock with fallback enabled to local
    cfg = LLMConfig(
        provider=LLMProvider.BEDROCK,
        model_name="amazon.nova-lite-v1:0",
        region="us-east-1",
        timeout=5,
        max_concurrency=2,
    )
    extractor = LLMContentExtractor(
        llm_config=cfg,
        config={
            'fallback': {'enabled': True, 'providers': ['local']},
            'providers': {'local': {'base_url': 'http://dummy', 'model': 'stub-model'}},
        },
    )

    # Force Bedrock to fail immediately
    async def fail_bedrock(self, prompt, system_prompt=None):
        raise RuntimeError("simulated bedrock outage")

    # Make local return a simple JSON content and record usage via _record_usage
    async def ok_local(self, prompt, system_prompt=None):
        # Simulate token usage for cost tracking
        self._record_usage('local', self.config.model_name, prompt_tokens=10, completion_tokens=20)
        return {"content": "fallback-ok"}

    monkeypatch.setattr(LLMContentExtractor, "_bedrock_request", fail_bedrock, raising=True)
    monkeypatch.setattr(LLMContentExtractor, "_local_request", ok_local, raising=True)

    res = await extractor._make_llm_request("hi")
    assert res.get('content') == 'fallback-ok'

    stats = extractor.get_usage_stats()
    # Ensure per-provider breakdown includes local (fallback) path
    byp = stats.get('by_provider', {})
    assert 'local' in byp and byp['local']['requests'] >= 1

