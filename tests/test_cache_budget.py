import asyncio
import types
import pytest
from pathlib import Path

from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider


@pytest.mark.asyncio
async def test_file_cache_hit_avoids_call_and_cost(tmp_path, monkeypatch):
    calls = {'bedrock': 0}

    async def fake_bedrock(self, prompt, system_prompt=None):
        calls['bedrock'] += 1
        # Simulate token usage
        self._record_usage('bedrock', self.config.model_name, input_tokens=10, output_tokens=20)
        return {"content": "ok"}

    cfg = LLMConfig(provider=LLMProvider.BEDROCK, model_name="test", region="us-east-1", timeout=5)
    extractor = LLMContentExtractor(llm_config=cfg, config={'cache': {'enabled': True, 'dir': str(tmp_path)}})
    monkeypatch.setattr(LLMContentExtractor, "_bedrock_request", fake_bedrock, raising=True)

    # First call populates cache
    res1 = await extractor._make_llm_request("hello")
    assert res1.get('content') == 'ok'
    cost_after_first = extractor.get_usage_stats().get('estimated_cost_usd', 0.0)
    assert calls['bedrock'] == 1

    # Second call hits cache; no new provider call, cost unchanged
    res2 = await extractor._make_llm_request("hello")
    assert res2.get('content') == 'ok'
    assert calls['bedrock'] == 1
    assert extractor.get_usage_stats().get('estimated_cost_usd', 0.0) == cost_after_first


@pytest.mark.asyncio
async def test_budget_guard_blocks_calls(monkeypatch):
    async def fake_bedrock(self, prompt, system_prompt=None):
        # Should not be called when budget exceeded
        raise AssertionError("Provider should not be called when budget exceeded")

    cfg = LLMConfig(provider=LLMProvider.BEDROCK, model_name="test", region="us-east-1", timeout=5)
    extractor = LLMContentExtractor(llm_config=cfg, config={'budget': {'hourly_usd': 0.0, 'daily_usd': 0.0}})
    monkeypatch.setattr(LLMContentExtractor, "_bedrock_request", fake_bedrock, raising=True)

    with pytest.raises(RuntimeError) as ei:
        await extractor._make_llm_request("anything")
    assert 'BudgetExceededError' in str(ei.value)
