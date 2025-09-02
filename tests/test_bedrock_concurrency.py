import asyncio
import types
import pytest

from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider


@pytest.mark.asyncio
async def test_concurrency_cap_limits_inflight_calls():
    # Configure extractor with Bedrock provider and small concurrency cap
    cap = 2
    cfg = LLMConfig(
        provider=LLMProvider.BEDROCK,
        model_name="dummy",
        region="us-east-1",
        timeout=5,
        max_concurrency=cap,
    )
    extractor = LLMContentExtractor(llm_config=cfg)

    # Track concurrent executions inside the stub
    active = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def fake_bedrock_request(self, prompt, system_prompt=None):
        nonlocal active, max_seen
        async with lock:
            active += 1
            if active > max_seen:
                max_seen = active
        try:
            await asyncio.sleep(0.05)
            return {"content": "ok"}
        finally:
            async with lock:
                active -= 1

    # Monkeypatch the Bedrock call to our stub
    extractor._bedrock_request = types.MethodType(fake_bedrock_request, extractor)

    # Fire more tasks than the cap
    N = 10
    await asyncio.gather(*[extractor._make_llm_request(f"p{i}") for i in range(N)])

    # Assert we never exceeded the cap
    assert max_seen <= cap


@pytest.mark.asyncio
async def test_throttle_events_reduce_with_capped_concurrency():
    # Helper to run a scenario with a given concurrency cap
    async def run_with_cap(cap: int, throttle_threshold: int, tasks: int = 20):
        cfg = LLMConfig(
            provider=LLMProvider.BEDROCK,
            model_name="dummy",
            region="us-east-1",
            timeout=5,
            max_concurrency=cap,
        )
        extractor = LLMContentExtractor(llm_config=cfg)

        active = 0
        lock = asyncio.Lock()

        async def fake_bedrock_request(self, prompt, system_prompt=None):
            nonlocal active
            # Simulate backend throttling when more than threshold are in-flight
            async with lock:
                active += 1
                over_capacity = active > throttle_threshold
            try:
                if over_capacity:
                    # Simulate a throttling event observed by the extractor
                    self._limits['throttle_events'] += 1
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.02)
                return {"content": "ok"}
            finally:
                async with lock:
                    active -= 1

        extractor._bedrock_request = types.MethodType(fake_bedrock_request, extractor)
        await asyncio.gather(*[extractor._make_llm_request(f"q{i}") for i in range(tasks)])
        return extractor.get_usage_stats().get('limits', {}).get('throttle_events', 0)

    # Unbounded-ish concurrency (larger than threshold) should see more throttle events
    high_cap_events = await run_with_cap(cap=16, throttle_threshold=3, tasks=40)
    # Capped concurrency at/below threshold should reduce throttle events
    low_cap_events = await run_with_cap(cap=2, throttle_threshold=3, tasks=40)

    assert low_cap_events < high_cap_events

