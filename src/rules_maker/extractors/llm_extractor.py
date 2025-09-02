"""
LLM-powered content extractor.

Uses Language Models to intelligently extract and understand
documentation content for rule generation.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from bs4 import BeautifulSoup
import httpx
import time
import uuid
import hashlib

from .base import ContentExtractor
from ..models import ContentSection, Rule, RuleSet


logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    BEDROCK = "bedrock"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 30
    region: Optional[str] = None  # For AWS Bedrock
    aws_access_key_id: Optional[str] = None  # For AWS Bedrock
    aws_secret_access_key: Optional[str] = None  # For AWS Bedrock
    aws_session_token: Optional[str] = None  # For AWS Bedrock
    # Retry/backoff configuration (for Bedrock/transient errors)
    retry_max_attempts: int = 3
    retry_base_ms: int = 250
    retry_max_ms: int = 2000
    # Concurrency control for LLM calls
    max_concurrency: int = 4


class LLMContentExtractor(ContentExtractor):
    """LLM-powered content extraction and rule generation."""
    
    def __init__(
        self,
        patterns: Optional[List] = None,
        llm_config: Optional[LLMConfig] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the LLM extractor."""
        super().__init__(patterns)
        
        self.config = llm_config or LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo"
        )

        # Optional config dict override (env fallback already handled below for specific knobs)
        if isinstance(config, dict):
            bed = (config or {}).get('bedrock') if config else None
            if bed:
                self.config.provider = LLMProvider.BEDROCK
                self.config.model_name = str(bed.get('model_id') or self.config.model_name)
                self.config.region = str(bed.get('region') or self.config.region or 'us-east-1')
                if 'timeout' in bed:
                    self.config.timeout = int(bed['timeout'])
                if 'concurrency' in bed:
                    self.config.max_concurrency = int(bed['concurrency'])
                retry = bed.get('retry') or {}
                if retry:
                    if 'max_attempts' in retry:
                        self.config.retry_max_attempts = int(retry['max_attempts'])
                    if 'base_ms' in retry:
                        self.config.retry_base_ms = int(retry['base_ms'])
                    if 'max_ms' in retry:
                        self.config.retry_max_ms = int(retry['max_ms'])
        # Multi-provider fallback and credentials config
        self._fallback_enabled = False
        self._fallback_providers: List[str] = []
        self._provider_settings: Dict[str, Dict[str, Any]] = {}
        if isinstance(config, dict):
            # Providers API keys / settings
            providers = config.get('providers') or {}
            if isinstance(providers, dict):
                self._provider_settings = providers
            # Fallback chain
            fb = (config.get('bedrock') or {}).get('fallback') or config.get('fallback') or {}
            if isinstance(fb, dict):
                self._fallback_enabled = bool(fb.get('enabled', False))
                prov = fb.get('providers') or []
                if isinstance(prov, list):
                    self._fallback_providers = [str(p).lower() for p in prov]
        # Env overrides for fallback flags
        import os
        env_fb = (os.environ.get('FALLBACK_ENABLED') or '').lower()
        if env_fb in {'1', 'true', 'yes', 'on'}:
            self._fallback_enabled = True
        env_list = os.environ.get('FALLBACK_PROVIDERS')
        if env_list:
            self._fallback_providers = [p.strip().lower() for p in env_list.split(',') if p.strip()]
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=self.config.timeout)

        # Concurrency limiting (semaphore)
        # Allow env override to quickly tune without code changes
        try:
            env_conc = int((__import__('os').environ.get('BEDROCK_MAX_CONCURRENCY') or '').strip() or 0)
        except Exception:
            env_conc = 0
        max_concurrency = env_conc or getattr(self.config, 'max_concurrency', 4) or 4
        self._semaphore = asyncio.Semaphore(max_concurrency)
        logger.info(f"LLMContentExtractor concurrency cap set to {max_concurrency}")

        # Telemetry settings (JSON logs, prompt redaction)
        self._telemetry_json = False
        self._telemetry_redact = True
        if isinstance(config, dict):
            tel = (config.get('telemetry') if config else None) or {}
            if isinstance(tel, dict):
                self._telemetry_json = bool(tel.get('json', False))
                self._telemetry_redact = bool(tel.get('redact_prompts', True))
        try:
            import os as _os
            env_json = (_os.environ.get('RULES_MAKER_LOG_JSON') or '').lower()
            if env_json in {'1','true','yes','on'}:
                self._telemetry_json = True
            env_redact = (_os.environ.get('RULES_MAKER_REDACT_PROMPTS') or '').lower()
            if env_redact in {'0','false','no','off'}:
                self._telemetry_redact = False
        except Exception:
            pass

        # Caching settings
        self._cache_enabled = False
        self._cache_dir: Optional[str] = None
        self._memory_cache: Dict[str, Any] = {}
        if isinstance(config, dict):
            cache = (config.get('cache') if config else None) or {}
            if isinstance(cache, dict):
                self._cache_enabled = bool(cache.get('enabled', False))
                self._cache_dir = str(cache.get('dir') or '') or None
        try:
            _os = __import__('os')
            env_cache = (_os.environ.get('RULES_MAKER_CACHE_ENABLED') or '').lower()
            if env_cache in {'1','true','yes','on'}:
                self._cache_enabled = True
            if _os.environ.get('RULES_MAKER_CACHE_DIR'):
                self._cache_dir = _os.environ.get('RULES_MAKER_CACHE_DIR')
        except Exception:
            pass
        if self._cache_enabled and not self._cache_dir:
            # Default cache dir under user cache
            from pathlib import Path as _Path
            self._cache_dir = str(_Path.home() / '.cache' / 'rules_maker' / 'llm')

        # Budget guardrails
        self._budget_hourly_usd: Optional[float] = None
        self._budget_daily_usd: Optional[float] = None
        self._hourly_cost_accum = 0.0
        self._daily_cost_accum = 0.0
        self._hourly_started_at = time.time()
        self._daily_started_at = time.time()
        if isinstance(config, dict):
            budget = (config.get('budget') if config else None) or {}
            if isinstance(budget, dict):
                if budget.get('hourly_usd') is not None:
                    self._budget_hourly_usd = float(budget.get('hourly_usd'))
                if budget.get('daily_usd') is not None:
                    self._budget_daily_usd = float(budget.get('daily_usd'))
        try:
            _os = __import__('os')
            if _os.environ.get('RULES_MAKER_BUDGET_HOURLY_USD'):
                self._budget_hourly_usd = float(_os.environ.get('RULES_MAKER_BUDGET_HOURLY_USD'))
            if _os.environ.get('RULES_MAKER_BUDGET_DAILY_USD'):
                self._budget_daily_usd = float(_os.environ.get('RULES_MAKER_BUDGET_DAILY_USD'))
        except Exception:
            pass
        
        # Token usage/cost tracking
        self._usage: Dict[str, Dict[str, float]] = {
            'prompt_tokens': 0.0,
            'completion_tokens': 0.0,
            'total_tokens': 0.0,
            'input_tokens': 0.0,   # Anthropic naming
            'output_tokens': 0.0,  # Anthropic naming
            'estimated_cost_usd': 0.0,
            'requests': 0.0
        }
        # Per-provider usage breakdown
        self._usage_by_provider: Dict[str, Dict[str, float]] = {}
        # Basic rate-limit and retry metrics
        self._limits: Dict[str, Any] = {
            'retries': 0,
            'throttle_events': 0,
            'last_error_code': None,
            'last_retry_delay_ms': 0
        }
        # Simple price map (USD per 1K tokens). These are placeholders; adjust as needed.
        self._price_map: Dict[str, Dict[str, float]] = {
            'openai': {
                'gpt-3.5-turbo':  {'input': 0.0005, 'output': 0.0015},
                'gpt-4o-mini':    {'input': 0.00015, 'output': 0.0006},
                'gpt-4o':         {'input': 0.005,  'output': 0.015},
                'gpt-4-turbo':    {'input': 0.01,   'output': 0.03},
            },
            'anthropic': {
                'claude-3-haiku':   {'input': 0.00025, 'output': 0.00125},
                'claude-3-sonnet':  {'input': 0.003,   'output': 0.015},
                'claude-3-opus':    {'input': 0.015,   'output': 0.075},
            },
            'bedrock': {
                'amazon.nova-lite-v1:0':     {'input': 0.00006, 'output': 0.00024},
                'eu.amazon.nova-lite-v1:0':  {'input': 0.00006, 'output': 0.00024},
                'amazon.nova-micro-v1:0':    {'input': 0.000035, 'output': 0.00014},
                'amazon.nova-pro-v1:0':      {'input': 0.0008, 'output': 0.0032},
                'anthropic.claude-3-sonnet-20240229-v1:0': {'input': 0.003, 'output': 0.015},
                'anthropic.claude-3-haiku-20240307-v1:0':  {'input': 0.00025, 'output': 0.00125},
                'anthropic.claude-3-5-sonnet-20240620-v1:0': {'input': 0.003, 'output': 0.015},
                'anthropic.claude-3-5-haiku-20241022-v1:0':  {'input': 0.00025, 'output': 0.00125},
            }
        }
        
        # Prompt templates
        self.extraction_prompt = self._load_extraction_prompt()
        self.rule_generation_prompt = self._load_rule_generation_prompt()
        
    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured content using LLM analysis."""
        try:
            # Get clean text content
            content_text = self._extract_clean_text(soup)
            
            # Use LLM to analyze and structure content
            structured_content = self._analyze_content_with_llm(content_text, url)
            
            # Extract sections using LLM
            sections = self.extract_sections(soup, url)
            
            return {
                'title': structured_content.get('title', ''),
                'content': content_text,
                'sections': [
                    (section.model_dump() if hasattr(section, 'model_dump') else section.dict() if hasattr(section, 'dict') else section)
                    for section in sections
                ],
                'document_type': structured_content.get('document_type', 'unknown'),
                'key_concepts': structured_content.get('key_concepts', []),
                'technologies': structured_content.get('technologies', []),
                'complexity_level': structured_content.get('complexity_level', 'intermediate'),
                'summary': structured_content.get('summary', ''),
                'metadata': structured_content.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Error in LLM extraction for {url}: {str(e)}")
            return self._fallback_extraction(soup, url)
    
    def extract_sections(self, soup: BeautifulSoup, url: str) -> List[ContentSection]:
        """Extract content sections using LLM understanding."""
        try:
            # Get structured content
            content_text = self._extract_clean_text(soup)
            
            # Use LLM to identify and categorize sections
            sections_data = self._extract_sections_with_llm(content_text, url)
            
            sections = []
            for section_data in sections_data:
                section = ContentSection(
                    title=section_data.get('title', ''),
                    content=section_data.get('content', ''),
                    level=section_data.get('level', 1),
                    url=url,
                    metadata={
                        'section_type': section_data.get('type', 'general'),
                        'importance': section_data.get('importance', 0.5),
                        'concepts': section_data.get('concepts', []),
                        'code_examples': section_data.get('code_examples', []),
                        'llm_generated': True
                    }
                )
                sections.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in LLM section extraction for {url}: {str(e)}")
            return self._fallback_section_extraction(soup, url)
    
    async def generate_rules(
        self, 
        content: Union[str, List[ContentSection]], 
        target_format: str = "cursor",
        context: Optional[Dict[str, Any]] = None
    ) -> RuleSet:
        """Generate coding rules from extracted content."""
        try:
            # Prepare content for rule generation
            if isinstance(content, list):
                content_text = self._sections_to_text(content)
            else:
                content_text = content
            
            # Generate rules using LLM
            rules_data = await self._generate_rules_with_llm(
                content_text, target_format, context
            )
            
            # Convert to Rule objects
            rules = []
            for rule_data in rules_data.get('rules', []):
                rule = Rule(
                    id=rule_data.get('id', f"rule_{len(rules)}"),
                    title=rule_data.get('title', ''),
                    description=rule_data.get('description', ''),
                    category=rule_data.get('category', 'general'),
                    priority=rule_data.get('priority', 1),
                    tags=rule_data.get('tags', []),
                    examples=rule_data.get('examples', []),
                    anti_patterns=rule_data.get('anti_patterns', []),
                    metadata={
                        'generated_by': 'llm',
                        'confidence': rule_data.get('confidence', 0.5),
                        'source_url': context.get('url') if context else None
                    }
                )
                rules.append(rule)
            
            return RuleSet(
                name=rules_data.get('name', 'Generated Rules'),
                description=rules_data.get('description', ''),
                rules=rules,
                metadata={
                    'generated_by': 'llm',
                    'model': self.config.model_name,
                    'target_format': target_format,
                    'context': context or {}
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating rules: {str(e)}")
            return RuleSet(name="Error", description=f"Failed to generate rules: {str(e)}", rules=[])
    
    async def _make_llm_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a request to the configured LLM provider."""
        async with self._semaphore:
            trace_id = str(uuid.uuid4())
            start = time.monotonic()
            # Snapshot limits/usage for deltas
            limits_before = dict(self._limits)
            usage_before = dict(self._usage)
            # Prepare prompt hash/length; avoid logging raw prompts
            phash = hashlib.sha256(prompt.encode('utf-8')).hexdigest() if prompt else None
            plength = len(prompt or '')
            region = getattr(self.config, 'region', None)
            provider_str = str(self.config.provider.value if isinstance(self.config.provider, LLMProvider) else self.config.provider)
            model = self.config.model_name
            self._log_event('llm_request_start', {
                'trace_id': trace_id,
                'provider': provider_str,
                'model': model,
                'region': region,
                'timeout_s': self.config.timeout,
                'prompt_len': plength,
                **({} if not phash else {'prompt_hash': phash}),
            }, redact=True)
            # Cache check before any provider calls
            cache_key = self._build_cache_key(provider_str, model, prompt, system_prompt)
            if self._cache_enabled:
                cached = self._cache_get(cache_key)
                if cached is not None:
                    self._log_event('llm_cache_hit', {
                        'trace_id': trace_id,
                        'provider': provider_str,
                        'model': model,
                        'region': region,
                        'cache_key': cache_key[:16],
                    })
                    end = time.monotonic()
                    self._emit_end_event(trace_id, provider_str, model, region, start, end, limits_before, usage_before, success=True)
                    return cached

            # Budget guardrails: block if exceeded (cache miss only)
            if self._budget_exceeded():
                reason = self._budget_reason()
                self._log_event('llm_budget_block', {
                    'trace_id': trace_id,
                    'provider': provider_str,
                    'model': model,
                    'region': region,
                    'reason': reason,
                })
                end = time.monotonic()
                self._emit_end_event(trace_id, provider_str, model, region, start, end, limits_before, usage_before, success=False, error=RuntimeError('BudgetExceededError'))
                raise RuntimeError(f"BudgetExceededError: {reason}")

            primary = str(self.config.provider.value if isinstance(self.config.provider, LLMProvider) else self.config.provider).lower()
            sequence: List[str] = [primary]
            # Only add fallback providers when primary is bedrock and fallback enabled
            if primary == 'bedrock' and self._fallback_enabled and self._fallback_providers:
                sequence.extend([p for p in self._fallback_providers if p != primary])

            last_err: Optional[Exception] = None
            for prov in sequence:
                try:
                    if prov == 'openai':
                        res = await self._openai_request(prompt, system_prompt)
                    elif prov == 'anthropic':
                        res = await self._anthropic_request(prompt, system_prompt)
                    elif prov == 'huggingface':
                        res = await self._huggingface_request(prompt, system_prompt)
                    elif prov == 'bedrock':
                        res = await self._bedrock_request(prompt, system_prompt)
                    elif prov == 'local':
                        res = await self._local_request(prompt, system_prompt)
                    else:
                        raise ValueError(f"Unsupported provider in fallback: {prov}")

                    # Success path for primary provider call: emit end, cache, return
                    end = time.monotonic()
                    self._emit_end_event(trace_id, provider_str, model, region, start, end, limits_before, usage_before, success=True)
                    if self._cache_enabled:
                        self._cache_put(cache_key, res)
                    return res
                except Exception as e:
                    # Log per-attempt failure
                    self._log_event('llm_attempt_error', {
                        'trace_id': trace_id,
                        'provider': prov,
                        'model': model,
                        'region': region,
                        'error': type(e).__name__,
                        'message': str(e)[:300],
                    }, redact=True)
                    last_err = e
                    # Prepare and try next provider by spinning a temp extractor with per-provider settings
                    # Skip if this was the last one
                    if prov == sequence[-1]:
                        break
                    # Build temp config for next provider
                    next_idx = sequence.index(prov) + 1
                    next_prov = sequence[next_idx]
                    tmp_cfg = self._build_provider_config(next_prov)
                    if not tmp_cfg:
                        continue
                    # Use a fresh extractor so we don't mutate shared config; disable nested fallback to avoid loops
                    tmp_extractor = LLMContentExtractor(llm_config=tmp_cfg, config={'fallback': {'enabled': False}})
                    try:
                        res = await tmp_extractor._make_llm_request(prompt, system_prompt)
                        # Merge usage into this extractor for unified reporting
                        self._merge_usage_stats(tmp_extractor.get_usage_stats())
                        end = time.monotonic()
                        self._emit_end_event(trace_id, provider_str, model, region, start, end, limits_before, usage_before, success=True)
                        if self._cache_enabled:
                            self._cache_put(cache_key, res)
                        return res
                    except Exception as e2:
                        last_err = e2
                        continue
            # If we reach here, all attempts failed
            end = time.monotonic()
            self._emit_end_event(trace_id, provider_str, model, region, start, end, limits_before, usage_before, success=False, error=last_err)
            raise last_err or RuntimeError("All provider attempts failed")

    def _build_cache_key(self, provider: str, model: str, prompt: str, system_prompt: Optional[str]) -> str:
        h = hashlib.sha256()
        h.update((provider or '').encode('utf-8'))
        h.update(b'|')
        h.update((model or '').encode('utf-8'))
        h.update(b'|')
        h.update((system_prompt or '').encode('utf-8'))
        h.update(b'|')
        h.update((prompt or '').encode('utf-8'))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._memory_cache:
            return self._memory_cache[key]
        if not self._cache_dir:
            return None
        from pathlib import Path
        p = Path(self._cache_dir)
        try:
            if p.is_dir():
                f = p / f"{key}.json"
                if f.exists():
                    try:
                        data = json.loads(f.read_text(encoding='utf-8'))
                        self._memory_cache[key] = data
                        return data
                    except Exception:
                        return None
        except Exception:
            return None
        return None

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        self._memory_cache[key] = value
        if not self._cache_dir:
            return
        from pathlib import Path
        p = Path(self._cache_dir)
        try:
            p.mkdir(parents=True, exist_ok=True)
            f = p / f"{key}.json"
            try:
                f.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
                self._log_event('llm_cache_store', {'cache_key': key[:16], 'path': str(f)})
            except Exception:
                pass
        except Exception:
            pass

    def _budget_exceeded(self) -> bool:
        now = time.time()
        # reset windows
        if now - self._hourly_started_at >= 3600:
            self._hourly_started_at = now
            self._hourly_cost_accum = 0.0
        if now - self._daily_started_at >= 86400:
            self._daily_started_at = now
            self._daily_cost_accum = 0.0
        if self._budget_hourly_usd is not None and self._hourly_cost_accum >= self._budget_hourly_usd:
            return True
        if self._budget_daily_usd is not None and self._daily_cost_accum >= self._budget_daily_usd:
            return True
        return False

    def _budget_reason(self) -> str:
        reasons = []
        if self._budget_hourly_usd is not None and self._hourly_cost_accum >= self._budget_hourly_usd:
            reasons.append(f"hourly cap {self._budget_hourly_usd:.4f} reached")
        if self._budget_daily_usd is not None and self._daily_cost_accum >= self._budget_daily_usd:
            reasons.append(f"daily cap {self._budget_daily_usd:.4f} reached")
        return '; '.join(reasons) or 'budget reached'

    def _emit_end_event(self, trace_id: str, provider: str, model: str, region: Optional[str], start: float, end: float,
                         limits_before: Dict[str, Any], usage_before: Dict[str, Any], success: bool, error: Optional[Exception] = None) -> None:
        latency_ms = int((end - start) * 1000)
        # Compute deltas
        retries_delta = int(self._limits.get('retries', 0) or 0) - int(limits_before.get('retries', 0) or 0)
        throttle_delta = int(self._limits.get('throttle_events', 0) or 0) - int(limits_before.get('throttle_events', 0) or 0)
        in_delta = float(self._usage.get('input_tokens', 0.0) or 0.0) - float(usage_before.get('input_tokens', 0.0) or 0.0)
        out_delta = float(self._usage.get('output_tokens', 0.0) or 0.0) - float(usage_before.get('output_tokens', 0.0) or 0.0)
        cost_delta = float(self._usage.get('estimated_cost_usd', 0.0) or 0.0) - float(usage_before.get('estimated_cost_usd', 0.0) or 0.0)
        payload = {
            'trace_id': trace_id,
            'provider': provider,
            'model': model,
            'region': region,
            'success': bool(success),
            'latency_ms': latency_ms,
            'usage_delta': {
                'input_tokens': int(in_delta),
                'output_tokens': int(out_delta),
                'estimated_cost_usd': float(f"{cost_delta:.6f}")
            },
            'limits_delta': {
                'retries': int(retries_delta),
                'throttle_events': int(throttle_delta)
            }
        }
        if not success and error is not None:
            payload['error'] = {'type': type(error).__name__, 'message': str(error)[:300]}
        self._log_event('llm_request_end', payload, redact=True)

    def _log_event(self, event: str, payload: Dict[str, Any], redact: bool = True) -> None:
        record = {'event': event, **payload}
        if self._telemetry_json:
            try:
                logger.info(json.dumps(record, ensure_ascii=False))
                return
            except Exception:
                pass
        # Fallback to compact key=value line
        parts = [f"event={event}"] + [f"{k}={v}" for k, v in record.items() if k != 'event']
        logger.info(' '.join(parts))

    def _build_provider_config(self, prov: str) -> Optional[LLMConfig]:
        prov = prov.lower()
        import os
        if prov == 'openai':
            api_key = (self._provider_settings.get('openai') or {}).get('api_key') or os.environ.get('OPENAI_API_KEY')
            model = (self._provider_settings.get('openai') or {}).get('model') or 'gpt-4o-mini'
            if not api_key:
                return None
            return LLMConfig(provider=LLMProvider.OPENAI, api_key=api_key, model_name=model,
                             temperature=self.config.temperature, max_tokens=self.config.max_tokens, timeout=self.config.timeout)
        if prov == 'anthropic':
            api_key = (self._provider_settings.get('anthropic') or {}).get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
            model = (self._provider_settings.get('anthropic') or {}).get('model') or 'claude-3-haiku-20240307'
            if not api_key:
                return None
            return LLMConfig(provider=LLMProvider.ANTHROPIC, api_key=api_key, model_name=model,
                             temperature=self.config.temperature, max_tokens=self.config.max_tokens, timeout=self.config.timeout)
        if prov == 'huggingface':
            api_key = (self._provider_settings.get('huggingface') or {}).get('api_key') or os.environ.get('HUGGINGFACE_API_KEY')
            model = (self._provider_settings.get('huggingface') or {}).get('model') or 'meta-llama/Meta-Llama-3-8B-Instruct'
            if not api_key:
                return None
            return LLMConfig(provider=LLMProvider.HUGGINGFACE, api_key=api_key, model_name=model,
                             temperature=self.config.temperature, max_tokens=self.config.max_tokens, timeout=self.config.timeout)
        if prov == 'local':
            base_url = (self._provider_settings.get('local') or {}).get('base_url') or os.environ.get('LOCAL_LLM_BASE_URL')
            model = (self._provider_settings.get('local') or {}).get('model') or 'qwen2.5:14b'
            if not base_url:
                return None
            return LLMConfig(provider=LLMProvider.LOCAL, base_url=base_url, model_name=model,
                             temperature=self.config.temperature, max_tokens=self.config.max_tokens, timeout=self.config.timeout)
        return None
    
    async def _openai_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        response = await self.client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        # Track usage/cost
        usage = result.get('usage', {})
        if usage:
            self._record_usage('openai', self.config.model_name, 
                               prompt_tokens=usage.get('prompt_tokens', 0),
                               completion_tokens=usage.get('completion_tokens', 0),
                               total_tokens=usage.get('total_tokens', 0))
        content = result["choices"][0]["message"]["content"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    async def _anthropic_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "model": self.config.model_name or "claude-3-sonnet-20240229",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": full_prompt}]
        }
        
        response = await self.client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        # Track usage/cost (Anthropic)
        if 'usage' in result:
            u = result['usage']
            self._record_usage('anthropic', self.config.model_name, 
                               input_tokens=u.get('input_tokens', 0),
                               output_tokens=u.get('output_tokens', 0))
        content = result["content"][0]["text"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    async def _huggingface_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to Hugging Face API."""
        url = f"https://api-inference.huggingface.co/models/{self.config.model_name}"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens
            }
        }
        
        response = await self.client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    async def _bedrock_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to AWS Bedrock."""
        try:
            import boto3
            from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError, ConnectionClosedError
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError("boto3 is required for Bedrock. Install with: pip install boto3")

        # Setup AWS session with credentials from config or environment
        session_kwargs = {}
        if self.config.aws_access_key_id:
            session_kwargs['aws_access_key_id'] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = self.config.aws_secret_access_key  
        if self.config.aws_session_token:
            session_kwargs['aws_session_token'] = self.config.aws_session_token
        
        session = boto3.Session(**session_kwargs)
        region = self.config.region or "us-east-1"
        # Botocore client config with tighter timeouts; we use our own retry/backoff below
        timeout = int(self.config.timeout or 30)
        boto_config = BotoConfig(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={'max_attempts': 0, 'mode': 'standard'}
        )
        # Some test stubs may not accept 'config' kwarg; fall back gracefully
        try:
            client = session.client('bedrock-runtime', region_name=region, config=boto_config)
        except TypeError:
            client = session.client('bedrock-runtime', region_name=region)
        
        # Prepare messages
        messages = []
        if system_prompt:
            # For models that support system messages, we'll include it in the conversation
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
            
        messages = [{"role": "user", "content": [{"text": full_prompt}]}]
        
        # Make the request
        model_id = self.config.model_name or "amazon.nova-lite-v1:0"
        
        # Retry/backoff configuration (allow env overrides)
        import os, random, asyncio
        max_attempts = int(os.environ.get('BEDROCK_RETRY_MAX_ATTEMPTS', self.config.retry_max_attempts))
        base_ms = int(os.environ.get('BEDROCK_RETRY_BASE_MS', self.config.retry_base_ms))
        max_ms = int(os.environ.get('BEDROCK_RETRY_MAX_MS', self.config.retry_max_ms))

        def _is_transient(err) -> bool:
            # Botocore ClientError with throttling or 5xx
            if isinstance(err, ClientError):
                code = (err.response or {}).get('Error', {}).get('Code', '')
                status = (err.response or {}).get('ResponseMetadata', {}).get('HTTPStatusCode')
                if code in {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}:
                    return True
                if status and int(status) in {429, 500, 502, 503, 504}:
                    return True
            # Network/transient botocore exceptions
            if isinstance(err, (EndpointConnectionError, ReadTimeoutError, ConnectionClosedError)):
                return True
            return False

        attempt = 0
        last_error = None
        while True:
            try:
                response = client.converse(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig={
                        'maxTokens': self.config.max_tokens,
                        'temperature': self.config.temperature
                    }
                )
                # Extract content from response
                output = response.get('output', {}).get('message', {}).get('content', [])
                content = ''
                if output and isinstance(output, list):
                    texts = [c.get('text', '') for c in output if isinstance(c, dict)]
                    content = '\n'.join(texts).strip()
                # Track usage
                usage = response.get('usage', {})
                if usage:
                    input_tokens = usage.get('inputTokens', 0)
                    output_tokens = usage.get('outputTokens', 0)
                    self._record_usage('bedrock', model_id,
                                       input_tokens=input_tokens,
                                       output_tokens=output_tokens)
                # Try to parse as JSON, fallback to plain content
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"content": content}

            except Exception as e:
                last_error = e
                transient = _is_transient(e)
                # If not transient or no more retries, raise
                if (not transient) or (attempt >= max_attempts):
                    if isinstance(e, ClientError):
                        error_code = (e.response or {}).get('Error', {}).get('Code', 'Unknown')
                        error_message = (e.response or {}).get('Error', {}).get('Message', str(e))
                        logger.error(f"Bedrock ClientError {error_code}: {error_message}")
                        # Update limits metadata
                        self._limits['last_error_code'] = error_code
                        raise RuntimeError(f"Bedrock request failed: {error_code} - {error_message}")
                    logger.error(f"Bedrock request error: {str(e)}")
                    raise

                # Compute backoff with jitter, prefer Retry-After if present
                retry_after_s = None
                if isinstance(e, ClientError):
                    try:
                        headers = (getattr(e, 'response', {}) or {}).get('ResponseMetadata', {}).get('HTTPHeaders', {})
                        ra = headers.get('retry-after') or headers.get('Retry-After')
                        if ra:
                            retry_after_s = float(ra)
                    except Exception:
                        retry_after_s = None
                delay_ms = min(base_ms * (2 ** attempt), max_ms)
                jitter_ms = random.randint(0, max(base_ms // 2, 1))
                total_delay = retry_after_s if retry_after_s is not None else (delay_ms + jitter_ms) / 1000.0
                # Log structured retry info
                error_code = ''
                status = None
                if isinstance(e, ClientError):
                    error_code = (e.response or {}).get('Error', {}).get('Code', '')
                    status = (e.response or {}).get('ResponseMetadata', {}).get('HTTPStatusCode')
                delay_log_ms = int(total_delay * 1000)
                logger.warning(
                    f"bedrock_retry attempt={attempt+1} delay_ms={delay_log_ms} code={error_code} status={status}"
                )
                # Update limits counters
                self._limits['retries'] += 1
                self._limits['last_retry_delay_ms'] = delay_log_ms
                self._limits['last_error_code'] = error_code or status
                if error_code in {"ThrottlingException", "TooManyRequestsException"} or status == 429:
                    self._limits['throttle_events'] += 1
                attempt += 1
                await asyncio.sleep(total_delay)
    
    async def _local_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to local LLM server."""
        if not self.config.base_url:
            raise ValueError("base_url required for local LLM provider")
        
        url = f"{self.config.base_url}/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        response = await self.client.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    def _analyze_content_with_llm(self, content: str, url: str) -> Dict[str, Any]:
        """Analyze content using LLM to extract structured information."""
        prompt = self.extraction_prompt.format(
            content=content[:4000],  # Limit content length
            url=url
        )
        
        # Synchronous wrapper for async function
        import asyncio
        
        async def _async_analyze():
            return await self._make_llm_request(prompt)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_analyze())
                    return future.result()
            else:
                return loop.run_until_complete(_async_analyze())
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {}
    
    def _extract_sections_with_llm(self, content: str, url: str) -> List[Dict[str, Any]]:
        """Extract sections using LLM."""
        prompt = f"""
        Analyze this documentation content and identify distinct sections:
        
        Content: {content[:3000]}
        URL: {url}
        
        Return a JSON array of sections with this structure:
        {{
            "title": "section title",
            "content": "section content", 
            "type": "introduction|installation|tutorial|api|examples|configuration|troubleshooting",
            "level": 1-6,
            "importance": 0.0-1.0,
            "concepts": ["concept1", "concept2"],
            "code_examples": ["example1", "example2"]
        }}
        """
        
        try:
            import asyncio
            async def _async_extract():
                return await self._make_llm_request(prompt)
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_extract())
                    result = future.result()
            else:
                result = loop.run_until_complete(_async_extract())
            
            if isinstance(result, dict) and 'sections' in result:
                return result['sections']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            logger.error(f"LLM section extraction failed: {str(e)}")
            return []
    
    async def _generate_rules_with_llm(
        self, 
        content: str, 
        target_format: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate coding rules using LLM."""
        prompt = self.rule_generation_prompt.format(
            content=content[:4000],
            target_format=target_format,
            context=json.dumps(context or {}, indent=2)
        )
        
        try:
            result = await self._make_llm_request(prompt)
            return result
        except Exception as e:
            logger.error(f"LLM rule generation failed: {str(e)}")
            return {"rules": []}
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and normalize whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _sections_to_text(self, sections: List[ContentSection]) -> str:
        """Convert sections to text for processing."""
        text_parts = []
        for section in sections:
            text_parts.append(f"# {section.title}\n{section.content}\n")
        return '\n'.join(text_parts)
    
    def _fallback_extraction(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Fallback extraction without LLM."""
        title = ""
        if soup.title:
            title = soup.title.get_text().strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        
        content = self._extract_clean_text(soup)
        
        return {
            'title': title,
            'content': content,
            'sections': [],
            'document_type': 'unknown',
            'key_concepts': [],
            'technologies': [],
            'complexity_level': 'unknown',
            'summary': '',
            'metadata': {}
        }
    
    def _fallback_section_extraction(self, soup: BeautifulSoup, url: str) -> List[ContentSection]:
        """Fallback section extraction without LLM."""
        sections = []
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            title = heading.get_text().strip()
            level = int(heading.name[1])
            
            # Get content until next heading
            content_parts = []
            current = heading.next_sibling
            
            while current:
                if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                
                if hasattr(current, 'get_text'):
                    text = current.get_text().strip()
                    if text:
                        content_parts.append(text)
                
                current = current.next_sibling
            
            content = '\n'.join(content_parts)
            
            if title and content:
                section = ContentSection(
                    title=title,
                    content=content,
                    level=level,
                    url=url,
                    metadata={'llm_generated': False}
                )
                sections.append(section)
        
        return sections
    
    def _load_extraction_prompt(self) -> str:
        """Load the content extraction prompt template."""
        return """
        Analyze this documentation content and extract structured information:
        
        Content: {content}
        URL: {url}
        
        Return a JSON object with this structure:
        {{
            "title": "document title",
            "document_type": "api|tutorial|guide|reference|installation|troubleshooting",
            "key_concepts": ["concept1", "concept2"],
            "technologies": ["tech1", "tech2"],
            "complexity_level": "beginner|intermediate|advanced",
            "summary": "brief summary of the content",
            "metadata": {{
                "language": "programming language if applicable",
                "framework": "framework name if applicable",
                "version": "version if mentioned"
            }}
        }}
        
        Focus on extracting actionable information that would be useful for generating coding rules.
        """
    
    def _load_rule_generation_prompt(self) -> str:
        """Load the rule generation prompt template."""
        return """
        Generate coding rules from this documentation content for {target_format} format:
        
        Content: {content}
        Context: {context}
        
        Return a JSON object with this structure:
        {{
            "name": "Rule set name",
            "description": "Description of the rule set",
            "rules": [
                {{
                    "id": "unique_rule_id",
                    "title": "Rule title",
                    "description": "Detailed rule description",
                    "category": "category name",
                    "priority": 1-5,
                    "tags": ["tag1", "tag2"],
                    "examples": ["good example 1", "good example 2"],
                    "anti_patterns": ["what to avoid 1", "what to avoid 2"],
                    "confidence": 0.0-1.0
                }}
            ]
        }}
        
        Generate practical, actionable rules that would help developers write better code.
        Focus on best practices, common patterns, and error prevention.
        """
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    # Usage tracking helpers
    def _record_usage(
        self,
        provider_key: str,
        model_name: str,
        prompt_tokens: float = 0.0,
        completion_tokens: float = 0.0,
        total_tokens: float = 0.0,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
    ) -> None:
        # Aggregate token counts
        self._usage['prompt_tokens'] += float(prompt_tokens)
        self._usage['completion_tokens'] += float(completion_tokens)
        self._usage['total_tokens'] += float(total_tokens or (prompt_tokens + completion_tokens))
        self._usage['input_tokens'] += float(input_tokens or prompt_tokens)
        self._usage['output_tokens'] += float(output_tokens or completion_tokens)
        self._usage['requests'] += 1.0
        
        # Estimate cost if we have price info
        provider_prices = self._price_map.get(provider_key.lower()) or {}
        # Try direct lookup first
        model_prices = provider_prices.get(model_name) or {}
        # Fallbacks: handle Bedrock ARNs or region-prefixed variants by substring matching
        if not model_prices and provider_prices:
            # Prefer the longest matching key contained in the model_name
            candidates = [k for k in provider_prices.keys() if k in str(model_name)]
            if candidates:
                best = max(candidates, key=len)
                model_prices = provider_prices.get(best) or {}
        in_price = model_prices.get('input')
        out_price = model_prices.get('output')
        inc_cost = None
        if in_price is not None and out_price is not None:
            # Incremental cost for this call
            inc_cost = (float(input_tokens or prompt_tokens) / 1000.0) * in_price + (float(output_tokens or completion_tokens) / 1000.0) * out_price
            self._usage['estimated_cost_usd'] = (self._usage.get('estimated_cost_usd', 0.0) or 0.0) + inc_cost

        # Per-provider accumulation
        p = provider_key.lower()
        byp = self._usage_by_provider.setdefault(p, {
            'requests': 0.0, 'prompt_tokens': 0.0, 'completion_tokens': 0.0,
            'input_tokens': 0.0, 'output_tokens': 0.0, 'estimated_cost_usd': 0.0
        })
        byp['requests'] += 1.0
        byp['prompt_tokens'] += float(prompt_tokens)
        byp['completion_tokens'] += float(completion_tokens)
        byp['input_tokens'] += float(input_tokens or prompt_tokens)
        byp['output_tokens'] += float(output_tokens or completion_tokens)
        if in_price is not None and out_price is not None:
            byp['estimated_cost_usd'] += (float(input_tokens or prompt_tokens) / 1000.0) * in_price + (float(output_tokens or completion_tokens) / 1000.0) * out_price

        # Update budget windows with inc_cost
        if inc_cost is not None:
            now = time.time()
            if now - self._hourly_started_at >= 3600:
                self._hourly_started_at = now
                self._hourly_cost_accum = 0.0
            if now - self._daily_started_at >= 86400:
                self._daily_started_at = now
                self._daily_cost_accum = 0.0
            self._hourly_cost_accum += float(inc_cost)
            self._daily_cost_accum += float(inc_cost)

    def get_usage_stats(self) -> Dict[str, float]:
        """Return aggregate token usage and estimated cost."""
        # Return a shallow copy to avoid external mutation
        out = dict(self._usage)
        # Back-compat alias
        if 'estimated_cost' not in out:
            out['estimated_cost'] = out.get('estimated_cost_usd', 0.0)
        # Include limits snapshot
        out['limits'] = dict(self._limits)
        # Include per-provider breakdown
        out['by_provider'] = {k: dict(v) for k, v in self._usage_by_provider.items()}
        return out

    def _merge_usage_stats(self, stats: Dict[str, Any]) -> None:
        # Merge per-provider breakdown and totals from another extractor stats dict
        byp = stats.get('by_provider') or {}
        for k, v in byp.items():
            dst = self._usage_by_provider.setdefault(k, {
                'requests': 0.0, 'prompt_tokens': 0.0, 'completion_tokens': 0.0,
                'input_tokens': 0.0, 'output_tokens': 0.0, 'estimated_cost_usd': 0.0
            })
            for fld in ('requests','prompt_tokens','completion_tokens','input_tokens','output_tokens','estimated_cost_usd'):
                dst[fld] = float(dst.get(fld, 0.0)) + float(v.get(fld, 0.0) or 0.0)
        # Merge overall totals
        for fld in ('prompt_tokens','completion_tokens','total_tokens','input_tokens','output_tokens','estimated_cost_usd','requests'):
            self._usage[fld] = float(self._usage.get(fld, 0.0) or 0.0) + float(stats.get(fld, 0.0) or 0.0)
