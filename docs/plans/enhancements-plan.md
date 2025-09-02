# Enhancements Plan — AWS Bedrock Integration and Rule Generation

This plan outlines concrete, incremental enhancements to strengthen production readiness, align documentation with code, improve resilience and observability, and expand capabilities. Items are grouped by phases, each with clear acceptance criteria.

## Objectives
- Align public docs and examples with actual API and behavior
- Improve resilience (retry/backoff, error mapping) and telemetry
- Expand supported models and accurate cost tracking
- Increase test coverage for Bedrock paths and credentials parsing
- Provide clearer CLI/config UX for Bedrock usage

## Phase 2 — Stability, Docs Alignment, Cost Tracking
Status: Completed (35 tests passed in conda env `rulescraper`)

### 2.1 Documentation Corrections and Consistency
Status: Completed

- Fix function names and usage in `docs/IMPLEMENTATION_PROGRESS.md`:
  - Replace `generate_cursor_rules_quick` with `quick_cursor_rules`.
  - Remove `track_usage=True` from `generate_cursor_rules(...)` examples.
  - Use `estimated_cost_usd` instead of `estimated_cost`.
- Clarify “Working Models”: Nova Lite/region support vs. Claude via Bedrock, and cross-inference restrictions.


Applied changes

- Updated code examples to import/use `quick_cursor_rules`.
- Removed unused `track_usage` argument from examples.
- Standardized cost key to `estimated_cost_usd`.
- Rewrote Working Models section with Nova/Claude + cross-inference notes.


Acceptance

- Running the example snippets as written (with `PYTHONPATH=src`) succeeds (assuming creds).
- No stale function names or mismatched keys remain.

### 2.2 Cost Map Expansion and Back-Compat Key
Status: Completed

- Extend price map in `LLMContentExtractor` to include:
  - `anthropic.claude-3-5-sonnet-20240620-v1:0`
  - `anthropic.claude-3-5-haiku-20241022-v1:0` (or current) if used
  - Region/profile variant from examples: `eu.amazon.nova-lite-v1:0`
- Add `estimated_cost` alias to `get_usage_stats()` output (canonical remains `estimated_cost_usd`).
- Improve pricing lookup for Bedrock ARNs/variants by substring-matching known model keys.


Acceptance

- Cost estimates increment for the above model IDs in mocked responses.
- Accessing either `estimated_cost` or `estimated_cost_usd` returns a float and stays in sync.

### 2.3 Retry/Backoff for Bedrock Throttling
Status: Completed

- In `LLMContentExtractor._bedrock_request`:
  - Implemented exponential backoff with jitter (defaults: 250ms → 2s, 3 attempts) for throttling/transient errors (`ThrottlingException`, `TooManyRequestsException`, `ServiceUnavailableException`, 5xx, and botocore connection timeouts).
  - Configurable via `LLMConfig` (`retry_max_attempts`, `retry_base_ms`, `retry_max_ms`) and env (`BEDROCK_RETRY_MAX_ATTEMPTS`, `BEDROCK_RETRY_BASE_MS`, `BEDROCK_RETRY_MAX_MS`).
- Logs structured retry metadata: `bedrock_retry attempt=<n> delay_ms=<ms> code=<err> status=<http>`.


Acceptance

- Transient errors trigger retries with increasing delays; final success returns content; final failure surfaces a clear runtime error with code/message.
- Unit test stubs simulate throttling twice then success; request returns parsed content and records usage once.
- Logged messages include retry attempt count and backoff duration.

### 2.4 Basic Rate-Limit Monitoring Metrics
Status: Completed

- Track counters in extractor for:
  - `retries`, `throttle_events`, `last_error_code`, `last_retry_delay_ms`.
- Expose via `get_usage_stats()` under a `limits` sub-dict.


Acceptance

- After simulated throttling, `limits.throttle_events > 0` and `limits.retries` reflects attempts.

### 2.5 Tests for Credentials Parsing and Validation
Status: Completed

- Added tests for `CredentialManager.load_bedrock_credentials_from_csv` handling:
  - Composite `name:base64` format
  - Pure base64 blob
  - Access-key-only (warn path)
- Added tests for Bedrock validation using stubbed boto3 Session/client.


Acceptance

- Tests run without network and assert parsing/validation behavior.

## Phase 3 — Developer UX, Performance, and CLI

### 3.1 CLI Improvements
Status: Completed

- Add CLI flags to choose provider `--provider bedrock`, set `--model-id`, `--region`, and `--credentials-csv`.
- Add `rules-maker bedrock validate` command to run credentials setup + connection test and print endpoint, usage, and identity summary.

Implementation

- Updated entry `rules-maker` (module: `rules_maker.cli:main`) to accept global flags: `--provider`, `--model-id`, `--region`, `--credentials-csv`.
- `scrape` supports Bedrock via per-command flags (`--llm-provider bedrock`, `--llm-model`, `--region`, `--credentials-csv`) or via globals.
- New command group `bedrock` with `validate`:
  - Loads credentials from CSV when provided, otherwise uses env/`~/.aws`.
  - Calls Bedrock `converse` to verify access and prints: endpoint URL, token usage, and AWS identity (account, user, ARN).
  - Attempts an end-to-end request via `BedrockRulesMaker`; throttling is handled gracefully with clear messaging.

Usage Examples

- Help: `PYTHONPATH=src python -m rules_maker.cli --help`
- Validate (Claude 3.5 Sonnet):
  - `PYTHONPATH=src python -m rules_maker.cli bedrock validate --region us-east-1 --model-id anthropic.claude-3-5-sonnet-20240620-v1:0`
  - With CSV: `... bedrock validate --credentials-csv docs/plans/bedrock-long-term-api-key.csv --region us-east-1 --model-id anthropic.claude-3-5-sonnet-20240620-v1:0`
- Adaptive scrape using Bedrock (globals):
  - `PYTHONPATH=src python -m rules_maker.cli --provider bedrock --model-id anthropic.claude-3-5-sonnet-20240620-v1:0 --region us-east-1 --credentials-csv docs/plans/bedrock-long-term-api-key.csv scrape https://example.com --adaptive -o out.cursor`
- Adaptive scrape (per-command flags):
  - `PYTHONPATH=src python -m rules_maker.cli scrape https://example.com --adaptive --llm-provider bedrock --llm-model anthropic.claude-3-5-sonnet-20240620-v1:0 --region us-east-1 --credentials-csv docs/plans/bedrock-long-term-api-key.csv -o out.cursor`

Acceptance

- Verified end-to-end with `PYTHONPATH=src`:
  - `bedrock validate` prints success status, endpoint, usage, and identity (when entitled; otherwise prints clear failure).
  - `scrape` runs with Bedrock provider flags; reports stats and saves output.

### 3.2 Config Surface
Status: Completed

- Extended `config.example.yaml` with `bedrock:` section:
  - `model_id`, `region`, `credentials_csv`, `timeout`, `concurrency`, `retry:{max_attempts,base_ms,max_ms}`.
- Wired optional config dict + env fallback into `BedrockRulesMaker` and `LLMContentExtractor`.
- Added CLI visibility flag: `rules-maker bedrock validate --show-config` prints effective model/region/timeout/concurrency/retry.

Implementation

- File: `config.example.yaml` — new `bedrock:` block with defaults.
- Files: `src/rules_maker/bedrock_integration.py`, `src/rules_maker/extractors/llm_extractor.py` — accept `config={'bedrock': {...}}` and read env:
  - `BEDROCK_MODEL_ID`, `AWS_REGION`/`BEDROCK_REGION`, `BEDROCK_TIMEOUT`, `BEDROCK_MAX_CONCURRENCY`, `BEDROCK_RETRY_MAX_ATTEMPTS`, `BEDROCK_RETRY_BASE_MS`, `BEDROCK_RETRY_MAX_MS`.
- File: `src/rules_maker/cli.py` — loads `-c config.yaml` and applies `bedrock` to adaptive scraper and `bedrock validate`; `--show-config` displays effective settings.

Acceptance

- Changing YAML/env updates behavior (e.g., region/timeout/retry) and is reflected by `bedrock validate --show-config` and in logs.

### 3.3 Concurrency and Safety
Status: Completed (concurrency); batching deferred

- Added concurrency cap for LLM calls via semaphore in `LLMContentExtractor._make_llm_request`.
  - Configurable from YAML (`bedrock.concurrency`), `LLMConfig.max_concurrency`, or env `BEDROCK_MAX_CONCURRENCY`.
- Improved Bedrock safety/backoff:
  - Exponential backoff + jitter with support for `Retry-After` header; explicit botocore timeouts.
  - Structured logs: `bedrock_retry attempt=<n> delay_ms=<ms> code=<err> status=<http>`.
- Tracking: `limits.retries`, `limits.throttle_events`, `limits.last_retry_delay_ms`, `limits.last_error_code` in `get_usage_stats()`.
- Optional batching for multi-page inputs is out of scope for this change and will be revisited alongside token budget modeling.

Acceptance

- Tests demonstrate: with N parallel requests, concurrent calls never exceed the cap; throttle events are lower under capped concurrency vs. higher concurrency.
- Files: `tests/test_bedrock_concurrency.py` (no network; stubs simulate latency/throttling).

## Phase 4 — Capability Expansion

### 4.1 Multi-Provider Fallback
Status: Completed (feature-flagged)

- Added optional fallback chain: Bedrock → OpenAI/Anthropic/HF/Local.
- Controlled via YAML (`bedrock.fallback.enabled`, `bedrock.fallback.providers`) or env (`FALLBACK_ENABLED`, `FALLBACK_PROVIDERS`).
- Provider credentials/models are read from `providers:` in YAML or env vars.
- Preserves cost/usage breakdown per provider in `get_usage_stats()['by_provider']`.

Implementation

- `config.example.yaml`: added `bedrock.fallback` and top-level `providers` section with example keys/models.
- `LLMContentExtractor`:
  - Parses fallback settings and provider credentials.
  - `_make_llm_request` tries primary provider first, then iterates fallbacks on exception (no network in tests).
  - Usage tracking now includes a `by_provider` breakdown with tokens, requests, and estimated cost.
- `BedrockRulesMaker` and `AdaptiveDocumentationScraper` pass loaded YAML config to extractor so fallback works in both CLI flows.

Acceptance

- Simulated Bedrock outage triggers fallback; rules still generated; usage stats split by provider.
- Tests: `tests/test_fallback.py` validates the fallback to `local` path and checks `by_provider` accounting.

### 4.2 Enhanced Telemetry and Logging
Status: Completed

- Structured logs (optional JSON) for request lifecycle with: provider, model, region, timeout, prompt_len, prompt_hash, latency_ms, usage deltas (tokens/cost), and retry/throttle deltas.
- Start/end/attempt-error events: `llm_request_start`, `llm_request_end`, `llm_attempt_error`.
- Redaction option hides raw prompts and includes only SHA-256 `prompt_hash` and `prompt_len`.
- Configurable via YAML `telemetry:` or env vars.

Implementation

- `config.example.yaml`: added `telemetry: { json, redact_prompts }`.
- `LLMContentExtractor`:
  - Emits structured events; JSON when `telemetry.json: true` or `RULES_MAKER_LOG_JSON=1`.
  - Redacts prompts by default (configurable); includes SHA-256 hash and length only.
  - Records per-call deltas for tokens/cost and retry/throttle counters in logs.

Config

- YAML:
  - `telemetry.json: true|false`
  - `telemetry.redact_prompts: true|false`
- Env:
  - `RULES_MAKER_LOG_JSON=1` enables JSON logs
  - `RULES_MAKER_REDACT_PROMPTS=0` disables redaction

Acceptance

- Unit tests validate presence of fields and redaction:
  - `tests/test_telemetry.py` asserts `llm_request_start` has `prompt_hash`/`prompt_len` and no raw prompt; `llm_request_end` includes `latency_ms`, `usage_delta`, and `limits_delta`.

### 4.3 Caching and Cost Controls
Status: Completed

- Added in-memory and file cache for LLM request results, keyed by provider|model|system|prompt hash.
- Cache hits bypass provider calls and incur no cost; writes stored under `~/.cache/rules_maker/llm` by default.
- Budget guardrails for hourly/daily cost caps; cache hits allowed even when caps reached; new provider calls are blocked with clear reason.

Implementation

- `config.example.yaml`: added `cache: {enabled, dir}` and `budget: {hourly_usd, daily_usd}`.
- `LLMContentExtractor`:
  - Cache: `_build_cache_key`, `_cache_get`, `_cache_put`; events `llm_cache_hit`, `llm_cache_store`.
  - Budget: windowed counters (hourly/daily) and preflight block on cache miss; event `llm_budget_block`.
  - Env overrides: `RULES_MAKER_CACHE_ENABLED`, `RULES_MAKER_CACHE_DIR`, `RULES_MAKER_BUDGET_HOURLY_USD`, `RULES_MAKER_BUDGET_DAILY_USD`.

Acceptance

- Tests: `tests/test_cache_budget.py` verifies file cache avoids calls and cost, and budget guard blocks provider calls with error.

## Security & Compliance
Status: Completed (credentials sanitation); IAM policy doc pending

- Rotated example credentials: `docs/plans/bedrock-long-term-api-key.csv` now contains placeholders and a warning comment.
- Next: add a minimal IAM policy example for Bedrock runtime in docs (separate PR).

Acceptance

- Repo contains only example credentials; no live keys present.

## Work Items Checklist (Condensed)
- [x] Docs: Fix names/keys in `docs/IMPLEMENTATION_PROGRESS.md`
- [x] Extractor: Add retry/backoff + config
- [x] Extractor: Expand price map + `estimated_cost` alias
- [x] Extractor: Limits metrics in `get_usage_stats()`
- [x] Tests: Credentials CSV parsing cases
- [x] Tests: Bedrock `converse` success/throttle via Stubber
- [x] CLI: `bedrock validate` + flags
- [x] Config: YAML + env wiring
- [x] Concurrency cap in extractor
- [x] Fallback: Multi-provider chain (feature-flagged)
- [x] Tests: Fallback to local and provider usage breakdown
- [x] Security: Replace/rotate CSV creds

## Milestones & Rough Timeline
- Phase 2: 1–2 days (docs, cost, retry, tests)
- Phase 3: 2–3 days (CLI, config, concurrency)
- Phase 4: 3–5 days (fallback, telemetry, caching, budgets)

## Notes
- We will not enable network calls in unit tests; use botocore `Stubber` for Bedrock.
- Keep changes incremental and backward compatible; avoid breaking public symbols.

## Test Execution (Reference)
- Env: conda activate `rulescraper`
- Install test deps: `pip install -r requirements.txt -r requirements-dev.txt`
- Run full suite: `PYTHONPATH=src pytest -q`
- Current status: 35 passed, 6 warnings
