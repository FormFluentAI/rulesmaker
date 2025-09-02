# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-02

- Pydantic V2 compliance: migrated to `model_validate`/`model_dump` in CLI and transformers; removed deprecation warnings during pipeline runs.
- Lazy CLI imports: avoid importing scrapers/LLM dependencies at module import time; commands import their dependencies on demand.
- Added `pipeline` CLI fixtures for local testing: sample rules, events (JSON/JSONL), and content files under `tests/fixtures/pipeline/`.

## [0.1.x] - 2025-09-01

- Phase 1 foundation complete (Cursor/Windsurf transformers, core models, initial CLI).

