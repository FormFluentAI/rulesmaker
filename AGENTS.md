# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/rules_maker/` (modular packages: `scrapers/`, `extractors/`, `transformers/`, `utils/`, `nlp/`, `interactive/`).
- CLI entrypoints (installed): `rules-maker`, `rm-setup`, `rm-doctor` → see `pyproject.toml`.
- Tests: `tests/` (pytest), examples: `examples/`, docs: `docs/`, scripts: `scripts/`.
- Config: `config/`, `config.example.yaml`, environment files: `.env`, `.env.example`.

## Build, Test, and Development Commands
- `make venv`: Create `.venv` and upgrade build tools.
- `make install`: Install runtime + dev requirements into `.venv`.
- `make setup-cu129`: Setup venv, deps, and PyTorch CUDA 12.9 wheels.
- `make torch-info`: Print Torch/CUDA availability.
- `make test` or `.venv/bin/pytest -q`: Run tests.
- Local run without install: `PYTHONPATH=src python -m rules_maker.cli --help`.
- After editable install: `pip install -e .` then `rules-maker --help`.

## Coding Style & Naming Conventions
- Python 3.9+; src-layout package `rules_maker` (snake_case for modules, CapWords for classes, lower_snake_case for functions/vars).
- Formatting: Black (`line-length = 88`), linting: Flake8, typing: mypy (no untyped defs).
- Keep public APIs minimal and documented; prefer small, composable modules under the relevant subpackage.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio`; tests live in `tests/`, named `test_*.py`.
- Run with coverage: `pytest --cov=src --cov-report=term-missing`.
- Prefer fast, deterministic tests; use fixtures in `tests/fixtures/`. For async code, mark with `@pytest.mark.asyncio`.

## Commit & Pull Request Guidelines
- Commit style: Conventional Commits (e.g., `feat: ...`, `chore: ...`, `fix: ...`) as seen in history.
- PRs should include: concise description, linked issues, before/after notes, and any relevant screenshots or CLI output.
- Add tests for new behavior and update docs (`README.md`, `ARCHITECTURE.md`) when interfaces change.

## Security & Configuration Tips
- Never commit secrets. Use `.env.local` for machine-specific values and `.env.example` for templates.
- Validate configs with `rm-doctor` and start from `config.example.yaml`.
- When using GPU, prefer `make setup-cu129` and verify with `make torch-info`.
