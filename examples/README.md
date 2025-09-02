# Examples

Short, copy‑pasteable commands to exercise the intelligent, interactive features.

## Interactive Analyze

- Analyze a local file and save JSON:
  - `PYTHONPATH=src python -m rules_maker.cli interactive analyze --file README.md -o tmp/analysis.json`

Expected output (truncated):
- Primary Technology: python
- Complexity Level: advanced
- Content Metrics (code examples, links, length)

## Natural Language Query

- Ask a question with technology context:
  - `PYTHONPATH=src python -m rules_maker.cli interactive query "What are best practices for React hooks?" --technologies react`

Expected output:
- A concise answer, 1–3 relevant sources, suggested rules, related topics, and confidence.

## Predict Rule Needs

- Predict rules from an analysis report:
  - `PYTHONPATH=src python -m rules_maker.cli interactive predict --project-analysis pipeline_report.json -o tmp/predictions.json`

Expected output:
- 3–6 rule predictions with priority, confidence, reason, and optional dependencies.

## User Insights

- Show user insights (requires prior interactive sessions or saved profile):
  - `PYTHONPATH=src python -m rules_maker.cli interactive insights --user-id default`

Notes:
- Add `--bedrock` plus `--model-id` and `--region` to enable Bedrock‑enhanced responses.
- Use `make venv && make install` or `pip install -e .` before running without `PYTHONPATH`.
