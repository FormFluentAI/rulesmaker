# CLI Commands Reference

This page lists the main CLI commands available in this project, with quick examples for common tasks. You can run the CLI either directly from the repo or after installing the package.

## Getting Help

- Show top-level help:
  - Local (without install): `PYTHONPATH=src python -m rules_maker.cli --help`
  - After install: `rules-maker --help`

## Installed Entrypoints

- `rules-maker`: Main CLI with subcommands (listed below)
- `rm-setup`: Prints recommended install steps and extras
- `rm-doctor`: Quick environment checks (imports, optional deps)

Examples:
- `rm-setup`
- `rm-doctor`

## Core Commands (rules-maker)

- `rules-maker scrape <url> [options]`
  - Scrape a documentation URL and generate rules.
  - Example: `rules-maker scrape https://docs.example.com --format cursor -o rules/example.mdc`

- `rules-maker batch <urls_file> [options]`
  - Scrape multiple URLs from a newline-delimited file.
  - Example: `rules-maker batch urls.txt --output-dir rules/batch --format windsurf --parallel`

- `rules-maker pipeline --rules <rules_file> [--content-file <file>] [--events <events.json|jsonl>] [--output <report.json>]`
  - Run the learning pipeline and emit a JSON report.
  - Example: `rules-maker pipeline --rules ruleset.yaml --events usage.jsonl --output pipeline_report.json`

- `rules-maker templates [--template <name>]`
  - List templates or print a specific template’s content.
  - Example: `rules-maker templates --template cursor_rule.j2`

- `rules-maker setup [--check-deps] [--install-deps]`
  - Check and/or install dependencies (non-destructive apart from pip installs when requested).
  - Examples:
    - `rules-maker setup --check-deps`
    - `rules-maker setup --install-deps`

## ML Commands

- Group: `rules-maker ml`
  - `rules-maker ml train <training_data_dir> --model-output <path> [--test-split <float>] [--checkpoint/--no-checkpoint]`
    - Train an ML extractor on labeled examples.
    - Example: `rules-maker ml train data/training --model-output models/extractor.bin`
  - `rules-maker ml test <model_path> <test_url>`
    - Test a trained model against a page URL.
    - Example: `rules-maker ml test models/extractor.bin https://docs.example.com/page`

## Bedrock Commands

- Group: `rules-maker bedrock`
  - `rules-maker bedrock validate [--model-id <id>] [--region <aws-region>] [--credentials-csv <path>] [--show-config]`
    - Validate credentials and print endpoint/usage/identity; optional config preview.
  - `rules-maker bedrock batch <sources_file> --output-dir <dir> [--model-id <id>] [--parallel-requests N] [--cost-limit USD] [--quality-threshold F] [--formats cursor|windsurf ...] [--dry-run]`
    - Bedrock-enhanced batch processing from a JSON/YAML sources file.

## ML Batch Processing

- Group: `rules-maker ml-batch` (hyphenated)
  - `rules-maker ml-batch frameworks [--output-dir <dir>] [--bedrock] [--config <file>] [--formats ...] [--max-concurrent N] [--quality-threshold F] [--dry-run]`
    - Process popular web frameworks.
  - `rules-maker ml-batch cloud [--output-dir <dir>] [--bedrock] [--config <file>] [--formats ...] [--max-concurrent N] [--quality-threshold F] [--dry-run]`
    - Process cloud platform docs.
  - `rules-maker ml-batch custom <sources_file> --output-dir <dir> [--bedrock] [--config <file>] [--formats ...] [--max-concurrent N] [--quality-threshold F] [--dry-run]`
    - Process custom documentation sources.

## Config Management

- Group: `rules-maker config`
  - `rules-maker config init [--output config/ml_batch_config.yaml] [--template minimal|standard|advanced] [--force]`
    - Create a starter ML batch configuration file.
  - `rules-maker config validate <config_file>`
    - Validate a configuration and show summary/errors.

## Learning System

- Group: `rules-maker learning`
  - `rules-maker learning feedback --rule-id <id> --signal-type <usage_success|user_rating|effectiveness|relevance> --value <0..1> [--context JSON] [--source <name>]`
    - Record feedback signals for rule improvement.
  - `rules-maker learning analyze <rules_dir> [--output <path>] [--format json|yaml|md]`
    - Analyze learning patterns and rule effectiveness.

## Quality Assessment

- Group: `rules-maker quality`
  - `rules-maker quality assess <rules_dir> [--format cursor|windsurf|all] [--output <path>] [--threshold F]`
    - Assess rule quality and emit a report + summary.
  - `rules-maker quality cluster <processing_results_dir> [--output <path>] [--min-coherence F]`
    - Analyze clusters from prior batch results.

## Analytics

- Group: `rules-maker analytics`
  - `rules-maker analytics insights <processing_results_dir> [--output <path>] [--format json|yaml|md] [--include-recommendations]`
    - Generate consolidated insights from batch processing outputs.

## Interactive (Intelligent) Commands

- Group: `rules-maker interactive`
  - `rules-maker interactive session [--project-type <str>] [--technologies <csv>] [--experience-level <level>] [--session-id <id>] [--bedrock]`
    - Guided session to generate personalized rules.
  - `rules-maker interactive analyze [<content>] [--url <u>] [--file <path>] [--output <path>] [--bedrock]`
    - Semantic analysis of content/doc pages.
  - `rules-maker interactive query [<question>] [--technologies <csv>] [--project-type <str>] [--experience-level <level>] [--bedrock]`
    - Ask questions with optional project context.
  - `rules-maker interactive predict [--project-analysis <path>] [--user-id <id>] [--current-rules <path>] [--output <path>] [--bedrock]`
    - Predict rule needs based on your project and usage.
  - `rules-maker interactive insights [--user-id <id>] [--global-insights] [--output <path>]`
    - User or global usage insights and recommendations.

## Tips

- Use `--help` on any command or group to see all options (e.g., `rules-maker quality --help`, `rules-maker bedrock batch --help`).
- For development without install, prefix commands with `PYTHONPATH=src python -m rules_maker.cli ...`.
- For AWS Bedrock features, prefer `rm-setup`, set credentials via env or `--credentials-csv`, and validate with `rules-maker bedrock validate`.

