# CLI Commands Reference

This page lists the main CLI commands available in this project, with quick examples for common tasks. You can run the CLI either directly from the repo or after installing the package.

## Overview

The rules-maker CLI provides comprehensive cursor rules generation with ML-powered enhancements, learning integration, and intelligent quality assessment. All commands support cursor rules knowledge integration and can generate high-quality `.cursorrules` files.

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
  - Scrape a documentation URL and generate rules with cursor rules knowledge integration.
  - **New Features**: ML enhancement, learning feedback collection, cursor rules validation
  - Example: `rules-maker scrape https://docs.example.com --format cursor -o rules/example.mdc --ml-enhanced --learning-feedback`

- `rules-maker batch <urls_file> [options]`
  - Scrape multiple URLs from a newline-delimited file with enhanced processing.
  - **New Features**: ML quality assessment, intelligent clustering, cursor rules validation
  - Example: `rules-maker batch urls.txt --output-dir rules/batch --format cursor --parallel --quality-assessment`

- `rules-maker pipeline --rules <rules_file> [--content-file <file>] [--events <events.json|jsonl>] [--output <report.json>]`
  - Run the learning pipeline and emit a JSON report with enhanced analytics.
  - **New Features**: Cursor rules quality assessment, learning insights, performance analytics
  - Example: `rules-maker pipeline --rules ruleset.yaml --events usage.jsonl --output pipeline_report.json --cursor-rules-validation`

- `rules-maker templates [--template <name>]`
  - List templates or print a specific template’s content.
  - Example: `rules-maker templates --template cursor_rule.j2`

- `rules-maker setup [--check-deps] [--install-deps]`
  - Check and/or install dependencies (non-destructive apart from pip installs when requested).
  - Examples:
    - `rules-maker setup --check-deps`
    - `rules-maker setup --install-deps`

## Enhanced Core Commands

### Cursor Rules Generation
- `rules-maker scrape <url> --format cursor --ml-enhanced --learning-feedback`
  - Generate cursor rules with ML enhancement and learning integration
  - **Features**: Cursor rules validation, quality assessment, learning feedback collection
  - Example: `rules-maker scrape https://docs.example.com --format cursor -o rules/example.mdc --ml-enhanced --learning-feedback`

- `rules-maker batch <urls_file> --format cursor --quality-assessment --cursor-rules-validation`
  - Batch process with cursor rules validation and quality assessment
  - **Features**: Intelligent clustering, cursor rules compliance scoring, technology-specific insights
  - Example: `rules-maker batch urls.txt --output-dir rules/batch --format cursor --quality-assessment`

### Cursor Rules Validation
- `rules-maker validate-cursor-rules <rules_dir> [--output <report.json>]`
  - Validate cursor rules structure and quality
  - **Features**: Structure validation, pattern matching, quality scoring, recommendations
  - Example: `rules-maker validate-cursor-rules rules/ --output validation_report.json`

- `rules-maker enhance-cursor-rules <rules_dir> [--output <enhanced_dir>]`
  - Enhance cursor rules with missing sections and best practices
  - **Features**: Automatic section addition, technology-specific guidelines, quality improvement
  - Example: `rules-maker enhance-cursor-rules rules/ --output enhanced_rules/`

## ML Commands

- Group: `rules-maker ml`
  - `rules-maker ml train <training_data_dir> --model-output <path> [--test-split <float>] [--checkpoint/--no-checkpoint]`
    - Train an ML extractor on labeled examples with cursor rules knowledge.
    - **New Features**: Cursor rules pattern recognition, quality prediction models
    - Example: `rules-maker ml train data/training --model-output models/extractor.bin --cursor-rules-mode`
  - `rules-maker ml test <model_path> <test_url>`
    - Test a trained model against a page URL with cursor rules validation.
    - **New Features**: Cursor rules compliance testing, quality assessment
    - Example: `rules-maker ml test models/extractor.bin https://docs.example.com/page --validate-cursor-rules`

## Bedrock Commands

- Group: `rules-maker bedrock`
  - `rules-maker bedrock validate [--model-id <id>] [--region <aws-region>] [--credentials-csv <path>] [--show-config]`
    - Validate credentials and print endpoint/usage/identity; optional config preview.
  - `rules-maker bedrock batch <sources_file> --output-dir <dir> [--model-id <id>] [--parallel-requests N] [--cost-limit USD] [--quality-threshold F] [--formats cursor|windsurf ...] [--dry-run]`
    - Bedrock-enhanced batch processing from a JSON/YAML sources file.

## ML Batch Processing

- Group: `rules-maker ml-batch` (hyphenated)
  - `rules-maker ml-batch frameworks [--output-dir <dir>] [--bedrock] [--config <file>] [--formats ...] [--max-concurrent N] [--quality-threshold F] [--dry-run]`
    - Process popular web frameworks with cursor rules knowledge integration.
    - **New Features**: Cursor rules validation, technology-specific guidelines, quality assessment
    - Example: `rules-maker ml-batch frameworks --output-dir rules/frameworks --format cursor --cursor-rules-validation`
  - `rules-maker ml-batch cloud [--output-dir <dir>] [--bedrock] [--config <file>] [--formats ...] [--max-concurrent N] [--quality-threshold F] [--dry-run]`
    - Process cloud platform docs with enhanced cursor rules generation.
    - **New Features**: Cloud-specific cursor rules, security guidelines, best practices
    - Example: `rules-maker ml-batch cloud --output-dir rules/cloud --format cursor --quality-threshold 0.8`
  - `rules-maker ml-batch custom <sources_file> --output-dir <dir> [--bedrock] [--config <file>] [--formats ...] [--max-concurrent N] [--quality-threshold F] [--dry-run]`
    - Process custom documentation sources with intelligent cursor rules generation.
    - **New Features**: Custom technology detection, adaptive guidelines, quality optimization
    - Example: `rules-maker ml-batch custom sources.json --output-dir rules/custom --format cursor --ml-enhanced`

## Config Management

- Group: `rules-maker config`
  - `rules-maker config init [--output config/ml_batch_config.yaml] [--template minimal|standard|advanced] [--force]`
    - Create a starter ML batch configuration file.
  - `rules-maker config validate <config_file>`
    - Validate a configuration and show summary/errors.

## Learning System

- Group: `rules-maker learning`
  - `rules-maker learning feedback --rule-id <id> --signal-type <usage_success|user_rating|effectiveness|relevance> --value <0..1> [--context JSON] [--source <name>]`
    - Record feedback signals for rule improvement with cursor rules context.
    - **New Features**: Cursor rules quality feedback, technology-specific learning, adaptive improvement
    - Example: `rules-maker learning feedback --rule-id "python_guidelines" --signal-type effectiveness --value 0.9 --context '{"technology": "python", "cursor_rules_compliance": 0.85}'`
  - `rules-maker learning analyze <rules_dir> [--output <path>] [--format json|yaml|md]`
    - Analyze learning patterns and rule effectiveness with cursor rules insights.
    - **New Features**: Cursor rules compliance analysis, technology-specific insights, quality trends
    - Example: `rules-maker learning analyze rules/ --output analysis.json --format json --cursor-rules-analysis`

## Quality Assessment

- Group: `rules-maker quality`
  - `rules-maker quality assess <rules_dir> [--format cursor|windsurf|all] [--output <path>] [--threshold F]`
    - Assess rule quality and emit a report + summary with cursor rules validation.
    - **New Features**: Cursor rules structure validation, compliance scoring, technology-specific quality metrics
    - Example: `rules-maker quality assess rules/ --format cursor --output quality_report.json --threshold 0.8 --cursor-rules-validation`
  - `rules-maker quality cluster <processing_results_dir> [--output <path>] [--min-coherence F]`
    - Analyze clusters from prior batch results with cursor rules insights.
    - **New Features**: Cursor rules clustering, technology-based grouping, quality coherence analysis
    - Example: `rules-maker quality cluster results/ --output cluster_analysis.json --min-coherence 0.6 --cursor-rules-clustering`

## Analytics

- Group: `rules-maker analytics`
  - `rules-maker analytics insights <processing_results_dir> [--output <path>] [--format json|yaml|md] [--include-recommendations]`
    - Generate consolidated insights from batch processing outputs with cursor rules analytics.
    - **New Features**: Cursor rules compliance analytics, technology-specific insights, quality trend analysis
    - Example: `rules-maker analytics insights results/ --output insights.json --format json --include-recommendations --cursor-rules-analytics`

## Interactive (Intelligent) Commands

- Group: `rules-maker interactive`
  - `rules-maker interactive session [--project-type <str>] [--technologies <csv>] [--experience-level <level>] [--session-id <id>] [--bedrock]`
    - Guided session to generate personalized cursor rules with learning integration.
    - **New Features**: Cursor rules knowledge integration, technology-specific guidance, adaptive learning
    - Example: `rules-maker interactive session --project-type "web-app" --technologies "react,typescript" --experience-level "intermediate" --bedrock`
  - `rules-maker interactive analyze [<content>] [--url <u>] [--file <path>] [--output <path>] [--bedrock]`
    - Semantic analysis of content/doc pages with cursor rules insights.
    - **New Features**: Cursor rules compliance analysis, technology detection, quality assessment
    - Example: `rules-maker interactive analyze --url https://docs.example.com --output analysis.json --cursor-rules-analysis`
  - `rules-maker interactive query [<question>] [--technologies <csv>] [--project-type <str>] [--experience-level <level>] [--bedrock]`
    - Ask questions with optional project context and cursor rules knowledge.
    - **New Features**: Cursor rules context, technology-specific answers, best practices guidance
    - Example: `rules-maker interactive query "How to structure React components?" --technologies "react,typescript" --project-type "web-app"`
  - `rules-maker interactive predict [--project-analysis <path>] [--user-id <id>] [--current-rules <path>] [--output <path>] [--bedrock]`
    - Predict rule needs based on your project and usage with cursor rules optimization.
    - **New Features**: Cursor rules prediction, technology-specific recommendations, quality optimization
    - Example: `rules-maker interactive predict --project-analysis project.json --current-rules rules/ --output predictions.json`
  - `rules-maker interactive insights [--user-id <id>] [--global-insights] [--output <path>]`
    - User or global usage insights and recommendations with cursor rules analytics.
    - **New Features**: Cursor rules usage analytics, technology trends, quality insights
    - Example: `rules-maker interactive insights --user-id "user123" --global-insights --output insights.json`

## Cursor Rules Specific Commands

### Cursor Rules Generation
- `rules-maker generate-cursor-rules <source> [--technology <tech>] [--output <path>]`
  - Generate cursor rules from various sources with technology-specific knowledge
  - **Features**: Technology detection, best practices integration, quality validation
  - Example: `rules-maker generate-cursor-rules https://docs.example.com --technology python --output rules/python_guidelines.mdc`

### Cursor Rules Validation
- `rules-maker validate-cursor-structure <rules_file> [--fix] [--output <path>]`
  - Validate and optionally fix cursor rules structure
  - **Features**: Structure validation, automatic fixes, quality scoring
  - Example: `rules-maker validate-cursor-structure rules/my_rules.mdc --fix --output fixed_rules.mdc`

### Cursor Rules Enhancement
- `rules-maker enhance-cursor-rules <rules_dir> [--technology <tech>] [--output <enhanced_dir>]`
  - Enhance cursor rules with missing sections and technology-specific guidelines
  - **Features**: Section addition, technology guidelines, quality improvement
  - Example: `rules-maker enhance-cursor-rules rules/ --technology react --output enhanced_rules/`

## Tips

- Use `--help` on any command or group to see all options (e.g., `rules-maker quality --help`, `rules-maker bedrock batch --help`).
- For development without install, prefix commands with `PYTHONPATH=src python -m rules_maker.cli ...`.
- For AWS Bedrock features, prefer `rm-setup`, set credentials via env or `--credentials-csv`, and validate with `rules-maker bedrock validate`.
- For cursor rules generation, use `--format cursor` to ensure proper `.cursorrules` format output.
- Enable ML enhancement with `--ml-enhanced` for better quality cursor rules generation.
- Use `--cursor-rules-validation` to validate and improve cursor rules structure automatically.

