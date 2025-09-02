# 🤖📋 Rules Maker - AI Coding Assistant Rule Generator

Transform web documentation into professional AI coding assistant rules for Cursor, Windsurf, and other AI development tools.

## ✅ Status: Phase 1 Complete & Verified (September 1, 2025)

**Core functionality is working and tested!** Generate professional-grade Cursor and Windsurf rules from documentation content.

Note: Pydantic V2 compliant (model_validate/model_dump). CLI imports are now lazy to avoid optional scraping deps unless used.

### 🎯 What Works Right Now

- ✅ **Cursor Rules Generation**: Professional developer guidelines with expert roles and critical instructions
- ✅ **Windsurf Workflow Rules**: Development process rules with quality gates and workflow phases  
- ✅ **Technology Detection**: Automatic identification of Python, JavaScript, React, FastAPI, etc.
- ✅ **Professional Formatting**: Industry-standard structure matching docs.cursor.com patterns
- ✅ **Data Validation**: Robust content processing with Pydantic models

## 🚀 Quick Start (5 Minutes)

### 1. Setup Environment
```bash
# Clone and navigate
git clone <repository-url>
cd rules-maker

# One-command setup with venv + deps
make setup-cu129  # installs deps and torch/torchvision from cu129 channel

# Use manually instead:
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt -r requirements-dev.txt
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### Make Targets (one-liners)
```bash
# Create venv and upgrade build tools
make venv

# Install runtime + dev requirements into .venv
make install

# Setup with PyTorch CUDA 12.9 wheels (recommended for NVIDIA GPUs)
make setup-cu129

# Verify Torch/CUDA availability
make torch-info

# Run the test suite
make test

# Remove local virtualenv
make clean-venv
```

### 2. Generate Cursor Rules (Verified Working)
```bash
PYTHONPATH=src python -c "
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.models import ScrapingResult

# Create content from your documentation
result = ScrapingResult(
    url='https://your-docs.com/',
    title='Your Framework Documentation',
    content='Your documentation content here - API guides, tutorials, best practices...'
)

# Generate professional Cursor rules
transformer = CursorRuleTransformer()
rules = transformer.transform([result])
print(rules)
"
```

### 3. Generate Windsurf Rules (Verified Working)
```bash
PYTHONPATH=src python -c "
from rules_maker.transformers.windsurf_transformer import WindsurfRuleTransformer
from rules_maker.models import ScrapingResult

# Same content, different rule format
result = ScrapingResult(
    url='https://your-docs.com/',
    title='Your Framework Documentation',
    content='Your documentation content here...'
)

# Generate workflow-focused Windsurf rules
transformer = WindsurfRuleTransformer()
rules = transformer.transform([result])
print(rules)
"
```

## 📋 Example Output

### Cursor Rules
```markdown
You are an expert in Python and modern software development.

## Key Principles
- Implement proper HTTP status codes and error responses
- Follow RESTful principles for API design
- Include comprehensive request/response validation

## 🚨 Critical Instructions

**NEVER:**
- Ignore error handling or edge cases
- Use deprecated APIs or methods

**ALWAYS:**
- Follow security best practices
- Validate all input parameters
```

### Windsurf Rules
```markdown
# Windsurf Workflow Rules

## Expert Role
You are a Python development expert focusing on clean architecture.

## Development Workflow
1. **Analysis Phase** - Understand requirements thoroughly
2. **Implementation Phase** - Write clean, well-documented code
3. **Testing Phase** - Write comprehensive tests
4. **Review Phase** - Code review for quality and standards

## Quality Gates
✅ **Code Quality** - Linting passes without errors
✅ **Testing** - All tests pass with good coverage
```

## 🎯 Key Features

### Professional Rule Generation
- **Expert Role Definitions**: Context-aware developer identities
- **Technology Detection**: Automatic framework/language identification
- **Critical Instructions**: Clear do's and don'ts with visual formatting
- **Industry Standards**: Based on analysis of docs.cursor.com and cursor.directory

### Advanced Capabilities  
- **Multi-Format Output**: Cursor, Windsurf, and custom rule formats
- **Content Analysis**: Intelligent extraction of best practices and patterns
- **Workflow Integration**: Development process guidelines and quality gates
- **Extensible Architecture**: Easy to add new rule formats and transformers

## 🔧 Advanced Setup (Optional)

### For ML-Enhanced Extraction
```bash
pip install scikit-learn sentence-transformers nltk
```

### For LLM Integration
```bash
pip install openai anthropic
export OPENAI_API_KEY="your-key-here"
```

### For Full Web Scraping
```bash
pip install lxml httpx aiofiles more-itertools python-dotenv rich jsonschema
```

### CUDA 12.9 (cu129) PyTorch
- For NVIDIA 50xx GPUs (e.g., 5070 Ti) using CUDA 12.9-compatible drivers, install PyTorch from the cu129 wheel index.
- Use the Makefile target:
  - `make setup-cu129`
- Or run manually inside your venv:
  - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129`
- Verify:
  - `make torch-info` or
  - `python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

Note: Recent torch wheels may report `+cu128` while still being compatible with CUDA 12.9 drivers (minor-version compatibility). This is expected; `torchvision` shows `+cu129`.

## 🖥️ CLI Commands

See docs/cli-commands.md for full usage. Entrypoints: `rules-maker`, `rm-setup`, `rm-doctor`.

## 📁 Project Structure

```
rules-maker/
├── src/rules_maker/
│   ├── transformers/           # Rule generation engines
│   │   ├── cursor_transformer.py      # ✅ Cursor rules (working)
│   │   └── windsurf_transformer.py    # ✅ Windsurf rules (working)
│   ├── models.py              # ✅ Data models (working)
│   ├── scrapers/              # Web scraping components
│   ├── extractors/            # Content extraction tools
│   └── cli.py                 # Command-line interface
├── examples/                  # Usage examples
├── tests/                     # Test suite
├── docs/                      # Documentation
│   ├── PHASE1_COMPLETE.md     # ✅ Completion status
│   └── plans/phase-01.md      # ✅ Implementation details
├── cursor_rules.md            # ✅ Generated Cursor rules example
├── PHASE1_SUMMARY.md          # ✅ Current status overview
└── STATUS.md                  # ✅ Quick usage guide
```

## 🧪 Testing & Verification

Run tests:
```bash
source .venv/bin/activate  # if not already active
make test                  # or: pytest -q
```

**Latest Test Results (Local):**
```
✅ ScrapingResult model working correctly
✅ Cursor transformer working - generated 1880 chars
✅ Technology detection working
✅ Professional formatting working
✅ Windsurf transformer working - generated 1748 chars
✅ Workflow formatting working

🎉 Core rule generation functionality VERIFIED and WORKING!
```

## 🧭 Interactive & Intelligent CLI

Use the interactive command group for guided analysis, Q&A, and predictions.

- Analyze content (save JSON):
  - `PYTHONPATH=src python -m rules_maker.cli interactive analyze --file README.md -o tmp/analysis.json`
- Ask a question (NLP):
  - `PYTHONPATH=src python -m rules_maker.cli interactive query "What are best practices for React hooks?" --technologies react`
- Predict rule needs from a report:
  - `PYTHONPATH=src python -m rules_maker.cli interactive predict --project-analysis pipeline_report.json -o tmp/predictions.json`
- View insights (requires prior sessions or a saved profile):
  - `PYTHONPATH=src python -m rules_maker.cli interactive insights --user-id default`

Tip: Add `--bedrock` (and/or global `--provider bedrock --model-id amazon.nova-lite-v1:0 --region us-east-1`) for enhanced responses.

## 🧠 ML Batch Processing (Dry‑Run Friendly)

Batch commands live under `ml-batch`. Dry‑run works even if ML deps are not installed.

- Popular frameworks (dry run):
  - `PYTHONPATH=src python -m rules_maker.cli ml-batch frameworks --dry-run`
- Cloud platforms (dry run):
  - `PYTHONPATH=src python -m rules_maker.cli ml-batch cloud --dry-run`
- Custom sources (dry run):
  - `PYTHONPATH=src python -m rules_maker.cli ml-batch custom sources.json --dry-run`

Install ML extras for full processing: `pip install -r requirements-dev.txt` (scikit-learn, numpy, etc.).

## 📈 Learning & Quality Tools

- Provide feedback (0.0–1.0):
  - `PYTHONPATH=src python -m rules_maker.cli learning feedback --rule-id R1 --signal-type user_rating --value 0.8`
- Analyze a rules directory (empty dirs supported):
  - `PYTHONPATH=src python -m rules_maker.cli learning analyze rules_out --format json -o tmp/learning_report.json`
- Assess quality of generated rules:
  - `PYTHONPATH=src python -m rules_maker.cli quality assess rules_out --format all -o tmp/quality_report.json`

## 🔗 More Examples

See examples/README.md for copy‑pasteable commands and expected outputs.

## 🎊 What's Next

### Immediate Use Cases
1. **Generate Cursor Rules**: Create professional coding guidelines from your documentation
2. **Create Windsurf Workflows**: Build development process rules for your team
3. **Customize Output**: Extend transformers for your specific AI assistant needs
4. **Integration**: Use in CI/CD pipelines to auto-generate rules from documentation

### Phase 2 Roadmap  
- **Automated Setup**: One-command dependency installation
- **CLI Improvements**: Direct command-line usage without import path issues
- **More Formats**: Support for additional AI assistants
- **Learning System**: Adaptive rule generation based on user feedback

## 🧠 Learning Engine (Phase 2)

Record usage events and optimize your rules using the new LearningEngine.

### Record Usage Events
```python
from rules_maker.models import Rule, RuleType
from rules_maker.learning.models import GeneratedRule, UsageEvent

# Your existing rule
rule = Rule(
    id="naming-001",
    title="Use descriptive names",
    description="Prefer descriptive variable and function names.",
    content="...",
    type=RuleType.BEST_PRACTICE,
    priority=2,
    confidence_score=0.4,
)

# Collect usage signals over time
events = [
    UsageEvent(rule_id="naming-001", success=True, feedback_score=0.5, context={"section": "naming"}),
    UsageEvent(rule_id="naming-001", success=True, context={"section": "naming"}),
    UsageEvent(rule_id="naming-001", success=False, context={"section": "examples"}),
]

generated = GeneratedRule(rule=rule, usage_events=events)
```

### Analyze and Optimize
```python
from rules_maker.learning import LearningEngine

# Provide a rule_map so the engine can update your actual Rule objects
engine = LearningEngine(config={"rule_map": {rule.id: rule}})

insights = engine.analyze_usage_patterns([generated])
optimized = engine.optimize_rules(insights)

print("Top rules:", insights.top_rules)
print("Underperforming:", insights.underperforming_rules)
for change in optimized.changes:
    print(change.rule_id, change.change_type, change.before, "=>", change.after)

# Updated Rule objects are in optimized.rules
updated_rule = next(r for r in optimized.rules if r.id == rule.id)
print("New priority:", updated_rule.priority)
print("New confidence:", updated_rule.confidence_score)
```

### Validate Improvements
```python
qm = engine.validate_improvements(before=rule, after=updated_rule)
print("Overall improvement score:", qm.overall_score)
```

### Simple A/B Aggregation (optional)
```python
from rules_maker.learning.models import ABTest

experiment = ABTest(experiment_key="exp-1", rule_id=rule.id, variants=[])
# Tag events with variants via `context={"variant": "A"}` or "B"
ab_events = [
    UsageEvent(rule_id=rule.id, success=True, context={"variant": "A"}),
    UsageEvent(rule_id=rule.id, success=False, context={"variant": "B"}),
]
result = engine.summarize_ab_test(experiment, ab_events)
print("Winner variant:", result.winner_variant)
```

## 🤝 Contributing

Rules Maker is designed to be extensible. Add new transformers, improve existing ones, or integrate with additional AI assistants.

## 📞 Support

- **Quick Issues**: Check STATUS.md for common setup problems
- **Examples**: See PHASE1_SUMMARY.md for tested working examples
- **Implementation Details**: Review docs/PHASE1_COMPLETE.md for technical details


---

**Ready to generate professional AI coding assistant rules?** Follow the Quick Start guide above and you'll have working Cursor and Windsurf rules in 5 minutes! 🚀
