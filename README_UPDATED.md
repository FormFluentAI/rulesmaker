# 🤖📋 Rules Maker - AI Coding Assistant Rule Generator

Transform web documentation into professional AI coding assistant rules for Cursor, Windsurf, and other AI development tools.

## ✅ Status: Phase 1 Complete & Verified (September 1, 2025)

**Core functionality is working and tested!** Generate professional-grade Cursor and Windsurf rules from documentation content.

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

# Setup virtual environment
source rm/bin/activate  # or python -m venv venv && source venv/bin/activate

# Install core dependencies
pip install pydantic requests beautifulsoup4 click fake-useragent jinja2 aiohttp numpy
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

**Latest Test Results (September 1, 2025):**
```
✅ ScrapingResult model working correctly
✅ Cursor transformer working - generated 1880 chars
✅ Technology detection working
✅ Professional formatting working
✅ Windsurf transformer working - generated 1748 chars
✅ Workflow formatting working

🎉 Core rule generation functionality VERIFIED and WORKING!
```

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

## 🤝 Contributing

Rules Maker is designed to be extensible. Add new transformers, improve existing ones, or integrate with additional AI assistants.

## 📞 Support

- **Quick Issues**: Check STATUS.md for common setup problems
- **Examples**: See PHASE1_SUMMARY.md for tested working examples
- **Implementation Details**: Review docs/PHASE1_COMPLETE.md for technical details


---

**Ready to generate professional AI coding assistant rules?** Follow the Quick Start guide above and you'll have working Cursor and Windsurf rules in 5 minutes! 🚀
