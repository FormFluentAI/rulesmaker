# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rules Maker is a Python tool that transforms web documentation into professional AI coding assistant rules for Cursor, Windsurf, and other AI development tools. The core functionality is **working and production-ready** with sophisticated rule generation capabilities.

## Development Commands

### Environment Setup
```bash
# Option 1: Use Makefile (Recommended)
make setup-cu129  # Creates venv, installs all deps + PyTorch with CUDA
make venv         # Creates virtual environment only
make install      # Installs from requirements.txt files
make test         # Runs pytest
make clean-venv   # Removes virtual environment

# Option 2: Manual setup with existing rm/ environment
source rm/bin/activate

# Option 3: Standard pip installation
pip install -e .                    # Install in development mode
pip install -e ".[dev]"             # Install with dev dependencies
pip install -e ".[all]"             # Install all optional dependencies
pip install -e ".[bedrock]"         # Install with Bedrock support
pip install -e ".[openai,anthropic]" # Install with LLM providers

# Alternative: From requirements files
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### AWS Bedrock Integration Commands

**Setup Bedrock with Nova Lite (NEW - VERIFIED WORKING):**
```bash
# Install Bedrock dependencies
pip install boto3

# Test Bedrock integration
PYTHONPATH=src python examples/simple_bedrock_test.py

# Full Bedrock examples  
PYTHONPATH=src python examples/bedrock_rules_example.py
```

**Generate Rules with Bedrock (VERIFIED WORKING):**
```bash
PYTHONPATH=src python -c "
from rules_maker.bedrock_integration import quick_cursor_rules

# Your documentation content
docs = '''FastAPI is a modern web framework for building APIs with Python.
Key features: Fast performance, type hints, automatic validation.
Best practices: Use Pydantic models, implement error handling, follow REST principles.'''

# Generate Cursor rules with Bedrock Nova Lite
rules = quick_cursor_rules(docs)
print(rules)
"
```

**Enhanced LLM-Powered Generation:**
```bash
PYTHONPATH=src python -c "
import asyncio
from rules_maker.bedrock_integration import quick_enhanced_cursor_rules

async def generate():
    docs = '''Your documentation content here...'''
    rules = await quick_enhanced_cursor_rules(docs)
    print(rules)
    
asyncio.run(generate())
"
```

### Core Functionality Commands

**Generate Cursor Rules (VERIFIED WORKING):**
```bash
PYTHONPATH=src python -c "
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.models import ScrapingResult

result = ScrapingResult(
    url='https://fastapi.tiangolo.com/',
    title='FastAPI Documentation',
    content='FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. Create API endpoints, handle request/response validation, automatic interactive API documentation with Swagger UI, dependency injection system, async/await support.'
)

transformer = CursorRuleTransformer()
rules = transformer.transform([result])
print(rules)
"
```

**Generate Windsurf Rules (VERIFIED WORKING):**
```bash
PYTHONPATH=src python -c "
from rules_maker.transformers.windsurf_transformer import WindsurfRuleTransformer
from rules_maker.models import ScrapingResult

result = ScrapingResult(
    url='https://fastapi.tiangolo.com/',
    title='FastAPI Documentation',
    content='FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. Create API endpoints, handle request/response validation, automatic interactive API documentation with Swagger UI, dependency injection system, async/await support.'
)

transformer = WindsurfRuleTransformer()
rules = transformer.transform([result])
print(rules)
"
```

### Testing Commands

```bash
# Using Makefile (Recommended)
make test

# Manual pytest commands
PYTHONPATH=src pytest                                    # Run all tests
PYTHONPATH=src pytest --cov=rules_maker                  # Run with coverage
PYTHONPATH=src pytest tests/test_phase1.py               # Run specific test file
PYTHONPATH=src pytest tests/test_phase1.py::test_cursor_transformer  # Run single test

# Test specific components
PYTHONPATH=src pytest tests/test_bedrock_concurrency.py  # Bedrock integration tests
PYTHONPATH=src pytest tests/test_learning_engine.py      # Learning pipeline tests
PYTHONPATH=src pytest tests/test_templates.py            # Template system tests
```

### Linting and Formatting

```bash
# Format code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/rules_maker/

# Run all quality checks
black src/ tests/ && flake8 src/ tests/ && mypy src/rules_maker/
```

### CLI Commands

```bash
# Using installed package entry points (if installed with pip install -e .)
rules-maker scrape https://docs.python.org/3/
rm-setup --check-deps
rm-doctor  # Check system health

# Direct module execution (Note: requires PYTHONPATH=src)
PYTHONPATH=src python -m rules_maker.cli scrape https://docs.python.org/3/
PYTHONPATH=src python -m rules_maker.cli scrape https://docs.python.org/3/ --async-scrape
PYTHONPATH=src python -m rules_maker.cli scrape https://docs.python.org/3/ --deep --max-pages 5
PYTHONPATH=src python -m rules_maker.cli scrape https://docs.python.org/3/ --format windsurf
PYTHONPATH=src python -m rules_maker.cli setup --check-deps
```

## Architecture Overview

### Core Pipeline Architecture

The Rules Maker follows a **4-stage transformation pipeline**:

1. **Scraping** → Documentation content extraction from URLs
2. **Extraction** → Content analysis and feature extraction
3. **Transformation** → Rule generation using detected technology patterns
4. **Templating** → Professional formatting via Jinja2 templates

### Key Architecture Patterns

**Strategy Pattern Implementation**: All major components (scrapers, extractors, transformers) use strategy pattern for extensibility. New implementations can be added by inheriting from base classes.

**Async-First Design**: Core operations support async processing with `AsyncDocumentationScraper` providing 5x+ performance improvements over synchronous scraping.

**Type-Safe Pipeline**: Complete Pydantic model coverage ensures data validation throughout the transformation pipeline - from `ScrapingResult` → `ContentSection` → `Rule` → final output.

**Technology Detection Engine**: Regex-based scoring system automatically identifies frameworks (React, Vue, Python, etc.) from documentation content and URLs, enabling technology-specific rule customization.

### Module Architecture

**Core Module Architecture:**

- **scrapers/**: Multi-strategy scraping (base, async_documentation_scraper, adaptive_documentation_scraper)  
- **transformers/**: Rule generation engines (cursor_transformer, windsurf_transformer, workflow_transformer)
- **extractors/**: Content extraction (ml_extractor, llm_extractor, structured_extractor)
- **models.py**: Comprehensive Pydantic data models with type safety
- **templates/**: Jinja2 template system with engine.py and rule templates
- **processors/**: Content processing pipeline (documentation_processor, code_processor, api_processor)
- **strategies/**: Strategy pattern implementations including learning_strategy
- **filters/**: Content filtering and validation (quality_filter, relevance_filter, duplicate_filter)
- **learning/**: ML pipeline components (engine.py, pattern_analyzer.py, usage_tracker.py)
- **utils/**: Utility functions and credential management

### Key Data Models

**ScrapingResult**: Primary data structure for scraped content
```python
class ScrapingResult(BaseModel):
    url: HttpUrl
    title: str
    content: str
    sections: List[ContentSection] = []
    documentation_type: DocumentationType = DocumentationType.UNKNOWN
    status: ScrapingStatus = ScrapingStatus.PENDING
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = {}
```

**ContentSection**: Hierarchical content organization
```python
class ContentSection(BaseModel):
    title: str
    content: str
    level: int = 1
    url: Optional[HttpUrl] = None
    metadata: Dict[str, Any] = {}
    subsections: List['ContentSection'] = []
```

### Rule Generation System

**Two-Phase Rule Generation**:

1. **Analysis Phase**: Technology detection using weighted regex scoring across 12+ frameworks (Python, JavaScript, React, Vue, Angular, Next.js, etc.)
2. **Generation Phase**: Professional rule formatting using technology-specific templates


**Rule Format Specifications**:

- **Cursor Rules**: Expert role definitions, critical instructions (🚨 NEVER/ALWAYS), 6 structured sections (principles, code style, tech guidelines, error handling, performance, critical instructions)
- **Windsurf Rules**: 4-phase development workflow (Analysis → Implementation → Testing → Review), quality gates with checkboxes (✅), measurable criteria


**Template System**: Jinja2 templates in `templates/templates/*.j2` provide professional formatting. Technology detection drives template variable population for framework-specific customization.

**Content Processing Pipeline**: `ScrapingResult` → Technology Detection → Template Variable Population → Jinja2 Rendering → Professional Rule Output

**Performance**: CursorRuleTransformer averages 1880+ chars, WindsurfRuleTransformer averages 1748+ chars, both with industry-standard professional formatting.

## AWS Bedrock Integration

### Quick Start with Bedrock

**Prerequisites:**

1. AWS Bedrock credentials in `docs/plans/bedrock-long-term-api-key.csv`
2. Install boto3: `pip install boto3`
3. Ensure access to `amazon.nova-lite-v1:0` model


**Simple Usage:**
```python
from rules_maker.bedrock_integration import BedrockRulesMaker

# Initialize with automatic credential loading
maker = BedrockRulesMaker(model_id="amazon.nova-lite-v1:0")

# Generate rules from documentation
documentation = "Your framework documentation content..."
cursor_rules = maker.generate_cursor_rules(documentation)
windsurf_rules = maker.generate_windsurf_rules(documentation)

# View usage stats
stats = maker.get_usage_stats()
print(f"Cost: ${stats['estimated_cost_usd']:.4f}")
```

**Enhanced LLM-Powered Generation:**
```python
import asyncio
from rules_maker.bedrock_integration import BedrockRulesMaker

async def generate_enhanced():
    maker = BedrockRulesMaker(model_id="amazon.nova-lite-v1:0")
    
    # Enhanced generation with LLM analysis
    enhanced_rules = await maker.generate_enhanced_cursor_rules(documentation)
    await maker.close()
    return enhanced_rules

rules = asyncio.run(generate_enhanced())
```

**Supported Bedrock Models:**

- `amazon.nova-lite-v1:0` (recommended - fast & cost-effective)
- `amazon.nova-micro-v1:0` (ultra low-cost)
- `amazon.nova-pro-v1:0` (high capability)
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`

## Working Examples

### Python API Usage (Preferred Method)

```python
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.transformers.windsurf_transformer import WindsurfRuleTransformer
from rules_maker.models import ScrapingResult

# Create documentation content
doc_result = ScrapingResult(
    url='https://your-docs.com/',
    title='Your Framework Documentation',
    content='Your documentation content here - API guides, tutorials, best practices...'
)

# Generate Cursor rules
cursor_transformer = CursorRuleTransformer()
cursor_rules = cursor_transformer.transform([doc_result])

# Generate Windsurf rules
windsurf_transformer = WindsurfRuleTransformer()
windsurf_rules = windsurf_transformer.transform([doc_result])
```

### Async Scraping Usage

```python
from rules_maker.scrapers.async_documentation_scraper import AsyncDocumentationScraper

async with AsyncDocumentationScraper() as scraper:
    # High-performance concurrent scraping
    results = await scraper.scrape_documentation_site(url, max_pages=10)
    
    # Transform results to rules
    transformer = CursorRuleTransformer()
    rules = transformer.transform(results)
```

## Development Notes

### Code Quality Standards

- **Type Safety**: 100% Pydantic model coverage with comprehensive validation
- **Error Handling**: Comprehensive exception management with graceful degradation
- **Testing**: Use pytest with asyncio support for async components
- **Code Style**: Follow PEP 8, use black formatter, flake8 linting
- **Documentation**: Document all public APIs and complex logic

### Key Patterns

**Strategy Pattern**: Used throughout for scrapers, extractors, and transformers - enables easy extension with new implementations.

**Async-First Design**: All scraping operations support both sync and async patterns for performance.

**Configuration-Driven**: YAML and programmatic configuration with Pydantic validation.

**Template-Based**: Jinja2 templates enable easy customization of rule formats.

### Performance Considerations

- **Async Operations**: 5x+ performance improvement with concurrent processing
- **Memory Efficiency**: Optimized for processing large documentation sites
- **Rate Limiting**: Built-in rate limiting and session management
- **Caching**: Content extraction results cached to avoid reprocessing

### Extension Points

**Adding New Rule Formats**: Create new transformer class inheriting from `RuleTransformer`

**Adding New Scrapers**: Implement `BaseScraper` interface with strategy pattern

**Custom Templates**: Add Jinja2 templates to `templates/` directory

**Content Extraction**: Extend extractors for domain-specific content processing

## Common Issues

### Import Path Issues
If CLI commands fail with import errors, always use `PYTHONPATH=src` prefix:
```bash
PYTHONPATH=src python -m rules_maker.cli [command]
```

### Dependency Issues
Check and install missing dependencies:
```bash
PYTHONPATH=src python -m rules_maker.cli setup --check-deps
```

### Virtual Environment
Ensure you're in the correct virtual environment:
```bash
# Option 1: Use Makefile to create and activate
make venv && source .venv/bin/activate

# Option 2: Use existing rm/ environment
source rm/bin/activate

# Option 3: Create manually
python -m venv venv && source venv/bin/activate
```

### PyTorch and CUDA
Check PyTorch CUDA availability for ML components:
```bash
make torch-info  # Shows PyTorch version and CUDA availability
```

## Production Status & Key Limitations

**✅ Working Components:**

- Core rule generation (CursorRuleTransformer, WindsurfRuleTransformer)
- Technology detection engine with 12+ framework support
- Pydantic model validation and type safety
- Async scraping architecture with 5x+ performance improvement
- Professional rule formatting matching industry standards


**⚠️ Known Limitations:**

- **CLI Robustness**: Command-line interface requires `PYTHONPATH=src` prefix for imports
- **Dependency Management**: Manual dependency installation required (no automated setup)
- **Template Extensions**: Limited to Cursor/Windsurf formats (extensible via strategy pattern)


**🔧 Recommended Usage Pattern:**
Python API usage is strongly recommended over CLI for production applications due to import path complexity.

## ML-Powered Batch Processing (NEW)

### Advanced Batch Processing System

**Process 100+ Documentation Sources with ML Intelligence:**
```bash
# Process popular frameworks with self-improving ML pipeline
PYTHONPATH=src python examples/batch_processing_demo.py --bedrock

# Custom batch processing with intelligent clustering
PYTHONPATH=src python -c "
from rules_maker.batch_processor import MLBatchProcessor, DocumentationSource
import asyncio

sources = [
    DocumentationSource('https://reactjs.org/docs/', 'React', 'javascript', 'react', priority=5),
    DocumentationSource('https://fastapi.tiangolo.com/', 'FastAPI', 'python', 'fastapi', priority=5),
    # Add 100+ sources...
]

async def process():
    processor = MLBatchProcessor(
        bedrock_config={'model_id': 'amazon.nova-lite-v1:0'},
        output_dir='rules/intelligent_batch',
        quality_threshold=0.7
    )
    result = await processor.process_documentation_batch(sources)
    print(f'Generated {result.total_rules_generated} rules with {result.quality_metrics}')

asyncio.run(process())
"
```

**Self-Improving Feedback System:**
```bash
# Demonstrate self-improving ML engine with quality scoring
PYTHONPATH=src python -c "
from rules_maker.learning.self_improving_engine import SelfImprovingEngine
import asyncio

async def demo_feedback():
    engine = SelfImprovingEngine(quality_threshold=0.7)
    
    # Collect feedback signals
    await engine.collect_feedback_signal('rule_123', 'usage_success', 0.8, source='user')
    await engine.collect_feedback_signal('rule_123', 'user_rating', 0.9, source='user')
    
    # Self-awarding system automatically boosts quality scores
    awards = await engine.self_award_quality_improvements(rules, batch_performance)
    print(f'Self-awarded improvements: {awards}')

asyncio.run(demo_feedback())
"
```

### Key ML Features

- **Intelligent Clustering**: TF-IDF vectorization + K-means for semantic rule grouping
- **Self-Awarding System**: Automatic quality score boosting based on performance trends
- **Quality Prediction**: ML models predict rule effectiveness before deployment
- **Adaptive Thresholds**: Dynamic quality standards based on system performance
- **Coherence Optimization**: Cosine similarity analysis for logically coherent rule sets

### Comprehensive Testing Commands

```bash
# Test ML batch processing system
PYTHONPATH=src pytest tests/test_batch_processing.py -v

# Test self-improving engine
PYTHONPATH=src pytest tests/test_batch_processing.py::TestSelfImprovingEngine -v

# Integration tests
PYTHONPATH=src pytest tests/test_batch_processing.py::TestIntegration -v
```

**🚨 Critical Implementation Notes:**

- Always use `PYTHONPATH=src` prefix for CLI commands
- Async operations provide 5x+ performance improvements over synchronous alternatives
- Technology detection is regex-based and may require content preprocessing for optimal results
- ML batch processing requires scikit-learn and numpy for clustering algorithms
- Self-improving engine automatically saves state and learns from feedback patterns