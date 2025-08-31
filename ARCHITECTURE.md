# Rules Maker - Project Structure Overview

## 📁 Project Organization

Based on analysis of the `c4ai` (Crawl4AI) and `mlscraper` projects, I've created an improved and more structured architecture for Rules Maker that combines the best aspects of both projects.

## 🏗️ Architecture Improvements

### **From Crawl4AI (c4ai):**
- ✅ **Modular Design**: Clear separation of concerns with dedicated modules
- ✅ **Async Support**: Built for high-performance async operations
- ✅ **Comprehensive Configuration**: Rich configuration system via Pydantic models
- ✅ **Strategy Pattern**: Pluggable strategies for different extraction methods
- ✅ **CLI Interface**: Professional command-line interface with Click
- ✅ **Template System**: Jinja2-based template engine for rule generation

### **From MLScraper:**
- ✅ **Machine Learning Focus**: ML-based content extraction capabilities
- ✅ **Training-Based Approach**: Learn from examples to improve extraction
- ✅ **Lightweight Core**: Simple, focused API design
- ✅ **Pattern Recognition**: Automatic pattern detection in documentation

### **Our Improvements:**
- ✅ **AI-Focused**: Specifically designed for AI coding assistant rules
- ✅ **Multiple Output Formats**: Support for Cursor, Windsurf, and custom formats
- ✅ **Documentation-Aware**: Specialized for documentation website patterns
- ✅ **Better Type Safety**: Full Pydantic models with proper validation
- ✅ **Enhanced Template System**: Rich template engine with custom filters

## 📂 Directory Structure

```
rules-maker/
├── src/
│   └── rules_maker/
│       ├── __init__.py           # Main package exports
│       ├── version.py            # Version information
│       ├── models.py             # Pydantic data models
│       ├── utils.py              # Utility functions
│       ├── cli.py                # Command-line interface
│       │
│       ├── scrapers/             # Web scraping components
│       │   ├── __init__.py
│       │   ├── base.py           # Base scraper class
│       │   ├── documentation_scraper.py    # Sync documentation scraper
│       │   ├── async_documentation_scraper.py  # Async scraper
│       │   └── adaptive_documentation_scraper.py  # ML-enhanced scraper
│       │
│       ├── extractors/           # Content extraction strategies
│       │   ├── __init__.py
│       │   ├── base.py           # Base extractor class
│       │   ├── ml_extractor.py   # Machine learning extractor
│       │   ├── llm_extractor.py  # LLM-powered extractor
│       │   └── structured_extractor.py  # Rule-based extractor
│       │
│       ├── transformers/         # Content transformation
│       │   ├── __init__.py
│       │   ├── base.py           # Base transformer class
│       │   ├── rule_transformer.py      # Generic rule transformer
│       │   ├── cursor_transformer.py    # Cursor rules format
│       │   ├── windsurf_transformer.py  # Windsurf rules format
│       │   └── workflow_transformer.py  # Workflow generator
│       │
│       ├── templates/            # Template engine and templates
│       │   ├── __init__.py
│       │   ├── engine.py         # Jinja2 template engine
│       │   └── templates/        # Template files
│       │       ├── cursor_rules.j2      # Cursor rules template
│       │       ├── windsurf_rules.j2    # Windsurf rules template
│       │       └── workflow.j2          # Workflow template
│       │
│       ├── processors/           # Content processors
│       │   ├── __init__.py
│       │   └── base.py           # Base processor (placeholder)
│       │
│       ├── strategies/           # Strategy pattern implementations
│       │   ├── __init__.py
│       │   └── base.py           # Base strategies (placeholder)
│       │
│       └── filters/              # Content filtering
│           ├── __init__.py
│           └── base.py           # Base filters (placeholder)
│
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_scrapers.py          # Basic scraper tests
│
├── examples/                     # Usage examples
│   ├── basic_usage.py            # Basic Python API usage
│   └── cli_usage.py              # CLI usage examples
│
├── docs/                         # Documentation (future)
├── config.example.yaml           # Example configuration
├── requirements.txt              # Core dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Modern Python packaging
├── LICENSE                      # MIT license
└── README.md                    # This file
```

## 🔧 Key Components

### **1. Scrapers Module**
- **Base Scraper**: Abstract base class with common functionality
- **Documentation Scraper**: Optimized for documentation websites
- **Async Scraper**: High-performance async scraping (future)
- **Adaptive Scraper**: ML-enhanced pattern recognition (future)

### **2. Extractors Module** 
- **Content Extractor**: Base class for content extraction
- **ML Extractor**: Machine learning-based extraction
- **LLM Extractor**: Large Language Model powered extraction
- **Structured Extractor**: Rule-based structured extraction

### **3. Transformers Module**
- **Rule Transformer**: Convert scraped content to rules
- **Cursor Transformer**: Generate .cursorrules files
- **Windsurf Transformer**: Generate Windsurf-compatible rules
- **Workflow Transformer**: Create workflow definitions

### **4. Templates Module**
- **Template Engine**: Jinja2-based rendering engine
- **Rule Templates**: Templates for different rule formats
- **Custom Filters**: Template filters for text processing

## 🚀 Usage Patterns

### **Python API**
```python
from rules_maker import DocumentationScraper, CursorRuleTransformer

# Configure and scrape
scraper = DocumentationScraper()
results = scraper.scrape_documentation_site("https://docs.example.com")

# Transform to rules
transformer = CursorRuleTransformer()
rules = transformer.transform(results)

# Save rules
with open(".cursorrules", "w") as f:
    f.write(rules)
```

### **CLI Interface**
```bash
# Simple scraping
rules-maker scrape https://docs.example.com --output .cursorrules

# Deep scraping with configuration
rules-maker scrape https://docs.example.com \
  --deep \
  --max-pages 50 \
  --format cursor \
  --output ./rules.txt

# Batch processing
rules-maker batch urls.txt --output-dir ./rules --format windsurf
```

## 🔄 Comparison with Source Projects

| Feature | Crawl4AI | MLScraper | Rules Maker |
|---------|-----------|-----------|-------------|
| **Async Support** | ✅ Advanced | ❌ None | ✅ Planned |
| **ML Integration** | 🔶 Basic | ✅ Core Feature | ✅ Enhanced |
| **Documentation Focus** | 🔶 General Web | ❌ General | ✅ Specialized |
| **AI Rules Output** | ❌ Raw Content | ❌ Data Only | ✅ AI-Ready Rules |
| **Template System** | ❌ None | ❌ None | ✅ Jinja2-based |
| **CLI Interface** | ✅ Comprehensive | ❌ Limited | ✅ Feature-rich |
| **Type Safety** | ✅ Pydantic | 🔶 Basic | ✅ Full Pydantic |
| **Strategy Pattern** | ✅ Extensive | ❌ Limited | ✅ Simplified |

## 🎯 Next Steps

1. **Complete Core Implementation**
   - Implement async scraper
   - Add ML-based extractors
   - Create LLM integration

2. **Enhanced Features**
   - Add more output formats
   - Implement learning strategies
   - Create GUI interface

3. **Testing & Documentation**
   - Comprehensive test suite
   - API documentation
   - Usage tutorials

4. **Performance Optimization**
   - Async/await optimization
   - Caching strategies
   - Memory management

This architecture provides a solid foundation that's both scalable and maintainable, incorporating the best practices from both reference projects while adding significant improvements for AI coding assistant integration.
