# Rules Maker 🤖📋

A powerful Python tool that scrapes documentation from the web and transforms it into AI coding assistant rules and workflows.

## 🎯 Overview

Rules Maker automates the process of creating coding rules and guidelines by:

1. **Scraping documentation** from various web sources
2. **Transforming content** into structured rules
3. **Generating outputs** for popular AI coding assistants like Cursor and Windsurf

## ✨ Features

- 🕷️ **Web Scraping**: Extract documentation from websites using advanced scraping techniques
- 🔄 **Content Transformation**: Convert raw documentation into structured rules
- 🎛️ **Multiple Output Formats**: Generate rules for different AI coding assistants
- 📚 **Documentation Sources**: Support for various documentation formats and websites
- ⚙️ **Configurable**: Customizable scraping and transformation rules

## 🛠️ Supported Output Formats

- **Cursor Rules** (.cursorrules files)
- **Windsurf Rules** (Windsurf-compatible rule formats)
- **Custom Workflows** (Configurable workflow definitions)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rules-maker.git
cd rules-maker

# Install dependencies
pip install -r requirements.txt

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.8+
- BeautifulSoup4
- Requests
- Selenium (for dynamic content)
- Additional scraping libraries (see requirements.txt)

## 🚀 Quick Start

```python
from rules_maker import DocumentationScraper, RuleTransformer

# Initialize scraper
scraper = DocumentationScraper()

# Scrape documentation
docs = scraper.scrape_url("https://docs.example.com/api")

# Transform to Cursor rules
transformer = RuleTransformer(output_format="cursor")
rules = transformer.transform(docs)

# Save rules
with open(".cursorrules", "w") as f:
    f.write(rules)
```

## 📖 Usage Examples

### Scraping API Documentation

```python
# Scrape REST API documentation
api_docs = scraper.scrape_url(
    "https://api.example.com/docs",
    scraper_type="api"
)

# Generate Cursor rules for API usage
cursor_rules = transformer.transform(
    api_docs, 
    template="api_guidelines"
)
```

### Creating Windsurf Workflows

```python
# Transform documentation to Windsurf workflows
windsurf_transformer = RuleTransformer(output_format="windsurf")
workflows = windsurf_transformer.create_workflows(docs)
```

### Batch Processing

```python
# Process multiple documentation sources
urls = [
    "https://docs.framework1.com",
    "https://docs.framework2.com",
    "https://api.service.com/docs"
]

for url in urls:
    docs = scraper.scrape_url(url)
    rules = transformer.transform(docs)
    # Save or process rules
```

## 🔍 Scraping Methods

The project supports multiple scraping approaches inspired by:

- **[mlscraper](https://github.com/lorey/mlscraper)**: Machine learning-based content extraction
- **[WebScraper](https://github.com/MLArtist/WebScraper)**: Advanced web scraping techniques
- **Custom scrapers**: Tailored for specific documentation formats

## 📁 Project Structure

```text
rules-maker/
├── src/
│   ├── scrapers/           # Web scraping modules
│   ├── transformers/       # Content transformation logic
│   ├── templates/          # Rule templates
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── examples/              # Usage examples
├── tests/                 # Test suite
└── docs/                  # Project documentation
```

## ⚙️ Configuration

Create a `config.yaml` file to customize scraping and transformation behavior:

```yaml
scraping:
  timeout: 30
  user_agent: "RulesMaker/1.0"
  rate_limit: 1.0
  
transformers:
  cursor:
    template: "cursor_rules_template.j2"
    max_rules: 50
  windsurf:
    template: "windsurf_template.j2"
    include_workflows: true

output:
  format_rules: true
  include_metadata: true
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/test_scrapers.py
python -m pytest tests/test_transformers.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/

# Run type checking
mypy src/
```

## 📝 Roadmap

- [ ] Support for more AI coding assistants
- [ ] GUI interface for non-technical users
- [ ] Integration with popular documentation platforms
- [ ] Real-time documentation monitoring
- [ ] Cloud-based processing
- [ ] Plugin system for custom transformers

## 🐛 Known Issues

- Some dynamic websites may require additional configuration
- Rate limiting may be needed for large-scale scraping
- Complex documentation structures may need manual template adjustments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [mlscraper](https://github.com/lorey/mlscraper) for ML-based content extraction inspiration
- [WebScraper](https://github.com/MLArtist/WebScraper) for advanced scraping techniques
- The open-source community for various scraping and parsing libraries

## 📞 Support

- 📧 Email: [support@rules-maker.com](mailto:support@rules-maker.com)
- 💬 Discord: [Join our community](https://discord.gg/rules-maker)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/rules-maker/issues)

---

Made with ❤️ for the developer community
