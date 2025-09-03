# Next.js Documentation Pipeline Guide

This guide explains how to use the comprehensive Next.js documentation processing pipeline that imports, categorizes, formats, and learns from Next.js documentation to generate intelligent cursor rules.

## 🚀 Overview

The Next.js pipeline is a sophisticated system that:

1. **Imports** Next.js documentation from multiple sources
2. **Categorizes** content using intelligent pattern matching and ML
3. **Formats** documentation into proper `.cursor/rules` format
4. **Learns** from user interactions and feedback for continuous improvement

## 📋 Prerequisites

- Python 3.9+
- Required dependencies (see `requirements.txt`)
- Access to Next.js documentation websites
- Optional: AWS Bedrock for ML-enhanced processing

## 🛠️ Installation

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   rules-maker --help
   rules-maker nextjs --help
   ```

## 🎯 Quick Start

### Basic Usage

Process Next.js official documentation and generate cursor rules:

```bash
rules-maker nextjs process --sources official --output .cursor/rules/nextjs
```

### Advanced Usage

Process all sources with ML enhancement and learning:

```bash
rules-maker nextjs process \
  --sources all \
  --ml-enhanced \
  --learning-enabled \
  --quality-threshold 0.8 \
  --format mdc \
  --verbose
```

## 📚 Available Sources

The pipeline supports three source groups:

### 1. Official Sources (`--sources official`)
- Next.js Official Documentation
- App Router Documentation
- Pages Router Documentation
- API Reference

### 2. Community Sources (`--sources community`)
- Next.js Learn Course
- Examples and Tutorials
- Community Resources

### 3. Ecosystem Sources (`--sources ecosystem`)
- Vercel Next.js Guide
- Tailwind CSS with Next.js
- TypeScript Integration Guide

## 🧠 Intelligent Categorization

The pipeline automatically categorizes content into Next.js-specific categories:

### Core Categories
- **Routing**: App Router, Pages Router, dynamic routes, middleware
- **Data Fetching**: Server Components, Client Components, SSR, SSG, ISR
- **Styling**: CSS Modules, Tailwind CSS, styled-components
- **Deployment**: Vercel, Docker, static export
- **Performance**: Image optimization, font optimization, bundle analysis
- **API Routes**: Route handlers, serverless functions
- **Configuration**: next.config.js, environment variables
- **Testing**: Jest, Playwright, Cypress
- **Security**: Authentication, CORS, security headers

### Advanced Categories
- **Migration**: Version upgrades, breaking changes
- **Advanced Patterns**: Parallel routes, intercepting routes, streaming
- **Troubleshooting**: Common issues, debugging guides

## 📝 Output Formats

The pipeline supports multiple output formats:

### MDC Format (Default)
```markdown
---
description: Next.js routing patterns for App Router and Pages Router
globs: ["**/app/**/*", "**/pages/**/*", "**/middleware.ts"]
alwaysApply: false
tags: ["nextjs", "routing", "app-router"]
---

# Next.js Routing Development Rules

## Overview
Comprehensive routing guidelines for Next.js applications...

## 1. App Router Patterns
Guidelines for using the App Router...

### Guidelines
- Use the app directory for new projects
- Leverage Server Components by default
- Use Client Components only when necessary

### Examples
```typescript
// app/page.tsx
export default function HomePage() {
  return <h1>Welcome to Next.js</h1>
}
```

### Anti-Patterns
- ❌ Don't mix App Router and Pages Router in the same project
- ❌ Avoid using Client Components for static content
```

### JSON Format
```json
{
  "metadata": {
    "description": "Next.js routing patterns",
    "globs": ["**/app/**/*", "**/pages/**/*"],
    "alwaysApply": false
  },
  "content": [
    {
      "title": "App Router Patterns",
      "description": "Guidelines for using the App Router...",
      "examples": ["..."],
      "guidelines": ["..."],
      "difficultyLevel": "intermediate"
    }
  ]
}
```

### YAML Format
```yaml
metadata:
  description: Next.js routing patterns
  globs:
    - "**/app/**/*"
    - "**/pages/**/*"
  alwaysApply: false

content:
  - title: App Router Patterns
    description: Guidelines for using the App Router...
    examples: ["..."]
    guidelines: ["..."]
    difficultyLevel: intermediate
```

## 🧪 Testing

### Run All Tests
```bash
rules-maker nextjs test
```

### Quick Tests
```bash
rules-maker nextjs test --quick
```

### Verbose Testing
```bash
rules-maker nextjs test --verbose
```

### Test Output
Tests generate comprehensive reports in the `test_output` directory:
- Component test results
- Integration test results
- Performance metrics
- Recommendations for improvement

## 🔧 Configuration

### Quality Threshold
Control the minimum quality for generated rules:
```bash
rules-maker nextjs process --quality-threshold 0.8
```

### ML Enhancement
Enable ML-enhanced processing for better categorization:
```bash
rules-maker nextjs process --ml-enhanced
```

### Learning System
Enable learning from user interactions:
```bash
rules-maker nextjs process --learning-enabled
```

### Parallel Processing
Enable parallel processing for faster execution:
```bash
rules-maker nextjs process --parallel
```

## 📊 Learning and Analytics

### Learning Events
The system records various learning events:
- **Categorization Events**: Content categorization results
- **Rule Generation Events**: Generated rule quality scores
- **User Feedback**: User satisfaction and corrections
- **Usage Events**: Rule effectiveness in practice

### Learning Reports
Generate comprehensive learning reports:
```bash
# Learning report is automatically generated when --learning-enabled is used
# Report saved to: .cursor/rules/nextjs/learning_report.json
```

### Learning Metrics
- **Categorization Accuracy**: How well content is categorized
- **Rule Quality Score**: Quality of generated rules
- **User Satisfaction**: User feedback scores
- **Improvement Trends**: Performance over time

## 🏗️ Architecture

### Core Components

1. **NextJSCategorizer**: Intelligent categorization system
2. **CursorRulesFormatter**: Formats content into cursor rules
3. **NextJSLearningIntegration**: Learning and improvement system
4. **AsyncDocumentationScraper**: High-performance web scraping
5. **AdaptiveDocumentationScraper**: ML-enhanced scraping

### Data Flow

```
Documentation Sources → Scraping → Categorization → Formatting → Learning → Cursor Rules
```

### Integration Points

- **ML Batch Processor**: For large-scale processing
- **Bedrock Integration**: For LLM-enhanced processing
- **Learning System**: For continuous improvement
- **Quality Assessment**: For rule validation

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -e .
   ```

2. **Scraping Failures**
   ```bash
   # Check network connectivity
   # Verify URLs are accessible
   # Use --verbose for detailed logs
   ```

3. **Categorization Issues**
   ```bash
   # Check taxonomy configuration
   # Verify pattern matching
   # Review content quality
   ```

4. **Learning System Errors**
   ```bash
   # Check data directory permissions
   # Verify learning configuration
   # Review event recording
   ```

### Debug Mode

Enable verbose logging for debugging:
```bash
rules-maker nextjs process --verbose
```

### Health Checks

Run system health checks:
```bash
rules-maker nextjs test --quick
```

## 📈 Performance Optimization

### Memory Usage
- **Base System**: ~500MB for 10 sources
- **ML-Enhanced**: ~2-4GB for 100 sources
- **Optimization**: Use quality thresholds to reduce memory

### Processing Speed
- **ML Overhead**: ~20-30% additional time
- **Parallel Processing**: 3-5x speed improvement
- **Clustering**: 5x throughput for large batches

### Quality Improvements
- **Semantic Analysis**: Enhanced technology detection
- **Quality Scoring**: Automated rule effectiveness
- **Self-Improvement**: Continuous optimization

## 🔄 Continuous Integration

### Automated Processing
Set up automated processing with cron:
```bash
# Daily processing of official sources
0 2 * * * rules-maker nextjs process --sources official --learning-enabled
```

### Quality Monitoring
Monitor rule quality over time:
```bash
# Weekly quality assessment
0 3 * * 0 rules-maker nextjs test --output-dir /var/log/nextjs-pipeline
```

## 📚 Examples

### Example 1: Basic Processing
```bash
# Process official Next.js docs
rules-maker nextjs process --sources official

# Output: .cursor/rules/nextjs/
# ├── routing.mdc
# ├── data-fetching.mdc
# ├── styling.mdc
# └── README.md
```

### Example 2: ML-Enhanced Processing
```bash
# Process with ML enhancement
rules-maker nextjs process \
  --sources all \
  --ml-enhanced \
  --learning-enabled \
  --quality-threshold 0.8

# Output: Enhanced categorization and learning reports
```

### Example 3: Custom Configuration
```bash
# Process specific sources with custom settings
rules-maker nextjs process \
  --sources ecosystem \
  --output .cursor/rules/nextjs-ecosystem \
  --format json \
  --parallel \
  --verbose
```

## 🤝 Contributing

### Adding New Sources
1. Update source definitions in `cli.py`
2. Add categorization patterns in `nextjs_categorizer.py`
3. Test with new sources

### Improving Categorization
1. Add new patterns to `NextJSCategorizer`
2. Update taxonomy configuration
3. Test categorization accuracy

### Enhancing Learning
1. Add new learning events
2. Improve pattern extraction
3. Update learning algorithms

## 📄 License

This pipeline is part of the rules-maker project and follows the same license terms.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Run tests to identify problems
3. Review logs with `--verbose`
4. Open an issue with detailed information

---

**Happy coding with Next.js and intelligent cursor rules! 🚀**
