# ML-Powered Batch Processing Guide

## Overview

The Rules Maker now includes a sophisticated ML-powered batch processing system that can scrape documentation from 100+ sources concurrently and generate intelligent, coherent rule sets using machine learning algorithms.

## Key Features

### 🧠 Intelligent Processing Pipeline

1. **Concurrent Documentation Scraping** - Process multiple sources in parallel with rate limiting
2. **ML-Powered Rule Generation** - Use Bedrock Nova Lite or standard transformers for enhanced rule creation
3. **Semantic Clustering** - Group related rules using TF-IDF vectorization and K-means clustering
4. **Self-Improving Feedback Loop** - Learn from usage patterns and automatically improve rule quality
5. **Quality Assessment** - Comprehensive metrics and scoring system for rule effectiveness

### 🚀 Self-Awarding System

The system includes a sophisticated self-awarding mechanism that:
- Tracks rule performance over time
- Automatically boosts quality scores for improving rules
- Awards bonuses for rules that exceed predictions
- Provides global performance rewards for high-quality batches

## Quick Start

### Basic Usage

```python
from rules_maker.batch_processor import process_popular_frameworks
import asyncio

# Process popular web frameworks
result = await process_popular_frameworks(
    output_dir="rules/frameworks",
    bedrock_config={
        'model_id': 'amazon.nova-lite-v1:0',
        'region': 'us-east-1'
    }
)

print(f"Generated {result.total_rules_generated} rules from {result.sources_processed} sources")
```

### Advanced Custom Processing

```python
from rules_maker.batch_processor import MLBatchProcessor, DocumentationSource
from rules_maker.models import RuleFormat

# Define your documentation sources
sources = [
    DocumentationSource("https://reactjs.org/docs/", "React", "javascript", "react", priority=5),
    DocumentationSource("https://fastapi.tiangolo.com/", "FastAPI", "python", "fastapi", priority=5),
    DocumentationSource("https://kubernetes.io/docs/", "Kubernetes", "cloud", "kubernetes", priority=4),
    # ... add up to 100+ sources
]

# Configure processor
processor = MLBatchProcessor(
    bedrock_config={
        'model_id': 'amazon.nova-lite-v1:0',
        'temperature': 0.3,
        'max_tokens': 2000
    },
    output_dir="rules/custom_batch",
    quality_threshold=0.7,
    max_concurrent=15
)

# Process all sources
result = await processor.process_documentation_batch(
    sources,
    formats=[RuleFormat.CURSOR, RuleFormat.WINDSURF]
)
```

## Self-Improving Feedback System

### Setting Up Feedback Collection

```python
from rules_maker.learning.self_improving_engine import SelfImprovingEngine

# Initialize the engine
engine = SelfImprovingEngine(
    feedback_window_hours=168,  # 1 week
    min_feedback_signals=5,
    quality_threshold=0.7,
    learning_rate=0.1
)

# Collect feedback signals
await engine.collect_feedback_signal(
    rule_id="rule_cursor_abc123",
    signal_type="usage_success",
    value=0.8,
    context={"usage_type": "cursor_ide", "project_type": "react"},
    source="user"
)

await engine.collect_feedback_signal(
    rule_id="rule_cursor_abc123",
    signal_type="user_rating",
    value=0.9,
    context={"rating_context": "rule_helpfulness"},
    source="user"
)
```

### Quality Prediction and Improvement

```python
# Predict rule quality using ML models
prediction = await engine.predict_rule_quality(rule)
print(f"Predicted quality: {prediction.predicted_quality:.3f}")
print(f"Confidence interval: {prediction.confidence_interval}")

# Generate improvement recommendations
recommendations = await engine.generate_improvement_recommendations(
    rule, effectiveness, prediction
)

for rec in recommendations:
    print(f"• {rec.description} (Expected improvement: {rec.expected_improvement:.3f})")
```

## Comprehensive Examples

### Processing 100+ Documentation Sources

See `examples/batch_processing_demo.py` for a complete demonstration:

```bash
# Basic demo with standard transformers
PYTHONPATH=src python examples/batch_processing_demo.py

# Enhanced demo with Bedrock integration
PYTHONPATH=src python examples/batch_processing_demo.py --bedrock

# Run specific demo modes
PYTHONPATH=src python examples/batch_processing_demo.py --demo-mode frameworks
PYTHONPATH=src python examples/batch_processing_demo.py --demo-mode cloud
PYTHONPATH=src python examples/batch_processing_demo.py --demo-mode comprehensive
```

## Architecture Details

### ML Pipeline Components

1. **Semantic Analyzer** (`learning/pattern_analyzer.py`)
   - Extracts code patterns, best practices, and anti-patterns
   - Performs technology detection and framework identification
   - Generates semantic keywords and topics

2. **Batch Processor** (`batch_processor.py`)
   - Orchestrates the entire pipeline
   - Handles concurrent scraping with rate limiting
   - Implements clustering algorithms using scikit-learn

3. **Self-Improving Engine** (`learning/self_improving_engine.py`)
   - Collects and analyzes feedback signals
   - Trains ML models for quality prediction
   - Implements self-awarding mechanisms

4. **Learning Engine** (`learning/engine.py`)
   - Provides usage pattern analysis
   - Optimizes rules based on effectiveness metrics
   - Supports A/B testing for rule variants

### Clustering Algorithm

The system uses TF-IDF vectorization combined with K-means clustering:

```python
# Vectorize rule content
feature_matrix = self.vectorizer.fit_transform(rule_texts)

# Determine optimal clusters (2-8 per technology group)
n_clusters = min(max(2, len(rules) // 3), 8)

# Perform clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(feature_matrix)

# Calculate coherence scores using cosine similarity
similarities = cosine_similarity(cluster_features)
coherence_score = 0.6 * avg_similarity + 0.4 * avg_quality
```

### Quality Metrics

The system tracks comprehensive quality metrics:

- **Overall Coherence**: Average coherence scores across all clusters
- **Technology Coverage**: Percentage of expected technologies covered
- **Rule Diversity**: Distribution of rules across clusters
- **Semantic Richness**: Variety of semantic keywords extracted
- **Improvement Score**: ML-driven improvement assessment

## Output Structure

The batch processor generates organized output:

```
rules/
├── frameworks/
│   ├── javascript/
│   │   ├── javascript_cursor_cluster_0/
│   │   │   ├── javascript_cursor_rules.md
│   │   │   ├── javascript_windsurf_rules.md
│   │   │   └── metadata.json
│   │   └── javascript_cursor_cluster_1/
│   └── python/
├── cloud/
└── comprehensive_demo/
```

Each cluster contains:
- **Rule files**: Formatted rules for Cursor/Windsurf
- **Metadata**: Cluster information, coherence scores, source attribution

## Configuration Options

### Batch Processor Configuration

```python
processor = MLBatchProcessor(
    bedrock_config={
        'model_id': 'amazon.nova-lite-v1:0',
        'region': 'us-east-1',
        'temperature': 0.3,
        'max_tokens': 2000
    },
    output_dir="rules/output",
    quality_threshold=0.6,  # Minimum quality for rule inclusion
    max_concurrent=10       # Maximum concurrent scraping operations
)
```

### Self-Improving Engine Configuration

```python
engine = SelfImprovingEngine(
    feedback_window_hours=168,      # Feedback collection window
    min_feedback_signals=5,         # Minimum signals for ML model training
    quality_threshold=0.7,          # Quality threshold for recommendations
    learning_rate=0.1,              # ML model learning rate
    model_update_interval_hours=24  # How often to retrain models
)
```

## Testing

Run comprehensive tests:

```bash
# Run all batch processing tests
PYTHONPATH=src pytest tests/test_batch_processing.py -v

# Run specific test categories
PYTHONPATH=src pytest tests/test_batch_processing.py::TestMLBatchProcessor -v
PYTHONPATH=src pytest tests/test_batch_processing.py::TestSelfImprovingEngine -v
```

## Performance Considerations

### Scalability

- **Concurrent Processing**: Default 10-15 concurrent scrapers
- **Memory Usage**: ~2-4GB for 100 sources with full ML pipeline
- **Processing Time**: ~10-20 minutes for 100 sources (varies by source size)

### Optimization Tips

1. **Adjust Concurrency**: Increase `max_concurrent` for faster processing
2. **Use Bedrock**: AWS Bedrock Nova Lite provides better rule quality
3. **Filter Sources**: Use priority levels to focus on high-value documentation
4. **Cache Results**: Intermediate results are cached for repeated processing

## Troubleshooting

### Common Issues

1. **Rate Limiting**: If you hit rate limits, reduce `max_concurrent`
2. **Memory Issues**: Process sources in smaller batches
3. **Bedrock Errors**: Ensure proper AWS credentials and model access
4. **Empty Results**: Check source URLs and scraping configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Features

### Custom Source Lists

Create domain-specific source collections:

```python
def create_ml_ai_sources():
    return [
        DocumentationSource("https://pytorch.org/docs/", "PyTorch", "python", "pytorch", priority=5),
        DocumentationSource("https://tensorflow.org/guide", "TensorFlow", "python", "tensorflow", priority=5),
        DocumentationSource("https://scikit-learn.org/stable/", "Scikit-learn", "python", "sklearn", priority=4),
        # ... more ML/AI sources
    ]

sources = create_ml_ai_sources()
result = await processor.process_documentation_batch(sources)
```

### Integration with Existing Workflows

```python
# Save self-improving engine state for persistence
await engine.save_state("production_engine_state.json")

# Load state in future sessions
await engine.load_state("production_engine_state.json")

# Export results for external systems
with open("batch_results.json", "w") as f:
    json.dump({
        'sources_processed': result.sources_processed,
        'clusters': [cluster.__dict__ for cluster in result.clusters],
        'quality_metrics': result.quality_metrics
    }, f, indent=2)
```

## Future Enhancements

The ML batch processing system is designed for continuous improvement:

1. **Enhanced ML Models**: Integration with larger language models
2. **Real-time Feedback**: Live feedback collection from IDE usage
3. **Collaborative Filtering**: Community-driven rule quality assessment
4. **Domain Adaptation**: Specialized models for different technology domains
5. **Performance Optimization**: GPU acceleration for large-scale processing