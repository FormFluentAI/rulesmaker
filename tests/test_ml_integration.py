"""
Comprehensive test suite for ML integration components.

Tests ML-enhanced processors, strategies, transformers, and integrated learning system
with full backward compatibility validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any, List

# Test data setup
@pytest.fixture
def sample_html_content():
    return """
    <html>
    <head><title>FastAPI Documentation</title></head>
    <body>
        <h1>FastAPI</h1>
        <p>FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.</p>
        <h2>Installation</h2>
        <pre><code>pip install fastapi</code></pre>
        <h2>Example</h2>
        <pre><code>
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/")
        async def read_root():
            return {"Hello": "World"}
        </code></pre>
        <h2>API Reference</h2>
        <p>FastAPI provides automatic API documentation with OpenAPI and JSON Schema.</p>
    </body>
    </html>
    """

@pytest.fixture
def sample_scraping_result(sample_html_content):
    from rules_maker.models import ScrapingResult, ScrapingStatus
    return ScrapingResult(
        url="https://fastapi.tiangolo.com/",
        title="FastAPI Documentation",
        content=sample_html_content,
        status=ScrapingStatus.COMPLETED,
        timestamp=datetime.now()
    )

@pytest.fixture
def sample_learning_example():
    from rules_maker.models import LearningExample, DocumentationType
    return LearningExample(
        input_html="""
        <html><head><title>Python Tutorial</title></head>
        <body><h1>Python Tutorial</h1><p>Learn Python programming</p>
        <pre><code>print("Hello World")</code></pre></body></html>
        """,
        url="https://docs.python.org/tutorial/",
        expected_output="Python programming tutorial with examples",
        documentation_type=DocumentationType.TUTORIAL,
        feedback_score=0.8
    )

@pytest.fixture
def sample_training_set(sample_learning_example):
    from rules_maker.models import TrainingSet
    return TrainingSet(
        examples=[sample_learning_example] * 5,  # Create 5 similar examples
        version="1.0",
        created_at=datetime.now()
    )


class TestMLDocumentationProcessor:
    """Test suite for ML-enhanced documentation processor."""
    
    def test_ml_processor_initialization(self):
        """Test ML processor initializes correctly."""
        from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
        
        processor = MLDocumentationProcessor()
        assert processor is not None
        assert hasattr(processor, 'ml_enabled')
        assert hasattr(processor, 'semantic_analyzer')
        
    def test_ml_processor_with_config(self):
        """Test ML processor initialization with custom config."""
        from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
        
        config = {'enable_ml': True, 'quality_threshold': 0.8}
        processor = MLDocumentationProcessor(config)
        assert processor.config == config
        
    def test_processor_fallback_when_ml_unavailable(self):
        """Test processor falls back gracefully when ML components fail."""
        from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
        
        with patch('rules_maker.processors.ml_documentation_processor.SemanticAnalyzer', side_effect=ImportError("Mock ML failure")):
            processor = MLDocumentationProcessor()
            assert processor.ml_enabled == False
            assert processor.semantic_analyzer is None
    
    def test_ml_processor_enhancement_with_ml_enabled(self, sample_html_content):
        """Test ML processor adds enhancements when ML is enabled."""
        from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
        
        processor = MLDocumentationProcessor()
        
        # Mock the semantic analyzer if ML is enabled
        if processor.ml_enabled:
            processor.semantic_analyzer.extract_semantic_keywords = Mock(return_value=['fastapi', 'python', 'web framework'])
        
        result = processor.process(sample_html_content, "https://fastapi.tiangolo.com/", {})
        
        assert result is not None
        if processor.ml_enabled:
            assert result.metadata.get('ml_enhanced', False)
            assert 'detected_technologies' in result.metadata
            assert 'content_complexity' in result.metadata
            assert 'quality_metrics' in result.metadata
        else:
            assert result.metadata.get('ml_enhanced', True) == False
    
    def test_technology_detection(self, sample_html_content):
        """Test technology detection capabilities."""
        from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
        
        processor = MLDocumentationProcessor()
        technologies = processor._detect_technologies(sample_html_content, "https://fastapi.tiangolo.com/")
        
        # Should detect FastAPI and Python
        tech_names = [tech['name'] for tech in technologies]
        assert 'fastapi' in tech_names or 'python' in tech_names
        
        # Check confidence scores are reasonable
        for tech in technologies:
            assert 0.0 <= tech['confidence'] <= 1.0
            assert tech['detection_method'] == 'ml_enhanced_pattern_matching'
    
    def test_content_complexity_scoring(self):
        """Test content complexity scoring."""
        from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
        
        processor = MLDocumentationProcessor()
        
        # Simple content
        simple_content = "Hello world"
        simple_complexity = processor._calculate_content_complexity(simple_content)
        
        # Complex content with code and technical terms
        complex_content = """
        FastAPI is a web framework with API endpoints and authentication.
        <pre><code>app = FastAPI()</code></pre>
        <a href="https://example.com">Link</a>
        Methods include GET, POST, PUT, DELETE for endpoint configuration.
        """
        complex_complexity = processor._calculate_content_complexity(complex_content)
        
        # Complex content should have higher complexity score
        assert complex_complexity['overall_score'] > simple_complexity['overall_score']
        assert complex_complexity['complexity_level'] in ['low', 'medium', 'high']


class TestMLQualityStrategy:
    """Test suite for ML quality strategy."""
    
    def test_ml_strategy_initialization(self):
        """Test ML strategy initializes correctly."""
        from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
        
        strategy = MLQualityStrategy()
        assert strategy is not None
        assert hasattr(strategy, 'quality_threshold')
        assert hasattr(strategy, 'heuristic_weights')
        assert strategy.quality_threshold == 0.7  # Default value
    
    def test_ml_strategy_with_config(self):
        """Test ML strategy initialization with custom config."""
        from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
        
        config = {'quality_threshold': 0.8, 'model_directory': 'custom_models/'}
        strategy = MLQualityStrategy(config)
        assert strategy.quality_threshold == 0.8
    
    def test_heuristic_prediction(self):
        """Test heuristic-based quality prediction."""
        from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
        
        strategy = MLQualityStrategy()
        
        # High-quality content with code, examples, and structure
        high_quality_content = """
        <h1>FastAPI Tutorial</h1>
        <p>Complete guide with examples</p>
        <pre><code>from fastapi import FastAPI</code></pre>
        <p>API endpoint documentation with parameters</p>
        <p>How to get started tutorial</p>
        """
        
        prediction = asyncio.run(strategy._heuristic_predict(high_quality_content, "https://fastapi.tiangolo.com/"))
        
        assert prediction['quality_score'] > 0.5
        assert prediction['quality_class'] in ['low', 'medium', 'high']
        assert prediction['method'] == 'heuristic_prediction'
        assert 'recommendations' in prediction
        assert prediction['is_high_quality'] == (prediction['quality_score'] >= strategy.quality_threshold)
    
    def test_quality_recommendations(self):
        """Test quality improvement recommendations."""
        from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
        
        strategy = MLQualityStrategy()
        
        # Poor quality content missing key elements
        poor_content = "This is just plain text without code examples or structure."
        recommendations = strategy._generate_recommendations(poor_content, 0.3)
        
        assert len(recommendations) > 0
        assert any('code examples' in rec for rec in recommendations)
        assert any('headings' in rec or 'structure' in rec for rec in recommendations)
    
    def test_training_with_heuristic_fallback(self, sample_training_set):
        """Test training with heuristic fallback when ML is unavailable."""
        from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
        
        # Force heuristic fallback
        with patch('rules_maker.strategies.ml_quality_strategy.ML_AVAILABLE', False):
            strategy = MLQualityStrategy()
            metrics = strategy.train(sample_training_set)
            
            assert metrics is not None
            assert strategy.is_trained
            assert 0.0 <= metrics.accuracy <= 1.0
            assert metrics.training_examples == len(sample_training_set.examples)
    
    @pytest.mark.asyncio
    async def test_async_predict_interface(self):
        """Test async prediction interface."""
        from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
        
        strategy = MLQualityStrategy()
        
        content = "<h1>Test</h1><p>Test content with <code>example</code></p>"
        prediction = await strategy.predict(content, "https://test.com")
        
        assert prediction is not None
        assert 'quality_score' in prediction
        assert 'quality_class' in prediction
        assert 'confidence' in prediction
        assert 'is_high_quality' in prediction


class TestMLCursorTransformer:
    """Test suite for ML-enhanced Cursor transformer."""
    
    def test_ml_transformer_initialization(self):
        """Test ML transformer initializes correctly."""
        from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
        
        transformer = MLCursorTransformer()
        assert transformer is not None
        assert hasattr(transformer, 'ml_enabled')
        assert hasattr(transformer, 'quality_threshold')
        assert transformer.quality_threshold == 0.7  # Default value
    
    def test_ml_transformer_with_config(self):
        """Test ML transformer initialization with config."""
        from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
        
        ml_config = {
            'quality_threshold': 0.8,
            'enable_clustering': False,
            'coherence_threshold': 0.7
        }
        transformer = MLCursorTransformer(ml_config)
        assert transformer.quality_threshold == 0.8
        assert transformer.enable_clustering == False
        assert transformer.coherence_threshold == 0.7
    
    def test_transformer_fallback_when_ml_unavailable(self, sample_scraping_result):
        """Test transformer falls back gracefully when ML components fail."""
        from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
        
        with patch('rules_maker.transformers.ml_cursor_transformer.MLQualityStrategy', side_effect=ImportError("Mock ML failure")):
            transformer = MLCursorTransformer()
            assert transformer.ml_enabled == False
            
            # Should still produce rules with fallback enhancement
            rules = asyncio.run(transformer.transform([sample_scraping_result]))
            assert rules is not None
            assert len(rules) > 0
            assert "Source Analysis" in rules
    
    @pytest.mark.asyncio
    async def test_ml_enhancement_integration(self, sample_scraping_result):
        """Test ML enhancement integration in transformation."""
        from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
        
        transformer = MLCursorTransformer()
        
        # Mock quality strategy if ML is enabled
        if transformer.ml_enabled:
            mock_prediction = {
                'quality_score': 0.8,
                'quality_class': 'high',
                'confidence': 0.9,
                'is_high_quality': True,
                'recommendations': ['Add more examples'],
                'method': 'ml_prediction'
            }
            transformer.quality_strategy.predict = AsyncMock(return_value=mock_prediction)
        
        rules = await transformer.transform([sample_scraping_result])
        
        assert rules is not None
        assert len(rules) > 0
        
        if transformer.ml_enabled:
            # Should contain ML quality assessment section
            assert "ML Quality Assessment" in rules or "Source Analysis" in rules
        else:
            # Should contain basic source analysis
            assert "Source Analysis" in rules
    
    @pytest.mark.asyncio  
    async def test_quality_insights_generation(self, sample_scraping_result):
        """Test quality insights generation."""
        from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
        
        transformer = MLCursorTransformer()
        
        if transformer.ml_enabled:
            # Mock the quality strategy
            mock_prediction = {
                'quality_score': 0.75,
                'is_high_quality': True,
                'recommendations': ['Test recommendation']
            }
            transformer.quality_strategy.predict = AsyncMock(return_value=mock_prediction)
        
        insights = await transformer.get_quality_insights([sample_scraping_result])
        
        assert insights is not None
        if transformer.ml_enabled:
            assert 'total_sources' in insights
            assert 'quality_assessments' in insights
            assert insights['total_sources'] == 1
        else:
            assert 'error' in insights
            assert insights['method'] == 'fallback'


class TestIntegratedLearningSystem:
    """Test suite for integrated learning system."""
    
    def test_integrated_system_initialization(self):
        """Test integrated learning system initializes correctly."""
        from rules_maker.learning.integrated_learning_system import IntegratedLearningSystem
        
        system = IntegratedLearningSystem()
        assert system is not None
        assert hasattr(system, 'base_engine')
        assert hasattr(system, 'ml_engine')
        assert hasattr(system, 'ml_strategy')
        assert hasattr(system, 'enable_ml')
        assert system.enable_ml == True  # Default value
        assert system.ml_weight == 0.6   # Default value
    
    def test_integrated_system_with_config(self):
        """Test integrated system initialization with config."""
        from rules_maker.learning.integrated_learning_system import IntegratedLearningSystem
        
        config = {
            'enable_ml': True,
            'ml_weight': 0.7,
            'feedback_integration': True,
            'quality_threshold': 0.8
        }
        system = IntegratedLearningSystem(config)
        assert system.enable_ml == True
        assert system.ml_weight == 0.7
        assert system.feedback_integration == True
    
    @pytest.mark.asyncio
    async def test_system_performance_stats(self):
        """Test system performance statistics generation."""
        from rules_maker.learning.integrated_learning_system import IntegratedLearningSystem
        
        system = IntegratedLearningSystem()
        stats = await system.get_system_performance_stats()
        
        assert stats is not None
        assert 'system_type' in stats
        assert stats['system_type'] == 'integrated_learning'
        assert 'components' in stats
        assert 'configuration' in stats
        assert stats['components']['base_engine'] == 'active'
        assert stats['configuration']['ml_enabled'] == system.enable_ml


class TestMLIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_ml_pipeline(self, sample_scraping_result):
        """Test complete ML pipeline from processing to rule generation."""
        from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
        from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
        from rules_maker.learning.integrated_learning_system import IntegratedLearningSystem
        
        # Initialize components
        processor = MLDocumentationProcessor()
        transformer = MLCursorTransformer()
        learning_system = IntegratedLearningSystem()
        
        # Process content
        processed_result = processor.process(
            sample_scraping_result.content, 
            str(sample_scraping_result.url), 
            {}
        )
        assert processed_result is not None
        
        # Transform to rules
        rules = await transformer.transform([sample_scraping_result])
        assert rules is not None
        assert len(rules) > 0
        
        # Get system stats
        stats = await learning_system.get_system_performance_stats()
        assert stats is not None
        assert stats['system_type'] == 'integrated_learning'
    
    def test_backward_compatibility(self, sample_scraping_result):
        """Test that existing functionality still works with ML components."""
        from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
        from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
        
        # Test that base transformer still works
        base_transformer = CursorRuleTransformer()
        base_rules = base_transformer.transform([sample_scraping_result])
        assert base_rules is not None
        assert len(base_rules) > 0
        
        # Test that ML transformer produces similar structure
        ml_transformer = MLCursorTransformer()
        ml_rules = asyncio.run(ml_transformer.transform([sample_scraping_result]))
        assert ml_rules is not None
        assert len(ml_rules) > 0
        
        # Both should contain similar core elements
        assert "# Cursor Rules" in base_rules or "FastAPI" in base_rules
        assert "# Cursor Rules" in ml_rules or "FastAPI" in ml_rules or "Source Analysis" in ml_rules
    
    def test_configuration_loading(self):
        """Test ML configuration loading."""
        import yaml
        import os
        
        config_path = '/home/ollie/dev/rules-maker/config/ml_batch_config.yaml'
        
        # Test that config file exists and is valid YAML
        assert os.path.exists(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert 'batch_processing' in config
        assert 'ml_engine' in config
        assert 'integrated_learning' in config
        
        # Test that config values are reasonable
        assert config['batch_processing']['quality_threshold'] > 0.0
        assert config['batch_processing']['quality_threshold'] <= 1.0
        assert config['ml_engine']['quality_threshold'] > 0.0
        assert config['integrated_learning']['ml_weight'] >= 0.0
        assert config['integrated_learning']['ml_weight'] <= 1.0


class TestMLComponentsWithMockedDependencies:
    """Test ML components with mocked scikit-learn dependencies."""
    
    def test_ml_components_without_sklearn(self):
        """Test components gracefully handle missing scikit-learn."""
        
        with patch('rules_maker.strategies.ml_quality_strategy.ML_AVAILABLE', False):
            from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
            
            strategy = MLQualityStrategy()
            assert strategy.quality_classifier is None
            assert strategy.quality_regressor is None
            assert strategy.vectorizer is None
    
    @pytest.mark.asyncio
    async def test_fallback_predictions_without_ml(self):
        """Test fallback predictions when ML is unavailable."""
        
        with patch('rules_maker.strategies.ml_quality_strategy.ML_AVAILABLE', False):
            from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
            
            strategy = MLQualityStrategy()
            prediction = await strategy.predict("<h1>Test</h1><p>API documentation</p>", "https://test.com")
            
            assert prediction is not None
            assert prediction['method'] == 'heuristic_prediction'
            assert 'quality_score' in prediction
            assert 'recommendations' in prediction


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])