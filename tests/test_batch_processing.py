"""
Comprehensive test suite for ML-powered batch processing system.

Tests the batch processor, self-improving engine, and rule clustering algorithms.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from rules_maker.batch_processor import (
    MLBatchProcessor, DocumentationSource, RuleCluster
)
from rules_maker.learning.self_improving_engine import (
    SelfImprovingEngine, FeedbackSignal, QualityPrediction
)
from rules_maker.models import (
    ScrapingResult, ScrapingStatus, RuleFormat, Rule, RuleType
)


class TestMLBatchProcessor:
    """Test suite for ML-powered batch processor."""
    
    @pytest.fixture
    def processor(self):
        """Create a test batch processor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield MLBatchProcessor(
                output_dir=temp_dir,
                quality_threshold=0.6,
                max_concurrent=2
            )
    
    @pytest.fixture
    def sample_sources(self):
        """Create sample documentation sources for testing."""
        return [
            DocumentationSource(
                url="https://reactjs.org/docs/",
                name="React",
                technology="javascript",
                framework="react",
                priority=5
            ),
            DocumentationSource(
                url="https://fastapi.tiangolo.com/",
                name="FastAPI",
                technology="python",
                framework="fastapi",
                priority=4
            ),
            DocumentationSource(
                url="https://kubernetes.io/docs/",
                name="Kubernetes",
                technology="cloud",
                framework="kubernetes",
                priority=5
            )
        ]
    
    @pytest.fixture
    def sample_scraped_results(self):
        """Create sample scraped results for testing."""
        return [
            ScrapingResult(
                url="https://reactjs.org/docs/",
                title="React Documentation",
                content="React is a JavaScript library for building user interfaces. It uses components and JSX syntax.",
                status=ScrapingStatus.COMPLETED,
                metadata={
                    'source_name': 'React',
                    'technology': 'javascript',
                    'framework': 'react'
                }
            ),
            ScrapingResult(
                url="https://fastapi.tiangolo.com/",
                title="FastAPI Documentation", 
                content="FastAPI is a modern web framework for building APIs with Python. It supports async/await and type hints.",
                status=ScrapingStatus.COMPLETED,
                metadata={
                    'source_name': 'FastAPI',
                    'technology': 'python',
                    'framework': 'fastapi'
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_scrape_documentation_batch(self, processor, sample_sources):
        """Test batch documentation scraping."""
        with patch.object(processor, '_scrape_documentation_batch') as mock_scrape:
            mock_results = [
                ScrapingResult(
                    url=source.url,
                    title=source.name,
                    content=f"Sample content for {source.name}",
                    status=ScrapingStatus.COMPLETED,
                    metadata={
                        'source_name': source.name,
                        'technology': source.technology,
                        'framework': source.framework
                    }
                )
                for source in sample_sources
            ]
            mock_scrape.return_value = mock_results
            
            results = await processor._scrape_documentation_batch(sample_sources)
            
            assert len(results) == 3
            assert all(r.status == ScrapingStatus.COMPLETED for r in results)
            assert results[0].metadata['technology'] == 'javascript'
            assert results[1].metadata['technology'] == 'python'
    
    @pytest.mark.asyncio
    async def test_generate_rules_batch(self, processor, sample_scraped_results):
        """Test batch rule generation."""
        rules = await processor._generate_rules_batch(
            sample_scraped_results,
            [RuleFormat.CURSOR, RuleFormat.WINDSURF]
        )
        
        assert len(rules) >= 2  # At least one per successful result
        assert all('id' in rule for rule in rules)
        assert all('technology' in rule for rule in rules)
        assert all('quality_score' in rule for rule in rules)
        assert any(rule['format'] == 'cursor' for rule in rules)
        assert any(rule['format'] == 'windsurf' for rule in rules)
    
    @pytest.mark.asyncio
    async def test_cluster_and_optimize_rules(self, processor):
        """Test rule clustering and optimization."""
        # Create sample rules
        sample_rules = [
            {
                'id': 'rule_1',
                'technology': 'javascript',
                'format': 'cursor',
                'content': 'Use React hooks for state management in functional components.',
                'quality_score': 0.8
            },
            {
                'id': 'rule_2',
                'technology': 'javascript',
                'format': 'cursor',
                'content': 'Implement proper error boundaries in React applications.',
                'quality_score': 0.7
            },
            {
                'id': 'rule_3',
                'technology': 'python',
                'format': 'windsurf',
                'content': 'Use FastAPI dependency injection for database connections.',
                'quality_score': 0.9
            }
        ]
        
        clusters = await processor._cluster_and_optimize_rules(sample_rules)
        
        assert len(clusters) >= 1
        assert all(isinstance(cluster, RuleCluster) for cluster in clusters)
        assert all(cluster.coherence_score >= 0 for cluster in clusters)
        assert all(len(cluster.rules) > 0 for cluster in clusters)
    
    @pytest.mark.asyncio
    async def test_assess_and_improve_quality(self, processor):
        """Test quality assessment and improvement."""
        # Create sample clusters
        sample_clusters = [
            RuleCluster(
                id="test_cluster_1",
                name="JavaScript Rules",
                rules=[{'id': 'rule_1', 'quality_score': 0.8}],
                coherence_score=0.8,
                technology="javascript",
                framework="react",
                semantic_keywords=["react", "hooks", "component"]
            ),
            RuleCluster(
                id="test_cluster_2",
                name="Python Rules",
                rules=[{'id': 'rule_2', 'quality_score': 0.7}],
                coherence_score=0.7,
                technology="python",
                framework="fastapi",
                semantic_keywords=["fastapi", "async", "api"]
            )
        ]
        
        quality_metrics = await processor._assess_and_improve_quality(sample_clusters)
        
        assert 'overall_coherence' in quality_metrics
        assert 'technology_coverage' in quality_metrics
        assert 'rule_diversity' in quality_metrics
        assert 'semantic_richness' in quality_metrics
        assert 'improvement_score' in quality_metrics
        
        assert 0 <= quality_metrics['overall_coherence'] <= 1
        assert 0 <= quality_metrics['technology_coverage'] <= 1
    
    def test_generate_rule_id(self, processor):
        """Test rule ID generation."""
        result = ScrapingResult(
            url="https://example.com",
            title="Test Doc",
            content="Test content"
        )
        
        rule_id = processor._generate_rule_id(result, RuleFormat.CURSOR)
        
        assert rule_id.startswith('rule_cursor_')
        assert len(rule_id) > 12  # Should have hash suffix
    
    def test_estimate_initial_quality(self, processor):
        """Test initial quality estimation."""
        # Mock semantic analysis
        mock_analysis = Mock()
        mock_analysis.best_practices.items = [Mock(), Mock()]  # 2 best practices
        mock_analysis.patterns.patterns = [Mock()]  # 1 pattern
        
        # Test with good content
        quality = processor._estimate_initial_quality("A long rule with examples and good structure", mock_analysis)
        assert 0.5 <= quality <= 1.0
        
        # Test with poor content
        quality = processor._estimate_initial_quality("Short", Mock())
        assert quality < 0.5
    
    def test_extract_keywords_from_rules(self, processor):
        """Test keyword extraction from rules."""
        rules = [
            {'content': 'Use async functions for better performance'},
            {'content': 'Implement error handling with try-catch patterns'},
            {'content': 'Write test cases for all API endpoints'}
        ]
        
        keywords = processor._extract_keywords_from_rules(rules)
        
        assert isinstance(keywords, list)
        assert all(isinstance(kw, str) for kw in keywords)
        # Should contain technical terms
        assert any(kw in ['async', 'function', 'error', 'handler', 'test', 'api'] for kw in keywords)


class TestSelfImprovingEngine:
    """Test suite for self-improving engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test self-improving engine."""
        return SelfImprovingEngine(
            feedback_window_hours=1,
            min_feedback_signals=2,
            quality_threshold=0.6,
            model_update_interval_hours=0.1  # Short interval for testing
        )
    
    @pytest.fixture
    def sample_rule(self):
        """Create a sample rule for testing."""
        return Rule(
            id="test_rule_1",
            title="Sample Rule",
            description="A sample rule for testing",
            content="Use proper error handling in your code with try-catch blocks and meaningful error messages.",
            type=RuleType.BEST_PRACTICE,
            category="error-handling",
            priority=3,
            confidence_score=0.7,
            tags=["error-handling", "best-practice"],
            examples=["try { /* code */ } catch (error) { /* handle */ }"]
        )
    
    @pytest.mark.asyncio
    async def test_collect_feedback_signal(self, engine, sample_rule):
        """Test feedback signal collection."""
        await engine.collect_feedback_signal(
            rule_id=sample_rule.id,
            signal_type="usage_success",
            value=0.8,
            context={"usage_type": "cursor_ide"},
            source="user"
        )
        
        assert len(engine.feedback_signals) == 1
        signal = engine.feedback_signals[0]
        assert signal.rule_id == sample_rule.id
        assert signal.signal_type == "usage_success"
        assert signal.value == 0.8
        assert signal.source == "user"
    
    @pytest.mark.asyncio
    async def test_predict_rule_quality_heuristic(self, engine, sample_rule):
        """Test quality prediction with heuristic fallback."""
        prediction = await engine.predict_rule_quality(sample_rule)
        
        assert isinstance(prediction, QualityPrediction)
        assert prediction.rule_id == sample_rule.id
        assert 0 <= prediction.predicted_quality <= 1
        assert len(prediction.confidence_interval) == 2
        assert prediction.confidence_interval[0] <= prediction.predicted_quality <= prediction.confidence_interval[1]
    
    @pytest.mark.asyncio
    async def test_self_award_quality_improvements(self, engine):
        """Test self-awarding system."""
        from rules_maker.learning.models import GeneratedRule
        
        # Create mock rules with improving quality history
        rules = [
            Mock(rule=Mock(id="rule_1")),
            Mock(rule=Mock(id="rule_2"))
        ]
        
        # Simulate quality history showing improvement
        engine.quality_history["rule_1"] = [0.5, 0.6, 0.7, 0.8]  # Improving trend
        engine.quality_history["rule_2"] = [0.8, 0.7, 0.6, 0.5]  # Declining trend
        
        batch_performance = {
            'improvement_score': 0.8,
            'quality_scores': {'rule_1': 0.8, 'rule_2': 0.5},
            'predicted_qualities': {'rule_1': 0.6, 'rule_2': 0.7}  # rule_1 exceeded prediction
        }
        
        awards = await engine.self_award_quality_improvements(rules, batch_performance)
        
        assert 'rule_1' in awards
        assert awards['rule_1'] > 0  # Should get award for improvement and exceeding prediction
        # rule_2 might get global award but not improvement award
    
    @pytest.mark.asyncio
    async def test_update_adaptive_thresholds(self, engine):
        """Test adaptive threshold updating."""
        initial_quality_threshold = engine.adaptive_thresholds['quality_threshold']
        
        # Test with high performance
        high_performance = {
            'overall_coherence': 0.9,
            'average_success_rate': 0.8
        }
        
        await engine.update_adaptive_thresholds(high_performance)
        
        assert engine.adaptive_thresholds['quality_threshold'] >= initial_quality_threshold
        
        # Test with low performance
        low_performance = {
            'overall_coherence': 0.3,
            'average_success_rate': 0.4
        }
        
        await engine.update_adaptive_thresholds(low_performance)
        
        # Should lower thresholds
        assert engine.adaptive_thresholds['quality_threshold'] < 0.9
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self, engine, sample_rule):
        """Test saving and loading engine state."""
        # Add some feedback signals
        await engine.collect_feedback_signal(
            rule_id=sample_rule.id,
            signal_type="quality_score",
            value=0.8
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save state
            await engine.save_state(tmp_path)
            assert Path(tmp_path).exists()
            
            # Create new engine and load state
            new_engine = SelfImprovingEngine()
            await new_engine.load_state(tmp_path)
            
            assert len(new_engine.feedback_signals) == 1
            assert new_engine.feedback_signals[0].rule_id == sample_rule.id
            assert new_engine.feedback_signals[0].value == 0.8
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_extract_rule_features(self, engine, sample_rule):
        """Test rule feature extraction."""
        features = engine._extract_rule_features(sample_rule)
        
        assert isinstance(features, dict)
        assert 'content_length' in features
        assert 'title_length' in features
        assert 'has_examples' in features
        assert 'confidence_score' in features
        assert 'priority' in features
        
        # Check feature values are normalized
        assert 0 <= features['content_length'] <= 1
        assert 0 <= features['has_examples'] <= 1
        assert features['confidence_score'] == sample_rule.confidence_score
    
    def test_heuristic_quality_estimation(self, engine, sample_rule):
        """Test heuristic quality estimation."""
        quality = engine._heuristic_quality_estimation(sample_rule)
        
        assert 0 <= quality <= 1
        assert quality > 0.5  # Should be decent quality given the sample rule


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_batch_processing(self):
        """Test complete end-to-end batch processing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create processor
            processor = MLBatchProcessor(
                output_dir=temp_dir,
                max_concurrent=1  # Limit concurrency for testing
            )
            
            # Create minimal test sources
            sources = [
                DocumentationSource(
                    url="https://example.com/react",
                    name="React Test",
                    technology="javascript",
                    framework="react"
                )
            ]
            
            # Mock the scraping to avoid actual HTTP requests
            with patch.object(processor, '_scrape_documentation_batch') as mock_scrape:
                mock_scrape.return_value = [
                    ScrapingResult(
                        url="https://example.com/react",
                        title="React Test Doc",
                        content="React is a JavaScript library. Use hooks for state management.",
                        status=ScrapingStatus.COMPLETED,
                        metadata={
                            'source_name': 'React Test',
                            'technology': 'javascript',
                            'framework': 'react'
                        }
                    )
                ]
                
                # Run batch processing
                result = await processor.process_documentation_batch(sources)
                
                # Verify results
                assert result.sources_processed == 1
                assert result.total_rules_generated > 0
                assert len(result.clusters) > 0
                assert result.processing_time > 0
                
                # Check that files were created
                output_path = Path(temp_dir)
                assert any(output_path.glob("**/*.md"))  # Should have rule files
                assert any(output_path.glob("**/metadata.json"))  # Should have metadata
    
    @pytest.mark.asyncio
    async def test_self_improving_feedback_integration(self):
        """Test integration between batch processor and self-improving engine."""
        engine = SelfImprovingEngine(min_feedback_signals=1)
        
        # Simulate collecting feedback over time
        rule_id = "test_rule_integration"
        
        # Initial feedback
        await engine.collect_feedback_signal(rule_id, "usage_success", 0.6)
        await engine.collect_feedback_signal(rule_id, "quality_score", 0.7)
        
        # Check that quality history is updated
        assert rule_id in engine.quality_history
        assert len(engine.quality_history[rule_id]) == 1
        assert engine.quality_history[rule_id][0] == 0.7
        
        # More feedback showing improvement
        await engine.collect_feedback_signal(rule_id, "quality_score", 0.8)
        assert len(engine.quality_history[rule_id]) == 2
        
        # Test self-awarding
        mock_rules = [Mock(rule=Mock(id=rule_id))]
        batch_performance = {
            'improvement_score': 0.5,
            'quality_scores': {rule_id: 0.8}
        }
        
        awards = await engine.self_award_quality_improvements(mock_rules, batch_performance)
        
        # Should get award for improvement trend
        assert rule_id in awards
        assert awards[rule_id] > 0


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])