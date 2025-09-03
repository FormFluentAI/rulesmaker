"""
Tests for learning and intelligence module integration across transformers.

Tests the integration of learning engines, intelligence engines, semantic analyzers,
and usage trackers across all transformer components with comprehensive validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from rules_maker.transformers.rule_transformer import RuleTransformer
from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.models import ContentSection, ScrapingResult, ScrapingStatus


@pytest.fixture
def sample_content_sections():
    """Sample content sections for testing."""
    return [
        ContentSection(
            title="Advanced React Patterns",
            content="Advanced React patterns and architectural decisions",
            section_type="code_example",
            metadata={"language": "javascript", "framework": "react", "topic": "patterns", "complexity": "advanced"}
        ),
        ContentSection(
            title="TypeScript Best Practices",
            content="TypeScript best practices for large-scale applications",
            section_type="guideline",
            metadata={"language": "typescript", "framework": "react", "topic": "typescript", "complexity": "intermediate"}
        ),
        ContentSection(
            title="Testing Architecture",
            content="Comprehensive testing architecture and strategies",
            section_type="guideline",
            metadata={"topic": "testing", "framework": "react", "tools": ["jest", "testing-library"], "complexity": "advanced"}
        )
    ]


@pytest.fixture
def mock_learning_engine():
    """Mock learning engine with comprehensive responses."""
    mock_engine = Mock()
    mock_engine.analyze_content = AsyncMock(return_value={
        "semantic_analysis": {
            "key_concepts": ["react", "typescript", "testing", "patterns", "architecture"],
            "complexity_score": 0.88,
            "technical_depth": "advanced",
            "learning_difficulty": "high",
            "prerequisites": ["javascript", "react_basics", "typescript", "testing_fundamentals"],
            "estimated_learning_time": "4-6 weeks"
        },
        "learning_insights": {
            "difficulty_level": "advanced",
            "prerequisites": ["javascript", "react_basics", "typescript", "testing_fundamentals"],
            "learning_path": ["basics", "components", "typescript", "testing", "patterns", "architecture"],
            "common_mistakes": [
                "Not using TypeScript properly",
                "Inadequate testing coverage",
                "Poor architectural decisions",
                "Not following patterns consistently"
            ],
            "best_practices": [
                "Use TypeScript for type safety",
                "Implement comprehensive testing",
                "Follow established patterns",
                "Use proper architecture"
            ]
        },
        "content_quality": {
            "completeness_score": 0.92,
            "accuracy_score": 0.96,
            "clarity_score": 0.88,
            "practical_applicability": 0.94
        }
    })
    
    mock_engine.track_usage = AsyncMock(return_value={
        "success": True,
        "usage_id": "integration_test_usage_123",
        "timestamp": datetime.now(),
        "processing_time": 1.8
    })
    
    mock_engine.get_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add advanced TypeScript patterns",
            "Include comprehensive testing examples",
            "Add architectural decision records",
            "Include performance optimization techniques",
            "Add accessibility guidelines",
            "Include error handling patterns"
        ],
        "related_topics": [
            "Advanced TypeScript patterns",
            "React testing best practices",
            "Software architecture",
            "Performance optimization",
            "Accessibility in React"
        ],
        "learning_resources": [
            "Advanced React patterns",
            "TypeScript handbook",
            "Testing guide",
            "Architecture patterns"
        ]
    })
    
    return mock_engine


@pytest.fixture
def mock_intelligence_engine():
    """Mock intelligence engine with comprehensive responses."""
    mock_engine = Mock()
    mock_engine.categorize_content = AsyncMock(return_value={
        "primary_category": "frontend_framework",
        "subcategories": ["react", "typescript", "testing", "architecture", "patterns"],
        "confidence": 0.98,
        "related_technologies": ["typescript", "nextjs", "jest", "testing-library", "webpack", "storybook"],
        "complexity_assessment": {
            "overall_complexity": "advanced",
            "technical_depth": "expert",
            "learning_curve": "steep",
            "prerequisite_knowledge": ["javascript", "react_basics", "typescript", "testing_fundamentals", "html", "css"]
        },
        "content_analysis": {
            "main_topics": ["patterns", "typescript", "testing", "architecture"],
            "target_audience": "expert_developers",
            "practical_focus": "production_ready"
        }
    })
    
    mock_engine.generate_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add advanced TypeScript patterns",
            "Include comprehensive testing examples",
            "Add architectural decision records",
            "Include performance optimization techniques",
            "Add accessibility guidelines",
            "Include error handling patterns"
        ],
        "related_topics": [
            "Advanced TypeScript patterns",
            "React testing best practices",
            "Software architecture",
            "Performance optimization",
            "Accessibility in React"
        ],
        "technology_suggestions": [
            "Consider adding Next.js for SSR",
            "Include Redux for complex state management",
            "Add Storybook for component development",
            "Include Jest for testing",
            "Add Webpack for bundling optimization",
            "Include ESLint for code quality"
        ],
        "complexity_insights": {
            "current_level": "advanced",
            "next_level": "expert",
            "progression_path": ["typescript", "testing", "patterns", "architecture", "advanced_patterns"]
        }
    })
    
    return mock_engine


@pytest.fixture
def mock_semantic_analyzer():
    """Mock semantic analyzer with comprehensive responses."""
    mock_analyzer = Mock()
    mock_analyzer.analyze = AsyncMock(return_value={
        "entities": ["React", "TypeScript", "Testing", "Patterns", "Architecture"],
        "concepts": ["frontend", "framework", "development", "patterns", "architecture"],
        "sentiment": "positive",
        "complexity": "advanced",
        "technical_terms": [
            "functional components", "hooks", "type safety", 
            "testing patterns", "architectural decisions", "design patterns"
        ],
        "semantic_similarity": {
            "related_concepts": ["vue", "angular", "svelte"],
            "similarity_scores": {"vue": 0.8, "angular": 0.6, "svelte": 0.7}
        },
        "content_structure": {
            "has_examples": True,
            "has_best_practices": True,
            "has_architecture_guidance": True,
            "completeness_score": 0.90
        }
    })
    return mock_analyzer


@pytest.fixture
def mock_usage_tracker():
    """Mock usage tracker with comprehensive responses."""
    mock_tracker = Mock()
    mock_tracker.track = AsyncMock(return_value={
        "success": True,
        "tracking_id": "integration_track_123",
        "timestamp": datetime.now(),
        "processing_time": 1.8,
        "content_length": 2000,
        "sections_count": 3
    })
    
    mock_tracker.get_usage_stats = AsyncMock(return_value={
        "total_requests": 200,
        "success_rate": 0.97,
        "avg_processing_time": 1.5,
        "popular_topics": ["react", "typescript", "testing", "patterns"],
        "user_feedback": {
            "positive": 0.88,
            "neutral": 0.10,
            "negative": 0.02
        }
    })
    
    return mock_tracker


class TestLearningIntelligenceIntegrationRuleTransformer:
    """Test learning and intelligence integration in RuleTransformer."""

    @pytest.fixture
    def rule_transformer(self, mock_learning_engine, mock_intelligence_engine, 
                        mock_semantic_analyzer, mock_usage_tracker):
        """Create RuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence, \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer') as mock_semantic, \
             patch('rules_maker.transformers.rule_transformer.UsageTracker') as mock_usage:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            mock_semantic.return_value = mock_semantic_analyzer
            mock_usage.return_value = mock_usage_tracker
            
            return RuleTransformer()

    @pytest.mark.asyncio
    async def test_learning_engine_integration(self, rule_transformer, sample_content_sections):
        """Test learning engine integration in RuleTransformer."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_insights' in result
        assert 'semantic_analysis' in result
        assert 'usage_tracking' in result
        
        # Verify learning engine methods were called
        rule_transformer.learning_engine.analyze_content.assert_called_once()
        rule_transformer.learning_engine.track_usage.assert_called_once()
        rule_transformer.learning_engine.get_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligence_engine_integration(self, rule_transformer, sample_content_sections):
        """Test intelligence engine integration in RuleTransformer."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligent_categorization' in result
        assert 'intelligent_recommendations' in result
        
        # Verify intelligence engine methods were called
        rule_transformer.intelligence_engine.categorize_content.assert_called_once()
        rule_transformer.intelligence_engine.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_analyzer_integration(self, rule_transformer, sample_content_sections):
        """Test semantic analyzer integration in RuleTransformer."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'semantic_analysis' in result
        
        # Verify semantic analyzer was called
        rule_transformer.semantic_analyzer.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_usage_tracker_integration(self, rule_transformer, sample_content_sections):
        """Test usage tracker integration in RuleTransformer."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'usage_tracking' in result
        
        # Verify usage tracker was called
        rule_transformer.usage_tracker.track.assert_called_once()


class TestLearningIntelligenceIntegrationMLCursorTransformer:
    """Test learning and intelligence integration in MLCursorTransformer."""

    @pytest.fixture
    def ml_cursor_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create MLCursorTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return MLCursorTransformer()

    @pytest.mark.asyncio
    async def test_learning_engine_integration(self, ml_cursor_transformer, sample_content_sections):
        """Test learning engine integration in MLCursorTransformer."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_insights' in result
        assert 'semantic_analysis' in result
        assert 'usage_tracking' in result
        
        # Verify learning engine methods were called
        ml_cursor_transformer.learning_engine.analyze_content.assert_called_once()
        ml_cursor_transformer.learning_engine.track_usage.assert_called_once()
        ml_cursor_transformer.learning_engine.get_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligence_engine_integration(self, ml_cursor_transformer, sample_content_sections):
        """Test intelligence engine integration in MLCursorTransformer."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligent_categorization' in result
        assert 'intelligent_recommendations' in result
        
        # Verify intelligence engine methods were called
        ml_cursor_transformer.intelligence_engine.categorize_content.assert_called_once()
        ml_cursor_transformer.intelligence_engine.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_ml_quality_with_learning_intelligence(self, ml_cursor_transformer, sample_content_sections):
        """Test ML quality assessment with learning and intelligence integration."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'ml_quality' in result
        
        ml_quality = result['ml_quality']
        assert 'cursor_rules_compliance' in ml_quality
        assert 'cursor_rules_insights' in ml_quality
        assert 'cursor_rules_recommendations' in ml_quality


class TestLearningIntelligenceIntegrationCursorRuleTransformer:
    """Test learning and intelligence integration in CursorRuleTransformer."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_learning_engine_integration(self, cursor_rule_transformer, sample_content_sections):
        """Test learning engine integration in CursorRuleTransformer."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_enhancement' in result
        
        learning_enhancement = result['learning_enhancement']
        assert 'semantic_analysis' in learning_enhancement
        assert 'learning_insights' in learning_enhancement
        assert 'recommendations' in learning_enhancement
        
        # Verify learning engine methods were called
        cursor_rule_transformer.learning_engine.analyze_content.assert_called_once()
        cursor_rule_transformer.learning_engine.track_usage.assert_called_once()
        cursor_rule_transformer.learning_engine.get_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligence_engine_integration(self, cursor_rule_transformer, sample_content_sections):
        """Test intelligence engine integration in CursorRuleTransformer."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligence_enhancement' in result
        
        intelligence_enhancement = result['intelligence_enhancement']
        assert 'categorization' in intelligence_enhancement
        assert 'enhancement_suggestions' in intelligence_enhancement
        assert 'related_topics' in intelligence_enhancement
        
        # Verify intelligence engine methods were called
        cursor_rule_transformer.intelligence_engine.categorize_content.assert_called_once()
        cursor_rule_transformer.intelligence_engine.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_cursor_rule_generation_with_learning_intelligence(self, cursor_rule_transformer, sample_content_sections):
        """Test cursor rule generation with learning and intelligence integration."""
        result = await cursor_rule_transformer.generate_cursor_rule(
            sample_content_sections, 
            "Learning-Intelligence Enhanced Guidelines"
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify learning and intelligence engines were called
        cursor_rule_transformer.learning_engine.analyze_content.assert_called_once()
        cursor_rule_transformer.intelligence_engine.categorize_content.assert_called_once()


class TestLearningIntelligenceConcurrentOperations:
    """Test learning and intelligence concurrent operations."""

    @pytest.fixture
    def rule_transformer(self, mock_learning_engine, mock_intelligence_engine, 
                        mock_semantic_analyzer, mock_usage_tracker):
        """Create RuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence, \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer') as mock_semantic, \
             patch('rules_maker.transformers.rule_transformer.UsageTracker') as mock_usage:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            mock_semantic.return_value = mock_semantic_analyzer
            mock_usage.return_value = mock_usage_tracker
            
            return RuleTransformer()

    @pytest.mark.asyncio
    async def test_concurrent_learning_operations(self, rule_transformer, sample_content_sections):
        """Test concurrent learning operations."""
        # Test concurrent learning engine operations
        tasks = [
            rule_transformer.learning_engine.analyze_content(sample_content_sections),
            rule_transformer.learning_engine.track_usage(sample_content_sections),
            rule_transformer.learning_engine.get_recommendations(sample_content_sections)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert results[0] is not None  # analyze_content result
        assert results[1] is not None  # track_usage result
        assert results[2] is not None  # get_recommendations result

    @pytest.mark.asyncio
    async def test_concurrent_intelligence_operations(self, rule_transformer, sample_content_sections):
        """Test concurrent intelligence operations."""
        # Test concurrent intelligence engine operations
        tasks = [
            rule_transformer.intelligence_engine.categorize_content(sample_content_sections),
            rule_transformer.intelligence_engine.generate_recommendations(sample_content_sections)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert results[0] is not None  # categorize_content result
        assert results[1] is not None  # generate_recommendations result

    @pytest.mark.asyncio
    async def test_concurrent_transformations(self, rule_transformer, sample_content_sections):
        """Test concurrent transformations with learning and intelligence."""
        # Create multiple content sections for concurrent processing
        content_sets = [sample_content_sections for _ in range(3)]
        
        # Run concurrent transformations
        tasks = [rule_transformer.transform(content) for content in content_sets]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'learning_insights' in result
            assert 'intelligent_categorization' in result
            assert 'semantic_analysis' in result
            assert 'usage_tracking' in result


class TestLearningIntelligenceGracefulDegradation:
    """Test learning and intelligence graceful degradation."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_learning_engine(self, sample_content_sections):
        """Test graceful degradation when learning engine is unavailable."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning:
            mock_learning.side_effect = Exception("Learning engine unavailable")
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without learning engine
            assert result is not None
            assert 'content' in result
            # Learning-specific fields should not be present
            assert 'learning_insights' not in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_intelligence_engine(self, sample_content_sections):
        """Test graceful degradation when intelligence engine is unavailable."""
        with patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence:
            mock_intelligence.side_effect = Exception("Intelligence engine unavailable")
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without intelligence engine
            assert result is not None
            assert 'content' in result
            # Intelligence-specific fields should not be present
            assert 'intelligent_categorization' not in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_semantic_analyzer(self, sample_content_sections):
        """Test graceful degradation when semantic analyzer is unavailable."""
        with patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer') as mock_semantic:
            mock_semantic.side_effect = Exception("Semantic analyzer unavailable")
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without semantic analyzer
            assert result is not None
            assert 'content' in result
            # Semantic analysis should still be present from learning engine
            assert 'semantic_analysis' in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_usage_tracker(self, sample_content_sections):
        """Test graceful degradation when usage tracker is unavailable."""
        with patch('rules_maker.transformers.rule_transformer.UsageTracker') as mock_usage:
            mock_usage.side_effect = Exception("Usage tracker unavailable")
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without usage tracker
            assert result is not None
            assert 'content' in result
            # Usage tracking should still be present from learning engine
            assert 'usage_tracking' in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_all_modules(self, sample_content_sections):
        """Test graceful degradation when all modules are unavailable."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence, \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer') as mock_semantic, \
             patch('rules_maker.transformers.rule_transformer.UsageTracker') as mock_usage:
            
            mock_learning.side_effect = Exception("Learning engine unavailable")
            mock_intelligence.side_effect = Exception("Intelligence engine unavailable")
            mock_semantic.side_effect = Exception("Semantic analyzer unavailable")
            mock_usage.side_effect = Exception("Usage tracker unavailable")
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work with basic functionality
            assert result is not None
            assert 'content' in result
            # Enhanced fields should not be present
            assert 'learning_insights' not in result
            assert 'intelligent_categorization' not in result


class TestLearningIntelligenceDataFlow:
    """Test learning and intelligence data flow."""

    @pytest.fixture
    def rule_transformer(self, mock_learning_engine, mock_intelligence_engine, 
                        mock_semantic_analyzer, mock_usage_tracker):
        """Create RuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence, \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer') as mock_semantic, \
             patch('rules_maker.transformers.rule_transformer.UsageTracker') as mock_usage:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            mock_semantic.return_value = mock_semantic_analyzer
            mock_usage.return_value = mock_usage_tracker
            
            return RuleTransformer()

    @pytest.mark.asyncio
    async def test_learning_data_flow(self, rule_transformer, sample_content_sections):
        """Test learning data flow through the system."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        
        # Check learning insights data flow
        learning_insights = result['learning_insights']
        assert 'difficulty_level' in learning_insights
        assert 'prerequisites' in learning_insights
        assert 'learning_path' in learning_insights
        assert 'common_mistakes' in learning_insights
        assert 'best_practices' in learning_insights
        
        # Check semantic analysis data flow
        semantic_analysis = result['semantic_analysis']
        assert 'key_concepts' in semantic_analysis
        assert 'complexity_score' in semantic_analysis
        assert 'technical_depth' in semantic_analysis
        
        # Check usage tracking data flow
        usage_tracking = result['usage_tracking']
        assert 'success' in usage_tracking
        assert 'usage_id' in usage_tracking
        assert 'timestamp' in usage_tracking
        assert 'processing_time' in usage_tracking

    @pytest.mark.asyncio
    async def test_intelligence_data_flow(self, rule_transformer, sample_content_sections):
        """Test intelligence data flow through the system."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        
        # Check intelligent categorization data flow
        intelligent_categorization = result['intelligent_categorization']
        assert 'primary_category' in intelligent_categorization
        assert 'subcategories' in intelligent_categorization
        assert 'confidence' in intelligent_categorization
        assert 'related_technologies' in intelligent_categorization
        assert 'complexity_assessment' in intelligent_categorization
        assert 'content_analysis' in intelligent_categorization
        
        # Check intelligent recommendations data flow
        intelligent_recommendations = result['intelligent_recommendations']
        assert 'enhancement_suggestions' in intelligent_recommendations
        assert 'related_topics' in intelligent_recommendations
        assert 'technology_suggestions' in intelligent_recommendations
        assert 'complexity_insights' in intelligent_recommendations

    @pytest.mark.asyncio
    async def test_combined_learning_intelligence_data_flow(self, rule_transformer, sample_content_sections):
        """Test combined learning and intelligence data flow."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        
        # Verify all expected fields are present
        expected_fields = [
            'learning_insights', 'semantic_analysis', 'usage_tracking',
            'intelligent_categorization', 'intelligent_recommendations'
        ]
        
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"
            assert result[field] is not None, f"Field {field} is None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
