"""
Tests for RuleTransformer enhancements with learning and intelligence integration.

Tests the enhanced RuleTransformer with cursor rules knowledge, learning engine,
intelligence engine, semantic analysis, and usage tracking capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from rules_maker.transformers.rule_transformer import RuleTransformer
from rules_maker.models import ContentSection, ScrapingResult, ScrapingStatus


@pytest.fixture
def sample_content_sections():
    """Sample content sections for testing."""
    return [
        ContentSection(
            title="React Component Structure",
            content="Guidelines for React component structure and organization",
            section_type="code_example",
            metadata={"language": "javascript", "framework": "react", "topic": "components"}
        ),
        ContentSection(
            title="State Management",
            content="Best practices for React state management with hooks",
            section_type="guideline",
            metadata={"topic": "state", "framework": "react", "complexity": "intermediate"}
        ),
        ContentSection(
            title="Performance Optimization",
            content="React performance optimization techniques and patterns",
            section_type="guideline",
            metadata={"topic": "performance", "framework": "react", "complexity": "advanced"}
        )
    ]


@pytest.fixture
def mock_learning_engine():
    """Mock learning engine with comprehensive responses."""
    mock_engine = Mock()
    mock_engine.analyze_content = AsyncMock(return_value={
        "semantic_analysis": {
            "key_concepts": ["react", "components", "state", "performance", "hooks"],
            "complexity_score": 0.75,
            "technical_depth": "intermediate",
            "learning_difficulty": "moderate",
            "prerequisites": ["javascript", "html", "css", "es6"],
            "estimated_learning_time": "2-3 weeks"
        },
        "learning_insights": {
            "difficulty_level": "intermediate",
            "prerequisites": ["javascript", "html", "css"],
            "learning_path": ["basics", "components", "state", "performance"],
            "common_mistakes": [
                "Not using keys in lists",
                "Mutating state directly",
                "Not optimizing re-renders"
            ],
            "best_practices": [
                "Use functional components",
                "Implement proper error boundaries",
                "Use TypeScript for type safety"
            ]
        },
        "content_quality": {
            "completeness_score": 0.85,
            "accuracy_score": 0.90,
            "clarity_score": 0.80,
            "practical_applicability": 0.88
        }
    })
    
    mock_engine.track_usage = AsyncMock(return_value={
        "success": True,
        "usage_id": "test_usage_123",
        "timestamp": datetime.now(),
        "processing_time": 1.2
    })
    
    mock_engine.get_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add TypeScript integration examples",
            "Include testing patterns with Jest and React Testing Library",
            "Add accessibility guidelines and ARIA patterns",
            "Include error handling and loading states",
            "Add performance monitoring and profiling techniques"
        ],
        "related_topics": [
            "React Hooks advanced patterns",
            "Context API and state management",
            "Server-side rendering with Next.js",
            "React performance profiling"
        ],
        "learning_resources": [
            "React official documentation",
            "Advanced React patterns",
            "React performance optimization guide"
        ]
    })
    
    return mock_engine


@pytest.fixture
def mock_intelligence_engine():
    """Mock intelligence engine with comprehensive responses."""
    mock_engine = Mock()
    mock_engine.categorize_content = AsyncMock(return_value={
        "primary_category": "frontend_framework",
        "subcategories": ["react", "javascript", "ui_development", "state_management"],
        "confidence": 0.95,
        "related_technologies": ["typescript", "nextjs", "redux", "jest", "storybook"],
        "complexity_assessment": {
            "overall_complexity": "intermediate",
            "technical_depth": "moderate",
            "learning_curve": "gradual",
            "prerequisite_knowledge": ["javascript", "html", "css"]
        },
        "content_analysis": {
            "main_topics": ["components", "state", "performance"],
            "target_audience": "intermediate_developers",
            "practical_focus": "production_ready"
        }
    })
    
    mock_engine.generate_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add TypeScript integration examples",
            "Include testing patterns and best practices",
            "Add accessibility guidelines and ARIA patterns",
            "Include error handling and loading states",
            "Add performance monitoring and profiling techniques"
        ],
        "related_topics": [
            "React Hooks advanced patterns",
            "Context API and state management",
            "Server-side rendering with Next.js",
            "React performance profiling"
        ],
        "technology_suggestions": [
            "Consider adding Next.js for SSR",
            "Include Redux for complex state management",
            "Add Storybook for component development",
            "Include Jest for testing"
        ],
        "complexity_insights": {
            "current_level": "intermediate",
            "next_level": "advanced",
            "progression_path": ["hooks", "context", "performance", "testing"]
        }
    })
    
    return mock_engine


@pytest.fixture
def mock_semantic_analyzer():
    """Mock semantic analyzer with comprehensive responses."""
    mock_analyzer = Mock()
    mock_analyzer.analyze = AsyncMock(return_value={
        "entities": ["React", "JavaScript", "Components", "State", "Hooks", "Performance"],
        "concepts": ["frontend", "framework", "development", "optimization", "patterns"],
        "sentiment": "positive",
        "complexity": "intermediate",
        "technical_terms": [
            "functional components", "hooks", "state management", 
            "performance optimization", "re-rendering", "memoization"
        ],
        "semantic_similarity": {
            "related_concepts": ["vue", "angular", "svelte"],
            "similarity_scores": {"vue": 0.8, "angular": 0.6, "svelte": 0.7}
        },
        "content_structure": {
            "has_examples": True,
            "has_best_practices": True,
            "has_performance_tips": True,
            "completeness_score": 0.85
        }
    })
    return mock_analyzer


@pytest.fixture
def mock_usage_tracker():
    """Mock usage tracker with comprehensive responses."""
    mock_tracker = Mock()
    mock_tracker.track = AsyncMock(return_value={
        "success": True,
        "tracking_id": "track_123",
        "timestamp": datetime.now(),
        "processing_time": 1.2,
        "content_length": 1500,
        "sections_count": 3
    })
    
    mock_tracker.get_usage_stats = AsyncMock(return_value={
        "total_requests": 150,
        "success_rate": 0.96,
        "avg_processing_time": 1.3,
        "popular_topics": ["react", "javascript", "performance"],
        "user_feedback": {
            "positive": 0.85,
            "neutral": 0.12,
            "negative": 0.03
        }
    })
    
    return mock_tracker


class TestRuleTransformerLearningIntegration:
    """Test RuleTransformer learning engine integration."""

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
    async def test_learning_engine_initialization(self, rule_transformer):
        """Test learning engine initialization."""
        assert rule_transformer.learning_engine is not None
        assert hasattr(rule_transformer.learning_engine, 'analyze_content')
        assert hasattr(rule_transformer.learning_engine, 'track_usage')
        assert hasattr(rule_transformer.learning_engine, 'get_recommendations')

    @pytest.mark.asyncio
    async def test_learning_content_analysis(self, rule_transformer, sample_content_sections):
        """Test learning content analysis integration."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_insights' in result
        assert 'semantic_analysis' in result
        
        learning_insights = result['learning_insights']
        assert 'difficulty_level' in learning_insights
        assert 'prerequisites' in learning_insights
        assert 'learning_path' in learning_insights
        assert 'common_mistakes' in learning_insights
        assert 'best_practices' in learning_insights
        
        semantic_analysis = result['semantic_analysis']
        assert 'key_concepts' in semantic_analysis
        assert 'complexity_score' in semantic_analysis
        assert 'technical_depth' in semantic_analysis
        
        # Verify learning engine was called
        rule_transformer.learning_engine.analyze_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_usage_tracking(self, rule_transformer, sample_content_sections):
        """Test learning usage tracking integration."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'usage_tracking' in result
        
        usage_tracking = result['usage_tracking']
        assert 'success' in usage_tracking
        assert 'usage_id' in usage_tracking
        assert 'timestamp' in usage_tracking
        assert 'processing_time' in usage_tracking
        
        # Verify usage tracking was called
        rule_transformer.learning_engine.track_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_recommendations(self, rule_transformer, sample_content_sections):
        """Test learning recommendations integration."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_recommendations' in result
        
        recommendations = result['learning_recommendations']
        assert 'enhancement_suggestions' in recommendations
        assert 'related_topics' in recommendations
        assert 'learning_resources' in recommendations
        
        # Verify recommendations were generated
        rule_transformer.learning_engine.get_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_content_quality_assessment(self, rule_transformer, sample_content_sections):
        """Test learning content quality assessment."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'content_quality' in result
        
        content_quality = result['content_quality']
        assert 'completeness_score' in content_quality
        assert 'accuracy_score' in content_quality
        assert 'clarity_score' in content_quality
        assert 'practical_applicability' in content_quality


class TestRuleTransformerIntelligenceIntegration:
    """Test RuleTransformer intelligence engine integration."""

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
    async def test_intelligence_engine_initialization(self, rule_transformer):
        """Test intelligence engine initialization."""
        assert rule_transformer.intelligence_engine is not None
        assert hasattr(rule_transformer.intelligence_engine, 'categorize_content')
        assert hasattr(rule_transformer.intelligence_engine, 'generate_recommendations')

    @pytest.mark.asyncio
    async def test_intelligent_categorization(self, rule_transformer, sample_content_sections):
        """Test intelligent content categorization."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligent_categorization' in result
        
        categorization = result['intelligent_categorization']
        assert 'primary_category' in categorization
        assert 'subcategories' in categorization
        assert 'confidence' in categorization
        assert 'related_technologies' in categorization
        assert 'complexity_assessment' in categorization
        assert 'content_analysis' in categorization
        
        # Verify intelligence engine was called
        rule_transformer.intelligence_engine.categorize_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligent_recommendations(self, rule_transformer, sample_content_sections):
        """Test intelligent recommendations generation."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligent_recommendations' in result
        
        recommendations = result['intelligent_recommendations']
        assert 'enhancement_suggestions' in recommendations
        assert 'related_topics' in recommendations
        assert 'technology_suggestions' in recommendations
        assert 'complexity_insights' in recommendations
        
        # Verify recommendations were generated
        rule_transformer.intelligence_engine.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligence_complexity_assessment(self, rule_transformer, sample_content_sections):
        """Test intelligence complexity assessment."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        categorization = result['intelligent_categorization']
        
        complexity_assessment = categorization['complexity_assessment']
        assert 'overall_complexity' in complexity_assessment
        assert 'technical_depth' in complexity_assessment
        assert 'learning_curve' in complexity_assessment
        assert 'prerequisite_knowledge' in complexity_assessment

    @pytest.mark.asyncio
    async def test_intelligence_technology_suggestions(self, rule_transformer, sample_content_sections):
        """Test intelligence technology suggestions."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        recommendations = result['intelligent_recommendations']
        
        technology_suggestions = recommendations['technology_suggestions']
        assert isinstance(technology_suggestions, list)
        assert len(technology_suggestions) > 0
        
        # Should suggest relevant technologies
        suggested_techs = [suggestion.lower() for suggestion in technology_suggestions]
        assert any('next' in tech for tech in suggested_techs)
        assert any('redux' in tech for tech in suggested_techs)


class TestRuleTransformerSemanticAnalysis:
    """Test RuleTransformer semantic analysis integration."""

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
    async def test_semantic_analyzer_initialization(self, rule_transformer):
        """Test semantic analyzer initialization."""
        assert rule_transformer.semantic_analyzer is not None
        assert hasattr(rule_transformer.semantic_analyzer, 'analyze')

    @pytest.mark.asyncio
    async def test_semantic_analysis_integration(self, rule_transformer, sample_content_sections):
        """Test semantic analysis integration."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'semantic_analysis' in result
        
        semantic_analysis = result['semantic_analysis']
        assert 'entities' in semantic_analysis
        assert 'concepts' in semantic_analysis
        assert 'sentiment' in semantic_analysis
        assert 'complexity' in semantic_analysis
        assert 'technical_terms' in semantic_analysis
        assert 'semantic_similarity' in semantic_analysis
        assert 'content_structure' in semantic_analysis
        
        # Verify semantic analyzer was called
        rule_transformer.semantic_analyzer.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_entities_extraction(self, rule_transformer, sample_content_sections):
        """Test semantic entities extraction."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        semantic_analysis = result['semantic_analysis']
        
        entities = semantic_analysis['entities']
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert 'React' in entities
        assert 'JavaScript' in entities

    @pytest.mark.asyncio
    async def test_semantic_concepts_identification(self, rule_transformer, sample_content_sections):
        """Test semantic concepts identification."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        semantic_analysis = result['semantic_analysis']
        
        concepts = semantic_analysis['concepts']
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert 'frontend' in concepts
        assert 'framework' in concepts

    @pytest.mark.asyncio
    async def test_semantic_similarity_analysis(self, rule_transformer, sample_content_sections):
        """Test semantic similarity analysis."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        semantic_analysis = result['semantic_analysis']
        
        similarity = semantic_analysis['semantic_similarity']
        assert 'related_concepts' in similarity
        assert 'similarity_scores' in similarity
        
        related_concepts = similarity['related_concepts']
        assert isinstance(related_concepts, list)
        assert 'vue' in related_concepts
        assert 'angular' in related_concepts


class TestRuleTransformerUsageTracking:
    """Test RuleTransformer usage tracking integration."""

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
    async def test_usage_tracker_initialization(self, rule_transformer):
        """Test usage tracker initialization."""
        assert rule_transformer.usage_tracker is not None
        assert hasattr(rule_transformer.usage_tracker, 'track')
        assert hasattr(rule_transformer.usage_tracker, 'get_usage_stats')

    @pytest.mark.asyncio
    async def test_usage_tracking_integration(self, rule_transformer, sample_content_sections):
        """Test usage tracking integration."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'usage_tracking' in result
        
        usage_tracking = result['usage_tracking']
        assert 'success' in usage_tracking
        assert 'tracking_id' in usage_tracking
        assert 'timestamp' in usage_tracking
        assert 'processing_time' in usage_tracking
        assert 'content_length' in usage_tracking
        assert 'sections_count' in usage_tracking
        
        # Verify usage tracker was called
        rule_transformer.usage_tracker.track.assert_called_once()

    @pytest.mark.asyncio
    async def test_usage_stats_retrieval(self, rule_transformer):
        """Test usage stats retrieval."""
        stats = await rule_transformer.usage_tracker.get_usage_stats()
        
        assert stats is not None
        assert 'total_requests' in stats
        assert 'success_rate' in stats
        assert 'avg_processing_time' in stats
        assert 'popular_topics' in stats
        assert 'user_feedback' in stats
        
        # Verify usage stats method was called
        rule_transformer.usage_tracker.get_usage_stats.assert_called_once()


class TestRuleTransformerCursorRulesIntegration:
    """Test RuleTransformer cursor rules integration."""

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
    async def test_cursor_rules_validation(self, rule_transformer):
        """Test cursor rules validation functionality."""
        # Test valid cursor rules
        valid_rules = """
        ---
        description: React development guidelines
        globs: ["**/*.jsx", "**/*.tsx"]
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        - Prefer hooks over class components
        
        ## Best Practices
        - Use TypeScript for type safety
        - Implement proper error boundaries
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(valid_rules)
        assert is_valid is True
        
        # Test invalid cursor rules
        invalid_rules = "Just plain text without proper structure"
        is_valid = rule_transformer._validate_cursor_rules_structure(invalid_rules)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement(self, rule_transformer):
        """Test cursor rules enhancement functionality."""
        # Test incomplete cursor rules
        incomplete_rules = """
        ---
        description: Basic React guidelines
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = rule_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
        assert enhanced_rules is not None
        assert "## Key Principles" in enhanced_rules
        assert "## Best Practices" in enhanced_rules
        assert "## Error Handling" in enhanced_rules
        assert "## Performance" in enhanced_rules
        assert "## Critical Instructions" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_metadata_generation(self, rule_transformer, sample_content_sections):
        """Test cursor rules metadata generation."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'cursor_rules_metadata' in result
        
        metadata = result['cursor_rules_metadata']
        assert 'structure_valid' in metadata
        assert 'has_frontmatter' in metadata
        assert 'has_required_sections' in metadata
        assert 'quality_score' in metadata


class TestRuleTransformerGracefulDegradation:
    """Test RuleTransformer graceful degradation when modules are unavailable."""

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


class TestRuleTransformerConcurrentOperations:
    """Test RuleTransformer concurrent operations."""

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
    async def test_concurrent_transformations(self, rule_transformer, sample_content_sections):
        """Test concurrent transformations."""
        # Create multiple content sections for concurrent processing
        content_sets = [sample_content_sections for _ in range(3)]
        
        # Run concurrent transformations
        tasks = [rule_transformer.transform(content) for content in content_sets]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'content' in result
            assert 'learning_insights' in result
            assert 'intelligent_categorization' in result

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
