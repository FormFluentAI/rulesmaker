"""
Tests for MLCursorTransformer enhancements with cursor rules validation and ML quality.

Tests the enhanced MLCursorTransformer with cursor rules validation, enhancement,
ML quality assessment, and learning/intelligence integration capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
from rules_maker.models import ContentSection, ScrapingResult, ScrapingStatus


@pytest.fixture
def sample_content_sections():
    """Sample content sections for testing."""
    return [
        ContentSection(
            title="React Component Best Practices",
            content="Guidelines for creating maintainable React components",
            section_type="code_example",
            metadata={"language": "javascript", "framework": "react", "topic": "components", "complexity": "intermediate"}
        ),
        ContentSection(
            title="State Management Patterns",
            content="Advanced state management patterns with React hooks and context",
            section_type="guideline",
            metadata={"topic": "state", "framework": "react", "complexity": "advanced", "patterns": ["hooks", "context"]}
        ),
        ContentSection(
            title="Performance Optimization",
            content="React performance optimization techniques and profiling",
            section_type="guideline",
            metadata={"topic": "performance", "framework": "react", "complexity": "advanced", "tools": ["profiler", "memo"]}
        )
    ]


@pytest.fixture
def mock_learning_engine():
    """Mock learning engine with comprehensive responses."""
    mock_engine = Mock()
    mock_engine.analyze_content = AsyncMock(return_value={
        "semantic_analysis": {
            "key_concepts": ["react", "components", "state", "performance", "hooks", "context"],
            "complexity_score": 0.85,
            "technical_depth": "advanced",
            "learning_difficulty": "high",
            "prerequisites": ["javascript", "react_basics", "es6"],
            "estimated_learning_time": "3-4 weeks"
        },
        "learning_insights": {
            "difficulty_level": "advanced",
            "prerequisites": ["javascript", "react_basics", "es6"],
            "learning_path": ["basics", "components", "state", "performance", "advanced_patterns"],
            "common_mistakes": [
                "Not using useCallback for expensive functions",
                "Creating objects in render",
                "Not memoizing expensive calculations",
                "Overusing useEffect"
            ],
            "best_practices": [
                "Use functional components with hooks",
                "Implement proper error boundaries",
                "Use TypeScript for type safety",
                "Profile before optimizing"
            ]
        },
        "content_quality": {
            "completeness_score": 0.90,
            "accuracy_score": 0.95,
            "clarity_score": 0.85,
            "practical_applicability": 0.92
        }
    })
    
    mock_engine.track_usage = AsyncMock(return_value={
        "success": True,
        "usage_id": "ml_test_usage_123",
        "timestamp": datetime.now(),
        "processing_time": 1.5
    })
    
    mock_engine.get_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add TypeScript integration examples",
            "Include testing patterns with Jest and React Testing Library",
            "Add accessibility guidelines and ARIA patterns",
            "Include error handling and loading states",
            "Add performance monitoring and profiling techniques",
            "Include advanced React patterns like render props and HOCs"
        ],
        "related_topics": [
            "React Hooks advanced patterns",
            "Context API and state management",
            "Server-side rendering with Next.js",
            "React performance profiling",
            "Advanced React patterns"
        ],
        "learning_resources": [
            "React official documentation",
            "Advanced React patterns",
            "React performance optimization guide",
            "React testing best practices"
        ]
    })
    
    return mock_engine


@pytest.fixture
def mock_intelligence_engine():
    """Mock intelligence engine with comprehensive responses."""
    mock_engine = Mock()
    mock_engine.categorize_content = AsyncMock(return_value={
        "primary_category": "frontend_framework",
        "subcategories": ["react", "javascript", "ui_development", "state_management", "performance"],
        "confidence": 0.98,
        "related_technologies": ["typescript", "nextjs", "redux", "jest", "storybook", "webpack"],
        "complexity_assessment": {
            "overall_complexity": "advanced",
            "technical_depth": "high",
            "learning_curve": "steep",
            "prerequisite_knowledge": ["javascript", "react_basics", "es6", "html", "css"]
        },
        "content_analysis": {
            "main_topics": ["components", "state", "performance", "patterns"],
            "target_audience": "advanced_developers",
            "practical_focus": "production_ready"
        }
    })
    
    mock_engine.generate_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add TypeScript integration examples",
            "Include testing patterns and best practices",
            "Add accessibility guidelines and ARIA patterns",
            "Include error handling and loading states",
            "Add performance monitoring and profiling techniques",
            "Include advanced React patterns and anti-patterns"
        ],
        "related_topics": [
            "React Hooks advanced patterns",
            "Context API and state management",
            "Server-side rendering with Next.js",
            "React performance profiling",
            "Advanced React patterns and anti-patterns"
        ],
        "technology_suggestions": [
            "Consider adding Next.js for SSR",
            "Include Redux for complex state management",
            "Add Storybook for component development",
            "Include Jest for testing",
            "Add Webpack for bundling optimization"
        ],
        "complexity_insights": {
            "current_level": "advanced",
            "next_level": "expert",
            "progression_path": ["hooks", "context", "performance", "testing", "advanced_patterns"]
        }
    })
    
    return mock_engine


class TestMLCursorTransformerInitialization:
    """Test MLCursorTransformer initialization and setup."""

    @pytest.fixture
    def ml_cursor_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create MLCursorTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return MLCursorTransformer()

    @pytest.mark.asyncio
    async def test_ml_cursor_transformer_initialization(self, ml_cursor_transformer):
        """Test MLCursorTransformer initialization."""
        assert ml_cursor_transformer is not None
        assert hasattr(ml_cursor_transformer, 'learning_engine')
        assert hasattr(ml_cursor_transformer, 'intelligence_engine')
        assert hasattr(ml_cursor_transformer, '_validate_cursor_rules_structure')
        assert hasattr(ml_cursor_transformer, '_enhance_cursor_rules_structure')

    @pytest.mark.asyncio
    async def test_learning_engine_integration(self, ml_cursor_transformer):
        """Test learning engine integration."""
        assert ml_cursor_transformer.learning_engine is not None
        assert hasattr(ml_cursor_transformer.learning_engine, 'analyze_content')
        assert hasattr(ml_cursor_transformer.learning_engine, 'track_usage')
        assert hasattr(ml_cursor_transformer.learning_engine, 'get_recommendations')

    @pytest.mark.asyncio
    async def test_intelligence_engine_integration(self, ml_cursor_transformer):
        """Test intelligence engine integration."""
        assert ml_cursor_transformer.intelligence_engine is not None
        assert hasattr(ml_cursor_transformer.intelligence_engine, 'categorize_content')
        assert hasattr(ml_cursor_transformer.intelligence_engine, 'generate_recommendations')


class TestMLCursorTransformerCursorRulesValidation:
    """Test MLCursorTransformer cursor rules validation functionality."""

    @pytest.fixture
    def ml_cursor_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create MLCursorTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return MLCursorTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_comprehensive(self, ml_cursor_transformer):
        """Test comprehensive cursor rules validation."""
        # Test high-quality cursor rules
        comprehensive_rules = """
        ---
        description: Comprehensive React development guidelines
        globs: ["**/*.jsx", "**/*.tsx", "**/*.js", "**/*.ts"]
        ---
        # React Development Guidelines
        
        ## Key Principles
        - Use functional components with hooks
        - Prefer composition over inheritance
        - Keep components small and focused
        - Follow single responsibility principle
        
        ## Code Style
        - Use camelCase for variables and functions
        - Use PascalCase for components
        - Use 2-space indentation
        - Use semicolons consistently
        - Use meaningful variable names
        
        ## Best Practices
        - Use TypeScript for type safety
        - Implement proper error boundaries
        - Use React.memo for performance optimization
        - Prefer controlled components
        - Use custom hooks for reusable logic
        
        ## Error Handling
        - Use try-catch blocks for async operations
        - Implement error boundaries for component errors
        - Log errors appropriately
        - Provide fallback UI for errors
        
        ## Performance
        - Use React.memo for expensive components
        - Use useMemo and useCallback appropriately
        - Avoid unnecessary re-renders
        - Profile components before optimizing
        - Use React.lazy for code splitting
        
        ## Critical Instructions
        - Never mutate props or state directly
        - Always use keys in lists
        - Handle loading and error states
        - Validate props with PropTypes or TypeScript
        - Test components thoroughly
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(comprehensive_rules)
        
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.9
        assert len(validation_result['missing_sections']) == 0
        assert 'quality_indicators' in validation_result
        assert 'structure_analysis' in validation_result

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_incomplete(self, ml_cursor_transformer):
        """Test cursor rules validation with incomplete rules."""
        # Test incomplete cursor rules
        incomplete_rules = """
        ---
        description: Basic React guidelines
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        - Use camelCase
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(incomplete_rules)
        
        assert validation_result['is_valid'] is False
        assert validation_result['score'] < 0.5
        assert len(validation_result['missing_sections']) > 0
        assert 'Key Principles' in validation_result['missing_sections']
        assert 'Best Practices' in validation_result['missing_sections']
        assert 'Error Handling' in validation_result['missing_sections']
        assert 'Performance' in validation_result['missing_sections']
        assert 'Critical Instructions' in validation_result['missing_sections']

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_invalid_structure(self, ml_cursor_transformer):
        """Test cursor rules validation with invalid structure."""
        # Test invalid cursor rules
        invalid_rules = "Just plain text without proper structure"
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(invalid_rules)
        
        assert validation_result['is_valid'] is False
        assert validation_result['score'] == 0.0
        assert 'missing_frontmatter' in validation_result['missing_sections']
        assert 'missing_title' in validation_result['missing_sections']

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_quality_indicators(self, ml_cursor_transformer):
        """Test cursor rules validation quality indicators."""
        # Test rules with quality indicators
        quality_rules = """
        ---
        description: High-quality React guidelines
        globs: ["**/*.jsx", "**/*.tsx"]
        ---
        # React Guidelines
        
        ## Key Principles
        - Use functional components with hooks
        - Keep components small and focused
        
        ## Code Style
        - Use camelCase for variables
        - Use PascalCase for components
        - Use 2-space indentation
        
        ## Best Practices
        - Use TypeScript for type safety
        - Implement proper error boundaries
        
        ## Error Handling
        - Use try-catch blocks
        - Implement error boundaries
        
        ## Performance
        - Use React.memo
        - Use useMemo appropriately
        
        ## Critical Instructions
        - Never mutate props
        - Always use keys in lists
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(quality_rules)
        
        assert validation_result['is_valid'] is True
        assert 'quality_indicators' in validation_result
        
        quality_indicators = validation_result['quality_indicators']
        assert 'has_examples' in quality_indicators
        assert 'has_best_practices' in quality_indicators
        assert 'has_performance_tips' in quality_indicators
        assert 'has_error_handling' in quality_indicators
        assert 'has_critical_instructions' in quality_indicators

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_scoring(self, ml_cursor_transformer):
        """Test cursor rules validation scoring system."""
        # Test different quality levels
        test_cases = [
            {
                "rules": """
                ---
                description: Perfect rules
                globs: ["**/*.jsx"]
                ---
                # Perfect Rules
                ## Key Principles
                ## Code Style
                ## Best Practices
                ## Error Handling
                ## Performance
                ## Critical Instructions
                """,
                "expected_score": 1.0
            },
            {
                "rules": """
                ---
                description: Good rules
                globs: ["**/*.jsx"]
                ---
                # Good Rules
                ## Key Principles
                ## Code Style
                ## Best Practices
                """,
                "expected_score": 0.5
            },
            {
                "rules": """
                ---
                description: Basic rules
                ---
                # Basic Rules
                ## Code Style
                """,
                "expected_score": 0.2
            }
        ]
        
        for test_case in test_cases:
            validation_result = ml_cursor_transformer._validate_cursor_rules_structure(test_case["rules"])
            assert validation_result['score'] >= test_case["expected_score"] - 0.1


class TestMLCursorTransformerCursorRulesEnhancement:
    """Test MLCursorTransformer cursor rules enhancement functionality."""

    @pytest.fixture
    def ml_cursor_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create MLCursorTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return MLCursorTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_basic(self, ml_cursor_transformer):
        """Test basic cursor rules enhancement."""
        # Test incomplete cursor rules
        incomplete_rules = """
        ---
        description: Basic React guidelines
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = ml_cursor_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
        assert enhanced_rules is not None
        assert "## Key Principles" in enhanced_rules
        assert "## Best Practices" in enhanced_rules
        assert "## Error Handling" in enhanced_rules
        assert "## Performance" in enhanced_rules
        assert "## Critical Instructions" in enhanced_rules
        
        # Original content should be preserved
        assert "## Code Style" in enhanced_rules
        assert "Use functional components" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_comprehensive(self, ml_cursor_transformer):
        """Test comprehensive cursor rules enhancement."""
        # Test rules missing multiple sections
        incomplete_rules = """
        ---
        description: React guidelines
        globs: ["**/*.jsx"]
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        - Use camelCase
        
        ## Best Practices
        - Use TypeScript
        """
        
        enhanced_rules = ml_cursor_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
        assert enhanced_rules is not None
        
        # Check that missing sections were added
        required_sections = ["Key Principles", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in enhanced_rules
        
        # Check that existing sections were preserved
        assert "## Code Style" in enhanced_rules
        assert "## Best Practices" in enhanced_rules
        assert "Use functional components" in enhanced_rules
        assert "Use TypeScript" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_frontmatter(self, ml_cursor_transformer):
        """Test cursor rules enhancement with frontmatter."""
        # Test rules without frontmatter
        no_frontmatter_rules = """
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = ml_cursor_transformer._enhance_cursor_rules_structure(no_frontmatter_rules)
        
        assert enhanced_rules is not None
        assert "---" in enhanced_rules
        assert "description:" in enhanced_rules
        assert "globs:" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_quality_improvement(self, ml_cursor_transformer):
        """Test cursor rules enhancement quality improvement."""
        # Test rules with basic content
        basic_rules = """
        ---
        description: Basic guidelines
        ---
        # Basic Guidelines
        
        ## Code Style
        - Use consistent formatting
        """
        
        enhanced_rules = ml_cursor_transformer._enhance_cursor_rules_structure(basic_rules)
        
        assert enhanced_rules is not None
        
        # Check that enhanced sections have meaningful content
        assert "functional components" in enhanced_rules.lower()
        assert "error handling" in enhanced_rules.lower()
        assert "performance" in enhanced_rules.lower()
        assert "best practices" in enhanced_rules.lower()


class TestMLCursorTransformerMLQualityAssessment:
    """Test MLCursorTransformer ML quality assessment with cursor rules."""

    @pytest.fixture
    def ml_cursor_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create MLCursorTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return MLCursorTransformer()

    @pytest.mark.asyncio
    async def test_ml_quality_with_cursor_rules_compliance(self, ml_cursor_transformer, sample_content_sections):
        """Test ML quality assessment with cursor rules compliance."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'ml_quality' in result
        
        ml_quality = result['ml_quality']
        assert 'cursor_rules_compliance' in ml_quality
        assert 'cursor_rules_insights' in ml_quality
        assert 'cursor_rules_recommendations' in ml_quality
        
        # Check cursor rules compliance
        compliance = ml_quality['cursor_rules_compliance']
        assert 'score' in compliance
        assert 'validation_status' in compliance
        assert 'missing_sections' in compliance
        assert 'quality_indicators' in compliance

    @pytest.mark.asyncio
    async def test_ml_quality_cursor_rules_insights(self, ml_cursor_transformer, sample_content_sections):
        """Test ML quality cursor rules insights."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        ml_quality = result['ml_quality']
        
        insights = ml_quality['cursor_rules_insights']
        assert 'technology_specific' in insights
        assert 'quality_assessment' in insights
        assert 'enhancement_opportunities' in insights
        
        # Check technology-specific insights
        tech_insights = insights['technology_specific']
        assert 'react' in tech_insights
        assert 'javascript' in tech_insights

    @pytest.mark.asyncio
    async def test_ml_quality_cursor_rules_recommendations(self, ml_cursor_transformer, sample_content_sections):
        """Test ML quality cursor rules recommendations."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        ml_quality = result['ml_quality']
        
        recommendations = ml_quality['cursor_rules_recommendations']
        assert 'structure_improvements' in recommendations
        assert 'content_enhancements' in recommendations
        assert 'technology_suggestions' in recommendations
        
        # Check that recommendations are actionable
        structure_improvements = recommendations['structure_improvements']
        assert isinstance(structure_improvements, list)
        assert len(structure_improvements) > 0

    @pytest.mark.asyncio
    async def test_ml_quality_technology_specific_insights(self, ml_cursor_transformer):
        """Test ML quality technology-specific insights."""
        # Test React-specific content
        react_content = [ContentSection(
            title="React Component",
            content="React component example with hooks",
            section_type="code_example",
            metadata={"framework": "react", "language": "javascript", "topic": "components"}
        )]
        
        result = await ml_cursor_transformer.transform(react_content)
        
        assert result is not None
        ml_quality = result['ml_quality']
        insights = ml_quality['cursor_rules_insights']
        
        tech_insights = insights['technology_specific']
        assert 'react' in tech_insights
        
        react_insights = tech_insights['react']
        assert 'guidelines' in react_insights
        assert 'best_practices' in react_insights
        assert 'common_patterns' in react_insights

    @pytest.mark.asyncio
    async def test_ml_quality_learning_intelligence_integration(self, ml_cursor_transformer, sample_content_sections):
        """Test ML quality with learning and intelligence integration."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_insights' in result
        assert 'intelligent_categorization' in result
        assert 'ml_quality' in result
        
        # Verify learning and intelligence engines were called
        ml_cursor_transformer.learning_engine.analyze_content.assert_called_once()
        ml_cursor_transformer.intelligence_engine.categorize_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_ml_quality_cursor_rules_metadata(self, ml_cursor_transformer, sample_content_sections):
        """Test ML quality cursor rules metadata."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'cursor_rules_metadata' in result
        
        metadata = result['cursor_rules_metadata']
        assert 'validation_status' in metadata
        assert 'enhancement_applied' in metadata
        assert 'quality_score' in metadata
        assert 'technology_detected' in metadata


class TestMLCursorTransformerLearningIntelligenceIntegration:
    """Test MLCursorTransformer learning and intelligence integration."""

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
        """Test learning engine integration."""
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
        """Test intelligence engine integration."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligent_categorization' in result
        assert 'intelligent_recommendations' in result
        
        # Verify intelligence engine methods were called
        ml_cursor_transformer.intelligence_engine.categorize_content.assert_called_once()
        ml_cursor_transformer.intelligence_engine.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_intelligence_combined_insights(self, ml_cursor_transformer, sample_content_sections):
        """Test combined learning and intelligence insights."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        
        # Check learning insights
        learning_insights = result['learning_insights']
        assert 'difficulty_level' in learning_insights
        assert 'prerequisites' in learning_insights
        assert 'learning_path' in learning_insights
        assert 'common_mistakes' in learning_insights
        assert 'best_practices' in learning_insights
        
        # Check intelligence insights
        intelligent_categorization = result['intelligent_categorization']
        assert 'primary_category' in intelligent_categorization
        assert 'subcategories' in intelligent_categorization
        assert 'confidence' in intelligent_categorization
        assert 'related_technologies' in intelligent_categorization
        assert 'complexity_assessment' in intelligent_categorization

    @pytest.mark.asyncio
    async def test_learning_intelligence_recommendations(self, ml_cursor_transformer, sample_content_sections):
        """Test learning and intelligence recommendations."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        
        # Check learning recommendations
        learning_recommendations = result.get('learning_recommendations', {})
        if learning_recommendations:
            assert 'enhancement_suggestions' in learning_recommendations
            assert 'related_topics' in learning_recommendations
            assert 'learning_resources' in learning_recommendations
        
        # Check intelligence recommendations
        intelligent_recommendations = result['intelligent_recommendations']
        assert 'enhancement_suggestions' in intelligent_recommendations
        assert 'related_topics' in intelligent_recommendations
        assert 'technology_suggestions' in intelligent_recommendations
        assert 'complexity_insights' in intelligent_recommendations


class TestMLCursorTransformerGracefulDegradation:
    """Test MLCursorTransformer graceful degradation."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_learning_engine(self, sample_content_sections):
        """Test graceful degradation when learning engine is unavailable."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning:
            mock_learning.side_effect = Exception("Learning engine unavailable")
            
            transformer = MLCursorTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without learning engine
            assert result is not None
            assert 'content' in result
            # Learning-specific fields should not be present
            assert 'learning_insights' not in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_intelligence_engine(self, sample_content_sections):
        """Test graceful degradation when intelligence engine is unavailable."""
        with patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            mock_intelligence.side_effect = Exception("Intelligence engine unavailable")
            
            transformer = MLCursorTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without intelligence engine
            assert result is not None
            assert 'content' in result
            # Intelligence-specific fields should not be present
            assert 'intelligent_categorization' not in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_all_modules(self, sample_content_sections):
        """Test graceful degradation when all modules are unavailable."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.side_effect = Exception("Learning engine unavailable")
            mock_intelligence.side_effect = Exception("Intelligence engine unavailable")
            
            transformer = MLCursorTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work with basic functionality
            assert result is not None
            assert 'content' in result
            # Enhanced fields should not be present
            assert 'learning_insights' not in result
            assert 'intelligent_categorization' not in result


class TestMLCursorTransformerConcurrentOperations:
    """Test MLCursorTransformer concurrent operations."""

    @pytest.fixture
    def ml_cursor_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create MLCursorTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return MLCursorTransformer()

    @pytest.mark.asyncio
    async def test_concurrent_transformations(self, ml_cursor_transformer, sample_content_sections):
        """Test concurrent transformations."""
        # Create multiple content sections for concurrent processing
        content_sets = [sample_content_sections for _ in range(3)]
        
        # Run concurrent transformations
        tasks = [ml_cursor_transformer.transform(content) for content in content_sets]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'content' in result
            assert 'ml_quality' in result
            assert 'cursor_rules_compliance' in result['ml_quality']

    @pytest.mark.asyncio
    async def test_concurrent_cursor_rules_validation(self, ml_cursor_transformer):
        """Test concurrent cursor rules validation."""
        # Test concurrent validation operations
        test_rules = [
            """
            ---
            description: Test rules 1
            ---
            # Test Rules 1
            ## Code Style
            """,
            """
            ---
            description: Test rules 2
            ---
            # Test Rules 2
            ## Best Practices
            """,
            """
            ---
            description: Test rules 3
            ---
            # Test Rules 3
            ## Performance
            """
        ]
        
        tasks = [ml_cursor_transformer._validate_cursor_rules_structure(rules) for rules in test_rules]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert 'is_valid' in result
            assert 'score' in result
            assert 'missing_sections' in result

    @pytest.mark.asyncio
    async def test_concurrent_cursor_rules_enhancement(self, ml_cursor_transformer):
        """Test concurrent cursor rules enhancement."""
        # Test concurrent enhancement operations
        test_rules = [
            """
            ---
            description: Basic rules 1
            ---
            # Basic Rules 1
            ## Code Style
            """,
            """
            ---
            description: Basic rules 2
            ---
            # Basic Rules 2
            ## Best Practices
            """,
            """
            ---
            description: Basic rules 3
            ---
            # Basic Rules 3
            ## Performance
            """
        ]
        
        tasks = [ml_cursor_transformer._enhance_cursor_rules_structure(rules) for rules in test_rules]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert "## Key Principles" in result
            assert "## Error Handling" in result
            assert "## Critical Instructions" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
