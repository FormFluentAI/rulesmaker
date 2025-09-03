"""
Comprehensive test suite for cursor rules integration in transformers.

Tests the integration of cursor rules knowledge, learning, and intelligence modules
across all transformer components with full validation and enhancement capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List, Optional

from rules_maker.transformers.rule_transformer import RuleTransformer
from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.models import (
    ScrapingResult, ScrapingStatus, ContentSection, 
    TrainingSet, LearningExample, DocumentationType
)


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return """
    <html>
    <head><title>React Best Practices</title></head>
    <body>
        <h1>React Development Guidelines</h1>
        <p>This document outlines best practices for React development.</p>
        <h2>Component Structure</h2>
        <pre><code>
        function MyComponent({ prop1, prop2 }) {
            const [state, setState] = useState(initialValue);
            return (
                <div className="component">
                    <h3>{prop1}</h3>
                    <p>{prop2}</p>
                </div>
            );
        }
        </code></pre>
        <h2>State Management</h2>
        <p>Use useState for local state, useContext for shared state.</p>
        <h2>Performance</h2>
        <p>Use React.memo, useMemo, and useCallback for optimization.</p>
    </body>
    </html>
    """


@pytest.fixture
def sample_scraping_result(sample_html_content):
    """Sample scraping result for testing."""
    return ScrapingResult(
        url="https://react.dev/learn",
        title="React Best Practices",
        content=sample_html_content,
        status=ScrapingStatus.COMPLETED,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_content_sections():
    """Sample content sections for testing."""
    return [
        ContentSection(
            title="Component Structure",
            content="Guidelines for React component structure and organization",
            section_type="code_example",
            metadata={"language": "javascript", "framework": "react"}
        ),
        ContentSection(
            title="State Management",
            content="Best practices for React state management",
            section_type="guideline",
            metadata={"topic": "state", "framework": "react"}
        ),
        ContentSection(
            title="Performance",
            content="React performance optimization techniques",
            section_type="guideline",
            metadata={"topic": "performance", "framework": "react"}
        )
    ]


@pytest.fixture
def mock_learning_engine():
    """Mock learning engine for testing."""
    mock_engine = Mock()
    mock_engine.analyze_content = AsyncMock(return_value={
        "semantic_analysis": {
            "key_concepts": ["react", "components", "state", "performance"],
            "complexity_score": 0.7,
            "technical_depth": "intermediate"
        },
        "learning_insights": {
            "difficulty_level": "intermediate",
            "prerequisites": ["javascript", "html", "css"],
            "learning_path": ["basics", "components", "state", "performance"]
        }
    })
    mock_engine.track_usage = AsyncMock(return_value=True)
    mock_engine.get_recommendations = AsyncMock(return_value=[
        "Consider adding TypeScript examples",
        "Include error handling patterns",
        "Add accessibility guidelines"
    ])
    return mock_engine


@pytest.fixture
def mock_intelligence_engine():
    """Mock intelligence engine for testing."""
    mock_engine = Mock()
    mock_engine.categorize_content = AsyncMock(return_value={
        "primary_category": "frontend_framework",
        "subcategories": ["react", "javascript", "ui_development"],
        "confidence": 0.95,
        "related_technologies": ["typescript", "nextjs", "redux"]
    })
    mock_engine.generate_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add TypeScript integration examples",
            "Include testing patterns",
            "Add accessibility guidelines"
        ],
        "related_topics": ["hooks", "context", "performance"],
        "complexity_assessment": "intermediate"
    })
    return mock_engine


@pytest.fixture
def mock_semantic_analyzer():
    """Mock semantic analyzer for testing."""
    mock_analyzer = Mock()
    mock_analyzer.analyze = AsyncMock(return_value={
        "entities": ["React", "JavaScript", "Components", "State"],
        "concepts": ["frontend", "framework", "development"],
        "sentiment": "positive",
        "complexity": "intermediate"
    })
    return mock_analyzer


@pytest.fixture
def mock_usage_tracker():
    """Mock usage tracker for testing."""
    mock_tracker = Mock()
    mock_tracker.track = AsyncMock(return_value=True)
    mock_tracker.get_usage_stats = AsyncMock(return_value={
        "total_requests": 100,
        "success_rate": 0.95,
        "avg_processing_time": 1.2
    })
    return mock_tracker


class TestRuleTransformerCursorRulesIntegration:
    """Test RuleTransformer with cursor rules integration."""

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
    async def test_rule_transformer_initialization(self, rule_transformer):
        """Test RuleTransformer initialization with learning and intelligence."""
        assert rule_transformer is not None
        assert hasattr(rule_transformer, 'learning_engine')
        assert hasattr(rule_transformer, 'intelligence_engine')
        assert hasattr(rule_transformer, 'semantic_analyzer')
        assert hasattr(rule_transformer, 'usage_tracker')

    @pytest.mark.asyncio
    async def test_transform_with_learning_integration(self, rule_transformer, sample_content_sections):
        """Test transform method with learning integration."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_insights' in result
        assert 'semantic_analysis' in result
        assert 'intelligent_categorization' in result
        
        # Verify learning engine was called
        rule_transformer.learning_engine.analyze_content.assert_called_once()
        rule_transformer.learning_engine.track_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_transform_with_intelligence_integration(self, rule_transformer, sample_content_sections):
        """Test transform method with intelligence integration."""
        result = await rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligent_categorization' in result
        assert 'enhancement_suggestions' in result
        
        # Verify intelligence engine was called
        rule_transformer.intelligence_engine.categorize_content.assert_called_once()
        rule_transformer.intelligence_engine.generate_recommendations.assert_called_once()

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
    async def test_graceful_degradation(self, sample_content_sections):
        """Test graceful degradation when learning/intelligence modules fail."""
        # Create transformer with failing dependencies
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.side_effect = Exception("Learning engine unavailable")
            mock_intelligence.side_effect = Exception("Intelligence engine unavailable")
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without learning/intelligence
            assert result is not None
            assert 'content' in result


class TestMLCursorTransformerCursorRulesIntegration:
    """Test MLCursorTransformer with cursor rules integration."""

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

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_scoring(self, ml_cursor_transformer):
        """Test cursor rules validation and scoring."""
        # Test comprehensive cursor rules
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
        
        ## Code Style
        - Use camelCase for variables and functions
        - Use PascalCase for components
        - Use 2-space indentation
        - Use semicolons consistently
        
        ## Best Practices
        - Use TypeScript for type safety
        - Implement proper error boundaries
        - Use React.memo for performance optimization
        - Prefer controlled components
        
        ## Error Handling
        - Use try-catch blocks for async operations
        - Implement error boundaries for component errors
        - Log errors appropriately
        
        ## Performance
        - Use React.memo for expensive components
        - Use useMemo and useCallback appropriately
        - Avoid unnecessary re-renders
        
        ## Critical Instructions
        - Never mutate props or state directly
        - Always use keys in lists
        - Handle loading and error states
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(comprehensive_rules)
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.8
        assert 'missing_sections' in validation_result
        assert 'quality_indicators' in validation_result

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement(self, ml_cursor_transformer):
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
        
        enhanced_rules = ml_cursor_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
        assert enhanced_rules is not None
        assert "## Key Principles" in enhanced_rules
        assert "## Best Practices" in enhanced_rules
        assert "## Error Handling" in enhanced_rules
        assert "## Performance" in enhanced_rules
        assert "## Critical Instructions" in enhanced_rules

    @pytest.mark.asyncio
    async def test_ml_quality_with_cursor_rules(self, ml_cursor_transformer, sample_content_sections):
        """Test ML quality assessment with cursor rules integration."""
        result = await ml_cursor_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'ml_quality' in result
        assert 'cursor_rules_compliance' in result['ml_quality']
        assert 'cursor_rules_insights' in result['ml_quality']
        assert 'cursor_rules_recommendations' in result['ml_quality']

    @pytest.mark.asyncio
    async def test_technology_specific_insights(self, ml_cursor_transformer):
        """Test technology-specific cursor rules insights."""
        # Test React-specific insights
        react_content = [ContentSection(
            title="React Component",
            content="React component example",
            section_type="code_example",
            metadata={"framework": "react", "language": "javascript"}
        )]
        
        result = await ml_cursor_transformer.transform(react_content)
        
        assert result is not None
        assert 'cursor_rules_insights' in result['ml_quality']
        insights = result['ml_quality']['cursor_rules_insights']
        assert 'technology_specific' in insights
        assert 'react' in insights['technology_specific']


class TestCursorRuleTransformerComprehensiveKnowledge:
    """Test CursorRuleTransformer with comprehensive knowledge base."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rule_transformer_initialization(self, cursor_rule_transformer):
        """Test CursorRuleTransformer initialization with knowledge base."""
        assert cursor_rule_transformer is not None
        assert hasattr(cursor_rule_transformer, 'cursor_rules_knowledge')
        assert hasattr(cursor_rule_transformer, 'learning_engine')
        assert hasattr(cursor_rule_transformer, 'intelligence_engine')
        
        # Verify knowledge base is populated
        assert 'required_sections' in cursor_rule_transformer.cursor_rules_knowledge
        assert 'technology_guidelines' in cursor_rule_transformer.cursor_rules_knowledge
        assert 'cursor_patterns' in cursor_rule_transformer.cursor_rules_knowledge

    @pytest.mark.asyncio
    async def test_generate_cursor_rule_with_knowledge(self, cursor_rule_transformer, sample_content_sections):
        """Test cursor rule generation with comprehensive knowledge."""
        result = await cursor_rule_transformer.generate_cursor_rule(
            sample_content_sections, 
            "React Development Guidelines"
        )
        
        assert result is not None
        assert "---" in result  # YAML frontmatter
        assert "description:" in result
        assert "globs:" in result
        assert "# React Development Guidelines" in result
        assert "## Key Principles" in result
        assert "## Code Style" in result
        assert "## Best Practices" in result
        assert "## Error Handling" in result
        assert "## Performance" in result
        assert "## Critical Instructions" in result

    @pytest.mark.asyncio
    async def test_technology_specific_guidelines(self, cursor_rule_transformer):
        """Test technology-specific guidelines generation."""
        # Test React guidelines
        react_guidelines = cursor_rule_transformer._get_technology_guidelines("react")
        assert react_guidelines is not None
        assert "functional components" in react_guidelines.lower()
        assert "hooks" in react_guidelines.lower()
        
        # Test Python guidelines
        python_guidelines = cursor_rule_transformer._get_technology_guidelines("python")
        assert python_guidelines is not None
        assert "pep 8" in python_guidelines.lower()
        assert "type hints" in python_guidelines.lower()
        
        # Test JavaScript guidelines
        js_guidelines = cursor_rule_transformer._get_technology_guidelines("javascript")
        assert js_guidelines is not None
        assert "es6" in js_guidelines.lower()
        assert "async/await" in js_guidelines.lower()

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_comprehensive(self, cursor_rule_transformer):
        """Test comprehensive cursor rules validation."""
        # Test valid comprehensive rules
        valid_rules = """
        ---
        description: Comprehensive React guidelines
        globs: ["**/*.jsx", "**/*.tsx"]
        ---
        # React Guidelines
        
        ## Key Principles
        - Use functional components
        - Keep components small
        
        ## Code Style
        - Use camelCase
        - Use 2-space indentation
        
        ## Best Practices
        - Use TypeScript
        - Implement error boundaries
        
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
        
        validation_result = cursor_rule_transformer._validate_cursor_rules_structure(valid_rules)
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.9
        assert len(validation_result['missing_sections']) == 0

    @pytest.mark.asyncio
    async def test_enhance_with_cursor_rules_knowledge(self, cursor_rule_transformer, sample_content_sections):
        """Test enhancement with cursor rules knowledge."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'cursor_rules_validation' in result
        assert 'cursor_rules_enhancement' in result
        assert 'learning_enhancement' in result
        assert 'intelligence_enhancement' in result
        
        # Verify learning and intelligence integration
        cursor_rule_transformer.learning_engine.analyze_content.assert_called_once()
        cursor_rule_transformer.intelligence_engine.categorize_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_sections_generation(self, cursor_rule_transformer):
        """Test default sections generation for missing sections."""
        # Test missing sections
        incomplete_rules = """
        ---
        description: Basic guidelines
        ---
        # Basic Guidelines
        
        ## Code Style
        - Use consistent formatting
        """
        
        enhanced_rules = cursor_rule_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
        assert enhanced_rules is not None
        assert "## Key Principles" in enhanced_rules
        assert "## Best Practices" in enhanced_rules
        assert "## Error Handling" in enhanced_rules
        assert "## Performance" in enhanced_rules
        assert "## Critical Instructions" in enhanced_rules

    @pytest.mark.asyncio
    async def test_learning_intelligence_enhancement(self, cursor_rule_transformer, sample_content_sections):
        """Test learning and intelligence enhancement of results."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_enhancement' in result
        assert 'intelligence_enhancement' in result
        
        learning_enhancement = result['learning_enhancement']
        assert 'semantic_analysis' in learning_enhancement
        assert 'learning_insights' in learning_enhancement
        assert 'recommendations' in learning_enhancement
        
        intelligence_enhancement = result['intelligence_enhancement']
        assert 'categorization' in intelligence_enhancement
        assert 'enhancement_suggestions' in intelligence_enhancement
        assert 'related_topics' in intelligence_enhancement


class TestCursorRulesValidationIntegration:
    """Test cursor rules validation integration across all transformers."""

    @pytest.mark.asyncio
    async def test_cursor_rules_pattern_matching(self):
        """Test cursor rules pattern matching across transformers."""
        # Test valid patterns
        valid_patterns = [
            "---\ndescription: Test\nglobs: ['**/*.js']\n---\n# Title\n## Section",
            "---\ndescription: Test\nglobs: ['**/*.py', '**/*.pyi']\n---\n# Title\n## Section",
            "---\ndescription: Test\nglobs: ['**/*.tsx', '**/*.jsx']\n---\n# Title\n## Section"
        ]
        
        for pattern in valid_patterns:
            # Test with RuleTransformer
            transformer = RuleTransformer()
            is_valid = transformer._validate_cursor_rules_structure(pattern)
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_cursor_rules_quality_scoring(self):
        """Test cursor rules quality scoring across transformers."""
        # Test high-quality rules
        high_quality_rules = """
        ---
        description: Comprehensive React development guidelines
        globs: ["**/*.jsx", "**/*.tsx", "**/*.js", "**/*.ts"]
        ---
        # React Development Guidelines
        
        ## Key Principles
        - Use functional components with hooks
        - Prefer composition over inheritance
        - Keep components small and focused
        
        ## Code Style
        - Use camelCase for variables and functions
        - Use PascalCase for components
        - Use 2-space indentation
        - Use semicolons consistently
        
        ## Best Practices
        - Use TypeScript for type safety
        - Implement proper error boundaries
        - Use React.memo for performance optimization
        - Prefer controlled components
        
        ## Error Handling
        - Use try-catch blocks for async operations
        - Implement error boundaries for component errors
        - Log errors appropriately
        
        ## Performance
        - Use React.memo for expensive components
        - Use useMemo and useCallback appropriately
        - Avoid unnecessary re-renders
        
        ## Critical Instructions
        - Never mutate props or state directly
        - Always use keys in lists
        - Handle loading and error states
        """
        
        # Test with MLCursorTransformer
        ml_transformer = MLCursorTransformer()
        validation_result = ml_transformer._validate_cursor_rules_structure(high_quality_rules)
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.8

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_consistency(self):
        """Test cursor rules enhancement consistency across transformers."""
        incomplete_rules = """
        ---
        description: Basic guidelines
        ---
        # Basic Guidelines
        
        ## Code Style
        - Use consistent formatting
        """
        
        # Test enhancement with different transformers
        rule_transformer = RuleTransformer()
        ml_transformer = MLCursorTransformer()
        cursor_transformer = CursorRuleTransformer()
        
        enhanced_rule = rule_transformer._enhance_cursor_rules_structure(incomplete_rules)
        enhanced_ml = ml_transformer._enhance_cursor_rules_structure(incomplete_rules)
        enhanced_cursor = cursor_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
        # All should add the same required sections
        required_sections = ["Key Principles", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        
        for section in required_sections:
            assert f"## {section}" in enhanced_rule
            assert f"## {section}" in enhanced_ml
            assert f"## {section}" in enhanced_cursor


class TestLearningIntelligenceIntegration:
    """Test learning and intelligence integration across all transformers."""

    @pytest.mark.asyncio
    async def test_learning_engine_integration(self, mock_learning_engine, sample_content_sections):
        """Test learning engine integration across transformers."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning:
            mock_learning.return_value = mock_learning_engine
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            assert result is not None
            assert 'learning_insights' in result
            assert 'semantic_analysis' in result
            
            # Verify learning engine methods were called
            mock_learning_engine.analyze_content.assert_called_once()
            mock_learning_engine.track_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligence_engine_integration(self, mock_intelligence_engine, sample_content_sections):
        """Test intelligence engine integration across transformers."""
        with patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence:
            mock_intelligence.return_value = mock_intelligence_engine
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            assert result is not None
            assert 'intelligent_categorization' in result
            assert 'enhancement_suggestions' in result
            
            # Verify intelligence engine methods were called
            mock_intelligence_engine.categorize_content.assert_called_once()
            mock_intelligence_engine.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_degradation_learning_intelligence(self, sample_content_sections):
        """Test graceful degradation when learning/intelligence modules are unavailable."""
        # Test with failing learning engine
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning:
            mock_learning.side_effect = Exception("Learning engine unavailable")
            
            transformer = RuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without learning
            assert result is not None
            assert 'content' in result

    @pytest.mark.asyncio
    async def test_async_learning_intelligence_operations(self, mock_learning_engine, mock_intelligence_engine, sample_content_sections):
        """Test async operations with learning and intelligence engines."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            transformer = RuleTransformer()
            
            # Test concurrent operations
            tasks = [
                transformer.transform(sample_content_sections),
                transformer.transform(sample_content_sections),
                transformer.transform(sample_content_sections)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result in results:
                assert result is not None
                assert 'learning_insights' in result
                assert 'intelligent_categorization' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
