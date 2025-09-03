"""
Tests for CursorRuleTransformer enhancements with comprehensive knowledge base.

Tests the enhanced CursorRuleTransformer with comprehensive cursor rules knowledge,
learning and intelligence integration, and advanced cursor rules generation capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.models import ContentSection, ScrapingResult, ScrapingStatus


@pytest.fixture
def sample_content_sections():
    """Sample content sections for testing."""
    return [
        ContentSection(
            title="React Component Architecture",
            content="Advanced React component architecture patterns and best practices",
            section_type="code_example",
            metadata={"language": "javascript", "framework": "react", "topic": "architecture", "complexity": "advanced"}
        ),
        ContentSection(
            title="TypeScript Integration",
            content="TypeScript integration patterns for React applications",
            section_type="guideline",
            metadata={"language": "typescript", "framework": "react", "topic": "typescript", "complexity": "intermediate"}
        ),
        ContentSection(
            title="Testing Strategies",
            content="Comprehensive testing strategies for React applications",
            section_type="guideline",
            metadata={"topic": "testing", "framework": "react", "tools": ["jest", "testing-library"], "complexity": "intermediate"}
        ),
        ContentSection(
            title="Performance Optimization",
            content="Advanced performance optimization techniques for React",
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
            "key_concepts": ["react", "typescript", "testing", "performance", "architecture", "patterns"],
            "complexity_score": 0.90,
            "technical_depth": "advanced",
            "learning_difficulty": "high",
            "prerequisites": ["javascript", "react_basics", "typescript", "testing_fundamentals"],
            "estimated_learning_time": "4-6 weeks"
        },
        "learning_insights": {
            "difficulty_level": "advanced",
            "prerequisites": ["javascript", "react_basics", "typescript", "testing_fundamentals"],
            "learning_path": ["basics", "components", "typescript", "testing", "performance", "architecture"],
            "common_mistakes": [
                "Not using TypeScript properly",
                "Inadequate testing coverage",
                "Performance anti-patterns",
                "Poor component architecture"
            ],
            "best_practices": [
                "Use TypeScript for type safety",
                "Implement comprehensive testing",
                "Follow performance best practices",
                "Use proper component architecture"
            ]
        },
        "content_quality": {
            "completeness_score": 0.95,
            "accuracy_score": 0.98,
            "clarity_score": 0.90,
            "practical_applicability": 0.95
        }
    })
    
    mock_engine.track_usage = AsyncMock(return_value={
        "success": True,
        "usage_id": "cursor_test_usage_123",
        "timestamp": datetime.now(),
        "processing_time": 2.0
    })
    
    mock_engine.get_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add advanced TypeScript patterns",
            "Include comprehensive testing examples",
            "Add performance profiling techniques",
            "Include accessibility guidelines",
            "Add error handling patterns",
            "Include advanced React patterns"
        ],
        "related_topics": [
            "Advanced TypeScript patterns",
            "React testing best practices",
            "Performance optimization",
            "Component architecture",
            "Accessibility in React"
        ],
        "learning_resources": [
            "Advanced React patterns",
            "TypeScript handbook",
            "React testing guide",
            "Performance optimization guide"
        ]
    })
    
    return mock_engine


@pytest.fixture
def mock_intelligence_engine():
    """Mock intelligence engine with comprehensive responses."""
    mock_engine = Mock()
    mock_engine.categorize_content = AsyncMock(return_value={
        "primary_category": "frontend_framework",
        "subcategories": ["react", "typescript", "testing", "performance", "architecture"],
        "confidence": 0.99,
        "related_technologies": ["typescript", "nextjs", "jest", "testing-library", "webpack", "storybook"],
        "complexity_assessment": {
            "overall_complexity": "advanced",
            "technical_depth": "expert",
            "learning_curve": "steep",
            "prerequisite_knowledge": ["javascript", "react_basics", "typescript", "testing_fundamentals", "html", "css"]
        },
        "content_analysis": {
            "main_topics": ["architecture", "typescript", "testing", "performance"],
            "target_audience": "expert_developers",
            "practical_focus": "production_ready"
        }
    })
    
    mock_engine.generate_recommendations = AsyncMock(return_value={
        "enhancement_suggestions": [
            "Add advanced TypeScript patterns",
            "Include comprehensive testing examples",
            "Add performance profiling techniques",
            "Include accessibility guidelines",
            "Add error handling patterns",
            "Include advanced React patterns"
        ],
        "related_topics": [
            "Advanced TypeScript patterns",
            "React testing best practices",
            "Performance optimization",
            "Component architecture",
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
            "progression_path": ["typescript", "testing", "performance", "architecture", "advanced_patterns"]
        }
    })
    
    return mock_engine


class TestCursorRuleTransformerInitialization:
    """Test CursorRuleTransformer initialization and knowledge base setup."""

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
        knowledge = cursor_rule_transformer.cursor_rules_knowledge
        assert 'required_sections' in knowledge
        assert 'technology_guidelines' in knowledge
        assert 'cursor_patterns' in knowledge
        assert 'quality_indicators' in knowledge

    @pytest.mark.asyncio
    async def test_cursor_rules_knowledge_base_structure(self, cursor_rule_transformer):
        """Test cursor rules knowledge base structure."""
        knowledge = cursor_rule_transformer.cursor_rules_knowledge
        
        # Test required sections
        required_sections = knowledge['required_sections']
        assert isinstance(required_sections, list)
        assert 'Key Principles' in required_sections
        assert 'Code Style' in required_sections
        assert 'Best Practices' in required_sections
        assert 'Error Handling' in required_sections
        assert 'Performance' in required_sections
        assert 'Critical Instructions' in required_sections
        
        # Test technology guidelines
        tech_guidelines = knowledge['technology_guidelines']
        assert isinstance(tech_guidelines, dict)
        assert 'react' in tech_guidelines
        assert 'python' in tech_guidelines
        assert 'javascript' in tech_guidelines
        assert 'typescript' in tech_guidelines
        
        # Test cursor patterns
        cursor_patterns = knowledge['cursor_patterns']
        assert isinstance(cursor_patterns, dict)
        assert 'frontmatter_pattern' in cursor_patterns
        assert 'section_pattern' in cursor_patterns
        assert 'quality_indicators' in cursor_patterns

    @pytest.mark.asyncio
    async def test_learning_engine_integration(self, cursor_rule_transformer):
        """Test learning engine integration."""
        assert cursor_rule_transformer.learning_engine is not None
        assert hasattr(cursor_rule_transformer.learning_engine, 'analyze_content')
        assert hasattr(cursor_rule_transformer.learning_engine, 'track_usage')
        assert hasattr(cursor_rule_transformer.learning_engine, 'get_recommendations')

    @pytest.mark.asyncio
    async def test_intelligence_engine_integration(self, cursor_rule_transformer):
        """Test intelligence engine integration."""
        assert cursor_rule_transformer.intelligence_engine is not None
        assert hasattr(cursor_rule_transformer.intelligence_engine, 'categorize_content')
        assert hasattr(cursor_rule_transformer.intelligence_engine, 'generate_recommendations')


class TestCursorRuleTransformerKnowledgeBase:
    """Test CursorRuleTransformer knowledge base functionality."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_technology_guidelines_retrieval(self, cursor_rule_transformer):
        """Test technology guidelines retrieval."""
        # Test React guidelines
        react_guidelines = cursor_rule_transformer._get_technology_guidelines("react")
        assert react_guidelines is not None
        assert isinstance(react_guidelines, str)
        assert len(react_guidelines) > 0
        assert "functional components" in react_guidelines.lower()
        assert "hooks" in react_guidelines.lower()
        
        # Test Python guidelines
        python_guidelines = cursor_rule_transformer._get_technology_guidelines("python")
        assert python_guidelines is not None
        assert isinstance(python_guidelines, str)
        assert len(python_guidelines) > 0
        assert "pep 8" in python_guidelines.lower()
        assert "type hints" in python_guidelines.lower()
        
        # Test JavaScript guidelines
        js_guidelines = cursor_rule_transformer._get_technology_guidelines("javascript")
        assert js_guidelines is not None
        assert isinstance(js_guidelines, str)
        assert len(js_guidelines) > 0
        assert "es6" in js_guidelines.lower()
        assert "async/await" in js_guidelines.lower()
        
        # Test TypeScript guidelines
        ts_guidelines = cursor_rule_transformer._get_technology_guidelines("typescript")
        assert ts_guidelines is not None
        assert isinstance(ts_guidelines, str)
        assert len(ts_guidelines) > 0
        assert "type safety" in ts_guidelines.lower()
        assert "interfaces" in ts_guidelines.lower()

    @pytest.mark.asyncio
    async def test_technology_guidelines_unknown_technology(self, cursor_rule_transformer):
        """Test technology guidelines for unknown technology."""
        unknown_guidelines = cursor_rule_transformer._get_technology_guidelines("unknown_tech")
        assert unknown_guidelines is not None
        assert isinstance(unknown_guidelines, str)
        assert len(unknown_guidelines) > 0
        # Should return generic guidelines
        assert "general" in unknown_guidelines.lower() or "common" in unknown_guidelines.lower()

    @pytest.mark.asyncio
    async def test_cursor_rules_patterns_validation(self, cursor_rule_transformer):
        """Test cursor rules patterns validation."""
        patterns = cursor_rule_transformer.cursor_rules_knowledge['cursor_patterns']
        
        # Test frontmatter pattern
        frontmatter_pattern = patterns['frontmatter_pattern']
        assert frontmatter_pattern is not None
        assert isinstance(frontmatter_pattern, str)
        
        # Test section pattern
        section_pattern = patterns['section_pattern']
        assert section_pattern is not None
        assert isinstance(section_pattern, str)
        
        # Test quality indicators
        quality_indicators = patterns['quality_indicators']
        assert isinstance(quality_indicators, list)
        assert len(quality_indicators) > 0

    @pytest.mark.asyncio
    async def test_quality_indicators_retrieval(self, cursor_rule_transformer):
        """Test quality indicators retrieval."""
        quality_indicators = cursor_rule_transformer.cursor_rules_knowledge['quality_indicators']
        
        assert isinstance(quality_indicators, dict)
        assert 'clean_code' in quality_indicators
        assert 'best_practices' in quality_indicators
        assert 'security' in quality_indicators
        assert 'performance' in quality_indicators
        assert 'maintainability' in quality_indicators
        
        # Test each category
        for category, indicators in quality_indicators.items():
            assert isinstance(indicators, list)
            assert len(indicators) > 0


class TestCursorRuleTransformerGeneration:
    """Test CursorRuleTransformer cursor rules generation."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_generate_cursor_rule_basic(self, cursor_rule_transformer, sample_content_sections):
        """Test basic cursor rule generation."""
        result = await cursor_rule_transformer.generate_cursor_rule(
            sample_content_sections, 
            "React Development Guidelines"
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Check YAML frontmatter
        assert "---" in result
        assert "description:" in result
        assert "globs:" in result
        
        # Check title
        assert "# React Development Guidelines" in result
        
        # Check required sections
        required_sections = ["Key Principles", "Code Style", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in result

    @pytest.mark.asyncio
    async def test_generate_cursor_rule_with_metadata(self, cursor_rule_transformer, sample_content_sections):
        """Test cursor rule generation with metadata."""
        result = await cursor_rule_transformer.generate_cursor_rule(
            sample_content_sections, 
            "Advanced React Guidelines",
            metadata={"framework": "react", "language": "typescript", "complexity": "advanced"}
        )
        
        assert result is not None
        assert isinstance(result, str)
        
        # Check that metadata influenced the content
        assert "typescript" in result.lower()
        assert "advanced" in result.lower()
        assert "react" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_cursor_rule_technology_specific(self, cursor_rule_transformer):
        """Test technology-specific cursor rule generation."""
        # Test React-specific content
        react_content = [ContentSection(
            title="React Component",
            content="React component with hooks",
            section_type="code_example",
            metadata={"framework": "react", "language": "javascript"}
        )]
        
        result = await cursor_rule_transformer.generate_cursor_rule(
            react_content, 
            "React Guidelines"
        )
        
        assert result is not None
        assert "react" in result.lower()
        assert "hooks" in result.lower()
        assert "functional components" in result.lower()
        
        # Test Python-specific content
        python_content = [ContentSection(
            title="Python Function",
            content="Python function with type hints",
            section_type="code_example",
            metadata={"language": "python", "framework": "django"}
        )]
        
        result = await cursor_rule_transformer.generate_cursor_rule(
            python_content, 
            "Python Guidelines"
        )
        
        assert result is not None
        assert "python" in result.lower()
        assert "pep 8" in result.lower()
        assert "type hints" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_cursor_rule_comprehensive(self, cursor_rule_transformer, sample_content_sections):
        """Test comprehensive cursor rule generation."""
        result = await cursor_rule_transformer.generate_cursor_rule(
            sample_content_sections, 
            "Comprehensive React Guidelines"
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 500  # Should be comprehensive
        
        # Check all required sections are present with content
        required_sections = ["Key Principles", "Code Style", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            section_start = result.find(f"## {section}")
            assert section_start != -1
            
            # Check that section has content (not just the header)
            next_section = result.find("## ", section_start + 1)
            if next_section == -1:
                section_content = result[section_start:]
            else:
                section_content = result[section_start:next_section]
            
            assert len(section_content) > 50  # Should have substantial content

    @pytest.mark.asyncio
    async def test_generate_cursor_rule_learning_integration(self, cursor_rule_transformer, sample_content_sections):
        """Test cursor rule generation with learning integration."""
        result = await cursor_rule_transformer.generate_cursor_rule(
            sample_content_sections, 
            "Learning-Enhanced Guidelines"
        )
        
        assert result is not None
        
        # Verify learning engine was called
        cursor_rule_transformer.learning_engine.analyze_content.assert_called_once()
        
        # Check that learning insights influenced the content
        assert "typescript" in result.lower()  # From learning insights
        assert "testing" in result.lower()  # From learning insights
        assert "performance" in result.lower()  # From learning insights

    @pytest.mark.asyncio
    async def test_generate_cursor_rule_intelligence_integration(self, cursor_rule_transformer, sample_content_sections):
        """Test cursor rule generation with intelligence integration."""
        result = await cursor_rule_transformer.generate_cursor_rule(
            sample_content_sections, 
            "Intelligence-Enhanced Guidelines"
        )
        
        assert result is not None
        
        # Verify intelligence engine was called
        cursor_rule_transformer.intelligence_engine.categorize_content.assert_called_once()
        cursor_rule_transformer.intelligence_engine.generate_recommendations.assert_called_once()
        
        # Check that intelligence insights influenced the content
        assert "advanced" in result.lower()  # From intelligence insights
        assert "architecture" in result.lower()  # From intelligence insights


class TestCursorRuleTransformerValidation:
    """Test CursorRuleTransformer cursor rules validation."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_comprehensive(self, cursor_rule_transformer):
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
        
        validation_result = cursor_rule_transformer._validate_cursor_rules_structure(comprehensive_rules)
        
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.9
        assert len(validation_result['missing_sections']) == 0
        assert 'quality_indicators' in validation_result
        assert 'structure_analysis' in validation_result

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_incomplete(self, cursor_rule_transformer):
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
        
        validation_result = cursor_rule_transformer._validate_cursor_rules_structure(incomplete_rules)
        
        assert validation_result['is_valid'] is False
        assert validation_result['score'] < 0.5
        assert len(validation_result['missing_sections']) > 0
        assert 'Key Principles' in validation_result['missing_sections']
        assert 'Best Practices' in validation_result['missing_sections']
        assert 'Error Handling' in validation_result['missing_sections']
        assert 'Performance' in validation_result['missing_sections']
        assert 'Critical Instructions' in validation_result['missing_sections']

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_quality_scoring(self, cursor_rule_transformer):
        """Test cursor rules validation quality scoring."""
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
            validation_result = cursor_rule_transformer._validate_cursor_rules_structure(test_case["rules"])
            assert validation_result['score'] >= test_case["expected_score"] - 0.1

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_pattern_matching(self, cursor_rule_transformer):
        """Test cursor rules validation pattern matching."""
        # Test valid patterns
        valid_patterns = [
            "---\ndescription: Test\nglobs: ['**/*.js']\n---\n# Title\n## Section",
            "---\ndescription: Test\nglobs: ['**/*.py', '**/*.pyi']\n---\n# Title\n## Section",
            "---\ndescription: Test\nglobs: ['**/*.tsx', '**/*.jsx']\n---\n# Title\n## Section"
        ]
        
        for pattern in valid_patterns:
            validation_result = cursor_rule_transformer._validate_cursor_rules_structure(pattern)
            assert validation_result['is_valid'] is True


class TestCursorRuleTransformerEnhancement:
    """Test CursorRuleTransformer cursor rules enhancement."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_basic(self, cursor_rule_transformer):
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
        
        enhanced_rules = cursor_rule_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
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
    async def test_cursor_rules_enhancement_comprehensive(self, cursor_rule_transformer):
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
        
        enhanced_rules = cursor_rule_transformer._enhance_cursor_rules_structure(incomplete_rules)
        
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
    async def test_cursor_rules_enhancement_frontmatter(self, cursor_rule_transformer):
        """Test cursor rules enhancement with frontmatter."""
        # Test rules without frontmatter
        no_frontmatter_rules = """
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = cursor_rule_transformer._enhance_cursor_rules_structure(no_frontmatter_rules)
        
        assert enhanced_rules is not None
        assert "---" in enhanced_rules
        assert "description:" in enhanced_rules
        assert "globs:" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_quality_improvement(self, cursor_rule_transformer):
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
        
        enhanced_rules = cursor_rule_transformer._enhance_cursor_rules_structure(basic_rules)
        
        assert enhanced_rules is not None
        
        # Check that enhanced sections have meaningful content
        assert "functional components" in enhanced_rules.lower()
        assert "error handling" in enhanced_rules.lower()
        assert "performance" in enhanced_rules.lower()
        assert "best practices" in enhanced_rules.lower()


class TestCursorRuleTransformerLearningIntelligenceIntegration:
    """Test CursorRuleTransformer learning and intelligence integration."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_learning_enhancement(self, cursor_rule_transformer, sample_content_sections):
        """Test learning enhancement of results."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_enhancement' in result
        
        learning_enhancement = result['learning_enhancement']
        assert 'semantic_analysis' in learning_enhancement
        assert 'learning_insights' in learning_enhancement
        assert 'recommendations' in learning_enhancement
        
        # Verify learning engine was called
        cursor_rule_transformer.learning_engine.analyze_content.assert_called_once()
        cursor_rule_transformer.learning_engine.track_usage.assert_called_once()
        cursor_rule_transformer.learning_engine.get_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligence_enhancement(self, cursor_rule_transformer, sample_content_sections):
        """Test intelligence enhancement of results."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'intelligence_enhancement' in result
        
        intelligence_enhancement = result['intelligence_enhancement']
        assert 'categorization' in intelligence_enhancement
        assert 'enhancement_suggestions' in intelligence_enhancement
        assert 'related_topics' in intelligence_enhancement
        
        # Verify intelligence engine was called
        cursor_rule_transformer.intelligence_engine.categorize_content.assert_called_once()
        cursor_rule_transformer.intelligence_engine.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_combined_learning_intelligence_enhancement(self, cursor_rule_transformer, sample_content_sections):
        """Test combined learning and intelligence enhancement."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'learning_enhancement' in result
        assert 'intelligence_enhancement' in result
        
        # Check learning enhancement details
        learning_enhancement = result['learning_enhancement']
        assert 'semantic_analysis' in learning_enhancement
        assert 'learning_insights' in learning_enhancement
        assert 'recommendations' in learning_enhancement
        
        # Check intelligence enhancement details
        intelligence_enhancement = result['intelligence_enhancement']
        assert 'categorization' in intelligence_enhancement
        assert 'enhancement_suggestions' in intelligence_enhancement
        assert 'related_topics' in intelligence_enhancement

    @pytest.mark.asyncio
    async def test_learning_intelligence_metadata(self, cursor_rule_transformer, sample_content_sections):
        """Test learning and intelligence metadata."""
        result = await cursor_rule_transformer.transform(sample_content_sections)
        
        assert result is not None
        assert 'cursor_rules_metadata' in result
        
        metadata = result['cursor_rules_metadata']
        assert 'learning_enhancement_applied' in metadata
        assert 'intelligence_enhancement_applied' in metadata
        assert 'quality_score' in metadata
        assert 'technology_detected' in metadata


class TestCursorRuleTransformerDefaultSections:
    """Test CursorRuleTransformer default sections generation."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_default_key_principles_section(self, cursor_rule_transformer):
        """Test default key principles section generation."""
        section = cursor_rule_transformer._get_default_key_principles_section("react")
        
        assert section is not None
        assert isinstance(section, str)
        assert len(section) > 0
        assert "key principles" in section.lower()
        assert "react" in section.lower()

    @pytest.mark.asyncio
    async def test_default_code_style_section(self, cursor_rule_transformer):
        """Test default code style section generation."""
        section = cursor_rule_transformer._get_default_code_style_section("python")
        
        assert section is not None
        assert isinstance(section, str)
        assert len(section) > 0
        assert "code style" in section.lower()
        assert "python" in section.lower()

    @pytest.mark.asyncio
    async def test_default_best_practices_section(self, cursor_rule_transformer):
        """Test default best practices section generation."""
        section = cursor_rule_transformer._get_default_best_practices_section("javascript")
        
        assert section is not None
        assert isinstance(section, str)
        assert len(section) > 0
        assert "best practices" in section.lower()
        assert "javascript" in section.lower()

    @pytest.mark.asyncio
    async def test_default_error_handling_section(self, cursor_rule_transformer):
        """Test default error handling section generation."""
        section = cursor_rule_transformer._get_default_error_handling_section("typescript")
        
        assert section is not None
        assert isinstance(section, str)
        assert len(section) > 0
        assert "error handling" in section.lower()
        assert "typescript" in section.lower()

    @pytest.mark.asyncio
    async def test_default_performance_section(self, cursor_rule_transformer):
        """Test default performance section generation."""
        section = cursor_rule_transformer._get_default_performance_section("react")
        
        assert section is not None
        assert isinstance(section, str)
        assert len(section) > 0
        assert "performance" in section.lower()
        assert "react" in section.lower()

    @pytest.mark.asyncio
    async def test_default_critical_instructions_section(self, cursor_rule_transformer):
        """Test default critical instructions section generation."""
        section = cursor_rule_transformer._get_default_critical_instructions_section("python")
        
        assert section is not None
        assert isinstance(section, str)
        assert len(section) > 0
        assert "critical instructions" in section.lower()
        assert "python" in section.lower()


class TestCursorRuleTransformerGracefulDegradation:
    """Test CursorRuleTransformer graceful degradation."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_learning_engine(self, sample_content_sections):
        """Test graceful degradation when learning engine is unavailable."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning:
            mock_learning.side_effect = Exception("Learning engine unavailable")
            
            transformer = CursorRuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without learning engine
            assert result is not None
            assert 'content' in result
            # Learning-specific fields should not be present
            assert 'learning_enhancement' not in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_intelligence_engine(self, sample_content_sections):
        """Test graceful degradation when intelligence engine is unavailable."""
        with patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            mock_intelligence.side_effect = Exception("Intelligence engine unavailable")
            
            transformer = CursorRuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work without intelligence engine
            assert result is not None
            assert 'content' in result
            # Intelligence-specific fields should not be present
            assert 'intelligence_enhancement' not in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_all_modules(self, sample_content_sections):
        """Test graceful degradation when all modules are unavailable."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.side_effect = Exception("Learning engine unavailable")
            mock_intelligence.side_effect = Exception("Intelligence engine unavailable")
            
            transformer = CursorRuleTransformer()
            result = await transformer.transform(sample_content_sections)
            
            # Should still work with basic functionality
            assert result is not None
            assert 'content' in result
            # Enhanced fields should not be present
            assert 'learning_enhancement' not in result
            assert 'intelligence_enhancement' not in result


class TestCursorRuleTransformerConcurrentOperations:
    """Test CursorRuleTransformer concurrent operations."""

    @pytest.fixture
    def cursor_rule_transformer(self, mock_learning_engine, mock_intelligence_engine):
        """Create CursorRuleTransformer with mocked dependencies."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine') as mock_learning, \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine') as mock_intelligence:
            
            mock_learning.return_value = mock_learning_engine
            mock_intelligence.return_value = mock_intelligence_engine
            
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_concurrent_transformations(self, cursor_rule_transformer, sample_content_sections):
        """Test concurrent transformations."""
        # Create multiple content sections for concurrent processing
        content_sets = [sample_content_sections for _ in range(3)]
        
        # Run concurrent transformations
        tasks = [cursor_rule_transformer.transform(content) for content in content_sets]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'content' in result
            assert 'cursor_rules_validation' in result
            assert 'cursor_rules_enhancement' in result

    @pytest.mark.asyncio
    async def test_concurrent_cursor_rule_generation(self, cursor_rule_transformer, sample_content_sections):
        """Test concurrent cursor rule generation."""
        # Test concurrent generation operations
        tasks = [
            cursor_rule_transformer.generate_cursor_rule(sample_content_sections, "React Guidelines 1"),
            cursor_rule_transformer.generate_cursor_rule(sample_content_sections, "React Guidelines 2"),
            cursor_rule_transformer.generate_cursor_rule(sample_content_sections, "React Guidelines 3")
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            assert "---" in result
            assert "description:" in result

    @pytest.mark.asyncio
    async def test_concurrent_validation_enhancement(self, cursor_rule_transformer):
        """Test concurrent validation and enhancement operations."""
        # Test concurrent validation and enhancement
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
        
        # Concurrent validation
        validation_tasks = [cursor_rule_transformer._validate_cursor_rules_structure(rules) for rules in test_rules]
        validation_results = await asyncio.gather(*validation_tasks)
        
        # Concurrent enhancement
        enhancement_tasks = [cursor_rule_transformer._enhance_cursor_rules_structure(rules) for rules in test_rules]
        enhancement_results = await asyncio.gather(*enhancement_tasks)
        
        assert len(validation_results) == 3
        assert len(enhancement_results) == 3
        
        for validation_result in validation_results:
            assert 'is_valid' in validation_result
            assert 'score' in validation_result
        
        for enhancement_result in enhancement_results:
            assert enhancement_result is not None
            assert "## Key Principles" in enhancement_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
