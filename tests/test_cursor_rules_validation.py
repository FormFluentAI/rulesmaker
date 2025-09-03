"""
Tests for cursor rules validation and enhancement functionality.

Tests the cursor rules validation, enhancement, and quality assessment
functionality across all transformer components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from rules_maker.transformers.rule_transformer import RuleTransformer
from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer


class TestCursorRulesValidationCore:
    """Test core cursor rules validation functionality."""

    @pytest.fixture
    def rule_transformer(self):
        """Create RuleTransformer for testing."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine'), \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine'), \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer'), \
             patch('rules_maker.transformers.rule_transformer.UsageTracker'):
            return RuleTransformer()

    @pytest.fixture
    def ml_cursor_transformer(self):
        """Create MLCursorTransformer for testing."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine'), \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine'):
            return MLCursorTransformer()

    @pytest.fixture
    def cursor_rule_transformer(self):
        """Create CursorRuleTransformer for testing."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine'), \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine'):
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_perfect_structure(self, rule_transformer):
        """Test validation of perfectly structured cursor rules."""
        perfect_rules = """
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
        
        is_valid = rule_transformer._validate_cursor_rules_structure(perfect_rules)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_missing_frontmatter(self, rule_transformer):
        """Test validation of cursor rules missing frontmatter."""
        no_frontmatter_rules = """
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(no_frontmatter_rules)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_missing_title(self, rule_transformer):
        """Test validation of cursor rules missing title."""
        no_title_rules = """
        ---
        description: React guidelines
        ---
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(no_title_rules)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_missing_sections(self, rule_transformer):
        """Test validation of cursor rules missing required sections."""
        incomplete_rules = """
        ---
        description: Basic React guidelines
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(incomplete_rules)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_invalid_yaml(self, rule_transformer):
        """Test validation of cursor rules with invalid YAML."""
        invalid_yaml_rules = """
        ---
        description: React guidelines
        globs: ["**/*.jsx"  # Missing closing bracket
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(invalid_yaml_rules)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_empty_content(self, rule_transformer):
        """Test validation of empty cursor rules."""
        empty_rules = ""
        
        is_valid = rule_transformer._validate_cursor_rules_structure(empty_rules)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_cursor_rules_validation_whitespace_only(self, rule_transformer):
        """Test validation of whitespace-only cursor rules."""
        whitespace_rules = "   \n  \t  \n  "
        
        is_valid = rule_transformer._validate_cursor_rules_structure(whitespace_rules)
        assert is_valid is False


class TestCursorRulesValidationScoring:
    """Test cursor rules validation scoring system."""

    @pytest.fixture
    def ml_cursor_transformer(self):
        """Create MLCursorTransformer for testing."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine'), \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine'):
            return MLCursorTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rules_scoring_perfect(self, ml_cursor_transformer):
        """Test scoring of perfect cursor rules."""
        perfect_rules = """
        ---
        description: Perfect React guidelines
        globs: ["**/*.jsx", "**/*.tsx"]
        ---
        # Perfect React Guidelines
        
        ## Key Principles
        - Use functional components
        - Keep components small
        
        ## Code Style
        - Use camelCase
        - Use PascalCase for components
        
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
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(perfect_rules)
        
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.9
        assert len(validation_result['missing_sections']) == 0

    @pytest.mark.asyncio
    async def test_cursor_rules_scoring_good(self, ml_cursor_transformer):
        """Test scoring of good cursor rules."""
        good_rules = """
        ---
        description: Good React guidelines
        globs: ["**/*.jsx"]
        ---
        # Good React Guidelines
        
        ## Key Principles
        - Use functional components
        
        ## Code Style
        - Use camelCase
        
        ## Best Practices
        - Use TypeScript
        
        ## Error Handling
        - Use try-catch blocks
        
        ## Performance
        - Use React.memo
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(good_rules)
        
        assert validation_result['is_valid'] is True
        assert 0.7 <= validation_result['score'] < 0.9
        assert 'Critical Instructions' in validation_result['missing_sections']

    @pytest.mark.asyncio
    async def test_cursor_rules_scoring_basic(self, ml_cursor_transformer):
        """Test scoring of basic cursor rules."""
        basic_rules = """
        ---
        description: Basic React guidelines
        ---
        # Basic React Guidelines
        
        ## Code Style
        - Use functional components
        
        ## Best Practices
        - Use TypeScript
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(basic_rules)
        
        assert validation_result['is_valid'] is False
        assert 0.3 <= validation_result['score'] < 0.7
        assert len(validation_result['missing_sections']) >= 4

    @pytest.mark.asyncio
    async def test_cursor_rules_scoring_poor(self, ml_cursor_transformer):
        """Test scoring of poor cursor rules."""
        poor_rules = """
        ---
        description: Poor guidelines
        ---
        # Poor Guidelines
        
        ## Code Style
        - Use consistent formatting
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(poor_rules)
        
        assert validation_result['is_valid'] is False
        assert validation_result['score'] < 0.3
        assert len(validation_result['missing_sections']) >= 5

    @pytest.mark.asyncio
    async def test_cursor_rules_scoring_quality_indicators(self, ml_cursor_transformer):
        """Test scoring with quality indicators."""
        quality_rules = """
        ---
        description: High-quality React guidelines
        globs: ["**/*.jsx", "**/*.tsx"]
        ---
        # High-Quality React Guidelines
        
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
        - Use try-catch blocks for async operations
        - Implement error boundaries for component errors
        
        ## Performance
        - Use React.memo for expensive components
        - Use useMemo and useCallback appropriately
        
        ## Critical Instructions
        - Never mutate props or state directly
        - Always use keys in lists
        """
        
        validation_result = ml_cursor_transformer._validate_cursor_rules_structure(quality_rules)
        
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.8
        
        quality_indicators = validation_result['quality_indicators']
        assert 'has_examples' in quality_indicators
        assert 'has_best_practices' in quality_indicators
        assert 'has_performance_tips' in quality_indicators
        assert 'has_error_handling' in quality_indicators
        assert 'has_critical_instructions' in quality_indicators


class TestCursorRulesEnhancement:
    """Test cursor rules enhancement functionality."""

    @pytest.fixture
    def rule_transformer(self):
        """Create RuleTransformer for testing."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine'), \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine'), \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer'), \
             patch('rules_maker.transformers.rule_transformer.UsageTracker'):
            return RuleTransformer()

    @pytest.fixture
    def ml_cursor_transformer(self):
        """Create MLCursorTransformer for testing."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine'), \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine'):
            return MLCursorTransformer()

    @pytest.fixture
    def cursor_rule_transformer(self):
        """Create CursorRuleTransformer for testing."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine'), \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine'):
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_basic(self, rule_transformer):
        """Test basic cursor rules enhancement."""
        basic_rules = """
        ---
        description: Basic React guidelines
        ---
        # Basic React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = rule_transformer._enhance_cursor_rules_structure(basic_rules)
        
        assert enhanced_rules is not None
        assert isinstance(enhanced_rules, str)
        assert len(enhanced_rules) > len(basic_rules)
        
        # Check that required sections were added
        required_sections = ["Key Principles", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in enhanced_rules
        
        # Check that original content was preserved
        assert "## Code Style" in enhanced_rules
        assert "Use functional components" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_no_frontmatter(self, rule_transformer):
        """Test cursor rules enhancement without frontmatter."""
        no_frontmatter_rules = """
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = rule_transformer._enhance_cursor_rules_structure(no_frontmatter_rules)
        
        assert enhanced_rules is not None
        assert "---" in enhanced_rules
        assert "description:" in enhanced_rules
        assert "globs:" in enhanced_rules
        
        # Check that required sections were added
        required_sections = ["Key Principles", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_partial_sections(self, rule_transformer):
        """Test cursor rules enhancement with partial sections."""
        partial_rules = """
        ---
        description: Partial React guidelines
        ---
        # Partial React Guidelines
        
        ## Code Style
        - Use functional components
        - Use camelCase
        
        ## Best Practices
        - Use TypeScript
        """
        
        enhanced_rules = rule_transformer._enhance_cursor_rules_structure(partial_rules)
        
        assert enhanced_rules is not None
        
        # Check that missing sections were added
        missing_sections = ["Key Principles", "Error Handling", "Performance", "Critical Instructions"]
        for section in missing_sections:
            assert f"## {section}" in enhanced_rules
        
        # Check that existing sections were preserved
        assert "## Code Style" in enhanced_rules
        assert "## Best Practices" in enhanced_rules
        assert "Use functional components" in enhanced_rules
        assert "Use TypeScript" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_ml_transformer(self, ml_cursor_transformer):
        """Test cursor rules enhancement with MLCursorTransformer."""
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
        assert isinstance(enhanced_rules, str)
        assert len(enhanced_rules) > len(basic_rules)
        
        # Check that required sections were added
        required_sections = ["Key Principles", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_cursor_transformer(self, cursor_rule_transformer):
        """Test cursor rules enhancement with CursorRuleTransformer."""
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
        assert isinstance(enhanced_rules, str)
        assert len(enhanced_rules) > len(basic_rules)
        
        # Check that required sections were added
        required_sections = ["Key Principles", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in enhanced_rules

    @pytest.mark.asyncio
    async def test_cursor_rules_enhancement_quality_improvement(self, rule_transformer):
        """Test cursor rules enhancement quality improvement."""
        basic_rules = """
        ---
        description: Basic guidelines
        ---
        # Basic Guidelines
        
        ## Code Style
        - Use consistent formatting
        """
        
        enhanced_rules = rule_transformer._enhance_cursor_rules_structure(basic_rules)
        
        assert enhanced_rules is not None
        
        # Check that enhanced sections have meaningful content
        assert "functional components" in enhanced_rules.lower()
        assert "error handling" in enhanced_rules.lower()
        assert "performance" in enhanced_rules.lower()
        assert "best practices" in enhanced_rules.lower()
        assert "critical instructions" in enhanced_rules.lower()


class TestCursorRulesValidationConsistency:
    """Test cursor rules validation consistency across transformers."""

    @pytest.fixture
    def rule_transformer(self):
        """Create RuleTransformer for testing."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine'), \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine'), \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer'), \
             patch('rules_maker.transformers.rule_transformer.UsageTracker'):
            return RuleTransformer()

    @pytest.fixture
    def ml_cursor_transformer(self):
        """Create MLCursorTransformer for testing."""
        with patch('rules_maker.transformers.ml_cursor_transformer.LearningEngine'), \
             patch('rules_maker.transformers.ml_cursor_transformer.IntelligentCategoryEngine'):
            return MLCursorTransformer()

    @pytest.fixture
    def cursor_rule_transformer(self):
        """Create CursorRuleTransformer for testing."""
        with patch('rules_maker.transformers.cursor_transformer.LearningEngine'), \
             patch('rules_maker.transformers.cursor_transformer.IntelligentCategoryEngine'):
            return CursorRuleTransformer()

    @pytest.mark.asyncio
    async def test_validation_consistency_across_transformers(self, rule_transformer, ml_cursor_transformer, cursor_rule_transformer):
        """Test validation consistency across all transformers."""
        test_rules = """
        ---
        description: Test React guidelines
        globs: ["**/*.jsx"]
        ---
        # Test React Guidelines
        
        ## Key Principles
        - Use functional components
        
        ## Code Style
        - Use camelCase
        
        ## Best Practices
        - Use TypeScript
        
        ## Error Handling
        - Use try-catch blocks
        
        ## Performance
        - Use React.memo
        
        ## Critical Instructions
        - Never mutate props
        """
        
        # Test with RuleTransformer
        rt_valid = rule_transformer._validate_cursor_rules_structure(test_rules)
        
        # Test with MLCursorTransformer
        ml_valid = ml_cursor_transformer._validate_cursor_rules_structure(test_rules)
        
        # Test with CursorRuleTransformer
        cr_valid = cursor_rule_transformer._validate_cursor_rules_structure(test_rules)
        
        # All should return the same validation result
        assert rt_valid == ml_valid['is_valid']
        assert rt_valid == cr_valid['is_valid']

    @pytest.mark.asyncio
    async def test_enhancement_consistency_across_transformers(self, rule_transformer, ml_cursor_transformer, cursor_rule_transformer):
        """Test enhancement consistency across all transformers."""
        basic_rules = """
        ---
        description: Basic guidelines
        ---
        # Basic Guidelines
        
        ## Code Style
        - Use consistent formatting
        """
        
        # Test enhancement with all transformers
        rt_enhanced = rule_transformer._enhance_cursor_rules_structure(basic_rules)
        ml_enhanced = ml_cursor_transformer._enhance_cursor_rules_structure(basic_rules)
        cr_enhanced = cursor_rule_transformer._enhance_cursor_rules_structure(basic_rules)
        
        # All should add the same required sections
        required_sections = ["Key Principles", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        
        for section in required_sections:
            assert f"## {section}" in rt_enhanced
            assert f"## {section}" in ml_enhanced
            assert f"## {section}" in cr_enhanced

    @pytest.mark.asyncio
    async def test_pattern_matching_consistency(self, rule_transformer, ml_cursor_transformer, cursor_rule_transformer):
        """Test pattern matching consistency across transformers."""
        valid_patterns = [
            "---\ndescription: Test\nglobs: ['**/*.js']\n---\n# Title\n## Section",
            "---\ndescription: Test\nglobs: ['**/*.py', '**/*.pyi']\n---\n# Title\n## Section",
            "---\ndescription: Test\nglobs: ['**/*.tsx', '**/*.jsx']\n---\n# Title\n## Section"
        ]
        
        for pattern in valid_patterns:
            # Test with RuleTransformer
            rt_valid = rule_transformer._validate_cursor_rules_structure(pattern)
            
            # Test with MLCursorTransformer
            ml_valid = ml_cursor_transformer._validate_cursor_rules_structure(pattern)
            
            # Test with CursorRuleTransformer
            cr_valid = cursor_rule_transformer._validate_cursor_rules_structure(pattern)
            
            # All should return the same validation result
            assert rt_valid == ml_valid['is_valid']
            assert rt_valid == cr_valid['is_valid']

    @pytest.mark.asyncio
    async def test_quality_scoring_consistency(self, ml_cursor_transformer, cursor_rule_transformer):
        """Test quality scoring consistency across transformers."""
        test_rules = """
        ---
        description: Test guidelines
        globs: ["**/*.jsx"]
        ---
        # Test Guidelines
        
        ## Key Principles
        - Use functional components
        
        ## Code Style
        - Use camelCase
        
        ## Best Practices
        - Use TypeScript
        
        ## Error Handling
        - Use try-catch blocks
        
        ## Performance
        - Use React.memo
        
        ## Critical Instructions
        - Never mutate props
        """
        
        # Test with MLCursorTransformer
        ml_result = ml_cursor_transformer._validate_cursor_rules_structure(test_rules)
        
        # Test with CursorRuleTransformer
        cr_result = cursor_rule_transformer._validate_cursor_rules_structure(test_rules)
        
        # Scores should be similar (within 0.1)
        assert abs(ml_result['score'] - cr_result['score']) <= 0.1
        
        # Missing sections should be the same
        assert set(ml_result['missing_sections']) == set(cr_result['missing_sections'])


class TestCursorRulesValidationEdgeCases:
    """Test cursor rules validation edge cases."""

    @pytest.fixture
    def rule_transformer(self):
        """Create RuleTransformer for testing."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine'), \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine'), \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer'), \
             patch('rules_maker.transformers.rule_transformer.UsageTracker'):
            return RuleTransformer()

    @pytest.mark.asyncio
    async def test_validation_unicode_content(self, rule_transformer):
        """Test validation with unicode content."""
        unicode_rules = """
        ---
        description: Guidelines with unicode: 测试
        globs: ["**/*.jsx"]
        ---
        # Guidelines with Unicode: 测试
        
        ## Code Style
        - Use functional components: 使用函数组件
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(unicode_rules)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validation_special_characters(self, rule_transformer):
        """Test validation with special characters."""
        special_rules = """
        ---
        description: Guidelines with special chars: @#$%^&*()
        globs: ["**/*.jsx", "**/*.tsx"]
        ---
        # Guidelines with Special Characters
        
        ## Code Style
        - Use functional components
        - Handle special cases: @#$%^&*()
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(special_rules)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validation_very_long_content(self, rule_transformer):
        """Test validation with very long content."""
        long_content = "## Code Style\n" + "- Use functional components\n" * 1000
        long_rules = f"""
        ---
        description: Guidelines with very long content
        globs: ["**/*.jsx"]
        ---
        # Guidelines with Long Content
        
        {long_content}
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(long_rules)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validation_nested_yaml(self, rule_transformer):
        """Test validation with nested YAML structure."""
        nested_rules = """
        ---
        description: Guidelines with nested YAML
        globs: ["**/*.jsx"]
        metadata:
          author: "Test Author"
          version: "1.0.0"
          tags: ["react", "javascript"]
        ---
        # Guidelines with Nested YAML
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(nested_rules)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validation_multiple_frontmatter(self, rule_transformer):
        """Test validation with multiple frontmatter blocks."""
        multiple_frontmatter_rules = """
        ---
        description: First frontmatter
        ---
        ---
        description: Second frontmatter
        ---
        # Guidelines with Multiple Frontmatter
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(multiple_frontmatter_rules)
        # Should handle gracefully, might be valid or invalid depending on implementation
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_validation_malformed_markdown(self, rule_transformer):
        """Test validation with malformed markdown."""
        malformed_rules = """
        ---
        description: Guidelines with malformed markdown
        globs: ["**/*.jsx"]
        ---
        # Guidelines with Malformed Markdown
        
        ## Code Style
        - Use functional components
        ### Nested section without proper structure
        - Another item
        #### Too many levels
        """
        
        is_valid = rule_transformer._validate_cursor_rules_structure(malformed_rules)
        # Should handle gracefully
        assert isinstance(is_valid, bool)


class TestCursorRulesValidationPerformance:
    """Test cursor rules validation performance."""

    @pytest.fixture
    def rule_transformer(self):
        """Create RuleTransformer for testing."""
        with patch('rules_maker.transformers.rule_transformer.LearningEngine'), \
             patch('rules_maker.transformers.rule_transformer.IntelligentCategoryEngine'), \
             patch('rules_maker.transformers.rule_transformer.SemanticAnalyzer'), \
             patch('rules_maker.transformers.rule_transformer.UsageTracker'):
            return RuleTransformer()

    @pytest.mark.asyncio
    async def test_validation_performance_large_rules(self, rule_transformer):
        """Test validation performance with large rules."""
        # Create large rules content
        large_sections = []
        for i in range(100):
            large_sections.append(f"## Section {i}\n" + "- Item 1\n" * 50)
        
        large_rules = f"""
        ---
        description: Large guidelines
        globs: ["**/*.jsx"]
        ---
        # Large Guidelines
        
        {chr(10).join(large_sections)}
        """
        
        import time
        start_time = time.time()
        is_valid = rule_transformer._validate_cursor_rules_structure(large_rules)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_validation_performance_concurrent(self, rule_transformer):
        """Test validation performance with concurrent operations."""
        test_rules = """
        ---
        description: Test guidelines
        globs: ["**/*.jsx"]
        ---
        # Test Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        # Run concurrent validations
        tasks = [rule_transformer._validate_cursor_rules_structure(test_rules) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_enhancement_performance_large_rules(self, rule_transformer):
        """Test enhancement performance with large rules."""
        # Create large rules content
        large_sections = []
        for i in range(50):
            large_sections.append(f"## Section {i}\n" + "- Item 1\n" * 20)
        
        large_rules = f"""
        ---
        description: Large guidelines
        globs: ["**/*.jsx"]
        ---
        # Large Guidelines
        
        {chr(10).join(large_sections)}
        """
        
        import time
        start_time = time.time()
        enhanced_rules = rule_transformer._enhance_cursor_rules_structure(large_rules)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 2 seconds)
        assert (end_time - start_time) < 2.0
        assert enhanced_rules is not None
        assert isinstance(enhanced_rules, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
