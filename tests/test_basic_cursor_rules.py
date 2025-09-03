"""
Basic tests for cursor rules functionality without learning/intelligence dependencies.

These tests verify the basic cursor rules validation and enhancement functionality
without requiring the learning and intelligence modules that may not be implemented yet.
"""

import pytest
from rules_maker.transformers.rule_transformer import RuleTransformer
from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer


class TestBasicCursorRulesValidation:
    """Test basic cursor rules validation functionality."""

    def test_rule_transformer_initialization(self):
        """Test RuleTransformer initialization."""
        transformer = RuleTransformer()
        assert transformer is not None

    def test_ml_cursor_transformer_initialization(self):
        """Test MLCursorTransformer initialization."""
        transformer = MLCursorTransformer()
        assert transformer is not None

    def test_cursor_rule_transformer_initialization(self):
        """Test CursorRuleTransformer initialization."""
        transformer = CursorRuleTransformer()
        assert transformer is not None

    def test_cursor_rules_validation_perfect_structure(self):
        """Test validation of perfectly structured cursor rules."""
        transformer = RuleTransformer()
        
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
        
        is_valid = transformer._validate_cursor_rules_structure(perfect_rules)
        assert is_valid is True

    def test_cursor_rules_validation_missing_frontmatter(self):
        """Test validation of cursor rules missing frontmatter."""
        transformer = RuleTransformer()
        
        no_frontmatter_rules = """
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = transformer._validate_cursor_rules_structure(no_frontmatter_rules)
        assert is_valid is False

    def test_cursor_rules_validation_missing_title(self):
        """Test validation of cursor rules missing title."""
        transformer = RuleTransformer()
        
        no_title_rules = """
        ---
        description: React guidelines
        ---
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = transformer._validate_cursor_rules_structure(no_title_rules)
        assert is_valid is False

    def test_cursor_rules_validation_missing_sections(self):
        """Test validation of cursor rules missing required sections."""
        transformer = RuleTransformer()
        
        incomplete_rules = """
        ---
        description: Basic React guidelines
        ---
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        is_valid = transformer._validate_cursor_rules_structure(incomplete_rules)
        assert is_valid is False

    def test_cursor_rules_enhancement_basic(self):
        """Test basic cursor rules enhancement."""
        transformer = RuleTransformer()
        
        basic_rules = """
        ---
        description: Basic React guidelines
        ---
        # Basic React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = transformer._enhance_cursor_rules_structure(basic_rules)
        
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

    def test_cursor_rules_enhancement_no_frontmatter(self):
        """Test cursor rules enhancement without frontmatter."""
        transformer = RuleTransformer()
        
        no_frontmatter_rules = """
        # React Guidelines
        
        ## Code Style
        - Use functional components
        """
        
        enhanced_rules = transformer._enhance_cursor_rules_structure(no_frontmatter_rules)
        
        assert enhanced_rules is not None
        assert "---" in enhanced_rules
        assert "description:" in enhanced_rules
        assert "globs:" in enhanced_rules
        
        # Check that required sections were added
        required_sections = ["Key Principles", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in enhanced_rules


class TestMLCursorTransformerBasic:
    """Test basic MLCursorTransformer functionality."""

    def test_ml_cursor_transformer_validation_scoring(self):
        """Test MLCursorTransformer validation scoring."""
        transformer = MLCursorTransformer()
        
        perfect_rules = """
        ---
        description: Perfect React guidelines
        globs: ["**/*.jsx", "**/*.tsx"]
        ---
        # Perfect React Guidelines
        
        You are an expert React developer with deep knowledge of best practices.
        
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
        - **NEVER:** mutate props directly
        - **ALWAYS:** use keys in lists
        """
        
        validation_result = transformer._validate_cursor_rules_structure(perfect_rules)
        
        assert validation_result['is_valid'] is True
        assert validation_result['score'] >= 0.9
        assert len(validation_result['missing_sections']) == 0

    def test_ml_cursor_transformer_enhancement(self):
        """Test MLCursorTransformer enhancement functionality."""
        transformer = MLCursorTransformer()
        
        basic_rules = """
        ---
        description: Basic guidelines
        ---
        # Basic Guidelines
        
        ## Code Style
        - Use consistent formatting
        """
        
        # First validate to get validation result
        validation_result = transformer._validate_cursor_rules_structure(basic_rules)
        
        # Then enhance using the validation result
        enhanced_rules = transformer._enhance_cursor_rules_structure(basic_rules, validation_result)
        
        assert enhanced_rules is not None
        assert isinstance(enhanced_rules, str)
        assert len(enhanced_rules) > len(basic_rules)
        
        # Check that some required sections were added (based on what's actually missing)
        # The basic rules only has "Code Style", so Key Principles and Best Practices should be added
        assert "## Key Principles" in enhanced_rules
        assert "## Best Practices" in enhanced_rules
        # The original Code Style section should still be there
        assert "## Code Style" in enhanced_rules


class TestCursorRuleTransformerBasic:
    """Test basic CursorRuleTransformer functionality."""

    def test_cursor_rule_transformer_knowledge_base(self):
        """Test CursorRuleTransformer knowledge base structure."""
        transformer = CursorRuleTransformer()
        
        assert transformer is not None
        assert hasattr(transformer, 'cursor_rules_knowledge')
        
        knowledge = transformer.cursor_rules_knowledge
        assert 'required_sections' in knowledge
        assert 'technology_guidelines' in knowledge
        assert 'cursor_patterns' in knowledge
        assert 'quality_indicators' in knowledge

    def test_technology_guidelines_retrieval(self):
        """Test technology guidelines retrieval."""
        transformer = CursorRuleTransformer()
        
        # Test React guidelines
        react_guidelines = transformer._get_technology_guidelines("react")
        assert react_guidelines is not None
        assert isinstance(react_guidelines, str)
        assert len(react_guidelines) > 0
        assert "functional components" in react_guidelines.lower()
        assert "hooks" in react_guidelines.lower()
        
        # Test Python guidelines
        python_guidelines = transformer._get_technology_guidelines("python")
        assert python_guidelines is not None
        assert isinstance(python_guidelines, str)
        assert len(python_guidelines) > 0
        assert "pep 8" in python_guidelines.lower()
        assert "type hints" in python_guidelines.lower()

    def test_cursor_rule_generation_basic(self):
        """Test basic cursor rule generation."""
        transformer = CursorRuleTransformer()
        
        # Create simple content sections
        from rules_maker.models import ContentSection
        
        content_sections = [
            ContentSection(
                title="React Component",
                content="React component example",
                section_type="code_example",
                metadata={"framework": "react", "language": "javascript"}
            )
        ]
        
        result = transformer.generate_cursor_rule(content_sections, "React Guidelines")
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Check YAML frontmatter
        assert "---" in result
        assert "description:" in result
        assert "globs:" in result
        
        # Check title
        assert "# React Guidelines" in result
        
        # Check required sections
        required_sections = ["Key Principles", "Code Style", "Best Practices", "Error Handling", "Performance", "Critical Instructions"]
        for section in required_sections:
            assert f"## {section}" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
