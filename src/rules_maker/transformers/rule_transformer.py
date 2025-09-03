"""
Base rule transformer class with cursor rules knowledge and learning integration.
"""

import logging
from typing import List, Any, Dict, Optional
from datetime import datetime
import re

from .base import BaseTransformer
from ..models import ScrapingResult, RuleSet, Rule

# Import learning and intelligence modules
try:
    from ..learning import LearningEngine, SemanticAnalyzer, UsageTracker
    from ..intelligence import IntelligentCategoryEngine, SmartRecommendationEngine
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)


class RuleTransformer(BaseTransformer):
    """Generic rule transformer with cursor rules knowledge and learning integration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize learning and intelligence components
        self.learning_available = LEARNING_AVAILABLE
        if self.learning_available:
            try:
                self.learning_engine = LearningEngine()
                self.semantic_analyzer = SemanticAnalyzer()
                self.usage_tracker = UsageTracker()
                self.category_engine = IntelligentCategoryEngine()
                self.recommendation_engine = SmartRecommendationEngine()
                logger.info("Learning and intelligence components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize learning components: {e}")
                self.learning_available = False
        else:
            logger.info("Learning components not available - using base functionality")
    
    def transform(self, results: List[ScrapingResult]) -> str:
        """Transform scraping results into rules format."""
        ruleset = self.generate_rules(results)
        
        if self.config.rule_format.value == "json":
            import json
            data = ruleset.model_dump() if hasattr(ruleset, 'model_dump') else ruleset.dict()
            return json.dumps(data, indent=2)
        elif self.config.rule_format.value == "yaml":
            import yaml
            data = ruleset.model_dump() if hasattr(ruleset, 'model_dump') else ruleset.dict()
            return yaml.dump(data, default_flow_style=False)
        else:
            # Default text format
            output = f"# {ruleset.name}\n\n{ruleset.description}\n\n"
            for rule in ruleset.rules:
                output += f"## {rule.title}\n{rule.description}\n\n"
            return output
    
    def generate_rules(self, results: List[ScrapingResult]) -> RuleSet:
        """Generate rules from scraping results with learning and intelligence enhancement."""
        filtered_results = self._filter_relevant_content(results)
        
        # Apply learning and intelligence enhancements if available
        if self.learning_available:
            try:
                filtered_results = self._enhance_with_learning(filtered_results)
                filtered_results = self._enhance_with_intelligence(filtered_results)
            except Exception as e:
                logger.warning(f"Learning/intelligence enhancement failed: {e}")
        
        rules = []
        for i, result in enumerate(filtered_results):
            # Extract enhanced metadata
            enhanced_metadata = self._extract_enhanced_metadata(result)
            
            rule = Rule(
                id=f"rule_{i+1}",
                title=self._generate_enhanced_title(result, enhanced_metadata),
                description=self._generate_enhanced_description(result, enhanced_metadata),
                category=self._determine_category(result, enhanced_metadata),
                tags=self._extract_enhanced_tags(result, enhanced_metadata)
            )
            rules.append(rule)
        
        # Apply cursor rules knowledge and best practices
        rules = self._apply_cursor_rules_knowledge(rules, filtered_results)
        
        return RuleSet(
            name="Generated Rules",
            description="Rules generated from scraped documentation with learning enhancement",
            rules=rules,
            format=self.config.rule_format
        )
    
    def _enhance_with_learning(self, results: List[ScrapingResult]) -> List[ScrapingResult]:
        """Enhance results using learning engine."""
        try:
            # Use semantic analyzer to enhance content understanding
            enhanced_results = []
            for result in results:
                if hasattr(self.semantic_analyzer, 'analyze_content'):
                    analysis = self.semantic_analyzer.analyze_content(result.content)
                    # Add semantic analysis to result metadata
                    if not hasattr(result, 'metadata'):
                        result.metadata = {}
                    result.metadata['semantic_analysis'] = analysis
                enhanced_results.append(result)
            return enhanced_results
        except Exception as e:
            logger.warning(f"Learning enhancement failed: {e}")
            return results
    
    def _enhance_with_intelligence(self, results: List[ScrapingResult]) -> List[ScrapingResult]:
        """Enhance results using intelligence engine."""
        try:
            # Use category engine for intelligent categorization
            enhanced_results = []
            for result in results:
                if hasattr(self.category_engine, 'categorize_content'):
                    category = self.category_engine.categorize_content(result.content)
                    if not hasattr(result, 'metadata'):
                        result.metadata = {}
                    result.metadata['intelligent_category'] = category
                enhanced_results.append(result)
            return enhanced_results
        except Exception as e:
            logger.warning(f"Intelligence enhancement failed: {e}")
            return results
    
    def _extract_enhanced_metadata(self, result: ScrapingResult) -> Dict[str, Any]:
        """Extract enhanced metadata from result."""
        metadata = {}
        
        # Extract basic metadata
        metadata['url'] = str(result.url)
        metadata['title'] = result.title
        metadata['content_length'] = len(result.content) if result.content else 0
        
        # Extract learning metadata if available
        if hasattr(result, 'metadata') and result.metadata:
            if 'semantic_analysis' in result.metadata:
                metadata['semantic_analysis'] = result.metadata['semantic_analysis']
            if 'intelligent_category' in result.metadata:
                metadata['intelligent_category'] = result.metadata['intelligent_category']
        
        return metadata
    
    def _generate_enhanced_title(self, result: ScrapingResult, metadata: Dict[str, Any]) -> str:
        """Generate enhanced title using learning insights."""
        base_title = f"Rule from {result.url}"
        
        # Use intelligent category if available
        if 'intelligent_category' in metadata:
            category = metadata['intelligent_category']
            if isinstance(category, dict) and 'primary_category' in category:
                return f"{category['primary_category'].title()} Rule from {result.url}"
        
        return base_title
    
    def _generate_enhanced_description(self, result: ScrapingResult, metadata: Dict[str, Any]) -> str:
        """Generate enhanced description using learning insights."""
        content = result.content or ""
        
        # Truncate content intelligently
        if len(content) > 500:
            # Try to find a good breaking point
            truncated = content[:500]
            last_period = truncated.rfind('.')
            if last_period > 400:  # If we can find a period near the end
                content = content[:last_period + 1] + "..."
            else:
                content = content[:500] + "..."
        
        # Add semantic insights if available
        if 'semantic_analysis' in metadata:
            analysis = metadata['semantic_analysis']
            if isinstance(analysis, dict) and 'key_concepts' in analysis:
                concepts = analysis['key_concepts'][:3]  # Top 3 concepts
                if concepts:
                    content += f"\n\nKey concepts: {', '.join(concepts)}"
        
        return content
    
    def _determine_category(self, result: ScrapingResult, metadata: Dict[str, Any]) -> str:
        """Determine category using intelligence engine."""
        # Use intelligent category if available
        if 'intelligent_category' in metadata:
            category = metadata['intelligent_category']
            if isinstance(category, dict) and 'primary_category' in category:
                return category['primary_category']
        
        # Fallback to content-based categorization
        content = (result.content or "").lower()
        if any(tech in content for tech in ['react', 'vue', 'angular', 'svelte']):
            return 'frontend'
        elif any(tech in content for tech in ['python', 'django', 'flask', 'fastapi']):
            return 'backend'
        elif any(tech in content for tech in ['api', 'endpoint', 'rest', 'graphql']):
            return 'api'
        else:
            return 'documentation'
    
    def _extract_enhanced_tags(self, result: ScrapingResult, metadata: Dict[str, Any]) -> List[str]:
        """Extract enhanced tags using learning insights."""
        tags = set()
        
        # Extract from semantic analysis if available
        if 'semantic_analysis' in metadata:
            analysis = metadata['semantic_analysis']
            if isinstance(analysis, dict):
                if 'key_concepts' in analysis:
                    tags.update(analysis['key_concepts'][:5])  # Top 5 concepts
                if 'technologies' in analysis:
                    tags.update(analysis['technologies'][:3])  # Top 3 technologies
        
        # Extract from intelligent category
        if 'intelligent_category' in metadata:
            category = metadata['intelligent_category']
            if isinstance(category, dict):
                if 'tags' in category:
                    tags.update(category['tags'][:3])  # Top 3 tags
        
        # Fallback to basic concept extraction
        if not tags:
            tags.update(self._extract_key_concepts(result.content))
        
        return list(tags)[:10]  # Limit to 10 tags
    
    def _apply_cursor_rules_knowledge(self, rules: List[Rule], results: List[ScrapingResult]) -> List[Rule]:
        """Apply cursor rules knowledge and best practices."""
        enhanced_rules = []
        
        for rule in rules:
            # Apply cursor rules best practices
            enhanced_rule = self._enhance_rule_with_cursor_knowledge(rule, results)
            enhanced_rules.append(enhanced_rule)
        
        return enhanced_rules
    
    def _enhance_rule_with_cursor_knowledge(self, rule: Rule, results: List[ScrapingResult]) -> Rule:
        """Enhance individual rule with cursor rules knowledge."""
        # Apply cursor rules formatting and structure
        enhanced_description = self._apply_cursor_rules_formatting(rule.description)
        
        # Create enhanced rule
        enhanced_rule = Rule(
            id=rule.id,
            title=rule.title,
            description=enhanced_description,
            category=rule.category,
            tags=rule.tags
        )
        
        return enhanced_rule
    
    def _apply_cursor_rules_formatting(self, description: str) -> str:
        """Apply cursor rules formatting and structure."""
        # Add cursor rules header if not present
        if not description.startswith("# "):
            description = f"# Cursor Rules\n\n{description}"
        
        # Ensure proper markdown structure
        lines = description.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Ensure proper heading hierarchy
            if line.startswith('##') and not line.startswith('###'):
                # Check if this should be a main section
                if any(keyword in line.lower() for keyword in ['principles', 'style', 'practices', 'guidelines']):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(f"### {line[2:].strip()}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _validate_cursor_rules_structure(self, rules_content: str) -> bool:
        """Validate cursor rules structure and return True if valid."""
        if not rules_content or not rules_content.strip():
            return False
        
        # Check for YAML frontmatter
        if not rules_content.strip().startswith('---'):
            return False
        
        # Check for proper YAML frontmatter structure
        lines = rules_content.strip().split('\n')
        if len(lines) < 3 or lines[0] != '---' or lines[1].strip() == '':
            return False
        
        # Find the end of frontmatter
        frontmatter_end = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                frontmatter_end = i
                break
        
        if frontmatter_end == -1:
            return False
        
        # Check for title after frontmatter
        content_after_frontmatter = '\n'.join(lines[frontmatter_end + 1:])
        if not content_after_frontmatter.strip().startswith('# '):
            return False
        
        # Check for at least one section
        if '## ' not in content_after_frontmatter:
            return False
        
        # Check for required sections (at least 3 of the main sections)
        required_sections = ['Key Principles', 'Code Style', 'Best Practices', 'Error Handling', 'Performance', 'Critical Instructions']
        found_sections = 0
        for section in required_sections:
            if f"## {section}" in content_after_frontmatter:
                found_sections += 1
        
        # Require at least 3 sections for a valid cursor rules file
        if found_sections < 3:
            return False
        
        return True
    
    def _enhance_cursor_rules_structure(self, rules_content: str) -> str:
        """Enhance cursor rules structure by adding missing sections."""
        if not rules_content or not rules_content.strip():
            # Return basic template if empty
            return """---
description: Basic guidelines
globs: ["**/*"]
---
# Basic Guidelines

## Key Principles
- Follow best practices
- Write clean, maintainable code
- Document your code

## Code Style
- Use consistent formatting
- Follow naming conventions
- Keep functions small and focused

## Best Practices
- Write tests for your code
- Use version control
- Review code before merging

## Error Handling
- Handle errors gracefully
- Log important events
- Provide meaningful error messages

## Performance
- Optimize for readability first
- Profile before optimizing
- Use appropriate data structures

## Critical Instructions
- Never commit secrets
- Always validate input
- Keep dependencies up to date
"""
        
        enhanced_content = rules_content
        
        # Add frontmatter if missing
        if not enhanced_content.strip().startswith('---'):
            enhanced_content = """---
description: Enhanced guidelines
globs: ["**/*"]
---
""" + enhanced_content
        
        # Add missing sections
        required_sections = [
            ("## Key Principles", "- Follow best practices\n- Write clean, maintainable code\n- Document your code"),
            ("## Best Practices", "- Write tests for your code\n- Use version control\n- Review code before merging"),
            ("## Error Handling", "- Handle errors gracefully\n- Log important events\n- Provide meaningful error messages"),
            ("## Performance", "- Optimize for readability first\n- Profile before optimizing\n- Use appropriate data structures"),
            ("## Critical Instructions", "- Never commit secrets\n- Always validate input\n- Keep dependencies up to date")
        ]
        
        for section_header, section_content in required_sections:
            if section_header not in enhanced_content:
                enhanced_content += f"\n\n{section_header}\n{section_content}"
        
        return enhanced_content