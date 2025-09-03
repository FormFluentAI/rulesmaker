"""
Cursor Rules Formatter

Specialized formatter for converting processed documentation into proper
.cursor/rules format with frontmatter, structured content, and best practices.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import yaml
import json

from ..models import ScrapingResult, RuleSet, Rule

# Conditional import to avoid circular dependencies
try:
    from ..intelligence.nextjs_categorizer import NextJSCategorizer, NextJSCategory
    NEXTJS_CATEGORIZER_AVAILABLE = True
except ImportError:
    NEXTJS_CATEGORIZER_AVAILABLE = False
    NextJSCategorizer = None
    NextJSCategory = None

logger = logging.getLogger(__name__)


@dataclass
class CursorRuleMetadata:
    """Metadata for cursor rule files."""
    description: str
    globs: List[str]
    always_apply: bool = False
    trigger: Optional[str] = None
    priority: int = 0
    tags: List[str] = None
    version: str = "1.0.0"
    last_updated: str = None


@dataclass
class CursorRuleContent:
    """Structured content for cursor rules."""
    title: str
    description: str
    examples: List[str] = None
    guidelines: List[str] = None
    anti_patterns: List[str] = None
    related_concepts: List[str] = None
    difficulty_level: str = "intermediate"
    category: str = "general"


class CursorRulesFormatter:
    """Formatter for converting documentation into cursor rules format."""
    
    def __init__(self, nextjs_categorizer: Optional[NextJSCategorizer] = None):
        """Initialize the cursor rules formatter.
        
        Args:
            nextjs_categorizer: Optional Next.js categorizer for enhanced categorization
        """
        if NEXTJS_CATEGORIZER_AVAILABLE and nextjs_categorizer is None:
            self.nextjs_categorizer = NextJSCategorizer()
        else:
            self.nextjs_categorizer = nextjs_categorizer
        
        # Default globs for Next.js projects
        self.default_globs = [
            "**/*.js",
            "**/*.jsx", 
            "**/*.ts",
            "**/*.tsx",
            "**/next.config.*",
            "**/middleware.ts",
            "**/app/**/*",
            "**/pages/**/*",
            "**/components/**/*",
            "**/lib/**/*",
            "**/utils/**/*"
        ]
        
        # Category-specific globs
        self.category_globs = {
            "routing": [
                "**/app/**/*",
                "**/pages/**/*",
                "**/middleware.ts",
                "**/route.ts"
            ],
            "styling": [
                "**/*.css",
                "**/*.scss",
                "**/*.sass",
                "**/styles/**/*",
                "**/components/**/*"
            ],
            "api-routes": [
                "**/api/**/*",
                "**/route.ts",
                "**/pages/api/**/*"
            ],
            "configuration": [
                "**/next.config.*",
                "**/.env*",
                "**/package.json",
                "**/tsconfig.json"
            ],
            "testing": [
                "**/*.test.*",
                "**/*.spec.*",
                "**/__tests__/**/*",
                "**/tests/**/*"
            ]
        }
    
    async def format_scraping_results(
        self,
        results: List[ScrapingResult],
        category_hint: Optional[str] = None,
        output_format: str = "mdc"
    ) -> Dict[str, str]:
        """Format scraping results into cursor rules.
        
        Args:
            results: List of scraping results to format
            category_hint: Optional category hint for focused formatting
            output_format: Output format (mdc, json, yaml)
            
        Returns:
            Dictionary mapping filenames to formatted content
        """
        logger.info(f"Formatting {len(results)} scraping results into cursor rules")
        
        formatted_rules = {}
        
        # Group results by category
        categorized_results = await self._categorize_results(results, category_hint)
        
        # Format each category
        for category, category_results in categorized_results.items():
            if category_results:
                # Generate metadata
                metadata = self._generate_metadata(category, category_results)
                
                # Generate content
                content = self._generate_content(category, category_results)
                
                # Format according to output format
                if output_format == "mdc":
                    formatted_content = self._format_as_mdc(metadata, content)
                elif output_format == "json":
                    formatted_content = self._format_as_json(metadata, content)
                elif output_format == "yaml":
                    formatted_content = self._format_as_yaml(metadata, content)
                else:
                    formatted_content = self._format_as_mdc(metadata, content)
                
                # Generate filename
                filename = self._generate_filename(category, output_format)
                formatted_rules[filename] = formatted_content
        
        return formatted_rules
    
    async def _categorize_results(
        self,
        results: List[ScrapingResult],
        category_hint: Optional[str] = None
    ) -> Dict[str, List[ScrapingResult]]:
        """Categorize scraping results."""
        categorized = {}
        
        for result in results:
            # Use category hint if provided
            if category_hint:
                category = category_hint
            else:
                # Use Next.js categorizer for intelligent categorization if available
                if self.nextjs_categorizer and NEXTJS_CATEGORIZER_AVAILABLE:
                    try:
                        categories = await self.nextjs_categorizer.categorize_nextjs_content(
                            result.content, str(result.url)
                        )
                        # Get the highest confidence category
                        if categories:
                            category = max(categories.items(), key=lambda x: x[1].confidence)[0]
                        else:
                            category = "general"
                    except Exception as e:
                        logger.warning(f"Categorization failed for {result.url}: {e}")
                        category = "general"
                else:
                    category = "general"
            
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(result)
        
        return categorized
    
    def _generate_metadata(
        self,
        category: str,
        results: List[ScrapingResult]
    ) -> CursorRuleMetadata:
        """Generate metadata for cursor rules."""
        # Get category-specific globs
        globs = self.category_globs.get(category, self.default_globs)
        
        # Generate description
        description = self._generate_description(category, results)
        
        # Determine if rules should always apply
        always_apply = category in ["routing", "configuration", "performance"]
        
        # Extract tags from results
        tags = self._extract_tags(results)
        
        return CursorRuleMetadata(
            description=description,
            globs=globs,
            always_apply=always_apply,
            tags=tags,
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_description(
        self,
        category: str,
        results: List[ScrapingResult]
    ) -> str:
        """Generate description for cursor rules."""
        descriptions = {
            "routing": "Next.js routing and navigation patterns for App Router and Pages Router",
            "data-fetching": "Data fetching patterns including Server Components, Client Components, and API integration",
            "styling": "Styling approaches including CSS Modules, Tailwind CSS, and styled-components",
            "deployment": "Deployment strategies for Next.js applications including Vercel, Docker, and static export",
            "performance": "Performance optimization techniques for Next.js applications",
            "api-routes": "API route handlers and serverless function patterns",
            "middleware": "Next.js middleware patterns for request/response handling",
            "configuration": "Next.js configuration and environment setup",
            "testing": "Testing strategies for Next.js applications",
            "security": "Security best practices for Next.js applications",
            "optimization": "Advanced optimization techniques and patterns",
            "troubleshooting": "Common issues and troubleshooting guides",
            "migration": "Migration guides and upgrade strategies",
            "advanced-patterns": "Advanced Next.js patterns and techniques"
        }
        
        return descriptions.get(category, f"Next.js {category} development guidelines and best practices")
    
    def _extract_tags(self, results: List[ScrapingResult]) -> List[str]:
        """Extract tags from scraping results."""
        tags = set()
        
        for result in results:
            # Extract tags from URL
            url_str = str(result.url)
            url_parts = url_str.split('/')
            for part in url_parts:
                if part and len(part) > 2:
                    tags.add(part.lower())
            
            # Extract tags from content
            content_tags = re.findall(r'#(\w+)', result.content)
            tags.update(content_tags)
            
            # Extract technology tags
            tech_tags = re.findall(r'\b(React|TypeScript|JavaScript|CSS|HTML|Node\.js)\b', result.content)
            tags.update(tag.lower() for tag in tech_tags)
        
        return list(tags)[:10]  # Limit to 10 tags
    
    def _generate_content(
        self,
        category: str,
        results: List[ScrapingResult]
    ) -> List[CursorRuleContent]:
        """Generate structured content for cursor rules."""
        content_items = []
        
        for result in results:
            # Extract title
            title = self._extract_title(result)
            
            # Extract description
            description = self._extract_description(result)
            
            # Extract examples
            examples = self._extract_examples(result)
            
            # Extract guidelines
            guidelines = self._extract_guidelines(result)
            
            # Extract anti-patterns
            anti_patterns = self._extract_anti_patterns(result)
            
            # Determine difficulty level
            difficulty_level = self._determine_difficulty(result)
            
            content_item = CursorRuleContent(
                title=title,
                description=description,
                examples=examples,
                guidelines=guidelines,
                anti_patterns=anti_patterns,
                difficulty_level=difficulty_level,
                category=category
            )
            
            content_items.append(content_item)
        
        return content_items
    
    def _extract_title(self, result: ScrapingResult) -> str:
        """Extract title from scraping result."""
        if result.title:
            return result.title
        
        # Extract from URL
        url_str = str(result.url)
        url_parts = url_str.split('/')
        if url_parts:
            last_part = url_parts[-1]
            if last_part and last_part != 'docs':
                return last_part.replace('-', ' ').replace('_', ' ').title()
        
        return "Next.js Development Rule"
    
    def _extract_description(self, result: ScrapingResult) -> str:
        """Extract description from scraping result."""
        if result.content:
            # Extract first paragraph or summary
            paragraphs = result.content.split('\n\n')
            for paragraph in paragraphs:
                if len(paragraph.strip()) > 50:  # Meaningful paragraph
                    return paragraph.strip()[:500] + "..." if len(paragraph) > 500 else paragraph.strip()
        
        return "Next.js development guideline and best practice."
    
    def _extract_examples(self, result: ScrapingResult) -> List[str]:
        """Extract code examples from scraping result."""
        examples = []
        
        if result.content:
            # Extract code blocks
            code_blocks = re.findall(r'```(?:typescript|javascript|jsx|tsx|js|ts)?\n(.*?)\n```', result.content, re.DOTALL)
            examples.extend(code_blocks)
            
            # Extract inline code
            inline_code = re.findall(r'`([^`]+)`', result.content)
            examples.extend([code for code in inline_code if len(code) > 10])
        
        return examples[:5]  # Limit to 5 examples
    
    def _extract_guidelines(self, result: ScrapingResult) -> List[str]:
        """Extract guidelines from scraping result."""
        guidelines = []
        
        if result.content:
            # Extract bullet points
            bullet_points = re.findall(r'^[\s]*[-*+]\s+(.+)$', result.content, re.MULTILINE)
            guidelines.extend(bullet_points)
            
            # Extract numbered lists
            numbered_points = re.findall(r'^\s*\d+\.\s+(.+)$', result.content, re.MULTILINE)
            guidelines.extend(numbered_points)
        
        return guidelines[:10]  # Limit to 10 guidelines
    
    def _extract_anti_patterns(self, result: ScrapingResult) -> List[str]:
        """Extract anti-patterns from scraping result."""
        anti_patterns = []
        
        if result.content:
            # Look for anti-pattern indicators
            anti_pattern_indicators = [
                r'don\'t\s+(.+)',
                r'avoid\s+(.+)',
                r'never\s+(.+)',
                r'anti-pattern[:\s]+(.+)',
                r'common mistake[:\s]+(.+)'
            ]
            
            for pattern in anti_pattern_indicators:
                matches = re.findall(pattern, result.content, re.IGNORECASE)
                anti_patterns.extend(matches)
        
        return anti_patterns[:5]  # Limit to 5 anti-patterns
    
    def _determine_difficulty(self, result: ScrapingResult) -> str:
        """Determine difficulty level from scraping result."""
        if not result.content:
            return "intermediate"
        
        content_lower = result.content.lower()
        
        # Beginner indicators
        beginner_indicators = ['basic', 'simple', 'getting started', 'introduction', 'tutorial']
        if any(indicator in content_lower for indicator in beginner_indicators):
            return "beginner"
        
        # Advanced indicators
        advanced_indicators = ['advanced', 'complex', 'optimization', 'performance', 'architecture']
        if any(indicator in content_lower for indicator in advanced_indicators):
            return "advanced"
        
        # Expert indicators
        expert_indicators = ['expert', 'internals', 'deep dive', 'custom implementation']
        if any(indicator in content_lower for indicator in expert_indicators):
            return "expert"
        
        return "intermediate"
    
    def _format_as_mdc(
        self,
        metadata: CursorRuleMetadata,
        content_items: List[CursorRuleContent]
    ) -> str:
        """Format as Markdown with frontmatter (.mdc format)."""
        # Generate frontmatter
        frontmatter = {
            'description': metadata.description,
            'globs': metadata.globs,
            'alwaysApply': metadata.always_apply,
            'tags': metadata.tags,
            'version': metadata.version,
            'lastUpdated': metadata.last_updated
        }
        
        if metadata.trigger:
            frontmatter['trigger'] = metadata.trigger
        
        # Format frontmatter
        frontmatter_yaml = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        
        # Generate content
        content_lines = [
            f"---\n{frontmatter_yaml}---\n",
            f"# {content_items[0].category.title()} Development Rules\n",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        ]
        
        # Add overview
        content_lines.extend([
            "## Overview\n",
            f"{metadata.description}\n",
            f"**Difficulty Level**: {content_items[0].difficulty_level.title()}\n",
            f"**Category**: {content_items[0].category}\n\n"
        ])
        
        # Add content items
        for i, item in enumerate(content_items, 1):
            content_lines.extend([
                f"## {i}. {item.title}\n",
                f"{item.description}\n\n"
            ])
            
            # Add guidelines
            if item.guidelines:
                content_lines.extend([
                    "### Guidelines\n",
                    *[f"- {guideline}\n" for guideline in item.guidelines],
                    "\n"
                ])
            
            # Add examples
            if item.examples:
                content_lines.extend([
                    "### Examples\n",
                    *[f"```typescript\n{example}\n```\n\n" for example in item.examples]
                ])
            
            # Add anti-patterns
            if item.anti_patterns:
                content_lines.extend([
                    "### Anti-Patterns\n",
                    *[f"- ❌ {anti_pattern}\n" for anti_pattern in item.anti_patterns],
                    "\n"
                ])
            
            content_lines.append("---\n\n")
        
        return "".join(content_lines)
    
    def _format_as_json(
        self,
        metadata: CursorRuleMetadata,
        content_items: List[CursorRuleContent]
    ) -> str:
        """Format as JSON."""
        data = {
            'metadata': {
                'description': metadata.description,
                'globs': metadata.globs,
                'alwaysApply': metadata.always_apply,
                'tags': metadata.tags,
                'version': metadata.version,
                'lastUpdated': metadata.last_updated
            },
            'content': [
                {
                    'title': item.title,
                    'description': item.description,
                    'examples': item.examples,
                    'guidelines': item.guidelines,
                    'antiPatterns': item.anti_patterns,
                    'difficultyLevel': item.difficulty_level,
                    'category': item.category
                }
                for item in content_items
            ]
        }
        
        return json.dumps(data, indent=2)
    
    def _format_as_yaml(
        self,
        metadata: CursorRuleMetadata,
        content_items: List[CursorRuleContent]
    ) -> str:
        """Format as YAML."""
        data = {
            'metadata': {
                'description': metadata.description,
                'globs': metadata.globs,
                'alwaysApply': metadata.always_apply,
                'tags': metadata.tags,
                'version': metadata.version,
                'lastUpdated': metadata.last_updated
            },
            'content': [
                {
                    'title': item.title,
                    'description': item.description,
                    'examples': item.examples,
                    'guidelines': item.guidelines,
                    'antiPatterns': item.anti_patterns,
                    'difficultyLevel': item.difficulty_level,
                    'category': item.category
                }
                for item in content_items
            ]
        }
        
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    def _generate_filename(self, category: str, output_format: str) -> str:
        """Generate filename for cursor rules."""
        # Clean category name
        clean_category = re.sub(r'[^a-zA-Z0-9_-]', '-', category.lower())
        clean_category = re.sub(r'-+', '-', clean_category).strip('-')
        
        # Add extension
        if output_format == "mdc":
            extension = "mdc"
        elif output_format == "json":
            extension = "json"
        elif output_format == "yaml":
            extension = "yaml"
        else:
            extension = "mdc"
        
        return f"{clean_category}.{extension}"
    
    def save_formatted_rules(
        self,
        formatted_rules: Dict[str, str],
        output_dir: str,
        create_subdirs: bool = True
    ):
        """Save formatted rules to files.
        
        Args:
            formatted_rules: Dictionary of filename -> content
            output_dir: Output directory
            create_subdirs: Whether to create subdirectories for organization
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for filename, content in formatted_rules.items():
            if create_subdirs:
                # Create category subdirectory
                category = filename.split('.')[0]
                category_dir = output_path / category
                category_dir.mkdir(exist_ok=True)
                file_path = category_dir / filename
            else:
                file_path = output_path / filename
            
            # Write content
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Saved cursor rule: {file_path}")
    
    def generate_rule_index(
        self,
        formatted_rules: Dict[str, str],
        output_dir: str
    ) -> str:
        """Generate an index file for all cursor rules."""
        index_content = [
            "# Next.js Cursor Rules Index\n",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "## Available Rules\n\n"
        ]
        
        # Group rules by category
        categories = {}
        for filename in formatted_rules.keys():
            category = filename.split('.')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(filename)
        
        # Generate index entries
        for category, files in categories.items():
            index_content.extend([
                f"### {category.title()}\n",
                *[f"- [{filename}]({category}/{filename})\n" for filename in files],
                "\n"
            ])
        
        # Save index
        index_path = Path(output_dir) / "README.md"
        index_path.write_text("".join(index_content), encoding='utf-8')
        
        logger.info(f"Generated rule index: {index_path}")
        return str(index_path)
