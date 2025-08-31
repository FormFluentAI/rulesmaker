"""
Template engine for generating rules and workflows.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from jinja2 import Environment, FileSystemLoader, Template

from ..models import RuleSet, Workflow, RuleFormat


class TemplateEngine:
    """Jinja2-based template engine for generating rules and workflows."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the template engine."""
        if template_dir is None:
            # Default to templates directory in package
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['snake_case'] = self._snake_case
        self.env.filters['camel_case'] = self._camel_case
        self.env.filters['clean_text'] = self._clean_text
    
    def render_cursor_rules(self, ruleset: RuleSet, **kwargs) -> str:
        """Render a Cursor rules file."""
        template = self.env.get_template('cursor_rules.j2')
        return template.render(ruleset=ruleset, **kwargs)
    
    def render_windsurf_rules(self, ruleset: RuleSet, **kwargs) -> str:
        """Render Windsurf-compatible rules."""
        template = self.env.get_template('windsurf_rules.j2')
        return template.render(ruleset=ruleset, **kwargs)
    
    def render_workflow(self, workflow: Workflow, **kwargs) -> str:
        """Render a workflow definition."""
        template = self.env.get_template('workflow.j2')
        return template.render(workflow=workflow, **kwargs)
    
    def render_custom(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a custom template with provided context."""
        template = self.env.get_template(template_name)
        return template.render(**context)
    
    def render_from_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """Render a template from a string."""
        template = Template(template_string, environment=self.env)
        return template.render(**context)
    
    def list_templates(self) -> List[str]:
        """List available templates."""
        templates = []
        for file_path in self.template_dir.glob("**/*.j2"):
            relative_path = file_path.relative_to(self.template_dir)
            templates.append(str(relative_path))
        return templates
    
    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists."""
        template_path = self.template_dir / template_name
        return template_path.exists()
    
    @staticmethod
    def _snake_case(text: str) -> str:
        """Convert text to snake_case."""
        import re
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
        text = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', text)
        return text.lower().replace(' ', '_').replace('-', '_')
    
    @staticmethod
    def _camel_case(text: str) -> str:
        """Convert text to camelCase."""
        words = text.replace('_', ' ').replace('-', ' ').split()
        if not words:
            return text
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for template output."""
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might break templates
        text = re.sub(r'[^\w\s\-.,!?()[\]{}:;]', '', text)
        return text.strip()
