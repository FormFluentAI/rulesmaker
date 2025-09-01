"""
Rule template management for Rules Maker.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from jinja2 import Template

from ..models import RuleFormat


class RuleTemplate:
    """Template for generating rules from documentation."""
    
    def __init__(self, template_content: str, format_type: RuleFormat = RuleFormat.CURSOR):
        """Initialize the rule template.
        
        Args:
            template_content: Jinja2 template content
            format_type: Target rule format
        """
        self.template = Template(template_content)
        self.format_type = format_type
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with the given context.
        
        Args:
            context: Template variables
            
        Returns:
            Rendered template content
        """
        return self.template.render(**context)
    
    @classmethod
    def from_file(cls, template_path: Path, format_type: RuleFormat = RuleFormat.CURSOR) -> 'RuleTemplate':
        """Create template from file.
        
        Args:
            template_path: Path to template file
            format_type: Target rule format
            
        Returns:
            RuleTemplate instance
        """
        content = template_path.read_text()
        return cls(content, format_type)
    
    @classmethod
    def get_default_cursor_template(cls) -> 'RuleTemplate':
        """Get default Cursor rules template."""
        template_content = """# {{ title }}

{{ description }}

## Usage Rules

{% for rule in rules %}
- {{ rule }}
{% endfor %}

## Code Examples

{% for example in examples %}
```{{ example.language }}
{{ example.code }}
```
{% endfor %}

## Best Practices

{% for practice in best_practices %}
- {{ practice }}
{% endfor %}
"""
        return cls(template_content, RuleFormat.CURSOR)
    
    @classmethod
    def get_default_windsurf_template(cls) -> 'RuleTemplate':
        """Get default Windsurf workflow template."""
        template_content = """workflow:
  name: "{{ title }}"
  description: "{{ description }}"
  
  steps:
{% for step in steps %}
    - name: "{{ step.name }}"
      action: "{{ step.action }}"
      parameters:
{% for key, value in step.parameters.items() %}
        {{ key }}: "{{ value }}"
{% endfor %}
{% endfor %}
  
  rules:
{% for rule in rules %}
    - "{{ rule }}"
{% endfor %}
"""
        return cls(template_content, RuleFormat.WINDSURF)
