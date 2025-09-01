"""
Workflow template management for Rules Maker.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from jinja2 import Template

from ..models import RuleFormat


class WorkflowTemplate:
    """Template for generating workflows from documentation."""
    
    def __init__(self, template_content: str, format_type: RuleFormat = RuleFormat.WINDSURF):
        """Initialize the workflow template.
        
        Args:
            template_content: Jinja2 template content
            format_type: Target format
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
    def from_file(cls, template_path: Path, format_type: RuleFormat = RuleFormat.WINDSURF) -> 'WorkflowTemplate':
        """Create template from file.
        
        Args:
            template_path: Path to template file
            format_type: Target format
            
        Returns:
            WorkflowTemplate instance
        """
        content = template_path.read_text()
        return cls(content, format_type)
    
    @classmethod
    def get_default_windsurf_workflow(cls) -> 'WorkflowTemplate':
        """Get default Windsurf workflow template."""
        template_content = """workflow:
  name: "{{ workflow_name }}"
  description: "{{ description }}"
  version: "{{ version | default('1.0.0') }}"
  
  environment:
    variables:
{% for var_name, var_value in environment_variables.items() %}
      {{ var_name }}: "{{ var_value }}"
{% endfor %}
  
  steps:
{% for step in steps %}
    - step_id: "{{ step.id }}"
      name: "{{ step.name }}"
      type: "{{ step.type }}"
      action: "{{ step.action }}"
      {% if step.condition %}
      condition: "{{ step.condition }}"
      {% endif %}
      parameters:
{% for key, value in step.parameters.items() %}
        {{ key }}: {{ value | tojson }}
{% endfor %}
      {% if step.retry %}
      retry:
        attempts: {{ step.retry.attempts }}
        delay: {{ step.retry.delay }}
      {% endif %}
      {% if step.timeout %}
      timeout: {{ step.timeout }}
      {% endif %}
{% endfor %}
  
  rules:
{% for rule in rules %}
    - rule_id: "{{ rule.id }}"
      name: "{{ rule.name }}"
      condition: "{{ rule.condition }}"
      action: "{{ rule.action }}"
      {% if rule.parameters %}
      parameters:
{% for key, value in rule.parameters.items() %}
        {{ key }}: {{ value | tojson }}
{% endfor %}
      {% endif %}
{% endfor %}
  
  error_handling:
    on_failure: "{{ error_handling.on_failure | default('stop') }}"
    {% if error_handling.retry_policy %}
    retry_policy:
      max_attempts: {{ error_handling.retry_policy.max_attempts }}
      backoff_factor: {{ error_handling.retry_policy.backoff_factor }}
    {% endif %}
  
  metadata:
    created_by: "Rules Maker"
    created_at: "{{ metadata.created_at }}"
    source_url: "{{ metadata.source_url }}"
    documentation_type: "{{ metadata.documentation_type }}"
"""
        return cls(template_content, RuleFormat.WINDSURF)
    
    @classmethod
    def get_simple_workflow_template(cls) -> 'WorkflowTemplate':
        """Get simplified workflow template."""
        template_content = """# {{ workflow_name }}

{{ description }}

## Steps

{% for step in steps %}
### {{ loop.index }}. {{ step.name }}

{{ step.description }}

{% if step.code_example %}
```{{ step.language | default('bash') }}
{{ step.code_example }}
```
{% endif %}

{% endfor %}

## Rules

{% for rule in rules %}
- {{ rule }}
{% endfor %}
"""
        return cls(template_content, RuleFormat.MARKDOWN)
