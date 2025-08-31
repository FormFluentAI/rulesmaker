"""
Workflow transformer.
"""

from typing import List
from .base import BaseTransformer
from ..models import ScrapingResult, Workflow, WorkflowStep, RuleSet


class WorkflowTransformer(BaseTransformer):
    """Transformer for creating workflows from documentation."""
    
    def transform(self, results: List[ScrapingResult]) -> str:
        """Transform scraping results into workflow format."""
        workflow = self.generate_workflow(results)
        return f"# Workflow: {workflow.name}\n\n{workflow.description}"
    
    def generate_rules(self, results: List[ScrapingResult]) -> RuleSet:
        """Generate rules (not applicable for workflow transformer)."""
        raise NotImplementedError("Use generate_workflow instead")
    
    def generate_workflow(self, results: List[ScrapingResult]) -> Workflow:
        """Generate workflow from scraping results."""
        steps = []
        for i, result in enumerate(results):
            step = WorkflowStep(
                id=f"step_{i+1}",
                name=f"Process {result.title}",
                description=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                action="process_documentation",
                parameters={"url": str(result.url)}
            )
            steps.append(step)
        
        return Workflow(
            name="Documentation Processing Workflow",
            description="Workflow for processing scraped documentation",
            steps=steps
        )
