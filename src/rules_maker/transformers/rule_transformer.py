"""
Base rule transformer class.
"""

from typing import List, Any
from .base import BaseTransformer
from ..models import ScrapingResult, RuleSet, Rule


class RuleTransformer(BaseTransformer):
    """Generic rule transformer."""
    
    def transform(self, results: List[ScrapingResult]) -> str:
        """Transform scraping results into rules format."""
        ruleset = self.generate_rules(results)
        
        if self.config.rule_format.value == "json":
            import json
            return json.dumps(ruleset.dict(), indent=2)
        elif self.config.rule_format.value == "yaml":
            import yaml
            return yaml.dump(ruleset.dict(), default_flow_style=False)
        else:
            # Default text format
            output = f"# {ruleset.name}\n\n{ruleset.description}\n\n"
            for rule in ruleset.rules:
                output += f"## {rule.title}\n{rule.description}\n\n"
            return output
    
    def generate_rules(self, results: List[ScrapingResult]) -> RuleSet:
        """Generate rules from scraping results."""
        filtered_results = self._filter_relevant_content(results)
        
        rules = []
        for i, result in enumerate(filtered_results):
            rule = Rule(
                id=f"rule_{i+1}",
                title=f"Rule from {result.url}",
                description=result.content[:500] + "..." if len(result.content) > 500 else result.content,
                category="documentation",
                tags=self._extract_key_concepts(result.content)
            )
            rules.append(rule)
        
        return RuleSet(
            name="Generated Rules",
            description="Rules generated from scraped documentation",
            rules=rules,
            format=self.config.rule_format
        )
