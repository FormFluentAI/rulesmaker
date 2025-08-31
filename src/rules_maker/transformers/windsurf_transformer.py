"""
Windsurf rules transformer.
"""

from typing import List
from .rule_transformer import RuleTransformer
from ..models import ScrapingResult


class WindsurfRuleTransformer(RuleTransformer):
    """Transformer for Windsurf rules format."""
    
    def transform(self, results: List[ScrapingResult]) -> str:
        """Transform scraping results into Windsurf rules format."""
        ruleset = self.generate_rules(results)
        # Placeholder for Windsurf-specific formatting
        return f"# Windsurf Rules: {ruleset.name}\n\n{ruleset.description}"
