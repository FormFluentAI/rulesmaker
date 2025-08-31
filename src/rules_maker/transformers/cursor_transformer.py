"""
Cursor rules transformer.
"""

from typing import List
from .rule_transformer import RuleTransformer
from ..models import ScrapingResult
from ..templates import TemplateEngine


class CursorRuleTransformer(RuleTransformer):
    """Transformer for Cursor rules format."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_engine = TemplateEngine()
    
    def transform(self, results: List[ScrapingResult]) -> str:
        """Transform scraping results into Cursor rules format."""
        ruleset = self.generate_rules(results)
        return self.template_engine.render_cursor_rules(ruleset)
