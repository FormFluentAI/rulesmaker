"""
Base transformer class for Rules Maker.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..models import ScrapingResult, RuleSet, Workflow, TransformationConfig


class BaseTransformer(ABC):
    """Base class for all content transformers."""
    
    def __init__(self, config: Optional[TransformationConfig] = None):
        """Initialize the transformer with configuration."""
        self.config = config or TransformationConfig()
    
    @abstractmethod
    def transform(self, results: List[ScrapingResult]) -> Any:
        """Transform scraping results into the target format."""
        pass
    
    @abstractmethod
    def generate_rules(self, results: List[ScrapingResult]) -> RuleSet:
        """Generate rules from scraping results."""
        pass
    
    def _filter_relevant_content(self, results: List[ScrapingResult]) -> List[ScrapingResult]:
        """Filter and prepare content for transformation."""
        filtered = []
        
        for result in results:
            if result.status.value == "completed" and result.content.strip():
                filtered.append(result)
        
        return filtered
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content."""
        # Simple keyword extraction - can be enhanced with NLP
        words = content.lower().split()
        
        # Common programming and documentation keywords
        keywords = [
            'function', 'method', 'class', 'variable', 'parameter',
            'return', 'example', 'usage', 'syntax', 'option',
            'configuration', 'setting', 'property', 'attribute',
            'api', 'endpoint', 'request', 'response', 'error'
        ]
        
        found_concepts = []
        for word in words:
            if word in keywords and word not in found_concepts:
                found_concepts.append(word)
        
        return found_concepts[:10]  # Limit to top 10
