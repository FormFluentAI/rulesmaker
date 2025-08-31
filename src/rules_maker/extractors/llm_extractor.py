"""
Placeholder for LLM content extractor.
"""

from typing import Dict, List, Any
from bs4 import BeautifulSoup
from .base import ContentExtractor
from ..models import ContentSection


class LLMContentExtractor(ContentExtractor):
    """LLM-powered content extraction."""
    
    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured content (placeholder)."""
        raise NotImplementedError("LLM extractor not yet implemented")
    
    def extract_sections(self, soup: BeautifulSoup, url: str) -> List[ContentSection]:
        """Extract content sections (placeholder)."""
        raise NotImplementedError("LLM extractor not yet implemented")
