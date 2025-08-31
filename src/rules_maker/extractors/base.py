"""
Base content extractor class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

from ..models import ContentSection, ExtractionPattern


class ContentExtractor(ABC):
    """Base class for content extraction strategies."""
    
    def __init__(self, patterns: Optional[List[ExtractionPattern]] = None):
        """Initialize the extractor with extraction patterns."""
        self.patterns = patterns or []
    
    @abstractmethod
    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured content from HTML."""
        pass
    
    @abstractmethod
    def extract_sections(self, soup: BeautifulSoup, url: str) -> List[ContentSection]:
        """Extract content sections from HTML."""
        pass
    
    def add_pattern(self, pattern: ExtractionPattern) -> None:
        """Add an extraction pattern."""
        self.patterns.append(pattern)
    
    def remove_pattern(self, pattern_name: str) -> None:
        """Remove an extraction pattern by name."""
        self.patterns = [p for p in self.patterns if p.name != pattern_name]
