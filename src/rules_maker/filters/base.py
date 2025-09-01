"""
Base content filter for Rules Maker.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup

from ..models import ScrapingResult, DocumentationStructure


class ContentFilter(ABC):
    """Base class for content filters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config or {}
    
    @abstractmethod
    def filter(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Filter content based on criteria.
        
        Args:
            content: Content to filter
            metadata: Content metadata
            
        Returns:
            True if content should be kept, False otherwise
        """
        pass
    
    @abstractmethod
    def score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Score content quality/relevance.
        
        Args:
            content: Content to score
            metadata: Content metadata
            
        Returns:
            Score from 0.0 to 1.0
        """
        pass
    
    def filter_results(self, results: List[ScrapingResult]) -> List[ScrapingResult]:
        """Filter a list of scraping results.
        
        Args:
            results: List of scraping results
            
        Returns:
            Filtered list of results
        """
        filtered = []
        
        for result in results:
            if self.filter(result.content, result.metadata):
                filtered.append(result)
        
        return filtered
    
    def filter_documentation(self, docs: List[DocumentationStructure]) -> List[DocumentationStructure]:
        """Filter a list of documentation structures.
        
        Args:
            docs: List of documentation structures
            
        Returns:
            Filtered list of documentation
        """
        filtered = []
        
        for doc in docs:
            # Convert doc to content for filtering
            content = f"{doc.title}\n"
            for section in doc.sections:
                content += f"{section.title}\n{section.content}\n"
            
            if self.filter(content, doc.metadata):
                filtered.append(doc)
        
        return filtered


class BaseContentFilter(ContentFilter):
    """Basic content filter implementation."""
    
    def __init__(self, min_length: int = 100, max_length: int = 50000, 
                 require_text: bool = True, config: Optional[Dict[str, Any]] = None):
        """Initialize basic filter.
        
        Args:
            min_length: Minimum content length
            max_length: Maximum content length
            require_text: Whether to require text content
            config: Additional configuration
        """
        super().__init__(config)
        self.min_length = min_length
        self.max_length = max_length
        self.require_text = require_text
    
    def filter(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Basic content filtering."""
        if not content or not content.strip():
            return False
        
        # Length checks
        content_length = len(content.strip())
        if content_length < self.min_length or content_length > self.max_length:
            return False
        
        # Text content check
        if self.require_text:
            # Remove HTML tags and check for meaningful text
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(strip=True)
            
            if len(text) < self.min_length:
                return False
            
            # Check for minimum word count
            words = text.split()
            if len(words) < 10:  # Minimum 10 words
                return False
        
        return True
    
    def score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Basic content scoring."""
        if not self.filter(content, metadata):
            return 0.0
        
        score = 0.5  # Base score
        
        # Length score (optimal around 1000-5000 characters)
        content_length = len(content.strip())
        if 1000 <= content_length <= 5000:
            score += 0.2
        elif 500 <= content_length <= 10000:
            score += 0.1
        
        # Text quality score
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(strip=True)
        words = text.split()
        
        # Word count score
        word_count = len(words)
        if 50 <= word_count <= 1000:
            score += 0.2
        elif 20 <= word_count <= 2000:
            score += 0.1
        
        # Sentence structure score (simple heuristic)
        sentences = text.split('.')
        if len(sentences) >= 3:  # At least 3 sentences
            score += 0.1
        
        return min(score, 1.0)
