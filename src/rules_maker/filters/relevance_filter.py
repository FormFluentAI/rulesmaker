"""
Relevance filter for Rules Maker.
"""

from typing import Dict, Any, List, Set
import re
from bs4 import BeautifulSoup

from .base import ContentFilter


class RelevanceFilter(ContentFilter):
    """Filter content based on relevance to documentation."""
    
    def __init__(self, keywords: List[str] = None, doc_indicators: List[str] = None,
                 exclude_patterns: List[str] = None, config: Dict[str, Any] = None):
        """Initialize relevance filter.
        
        Args:
            keywords: Keywords that indicate relevant content
            doc_indicators: Indicators of documentation content
            exclude_patterns: Patterns to exclude
            config: Additional configuration
        """
        super().__init__(config)
        
        self.keywords = set(keywords or [])
        self.doc_indicators = set(doc_indicators or [
            'documentation', 'docs', 'guide', 'tutorial', 'reference',
            'api', 'usage', 'example', 'install', 'setup', 'configuration',
            'getting started', 'quickstart', 'readme'
        ])
        self.exclude_patterns = exclude_patterns or [
            r'\b(login|register|sign\s*up|sign\s*in)\b',
            r'\b(404|not\s*found|error|broken)\b',
            r'\b(coming\s*soon|under\s*construction)\b',
            r'\b(blog|news|press|contact)\b'
        ]
        
        self.exclude_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.exclude_patterns]
    
    def filter(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Filter based on relevance."""
        if not content or not content.strip():
            return False
        
        # Check for exclusion patterns
        if self._has_exclusion_patterns(content, metadata):
            return False
        
        # Check for documentation indicators
        if not self._has_doc_indicators(content, metadata):
            return False
        
        # Check for keywords if specified
        if self.keywords and not self._has_keywords(content, metadata):
            return False
        
        return True
    
    def score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Score relevance of content."""
        if not content:
            return 0.0
        
        score = 0.0
        content_lower = content.lower()
        
        # Check URL for documentation indicators
        url = metadata.get('url', '')
        if url:
            url_lower = url.lower()
            for indicator in self.doc_indicators:
                if indicator in url_lower:
                    score += 0.3
                    break
        
        # Check title for documentation indicators
        title = metadata.get('title', '')
        if title:
            title_lower = title.lower()
            for indicator in self.doc_indicators:
                if indicator in title_lower:
                    score += 0.2
                    break
        
        # Check content for documentation indicators
        doc_indicator_count = sum(1 for indicator in self.doc_indicators 
                                 if indicator in content_lower)
        score += min(doc_indicator_count * 0.1, 0.3)
        
        # Check for keywords
        if self.keywords:
            keyword_count = sum(1 for keyword in self.keywords 
                              if keyword.lower() in content_lower)
            score += min(keyword_count * 0.05, 0.2)
        
        # Check for code examples (good indicator of documentation)
        soup = BeautifulSoup(content, 'html.parser')
        code_blocks = soup.find_all(['pre', 'code'])
        if code_blocks:
            score += min(len(code_blocks) * 0.02, 0.1)
        
        # Check for structured content (headings, lists)
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings:
            score += min(len(headings) * 0.01, 0.1)
        
        lists = soup.find_all(['ul', 'ol'])
        if lists:
            score += min(len(lists) * 0.01, 0.05)
        
        # Penalty for exclusion indicators
        exclusion_score = 0
        for pattern in self.exclude_regex:
            if pattern.search(content):
                exclusion_score += 0.1
        
        score = max(0, score - exclusion_score)
        
        return min(score, 1.0)
    
    def _has_exclusion_patterns(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if content has exclusion patterns."""
        # Check content
        for pattern in self.exclude_regex:
            if pattern.search(content):
                return True
        
        # Check URL
        url = metadata.get('url', '')
        if url:
            for pattern in self.exclude_regex:
                if pattern.search(url):
                    return True
        
        # Check title
        title = metadata.get('title', '')
        if title:
            for pattern in self.exclude_regex:
                if pattern.search(title):
                    return True
        
        return False
    
    def _has_doc_indicators(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if content has documentation indicators."""
        content_lower = content.lower()
        
        # Check content
        for indicator in self.doc_indicators:
            if indicator in content_lower:
                return True
        
        # Check URL
        url = metadata.get('url', '')
        if url:
            url_lower = url.lower()
            for indicator in self.doc_indicators:
                if indicator in url_lower:
                    return True
        
        # Check title
        title = metadata.get('title', '')
        if title:
            title_lower = title.lower()
            for indicator in self.doc_indicators:
                if indicator in title_lower:
                    return True
        
        return False
    
    def _has_keywords(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if content has specified keywords."""
        if not self.keywords:
            return True
        
        content_lower = content.lower()
        
        # Check content
        for keyword in self.keywords:
            if keyword.lower() in content_lower:
                return True
        
        # Check URL
        url = metadata.get('url', '')
        if url:
            url_lower = url.lower()
            for keyword in self.keywords:
                if keyword.lower() in url_lower:
                    return True
        
        # Check title
        title = metadata.get('title', '')
        if title:
            title_lower = title.lower()
            for keyword in self.keywords:
                if keyword.lower() in title_lower:
                    return True
        
        return False
