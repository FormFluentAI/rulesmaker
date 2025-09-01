"""
Quality filter for Rules Maker.
"""

from typing import Dict, Any, List
import re
from bs4 import BeautifulSoup

from .base import ContentFilter


class QualityFilter(ContentFilter):
    """Filter content based on quality metrics."""
    
    def __init__(self, min_word_count: int = 50, min_sentence_count: int = 3,
                 max_duplicate_ratio: float = 0.5, require_structure: bool = True,
                 config: Dict[str, Any] = None):
        """Initialize quality filter.
        
        Args:
            min_word_count: Minimum number of words
            min_sentence_count: Minimum number of sentences
            max_duplicate_ratio: Maximum ratio of duplicate content
            require_structure: Whether to require structured content
            config: Additional configuration
        """
        super().__init__(config)
        self.min_word_count = min_word_count
        self.min_sentence_count = min_sentence_count
        self.max_duplicate_ratio = max_duplicate_ratio
        self.require_structure = require_structure
    
    def filter(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Filter based on quality metrics."""
        if not content or not content.strip():
            return False
        
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(strip=True)
        
        # Word count check
        words = text.split()
        if len(words) < self.min_word_count:
            return False
        
        # Sentence count check
        sentences = self._split_sentences(text)
        if len(sentences) < self.min_sentence_count:
            return False
        
        # Structure check
        if self.require_structure and not self._has_structure(soup):
            return False
        
        # Duplicate content check
        if self._has_excessive_duplicates(text):
            return False
        
        # Spam/low quality indicators
        if self._is_spam_like(text, soup):
            return False
        
        return True
    
    def score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Score content quality."""
        if not content:
            return 0.0
        
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(strip=True)
        score = 0.0
        
        # Word count score
        words = text.split()
        word_count = len(words)
        if word_count >= 200:
            score += 0.3
        elif word_count >= 100:
            score += 0.2
        elif word_count >= 50:
            score += 0.1
        
        # Sentence structure score
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)
        if sentence_count >= 10:
            score += 0.2
        elif sentence_count >= 5:
            score += 0.1
        
        # Average sentence length (indicates quality)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 8 <= avg_sentence_length <= 25:  # Good range
                score += 0.1
        
        # Structure score
        if self._has_structure(soup):
            score += 0.2
            
            # Bonus for good structure
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if len(headings) >= 2:
                score += 0.1
        
        # Code examples score (good for documentation)
        code_blocks = soup.find_all(['pre', 'code'])
        if code_blocks:
            score += min(len(code_blocks) * 0.02, 0.1)
        
        # Links score (indicates interconnected content)
        links = soup.find_all('a', href=True)
        if links:
            score += min(len(links) * 0.005, 0.05)
        
        # Penalty for poor quality indicators
        penalty = 0
        
        # Excessive repetition penalty
        if self._has_excessive_duplicates(text):
            penalty += 0.3
        
        # Spam-like content penalty
        if self._is_spam_like(text, soup):
            penalty += 0.4
        
        # Very short sentences penalty
        short_sentences = sum(1 for s in sentences if len(s.split()) < 3)
        if short_sentences > len(sentences) * 0.3:  # More than 30% short sentences
            penalty += 0.1
        
        score = max(0, score - penalty)
        return min(score, 1.0)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _has_structure(self, soup: BeautifulSoup) -> bool:
        """Check if content has good structure."""
        # Check for headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings:
            return True
        
        # Check for lists
        lists = soup.find_all(['ul', 'ol'])
        if lists:
            return True
        
        # Check for paragraphs
        paragraphs = soup.find_all('p')
        if len(paragraphs) >= 2:
            return True
        
        return False
    
    def _has_excessive_duplicates(self, text: str) -> bool:
        """Check for excessive duplicate content."""
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return False
        
        # Check for repeated sentences
        sentence_counts = {}
        for sentence in sentences:
            sentence_clean = re.sub(r'\s+', ' ', sentence.lower().strip())
            if len(sentence_clean) > 10:  # Only count meaningful sentences
                sentence_counts[sentence_clean] = sentence_counts.get(sentence_clean, 0) + 1
        
        # Calculate duplicate ratio
        total_sentences = len([s for s in sentences if len(s.strip()) > 10])
        duplicate_sentences = sum(count - 1 for count in sentence_counts.values() if count > 1)
        
        if total_sentences == 0:
            return False
        
        duplicate_ratio = duplicate_sentences / total_sentences
        return duplicate_ratio > self.max_duplicate_ratio
    
    def _is_spam_like(self, text: str, soup: BeautifulSoup) -> bool:
        """Check for spam-like characteristics."""
        text_lower = text.lower()
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:  # More than 30% caps
            return True
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in '!?.,;:') / len(text) if text else 0
        if punct_ratio > 0.1:  # More than 10% punctuation
            return True
        
        # Check for spam keywords
        spam_keywords = [
            'click here', 'buy now', 'limited time', 'act now',
            'free money', 'earn money', 'make money fast',
            'weight loss', 'lose weight', 'diet pills'
        ]
        
        spam_count = sum(1 for keyword in spam_keywords if keyword in text_lower)
        if spam_count > 2:
            return True
        
        # Check for excessive links
        links = soup.find_all('a', href=True)
        words = text.split()
        if len(words) > 0:
            link_ratio = len(links) / len(words)
            if link_ratio > 0.1:  # More than 1 link per 10 words
                return True
        
        return False
