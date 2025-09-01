"""
Duplicate filter for Rules Maker.
"""

from typing import Dict, Any, List, Set
import hashlib
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

from .base import ContentFilter


class DuplicateFilter(ContentFilter):
    """Filter duplicate content."""
    
    def __init__(self, similarity_threshold: float = 0.8, hash_threshold: float = 0.95,
                 use_content_hash: bool = True, use_text_similarity: bool = True,
                 config: Dict[str, Any] = None):
        """Initialize duplicate filter.
        
        Args:
            similarity_threshold: Threshold for text similarity (0.0-1.0)
            hash_threshold: Threshold for hash similarity (0.0-1.0)
            use_content_hash: Whether to use content hashing
            use_text_similarity: Whether to use text similarity comparison
            config: Additional configuration
        """
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self.hash_threshold = hash_threshold
        self.use_content_hash = use_content_hash
        self.use_text_similarity = use_text_similarity
        
        # Store seen content for duplicate detection
        self.seen_hashes: Set[str] = set()
        self.seen_content: List[str] = []
        self.content_metadata: List[Dict[str, Any]] = []
    
    def filter(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Filter duplicates."""
        if not content or not content.strip():
            return False
        
        # Clean content for comparison
        cleaned_content = self._clean_content(content)
        
        # Content hash check
        if self.use_content_hash:
            content_hash = self._get_content_hash(cleaned_content)
            if content_hash in self.seen_hashes:
                return False
        
        # Text similarity check
        if self.use_text_similarity:
            for seen_content in self.seen_content:
                similarity = self._calculate_similarity(cleaned_content, seen_content)
                if similarity >= self.similarity_threshold:
                    return False
        
        # Add to seen content
        if self.use_content_hash:
            self.seen_hashes.add(content_hash)
        
        if self.use_text_similarity:
            self.seen_content.append(cleaned_content)
            self.content_metadata.append(metadata)
            
            # Limit memory usage by keeping only recent content
            if len(self.seen_content) > 1000:
                self.seen_content = self.seen_content[-500:]
                self.content_metadata = self.content_metadata[-500:]
        
        return True
    
    def score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Score uniqueness of content."""
        if not content:
            return 0.0
        
        cleaned_content = self._clean_content(content)
        
        # Base uniqueness score
        score = 1.0
        
        # Check against seen content
        max_similarity = 0.0
        
        for seen_content in self.seen_content:
            similarity = self._calculate_similarity(cleaned_content, seen_content)
            max_similarity = max(max_similarity, similarity)
        
        # Reduce score based on highest similarity
        if max_similarity > 0.5:
            score *= (1.0 - max_similarity)
        
        # Bonus for completely unique content
        if max_similarity < 0.1:
            score = min(score + 0.1, 1.0)
        
        return score
    
    def reset(self) -> None:
        """Reset seen content tracking."""
        self.seen_hashes.clear()
        self.seen_content.clear()
        self.content_metadata.clear()
    
    def get_duplicate_groups(self) -> List[List[int]]:
        """Get groups of duplicate content indices."""
        if not self.use_text_similarity:
            return []
        
        groups = []
        processed = set()
        
        for i, content1 in enumerate(self.seen_content):
            if i in processed:
                continue
            
            group = [i]
            processed.add(i)
            
            for j, content2 in enumerate(self.seen_content[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(content1, content2)
                if similarity >= self.similarity_threshold:
                    group.append(j)
                    processed.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _clean_content(self, content: str) -> str:
        """Clean content for comparison."""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Normalize whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        return text
    
    def _get_content_hash(self, content: str) -> str:
        """Get hash of content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        if not content1 or not content2:
            return 0.0
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, content1, content2)
        return matcher.ratio()
    
    def _calculate_jaccard_similarity(self, content1: str, content2: str) -> float:
        """Calculate Jaccard similarity between two pieces of content."""
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def find_near_duplicates(self, threshold: float = None) -> List[List[int]]:
        """Find near-duplicate content groups.
        
        Args:
            threshold: Similarity threshold (uses instance threshold if None)
            
        Returns:
            List of groups containing indices of similar content
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        groups = []
        processed = set()
        
        for i, content1 in enumerate(self.seen_content):
            if i in processed:
                continue
            
            group = [i]
            for j, content2 in enumerate(self.seen_content[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(content1, content2)
                if similarity >= threshold:
                    group.append(j)
            
            if len(group) > 1:
                groups.append(group)
                processed.update(group)
        
        return groups
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get duplicate detection statistics."""
        total_content = len(self.seen_content)
        duplicate_groups = self.get_duplicate_groups()
        duplicate_count = sum(len(group) - 1 for group in duplicate_groups)  # Subtract 1 to count only duplicates
        
        return {
            'total_content_pieces': total_content,
            'duplicate_groups': len(duplicate_groups),
            'duplicate_count': duplicate_count,
            'unique_count': total_content - duplicate_count,
            'duplicate_ratio': duplicate_count / total_content if total_content > 0 else 0.0
        }
