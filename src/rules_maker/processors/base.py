"""
Base content processor for Rules Maker.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup

from ..models import ScrapingResult, DocumentationStructure, ContentSection


class ContentProcessor(ABC):
    """Base class for content processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the processor.
        
        Args:
            config: Processor configuration
        """
        self.config = config or {}
    
    @abstractmethod
    def process(self, content: str, url: str, metadata: Dict[str, Any]) -> DocumentationStructure:
        """Process content and return structured documentation.
        
        Args:
            content: Raw content to process
            url: Source URL
            metadata: Additional metadata
            
        Returns:
            Structured documentation
        """
        pass
    
    def extract_sections(self, soup: BeautifulSoup) -> List[ContentSection]:
        """Extract sections from HTML content.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of content sections
        """
        sections = []
        
        # Look for headings and their content
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(heading.name[1])
            title = heading.get_text(strip=True)
            
            # Get content until next heading of same or higher level
            content_elements = []
            for sibling in heading.next_siblings:
                if hasattr(sibling, 'name'):
                    if sibling.name and sibling.name.startswith('h'):
                        sibling_level = int(sibling.name[1])
                        if sibling_level <= level:
                            break
                    content_elements.append(sibling)
            
            # Extract text content
            content = ""
            code_examples = []
            
            for element in content_elements:
                if hasattr(element, 'get_text'):
                    if element.name in ['pre', 'code']:
                        code_examples.append({
                            'language': element.get('class', ['text'])[0] if element.get('class') else 'text',
                            'code': element.get_text(strip=True)
                        })
                    else:
                        content += element.get_text(strip=True) + " "
            
            if title:
                section = ContentSection(
                    title=title,
                    content=content.strip(),
                    level=level,
                    metadata={
                        'heading_tag': heading.name,
                        'code_examples': code_examples
                    }
                )
                sections.append(section)
        
        return sections
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML.
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            
        Returns:
            Extracted metadata
        """
        metadata = {'source_url': url}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Meta description
        desc_meta = soup.find('meta', attrs={'name': 'description'})
        if desc_meta:
            metadata['description'] = desc_meta.get('content', '')
        
        # Language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        # Keywords
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta:
            metadata['keywords'] = keywords_meta.get('content', '').split(',')
        
        return metadata
