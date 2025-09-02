"""
Documentation processor for Rules Maker.
"""

from typing import Dict, Any, List
from bs4 import BeautifulSoup
from datetime import datetime

from .base import ContentProcessor
from ..models import DocumentationStructure
from ..utils import detect_documentation_type


class DocumentationProcessor(ContentProcessor):
    """Processor for general documentation content."""
    
    def process(self, content: str, url: str, metadata: Dict[str, Any]) -> DocumentationStructure:
        """Process documentation content.
        
        Args:
            content: HTML content
            url: Source URL
            metadata: Additional metadata
            
        Returns:
            Structured documentation
        """
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract basic metadata
        doc_metadata = self.extract_metadata(soup, url)
        doc_metadata.update(metadata)
        
        # Detect documentation type
        title = doc_metadata.get('title', '')
        doc_type = detect_documentation_type(url, title, content)
        
        # Extract sections
        sections = self.extract_sections(soup)
        
        # Extract navigation links
        nav_links = self._extract_navigation(soup, url)
        
        return DocumentationStructure(
            name=title,
            base_url=url,
            documentation_type=doc_type,
            sections=sections,
            metadata=doc_metadata,
            navigation={'links': nav_links},
            last_updated=datetime.now()
        )
    
    def _extract_navigation(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract navigation links."""
        nav_links = []
        
        # Common navigation selectors
        nav_selectors = [
            'nav a',
            '.navigation a',
            '.sidebar a',
            '.menu a',
            '.toc a',
            '.table-of-contents a'
        ]
        
        for selector in nav_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                
                if href and text:
                    nav_links.append({
                        'text': text,
                        'url': href,
                        'type': 'navigation'
                    })
        
        return nav_links
    
    def _extract_code_examples(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code examples from content."""
        code_examples = []
        
        # Find code blocks
        for code_element in soup.find_all(['pre', 'code']):
            # Skip inline code
            if code_element.name == 'code' and code_element.parent.name != 'pre':
                continue
            
            code_text = code_element.get_text(strip=True)
            if not code_text:
                continue
            
            # Try to determine language
            language = 'text'
            
            # Check for class indicating language
            classes = code_element.get('class', [])
            for cls in classes:
                if cls.startswith('language-'):
                    language = cls.replace('language-', '')
                    break
                elif cls in ['python', 'javascript', 'bash', 'html', 'css', 'json']:
                    language = cls
                    break
            
            # Check parent for language info
            if language == 'text' and code_element.parent:
                parent_classes = code_element.parent.get('class', [])
                for cls in parent_classes:
                    if cls.startswith('language-'):
                        language = cls.replace('language-', '')
                        break
                    elif cls in ['python', 'javascript', 'bash', 'html', 'css', 'json']:
                        language = cls
                        break
            
            code_examples.append({
                'language': language,
                'code': code_text,
                'context': self._get_code_context(code_element)
            })
        
        return code_examples
    
    def _get_code_context(self, code_element) -> str:
        """Get context around a code example."""
        context = ""
        
        # Look for preceding heading or paragraph
        prev_element = code_element.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
        if prev_element:
            context = prev_element.get_text(strip=True)[:200]
        
        return context
