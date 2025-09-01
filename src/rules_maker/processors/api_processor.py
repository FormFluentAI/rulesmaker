"""
API documentation processor for Rules Maker.
"""

from typing import Dict, Any, List
from bs4 import BeautifulSoup
import re

from .documentation_processor import DocumentationProcessor
from ..models import DocumentationStructure, ContentSection, DocumentationType


class APIDocumentationProcessor(DocumentationProcessor):
    """Processor specialized for API documentation."""
    
    def process(self, content: str, url: str, metadata: Dict[str, Any]) -> DocumentationStructure:
        """Process API documentation content.
        
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
        
        # Force API documentation type
        doc_type = DocumentationType.API
        
        # Extract sections with API-specific handling
        sections = self._extract_api_sections(soup)
        
        # Extract API endpoints
        endpoints = self._extract_endpoints(soup)
        doc_metadata['endpoints'] = endpoints
        
        # Extract parameters
        parameters = self._extract_parameters(soup)
        doc_metadata['parameters'] = parameters
        
        # Extract response examples
        responses = self._extract_responses(soup)
        doc_metadata['responses'] = responses
        
        # Extract navigation links
        nav_links = self._extract_navigation(soup, url)
        
        # Extract code examples with API focus
        code_examples = self._extract_api_code_examples(soup)
        
        return DocumentationStructure(
            title=doc_metadata.get('title', ''),
            url=url,
            documentation_type=doc_type,
            sections=sections,
            metadata=doc_metadata,
            navigation_links=nav_links,
            code_examples=code_examples
        )
    
    def _extract_api_sections(self, soup: BeautifulSoup) -> List[ContentSection]:
        """Extract sections with API-specific handling."""
        sections = self.extract_sections(soup)
        
        # Enhance sections with API-specific metadata
        for section in sections:
            section.metadata['section_type'] = self._classify_api_section(section.title, section.content)
        
        return sections
    
    def _classify_api_section(self, title: str, content: str) -> str:
        """Classify API documentation section type."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        if any(word in title_lower for word in ['endpoint', 'route', 'api']):
            return 'endpoint'
        elif any(word in title_lower for word in ['parameter', 'param', 'argument']):
            return 'parameters'
        elif any(word in title_lower for word in ['response', 'return', 'output']):
            return 'response'
        elif any(word in title_lower for word in ['example', 'sample', 'demo']):
            return 'example'
        elif any(word in title_lower for word in ['error', 'exception', 'status']):
            return 'error_handling'
        elif any(word in title_lower for word in ['auth', 'authentication', 'token']):
            return 'authentication'
        else:
            return 'general'
    
    def _extract_endpoints(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract API endpoints from documentation."""
        endpoints = []
        
        # Look for HTTP method indicators
        method_patterns = {
            'GET': r'\bGET\b',
            'POST': r'\bPOST\b',
            'PUT': r'\bPUT\b',
            'DELETE': r'\bDELETE\b',
            'PATCH': r'\bPATCH\b'
        }
        
        # Find elements that might contain endpoint information
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'code', 'pre']):
            text = element.get_text()
            
            for method, pattern in method_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    # Try to extract the endpoint path
                    path_match = re.search(r'(/[^\s]*)', text)
                    if path_match:
                        path = path_match.group(1)
                        
                        endpoint = {
                            'method': method,
                            'path': path,
                            'description': self._get_endpoint_description(element),
                            'source_element': element.name
                        }
                        endpoints.append(endpoint)
                        break
        
        return endpoints
    
    def _get_endpoint_description(self, element) -> str:
        """Get description for an endpoint."""
        # Look for description in nearby elements
        description = ""
        
        # Check next sibling
        next_elem = element.find_next(['p', 'div'])
        if next_elem:
            description = next_elem.get_text(strip=True)[:200]
        
        return description
    
    def _extract_parameters(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract API parameters from documentation."""
        parameters = []
        
        # Look for parameter tables
        for table in soup.find_all('table'):
            headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
            
            # Check if this looks like a parameters table
            if any(word in ' '.join(headers) for word in ['parameter', 'name', 'type', 'description']):
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = [td.get_text(strip=True) for td in row.find_all('td')]
                    
                    if len(cells) >= 2:
                        param = {
                            'name': cells[0],
                            'type': cells[1] if len(cells) > 1 else 'string',
                            'description': cells[2] if len(cells) > 2 else '',
                            'required': 'required' in ' '.join(cells).lower()
                        }
                        parameters.append(param)
        
        return parameters
    
    def _extract_responses(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract API response examples from documentation."""
        responses = []
        
        # Look for response sections
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = element.get_text().lower()
            
            if any(word in text for word in ['response', 'return', 'output', 'example']):
                # Find associated code blocks
                next_elem = element.find_next(['pre', 'code'])
                if next_elem:
                    response_text = next_elem.get_text(strip=True)
                    
                    # Try to determine format
                    format_type = 'text'
                    if response_text.strip().startswith('{'):
                        format_type = 'json'
                    elif response_text.strip().startswith('<'):
                        format_type = 'xml'
                    
                    response = {
                        'format': format_type,
                        'content': response_text,
                        'description': element.get_text(strip=True)
                    }
                    responses.append(response)
        
        return responses
    
    def _extract_api_code_examples(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code examples with API focus."""
        code_examples = self._extract_code_examples(soup)
        
        # Enhance with API-specific metadata
        for example in code_examples:
            content = example['code'].lower()
            
            # Detect if this is a request example
            if any(word in content for word in ['curl', 'fetch', 'request', 'axios']):
                example['type'] = 'request'
            elif any(word in content for word in ['response', 'return', '200', '201']):
                example['type'] = 'response'
            else:
                example['type'] = 'general'
        
        return code_examples
