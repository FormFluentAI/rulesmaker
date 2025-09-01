"""
Code documentation processor for Rules Maker.
"""

from typing import Dict, Any, List
from bs4 import BeautifulSoup
import re

from .documentation_processor import DocumentationProcessor
from ..models import DocumentationStructure, ContentSection, DocumentationType


class CodeDocumentationProcessor(DocumentationProcessor):
    """Processor specialized for code documentation (libraries, frameworks)."""
    
    def process(self, content: str, url: str, metadata: Dict[str, Any]) -> DocumentationStructure:
        """Process code documentation content.
        
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
        
        # Detect specific documentation type
        title = doc_metadata.get('title', '')
        if 'framework' in url.lower() or 'framework' in title.lower():
            doc_type = DocumentationType.FRAMEWORK
        else:
            doc_type = DocumentationType.LIBRARY
        
        # Extract sections with code-specific handling
        sections = self._extract_code_sections(soup)
        
        # Extract classes and functions
        classes = self._extract_classes(soup)
        doc_metadata['classes'] = classes
        
        functions = self._extract_functions(soup)
        doc_metadata['functions'] = functions
        
        # Extract installation instructions
        installation = self._extract_installation(soup)
        if installation:
            doc_metadata['installation'] = installation
        
        # Extract usage examples
        usage_examples = self._extract_usage_examples(soup)
        doc_metadata['usage_examples'] = usage_examples
        
        # Extract navigation links
        nav_links = self._extract_navigation(soup, url)
        
        # Extract code examples with code focus
        code_examples = self._extract_code_examples(soup)
        
        return DocumentationStructure(
            title=title,
            url=url,
            documentation_type=doc_type,
            sections=sections,
            metadata=doc_metadata,
            navigation_links=nav_links,
            code_examples=code_examples
        )
    
    def _extract_code_sections(self, soup: BeautifulSoup) -> List[ContentSection]:
        """Extract sections with code-specific handling."""
        sections = self.extract_sections(soup)
        
        # Enhance sections with code-specific metadata
        for section in sections:
            section.metadata['section_type'] = self._classify_code_section(section.title, section.content)
        
        return sections
    
    def _classify_code_section(self, title: str, content: str) -> str:
        """Classify code documentation section type."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        if any(word in title_lower for word in ['class', 'classes']):
            return 'class_reference'
        elif any(word in title_lower for word in ['function', 'method', 'api']):
            return 'function_reference'
        elif any(word in title_lower for word in ['install', 'setup', 'getting started']):
            return 'installation'
        elif any(word in title_lower for word in ['usage', 'example', 'tutorial']):
            return 'usage'
        elif any(word in title_lower for word in ['config', 'configuration', 'settings']):
            return 'configuration'
        elif any(word in title_lower for word in ['changelog', 'changes', 'release']):
            return 'changelog'
        else:
            return 'general'
    
    def _extract_classes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract class definitions from documentation."""
        classes = []
        
        # Look for class headers or signatures
        class_patterns = [
            r'class\s+([A-Z][a-zA-Z0-9_]*)',
            r'([A-Z][a-zA-Z0-9_]*)\s*\(',
        ]
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'code', 'pre']):
            text = element.get_text()
            
            for pattern in class_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    class_name = match.group(1)
                    
                    class_info = {
                        'name': class_name,
                        'description': self._get_element_description(element),
                        'methods': self._extract_class_methods(element),
                        'source_element': element.name
                    }
                    classes.append(class_info)
        
        return classes
    
    def _extract_functions(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract function definitions from documentation."""
        functions = []
        
        # Look for function signatures
        function_patterns = [
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*->',
        ]
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'code', 'pre']):
            text = element.get_text()
            
            for pattern in function_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    function_name = match.group(1)
                    
                    function_info = {
                        'name': function_name,
                        'signature': match.group(0),
                        'description': self._get_element_description(element),
                        'parameters': self._extract_function_parameters(element),
                        'source_element': element.name
                    }
                    functions.append(function_info)
        
        return functions
    
    def _extract_class_methods(self, class_element) -> List[str]:
        """Extract methods for a class."""
        methods = []
        
        # Look for method definitions in nearby content
        current = class_element
        for _ in range(10):  # Look ahead a few elements
            current = current.find_next()
            if not current:
                break
            
            text = current.get_text() if hasattr(current, 'get_text') else str(current)
            method_matches = re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', text)
            
            for match in method_matches:
                methods.append(match.group(1))
        
        return methods
    
    def _extract_function_parameters(self, function_element) -> List[Dict[str, str]]:
        """Extract parameters for a function."""
        parameters = []
        
        # Look for parameter documentation in nearby elements
        next_elem = function_element.find_next(['p', 'div', 'ul', 'table'])
        if next_elem:
            text = next_elem.get_text()
            
            # Simple parameter extraction (can be enhanced)
            param_matches = re.finditer(r'(\w+)\s*:\s*([^,\n]*)', text)
            for match in param_matches:
                parameters.append({
                    'name': match.group(1),
                    'type': match.group(2).strip()
                })
        
        return parameters
    
    def _get_element_description(self, element) -> str:
        """Get description for an element."""
        description = ""
        
        # Look for description in nearby elements
        next_elem = element.find_next(['p', 'div'])
        if next_elem:
            description = next_elem.get_text(strip=True)[:300]
        
        return description
    
    def _extract_installation(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract installation instructions."""
        installation = {}
        
        # Look for installation sections
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = element.get_text().lower()
            
            if any(word in text for word in ['install', 'setup', 'getting started']):
                # Find associated code blocks
                code_elem = element.find_next(['pre', 'code'])
                if code_elem:
                    install_command = code_elem.get_text(strip=True)
                    
                    # Detect package manager
                    if 'pip install' in install_command:
                        installation['pip'] = install_command
                    elif 'npm install' in install_command:
                        installation['npm'] = install_command
                    elif 'yarn add' in install_command:
                        installation['yarn'] = install_command
                    else:
                        installation['generic'] = install_command
        
        return installation
    
    def _extract_usage_examples(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract usage examples."""
        examples = []
        
        # Look for example sections
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = element.get_text().lower()
            
            if any(word in text for word in ['usage', 'example', 'quickstart', 'basic']):
                # Find associated code blocks
                current = element
                for _ in range(5):  # Look ahead a few elements
                    current = current.find_next(['pre', 'code'])
                    if not current:
                        break
                    
                    code_text = current.get_text(strip=True)
                    if code_text:
                        example = {
                            'title': element.get_text(strip=True),
                            'code': code_text,
                            'language': self._detect_code_language(code_text),
                            'description': self._get_element_description(element)
                        }
                        examples.append(example)
                        break
        
        return examples
    
    def _detect_code_language(self, code: str) -> str:
        """Detect programming language from code content."""
        code_lower = code.lower()
        
        if any(keyword in code_lower for keyword in ['import ', 'def ', 'class ', 'print(']):
            return 'python'
        elif any(keyword in code_lower for keyword in ['function', 'var ', 'const ', 'let ']):
            return 'javascript'
        elif any(keyword in code_lower for keyword in ['public class', 'private ', 'static ']):
            return 'java'
        elif any(keyword in code_lower for keyword in ['#include', 'int main', 'printf']):
            return 'c'
        elif any(keyword in code_lower for keyword in ['<?php', 'echo ', '$']):
            return 'php'
        else:
            return 'text'
