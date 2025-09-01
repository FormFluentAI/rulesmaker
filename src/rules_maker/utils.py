"""
Utility functions for Rules Maker.
"""

import re
import logging
import hashlib
import mimetypes
from urllib.parse import urljoin, urlparse, parse_qs
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .models import DocumentationType


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """Set up logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
        ]
    )


def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """Normalize a URL by resolving relative paths."""
    if base_url:
        return urljoin(base_url, url)
    return url


def clean_content(content: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    # Remove special characters
    content = re.sub(r'[^\w\s\-_.,!?()[\]{}:;]', '', content)
    # Strip leading/trailing whitespace
    content = content.strip()
    return content


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.netloc


def generate_content_hash(content: str) -> str:
    """Generate a hash for content deduplication."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def detect_documentation_type(url: str, title: str = "", content: str = "") -> DocumentationType:
    """Detect the type of documentation based on URL, title, and content."""
    url_lower = url.lower()
    title_lower = title.lower()
    content_lower = content.lower()
    
    # API documentation patterns
    api_patterns = [
        r'/api[s]?/',
        r'api\..*\.com',
        r'developer\..*\.com',
        r'/docs/api',
        r'rest',
        r'graphql',
        r'openapi',
        r'swagger'
    ]
    
    if any(re.search(pattern, url_lower) for pattern in api_patterns):
        return DocumentationType.API
    
    if any(word in title_lower or word in content_lower for word in 
           ['api reference', 'api documentation', 'rest api', 'api guide']):
        return DocumentationType.API
    
    # Framework documentation
    framework_patterns = [
        r'/(react|vue|angular|django|flask|rails|laravel)',
        r'docs\..*framework',
        r'framework.*docs'
    ]
    
    if any(re.search(pattern, url_lower) for pattern in framework_patterns):
        return DocumentationType.FRAMEWORK
    
    # Library documentation
    if any(word in url_lower for word in ['library', 'lib', 'package']):
        return DocumentationType.LIBRARY
    
    # Tutorial detection
    if any(word in title_lower or word in url_lower for word in 
           ['tutorial', 'guide', 'getting-started', 'quickstart']):
        return DocumentationType.TUTORIAL
    
    # Reference documentation
    if any(word in title_lower or word in url_lower for word in 
           ['reference', 'ref', 'docs']):
        return DocumentationType.REFERENCE
    
    # README detection
    if 'readme' in url_lower or 'readme' in title_lower:
        return DocumentationType.README
    
    # Changelog detection
    if any(word in url_lower or word in title_lower for word in 
           ['changelog', 'changes', 'releases', 'history']):
        return DocumentationType.CHANGELOG
    
    return DocumentationType.UNKNOWN


def extract_navigation_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """Extract navigation links from a documentation page."""
    links = []
    
    # Common navigation selectors
    nav_selectors = [
        'nav a',
        '.nav a',
        '.navigation a',
        '.sidebar a',
        '.menu a',
        '.toc a',
        '.table-of-contents a'
    ]
    
    for selector in nav_selectors:
        nav_links = soup.select(selector)
        for link in nav_links:
            href = link.get('href')
            text = link.get_text(strip=True)
            
            if href and text:
                full_url = normalize_url(href, base_url)
                if validate_url(full_url):
                    links.append({
                        'text': text,
                        'url': full_url,
                        'type': 'navigation'
                    })
    
    return links


def extract_code_blocks(content: str) -> List[Dict[str, str]]:
    """Extract code blocks from content."""
    code_blocks = []
    
    # Markdown code blocks
    markdown_pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.finditer(markdown_pattern, content, re.DOTALL)
    
    for match in matches:
        language = match.group(1) or 'text'
        code = match.group(2)
        code_blocks.append({
            'language': language,
            'code': code,
            'type': 'markdown'
        })
    
    # HTML code blocks
    soup = BeautifulSoup(content, 'html.parser')
    for code_tag in soup.find_all(['code', 'pre']):
        language = code_tag.get('class', ['text'])[0] if code_tag.get('class') else 'text'
        code = code_tag.get_text()
        code_blocks.append({
            'language': language,
            'code': code,
            'type': 'html'
        })
    
    return code_blocks


def is_documentation_url(url: str) -> bool:
    """Check if a URL likely points to documentation."""
    doc_indicators = [
        'docs',
        'documentation',
        'api',
        'reference',
        'guide',
        'tutorial',
        'manual',
        'wiki',
        'help'
    ]
    
    url_lower = url.lower()
    return any(indicator in url_lower for indicator in doc_indicators)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe file system usage."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces and dots
    filename = re.sub(r'\.+', '.', filename)
    filename = re.sub(r'\s+', '_', filename)
    # Limit length
    return filename[:100]


def get_content_type(url: str) -> str:
    """Get the content type of a URL."""
    try:
        response = requests.head(url, timeout=5)
        return response.headers.get('content-type', 'text/html')
    except Exception:
        # Fallback to guessing based on URL
        content_type, _ = mimetypes.guess_type(url)
        return content_type or 'text/html'


def split_text_into_chunks(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_length, len(text))
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + max_length // 2:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = max(start + max_length - overlap, end)
    
    return chunks


def extract_metadata_from_html(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract metadata from HTML document."""
    metadata = {}
    
    # Title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text(strip=True)
    
    # Meta tags
    meta_tags = soup.find_all('meta')
    for meta in meta_tags:
        name = meta.get('name') or meta.get('property')
        content = meta.get('content')
        
        if name and content:
            metadata[name] = content
    
    # Language
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang'):
        metadata['language'] = html_tag.get('lang')
    
    # Canonical URL
    canonical = soup.find('link', rel='canonical')
    if canonical and canonical.get('href'):
        metadata['canonical_url'] = canonical.get('href')
    
    return metadata


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main content from a BeautifulSoup object."""
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Try to find main content areas
    main_selectors = [
        'main',
        '[role="main"]',
        '.main-content',
        '.content',
        '.article',
        '.post',
        '#content',
        '#main'
    ]
    
    for selector in main_selectors:
        main_element = soup.select_one(selector)
        if main_element:
            return main_element.get_text(strip=True, separator=' ')
    
    # Fallback: try to get body content excluding navigation, sidebars, etc.
    body = soup.find('body')
    if body:
        # Remove common non-content elements
        for element in body.select('nav, .nav, .navigation, .sidebar, .menu, header, footer'):
            element.decompose()
        
        return body.get_text(strip=True, separator=' ')
    
    # Final fallback: get all text
    return soup.get_text(strip=True, separator=' ')
