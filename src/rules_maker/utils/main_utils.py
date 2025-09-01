"""
Core utility functions re-implemented for utils module.

These functions are copies/adaptations of the main utils.py functions
to avoid import issues with relative imports.
"""

import re
import logging
import hashlib
from urllib.parse import urlparse
from typing import Optional, Dict, Any


def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """Normalize URL by removing fragments and query parameters."""
    from urllib.parse import urljoin, urlparse, urlunparse
    
    if base_url:
        url = urljoin(base_url, url)
    
    parsed = urlparse(url)
    # Remove fragment and some query params that don't matter
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc, 
        parsed.path,
        parsed.params,
        '',  # Remove query for now
        ''   # Remove fragment
    ))
    
    return normalized


def clean_content(content: str) -> str:
    """Clean and normalize text content."""
    if not content:
        return ""
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    # Remove some common unwanted patterns
    content = re.sub(r'^\s*[•·*-]\s*', '', content, flags=re.MULTILINE)
    
    return content


def extract_metadata_from_html(soup) -> Dict[str, Any]:
    """Extract metadata from HTML soup object."""
    metadata = {}
    
    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text().strip()
    
    # Extract meta description
    desc_tag = soup.find('meta', attrs={'name': 'description'})
    if desc_tag and desc_tag.get('content'):
        metadata['description'] = desc_tag.get('content').strip()
    
    # Extract language
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang'):
        metadata['language'] = html_tag.get('lang')
    
    return metadata


def detect_documentation_type(content: str, url: str = "") -> str:
    """Detect the type of documentation based on content and URL."""
    content_lower = content.lower()
    url_lower = url.lower()
    
    # API documentation indicators
    if any(term in content_lower for term in ['api', 'endpoint', 'rest', 'graphql', 'swagger', 'openapi']):
        return "api"
    
    # Tutorial indicators
    if any(term in content_lower for term in ['tutorial', 'getting started', 'quickstart', 'guide']):
        return "tutorial"
    
    # Installation indicators  
    if any(term in content_lower for term in ['install', 'setup', 'pip install', 'npm install']):
        return "installation"
    
    # Reference documentation
    if any(term in content_lower for term in ['reference', 'docs', 'documentation']):
        return "reference"
    
    # Framework documentation
    if any(term in content_lower for term in ['framework', 'library']):
        return "framework"
    
    return "unknown"


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


def extract_navigation_links(soup, base_url: str) -> list:
    """Extract navigation links from a documentation page."""
    from urllib.parse import urljoin
    
    links = []
    
    # Look for common navigation patterns
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
            text = link.get_text().strip()
            
            if href and text:
                full_url = urljoin(base_url, href)
                links.append({
                    'url': full_url,
                    'text': text,
                    'type': 'navigation'
                })
    
    return links


def is_documentation_url(url: str) -> bool:
    """Check if a URL likely points to documentation."""
    doc_indicators = [
        'docs',
        'documentation',
        'guide',
        'manual',
        'reference',
        'wiki',
        'help',
        'tutorial',
        'api',
        'readme'
    ]
    
    url_lower = url.lower()
    return any(indicator in url_lower for indicator in doc_indicators)


def extract_main_content(soup) -> str:
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
            return main_element.get_text(separator=' ', strip=True)
    
    # Fallback: get body text
    body = soup.find('body')
    if body:
        return body.get_text(separator=' ', strip=True)
    
    # Last resort: get all text
    return soup.get_text(separator=' ', strip=True)


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""


def generate_content_hash(content: str) -> str:
    """Generate hash for content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def extract_code_blocks(content: str) -> list:
    """Extract code blocks from content."""
    import re
    
    # Look for common code block patterns
    patterns = [
        r'```[\s\S]*?```',  # Markdown code blocks
        r'`[^`\n]+`',       # Inline code
        r'<code>[\s\S]*?</code>',  # HTML code tags
        r'<pre>[\s\S]*?</pre>',    # HTML pre tags
    ]
    
    code_blocks = []
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # Clean up the match
            clean_match = match.strip('`').strip('<code>').strip('</code>')
            clean_match = clean_match.strip('<pre>').strip('</pre>')
            if clean_match.strip():
                code_blocks.append({
                    'content': clean_match.strip(),
                    'language': 'unknown'
                })
    
    return code_blocks


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem."""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')
    return filename or 'unnamed'


def get_content_type(url: str) -> str:
    """Get content type from URL."""
    try:
        import requests
        response = requests.head(url, timeout=5)
        return response.headers.get('content-type', 'unknown')
    except:
        return 'unknown'


def split_text_into_chunks(text: str, max_length: int = 1000, overlap: int = 100) -> list:
    """Split text into overlapping chunks."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at word boundary
        if end < len(text):
            # Find last space before max_length
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position considering overlap
        start = max(start + max_length - overlap, end)
        
        if start >= len(text):
            break
    
    return chunks


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity."""
    if not text1 or not text2:
        return 0.0
    
    # Simple character-based similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0