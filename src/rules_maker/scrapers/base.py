"""
Base scraper class for Rules Maker.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from ..models import ScrapingResult, ScrapingConfig, ScrapingStatus, DocumentationType
from ..utils import validate_url, normalize_url, extract_metadata_from_html, detect_documentation_type


logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all scrapers."""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        """Initialize the scraper with configuration."""
        self.config = config or ScrapingConfig()
        self.session = requests.Session()
        self._setup_session()
        
    def _setup_session(self) -> None:
        """Configure the requests session."""
        # Set headers
        headers = {
            'User-Agent': self.config.user_agent,
            **self.config.headers
        }
        self.session.headers.update(headers)
        
        # Set cookies
        if self.config.cookies:
            self.session.cookies.update(self.config.cookies)
            
        # Set proxy
        if self.config.proxy:
            self.session.proxies = {
                'http': self.config.proxy,
                'https': self.config.proxy
            }
    
    @abstractmethod
    def scrape_url(self, url: str, **kwargs) -> ScrapingResult:
        """Scrape a single URL and return the result."""
        pass
    
    @abstractmethod
    def scrape_multiple(self, urls: List[str], **kwargs) -> List[ScrapingResult]:
        """Scrape multiple URLs and return results."""
        pass
    
    def _fetch_page(self, url: str) -> requests.Response:
        """Fetch a web page using the configured session."""
        if not validate_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        try:
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            return response
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def _parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content using BeautifulSoup."""
        return BeautifulSoup(html_content, 'lxml')
    
    def _extract_basic_info(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract basic information from a parsed HTML page."""
        metadata = extract_metadata_from_html(soup)
        
        # Extract title
        title = ""
        if 'title' in metadata:
            title = metadata['title']
        else:
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ""
        
        # Extract main content
        content = self._extract_main_content(soup)
        
        # Detect documentation type
        doc_type = detect_documentation_type(url, title, content)
        
        return {
            'title': title,
            'content': content,
            'documentation_type': doc_type,
            'metadata': metadata
        }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML, excluding navigation and footer."""
        # Remove unwanted elements
        for element in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            element.decompose()
        
        # Try common content selectors
        content_selectors = [
            'main',
            '.content',
            '.main-content', 
            '#content',
            '#main',
            'article',
            '.article',
            '.documentation',
            '.docs'
        ]
        
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                return content_element.get_text(strip=True, separator=' ')
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text(strip=True, separator=' ')
        
        return soup.get_text(strip=True, separator=' ')
    
    def _create_scraping_result(
        self, 
        url: str, 
        response: Optional[requests.Response] = None,
        error: Optional[str] = None
    ) -> ScrapingResult:
        """Create a ScrapingResult object."""
        if error:
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message=error
            )
        
        if not response:
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message="No response received"
            )
        
        try:
            soup = self._parse_html(response.text)
            info = self._extract_basic_info(soup, url)
            
            return ScrapingResult(
                url=url,
                title=info['title'],
                content=info['content'],
                documentation_type=info['documentation_type'],
                status=ScrapingStatus.COMPLETED,
                metadata=info['metadata'],
                raw_html=response.text
            )
            
        except Exception as e:
            logger.error(f"Failed to parse content from {url}: {e}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message=str(e)
            )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
