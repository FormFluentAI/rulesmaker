"""
Synchronous documentation scraper implementation.
"""

import time
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .base import BaseScraper
from ..models import ScrapingResult, ScrapingConfig, ContentSection
from ..utils import extract_navigation_links, is_documentation_url


logger = logging.getLogger(__name__)


class DocumentationScraper(BaseScraper):
    """Synchronous scraper optimized for documentation websites."""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        """Initialize the documentation scraper."""
        super().__init__(config)
        self.scraped_urls = set()
        
    def scrape_url(self, url: str, **kwargs) -> ScrapingResult:
        """Scrape a single documentation URL."""
        logger.info(f"Scraping URL: {url}")
        
        try:
            response = self._fetch_page(url)
            result = self._create_scraping_result(url, response)
            
            # Extract sections if this is documentation
            if result.status.value == "completed":
                soup = self._parse_html(response.text)
                result.sections = self._extract_sections(soup, url)
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return self._create_scraping_result(url, error=str(e))
    
    def scrape_multiple(self, urls: List[str], **kwargs) -> List[ScrapingResult]:
        """Scrape multiple URLs sequentially."""
        results = []
        
        for i, url in enumerate(urls):
            if i > 0 and self.config.delay > 0:
                time.sleep(self.config.delay)
            
            result = self.scrape_url(url)
            results.append(result)
            
            logger.info(f"Scraped {i+1}/{len(urls)}: {url}")
        
        return results
    
    def scrape_documentation_site(
        self, 
        base_url: str, 
        max_pages: Optional[int] = None,
        follow_external_links: bool = False
    ) -> List[ScrapingResult]:
        """
        Scrape an entire documentation site by following navigation links.
        
        Args:
            base_url: The starting URL for the documentation site
            max_pages: Maximum number of pages to scrape
            follow_external_links: Whether to follow links to external domains
        
        Returns:
            List of scraping results
        """
        max_pages = max_pages or self.config.max_pages
        results = []
        urls_to_scrape = [base_url]
        self.scraped_urls = set()
        
        base_domain = urlparse(base_url).netloc
        
        while urls_to_scrape and len(results) < max_pages:
            current_url = urls_to_scrape.pop(0)
            
            if current_url in self.scraped_urls:
                continue
                
            self.scraped_urls.add(current_url)
            
            # Apply rate limiting
            if results and self.config.rate_limit > 0:
                time.sleep(self.config.rate_limit)
            
            try:
                result = self.scrape_url(current_url)
                results.append(result)
                
                if result.status.value == "completed":
                    # Extract and queue new URLs
                    new_urls = self._extract_documentation_links(
                        result.raw_html, 
                        current_url,
                        base_domain,
                        follow_external_links
                    )
                    
                    for new_url in new_urls:
                        if new_url not in self.scraped_urls and new_url not in urls_to_scrape:
                            urls_to_scrape.append(new_url)
                
                logger.info(f"Scraped {len(results)}/{max_pages}: {current_url}")
                
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {e}")
                continue
        
        logger.info(f"Completed scraping {len(results)} pages from {base_url}")
        return results
    
    def _extract_sections(self, soup: BeautifulSoup, base_url: str) -> List[ContentSection]:
        """Extract content sections from a documentation page."""
        sections = []
        
        # Find all heading elements
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            level = int(heading.name[1])  # Extract number from h1, h2, etc.
            title = heading.get_text(strip=True)
            
            if not title:
                continue
            
            # Extract content until the next heading of same or higher level
            content_elements = []
            current = heading.next_sibling
            
            while current:
                if hasattr(current, 'name'):
                    if (current.name and 
                        current.name.startswith('h') and 
                        len(current.name) == 2 and
                        current.name[1].isdigit() and
                        int(current.name[1]) <= level):
                        break
                    content_elements.append(current)
                current = current.next_sibling
            
            # Extract text content
            content = ""
            for element in content_elements:
                if hasattr(element, 'get_text'):
                    content += element.get_text(strip=True, separator=' ') + " "
            
            content = content.strip()
            
            if content:
                section = ContentSection(
                    title=title,
                    content=content,
                    level=level,
                    url=base_url
                )
                sections.append(section)
        
        return sections
    
    def _extract_documentation_links(
        self, 
        html_content: str, 
        base_url: str, 
        base_domain: str,
        follow_external: bool = False
    ) -> List[str]:
        """Extract relevant documentation links from HTML content."""
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'lxml')
        links = extract_navigation_links(soup, base_url)
        
        filtered_urls = []
        
        for link in links:
            url = link['url']
            parsed = urlparse(url)
            
            # Filter by domain
            if not follow_external and parsed.netloc != base_domain:
                continue
            
            # Only include documentation-like URLs
            if is_documentation_url(url):
                filtered_urls.append(url)
        
        return filtered_urls
