"""
Async documentation scraper for high-performance scraping.

Implements concurrent scraping with rate limiting, session management,
and robust error handling using aiohttp and asyncio.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urljoin, urlparse
import logging
import warnings

import aiohttp
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from fake_useragent import UserAgent

from .base import BaseScraper
from ..models import (
    ScrapingResult, ScrapingConfig, ScrapingStatus, 
    ContentSection
)
from ..utils import detect_documentation_type, extract_main_content


logger = logging.getLogger(__name__)

# Suppress XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class AsyncDocumentationScraper(BaseScraper):
    """Async scraper for high-performance documentation scraping."""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        """Initialize the async scraper."""
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.user_agent = UserAgent()
        self._visited_urls: Set[str] = set()
        
    async def __aenter__(self):
        """Async context manager entry."""
        logger.debug("AsyncDocumentationScraper: entering context manager")
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        logger.debug("AsyncDocumentationScraper: exiting context manager")
        await self.close()
        
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            # Configure connection limits and timeouts
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=10,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=10,
                sock_read=self.config.timeout
            )
            
            headers = {
                'User-Agent': self.config.user_agent,
                **self.config.headers
            }
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                cookies=self.config.cookies
            )
            
        # Create semaphore for rate limiting
        if self.semaphore is None:
            max_concurrent = min(50, max(5, int(1.0 / self.config.rate_limit)))
            self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def close(self):
        """Close the session and cleanup resources."""
        if self.session and not self.session.closed:
            logger.debug(f"Closing aiohttp session: {self.session}")
            try:
                await self.session.close()
                # Wait a bit for the session to fully close
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
        self.session = None
        self.semaphore = None
        
    async def scrape_url(self, url: str, **kwargs) -> ScrapingResult:
        """Scrape a single URL asynchronously."""
        # Ensure session exists
        if self.session is None or self.session.closed:
            await self._ensure_session()
        
        try:
            async with self.semaphore:
                # Apply rate limiting
                await asyncio.sleep(self.config.rate_limit)
                
                # Make the request
                async with self.session.get(url) as response:
                    if response.status == 200:
                        # Handle encoding properly
                        encoding = response.charset or 'utf-8'
                        html_content = await response.text(encoding=encoding, errors='replace')
                        return await self._process_content(url, html_content)
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return ScrapingResult(
                            url=url,
                            title="",
                            content="",
                            status=ScrapingStatus.FAILED,
                            error_message=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while scraping {url}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message="Request timeout"
            )
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message=str(e)
            )
    
    async def scrape_multiple(self, urls: List[str], **kwargs) -> List[ScrapingResult]:
        """Scrape multiple URLs concurrently."""
        await self._ensure_session()
        
        # Create tasks for all URLs
        tasks = [self.scrape_url(url, **kwargs) for url in urls]
        
        # Execute with progress tracking
        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                logger.info(f"Completed {len(results)}/{len(urls)} URLs")
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
                
        return results
    
    async def scrape_documentation_site(
        self, 
        base_url: str, 
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        **kwargs
    ) -> List[ScrapingResult]:
        """Scrape an entire documentation site with link following."""
        await self._ensure_session()
        
        max_pages = max_pages or self.config.max_pages
        max_depth = max_depth or self.config.max_depth
        
        urls_to_scrape = [base_url]
        results = []
        depth = 0
        
        while urls_to_scrape and len(results) < max_pages and depth < max_depth:
            current_batch = urls_to_scrape.copy()
            urls_to_scrape.clear()
            
            # Scrape current batch
            batch_results = await self.scrape_multiple(current_batch, **kwargs)
            results.extend(batch_results)
            
            # Extract links from successful results for next depth level
            if depth < max_depth - 1 and self.config.follow_links:
                new_urls = set()
                for result in batch_results:
                    if result.status == ScrapingStatus.COMPLETED and result.raw_html:
                        links = self._extract_documentation_links(
                            result.raw_html, 
                            str(result.url), 
                            base_url
                        )
                        new_urls.update(links)
                
                # Filter out already visited URLs
                new_urls = new_urls - self._visited_urls
                urls_to_scrape = list(new_urls)[:max_pages - len(results)]
                self._visited_urls.update(urls_to_scrape)
            
            depth += 1
            
        return results
    
    async def _process_content(self, url: str, html_content: str) -> ScrapingResult:
        """Process scraped HTML content into structured result."""
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.get_text().strip()
            elif soup.find('h1'):
                title = soup.find('h1').get_text().strip()
            
            # Extract main content
            content = extract_main_content(soup)
            
            # Extract sections
            sections = self._extract_sections(soup)
            
            # Detect documentation type
            doc_type = detect_documentation_type(url, title, content)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            return ScrapingResult(
                url=url,
                title=title,
                content=content,
                sections=sections,
                documentation_type=doc_type,
                status=ScrapingStatus.COMPLETED,
                metadata=metadata,
                raw_html=html_content
            )
            
        except Exception as e:
            logger.error(f"Error processing content from {url}: {str(e)}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message=f"Content processing error: {str(e)}"
            )
    
    def _extract_sections(self, soup: BeautifulSoup) -> List[ContentSection]:
        """Extract content sections from HTML."""
        sections = []
        
        # Find all headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            level = int(heading.name[1])  # Extract number from h1, h2, etc.
            title = heading.get_text().strip()
            
            # Get content until next heading of same or higher level
            content_parts = []
            current = heading.next_sibling
            
            while current:
                if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    next_level = int(current.name[1])
                    if next_level <= level:
                        break
                
                if hasattr(current, 'get_text'):
                    text = current.get_text().strip()
                    if text:
                        content_parts.append(text)
                        
                current = current.next_sibling
            
            content = '\n'.join(content_parts)
            
            if title and content:
                section = ContentSection(
                    title=title,
                    content=content,
                    level=level,
                    metadata={'tag': heading.name}
                )
                sections.append(section)
        
        return sections
    
    def _extract_documentation_links(
        self, 
        html_content: str, 
        current_url: str, 
        base_url: str
    ) -> Set[str]:
        """Extract relevant documentation links from HTML."""
        soup = BeautifulSoup(html_content, 'lxml')
        links = set()
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            full_url = urljoin(current_url, href)
            
            # Only include links within the same domain
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                # Filter for documentation-like URLs
                if self._is_documentation_link(full_url, link.get_text().strip()):
                    links.add(full_url)
        
        return links
    
    def _is_documentation_link(self, url: str, link_text: str) -> bool:
        """Determine if a link is likely to be documentation."""
        # Common documentation URL patterns
        doc_patterns = [
            '/docs/', '/doc/', '/documentation/',
            '/guide/', '/tutorial/', '/reference/',
            '/api/', '/help/', '/manual/'
        ]
        
        # Exclude common non-documentation patterns
        exclude_patterns = [
            '/download', '/login', '/register', '/search',
            '.pdf', '.zip', '.tar', '.exe', '/edit',
            '/delete', '/admin', '/user', '?', '#'
        ]
        
        url_lower = url.lower()
        text_lower = link_text.lower()
        
        # Check exclusions first
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False
        
        # Check for documentation patterns
        for pattern in doc_patterns:
            if pattern in url_lower:
                return True
        
        # Check link text for documentation keywords
        doc_keywords = [
            'guide', 'tutorial', 'documentation', 'reference',
            'api', 'manual', 'help', 'docs', 'getting started'
        ]
        
        for keyword in doc_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {
            'url': url,
            'scraped_at': time.time()
        }
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                metadata[f"meta_{meta['name']}"] = meta.get('content', '')
            elif meta.get('property'):
                metadata[f"meta_{meta['property']}"] = meta.get('content', '')
        
        # Extract language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag['lang']
        
        return metadata
    
    # Synchronous interface compatibility
    def scrape_url_sync(self, url: str, **kwargs) -> ScrapingResult:
        """Synchronous wrapper for async scrape_url."""
        # Use our safe async runner
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We're in an existing event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.scrape_url(url, **kwargs))
                    return future.result()
            else:
                # No running loop, safe to use asyncio.run()
                return asyncio.run(self.scrape_url(url, **kwargs))
        except RuntimeError:
            # No event loop, safe to use asyncio.run()
            return asyncio.run(self.scrape_url(url, **kwargs))
    
    def scrape_multiple_sync(self, urls: List[str], **kwargs) -> List[ScrapingResult]:
        """Synchronous wrapper for async scrape_multiple."""
        # Use our safe async runner
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We're in an existing event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.scrape_multiple(urls, **kwargs))
                    return future.result()
            else:
                # No running loop, safe to use asyncio.run()
                return asyncio.run(self.scrape_multiple(urls, **kwargs))
        except RuntimeError:
            # No event loop, safe to use asyncio.run()
            return asyncio.run(self.scrape_multiple(urls, **kwargs))
