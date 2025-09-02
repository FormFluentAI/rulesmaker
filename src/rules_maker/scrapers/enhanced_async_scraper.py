"""
Enhanced async documentation scraper with URL validation, redirect handling,
and intelligent error recovery mechanisms.

This enhanced version addresses common scraping failures by:
1. Following redirects automatically
2. Validating URLs before scraping
3. Implementing fallback strategies for failed requests
4. Providing detailed failure diagnostics
"""

import asyncio
import aiohttp
import time
import logging
from typing import List, Optional, Dict, Any, Set, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass

from .async_documentation_scraper import AsyncDocumentationScraper
from ..models import ScrapingResult, ScrapingConfig, ScrapingStatus, DocumentationType
from ..utils import detect_documentation_type, extract_main_content

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of URL validation."""
    is_valid: bool
    final_url: str
    status_code: int
    error_message: Optional[str] = None
    redirects: List[str] = None


class EnhancedAsyncDocumentationScraper(AsyncDocumentationScraper):
    """Enhanced async scraper with validation, redirect handling, and error recovery."""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        super().__init__(config)
        self.url_cache: Dict[str, ValidationResult] = {}
        self.failed_urls: Set[str] = set()
        self.redirect_patterns: Dict[str, str] = {
            # Known redirect patterns for popular sites
            'reactjs.org/docs/': 'react.dev/learn',
            'fastify.io/docs/': 'www.fastify.io/docs/latest/',
            'rubyonrails.org/guides': 'guides.rubyonrails.org/',
            'httpx.python-requests.org': 'www.python-httpx.org',
            'jestjs.io/docs/': 'jestjs.io/',
            'rollupjs.org/guide/': 'rollupjs.org/introduction/',
        }
        
    async def validate_url(self, url: str) -> ValidationResult:
        """Validate a URL and handle redirects."""
        if url in self.url_cache:
            return self.url_cache[url]
            
        if url in self.failed_urls:
            return ValidationResult(
                is_valid=False,
                final_url=url,
                status_code=0,
                error_message="Previously failed URL"
            )
            
        try:
            await self._ensure_session()
            
            # Try the original URL first
            async with self.session.head(url, allow_redirects=True) as response:
                redirects = []
                if response.history:
                    redirects = [str(r.url) for r in response.history]
                
                result = ValidationResult(
                    is_valid=200 <= response.status < 400,
                    final_url=str(response.url),
                    status_code=response.status,
                    redirects=redirects
                )
                
                if result.is_valid:
                    self.url_cache[url] = result
                    logger.debug(f"✅ Validated URL: {url} -> {result.final_url}")
                else:
                    # Try fallback patterns
                    fallback_result = await self._try_fallback_url(url)
                    if fallback_result.is_valid:
                        result = fallback_result
                        self.url_cache[url] = result
                    else:
                        self.failed_urls.add(url)
                        logger.warning(f"❌ URL validation failed: {url} (Status: {response.status})")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ URL validation error for {url}: {e}")
            
            # Try fallback patterns on exception
            fallback_result = await self._try_fallback_url(url)
            if fallback_result.is_valid:
                self.url_cache[url] = fallback_result
                return fallback_result
            
            self.failed_urls.add(url)
            return ValidationResult(
                is_valid=False,
                final_url=url,
                status_code=0,
                error_message=str(e)
            )
    
    async def _try_fallback_url(self, original_url: str) -> ValidationResult:
        """Try fallback URL patterns for known redirected sites."""
        for pattern, replacement in self.redirect_patterns.items():
            if pattern in original_url:
                fallback_url = original_url.replace(pattern, replacement)
                if not fallback_url.startswith('http'):
                    fallback_url = f"https://{fallback_url}"
                
                try:
                    async with self.session.head(fallback_url, allow_redirects=True) as response:
                        if 200 <= response.status < 400:
                            logger.info(f"🔄 Found working fallback: {original_url} -> {fallback_url}")
                            return ValidationResult(
                                is_valid=True,
                                final_url=fallback_url,
                                status_code=response.status,
                                redirects=[original_url]
                            )
                except Exception:
                    continue
        
        return ValidationResult(
            is_valid=False,
            final_url=original_url,
            status_code=0,
            error_message="No valid fallback found"
        )
    
    async def scrape_url(self, url: str, **kwargs) -> ScrapingResult:
        """Enhanced URL scraping with validation and error recovery."""
        # First, validate the URL
        validation = await self.validate_url(url)
        
        if not validation.is_valid:
            logger.warning(f"⚠️ Skipping invalid URL: {url} - {validation.error_message}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message=f"URL validation failed: {validation.error_message}"
            )
        
        # Use the validated/redirected URL for scraping
        scraping_url = validation.final_url
        
        try:
            await self._ensure_session()
            
            async with self.semaphore:
                # Apply rate limiting
                await asyncio.sleep(self.config.rate_limit)
                
                # Make the request with enhanced error handling
                async with self.session.get(
                    scraping_url,
                    allow_redirects=True,
                    ssl=False  # More lenient SSL for docs sites
                ) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        result = await self._process_content(scraping_url, html_content)
                        
                        # Update metadata with validation info
                        result.metadata.update({
                            'original_url': url,
                            'final_url': scraping_url,
                            'redirected': scraping_url != url,
                            'validation_status': 'success'
                        })
                        
                        logger.info(f"✅ Successfully scraped: {scraping_url}")
                        return result
                    else:
                        logger.warning(f"⚠️ HTTP {response.status} for {scraping_url}")
                        return ScrapingResult(
                            url=url,
                            title="",
                            content="",
                            status=ScrapingStatus.FAILED,
                            error_message=f"HTTP {response.status}",
                            metadata={'final_url': scraping_url}
                        )
                        
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Timeout while scraping {scraping_url}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message="Request timeout",
                metadata={'final_url': scraping_url}
            )
        except Exception as e:
            logger.error(f"❌ Error scraping {scraping_url}: {str(e)}")
            return ScrapingResult(
                url=url,
                title="",
                content="",
                status=ScrapingStatus.FAILED,
                error_message=str(e),
                metadata={'final_url': scraping_url}
            )
    
    async def scrape_multiple_with_validation(
        self, 
        urls: List[str], 
        **kwargs
    ) -> Tuple[List[ScrapingResult], Dict[str, Any]]:
        """Scrape multiple URLs with validation and detailed reporting."""
        
        logger.info(f"🔍 Pre-validating {len(urls)} URLs...")
        
        # Pre-validate all URLs
        validation_tasks = [self.validate_url(url) for url in urls]
        validations = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Separate valid and invalid URLs
        valid_urls = []
        validation_stats = {
            'total_urls': len(urls),
            'valid_urls': 0,
            'invalid_urls': 0,
            'redirected_urls': 0,
            'failed_validations': [],
            'successful_redirects': []
        }
        
        for i, (url, validation) in enumerate(zip(urls, validations)):
            if isinstance(validation, Exception):
                validation_stats['failed_validations'].append({
                    'url': url,
                    'error': str(validation)
                })
                continue
                
            if validation.is_valid:
                valid_urls.append(url)
                validation_stats['valid_urls'] += 1
                
                if validation.final_url != url:
                    validation_stats['redirected_urls'] += 1
                    validation_stats['successful_redirects'].append({
                        'original': url,
                        'final': validation.final_url
                    })
            else:
                validation_stats['invalid_urls'] += 1
                validation_stats['failed_validations'].append({
                    'url': url,
                    'error': validation.error_message,
                    'status_code': validation.status_code
                })
        
        logger.info(f"✅ Validation complete: {validation_stats['valid_urls']}/{len(urls)} URLs valid")
        if validation_stats['redirected_urls'] > 0:
            logger.info(f"🔄 {validation_stats['redirected_urls']} URLs were redirected")
        
        # Scrape only valid URLs
        if valid_urls:
            results = await self.scrape_multiple(valid_urls, **kwargs)
        else:
            results = []
            
        return results, validation_stats
    
    async def scrape_documentation_site_enhanced(
        self,
        base_url: str,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[ScrapingResult], Dict[str, Any]]:
        """Enhanced site scraping with validation and detailed reporting."""
        
        logger.info(f"🚀 Starting enhanced documentation site scraping: {base_url}")
        
        # Validate the base URL first
        validation = await self.validate_url(base_url)
        if not validation.is_valid:
            logger.error(f"❌ Base URL validation failed: {base_url}")
            return [], {
                'error': f"Base URL validation failed: {validation.error_message}",
                'base_url': base_url
            }
        
        # Use the validated URL
        validated_base_url = validation.final_url
        logger.info(f"✅ Validated base URL: {validated_base_url}")
        
        max_pages = max_pages or self.config.max_pages
        max_depth = max_depth or self.config.max_depth
        
        urls_to_scrape = [validated_base_url]
        results = []
        depth = 0
        
        scraping_stats = {
            'base_url': base_url,
            'final_base_url': validated_base_url,
            'total_pages_found': 0,
            'pages_scraped': 0,
            'pages_failed': 0,
            'max_depth_reached': 0,
            'link_discovery_stats': []
        }
        
        while urls_to_scrape and len(results) < max_pages and depth < max_depth:
            logger.info(f"🔍 Depth {depth + 1}: Processing {len(urls_to_scrape)} URLs")
            
            current_batch = urls_to_scrape.copy()
            urls_to_scrape.clear()
            
            # Scrape current batch with validation
            batch_results, batch_stats = await self.scrape_multiple_with_validation(current_batch, **kwargs)
            results.extend(batch_results)
            
            scraping_stats['pages_scraped'] += batch_stats['valid_urls']
            scraping_stats['pages_failed'] += batch_stats['invalid_urls']
            
            # Extract links for next depth level
            if depth < max_depth - 1 and self.config.follow_links:
                new_urls = set()
                for result in batch_results:
                    if result.status == ScrapingStatus.COMPLETED and result.raw_html:
                        links = self._extract_documentation_links(
                            result.raw_html,
                            str(result.url),
                            validated_base_url
                        )
                        new_urls.update(links)
                
                # Filter out already visited URLs
                new_urls = new_urls - self._visited_urls
                urls_to_scrape = list(new_urls)[:max_pages - len(results)]
                self._visited_urls.update(urls_to_scrape)
                
                scraping_stats['link_discovery_stats'].append({
                    'depth': depth,
                    'links_found': len(new_urls),
                    'links_selected': len(urls_to_scrape)
                })
                
                scraping_stats['total_pages_found'] += len(new_urls)
            
            depth += 1
            scraping_stats['max_depth_reached'] = depth
        
        logger.info(f"🎉 Enhanced scraping complete: {len(results)} pages scraped")
        return results, scraping_stats
    
    def get_failure_report(self) -> Dict[str, Any]:
        """Generate a detailed failure report."""
        return {
            'failed_urls': list(self.failed_urls),
            'cached_validations': len(self.url_cache),
            'successful_validations': len([v for v in self.url_cache.values() if v.is_valid]),
            'redirect_patterns_used': list(self.redirect_patterns.keys()),
            'validation_cache': {
                url: {
                    'is_valid': result.is_valid,
                    'final_url': result.final_url,
                    'status_code': result.status_code,
                    'redirects': len(result.redirects or [])
                }
                for url, result in self.url_cache.items()
            }
        }
    
    async def clear_cache(self):
        """Clear validation cache and failed URLs."""
        self.url_cache.clear()
        self.failed_urls.clear()
        logger.info("🧹 Cleared validation cache and failed URLs list")