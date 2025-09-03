"""
Base strategy classes for Rules Maker.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from bs4 import BeautifulSoup

from ..models import ScrapingResult, DocumentationStructure, RuleSet, ScrapingConfig, LearningExample


class ScrapingStrategy(ABC):
    """Base strategy for web scraping."""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        """Initialize the strategy.
        
        Args:
            config: Scraping configuration
        """
        self.config = config or ScrapingConfig()
    
    @abstractmethod
    async def scrape_url(self, url: str) -> ScrapingResult:
        """Scrape a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Scraping result
        """
        pass
    
    @abstractmethod
    async def scrape_multiple(self, urls: List[str]) -> List[ScrapingResult]:
        """Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraping results
        """
        pass


class ContentExtractionStrategy(ABC):
    """Base strategy for content extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy.
        
        Args:
            config: Extraction configuration
        """
        self.config = config or {}
    
    @abstractmethod
    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content from HTML.
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            
        Returns:
            Extracted content
        """
        pass
    
    @abstractmethod
    def can_handle(self, url: str, content: str) -> bool:
        """Check if this strategy can handle the content.
        
        Args:
            url: Source URL
            content: HTML content
            
        Returns:
            True if strategy can handle this content
        """
        pass


class RuleGenerationStrategy(ABC):
    """Base strategy for rule generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy.
        
        Args:
            config: Generation configuration
        """
        self.config = config or {}
    
    @abstractmethod
    def generate_rules(self, documentation: DocumentationStructure) -> RuleSet:
        """Generate rules from documentation.
        
        Args:
            documentation: Structured documentation
            
        Returns:
            Generated rule set
        """
        pass
    
    @abstractmethod
    def get_rule_quality_score(self, rules: RuleSet) -> float:
        """Calculate quality score for generated rules.
        
        Args:
            rules: Rule set to evaluate
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        pass


class BaseScrapingStrategy(ScrapingStrategy):
    """Basic implementation of scraping strategy."""
    
    async def scrape_url(self, url: str) -> ScrapingResult:
        """Basic URL scraping implementation."""
        import aiohttp
        from ..models import ScrapingStatus
        
        try:
            # Use proper session management with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    content = await response.text()
                    
                    return ScrapingResult(
                        url=url,
                        content=content,
                        status=ScrapingStatus.COMPLETED,
                        metadata={
                            'status_code': response.status,
                            'content_type': response.headers.get('content-type', ''),
                            'content_length': len(content)
                        }
                    )
        except Exception as e:
            return ScrapingResult(
                url=url,
                content="",
                status=ScrapingStatus.FAILED,
                error_message=str(e),
                metadata={}
            )
    
    async def scrape_multiple(self, urls: List[str]) -> List[ScrapingResult]:
        """Basic multiple URL scraping implementation."""
        import asyncio
        
        tasks = [self.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)


class BasicContentExtractionStrategy(ContentExtractionStrategy):
    """Basic content extraction strategy."""
    
    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Basic content extraction."""
        from ..utils import extract_main_content, detect_documentation_type
        
        # Extract title
        title_element = soup.find('title')
        title = title_element.get_text(strip=True) if title_element else ""
        
        # Extract main content
        content = extract_main_content(soup)
        
        # Detect documentation type
        doc_type = detect_documentation_type(url, title, content)
        
        return {
            'title': title,
            'content': content,
            'url': url,
            'documentation_type': doc_type,
            'metadata': {
                'extraction_method': 'basic',
                'content_length': len(content)
            }
        }
    
    def can_handle(self, url: str, content: str) -> bool:
        """Basic strategy can handle any content."""
        return True


class BasicRuleGenerationStrategy(RuleGenerationStrategy):
    """Basic rule generation strategy."""
    
    def generate_rules(self, documentation: DocumentationStructure) -> RuleSet:
        """Basic rule generation."""
        from ..models import Rule, RuleType, RuleSet
        
        rules = []
        
        # Generate basic rules from content
        if documentation.title:
            rules.append(Rule(
                id=f"title_rule_{len(rules)}",
                type=RuleType.FORMATTING,
                description=f"Use proper formatting for {documentation.title}",
                content=f"When working with {documentation.title}, follow the documentation guidelines.",
                confidence_score=0.7
            ))
        
        # Generate rules from sections
        for section in documentation.sections:
            if section.title and section.content:
                rules.append(Rule(
                    id=f"section_rule_{len(rules)}",
                    type=RuleType.BEST_PRACTICE,
                    description=f"Guidelines for {section.title}",
                    content=section.content[:200] + "..." if len(section.content) > 200 else section.content,
                    confidence_score=0.6
                ))
        
        # Generate rules from code examples
        for example in documentation.code_examples:
            if example.get('code'):
                rules.append(Rule(
                    id=f"code_rule_{len(rules)}",
                    type=RuleType.CODE_PATTERN,
                    description=f"Code pattern for {example.get('language', 'unknown')}",
                    content=f"Example usage:\n```{example.get('language', '')}\n{example['code']}\n```",
                    confidence_score=0.8
                ))
        
        return RuleSet(
            name=f"Rules for {documentation.title}",
            description=f"Generated rules from {documentation.url}",
            rules=rules,
            metadata={
                'source_url': documentation.url,
                'generation_method': 'basic',
                'total_rules': len(rules)
            }
        )
    
    def get_rule_quality_score(self, rules: RuleSet) -> float:
        """Calculate basic quality score."""
        if not rules.rules:
            return 0.0
        
        # Simple quality metric based on rule count and average confidence
        rule_count_score = min(len(rules.rules) / 10, 1.0)  # Max score at 10+ rules
        avg_confidence = sum(rule.confidence_score for rule in rules.rules) / len(rules.rules)
        
        return (rule_count_score + avg_confidence) / 2
