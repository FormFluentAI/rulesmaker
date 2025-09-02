"""
Adaptive documentation scraper with ML-based content recognition.

Combines the async scraper with ML extraction capabilities
to intelligently identify and extract documentation content.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .async_documentation_scraper import AsyncDocumentationScraper
from ..extractors.ml_extractor import MLContentExtractor
from ..extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider
from ..models import (
    ScrapingResult, ScrapingConfig, ScrapingStatus,
    DocumentationType, ContentSection, TrainingSet
)

logger = logging.getLogger(__name__)


class AdaptiveDocumentationScraper(AsyncDocumentationScraper):
    """ML-enhanced scraper that learns from documentation patterns."""
    
    def __init__(
        self, 
        config: Optional[ScrapingConfig] = None,
        ml_model_path: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_ml: bool = True,
        use_llm: bool = False,
        app_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the adaptive scraper."""
        super().__init__(config)
        
        self.use_ml = use_ml
        self.use_llm = use_llm
        
        # Initialize extractors
        if self.use_ml:
            self.ml_extractor = MLContentExtractor(
                model_path=ml_model_path,
                use_transformers=True
            )
        else:
            self.ml_extractor = None
        
        if self.use_llm:
            self.llm_extractor = LLMContentExtractor(llm_config=llm_config, config=app_config or {})
        else:
            self.llm_extractor = None
        
        # Performance tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'ml_extractions': 0,
            'llm_extractions': 0,
            'fallback_extractions': 0,
            'avg_confidence': 0.0
        }
    
    async def scrape_url(self, url: str, **kwargs) -> ScrapingResult:
        """Scrape a single URL with adaptive content extraction."""
        # First get the basic scraping result
        result = await super().scrape_url(url, **kwargs)
        
        if result.status != ScrapingStatus.COMPLETED or not result.raw_html:
            return result
        
        # Enhance with adaptive extraction
        try:
            enhanced_result = await self._enhance_with_adaptive_extraction(result)
            return enhanced_result
        except Exception as e:
            logger.error(f"Adaptive extraction failed for {url}: {str(e)}")
            return result
    
    async def scrape_multiple(self, urls: List[str], **kwargs) -> List[ScrapingResult]:
        """Scrape multiple URLs with adaptive extraction."""
        # Get basic results first
        results = await super().scrape_multiple(urls, **kwargs)
        
        # Enhance each result
        enhanced_results = []
        for result in results:
            if result.status == ScrapingStatus.COMPLETED and result.raw_html:
                try:
                    enhanced_result = await self._enhance_with_adaptive_extraction(result)
                    enhanced_results.append(enhanced_result)
                except Exception as e:
                    logger.error(f"Adaptive extraction failed for {result.url}: {str(e)}")
                    enhanced_results.append(result)
            else:
                enhanced_results.append(result)
        
        return enhanced_results
    
    async def _enhance_with_adaptive_extraction(self, result: ScrapingResult) -> ScrapingResult:
        """Enhance scraping result with adaptive ML/LLM extraction."""
        from bs4 import BeautifulSoup
        # If there is no HTML to enhance, return as-is
        if not result.raw_html:
            return result

        soup = BeautifulSoup(result.raw_html, 'html.parser')
        url = str(result.url)
        
        # Try ML extraction first
        if self.use_ml and self.ml_extractor:
            try:
                ml_data = self.ml_extractor.extract(soup, url)
                ml_sections = self.ml_extractor.extract_sections(soup, url)
                
                # Update result with ML extraction
                result.sections = ml_sections
                result.metadata.update({
                    'ml_extraction': ml_data,
                    'extraction_method': 'ml',
                    'confidence_score': ml_data.get('confidence_score', 0.0)
                })
                
                self.extraction_stats['ml_extractions'] += 1
                self.extraction_stats['total_extractions'] += 1
                
                logger.info(f"ML extraction completed for {url}")
                return result
                
            except Exception as e:
                logger.warning(f"ML extraction failed for {url}: {str(e)}")
        
        # Fallback to LLM extraction
        if self.use_llm and self.llm_extractor:
            try:
                llm_data = self.llm_extractor.extract(soup, url)
                llm_sections = self.llm_extractor.extract_sections(soup, url)
                
                # Update result with LLM extraction
                result.sections = llm_sections
                result.metadata.update({
                    'llm_extraction': llm_data,
                    'extraction_method': 'llm',
                    'confidence_score': 0.8  # LLM generally reliable
                })
                
                self.extraction_stats['llm_extractions'] += 1
                self.extraction_stats['total_extractions'] += 1
                
                logger.info(f"LLM extraction completed for {url}")
                return result
                
            except Exception as e:
                logger.warning(f"LLM extraction failed for {url}: {str(e)}")
        
        # Fallback to basic extraction
        result.metadata.update({
            'extraction_method': 'fallback',
            'confidence_score': 0.5
        })
        
        self.extraction_stats['fallback_extractions'] += 1
        self.extraction_stats['total_extractions'] += 1
        
        return result
    
    async def train_ml_extractor(self, training_set: TrainingSet) -> Dict[str, float]:
        """Train the ML extractor with provided examples."""
        if not self.use_ml or not self.ml_extractor:
            raise ValueError("ML extractor not enabled")
        
        logger.info(f"Training ML extractor with {len(training_set.examples)} examples")
        
        # Train the ML model
        performance = self.ml_extractor.train(training_set)
        
        logger.info(f"ML training completed. Performance: {performance}")
        return performance
    
    def save_ml_model(self, path: str) -> None:
        """Save the trained ML model."""
        if not self.use_ml or not self.ml_extractor:
            raise ValueError("ML extractor not enabled")
        
        self.ml_extractor.save_model(path)
        logger.info(f"ML model saved to {path}")
    
    def load_ml_model(self, path: str) -> None:
        """Load a trained ML model."""
        if not self.use_ml or not self.ml_extractor:
            raise ValueError("ML extractor not enabled")
        
        self.ml_extractor.load_model(path)
        logger.info(f"ML model loaded from {path}")
    
    async def generate_rules_from_content(
        self, 
        content: List[ContentSection],
        target_format: str = "cursor",
        context: Optional[Dict[str, Any]] = None
    ):
        """Generate coding rules from extracted content."""
        if not self.use_llm or not self.llm_extractor:
            raise ValueError("LLM extractor not enabled for rule generation")
        
        return await self.llm_extractor.generate_rules(content, target_format, context)
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction performance statistics."""
        stats = self.extraction_stats.copy()
        
        if stats['total_extractions'] > 0:
            stats['ml_success_rate'] = stats['ml_extractions'] / stats['total_extractions']
            stats['llm_success_rate'] = stats['llm_extractions'] / stats['total_extractions']
            stats['fallback_rate'] = stats['fallback_extractions'] / stats['total_extractions']
        else:
            stats['ml_success_rate'] = 0.0
            stats['llm_success_rate'] = 0.0
            stats['fallback_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset extraction statistics."""
        self.extraction_stats = {
            'total_extractions': 0,
            'ml_extractions': 0,
            'llm_extractions': 0,
            'fallback_extractions': 0,
            'avg_confidence': 0.0
        }
    
    async def close(self):
        """Close the scraper and cleanup resources."""
        await super().close()
        
        if self.llm_extractor:
            await self.llm_extractor.close()
    
    # Synchronous interface compatibility (use distinct names to avoid overriding async methods)
    def scrape_url_sync(self, url: str, **kwargs) -> ScrapingResult:
        """Synchronous wrapper for adaptive scrape_url (does not override async method)."""
        return asyncio.run(self.scrape_url_async(url, **kwargs))
    
    def scrape_multiple_sync(self, urls: List[str], **kwargs) -> List[ScrapingResult]:
        """Synchronous wrapper for adaptive scrape_multiple (does not override async method)."""
        return asyncio.run(self.scrape_multiple_async(urls, **kwargs))
    
    async def scrape_url_async(self, url: str, **kwargs) -> ScrapingResult:
        """Async version of scrape_url."""
        return await super().scrape_url(url, **kwargs)
    
    async def scrape_multiple_async(self, urls: List[str], **kwargs) -> List[ScrapingResult]:
        """Async version of scrape_multiple."""
        return await super().scrape_multiple(urls, **kwargs)
