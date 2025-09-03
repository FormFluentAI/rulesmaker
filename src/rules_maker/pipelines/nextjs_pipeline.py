"""
Next.js Documentation Pipeline

Comprehensive pipeline for scraping, categorizing, and generating cursor rules
for Next.js documentation. Integrates all components: scrapers, categorizers,
transformers, and formatters.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import yaml

from ..scrapers import AsyncDocumentationScraper, AdaptiveDocumentationScraper
from ..intelligence.nextjs_categorizer import NextJSCategorizer, NextJSCategory
from ..transformers.cursor_transformer import CursorRuleTransformer
from ..formatters.cursor_rules_formatter import CursorRulesFormatter
from ..models import ScrapingConfig, ScrapingResult, RuleSet, Rule
from ..batch_processor import DocumentationSource
from ..learning.nextjs_learning_integration import NextJSLearningIntegration

logger = logging.getLogger(__name__)


class NextJSPipeline:
    """Comprehensive Next.js documentation processing pipeline."""
    
    def __init__(
        self,
        output_dir: str = ".cursor/rules/nextjs",
        use_ml: bool = True,
        use_learning: bool = True,
        quality_threshold: float = 0.7,
        max_pages: int = 50,
        rate_limit: float = 0.5
    ):
        """Initialize the Next.js pipeline.
        
        Args:
            output_dir: Directory to save generated cursor rules
            use_ml: Enable ML-enhanced processing
            use_learning: Enable learning system integration
            quality_threshold: Minimum quality threshold for content
            max_pages: Maximum pages to scrape per source
            rate_limit: Rate limit for scraping (seconds between requests)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_ml = use_ml
        self.use_learning = use_learning
        self.quality_threshold = quality_threshold
        
        # Initialize components
        self.categorizer = NextJSCategorizer()
        self.formatter = CursorRulesFormatter(self.categorizer)
        self.transformer = CursorRuleTransformer()
        
        # Initialize learning integration if enabled
        self.learning_integration = None
        if use_learning:
            try:
                self.learning_integration = NextJSLearningIntegration()
            except ImportError:
                logger.warning("Learning integration not available")
        
        # Configure scraper
        self.scraping_config = ScrapingConfig(
            max_pages=max_pages,
            rate_limit=rate_limit,
            max_depth=3
        )
        
        if use_ml:
            self.scraper = AdaptiveDocumentationScraper(
                config=self.scraping_config,
                use_ml=True,
                use_llm=False
            )
        else:
            self.scraper = AsyncDocumentationScraper(config=self.scraping_config)
        
        # Define comprehensive Next.js sources
        self.nextjs_sources = self._initialize_nextjs_sources()
    
    async def close(self):
        """Close the pipeline and clean up resources."""
        if hasattr(self.scraper, 'close'):
            await self.scraper.close()
        if self.learning_integration and hasattr(self.learning_integration, 'close'):
            await self.learning_integration.close()
        
    def _initialize_nextjs_sources(self) -> Dict[str, List[DocumentationSource]]:
        """Initialize comprehensive Next.js documentation sources."""
        return {
            'official': [
                # Core Documentation
                DocumentationSource("https://nextjs.org/docs", "Next.js Official Docs", "nextjs", "overview", 10),
                DocumentationSource("https://nextjs.org/docs/getting-started", "Getting Started", "nextjs", "getting-started", 10),
                DocumentationSource("https://nextjs.org/docs/getting-started/installation", "Installation", "nextjs", "setup", 9),
                
                # Routing & Navigation
                DocumentationSource("https://nextjs.org/docs/app", "App Router", "nextjs", "routing", 10),
                DocumentationSource("https://nextjs.org/docs/pages", "Pages Router", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing", "App Router Routing", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing", "Pages Router Routing", "nextjs", "routing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/linking-and-navigating", "Linking and Navigating", "nextjs", "routing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/middleware", "Middleware", "nextjs", "routing", 8),
                
                # Components & Rendering
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering", "Rendering", "nextjs", "rendering", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/server-components", "Server Components", "nextjs", "components", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/client-components", "Client Components", "nextjs", "components", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/composition-patterns", "Composition Patterns", "nextjs", "components", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/partial-prerendering", "Partial Prerendering", "nextjs", "rendering", 8),
                
                # Data Fetching
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching", "Data Fetching", "nextjs", "data-fetching", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/fetching", "Fetching Data", "nextjs", "data-fetching", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/caching", "Caching", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/revalidating", "Revalidating", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/forms-and-mutations", "Forms and Mutations", "nextjs", "data-fetching", 8),
                
                # Styling & UI
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling", "Styling", "nextjs", "styling", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/css-modules", "CSS Modules", "nextjs", "styling", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/tailwind-css", "Tailwind CSS", "nextjs", "styling", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/sass", "Sass", "nextjs", "styling", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/css-in-js", "CSS-in-JS", "nextjs", "styling", 7),
                
                # Optimization
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing", "Optimizing", "nextjs", "optimization", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/images", "Image Optimization", "nextjs", "optimization", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/fonts", "Font Optimization", "nextjs", "optimization", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/static-assets", "Static Assets", "nextjs", "optimization", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/lazy-loading", "Lazy Loading", "nextjs", "optimization", 7),
                
                # Configuration
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "Next.js Config", "nextjs", "configuration", 8),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "App Directory Config", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "Compiler Options", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "Experimental Features", "nextjs", "configuration", 6),
                
                # API Reference
                DocumentationSource("https://nextjs.org/docs/app/api-reference", "API Reference", "nextjs", "api", 8),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/functions", "Functions", "nextjs", "api", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/components", "Components", "nextjs", "api", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/file-conventions", "File Conventions", "nextjs", "api", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "Config Reference", "nextjs", "api", 7),
                
                # Advanced Features
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication", "Authentication", "nextjs", "authentication", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/internationalization", "Internationalization", "nextjs", "i18n", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/analytics", "Analytics", "nextjs", "analytics", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/upgrading", "Upgrading", "nextjs", "migration", 6),
                
                # Error Handling
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/error-handling", "Error Handling", "nextjs", "error-handling", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/file-conventions/error", "Error Pages", "nextjs", "error-handling", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/file-conventions/not-found", "Not Found Pages", "nextjs", "error-handling", 6),
                
                # Deployment
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying", "Deploying", "nextjs", "deployment", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying/static-exports", "Static Exports", "nextjs", "deployment", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying", "Docker", "nextjs", "deployment", 7),
            ],
            'community': [
                DocumentationSource("https://nextjs.org/learn", "Next.js Learn", "nextjs", "tutorial", 9),
                DocumentationSource("https://nextjs.org/learn/dashboard-app", "Dashboard App Tutorial", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/about-nextjs", "About Next.js", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/from-javascript-to-react", "From JavaScript to React", "nextjs", "tutorial", 7),
                DocumentationSource("https://nextjs.org/learn/foundations/from-react-to-nextjs", "From React to Next.js", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/how-nextjs-works", "How Next.js Works", "nextjs", "tutorial", 8),
            ],
            'ecosystem': [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication", "Authentication Patterns", "nextjs", "authentication", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/internationalization", "i18n Patterns", "nextjs", "i18n", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/analytics", "Analytics Integration", "nextjs", "analytics", 6),
            ]
        }
    
    async def process_sources(
        self, 
        source_types: List[str] = None,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Process Next.js documentation sources and generate cursor rules.
        
        Args:
            source_types: Types of sources to process ('official', 'community', 'ecosystem', 'all')
            parallel: Enable parallel processing
            
        Returns:
            Dictionary with processing results and statistics
        """
        if source_types is None:
            source_types = ['all']
        
        if 'all' in source_types:
            sources_to_process = []
            for source_list in self.nextjs_sources.values():
                sources_to_process.extend(source_list)
        else:
            sources_to_process = []
            for source_type in source_types:
                if source_type in self.nextjs_sources:
                    sources_to_process.extend(self.nextjs_sources[source_type])
        
        logger.info(f"Processing {len(sources_to_process)} Next.js documentation sources")
        
        # Use async context manager for proper session cleanup
        try:
            async with self.scraper:
                logger.debug("Using scraper context manager")
                # Process sources
                if parallel:
                    results = await self._process_sources_parallel(sources_to_process)
                else:
                    results = await self._process_sources_sequential(sources_to_process)
                logger.debug("Finished processing sources in context manager")
        except Exception as e:
            logger.error(f"Error in scraper context manager: {e}")
            # Fallback: ensure scraper is closed
            if hasattr(self.scraper, 'close'):
                await self.scraper.close()
            raise
        
        # Generate cursor rules
        cursor_rules = await self._generate_cursor_rules(results)
        
        # Save results
        await self._save_results(cursor_rules, results)
        
        return {
            'sources_processed': len(sources_to_process),
            'pages_scraped': len([r for r in results.values() if r.content]),
            'cursor_rules_generated': len(cursor_rules),
            'output_directory': str(self.output_dir),
            'processing_time': datetime.now().isoformat(),
            'results': results,
            'cursor_rules': cursor_rules
        }
    
    async def _process_sources_parallel(
        self, 
        sources: List[DocumentationSource]
    ) -> Dict[str, ScrapingResult]:
        """Process sources in parallel."""
        tasks = []
        for source in sources:
            task = self._process_single_source(source)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and create results dict
        processed_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process source {sources[i].url}: {result}")
            else:
                processed_results[sources[i].url] = result
        
        return processed_results
    
    async def _process_sources_sequential(
        self, 
        sources: List[DocumentationSource]
    ) -> Dict[str, ScrapingResult]:
        """Process sources sequentially."""
        results = {}
        
        for source in sources:
            try:
                result = await self._process_single_source(source)
                results[source.url] = result
            except Exception as e:
                logger.error(f"Failed to process source {source.url}: {e}")
        
        return results
    
    async def _process_single_source(self, source: DocumentationSource) -> ScrapingResult:
        """Process a single documentation source."""
        logger.info(f"Processing source: {source.name} ({source.url})")
        
        # Scrape content
        scraping_result = await self.scraper.scrape_url(source.url)
        
        # Categorize content
        if scraping_result.content and len(scraping_result.content.strip()) > 100:
            categories = await self.categorizer.categorize_nextjs_content(
                scraping_result.content, 
                str(scraping_result.url),
                {'source_type': source.category, 'priority': source.priority}
            )
            
            # Filter by quality threshold
            if any(cat.confidence >= self.quality_threshold for cat in categories.values()):
                scraping_result.metadata = scraping_result.metadata or {}
                scraping_result.metadata['categories'] = categories
                scraping_result.metadata['source_info'] = {
                    'name': source.name,
                    'category': source.category,
                    'priority': source.priority
                }
        else:
            # No content or too short
            scraping_result.metadata = scraping_result.metadata or {}
            scraping_result.metadata['categories'] = {}
        
        # Learn from results if learning is enabled
        if self.learning_integration:
            await self.learning_integration.learn_from_scraping_result(scraping_result)
        
        return scraping_result
    
    async def _generate_cursor_rules(
        self, 
        results: Dict[str, ScrapingResult]
    ) -> Dict[str, str]:
        """Generate cursor rules from processed results."""
        logger.info("Generating cursor rules from processed results")
        
        # Group pages by category
        categorized_content = self._group_content_by_category(results)
        
        # Generate rules for each category
        cursor_rules = {}
        
        for category, pages in categorized_content.items():
            if not pages:  # Skip empty categories
                continue
                
            logger.info(f"Generating cursor rules for category: {category} ({len(pages)} pages)")
            
            # Transform content to rules
            rule_content = await self.transformer.transform_with_formatter(
                pages, 
                category_hint=category
            )
            
            # Format as cursor rule using the formatter
            formatted_rules = await self.formatter.format_scraping_results(
                pages, 
                output_format='mdc',
                category_hint=category
            )
            
            # Add formatted rules to our collection
            for filename, content in formatted_rules.items():
                cursor_rules[filename] = content
        
        return cursor_rules
    
    def _group_content_by_category(
        self, 
        results: Dict[str, ScrapingResult]
    ) -> Dict[str, List[Any]]:
        """Group scraped content by Next.js category."""
        categorized = {}
        
        for result in results.values():
            if not result.metadata or 'categories' not in result.metadata:
                continue
            
            categories = result.metadata['categories']
            
            # Find the highest confidence category
            best_category = None
            best_confidence = 0.0
            
            for category_name, category_data in categories.items():
                if category_data.confidence > best_confidence:
                    best_confidence = category_data.confidence
                    best_category = category_name
            
            if best_category and best_confidence >= self.quality_threshold:
                if best_category not in categorized:
                    categorized[best_category] = []
                categorized[best_category].append(result)
        
        return categorized
    
    def _get_globs_for_category(self, category: str) -> List[str]:
        """Get appropriate file globs for a Next.js category."""
        glob_mapping = {
            'routing': ['**/app/**/*', '**/pages/**/*', '**/middleware.ts', '**/route.ts'],
            'data-fetching': ['**/app/**/*', '**/pages/**/*', '**/api/**/*'],
            'styling': ['**/*.css', '**/*.scss', '**/*.sass', '**/*.module.css', '**/tailwind.config.*'],
            'deployment': ['**/next.config.*', '**/vercel.json', '**/Dockerfile', '**/.env*'],
            'performance': ['**/next.config.*', '**/app/**/*', '**/pages/**/*'],
            'security': ['**/middleware.ts', '**/api/**/*', '**/app/**/*'],
            'testing': ['**/*.test.*', '**/*.spec.*', '**/__tests__/**/*', '**/tests/**/*'],
            'api-routes': ['**/api/**/*', '**/route.ts', '**/route.js'],
            'middleware': ['**/middleware.ts', '**/middleware.js'],
            'configuration': ['**/next.config.*', '**/package.json', '**/tsconfig.json'],
            'optimization': ['**/next.config.*', '**/app/**/*', '**/pages/**/*'],
            'troubleshooting': ['**/*.md', '**/README.md', '**/docs/**/*'],
            'migration': ['**/package.json', '**/next.config.*', '**/*.md'],
            'advanced-patterns': ['**/app/**/*', '**/pages/**/*', '**/components/**/*']
        }
        
        return glob_mapping.get(category, ['**/*.tsx', '**/*.ts', '**/*.jsx', '**/*.js'])
    
    async def _save_results(
        self, 
        cursor_rules: Dict[str, str], 
        results: Dict[str, ScrapingResult]
    ):
        """Save generated cursor rules and processing results."""
        logger.info(f"Saving results to {self.output_dir}")
        
        # Save cursor rules
        for filename, content in cursor_rules.items():
            rule_path = self.output_dir / filename
            with open(rule_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved cursor rule: {rule_path}")
        
        # Save processing summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'sources_processed': len(results),
            'pages_scraped': len([r for r in results.values() if r.content]),
            'cursor_rules_generated': len(cursor_rules),
            'categories': list(cursor_rules.keys()),
            'processing_config': {
                'use_ml': self.use_ml,
                'use_learning': self.use_learning,
                'quality_threshold': self.quality_threshold,
                'max_pages': self.scraping_config.max_pages,
                'rate_limit': self.scraping_config.rate_limit
            }
        }
        
        summary_path = self.output_dir / 'processing_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved processing summary: {summary_path}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and performance metrics."""
        stats = {
            'categorizer_stats': self.categorizer.get_categorization_stats(),
            'output_directory': str(self.output_dir),
            'configuration': {
                'use_ml': self.use_ml,
                'use_learning': self.use_learning,
                'quality_threshold': self.quality_threshold,
                'max_pages': self.scraping_config.max_pages,
                'rate_limit': self.scraping_config.rate_limit
            },
            'sources_available': {
                category: len(sources) 
                for category, sources in self.nextjs_sources.items()
            }
        }
        
        if self.learning_integration:
            stats['learning_stats'] = self.learning_integration.get_learning_stats()
        
        return stats
