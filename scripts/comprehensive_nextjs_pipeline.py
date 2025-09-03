#!/usr/bin/env python3
"""
Comprehensive Next.js Documentation Pipeline

This script creates a complete pipeline to gather ALL relevant Next.js documentation,
categorize it properly, and generate comprehensive cursor rules for each topic.

Usage:
    python scripts/comprehensive_nextjs_pipeline.py --help
    python scripts/comprehensive_nextjs_pipeline.py --all-topics --output .cursor/rules/nextjs
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rules_maker.scrapers import AsyncDocumentationScraper, AdaptiveDocumentationScraper
from rules_maker.transformers import CursorRuleTransformer, MLCursorTransformer
from rules_maker.batch_processor import MLBatchProcessor, DocumentationSource
from rules_maker.models import ScrapingConfig, RuleFormat
from rules_maker.intelligence import IntelligentCategoryEngine, SemanticAnalyzer
from rules_maker.intelligence.nextjs_categorizer import NextJSCategorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveNextJSPipeline:
    """Comprehensive Next.js documentation processing pipeline with complete coverage."""
    
    def __init__(
        self,
        output_dir: str = ".cursor/rules/nextjs",
        use_ml: bool = True,
        enable_learning: bool = True,
        quality_threshold: float = 0.7
    ):
        """Initialize the comprehensive Next.js docs pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_ml = use_ml
        self.enable_learning = enable_learning
        self.quality_threshold = quality_threshold
        
        # Initialize components
        self._initialize_components()
        
        # Get comprehensive Next.js documentation sources
        self.nextjs_sources = self._get_comprehensive_nextjs_sources()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        logger.info("Initializing comprehensive pipeline components...")
        
        # Scraping configuration optimized for comprehensive coverage
        self.scraping_config = ScrapingConfig(
            max_pages=200,  # Increased for comprehensive coverage
            rate_limit=0.3,  # Slower rate for better quality
            max_depth=4,     # Deeper crawling
            user_agent="Comprehensive-NextJS-Pipeline/1.0"
        )
        
        # Initialize scrapers
        if self.use_ml:
            self.scraper = AdaptiveDocumentationScraper(
                config=self.scraping_config,
                use_ml=True,
                use_llm=False
            )
            self.transformer = MLCursorTransformer(ml_config={
                'quality_threshold': self.quality_threshold,
                'enable_clustering': True,
                'coherence_threshold': 0.6
            })
        else:
            self.scraper = AsyncDocumentationScraper(config=self.scraping_config)
            self.transformer = CursorRuleTransformer()
        
        # Initialize Next.js specific categorizer
        self.nextjs_categorizer = NextJSCategorizer()
        
        # Initialize intelligence components
        try:
            self.category_engine = IntelligentCategoryEngine()
            self.semantic_analyzer = SemanticAnalyzer()
            logger.info("Intelligence components initialized")
        except Exception as e:
            logger.warning(f"Intelligence components not available: {e}")
            self.category_engine = None
            self.semantic_analyzer = None
        
        # Initialize batch processor
        self.batch_processor = MLBatchProcessor(
            output_dir=str(self.output_dir / "batch_output"),
            quality_threshold=self.quality_threshold,
            max_concurrent=8  # Reduced for stability
        )
        
    def _get_comprehensive_nextjs_sources(self) -> Dict[str, List[DocumentationSource]]:
        """Get comprehensive Next.js documentation sources covering ALL topics."""
        return {
            "core_documentation": [
                # Core Documentation
                DocumentationSource("https://nextjs.org/docs", "Next.js Official Docs", "nextjs", "overview", 10),
                DocumentationSource("https://nextjs.org/docs/getting-started", "Getting Started", "nextjs", "getting-started", 10),
                DocumentationSource("https://nextjs.org/docs/getting-started/installation", "Installation", "nextjs", "setup", 9),
                DocumentationSource("https://nextjs.org/docs/getting-started/project-structure", "Project Structure", "nextjs", "setup", 8),
                DocumentationSource("https://nextjs.org/docs/getting-started/react-essentials", "React Essentials", "nextjs", "react", 9),
                DocumentationSource("https://nextjs.org/docs/getting-started/typescript", "TypeScript", "nextjs", "typescript", 8),
            ],
            
            "routing_navigation": [
                # App Router
                DocumentationSource("https://nextjs.org/docs/app", "App Router", "nextjs", "routing", 10),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing", "App Router Routing", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/layouts-and-pages", "Layouts and Pages", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/linking-and-navigating", "Linking and Navigating", "nextjs", "routing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/loading-ui-and-streaming", "Loading UI and Streaming", "nextjs", "routing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/error-handling", "Error Handling", "nextjs", "routing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/parallel-and-intercepting-routes", "Parallel and Intercepting Routes", "nextjs", "routing", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/route-groups", "Route Groups", "nextjs", "routing", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/dynamic-routes", "Dynamic Routes", "nextjs", "routing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/middleware", "Middleware", "nextjs", "routing", 8),
                
                # Pages Router
                DocumentationSource("https://nextjs.org/docs/pages", "Pages Router", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing", "Pages Router Routing", "nextjs", "routing", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing/dynamic-routes", "Pages Dynamic Routes", "nextjs", "routing", 7),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing/imperative-routing", "Imperative Routing", "nextjs", "routing", 6),
            ],
            
            "components_rendering": [
                # Server and Client Components
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering", "Rendering", "nextjs", "rendering", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/server-components", "Server Components", "nextjs", "components", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/client-components", "Client Components", "nextjs", "components", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/composition-patterns", "Composition Patterns", "nextjs", "components", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/partial-prerendering", "Partial Prerendering", "nextjs", "rendering", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/static-and-dynamic", "Static and Dynamic", "nextjs", "rendering", 7),
                
                # Pages Router Rendering
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/rendering", "Pages Rendering", "nextjs", "rendering", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/rendering/static-site-generation", "Static Site Generation", "nextjs", "rendering", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/rendering/server-side-rendering", "Server-Side Rendering", "nextjs", "rendering", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/rendering/incremental-static-regeneration", "Incremental Static Regeneration", "nextjs", "rendering", 7),
            ],
            
            "data_fetching": [
                # Data Fetching
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching", "Data Fetching", "nextjs", "data-fetching", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/fetching", "Fetching Data", "nextjs", "data-fetching", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/caching-and-revalidating", "Caching and Revalidating", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/forms-and-mutations", "Forms and Mutations", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/patterns", "Data Fetching Patterns", "nextjs", "data-fetching", 7),
                
                # Pages Router Data Fetching
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/data-fetching", "Pages Data Fetching", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/data-fetching/get-static-props", "getStaticProps", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/data-fetching/get-server-side-props", "getServerSideProps", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/data-fetching/get-static-paths", "getStaticPaths", "nextjs", "data-fetching", 7),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/data-fetching/incremental-static-regeneration", "ISR", "nextjs", "data-fetching", 7),
            ],
            
            "styling_ui": [
                # Styling
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling", "Styling", "nextjs", "styling", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/css-modules", "CSS Modules", "nextjs", "styling", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/tailwind-css", "Tailwind CSS", "nextjs", "styling", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/sass", "Sass", "nextjs", "styling", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/css-in-js", "CSS-in-JS", "nextjs", "styling", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/styled-jsx", "Styled JSX", "nextjs", "styling", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/global-styles", "Global Styles", "nextjs", "styling", 6),
            ],
            
            "optimization": [
                # Optimization
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing", "Optimizing", "nextjs", "optimization", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/images", "Image Optimization", "nextjs", "optimization", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/fonts", "Font Optimization", "nextjs", "optimization", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/static-assets", "Static Assets", "nextjs", "optimization", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/lazy-loading", "Lazy Loading", "nextjs", "optimization", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/analytics", "Analytics", "nextjs", "optimization", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/third-party-libraries", "Third-Party Libraries", "nextjs", "optimization", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/bundle-analyzer", "Bundle Analyzer", "nextjs", "optimization", 6),
            ],
            
            "configuration": [
                # Configuration
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "Next.js Config", "nextjs", "configuration", 8),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/appdir", "App Directory Config", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/compiler", "Compiler Options", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/experimental", "Experimental Features", "nextjs", "configuration", 6),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/headers", "Headers", "nextjs", "configuration", 6),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/redirects", "Redirects", "nextjs", "configuration", 6),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/rewrites", "Rewrites", "nextjs", "configuration", 6),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/environment-variables", "Environment Variables", "nextjs", "configuration", 7),
            ],
            
            "api_routes": [
                # API Routes
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/route-handlers", "Route Handlers", "nextjs", "api-routes", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/route-handlers", "API Routes", "nextjs", "api-routes", 8),
                
                # Pages Router API Routes
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing/api-routes", "Pages API Routes", "nextjs", "api-routes", 8),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing/api-routes/request-helpers", "Request Helpers", "nextjs", "api-routes", 6),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing/api-routes/edge-api-routes", "Edge API Routes", "nextjs", "api-routes", 6),
            ],
            
            "middleware": [
                # Middleware
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/middleware", "Middleware", "nextjs", "middleware", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/middleware/middleware-matcher", "Middleware Matcher", "nextjs", "middleware", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/middleware/edge-runtime", "Edge Runtime", "nextjs", "middleware", 6),
            ],
            
            "authentication": [
                # Authentication
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication", "Authentication", "nextjs", "authentication", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication/nextauth", "NextAuth.js", "nextjs", "authentication", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication/with-auth0", "Auth0", "nextjs", "authentication", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication/with-clerk", "Clerk", "nextjs", "authentication", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication/with-firebase", "Firebase", "nextjs", "authentication", 6),
            ],
            
            "internationalization": [
                # Internationalization
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/internationalization", "Internationalization", "nextjs", "i18n", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/internationalization/getting-started", "i18n Getting Started", "nextjs", "i18n", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/internationalization/routing", "i18n Routing", "nextjs", "i18n", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/internationalization/advanced", "Advanced i18n", "nextjs", "i18n", 5),
            ],
            
            "deployment": [
                # Deployment
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying", "Deploying", "nextjs", "deployment", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying/static-exports", "Static Exports", "nextjs", "deployment", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying/docker", "Docker", "nextjs", "deployment", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying/vercel", "Vercel", "nextjs", "deployment", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying/other-hosting", "Other Hosting", "nextjs", "deployment", 6),
            ],
            
            "testing": [
                # Testing
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing", "Testing", "nextjs", "testing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing/jest", "Jest", "nextjs", "testing", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing/playwright", "Playwright", "nextjs", "testing", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing/cypress", "Cypress", "nextjs", "testing", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing/react-testing-library", "React Testing Library", "nextjs", "testing", 7),
            ],
            
            "upgrading": [
                # Upgrading
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/upgrading", "Upgrading", "nextjs", "migration", 6),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/upgrading/version-14", "Version 14", "nextjs", "migration", 5),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/upgrading/version-13", "Version 13", "nextjs", "migration", 5),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/upgrading/version-12", "Version 12", "nextjs", "migration", 4),
            ],
            
            "api_reference": [
                # API Reference
                DocumentationSource("https://nextjs.org/docs/app/api-reference", "API Reference", "nextjs", "api", 8),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/functions", "Functions", "nextjs", "api", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/components", "Components", "nextjs", "api", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/file-conventions", "File Conventions", "nextjs", "api", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "Config Reference", "nextjs", "api", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/edge-runtime", "Edge Runtime", "nextjs", "api", 6),
            ],
            
            "learn_tutorials": [
                # Learn and Tutorials
                DocumentationSource("https://nextjs.org/learn", "Next.js Learn", "nextjs", "tutorial", 9),
                DocumentationSource("https://nextjs.org/learn/dashboard-app", "Dashboard App Tutorial", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/about-nextjs", "About Next.js", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/from-javascript-to-react", "From JavaScript to React", "nextjs", "tutorial", 7),
                DocumentationSource("https://nextjs.org/learn/foundations/from-react-to-nextjs", "From React to Next.js", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/how-nextjs-works", "How Next.js Works", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/css-styling", "CSS Styling", "nextjs", "tutorial", 7),
                DocumentationSource("https://nextjs.org/learn/foundations/typescript", "TypeScript", "nextjs", "tutorial", 7),
            ],
            
            "examples": [
                # Examples
                DocumentationSource("https://nextjs.org/docs/examples", "Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/authentication", "Authentication Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/with-typescript", "TypeScript Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/with-tailwindcss", "Tailwind Examples", "nextjs", "examples", 6),
                DocumentationSource("https://nextjs.org/docs/examples/with-mongodb", "MongoDB Examples", "nextjs", "examples", 6),
                DocumentationSource("https://nextjs.org/docs/examples/with-postgres", "PostgreSQL Examples", "nextjs", "examples", 6),
                DocumentationSource("https://nextjs.org/docs/examples/with-prisma", "Prisma Examples", "nextjs", "examples", 6),
                DocumentationSource("https://nextjs.org/docs/examples/with-supabase", "Supabase Examples", "nextjs", "examples", 6),
            ],
            
            "ecosystem": [
                # Ecosystem
                DocumentationSource("https://vercel.com/docs/frameworks/nextjs", "Vercel Next.js Guide", "nextjs", "deployment", 8),
                DocumentationSource("https://vercel.com/docs/frameworks/nextjs/nextjs-config", "Vercel Config", "nextjs", "deployment", 7),
                DocumentationSource("https://tailwindcss.com/docs/guides/nextjs", "Tailwind with Next.js", "nextjs", "styling", 7),
                DocumentationSource("https://www.typescriptlang.org/docs/handbook/react.html", "TypeScript with React", "nextjs", "typescript", 7),
                DocumentationSource("https://www.prisma.io/docs/getting-started/setup-prisma/start-from-scratch/nextjs", "Prisma with Next.js", "nextjs", "database", 6),
                DocumentationSource("https://supabase.com/docs/guides/getting-started/quickstarts/nextjs", "Supabase with Next.js", "nextjs", "database", 6),
                DocumentationSource("https://next-auth.js.org/getting-started/example", "NextAuth.js", "nextjs", "authentication", 7),
                DocumentationSource("https://www.framer.com/motion/", "Framer Motion", "nextjs", "animation", 6),
                DocumentationSource("https://react-hook-form.com/get-started", "React Hook Form", "nextjs", "forms", 6),
                DocumentationSource("https://zustand-demo.pmnd.rs/", "Zustand", "nextjs", "state-management", 6),
            ]
        }
    
    async def process_all_topics(self) -> Dict[str, Any]:
        """Process all Next.js documentation topics comprehensively."""
        logger.info("🚀 Starting comprehensive Next.js documentation processing...")
        
        results = {
            'total_sources': 0,
            'total_rules': 0,
            'categories_processed': 0,
            'processing_time': 0,
            'topic_results': {},
            'errors': []
        }
        
        start_time = datetime.now()
        
        for topic_name, sources in self.nextjs_sources.items():
            logger.info(f"📚 Processing topic: {topic_name} ({len(sources)} sources)")
            
            try:
                topic_result = await self._process_topic(topic_name, sources)
                results['topic_results'][topic_name] = topic_result
                results['total_sources'] += topic_result['sources_processed']
                results['total_rules'] += topic_result['rules_generated']
                results['categories_processed'] += 1
                
                logger.info(f"✅ {topic_name}: {topic_result['sources_processed']} sources, {topic_result['rules_generated']} rules")
                
            except Exception as e:
                logger.error(f"❌ Error processing {topic_name}: {e}")
                results['errors'].append(f"{topic_name}: {str(e)}")
        
        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        # Generate comprehensive report
        await self._generate_comprehensive_report(results)
        
        return results
    
    async def _process_topic(
        self, 
        topic_name: str, 
        sources: List[DocumentationSource]
    ) -> Dict[str, Any]:
        """Process a specific topic with its sources."""
        topic_result = {
            'topic': topic_name,
            'sources_processed': 0,
            'rules_generated': 0,
            'categories': {},
            'quality_scores': [],
            'processing_time': 0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            # Use batch processor for efficient processing
            batch_result = await self.batch_processor.process_documentation_batch(
                sources, 
                formats=[RuleFormat.CURSOR]
            )
            
            # Process results
            topic_result['sources_processed'] = batch_result.sources_processed
            topic_result['rules_generated'] = batch_result.total_rules_generated
            topic_result['processing_time'] = batch_result.processing_time
            
            # Analyze categories and quality
            for cluster in batch_result.clusters:
                category = cluster.category
                if category not in topic_result['categories']:
                    topic_result['categories'][category] = 0
                topic_result['categories'][category] += len(cluster.rules)
                
                # Collect quality scores
                if hasattr(cluster, 'quality_score'):
                    topic_result['quality_scores'].append(cluster.quality_score)
            
            # Save topic-specific rules
            await self._save_topic_rules(topic_name, batch_result.clusters)
            
        except Exception as e:
            logger.error(f"Error processing topic {topic_name}: {e}")
            topic_result['errors'].append(str(e))
        
        return topic_result
    
    async def _save_topic_rules(
        self, 
        topic_name: str, 
        clusters: List[Any]
    ):
        """Save rules for a specific topic."""
        topic_dir = self.output_dir / topic_name
        topic_dir.mkdir(parents=True, exist_ok=True)
        
        # Group clusters by category
        category_clusters = {}
        for cluster in clusters:
            category = cluster.category
            if category not in category_clusters:
                category_clusters[category] = []
            category_clusters[category].append(cluster)
        
        # Generate rule files for each category
        for category, category_cluster_list in category_clusters.items():
            filename = f"{category}.mdc"
            filepath = topic_dir / filename
            
            # Generate consolidated content
            consolidated_content = self._generate_topic_category_content(
                topic_name, category, category_cluster_list
            )
            
            # Write to file
            filepath.write_text(consolidated_content, encoding='utf-8')
            logger.info(f"Generated topic rule: {filepath}")
    
    def _generate_topic_category_content(
        self, 
        topic_name: str, 
        category: str, 
        clusters: List[Any]
    ) -> str:
        """Generate consolidated content for a topic category."""
        # Generate frontmatter
        frontmatter = f"""---
description: Comprehensive {category} rules for Next.js {topic_name.replace('_', ' ').title()} development
globs: ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx", "**/next.config.*"]
alwaysApply: false
---

# Next.js {topic_name.replace('_', ' ').title()} - {category.title()} Rules

"""
        
        content = frontmatter
        
        # Add category overview
        content += f"## Overview\n\n"
        content += f"This rule set provides comprehensive guidelines for {category} in Next.js {topic_name.replace('_', ' ')} applications.\n\n"
        
        # Add all rules from clusters
        for cluster in clusters:
            content += f"### {cluster.category.title()} Guidelines\n\n"
            
            for rule in getattr(cluster, 'rules', []):
                content += f"#### {rule.title}\n\n"
                content += f"{rule.description}\n\n"
                
                if hasattr(rule, 'examples') and rule.examples:
                    content += "**Examples:**\n\n"
                    for example in rule.examples:
                        content += f"```typescript\n{example}\n```\n\n"
                
                content += "---\n\n"
        
        return content
    
    async def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive pipeline report."""
        report = {
            'pipeline_info': {
                'name': 'Comprehensive Next.js Documentation Pipeline',
                'version': '2.0.0',
                'timestamp': datetime.now().isoformat(),
                'ml_enabled': self.use_ml,
                'learning_enabled': self.enable_learning,
                'quality_threshold': self.quality_threshold
            },
            'processing_summary': {
                'total_topics': results['categories_processed'],
                'total_sources': results['total_sources'],
                'total_rules': results['total_rules'],
                'total_processing_time': results['processing_time']
            },
            'topic_breakdown': {},
            'quality_metrics': {
                'average_quality': 0.0,
                'quality_distribution': {}
            },
            'topic_results': results['topic_results'],
            'errors': results['errors']
        }
        
        # Calculate topic breakdown
        for topic_name, topic_result in results['topic_results'].items():
            report['topic_breakdown'][topic_name] = {
                'sources_processed': topic_result['sources_processed'],
                'rules_generated': topic_result['rules_generated'],
                'categories': topic_result['categories'],
                'processing_time': topic_result['processing_time']
            }
        
        # Calculate quality metrics
        all_quality_scores = []
        for topic_result in results['topic_results'].values():
            all_quality_scores.extend(topic_result.get('quality_scores', []))
        
        if all_quality_scores:
            report['quality_metrics']['average_quality'] = sum(all_quality_scores) / len(all_quality_scores)
        
        # Save report
        report_file = self.output_dir / "comprehensive_pipeline_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive pipeline report saved to {report_file}")
        
        # Print summary
        self._print_comprehensive_summary(report)
    
    def _print_comprehensive_summary(self, report: Dict[str, Any]):
        """Print comprehensive pipeline execution summary."""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE NEXT.JS DOCUMENTATION PIPELINE SUMMARY")
        print("="*80)
        
        summary = report['processing_summary']
        print(f"📊 Total Topics Processed: {summary['total_topics']}")
        print(f"📚 Total Sources Processed: {summary['total_sources']}")
        print(f"📝 Total Rules Generated: {summary['total_rules']}")
        print(f"⏱️  Total Processing Time: {summary['total_processing_time']:.2f}s")
        print(f"🎯 Average Quality Score: {report['quality_metrics']['average_quality']:.2f}")
        
        print(f"\n📁 Topics Processed:")
        for topic_name, topic_data in report['topic_breakdown'].items():
            print(f"   • {topic_name}: {topic_data['sources_processed']} sources, {topic_data['rules_generated']} rules")
        
        if report['errors']:
            print(f"\n❌ Errors ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"   • {error}")
        
        print(f"\n📂 Output Directory: {self.output_dir}")
        print("="*80)


async def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Next.js Documentation Pipeline"
    )
    
    parser.add_argument(
        '--all-topics',
        action='store_true',
        help='Process all Next.js documentation topics'
    )
    
    parser.add_argument(
        '--topics',
        nargs='+',
        help='Specific topics to process (e.g., routing_navigation styling_ui)'
    )
    
    parser.add_argument(
        '--output',
        default='.cursor/rules/nextjs',
        help='Output directory for generated rules'
    )
    
    parser.add_argument(
        '--ml-enhanced',
        action='store_true',
        help='Enable ML-enhanced processing'
    )
    
    parser.add_argument(
        '--learning-enabled',
        action='store_true',
        help='Enable learning system integration'
    )
    
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.7,
        help='Minimum quality threshold for rules'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = ComprehensiveNextJSPipeline(
        output_dir=args.output,
        use_ml=args.ml_enhanced,
        enable_learning=args.learning_enabled,
        quality_threshold=args.quality_threshold
    )
    
    # Process topics
    if args.all_topics:
        results = await pipeline.process_all_topics()
    elif args.topics:
        # Process specific topics
        results = {'topic_results': {}}
        for topic in args.topics:
            if topic in pipeline.nextjs_sources:
                topic_result = await pipeline._process_topic(topic, pipeline.nextjs_sources[topic])
                results['topic_results'][topic] = topic_result
            else:
                logger.warning(f"Unknown topic: {topic}")
    else:
        logger.error("Please specify --all-topics or --topics")
        return
    
    logger.info("🎉 Comprehensive Next.js documentation pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
