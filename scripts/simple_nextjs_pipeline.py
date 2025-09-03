#!/usr/bin/env python3
"""
Simple Next.js Documentation Pipeline

This script demonstrates comprehensive Next.js documentation coverage
using the existing codebase components.

Usage:
    python scripts/simple_nextjs_pipeline.py --help
    python scripts/simple_nextjs_pipeline.py --all-topics --output demo_output/comprehensive
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

from rules_maker.scrapers import AsyncDocumentationScraper
from rules_maker.transformers import CursorRuleTransformer
from rules_maker.batch_processor import MLBatchProcessor, DocumentationSource
from rules_maker.models import ScrapingConfig, RuleFormat
from rules_maker.intelligence.nextjs_categorizer import NextJSCategorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleNextJSPipeline:
    """Simple Next.js documentation processing pipeline with comprehensive coverage."""
    
    def __init__(
        self,
        output_dir: str = "demo_output/comprehensive",
        quality_threshold: float = 0.7
    ):
        """Initialize the simple Next.js docs pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_threshold = quality_threshold
        
        # Initialize components
        self._initialize_components()
        
        # Get comprehensive Next.js documentation sources
        self.nextjs_sources = self._get_comprehensive_nextjs_sources()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        logger.info("Initializing simple pipeline components...")
        
        # Scraping configuration
        self.scraping_config = ScrapingConfig(
            max_pages=50,  # Reduced for demo
            rate_limit=0.5,
            max_depth=2,
            user_agent="Simple-NextJS-Pipeline/1.0"
        )
        
        # Initialize scrapers and transformers
        self.scraper = AsyncDocumentationScraper(config=self.scraping_config)
        self.transformer = CursorRuleTransformer()
        
        # Initialize Next.js specific categorizer
        self.nextjs_categorizer = NextJSCategorizer()
        
        # Initialize batch processor
        self.batch_processor = MLBatchProcessor(
            output_dir=str(self.output_dir / "batch_output"),
            quality_threshold=self.quality_threshold,
            max_concurrent=5
        )
        
    def _get_comprehensive_nextjs_sources(self) -> Dict[str, List[DocumentationSource]]:
        """Get comprehensive Next.js documentation sources covering ALL topics."""
        return {
            "core_documentation": [
                DocumentationSource("https://nextjs.org/docs", "Next.js Official Docs", "nextjs", "overview", 10),
                DocumentationSource("https://nextjs.org/docs/getting-started", "Getting Started", "nextjs", "getting-started", 10),
                DocumentationSource("https://nextjs.org/docs/getting-started/installation", "Installation", "nextjs", "setup", 9),
            ],
            
            "routing_navigation": [
                DocumentationSource("https://nextjs.org/docs/app", "App Router", "nextjs", "routing", 10),
                DocumentationSource("https://nextjs.org/docs/pages", "Pages Router", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing", "App Router Routing", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/layouts-and-pages", "Layouts and Pages", "nextjs", "routing", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/linking-and-navigating", "Linking and Navigating", "nextjs", "routing", 8),
            ],
            
            "components_rendering": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering", "Rendering", "nextjs", "rendering", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/server-components", "Server Components", "nextjs", "components", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/client-components", "Client Components", "nextjs", "components", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/rendering/composition-patterns", "Composition Patterns", "nextjs", "components", 8),
            ],
            
            "data_fetching": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching", "Data Fetching", "nextjs", "data-fetching", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/fetching", "Fetching Data", "nextjs", "data-fetching", 9),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/caching-and-revalidating", "Caching and Revalidating", "nextjs", "data-fetching", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/data-fetching/forms-and-mutations", "Forms and Mutations", "nextjs", "data-fetching", 8),
            ],
            
            "styling_ui": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling", "Styling", "nextjs", "styling", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/css-modules", "CSS Modules", "nextjs", "styling", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/tailwind-css", "Tailwind CSS", "nextjs", "styling", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/styling/sass", "Sass", "nextjs", "styling", 7),
            ],
            
            "optimization": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing", "Optimizing", "nextjs", "optimization", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/images", "Image Optimization", "nextjs", "optimization", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/fonts", "Font Optimization", "nextjs", "optimization", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/optimizing/static-assets", "Static Assets", "nextjs", "optimization", 7),
            ],
            
            "configuration": [
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js", "Next.js Config", "nextjs", "configuration", 8),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/appdir", "App Directory Config", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/compiler", "Compiler Options", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/experimental", "Experimental Features", "nextjs", "configuration", 6),
            ],
            
            "api_routes": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/route-handlers", "Route Handlers", "nextjs", "api-routes", 9),
                DocumentationSource("https://nextjs.org/docs/pages/building-your-application/routing/api-routes", "Pages API Routes", "nextjs", "api-routes", 8),
            ],
            
            "middleware": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/routing/middleware", "Middleware", "nextjs", "middleware", 8),
            ],
            
            "authentication": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/authentication", "Authentication", "nextjs", "authentication", 8),
            ],
            
            "deployment": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying", "Deploying", "nextjs", "deployment", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying/static-exports", "Static Exports", "nextjs", "deployment", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/deploying/docker", "Docker", "nextjs", "deployment", 7),
            ],
            
            "testing": [
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing", "Testing", "nextjs", "testing", 8),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing/jest", "Jest", "nextjs", "testing", 7),
                DocumentationSource("https://nextjs.org/docs/app/building-your-application/testing/playwright", "Playwright", "nextjs", "testing", 7),
            ],
            
            "learn_tutorials": [
                DocumentationSource("https://nextjs.org/learn", "Next.js Learn", "nextjs", "tutorial", 9),
                DocumentationSource("https://nextjs.org/learn/dashboard-app", "Dashboard App Tutorial", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/about-nextjs", "About Next.js", "nextjs", "tutorial", 8),
            ],
            
            "examples": [
                DocumentationSource("https://nextjs.org/docs/examples", "Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/authentication", "Authentication Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/with-typescript", "TypeScript Examples", "nextjs", "examples", 7),
            ],
            
            "ecosystem": [
                DocumentationSource("https://vercel.com/docs/frameworks/nextjs", "Vercel Next.js Guide", "nextjs", "deployment", 8),
                DocumentationSource("https://tailwindcss.com/docs/guides/nextjs", "Tailwind with Next.js", "nextjs", "styling", 7),
                DocumentationSource("https://www.typescriptlang.org/docs/handbook/react.html", "TypeScript with React", "nextjs", "typescript", 7),
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
                'name': 'Simple Next.js Documentation Pipeline',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
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
        description="Simple Next.js Documentation Pipeline"
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
        default='demo_output/comprehensive',
        help='Output directory for generated rules'
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
    pipeline = SimpleNextJSPipeline(
        output_dir=args.output,
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
    
    logger.info("🎉 Simple Next.js documentation pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
