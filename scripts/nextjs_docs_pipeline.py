#!/usr/bin/env python3
"""
Next.js Documentation Import and Processing Pipeline

This script creates a comprehensive pipeline to:
1. Import Next.js documentation from multiple sources
2. Categorize content using intelligent taxonomy
3. Format into proper .cursor/rules format
4. Integrate with ML learning system for continuous improvement

Usage:
    python scripts/nextjs_docs_pipeline.py --help
    python scripts/nextjs_docs_pipeline.py --sources nextjs_official --output .cursor/rules/nextjs
    python scripts/nextjs_docs_pipeline.py --sources all --ml-enhanced --learning-enabled
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
from rules_maker.learning import IntegratedLearningSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NextJSDocsPipeline:
    """Comprehensive Next.js documentation processing pipeline."""
    
    def __init__(
        self,
        output_dir: str = ".cursor/rules/nextjs",
        use_ml: bool = True,
        enable_learning: bool = True,
        quality_threshold: float = 0.7
    ):
        """Initialize the Next.js docs pipeline.
        
        Args:
            output_dir: Output directory for generated cursor rules
            use_ml: Enable ML-enhanced processing
            enable_learning: Enable learning system integration
            quality_threshold: Minimum quality threshold for rules
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_ml = use_ml
        self.enable_learning = enable_learning
        self.quality_threshold = quality_threshold
        
        # Initialize components
        self._initialize_components()
        
        # Next.js documentation sources
        self.nextjs_sources = self._get_nextjs_sources()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Scraping configuration optimized for Next.js docs
        self.scraping_config = ScrapingConfig(
            max_pages=100,
            rate_limit=0.5,
            max_depth=3,
            user_agent="NextJS-Docs-Pipeline/1.0"
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
        
        # Initialize intelligence components
        try:
            self.category_engine = IntelligentCategoryEngine()
            self.semantic_analyzer = SemanticAnalyzer()
            logger.info("Intelligence components initialized")
        except Exception as e:
            logger.warning(f"Intelligence components not available: {e}")
            self.category_engine = None
            self.semantic_analyzer = None
        
        # Initialize learning system
        if self.enable_learning:
            try:
                self.learning_system = IntegratedLearningSystem({
                    'ml_weight': 0.7,
                    'enable_ml': self.use_ml,
                    'feedback_integration': True
                })
                logger.info("Learning system initialized")
            except Exception as e:
                logger.warning(f"Learning system not available: {e}")
                self.learning_system = None
        
        # Initialize batch processor for large-scale processing
        self.batch_processor = MLBatchProcessor(
            output_dir=str(self.output_dir / "batch_output"),
            quality_threshold=self.quality_threshold,
            max_concurrent=10
        )
        
    def _get_nextjs_sources(self) -> Dict[str, List[DocumentationSource]]:
        """Get comprehensive Next.js documentation sources organized by category."""
        return {
            "nextjs_official": [
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
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/appdir", "App Directory Config", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/compiler", "Compiler Options", "nextjs", "configuration", 7),
                DocumentationSource("https://nextjs.org/docs/app/api-reference/next-config-js/experimental", "Experimental Features", "nextjs", "configuration", 6),
                
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
            
            "nextjs_community": [
                DocumentationSource("https://nextjs.org/learn", "Next.js Learn", "nextjs", "tutorial", 9),
                DocumentationSource("https://nextjs.org/learn/dashboard-app", "Dashboard App Tutorial", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/about-nextjs", "About Next.js", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/from-javascript-to-react", "From JavaScript to React", "nextjs", "tutorial", 7),
                DocumentationSource("https://nextjs.org/learn/foundations/from-react-to-nextjs", "From React to Next.js", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/learn/foundations/how-nextjs-works", "How Next.js Works", "nextjs", "tutorial", 8),
                DocumentationSource("https://nextjs.org/docs/examples", "Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/authentication", "Authentication Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/with-typescript", "TypeScript Examples", "nextjs", "examples", 7),
                DocumentationSource("https://nextjs.org/docs/examples/with-tailwindcss", "Tailwind Examples", "nextjs", "examples", 6),
            ],
            
            "nextjs_ecosystem": [
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
    
    async def process_source_group(
        self, 
        source_group: str, 
        sources: List[DocumentationSource]
    ) -> Dict[str, Any]:
        """Process a group of documentation sources.
        
        Args:
            source_group: Name of the source group
            sources: List of documentation sources to process
            
        Returns:
            Processing results and statistics
        """
        logger.info(f"Processing {source_group} with {len(sources)} sources...")
        
        results = {
            'group': source_group,
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
            results['sources_processed'] = batch_result.sources_processed
            results['rules_generated'] = batch_result.total_rules_generated
            results['processing_time'] = batch_result.processing_time
            
            # Analyze categories and quality
            for cluster in batch_result.clusters:
                category = cluster.category
                if category not in results['categories']:
                    results['categories'][category] = 0
                results['categories'][category] += len(cluster.rules)
                
                # Collect quality scores
                if hasattr(cluster, 'quality_score'):
                    results['quality_scores'].append(cluster.quality_score)
            
            # Save individual rule files
            await self._save_individual_rules(source_group, batch_result.clusters)
            
            # Generate category-specific rule files
            await self._generate_category_rules(source_group, batch_result.clusters)
            
            logger.info(f"✅ {source_group} processing completed: {results['sources_processed']} sources, {results['rules_generated']} rules")
            
        except Exception as e:
            logger.error(f"❌ Error processing {source_group}: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def _save_individual_rules(
        self, 
        source_group: str, 
        clusters: List[Any]
    ):
        """Save individual rule files for each cluster."""
        individual_dir = self.output_dir / source_group / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        for i, cluster in enumerate(clusters):
            filename = f"{cluster.category}_{i+1}.mdc"
            filepath = individual_dir / filename
            
            # Generate rule content
            rule_content = self._generate_cluster_rule_content(cluster)
            
            # Write to file
            filepath.write_text(rule_content, encoding='utf-8')
            logger.debug(f"Saved individual rule: {filepath}")
    
    async def _generate_category_rules(
        self, 
        source_group: str, 
        clusters: List[Any]
    ):
        """Generate consolidated category-specific rule files."""
        category_dir = self.output_dir / source_group / "categories"
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Group clusters by category
        category_clusters = {}
        for cluster in clusters:
            category = cluster.category
            if category not in category_clusters:
                category_clusters[category] = []
            category_clusters[category].append(cluster)
        
        # Generate consolidated rule files for each category
        for category, category_cluster_list in category_clusters.items():
            filename = f"{category}.mdc"
            filepath = category_dir / filename
            
            # Generate consolidated content
            consolidated_content = self._generate_consolidated_category_content(
                category, category_cluster_list
            )
            
            # Write to file
            filepath.write_text(consolidated_content, encoding='utf-8')
            logger.info(f"Generated category rule: {filepath}")
    
    def _generate_cluster_rule_content(self, cluster: Any) -> str:
        """Generate rule content for a single cluster."""
        # Extract cluster information
        category = getattr(cluster, 'category', 'general')
        rules = getattr(cluster, 'rules', [])
        quality_score = getattr(cluster, 'quality_score', 0.0)
        
        # Generate frontmatter
        frontmatter = f"""---
description: {category.title()} rules for Next.js development
globs: ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx", "**/next.config.*"]
alwaysApply: false
---

# Next.js {category.title()} Rules

"""
        
        # Add quality assessment if available
        if quality_score > 0:
            frontmatter += f"<!-- Quality Score: {quality_score:.2f} -->\n\n"
        
        # Generate rule content
        content = frontmatter
        
        for rule in rules:
            content += f"## {rule.title}\n\n"
            content += f"{rule.description}\n\n"
            
            # Add examples if available
            if hasattr(rule, 'examples') and rule.examples:
                content += "### Examples\n\n"
                for example in rule.examples:
                    content += f"```typescript\n{example}\n```\n\n"
            
            content += "---\n\n"
        
        return content
    
    def _generate_consolidated_category_content(
        self, 
        category: str, 
        clusters: List[Any]
    ) -> str:
        """Generate consolidated content for a category."""
        # Generate frontmatter
        frontmatter = f"""---
description: Comprehensive {category} rules for Next.js development
globs: ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx", "**/next.config.*"]
alwaysApply: false
---

# Next.js {category.title()} Development Rules

"""
        
        content = frontmatter
        
        # Add category overview
        content += f"## Overview\n\n"
        content += f"This rule set provides comprehensive guidelines for {category} in Next.js applications.\n\n"
        
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
    
    async def run_learning_analysis(self, results: List[Dict[str, Any]]):
        """Run learning analysis on processing results."""
        if not self.learning_system:
            logger.info("Learning system not available, skipping analysis")
            return
        
        logger.info("Running learning analysis...")
        
        try:
            # Collect usage patterns from results
            usage_patterns = []
            for result in results:
                for category, count in result.get('categories', {}).items():
                    usage_patterns.append({
                        'category': category,
                        'usage_count': count,
                        'quality_scores': result.get('quality_scores', []),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Analyze patterns
            analysis = await self.learning_system.analyze_usage_patterns(usage_patterns)
            
            # Save learning insights
            insights_file = self.output_dir / "learning_insights.json"
            with open(insights_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Learning analysis completed, insights saved to {insights_file}")
            
        except Exception as e:
            logger.error(f"Learning analysis failed: {e}")
    
    async def generate_pipeline_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive pipeline report."""
        report = {
            'pipeline_info': {
                'name': 'Next.js Documentation Pipeline',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'ml_enabled': self.use_ml,
                'learning_enabled': self.enable_learning,
                'quality_threshold': self.quality_threshold
            },
            'processing_summary': {
                'total_groups': len(results),
                'total_sources': sum(r.get('sources_processed', 0) for r in results),
                'total_rules': sum(r.get('rules_generated', 0) for r in results),
                'total_processing_time': sum(r.get('processing_time', 0) for r in results)
            },
            'category_breakdown': {},
            'quality_metrics': {
                'average_quality': 0.0,
                'quality_distribution': {}
            },
            'results': results
        }
        
        # Calculate category breakdown
        for result in results:
            for category, count in result.get('categories', {}).items():
                if category not in report['category_breakdown']:
                    report['category_breakdown'][category] = 0
                report['category_breakdown'][category] += count
        
        # Calculate quality metrics
        all_quality_scores = []
        for result in results:
            all_quality_scores.extend(result.get('quality_scores', []))
        
        if all_quality_scores:
            report['quality_metrics']['average_quality'] = sum(all_quality_scores) / len(all_quality_scores)
        
        # Save report
        report_file = self.output_dir / "pipeline_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Pipeline report saved to {report_file}")
        
        # Print summary
        self._print_pipeline_summary(report)
    
    def _print_pipeline_summary(self, report: Dict[str, Any]):
        """Print pipeline execution summary."""
        print("\n" + "="*60)
        print("🚀 NEXT.JS DOCUMENTATION PIPELINE SUMMARY")
        print("="*60)
        
        summary = report['processing_summary']
        print(f"📊 Total Sources Processed: {summary['total_sources']}")
        print(f"📝 Total Rules Generated: {summary['total_rules']}")
        print(f"⏱️  Total Processing Time: {summary['total_processing_time']:.2f}s")
        print(f"🎯 Average Quality Score: {report['quality_metrics']['average_quality']:.2f}")
        
        print(f"\n📁 Categories Generated:")
        for category, count in report['category_breakdown'].items():
            print(f"   • {category}: {count} rules")
        
        print(f"\n📂 Output Directory: {self.output_dir}")
        print("="*60)


async def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(
        description="Next.js Documentation Import and Processing Pipeline"
    )
    
    parser.add_argument(
        '--sources',
        choices=['nextjs_official', 'nextjs_community', 'nextjs_ecosystem', 'all'],
        default='all',
        help='Source groups to process'
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
    pipeline = NextJSDocsPipeline(
        output_dir=args.output,
        use_ml=args.ml_enhanced,
        enable_learning=args.learning_enabled,
        quality_threshold=args.quality_threshold
    )
    
    # Determine which source groups to process
    if args.sources == 'all':
        source_groups = pipeline.nextjs_sources.keys()
    else:
        source_groups = [args.sources]
    
    # Process source groups
    results = []
    for source_group in source_groups:
        if source_group in pipeline.nextjs_sources:
            result = await pipeline.process_source_group(
                source_group, 
                pipeline.nextjs_sources[source_group]
            )
            results.append(result)
        else:
            logger.warning(f"Unknown source group: {source_group}")
    
    # Run learning analysis
    if args.learning_enabled:
        await pipeline.run_learning_analysis(results)
    
    # Generate pipeline report
    await pipeline.generate_pipeline_report(results)
    
    logger.info("🎉 Next.js documentation pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
