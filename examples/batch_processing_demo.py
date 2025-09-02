#!/usr/bin/env python3
"""
Comprehensive demonstration of ML-powered batch documentation processing.

This example shows how to use the Rules Maker system to:
1. Process 100+ documentation sources in parallel
2. Generate intelligent rule clusters using ML algorithms
3. Apply self-improving feedback loops for quality optimization
4. Export optimized rule sets for Cursor and Windsurf

Usage:
    PYTHONPATH=src python examples/batch_processing_demo.py [--bedrock] [--demo-mode]
"""

import asyncio
import logging
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rules_maker.batch_processor import (
    MLBatchProcessor, 
    DocumentationSource,
    process_popular_frameworks,
    process_cloud_platforms
)
from rules_maker.learning.self_improving_engine import SelfImprovingEngine
from rules_maker.models import RuleFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_comprehensive_batch_processing():
    """Demonstrate comprehensive batch processing with 100+ sources."""
    logger.info("🚀 Starting comprehensive batch processing demo")
    
    # Create comprehensive documentation sources
    sources = create_comprehensive_source_list()
    
    # Initialize processor with self-improving engine
    bedrock_config = {
        'model_id': 'amazon.nova-lite-v1:0',
        'region': 'us-east-1',
        'temperature': 0.3,
        'max_tokens': 2000
    } if '--bedrock' in sys.argv else None
    
    processor = MLBatchProcessor(
        bedrock_config=bedrock_config,
        output_dir="rules/comprehensive_demo",
        quality_threshold=0.65,
        max_concurrent=12
    )
    
    # Process all sources
    start_time = time.time()
    result = await processor.process_documentation_batch(
        sources,
        formats=[RuleFormat.CURSOR, RuleFormat.WINDSURF]
    )
    processing_time = time.time() - start_time
    
    # Display comprehensive results
    print_comprehensive_results(result, processing_time)
    
    # Demonstrate self-improving feedback loop
    await demo_self_improving_feedback_loop(result.clusters)
    
    return result


async def demo_framework_specific_processing():
    """Demonstrate framework-specific processing."""
    logger.info("🔧 Processing popular frameworks")
    
    bedrock_config = {
        'model_id': 'amazon.nova-lite-v1:0',
        'region': 'us-east-1'
    } if '--bedrock' in sys.argv else None
    
    # Process popular frameworks
    framework_result = await process_popular_frameworks(
        output_dir="rules/frameworks_demo",
        bedrock_config=bedrock_config
    )
    
    print("\n" + "="*80)
    print("📊 FRAMEWORK PROCESSING RESULTS")
    print("="*80)
    print(f"Sources processed: {framework_result.sources_processed}")
    print(f"Total rules generated: {framework_result.total_rules_generated}")
    print(f"Processing time: {framework_result.processing_time:.2f}s")
    print(f"Quality metrics: {framework_result.quality_metrics}")
    
    return framework_result


async def demo_cloud_platform_processing():
    """Demonstrate cloud platform processing."""
    logger.info("☁️ Processing cloud platforms")
    
    bedrock_config = {
        'model_id': 'amazon.nova-lite-v1:0',
        'region': 'us-east-1'
    } if '--bedrock' in sys.argv else None
    
    # Process cloud platforms
    cloud_result = await process_cloud_platforms(
        output_dir="rules/cloud_demo",
        bedrock_config=bedrock_config
    )
    
    print("\n" + "="*80)
    print("☁️ CLOUD PLATFORM PROCESSING RESULTS")
    print("="*80)
    print(f"Sources processed: {cloud_result.sources_processed}")
    print(f"Total rules generated: {cloud_result.total_rules_generated}")
    print(f"Processing time: {cloud_result.processing_time:.2f}s")
    print(f"Quality metrics: {cloud_result.quality_metrics}")
    
    return cloud_result


async def demo_self_improving_feedback_loop(clusters):
    """Demonstrate the self-improving feedback loop system."""
    logger.info("🧠 Demonstrating self-improving feedback loop")
    
    engine = SelfImprovingEngine(
        feedback_window_hours=24,
        min_feedback_signals=3,
        quality_threshold=0.6
    )
    
    # Simulate feedback signals for demonstration
    for cluster in clusters[:5]:  # Demo with first 5 clusters
        for rule in cluster.rules[:2]:  # Demo with first 2 rules per cluster
            rule_id = rule['id']
            
            # Simulate various feedback signals
            await engine.collect_feedback_signal(
                rule_id=rule_id,
                signal_type="usage_success",
                value=0.8,
                context={"usage_type": "cursor_ide"},
                source="user"
            )
            
            await engine.collect_feedback_signal(
                rule_id=rule_id,
                signal_type="quality_score",
                value=0.75,
                context={"evaluator": "ml_model"},
                source="system"
            )
            
            await engine.collect_feedback_signal(
                rule_id=rule_id,
                signal_type="user_rating",
                value=0.7,
                context={"rating_context": "rule_helpfulness"},
                source="user"
            )
    
    print("\n" + "="*80)
    print("🧠 SELF-IMPROVING FEEDBACK DEMONSTRATION")
    print("="*80)
    print(f"Total feedback signals collected: {len(engine.feedback_signals)}")
    print(f"Adaptive thresholds: {engine.adaptive_thresholds}")
    
    # Demonstrate quality predictions (would work better with more training data)
    if clusters and clusters[0].rules:
        sample_rule_data = clusters[0].rules[0]
        print(f"\nSample feedback analysis for rule: {sample_rule_data['id']}")
        print(f"Initial quality score: {sample_rule_data['quality_score']:.3f}")
    
    # Save engine state
    await engine.save_state("demo_engine_state.json")
    print("💾 Saved self-improving engine state for future use")


def create_comprehensive_source_list() -> List[DocumentationSource]:
    """Create a comprehensive list of 100+ documentation sources with updated URLs."""
    
    # Import updated sources
    try:
        from rules_maker.sources.updated_documentation_sources import get_comprehensive_updated_sources
        return get_comprehensive_updated_sources()
    except ImportError:
        # Fallback to updated URLs inline
        web_frameworks = [
            DocumentationSource("https://react.dev/learn", "React", "javascript", "react", priority=5),
            DocumentationSource("https://vuejs.org/guide/", "Vue.js", "javascript", "vue", priority=5),
            DocumentationSource("https://angular.io/docs", "Angular", "javascript", "angular", priority=4),
            DocumentationSource("https://nextjs.org/docs", "Next.js", "javascript", "nextjs", priority=4),
            DocumentationSource("https://nuxt.com/docs", "Nuxt.js", "javascript", "nuxtjs", priority=3),
            DocumentationSource("https://svelte.dev/docs/introduction", "Svelte", "javascript", "svelte", priority=3),
            DocumentationSource("https://kit.svelte.dev/docs/introduction", "SvelteKit", "javascript", "sveltekit", priority=3),
            DocumentationSource("https://remix.run/docs/en/main", "Remix", "javascript", "remix", priority=3),
            DocumentationSource("https://expressjs.com/en/4x/api.html", "Express.js", "javascript", "express", priority=3),
            DocumentationSource("https://www.fastify.io/docs/latest/", "Fastify", "javascript", "fastify", priority=2),
        ]
    
    python_frameworks = [
        DocumentationSource("https://fastapi.tiangolo.com/", "FastAPI", "python", "fastapi", priority=5),
        DocumentationSource("https://flask.palletsprojects.com/", "Flask", "python", "flask", priority=4),
        DocumentationSource("https://docs.djangoproject.com/", "Django", "python", "django", priority=4),
        DocumentationSource("https://pydantic-docs.helpmanual.io/", "Pydantic", "python", "pydantic", priority=4),
        DocumentationSource("https://docs.sqlalchemy.org/", "SQLAlchemy", "python", "sqlalchemy", priority=3),
        DocumentationSource("https://docs.celeryproject.org/", "Celery", "python", "celery", priority=3),
        DocumentationSource("https://docs.pytest.org/", "Pytest", "python", "pytest", priority=4),
        DocumentationSource("https://click.palletsprojects.com/", "Click", "python", "click", priority=2),
        DocumentationSource("https://requests.readthedocs.io/", "Requests", "python", "requests", priority=3),
        DocumentationSource("https://httpx.python-requests.org/", "HTTPX", "python", "httpx", priority=2),
    ]
    
    backend_frameworks = [
        DocumentationSource("https://spring.io/guides", "Spring Boot", "java", "spring", priority=4),
        DocumentationSource("https://docs.spring.io/spring-framework/docs/current/reference/html/", "Spring Framework", "java", "spring", priority=3),
        DocumentationSource("https://quarkus.io/guides/", "Quarkus", "java", "quarkus", priority=3),
        DocumentationSource("https://micronaut.io/docs/", "Micronaut", "java", "micronaut", priority=2),
        DocumentationSource("https://rubyonrails.org/guides", "Ruby on Rails", "ruby", "rails", priority=3),
        DocumentationSource("https://laravel.com/docs", "Laravel", "php", "laravel", priority=3),
        DocumentationSource("https://symfony.com/doc/", "Symfony", "php", "symfony", priority=2),
        DocumentationSource("https://docs.microsoft.com/en-us/aspnet/core/", "ASP.NET Core", "csharp", "dotnet", priority=3),
        DocumentationSource("https://go.dev/doc/", "Go Documentation", "go", framework=None, priority=4),
        DocumentationSource("https://doc.rust-lang.org/book/", "Rust Book", "rust", framework=None, priority=3),
    ]
    
    cloud_platforms = [
        DocumentationSource("https://docs.aws.amazon.com/", "AWS", "cloud", "aws", priority=5),
        DocumentationSource("https://docs.microsoft.com/en-us/azure/", "Azure", "cloud", "azure", priority=4),
        DocumentationSource("https://cloud.google.com/docs", "Google Cloud", "cloud", "gcp", priority=4),
        DocumentationSource("https://kubernetes.io/docs/", "Kubernetes", "cloud", "kubernetes", priority=5),
        DocumentationSource("https://docs.docker.com/", "Docker", "cloud", "docker", priority=4),
        DocumentationSource("https://www.terraform.io/docs", "Terraform", "cloud", "terraform", priority=4),
        DocumentationSource("https://docs.ansible.com/", "Ansible", "cloud", "ansible", priority=3),
        DocumentationSource("https://helm.sh/docs/", "Helm", "cloud", "helm", priority=3),
        DocumentationSource("https://istio.io/docs/", "Istio", "cloud", "istio", priority=2),
        DocumentationSource("https://prometheus.io/docs/", "Prometheus", "cloud", "prometheus", priority=3),
    ]
    
    databases = [
        DocumentationSource("https://docs.mongodb.com/", "MongoDB", "database", "mongodb", priority=4),
        DocumentationSource("https://www.postgresql.org/docs/", "PostgreSQL", "database", "postgresql", priority=4),
        DocumentationSource("https://dev.mysql.com/doc/", "MySQL", "database", "mysql", priority=3),
        DocumentationSource("https://redis.io/documentation", "Redis", "database", "redis", priority=4),
        DocumentationSource("https://www.elastic.co/guide/", "Elasticsearch", "database", "elasticsearch", priority=3),
        DocumentationSource("https://cassandra.apache.org/doc/", "Apache Cassandra", "database", "cassandra", priority=2),
        DocumentationSource("https://neo4j.com/docs/", "Neo4j", "database", "neo4j", priority=2),
        DocumentationSource("https://docs.influxdata.com/", "InfluxDB", "database", "influxdb", priority=2),
    ]
    
    ml_ai_tools = [
        DocumentationSource("https://pytorch.org/docs/stable/", "PyTorch", "python", "pytorch", priority=4),
        DocumentationSource("https://www.tensorflow.org/guide", "TensorFlow", "python", "tensorflow", priority=4),
        DocumentationSource("https://scikit-learn.org/stable/user_guide.html", "Scikit-learn", "python", "sklearn", priority=3),
        DocumentationSource("https://docs.huggingface.co/transformers/", "Hugging Face Transformers", "python", "transformers", priority=3),
        DocumentationSource("https://pandas.pydata.org/docs/", "Pandas", "python", "pandas", priority=4),
        DocumentationSource("https://numpy.org/doc/", "NumPy", "python", "numpy", priority=3),
        DocumentationSource("https://matplotlib.org/stable/", "Matplotlib", "python", "matplotlib", priority=2),
        DocumentationSource("https://docs.opencv.org/", "OpenCV", "python", "opencv", priority=2),
    ]
    
    devtools = [
        DocumentationSource("https://docs.github.com/", "GitHub", "devtools", "github", priority=4),
        DocumentationSource("https://docs.gitlab.com/", "GitLab", "devtools", "gitlab", priority=3),
        DocumentationSource("https://docs.docker.com/compose/", "Docker Compose", "devtools", "docker-compose", priority=3),
        DocumentationSource("https://docs.nginx.com/", "Nginx", "devtools", "nginx", priority=3),
        DocumentationSource("https://httpd.apache.org/docs/", "Apache HTTP Server", "devtools", "apache", priority=2),
        DocumentationSource("https://www.jenkins.io/doc/", "Jenkins", "devtools", "jenkins", priority=3),
        DocumentationSource("https://docs.sonarqube.org/", "SonarQube", "devtools", "sonarqube", priority=2),
        DocumentationSource("https://jestjs.io/docs/", "Jest", "javascript", "jest", priority=3),
        DocumentationSource("https://mochajs.org/", "Mocha", "javascript", "mocha", priority=2),
        DocumentationSource("https://www.cypress.io/guides/", "Cypress", "javascript", "cypress", priority=3),
    ]
    
    mobile_frameworks = [
        DocumentationSource("https://reactnative.dev/docs/", "React Native", "javascript", "react-native", priority=4),
        DocumentationSource("https://flutter.dev/docs", "Flutter", "dart", "flutter", priority=4),
        DocumentationSource("https://ionicframework.com/docs", "Ionic", "javascript", "ionic", priority=3),
        DocumentationSource("https://docs.expo.dev/", "Expo", "javascript", "expo", priority=3),
        DocumentationSource("https://developer.android.com/docs", "Android", "kotlin", "android", priority=4),
        DocumentationSource("https://developer.apple.com/documentation/", "iOS", "swift", "ios", priority=3),
    ]
    
    # Combine all sources
    all_sources = (
        web_frameworks + python_frameworks + backend_frameworks + 
        cloud_platforms + databases + ml_ai_tools + devtools + mobile_frameworks
    )
    
    # Add some additional specialized sources to reach 100+
    additional_sources = [
        DocumentationSource("https://www.electronjs.org/docs", "Electron", "javascript", "electron", priority=2),
        DocumentationSource("https://tauri.app/v1/guides/", "Tauri", "rust", "tauri", priority=2),
        DocumentationSource("https://docs.solidjs.com/", "Solid.js", "javascript", "solid", priority=2),
        DocumentationSource("https://qwik.builder.io/docs/", "Qwik", "javascript", "qwik", priority=2),
        DocumentationSource("https://lit.dev/docs/", "Lit", "javascript", "lit", priority=2),
        DocumentationSource("https://stenciljs.com/docs/", "Stencil", "javascript", "stencil", priority=1),
        DocumentationSource("https://docs.astro.build/", "Astro", "javascript", "astro", priority=3),
        DocumentationSource("https://vitejs.dev/guide/", "Vite", "javascript", "vite", priority=3),
        DocumentationSource("https://webpack.js.org/guides/", "Webpack", "javascript", "webpack", priority=2),
        DocumentationSource("https://parceljs.org/docs/", "Parcel", "javascript", "parcel", priority=1),
        DocumentationSource("https://rollupjs.org/guide/", "Rollup", "javascript", "rollup", priority=2),
        DocumentationSource("https://esbuild.github.io/", "esbuild", "javascript", "esbuild", priority=2),
    ]
    
    all_sources.extend(additional_sources)
    
    logger.info(f"📚 Created comprehensive source list with {len(all_sources)} documentation sources")
    return all_sources


def print_comprehensive_results(result, processing_time):
    """Print comprehensive batch processing results."""
    print("\n" + "="*100)
    print("🎉 COMPREHENSIVE BATCH PROCESSING RESULTS")
    print("="*100)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   • Sources processed: {result.sources_processed}")
    print(f"   • Total rules generated: {result.total_rules_generated}")
    print(f"   • Processing time: {processing_time:.2f} seconds")
    print(f"   • Average rules per source: {result.total_rules_generated / max(result.sources_processed, 1):.1f}")
    print(f"   • Failed sources: {len(result.failed_sources)}")
    
    print(f"\n🧠 QUALITY METRICS:")
    for metric, value in result.quality_metrics.items():
        print(f"   • {metric.replace('_', ' ').title()}: {value:.3f}")
    
    print(f"\n🏆 TOP PERFORMING CLUSTERS:")
    top_clusters = sorted(result.clusters, key=lambda c: c.coherence_score, reverse=True)[:5]
    for i, cluster in enumerate(top_clusters, 1):
        print(f"   {i}. {cluster.name} (Score: {cluster.coherence_score:.3f}, Rules: {len(cluster.rules)})")
    
    print(f"\n🔍 TECHNOLOGY DISTRIBUTION:")
    tech_counts = {}
    for cluster in result.clusters:
        tech_counts[cluster.technology] = tech_counts.get(cluster.technology, 0) + len(cluster.rules)
    
    sorted_techs = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for tech, count in sorted_techs:
        print(f"   • {tech.title()}: {count} rules")
    
    if result.insights:
        print(f"\n💡 KEY INSIGHTS:")
        for insight in result.insights.get('recommendations', [])[:3]:
            print(f"   • {insight}")
    
    if result.failed_sources:
        print(f"\n❌ FAILED SOURCES ({len(result.failed_sources)}):")
        for url in result.failed_sources[:5]:  # Show first 5
            print(f"   • {url}")
        if len(result.failed_sources) > 5:
            print(f"   ... and {len(result.failed_sources) - 5} more")
    
    print("\n" + "="*100)


async def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="ML-Powered Batch Documentation Processing Demo")
    parser.add_argument('--bedrock', action='store_true', help='Use AWS Bedrock for enhanced rule generation')
    parser.add_argument('--demo-mode', choices=['comprehensive', 'frameworks', 'cloud', 'all'], 
                       default='all', help='Demo mode to run')
    
    args = parser.parse_args()
    
    print("🚀 RULES MAKER - ML-POWERED BATCH PROCESSING DEMO")
    print("="*60)
    
    if args.bedrock:
        print("🤖 Using AWS Bedrock for enhanced rule generation")
    else:
        print("🔧 Using standard transformers (add --bedrock for enhanced generation)")
    
    print(f"📋 Demo mode: {args.demo_mode}")
    print()
    
    try:
        if args.demo_mode == 'comprehensive':
            await demo_comprehensive_batch_processing()
        elif args.demo_mode == 'frameworks':
            await demo_framework_specific_processing()
        elif args.demo_mode == 'cloud':
            await demo_cloud_platform_processing()
        else:  # all
            # Run all demos
            await demo_framework_specific_processing()
            await demo_cloud_platform_processing()
            await demo_comprehensive_batch_processing()
        
        print("\n✅ Demo completed successfully!")
        print("\n📁 Check the 'rules/' directory for generated rule sets")
        print("💾 Engine state saved to 'demo_engine_state.json'")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo failed with exception")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())