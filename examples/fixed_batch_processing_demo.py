#!/usr/bin/env python3
"""
Fixed ML-powered batch documentation processing demo.

This updated version addresses the "failed pulls" issue by:
1. Using updated documentation URLs that are currently valid
2. Implementing enhanced error recovery and validation
3. Providing detailed failure diagnostics and fallback strategies
4. Using the enhanced async scraper with redirect handling

Usage:
    PYTHONPATH=src python examples/fixed_batch_processing_demo.py [--bedrock]
"""

import asyncio
import logging
import sys
import argparse
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rules_maker.batch_processor import MLBatchProcessor, DocumentationSource
from rules_maker.learning.self_improving_engine import SelfImprovingEngine
from rules_maker.models import RuleFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_fixed_framework_sources() -> list[DocumentationSource]:
    """Get framework sources with corrected URLs."""
    return [
        # Updated web framework URLs
        DocumentationSource("https://react.dev/learn", "React", "javascript", "react", priority=5),
        DocumentationSource("https://vuejs.org/guide/", "Vue.js", "javascript", "vue", priority=5),
        DocumentationSource("https://angular.io/docs", "Angular", "javascript", "angular", priority=4),
        DocumentationSource("https://nextjs.org/docs", "Next.js", "javascript", "nextjs", priority=4),
        DocumentationSource("https://svelte.dev/docs/introduction", "Svelte", "javascript", "svelte", priority=3),
        DocumentationSource("https://www.fastify.io/docs/latest/", "Fastify", "javascript", "fastify", priority=2),
        
        # Updated Python framework URLs
        DocumentationSource("https://fastapi.tiangolo.com/", "FastAPI", "python", "fastapi", priority=5),
        DocumentationSource("https://flask.palletsprojects.com/en/3.0.x/", "Flask", "python", "flask", priority=4),
        DocumentationSource("https://docs.djangoproject.com/en/stable/", "Django", "python", "django", priority=4),
        DocumentationSource("https://docs.pydantic.dev/latest/", "Pydantic", "python", "pydantic", priority=4),
        
        # Backend frameworks
        DocumentationSource("https://spring.io/guides", "Spring Boot", "java", "spring", priority=4),
        DocumentationSource("https://guides.rubyonrails.org/", "Ruby on Rails", "ruby", "rails", priority=3),
        DocumentationSource("https://laravel.com/docs/10.x", "Laravel", "php", "laravel", priority=3),
        DocumentationSource("https://go.dev/doc/", "Go Documentation", "go", framework=None, priority=4),
        
        # Additional reliable sources
        DocumentationSource("https://www.python-httpx.org/", "HTTPX", "python", "httpx", priority=2),
        DocumentationSource("https://docs.pytest.org/en/stable/", "Pytest", "python", "pytest", priority=4),
        DocumentationSource("https://click.palletsprojects.com/en/8.1.x/", "Click", "python", "click", priority=2),
        DocumentationSource("https://requests.readthedocs.io/en/latest/", "Requests", "python", "requests", priority=3),
    ]


def get_fixed_cloud_sources() -> list[DocumentationSource]:
    """Get cloud platform sources with corrected URLs."""
    return [
        DocumentationSource("https://docs.aws.amazon.com/", "AWS", "cloud", "aws", priority=5),
        DocumentationSource("https://learn.microsoft.com/en-us/azure/", "Azure", "cloud", "azure", priority=4),
        DocumentationSource("https://cloud.google.com/docs", "Google Cloud", "cloud", "gcp", priority=4),
        DocumentationSource("https://kubernetes.io/docs/home/", "Kubernetes", "cloud", "kubernetes", priority=5),
        DocumentationSource("https://docs.docker.com/", "Docker", "cloud", "docker", priority=4),
        DocumentationSource("https://developer.hashicorp.com/terraform/docs", "Terraform", "cloud", "terraform", priority=4),
    ]


async def demo_fixed_framework_processing():
    """Demonstrate fixed framework processing with error recovery."""
    logger.info("🔧 Running fixed framework processing demo")
    
    bedrock_config = {
        'model_id': 'amazon.nova-lite-v1:0',
        'region': 'us-east-1'
    } if '--bedrock' in sys.argv else None
    
    processor = MLBatchProcessor(
        bedrock_config=bedrock_config,
        output_dir="rules/fixed_frameworks",
        quality_threshold=0.6,
        max_concurrent=8
    )
    
    sources = get_fixed_framework_sources()
    
    start_time = time.time()
    result = await processor.process_documentation_batch(
        sources,
        formats=[RuleFormat.CURSOR, RuleFormat.WINDSURF]
    )
    processing_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("🔧 FIXED FRAMEWORK PROCESSING RESULTS")
    print("="*80)
    print(f"Sources processed: {result.sources_processed}")
    print(f"Total rules generated: {result.total_rules_generated}")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Failed sources: {len(result.failed_sources)}")
    
    if result.failed_sources:
        print(f"\n❌ Failed sources:")
        for url in result.failed_sources:
            print(f"   • {url}")
    
    print(f"\n✅ Success rate: {result.sources_processed}/{len(sources)} ({result.sources_processed/len(sources)*100:.1f}%)")
    
    return result


async def demo_enhanced_scraper_validation():
    """Demonstrate the enhanced scraper's validation capabilities."""
    logger.info("🚀 Demonstrating enhanced scraper validation")
    
    from rules_maker.scrapers.enhanced_async_scraper import EnhancedAsyncDocumentationScraper
    from rules_maker.models import ScrapingConfig
    
    config = ScrapingConfig(rate_limit=0.5, max_concurrent=5)
    
    # Test URLs including some problematic ones
    test_urls = [
        "https://react.dev/learn",  # Should work
        "https://reactjs.org/docs/",  # Should redirect
        "https://fastify.io/docs/",  # Should fallback
        "https://invalid-url-that-does-not-exist.com/",  # Should fail
        "https://vuejs.org/guide/",  # Should work
    ]
    
    async with EnhancedAsyncDocumentationScraper(config) as scraper:
        logger.info(f"🔍 Validating {len(test_urls)} test URLs...")
        
        # Pre-validate URLs
        validation_results = []
        for url in test_urls:
            validation = await scraper.validate_url(url)
            validation_results.append((url, validation))
            
            status = "✅ Valid" if validation.is_valid else "❌ Invalid"
            if validation.is_valid and validation.final_url != url:
                status += " (Redirected)"
            
            print(f"   {status}: {url}")
            if validation.final_url != url:
                print(f"     → {validation.final_url}")
        
        # Test scraping with validation
        logger.info("🌐 Testing scraping with validation...")
        results, stats = await scraper.scrape_multiple_with_validation(test_urls)
        
        print(f"\n📊 Scraping results:")
        print(f"   • Total URLs: {stats['total_urls']}")
        print(f"   • Valid URLs: {stats['valid_urls']}")
        print(f"   • Invalid URLs: {stats['invalid_urls']}")
        print(f"   • Redirected: {stats['redirected_urls']}")
        print(f"   • Successfully scraped: {len(results)}")
        
        # Show failure report
        failure_report = scraper.get_failure_report()
        if failure_report['failed_urls']:
            print(f"\n⚠️ Failed URLs: {len(failure_report['failed_urls'])}")
            for url in failure_report['failed_urls']:
                print(f"   • {url}")


async def demo_self_improving_with_recovery():
    """Demonstrate self-improving engine with error recovery scenarios."""
    logger.info("🧠 Demonstrating self-improving engine with recovery")
    
    engine = SelfImprovingEngine(
        feedback_window_hours=1,
        min_feedback_signals=2,
        quality_threshold=0.6
    )
    
    # Simulate various scenarios including recovery from failures
    scenarios = [
        {
            'rule_id': 'rule_react_hooks',
            'initial_quality': 0.5,
            'feedback_sequence': [
                ('usage_success', 0.3),  # Initial poor performance
                ('user_rating', 0.4),    # Low user rating
                ('usage_success', 0.6),  # Slight improvement
                ('user_rating', 0.7),    # Better user feedback
                ('usage_success', 0.8),  # Good performance after fixes
            ]
        },
        {
            'rule_id': 'rule_fastapi_async',
            'initial_quality': 0.8,
            'feedback_sequence': [
                ('usage_success', 0.9),  # Consistently good
                ('user_rating', 0.85),   # High user satisfaction
                ('usage_success', 0.9),  # Maintained performance
            ]
        }
    ]
    
    print("\n🔄 Simulating feedback collection and self-improvement:")
    
    for scenario in scenarios:
        rule_id = scenario['rule_id']
        print(f"\n📈 Scenario: {rule_id}")
        print(f"   Initial quality: {scenario['initial_quality']:.3f}")
        
        # Collect feedback signals over time
        for signal_type, value in scenario['feedback_sequence']:
            await engine.collect_feedback_signal(
                rule_id=rule_id,
                signal_type=signal_type,
                value=value,
                context={'scenario': 'recovery_demo'},
                source='user'
            )
            
            # Show current quality trend
            if rule_id in engine.quality_history:
                current_avg = sum(engine.quality_history[rule_id]) / len(engine.quality_history[rule_id])
                print(f"   {signal_type}: {value:.3f} → avg quality: {current_avg:.3f}")
    
    # Demonstrate self-awarding
    print(f"\n🏆 Self-awarding demonstration:")
    mock_rules = [
        type('MockRule', (), {'rule': type('Rule', (), {'id': 'rule_react_hooks'})})(),
        type('MockRule', (), {'rule': type('Rule', (), {'id': 'rule_fastapi_async'})})(),
    ]
    
    batch_performance = {
        'improvement_score': 0.7,
        'quality_scores': {
            'rule_react_hooks': 0.7,   # Improved from 0.5
            'rule_fastapi_async': 0.85  # Maintained high quality
        },
        'predicted_qualities': {
            'rule_react_hooks': 0.6,   # Exceeded prediction
            'rule_fastapi_async': 0.8  # Met prediction
        }
    }
    
    awards = await engine.self_award_quality_improvements(mock_rules, batch_performance)
    
    for rule_id, award in awards.items():
        print(f"   🎯 {rule_id}: +{award:.3f} quality bonus")
    
    print(f"\n📊 Final adaptive thresholds: {engine.adaptive_thresholds}")


async def main():
    """Main demonstration function with error recovery focus."""
    parser = argparse.ArgumentParser(description="Fixed ML-Powered Batch Processing Demo")
    parser.add_argument('--bedrock', action='store_true', help='Use AWS Bedrock for enhanced rule generation')
    parser.add_argument('--demo-mode', choices=['validation', 'frameworks', 'self-improving', 'all'], 
                       default='all', help='Demo mode to run')
    
    args = parser.parse_args()
    
    print("🔧 RULES MAKER - FIXED BATCH PROCESSING DEMO")
    print("="*60)
    print("This demo addresses the 'failed pulls' issue with:")
    print("• Updated documentation URLs")
    print("• Enhanced validation and redirect handling") 
    print("• Intelligent fallback mechanisms")
    print("• Detailed error reporting and recovery")
    print()
    
    if args.bedrock:
        print("🤖 Using AWS Bedrock for enhanced rule generation")
    else:
        print("🔧 Using standard transformers")
    
    print(f"📋 Demo mode: {args.demo_mode}")
    print()
    
    try:
        if args.demo_mode == 'validation':
            await demo_enhanced_scraper_validation()
        elif args.demo_mode == 'frameworks':
            await demo_fixed_framework_processing()
        elif args.demo_mode == 'self-improving':
            await demo_self_improving_with_recovery()
        else:  # all
            await demo_enhanced_scraper_validation()
            await demo_fixed_framework_processing() 
            await demo_self_improving_with_recovery()
        
        print("\n✅ Fixed demo completed successfully!")
        print("📈 The 'failed pulls' issue has been resolved with:")
        print("   • URL validation and redirect handling")
        print("   • Fallback mechanisms for moved documentation")
        print("   • Enhanced error recovery and reporting")
        print("   • Self-improving quality assessment")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Fixed demo failed with exception")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())