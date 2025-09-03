#!/usr/bin/env python3
"""
Test Script for Next.js Documentation Pipeline

This script tests the complete Next.js documentation processing pipeline
including scraping, categorization, formatting, and learning integration.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rules_maker.scrapers import AsyncDocumentationScraper, AdaptiveDocumentationScraper
from rules_maker.transformers import CursorRuleTransformer
try:
    from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
    ML_TRANSFORMER_AVAILABLE = True
except ImportError:
    ML_TRANSFORMER_AVAILABLE = False
from rules_maker.models import ScrapingResult, ScrapingConfig, RuleFormat
from rules_maker.intelligence.nextjs_categorizer import NextJSCategorizer, NextJSCategory
from rules_maker.formatters.cursor_rules_formatter import CursorRulesFormatter
from rules_maker.learning.nextjs_learning_integration import NextJSLearningIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NextJSPipelineTester:
    """Test suite for the Next.js documentation pipeline."""
    
    def __init__(self, test_output_dir: str = "test_output"):
        """Initialize the pipeline tester.
        
        Args:
            test_output_dir: Directory for test outputs
        """
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test data
        self.test_urls = [
            "https://nextjs.org/docs/app",
            "https://nextjs.org/docs/pages",
            "https://nextjs.org/docs/api-reference"
        ]
        
        # Test results storage
        self.test_results = {
            'scraping_tests': {},
            'categorization_tests': {},
            'formatting_tests': {},
            'learning_tests': {},
            'integration_tests': {}
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all pipeline tests."""
        logger.info("🚀 Starting Next.js Pipeline Test Suite")
        
        # Test individual components
        await self.test_scraping_components()
        await self.test_categorization_system()
        await self.test_formatting_system()
        await self.test_learning_integration()
        
        # Test integration
        await self.test_end_to_end_pipeline()
        
        # Generate test report
        report = self.generate_test_report()
        
        logger.info("✅ All tests completed")
        return report
    
    async def test_scraping_components(self):
        """Test scraping components."""
        logger.info("Testing scraping components...")
        
        # Test async scraper
        try:
            scraper = AsyncDocumentationScraper()
            async with scraper:
                result = await scraper.scrape_url(self.test_urls[0])
                
                self.test_results['scraping_tests']['async_scraper'] = {
                    'status': 'passed',
                    'url': self.test_urls[0],
                    'content_length': len(result.content) if result.content else 0,
                    'status_code': result.status.value if hasattr(result.status, 'value') else str(result.status)
                }
                
                logger.info(f"✅ Async scraper test passed: {len(result.content)} chars")
                
        except Exception as e:
            self.test_results['scraping_tests']['async_scraper'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Async scraper test failed: {e}")
        
        # Test adaptive scraper
        try:
            adaptive_scraper = AdaptiveDocumentationScraper(use_ml=True, use_llm=False)
            async with adaptive_scraper:
                result = await adaptive_scraper.scrape_url(self.test_urls[1])
                
                self.test_results['scraping_tests']['adaptive_scraper'] = {
                    'status': 'passed',
                    'url': self.test_urls[1],
                    'content_length': len(result.content) if result.content else 0,
                    'ml_enhanced': hasattr(result, 'metadata') and result.metadata.get('ml_enhanced', False)
                }
                
                logger.info(f"✅ Adaptive scraper test passed: {len(result.content)} chars")
                
        except Exception as e:
            self.test_results['scraping_tests']['adaptive_scraper'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Adaptive scraper test failed: {e}")
    
    async def test_categorization_system(self):
        """Test categorization system."""
        logger.info("Testing categorization system...")
        
        try:
            categorizer = NextJSCategorizer()
            
            # Test content samples
            test_contents = [
                {
                    'content': 'Next.js App Router provides a new way to build applications with the app directory structure.',
                    'url': 'https://nextjs.org/docs/app',
                    'expected_category': 'routing'
                },
                {
                    'content': 'Server Components allow you to fetch data directly on the server without client-side JavaScript.',
                    'url': 'https://nextjs.org/docs/app/building-your-application/rendering/server-components',
                    'expected_category': 'data-fetching'
                },
                {
                    'content': 'CSS Modules provide a way to scope CSS locally to components.',
                    'url': 'https://nextjs.org/docs/app/building-your-application/styling/css-modules',
                    'expected_category': 'styling'
                }
            ]
            
            categorization_results = []
            
            for test_case in test_contents:
                categories = await categorizer.categorize_nextjs_content(
                    test_case['content'],
                    test_case['url']
                )
                
                # Get highest confidence category
                if categories:
                    top_category = max(categories.items(), key=lambda x: x[1].confidence)
                    categorization_results.append({
                        'input': test_case,
                        'predicted_category': top_category[0],
                        'confidence': top_category[1].confidence,
                        'all_categories': {k: v.confidence for k, v in categories.items()}
                    })
                else:
                    categorization_results.append({
                        'input': test_case,
                        'predicted_category': 'none',
                        'confidence': 0.0,
                        'all_categories': {}
                    })
            
            self.test_results['categorization_tests'] = {
                'status': 'passed',
                'results': categorization_results,
                'accuracy': self._calculate_categorization_accuracy(categorization_results)
            }
            
            logger.info(f"✅ Categorization test passed: {len(categorization_results)} test cases")
            
        except Exception as e:
            self.test_results['categorization_tests'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Categorization test failed: {e}")
    
    def _calculate_categorization_accuracy(self, results: List[Dict]) -> float:
        """Calculate categorization accuracy."""
        correct = 0
        total = len(results)
        
        for result in results:
            expected = result['input']['expected_category']
            predicted = result['predicted_category']
            
            if expected in predicted or predicted in expected:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    async def test_formatting_system(self):
        """Test formatting system."""
        logger.info("Testing formatting system...")
        
        try:
            formatter = CursorRulesFormatter()
            
            # Create mock scraping results
            mock_results = [
                ScrapingResult(
                    url="https://nextjs.org/docs/app",
                    title="App Router Documentation",
                    content="The App Router is a new paradigm for building applications using React's latest features.",
                    status="completed"
                ),
                ScrapingResult(
                    url="https://nextjs.org/docs/pages",
                    title="Pages Router Documentation", 
                    content="The Pages Router is the original Next.js routing system based on file system routing.",
                    status="completed"
                )
            ]
            
            # Test different output formats
            formats = ['mdc', 'json', 'yaml']
            formatting_results = {}
            
            for format_type in formats:
                formatted_rules = await formatter.format_scraping_results(
                    mock_results,
                    category_hint="routing",
                    output_format=format_type
                )
                
                formatting_results[format_type] = {
                    'files_generated': len(formatted_rules),
                    'sample_content_length': len(list(formatted_rules.values())[0]) if formatted_rules else 0
                }
                
                # Save test output
                test_dir = self.test_output_dir / f"formatting_test_{format_type}"
                test_dir.mkdir(exist_ok=True)
                formatter.save_formatted_rules(formatted_rules, str(test_dir))
            
            self.test_results['formatting_tests'] = {
                'status': 'passed',
                'results': formatting_results,
                'formats_tested': formats
            }
            
            logger.info(f"✅ Formatting test passed: {len(formats)} formats tested")
            
        except Exception as e:
            self.test_results['formatting_tests'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Formatting test failed: {e}")
    
    async def test_learning_integration(self):
        """Test learning integration system."""
        logger.info("Testing learning integration...")
        
        try:
            learning_integration = NextJSLearningIntegration()
            
            # Test recording events
            await learning_integration.record_categorization_event(
                content="Test content for learning",
                url="https://test.example.com",
                categories={'routing': type('obj', (object,), {'confidence': 0.8})()},
                confidence=0.8
            )
            
            await learning_integration.record_user_feedback(
                content_hash=hash("test content"),
                url="https://test.example.com",
                feedback={'satisfaction': 0.9, 'category': 'routing'}
            )
            
            # Test learning analysis
            patterns = await learning_integration.analyze_learning_patterns()
            
            # Test metrics generation
            metrics = await learning_integration.get_learning_metrics()
            
            self.test_results['learning_tests'] = {
                'status': 'passed',
                'events_recorded': len(learning_integration.learning_events),
                'patterns_analyzed': patterns.get('total_events', 0),
                'metrics_generated': metrics.total_events > 0
            }
            
            logger.info(f"✅ Learning integration test passed: {len(learning_integration.learning_events)} events recorded")
            
        except Exception as e:
            self.test_results['learning_tests'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Learning integration test failed: {e}")
    
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")
        
        try:
            # Initialize components
            scraper = AsyncDocumentationScraper()
            categorizer = NextJSCategorizer()
            formatter = CursorRulesFormatter(categorizer)
            learning_integration = NextJSLearningIntegration()
            
            # Step 1: Scrape content
            scraping_results = []
            async with scraper:
                for url in self.test_urls[:2]:  # Test with 2 URLs
                    result = await scraper.scrape_url(url)
                    if result.content:
                        scraping_results.append(result)
            
            # Step 2: Categorize content
            categorized_results = {}
            for result in scraping_results:
                categories = await categorizer.categorize_nextjs_content(
                    result.content, result.url
                )
                if categories:
                    top_category = max(categories.items(), key=lambda x: x[1].confidence)[0]
                    if top_category not in categorized_results:
                        categorized_results[top_category] = []
                    categorized_results[top_category].append(result)
                    
                    # Record learning event
                    await learning_integration.record_categorization_event(
                        result.content, result.url, categories, 
                        max(categories.values(), key=lambda x: x.confidence).confidence
                    )
            
            # Step 3: Format rules
            all_formatted_rules = {}
            for category, results in categorized_results.items():
                formatted_rules = await formatter.format_scraping_results(
                    results, category_hint=category, output_format='mdc'
                )
                all_formatted_rules.update(formatted_rules)
                
                # Record rule generation event
                await learning_integration.record_rule_generation_event(
                    results, formatted_rules, 0.8  # Mock quality score
                )
            
            # Step 4: Save results
            pipeline_output_dir = self.test_output_dir / "end_to_end_pipeline"
            formatter.save_formatted_rules(all_formatted_rules, str(pipeline_output_dir))
            
            # Step 5: Generate learning report
            learning_report = await learning_integration.generate_learning_report()
            
            self.test_results['integration_tests'] = {
                'status': 'passed',
                'urls_processed': len(scraping_results),
                'categories_generated': len(categorized_results),
                'rules_generated': len(all_formatted_rules),
                'learning_events': len(learning_integration.learning_events),
                'output_directory': str(pipeline_output_dir)
            }
            
            logger.info(f"✅ End-to-end pipeline test passed: {len(all_formatted_rules)} rules generated")
            
        except Exception as e:
            self.test_results['integration_tests'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ End-to-end pipeline test failed: {e}")
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            'test_suite': 'Next.js Documentation Pipeline',
            'timestamp': str(datetime.now()),
            'test_results': self.test_results,
            'summary': self._generate_test_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_file = self.test_output_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {report_file}")
        return report
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_category, results in self.test_results.items():
            if isinstance(results, dict) and 'status' in results:
                total_tests += 1
                if results['status'] == 'passed':
                    passed_tests += 1
                else:
                    failed_tests += 1
            elif isinstance(results, dict):
                # Handle nested test results
                for test_name, test_result in results.items():
                    if isinstance(test_result, dict) and 'status' in test_result:
                        total_tests += 1
                        if test_result['status'] == 'passed':
                            passed_tests += 1
                        else:
                            failed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if failed_tests == 0 else 'FAILED'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check scraping tests
        scraping_tests = self.test_results.get('scraping_tests', {})
        if any(test.get('status') == 'failed' for test in scraping_tests.values()):
            recommendations.append("Fix scraping component issues - check network connectivity and scraper configuration")
        
        # Check categorization tests
        categorization_tests = self.test_results.get('categorization_tests', {})
        if categorization_tests.get('status') == 'failed':
            recommendations.append("Fix categorization system - check pattern matching and taxonomy configuration")
        elif categorization_tests.get('accuracy', 0) < 0.7:
            recommendations.append("Improve categorization accuracy - consider updating patterns and training data")
        
        # Check formatting tests
        formatting_tests = self.test_results.get('formatting_tests', {})
        if formatting_tests.get('status') == 'failed':
            recommendations.append("Fix formatting system - check template generation and output format handling")
        
        # Check learning tests
        learning_tests = self.test_results.get('learning_tests', {})
        if learning_tests.get('status') == 'failed':
            recommendations.append("Fix learning integration - check event recording and analysis systems")
        
        # Check integration tests
        integration_tests = self.test_results.get('integration_tests', {})
        if integration_tests.get('status') == 'failed':
            recommendations.append("Fix end-to-end pipeline - check component integration and data flow")
        
        if not recommendations:
            recommendations.append("All tests passed - pipeline is ready for production use")
        
        return recommendations
    
    def print_test_summary(self, report: Dict[str, Any]):
        """Print test summary to console."""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("🧪 NEXT.JS PIPELINE TEST SUMMARY")
        print("="*60)
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"🎯 Overall Status: {summary['overall_status']}")
        
        print(f"\n📁 Test Output Directory: {self.test_output_dir}")
        
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"   {i}. {recommendation}")
        
        print("="*60)


async def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(
        description="Test Next.js Documentation Pipeline"
    )
    
    parser.add_argument(
        '--output-dir',
        default='test_output',
        help='Output directory for test results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip end-to-end pipeline)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = NextJSPipelineTester(test_output_dir=args.output_dir)
    
    # Run tests
    if args.quick:
        logger.info("Running quick tests only...")
        await tester.test_scraping_components()
        await tester.test_categorization_system()
        await tester.test_formatting_system()
        await tester.test_learning_integration()
    else:
        report = await tester.run_all_tests()
        tester.print_test_summary(report)
    
    logger.info("🎉 Test execution completed!")


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(main())
