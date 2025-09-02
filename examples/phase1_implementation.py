"""
Phase 1 Implementation Examples.

This script demonstrates the usage of all Phase 1 components:
1. Async Documentation Scraper
2. ML Content Extractor  
3. LLM Integration
"""

import asyncio
import logging
from pathlib import Path
from typing import List

from rules_maker import (
    AsyncDocumentationScraper,
    AdaptiveDocumentationScraper,
    MLContentExtractor,
    LLMContentExtractor,
    ScrapingConfig,
)
from rules_maker.extractors.llm_extractor import LLMConfig, LLMProvider
from rules_maker.models import TrainingSet, LearningExample, DocumentationType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_async_scraper():
    """Example of using the async documentation scraper."""
    print("\n🚀 Async Scraper Example")
    print("=" * 50)
    
    # Configure scraping
    config = ScrapingConfig(
        max_pages=5,
        max_depth=2,
        rate_limit=1.0,
        timeout=30
    )
    
    # Create async scraper
    async with AsyncDocumentationScraper(config) as scraper:
        # Scrape a single URL
        result = await scraper.scrape_url("https://docs.python.org/3/tutorial/")
        print(f"✅ Scraped: {result.title}")
        print(f"📝 Content length: {len(result.content)} characters")
        print(f"📑 Sections found: {len(result.sections)}")
        
        # Scrape multiple URLs concurrently
        urls = [
            "https://docs.python.org/3/tutorial/introduction.html",
            "https://docs.python.org/3/tutorial/controlflow.html",
            "https://docs.python.org/3/tutorial/datastructures.html"
        ]
        
        results = await scraper.scrape_multiple(urls)
        print(f"\n📚 Scraped {len(results)} pages concurrently")
        
        for result in results:
            if result.status.value == "completed":
                print(f"  ✅ {result.title}: {len(result.sections)} sections")
            else:
                print(f"  ❌ Failed: {result.error_message}")


async def example_ml_extractor():
    """Example of using the ML content extractor."""
    print("\n🧠 ML Extractor Example") 
    print("=" * 50)
    
    # Create ML extractor
    ml_extractor = MLContentExtractor(use_transformers=True)
    
    # Simulate some HTML content for extraction
    html_content = """
    <html>
    <head><title>Python Tutorial</title></head>
    <body>
        <h1>Introduction to Python</h1>
        <p>Python is a powerful programming language...</p>
        
        <h2>Installation</h2>
        <p>To install Python, download from python.org...</p>
        <pre><code>pip install python</code></pre>
        
        <h2>Basic Syntax</h2>
        <p>Python uses indentation for code blocks...</p>
        <code>print("Hello, World!")</code>
        
        <h2>Examples</h2>
        <p>Here are some basic examples:</p>
        <pre><code>
        # This is a comment
        x = 10
        y = 20
        print(x + y)
        </code></pre>
    </body>
    </html>
    """
    
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract structured content
    extraction_result = ml_extractor.extract(soup, "https://example.com/tutorial")
    print(f"📄 Document title: {extraction_result.get('title')}")
    print(f"📋 Document type: {extraction_result.get('document_type')}")
    print(f"🎯 Confidence score: {extraction_result.get('confidence_score', 0):.2f}")
    
    # Extract sections
    sections = ml_extractor.extract_sections(soup, "https://example.com/tutorial")
    print(f"\n📑 Extracted {len(sections)} sections:")
    
    for i, section in enumerate(sections, 1):
        print(f"  {i}. {section.title} (Level {section.level})")
        print(f"     Type: {section.metadata.get('section_type', 'unknown')}")
        print(f"     Confidence: {section.metadata.get('confidence', 0):.2f}")
        print(f"     Has code: {section.metadata.get('has_code', False)}")


async def example_llm_extractor():
    """Example of using the LLM content extractor."""
    print("\n🤖 LLM Extractor Example")
    print("=" * 50)
    
    # Note: This example requires API keys to work
    # For demo purposes, we'll show the setup
    
    # Configure LLM (you would need to set your API key)
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key-here",  # Set your actual API key
        temperature=0.3,
        max_tokens=2000
    )
    
    # Create LLM extractor
    llm_extractor = LLMContentExtractor(llm_config=llm_config)
    
    # Simulate content for rule generation
    from rules_maker.models import ContentSection
    
    sections = [
        ContentSection(
            title="Best Practices",
            content="Always use meaningful variable names. Write comments for complex logic. Follow PEP 8 style guide.",
            level=1
        ),
        ContentSection(
            title="Error Handling", 
            content="Use try-except blocks for error handling. Always catch specific exceptions. Provide meaningful error messages.",
            level=1
        )
    ]
    
    print("🎯 Configured LLM extractor")
    print(f"   Provider: {llm_config.provider}")
    print(f"   Model: {llm_config.model_name}")
    print(f"   Content sections: {len(sections)}")
    
    # Note: Actual rule generation would require API key
    print("\n💡 To generate rules, you would call:")
    print("   rules = await llm_extractor.generate_rules(sections, 'cursor')")
    
    await llm_extractor.close()


async def example_adaptive_scraper():
    """Example of using the adaptive scraper with ML and LLM."""
    print("\n🔮 Adaptive Scraper Example")
    print("=" * 50)
    
    # Configure scraping
    config = ScrapingConfig(
        max_pages=3,
        rate_limit=1.0,
        timeout=30
    )
    
    # Configure LLM (optional)
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key-here"  # Set your actual API key
    )
    
    # Create adaptive scraper with ML enabled, LLM optional
    adaptive_scraper = AdaptiveDocumentationScraper(
        config=config,
        use_ml=True,
        use_llm=False,  # Set to True if you have LLM API key
        llm_config=llm_config
    )
    
    try:
        # Scrape with adaptive extraction
        result = await adaptive_scraper.scrape_url("https://docs.python.org/3/tutorial/")
        
        print(f"📄 Title: {result.title}")
        print(f"📋 Type: {result.documentation_type}")
        print(f"📑 Sections: {len(result.sections)}")
        print(f"🎯 Extraction method: {result.metadata.get('extraction_method', 'unknown')}")
        print(f"🎯 Confidence: {result.metadata.get('confidence_score', 0):.2f}")
        
        # Show extraction statistics
        stats = adaptive_scraper.get_extraction_stats()
        print(f"\n📊 Extraction Statistics:")
        print(f"   Total extractions: {stats['total_extractions']}")
        print(f"   ML success rate: {stats['ml_success_rate']:.2%}")
        print(f"   LLM success rate: {stats['llm_success_rate']:.2%}")
        print(f"   Fallback rate: {stats['fallback_rate']:.2%}")
        
    finally:
        await adaptive_scraper.close()


async def example_ml_training():
    """Example of training the ML extractor."""
    print("\n🎓 ML Training Example")
    print("=" * 50)
    
    # Create training examples (in practice, you'd load these from files)
    training_examples = [
        LearningExample(
            input_html="<html><h1>Installation</h1><p>Download and install...</p></html>",
            expected_output={"section_type": "installation"},
            url="https://example.com/install",
            documentation_type=DocumentationType.GUIDE
        ),
        LearningExample(
            input_html="<html><h1>API Reference</h1><p>function foo(x, y)...</p></html>",
            expected_output={"section_type": "api_reference"},
            url="https://example.com/api",
            documentation_type=DocumentationType.API
        )
    ]
    
    # Create training set
    training_set = TrainingSet(
        name="Documentation Training Set",
        description="Training examples for documentation classification",
        examples=training_examples,
        documentation_type=DocumentationType.GUIDE
    )
    
    # Create and train ML extractor
    ml_extractor = MLContentExtractor(use_transformers=True)
    
    print(f"📚 Training with {len(training_set.examples)} examples...")
    
    try:
        performance = ml_extractor.train(training_set)
        print(f"✅ Training completed!")
        print(f"   Accuracy: {performance.get('accuracy', 0):.2%}")
        
        # Save trained model
        model_path = "trained_model.pkl"
        ml_extractor.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("   This is expected without proper training data")


async def main():
    """Run all Phase 1 examples."""
    print("🎉 Rules Maker - Phase 1 Implementation Examples")
    print("=" * 60)
    
    # Run examples
    await example_async_scraper()
    await example_ml_extractor()
    await example_llm_extractor()
    await example_adaptive_scraper()
    await example_ml_training()
    
    print("\n✨ All Phase 1 examples completed!")
    print("\n📝 Next Steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Set up API keys for LLM integration")
    print("   3. Prepare training data for ML models")
    print("   4. Test with real documentation sites")


if __name__ == "__main__":
    asyncio.run(main())
