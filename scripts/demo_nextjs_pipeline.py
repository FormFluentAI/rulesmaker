#!/usr/bin/env python3
"""
Next.js Pipeline Demonstration Script

This script demonstrates the complete Next.js documentation processing pipeline
with a simple example that shows all the key features.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rules_maker.scrapers import AsyncDocumentationScraper
from rules_maker.intelligence.nextjs_categorizer import NextJSCategorizer
from rules_maker.formatters.cursor_rules_formatter import CursorRulesFormatter
from rules_maker.learning.nextjs_learning_integration import NextJSLearningIntegration
from rules_maker.models import ScrapingResult, ScrapingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_nextjs_pipeline():
    """Demonstrate the Next.js documentation pipeline."""
    print("🚀 Next.js Documentation Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Initialize components
    print("\n1️⃣ Initializing pipeline components...")
    
    categorizer = NextJSCategorizer()
    formatter = CursorRulesFormatter(categorizer)
    learning_integration = NextJSLearningIntegration()
    
    scraper_config = ScrapingConfig(
        max_pages=5,
        rate_limit=1.0,
        max_depth=2
    )
    scraper = AsyncDocumentationScraper(config=scraper_config)
    
    print("✅ Components initialized successfully")
    
    # Step 2: Create mock documentation content
    print("\n2️⃣ Creating mock Next.js documentation content...")
    
    mock_content = """
    # Next.js App Router Guide
    
    The App Router is a new paradigm for building applications using React's latest features.
    
    ## Key Features
    
    - **Server Components**: Fetch data directly on the server
    - **Client Components**: Interactive components with client-side JavaScript
    - **Nested Layouts**: Share UI between multiple pages
    - **Loading States**: Built-in loading UI with loading.tsx
    - **Error Handling**: Error boundaries with error.tsx
    
    ## Example: Server Component
    
    ```typescript
    // app/page.tsx
    async function HomePage() {
      const data = await fetch('https://api.example.com/data')
      const posts = await data.json()
      
      return (
        <div>
          <h1>My Blog</h1>
          {posts.map(post => (
            <article key={post.id}>
              <h2>{post.title}</h2>
              <p>{post.excerpt}</p>
            </article>
          ))}
        </div>
      )
    }
    
    export default HomePage
    ```
    
    ## Best Practices
    
    - Use Server Components by default
    - Only use Client Components when you need interactivity
    - Leverage nested layouts for shared UI
    - Use loading.tsx for loading states
    - Implement error.tsx for error boundaries
    """
    
    # Create mock scraping result
    mock_result = ScrapingResult(
        url="https://nextjs.org/docs/app",
        title="Next.js App Router Guide",
        content=mock_content,
        status="completed"
    )
    
    print("✅ Mock content created")
    
    # Step 3: Categorize content
    print("\n3️⃣ Categorizing content...")
    
    categories = await categorizer.categorize_nextjs_content(
        mock_content, 
        "https://nextjs.org/docs/app"
    )
    
    if categories:
        top_category = max(categories.items(), key=lambda x: x[1].confidence)
        print(f"✅ Content categorized as: {top_category[0]} (confidence: {top_category[1].confidence:.2f})")
        print(f"   All categories: {list(categories.keys())}")
    else:
        print("❌ No categories identified")
    
    # Step 4: Record learning event
    print("\n4️⃣ Recording learning event...")
    
    await learning_integration.record_categorization_event(
        mock_content,
        "https://nextjs.org/docs/app",
        categories,
        top_category[1].confidence if categories else 0.0
    )
    
    print("✅ Learning event recorded")
    
    # Step 5: Format into cursor rules
    print("\n5️⃣ Formatting content into cursor rules...")
    
    formatted_rules = await formatter.format_scraping_results(
        [mock_result],
        category_hint="routing",
        output_format="mdc"
    )
    
    print(f"✅ Generated {len(formatted_rules)} cursor rule files")
    
    # Step 6: Save formatted rules
    print("\n6️⃣ Saving formatted rules...")
    
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    formatter.save_formatted_rules(formatted_rules, str(output_dir))
    
    # Generate index
    index_file = formatter.generate_rule_index(formatted_rules, str(output_dir))
    
    print(f"✅ Rules saved to: {output_dir}")
    print(f"✅ Index file: {index_file}")
    
    # Step 7: Generate learning report
    print("\n7️⃣ Generating learning report...")
    
    learning_report = await learning_integration.generate_learning_report()
    
    report_file = output_dir / "learning_report.json"
    import json
    with open(report_file, 'w') as f:
        json.dump(learning_report, f, indent=2)
    
    print(f"✅ Learning report saved to: {report_file}")
    
    # Step 8: Display results
    print("\n8️⃣ Demo Results Summary")
    print("=" * 30)
    
    print(f"📄 Content processed: {len(mock_content)} characters")
    print(f"🏷️  Categories identified: {len(categories) if categories else 0}")
    print(f"📝 Cursor rules generated: {len(formatted_rules)}")
    print(f"🧠 Learning events recorded: {len(learning_integration.learning_events)}")
    print(f"📁 Output directory: {output_dir}")
    
    # Show sample rule content
    if formatted_rules:
        sample_file = list(formatted_rules.keys())[0]
        sample_content = formatted_rules[sample_file]
        print(f"\n📋 Sample rule content ({sample_file}):")
        print("-" * 40)
        print(sample_content[:500] + "..." if len(sample_content) > 500 else sample_content)
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Check the generated cursor rules in demo_output/")
    print("2. Use the rules in your .cursor/rules/ directory")
    print("3. Run the full pipeline with: rules-maker nextjs process")
    print("4. Test the pipeline with: rules-maker nextjs test")


if __name__ == "__main__":
    asyncio.run(demo_nextjs_pipeline())
