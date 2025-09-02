#!/usr/bin/env python3
"""
Example demonstrating the enhanced Semantic Content Analysis Engine
with Context7 integration for fetching latest documentation.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rules_maker.intelligence.semantic_analyzer import SemanticAnalyzer


async def main():
    """Demonstrate the enhanced semantic analyzer with Context7 integration."""

    # Initialize the analyzer
    analyzer = SemanticAnalyzer()

    # Example Next.js documentation content
    sample_content = """
    # Next.js App Router Documentation

    The App Router is the new way to build Next.js applications using React Server Components.

    ## Routing Fundamentals

    In Next.js 14, you can use the App Router to create nested routes with layouts.

    ### File Structure
    ```
    app/
      layout.tsx
      page.tsx
      blog/
        page.tsx
        [slug]/
          page.tsx
    ```

    ### Server Components
    Server Components allow you to render components on the server, reducing bundle size.

    ```tsx
    // app/page.tsx
    export default function Page() {
      return <h1>Hello, Next.js!</h1>
    }
    ```

    ### Data Fetching
    Use async/await in Server Components for data fetching:

    ```tsx
    async function getData() {
      const res = await fetch('https://api.example.com/data')
      return res.json()
    }

    export default async function Page() {
      const data = await getData()
      return <div>{data.title}</div>
    }
    ```
    """

    sample_url = "https://nextjs.org/docs/app/building-your-application/routing"

    print("🔍 Analyzing documentation content...")
    print(f"URL: {sample_url}")
    print("-" * 60)

    # Analyze the content
    analysis = await analyzer.analyze_content(sample_content, sample_url)

    # Display results
    print("📊 Analysis Results:")
    print(f"Primary Technology: {analysis.primary_technology}")
    print(f"Secondary Technologies: {', '.join(analysis.secondary_technologies)}")
    print(f"Complexity Level: {analysis.complexity_level.value}")
    print(f"Content Type: {analysis.content_type.value}")
    print(f"Framework Version: {analysis.framework_version or 'Not detected'}")
    print(f"Language Detected: {analysis.language_detected or 'Not detected'}")
    print(f"Quality Score: {analysis.quality_score:.2f}")
    print(f"Code Examples: {analysis.code_examples_count}")
    print(f"External Links: {analysis.external_links_count}")

    print("\n🏷️  Content Categories:")
    for category_name, category_data in analysis.content_categories.items():
        print(f"  • {category_name} (confidence: {category_data.confidence:.2f})")
        if category_data.topics:
            print(f"    Topics: {', '.join(category_data.topics[:3])}")
        if category_data.patterns:
            print(f"    Patterns: {', '.join(category_data.patterns[:3])}")

    if analysis.prerequisites:
        print("\n📚 Prerequisites:")
        for prereq in analysis.prerequisites:
            print(f"  • {prereq}")

    print("\n✅ Analysis complete!")
    print("\nNote: Context7 integration is configured to fetch latest documentation")
    print("from the resolved library repositories for enhanced analysis.")


if __name__ == "__main__":
    asyncio.run(main())
