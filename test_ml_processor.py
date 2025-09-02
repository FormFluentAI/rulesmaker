#!/usr/bin/env python3
"""
Test script for MLDocumentationProcessor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor

def test_ml_processor():
    """Test the ML documentation processor with sample content."""

    # Sample HTML content
    sample_content = """
    <html>
    <head><title>Python Best Practices</title></head>
    <body>
        <h1>Python Development Guide</h1>
        <p>This guide covers best practices for Python development.</p>

        <h2>Code Examples</h2>
        <pre><code>
def hello_world():
    print("Hello, World!")

# Best practice: Use descriptive variable names
user_name = "Alice"
print(f"Hello, {user_name}!")
        </code></pre>

        <h2>Important Rules</h2>
        <ul>
            <li>Always use meaningful variable names</li>
            <li>Write clear, readable code</li>
            <li>Add docstrings to functions</li>
        </ul>
    </body>
    </html>
    """

    # Initialize processor
    processor = MLDocumentationProcessor()

    # Process content
    result = processor.process(
        content=sample_content,
        url="https://example.com/python-guide",
        metadata={"source": "test"}
    )

    # Print results
    print("=== ML Documentation Processor Test ===")
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Documentation Type: {result.documentation_type}")
    print(f"ML Enhanced: {result.metadata.get('ml_enhanced', False)}")
    print(f"Semantic Keywords: {result.metadata.get('semantic_keywords', [])}")
    print(f"Content Complexity: {result.metadata.get('content_complexity', 'N/A')}")
    print(f"Number of sections: {len(result.sections)}")

    if result.sections:
        print(f"\nFirst section title: {result.sections[0].title}")
        print(f"First section has code: {result.sections[0].metadata.get('has_code_examples', False)}")

    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    test_ml_processor()
