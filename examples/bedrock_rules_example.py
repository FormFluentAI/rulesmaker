#!/usr/bin/env python3
"""
Example usage of Rules Maker with AWS Bedrock Nova Lite.

This example shows how to:
1. Load credentials from CSV
2. Generate Cursor rules using Bedrock
3. Generate Windsurf rules using Bedrock
4. Use enhanced LLM-powered rule generation

Prerequisites:
- pip install boto3
- Bedrock credentials in docs/plans/bedrock-long-term-api-key.csv
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rules_maker.bedrock_integration import (
    BedrockRulesMaker,
    quick_cursor_rules,
    quick_windsurf_rules,
    quick_enhanced_cursor_rules,
    quick_enhanced_windsurf_rules
)


# Example documentation content
FASTAPI_DOCS = """
FastAPI

FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

Key features:
- Fast: Very high performance, on par with NodeJS and Go
- Fast to code: Increase the speed to develop features by about 200% to 300%
- Fewer bugs: Reduce about 40% of human (developer) induced errors
- Intuitive: Great editor support. Completion everywhere. Less time debugging
- Easy: Designed to be easy to use and learn. Less time reading docs
- Short: Minimize code duplication. Multiple features from each parameter declaration
- Robust: Get production-ready code. With automatic interactive documentation

Installation:
pip install fastapi
pip install "uvicorn[standard]"

Basic Usage:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

Best Practices:
- Use Pydantic models for request/response validation
- Implement proper error handling with HTTPException
- Use dependency injection for database connections
- Follow RESTful API design principles
- Include comprehensive tests with pytest
"""


def basic_example():
    """Basic example using simple rule generation."""
    print("🚀 Basic Bedrock Rules Generation Example")
    print("=" * 50)
    
    try:
        # Create BedrockRulesMaker instance
        # This will automatically load credentials from CSV and validate access
        maker = BedrockRulesMaker(
            model_id="amazon.nova-lite-v1:0",
            region="us-east-1"
        )
        
        print("✅ Bedrock connection established")
        
        # Generate Cursor rules
        print("\n📋 Generating Cursor Rules...")
        cursor_rules = maker.generate_cursor_rules(
            documentation_content=FASTAPI_DOCS,
            title="FastAPI Documentation",
            url="https://fastapi.tiangolo.com/"
        )
        
        print("✅ Cursor rules generated!")
        print(f"Length: {len(cursor_rules)} characters")
        print("\n--- CURSOR RULES PREVIEW ---")
        print(cursor_rules[:500] + "..." if len(cursor_rules) > 500 else cursor_rules)
        
        # Generate Windsurf rules
        print("\n📋 Generating Windsurf Rules...")
        windsurf_rules = maker.generate_windsurf_rules(
            documentation_content=FASTAPI_DOCS,
            title="FastAPI Documentation", 
            url="https://fastapi.tiangolo.com/"
        )
        
        print("✅ Windsurf rules generated!")
        print(f"Length: {len(windsurf_rules)} characters")
        print("\n--- WINDSURF RULES PREVIEW ---")
        print(windsurf_rules[:500] + "..." if len(windsurf_rules) > 500 else windsurf_rules)
        
        # Show usage stats
        stats = maker.get_usage_stats()
        print(f"\n📊 Usage Stats:")
        print(f"- Model: {stats['model_id']}")
        print(f"- Region: {stats['region']}")
        print(f"- Requests: {stats['total_requests']}")
        print(f"- Input tokens: {stats['total_input_tokens']}")
        print(f"- Output tokens: {stats['total_output_tokens']}")
        print(f"- Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. boto3 installed: pip install boto3")
        print("2. Valid credentials in docs/plans/bedrock-long-term-api-key.csv")
        print("3. Proper AWS permissions for Bedrock")


async def enhanced_example():
    """Enhanced example using LLM-powered analysis."""
    print("\n🧠 Enhanced LLM-Powered Rules Generation Example")
    print("=" * 55)
    
    try:
        # Create BedrockRulesMaker instance
        maker = BedrockRulesMaker(
            model_id="amazon.nova-lite-v1:0",
            region="us-east-1",
            temperature=0.2,  # Lower temperature for more consistent output
            max_tokens=3000   # More tokens for detailed analysis
        )
        
        print("✅ Bedrock connection established for enhanced generation")
        
        # Test connection first
        connection_test = await maker.test_bedrock_connection()
        if connection_test['success']:
            print("✅ Bedrock connection test successful")
            print(f"Response: {connection_test['response']}")
        else:
            print(f"❌ Connection test failed: {connection_test['error']}")
            return
        
        # Generate enhanced Cursor rules with LLM analysis
        print("\n🧠 Generating Enhanced Cursor Rules with LLM Analysis...")
        enhanced_cursor = await maker.generate_enhanced_cursor_rules(
            documentation_content=FASTAPI_DOCS,
            title="FastAPI Documentation",
            url="https://fastapi.tiangolo.com/"
        )
        
        print("✅ Enhanced Cursor rules generated!")
        print(f"Length: {len(enhanced_cursor)} characters")
        print("\n--- ENHANCED CURSOR RULES PREVIEW ---")
        print(enhanced_cursor[:600] + "..." if len(enhanced_cursor) > 600 else enhanced_cursor)
        
        # Generate enhanced Windsurf rules with LLM analysis
        print("\n🧠 Generating Enhanced Windsurf Rules with LLM Analysis...")
        enhanced_windsurf = await maker.generate_enhanced_windsurf_rules(
            documentation_content=FASTAPI_DOCS,
            title="FastAPI Documentation",
            url="https://fastapi.tiangolo.com/"
        )
        
        print("✅ Enhanced Windsurf rules generated!")
        print(f"Length: {len(enhanced_windsurf)} characters")
        print("\n--- ENHANCED WINDSURF RULES PREVIEW ---")
        print(enhanced_windsurf[:600] + "..." if len(enhanced_windsurf) > 600 else enhanced_windsurf)
        
        # Final usage stats
        final_stats = maker.get_usage_stats()
        print(f"\n📊 Final Usage Stats:")
        print(f"- Total requests: {final_stats['total_requests']}")
        print(f"- Total input tokens: {final_stats['total_input_tokens']}")
        print(f"- Total output tokens: {final_stats['total_output_tokens']}")
        print(f"- Total estimated cost: ${final_stats['estimated_cost_usd']:.4f}")
        
        # Cleanup
        await maker.close()
        
    except Exception as e:
        print(f"❌ Enhanced example error: {e}")
        import traceback
        traceback.print_exc()


def quick_functions_example():
    """Example using quick utility functions."""
    print("\n⚡ Quick Functions Example")
    print("=" * 30)
    
    try:
        # Quick Cursor rules
        cursor_rules = quick_cursor_rules(
            FASTAPI_DOCS,
            model_id="amazon.nova-lite-v1:0"
        )
        print(f"✅ Quick Cursor rules generated ({len(cursor_rules)} chars)")
        
        # Quick Windsurf rules  
        windsurf_rules = quick_windsurf_rules(
            FASTAPI_DOCS,
            model_id="amazon.nova-lite-v1:0"
        )
        print(f"✅ Quick Windsurf rules generated ({len(windsurf_rules)} chars)")
        
    except Exception as e:
        print(f"❌ Quick functions error: {e}")


async def quick_enhanced_example():
    """Example using quick enhanced functions."""
    print("\n⚡🧠 Quick Enhanced Functions Example")  
    print("=" * 40)
    
    try:
        # Quick enhanced Cursor rules
        enhanced_cursor = await quick_enhanced_cursor_rules(
            FASTAPI_DOCS,
            model_id="amazon.nova-lite-v1:0"
        )
        print(f"✅ Quick enhanced Cursor rules generated ({len(enhanced_cursor)} chars)")
        
        # Quick enhanced Windsurf rules
        enhanced_windsurf = await quick_enhanced_windsurf_rules(
            FASTAPI_DOCS,
            model_id="amazon.nova-lite-v1:0"
        )
        print(f"✅ Quick enhanced Windsurf rules generated ({len(enhanced_windsurf)} chars)")
        
    except Exception as e:
        print(f"❌ Quick enhanced functions error: {e}")


def main():
    """Run all examples."""
    print("🤖 Rules Maker with AWS Bedrock Nova Lite - Examples")
    print("=" * 60)
    
    # Check for credentials file
    creds_path = Path(__file__).parent.parent / "docs" / "plans" / "bedrock-long-term-api-key.csv"
    if not creds_path.exists():
        print(f"⚠️  Credentials file not found at: {creds_path}")
        print("Please ensure your Bedrock credentials CSV exists at this path.")
        return
    
    print(f"✅ Found credentials file: {creds_path}")
    
    # Run examples
    basic_example()
    quick_functions_example()
    
    # Run async examples
    asyncio.run(enhanced_example())
    asyncio.run(quick_enhanced_example())
    
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Save generated rules to .cursorrules or windsurf_rules.md files")
    print("2. Integrate with your development workflow")
    print("3. Customize the BedrockRulesMaker settings for your needs")


if __name__ == "__main__":
    main()