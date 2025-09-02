#!/usr/bin/env python3
"""
Direct test of transformers functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_transformers():
    """Test transformers directly without issues."""
    
    print("=== Direct Transformer Test ===\n")
    
    # Create a properly formatted result
    try:
        from rules_maker.models import ScrapingResult
        result = ScrapingResult(
            url="https://docs.python.org/3/", 
            title="Python Documentation",
            content="This is a Python FastAPI tutorial showing how to create REST APIs with async/await patterns."
        )
        print("✅ ScrapingResult created successfully")
        print(f"   URL: {result.url}")
        print(f"   Title: {result.title}")
        print(f"   Content length: {len(result.content)} chars")
    except Exception as e:
        print(f"❌ ScrapingResult creation failed: {e}")
        return
    
    # Test Cursor Transformer
    print("\n--- Testing Cursor Transformer ---")
    try:
        from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
        cursor_transformer = CursorRuleTransformer()
        print("✅ Cursor transformer imported and instantiated")
        
        cursor_rules = cursor_transformer.transform([result])
        print(f"✅ Cursor transformation successful")
        print(f"   Generated {len(cursor_rules)} characters")
        print(f"   Preview: {cursor_rules[:200]}...")
        
        # Check if it contains expected sections
        if "Expert Role" in cursor_rules or "Python" in cursor_rules:
            print("✅ Rules contain expected content")
        else:
            print("⚠️  Rules may not contain expected content")
            
    except Exception as e:
        print(f"❌ Cursor transformer failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Windsurf Transformer
    print("\n--- Testing Windsurf Transformer ---")
    try:
        from rules_maker.transformers.windsurf_transformer import WindsurfRuleTransformer
        windsurf_transformer = WindsurfRuleTransformer()
        print("✅ Windsurf transformer imported and instantiated")
        
        windsurf_rules = windsurf_transformer.transform([result])
        print(f"✅ Windsurf transformation successful")
        print(f"   Generated {len(windsurf_rules)} characters")
        print(f"   Preview: {windsurf_rules[:200]}...")
        
        # Check if it contains expected sections
        if "Workflow" in windsurf_rules or "Quality Gates" in windsurf_rules:
            print("✅ Rules contain expected workflow content")
        else:
            print("⚠️  Rules may not contain expected workflow content")
            
    except Exception as e:
        print(f"❌ Windsurf transformer failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Transformer Test Complete ===")

if __name__ == "__main__":
    test_transformers()
