#!/usr/bin/env python3
"""
Test script to check Rules Maker status.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    
    print("=== Rules Maker Status Check ===\n")
    
    # Test 1: Basic imports
    print("1. Testing basic imports...")
    try:
        from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
        print("   ✅ Cursor transformer import successful")
    except Exception as e:
        print(f"   ❌ Cursor transformer import failed: {e}")
    
    try:
        from rules_maker.transformers.windsurf_transformer import WindsurfRuleTransformer
        print("   ✅ Windsurf transformer import successful")
    except Exception as e:
        print(f"   ❌ Windsurf transformer import failed: {e}")
    
    # Test 2: Model creation
    print("\n2. Testing model creation...")
    try:
        from rules_maker.models import ScrapingResult
        result = ScrapingResult(
            url="https://docs.python.org/3/", 
            title="Python Documentation",
            content="Python FastAPI tutorial"
        )
        print("   ✅ ScrapingResult creation successful")
    except Exception as e:
        print(f"   ❌ ScrapingResult creation failed: {e}")
    
    # Test 3: Basic transformation
    print("\n3. Testing transformations...")
    try:
        transformer = CursorRuleTransformer()
        rules = transformer.transform([result])
        print(f"   ✅ Cursor transformation successful ({len(rules)} chars)")
        print(f"   Sample: {rules[:100]}...")
    except Exception as e:
        print(f"   ❌ Cursor transformation failed: {e}")
    
    try:
        transformer = WindsurfRuleTransformer()
        rules = transformer.transform([result])
        print(f"   ✅ Windsurf transformation successful ({len(rules)} chars)")
        print(f"   Sample: {rules[:100]}...")
    except Exception as e:
        print(f"   ❌ Windsurf transformation failed: {e}")

    # Test 4: File structure check
    print("\n4. Checking file structure...")
    
    files_to_check = [
        'src/rules_maker/transformers/cursor_transformer.py',
        'src/rules_maker/transformers/windsurf_transformer.py',
        'src/rules_maker/models.py',
        'cursor_rules.md',
        'docs/PHASE1_COMPLETE.md',
        'PHASE1_SUMMARY.md'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} exists")
        else:
            print(f"   ❌ {file_path} missing")

    print("\n=== Status Check Complete ===")

if __name__ == "__main__":
    test_basic_functionality()
