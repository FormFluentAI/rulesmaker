#!/usr/bin/env python3
"""
Minimal Bedrock test that avoids complex imports.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_direct_bedrock():
    """Test Bedrock directly without going through the full import chain."""
    print("🧪 Minimal Bedrock Test")
    print("=" * 30)
    
    # Test credential loading first
    print("🔐 Testing credential loading...")
    try:
        from rules_maker.utils.credentials import CredentialManager
        
        manager = CredentialManager()
        
        # Load credentials
        credentials = manager.load_bedrock_credentials_from_csv()
        print(f"✅ Credentials loaded from {credentials.get('source', 'unknown')}")
        
        # Setup environment
        manager.setup_aws_environment(credentials)
        print("✅ AWS environment configured")
        
        # Validate access
        validation = manager.validate_bedrock_access()
        if validation['success']:
            print(f"✅ Bedrock validation successful")
            print(f"🤖 Model: {validation['model_id']}")
            print(f"🌎 Region: {validation['region']}")
            print(f"💬 Response: {validation['response'][:100]}...")
        else:
            print(f"❌ Validation failed: {validation['error']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rule_generation():
    """Test rule generation with minimal imports."""
    print("\n📋 Testing minimal rule generation...")
    
    try:
        # Import only what we need
        from rules_maker.models import ScrapingResult
        from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
        
        # Create test data
        result = ScrapingResult(
            url="https://example.com",
            title="Test Documentation",
            content="Python is a programming language. Use functions, classes, and modules for organization. Follow PEP 8 style guidelines."
        )
        
        # Generate rules
        transformer = CursorRuleTransformer()
        rules = transformer.transform([result])
        
        print(f"✅ Rules generated successfully!")
        print(f"📏 Length: {len(rules)} characters")
        print(f"🔍 Preview: {rules[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run minimal tests."""
    print("🧪 Minimal Bedrock Integration Test")
    print("=" * 40)
    
    # Check for credentials file
    creds_path = Path(__file__).parent.parent / "docs" / "plans" / "bedrock-long-term-api-key.csv"
    if not creds_path.exists():
        print(f"❌ Credentials file not found: {creds_path}")
        return
    
    print(f"✅ Found credentials file")
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    if test_direct_bedrock():
        tests_passed += 1
    
    if test_rule_generation():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 Minimal tests passed! Basic functionality is working.")
        print("\nYou can now use Bedrock integration like this:")
        print("```python")
        print("from rules_maker.bedrock_integration import quick_cursor_rules")
        print("rules = quick_cursor_rules('Your documentation content here')")
        print("```")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()