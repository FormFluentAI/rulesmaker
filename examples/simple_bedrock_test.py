#!/usr/bin/env python3
"""
Simple test script to verify Bedrock integration works.

This script does a minimal test of the Bedrock integration.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_credentials():
    """Test credential loading and setup."""
    print("🔐 Testing credential management...")
    
    try:
        from rules_maker.utils.credentials import setup_bedrock_credentials
        
        # Setup credentials from CSV
        result = setup_bedrock_credentials()
        
        print(f"✅ Credentials loaded: {result['credentials_loaded']}")
        print(f"📍 Source: {result['credentials_source']}")
        
        # Check validation
        validation = result['validation']
        if validation['success']:
            print(f"✅ Bedrock validation successful")
            print(f"🤖 Model: {validation['model_id']}")
            print(f"🌎 Region: {validation['region']}")
            print(f"💬 Response: {validation['response']}")
            if 'usage' in validation:
                usage = validation['usage']
                print(f"📊 Tokens - Input: {usage.get('inputTokens', 0)}, Output: {usage.get('outputTokens', 0)}")
        else:
            print(f"❌ Validation failed: {validation['error']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Credential test failed: {e}")
        return False


def test_rule_generation():
    """Test basic rule generation."""
    print("\n📋 Testing rule generation...")
    
    try:
        from rules_maker.bedrock_integration import quick_cursor_rules
        
        # Simple documentation content
        docs = """
        React is a JavaScript library for building user interfaces.
        
        Key concepts:
        - Components are the building blocks
        - Use JSX for templating
        - State management with useState
        - Effects with useEffect
        
        Example:
        ```jsx
        function App() {
            const [count, setCount] = useState(0);
            return <div onClick={() => setCount(count + 1)}>{count}</div>;
        }
        ```
        """
        
        # Generate rules
        rules = quick_cursor_rules(docs)
        
        print(f"✅ Rules generated successfully!")
        print(f"📏 Length: {len(rules)} characters")
        print(f"🔍 Preview: {rules[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule generation test failed: {e}")
        return False


def main():
    """Run simple tests."""
    print("🧪 Simple Bedrock Integration Test")
    print("=" * 40)
    
    # Check for credentials file
    creds_path = Path(__file__).parent.parent / "docs" / "plans" / "bedrock-long-term-api-key.csv"
    if not creds_path.exists():
        print(f"❌ Credentials file not found: {creds_path}")
        print("Please ensure your Bedrock credentials CSV exists.")
        return
    
    print(f"✅ Found credentials file")
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    if test_credentials():
        tests_passed += 1
    
    if test_rule_generation():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Bedrock integration is working.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Verify boto3 is installed: pip install boto3")
        print("2. Check credentials CSV format and content")
        print("3. Ensure AWS region supports your chosen model")
        print("4. Verify Bedrock permissions in your AWS account")


if __name__ == "__main__":
    main()