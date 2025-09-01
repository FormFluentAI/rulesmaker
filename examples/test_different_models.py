#!/usr/bin/env python3
"""
Test different Bedrock models to see which ones are available.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_models():
    """Test different Bedrock models."""
    print("🧪 Testing Different Bedrock Models")
    print("=" * 40)
    
    models_to_test = [
        "amazon.nova-micro-v1:0",
        "amazon.nova-lite-v1:0", 
        "amazon.nova-pro-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0"
    ]
    
    from rules_maker.utils.credentials import get_credential_manager
    
    try:
        manager = get_credential_manager()
        manager.setup_aws_environment()
        
        print("Testing model availability...")
        working_models = []
        
        for model in models_to_test:
            print(f"\n🤖 Testing {model}...")
            validation = manager.validate_bedrock_access(model_id=model)
            
            if validation['success']:
                print(f"✅ {model} - WORKING")
                print(f"   Response: {validation['response'][:100]}...")
                working_models.append(model)
            else:
                error = validation['error']
                if 'AccessDeniedException' in error:
                    print(f"🔒 {model} - ACCESS DENIED (not enabled in your account)")
                elif 'ValidationException' in error:
                    print(f"❌ {model} - NOT AVAILABLE (invalid model ID)")
                else:
                    print(f"⚠️ {model} - ERROR: {error}")
        
        print(f"\n📊 Summary:")
        print(f"✅ Working models: {len(working_models)}")
        for model in working_models:
            print(f"   - {model}")
        
        if working_models:
            print(f"\n💡 Recommendation: Use {working_models[0]} for best results")
            print("\nUpdate your code to use a working model:")
            print("```python")
            print("from rules_maker.bedrock_integration import BedrockRulesMaker")
            print(f'maker = BedrockRulesMaker(model_id="{working_models[0]}")')
            print("```")
        else:
            print("\n⚠️ No models are working. Please:")
            print("1. Check that you have Bedrock access in your AWS account")
            print("2. Enable model access in the AWS Bedrock console")
            print("3. Verify your region supports the models you want to use")
            print("4. Check that your credentials have the correct permissions")
            
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your credentials CSV file exists and is readable")
        print("2. Verify your AWS credentials have Bedrock permissions") 
        print("3. Ensure you're in a region that supports Bedrock")

if __name__ == "__main__":
    test_models()