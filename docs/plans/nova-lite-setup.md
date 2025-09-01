# Amazon Nova Lite Setup Guide with Python

## Overview
Amazon Nova Lite (`amazon.nova-lite-v1:0`) is a multimodal AI model available through Amazon Bedrock that supports text, image, and video inputs with text outputs. This guide shows you how to set it up and use it with Python.

## Prerequisites
- AWS Account with Amazon Bedrock access
- Python 3.7+ installed
- AWS credentials (you have temporary session credentials)

## Step 1: Install Required Libraries
```bash
pip install boto3
```

## Step 2: Set Up AWS Credentials

Since you have temporary session credentials, you need to configure all three values:

### Method 1: Environment Variables (Recommended)
```bash
# Linux/Mac
export AWS_ACCESS_KEY_ID="ASIAS74TMLK7TUG2Q35Z"
export AWS_SECRET_ACCESS_KEY="as+iu3pslvPTy+XJpe7zNxS+VYqIl6KIWR1oc40o"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjELP//////////wEaDGV1LWNlbnRyYWwtMSJIMEYCIQCO6UyVzZvFjNlwDV4Nm..."
export AWS_REGION="eu-central-1"

# Windows
set AWS_ACCESS_KEY_ID=ASIAS74TMLK7TUG2Q35Z
set AWS_SECRET_ACCESS_KEY=as+iu3pslvPTy+XJpe7zNxS+VYqIl6KIWR1oc40o
set AWS_SESSION_TOKEN=IQoJb3JpZ2luX2VjELP//////////wEaDGV1LWNlbnRyYWwtMSJIMEYCIQCO6UyVzZvFjNlwDV4Nm...
set AWS_REGION=eu-central-1
```

### Method 2: Direct Configuration in Python
```python
import boto3

session = boto3.Session(
    aws_access_key_id='ASIAS74TMLK7TUG2Q35Z',
    aws_secret_access_key='as+iu3pslvPTy+XJpe7zNxS+VYqIl6KIWR1oc40o',
    aws_session_token='IQoJb3JpZ2luX2VjELP//////////wEaDGV1LWNlbnRyYWwtMSJIMEYCIQCO6UyVzZvFjNlwDV4Nm...',
    region_name='eu-central-1'
)
client = session.client('bedrock-runtime')
```

## Step 3: Request Model Access
1. Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Ensure you're in a supported region (us-east-1 for Nova models)
3. Click "Model access" in the left sidebar
4. Click "Manage model access"
5. Select Amazon Nova models
6. Click "Request access" (usually approved instantly)

## Step 4: Basic Usage Examples

### Example 1: Simple Text Generation
```python
import boto3
import json

# Create Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Model configuration
model_id = "amazon.nova-lite-v1:0"

# Create conversation
conversation = [
    {
        "role": "user",
        "content": [{"text": "Explain machine learning in simple terms"}]
    }
]

# Make the API call
response = client.converse(
    modelId=model_id,
    messages=conversation,
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.7,
        "topP": 0.9
    }
)

# Extract and print response
response_text = response["output"]["message"]["content"][0]["text"]
print(response_text)
```

### Example 2: Streaming Response
```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.nova-lite-v1:0"

conversation = [
    {
        "role": "user",
        "content": [{"text": "Write a short story about AI"}]
    }
]

# Stream the response
streaming_response = client.converse_stream(
    modelId=model_id,
    messages=conversation,
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.7,
        "topP": 0.9
    }
)

# Process streamed response
for chunk in streaming_response["stream"]:
    if "contentBlockDelta" in chunk:
        text = chunk["contentBlockDelta"]["delta"]["text"]
        print(text, end="")
```

### Example 3: Complete Chatbot Class
```python
import boto3
import json
from typing import List, Dict

class NovaLiteChatbot:
    def __init__(self, region_name="us-east-1"):
        """Initialize the Nova Lite chatbot."""
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = "amazon.nova-lite-v1:0"
        self.conversation_history = []
        
    def chat(self, user_input: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Send a message and get response."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": [{"text": user_input}]
        })
        
        try:
            # Make API call
            response = self.client.converse(
                modelId=self.model_id,
                messages=self.conversation_history,
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            )
            
            # Extract response
            assistant_message = response["output"]["message"]["content"][0]["text"]
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": [{"text": assistant_message}]
            })
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self):
        """Get current conversation history."""
        return self.conversation_history

# Usage example
if __name__ == "__main__":
    chatbot = NovaLiteChatbot()
    
    print("Nova Lite Chatbot initialized!")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_history()
            print("Conversation cleared!")
            continue
        elif not user_input:
            continue
            
        response = chatbot.chat(user_input)
        print(f"\nAssistant: {response}")
```

## Step 5: Advanced Features

### Working with Images
```python
import boto3
import base64

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Include image in conversation
image_data = encode_image("path/to/your/image.jpg")

conversation = [
    {
        "role": "user",
        "content": [
            {"text": "What's in this image?"},
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": base64.b64decode(image_data)}
                }
            }
        ]
    }
]

response = client.converse(
    modelId="amazon.nova-lite-v1:0",
    messages=conversation,
    inferenceConfig={"maxTokens": 512, "temperature": 0.7}
)

print(response["output"]["message"]["content"][0]["text"])
```

## Troubleshooting

### Common Issues:

1. **Region Availability**: Nova Lite is currently only available in `us-east-1`
2. **Model Access**: Ensure you've requested access to Nova models in Bedrock console
3. **Session Token**: Temporary credentials require all three values (access key, secret key, session token)
4. **Timeout**: For long responses, increase boto3 timeout settings

### Error Solutions:

```python
# For timeout issues
import botocore

config = botocore.config.Config(
    read_timeout=900,
    retries={'max_attempts': 3}
)

client = boto3.client(
    "bedrock-runtime", 
    region_name="us-east-1",
    config=config
)
```

## Next Steps

1. Explore multimodal capabilities (text + image + video)
2. Implement streaming for better user experience
3. Add error handling and retry logic
4. Integrate with web frameworks (FastAPI, Flask)
5. Build production applications with proper authentication

## Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Nova Models User Guide](https://docs.aws.amazon.com/nova/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [AWS SDK Examples](https://github.com/awsdocs/aws-doc-sdk-examples)