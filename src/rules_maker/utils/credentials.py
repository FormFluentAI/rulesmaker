"""
Credential management utilities for Rules Maker.

Handles AWS Bedrock credentials and other API key management.
"""

import os
import csv
import base64
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class CredentialManager:
    """Manages credentials for various services."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize credential manager."""
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self._credentials_cache = {}
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def load_bedrock_credentials_from_csv(self, csv_path: Optional[str] = None) -> Dict[str, str]:
        """
        Load Bedrock credentials from CSV file.
        
        Args:
            csv_path: Path to CSV file. If None, looks for docs/plans/bedrock-long-term-api-key.csv
            
        Returns:
            Dict with AWS credentials
        """
        if csv_path is None:
            csv_path = self.project_root / "docs" / "plans" / "bedrock-long-term-api-key.csv"
        else:
            csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Bedrock credentials CSV not found: {csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if not rows:
                raise ValueError("Empty credentials CSV file")
            
            # Get the first row (assuming single credential set)
            cred_row = rows[0]
            
            # Extract credentials
            api_key_name = cred_row.get('API key name', '')
            api_key = cred_row.get('API key', '')
            
            if not api_key:
                raise ValueError("No API key found in CSV")
            
            # Check if this is a composite key format: name:encoded_credentials
            if ':' in api_key:
                key_parts = api_key.split(':', 1)
                if len(key_parts) == 2:
                    try:
                        # Decode base64 credentials
                        decoded_creds = base64.b64decode(key_parts[1]).decode('utf-8')
                        # Parse the decoded credentials (format may vary)
                        return self._parse_aws_credentials(decoded_creds, api_key_name)
                    except Exception as e:
                        logger.warning(f"Failed to decode credentials: {e}")
            
            # Check if this is a base64-encoded credential set
            try:
                if len(api_key) > 50 and api_key.isalnum() or '+' in api_key or '/' in api_key:
                    # Try to decode as base64
                    decoded_creds = base64.b64decode(api_key).decode('utf-8')
                    return self._parse_aws_credentials(decoded_creds, api_key_name)
            except Exception:
                pass
            
            # Check if this looks like an AWS access key ID (starts with AKIA, ASIA, etc.)
            if api_key.startswith(('AKIA', 'ASIA', 'AROA', 'AIPA', 'ANPA', 'ANVA', 'AGPA')):
                logger.warning("Found AWS Access Key ID but no Secret Key. You may need to provide additional credentials.")
                return {
                    'aws_access_key_id': api_key,
                    'api_key_name': api_key_name,
                    'source': 'csv_access_key_only'
                }
            
            # Fallback: treat as direct access key (may not work without secret)
            return {
                'aws_access_key_id': api_key,
                'api_key_name': api_key_name,
                'source': 'csv_raw',
                'warning': 'No AWS Secret Access Key provided - authentication may fail'
            }
            
        except Exception as e:
            logger.error(f"Failed to load Bedrock credentials from {csv_path}: {e}")
            raise
    
    def _parse_aws_credentials(self, decoded_creds: str, key_name: str) -> Dict[str, str]:
        """
        Parse decoded AWS credentials.
        
        This method handles various credential formats that might be returned
        by the AWS credential generation process.
        """
        # Try to parse as JSON first
        try:
            import json
            creds_data = json.loads(decoded_creds)
            if isinstance(creds_data, dict):
                return {
                    'aws_access_key_id': creds_data.get('AccessKeyId', ''),
                    'aws_secret_access_key': creds_data.get('SecretAccessKey', ''),
                    'aws_session_token': creds_data.get('SessionToken', ''),
                    'api_key_name': key_name,
                    'source': 'csv_json'
                }
        except:
            pass
        
        # Try to parse as key-value pairs
        cred_dict = {}
        for line in decoded_creds.split('\n'):
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                cred_dict[key.strip()] = value.strip()
        
        if cred_dict:
            return {
                'aws_access_key_id': cred_dict.get('AWS_ACCESS_KEY_ID', ''),
                'aws_secret_access_key': cred_dict.get('AWS_SECRET_ACCESS_KEY', ''),
                'aws_session_token': cred_dict.get('AWS_SESSION_TOKEN', ''),
                'api_key_name': key_name,
                'source': 'csv_keyvalue'
            }
        
        # Fallback: return raw credentials
        return {
            'aws_access_key_id': decoded_creds,
            'api_key_name': key_name,
            'source': 'csv_raw'
        }
    
    def setup_aws_environment(self, credentials: Optional[Dict[str, str]] = None) -> None:
        """
        Set up AWS environment variables from credentials.
        
        Args:
            credentials: AWS credentials dict. If None, loads from CSV.
        """
        if credentials is None:
            credentials = self.load_bedrock_credentials_from_csv()
        
        # Set AWS environment variables
        if credentials.get('aws_access_key_id'):
            os.environ['AWS_ACCESS_KEY_ID'] = credentials['aws_access_key_id']
        
        if credentials.get('aws_secret_access_key'):
            os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['aws_secret_access_key']
        
        if credentials.get('aws_session_token'):
            os.environ['AWS_SESSION_TOKEN'] = credentials['aws_session_token']
        
        # Set default region if not already set
        if not os.environ.get('AWS_REGION'):
            os.environ['AWS_REGION'] = 'us-east-1'  # Default region for Bedrock
        
        # Set default model if not already set
        if not os.environ.get('BEDROCK_MODEL_ID'):
            os.environ['BEDROCK_MODEL_ID'] = 'amazon.nova-lite-v1:0'
        
        logger.info(f"AWS environment configured with credentials from {credentials.get('source', 'unknown')}")
    
    def get_aws_session_info(self) -> Dict[str, str]:
        """Get information about current AWS session."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            session = boto3.Session()
            
            # Get credentials info
            creds = session.get_credentials()
            creds_info = {}
            if creds:
                frozen = creds.get_frozen_credentials()
                creds_info = {
                    'access_key': getattr(frozen, 'access_key', None),
                    'has_secret': bool(getattr(frozen, 'secret_key', None)),
                    'has_token': bool(getattr(frozen, 'token', None))
                }
            
            # Get caller identity
            try:
                sts = session.client('sts')
                identity = sts.get_caller_identity()
                creds_info.update({
                    'user_id': identity.get('UserId', ''),
                    'account': identity.get('Account', ''),
                    'arn': identity.get('Arn', '')
                })
            except ClientError as e:
                creds_info['sts_error'] = str(e)
            
            return creds_info
            
        except ImportError:
            return {'error': 'boto3 not installed'}
        except Exception as e:
            return {'error': str(e)}
    
    def validate_bedrock_access(self, model_id: str = None, region: str = None) -> Dict[str, any]:
        """
        Validate that Bedrock access is working.
        
        Args:
            model_id: Model to test (defaults to amazon.nova-lite-v1:0)
            region: AWS region (defaults to us-east-1)
            
        Returns:
            Dict with validation results
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            model_id = model_id or os.environ.get('BEDROCK_MODEL_ID', 'amazon.nova-lite-v1:0')
            region = region or os.environ.get('AWS_REGION', 'us-east-1')
            
            # Create Bedrock client
            session = boto3.Session()
            client = session.client('bedrock-runtime', region_name=region)
            
            # Test with minimal request
            response = client.converse(
                modelId=model_id,
                messages=[{'role': 'user', 'content': [{'text': 'Hello'}]}],
                inferenceConfig={'maxTokens': 10, 'temperature': 0.0}
            )
            
            # Extract response
            output = response.get('output', {}).get('message', {}).get('content', [])
            response_text = ''
            if output and isinstance(output, list):
                texts = [c.get('text', '') for c in output if isinstance(c, dict)]
                response_text = ' '.join(texts).strip()
            
            return {
                'success': True,
                'model_id': model_id,
                'region': region,
                'response': response_text,
                'usage': response.get('usage', {}),
                'endpoint': client.meta.endpoint_url
            }
            
        except ClientError as e:
            return {
                'success': False,
                'error': str(e),
                'error_code': e.response.get('Error', {}).get('Code', 'Unknown'),
                'model_id': model_id,
                'region': region
            }
        except ImportError:
            return {
                'success': False,
                'error': 'boto3 not installed',
                'install_command': 'pip install boto3'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id,
                'region': region
            }


# Singleton instance for easy access
_credential_manager = None

def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager


def setup_bedrock_credentials(csv_path: Optional[str] = None) -> Dict[str, str]:
    """
    Quick setup function for Bedrock credentials.
    
    Args:
        csv_path: Path to credentials CSV file
        
    Returns:
        Dict with validation results
    """
    manager = get_credential_manager()
    
    # Load and setup credentials
    credentials = manager.load_bedrock_credentials_from_csv(csv_path)
    manager.setup_aws_environment(credentials)
    
    # Validate access
    validation = manager.validate_bedrock_access()
    
    return {
        'credentials_loaded': True,
        'credentials_source': credentials.get('source', 'unknown'),
        'validation': validation
    }