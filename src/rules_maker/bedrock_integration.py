"""
Bedrock integration helpers for Rules Maker.

Provides easy-to-use functions for setting up and using AWS Bedrock
with Rules Maker transformers and extractors.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .utils.credentials import setup_bedrock_credentials, get_credential_manager
from .extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider
from .transformers.cursor_transformer import CursorRuleTransformer  
from .transformers.windsurf_transformer import WindsurfRuleTransformer
from .models import ScrapingResult

logger = logging.getLogger(__name__)


class BedrockRulesMaker:
    """Main interface for using Rules Maker with AWS Bedrock."""
    
    def __init__(
        self,
        model_id: str = "amazon.nova-lite-v1:0",
        region: str = "us-east-1",
        credentials_csv_path: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Bedrock Rules Maker.
        
        Args:
            model_id: Bedrock model ID (e.g., amazon.nova-lite-v1:0)
            region: AWS region
            credentials_csv_path: Path to CSV file with Bedrock credentials
            temperature: Model temperature for generation
            max_tokens: Maximum tokens to generate
        """
        # Keep raw config for downstream components
        self._config = config if isinstance(config, dict) else None
        # Allow overriding via config dict and environment
        cfg = (config or {}).get('bedrock') if isinstance(config, dict) else None
        env_model = os.environ.get('BEDROCK_MODEL_ID')
        env_region = os.environ.get('AWS_REGION') or os.environ.get('BEDROCK_REGION')
        env_timeout = os.environ.get('BEDROCK_TIMEOUT')
        env_conc = os.environ.get('BEDROCK_MAX_CONCURRENCY')

        self.model_id = (cfg or {}).get('model_id') or model_id or env_model or "amazon.nova-lite-v1:0"
        self.region = (cfg or {}).get('region') or region or env_region or "us-east-1"
        self.temperature = float((cfg or {}).get('temperature', temperature))
        self.max_tokens = int((cfg or {}).get('max_tokens', max_tokens))
        self.timeout = int((cfg or {}).get('timeout') or (env_timeout or 30))
        self.max_concurrency = int((cfg or {}).get('concurrency') or (env_conc or 4))
        retry_cfg = ((cfg or {}).get('retry') or {})
        self.retry_max_attempts = int(retry_cfg.get('max_attempts', int(os.environ.get('BEDROCK_RETRY_MAX_ATTEMPTS', 3))))
        self.retry_base_ms = int(retry_cfg.get('base_ms', int(os.environ.get('BEDROCK_RETRY_BASE_MS', 250))))
        self.retry_max_ms = int(retry_cfg.get('max_ms', int(os.environ.get('BEDROCK_RETRY_MAX_MS', 2000))))
        
        # Setup credentials
        # Credentials CSV may come from config dict as well
        if not credentials_csv_path and cfg:
            credentials_csv_path = cfg.get('credentials_csv') or None
        if credentials_csv_path:
            credentials_csv_path = str(credentials_csv_path)

        if credentials_csv_path or not self._has_aws_credentials():
            logger.info("Setting up Bedrock credentials from CSV...")
            self.credential_setup = setup_bedrock_credentials(credentials_csv_path)
            if not self.credential_setup['validation']['success']:
                logger.error(f"Bedrock validation failed: {self.credential_setup['validation']['error']}")
                raise RuntimeError("Failed to validate Bedrock access")
            logger.info("✅ Bedrock credentials validated successfully")
        else:
            logger.info("Using existing AWS credentials from environment")
            # Still validate access
            manager = get_credential_manager()
            validation = manager.validate_bedrock_access(self.model_id, self.region)
            if not validation['success']:
                raise RuntimeError(f"Bedrock validation failed: {validation['error']}")
        
        # Setup LLM config for Bedrock
        self.llm_config = LLMConfig(
            provider=LLMProvider.BEDROCK,
            model_name=self.model_id,
            region=self.region,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            retry_max_attempts=self.retry_max_attempts,
            retry_base_ms=self.retry_base_ms,
            retry_max_ms=self.retry_max_ms,
            max_concurrency=self.max_concurrency,
        )
        
        # Initialize components
        self.extractor = LLMContentExtractor(llm_config=self.llm_config, config=self._config or {})
        self.cursor_transformer = CursorRuleTransformer()
        self.windsurf_transformer = WindsurfRuleTransformer()
        
        logger.info(f"BedrockRulesMaker initialized with model: {self.model_id}")
    
    def _has_aws_credentials(self) -> bool:
        """Check if AWS credentials are already available."""
        return bool(
            os.environ.get('AWS_ACCESS_KEY_ID') or 
            os.environ.get('AWS_PROFILE') or
            Path('~/.aws/credentials').expanduser().exists()
        )
    
    def generate_cursor_rules(
        self, 
        documentation_content: str,
        title: str = "Documentation",
        url: str = "https://example.com"
    ) -> str:
        """
        Generate Cursor rules from documentation content.
        
        Args:
            documentation_content: Raw documentation text
            title: Title for the documentation
            url: Source URL
            
        Returns:
            Generated Cursor rules as string
        """
        # Create scraping result
        result = ScrapingResult(
            url=url,
            title=title,
            content=documentation_content
        )
        
        # Generate rules
        return self.cursor_transformer.transform([result])
    
    def generate_windsurf_rules(
        self, 
        documentation_content: str,
        title: str = "Documentation", 
        url: str = "https://example.com"
    ) -> str:
        """
        Generate Windsurf rules from documentation content.
        
        Args:
            documentation_content: Raw documentation text
            title: Title for the documentation
            url: Source URL
            
        Returns:
            Generated Windsurf rules as string
        """
        # Create scraping result
        result = ScrapingResult(
            url=url,
            title=title,
            content=documentation_content
        )
        
        # Generate rules
        return self.windsurf_transformer.transform([result])
    
    async def generate_enhanced_cursor_rules(
        self,
        documentation_content: str,
        title: str = "Documentation",
        url: str = "https://example.com"
    ) -> str:
        """
        Generate enhanced Cursor rules using LLM analysis.
        
        Args:
            documentation_content: Raw documentation text
            title: Title for the documentation
            url: Source URL
            
        Returns:
            Enhanced Cursor rules as string
        """
        # Use LLM extractor for enhanced analysis
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(f"<html><head><title>{title}</title></head><body>{documentation_content}</body></html>", 'html.parser')
        
        # Extract with LLM analysis
        extracted_data = self.extractor.extract(soup, url)
        
        # Create enhanced scraping result  
        result = ScrapingResult(
            url=url,
            title=extracted_data.get('title', title),
            content=extracted_data.get('content', documentation_content),
            sections=extracted_data.get('sections', []),
            metadata=extracted_data.get('metadata', {})
        )
        
        # Generate enhanced rules
        return self.cursor_transformer.transform([result])
    
    async def generate_enhanced_windsurf_rules(
        self,
        documentation_content: str, 
        title: str = "Documentation",
        url: str = "https://example.com"
    ) -> str:
        """
        Generate enhanced Windsurf rules using LLM analysis.
        
        Args:
            documentation_content: Raw documentation text
            title: Title for the documentation
            url: Source URL
            
        Returns:
            Enhanced Windsurf rules as string
        """
        # Use LLM extractor for enhanced analysis
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(f"<html><head><title>{title}</title></head><body>{documentation_content}</body></html>", 'html.parser')
        
        # Extract with LLM analysis
        extracted_data = self.extractor.extract(soup, url)
        
        # Create enhanced scraping result
        result = ScrapingResult(
            url=url,
            title=extracted_data.get('title', title),
            content=extracted_data.get('content', documentation_content),
            sections=extracted_data.get('sections', []),
            metadata=extracted_data.get('metadata', {})
        )
        
        # Generate enhanced rules
        return self.windsurf_transformer.transform([result])
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics from the LLM extractor."""
        stats = self.extractor.get_usage_stats()
        return {
            'model_id': self.model_id,
            'region': self.region,
            'total_requests': stats.get('requests', 0),
            'total_input_tokens': stats.get('input_tokens', 0),
            'total_output_tokens': stats.get('output_tokens', 0),
            'estimated_cost_usd': stats.get('estimated_cost_usd', 0.0)
        }
    
    async def test_bedrock_connection(self) -> Dict[str, Any]:
        """Test Bedrock connection with a simple request."""
        try:
            # Make a simple test request
            test_result = await self.extractor._make_llm_request(
                "Respond with 'Hello from Bedrock!' and nothing else.",
                "You are a helpful assistant."
            )
            
            return {
                'success': True,
                'model_id': self.model_id,
                'region': self.region,
                'response': test_result,
                'usage_stats': self.get_usage_stats()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_id': self.model_id,
                'region': self.region
            }
    
    async def close(self):
        """Close the extractor and cleanup resources."""
        await self.extractor.close()


def create_bedrock_rules_maker(
    credentials_csv_path: Optional[str] = None,
    model_id: str = "amazon.nova-lite-v1:0",
    region: str = "us-east-1",
    **kwargs
) -> BedrockRulesMaker:
    """
    Convenience function to create a BedrockRulesMaker instance.
    
    Args:
        credentials_csv_path: Path to CSV with Bedrock credentials
        model_id: Bedrock model ID
        region: AWS region
        **kwargs: Additional arguments for BedrockRulesMaker
        
    Returns:
        Configured BedrockRulesMaker instance
    """
    return BedrockRulesMaker(
        model_id=model_id,
        region=region,
        credentials_csv_path=credentials_csv_path,
        **kwargs
    )


# Quick usage functions
def quick_cursor_rules(documentation_content: str, **kwargs) -> str:
    """Quick function to generate Cursor rules with Bedrock."""
    maker = create_bedrock_rules_maker(**kwargs)
    return maker.generate_cursor_rules(documentation_content)


def quick_windsurf_rules(documentation_content: str, **kwargs) -> str:
    """Quick function to generate Windsurf rules with Bedrock."""
    maker = create_bedrock_rules_maker(**kwargs)
    return maker.generate_windsurf_rules(documentation_content)


async def quick_enhanced_cursor_rules(documentation_content: str, **kwargs) -> str:
    """Quick function to generate enhanced Cursor rules with Bedrock LLM analysis."""
    maker = create_bedrock_rules_maker(**kwargs)
    try:
        return await maker.generate_enhanced_cursor_rules(documentation_content)
    finally:
        await maker.close()


async def quick_enhanced_windsurf_rules(documentation_content: str, **kwargs) -> str:
    """Quick function to generate enhanced Windsurf rules with Bedrock LLM analysis."""
    maker = create_bedrock_rules_maker(**kwargs)
    try:
        return await maker.generate_enhanced_windsurf_rules(documentation_content)
    finally:
        await maker.close()
