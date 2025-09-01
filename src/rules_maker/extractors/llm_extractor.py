"""
LLM-powered content extractor.

Uses Language Models to intelligently extract and understand
documentation content for rule generation.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from bs4 import BeautifulSoup
import httpx

from .base import ContentExtractor
from ..models import ContentSection, Rule, RuleSet


logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    BEDROCK = "bedrock"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 30
    region: Optional[str] = None  # For AWS Bedrock
    aws_access_key_id: Optional[str] = None  # For AWS Bedrock
    aws_secret_access_key: Optional[str] = None  # For AWS Bedrock
    aws_session_token: Optional[str] = None  # For AWS Bedrock


class LLMContentExtractor(ContentExtractor):
    """LLM-powered content extraction and rule generation."""
    
    def __init__(
        self, 
        patterns: Optional[List] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        """Initialize the LLM extractor."""
        super().__init__(patterns)
        
        self.config = llm_config or LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo"
        )
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        
        # Token usage/cost tracking
        self._usage: Dict[str, Dict[str, float]] = {
            'prompt_tokens': 0.0,
            'completion_tokens': 0.0,
            'total_tokens': 0.0,
            'input_tokens': 0.0,   # Anthropic naming
            'output_tokens': 0.0,  # Anthropic naming
            'estimated_cost_usd': 0.0,
            'requests': 0.0
        }
        # Simple price map (USD per 1K tokens). These are placeholders; adjust as needed.
        self._price_map: Dict[str, Dict[str, float]] = {
            'openai': {
                'gpt-3.5-turbo':  {'input': 0.0005, 'output': 0.0015},
                'gpt-4o-mini':    {'input': 0.00015, 'output': 0.0006},
                'gpt-4o':         {'input': 0.005,  'output': 0.015},
                'gpt-4-turbo':    {'input': 0.01,   'output': 0.03},
            },
            'anthropic': {
                'claude-3-haiku':   {'input': 0.00025, 'output': 0.00125},
                'claude-3-sonnet':  {'input': 0.003,   'output': 0.015},
                'claude-3-opus':    {'input': 0.015,   'output': 0.075},
            },
            'bedrock': {
                'amazon.nova-lite-v1:0':     {'input': 0.00006, 'output': 0.00024},
                'amazon.nova-micro-v1:0':    {'input': 0.000035, 'output': 0.00014},
                'amazon.nova-pro-v1:0':      {'input': 0.0008, 'output': 0.0032},
                'anthropic.claude-3-sonnet-20240229-v1:0': {'input': 0.003, 'output': 0.015},
                'anthropic.claude-3-haiku-20240307-v1:0':  {'input': 0.00025, 'output': 0.00125},
            }
        }
        
        # Prompt templates
        self.extraction_prompt = self._load_extraction_prompt()
        self.rule_generation_prompt = self._load_rule_generation_prompt()
        
    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured content using LLM analysis."""
        try:
            # Get clean text content
            content_text = self._extract_clean_text(soup)
            
            # Use LLM to analyze and structure content
            structured_content = self._analyze_content_with_llm(content_text, url)
            
            # Extract sections using LLM
            sections = self.extract_sections(soup, url)
            
            return {
                'title': structured_content.get('title', ''),
                'content': content_text,
                'sections': [section.dict() for section in sections],
                'document_type': structured_content.get('document_type', 'unknown'),
                'key_concepts': structured_content.get('key_concepts', []),
                'technologies': structured_content.get('technologies', []),
                'complexity_level': structured_content.get('complexity_level', 'intermediate'),
                'summary': structured_content.get('summary', ''),
                'metadata': structured_content.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Error in LLM extraction for {url}: {str(e)}")
            return self._fallback_extraction(soup, url)
    
    def extract_sections(self, soup: BeautifulSoup, url: str) -> List[ContentSection]:
        """Extract content sections using LLM understanding."""
        try:
            # Get structured content
            content_text = self._extract_clean_text(soup)
            
            # Use LLM to identify and categorize sections
            sections_data = self._extract_sections_with_llm(content_text, url)
            
            sections = []
            for section_data in sections_data:
                section = ContentSection(
                    title=section_data.get('title', ''),
                    content=section_data.get('content', ''),
                    level=section_data.get('level', 1),
                    url=url,
                    metadata={
                        'section_type': section_data.get('type', 'general'),
                        'importance': section_data.get('importance', 0.5),
                        'concepts': section_data.get('concepts', []),
                        'code_examples': section_data.get('code_examples', []),
                        'llm_generated': True
                    }
                )
                sections.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in LLM section extraction for {url}: {str(e)}")
            return self._fallback_section_extraction(soup, url)
    
    async def generate_rules(
        self, 
        content: Union[str, List[ContentSection]], 
        target_format: str = "cursor",
        context: Optional[Dict[str, Any]] = None
    ) -> RuleSet:
        """Generate coding rules from extracted content."""
        try:
            # Prepare content for rule generation
            if isinstance(content, list):
                content_text = self._sections_to_text(content)
            else:
                content_text = content
            
            # Generate rules using LLM
            rules_data = await self._generate_rules_with_llm(
                content_text, target_format, context
            )
            
            # Convert to Rule objects
            rules = []
            for rule_data in rules_data.get('rules', []):
                rule = Rule(
                    id=rule_data.get('id', f"rule_{len(rules)}"),
                    title=rule_data.get('title', ''),
                    description=rule_data.get('description', ''),
                    category=rule_data.get('category', 'general'),
                    priority=rule_data.get('priority', 1),
                    tags=rule_data.get('tags', []),
                    examples=rule_data.get('examples', []),
                    anti_patterns=rule_data.get('anti_patterns', []),
                    metadata={
                        'generated_by': 'llm',
                        'confidence': rule_data.get('confidence', 0.5),
                        'source_url': context.get('url') if context else None
                    }
                )
                rules.append(rule)
            
            return RuleSet(
                name=rules_data.get('name', 'Generated Rules'),
                description=rules_data.get('description', ''),
                rules=rules,
                metadata={
                    'generated_by': 'llm',
                    'model': self.config.model_name,
                    'target_format': target_format,
                    'context': context or {}
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating rules: {str(e)}")
            return RuleSet(name="Error", description=f"Failed to generate rules: {str(e)}", rules=[])
    
    async def _make_llm_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a request to the configured LLM provider."""
        if self.config.provider == LLMProvider.OPENAI:
            return await self._openai_request(prompt, system_prompt)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_request(prompt, system_prompt)
        elif self.config.provider == LLMProvider.HUGGINGFACE:
            return await self._huggingface_request(prompt, system_prompt)
        elif self.config.provider == LLMProvider.BEDROCK:
            return await self._bedrock_request(prompt, system_prompt)
        elif self.config.provider == LLMProvider.LOCAL:
            return await self._local_request(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def _openai_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        response = await self.client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        # Track usage/cost
        usage = result.get('usage', {})
        if usage:
            self._record_usage('openai', self.config.model_name, 
                               prompt_tokens=usage.get('prompt_tokens', 0),
                               completion_tokens=usage.get('completion_tokens', 0),
                               total_tokens=usage.get('total_tokens', 0))
        content = result["choices"][0]["message"]["content"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    async def _anthropic_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "model": self.config.model_name or "claude-3-sonnet-20240229",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": full_prompt}]
        }
        
        response = await self.client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        # Track usage/cost (Anthropic)
        if 'usage' in result:
            u = result['usage']
            self._record_usage('anthropic', self.config.model_name, 
                               input_tokens=u.get('input_tokens', 0),
                               output_tokens=u.get('output_tokens', 0))
        content = result["content"][0]["text"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    async def _huggingface_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to Hugging Face API."""
        url = f"https://api-inference.huggingface.co/models/{self.config.model_name}"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens
            }
        }
        
        response = await self.client.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    async def _bedrock_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to AWS Bedrock."""
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 is required for Bedrock. Install with: pip install boto3")
        
        # Setup AWS session with credentials from config or environment
        session_kwargs = {}
        if self.config.aws_access_key_id:
            session_kwargs['aws_access_key_id'] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = self.config.aws_secret_access_key  
        if self.config.aws_session_token:
            session_kwargs['aws_session_token'] = self.config.aws_session_token
        
        session = boto3.Session(**session_kwargs)
        region = self.config.region or "us-east-1"
        client = session.client('bedrock-runtime', region_name=region)
        
        # Prepare messages
        messages = []
        if system_prompt:
            # For models that support system messages, we'll include it in the conversation
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
            
        messages = [{"role": "user", "content": [{"text": full_prompt}]}]
        
        # Make the request
        model_id = self.config.model_name or "amazon.nova-lite-v1:0"
        
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={
                    'maxTokens': self.config.max_tokens,
                    'temperature': self.config.temperature
                }
            )
            
            # Extract content from response
            output = response.get('output', {}).get('message', {}).get('content', [])
            content = ''
            if output and isinstance(output, list):
                texts = [c.get('text', '') for c in output if isinstance(c, dict)]
                content = '\n'.join(texts).strip()
            
            # Track usage
            usage = response.get('usage', {})
            if usage:
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                self._record_usage('bedrock', model_id,
                                   input_tokens=input_tokens,
                                   output_tokens=output_tokens)
            
            # Try to parse as JSON, fallback to plain content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"content": content}
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"Bedrock ClientError {error_code}: {error_message}")
            raise RuntimeError(f"Bedrock request failed: {error_code} - {error_message}")
        except Exception as e:
            logger.error(f"Bedrock request error: {str(e)}")
            raise
    
    async def _local_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make request to local LLM server."""
        if not self.config.base_url:
            raise ValueError("base_url required for local LLM provider")
        
        url = f"{self.config.base_url}/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        response = await self.client.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    
    def _analyze_content_with_llm(self, content: str, url: str) -> Dict[str, Any]:
        """Analyze content using LLM to extract structured information."""
        prompt = self.extraction_prompt.format(
            content=content[:4000],  # Limit content length
            url=url
        )
        
        # Synchronous wrapper for async function
        import asyncio
        
        async def _async_analyze():
            return await self._make_llm_request(prompt)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_analyze())
                    return future.result()
            else:
                return loop.run_until_complete(_async_analyze())
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {}
    
    def _extract_sections_with_llm(self, content: str, url: str) -> List[Dict[str, Any]]:
        """Extract sections using LLM."""
        prompt = f"""
        Analyze this documentation content and identify distinct sections:
        
        Content: {content[:3000]}
        URL: {url}
        
        Return a JSON array of sections with this structure:
        {{
            "title": "section title",
            "content": "section content", 
            "type": "introduction|installation|tutorial|api|examples|configuration|troubleshooting",
            "level": 1-6,
            "importance": 0.0-1.0,
            "concepts": ["concept1", "concept2"],
            "code_examples": ["example1", "example2"]
        }}
        """
        
        try:
            import asyncio
            async def _async_extract():
                return await self._make_llm_request(prompt)
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_extract())
                    result = future.result()
            else:
                result = loop.run_until_complete(_async_extract())
            
            if isinstance(result, dict) and 'sections' in result:
                return result['sections']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            logger.error(f"LLM section extraction failed: {str(e)}")
            return []
    
    async def _generate_rules_with_llm(
        self, 
        content: str, 
        target_format: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate coding rules using LLM."""
        prompt = self.rule_generation_prompt.format(
            content=content[:4000],
            target_format=target_format,
            context=json.dumps(context or {}, indent=2)
        )
        
        try:
            result = await self._make_llm_request(prompt)
            return result
        except Exception as e:
            logger.error(f"LLM rule generation failed: {str(e)}")
            return {"rules": []}
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and normalize whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _sections_to_text(self, sections: List[ContentSection]) -> str:
        """Convert sections to text for processing."""
        text_parts = []
        for section in sections:
            text_parts.append(f"# {section.title}\n{section.content}\n")
        return '\n'.join(text_parts)
    
    def _fallback_extraction(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Fallback extraction without LLM."""
        title = ""
        if soup.title:
            title = soup.title.get_text().strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        
        content = self._extract_clean_text(soup)
        
        return {
            'title': title,
            'content': content,
            'sections': [],
            'document_type': 'unknown',
            'key_concepts': [],
            'technologies': [],
            'complexity_level': 'unknown',
            'summary': '',
            'metadata': {}
        }
    
    def _fallback_section_extraction(self, soup: BeautifulSoup, url: str) -> List[ContentSection]:
        """Fallback section extraction without LLM."""
        sections = []
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            title = heading.get_text().strip()
            level = int(heading.name[1])
            
            # Get content until next heading
            content_parts = []
            current = heading.next_sibling
            
            while current:
                if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                
                if hasattr(current, 'get_text'):
                    text = current.get_text().strip()
                    if text:
                        content_parts.append(text)
                
                current = current.next_sibling
            
            content = '\n'.join(content_parts)
            
            if title and content:
                section = ContentSection(
                    title=title,
                    content=content,
                    level=level,
                    url=url,
                    metadata={'llm_generated': False}
                )
                sections.append(section)
        
        return sections
    
    def _load_extraction_prompt(self) -> str:
        """Load the content extraction prompt template."""
        return """
        Analyze this documentation content and extract structured information:
        
        Content: {content}
        URL: {url}
        
        Return a JSON object with this structure:
        {{
            "title": "document title",
            "document_type": "api|tutorial|guide|reference|installation|troubleshooting",
            "key_concepts": ["concept1", "concept2"],
            "technologies": ["tech1", "tech2"],
            "complexity_level": "beginner|intermediate|advanced",
            "summary": "brief summary of the content",
            "metadata": {{
                "language": "programming language if applicable",
                "framework": "framework name if applicable",
                "version": "version if mentioned"
            }}
        }}
        
        Focus on extracting actionable information that would be useful for generating coding rules.
        """
    
    def _load_rule_generation_prompt(self) -> str:
        """Load the rule generation prompt template."""
        return """
        Generate coding rules from this documentation content for {target_format} format:
        
        Content: {content}
        Context: {context}
        
        Return a JSON object with this structure:
        {{
            "name": "Rule set name",
            "description": "Description of the rule set",
            "rules": [
                {{
                    "id": "unique_rule_id",
                    "title": "Rule title",
                    "description": "Detailed rule description",
                    "category": "category name",
                    "priority": 1-5,
                    "tags": ["tag1", "tag2"],
                    "examples": ["good example 1", "good example 2"],
                    "anti_patterns": ["what to avoid 1", "what to avoid 2"],
                    "confidence": 0.0-1.0
                }}
            ]
        }}
        
        Generate practical, actionable rules that would help developers write better code.
        Focus on best practices, common patterns, and error prevention.
        """
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    # Usage tracking helpers
    def _record_usage(
        self,
        provider_key: str,
        model_name: str,
        prompt_tokens: float = 0.0,
        completion_tokens: float = 0.0,
        total_tokens: float = 0.0,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
    ) -> None:
        # Aggregate token counts
        self._usage['prompt_tokens'] += float(prompt_tokens)
        self._usage['completion_tokens'] += float(completion_tokens)
        self._usage['total_tokens'] += float(total_tokens or (prompt_tokens + completion_tokens))
        self._usage['input_tokens'] += float(input_tokens or prompt_tokens)
        self._usage['output_tokens'] += float(output_tokens or completion_tokens)
        self._usage['requests'] += 1.0
        
        # Estimate cost if we have price info
        provider_prices = self._price_map.get(provider_key.lower()) or {}
        model_prices = provider_prices.get(model_name) or {}
        in_price = model_prices.get('input')
        out_price = model_prices.get('output')
        if in_price is not None and out_price is not None:
            cost = (self._usage['input_tokens'] / 1000.0) * in_price + (self._usage['output_tokens'] / 1000.0) * out_price
            self._usage['estimated_cost_usd'] = cost

    def get_usage_stats(self) -> Dict[str, float]:
        """Return aggregate token usage and estimated cost."""
        # Return a shallow copy to avoid external mutation
        return dict(self._usage)
