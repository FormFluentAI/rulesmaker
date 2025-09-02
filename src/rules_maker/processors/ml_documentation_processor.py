"""
ML-enhanced documentation processor that extends existing DocumentationProcessor.

Adds semantic analysis, technology detection, and content complexity scoring
while preserving all existing functionality.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from .documentation_processor import DocumentationProcessor
from ..learning.pattern_analyzer import SemanticAnalyzer
from ..models import DocumentationStructure

logger = logging.getLogger(__name__)


class MLDocumentationProcessor(DocumentationProcessor):
    """ML-enhanced documentation processor with semantic analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced processor.
        
        Args:
            config: Configuration with ML-specific settings
        """
        super().__init__()
        self.config = config or {}
        
        # Initialize ML components
        try:
            self.semantic_analyzer = SemanticAnalyzer()
            self.ml_enabled = True
            logger.info("ML components initialized successfully")
        except Exception as e:
            logger.warning(f"ML components initialization failed: {e}, falling back to base processor")
            self.semantic_analyzer = None
            self.ml_enabled = False
    
    def process(self, content: str, url: str, metadata: Dict[str, Any] = None) -> DocumentationStructure:
        """Process content with ML enhancement.
        
        Args:
            content: HTML or text content to process
            url: Source URL
            metadata: Additional metadata
            
        Returns:
            DocumentationStructure with ML enhancements
        """
        if metadata is None:
            metadata = {}
            
        # Use existing base processing logic
        base_result = super().process(content, url, metadata)
        
        # Add ML enhancements if available
        if self.ml_enabled and self.semantic_analyzer:
            try:
                ml_enhancements = self._perform_ml_analysis(content, url)
                base_result.metadata.update(ml_enhancements)
                base_result.metadata['ml_enhanced'] = True
                logger.debug(f"ML analysis completed for {url}")
            except Exception as e:
                logger.warning(f"ML analysis failed for {url}: {e}, using base result")
                base_result.metadata['ml_enhanced'] = False
                base_result.metadata['ml_error'] = str(e)
        else:
            base_result.metadata['ml_enhanced'] = False
        
        return base_result
    
    def _perform_ml_analysis(self, content: str, url: str) -> Dict[str, Any]:
        """Perform ML-powered analysis on content.
        
        Args:
            content: Content to analyze
            url: Source URL
            
        Returns:
            Dictionary of ML analysis results
        """
        enhancements = {}
        
        # Semantic keyword extraction
        semantic_keywords = self.semantic_analyzer.extract_semantic_keywords(content)
        enhancements['semantic_keywords'] = semantic_keywords
        
        # Technology stack detection
        detected_technologies = self._detect_technologies(content, url)
        enhancements['detected_technologies'] = detected_technologies
        
        # Content complexity scoring
        complexity_score = self._calculate_content_complexity(content)
        enhancements['content_complexity'] = complexity_score
        
        # Documentation quality assessment
        quality_metrics = self._assess_documentation_quality(content)
        enhancements['quality_metrics'] = quality_metrics
        
        # Add analysis timestamp
        enhancements['ml_analysis_timestamp'] = datetime.now().isoformat()
        
        return enhancements
    
    def _detect_technologies(self, content: str, url: str) -> List[Dict[str, Any]]:
        """Detect technologies mentioned in content.
        
        Args:
            content: Content to analyze
            url: Source URL
            
        Returns:
            List of detected technologies with confidence scores
        """
        technologies = []
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Enhanced technology patterns with ML-informed scoring
        tech_patterns = {
            'react': {
                'patterns': ['react', 'jsx', 'create-react-app', 'react-dom', 'hooks', 'usestate', 'useeffect'],
                'url_indicators': ['react', 'reactjs'],
                'framework_type': 'javascript'
            },
            'vue': {
                'patterns': ['vue.js', 'vue', 'vuejs', 'nuxt', 'composition api', 'vue-router'],
                'url_indicators': ['vue', 'vuejs'],
                'framework_type': 'javascript'
            },
            'angular': {
                'patterns': ['angular', 'typescript', '@angular', 'ng-', 'angular cli'],
                'url_indicators': ['angular'],
                'framework_type': 'javascript'
            },
            'python': {
                'patterns': ['python', 'pip', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
                'url_indicators': ['python', 'pypi'],
                'framework_type': 'python'
            },
            'fastapi': {
                'patterns': ['fastapi', 'pydantic', 'uvicorn', 'async def', 'path parameters'],
                'url_indicators': ['fastapi', 'tiangolo'],
                'framework_type': 'python'
            },
            'nextjs': {
                'patterns': ['next.js', 'nextjs', 'getstaticprops', 'getserversideprops', 'app router'],
                'url_indicators': ['nextjs', 'vercel'],
                'framework_type': 'javascript'
            }
        }
        
        for tech_name, tech_info in tech_patterns.items():
            confidence = 0.0
            matches = []
            
            # Pattern matching in content
            for pattern in tech_info['patterns']:
                pattern_matches = len(re.findall(r'\b' + re.escape(pattern.lower()) + r'\b', content_lower))
                if pattern_matches > 0:
                    confidence += min(pattern_matches * 0.1, 0.5)  # Max 0.5 from content patterns
                    matches.append(pattern)
            
            # URL indicators
            for indicator in tech_info['url_indicators']:
                if indicator in url_lower:
                    confidence += 0.3
                    matches.append(f"url:{indicator}")
            
            # Add technology if confidence threshold met
            if confidence >= 0.2:  # Minimum confidence threshold
                technologies.append({
                    'name': tech_name,
                    'confidence': min(confidence, 1.0),
                    'framework_type': tech_info['framework_type'],
                    'matches': matches,
                    'detection_method': 'ml_enhanced_pattern_matching'
                })
        
        # Sort by confidence (highest first)
        technologies.sort(key=lambda x: x['confidence'], reverse=True)
        
        return technologies
    
    def _calculate_content_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate content complexity metrics.
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        # Basic complexity indicators
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        code_blocks = len(re.findall(r'```|<pre>|<code>', content))
        links = len(re.findall(r'https?://|href=', content))
        
        # Technical complexity indicators
        technical_terms = len(re.findall(r'\b(?:api|function|method|class|variable|parameter|endpoint|authentication|authorization|configuration|deployment)\b', content.lower()))
        
        # Calculate normalized complexity score (0-1 scale)
        complexity_factors = {
            'word_density': min(word_count / 10000, 1.0),  # Normalize to typical doc length
            'code_density': min(code_blocks / 20, 1.0),    # Normalize to typical code examples
            'link_density': min(links / 50, 1.0),          # Normalize to typical link count
            'technical_density': min(technical_terms / 100, 1.0)  # Technical term density
        }
        
        # Weighted complexity score
        weights = {'word_density': 0.2, 'code_density': 0.3, 'link_density': 0.2, 'technical_density': 0.3}
        overall_complexity = sum(complexity_factors[factor] * weight for factor, weight in weights.items())
        
        return {
            'overall_score': round(overall_complexity, 3),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'code_blocks': code_blocks,
            'links': links,
            'technical_terms': technical_terms,
            'complexity_level': 'low' if overall_complexity < 0.3 else 'medium' if overall_complexity < 0.7 else 'high'
        }
    
    def _assess_documentation_quality(self, content: str) -> Dict[str, Any]:
        """Assess documentation quality using ML-informed heuristics.
        
        Args:
            content: Content to assess
            
        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}
        
        # Structure quality indicators
        has_headings = bool(re.search(r'<h[1-6]>|#{1,6}\s', content))
        has_code_examples = bool(re.search(r'```|<pre>|<code>', content))
        has_links = bool(re.search(r'https?://|href=', content))
        
        # Content quality indicators
        has_installation = bool(re.search(r'\b(?:install|installation|setup|getting started)\b', content.lower()))
        has_examples = bool(re.search(r'\b(?:example|usage|tutorial|how to)\b', content.lower()))
        has_api_docs = bool(re.search(r'\b(?:api|endpoint|method|parameter|response)\b', content.lower()))
        
        # Calculate quality scores
        structure_score = sum([has_headings, has_code_examples, has_links]) / 3
        content_score = sum([has_installation, has_examples, has_api_docs]) / 3
        
        # Overall quality assessment
        overall_quality = (structure_score + content_score) / 2
        
        quality_metrics = {
            'overall_quality': round(overall_quality, 3),
            'structure_score': round(structure_score, 3),
            'content_score': round(content_score, 3),
            'quality_indicators': {
                'has_headings': has_headings,
                'has_code_examples': has_code_examples,
                'has_links': has_links,
                'has_installation': has_installation,
                'has_examples': has_examples,
                'has_api_docs': has_api_docs
            },
            'quality_level': 'low' if overall_quality < 0.4 else 'medium' if overall_quality < 0.8 else 'high'
        }
        
        return quality_metrics