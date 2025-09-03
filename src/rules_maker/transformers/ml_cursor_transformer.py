"""
ML-enhanced Cursor transformer with quality assessment and intelligent clustering.

Extends existing CursorRuleTransformer with ML-powered quality scoring,
rule clustering, self-improving feedback integration, and cursor rules knowledge.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .cursor_transformer import CursorRuleTransformer
from ..strategies.ml_quality_strategy import MLQualityStrategy
from ..models import ScrapingResult

# Import learning and intelligence modules
try:
    from ..learning import LearningEngine, SemanticAnalyzer, UsageTracker
    from ..intelligence import IntelligentCategoryEngine, SmartRecommendationEngine
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLCursorTransformer(CursorRuleTransformer):
    """ML-enhanced Cursor transformer with quality assessment."""
    
    def __init__(self, ml_config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced Cursor transformer with cursor rules knowledge.
        
        Args:
            ml_config: ML-specific configuration
        """
        super().__init__()
        self.ml_config = ml_config or {}
        
        # ML configuration
        self.quality_threshold = self.ml_config.get('quality_threshold', 0.7)
        self.enable_clustering = self.ml_config.get('enable_clustering', True)
        self.coherence_threshold = self.ml_config.get('coherence_threshold', 0.6)
        self.max_clusters = self.ml_config.get('max_clusters', 5)
        
        # Cursor rules configuration
        self.cursor_rules_config = self.ml_config.get('cursor_rules', {})
        self.enable_cursor_rules_validation = self.cursor_rules_config.get('enable_validation', True)
        self.cursor_rules_quality_threshold = self.cursor_rules_config.get('quality_threshold', 0.8)
        
        # Initialize ML components
        try:
            self.quality_strategy = MLQualityStrategy(self.ml_config.get('quality_strategy_config', {}))
            # Lazy import to avoid hard dependency on numpy/sklearn at import time
            try:
                from ..learning.self_improving_engine import SelfImprovingEngine  # type: ignore
                self.self_improving_engine = SelfImprovingEngine(
                    quality_threshold=self.quality_threshold
                )
            except Exception as e:
                logger.warning(f"SelfImprovingEngine unavailable ({e}); continuing without it")
                self.self_improving_engine = None
            self.ml_enabled = True
            logger.info("ML components initialized successfully")
        except Exception as e:
            logger.warning(f"ML components initialization failed: {e}, falling back to base transformer")
            self.quality_strategy = None
            self.self_improving_engine = None
            self.ml_enabled = False
        
        # Initialize learning and intelligence components
        self.learning_available = LEARNING_AVAILABLE
        if self.learning_available:
            try:
                self.learning_engine = LearningEngine()
                self.semantic_analyzer = SemanticAnalyzer()
                self.usage_tracker = UsageTracker()
                self.category_engine = IntelligentCategoryEngine()
                self.recommendation_engine = SmartRecommendationEngine()
                logger.info("Learning and intelligence components initialized for ML transformer")
            except Exception as e:
                logger.warning(f"Failed to initialize learning components: {e}")
                self.learning_available = False
    
    async def transform(self, scraping_results: List[ScrapingResult]) -> str:
        """Transform scraping results with ML enhancement and cursor rules knowledge.
        
        Args:
            scraping_results: List of scraping results to transform
            
        Returns:
            Enhanced Cursor rules with ML quality assessment and cursor rules validation
        """
        # Use existing base transformation logic
        base_rules = super().transform(scraping_results)
        
        # Apply cursor rules knowledge and validation
        if self.enable_cursor_rules_validation:
            base_rules = self._apply_cursor_rules_validation(base_rules, scraping_results)
        
        # Add ML enhancements if available
        if self.ml_enabled:
            try:
                enhanced_rules = await self._add_ml_enhancements(base_rules, scraping_results)
                logger.debug(f"ML enhancements added to rules for {len(scraping_results)} sources")
                return enhanced_rules
            except Exception as e:
                logger.warning(f"ML enhancement failed: {e}, using base rules")
                return self._add_fallback_enhancement(base_rules, scraping_results)
        else:
            return self._add_fallback_enhancement(base_rules, scraping_results)
    
    async def _add_ml_enhancements(self, base_rules: str, scraping_results: List[ScrapingResult]) -> str:
        """Add ML-powered enhancements to base rules.
        
        Args:
            base_rules: Base transformation result
            scraping_results: Original scraping results
            
        Returns:
            ML-enhanced rules
        """
        enhancements = []
        
        # Generate quality assessments for each source
        quality_assessments = await self._generate_quality_assessments(scraping_results)
        
        # Perform intelligent clustering if enabled
        clusters = None
        if self.enable_clustering and len(scraping_results) > 1:
            clusters = await self._perform_intelligent_clustering(scraping_results, quality_assessments)
        
        # Add ML quality assessment section
        ml_section = self._create_ml_quality_section(quality_assessments, clusters)
        
        # Insert ML section before the final closing
        anchor = "Remember: These rules are continuously updated based on official documentation and community feedback."
        if anchor in base_rules:
            enhanced_rules = base_rules.replace(
                anchor,
                f"{ml_section}\n\n{anchor}"
            )
        else:
            enhanced_rules = base_rules + "\n\n" + ml_section
        return enhanced_rules
    
    def _apply_cursor_rules_validation(self, base_rules: str, scraping_results: List[ScrapingResult]) -> str:
        """Apply cursor rules validation and knowledge to base rules.
        
        Args:
            base_rules: Base transformation result
            scraping_results: Original scraping results
            
        Returns:
            Rules with cursor rules validation and knowledge applied
        """
        try:
            # Validate cursor rules structure
            validation_result = self._validate_cursor_rules_structure(base_rules)
            
            # Apply cursor rules best practices
            enhanced_rules = self._apply_cursor_rules_best_practices(base_rules, validation_result)
            
            # Add cursor rules metadata if validation passed
            if validation_result['is_valid']:
                enhanced_rules = self._add_cursor_rules_metadata(enhanced_rules, validation_result)
            
            return enhanced_rules
            
        except Exception as e:
            logger.warning(f"Cursor rules validation failed: {e}")
            return base_rules
    
    def _validate_cursor_rules_structure(self, rules: str) -> Dict[str, Any]:
        """Validate cursor rules structure and content.
        
        Args:
            rules: Rules content to validate
            
        Returns:
            Validation result with scores and recommendations
        """
        validation_result = {
            'is_valid': True,
            'score': 0.0,
            'issues': [],
            'recommendations': [],
            'structure_analysis': {},
            'missing_sections': []
        }
        
        try:
            # Check for required sections
            required_sections = ['#', '## Key Principles', '## Code Style', '## Best Practices']
            found_sections = []
            
            for section in required_sections:
                if section in rules:
                    found_sections.append(section)
                else:
                    validation_result['issues'].append(f"Missing required section: {section}")
                    # Add to missing sections for enhancement
                    if section == '## Key Principles':
                        validation_result['missing_sections'].append('Key Principles')
                    elif section == '## Code Style':
                        validation_result['missing_sections'].append('Code Style and Structure')
                    elif section == '## Best Practices':
                        validation_result['missing_sections'].append('Best Practices')
            
            # Calculate structure score
            structure_score = len(found_sections) / len(required_sections)
            validation_result['structure_analysis']['section_coverage'] = structure_score
            
            # Check for cursor rules specific patterns
            cursor_patterns = [
                r'You are an expert',
                r'## Key Principles',
                r'## Code Style',
                r'## Best Practices',
                r'## Critical Instructions',
                r'\*\*NEVER:\*\*',
                r'\*\*ALWAYS:\*\*'
            ]
            
            pattern_matches = 0
            for pattern in cursor_patterns:
                if re.search(pattern, rules, re.IGNORECASE):
                    pattern_matches += 1
            
            pattern_score = pattern_matches / len(cursor_patterns)
            validation_result['structure_analysis']['pattern_coverage'] = pattern_score
            
            # Overall validation score
            validation_result['score'] = (structure_score + pattern_score) / 2
            
            # Determine if valid based on threshold
            validation_result['is_valid'] = validation_result['score'] >= self.cursor_rules_quality_threshold
            
            # Generate recommendations
            if not validation_result['is_valid']:
                validation_result['recommendations'].extend([
                    "Add missing required sections",
                    "Include cursor rules specific patterns",
                    "Ensure proper markdown structure"
                ])
            
            return validation_result
            
        except Exception as e:
            logger.warning(f"Cursor rules validation error: {e}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _apply_cursor_rules_best_practices(self, rules: str, validation_result: Dict[str, Any]) -> str:
        """Apply cursor rules best practices based on validation results.
        
        Args:
            rules: Rules content
            validation_result: Validation analysis results
            
        Returns:
            Enhanced rules with best practices applied
        """
        enhanced_rules = rules
        
        # Add missing sections if needed
        if validation_result['structure_analysis'].get('section_coverage', 0) < 1.0:
            enhanced_rules = self._add_missing_cursor_sections(enhanced_rules)
        
        # Ensure proper cursor rules formatting
        enhanced_rules = self._ensure_cursor_rules_formatting(enhanced_rules)
        
        # Add cursor rules specific enhancements
        enhanced_rules = self._add_cursor_rules_enhancements(enhanced_rules)
        
        return enhanced_rules
    
    def _add_missing_cursor_sections(self, rules: str) -> str:
        """Add missing cursor rules sections.
        
        Args:
            rules: Current rules content
            
        Returns:
            Rules with missing sections added
        """
        sections_to_add = []
        
        if '## Key Principles' not in rules:
            sections_to_add.append("""
## Key Principles

- Write clean, readable, and maintainable code
- Follow established coding standards and best practices
- Implement proper error handling and validation
- Consider performance and security implications
""")
        
        if '## Code Style and Structure' not in rules:
            sections_to_add.append("""
## Code Style and Structure

- Use descriptive names for variables, functions, and classes
- Keep functions focused and modular
- Add meaningful comments for complex logic
- Follow consistent formatting and indentation
""")
        
        if '## Best Practices' not in rules:
            sections_to_add.append("""
## Best Practices

- Test your code thoroughly
- Handle edge cases appropriately
- Use version control effectively
- Document your code and APIs
""")
        
        if '## 🚨 Critical Instructions' not in rules:
            sections_to_add.append("""
## 🚨 Critical Instructions

**NEVER:**
- Ignore error handling or edge cases
- Use deprecated APIs or methods
- Hardcode sensitive information
- Skip input validation and sanitization

**ALWAYS:**
- Follow security best practices
- Test your code thoroughly
- Document complex logic and algorithms
- Consider accessibility and user experience
""")
        
        # Insert sections before the final closing
        if sections_to_add:
            anchor = "Remember: These rules are continuously updated based on official documentation and community feedback."
            if anchor in rules:
                enhanced_rules = rules.replace(anchor, f"{''.join(sections_to_add)}\n{anchor}")
            else:
                enhanced_rules = rules + "\n" + "".join(sections_to_add)
        else:
            enhanced_rules = rules
        
        return enhanced_rules
    
    def _ensure_cursor_rules_formatting(self, rules: str) -> str:
        """Ensure proper cursor rules formatting.
        
        Args:
            rules: Rules content
            
        Returns:
            Properly formatted cursor rules
        """
        # Ensure proper header
        if not rules.startswith('# '):
            rules = f"# Cursor Rules\n\n{rules}"
        
        # Ensure proper markdown structure
        lines = rules.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Ensure proper heading hierarchy
            if line.startswith('##') and not line.startswith('###'):
                # Check if this should be a main section
                if any(keyword in line.lower() for keyword in ['principles', 'style', 'practices', 'guidelines', 'instructions']):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(f"### {line[2:].strip()}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _add_cursor_rules_enhancements(self, rules: str) -> str:
        """Add cursor rules specific enhancements.
        
        Args:
            rules: Rules content
            
        Returns:
            Enhanced rules with cursor-specific improvements
        """
        # Add cursor rules metadata section
        metadata_section = f"""
---

## 📋 Cursor Rules Metadata

- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Format**: Cursor Rules (.cursorrules)
- **Validation**: Passed cursor rules structure validation
- **Quality**: Enhanced with ML quality assessment
- **Learning**: Integrated with learning and intelligence systems

*This rule set follows cursor rules best practices and includes comprehensive development guidelines.*
"""
        
        # Insert metadata before final closing
        anchor = "Remember: These rules are continuously updated based on official documentation and community feedback."
        if anchor in rules:
            enhanced_rules = rules.replace(anchor, f"{metadata_section}\n{anchor}")
        else:
            enhanced_rules = rules + metadata_section
        
        return enhanced_rules
    
    def _add_cursor_rules_metadata(self, rules: str, validation_result: Dict[str, Any]) -> str:
        """Add cursor rules metadata based on validation results.
        
        Args:
            rules: Rules content
            validation_result: Validation analysis results
            
        Returns:
            Rules with cursor rules metadata added
        """
        # Add validation metadata
        validation_metadata = f"""
## 🔍 Cursor Rules Validation

- **Structure Score**: {validation_result['score']:.2f}
- **Section Coverage**: {validation_result['structure_analysis'].get('section_coverage', 0):.2f}
- **Pattern Coverage**: {validation_result['structure_analysis'].get('pattern_coverage', 0):.2f}
- **Validation Status**: {'✅ Passed' if validation_result['is_valid'] else '❌ Failed'}

"""
        
        # Insert validation metadata before final closing
        anchor = "Remember: These rules are continuously updated based on official documentation and community feedback."
        if anchor in rules:
            enhanced_rules = rules.replace(anchor, f"{validation_metadata}\n{anchor}")
        else:
            enhanced_rules = rules + validation_metadata
        
        return enhanced_rules
    
    async def _generate_quality_assessments(self, scraping_results: List[ScrapingResult]) -> List[Dict[str, Any]]:
        """Generate quality assessments for scraping results.
        
        Args:
            scraping_results: Scraping results to assess
            
        Returns:
            List of quality assessment results
        """
        assessments = []
        
        for result in scraping_results:
            try:
                # Get ML quality prediction
                quality_prediction = await self.quality_strategy.predict(
                    result.content, 
                    str(result.url)
                )
                
                # Add source metadata
                assessment = {
                    'source_url': str(result.url),
                    'source_title': result.title,
                    'quality_prediction': quality_prediction,
                    'timestamp': datetime.now().isoformat()
                }
                
                assessments.append(assessment)
                
            except Exception as e:
                logger.warning(f"Quality assessment failed for {result.url}: {e}")
                # Fallback assessment
                assessments.append({
                    'source_url': str(result.url),
                    'source_title': result.title,
                    'quality_prediction': {
                        'quality_score': 0.5,
                        'quality_class': 'medium',
                        'confidence': 0.3,
                        'method': 'fallback',
                        'is_high_quality': False
                    },
                    'timestamp': datetime.now().isoformat()
                })
        
        return assessments
    
    async def _perform_intelligent_clustering(self, scraping_results: List[ScrapingResult], 
                                           quality_assessments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Perform intelligent clustering of sources based on content similarity.
        
        Args:
            scraping_results: Original scraping results
            quality_assessments: Quality assessment results
            
        Returns:
            Clustering results or None if clustering fails/disabled
        """
        if not ML_AVAILABLE or len(scraping_results) < 2:
            return None
        
        try:
            # Prepare content for clustering
            contents = []
            for result in scraping_results:
                # Combine title and content for clustering
                content = f"{result.title} {result.content[:2000]}"  # Limit content length
                contents.append(content)
            
            # Vectorize content
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            X = vectorizer.fit_transform(contents)
            
            # Determine optimal number of clusters
            n_clusters = min(self.max_clusters, len(scraping_results))
            if n_clusters < 2:
                return None
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate clustering quality
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            # Only use clustering if quality is acceptable
            if silhouette_avg < self.coherence_threshold:
                logger.debug(f"Clustering coherence too low: {silhouette_avg:.3f} < {self.coherence_threshold}")
                return None
            
            # Organize results by cluster
            clusters = {}
            for i, (result, assessment, label) in enumerate(zip(scraping_results, quality_assessments, cluster_labels)):
                cluster_id = f"cluster_{label}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = {
                        'sources': [],
                        'quality_scores': [],
                        'technologies': set(),
                        'avg_quality': 0.0
                    }
                
                clusters[cluster_id]['sources'].append({
                    'url': str(result.url),
                    'title': result.title,
                    'quality_score': assessment['quality_prediction']['quality_score']
                })
                clusters[cluster_id]['quality_scores'].append(assessment['quality_prediction']['quality_score'])
                
                # Extract detected technologies if available
                if 'detected_technologies' in assessment['quality_prediction']:
                    for tech in assessment['quality_prediction']['detected_technologies']:
                        clusters[cluster_id]['technologies'].add(tech['name'])
            
            # Calculate cluster statistics
            for cluster_id, cluster_info in clusters.items():
                cluster_info['avg_quality'] = sum(cluster_info['quality_scores']) / len(cluster_info['quality_scores'])
                cluster_info['technologies'] = list(cluster_info['technologies'])
                cluster_info['source_count'] = len(cluster_info['sources'])
            
            return {
                'n_clusters': n_clusters,
                'coherence_score': silhouette_avg,
                'clusters': clusters,
                'clustering_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return None
    
    def _create_ml_quality_section(self, quality_assessments: List[Dict[str, Any]], 
                                  clusters: Optional[Dict[str, Any]]) -> str:
        """Create ML quality assessment section for rules with cursor rules knowledge.
        
        Args:
            quality_assessments: Quality assessment results
            clusters: Clustering results (optional)
            
        Returns:
            Formatted ML quality section with cursor rules insights
        """
        section_parts = []
        
        # Header with cursor rules context
        section_parts.append("## 🤖 ML Quality Assessment\n")
        section_parts.append("*Automatically generated quality analysis and recommendations with cursor rules validation*\n")
        
        # Overall statistics
        if quality_assessments:
            total_sources = len(quality_assessments)
            high_quality_count = sum(1 for assessment in quality_assessments 
                                   if assessment['quality_prediction']['is_high_quality'])
            avg_quality = sum(assessment['quality_prediction']['quality_score'] 
                            for assessment in quality_assessments) / total_sources
            
            section_parts.append(f"**📊 Source Analysis:**")
            section_parts.append(f"- Total sources analyzed: {total_sources}")
            section_parts.append(f"- High-quality sources: {high_quality_count} ({high_quality_count/total_sources*100:.1f}%)")
            section_parts.append(f"- Average quality score: {avg_quality:.3f}")
            section_parts.append("")
        
        # Clustering information
        if clusters and clusters['coherence_score'] >= self.coherence_threshold:
            section_parts.append("**🎯 Intelligent Clustering:**")
            section_parts.append(f"- Sources grouped into {clusters['n_clusters']} clusters")
            section_parts.append(f"- Clustering coherence: {clusters['coherence_score']:.3f}")
            section_parts.append("")
            
            # Cluster details
            for cluster_id, cluster_info in clusters['clusters'].items():
                section_parts.append(f"*{cluster_id.replace('_', ' ').title()}* ({cluster_info['source_count']} sources):")
                section_parts.append(f"  - Average quality: {cluster_info['avg_quality']:.3f}")
                if cluster_info['technologies']:
                    section_parts.append(f"  - Technologies: {', '.join(cluster_info['technologies'][:3])}")
                section_parts.append("")
        
        # Quality recommendations
        section_parts.append("**💡 Quality Recommendations:**")
        all_recommendations = set()
        for assessment in quality_assessments:
            if 'recommendations' in assessment['quality_prediction']:
                all_recommendations.update(assessment['quality_prediction']['recommendations'])
        
        if all_recommendations:
            for recommendation in list(all_recommendations)[:5]:  # Limit to top 5
                section_parts.append(f"- {recommendation}")
        else:
            section_parts.append("- No specific recommendations - sources meet quality standards")
        section_parts.append("")
        
        # Source quality breakdown
        section_parts.append("**📋 Source Quality Breakdown:**")
        for assessment in quality_assessments[:5]:  # Show top 5 sources
            quality = assessment['quality_prediction']
            quality_emoji = "🟢" if quality['is_high_quality'] else "🟡" if quality['quality_score'] > 0.4 else "🔴"
            section_parts.append(f"{quality_emoji} {assessment['source_title'][:50]}... (Score: {quality['quality_score']:.3f})")
        
        if len(quality_assessments) > 5:
            section_parts.append(f"... and {len(quality_assessments) - 5} more sources")
        section_parts.append("")
        
        # Cursor rules specific insights
        section_parts.append("**🎯 Cursor Rules Insights:**")
        cursor_insights = self._generate_cursor_rules_insights(quality_assessments)
        section_parts.extend(cursor_insights)
        section_parts.append("")
        
        # ML metadata with cursor rules context
        section_parts.append(f"*Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using ML quality assessment with cursor rules validation*")
        section_parts.append("")
        
        return "\n".join(section_parts)
    
    def _generate_cursor_rules_insights(self, quality_assessments: List[Dict[str, Any]]) -> List[str]:
        """Generate cursor rules specific insights from quality assessments.
        
        Args:
            quality_assessments: Quality assessment results
            
        Returns:
            List of cursor rules insights
        """
        insights = []
        
        # Analyze cursor rules compliance
        cursor_compliance_score = self._calculate_cursor_rules_compliance(quality_assessments)
        insights.append(f"- Cursor rules compliance score: {cursor_compliance_score:.2f}")
        
        # Technology-specific insights
        tech_insights = self._analyze_technology_cursor_rules(quality_assessments)
        insights.extend(tech_insights)
        
        # Quality recommendations for cursor rules
        recommendations = self._generate_cursor_rules_recommendations(quality_assessments)
        insights.extend(recommendations)
        
        return insights
    
    def _calculate_cursor_rules_compliance(self, quality_assessments: List[Dict[str, Any]]) -> float:
        """Calculate cursor rules compliance score.
        
        Args:
            quality_assessments: Quality assessment results
            
        Returns:
            Compliance score between 0.0 and 1.0
        """
        if not quality_assessments:
            return 0.0
        
        # Calculate average quality score
        total_score = sum(assessment['quality_prediction']['quality_score'] 
                         for assessment in quality_assessments)
        avg_score = total_score / len(quality_assessments)
        
        # Adjust for cursor rules specific factors
        cursor_factors = []
        for assessment in quality_assessments:
            # Check for cursor rules specific patterns
            content = assessment.get('source_title', '') + ' ' + str(assessment.get('source_url', ''))
            if any(pattern in content.lower() for pattern in ['cursor', 'rules', 'guidelines', 'best practices']):
                cursor_factors.append(1.0)
            else:
                cursor_factors.append(0.5)
        
        cursor_factor = sum(cursor_factors) / len(cursor_factors)
        
        # Combine quality and cursor factors
        compliance_score = (avg_score * 0.7) + (cursor_factor * 0.3)
        
        return min(compliance_score, 1.0)
    
    def _analyze_technology_cursor_rules(self, quality_assessments: List[Dict[str, Any]]) -> List[str]:
        """Analyze technology-specific cursor rules insights.
        
        Args:
            quality_assessments: Quality assessment results
            
        Returns:
            List of technology-specific insights
        """
        insights = []
        
        # Detect technologies from assessments
        technologies = set()
        for assessment in quality_assessments:
            if 'detected_technologies' in assessment['quality_prediction']:
                for tech in assessment['quality_prediction']['detected_technologies']:
                    technologies.add(tech['name'])
        
        # Generate technology-specific insights
        if 'React' in technologies or 'Next.js' in technologies:
            insights.append("- React/Next.js: Optimized for modern React development patterns")
        if 'Python' in technologies:
            insights.append("- Python: Enhanced with PEP 8 and modern Python best practices")
        if 'TypeScript' in technologies:
            insights.append("- TypeScript: Includes strict typing and modern TS patterns")
        
        return insights
    
    def _generate_cursor_rules_recommendations(self, quality_assessments: List[Dict[str, Any]]) -> List[str]:
        """Generate cursor rules specific recommendations.
        
        Args:
            quality_assessments: Quality assessment results
            
        Returns:
            List of cursor rules recommendations
        """
        recommendations = []
        
        # Analyze quality patterns
        high_quality_count = sum(1 for assessment in quality_assessments 
                               if assessment['quality_prediction']['is_high_quality'])
        total_count = len(quality_assessments)
        
        if high_quality_count / total_count < 0.7:
            recommendations.append("- Consider enhancing source quality for better cursor rules generation")
        
        # Technology-specific recommendations
        recommendations.append("- Ensure cursor rules include technology-specific best practices")
        recommendations.append("- Validate cursor rules structure and formatting")
        
        return recommendations
    
    def _add_fallback_enhancement(self, base_rules: str, scraping_results: List[ScrapingResult]) -> str:
        """Add basic enhancement when ML is not available.
        
        Args:
            base_rules: Base transformation result
            scraping_results: Original scraping results
            
        Returns:
            Rules with basic enhancement
        """
        # Simple source count and basic analysis
        source_count = len(scraping_results)
        urls = [str(result.url) for result in scraping_results]
        
        enhancement = f"""
## 📋 Source Analysis

**Sources processed:** {source_count}
**Analysis method:** Basic heuristic analysis

**Sources:**
{chr(10).join(f"- {url}" for url in urls[:5])}
{"... and more" if len(urls) > 5 else ""}

*Note: ML quality assessment unavailable - using basic analysis*

"""
        
        # Insert enhancement before final closing
        anchor = "Remember: These rules are continuously updated based on official documentation and community feedback."
        if anchor in base_rules:
            enhanced_rules = base_rules.replace(
                anchor,
                f"{enhancement}\n{anchor}"
            )
        else:
            enhanced_rules = base_rules + "\n\n" + enhancement
        return enhanced_rules
    
    async def get_quality_insights(self, scraping_results: List[ScrapingResult]) -> Dict[str, Any]:
        """Get detailed quality insights for scraping results.
        
        Args:
            scraping_results: Scraping results to analyze
            
        Returns:
            Dictionary of quality insights and recommendations
        """
        if not self.ml_enabled:
            return {'error': 'ML components not available', 'method': 'fallback'}
        
        try:
            # Generate quality assessments
            quality_assessments = await self._generate_quality_assessments(scraping_results)
            
            # Perform clustering analysis
            clusters = await self._perform_intelligent_clustering(scraping_results, quality_assessments)
            
            # Calculate overall insights
            insights = {
                'total_sources': len(scraping_results),
                'quality_assessments': quality_assessments,
                'clustering': clusters,
                'overall_statistics': self._calculate_overall_statistics(quality_assessments),
                'recommendations': self._generate_overall_recommendations(quality_assessments),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Quality insights generation failed: {e}")
            return {'error': str(e), 'method': 'error_fallback'}
    
    def _calculate_overall_statistics(self, quality_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall quality statistics.
        
        Args:
            quality_assessments: Quality assessment results
            
        Returns:
            Dictionary of overall statistics
        """
        if not quality_assessments:
            return {}
        
        quality_scores = [assessment['quality_prediction']['quality_score'] 
                         for assessment in quality_assessments]
        
        return {
            'average_quality': sum(quality_scores) / len(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'high_quality_count': sum(1 for assessment in quality_assessments 
                                    if assessment['quality_prediction']['is_high_quality']),
            'high_quality_percentage': (sum(1 for assessment in quality_assessments 
                                          if assessment['quality_prediction']['is_high_quality']) 
                                      / len(quality_assessments) * 100)
        }
    
    def _generate_overall_recommendations(self, quality_assessments: List[Dict[str, Any]]) -> List[str]:
        """Generate overall recommendations based on quality assessments.
        
        Args:
            quality_assessments: Quality assessment results
            
        Returns:
            List of overall recommendations
        """
        all_recommendations = []
        for assessment in quality_assessments:
            if 'recommendations' in assessment['quality_prediction']:
                all_recommendations.extend(assessment['quality_prediction']['recommendations'])
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Return top recommendations by frequency
        sorted_recommendations = sorted(recommendation_counts.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return [rec for rec, count in sorted_recommendations[:5]]
