"""
ML-enhanced Cursor transformer with quality assessment and intelligent clustering.

Extends existing CursorRuleTransformer with ML-powered quality scoring,
rule clustering, and self-improving feedback integration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

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

logger = logging.getLogger(__name__)


class MLCursorTransformer(CursorRuleTransformer):
    """ML-enhanced Cursor transformer with quality assessment."""
    
    def __init__(self, ml_config: Optional[Dict[str, Any]] = None):
        """Initialize ML-enhanced Cursor transformer.
        
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
    
    async def transform(self, scraping_results: List[ScrapingResult]) -> str:
        """Transform scraping results with ML enhancement.
        
        Args:
            scraping_results: List of scraping results to transform
            
        Returns:
            Enhanced Cursor rules with ML quality assessment
        """
        # Use existing base transformation logic
        base_rules = super().transform(scraping_results)
        
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
        """Create ML quality assessment section for rules.
        
        Args:
            quality_assessments: Quality assessment results
            clusters: Clustering results (optional)
            
        Returns:
            Formatted ML quality section
        """
        section_parts = []
        
        # Header
        section_parts.append("## 🤖 ML Quality Assessment\n")
        section_parts.append("*Automatically generated quality analysis and recommendations*\n")
        
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
        
        # ML metadata
        section_parts.append(f"*Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using ML quality assessment*")
        section_parts.append("")
        
        return "\n".join(section_parts)
    
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
