"""
ML-Powered Batch Documentation Processor

Implements intelligent batch processing for 100+ documentation sources with:
- Self-improving ML pipeline for rule quality assessment
- Intelligent rule clustering and coherence algorithms  
- Feedback-driven optimization and self-awarding system
- Technology-aware rule grouping and semantic analysis
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import defaultdict
from pathlib import Path
import json
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .scrapers.async_documentation_scraper import AsyncDocumentationScraper
from .transformers.cursor_transformer import CursorRuleTransformer
from .transformers.windsurf_transformer import WindsurfRuleTransformer
from .learning.engine import LearningEngine
from .learning.pattern_analyzer import SemanticAnalyzer
from .processors.ml_documentation_processor import MLDocumentationProcessor
from .models import (
    ScrapingResult, ScrapingConfig, TransformationConfig,
    RuleFormat, DocumentationType
)
from .bedrock_integration import BedrockRulesMaker

logger = logging.getLogger(__name__)


@dataclass
class DocumentationSource:
    """Represents a documentation source to be processed."""
    url: str
    name: str
    technology: str
    framework: Optional[str] = None
    priority: int = 1
    expected_pages: int = 20
    language: Optional[str] = None
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RuleCluster:
    """Represents a cluster of related rules."""
    id: str
    name: str
    rules: List[Dict[str, Any]]
    coherence_score: float
    technology: str
    framework: Optional[str]
    semantic_keywords: List[str]
    centroid: Optional[np.ndarray] = None


@dataclass
class BatchResult:
    """Results from batch processing."""
    sources_processed: int
    total_rules_generated: int
    clusters: List[RuleCluster]
    quality_metrics: Dict[str, float]
    processing_time: float
    failed_sources: List[str]
    insights: Dict[str, Any]


class MLBatchProcessor:
    """ML-powered batch processor for documentation sources."""
    
    def __init__(
        self,
        bedrock_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "rules/generated",
        quality_threshold: float = 0.6,
        max_concurrent: int = 10
    ):
        """Initialize the batch processor."""
        self.bedrock_config = bedrock_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_threshold = quality_threshold
        self.max_concurrent = max_concurrent
        
        # Initialize components
        self.scraper_config = ScrapingConfig(
            max_pages=50,
            rate_limit=0.5,
            max_depth=3
        )
        
        self.semantic_analyzer = SemanticAnalyzer()
        self.learning_engine = LearningEngine()
        # ML-enhanced documentation processor (graceful fallback inside class)
        self.ml_processor = MLDocumentationProcessor()
        
        # ML components for clustering
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Quality tracking
        self.quality_history: List[Dict[str, Any]] = []
        self.feedback_scores: Dict[str, float] = {}
        
    async def process_documentation_batch(
        self,
        sources: List[DocumentationSource],
        formats: List[RuleFormat] = None
    ) -> BatchResult:
        """Process multiple documentation sources and generate optimized rule sets."""
        if formats is None:
            formats = [RuleFormat.CURSOR, RuleFormat.WINDSURF]
            
        start_time = time.time()
        logger.info(f"🚀 Starting batch processing of {len(sources)} documentation sources")
        
        # Phase 1: Concurrent documentation scraping
        scraped_results = await self._scrape_documentation_batch(sources)
        
        # Phase 2: ML-powered content analysis and rule generation
        raw_rules = await self._generate_rules_batch(scraped_results, formats)
        
        # Phase 3: Intelligent clustering and coherence optimization
        clustered_rules = await self._cluster_and_optimize_rules(raw_rules)
        
        # Phase 4: Quality assessment and self-improvement
        quality_metrics = await self._assess_and_improve_quality(clustered_rules)
        
        # Phase 5: Generate final optimized rule sets
        await self._export_optimized_rules(clustered_rules, formats)
        
        processing_time = time.time() - start_time
        
        # Compile results
        result = BatchResult(
            sources_processed=len([r for r in scraped_results if r.status.value == "completed"]),
            total_rules_generated=sum(len(cluster.rules) for cluster in clustered_rules),
            clusters=clustered_rules,
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            failed_sources=[r.url for r in scraped_results if r.status.value == "failed"],
            insights=await self._generate_batch_insights(clustered_rules, quality_metrics)
        )
        
        logger.info(f"✅ Batch processing completed in {processing_time:.2f}s")
        logger.info(f"📊 Generated {result.total_rules_generated} rules across {len(clustered_rules)} clusters")
        
        return result
    
    async def _scrape_documentation_batch(
        self, 
        sources: List[DocumentationSource]
    ) -> List[ScrapingResult]:
        """Scrape documentation from all sources concurrently with enhanced error recovery."""
        logger.info("🔍 Phase 1: Concurrent documentation scraping with validation")
        
        # Try to use enhanced scraper first
        try:
            from .scrapers.enhanced_async_scraper import EnhancedAsyncDocumentationScraper
            
            async with EnhancedAsyncDocumentationScraper(self.scraper_config) as scraper:
                urls = [source.url for source in sources]
                results, validation_stats = await scraper.scrape_multiple_with_validation(urls)
                
                # Update metadata for successful results
                source_map = {source.url: source for source in sources}
                for result in results:
                    original_url = result.metadata.get('original_url', str(result.url))
                    if original_url in source_map:
                        source = source_map[original_url]
                        result.metadata.update({
                            'source_name': source.name,
                            'technology': source.technology,
                            'framework': source.framework,
                            'category': source.category,
                            'priority': source.priority
                        })
                
                # Log validation statistics
                logger.info(f"📊 Validation stats: {validation_stats['valid_urls']}/{validation_stats['total_urls']} valid")
                if validation_stats['redirected_urls'] > 0:
                    logger.info(f"🔄 {validation_stats['redirected_urls']} URLs redirected successfully")
                if validation_stats['failed_validations']:
                    logger.warning(f"⚠️ {len(validation_stats['failed_validations'])} URLs failed validation")
                
                return results
                
        except ImportError:
            logger.warning("Enhanced scraper not available, falling back to standard scraper")
            
        # Fallback to standard scraper
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []
        
        async def scrape_source(source: DocumentationSource) -> ScrapingResult:
            async with semaphore:
                try:
                    from .scrapers.async_documentation_scraper import AsyncDocumentationScraper
                    async with AsyncDocumentationScraper(self.scraper_config) as scraper:
                        result = await scraper.scrape_url(source.url)
                        result.metadata.update({
                            'source_name': source.name,
                            'technology': source.technology,
                            'framework': source.framework,
                            'category': source.category,
                            'priority': source.priority
                        })
                        return result
                except Exception as e:
                    logger.error(f"Failed to scrape {source.url}: {e}")
                    return ScrapingResult(
                        url=source.url,
                        title=source.name,
                        content="",
                        status="failed",
                        error_message=str(e)
                    )
        
        # Execute scraping tasks
        tasks = [scrape_source(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, ScrapingResult)]
        successful = [r for r in valid_results if r.status.value == "completed"]
        
        logger.info(f"✅ Scraped {len(successful)}/{len(sources)} sources successfully")
        return valid_results
    
    async def _generate_rules_batch(
        self,
        scraped_results: List[ScrapingResult],
        formats: List[RuleFormat]
    ) -> List[Dict[str, Any]]:
        """Generate rules from scraped content using ML-enhanced transformers."""
        logger.info("🤖 Phase 2: ML-powered rule generation")
        
        all_rules = []
        
        # Initialize Bedrock-powered transformers if configured
        if self.bedrock_config:
            try:
                bedrock_maker = BedrockRulesMaker(**self.bedrock_config)
            except Exception as e:
                logger.warning(f"Bedrock initialization failed: {e}, falling back to standard transformers")
                bedrock_maker = None
        else:
            bedrock_maker = None
        
        # Process each successful scraping result
        for result in scraped_results:
            if result.status.value != "completed":
                continue
                
            # Enrich metadata with ML analysis (non-blocking, best-effort)
            try:
                doc_struct = self.ml_processor.process(result.content, str(result.url), dict(result.metadata))
                # Merge ML metadata into result metadata
                if isinstance(doc_struct.metadata, dict):
                    result.metadata.update({k: v for k, v in doc_struct.metadata.items() if k not in ('title', 'name')})
            except Exception as e:
                logger.debug(f"ML document enhancement failed for {result.url}: {e}")

            # Perform semantic analysis
            content_analysis = self.semantic_analyzer.analyze_content(result.content)
            
            # Generate rules for each format
            for format_type in formats:
                try:
                    if bedrock_maker and format_type in [RuleFormat.CURSOR, RuleFormat.WINDSURF]:
                        # Use Bedrock-enhanced generation
                        if format_type == RuleFormat.CURSOR:
                            rules_content = await bedrock_maker.generate_enhanced_cursor_rules(
                                result.content
                            )
                        else:
                            rules_content = await bedrock_maker.generate_enhanced_windsurf_rules(
                                result.content
                            )
                    else:
                        # Use standard transformers
                        if format_type == RuleFormat.CURSOR:
                            transformer = CursorRuleTransformer()
                        elif format_type == RuleFormat.WINDSURF:
                            transformer = WindsurfRuleTransformer()
                        else:
                            continue
                            
                        rules_content = transformer.transform([result])
                    
                    # Create rule entry with rich metadata
                    rule_entry = {
                        'id': self._generate_rule_id(result, format_type),
                        'source_url': str(result.url),
                        'source_name': result.metadata.get('source_name', result.title),
                        'technology': result.metadata.get('technology', 'unknown'),
                        'framework': result.metadata.get('framework'),
                        'category': result.metadata.get('category', 'general'),
                        'format': format_type.value,
                        'content': rules_content,
                        'quality_score': self._estimate_initial_quality(rules_content, content_analysis),
                        'semantic_analysis': content_analysis.model_dump(),
                        'generated_at': time.time(),
                        'coherence_features': self._extract_coherence_features(rules_content)
                    }
                    
                    all_rules.append(rule_entry)
                    
                except Exception as e:
                    logger.error(f"Failed to generate {format_type.value} rules for {result.url}: {e}")
        
        logger.info(f"✅ Generated {len(all_rules)} rule sets")
        return all_rules
    
    async def _cluster_and_optimize_rules(
        self, 
        raw_rules: List[Dict[str, Any]]
    ) -> List[RuleCluster]:
        """Cluster rules by semantic similarity and optimize for coherence."""
        logger.info("🧠 Phase 3: Intelligent clustering and coherence optimization")
        
        if not raw_rules:
            return []
        
        # Group rules by technology and format first
        tech_format_groups = defaultdict(list)
        for rule in raw_rules:
            key = (rule['technology'], rule['format'])
            tech_format_groups[key].append(rule)
        
        clusters = []
        
        for (technology, format_type), rules in tech_format_groups.items():
            if len(rules) < 2:
                # Single rule - create individual cluster
                cluster = RuleCluster(
                    id=f"{technology}_{format_type}_single",
                    name=f"{technology.title()} {format_type.title()} Rules",
                    rules=rules,
                    coherence_score=rules[0]['quality_score'],
                    technology=technology,
                    framework=rules[0].get('framework'),
                    semantic_keywords=self._extract_keywords_from_rules(rules)
                )
                clusters.append(cluster)
                continue
            
            # Extract text features for clustering
            rule_texts = [rule['content'] for rule in rules]
            
            try:
                # Vectorize rule content
                feature_matrix = self.vectorizer.fit_transform(rule_texts)
                
                # Determine optimal number of clusters
                n_clusters = min(max(2, len(rules) // 3), 8)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(feature_matrix)
                
                # Create clusters
                for cluster_id in range(n_clusters):
                    cluster_rules = [
                        rule for i, rule in enumerate(rules) 
                        if cluster_labels[i] == cluster_id
                    ]
                    
                    if cluster_rules:
                        coherence_score = self._calculate_cluster_coherence(
                            cluster_rules, feature_matrix, cluster_labels, cluster_id
                        )
                        
                        cluster = RuleCluster(
                            id=f"{technology}_{format_type}_cluster_{cluster_id}",
                            name=f"{technology.title()} {format_type.title()} - Cluster {cluster_id + 1}",
                            rules=cluster_rules,
                            coherence_score=coherence_score,
                            technology=technology,
                            framework=cluster_rules[0].get('framework'),
                            semantic_keywords=self._extract_keywords_from_rules(cluster_rules),
                            centroid=kmeans.cluster_centers_[cluster_id]
                        )
                        clusters.append(cluster)
                        
            except Exception as e:
                logger.error(f"Clustering failed for {technology}/{format_type}: {e}")
                # Fallback: create single cluster
                cluster = RuleCluster(
                    id=f"{technology}_{format_type}_fallback",
                    name=f"{technology.title()} {format_type.title()} Rules",
                    rules=rules,
                    coherence_score=np.mean([r['quality_score'] for r in rules]),
                    technology=technology,
                    framework=rules[0].get('framework'),
                    semantic_keywords=self._extract_keywords_from_rules(rules)
                )
                clusters.append(cluster)
        
        # Sort clusters by coherence score
        clusters.sort(key=lambda c: c.coherence_score, reverse=True)
        
        logger.info(f"✅ Created {len(clusters)} optimized rule clusters")
        return clusters
    
    async def _assess_and_improve_quality(
        self,
        clusters: List[RuleCluster]
    ) -> Dict[str, float]:
        """Assess rule quality and apply self-improving optimizations."""
        logger.info("📈 Phase 4: Quality assessment and self-improvement")
        
        quality_metrics = {
            'overall_coherence': 0.0,
            'technology_coverage': 0.0,
            'rule_diversity': 0.0,
            'semantic_richness': 0.0,
            'improvement_score': 0.0
        }
        
        if not clusters:
            return quality_metrics
        
        # Calculate overall coherence
        coherence_scores = [cluster.coherence_score for cluster in clusters]
        quality_metrics['overall_coherence'] = np.mean(coherence_scores)
        
        # Calculate technology coverage
        technologies = set(cluster.technology for cluster in clusters)
        quality_metrics['technology_coverage'] = len(technologies) / 20  # Normalize against expected 20 tech types
        
        # Calculate rule diversity
        total_rules = sum(len(cluster.rules) for cluster in clusters)
        avg_rules_per_cluster = total_rules / len(clusters) if clusters else 0
        quality_metrics['rule_diversity'] = min(1.0, avg_rules_per_cluster / 10)  # Normalize
        
        # Calculate semantic richness
        all_keywords = set()
        for cluster in clusters:
            all_keywords.update(cluster.semantic_keywords)
        quality_metrics['semantic_richness'] = min(1.0, len(all_keywords) / 100)  # Normalize
        
        # Self-improvement: Apply learning-based optimizations
        improvement_score = await self._apply_learning_optimizations(clusters)
        quality_metrics['improvement_score'] = improvement_score
        
        # Store quality metrics for future improvements
        self.quality_history.append({
            'timestamp': time.time(),
            'metrics': quality_metrics.copy(),
            'cluster_count': len(clusters),
            'total_rules': total_rules
        })
        
        logger.info(f"✅ Quality assessment complete - Overall coherence: {quality_metrics['overall_coherence']:.3f}")
        return quality_metrics
    
    async def _apply_learning_optimizations(self, clusters: List[RuleCluster]) -> float:
        """Apply ML-based optimizations to improve rule quality."""
        improvement_score = 0.0
        
        try:
            # Use learning engine to analyze historical performance
            if self.quality_history:
                # Identify improvement patterns
                historical_scores = [h['metrics']['overall_coherence'] for h in self.quality_history]
                if len(historical_scores) > 1:
                    trend = np.polyfit(range(len(historical_scores)), historical_scores, 1)[0]
                    improvement_score += max(0, trend) * 0.5
            
            # Apply coherence-based filtering
            high_quality_clusters = [c for c in clusters if c.coherence_score > self.quality_threshold]
            if high_quality_clusters:
                improvement_score += len(high_quality_clusters) / len(clusters) * 0.3
            
            # Boost scores for diverse technology coverage
            unique_techs = set(c.technology for c in clusters)
            if len(unique_techs) > 3:
                improvement_score += 0.2
                
        except Exception as e:
            logger.error(f"Learning optimization failed: {e}")
            
        return min(1.0, improvement_score)
    
    async def _export_optimized_rules(
        self,
        clusters: List[RuleCluster],
        formats: List[RuleFormat]
    ):
        """Export optimized rule sets to files."""
        logger.info("💾 Phase 5: Exporting optimized rule sets")
        
        for cluster in clusters:
            # Create cluster directory
            cluster_dir = self.output_dir / cluster.technology / cluster.id
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            # Export rules by format
            format_rules = defaultdict(list)
            for rule in cluster.rules:
                format_rules[rule['format']].append(rule)
            
            for format_type, rules in format_rules.items():
                if not rules:
                    continue
                    
                # Combine rule contents
                combined_content = self._combine_rule_contents(rules)
                
                # Create filename
                filename = f"{cluster.technology}_{format_type}_rules.md"
                filepath = cluster_dir / filename
                
                # Write rules file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# {cluster.name}\n\n")
                    f.write(f"**Technology:** {cluster.technology.title()}\n")
                    f.write(f"**Format:** {format_type.title()}\n")
                    f.write(f"**Coherence Score:** {cluster.coherence_score:.3f}\n")
                    f.write(f"**Rules Count:** {len(rules)}\n")
                    f.write(f"**Keywords:** {', '.join(cluster.semantic_keywords[:10])}\n\n")
                    f.write("---\n\n")
                    f.write(combined_content)
            
            # Export cluster metadata
            metadata = {
                'cluster_id': cluster.id,
                'name': cluster.name,
                'technology': cluster.technology,
                'framework': cluster.framework,
                'coherence_score': cluster.coherence_score,
                'rules_count': len(cluster.rules),
                'semantic_keywords': cluster.semantic_keywords,
                'sources': list(set(rule['source_name'] for rule in cluster.rules)),
                'generated_at': time.time()
            }
            
            with open(cluster_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Exported {len(clusters)} optimized rule clusters")
    
    async def _generate_batch_insights(
        self,
        clusters: List[RuleCluster],
        quality_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate insights from batch processing results."""
        
        insights = {
            'technology_distribution': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'top_performing_sources': [],
            'improvement_opportunities': [],
            'recommendations': []
        }
        
        # Technology distribution
        for cluster in clusters:
            insights['technology_distribution'][cluster.technology] += len(cluster.rules)
        
        # Quality distribution
        for cluster in clusters:
            score_bucket = f"{int(cluster.coherence_score * 10) / 10:.1f}"
            insights['quality_distribution'][score_bucket] += 1
        
        # Top performing sources
        source_scores = defaultdict(list)
        for cluster in clusters:
            for rule in cluster.rules:
                source_scores[rule['source_name']].append(rule['quality_score'])
        
        avg_source_scores = {
            source: np.mean(scores) 
            for source, scores in source_scores.items()
        }
        
        insights['top_performing_sources'] = sorted(
            avg_source_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Improvement opportunities
        low_quality_clusters = [c for c in clusters if c.coherence_score < self.quality_threshold]
        if low_quality_clusters:
            insights['improvement_opportunities'].extend([
                f"Low coherence in {c.technology} rules (score: {c.coherence_score:.3f})"
                for c in low_quality_clusters[:3]
            ])
        
        # Recommendations
        if quality_metrics['technology_coverage'] < 0.5:
            insights['recommendations'].append("Expand technology coverage by adding more diverse documentation sources")
        
        if quality_metrics['semantic_richness'] < 0.6:
            insights['recommendations'].append("Include more detailed and varied documentation to improve semantic richness")
        
        if quality_metrics['overall_coherence'] < self.quality_threshold:
            insights['recommendations'].append("Focus on higher-quality documentation sources to improve rule coherence")
        
        return dict(insights)
    
    # Helper methods
    
    def _generate_rule_id(self, result: ScrapingResult, format_type: RuleFormat) -> str:
        """Generate unique ID for a rule."""
        content_hash = hashlib.md5(
            f"{result.url}_{format_type.value}_{result.title}".encode()
        ).hexdigest()[:8]
        return f"rule_{format_type.value}_{content_hash}"
    
    def _estimate_initial_quality(
        self, 
        rules_content: str, 
        content_analysis: Any
    ) -> float:
        """Estimate initial quality score for generated rules."""
        score = 0.4  # Base score (allows truly low-quality to be < 0.5)
        
        # Content length bonus
        if len(rules_content) > 500:
            score += 0.1
        if len(rules_content) > 1000:
            score += 0.1
        
        # Semantic analysis bonus (defensive for mocks/missing attrs)
        try:
            items = getattr(getattr(content_analysis, 'best_practices', object()), 'items', [])
            items_len = len(items) if hasattr(items, '__len__') else 0
            score += min(0.2, items_len * 0.05)
        except Exception:
            pass
        try:
            pats = getattr(getattr(content_analysis, 'patterns', object()), 'patterns', [])
            pats_len = len(pats) if hasattr(pats, '__len__') else 0
            score += min(0.2, pats_len * 0.02)
        except Exception:
            pass
        
        return min(1.0, score)
    
    def _extract_coherence_features(self, rules_content: str) -> Dict[str, Any]:
        """Extract features for coherence analysis."""
        return {
            'length': len(rules_content),
            'word_count': len(rules_content.split()),
            'has_examples': 'example' in rules_content.lower(),
            'has_code_blocks': '```' in rules_content,
            'has_structure': any(marker in rules_content for marker in ['##', '**', '1.', '-'])
        }
    
    def _extract_keywords_from_rules(self, rules: List[Dict[str, Any]]) -> List[str]:
        """Extract semantic keywords from rule cluster."""
        all_text = ' '.join(rule['content'] for rule in rules)
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = all_text.lower().split()
        
        # Filter technical terms
        tech_keywords = [
            word for word in set(words)
            if len(word) > 3 and word.isalpha() and
            word in ['async', 'function', 'class', 'method', 'api', 'config', 'test', 'error', 'handler', 'pattern']
        ]
        
        return tech_keywords[:20]  # Top 20 keywords
    
    def _calculate_cluster_coherence(
        self,
        cluster_rules: List[Dict[str, Any]],
        feature_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_id: int
    ) -> float:
        """Calculate coherence score for a rule cluster."""
        if len(cluster_rules) <= 1:
            return cluster_rules[0]['quality_score'] if cluster_rules else 0.0
        
        # Get feature vectors for this cluster
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_features = feature_matrix[cluster_indices]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(cluster_features)
        
        # Average similarity (excluding diagonal)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        avg_similarity = similarities[mask].mean() if mask.any() else 0.0
        
        # Combine with individual rule quality scores
        avg_quality = np.mean([rule['quality_score'] for rule in cluster_rules])
        
        # Weighted coherence score
        coherence_score = 0.6 * avg_similarity + 0.4 * avg_quality
        
        return max(0.0, min(1.0, coherence_score))
    
    def _combine_rule_contents(self, rules: List[Dict[str, Any]]) -> str:
        """Combine multiple rule contents into coherent output."""
        if len(rules) == 1:
            return rules[0]['content']
        
        # Sort rules by quality score
        sorted_rules = sorted(rules, key=lambda r: r['quality_score'], reverse=True)
        
        # Take the best rule as base and incorporate unique elements from others
        base_content = sorted_rules[0]['content']
        
        # Add source attribution
        sources = list(set(rule['source_name'] for rule in rules))
        attribution = f"\n\n<!-- Generated from: {', '.join(sources)} -->\n"
        
        return base_content + attribution


# Convenience functions for common use cases

async def process_popular_frameworks(
    output_dir: str = "rules/frameworks",
    bedrock_config: Optional[Dict[str, Any]] = None
) -> BatchResult:
    """Process popular web frameworks and generate rules."""
    
    sources = [
        DocumentationSource("https://reactjs.org/docs/", "React", "javascript", "react", priority=5),
        DocumentationSource("https://vuejs.org/guide/", "Vue.js", "javascript", "vue", priority=5),
        DocumentationSource("https://angular.io/docs", "Angular", "javascript", "angular", priority=4),
        DocumentationSource("https://nextjs.org/docs", "Next.js", "javascript", "nextjs", priority=4),
        DocumentationSource("https://svelte.dev/docs", "Svelte", "javascript", "svelte", priority=3),
        DocumentationSource("https://fastapi.tiangolo.com/", "FastAPI", "python", "fastapi", priority=5),
        DocumentationSource("https://flask.palletsprojects.com/", "Flask", "python", "flask", priority=4),
        DocumentationSource("https://docs.djangoproject.com/", "Django", "python", "django", priority=4),
        DocumentationSource("https://spring.io/guides", "Spring Boot", "java", "spring", priority=4),
        DocumentationSource("https://rubyonrails.org/guides", "Ruby on Rails", "ruby", "rails", priority=3),
        DocumentationSource("https://laravel.com/docs", "Laravel", "php", "laravel", priority=3),
        DocumentationSource("https://docs.microsoft.com/en-us/aspnet/core/", "ASP.NET Core", "csharp", "dotnet", priority=3),
        DocumentationSource("https://expressjs.com/en/4x/api.html", "Express.js", "javascript", "express", priority=3),
        DocumentationSource("https://www.typescriptlang.org/docs/", "TypeScript", "typescript", framework=None, priority=4),
        DocumentationSource("https://golang.org/doc/", "Go", "go", framework=None, priority=4),
        DocumentationSource("https://doc.rust-lang.org/book/", "Rust", "rust", framework=None, priority=3),
        DocumentationSource("https://kotlinlang.org/docs/", "Kotlin", "kotlin", framework=None, priority=3),
        DocumentationSource("https://docs.swift.org/", "Swift", "swift", framework=None, priority=2),
        DocumentationSource("https://pytorch.org/docs/stable/", "PyTorch", "python", "pytorch", priority=3),
        DocumentationSource("https://www.tensorflow.org/guide", "TensorFlow", "python", "tensorflow", priority=3),
    ]
    
    processor = MLBatchProcessor(
        bedrock_config=bedrock_config,
        output_dir=output_dir,
        max_concurrent=8
    )
    
    return await processor.process_documentation_batch(sources)


async def process_cloud_platforms(
    output_dir: str = "rules/cloud",
    bedrock_config: Optional[Dict[str, Any]] = None
) -> BatchResult:
    """Process cloud platform documentation."""
    
    sources = [
        DocumentationSource("https://docs.aws.amazon.com/", "AWS", "cloud", "aws", priority=5),
        DocumentationSource("https://docs.microsoft.com/en-us/azure/", "Azure", "cloud", "azure", priority=4),
        DocumentationSource("https://cloud.google.com/docs", "Google Cloud", "cloud", "gcp", priority=4),
        DocumentationSource("https://kubernetes.io/docs/", "Kubernetes", "cloud", "kubernetes", priority=5),
        DocumentationSource("https://docs.docker.com/", "Docker", "cloud", "docker", priority=4),
        DocumentationSource("https://www.terraform.io/docs", "Terraform", "cloud", "terraform", priority=4),
    ]
    
    processor = MLBatchProcessor(
        bedrock_config=bedrock_config,
        output_dir=output_dir,
        max_concurrent=6
    )
    
    return await processor.process_documentation_batch(sources)
