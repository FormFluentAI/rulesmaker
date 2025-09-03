"""
Next.js Intelligent Categorizer

Specialized categorization system for Next.js documentation that understands
the framework's unique concepts, patterns, and best practices.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import yaml
import json
from pathlib import Path

from .models import ContentAnalysis, CategoryConfidence, ComplexityLevel, ContentType
from .semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)


class NextJSCategory(Enum):
    """Next.js specific categories."""
    ROUTING = "routing"
    DATA_FETCHING = "data-fetching"
    STYLING = "styling"
    DEPLOYMENT = "deployment"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TESTING = "testing"
    API_ROUTES = "api-routes"
    MIDDLEWARE = "middleware"
    CONFIGURATION = "configuration"
    OPTIMIZATION = "optimization"
    TROUBLESHOOTING = "troubleshooting"
    MIGRATION = "migration"
    ADVANCED_PATTERNS = "advanced-patterns"


@dataclass
class NextJSPattern:
    """Next.js specific pattern definition."""
    pattern: str
    confidence: float
    category: NextJSCategory
    subcategory: Optional[str] = None
    difficulty: Optional[str] = None
    context_clues: List[str] = None


class NextJSCategorizer:
    """Intelligent categorizer specialized for Next.js documentation."""
    
    def __init__(self, taxonomy_path: str = "config/intelligent_taxonomy.yaml"):
        """Initialize the Next.js categorizer.
        
        Args:
            taxonomy_path: Path to the taxonomy configuration file
        """
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self._load_taxonomy()
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Initialize Next.js specific patterns
        self.nextjs_patterns = self._initialize_nextjs_patterns()
        
        # Load learned patterns from previous runs
        self.learned_patterns = self._load_learned_patterns()
        
    def _load_taxonomy(self) -> Dict:
        """Load taxonomy configuration."""
        if Path(self.taxonomy_path).exists():
            with open(self.taxonomy_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_learned_patterns(self) -> Dict[str, List[Dict]]:
        """Load learned patterns from previous categorization runs."""
        patterns_file = Path("data/nextjs_learned_patterns.json")
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_learned_patterns(self):
        """Save learned patterns for future use."""
        patterns_file = Path("data/nextjs_learned_patterns.json")
        patterns_file.parent.mkdir(exist_ok=True)
        with open(patterns_file, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)
    
    def _initialize_nextjs_patterns(self) -> List[NextJSPattern]:
        """Initialize Next.js specific patterns."""
        patterns = [
            # Routing Patterns
            NextJSPattern(
                pattern=r"app router|app directory|app/",
                confidence=0.9,
                category=NextJSCategory.ROUTING,
                subcategory="app-router",
                difficulty="intermediate",
                context_clues=["app/", "layout.tsx", "page.tsx", "route.ts"]
            ),
            NextJSPattern(
                pattern=r"pages router|pages directory|pages/",
                confidence=0.9,
                category=NextJSCategory.ROUTING,
                subcategory="page-router",
                difficulty="beginner",
                context_clues=["pages/", "_app.js", "_document.js", "getServerSideProps"]
            ),
            NextJSPattern(
                pattern=r"dynamic routes?|\[.*\]|catch-all",
                confidence=0.8,
                category=NextJSCategory.ROUTING,
                subcategory="dynamic-routing",
                difficulty="intermediate",
                context_clues=["[...slug]", "[id]", "getStaticPaths"]
            ),
            NextJSPattern(
                pattern=r"middleware|middleware\.ts",
                confidence=0.9,
                category=NextJSCategory.MIDDLEWARE,
                subcategory="request-middleware",
                difficulty="advanced",
                context_clues=["middleware.ts", "NextRequest", "NextResponse"]
            ),
            
            # Data Fetching Patterns
            NextJSPattern(
                pattern=r"server components?|use server",
                confidence=0.9,
                category=NextJSCategory.DATA_FETCHING,
                subcategory="server-components",
                difficulty="intermediate",
                context_clues=["use server", "async function", "fetch()"]
            ),
            NextJSPattern(
                pattern=r"client components?|use client",
                confidence=0.9,
                category=NextJSCategory.DATA_FETCHING,
                subcategory="client-components",
                difficulty="beginner",
                context_clues=["use client", "useEffect", "useState"]
            ),
            NextJSPattern(
                pattern=r"SSR|server-side rendering|getServerSideProps",
                confidence=0.8,
                category=NextJSCategory.DATA_FETCHING,
                subcategory="ssr",
                difficulty="intermediate",
                context_clues=["getServerSideProps", "getInitialProps"]
            ),
            NextJSPattern(
                pattern=r"SSG|static generation|getStaticProps",
                confidence=0.8,
                category=NextJSCategory.DATA_FETCHING,
                subcategory="ssg",
                difficulty="intermediate",
                context_clues=["getStaticProps", "getStaticPaths", "generateStaticParams"]
            ),
            NextJSPattern(
                pattern=r"ISR|incremental static regeneration",
                confidence=0.9,
                category=NextJSCategory.DATA_FETCHING,
                subcategory="isr",
                difficulty="advanced",
                context_clues=["revalidate", "on-demand revalidation"]
            ),
            
            # API Routes Patterns
            NextJSPattern(
                pattern=r"api routes?|route handlers?|api/",
                confidence=0.9,
                category=NextJSCategory.API_ROUTES,
                subcategory="route-handlers",
                difficulty="intermediate",
                context_clues=["route.ts", "api/", "GET", "POST", "PUT", "DELETE"]
            ),
            NextJSPattern(
                pattern=r"pages/api|api routes?",
                confidence=0.8,
                category=NextJSCategory.API_ROUTES,
                subcategory="pages-api",
                difficulty="intermediate",
                context_clues=["pages/api/", "req", "res"]
            ),
            
            # Styling Patterns
            NextJSPattern(
                pattern=r"CSS modules?|\.module\.css",
                confidence=0.9,
                category=NextJSCategory.STYLING,
                subcategory="css-modules",
                difficulty="beginner",
                context_clues=[".module.css", "styles.", "className"]
            ),
            NextJSPattern(
                pattern=r"Tailwind CSS|tailwind",
                confidence=0.9,
                category=NextJSCategory.STYLING,
                subcategory="tailwind",
                difficulty="beginner",
                context_clues=["tailwind", "tw-", "bg-", "text-"]
            ),
            NextJSPattern(
                pattern=r"styled-components?|styled-jsx",
                confidence=0.8,
                category=NextJSCategory.STYLING,
                subcategory="styled-components",
                difficulty="intermediate",
                context_clues=["styled-components", "styled-jsx", "styled."]
            ),
            
            # Performance Patterns
            NextJSPattern(
                pattern=r"image optimization|next/image|Image component",
                confidence=0.9,
                category=NextJSCategory.PERFORMANCE,
                subcategory="image-optimization",
                difficulty="beginner",
                context_clues=["next/image", "Image", "optimize", "lazy loading"]
            ),
            NextJSPattern(
                pattern=r"font optimization|next/font|Font component",
                confidence=0.9,
                category=NextJSCategory.PERFORMANCE,
                subcategory="font-optimization",
                difficulty="intermediate",
                context_clues=["next/font", "Font", "font optimization"]
            ),
            NextJSPattern(
                pattern=r"bundle analysis|webpack bundle|analyze",
                confidence=0.8,
                category=NextJSCategory.PERFORMANCE,
                subcategory="bundle-analysis",
                difficulty="advanced",
                context_clues=["@next/bundle-analyzer", "analyze", "webpack"]
            ),
            
            # Deployment Patterns
            NextJSPattern(
                pattern=r"Vercel|vercel\.json|deployment",
                confidence=0.8,
                category=NextJSCategory.DEPLOYMENT,
                subcategory="vercel",
                difficulty="beginner",
                context_clues=["vercel", "vercel.json", "deploy"]
            ),
            NextJSPattern(
                pattern=r"static export|next export|static generation",
                confidence=0.9,
                category=NextJSCategory.DEPLOYMENT,
                subcategory="static-export",
                difficulty="intermediate",
                context_clues=["next export", "output: 'export'", "static"]
            ),
            NextJSPattern(
                pattern=r"Docker|dockerfile|container",
                confidence=0.8,
                category=NextJSCategory.DEPLOYMENT,
                subcategory="docker",
                difficulty="advanced",
                context_clues=["Dockerfile", "docker", "container"]
            ),
            
            # Configuration Patterns
            NextJSPattern(
                pattern=r"next\.config\.|configuration|config",
                confidence=0.8,
                category=NextJSCategory.CONFIGURATION,
                subcategory="next-config",
                difficulty="intermediate",
                context_clues=["next.config.js", "next.config.ts", "config"]
            ),
            NextJSPattern(
                pattern=r"environment variables?|env|process\.env",
                confidence=0.7,
                category=NextJSCategory.CONFIGURATION,
                subcategory="environment",
                difficulty="beginner",
                context_clues=["process.env", ".env", "environment"]
            ),
            
            # Testing Patterns
            NextJSPattern(
                pattern=r"Jest|testing|test|spec",
                confidence=0.8,
                category=NextJSCategory.TESTING,
                subcategory="jest",
                difficulty="intermediate",
                context_clues=["jest", "test", "spec", "describe", "it"]
            ),
            NextJSPattern(
                pattern=r"Playwright|Cypress|e2e|end-to-end",
                confidence=0.9,
                category=NextJSCategory.TESTING,
                subcategory="e2e",
                difficulty="advanced",
                context_clues=["playwright", "cypress", "e2e", "end-to-end"]
            ),
            
            # Security Patterns
            NextJSPattern(
                pattern=r"authentication|auth|login|session",
                confidence=0.8,
                category=NextJSCategory.SECURITY,
                subcategory="authentication",
                difficulty="intermediate",
                context_clues=["auth", "login", "session", "jwt", "oauth"]
            ),
            NextJSPattern(
                pattern=r"CORS|cross-origin|security headers",
                confidence=0.8,
                category=NextJSCategory.SECURITY,
                subcategory="cors",
                difficulty="advanced",
                context_clues=["cors", "cross-origin", "security", "headers"]
            ),
            
            # Migration Patterns
            NextJSPattern(
                pattern=r"migration|migrate|upgrade|from.*to",
                confidence=0.8,
                category=NextJSCategory.MIGRATION,
                subcategory="version-migration",
                difficulty="advanced",
                context_clues=["migration", "upgrade", "from", "to", "breaking changes"]
            ),
            
            # Advanced Patterns
            NextJSPattern(
                pattern=r"parallel routes?|intercepting routes?|route groups?",
                confidence=0.9,
                category=NextJSCategory.ADVANCED_PATTERNS,
                subcategory="advanced-routing",
                difficulty="expert",
                context_clues=["@", "parallel", "intercepting", "route groups"]
            ),
            NextJSPattern(
                pattern=r"streaming|suspense|loading\.tsx",
                confidence=0.8,
                category=NextJSCategory.ADVANCED_PATTERNS,
                subcategory="streaming",
                difficulty="advanced",
                context_clues=["streaming", "suspense", "loading.tsx", "loading.js"]
            ),
        ]
        
        return patterns
    
    async def categorize_nextjs_content(
        self, 
        content: str, 
        url: str = "",
        metadata: Optional[Dict] = None
    ) -> Dict[str, CategoryConfidence]:
        """Categorize Next.js documentation content.
        
        Args:
            content: The documentation content
            url: Optional URL for additional context
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary of categories with confidence scores
        """
        logger.debug(f"Categorizing Next.js content from {url}")
        
        # Start with semantic analysis
        analysis = await self.semantic_analyzer.analyze_content(content, url)
        
        # Apply Next.js specific pattern matching
        nextjs_categories = self._apply_nextjs_patterns(content, url)
        
        # Apply learned patterns
        learned_categories = self._apply_learned_patterns(content, url)
        
        # Merge and weight results
        final_categories = self._merge_categorization_results(
            analysis.content_categories,
            nextjs_categories,
            learned_categories
        )
        
        # Apply context-based adjustments
        adjusted_categories = self._apply_context_adjustments(
            final_categories, content, url, metadata
        )
        
        return adjusted_categories
    
    def _apply_nextjs_patterns(
        self, 
        content: str, 
        url: str
    ) -> Dict[str, CategoryConfidence]:
        """Apply Next.js specific pattern matching."""
        categories = {}
        content_lower = content.lower()
        url_lower = str(url).lower()
        
        for pattern in self.nextjs_patterns:
            # Check pattern in content
            if re.search(pattern.pattern, content_lower):
                confidence = pattern.confidence
                
                # Boost confidence if context clues are present
                context_boost = 0.0
                if pattern.context_clues:
                    for clue in pattern.context_clues:
                        if clue.lower() in content_lower or clue.lower() in url_lower:
                            context_boost += 0.1
                
                confidence = min(confidence + context_boost, 1.0)
                
                # Apply difficulty-based adjustments
                if pattern.difficulty:
                    confidence = self._adjust_confidence_by_difficulty(
                        confidence, pattern.difficulty, content
                    )
                
                category_key = pattern.category.value
                if category_key not in categories:
                    categories[category_key] = CategoryConfidence(
                        confidence=confidence,
                        topics=[pattern.subcategory or pattern.category.value],
                        patterns=[pattern.pattern]
                    )
                else:
                    # Update existing category with higher confidence
                    if confidence > categories[category_key].confidence:
                        categories[category_key].confidence = confidence
                        categories[category_key].topics.append(pattern.subcategory or pattern.category.value)
                        categories[category_key].patterns.append(pattern.pattern)
        
        return categories
    
    def _apply_learned_patterns(
        self, 
        content: str, 
        url: str
    ) -> Dict[str, CategoryConfidence]:
        """Apply learned patterns from previous categorizations."""
        categories = {}
        
        for category, patterns in self.learned_patterns.items():
            for pattern_data in patterns:
                pattern = pattern_data.get('pattern', '')
                confidence = pattern_data.get('confidence', 0.5)
                
                if re.search(pattern, content.lower()):
                    if category not in categories:
                        categories[category] = CategoryConfidence(
                            confidence=confidence,
                            topics=[pattern_data.get('subcategory', category)],
                            patterns=[pattern]
                        )
                    else:
                        # Update with higher confidence
                        if confidence > categories[category].confidence:
                            categories[category].confidence = confidence
        
        return categories
    
    def _merge_categorization_results(
        self,
        semantic_categories: Dict[str, CategoryConfidence],
        nextjs_categories: Dict[str, CategoryConfidence],
        learned_categories: Dict[str, CategoryConfidence]
    ) -> Dict[str, CategoryConfidence]:
        """Merge different categorization results with appropriate weighting."""
        merged = {}
        
        # Weight the different sources
        weights = {
            'semantic': 0.3,
            'nextjs': 0.5,  # Higher weight for Next.js specific patterns
            'learned': 0.2
        }
        
        # Merge semantic categories
        for category, confidence in semantic_categories.items():
            merged[category] = CategoryConfidence(
                confidence=confidence.confidence * weights['semantic'],
                topics=confidence.topics.copy(),
                patterns=confidence.patterns.copy()
            )
        
        # Merge Next.js specific categories
        for category, confidence in nextjs_categories.items():
            if category in merged:
                # Combine with existing
                merged[category].confidence += confidence.confidence * weights['nextjs']
                merged[category].topics.extend(confidence.topics)
                merged[category].patterns.extend(confidence.patterns)
            else:
                merged[category] = CategoryConfidence(
                    confidence=confidence.confidence * weights['nextjs'],
                    topics=confidence.topics.copy(),
                    patterns=confidence.patterns.copy()
                )
        
        # Merge learned categories
        for category, confidence in learned_categories.items():
            if category in merged:
                merged[category].confidence += confidence.confidence * weights['learned']
                merged[category].topics.extend(confidence.topics)
                merged[category].patterns.extend(confidence.patterns)
            else:
                merged[category] = CategoryConfidence(
                    confidence=confidence.confidence * weights['learned'],
                    topics=confidence.topics.copy(),
                    patterns=confidence.patterns.copy()
                )
        
        # Normalize confidence scores
        for category in merged:
            merged[category].confidence = min(merged[category].confidence, 1.0)
        
        return merged
    
    def _apply_context_adjustments(
        self,
        categories: Dict[str, CategoryConfidence],
        content: str,
        url: str,
        metadata: Optional[Dict]
    ) -> Dict[str, CategoryConfidence]:
        """Apply context-based adjustments to categorization."""
        adjusted = categories.copy()
        
        # URL-based adjustments
        url_str = str(url).lower()
        if 'api' in url_str:
            if 'api-routes' not in adjusted:
                adjusted['api-routes'] = CategoryConfidence(
                    confidence=0.8,
                    topics=['api-routes'],
                    patterns=['url-based-detection']
                )
            else:
                adjusted['api-routes'].confidence = min(adjusted['api-routes'].confidence + 0.2, 1.0)
        
        if 'deploy' in url_str or 'vercel' in url_str:
            if 'deployment' not in adjusted:
                adjusted['deployment'] = CategoryConfidence(
                    confidence=0.7,
                    topics=['deployment'],
                    patterns=['url-based-detection']
                )
            else:
                adjusted['deployment'].confidence = min(adjusted['deployment'].confidence + 0.2, 1.0)
        
        # Content length adjustments
        content_length = len(content)
        if content_length < 500:  # Short content
            for category in adjusted:
                adjusted[category].confidence *= 0.8  # Reduce confidence for short content
        elif content_length > 5000:  # Long content
            for category in adjusted:
                adjusted[category].confidence *= 1.1  # Slightly boost confidence for detailed content
                adjusted[category].confidence = min(adjusted[category].confidence, 1.0)
        
        # Metadata-based adjustments
        if metadata:
            if metadata.get('is_tutorial', False):
                if 'tutorial' not in adjusted:
                    adjusted['tutorial'] = CategoryConfidence(
                        confidence=0.7,
                        topics=['tutorial'],
                        patterns=['metadata-based-detection']
                    )
            
            if metadata.get('is_reference', False):
                if 'reference' not in adjusted:
                    adjusted['reference'] = CategoryConfidence(
                        confidence=0.7,
                        topics=['reference'],
                        patterns=['metadata-based-detection']
                    )
        
        return adjusted
    
    def _adjust_confidence_by_difficulty(
        self, 
        confidence: float, 
        difficulty: str, 
        content: str
    ) -> float:
        """Adjust confidence based on content difficulty indicators."""
        difficulty_indicators = {
            'beginner': ['basic', 'simple', 'getting started', 'introduction'],
            'intermediate': ['guide', 'how to', 'examples', 'common'],
            'advanced': ['advanced', 'optimization', 'best practices', 'architecture'],
            'expert': ['internals', 'deep dive', 'custom', 'expert', 'complex']
        }
        
        content_lower = content.lower()
        indicators = difficulty_indicators.get(difficulty, [])
        
        # Boost confidence if difficulty indicators are present
        for indicator in indicators:
            if indicator in content_lower:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def learn_from_categorization(
        self,
        content: str,
        url: str,
        categories: Dict[str, CategoryConfidence],
        user_feedback: Optional[Dict] = None
    ):
        """Learn from categorization results and user feedback."""
        logger.debug(f"Learning from categorization of {url}")
        
        # Extract patterns that led to high-confidence categorizations
        for category, confidence in categories.items():
            if confidence.confidence > 0.7:  # High confidence
                if category not in self.learned_patterns:
                    self.learned_patterns[category] = []
                
                # Extract new patterns from content
                new_patterns = self._extract_patterns_from_content(content, category)
                
                for pattern in new_patterns:
                    # Check if pattern already exists
                    existing = any(
                        p.get('pattern') == pattern 
                        for p in self.learned_patterns[category]
                    )
                    
                    if not existing:
                        self.learned_patterns[category].append({
                            'pattern': pattern,
                            'confidence': confidence.confidence,
                            'subcategory': confidence.topics[0] if confidence.topics else category,
                            'source_url': url,
                            'timestamp': str(datetime.now())
                        })
        
        # Apply user feedback if available
        if user_feedback:
            self._apply_user_feedback(user_feedback, categories)
        
        # Save learned patterns
        self._save_learned_patterns()
    
    def _extract_patterns_from_content(
        self, 
        content: str, 
        category: str
    ) -> List[str]:
        """Extract potential patterns from content for learning."""
        patterns = []
        
        # Extract technical terms and phrases
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content)
        for term in technical_terms:
            if len(term) > 3:  # Filter out short terms
                patterns.append(term.lower())
        
        # Extract code patterns
        code_patterns = re.findall(r'`([^`]+)`', content)
        for pattern in code_patterns:
            if len(pattern) > 2 and len(pattern) < 50:  # Reasonable length
                patterns.append(pattern.lower())
        
        # Extract file extensions and paths
        file_patterns = re.findall(r'\.[a-z]+(?:\.[a-z]+)?', content)
        for pattern in file_patterns:
            patterns.append(pattern.lower())
        
        return patterns[:10]  # Limit to top 10 patterns
    
    def _apply_user_feedback(
        self,
        feedback: Dict,
        categories: Dict[str, CategoryConfidence]
    ):
        """Apply user feedback to improve categorization."""
        # This would implement feedback-based learning
        # For now, it's a placeholder for future enhancement
        logger.debug("Applying user feedback to categorization")
    
    def get_categorization_stats(self) -> Dict[str, Any]:
        """Get statistics about categorization performance."""
        stats = {
            'total_patterns': len(self.nextjs_patterns),
            'learned_patterns': sum(len(patterns) for patterns in self.learned_patterns.values()),
            'categories_covered': len(self.learned_patterns),
            'pattern_breakdown': {
                category: len(patterns) 
                for category, patterns in self.learned_patterns.items()
            }
        }
        
        return stats
