"""
Semantic Content Analysis Engine.

AI-powered content understanding and categorization system that provides
deep semantic analysis of documentation content.
"""

import re
import asyncio
from typing import Dict, List, Optional
import yaml
import os

# from ..bedrock_integration import BedrockRulesMaker  # Commented out for now
from .models import (
    ContentAnalysis, 
    CategoryConfidence, 
    ComplexityLevel, 
    ContentType
)


class SemanticAnalyzer:
    """AI-powered content understanding and categorization."""
    
    def __init__(self, bedrock_config: Optional[Dict] = None):
        """Initialize the semantic analyzer.
        
        Args:
            bedrock_config: Configuration for Bedrock integration
        """
        self.bedrock_config = bedrock_config or {}
        self._taxonomy = None
        self._load_taxonomy()
        
    def _load_taxonomy(self):
        """Load the intelligent taxonomy configuration."""
        taxonomy_path = "config/intelligent_taxonomy.yaml"
        if os.path.exists(taxonomy_path):
            with open(taxonomy_path, 'r') as f:
                self._taxonomy = yaml.safe_load(f)
        else:
            # Default taxonomy if file doesn't exist
            self._taxonomy = self._get_default_taxonomy()
    
    def _get_default_taxonomy(self) -> Dict:
        """Get default taxonomy configuration."""
        return {
            "frameworks": {
                "nextjs": {
                    "categories": {
                        "routing": {
                            "patterns": ["app router", "page router", "dynamic routes", "api routes"],
                            "subcategories": {
                                "app-router": ["layout", "page", "loading", "error", "not-found"],
                                "page-router": ["pages", "api", "dynamic", "_app", "_document"]
                            },
                            "difficulty_markers": {
                                "beginner": ["basic routing", "static routes"],
                                "advanced": ["parallel routes", "intercepting routes", "middleware"]
                            }
                        },
                        "data-fetching": {
                            "patterns": ["server components", "client components", "data fetching"],
                            "subcategories": {
                                "server-side": ["generateStaticParams", "revalidation", "streaming"],
                                "client-side": ["useEffect", "SWR", "React Query"]
                            },
                            "context_clues": [
                                "use server", "use client", "fetch()", "getServerSideProps"
                            ]
                        }
                    }
                },
                "react": {
                    "categories": {
                        "hooks": {
                            "patterns": ["useState", "useEffect", "useContext", "custom hooks"],
                            "difficulty_markers": {
                                "beginner": ["useState", "useEffect"],
                                "advanced": ["useCallback", "useMemo", "useReducer", "custom hooks"]
                            }
                        },
                        "components": {
                            "patterns": ["functional components", "class components", "JSX"],
                            "subcategories": {
                                "functional": ["arrow functions", "function declarations"],
                                "class": ["extends React.Component", "render method"]
                            }
                        }
                    }
                },
                "python": {
                    "categories": {
                        "web-frameworks": {
                            "patterns": ["fastapi", "flask", "django", "starlette"],
                            "subcategories": {
                                "fastapi": ["pydantic", "dependency injection", "async/await"],
                                "django": ["models", "views", "templates", "admin"]
                            }
                        }
                    }
                }
            }
        }
    
    async def analyze_content(self, content: str, url: str) -> ContentAnalysis:
        """Deep semantic analysis of documentation content.
        
        Args:
            content: The documentation content to analyze
            url: The URL of the content
            
        Returns:
            ContentAnalysis with detected technologies, categories, and metadata
        """
        # Basic technology detection
        primary_tech, secondary_techs = self._detect_technologies(content, url)
        
        # Content categorization
        categories = self._categorize_content(content, primary_tech)
        
        # Complexity analysis
        complexity = self._analyze_complexity(content, categories)
        
        # Content type detection
        content_type = self._detect_content_type(content, url)
        
        # Version detection
        version = self._detect_version(content, primary_tech)
        
        # Prerequisites detection
        prerequisites = self._detect_prerequisites(content, primary_tech)
        
        # Quality scoring
        quality_score = self._calculate_quality_score(content, categories)
        
        # Enhanced analysis with Bedrock if available
        if self.bedrock_config:
            enhanced_analysis = await self._enhance_with_llm(content, url, primary_tech)
            categories.update(enhanced_analysis.get('categories', {}))
        
        # Enhanced analysis with Context7 for latest documentation
        context7_content = await self._enhance_with_context7(content, url, primary_tech)
        if context7_content:
            # Re-analyze with enhanced content from Context7
            enhanced_primary_tech, enhanced_secondary_techs = self._detect_technologies(context7_content, url)
            enhanced_categories = self._categorize_content(context7_content, enhanced_primary_tech)
            
            # Merge results, preferring Context7 data where available
            if enhanced_primary_tech != "unknown":
                primary_tech = enhanced_primary_tech
            if enhanced_secondary_techs:
                secondary_techs = list(set(secondary_techs + enhanced_secondary_techs))
            
            # Update categories with Context7 enhanced data
            for cat_name, cat_data in enhanced_categories.items():
                if cat_name in categories:
                    # Merge confidence and topics
                    categories[cat_name].confidence = max(categories[cat_name].confidence, cat_data.confidence)
                    categories[cat_name].topics = list(set(categories[cat_name].topics + cat_data.topics))
                else:
                    categories[cat_name] = cat_data
        
        return ContentAnalysis(
            primary_technology=primary_tech,
            secondary_technologies=secondary_techs,
            content_categories=categories,
            complexity_level=complexity,
            content_type=content_type,
            framework_version=version,
            prerequisites=prerequisites,
            language_detected=self._detect_language(content),
            code_examples_count=len(re.findall(r'```[\s\S]*?```', content)),
            external_links_count=len(re.findall(r'https?://[^\s\)]+', content)),
            quality_score=quality_score,
            metadata={
                "url": url,
                "content_length": len(content),
                "analysis_timestamp": asyncio.get_event_loop().time()
            }
        )
    
    def _detect_technologies(self, content: str, url: str) -> tuple[str, List[str]]:
        """Detect primary and secondary technologies."""
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Technology patterns with weights
        tech_patterns = {
            "nextjs": {
                "patterns": [r"next\.?js", r"app router", r"page router", r"next/", r"getServerSideProps"],
                "weight": 1.0,
                "url_patterns": ["nextjs.org", "next"]
            },
            "react": {
                "patterns": [r"react", r"jsx", r"usestate", r"useeffect", r"component"],
                "weight": 0.9,
                "url_patterns": ["react.dev", "reactjs.org"]
            },
            "vue": {
                "patterns": [r"vue\.?js", r"@click", r"v-model", r"composition api"],
                "weight": 1.0,
                "url_patterns": ["vuejs.org"]
            },
            "angular": {
                "patterns": [r"angular", r"@component", r"@injectable", r"typescript"],
                "weight": 1.0,
                "url_patterns": ["angular.io"]
            },
            "python": {
                "patterns": [r"python", r"def\s+\w+", r"import\s+\w+", r"pip install"],
                "weight": 0.8,
                "url_patterns": ["python.org", "docs.python"]
            },
            "fastapi": {
                "patterns": [r"fastapi", r"@app\.get", r"pydantic", r"uvicorn"],
                "weight": 1.0,
                "url_patterns": ["fastapi.tiangolo.com"]
            },
            "django": {
                "patterns": [r"django", r"models\.Model", r"admin\.site", r"urls\.py"],
                "weight": 1.0,
                "url_patterns": ["djangoproject.com"]
            }
        }
        
        scores = {}
        for tech, config in tech_patterns.items():
            score = 0
            
            # Content pattern matching
            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, content_lower))
                score += matches * config["weight"]
            
            # URL pattern matching (higher weight)
            for url_pattern in config.get("url_patterns", []):
                if url_pattern in url_lower:
                    score += 10 * config["weight"]
            
            if score > 0:
                scores[tech] = score
        
        if not scores:
            return "unknown", []
        
        # Sort by score
        sorted_techs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_techs[0][0]
        secondary = [tech for tech, _ in sorted_techs[1:4] if scores[tech] > scores[primary] * 0.3]
        
        return primary, secondary
    
    def _categorize_content(self, content: str, primary_tech: str) -> Dict[str, CategoryConfidence]:
        """Categorize content based on detected technology."""
        categories = {}
        
        if primary_tech in self._taxonomy.get("frameworks", {}):
            tech_config = self._taxonomy["frameworks"][primary_tech]
            
            for category, config in tech_config.get("categories", {}).items():
                confidence = 0.0
                detected_topics = []
                detected_patterns = []
                
                # Pattern matching
                for pattern in config.get("patterns", []):
                    if re.search(pattern.lower(), content.lower()):
                        confidence += 0.3
                        detected_patterns.append(pattern)
                
                # Context clues
                for clue in config.get("context_clues", []):
                    if clue.lower() in content.lower():
                        confidence += 0.2
                        detected_topics.append(clue)
                
                # Subcategory detection
                for subcat, patterns in config.get("subcategories", {}).items():
                    for pattern in patterns:
                        if pattern.lower() in content.lower():
                            confidence += 0.1
                            detected_topics.append(f"{subcat}: {pattern}")
                
                # Normalize confidence
                confidence = min(confidence, 1.0)
                
                if confidence > 0.1:
                    categories[category] = CategoryConfidence(
                        confidence=confidence,
                        topics=detected_topics,
                        patterns=detected_patterns
                    )
        
        return categories
    
    def _analyze_complexity(self, content: str, categories: Dict) -> ComplexityLevel:
        """Analyze content complexity level."""
        complexity_indicators = {
            "advanced": [
                "advanced", "complex", "expert", "optimization", "performance",
                "architecture", "design patterns", "best practices"
            ],
            "expert": [
                "internals", "deep dive", "under the hood", "custom implementation",
                "advanced concepts", "expert guide"
            ],
            "beginner": [
                "getting started", "introduction", "basic", "tutorial", "quickstart",
                "hello world", "first steps"
            ]
        }
        
        content_lower = content.lower()
        scores = {"beginner": 0, "intermediate": 0, "advanced": 0, "expert": 0}
        
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    scores[level] += 1
        
        # Add points based on categories
        if len(categories) > 3:
            scores["advanced"] += 1
        if any(cat.confidence > 0.8 for cat in categories.values()):
            scores["intermediate"] += 1
        
        # Determine complexity level
        max_score = max(scores.values())
        if max_score == 0:
            return ComplexityLevel.INTERMEDIATE
        
        for level, score in scores.items():
            if score == max_score:
                return ComplexityLevel(level)
        
        return ComplexityLevel.INTERMEDIATE
    
    def _detect_content_type(self, content: str, url: str) -> ContentType:
        """Detect the type of documentation content."""
        content_lower = content.lower()
        url_lower = url.lower()
        
        type_indicators = {
            ContentType.TUTORIAL: ["tutorial", "getting started", "walkthrough", "step by step"],
            ContentType.REFERENCE: ["reference", "api", "documentation", "docs"],
            ContentType.GUIDE: ["guide", "how to", "best practices", "examples"],
            ContentType.API_DOCS: ["api", "endpoints", "methods", "parameters"],
            ContentType.TROUBLESHOOTING: ["troubleshooting", "common issues", "debugging", "faq"]
        }
        
        scores = {t: 0 for t in ContentType}
        
        for content_type, indicators in type_indicators.items():
            for indicator in indicators:
                if indicator in content_lower or indicator in url_lower:
                    scores[content_type] += 1
        
        max_score = max(scores.values())
        if max_score == 0:
            return ContentType.REFERENCE
        
        for content_type, score in scores.items():
            if score == max_score:
                return content_type
        
        return ContentType.REFERENCE
    
    def _detect_version(self, content: str, technology: str) -> Optional[str]:
        """Detect framework/technology version."""
        version_patterns = {
            "nextjs": [r"next\.?js\s+(\d+\.[\d\.]+)", r"version\s+(\d+\.[\d\.]+)"],
            "react": [r"react\s+(\d+\.[\d\.]+)", r"version\s+(\d+\.[\d\.]+)"],
            "python": [r"python\s+(\d+\.[\d\.]+)", r"version\s+(\d+\.[\d\.]+)"]
        }
        
        if technology in version_patterns:
            for pattern in version_patterns[technology]:
                match = re.search(pattern, content.lower())
                if match:
                    return match.group(1)
        
        return None
    
    def _detect_prerequisites(self, content: str, technology: str) -> List[str]:
        """Detect prerequisites mentioned in the content."""
        prereq_patterns = [
            r"requires?\s+([^.]+)",
            r"prerequisite[s]?[:\s]+([^.]+)",
            r"before\s+starting[^.]*you\s+should\s+know\s+([^.]+)",
            r"familiarity\s+with\s+([^.]+)"
        ]
        
        prerequisites = []
        content_lower = content.lower()
        
        for pattern in prereq_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                # Clean up the prerequisite text
                prereq = re.sub(r'[^\w\s,-]', '', match).strip()
                if len(prereq) > 3 and len(prereq) < 100:
                    prerequisites.append(prereq)
        
        return list(set(prerequisites))[:5]  # Limit to 5 most relevant
    
    def _detect_language(self, content: str) -> Optional[str]:
        """Detect the primary programming language in the content."""
        language_patterns = {
            "javascript": [r"function\s+\w+", r"const\s+\w+", r"=>\s*{", r"npm install"],
            "typescript": [r"interface\s+\w+", r"type\s+\w+", r":\s*string", r":\s*number"],
            "python": [r"def\s+\w+", r"import\s+\w+", r"class\s+\w+", r"pip install"],
            "java": [r"public\s+class", r"public\s+static", r"@Override"],
            "rust": [r"fn\s+\w+", r"let\s+mut", r"cargo"],
            "go": [r"func\s+\w+", r"package\s+\w+", r"go\s+mod"]
        }
        
        scores = {}
        content_lower = content.lower()
        
        for lang, patterns in language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            
            if score > 0:
                scores[lang] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return None
    
    def _calculate_quality_score(self, content: str, categories: Dict) -> float:
        """Calculate a quality score for the content."""
        score = 0.0
        
        # Content length
        if 500 < len(content) < 10000:
            score += 0.2
        elif len(content) >= 10000:
            score += 0.3
        
        # Code examples
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        score += min(code_blocks * 0.1, 0.3)
        
        # Categories detected
        if categories:
            avg_confidence = sum(cat.confidence for cat in categories.values()) / len(categories)
            score += avg_confidence * 0.3
        
        # Structure indicators (headings)
        headings = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        score += min(headings * 0.05, 0.2)
        
        return min(score, 1.0)
    
    async def _resolve_library_id(self, technology: str, url: str) -> Optional[str]:
        """Resolve library ID using context7 server for documentation fetching.
        
        Args:
            technology: Detected technology (e.g., 'nextjs', 'react')
            url: Original URL for context
            
        Returns:
            Library ID in format 'org/project' or None if not found
        """
        try:
            # Use the context7 resolve library ID tool
            # This would typically be called via the MCP tool
            # For now, we'll implement a mapping based on common libraries
            
            library_mappings = {
                "nextjs": "vercel/next.js",
                "react": "facebook/react",
                "vue": "vuejs/vue",
                "angular": "angular/angular",
                "python": "python/cpython",
                "fastapi": "tiangolo/fastapi",
                "django": "django/django",
                "flask": "pallets/flask",
                "typescript": "microsoft/typescript",
                "nodejs": "nodejs/node"
            }
            
            # Try to resolve from URL patterns
            url_lower = url.lower()
            for tech, lib_id in library_mappings.items():
                if tech in url_lower or any(pattern in url_lower for pattern in [f"{tech}.org", f"{tech}js.org", f"{tech}.com"]):
                    return lib_id
            
            # Fallback to direct mapping
            return library_mappings.get(technology)
            
        except Exception as e:
            print(f"Library ID resolution failed: {e}")
            return None
    
    async def _fetch_context7_docs(self, library_id: str, query: str = "") -> Optional[str]:
        """Fetch latest documentation from context7 server.
        
        Args:
            library_id: Library ID in format 'org/project'
            query: Optional search query for specific documentation
            
        Returns:
            Latest documentation content or None if fetch fails
        """
        try:
            # Use the context7 get library docs tool
            # This would typically be called via the MCP tool
            # For now, we'll simulate the integration
            
            # Simulate fetching documentation based on library_id
            if not library_id:
                return None
                
            # In a real implementation, this would call:
            # result = await mcp_context7_get-library-docs(library_id=library_id, query=query)
            
            # For demonstration, return a placeholder
            # The actual implementation would use the MCP tool to fetch real docs
            return f"Fetched latest documentation for {library_id}"
            
        except Exception as e:
            print(f"Context7 documentation fetch failed: {e}")
            return None
    
    async def _enhance_with_context7(self, content: str, url: str, primary_tech: str) -> Optional[str]:
        """Enhance content analysis by fetching latest docs from context7.
        
        Args:
            content: Original content
            url: Source URL
            primary_tech: Detected primary technology
            
        Returns:
            Enhanced content with latest documentation or None
        """
        try:
            # Resolve library ID
            library_id = await self._resolve_library_id(primary_tech, url)
            if not library_id:
                return None
            
            # Fetch latest documentation
            latest_docs = await self._fetch_context7_docs(library_id)
            if not latest_docs:
                return None
            
            # Combine original content with latest docs for enhanced analysis
            enhanced_content = f"""
            Original Content:
            {content}
            
            Latest Documentation from Context7 ({library_id}):
            {latest_docs}
            """
            
            return enhanced_content
            
        except Exception as e:
            print(f"Context7 enhancement failed: {e}")
            return None
    
    async def _enhance_with_llm(self, content: str, url: str, primary_tech: str) -> Dict:
        """Enhance analysis using LLM via Bedrock."""
        try:
            # bedrock_maker = BedrockRulesMaker(**self.bedrock_config)
            
            # Create a prompt for enhanced analysis
            # prompt = f"""
            # Analyze this {primary_tech} documentation content and identify:
            # 1. Specific technical categories and subcategories
            # 2. Confidence levels for each category (0-1)
            # 3. Key topics and patterns within each category
            # 
            # Content preview: {content[:1000]}...
            # 
            # Return a structured analysis focusing on technical categorization.
            # """
            
            # Use Bedrock to get enhanced analysis
            # This is a simplified example - in practice you'd use a more structured prompt
            # response = await bedrock_maker._call_bedrock_async(prompt)
            
            # Parse the response (this would need more sophisticated parsing)
            # For now, return a simple enhancement
            return {
                "categories": {
                    "llm_enhanced": {
                        "confidence": 0.7,
                        "topics": ["llm_analyzed_content"],
                        "patterns": ["advanced_pattern_recognition"]
                    }
                }
            }
            
        except Exception as e:
            # Graceful fallback if LLM enhancement fails
            print(f"LLM enhancement failed: {e}")
            return {"categories": {}}