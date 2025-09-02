"""
Intent Analyzer for Natural Language Queries.

Analyzes user queries to understand their intent, extract key entities,
and provide structured information for processing.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass

from ..intelligence.models import ComplexityLevel


class QueryType(str, Enum):
    """Types of queries users can make."""
    HOW_TO = "how_to"              # "How do I..."
    WHAT_IS = "what_is"            # "What is..."
    BEST_PRACTICES = "best_practices"  # "What are the best practices..."
    EXAMPLES = "examples"          # "Show me examples..."
    COMPARISON = "comparison"      # "What's the difference between..."
    TROUBLESHOOTING = "troubleshooting"  # "Why doesn't this work..."
    INSTALLATION = "installation"  # "How to install..."
    CONFIGURATION = "configuration"  # "How to configure..."
    GENERAL = "general"           # General questions


class IntentCategory(str, Enum):
    """Categories of user intent."""
    LEARNING = "learning"         # User wants to learn something
    PROBLEM_SOLVING = "problem_solving"  # User has a specific problem
    EXPLORATION = "exploration"   # User is exploring options
    IMPLEMENTATION = "implementation"  # User wants to implement something
    OPTIMIZATION = "optimization"  # User wants to improve something


@dataclass
class QueryIntent:
    """Structured representation of user query intent."""
    query_type: QueryType
    intent_category: IntentCategory
    technologies: List[str]
    topics: List[str]
    complexity_level: ComplexityLevel
    urgency: str  # low, medium, high
    specificity: float  # 0.0 to 1.0
    entities: Dict[str, List[str]]
    keywords: List[str]
    context_clues: List[str]


class IntentAnalyzer:
    """Analyzes user queries to understand intent and extract entities."""
    
    def __init__(self):
        """Initialize the intent analyzer."""
        self.technology_patterns = self._build_technology_patterns()
        self.query_patterns = self._build_query_patterns()
        self.topic_patterns = self._build_topic_patterns()
        self.urgency_indicators = self._build_urgency_indicators()
        self.complexity_indicators = self._build_complexity_indicators()
    
    def _build_technology_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for technology detection."""
        return {
            "javascript": ["javascript", "js", "node", "nodejs", "npm", "yarn"],
            "typescript": ["typescript", "ts", "tsc"],
            "react": ["react", "jsx", "reactjs", "react.js"],
            "nextjs": ["nextjs", "next.js", "next", "vercel"],
            "vue": ["vue", "vuejs", "vue.js", "nuxt"],
            "angular": ["angular", "ng", "angular cli"],
            "python": ["python", "py", "pip", "conda", "python3"],
            "django": ["django", "django rest", "drf"],
            "fastapi": ["fastapi", "fast api", "starlette"],
            "flask": ["flask", "werkzeug"],
            "java": ["java", "jdk", "jvm", "maven", "gradle"],
            "spring": ["spring", "spring boot", "spring framework"],
            "csharp": ["c#", "csharp", "dotnet", ".net", "asp.net"],
            "go": ["go", "golang", "go mod"],
            "rust": ["rust", "cargo", "rustc"],
            "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
            "azure": ["azure", "microsoft azure"],
            "docker": ["docker", "container", "dockerfile"],
            "kubernetes": ["kubernetes", "k8s", "kubectl"],
            "mongodb": ["mongodb", "mongo", "nosql"],
            "postgresql": ["postgresql", "postgres", "psql"],
            "mysql": ["mysql", "mariadb"],
            "redis": ["redis", "cache"]
        }
    
    def _build_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Build patterns for query type detection."""
        return {
            QueryType.HOW_TO: [
                r"how (?:do i|to|can i)",
                r"how (?:do you|would you)",
                r"what's the (?:way|process) to",
                r"steps to",
                r"guide (?:to|for)",
                r"tutorial (?:on|for)"
            ],
            QueryType.WHAT_IS: [
                r"what is",
                r"what are",
                r"what's",
                r"define",
                r"explain",
                r"meaning of"
            ],
            QueryType.BEST_PRACTICES: [
                r"best practices",
                r"recommended (?:way|approach)",
                r"good practices",
                r"standards for",
                r"conventions",
                r"guidelines"
            ],
            QueryType.EXAMPLES: [
                r"examples? of",
                r"show me",
                r"demonstrate",
                r"sample",
                r"code examples?",
                r"give me (?:an? )?examples?"
            ],
            QueryType.COMPARISON: [
                r"(?:difference|diff) between",
                r"compare",
                r"vs",
                r"versus",
                r"which is better",
                r"pros and cons"
            ],
            QueryType.TROUBLESHOOTING: [
                r"why (?:doesn't|does not|won't|will not)",
                r"(?:error|problem|issue) with",
                r"not working",
                r"(?:fix|solve|debug)",
                r"troubleshoot",
                r"broken"
            ],
            QueryType.INSTALLATION: [
                r"install",
                r"setup",
                r"set up",
                r"getting started",
                r"initial setup"
            ],
            QueryType.CONFIGURATION: [
                r"configur",
                r"settings",
                r"options",
                r"customize",
                r"setup"
            ]
        }
    
    def _build_topic_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for topic detection."""
        return {
            "authentication": ["auth", "login", "signin", "jwt", "oauth", "session", "token"],
            "authorization": ["permissions", "roles", "access control", "rbac"],
            "routing": ["routing", "routes", "navigation", "url", "path"],
            "state_management": ["state", "store", "redux", "vuex", "context"],
            "data_fetching": ["api", "fetch", "http", "ajax", "request", "rest", "graphql"],
            "styling": ["css", "styling", "styles", "theme", "design"],
            "testing": ["test", "testing", "unit test", "integration test", "e2e"],
            "deployment": ["deploy", "deployment", "production", "build", "ci/cd"],
            "performance": ["performance", "optimization", "speed", "cache", "lazy"],
            "security": ["security", "secure", "vulnerability", "xss", "csrf"],
            "database": ["database", "db", "sql", "query", "migration", "schema"],
            "api_design": ["api design", "rest api", "endpoints", "swagger", "openapi"],
            "error_handling": ["error", "exception", "try catch", "error handling"],
            "validation": ["validation", "validate", "form validation", "input validation"],
            "responsive_design": ["responsive", "mobile", "media queries", "breakpoints"],
            "accessibility": ["accessibility", "a11y", "wcag", "screen reader"],
            "internationalization": ["i18n", "internationalization", "localization", "l10n"],
            "monitoring": ["monitoring", "logging", "metrics", "observability"],
            "microservices": ["microservices", "distributed", "service mesh"],
            "devops": ["devops", "ci/cd", "pipeline", "automation"]
        }
    
    def _build_urgency_indicators(self) -> Dict[str, List[str]]:
        """Build patterns for urgency detection."""
        return {
            "high": ["urgent", "asap", "immediately", "critical", "broken", "not working", "stuck"],
            "medium": ["soon", "important", "needed", "deadline", "problem"],
            "low": ["when possible", "eventually", "curious", "wondering", "interested"]
        }
    
    def _build_complexity_indicators(self) -> Dict[ComplexityLevel, List[str]]:
        """Build patterns for complexity level detection."""
        return {
            ComplexityLevel.BEGINNER: [
                "beginner", "new to", "just started", "getting started", "basic",
                "simple", "first time", "introduction", "hello world"
            ],
            ComplexityLevel.INTERMEDIATE: [
                "intermediate", "some experience", "familiar with", "understand basics",
                "next level", "improve", "better way"
            ],
            ComplexityLevel.ADVANCED: [
                "advanced", "experienced", "complex", "sophisticated", "optimization",
                "performance", "scalable", "architecture", "best practices"
            ],
            ComplexityLevel.EXPERT: [
                "expert", "deep dive", "internals", "custom implementation", "enterprise",
                "production ready", "high performance", "at scale"
            ]
        }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """Analyze a user query to extract intent and entities.
        
        Args:
            query: User's natural language query
            
        Returns:
            QueryIntent object with structured analysis
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Detect intent category
        intent_category = self._detect_intent_category(query_lower, query_type)
        
        # Extract technologies
        technologies = self._extract_technologies(query_lower)
        
        # Extract topics
        topics = self._extract_topics(query_lower)
        
        # Detect complexity level
        complexity_level = self._detect_complexity_level(query_lower)
        
        # Detect urgency
        urgency = self._detect_urgency(query_lower)
        
        # Calculate specificity
        specificity = self._calculate_specificity(query_lower, technologies, topics)
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Extract context clues
        context_clues = self._extract_context_clues(query_lower)
        
        return QueryIntent(
            query_type=query_type,
            intent_category=intent_category,
            technologies=technologies,
            topics=topics,
            complexity_level=complexity_level,
            urgency=urgency,
            specificity=specificity,
            entities=entities,
            keywords=keywords,
            context_clues=context_clues
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query based on patterns."""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type
        
        return QueryType.GENERAL
    
    def _detect_intent_category(self, query: str, query_type: QueryType) -> IntentCategory:
        """Detect the user's intent category."""
        # Map query types to intent categories with some heuristics
        intent_mapping = {
            QueryType.HOW_TO: IntentCategory.IMPLEMENTATION,
            QueryType.WHAT_IS: IntentCategory.LEARNING,
            QueryType.BEST_PRACTICES: IntentCategory.OPTIMIZATION,
            QueryType.EXAMPLES: IntentCategory.LEARNING,
            QueryType.COMPARISON: IntentCategory.EXPLORATION,
            QueryType.TROUBLESHOOTING: IntentCategory.PROBLEM_SOLVING,
            QueryType.INSTALLATION: IntentCategory.IMPLEMENTATION,
            QueryType.CONFIGURATION: IntentCategory.IMPLEMENTATION
        }
        
        base_intent = intent_mapping.get(query_type, IntentCategory.EXPLORATION)
        
        # Refine based on query content
        if any(word in query for word in ["broken", "error", "not working", "fix"]):
            return IntentCategory.PROBLEM_SOLVING
        elif any(word in query for word in ["learn", "understand", "explain"]):
            return IntentCategory.LEARNING
        elif any(word in query for word in ["implement", "create", "build"]):
            return IntentCategory.IMPLEMENTATION
        elif any(word in query for word in ["optimize", "improve", "performance"]):
            return IntentCategory.OPTIMIZATION
        
        return base_intent
    
    def _extract_technologies(self, query: str) -> List[str]:
        """Extract mentioned technologies from the query."""
        detected_techs = []
        
        for tech, patterns in self.technology_patterns.items():
            for pattern in patterns:
                if pattern.lower() in query:
                    detected_techs.append(tech)
                    break
        
        return list(set(detected_techs))  # Remove duplicates
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract topics from the query."""
        detected_topics = []
        
        for topic, patterns in self.topic_patterns.items():
            for pattern in patterns:
                if pattern.lower() in query:
                    detected_topics.append(topic)
                    break
        
        return list(set(detected_topics))
    
    def _detect_complexity_level(self, query: str) -> ComplexityLevel:
        """Detect the complexity level of the query."""
        scores = {level: 0 for level in ComplexityLevel}
        
        for level, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in query:
                    scores[level] += 1
        
        # Find the level with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return ComplexityLevel.INTERMEDIATE  # Default
        
        for level, score in scores.items():
            if score == max_score:
                return level
        
        return ComplexityLevel.INTERMEDIATE
    
    def _detect_urgency(self, query: str) -> str:
        """Detect the urgency level of the query."""
        for urgency, indicators in self.urgency_indicators.items():
            for indicator in indicators:
                if indicator in query:
                    return urgency
        
        return "medium"  # Default urgency
    
    def _calculate_specificity(self, query: str, technologies: List[str], topics: List[str]) -> float:
        """Calculate how specific the query is."""
        base_specificity = 0.3  # Base level
        
        # Add specificity for technologies mentioned
        base_specificity += len(technologies) * 0.15
        
        # Add specificity for topics mentioned
        base_specificity += len(topics) * 0.1
        
        # Add specificity for specific terms
        specific_terms = [
            "version", "configuration", "setup", "implementation",
            "example", "code", "function", "class", "method"
        ]
        
        for term in specific_terms:
            if term in query:
                base_specificity += 0.05
        
        # Questions with "how" are usually more specific
        if query.startswith("how"):
            base_specificity += 0.1
        
        return min(base_specificity, 1.0)
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query."""
        entities = {
            "frameworks": [],
            "languages": [],
            "tools": [],
            "concepts": [],
            "files": [],
            "commands": []
        }
        
        # Framework entities
        framework_terms = ["react", "vue", "angular", "nextjs", "django", "flask", "spring"]
        for term in framework_terms:
            if term in query:
                entities["frameworks"].append(term)
        
        # Language entities
        language_terms = ["javascript", "python", "java", "typescript", "go", "rust"]
        for term in language_terms:
            if term in query:
                entities["languages"].append(term)
        
        # Tool entities
        tool_terms = ["docker", "kubernetes", "git", "npm", "pip", "maven"]
        for term in tool_terms:
            if term in query:
                entities["tools"].append(term)
        
        # File pattern entities
        file_patterns = [r"\\.[a-zA-Z]{1,4}\\b", r"\\w+\\.\\w+"]
        for pattern in file_patterns:
            matches = re.findall(pattern, query)
            entities["files"].extend(matches)
        
        # Command entities (things that look like commands)
        command_pattern = r"\\b(?:npm|pip|git|docker|kubectl)\\s+\\w+"
        commands = re.findall(command_pattern, query)
        entities["commands"].extend(commands)
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "how", "what", "when", "where", "why", "which",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "can", "may",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r"\\b\\w+\\b", query.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Sort by relevance (longer words first, then alphabetically)
        keywords.sort(key=lambda x: (-len(x), x))
        
        return keywords[:10]  # Return top 10 keywords
    
    def _extract_context_clues(self, query: str) -> List[str]:
        """Extract context clues from the query."""
        context_clues = []
        
        # Project type clues
        project_types = ["web app", "mobile app", "api", "website", "application", "service"]
        for ptype in project_types:
            if ptype in query:
                context_clues.append(f"project_type:{ptype}")
        
        # Experience level clues
        experience_clues = ["beginner", "new", "experienced", "expert", "learning"]
        for exp in experience_clues:
            if exp in query:
                context_clues.append(f"experience:{exp}")
        
        # Environment clues
        environments = ["development", "production", "staging", "local", "cloud"]
        for env in environments:
            if env in query:
                context_clues.append(f"environment:{env}")
        
        # Timeline clues
        timeline_words = ["quick", "fast", "urgent", "slowly", "step by step"]
        for timeline in timeline_words:
            if timeline in query:
                context_clues.append(f"timeline:{timeline}")
        
        return context_clues
    
    def suggest_clarifying_questions(self, intent: QueryIntent) -> List[str]:
        """Suggest clarifying questions based on the analyzed intent.
        
        Args:
            intent: Analyzed query intent
            
        Returns:
            List of clarifying questions to ask the user
        """
        questions = []
        
        # If no technologies detected, ask for clarification
        if not intent.technologies:
            questions.append("Which technology or framework are you working with?")
        
        # If query is vague, ask for more specifics
        if intent.specificity < 0.4:
            questions.append("Could you provide more details about what you're trying to achieve?")
        
        # If it's a how-to question but no specific context
        if intent.query_type == QueryType.HOW_TO and not intent.topics:
            questions.append("What specific aspect would you like help with?")
        
        # If it's a troubleshooting question but no error details
        if intent.query_type == QueryType.TROUBLESHOOTING and "error" not in intent.keywords:
            questions.append("What error message or unexpected behavior are you seeing?")
        
        # If complexity level is unclear and they're asking for implementation
        if intent.intent_category == IntentCategory.IMPLEMENTATION and intent.complexity_level == ComplexityLevel.INTERMEDIATE:
            questions.append("What's your experience level with this technology?")
        
        return questions[:3]  # Return top 3 questions