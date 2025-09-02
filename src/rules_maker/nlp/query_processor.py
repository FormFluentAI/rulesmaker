"""
Natural Language Query Processor.

Processes natural language queries about documentation and rules,
providing intelligent responses and contextual recommendations.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any
import json
import os
from collections import defaultdict, Counter

from .intent_analyzer import IntentAnalyzer, QueryIntent, QueryType, IntentCategory
from ..intelligence.models import (
    QueryResponse, RecommendedSource, UserIntent, ProjectAnalysis,
    ComplexityLevel
)
from ..intelligence.recommendation_engine import SmartRecommendationEngine
from ..intelligence.semantic_analyzer import SemanticAnalyzer
from ..sources.updated_documentation_sources import get_comprehensive_updated_sources
from ..bedrock_integration import BedrockRulesMaker


class ProjectContext:
    """Project context for natural language queries."""
    
    def __init__(self, technologies: List[str] = None, project_type: str = None,
                 experience_level: str = "intermediate"):
        """Initialize project context.
        
        Args:
            technologies: List of technologies being used
            project_type: Type of project (web-app, api, etc.)
            experience_level: User's experience level
        """
        self.technologies = technologies or []
        self.project_type = project_type or "general"
        self.experience_level = experience_level


class NaturalLanguageQueryProcessor:
    """Process natural language queries about documentation and rules."""
    
    def __init__(self, bedrock_config: Optional[Dict] = None):
        """Initialize the query processor.
        
        Args:
            bedrock_config: Configuration for Bedrock integration
        """
        self.bedrock_config = bedrock_config or {}
        self.intent_analyzer = IntentAnalyzer()
        self.recommendation_engine = SmartRecommendationEngine(bedrock_config)
        self.semantic_analyzer = SemanticAnalyzer(bedrock_config)
        
        # Load query history and response templates
        self.query_history = self._load_query_history()
        self.response_templates = self._load_response_templates()
        self.knowledge_base = self._build_knowledge_base()
        
    def _load_query_history(self) -> Dict:
        """Load query history for learning and improvement."""
        history_path = "data/query_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return {"queries": [], "successful_responses": [], "user_feedback": []}
    
    def _save_query_history(self):
        """Save query history."""
        os.makedirs("data", exist_ok=True)
        with open("data/query_history.json", 'w') as f:
            json.dump(self.query_history, f, indent=2, default=str)
    
    def _load_response_templates(self) -> Dict:
        """Load response templates for different query types."""
        templates_path = "data/response_templates.json"
        if os.path.exists(templates_path):
            with open(templates_path, 'r') as f:
                return json.load(f)
        return self._get_default_response_templates()
    
    def _get_default_response_templates(self) -> Dict:
        """Get default response templates."""
        return {
            "how_to": {
                "intro": "Here's how to {topic} in {technology}:",
                "steps": "\\n\\n**Steps:**\\n{steps}",
                "example": "\\n\\n**Example:**\\n```{language}\\n{code}\\n```",
                "notes": "\\n\\n**Additional Notes:**\\n{notes}"
            },
            "what_is": {
                "definition": "{concept} is {definition}",
                "context": "\\n\\n**In {technology} context:**\\n{context_info}",
                "use_cases": "\\n\\n**Common use cases:**\\n{use_cases}",
                "related": "\\n\\n**Related concepts:** {related_concepts}"
            },
            "best_practices": {
                "intro": "Here are the best practices for {topic} in {technology}:",
                "practices": "\\n\\n{practices_list}",
                "antipatterns": "\\n\\n**Avoid these antipatterns:**\\n{antipatterns}",
                "resources": "\\n\\n**Additional resources:**\\n{resources}"
            },
            "examples": {
                "intro": "Here are examples of {topic} in {technology}:",
                "basic": "\\n\\n**Basic Example:**\\n```{language}\\n{basic_code}\\n```",
                "advanced": "\\n\\n**Advanced Example:**\\n```{language}\\n{advanced_code}\\n```",
                "explanation": "\\n\\n**Explanation:**\\n{explanation}"
            },
            "troubleshooting": {
                "problem": "**Problem:** {problem_description}",
                "likely_causes": "\\n\\n**Likely causes:**\\n{causes}",
                "solutions": "\\n\\n**Solutions:**\\n{solutions}",
                "prevention": "\\n\\n**Prevention:**\\n{prevention_tips}"
            },
            "comparison": {
                "intro": "Here's a comparison between {option1} and {option2}:",
                "table": "\\n\\n{comparison_table}",
                "recommendation": "\\n\\n**Recommendation:** {recommendation}",
                "context": "\\n\\n**When to use:**\\n{usage_context}"
            }
        }
    
    def _build_knowledge_base(self) -> Dict:
        """Build a knowledge base from available documentation sources."""
        knowledge_base = {
            "technologies": defaultdict(dict),
            "topics": defaultdict(list),
            "patterns": defaultdict(list),
            "common_questions": defaultdict(list)
        }
        
        # Get available documentation sources
        sources = get_comprehensive_updated_sources()
        
        # Organize sources by technology and topic
        for source in sources:
            tech = source.framework or source.language or "general"
            knowledge_base["technologies"][tech]["sources"] = knowledge_base["technologies"][tech].get("sources", [])
            # DocumentationSource uses 'name' field, not 'title'
            knowledge_base["technologies"][tech]["sources"].append({
                "url": source.url,
                "title": getattr(source, 'title', None) or getattr(source, 'name', 'Documentation'),
                "priority": source.priority
            })
        
        # Add common patterns and topics (this would be expanded with real data)
        self._populate_knowledge_base_patterns(knowledge_base)
        
        return knowledge_base
    
    def _populate_knowledge_base_patterns(self, kb: Dict):
        """Populate knowledge base with common patterns and topics."""
        # Common React patterns
        kb["patterns"]["react"].extend([
            "functional components with hooks",
            "state management with useState",
            "side effects with useEffect",
            "component composition",
            "prop drilling solutions"
        ])
        
        # Common Next.js patterns
        kb["patterns"]["nextjs"].extend([
            "app router architecture",
            "server components vs client components",
            "data fetching patterns",
            "dynamic routing",
            "middleware usage"
        ])
        
        # Common Python patterns
        kb["patterns"]["python"].extend([
            "async/await for I/O operations",
            "context managers",
            "type hints usage",
            "error handling patterns",
            "virtual environment setup"
        ])
        
        # Common topics
        kb["topics"]["authentication"].extend([
            "JWT tokens", "OAuth2 flow", "session management", "password hashing"
        ])
        
        kb["topics"]["deployment"].extend([
            "Docker containerization", "CI/CD pipelines", "environment variables", "production builds"
        ])
        
        # Common questions
        kb["common_questions"]["react"].extend([
            "How to manage state in React?",
            "When to use useEffect?",
            "How to optimize React performance?",
            "What's the difference between props and state?"
        ])
    
    async def process_query(self, query: str, context: Optional[ProjectContext] = None) -> QueryResponse:
        """Answer natural language questions about documentation.
        
        Args:
            query: User's natural language query
            context: Optional project context
            
        Returns:
            QueryResponse with answer and recommendations
        """
        # Analyze query intent
        intent = self.intent_analyzer.analyze_query(query)
        
        # Record query for learning
        self._record_query(query, intent)
        
        # Find relevant documentation
        relevant_docs = await self.find_relevant_documentation(intent, context)
        
        # Generate contextual answer
        answer = await self.generate_contextual_answer(intent, relevant_docs, context)
        
        # Suggest related rules
        suggested_rules = await self.suggest_rules(intent, context)
        
        # Find related topics
        related_topics = await self.find_related_topics(intent)
        
        # Generate follow-up suggestions
        follow_up_suggestions = self._generate_follow_up_suggestions(intent, context)
        
        # Calculate confidence based on intent specificity and available docs
        confidence = self._calculate_response_confidence(intent, relevant_docs)
        
        response = QueryResponse(
            answer=answer,
            relevant_sources=relevant_docs,
            suggested_rules=suggested_rules,
            related_topics=related_topics,
            confidence=confidence,
            follow_up_suggestions=follow_up_suggestions
        )
        
        return response
    
    async def find_relevant_documentation(
        self, 
        intent: QueryIntent, 
        context: Optional[ProjectContext] = None
    ) -> List[RecommendedSource]:
        """Find relevant documentation sources based on query intent.
        
        Args:
            intent: Analyzed query intent
            context: Optional project context
            
        Returns:
            List of relevant documentation sources
        """
        relevant_sources = []
        
        # Get all available sources
        all_sources = get_comprehensive_updated_sources()
        
        # Score sources based on intent
        scored_sources = []
        
        for source in all_sources:
            score = self._score_source_relevance(source, intent, context)
            if score > 0.1:  # Minimum relevance threshold
                recommended = RecommendedSource(
                    source=source.url,
                    reason=self._generate_source_reason(source, intent, score),
                    priority=min(int(score * 5) + 1, 5),
                    estimated_value=self._estimate_source_value(source, intent, score),
                    category=source.language,
                    framework=source.framework,
                    confidence=score
                )
                scored_sources.append((recommended, score))
        
        # Sort by relevance score
        scored_sources.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 most relevant sources
        return [source for source, _ in scored_sources[:5]]
    
    def _score_source_relevance(
        self, 
        source, 
        intent: QueryIntent, 
        context: Optional[ProjectContext]
    ) -> float:
        """Score how relevant a source is to the query intent."""
        score = 0.0
        
        # Technology matching (40% weight)
        tech_score = 0.0
        source_tech = (source.framework or source.language or "").lower()
        
        for tech in intent.technologies:
            if tech.lower() == source_tech:
                tech_score = 1.0
                break
            elif tech.lower() in source_tech or source_tech in tech.lower():
                tech_score = max(tech_score, 0.7)
        
        # Context technology matching
        if context and context.technologies:
            for tech in context.technologies:
                if tech.lower() == source_tech:
                    tech_score = max(tech_score, 0.8)
        
        score += tech_score * 0.4
        
        # Topic matching (30% weight)
        topic_score = 0.0
        source_text = f"{getattr(source, 'title', None) or getattr(source, 'name', '')} {source.url}".lower()
        
        for topic in intent.topics:
            if topic in source_text:
                topic_score += 0.3
        
        score += min(topic_score, 1.0) * 0.3
        
        # Keyword matching (20% weight)
        keyword_score = 0.0
        for keyword in intent.keywords[:5]:  # Top 5 keywords
            if keyword in source_text:
                keyword_score += 0.1
        
        score += min(keyword_score, 1.0) * 0.2
        
        # Source priority (10% weight)
        priority_score = source.priority / 5.0
        score += priority_score * 0.1
        
        return min(score, 1.0)
    
    def _generate_source_reason(self, source, intent: QueryIntent, score: float) -> str:
        """Generate a reason why this source is recommended."""
        reasons = []
        
        source_tech = source.framework or source.language or "general"
        
        # Technology matching reason
        if any(tech.lower() == source_tech.lower() for tech in intent.technologies):
            reasons.append(f"Directly covers {source_tech}")
        elif intent.technologies:
            reasons.append(f"Related to {source_tech} technology")
        
        # Topic matching reason
        source_text = f"{getattr(source, 'title', None) or getattr(source, 'name', '')} {source.url}".lower()
        matching_topics = [topic for topic in intent.topics if topic in source_text]
        if matching_topics:
            reasons.append(f"Addresses {', '.join(matching_topics[:2])}")
        
        # Quality reason
        if source.priority >= 4:
            reasons.append("High-quality documentation source")
        
        # Default reason
        if not reasons:
            reasons.append("Relevant documentation resource")
        
        return reasons[0] if reasons else "Potentially useful documentation"
    
    def _estimate_source_value(self, source, intent: QueryIntent, score: float) -> str:
        """Estimate the value of a source for the user."""
        if score >= 0.8:
            return "Very High - Directly addresses your question"
        elif score >= 0.6:
            return "High - Contains relevant information"
        elif score >= 0.4:
            return "Medium - May provide useful context"
        else:
            return "Low - Limited relevance but may help"
    
    async def generate_contextual_answer(
        self, 
        intent: QueryIntent, 
        relevant_docs: List[RecommendedSource], 
        context: Optional[ProjectContext] = None
    ) -> str:
        """Generate a contextual answer based on intent and available documentation.
        
        Args:
            intent: Analyzed query intent
            relevant_docs: Relevant documentation sources
            context: Optional project context
            
        Returns:
            Generated answer string
        """
        # Get template for query type
        template_config = self.response_templates.get(
            intent.query_type.value, 
            self.response_templates["how_to"]
        )
        
        # Generate answer based on intent type
        if intent.query_type == QueryType.HOW_TO:
            answer = await self._generate_how_to_answer(intent, relevant_docs, context, template_config)
        elif intent.query_type == QueryType.WHAT_IS:
            answer = await self._generate_what_is_answer(intent, relevant_docs, context, template_config)
        elif intent.query_type == QueryType.BEST_PRACTICES:
            answer = await self._generate_best_practices_answer(intent, relevant_docs, context, template_config)
        elif intent.query_type == QueryType.EXAMPLES:
            answer = await self._generate_examples_answer(intent, relevant_docs, context, template_config)
        elif intent.query_type == QueryType.TROUBLESHOOTING:
            answer = await self._generate_troubleshooting_answer(intent, relevant_docs, context, template_config)
        elif intent.query_type == QueryType.COMPARISON:
            answer = await self._generate_comparison_answer(intent, relevant_docs, context, template_config)
        else:
            answer = await self._generate_general_answer(intent, relevant_docs, context)
        
        # Enhance with LLM if available and query is complex
        if self.bedrock_config and intent.specificity > 0.6:
            enhanced_answer = await self._enhance_answer_with_llm(answer, intent, relevant_docs, context)
            if enhanced_answer:
                answer = enhanced_answer
        
        return answer
    
    async def _generate_how_to_answer(
        self, 
        intent: QueryIntent, 
        docs: List[RecommendedSource], 
        context: Optional[ProjectContext],
        template: Dict
    ) -> str:
        """Generate a how-to answer."""
        primary_tech = intent.technologies[0] if intent.technologies else "your technology"
        main_topic = intent.topics[0] if intent.topics else "the task"
        
        answer = template["intro"].format(topic=main_topic, technology=primary_tech)
        
        # Add knowledge-based steps
        steps = self._get_knowledge_based_steps(intent, context)
        if steps:
            answer += template["steps"].format(steps=steps)
        
        # Add example if available
        example_code = self._get_example_code(intent, context)
        if example_code:
            language = self._infer_language(intent, context)
            answer += template["example"].format(
                language=language or "text", 
                code=example_code
            )
        
        # Add additional notes
        notes = self._get_additional_notes(intent, context)
        if notes:
            answer += template["notes"].format(notes=notes)
        
        return answer
    
    async def _generate_what_is_answer(
        self, 
        intent: QueryIntent, 
        docs: List[RecommendedSource], 
        context: Optional[ProjectContext],
        template: Dict
    ) -> str:
        """Generate a what-is answer."""
        concept = intent.keywords[0] if intent.keywords else "concept"
        
        # Try to get definition from knowledge base
        definition = self._get_concept_definition(concept, intent.technologies)
        
        answer = template["definition"].format(
            concept=concept.title(), 
            definition=definition
        )
        
        # Add technology-specific context
        if intent.technologies:
            tech = intent.technologies[0]
            context_info = self._get_technology_context(concept, tech)
            if context_info:
                answer += template["context"].format(
                    technology=tech.title(),
                    context_info=context_info
                )
        
        return answer
    
    async def _generate_best_practices_answer(
        self, 
        intent: QueryIntent, 
        docs: List[RecommendedSource], 
        context: Optional[ProjectContext],
        template: Dict
    ) -> str:
        """Generate a best practices answer."""
        topic = intent.topics[0] if intent.topics else "development"
        tech = intent.technologies[0] if intent.technologies else "your technology"
        
        answer = template["intro"].format(topic=topic, technology=tech)
        
        # Get best practices from knowledge base
        practices = self._get_best_practices(topic, tech, intent.complexity_level)
        if practices:
            practices_text = "\\n".join([f"• {practice}" for practice in practices])
            answer += template["practices"].format(practices_list=practices_text)
        
        return answer
    
    async def _generate_examples_answer(
        self, 
        intent: QueryIntent, 
        docs: List[RecommendedSource], 
        context: Optional[ProjectContext],
        template: Dict
    ) -> str:
        """Generate an examples answer."""
        topic = intent.topics[0] if intent.topics else "implementation"
        tech = intent.technologies[0] if intent.technologies else "code"
        
        answer = template["intro"].format(topic=topic, technology=tech)
        
        # Add basic example
        basic_code = self._get_basic_example(intent, context)
        if basic_code:
            language = self._infer_language(intent, context)
            answer += template["basic"].format(
                language=language or "text",
                basic_code=basic_code
            )
        
        return answer
    
    async def _generate_troubleshooting_answer(
        self, 
        intent: QueryIntent, 
        docs: List[RecommendedSource], 
        context: Optional[ProjectContext],
        template: Dict
    ) -> str:
        """Generate a troubleshooting answer."""
        # Extract problem description from query
        problem_keywords = [kw for kw in intent.keywords if kw in ["error", "not", "working", "broken"]]
        problem = " ".join(problem_keywords) if problem_keywords else "the issue you're experiencing"
        
        answer = template["problem"].format(problem_description=problem)
        
        # Get common causes and solutions
        causes = self._get_common_causes(intent)
        if causes:
            causes_text = "\\n".join([f"• {cause}" for cause in causes])
            answer += template["likely_causes"].format(causes=causes_text)
        
        solutions = self._get_common_solutions(intent)
        if solutions:
            solutions_text = "\\n".join([f"• {solution}" for solution in solutions])
            answer += template["solutions"].format(solutions=solutions_text)
        
        return answer
    
    async def _generate_comparison_answer(
        self, 
        intent: QueryIntent, 
        docs: List[RecommendedSource], 
        context: Optional[ProjectContext],
        template: Dict
    ) -> str:
        """Generate a comparison answer."""
        # Try to extract comparison items from keywords
        comparison_items = intent.keywords[:2] if len(intent.keywords) >= 2 else ["Option A", "Option B"]
        
        answer = template["intro"].format(
            option1=comparison_items[0].title(),
            option2=comparison_items[1].title()
        )
        
        # Add comparison table (simplified)
        table = f"""
| Aspect | {comparison_items[0].title()} | {comparison_items[1].title()} |
|--------|------------|-------------|
| Use Case | General purpose | Specific scenarios |
| Complexity | Medium | Varies |
| Community | Active | Growing |
        """
        answer += template["table"].format(comparison_table=table)
        
        return answer
    
    async def _generate_general_answer(
        self, 
        intent: QueryIntent, 
        docs: List[RecommendedSource], 
        context: Optional[ProjectContext]
    ) -> str:
        """Generate a general answer for queries that don't fit specific patterns."""
        if intent.technologies:
            tech = intent.technologies[0]
            answer = f"For {tech} development, here's what I found:\\n\\n"
        else:
            answer = "Based on your query, here's some relevant information:\\n\\n"
        
        # Add information based on detected topics
        if intent.topics:
            for topic in intent.topics[:2]:  # Top 2 topics
                topic_info = self._get_topic_information(topic, intent.technologies)
                if topic_info:
                    answer += f"**{topic.replace('_', ' ').title()}**: {topic_info}\\n\\n"
        
        # Add general guidance
        if not intent.topics and docs:
            answer += "I recommend checking these documentation sources for more detailed information."
        
        return answer
    
    def _get_knowledge_based_steps(self, intent: QueryIntent, context: Optional[ProjectContext]) -> str:
        """Get steps from knowledge base based on intent."""
        # This would be expanded with actual knowledge base lookups
        if "authentication" in intent.topics:
            return """1. Choose an authentication method (JWT, OAuth, etc.)
2. Set up authentication middleware
3. Create login/logout endpoints
4. Implement token validation
5. Add authentication to protected routes"""
        
        elif "deployment" in intent.topics:
            return """1. Build your application for production
2. Choose a deployment platform
3. Configure environment variables
4. Set up CI/CD pipeline
5. Deploy and test"""
        
        return ""
    
    def _get_example_code(self, intent: QueryIntent, context: Optional[ProjectContext]) -> str:
        """Get example code based on intent."""
        # This would be expanded with actual code examples
        if "react" in intent.technologies and "state_management" in intent.topics:
            return """const [count, setCount] = useState(0);

const increment = () => {
  setCount(prev => prev + 1);
};

return (
  <div>
    <p>Count: {count}</p>
    <button onClick={increment}>Increment</button>
  </div>
);"""
        
        return ""
    
    def _get_additional_notes(self, intent: QueryIntent, context: Optional[ProjectContext]) -> str:
        """Get additional notes based on intent."""
        notes = []
        
        if intent.complexity_level == ComplexityLevel.BEGINNER:
            notes.append("💡 If you're new to this, start with the basics and build up gradually.")
        
        if intent.urgency == "high":
            notes.append("⚡ For quick implementation, focus on the essential steps first.")
        
        if intent.technologies:
            tech = intent.technologies[0]
            notes.append(f"📚 Check the official {tech} documentation for more detailed information.")
        
        return "\\n".join(notes) if notes else ""
    
    def _get_concept_definition(self, concept: str, technologies: List[str]) -> str:
        """Get definition of a concept."""
        # This would be expanded with actual definitions
        definitions = {
            "hook": "a special function in React that lets you use state and lifecycle features in functional components",
            "component": "a reusable piece of UI that can accept inputs (props) and return elements describing what should appear on the screen",
            "middleware": "software that acts as a bridge between different applications or services, often handling cross-cutting concerns like authentication or logging",
            "api": "Application Programming Interface - a set of protocols and tools for building software applications",
            "state": "data that changes over time and affects how a component renders"
        }
        
        return definitions.get(concept.lower(), f"a concept in software development related to {concept}")
    
    def _get_technology_context(self, concept: str, technology: str) -> str:
        """Get technology-specific context for a concept."""
        # This would be expanded with actual context information
        if concept == "hook" and technology == "react":
            return "Hooks were introduced in React 16.8 to allow functional components to use state and lifecycle methods previously only available in class components."
        
        return f"In {technology}, {concept} is commonly used for various development tasks."
    
    def _get_best_practices(self, topic: str, technology: str, complexity: ComplexityLevel) -> List[str]:
        """Get best practices for a topic and technology."""
        practices = {
            "authentication": [
                "Always use HTTPS for authentication endpoints",
                "Implement proper session management",
                "Use strong password policies",
                "Add rate limiting to prevent brute force attacks",
                "Store passwords using secure hashing algorithms"
            ],
            "state_management": [
                "Keep state as local as possible",
                "Use immutable update patterns",
                "Separate concerns between UI and business logic",
                "Consider using state management libraries for complex apps",
                "Avoid deeply nested state structures"
            ]
        }
        
        return practices.get(topic, ["Follow established patterns", "Write clear, maintainable code", "Test your implementation"])
    
    def _get_basic_example(self, intent: QueryIntent, context: Optional[ProjectContext]) -> str:
        """Get a basic code example."""
        # This would be expanded with actual examples
        if "python" in intent.technologies and "api" in intent.topics:
            return """from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}"""
        
        return ""
    
    def _get_common_causes(self, intent: QueryIntent) -> List[str]:
        """Get common causes for troubleshooting."""
        return [
            "Configuration issues",
            "Missing dependencies",
            "Version compatibility problems",
            "Environment variable issues",
            "Network connectivity problems"
        ]
    
    def _get_common_solutions(self, intent: QueryIntent) -> List[str]:
        """Get common solutions for troubleshooting."""
        return [
            "Check your configuration files",
            "Verify all dependencies are installed",
            "Update to compatible versions",
            "Review error logs for specific messages",
            "Test in a clean environment"
        ]
    
    def _get_topic_information(self, topic: str, technologies: List[str]) -> str:
        """Get information about a specific topic."""
        topic_info = {
            "authentication": "the process of verifying user identity, typically involving usernames, passwords, tokens, or other credentials",
            "routing": "the mechanism that determines how an application responds to different URLs or endpoints",
            "state_management": "the practice of managing and organizing data that changes over time in an application",
            "deployment": "the process of making your application available to users, typically involving building, testing, and hosting",
            "testing": "the practice of writing code to verify that your application works as expected"
        }
        
        return topic_info.get(topic, f"an important concept in software development")
    
    def _infer_language(self, intent: QueryIntent, context: Optional[ProjectContext]) -> Optional[str]:
        """Infer programming language from intent."""
        tech_to_lang = {
            "javascript": "javascript",
            "typescript": "typescript", 
            "react": "jsx",
            "nextjs": "jsx",
            "vue": "vue",
            "python": "python",
            "java": "java",
            "csharp": "csharp",
            "go": "go",
            "rust": "rust"
        }
        
        for tech in intent.technologies:
            if tech in tech_to_lang:
                return tech_to_lang[tech]
        
        return None
    
    async def _enhance_answer_with_llm(
        self, 
        answer: str, 
        intent: QueryIntent, 
        docs: List[RecommendedSource],
        context: Optional[ProjectContext]
    ) -> Optional[str]:
        """Enhance answer using LLM for complex queries."""
        try:
            bedrock_maker = BedrockRulesMaker(**self.bedrock_config)
            
            prompt = f"""
            Enhance this answer to a technical question with more specific and helpful information:
            
            Original Question Context:
            - Technologies: {intent.technologies}
            - Topics: {intent.topics}
            - Query Type: {intent.query_type.value}
            - Experience Level: {intent.complexity_level.value}
            
            Current Answer:
            {answer}
            
            Please enhance this answer by:
            1. Adding more specific technical details
            2. Including relevant examples if missing
            3. Providing practical implementation tips
            4. Maintaining the same structure and tone
            
            Return only the enhanced answer.
            """
            
            enhanced = await bedrock_maker._call_bedrock_async(prompt)
            
            # Basic validation
            if enhanced and len(enhanced) > len(answer) * 0.8:
                return enhanced
                
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
        
        return None
    
    async def suggest_rules(self, intent: QueryIntent, context: Optional[ProjectContext] = None) -> List[str]:
        """Suggest relevant rules based on query intent."""
        rule_suggestions = []
        
        # Suggest rules based on technologies
        for tech in intent.technologies:
            if tech == "react":
                rule_suggestions.extend([
                    "Use functional components with hooks",
                    "Implement proper error boundaries",
                    "Optimize with React.memo when needed"
                ])
            elif tech == "python":
                rule_suggestions.extend([
                    "Use type hints for better code clarity",
                    "Follow PEP 8 style guidelines",
                    "Implement proper exception handling"
                ])
        
        # Suggest rules based on topics
        if "authentication" in intent.topics:
            rule_suggestions.extend([
                "Always validate user input",
                "Use secure password hashing",
                "Implement proper session management"
            ])
        
        if "api" in intent.topics:
            rule_suggestions.extend([
                "Follow RESTful API design principles",
                "Implement proper error responses",
                "Add request validation and rate limiting"
            ])
        
        return rule_suggestions[:5]  # Return top 5 suggestions
    
    async def find_related_topics(self, intent: QueryIntent) -> List[str]:
        """Find topics related to the query intent."""
        related = set()
        
        # Add related topics based on current topics
        topic_relations = {
            "authentication": ["authorization", "security", "sessions", "jwt", "oauth"],
            "routing": ["navigation", "urls", "middleware", "guards"],
            "state_management": ["data flow", "props", "context", "redux", "stores"],
            "api": ["http", "rest", "graphql", "endpoints", "requests"],
            "testing": ["unit tests", "integration tests", "e2e tests", "mocking"],
            "deployment": ["ci/cd", "docker", "cloud", "production", "build"]
        }
        
        for topic in intent.topics:
            if topic in topic_relations:
                related.update(topic_relations[topic])
        
        # Add related topics based on technologies
        tech_topics = {
            "react": ["components", "hooks", "jsx", "virtual dom", "lifecycle"],
            "nextjs": ["ssr", "ssg", "app router", "page router", "middleware"],
            "python": ["modules", "packages", "virtual environments", "pip", "decorators"],
            "javascript": ["es6", "promises", "async/await", "dom", "events"]
        }
        
        for tech in intent.technologies:
            if tech in tech_topics:
                related.update(tech_topics[tech])
        
        # Remove current topics from related topics
        related = related - set(intent.topics)
        
        return list(related)[:8]  # Return top 8 related topics
    
    def _generate_follow_up_suggestions(self, intent: QueryIntent, context: Optional[ProjectContext]) -> List[str]:
        """Generate follow-up suggestions based on the query."""
        suggestions = []
        
        # Suggest next steps based on query type
        if intent.query_type == QueryType.HOW_TO:
            suggestions.extend([
                "Would you like to see a complete example?",
                "Need help with testing this implementation?",
                "Want to know about common pitfalls to avoid?"
            ])
        
        elif intent.query_type == QueryType.WHAT_IS:
            suggestions.extend([
                "Would you like to see how to implement this?",
                "Need examples of this concept in practice?",
                "Want to compare this with similar concepts?"
            ])
        
        elif intent.query_type == QueryType.TROUBLESHOOTING:
            suggestions.extend([
                "Would you like help with debugging techniques?",
                "Need information about error prevention?",
                "Want to set up better logging and monitoring?"
            ])
        
        # Add technology-specific suggestions
        if intent.technologies:
            tech = intent.technologies[0]
            suggestions.append(f"Want to learn more advanced {tech} patterns?")
            suggestions.append(f"Need help with {tech} best practices?")
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def _calculate_response_confidence(self, intent: QueryIntent, docs: List[RecommendedSource]) -> float:
        """Calculate confidence in the response."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on intent specificity
        confidence += intent.specificity * 0.3
        
        # Increase confidence based on available documentation
        if docs:
            avg_doc_confidence = sum(doc.confidence for doc in docs) / len(docs)
            confidence += avg_doc_confidence * 0.2
        
        # Adjust based on query type (some types are easier to answer)
        easy_types = [QueryType.WHAT_IS, QueryType.INSTALLATION, QueryType.EXAMPLES]
        if intent.query_type in easy_types:
            confidence += 0.1
        
        # Adjust based on technology coverage
        if intent.technologies and any(
            tech in self.knowledge_base["technologies"] for tech in intent.technologies
        ):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _record_query(self, query: str, intent: QueryIntent):
        """Record query for learning and improvement."""
        query_record = {
            "timestamp": str(os.times()),
            "query": query,
            "intent": {
                "query_type": intent.query_type.value,
                "intent_category": intent.intent_category.value,
                "technologies": intent.technologies,
                "topics": intent.topics,
                "complexity_level": intent.complexity_level.value,
                "specificity": intent.specificity
            }
        }
        
        self.query_history["queries"].append(query_record)
        
        # Keep only last 1000 queries to avoid bloat
        if len(self.query_history["queries"]) > 1000:
            self.query_history["queries"] = self.query_history["queries"][-1000:]
        
        self._save_query_history()
    
    def collect_response_feedback(self, query: str, response: QueryResponse, 
                                 rating: float, feedback: str = ""):
        """Collect feedback on response quality.
        
        Args:
            query: Original query
            response: Generated response
            rating: User rating (1-5)
            feedback: Optional user feedback text
        """
        feedback_record = {
            "timestamp": str(os.times()),
            "query": query,
            "rating": rating,
            "feedback": feedback,
            "response_confidence": response.confidence,
            "sources_provided": len(response.relevant_sources)
        }
        
        self.query_history["user_feedback"].append(feedback_record)
        self._save_query_history()
    
    def get_query_insights(self) -> Dict:
        """Get insights about query processing performance."""
        insights = {
            "total_queries": len(self.query_history["queries"]),
            "popular_technologies": Counter(),
            "common_query_types": Counter(),
            "avg_rating": 0.0,
            "improvement_areas": []
        }
        
        # Analyze query patterns
        for query_record in self.query_history["queries"]:
            intent = query_record["intent"]
            insights["popular_technologies"].update(intent["technologies"])
            insights["common_query_types"][intent["query_type"]] += 1
        
        # Analyze feedback
        feedback_records = self.query_history["user_feedback"]
        if feedback_records:
            ratings = [fb["rating"] for fb in feedback_records]
            insights["avg_rating"] = sum(ratings) / len(ratings)
            
            # Identify improvement areas
            if insights["avg_rating"] < 3.5:
                insights["improvement_areas"].append("Response quality needs improvement")
            
            low_confidence = [fb for fb in feedback_records if fb.get("response_confidence", 1.0) < 0.6]
            if len(low_confidence) / len(feedback_records) > 0.3:
                insights["improvement_areas"].append("Many responses have low confidence - need better knowledge base")
        
        return insights
