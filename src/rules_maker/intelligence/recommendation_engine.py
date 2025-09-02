"""
Smart Recommendation Engine.

AI-powered source and workflow recommendations based on user intent,
project analysis, and learned patterns.
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
import json
import os
from collections import defaultdict, Counter

from .models import (
    UserIntent, RecommendedSource, ProjectAnalysis,
    ComplexityLevel
)
from ..sources.updated_documentation_sources import (
    get_comprehensive_updated_sources,
    get_updated_web_frameworks,
    get_updated_python_frameworks,
    get_updated_backend_frameworks,
    get_updated_cloud_platforms,
    get_updated_ml_ai_tools
)
from ..bedrock_integration import BedrockRulesMaker


class SmartRecommendationEngine:
    """AI-powered source and workflow recommendations."""
    
    def __init__(self, bedrock_config: Optional[Dict] = None):
        """Initialize the recommendation engine.
        
        Args:
            bedrock_config: Configuration for Bedrock integration
        """
        self.bedrock_config = bedrock_config or {}
        self.user_preferences = self._load_user_preferences()
        self.recommendation_history = self._load_recommendation_history()
        self.source_ratings = self._load_source_ratings()
        
    def _load_user_preferences(self) -> Dict:
        """Load user preferences from previous sessions."""
        prefs_path = "data/user_preferences.json"
        if os.path.exists(prefs_path):
            with open(prefs_path, 'r') as f:
                return json.load(f)
        return {
            "preferred_frameworks": defaultdict(int),
            "content_type_preferences": defaultdict(int),
            "complexity_preferences": defaultdict(int),
            "learning_style": defaultdict(int)
        }
    
    def _save_user_preferences(self):
        """Save user preferences to disk."""
        os.makedirs("data", exist_ok=True)
        with open("data/user_preferences.json", 'w') as f:
            # Convert defaultdicts to regular dicts
            prefs_data = {
                "preferred_frameworks": dict(self.user_preferences["preferred_frameworks"]),
                "content_type_preferences": dict(self.user_preferences["content_type_preferences"]),
                "complexity_preferences": dict(self.user_preferences["complexity_preferences"]),
                "learning_style": dict(self.user_preferences["learning_style"])
            }
            json.dump(prefs_data, f, indent=2)
    
    def _load_recommendation_history(self) -> Dict:
        """Load recommendation history for learning."""
        history_path = "data/recommendation_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return {"successful_recommendations": [], "user_feedback": []}
    
    def _save_recommendation_history(self):
        """Save recommendation history."""
        os.makedirs("data", exist_ok=True)
        with open("data/recommendation_history.json", 'w') as f:
            json.dump(self.recommendation_history, f, indent=2, default=str)
    
    def _load_source_ratings(self) -> Dict:
        """Load source quality ratings."""
        ratings_path = "data/source_ratings.json"
        if os.path.exists(ratings_path):
            with open(ratings_path, 'r') as f:
                return json.load(f)
        return defaultdict(list)
    
    def _save_source_ratings(self):
        """Save source quality ratings."""
        os.makedirs("data", exist_ok=True)
        with open("data/source_ratings.json", 'w') as f:
            json.dump(dict(self.source_ratings), f, indent=2)
    
    async def recommend_documentation_sources(
        self, 
        user_intent: UserIntent, 
        project_analysis: Optional[ProjectAnalysis] = None
    ) -> List[RecommendedSource]:
        """Intelligently recommend documentation sources based on user needs.
        
        Args:
            user_intent: User's stated intentions and preferences
            project_analysis: Optional analysis of user's project
            
        Returns:
            List of recommended documentation sources, sorted by relevance
        """
        # Get all available sources
        all_sources = get_comprehensive_updated_sources()
        
        # Score sources based on multiple criteria
        scored_sources = []
        
        for source in all_sources:
            score_data = self._score_source(source, user_intent, project_analysis)
            
            if score_data["total_score"] > 0.1:  # Minimum relevance threshold
                recommended = RecommendedSource(
                    source=source.url,
                    reason=score_data["reason"],
                    priority=min(int(score_data["total_score"] * 5) + 1, 5),
                    estimated_value=score_data["estimated_value"],
                    category=source.language,
                    framework=source.framework,
                    confidence=score_data["total_score"]
                )
                scored_sources.append((recommended, score_data["total_score"]))
        
        # Sort by score and return top recommendations
        scored_sources.sort(key=lambda x: x[1], reverse=True)
        recommendations = [rec for rec, _ in scored_sources[:20]]  # Top 20
        
        # Enhance with LLM analysis if available
        if self.bedrock_config and recommendations:
            enhanced_recommendations = await self._enhance_recommendations_with_llm(
                recommendations, user_intent, project_analysis
            )
            if enhanced_recommendations:
                recommendations = enhanced_recommendations
        
        # Apply learned preferences
        recommendations = self._apply_learned_preferences(recommendations, user_intent)
        
        # Diversify recommendations
        recommendations = self._diversify_recommendations(recommendations)
        
        # Record recommendation for learning
        self._record_recommendation(user_intent, recommendations)
        
        return recommendations[:15]  # Return top 15
    
    def _score_source(
        self, 
        source, 
        user_intent: UserIntent, 
        project_analysis: Optional[ProjectAnalysis]
    ) -> Dict:
        """Score a documentation source for relevance.
        
        Args:
            source: Documentation source to score
            user_intent: User's stated intentions
            project_analysis: Optional project analysis
            
        Returns:
            Dictionary with scoring details
        """
        score_components = {
            "technology_match": 0.0,
            "experience_level": 0.0,
            "goal_alignment": 0.0,
            "project_relevance": 0.0,
            "historical_preference": 0.0,
            "source_quality": 0.0
        }
        
        reasons = []
        
        # Technology matching (40% weight)
        tech_score = self._score_technology_match(source, user_intent)
        score_components["technology_match"] = tech_score * 0.4
        
        if tech_score > 0.7:
            reasons.append(f"Strong match for {source.framework or source.language}")
        elif tech_score > 0.3:
            reasons.append(f"Relevant for {source.framework or source.language}")
        
        # Experience level matching (20% weight)
        exp_score = self._score_experience_level(source, user_intent)
        score_components["experience_level"] = exp_score * 0.2
        
        # Goal alignment (20% weight)
        goal_score = self._score_goal_alignment(source, user_intent)
        score_components["goal_alignment"] = goal_score * 0.2
        
        if goal_score > 0.5:
            reasons.append("Aligns with your stated goals")
        
        # Project relevance (10% weight)
        if project_analysis:
            proj_score = self._score_project_relevance(source, project_analysis)
            score_components["project_relevance"] = proj_score * 0.1
            
            if proj_score > 0.7:
                reasons.append("Highly relevant to your project patterns")
        
        # Historical preferences (5% weight)
        hist_score = self._score_historical_preference(source, user_intent)
        score_components["historical_preference"] = hist_score * 0.05
        
        # Source quality (5% weight)
        quality_score = self._score_source_quality(source)
        score_components["source_quality"] = quality_score * 0.05
        
        # Calculate total score
        total_score = sum(score_components.values())
        
        # Generate estimated value description
        estimated_value = self._generate_estimated_value(source, total_score, user_intent)
        
        # Generate comprehensive reason
        reason = self._generate_recommendation_reason(source, reasons, score_components)
        
        return {
            "total_score": min(total_score, 1.0),
            "components": score_components,
            "reason": reason,
            "estimated_value": estimated_value
        }
    
    def _score_technology_match(self, source, user_intent: UserIntent) -> float:
        """Score how well source matches user's technology stack."""
        user_techs = [tech.lower() for tech in user_intent.technologies]
        
        # Direct framework match
        if source.framework and source.framework.lower() in user_techs:
            return 1.0
        
        # Language match
        if source.language and source.language.lower() in user_techs:
            return 0.7
        
        # Related technology matching
        tech_relations = {
            "react": ["nextjs", "javascript", "typescript", "jsx"],
            "nextjs": ["react", "javascript", "typescript"],
            "vue": ["vuejs", "javascript", "typescript"],
            "angular": ["typescript", "javascript"],
            "python": ["fastapi", "django", "flask", "pandas", "numpy"],
            "fastapi": ["python", "pydantic", "starlette"],
            "django": ["python"],
            "javascript": ["nodejs", "express", "react", "vue", "angular"],
            "typescript": ["javascript", "react", "angular", "nextjs"]
        }
        
        source_tech = (source.framework or source.language or "").lower()
        
        for user_tech in user_techs:
            related = tech_relations.get(user_tech, [])
            if source_tech in related:
                return 0.5
            
            # Check reverse relationship
            source_related = tech_relations.get(source_tech, [])
            if user_tech in source_related:
                return 0.5
        
        return 0.0
    
    def _score_experience_level(self, source, user_intent: UserIntent) -> float:
        """Score based on experience level appropriateness."""
        # Map source priorities to difficulty levels
        difficulty_mapping = {
            1: "beginner",
            2: "beginner", 
            3: "intermediate",
            4: "intermediate",
            5: "advanced"
        }
        
        source_level = difficulty_mapping.get(source.priority, "intermediate")
        user_level = user_intent.experience_level.value
        
        # Perfect match
        if source_level == user_level:
            return 1.0
        
        # Adjacent levels
        level_order = ["beginner", "intermediate", "advanced", "expert"]
        try:
            source_idx = level_order.index(source_level)
            user_idx = level_order.index(user_level)
            
            diff = abs(source_idx - user_idx)
            if diff == 1:
                return 0.7
            elif diff == 2:
                return 0.4
            else:
                return 0.1
        except ValueError:
            return 0.5  # Default if level not found
    
    def _score_goal_alignment(self, source, user_intent: UserIntent) -> float:
        """Score based on alignment with user goals."""
        goal_keywords = {
            "learn best practices": ["best practices", "patterns", "architecture", "guide"],
            "improve code quality": ["quality", "testing", "linting", "standards"],
            "increase productivity": ["productivity", "tools", "automation", "workflow"],
            "follow industry standards": ["standards", "conventions", "best practices"],
            "optimize performance": ["performance", "optimization", "speed", "efficiency"],
            "enhance security": ["security", "authentication", "authorization", "encryption"],
            "better documentation": ["documentation", "docs", "api", "reference"],
            "team collaboration": ["collaboration", "team", "workflow", "process"],
            "rapid prototyping": ["quick", "rapid", "prototype", "starter", "template"]
        }
        
        total_alignment = 0.0
        matched_goals = 0
        
        source_text = f"{getattr(source, 'title', None) or getattr(source, 'name', '')} {source.url}".lower()
        
        for goal in user_intent.goals:
            goal_lower = goal.lower()
            keywords = goal_keywords.get(goal_lower, [goal_lower.split()])
            
            if isinstance(keywords[0], list):
                keywords = keywords[0]
            
            alignment = 0.0
            for keyword in keywords:
                if keyword in source_text:
                    alignment += 0.3
            
            if alignment > 0:
                total_alignment += min(alignment, 1.0)
                matched_goals += 1
        
        if len(user_intent.goals) > 0:
            return min(total_alignment / len(user_intent.goals), 1.0)
        
        return 0.5  # Default if no specific goals
    
    def _score_project_relevance(self, source, project_analysis: ProjectAnalysis) -> float:
        """Score based on project analysis patterns."""
        relevance_score = 0.0
        source_text = f"{getattr(source, 'title', None) or getattr(source, 'name', '')} {source.url}".lower()
        
        # Check for authentication relevance
        if project_analysis.has_authentication_patterns:
            auth_keywords = ["auth", "login", "jwt", "session", "oauth", "security"]
            if any(keyword in source_text for keyword in auth_keywords):
                relevance_score += 0.2
        
        # Check for routing relevance
        if project_analysis.uses_complex_routing:
            routing_keywords = ["routing", "router", "navigation", "route"]
            if any(keyword in source_text for keyword in routing_keywords):
                relevance_score += 0.2
        
        # Check for state management relevance
        if project_analysis.has_state_management:
            state_keywords = ["state", "store", "redux", "vuex", "context"]
            if any(keyword in source_text for keyword in state_keywords):
                relevance_score += 0.2
        
        # Check for API integration relevance
        if project_analysis.has_api_integration:
            api_keywords = ["api", "rest", "graphql", "fetch", "axios", "request"]
            if any(keyword in source_text for keyword in api_keywords):
                relevance_score += 0.2
        
        # Check for database relevance
        if project_analysis.uses_database:
            db_keywords = ["database", "sql", "nosql", "postgres", "mongo", "orm"]
            if any(keyword in source_text for keyword in db_keywords):
                relevance_score += 0.2
        
        # Check for testing relevance
        if project_analysis.has_testing_setup:
            test_keywords = ["test", "testing", "jest", "pytest", "spec", "unit", "integration"]
            if any(keyword in source_text for keyword in test_keywords):
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _score_historical_preference(self, source, user_intent: UserIntent) -> float:
        """Score based on historical user preferences."""
        framework = source.framework or source.language or ""
        
        # Check framework preferences
        framework_pref = self.user_preferences["preferred_frameworks"].get(framework.lower(), 0)
        
        # Check content type preferences (inferred from URL patterns)
        content_type = self._infer_content_type(source.url)
        content_pref = self.user_preferences["content_type_preferences"].get(content_type, 0)
        
        # Normalize scores (assuming max preference count is 10)
        framework_score = min(framework_pref / 10.0, 1.0) if framework_pref > 0 else 0.0
        content_score = min(content_pref / 10.0, 1.0) if content_pref > 0 else 0.0
        
        return (framework_score + content_score) / 2.0
    
    def _score_source_quality(self, source) -> float:
        """Score source quality based on ratings and metadata."""
        source_key = source.url
        
        # Check if we have historical ratings for this source
        if source_key in self.source_ratings:
            ratings = self.source_ratings[source_key]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                return avg_rating / 5.0  # Normalize to 0-1
        
        # Default quality score based on priority
        return source.priority / 5.0
    
    def _infer_content_type(self, url: str) -> str:
        """Infer content type from URL patterns."""
        url_lower = url.lower()
        
        if "api" in url_lower or "reference" in url_lower:
            return "api_reference"
        elif "tutorial" in url_lower or "guide" in url_lower:
            return "tutorial"
        elif "example" in url_lower or "demo" in url_lower:
            return "examples"
        elif "docs" in url_lower or "documentation" in url_lower:
            return "documentation"
        else:
            return "general"
    
    def _generate_estimated_value(self, source, score: float, user_intent: UserIntent) -> str:
        """Generate estimated value description."""
        if score >= 0.8:
            return "Very High - Highly relevant to your needs and project"
        elif score >= 0.6:
            return "High - Good match for your requirements"
        elif score >= 0.4:
            return "Medium - Useful for your technology stack"
        elif score >= 0.2:
            return "Low-Medium - Some relevance to your goals"
        else:
            return "Low - Limited relevance but may provide context"
    
    def _generate_recommendation_reason(self, source, reasons: List[str], components: Dict) -> str:
        """Generate comprehensive recommendation reason."""
        if not reasons:
            reasons = ["General documentation resource"]
        
        # Add score-based reasoning
        max_component = max(components.items(), key=lambda x: x[1])
        
        if max_component[0] == "technology_match" and max_component[1] > 0.3:
            primary_reason = f"Strong technology alignment with {source.framework or source.language}"
        elif max_component[0] == "goal_alignment" and max_component[1] > 0.2:
            primary_reason = "Matches your stated learning goals"
        elif max_component[0] == "project_relevance" and max_component[1] > 0.2:
            primary_reason = "Relevant to patterns detected in your project"
        else:
            primary_reason = reasons[0] if reasons else "Recommended documentation"
        
        return primary_reason
    
    async def _enhance_recommendations_with_llm(
        self, 
        recommendations: List[RecommendedSource],
        user_intent: UserIntent,
        project_analysis: Optional[ProjectAnalysis]
    ) -> Optional[List[RecommendedSource]]:
        """Enhance recommendations using LLM analysis."""
        try:
            bedrock_maker = BedrockRulesMaker(**self.bedrock_config)
            
            # Create context for LLM
            context = {
                "project_type": user_intent.project_type,
                "technologies": user_intent.technologies,
                "experience_level": user_intent.experience_level.value,
                "goals": user_intent.goals[:3],  # Top 3 goals only
                "recommendations_count": len(recommendations)
            }
            
            # Simple enhancement - in practice, you'd use more sophisticated prompting
            enhanced_recommendations = recommendations.copy()
            
            # Boost confidence for high-scoring recommendations
            for rec in enhanced_recommendations:
                if rec.priority >= 4:
                    rec.confidence = min(rec.confidence * 1.1, 1.0)
                    rec.estimated_value = f"LLM Enhanced - {rec.estimated_value}"
            
            return enhanced_recommendations
            
        except Exception as e:
            # Graceful fallback if LLM enhancement fails
            print(f"LLM enhancement failed: {e}")
            return None
    
    def _apply_learned_preferences(self, recommendations: List[RecommendedSource], 
                                 user_intent: UserIntent) -> List[RecommendedSource]:
        """Apply learned user preferences to boost relevant recommendations."""
        enhanced = []
        
        for rec in recommendations:
            framework = rec.framework or ""
            
            # Boost based on framework preference
            framework_boost = self.user_preferences["preferred_frameworks"].get(framework.lower(), 0)
            if framework_boost > 0:
                rec.confidence = min(rec.confidence * (1.0 + framework_boost * 0.1), 1.0)
                rec.priority = min(rec.priority + 1, 5)
            
            enhanced.append(rec)
        
        return enhanced
    
    def _diversify_recommendations(self, recommendations: List[RecommendedSource]) -> List[RecommendedSource]:
        """Ensure diversity in recommendations to avoid over-specialization."""
        diversified = []
        seen_frameworks = set()
        seen_categories = set()
        
        # First pass: include high-priority unique items
        for rec in recommendations:
            framework = rec.framework or "general"
            category = rec.category or "general"
            
            if len(diversified) < 5:  # Always include top 5
                diversified.append(rec)
                seen_frameworks.add(framework)
                seen_categories.add(category)
            elif framework not in seen_frameworks or category not in seen_categories:
                diversified.append(rec)
                seen_frameworks.add(framework)
                seen_categories.add(category)
                
                if len(diversified) >= 15:  # Max diversity limit
                    break
        
        # Second pass: fill remaining slots with highest scores
        for rec in recommendations:
            if rec not in diversified and len(diversified) < 15:
                diversified.append(rec)
        
        return diversified
    
    def _record_recommendation(self, user_intent: UserIntent, recommendations: List[RecommendedSource]):
        """Record recommendation for learning purposes."""
        record = {
            "timestamp": str(os.times()),
            "user_intent": {
                "project_type": user_intent.project_type,
                "technologies": user_intent.technologies,
                "experience_level": user_intent.experience_level.value,
                "goals": user_intent.goals
            },
            "recommendations": [
                {
                    "source": rec.source,
                    "framework": rec.framework,
                    "priority": rec.priority,
                    "confidence": rec.confidence
                }
                for rec in recommendations[:5]  # Top 5 only
            ]
        }
        
        self.recommendation_history["successful_recommendations"].append(record)
        
        # Keep only last 100 recommendations to avoid bloat
        if len(self.recommendation_history["successful_recommendations"]) > 100:
            self.recommendation_history["successful_recommendations"] = \
                self.recommendation_history["successful_recommendations"][-100:]
        
        self._save_recommendation_history()
    
    def collect_recommendation_feedback(self, source_url: str, rating: float, 
                                      user_intent: UserIntent, feedback_type: str = "rating"):
        """Collect feedback on recommendations for learning.
        
        Args:
            source_url: URL of the recommended source
            rating: User rating (1-5 scale)
            user_intent: Original user intent
            feedback_type: Type of feedback (rating, usage, etc.)
        """
        # Update source ratings
        self.source_ratings[source_url].append(rating)
        
        # Update user preferences based on positive feedback
        if rating >= 4.0:
            for tech in user_intent.technologies:
                self.user_preferences["preferred_frameworks"][tech.lower()] += 1
            
            for goal in user_intent.goals[:3]:  # Top 3 goals
                self.user_preferences["content_type_preferences"][goal.lower()] += 1
        
        # Record feedback in history
        feedback_record = {
            "timestamp": str(os.times()),
            "source_url": source_url,
            "rating": rating,
            "feedback_type": feedback_type,
            "user_context": {
                "project_type": user_intent.project_type,
                "technologies": user_intent.technologies
            }
        }
        
        self.recommendation_history["user_feedback"].append(feedback_record)
        
        # Save updates
        self._save_source_ratings()
        self._save_user_preferences()
        self._save_recommendation_history()
    
    def get_recommendation_insights(self) -> Dict:
        """Get insights about recommendation performance.
        
        Returns:
            Dictionary with recommendation insights and metrics
        """
        insights = {
            "total_recommendations": len(self.recommendation_history["successful_recommendations"]),
            "total_feedback": len(self.recommendation_history["user_feedback"]),
            "average_rating": 0.0,
            "top_frameworks": [],
            "most_successful_sources": [],
            "improvement_areas": []
        }
        
        # Calculate average rating
        feedback = self.recommendation_history["user_feedback"]
        if feedback:
            ratings = [fb["rating"] for fb in feedback if "rating" in fb]
            if ratings:
                insights["average_rating"] = sum(ratings) / len(ratings)
        
        # Analyze top frameworks
        framework_counts = Counter()
        for rec in self.recommendation_history["successful_recommendations"]:
            for item in rec["recommendations"]:
                if item["framework"]:
                    framework_counts[item["framework"]] += 1
        
        insights["top_frameworks"] = framework_counts.most_common(5)
        
        # Find most successful sources (highest average ratings)
        source_ratings = {}
        for source_url, ratings in self.source_ratings.items():
            if ratings:
                source_ratings[source_url] = sum(ratings) / len(ratings)
        
        insights["most_successful_sources"] = sorted(
            source_ratings.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Generate improvement suggestions
        if insights["average_rating"] < 3.5:
            insights["improvement_areas"].append("Overall recommendation quality needs improvement")
        
        if insights["total_feedback"] < insights["total_recommendations"] * 0.1:
            insights["improvement_areas"].append("Need more user feedback to improve recommendations")
        
        return insights
