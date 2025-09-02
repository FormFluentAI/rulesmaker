"""
Intelligent Category Engine.

Advanced categorization system that evolves based on user feedback and usage patterns.
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import yaml
import os
import json

from .models import ContentAnalysis, CategoryConfidence
from .semantic_analyzer import SemanticAnalyzer


class IntelligentCategoryEngine:
    """Intelligent content categorization with adaptive learning."""
    
    def __init__(self, taxonomy_path: str = "config/intelligent_taxonomy.yaml"):
        """Initialize the category engine.
        
        Args:
            taxonomy_path: Path to the taxonomy configuration file
        """
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self._load_taxonomy()
        self.category_weights = self._load_category_weights()
        self.user_feedback = self._load_user_feedback()
        self.semantic_analyzer = SemanticAnalyzer()
        
    def _load_taxonomy(self) -> Dict:
        """Load taxonomy configuration."""
        if os.path.exists(self.taxonomy_path):
            with open(self.taxonomy_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_category_weights(self) -> Dict:
        """Load learned category weights from previous sessions."""
        weights_path = "data/category_weights.json"
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                return json.load(f)
        return defaultdict(lambda: 1.0)
    
    def _save_category_weights(self):
        """Save learned category weights."""
        os.makedirs("data", exist_ok=True)
        with open("data/category_weights.json", 'w') as f:
            json.dump(dict(self.category_weights), f, indent=2)
    
    def _load_user_feedback(self) -> Dict:
        """Load user feedback data."""
        feedback_path = "data/user_feedback.json"
        if os.path.exists(feedback_path):
            with open(feedback_path, 'r') as f:
                return json.load(f)
        return {"category_ratings": defaultdict(list), "pattern_feedback": defaultdict(list)}
    
    def _save_user_feedback(self):
        """Save user feedback data."""
        os.makedirs("data", exist_ok=True)
        # Convert defaultdict to regular dict for JSON serialization
        feedback_data = {
            "category_ratings": dict(self.user_feedback["category_ratings"]),
            "pattern_feedback": dict(self.user_feedback["pattern_feedback"])
        }
        with open("data/user_feedback.json", 'w') as f:
            json.dump(feedback_data, f, indent=2)
    
    async def categorize_content(self, content: str, url: str = "", 
                                technology_hint: Optional[str] = None) -> Dict[str, CategoryConfidence]:
        """Intelligently categorize documentation content.
        
        Args:
            content: The documentation content
            url: Optional URL for additional context
            technology_hint: Optional technology hint to improve categorization
            
        Returns:
            Dictionary of categories with confidence scores
        """
        # Use semantic analyzer for initial analysis
        analysis = await self.semantic_analyzer.analyze_content(content, url or "")
        
        # Get base categories from semantic analysis
        categories = analysis.content_categories
        
        # Apply learned weights and patterns
        enhanced_categories = self._apply_learned_patterns(
            content, categories, analysis.primary_technology
        )
        
        # Cross-reference with taxonomy
        taxonomy_categories = self._apply_taxonomy_matching(
            content, analysis.primary_technology
        )
        
        # Merge and weight the results
        final_categories = self._merge_category_results(
            enhanced_categories, taxonomy_categories
        )
        
        # Apply user feedback adjustments
        adjusted_categories = self._apply_user_feedback_adjustments(
            final_categories, analysis.primary_technology
        )
        
        return adjusted_categories
    
    def _apply_learned_patterns(self, content: str, base_categories: Dict[str, CategoryConfidence], 
                               technology: str) -> Dict[str, CategoryConfidence]:
        """Apply learned patterns to improve categorization."""
        enhanced = base_categories.copy()
        content_lower = content.lower()
        
        # Apply learned weights for this technology
        tech_weights = self.category_weights.get(technology, {})
        
        for category, confidence in enhanced.items():
            weight = tech_weights.get(category, 1.0)
            enhanced[category] = CategoryConfidence(
                confidence=min(confidence.confidence * weight, 1.0),
                topics=confidence.topics,
                patterns=confidence.patterns
            )
        
        # Look for new patterns based on feedback
        feedback_patterns = self.user_feedback.get("pattern_feedback", {})
        for pattern, feedback_data in feedback_patterns.items():
            if pattern.lower() in content_lower:
                # Extract suggested category from feedback
                suggested_categories = [fb.get("suggested_category") for fb in feedback_data 
                                      if fb.get("suggested_category")]
                if suggested_categories:
                    most_common = Counter(suggested_categories).most_common(1)[0][0]
                    if most_common not in enhanced:
                        enhanced[most_common] = CategoryConfidence(
                            confidence=0.6,
                            topics=[pattern],
                            patterns=["user_feedback_pattern"]
                        )
        
        return enhanced
    
    def _apply_taxonomy_matching(self, content: str, technology: str) -> Dict[str, CategoryConfidence]:
        """Apply taxonomy-based pattern matching."""
        categories = {}
        
        if technology not in self.taxonomy.get("frameworks", {}):
            return categories
        
        tech_config = self.taxonomy["frameworks"][technology]
        content_lower = content.lower()
        
        for category, config in tech_config.get("categories", {}).items():
            confidence = 0.0
            detected_topics = []
            detected_patterns = []
            
            # Pattern matching with enhanced scoring
            patterns = config.get("patterns", [])
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in content_lower:
                    # Count occurrences and weight by pattern importance
                    occurrences = content_lower.count(pattern_lower)
                    base_score = min(occurrences * 0.1, 0.3)
                    
                    # Apply learned weight for this pattern
                    pattern_weight = self.category_weights.get(f"{technology}_{category}_{pattern}", 1.0)
                    confidence += base_score * pattern_weight
                    detected_patterns.append(pattern)
            
            # Context clues matching
            context_clues = config.get("context_clues", [])
            for clue in context_clues:
                if clue.lower() in content_lower:
                    confidence += 0.15
                    detected_topics.append(f"context: {clue}")
            
            # Subcategory detection with enhanced weighting
            subcategories = config.get("subcategories", {})
            for subcat, sub_patterns in subcategories.items():
                for pattern in sub_patterns:
                    if pattern.lower() in content_lower:
                        confidence += 0.08
                        detected_topics.append(f"{subcat}: {pattern}")
            
            # Difficulty-based confidence adjustment
            difficulty_markers = config.get("difficulty_markers", {})
            for level, markers in difficulty_markers.items():
                for marker in markers:
                    if marker.lower() in content_lower:
                        # Weight confidence based on difficulty level
                        difficulty_weights = {
                            "beginner": 1.0,
                            "intermediate": 1.1,
                            "advanced": 1.2,
                            "expert": 1.3
                        }
                        weight = difficulty_weights.get(level, 1.0)
                        confidence *= weight
                        detected_topics.append(f"difficulty: {level}")
            
            # Normalize confidence
            confidence = min(confidence, 1.0)
            
            if confidence > 0.05:  # Lower threshold for more inclusive categorization
                categories[category] = CategoryConfidence(
                    confidence=confidence,
                    topics=detected_topics,
                    patterns=detected_patterns
                )
        
        return categories
    
    def _merge_category_results(self, semantic_categories: Dict[str, CategoryConfidence],
                              taxonomy_categories: Dict[str, CategoryConfidence]) -> Dict[str, CategoryConfidence]:
        """Merge results from different categorization methods."""
        merged = {}
        all_categories = set(semantic_categories.keys()) | set(taxonomy_categories.keys())
        
        for category in all_categories:
            semantic_conf = semantic_categories.get(category)
            taxonomy_conf = taxonomy_categories.get(category)
            
            if semantic_conf and taxonomy_conf:
                # Both methods detected this category - combine with weighted average
                combined_confidence = (semantic_conf.confidence * 0.4 + 
                                     taxonomy_conf.confidence * 0.6)
                combined_topics = list(set(semantic_conf.topics + taxonomy_conf.topics))
                combined_patterns = list(set(semantic_conf.patterns + taxonomy_conf.patterns))
                
                merged[category] = CategoryConfidence(
                    confidence=combined_confidence,
                    topics=combined_topics,
                    patterns=combined_patterns
                )
            elif taxonomy_conf:
                # Only taxonomy detected - use with slight reduction
                merged[category] = CategoryConfidence(
                    confidence=taxonomy_conf.confidence * 0.9,
                    topics=taxonomy_conf.topics,
                    patterns=taxonomy_conf.patterns
                )
            elif semantic_conf:
                # Only semantic detected - use as is
                merged[category] = semantic_conf
        
        return merged
    
    def _apply_user_feedback_adjustments(self, categories: Dict[str, CategoryConfidence], 
                                        technology: str) -> Dict[str, CategoryConfidence]:
        """Apply user feedback to adjust category scores."""
        adjusted = categories.copy()
        
        category_ratings = self.user_feedback.get("category_ratings", {})
        
        for category, confidence in adjusted.items():
            category_key = f"{technology}_{category}"
            if category_key in category_ratings:
                ratings = category_ratings[category_key]
                if ratings:
                    avg_rating = sum(ratings) / len(ratings)
                    # Adjust confidence based on average user rating (0-5 scale)
                    rating_multiplier = (avg_rating / 5.0) * 0.5 + 0.75  # 0.75-1.25 range
                    
                    adjusted[category] = CategoryConfidence(
                        confidence=min(confidence.confidence * rating_multiplier, 1.0),
                        topics=confidence.topics,
                        patterns=confidence.patterns
                    )
        
        return adjusted
    
    def collect_user_feedback(self, technology: str, category: str, rating: float, 
                            feedback_type: str = "rating"):
        """Collect user feedback for continuous improvement.
        
        Args:
            technology: The technology being categorized
            category: The category being rated
            rating: User rating (0-5 scale)
            feedback_type: Type of feedback (rating, pattern_suggestion, etc.)
        """
        if feedback_type == "rating":
            category_key = f"{technology}_{category}"
            self.user_feedback["category_ratings"][category_key].append(rating)
            
            # Update category weight based on rating
            if rating >= 4.0:
                self.category_weights[category_key] *= 1.05  # Boost good categories
            elif rating <= 2.0:
                self.category_weights[category_key] *= 0.95  # Reduce poor categories
        
        self._save_user_feedback()
        self._save_category_weights()
    
    def collect_pattern_feedback(self, pattern: str, suggested_category: str, 
                               is_helpful: bool):
        """Collect feedback about specific patterns.
        
        Args:
            pattern: The pattern that was detected
            suggested_category: Category the user suggests for this pattern
            is_helpful: Whether this pattern was helpful for categorization
        """
        self.user_feedback["pattern_feedback"][pattern].append({
            "suggested_category": suggested_category,
            "is_helpful": is_helpful,
            "timestamp": str(os.times())
        })
        
        self._save_user_feedback()
    
    def get_category_insights(self, technology: str) -> Dict:
        """Get insights about categorization performance for a technology.
        
        Args:
            technology: The technology to analyze
            
        Returns:
            Dictionary with categorization insights
        """
        insights = {
            "total_categories": 0,
            "high_confidence_categories": 0,
            "user_feedback_count": 0,
            "avg_user_rating": 0.0,
            "learned_patterns": 0,
            "improvement_suggestions": []
        }
        
        # Analyze taxonomy coverage
        if technology in self.taxonomy.get("frameworks", {}):
            tech_config = self.taxonomy["frameworks"][technology]
            insights["total_categories"] = len(tech_config.get("categories", {}))
        
        # Analyze user feedback
        category_ratings = self.user_feedback.get("category_ratings", {})
        tech_ratings = []
        for key, ratings in category_ratings.items():
            if key.startswith(f"{technology}_"):
                tech_ratings.extend(ratings)
        
        if tech_ratings:
            insights["user_feedback_count"] = len(tech_ratings)
            insights["avg_user_rating"] = sum(tech_ratings) / len(tech_ratings)
        
        # Count learned patterns
        tech_weights = {k: v for k, v in self.category_weights.items() 
                       if k.startswith(f"{technology}_")}
        insights["learned_patterns"] = len(tech_weights)
        
        # Generate improvement suggestions
        if insights["avg_user_rating"] < 3.5:
            insights["improvement_suggestions"].append(
                "Consider reviewing taxonomy patterns for better accuracy"
            )
        
        if insights["learned_patterns"] < 5:
            insights["improvement_suggestions"].append(
                "More user feedback needed to improve categorization"
            )
        
        return insights
    
    def export_learned_patterns(self) -> Dict:
        """Export learned patterns for backup or transfer.
        
        Returns:
            Dictionary containing all learned patterns and feedback
        """
        return {
            "category_weights": dict(self.category_weights),
            "user_feedback": {
                "category_ratings": dict(self.user_feedback["category_ratings"]),
                "pattern_feedback": dict(self.user_feedback["pattern_feedback"])
            },
            "export_timestamp": str(os.times())
        }
    
    def import_learned_patterns(self, patterns_data: Dict):
        """Import learned patterns from backup or another system.
        
        Args:
            patterns_data: Dictionary containing learned patterns
        """
        if "category_weights" in patterns_data:
            self.category_weights.update(patterns_data["category_weights"])
            
        if "user_feedback" in patterns_data:
            feedback = patterns_data["user_feedback"]
            if "category_ratings" in feedback:
                for key, ratings in feedback["category_ratings"].items():
                    self.user_feedback["category_ratings"][key].extend(ratings)
            
            if "pattern_feedback" in feedback:
                for pattern, feedback_list in feedback["pattern_feedback"].items():
                    self.user_feedback["pattern_feedback"][pattern].extend(feedback_list)
        
        self._save_category_weights()
        self._save_user_feedback()