"""
Intelligence module for Rules Maker.

This module provides AI-powered content understanding, categorization, and
intelligent user experience enhancements.
"""

from .semantic_analyzer import SemanticAnalyzer, ContentAnalysis
from .category_engine import IntelligentCategoryEngine
from .recommendation_engine import SmartRecommendationEngine, RecommendedSource, UserIntent

__all__ = [
    "SemanticAnalyzer",
    "ContentAnalysis", 
    "IntelligentCategoryEngine",
    "SmartRecommendationEngine",
    "RecommendedSource",
    "UserIntent",
]