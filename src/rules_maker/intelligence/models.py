"""
Data models for the intelligence module.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    """Content complexity levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentType(str, Enum):
    """Types of documentation content."""
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    GUIDE = "guide"
    EXAMPLES = "examples"
    API_DOCS = "api_docs"
    TROUBLESHOOTING = "troubleshooting"


class CategoryConfidence(BaseModel):
    """Confidence score and topics for a content category."""
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    topics: List[str] = Field(default_factory=list, description="Specific topics within this category")
    patterns: List[str] = Field(default_factory=list, description="Detected patterns")


class ContentAnalysis(BaseModel):
    """Results of semantic content analysis."""
    primary_technology: str = Field(description="Main technology/framework detected")
    secondary_technologies: List[str] = Field(default_factory=list, description="Additional technologies")
    content_categories: Dict[str, CategoryConfidence] = Field(
        default_factory=dict, description="Categories with confidence scores"
    )
    complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    content_type: ContentType = ContentType.REFERENCE
    framework_version: Optional[str] = None
    prerequisites: List[str] = Field(default_factory=list)
    language_detected: Optional[str] = None
    code_examples_count: int = 0
    external_links_count: int = 0
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendedSource(BaseModel):
    """Recommended documentation source."""
    source: str = Field(description="URL or path to the documentation")
    reason: str = Field(description="Why this source is recommended")
    priority: int = Field(ge=1, le=5, description="Priority level (1-5, 5 being highest)")
    estimated_value: str = Field(description="Estimated value to the user")
    category: Optional[str] = None
    framework: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class UserIntent(BaseModel):
    """User's intent and context for recommendation."""
    project_type: str = Field(description="Type of project (e.g., 'nextjs-ecommerce', 'react-dashboard')")
    technologies: List[str] = Field(default_factory=list, description="Technologies user is working with")
    experience_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    goals: List[str] = Field(default_factory=list, description="User's stated goals")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Any constraints or preferences")
    current_focus: Optional[str] = None
    time_budget: Optional[int] = None  # in minutes
    learning_preferences: List[str] = Field(default_factory=list)


class ProjectAnalysis(BaseModel):
    """Analysis of a user's project for predictive features."""
    has_authentication_patterns: bool = False
    uses_complex_routing: bool = False
    has_state_management: bool = False
    has_api_integration: bool = False
    uses_database: bool = False
    has_testing_setup: bool = False
    deployment_target: Optional[str] = None
    architectural_patterns: List[str] = Field(default_factory=list)
    technologies_detected: List[str] = Field(default_factory=list)
    complexity_indicators: Dict[str, float] = Field(default_factory=dict)
    potential_issues: List[str] = Field(default_factory=list)


class RulePrediction(BaseModel):
    """Predicted rule need with confidence."""
    rule_type: str = Field(description="Type of rule predicted to be needed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the prediction")
    reason: str = Field(description="Reason for the prediction")
    priority: str = Field(description="Priority level: low, medium, high, critical")
    estimated_impact: Optional[str] = None
    suggested_timing: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response to a natural language query."""
    answer: str = Field(description="Generated answer to the query")
    relevant_sources: List[RecommendedSource] = Field(default_factory=list)
    suggested_rules: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    follow_up_suggestions: List[str] = Field(default_factory=list)


class InteractiveSession(BaseModel):
    """Interactive user session data."""
    session_id: str = Field(description="Unique session identifier")
    user_context: Optional[UserIntent] = None
    project_analysis: Optional[ProjectAnalysis] = None
    recommended_sources: List[RecommendedSource] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)
    current_step: Optional[str] = None
    feedback_collected: List[Dict[str, Any]] = Field(default_factory=list)
    start_time: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)