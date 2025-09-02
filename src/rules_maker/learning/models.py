"""
Models for the Intelligent Learning Engine.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, UTC
from pydantic import BaseModel, Field

from ..models import Rule


class UsageEvent(BaseModel):
    """Represents a single usage event for a rule.

    Attributes:
        rule_id: The ID of the rule used.
        action: The kind of interaction (e.g., 'applied', 'suggested', 'skipped').
        success: Whether usage led to a positive outcome.
        timestamp: When the event occurred.
        context: Optional free-form metadata about where/how used.
        feedback_score: Optional explicit user feedback in [-1.0, 1.0].
    """

    rule_id: str
    action: str = "applied"
    success: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: Dict[str, Any] = Field(default_factory=dict)
    feedback_score: Optional[float] = None


class GeneratedRule(BaseModel):
    """A generated rule with attached usage and feedback signals."""

    rule: Rule
    usage_events: List[UsageEvent] = Field(default_factory=list)
    # Optional grouping for A/B tests
    experiment_key: Optional[str] = None
    variant_key: Optional[str] = None


class RuleEffectiveness(BaseModel):
    """Aggregated effectiveness metrics for a single rule."""

    rule_id: str
    title: str = ""
    usage_count: int
    success_count: int
    success_rate: float
    avg_feedback: float
    last_used_at: Optional[datetime] = None
    sections_effectiveness: Dict[str, float] = Field(default_factory=dict)


class UsageInsights(BaseModel):
    """Insights derived from usage data across rules."""

    by_rule: Dict[str, RuleEffectiveness] = Field(default_factory=dict)
    total_events: int = 0
    global_success_rate: float = 0.0
    underperforming_rules: List[str] = Field(default_factory=list)
    top_rules: List[str] = Field(default_factory=list)


class ChangeRecord(BaseModel):
    """Describes a change proposed or applied to a rule."""

    rule_id: str
    change_type: str  # e.g., 'priority_adjust', 'content_tweak', 'deprecate'
    description: str
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)
    score_delta: float = 0.0


class OptimizedRules(BaseModel):
    """Result of an optimization pass."""

    rules: List[Rule] = Field(default_factory=list)
    changes: List[ChangeRecord] = Field(default_factory=list)
    quality_score: float = 0.0


class QualityMetrics(BaseModel):
    """Quality metrics comparing rule versions."""

    relevance_improvement: float = 0.0
    clarity_improvement: float = 0.0
    duplication_reduction: float = 0.0
    overall_score: float = 0.0
    notes: Optional[str] = None


class ABVariant(BaseModel):
    """A single variant in an A/B test."""

    key: str
    description: str = ""
    rule_overrides: Dict[str, Any] = Field(default_factory=dict)


class ABTest(BaseModel):
    """Represents an A/B test experiment for a rule."""

    experiment_key: str
    rule_id: str
    variants: List[ABVariant]
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: Optional[datetime] = None


class ABTestResult(BaseModel):
    """Aggregated outcomes for an A/B test."""

    experiment_key: str
    winner_variant: Optional[str]
    variant_metrics: Dict[str, RuleEffectiveness] = Field(default_factory=dict)
    p_value: Optional[float] = None
    notes: Optional[str] = None


# ===== Semantic Content Understanding Models (Phase 2 - 1.2) =====

class CodePattern(BaseModel):
    """A single code/documentation pattern discovered in content."""

    name: str
    description: str = ""
    occurrences: int = 0
    confidence: float = 0.0  # [0,1]
    language: Optional[str] = None
    category: Optional[str] = None  # e.g., 'structure', 'async', 'config', 'testing'
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class CodePatterns(BaseModel):
    """Container for discovered code/documentation patterns."""

    patterns: List[CodePattern] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BestPractice(BaseModel):
    """Represents a best practice extracted from content/patterns."""

    name: str
    description: str
    rationale: str = ""
    references: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    tags: List[str] = Field(default_factory=list)


class BestPractices(BaseModel):
    items: List[BestPractice] = Field(default_factory=list)


class AntiPattern(BaseModel):
    """Represents an anti-pattern with remediation guidance."""

    name: str
    description: str
    impact: str = ""
    remediation: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    confidence: float = 0.0
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class AntiPatterns(BaseModel):
    items: List[AntiPattern] = Field(default_factory=list)


class ContentAnalysis(BaseModel):
    """Aggregated semantic analysis of content."""

    content_summary: str = ""
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    key_topics: List[str] = Field(default_factory=list)
    patterns: CodePatterns = Field(default_factory=CodePatterns)
    best_practices: BestPractices = Field(default_factory=BestPractices)
    anti_patterns: AntiPatterns = Field(default_factory=AntiPatterns)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CustomRules(BaseModel):
    """Custom rules generated from content analysis, aligned to project context."""

    rules: List[Rule] = Field(default_factory=list)
    coverage: Dict[str, float] = Field(default_factory=dict)  # e.g., {'best_practices':0.8}
    notes: str = ""


class Recommendation(BaseModel):
    """Recommendation produced by the pipeline (template/format/other)."""

    kind: Literal["template_enhancement", "format_optimization", "general"] = "general"
    message: str
    targets: List[str] = Field(default_factory=list)  # rule_ids or categories
    confidence: float = 0.0


class PipelineReport(BaseModel):
    """Aggregate report from the learning pipeline run."""

    insights: UsageInsights
    optimized: OptimizedRules
    validations: Dict[str, QualityMetrics] = Field(default_factory=dict)  # rule_id -> metrics
    content_analysis: Optional[ContentAnalysis] = None
    recommendations: List[Recommendation] = Field(default_factory=list)
