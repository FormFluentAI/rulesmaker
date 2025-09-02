"""
Learning Pipeline Architecture (Phase 2 - 1.3)

Coordinates:
 - data_collection: UsageTracker (usage_metrics, user_feedback, rule_modifications, success_indicators)
 - analysis_engine: SemanticAnalyzer + LearningEngine.analyze_usage_patterns (pattern_recognition, effectiveness_scoring, improvement_identification, trend_analysis[lightweight])
 - optimization_engine: LearningEngine.optimize_rules + validate_improvements (rule_refinement, template_enhancement[advice], format_optimization[advice], quality_validation)
"""

from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

from .usage_tracker import UsageTracker
from .pattern_analyzer import SemanticAnalyzer
from .engine import LearningEngine
from .models import (
    PipelineReport,
    Recommendation,
    UsageInsights,
    OptimizedRules,
)
from ..models import Rule


@dataclass
class LearningPipeline:
    usage_tracker: UsageTracker
    learning_engine: LearningEngine
    semantic_analyzer: SemanticAnalyzer

    @classmethod
    def default(cls) -> "LearningPipeline":
        return cls(UsageTracker(), LearningEngine(), SemanticAnalyzer())

    # ---- Orchestration ----
    def run(
        self,
        rule_map: Dict[str, Rule],
        content: Optional[str] = None,
    ) -> PipelineReport:
        # data_collection → events → GeneratedRule
        generated = self.usage_tracker.as_generated_rules(rule_map)

        # analysis_engine → usage insights
        insights: UsageInsights = self.learning_engine.analyze_usage_patterns(generated)

        # optimization_engine → optimized rules
        # provide map for in-place property updates
        self.learning_engine.config.setdefault("rule_map", rule_map)
        optimized: OptimizedRules = self.learning_engine.optimize_rules(insights)

        # quality_validation
        validations = {}
        for updated in optimized.rules:
            before = rule_map.get(updated.id)
            if before:
                validations[updated.id] = self.learning_engine.validate_improvements(before, updated)

        # pattern_recognition and content analysis if content provided
        content_analysis = self.semantic_analyzer.analyze_content(content) if content else None

        # template_enhancement / format_optimization (advice)
        recs: List[Recommendation] = []
        if content_analysis and any(p.category == "testing" for p in content_analysis.patterns.patterns):
            recs.append(
                Recommendation(
                    kind="template_enhancement",
                    message="Include a testing-focused rule section and examples in templates.",
                    targets=["templates"],
                    confidence=0.6,
                )
            )
        # Suggest Windsurf formatting when async-heavy
        if content_analysis and any(p.name == "Async Pattern" for p in content_analysis.patterns.patterns):
            recs.append(
                Recommendation(
                    kind="format_optimization",
                    message="Emphasize async guidelines and non-blocking IO patterns in Windsurf/Cursor formats.",
                    targets=["windsurf", "cursor"],
                    confidence=0.55,
                )
            )

        return PipelineReport(
            insights=insights,
            optimized=optimized,
            validations=validations,
            content_analysis=content_analysis,
            recommendations=recs,
        )

