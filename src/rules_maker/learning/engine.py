"""
Intelligent Learning Engine for adaptive rule optimization.

Implements:
 - Usage Pattern Analysis
 - Iterative Improvement of rules
 - Lightweight A/B testing aggregation
 - Quality scoring and validation
 - Feedback signal integration
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime
import math

from ..models import Rule
from .models import (
    GeneratedRule,
    UsageEvent,
    UsageInsights,
    RuleEffectiveness,
    OptimizedRules,
    ChangeRecord,
    QualityMetrics,
    ABTest,
    ABTestResult,
)


class LearningEngine:
    """Analyze rule usage and optimize rules based on outcomes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        # thresholds/config
        self.min_events = int(self.config.get("min_events", 3))
        self.success_threshold = float(self.config.get("success_threshold", 0.55))
        self.deprecate_threshold = float(self.config.get("deprecate_threshold", 0.15))
        self.priority_boost = int(self.config.get("priority_boost", 1))
        self.priority_cap = int(self.config.get("priority_cap", 5))

    # Intelligent rule refinement based on usage patterns
    def analyze_usage_patterns(self, rules: List[GeneratedRule]) -> UsageInsights:
        """Aggregate usage signals and compute rule effectiveness metrics."""
        by_rule: Dict[str, RuleEffectiveness] = {}
        total_events = 0
        global_success = 0

        for gr in rules:
            events = gr.usage_events
            if not events:
                # Initialize with zeros to keep rule present in insights
                by_rule[gr.rule.id] = RuleEffectiveness(
                    rule_id=gr.rule.id,
                    title=gr.rule.title,
                    usage_count=0,
                    success_count=0,
                    success_rate=0.0,
                    avg_feedback=0.0,
                    last_used_at=None,
                    sections_effectiveness={},
                )
                continue

            usage_count = len(events)
            success_count = sum(1 for e in events if e.success)
            feedback_vals = [e.feedback_score for e in events if e.feedback_score is not None]
            avg_feedback = sum(feedback_vals) / len(feedback_vals) if feedback_vals else 0.0
            last_used = max((e.timestamp for e in events), default=None)

            # Optional: section effectiveness via event.context.get('section')
            sections: Dict[str, List[bool]] = {}
            for e in events:
                section = str(e.context.get("section", "__all__"))
                sections.setdefault(section, []).append(e.success)
            sections_effectiveness = {
                s: (sum(1 for ok in oks if ok) / len(oks) if oks else 0.0)
                for s, oks in sections.items()
            }

            success_rate = success_count / usage_count if usage_count else 0.0
            by_rule[gr.rule.id] = RuleEffectiveness(
                rule_id=gr.rule.id,
                title=gr.rule.title,
                usage_count=usage_count,
                success_count=success_count,
                success_rate=success_rate,
                avg_feedback=avg_feedback,
                last_used_at=last_used,
                sections_effectiveness=sections_effectiveness,
            )

            total_events += usage_count
            global_success += success_count

        global_success_rate = (global_success / total_events) if total_events else 0.0

        # Rank rules
        ranked = sorted(
            by_rule.values(),
            key=lambda r: (r.success_rate, r.usage_count, r.avg_feedback),
            reverse=True,
        )
        top_rules = [r.rule_id for r in ranked[:5]]
        underperforming_rules = [
            r.rule_id for r in ranked if r.usage_count >= self.min_events and r.success_rate < self.success_threshold
        ]

        return UsageInsights(
            by_rule={k: v for k, v in by_rule.items()},
            total_events=total_events,
            global_success_rate=global_success_rate,
            underperforming_rules=underperforming_rules,
            top_rules=top_rules,
        )

    def optimize_rules(self, insights: UsageInsights) -> OptimizedRules:
        """Produce an optimized rule set with proposed changes.

        Strategies:
         - Boost priority for top performers
         - Lower priority or flag for deprecation underperformers
         - Apply light clarity tweaks suggestions via metadata
        """
        rules_out: List[Rule] = []
        changes: List[ChangeRecord] = []

        # If caller passed a rule map, perform real updates
        rule_map: Dict[str, Rule] = self.config.get("rule_map", {})
        for rid, eff in insights.by_rule.items():
            if rid not in rule_map:
                continue
            rule = rule_map[rid]
            before = {"priority": rule.priority, "confidence_score": rule.confidence_score}

            priority = rule.priority
            confidence = rule.confidence_score
            score_delta = 0.0

            if eff.usage_count >= self.min_events:
                # Adjust priority based on success rate vs threshold
                if eff.success_rate >= max(self.success_threshold, 0.7):
                    priority = min(self.priority_cap, priority + self.priority_boost)
                    score_delta += 0.1
                elif eff.success_rate < self.deprecate_threshold:
                    priority = max(1, priority - 1)
                    score_delta -= 0.1

            # Nudge confidence towards measured success rate blended with feedback
            blended = 0.7 * eff.success_rate + 0.3 * ((eff.avg_feedback + 1.0) / 2.0)
            confidence = max(0.0, min(1.0, 0.5 * rule.confidence_score + 0.5 * blended))

            # Suggest clarity tweak metadata when section variance is high
            sections = list(eff.sections_effectiveness.values())
            if len(sections) > 1:
                variance = _variance(sections)
                if variance > 0.05:
                    rule.metadata.setdefault("suggestions", []).append(
                        "Clarify uneven sections based on usage outcomes."
                    )

            # Apply updates (Pydantic v2: use model_copy)
            updated = rule.model_copy(update={"priority": priority, "confidence_score": confidence})
            rules_out.append(updated)

            changes.append(
                ChangeRecord(
                    rule_id=rid,
                    change_type="priority_confidence_update",
                    description=f"Adjusted priority/confidence based on {eff.usage_count} usages",
                    before=before,
                    after={"priority": priority, "confidence_score": confidence},
                    score_delta=score_delta,
                )
            )

        # Global quality score as average of success rates weighted by usage
        denom = sum(e.usage_count for e in insights.by_rule.values()) or 1
        weighted = sum(e.success_rate * e.usage_count for e in insights.by_rule.values()) / denom

        return OptimizedRules(rules=rules_out, changes=changes, quality_score=weighted)

    def validate_improvements(self, before: Rule, after: Rule) -> QualityMetrics:
        """Compare two versions of a rule and estimate quality deltas."""
        # Heuristics: clarity by content length and presence of examples,
        # relevance by confidence score, duplication by tags/categories.
        clarity_before = _clarity_score(before)
        clarity_after = _clarity_score(after)
        rel_before = before.confidence_score
        rel_after = after.confidence_score
        dup_before = _duplication_signal(before)
        dup_after = _duplication_signal(after)

        return QualityMetrics(
            relevance_improvement=rel_after - rel_before,
            clarity_improvement=clarity_after - clarity_before,
            duplication_reduction=dup_before - dup_after,
            overall_score=0.5 * (rel_after - rel_before) + 0.3 * (clarity_after - clarity_before) + 0.2 * (dup_before - dup_after),
            notes="Heuristic comparison based on rule fields",
        )

    # Optional helpers for A/B testing aggregation
    def summarize_ab_test(self, experiment: ABTest, events: List[UsageEvent]) -> ABTestResult:
        """Aggregate outcomes by variant and select a winner (simple heuristic)."""
        by_variant: Dict[str, List[UsageEvent]] = {}
        for ev in events:
            v = ev.context.get("variant") or "control"
            by_variant.setdefault(v, []).append(ev)

        metrics: Dict[str, RuleEffectiveness] = {}
        best_v = None
        best_score = -1.0

        for v, ves in by_variant.items():
            usage_count = len(ves)
            succ = sum(1 for e in ves if e.success)
            rate = succ / usage_count if usage_count else 0.0
            fb_vals = [e.feedback_score for e in ves if e.feedback_score is not None]
            avg_fb = sum(fb_vals) / len(fb_vals) if fb_vals else 0.0
            score = 0.7 * rate + 0.3 * ((avg_fb + 1.0) / 2.0)
            metrics[v] = RuleEffectiveness(
                rule_id=experiment.rule_id,
                title="",
                usage_count=usage_count,
                success_count=succ,
                success_rate=rate,
                avg_feedback=avg_fb,
                last_used_at=max((e.timestamp for e in ves), default=None),
                sections_effectiveness={},
            )
            if score > best_score:
                best_score = score
                best_v = v

        return ABTestResult(
            experiment_key=experiment.experiment_key,
            winner_variant=best_v,
            variant_metrics=metrics,
            p_value=None,
            notes="Winner selected by blended success/feedback score",
        )


def _clarity_score(rule: Rule) -> float:
    # Simple heuristic: presence of title, examples, and concise description
    score = 0.0
    if rule.title:
        score += 0.2
    if rule.description:
        # Favor 80-300 chars as concise
        n = len(rule.description)
        score += 0.4 if 80 <= n <= 300 else 0.2
    if rule.examples:
        score += min(0.4, 0.2 * len(rule.examples))
    return min(score, 1.0)


def _duplication_signal(rule: Rule) -> float:
    # Use count of tags/category overlap as proxy; without corpus we keep minimal
    # Here: more tags could indicate potential overlap, so invert to prefer fewer
    return min(1.0, 0.1 * len(rule.tags))


def _variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)
