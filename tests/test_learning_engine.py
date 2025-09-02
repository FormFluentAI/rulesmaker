import pytest
from datetime import datetime, timedelta, UTC

from rules_maker.models import Rule, RuleType
from rules_maker.learning import LearningEngine
from rules_maker.learning.models import GeneratedRule, UsageEvent


def make_rule(idx: int, priority: int = 1, confidence: float = 0.2) -> Rule:
    return Rule(
        id=f"r{idx}",
        title=f"Rule {idx}",
        description="Keep code simple and readable.",
        content="Prefer descriptive names.",
        type=RuleType.BEST_PRACTICE,
        category="style",
        priority=priority,
        confidence_score=confidence,
        tags=["style", "naming"],
        examples=["Use descriptive variable names"],
    )


def test_analyze_usage_patterns_basic():
    r1 = make_rule(1)
    r2 = make_rule(2)

    now = datetime.now(UTC)
    events_r1 = [
        UsageEvent(rule_id=r1.id, success=True, timestamp=now - timedelta(minutes=10), context={"section": "naming"}),
        UsageEvent(rule_id=r1.id, success=True, timestamp=now - timedelta(minutes=5), context={"section": "naming"}, feedback_score=0.5),
        UsageEvent(rule_id=r1.id, success=False, timestamp=now - timedelta(minutes=1), context={"section": "examples"}),
    ]
    events_r2 = [
        UsageEvent(rule_id=r2.id, success=False, timestamp=now - timedelta(minutes=3), context={"section": "general"}),
        UsageEvent(rule_id=r2.id, success=False, timestamp=now - timedelta(minutes=2), context={"section": "general"}),
        UsageEvent(rule_id=r2.id, success=False, timestamp=now - timedelta(minutes=1), context={"section": "general"}),
    ]

    gr1 = GeneratedRule(rule=r1, usage_events=events_r1)
    gr2 = GeneratedRule(rule=r2, usage_events=events_r2)

    engine = LearningEngine()
    insights = engine.analyze_usage_patterns([gr1, gr2])

    assert insights.total_events == 6
    assert r1.id in insights.by_rule and r2.id in insights.by_rule
    assert insights.by_rule[r1.id].success_rate > insights.by_rule[r2.id].success_rate
    assert r2.id in insights.underperforming_rules


def test_optimize_rules_updates_priority_and_confidence():
    r1 = make_rule(1, priority=2, confidence=0.4)
    r2 = make_rule(2, priority=3, confidence=0.6)

    now = datetime.now(UTC)
    events_r1 = [UsageEvent(rule_id=r1.id, success=True) for _ in range(5)]
    events_r2 = [UsageEvent(rule_id=r2.id, success=False) for _ in range(5)]

    engine = LearningEngine(config={"rule_map": {r1.id: r1, r2.id: r2}})
    insights = engine.analyze_usage_patterns([
        GeneratedRule(rule=r1, usage_events=events_r1),
        GeneratedRule(rule=r2, usage_events=events_r2),
    ])

    optimized = engine.optimize_rules(insights)
    ids = {r.id: r for r in optimized.rules}

    assert ids[r1.id].priority >= r1.priority  # boosted or unchanged
    assert ids[r1.id].confidence_score >= r1.confidence_score
    assert ids[r2.id].priority <= r2.priority  # potentially lowered


def test_validate_improvements_scores_directionally():
    before = make_rule(1, priority=2, confidence=0.3)
    after = before.model_copy(update={
        "confidence_score": 0.6,
        "description": before.description + " Includes examples.",
        "examples": before.examples + ["Add code sample."],
        "tags": before.tags[:-1],
    })

    engine = LearningEngine()
    qm = engine.validate_improvements(before, after)
    assert qm.relevance_improvement > 0
    assert qm.clarity_improvement >= 0
    assert qm.overall_score > 0
