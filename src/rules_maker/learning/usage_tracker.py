"""
Usage tracking for the Learning Pipeline (data_collection phase).

Collects usage metrics, user feedback, rule modifications, and success indicators.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import UsageEvent, GeneratedRule, ChangeRecord
from ..models import Rule


class UsageTracker:
    """In-memory usage tracker for rules.

    This can later be replaced by a persistent store.
    """

    def __init__(self) -> None:
        self._events: List[UsageEvent] = []
        self._modifications: List[ChangeRecord] = []

    # ---- Recording APIs ----
    def record_usage(
        self,
        rule_id: str,
        success: bool,
        action: str = "applied",
        feedback_score: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ev = UsageEvent(
            rule_id=rule_id,
            action=action,
            success=success,
            feedback_score=feedback_score,
            context=context or {},
            timestamp=timestamp or datetime.utcnow(),
        )
        self._events.append(ev)

    def record_rule_modification(
        self,
        rule_id: str,
        description: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        change_type: str = "manual_modification",
        score_delta: float = 0.0,
    ) -> None:
        self._modifications.append(
            ChangeRecord(
                rule_id=rule_id,
                change_type=change_type,
                description=description,
                before=before,
                after=after,
                score_delta=score_delta,
            )
        )

    # ---- Accessors ----
    @property
    def events(self) -> List[UsageEvent]:
        return list(self._events)

    @property
    def modifications(self) -> List[ChangeRecord]:
        return list(self._modifications)

    def clear(self) -> None:
        self._events.clear()
        self._modifications.clear()

    # ---- Utilities ----
    def as_generated_rules(self, rule_map: Dict[str, Rule]) -> List[GeneratedRule]:
        by_rule: Dict[str, List[UsageEvent]] = {rid: [] for rid in rule_map.keys()}
        for ev in self._events:
            by_rule.setdefault(ev.rule_id, []).append(ev)

        out: List[GeneratedRule] = []
        for rid, rule in rule_map.items():
            out.append(GeneratedRule(rule=rule, usage_events=by_rule.get(rid, [])))
        return out

