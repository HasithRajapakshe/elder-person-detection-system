"""Contextual alert engine.

Assigns NORMAL / MONITOR / ALERT based on configurable rules.
Each decision includes the triggering rule, reason, and confidence.
"""
from __future__ import annotations

from typing import List, Optional

from src.models import (
    ActivityState,
    AlertEvent,
    AlertLevel,
    BedEvent,
    BedEventType,
)
from src.config import PipelineConfig


class AlertEngine:
    """Evaluates current state and events to produce safety alerts.

    Rules are evaluated in priority order. The highest-priority
    matching rule determines the alert level.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._alerts: List[AlertEvent] = []

        # Tracking for duration-based alerts
        self._out_of_bed_since: Optional[float] = None
        self._unknown_since: Optional[float] = None
        self._sitting_edge_since: Optional[float] = None
        self._missing_since: Optional[float] = None

    @property
    def alerts(self) -> List[AlertEvent]:
        return list(self._alerts)

    def evaluate(
        self,
        state: ActivityState,
        timestamp: float,
        bed_event: Optional[BedEvent] = None,
        person_detected: bool = True,
    ) -> AlertEvent:
        """Evaluate the current situation and return an alert event.

        Always returns exactly one AlertEvent (NORMAL, MONITOR, or ALERT).
        """
        # --- Track durations ---
        if state in (
            ActivityState.STANDING,
            ActivityState.WALKING,
            ActivityState.OUT_OF_BED,
            ActivityState.SITTING_OUTSIDE_BED,
        ):
            if self._out_of_bed_since is None:
                self._out_of_bed_since = timestamp
        else:
            self._out_of_bed_since = None

        if state == ActivityState.UNKNOWN:
            if self._unknown_since is None:
                self._unknown_since = timestamp
        else:
            self._unknown_since = None

        if state == ActivityState.SITTING_ON_BED:
            if self._sitting_edge_since is None:
                self._sitting_edge_since = timestamp
        else:
            self._sitting_edge_since = None

        if not person_detected:
            if self._missing_since is None:
                self._missing_since = timestamp
        else:
            self._missing_since = None

        # --- ALERT rules (highest priority) ---

        # Rule: prolonged out of bed
        if self._out_of_bed_since is not None:
            elapsed = timestamp - self._out_of_bed_since
            if elapsed >= self._config.prolonged_out_of_bed_sec:
                alert = AlertEvent(
                    timestamp=timestamp,
                    level=AlertLevel.ALERT,
                    rule="prolonged_out_of_bed",
                    reason=(
                        f"Person out of bed for {elapsed:.0f}s "
                        f"(threshold: {self._config.prolonged_out_of_bed_sec:.0f}s)"
                    ),
                    confidence=0.85,
                    state=state,
                )
                self._alerts.append(alert)
                return alert

        # Rule: person missing after bed exit
        if self._missing_since is not None:
            elapsed = timestamp - self._missing_since
            if elapsed >= self._config.person_missing_alert_sec:
                alert = AlertEvent(
                    timestamp=timestamp,
                    level=AlertLevel.ALERT,
                    rule="person_missing_prolonged",
                    reason=(
                        f"Person not detected for {elapsed:.0f}s "
                        f"(threshold: {self._config.person_missing_alert_sec:.0f}s)"
                    ),
                    confidence=0.75,
                    state=state,
                )
                self._alerts.append(alert)
                return alert

        # --- MONITOR rules ---

        # Rule: recently confirmed bed exit
        if bed_event and bed_event.event_type == BedEventType.BED_EXIT:
            alert = AlertEvent(
                timestamp=timestamp,
                level=AlertLevel.MONITOR,
                rule="bed_exit_detected",
                reason="Bed exit recently confirmed",
                confidence=bed_event.confidence,
                state=state,
                trigger_event="BED_EXIT",
            )
            self._alerts.append(alert)
            return alert

        # Rule: prolonged unknown
        if self._unknown_since is not None:
            elapsed = timestamp - self._unknown_since
            if elapsed >= self._config.prolonged_unknown_sec:
                alert = AlertEvent(
                    timestamp=timestamp,
                    level=AlertLevel.MONITOR,
                    rule="prolonged_unknown",
                    reason=(
                        f"Ambiguous state persists for {elapsed:.0f}s"
                    ),
                    confidence=0.60,
                    state=state,
                )
                self._alerts.append(alert)
                return alert

        # Rule: prolonged sitting on bed edge
        if self._sitting_edge_since is not None:
            elapsed = timestamp - self._sitting_edge_since
            if elapsed >= self._config.long_sitting_on_bed_edge_sec:
                alert = AlertEvent(
                    timestamp=timestamp,
                    level=AlertLevel.MONITOR,
                    rule="prolonged_sitting_bed_edge",
                    reason=(
                        f"Sitting on bed edge for {elapsed:.0f}s"
                    ),
                    confidence=0.65,
                    state=state,
                )
                self._alerts.append(alert)
                return alert

        # Rule: out of bed (not yet prolonged but active)
        if self._out_of_bed_since is not None:
            elapsed = timestamp - self._out_of_bed_since
            if elapsed >= self._config.bed_exit_confirmation_sec:
                alert = AlertEvent(
                    timestamp=timestamp,
                    level=AlertLevel.MONITOR,
                    rule="out_of_bed_active",
                    reason=(
                        f"Person out of bed for {elapsed:.0f}s"
                    ),
                    confidence=0.60,
                    state=state,
                )
                self._alerts.append(alert)
                return alert

        # --- NORMAL ---

        # Bed return
        if bed_event and bed_event.event_type == BedEventType.BED_RETURN:
            alert = AlertEvent(
                timestamp=timestamp,
                level=AlertLevel.NORMAL,
                rule="bed_return_detected",
                reason="Successful bed return",
                confidence=bed_event.confidence,
                state=state,
                trigger_event="BED_RETURN",
            )
            self._alerts.append(alert)
            return alert

        # Normal lying / sitting / standing / walking
        rule_map = {
            ActivityState.LYING_IN_BED: ("normal_lying", "Normal lying in bed"),
            ActivityState.SITTING_ON_BED: ("normal_sitting_bed", "Normal sitting on bed"),
            ActivityState.STANDING: ("normal_standing", "Normal standing"),
            ActivityState.WALKING: ("normal_walking", "Normal walking"),
            ActivityState.SITTING_OUTSIDE_BED: ("normal_sitting", "Normal sitting"),
        }

        rule, reason = rule_map.get(
            state, ("normal_activity", "Normal activity")
        )

        alert = AlertEvent(
            timestamp=timestamp,
            level=AlertLevel.NORMAL,
            rule=rule,
            reason=reason,
            confidence=0.90,
            state=state,
        )
        self._alerts.append(alert)
        return alert

    def reset(self) -> None:
        self._alerts.clear()
        self._out_of_bed_since = None
        self._unknown_since = None
        self._sitting_edge_since = None
        self._missing_since = None
