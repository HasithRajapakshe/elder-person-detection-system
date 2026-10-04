"""Temporal state smoother.

Prevents false transitions from single-frame label noise by using a
rolling window with confidence-weighted majority voting and hysteresis.

Critical design choice: the system must NOT emit state transitions
based on single-frame label changes. This module ensures that a
transition is confirmed only when sufficient temporal evidence
accumulates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from src.models import ActivityState, StatePrediction
from src.config import PipelineConfig


@dataclass
class SmoothedState:
    """Result of temporal smoothing for one observation."""
    raw_state: ActivityState
    raw_confidence: float
    smoothed_state: ActivityState
    smoothed_confidence: float
    transition_pending: bool = False
    pending_state: Optional[ActivityState] = None
    window_size: int = 0
    timestamp_sec: float = 0.0


@dataclass
class _WindowEntry:
    state: ActivityState
    confidence: float
    timestamp_sec: float


class TemporalSmoother:
    """Rolling-window temporal smoother with hysteresis.

    Maintains the current stable state and only transitions when a new
    state is observed consistently for a configurable confirmation
    period with sufficient confidence.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._window: Deque[_WindowEntry] = deque()
        self._current_state: ActivityState = ActivityState.UNKNOWN
        self._current_confidence: float = 0.0
        self._pending_state: Optional[ActivityState] = None
        self._pending_since: Optional[float] = None
        self._pending_count: int = 0

    @property
    def current_state(self) -> ActivityState:
        return self._current_state

    def update(self, prediction: StatePrediction, timestamp_sec: float) -> SmoothedState:
        """Process a new raw prediction and return the smoothed state."""
        entry = _WindowEntry(
            state=prediction.state,
            confidence=prediction.confidence,
            timestamp_sec=timestamp_sec,
        )
        self._window.append(entry)

        # Remove entries outside the smoothing window
        cutoff = timestamp_sec - self._config.smoothing_window_sec
        while self._window and self._window[0].timestamp_sec < cutoff:
            self._window.popleft()

        # Compute weighted vote across window
        vote_weights: dict[ActivityState, float] = {}
        vote_counts: dict[ActivityState, int] = {}
        for e in self._window:
            vote_weights[e.state] = vote_weights.get(e.state, 0.0) + e.confidence
            vote_counts[e.state] = vote_counts.get(e.state, 0) + 1

        # Best candidate in window
        if vote_weights:
            window_best = max(vote_weights, key=lambda s: vote_weights[s])
            total_weight = sum(vote_weights.values())
            window_confidence = (
                vote_weights[window_best] / total_weight if total_weight > 0 else 0.0
            )
            window_count = vote_counts.get(window_best, 0)
        else:
            window_best = prediction.state
            window_confidence = prediction.confidence
            window_count = 1

        transition_pending = False
        pending_state = None

        if window_best != self._current_state:
            # Check if this is a new pending transition
            if window_best != self._pending_state:
                self._pending_state = window_best
                self._pending_since = timestamp_sec
                self._pending_count = 1
            else:
                self._pending_count += 1

            # Check confirmation criteria
            duration_met = False
            if self._pending_since is not None:
                elapsed = timestamp_sec - self._pending_since
                duration_met = elapsed >= self._config.transition_confirmation_sec

            count_met = self._pending_count >= self._config.minimum_observations_for_transition

            # Hysteresis: the new state must exceed current by hysteresis margin
            current_weight = vote_weights.get(self._current_state, 0.0)
            new_weight = vote_weights.get(window_best, 0.0)
            hysteresis_met = new_weight > current_weight + self._config.hysteresis_factor * total_weight if total_weight > 0 else True

            if duration_met and count_met and hysteresis_met:
                # Confirm transition
                self._current_state = window_best
                self._current_confidence = window_confidence
                self._pending_state = None
                self._pending_since = None
                self._pending_count = 0
            else:
                transition_pending = True
                pending_state = window_best
        else:
            # Same state — clear pending
            self._pending_state = None
            self._pending_since = None
            self._pending_count = 0
            self._current_confidence = window_confidence

        return SmoothedState(
            raw_state=prediction.state,
            raw_confidence=prediction.confidence,
            smoothed_state=self._current_state,
            smoothed_confidence=self._current_confidence,
            transition_pending=transition_pending,
            pending_state=pending_state,
            window_size=len(self._window),
            timestamp_sec=timestamp_sec,
        )

    def reset(self) -> None:
        """Reset smoother state."""
        self._window.clear()
        self._current_state = ActivityState.UNKNOWN
        self._current_confidence = 0.0
        self._pending_state = None
        self._pending_since = None
        self._pending_count = 0
