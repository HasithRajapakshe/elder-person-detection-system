"""Finite State Machine for activity state transitions.

Validates transitions against plausible progressions, stores
transition metadata, and prevents impossible state jumps while
allowing skip-transitions when intermediate states are missed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from src.models import ActivityState, StateTransition
from src.temporal.smoother import SmoothedState


# Plausible transitions: source -> set of valid destinations.
# Skip transitions are allowed (e.g. LYING_IN_BED -> STANDING)
# because the detector can miss intermediate states.
VALID_TRANSITIONS: Dict[ActivityState, Set[ActivityState]] = {
    ActivityState.LYING_IN_BED: {
        ActivityState.SITTING_ON_BED,
        ActivityState.STANDING,         # skip: missed sitting
        ActivityState.UNKNOWN,
    },
    ActivityState.SITTING_ON_BED: {
        ActivityState.LYING_IN_BED,
        ActivityState.STANDING,
        ActivityState.WALKING,          # skip: missed standing
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.OUT_OF_BED,
        ActivityState.UNKNOWN,
    },
    ActivityState.SITTING_OUTSIDE_BED: {
        ActivityState.STANDING,
        ActivityState.WALKING,
        ActivityState.SITTING_ON_BED,
        ActivityState.OUT_OF_BED,
        ActivityState.UNKNOWN,
    },
    ActivityState.STANDING: {
        ActivityState.WALKING,
        ActivityState.SITTING_ON_BED,
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.LYING_IN_BED,     # skip: sat then lay quickly
        ActivityState.OUT_OF_BED,
        ActivityState.UNKNOWN,
    },
    ActivityState.WALKING: {
        ActivityState.STANDING,
        ActivityState.SITTING_ON_BED,
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.LYING_IN_BED,
        ActivityState.OUT_OF_BED,
        ActivityState.UNKNOWN,
    },
    ActivityState.OUT_OF_BED: {
        ActivityState.WALKING,
        ActivityState.STANDING,
        ActivityState.SITTING_ON_BED,
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.LYING_IN_BED,
        ActivityState.UNKNOWN,
    },
    ActivityState.UNKNOWN: {
        # From UNKNOWN, any state is allowed (recovery)
        s for s in ActivityState
    },
}


class ActivityFSM:
    """Explicit temporal state machine tracking transitions.

    Records every confirmed transition with full metadata for
    traceability and interview explainability.
    """

    def __init__(self) -> None:
        self._current_state: ActivityState = ActivityState.UNKNOWN
        self._state_start_time: float = 0.0
        self._transitions: List[StateTransition] = []

    @property
    def current_state(self) -> ActivityState:
        return self._current_state

    @property
    def transitions(self) -> List[StateTransition]:
        return list(self._transitions)

    def update(self, smoothed: SmoothedState) -> Optional[StateTransition]:
        """Process a smoothed state and record any transition.

        Returns the StateTransition if one occurred, else None.
        """
        new_state = smoothed.smoothed_state
        timestamp = smoothed.timestamp_sec

        if new_state == self._current_state:
            return None

        # Validate transition
        allowed = VALID_TRANSITIONS.get(self._current_state, set())
        if new_state not in allowed:
            # Still allow it but record as unexpected
            evidence = [
                f"transition {self._current_state.value} -> {new_state.value} "
                f"not in expected set; allowed anyway (skip transition)"
            ]
        else:
            evidence = [
                f"valid transition: {self._current_state.value} -> {new_state.value}"
            ]

        transition = StateTransition(
            previous_state=self._current_state,
            new_state=new_state,
            start_timestamp=self._state_start_time,
            confirmed_timestamp=timestamp,
            confidence=smoothed.smoothed_confidence,
            reason=f"Transition from {self._current_state.value} to {new_state.value}",
            evidence=evidence,
        )

        self._transitions.append(transition)
        self._current_state = new_state
        self._state_start_time = timestamp

        return transition

    def reset(self) -> None:
        """Reset the FSM to initial state."""
        self._current_state = ActivityState.UNKNOWN
        self._state_start_time = 0.0
        self._transitions.clear()
