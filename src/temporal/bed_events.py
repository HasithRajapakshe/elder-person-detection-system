"""Bed exit and return detector.

Implements temporal evidence requirements for bed events:
- A bed exit is NOT simply sitting up or brief pose fluctuation
- A bed return requires persistent re-entry into bed
- No duplicate events within the same episode
"""
from __future__ import annotations

from typing import List, Optional

from src.models import (
    ActivityState,
    AlertLevel,
    BedEvent,
    BedEventType,
    StateTransition,
)
from src.config import PipelineConfig


# States considered "in bed"
IN_BED_STATES = {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}

# States considered "out of bed"
OUT_OF_BED_STATES = {
    ActivityState.STANDING,
    ActivityState.WALKING,
    ActivityState.OUT_OF_BED,
    ActivityState.SITTING_OUTSIDE_BED,
}


class BedEventDetector:
    """Detects bed exits and returns with temporal confirmation.

    Bed exit requires:
    1. Person previously confirmed in bed (LYING or SITTING_ON_BED)
    2. Transition to out-of-bed state
    3. Persistence in out-of-bed state for confirmation period

    Bed return requires:
    1. Person currently out of bed
    2. Re-entry into bed region
    3. Persistence in in-bed state for confirmation period
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._events: List[BedEvent] = []

        # Tracking state for event detection
        self._was_in_bed: bool = False
        self._is_out_of_bed: bool = False
        self._exit_pending: bool = False
        self._return_pending: bool = False
        self._exit_start_time: Optional[float] = None
        self._return_start_time: Optional[float] = None
        self._exit_previous_state: Optional[ActivityState] = None
        self._last_in_bed_state: Optional[ActivityState] = None

        # Episode tracking to prevent duplicates
        self._active_exit_episode: bool = False

    @property
    def events(self) -> List[BedEvent]:
        return list(self._events)

    @property
    def exit_count(self) -> int:
        return sum(1 for e in self._events if e.event_type == BedEventType.BED_EXIT)

    @property
    def return_count(self) -> int:
        return sum(1 for e in self._events if e.event_type == BedEventType.BED_RETURN)

    def update(
        self,
        current_state: ActivityState,
        timestamp: float,
        transition: Optional[StateTransition] = None,
    ) -> Optional[BedEvent]:
        """Process a state update and return a bed event if one is confirmed.

        Returns BedEvent if a new event is confirmed, else None.
        """
        is_in_bed = current_state in IN_BED_STATES
        is_out = current_state in OUT_OF_BED_STATES

        event = None

        # --- BED EXIT DETECTION ---
        if self._was_in_bed and is_out and not self._active_exit_episode:
            if not self._exit_pending:
                # Start exit confirmation period
                self._exit_pending = True
                self._exit_start_time = timestamp
                self._exit_previous_state = self._last_in_bed_state or ActivityState.LYING_IN_BED
            else:
                # Check if confirmation period has elapsed
                elapsed = timestamp - (self._exit_start_time or timestamp)
                if elapsed >= self._config.bed_exit_confirmation_sec:
                    event = BedEvent(
                        event_type=BedEventType.BED_EXIT,
                        start_time=self._exit_start_time or timestamp,
                        confirmed_time=timestamp,
                        previous_state=self._exit_previous_state or ActivityState.LYING_IN_BED,
                        current_state=current_state,
                        confidence=0.85,
                        decision=AlertLevel.MONITOR,
                        evidence=[
                            f"person was in bed ({self._exit_previous_state})",
                            f"transitioned to {current_state.value}",
                            f"persisted out of bed for {elapsed:.1f}s",
                            f"exit confirmed at {timestamp:.1f}s",
                        ],
                    )
                    self._events.append(event)
                    self._active_exit_episode = True
                    self._exit_pending = False
                    self._was_in_bed = False
                    self._is_out_of_bed = True

        # Cancel exit if person returns to bed before confirmation
        if self._exit_pending and is_in_bed:
            self._exit_pending = False
            self._exit_start_time = None

        # --- BED RETURN DETECTION ---
        if self._active_exit_episode and is_in_bed:
            if not self._return_pending:
                self._return_pending = True
                self._return_start_time = timestamp
            else:
                elapsed = timestamp - (self._return_start_time or timestamp)
                if elapsed >= self._config.bed_return_confirmation_sec:
                    event = BedEvent(
                        event_type=BedEventType.BED_RETURN,
                        start_time=self._return_start_time or timestamp,
                        confirmed_time=timestamp,
                        previous_state=ActivityState.OUT_OF_BED,
                        current_state=current_state,
                        confidence=0.85,
                        decision=AlertLevel.NORMAL,
                        evidence=[
                            "person was out of bed",
                            f"returned to {current_state.value}",
                            f"persisted in bed for {elapsed:.1f}s",
                            f"return confirmed at {timestamp:.1f}s",
                        ],
                    )
                    self._events.append(event)
                    self._active_exit_episode = False
                    self._return_pending = False
                    self._is_out_of_bed = False
                    self._was_in_bed = True

        # Cancel return if person leaves bed again
        if self._return_pending and is_out:
            self._return_pending = False
            self._return_start_time = None

        # Track bed status
        if is_in_bed:
            self._was_in_bed = True
            self._last_in_bed_state = current_state

        # UNKNOWN doesn't change bed tracking
        if current_state == ActivityState.UNKNOWN:
            pass  # retain current tracking state

        return event

    def reset(self) -> None:
        """Reset detector state."""
        self._events.clear()
        self._was_in_bed = False
        self._is_out_of_bed = False
        self._exit_pending = False
        self._return_pending = False
        self._active_exit_episode = False
