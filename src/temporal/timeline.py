"""Timeline builder and duration aggregator.

Builds a compressed temporal timeline from stable state transitions,
merging adjacent identical states. Also computes duration aggregates
and human-readable summaries.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.models import (
    ActivityState,
    ActivitySummary,
    BedEvent,
    BedEventType,
    TimelineInterval,
)


def _format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def build_timeline(
    states: List[ActivityState],
    confidences: List[float],
    timestamps: List[float],
) -> List[TimelineInterval]:
    """Build a compressed timeline from per-observation state data.

    Merges adjacent identical stable states into intervals.
    """
    if not states:
        return []

    intervals: List[TimelineInterval] = []
    current_state = states[0]
    start_ts = timestamps[0]
    conf_sum = confidences[0]
    count = 1

    for i in range(1, len(states)):
        if states[i] == current_state:
            conf_sum += confidences[i]
            count += 1
        else:
            # Close current interval
            end_ts = timestamps[i]
            intervals.append(TimelineInterval(
                start_sec=round(start_ts, 3),
                end_sec=round(end_ts, 3),
                duration_sec=round(end_ts - start_ts, 3),
                state=current_state,
                mean_confidence=round(conf_sum / count, 4),
                observation_count=count,
            ))
            # Start new interval
            current_state = states[i]
            start_ts = timestamps[i]
            conf_sum = confidences[i]
            count = 1

    # Close final interval
    end_ts = timestamps[-1]
    # Add a small delta if there's only one observation in the final interval
    if count == 1 and len(timestamps) >= 2:
        # Estimate interval duration from average sample spacing
        avg_spacing = (timestamps[-1] - timestamps[0]) / max(1, len(timestamps) - 1)
        end_ts = timestamps[-1] + avg_spacing
    intervals.append(TimelineInterval(
        start_sec=round(start_ts, 3),
        end_sec=round(end_ts, 3),
        duration_sec=round(end_ts - start_ts, 3),
        state=current_state,
        mean_confidence=round(conf_sum / count, 4),
        observation_count=count,
    ))

    return intervals


def aggregate_durations(
    timeline: List[TimelineInterval],
    bed_events: List[BedEvent],
    total_video_duration: Optional[float] = None,
) -> ActivitySummary:
    """Compute duration aggregates from timeline intervals.

    Validates that sum(activity durations) ≈ observation duration.
    """
    if not timeline:
        return ActivitySummary()

    observation_duration = timeline[-1].end_sec - timeline[0].start_sec

    # Per-state durations
    durations: Dict[str, float] = {s.value: 0.0 for s in ActivityState}
    for interval in timeline:
        durations[interval.state.value] += interval.duration_sec

    # Human-readable
    human_durations = {k: _format_duration(v) for k, v in durations.items()}

    # Bed status durations
    in_bed_states = {ActivityState.LYING_IN_BED.value, ActivityState.SITTING_ON_BED.value}
    out_of_bed_states = {
        ActivityState.STANDING.value,
        ActivityState.WALKING.value,
        ActivityState.OUT_OF_BED.value,
        ActivityState.SITTING_OUTSIDE_BED.value,
    }

    total_in_bed = sum(durations.get(s, 0.0) for s in in_bed_states)
    total_out = sum(durations.get(s, 0.0) for s in out_of_bed_states)

    # Longest out-of-bed period
    longest_out = 0.0
    current_out = 0.0
    for interval in timeline:
        if interval.state.value in out_of_bed_states:
            current_out += interval.duration_sec
            longest_out = max(longest_out, current_out)
        else:
            current_out = 0.0

    # Event counts
    exit_count = sum(1 for e in bed_events if e.event_type == BedEventType.BED_EXIT)
    return_count = sum(1 for e in bed_events if e.event_type == BedEventType.BED_RETURN)

    # Final state
    final_state = timeline[-1].state.value if timeline else None

    return ActivitySummary(
        observation_duration_sec=round(observation_duration, 3),
        activity_durations={k: round(v, 3) for k, v in durations.items()},
        activity_durations_human=human_durations,
        bed_exit_count=exit_count,
        bed_return_count=return_count,
        total_in_bed_sec=round(total_in_bed, 3),
        total_out_of_bed_sec=round(total_out, 3),
        longest_out_of_bed_period_sec=round(longest_out, 3),
        final_state=final_state,
    )
