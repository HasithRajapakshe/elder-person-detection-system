import pytest

from src.models import ActivityState, StatePrediction, TimelineInterval
from src.config import PipelineConfig
from src.temporal.smoother import TemporalSmoother
from src.temporal.timeline import build_timeline
from src.temporal.bed_events import BedEventDetector, BedEventType


def test_smoother_ignores_single_frame_noise():
    config = PipelineConfig(
        smoothing_window_sec=3.0,
        transition_confirmation_sec=2.0,
        minimum_observations_for_transition=3,
        hysteresis_factor=0.0
    )
    smoother = TemporalSmoother(config)

    # Base state
    pred_lying = StatePrediction(ActivityState.LYING_IN_BED, 0.9)
    smoother.update(pred_lying, 0.0)
    smoother.update(pred_lying, 1.0)
    smoother.update(pred_lying, 2.0)
    smoother.update(pred_lying, 3.0)
    
    assert smoother.current_state == ActivityState.LYING_IN_BED
    
    # Noise (one frame)
    pred_sit = StatePrediction(ActivityState.SITTING_ON_BED, 1.0)
    
    # We clear the window by waiting a bit, but keep current state
    s = smoother.update(pred_sit, 6.0)
    
    # Should not transition yet because it needs 2 seconds of confirmation
    assert s.smoothed_state == ActivityState.LYING_IN_BED
    assert s.transition_pending is True

    # Back to lying immediately after
    smoother.update(pred_lying, 6.5)
    s = smoother.update(pred_lying, 7.0)
    assert s.smoothed_state == ActivityState.LYING_IN_BED
    assert s.transition_pending is False


def test_smoother_transitions_after_confirmation():
    config = PipelineConfig(
        smoothing_window_sec=1.0,
        transition_confirmation_sec=1.0, # fast confirm
        minimum_observations_for_transition=2,
        hysteresis_factor=0.0
    )
    smoother = TemporalSmoother(config)

    # Start lying
    pred_lying = StatePrediction(ActivityState.LYING_IN_BED, 0.9)
    smoother.update(pred_lying, 0.0)
    smoother.update(pred_lying, 1.0)
    smoother.update(pred_lying, 2.0)
    
    # Transition to sitting, clear window of lying
    pred_sit = StatePrediction(ActivityState.SITTING_ON_BED, 0.8)
    smoother.update(pred_sit, 5.0) # wait to clear window
    smoother.update(pred_sit, 5.5)
    
    # At 6.0, duration is 1.0 sec (from 5.0 to 6.0)
    s = smoother.update(pred_sit, 6.0)
    
    assert s.smoothed_state == ActivityState.SITTING_ON_BED


def test_bed_exit_requires_temporal_confirmation():
    config = PipelineConfig(bed_exit_confirmation_sec=2.0)
    detector = BedEventDetector(config)

    # Start in bed
    detector.update(ActivityState.LYING_IN_BED, 0.0)
    detector.update(ActivityState.SITTING_ON_BED, 1.0)
    
    # Brief out of bed (e.g. noise or standing up and immediately sitting down)
    event = detector.update(ActivityState.STANDING, 2.0)
    assert event is None
    
    event = detector.update(ActivityState.SITTING_ON_BED, 3.0)
    assert event is None # Cancelled exit
    
    # Real exit
    event = detector.update(ActivityState.STANDING, 4.0)
    assert event is None # Not confirmed yet
    
    event = detector.update(ActivityState.WALKING, 5.0)
    assert event is None
    
    # Confirmed
    event = detector.update(ActivityState.OUT_OF_BED, 6.0)
    assert event is not None
    assert event.event_type == BedEventType.BED_EXIT
    assert detector.exit_count == 1
    
    # Duplicate exit suppression
    event = detector.update(ActivityState.WALKING, 7.0)
    assert event is None
    assert detector.exit_count == 1


def test_timeline_builder():
    states = [
        ActivityState.LYING_IN_BED,
        ActivityState.LYING_IN_BED,
        ActivityState.SITTING_ON_BED,
        ActivityState.SITTING_ON_BED,
        ActivityState.SITTING_ON_BED,
    ]
    confs = [0.9, 0.9, 0.8, 0.8, 0.8]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    timeline = build_timeline(states, confs, times)
    
    assert len(timeline) == 2
    assert timeline[0].state == ActivityState.LYING_IN_BED
    assert timeline[0].start_sec == 0.0
    assert timeline[0].end_sec == 2.0
    assert timeline[0].duration_sec == 2.0
    
    assert timeline[1].state == ActivityState.SITTING_ON_BED
    assert timeline[1].start_sec == 2.0
    assert timeline[1].end_sec == 4.0
    assert timeline[1].duration_sec == 2.0
