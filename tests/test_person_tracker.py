import pytest

from src.perception.person_tracker import TrackObservation, choose_primary_track


def test_primary_track_selection():
    observations = [
        # Track 1: Present in 5 frames, large area, high confidence
        TrackObservation(frame_index=0, timestamp_sec=0.0, track_id=1, x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        TrackObservation(frame_index=1, timestamp_sec=0.1, track_id=1, x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        TrackObservation(frame_index=2, timestamp_sec=0.2, track_id=1, x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        TrackObservation(frame_index=3, timestamp_sec=0.3, track_id=1, x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        TrackObservation(frame_index=4, timestamp_sec=0.4, track_id=1, x1=0, y1=0, x2=100, y2=100, confidence=0.9),

        # Track 2: Present in only 1 frame, small area, low confidence (noise)
        TrackObservation(frame_index=2, timestamp_sec=0.2, track_id=2, x1=10, y1=10, x2=20, y2=20, confidence=0.4),
    ]
    
    primary_id, stats = choose_primary_track(observations, frame_width=100, frame_height=100)
    
    assert primary_id == 1
    assert 1 in stats
    assert 2 in stats
    assert stats[1]["score"] > stats[2]["score"]
    assert stats[1]["presence_ratio"] == 1.0


def test_empty_detections():
    primary_id, stats = choose_primary_track([], frame_width=100, frame_height=100)
    assert primary_id is None
    assert stats == {}


def test_multiple_tracks_scoring():
    # Synthetic tracks to test scoring weight deterministically
    observations = [
        # Track 1: 5 frames, small area (10x10 = 100), high conf (0.9)
        TrackObservation(0, 0.0, 1, 0, 0, 10, 10, 0.9),
        TrackObservation(1, 0.1, 1, 0, 0, 10, 10, 0.9),
        TrackObservation(2, 0.2, 1, 0, 0, 10, 10, 0.9),
        TrackObservation(3, 0.3, 1, 0, 0, 10, 10, 0.9),
        TrackObservation(4, 0.4, 1, 0, 0, 10, 10, 0.9),
        
        # Track 2: 3 frames, large area (50x50 = 2500), high conf (0.9)
        TrackObservation(0, 0.0, 2, 0, 0, 50, 50, 0.9),
        TrackObservation(1, 0.1, 2, 0, 0, 50, 50, 0.9),
        TrackObservation(2, 0.2, 2, 0, 0, 50, 50, 0.9),
    ]
    
    primary_id, stats = choose_primary_track(observations, frame_width=100, frame_height=100)
    
    # Track 1: presence = 5/5 = 1.0, area = 100/10000 = 0.01, conf = 0.9
    # Score 1: 0.7*1.0 + 0.2*min(1.0, 0.04) + 0.1*0.9 = 0.7 + 0.008 + 0.09 = 0.798
    
    # Track 2: presence = 3/5 = 0.6, area = 2500/10000 = 0.25, conf = 0.9
    # Score 2: 0.7*0.6 + 0.2*min(1.0, 1.0) + 0.1*0.9 = 0.42 + 0.2 + 0.09 = 0.71
    
    assert primary_id == 1
    assert abs(stats[1]["score"] - 0.798) < 1e-4
    assert abs(stats[2]["score"] - 0.71) < 1e-4
