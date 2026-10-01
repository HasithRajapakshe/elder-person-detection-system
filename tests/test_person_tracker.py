from src.perception.person_tracker import (
    TrackObservation,
    choose_primary_track,
)


def create_observation(
    frame,
    track_id,
    box,
    confidence=0.90,
):
    x1, y1, x2, y2 = box

    return TrackObservation(
        frame_index=frame,
        timestamp_sec=frame / 30.0,
        track_id=track_id,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
    )


def test_primary_track_prefers_persistent_person():

    observations = []

    # Elderly/monitored person:
    # present throughout the video.
    for frame in range(10):

        observations.append(
            create_observation(
                frame,
                1,
                (10, 10, 60, 110),
            )
        )

    # Example caregiver:
    # larger bounding box but only
    # briefly enters the scene.
    for frame in range(7, 10):

        observations.append(
            create_observation(
                frame,
                2,
                (0, 0, 100, 150),
                0.95,
            )
        )

    primary_track_id, stats = (
        choose_primary_track(
            observations,
            200,
            200,
        )
    )

    assert primary_track_id == 1

    assert (
        stats[1]["presence_ratio"]
        >
        stats[2]["presence_ratio"]
    )


def test_no_observations_returns_none():

    primary_track_id, stats = (
        choose_primary_track(
            [],
            640,
            480,
        )
    )

    assert primary_track_id is None
    assert stats == {}