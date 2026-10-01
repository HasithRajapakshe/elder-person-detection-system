import pytest

from src.perception.bed_context import (
    BedRegion,
    bbox_overlap_fraction,
    spatial_features,
)


def test_normalized_bed_coordinates():
    region = BedRegion(0.1, 0.2, 0.8, 0.9)

    assert region.pixels(1000, 500) == (
        100, 100, 800, 450
    )


def test_invalid_bed_rectangle():
    with pytest.raises(ValueError):
        BedRegion(0.8, 0.2, 0.1, 0.9)


def test_overlap_uses_person_area():
    assert bbox_overlap_fraction(
        (0, 0, 100, 100),
        (50, 0, 150, 100),
    ) == pytest.approx(0.5)


def test_degenerate_box_has_zero_overlap():
    assert bbox_overlap_fraction(
        (10, 10, 10, 20),
        (0, 0, 100, 100),
    ) == 0


def test_missing_pose_remains_unknown():
    features = spatial_features(
        (20, 20, 80, 80),
        None,
        None,
        (10, 10, 90, 90),
        100,
        100,
    )

    assert features["bed_relation"] == "overlapping_bed"
    assert features["anchor_source"] == "bbox_center"
    assert features["torso_angle_from_vertical_deg"] is None
    assert features["pose_quality"] == "insufficient_torso_keypoints"


def test_vertical_torso():
    points = [[0.0, 0.0] for _ in range(17)]
    confidence = [0.0] * 17

    points[5] = [40, 20]
    points[6] = [60, 20]
    points[11] = [40, 60]
    points[12] = [60, 60]

    for index in (5, 6, 11, 12):
        confidence[index] = 0.9

    features = spatial_features(
        (20, 10, 80, 90),
        points,
        confidence,
        (0, 0, 100, 100),
        100,
        100,
    )

    assert features["anchor_source"] == "hips"
    assert features["torso_angle_from_vertical_deg"] == pytest.approx(0)


def test_low_confidence_keypoints_are_not_used():
    points = [[50.0, 50.0] for _ in range(17)]
    confidence = [0.1] * 17

    features = spatial_features(
        (20, 20, 80, 80),
        points,
        confidence,
        (0, 0, 100, 100),
        100,
        100,
    )

    assert features["anchor_source"] == "bbox_center"
    assert features["torso_angle_from_vertical_deg"] is None


def test_person_far_from_bed():
    features = spatial_features(
        (700, 700, 900, 900),
        None,
        None,
        (0, 0, 200, 200),
        1000,
        1000,
    )

    assert features["bed_relation"] == "away_from_bed"