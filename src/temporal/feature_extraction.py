"""Explainable feature extraction layer.

Computes normalised, interpretable features from TemporalObservation
for use by the state classifier. Every feature is a simple float,
making the classification logic transparent and testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.models import TemporalObservation
from src.config import PipelineConfig


@dataclass
class ExtractedFeatures:
    """All features extracted from a single observation."""

    # --- Pose features ---
    torso_verticality: float = 0.0      # 1.0 = perfectly vertical, 0.0 = horizontal
    torso_angle_deg: Optional[float] = None
    body_horizontal_score: float = 0.0  # 1.0 = perfectly horizontal
    hip_knee_angle: Optional[float] = None
    knee_bend_score: float = 0.0        # 1.0 = fully bent (sitting), 0.0 = straight
    shoulder_hip_vertical_distance: float = 0.0  # normalised
    hip_ankle_vertical_distance: float = 0.0      # normalised
    pose_visibility: float = 0.0        # fraction of key torso points visible

    # --- Geometry features ---
    bbox_aspect_ratio: float = 0.0      # height / width
    bbox_area_ratio: float = 0.0        # bbox_area / frame_area
    bbox_center_x: float = 0.0          # normalised 0–1
    bbox_center_y: float = 0.0          # normalised 0–1

    # --- Bed relation features ---
    bbox_bed_iou: float = 0.0
    bbox_bed_overlap_ratio: float = 0.0
    center_inside_bed: bool = False
    hip_inside_bed: bool = False
    shoulders_inside_bed: bool = False
    distance_to_bed: float = 1.0        # normalised, 0 = on bed
    near_bed: bool = False
    on_bed: bool = False

    # --- Motion features ---
    centroid_displacement: float = 0.0  # normalised
    smoothed_speed: float = 0.0         # normalised per second
    movement_persistence: int = 0       # consecutive frames with motion
    stationary_duration: float = 0.0    # seconds stationary

    # --- Confidence ---
    person_confidence: float = 0.0
    pose_quality_score: float = 0.0     # 0–1

    # --- Bed context available ---
    bed_configured: bool = False


def _norm_dist(
    p1: Optional[Tuple[float, float]],
    p2: Optional[Tuple[float, float]],
    frame_height: int,
) -> float:
    """Vertical distance between two points, normalised by frame height."""
    if p1 is None or p2 is None or frame_height <= 0:
        return 0.0
    return abs(p1[1] - p2[1]) / frame_height


def _point_in_rect(
    point: Optional[Tuple[float, float]],
    rect: Optional[Tuple],
) -> bool:
    if point is None or rect is None:
        return False
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def extract_features(
    obs: TemporalObservation,
    config: PipelineConfig,
    movement_history: Optional[list] = None,
    stationary_since: Optional[float] = None,
) -> ExtractedFeatures:
    """Extract all features from a single observation.

    Args:
        obs: The temporal observation.
        config: Pipeline configuration.
        movement_history: List of recent displacement values for persistence.
        stationary_since: Timestamp when the person became stationary.
    """
    feat = ExtractedFeatures()

    # --- Person confidence ---
    feat.person_confidence = obs.person_confidence

    # --- Pose features ---
    feat.torso_angle_deg = obs.torso_angle_deg

    if obs.torso_angle_deg is not None:
        # verticality: 1 at 0°, 0 at 90°
        feat.torso_verticality = max(0.0, 1.0 - obs.torso_angle_deg / 90.0)
        feat.body_horizontal_score = max(0.0, obs.torso_angle_deg / 90.0)
    else:
        feat.torso_verticality = 0.5  # uncertain default
        feat.body_horizontal_score = 0.5

    # Shoulder-hip vertical distance (normalised)
    feat.shoulder_hip_vertical_distance = _norm_dist(
        obs.shoulder_mid, obs.hip_mid, obs.frame_height
    )

    # Hip-ankle vertical distance (normalised)
    ankle_mid = None
    if obs.ankle_left and obs.ankle_right:
        ankle_mid = (
            (obs.ankle_left[0] + obs.ankle_right[0]) / 2.0,
            (obs.ankle_left[1] + obs.ankle_right[1]) / 2.0,
        )
    elif obs.ankle_left:
        ankle_mid = obs.ankle_left
    elif obs.ankle_right:
        ankle_mid = obs.ankle_right
    feat.hip_ankle_vertical_distance = _norm_dist(
        obs.hip_mid, ankle_mid, obs.frame_height
    )

    # Knee bend score
    knee_mid = None
    if obs.knee_left and obs.knee_right:
        knee_mid = (
            (obs.knee_left[0] + obs.knee_right[0]) / 2.0,
            (obs.knee_left[1] + obs.knee_right[1]) / 2.0,
        )
    elif obs.knee_left:
        knee_mid = obs.knee_left
    elif obs.knee_right:
        knee_mid = obs.knee_right

    if obs.hip_mid and knee_mid and ankle_mid and obs.frame_height > 0:
        # Angle at knee: straight leg ~ 180°, bent ~ 90°
        hip_knee_dy = knee_mid[1] - obs.hip_mid[1]
        hip_knee_dx = knee_mid[0] - obs.hip_mid[0]
        knee_ankle_dy = ankle_mid[1] - knee_mid[1]
        knee_ankle_dx = ankle_mid[0] - knee_mid[0]
        a1 = math.atan2(hip_knee_dy, hip_knee_dx)
        a2 = math.atan2(knee_ankle_dy, knee_ankle_dx)
        angle = abs(math.degrees(a2 - a1))
        if angle > 180:
            angle = 360 - angle
        feat.hip_knee_angle = angle
        # Normalise: 180° = 0.0 (straight), 90° = 1.0 (bent)
        feat.knee_bend_score = max(0.0, min(1.0, (180.0 - angle) / 90.0))

    # Pose visibility (fraction of 4 torso keypoints visible)
    torso_kps = [obs.shoulder_left, obs.shoulder_right, obs.hip_left, obs.hip_right]
    feat.pose_visibility = sum(1 for p in torso_kps if p is not None) / 4.0

    pose_quality_map = {"good": 1.0, "partial": 0.6, "poor": 0.3, "missing": 0.0}
    feat.pose_quality_score = pose_quality_map.get(obs.pose_quality, 0.0)

    # --- Geometry features ---
    feat.bbox_aspect_ratio = obs.body_aspect_ratio if obs.body_aspect_ratio else 0.0
    feat.bbox_area_ratio = obs.bbox_area_ratio

    if obs.bbox_center and obs.frame_width > 0 and obs.frame_height > 0:
        feat.bbox_center_x = obs.bbox_center[0] / obs.frame_width
        feat.bbox_center_y = obs.bbox_center[1] / obs.frame_height

    # --- Bed relation features ---
    feat.bed_configured = obs.bed_relation != "not_configured"
    feat.bbox_bed_overlap_ratio = obs.bbox_bed_overlap
    feat.bbox_bed_iou = obs.bbox_bed_overlap  # using overlap fraction as proxy

    feat.center_inside_bed = _point_in_rect(obs.bbox_center, obs.bed_region)
    feat.hip_inside_bed = _point_in_rect(obs.hip_mid, obs.bed_region)
    feat.shoulders_inside_bed = _point_in_rect(obs.shoulder_mid, obs.bed_region)

    feat.distance_to_bed = obs.distance_to_bed if obs.distance_to_bed is not None else 1.0
    feat.near_bed = obs.near_bed
    feat.on_bed = obs.on_bed

    # --- Motion features ---
    feat.centroid_displacement = obs.motion_displacement
    feat.smoothed_speed = obs.motion_speed

    if movement_history is not None:
        count = 0
        for d in reversed(movement_history):
            if d > config.stationary_speed_threshold:
                count += 1
            else:
                break
        feat.movement_persistence = count

    if stationary_since is not None and obs.timestamp_sec >= stationary_since:
        feat.stationary_duration = obs.timestamp_sec - stationary_since

    return feat
