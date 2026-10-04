"""Typed configuration for the temporal reasoning pipeline.

All thresholds live here rather than being scattered as magic numbers.
Defaults are sensible for a typical indoor elderly-care camera but
should be calibrated per camera/video environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Complete configuration for the analysis pipeline."""

    # --- State classification ---
    state_confidence_threshold: float = 0.45
    unknown_confidence_threshold: float = 0.35

    # --- Temporal smoothing ---
    smoothing_window_sec: float = 3.0
    transition_confirmation_sec: float = 2.0
    minimum_observations_for_transition: int = 3
    hysteresis_factor: float = 0.15  # extra confidence needed to leave current state

    # --- Motion thresholds (normalised by frame diagonal) ---
    walking_speed_threshold: float = 0.008   # normalised speed per second
    stationary_speed_threshold: float = 0.002
    walking_persistence_sec: float = 1.5     # minimum duration to confirm WALKING
    movement_persistence_frames: int = 3

    # --- Pose / geometry ---
    torso_vertical_threshold_deg: float = 35.0  # below = upright
    torso_horizontal_threshold_deg: float = 55.0 # above = horizontal/lying
    sitting_torso_max_deg: float = 50.0
    standing_aspect_ratio_min: float = 1.2  # bbox height/width
    lying_aspect_ratio_max: float = 1.0

    # --- Bed relation ---
    bed_overlap_threshold: float = 0.30
    near_bed_distance_threshold: float = 0.05  # normalised
    on_bed_overlap_threshold: float = 0.20
    bed_context_enabled: bool = True

    # --- Bed events ---
    bed_exit_confirmation_sec: float = 5.0
    bed_return_confirmation_sec: float = 4.0
    bed_exit_movement_threshold: float = 0.03  # normalised distance from bed

    # --- Alert thresholds ---
    prolonged_out_of_bed_sec: float = 60.0
    prolonged_unknown_sec: float = 30.0
    long_sitting_on_bed_edge_sec: float = 45.0
    person_missing_alert_sec: float = 30.0

    # --- Tracking / gaps ---
    tracking_gap_tolerance_sec: float = 3.0
    max_gap_retain_state_sec: float = 5.0

    # --- Keypoint confidence ---
    keypoint_confidence_threshold: float = 0.40

    # --- Sampling ---
    sample_fps: float = 2.0

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create config from a flat or nested dictionary.

        Accepts the settings.yaml structure or a flat dict of overrides.
        """
        cfg = cls()

        # Support nested 'temporal' or 'pipeline' section
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        for attr_name in vars(cfg):
            if attr_name in flat and flat[attr_name] is not None:
                expected_type = type(getattr(cfg, attr_name))
                try:
                    setattr(cfg, attr_name, expected_type(flat[attr_name]))
                except (ValueError, TypeError):
                    pass  # keep default

        return cfg
