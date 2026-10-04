"""Explainable rule-based state classifier.

Uses multi-evidence scoring rather than a single fragile threshold.
Each state has a score computed from multiple features; the state with
the highest score wins, subject to a minimum confidence threshold.

Returns a StatePrediction with human-readable reasons for interview
explainability.
"""
from __future__ import annotations

from typing import List

from src.models import ActivityState, BedStatus, StatePrediction
from src.config import PipelineConfig
from src.temporal.feature_extraction import ExtractedFeatures


def classify_state(
    features: ExtractedFeatures,
    config: PipelineConfig,
) -> StatePrediction:
    """Classify the activity state from extracted features.

    This is intentionally rule-based because the assignment does not
    require training a new model. Each rule contributes evidence
    (positive or negative) towards a state, and the highest-scoring
    state is selected.
    """
    scores: dict[str, float] = {s.value: 0.0 for s in ActivityState}
    reasons: dict[str, list[str]] = {s.value: [] for s in ActivityState}

    # Shorthand
    f = features
    has_bed = f.bed_configured
    has_pose = f.pose_visibility >= 0.5

    # ===================================================================
    # LYING_IN_BED
    # ===================================================================
    if has_bed and f.on_bed:
        scores["LYING_IN_BED"] += 0.25
        reasons["LYING_IN_BED"].append("person is on bed")
    if has_bed and f.bbox_bed_overlap_ratio >= config.bed_overlap_threshold:
        scores["LYING_IN_BED"] += 0.20
        reasons["LYING_IN_BED"].append(
            f"bed overlap {f.bbox_bed_overlap_ratio:.2f} >= threshold"
        )
    if f.body_horizontal_score >= 0.6:
        scores["LYING_IN_BED"] += 0.20
        reasons["LYING_IN_BED"].append("torso is horizontal / near horizontal")
    if f.bbox_aspect_ratio > 0 and f.bbox_aspect_ratio < config.lying_aspect_ratio_max:
        scores["LYING_IN_BED"] += 0.15
        reasons["LYING_IN_BED"].append(
            f"bbox aspect ratio {f.bbox_aspect_ratio:.2f} indicates horizontal body"
        )
    if f.smoothed_speed < config.stationary_speed_threshold:
        scores["LYING_IN_BED"] += 0.10
        reasons["LYING_IN_BED"].append("very low movement")
    if has_bed and f.hip_inside_bed:
        scores["LYING_IN_BED"] += 0.10
        reasons["LYING_IN_BED"].append("hips inside bed region")

    # ===================================================================
    # SITTING_ON_BED
    # ===================================================================
    if has_bed and (f.on_bed or f.near_bed):
        scores["SITTING_ON_BED"] += 0.20
        reasons["SITTING_ON_BED"].append("person on or near bed")
    if 0.3 <= f.torso_verticality <= 0.85:
        scores["SITTING_ON_BED"] += 0.15
        reasons["SITTING_ON_BED"].append("torso approximately upright (sitting)")
    elif f.torso_verticality > 0.85:
        # Very upright suggests standing more than sitting
        scores["SITTING_ON_BED"] += 0.05
    if has_bed and f.hip_inside_bed:
        scores["SITTING_ON_BED"] += 0.15
        reasons["SITTING_ON_BED"].append("hips positioned in/on bed")
    if f.knee_bend_score >= 0.3:
        scores["SITTING_ON_BED"] += 0.10
        reasons["SITTING_ON_BED"].append("knee geometry compatible with sitting")
    if f.smoothed_speed < config.stationary_speed_threshold * 2:
        scores["SITTING_ON_BED"] += 0.05
        reasons["SITTING_ON_BED"].append("limited movement")
    if f.bbox_aspect_ratio >= 0.8 and f.bbox_aspect_ratio <= 1.8:
        scores["SITTING_ON_BED"] += 0.05
        reasons["SITTING_ON_BED"].append("bbox shape compatible with sitting")
    # Penalty if strongly horizontal
    if f.body_horizontal_score >= 0.7:
        scores["SITTING_ON_BED"] -= 0.15
        reasons["SITTING_ON_BED"].append("penalty: body too horizontal for sitting")

    # ===================================================================
    # SITTING_OUTSIDE_BED
    # ===================================================================
    if 0.3 <= f.torso_verticality <= 0.85:
        scores["SITTING_OUTSIDE_BED"] += 0.15
        reasons["SITTING_OUTSIDE_BED"].append("torso suggests sitting posture")
    if has_bed and not f.on_bed and f.distance_to_bed > config.near_bed_distance_threshold:
        scores["SITTING_OUTSIDE_BED"] += 0.20
        reasons["SITTING_OUTSIDE_BED"].append("person away from bed")
    elif not has_bed:
        # Without bed, reduce sitting_outside_bed score
        scores["SITTING_OUTSIDE_BED"] += 0.05
    if f.knee_bend_score >= 0.3:
        scores["SITTING_OUTSIDE_BED"] += 0.10
        reasons["SITTING_OUTSIDE_BED"].append("knee bend consistent with sitting")
    if f.smoothed_speed < config.stationary_speed_threshold * 2:
        scores["SITTING_OUTSIDE_BED"] += 0.05
        reasons["SITTING_OUTSIDE_BED"].append("stationary or near-stationary")
    if f.bbox_aspect_ratio >= 0.8 and f.bbox_aspect_ratio <= 1.8:
        scores["SITTING_OUTSIDE_BED"] += 0.05
    # Penalty if on bed
    if has_bed and f.on_bed:
        scores["SITTING_OUTSIDE_BED"] -= 0.20
        reasons["SITTING_OUTSIDE_BED"].append("penalty: person is on bed")

    # ===================================================================
    # STANDING
    # ===================================================================
    if f.torso_verticality >= 0.6:
        scores["STANDING"] += 0.25
        reasons["STANDING"].append("torso orientation is vertical")
    if f.hip_ankle_vertical_distance >= 0.15:
        scores["STANDING"] += 0.15
        reasons["STANDING"].append(
            f"hip-to-ankle vertical separation {f.hip_ankle_vertical_distance:.2f} is large"
        )
    if f.bbox_aspect_ratio >= config.standing_aspect_ratio_min:
        scores["STANDING"] += 0.15
        reasons["STANDING"].append(
            f"bbox aspect ratio {f.bbox_aspect_ratio:.2f} compatible with standing"
        )
    if has_bed and not f.on_bed:
        scores["STANDING"] += 0.05
        reasons["STANDING"].append("person is not overlapping bed")
    if f.smoothed_speed <= config.walking_speed_threshold:
        scores["STANDING"] += 0.05
        reasons["STANDING"].append("low/moderate movement")
    # Penalty for horizontal
    if f.body_horizontal_score >= 0.5:
        scores["STANDING"] -= 0.20
        reasons["STANDING"].append("penalty: body orientation too horizontal")
    # Penalty for low aspect ratio
    if f.bbox_aspect_ratio > 0 and f.bbox_aspect_ratio < 0.8:
        scores["STANDING"] -= 0.10

    # ===================================================================
    # WALKING
    # ===================================================================
    if f.torso_verticality >= 0.5:
        scores["WALKING"] += 0.15
        reasons["WALKING"].append("upright posture")
    if f.smoothed_speed >= config.walking_speed_threshold:
        scores["WALKING"] += 0.25
        reasons["WALKING"].append(
            f"speed {f.smoothed_speed:.4f} exceeds walking threshold"
        )
    if f.movement_persistence >= config.movement_persistence_frames:
        scores["WALKING"] += 0.15
        reasons["WALKING"].append(
            f"persistent movement for {f.movement_persistence} frames"
        )
    if f.bbox_aspect_ratio >= config.standing_aspect_ratio_min:
        scores["WALKING"] += 0.05
        reasons["WALKING"].append("bbox suggests upright body")
    if f.centroid_displacement >= config.walking_speed_threshold * 0.5:
        scores["WALKING"] += 0.10
        reasons["WALKING"].append("significant centroid displacement")
    # Penalty for stationary
    if f.smoothed_speed < config.stationary_speed_threshold:
        scores["WALKING"] -= 0.20
        reasons["WALKING"].append("penalty: too stationary for walking")

    # ===================================================================
    # OUT_OF_BED (meta-state: away from bed, not further classifiable)
    # ===================================================================
    if has_bed and not f.on_bed and not f.near_bed:
        scores["OUT_OF_BED"] += 0.15
        reasons["OUT_OF_BED"].append("person away from bed")
    if has_bed and f.distance_to_bed > config.near_bed_distance_threshold * 2:
        scores["OUT_OF_BED"] += 0.10
        reasons["OUT_OF_BED"].append("significant distance from bed")

    # ===================================================================
    # UNKNOWN
    # ===================================================================
    if f.pose_visibility < 0.25:
        scores["UNKNOWN"] += 0.30
        reasons["UNKNOWN"].append("insufficient pose evidence")
    if f.person_confidence < 0.3:
        scores["UNKNOWN"] += 0.20
        reasons["UNKNOWN"].append("low detection confidence")
    if f.pose_quality_score <= 0.3:
        scores["UNKNOWN"] += 0.15
        reasons["UNKNOWN"].append("poor pose quality")

    # ===================================================================
    # Determine best state
    # ===================================================================
    # Remove OUT_OF_BED if a more specific state has higher score
    specific_states = ["STANDING", "WALKING", "SITTING_OUTSIDE_BED"]
    out_of_bed_score = scores["OUT_OF_BED"]
    for s in specific_states:
        if scores[s] > out_of_bed_score:
            scores["OUT_OF_BED"] = max(0.0, scores["OUT_OF_BED"] - 0.10)
            break

    # Find best state
    best_state_name = max(scores, key=lambda s: scores[s])
    best_score = scores[best_state_name]

    # Confidence: normalise top score relative to total positive evidence
    total_positive = sum(max(0.0, v) for v in scores.values())
    if total_positive > 0:
        confidence = best_score / total_positive
    else:
        confidence = 0.0

    # If confidence is too low, fall back to UNKNOWN
    if confidence < config.unknown_confidence_threshold:
        best_state_name = "UNKNOWN"
        confidence = max(confidence, 0.1)
        reasons["UNKNOWN"].append("confidence below threshold — insufficient evidence")

    best_state = ActivityState(best_state_name)

    # Determine bed status
    if best_state in (ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED):
        bed_status = BedStatus.IN_BED
    elif best_state == ActivityState.SITTING_ON_BED:
        bed_status = BedStatus.ON_BED_EDGE
    elif best_state in (
        ActivityState.STANDING,
        ActivityState.WALKING,
        ActivityState.OUT_OF_BED,
        ActivityState.SITTING_OUTSIDE_BED,
    ):
        bed_status = BedStatus.OUT_OF_BED
    else:
        bed_status = BedStatus.UNKNOWN

    return StatePrediction(
        state=best_state,
        confidence=round(confidence, 4),
        scores={k: round(v, 4) for k, v in scores.items()},
        reasons=reasons.get(best_state_name, []),
        bed_status=bed_status,
    )
