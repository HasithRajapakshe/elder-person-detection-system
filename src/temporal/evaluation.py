"""Evaluation engine for the activity monitoring pipeline.

Evaluates activity recognition, bed events, and duration estimation
against ground truth labels. Accepts simple CSV/JSON timeline format.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models import (
    ActivityState,
    BedEvent,
    BedEventType,
    ClassMetrics,
    EventMetrics,
    FailureCase,
    TimelineInterval,
)


# Mapping from ground truth labels to our ActivityState
LABEL_MAP = {
    "none": ActivityState.LYING_IN_BED,       # default resting state
    "lie": ActivityState.LYING_IN_BED,
    "lying": ActivityState.LYING_IN_BED,
    "lying_in_bed": ActivityState.LYING_IN_BED,
    "sit": ActivityState.SITTING_ON_BED,
    "sitting": ActivityState.SITTING_ON_BED,
    "sitting_on_bed": ActivityState.SITTING_ON_BED,
    "stand": ActivityState.STANDING,
    "standing": ActivityState.STANDING,
    "walk": ActivityState.WALKING,
    "walking": ActivityState.WALKING,
    "stand-sit": ActivityState.SITTING_ON_BED,  # transition to sitting
    "sit-stand": ActivityState.STANDING,         # transition to standing
    "stand-lie": ActivityState.LYING_IN_BED,     # transition to lying (fall context)
    "unknown": ActivityState.UNKNOWN,
}


def load_ground_truth_csv(path: Path) -> List[Dict[str, Any]]:
    """Load ground truth from a CSV file with columns:
    start_time, end_time, action, is_fall
    """
    labels = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append({
                "start_sec": float(row.get("start_time", row.get("start_sec", 0))),
                "end_sec": float(row.get("end_time", row.get("end_sec", 0))),
                "action": row.get("action", "").strip(),
                "is_fall": str(row.get("is_fall", "False")).lower() in ("true", "1", "yes"),
            })
    return labels


def load_ground_truth_json(path: Path) -> List[Dict[str, Any]]:
    """Load ground truth from a JSON timeline file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("timeline", data.get("labels", []))


def _get_gt_state_at_time(
    gt_labels: List[Dict[str, Any]],
    timestamp: float,
) -> Optional[ActivityState]:
    """Find the ground truth state at a given timestamp."""
    for label in gt_labels:
        if label["start_sec"] <= timestamp <= label["end_sec"]:
            action = label["action"].lower().strip()
            return LABEL_MAP.get(action, ActivityState.UNKNOWN)
    return None


def evaluate_activity_recognition(
    timeline: List[TimelineInterval],
    gt_labels: List[Dict[str, Any]],
    sample_interval_sec: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate activity recognition at sampled timestamps.

    Returns accuracy, per-class precision/recall/F1, confusion matrix.
    """
    if not timeline or not gt_labels:
        return {"accuracy": 0.0, "per_class": [], "confusion_matrix": {}}

    # Determine evaluation time range
    start = max(timeline[0].start_sec, gt_labels[0]["start_sec"])
    end = min(timeline[-1].end_sec, gt_labels[-1]["end_sec"])

    # Sample timestamps
    y_true = []
    y_pred = []
    t = start
    while t <= end:
        gt_state = _get_gt_state_at_time(gt_labels, t)
        pred_state = _get_pred_state_at_time(timeline, t)
        if gt_state is not None and pred_state is not None:
            y_true.append(gt_state.value)
            y_pred.append(pred_state.value)
        t += sample_interval_sec

    if not y_true:
        return {"accuracy": 0.0, "per_class": [], "confusion_matrix": {}}

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    # Per-class metrics
    all_labels = sorted(set(y_true + y_pred))
    per_class = []
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = sum(1 for t in y_true if t == label)
        per_class.append(ClassMetrics(
            label=label,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
        ))

    # Confusion matrix
    confusion: Dict[str, Dict[str, int]] = {}
    for t, p in zip(y_true, y_pred):
        if t not in confusion:
            confusion[t] = {}
        confusion[t][p] = confusion[t].get(p, 0) + 1

    return {
        "accuracy": round(accuracy, 4),
        "total_samples": len(y_true),
        "correct": correct,
        "per_class": [asdict(c) for c in per_class],
        "confusion_matrix": confusion,
    }


def _get_pred_state_at_time(
    timeline: List[TimelineInterval],
    timestamp: float,
) -> Optional[ActivityState]:
    """Find the predicted state at a given timestamp."""
    for interval in timeline:
        if interval.start_sec <= timestamp <= interval.end_sec:
            return interval.state
    return None


def evaluate_bed_events(
    predicted_events: List[BedEvent],
    gt_labels: List[Dict[str, Any]],
    event_type: BedEventType,
    tolerance_sec: float = 5.0,
) -> EventMetrics:
    """Evaluate bed exit/return events against ground truth.

    Currently uses a simple temporal tolerance matching.
    """
    # For this dataset, we don't have explicit bed event ground truth,
    # so we infer from state transitions in the labels
    gt_transitions = _extract_gt_events(gt_labels, event_type)
    pred_times = [
        e.confirmed_time for e in predicted_events
        if e.event_type == event_type
    ]

    tp = 0
    matched_gt = set()
    matched_pred = set()

    for pi, pt in enumerate(pred_times):
        for gi, gt in enumerate(gt_transitions):
            if gi in matched_gt:
                continue
            if abs(pt - gt) <= tolerance_sec:
                tp += 1
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    fp = len(pred_times) - tp
    fn = len(gt_transitions) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return EventMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def _extract_gt_events(
    gt_labels: List[Dict[str, Any]],
    event_type: BedEventType,
) -> List[float]:
    """Infer event timestamps from ground truth label transitions."""
    events = []
    prev_in_bed = True  # Assume starts in bed

    for label in gt_labels:
        action = label["action"].lower().strip()
        state = LABEL_MAP.get(action, ActivityState.UNKNOWN)

        is_bed_state = state in (ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED)

        if event_type == BedEventType.BED_EXIT:
            if prev_in_bed and not is_bed_state and state != ActivityState.UNKNOWN:
                events.append(label["start_sec"])
        elif event_type == BedEventType.BED_RETURN:
            if not prev_in_bed and is_bed_state:
                events.append(label["start_sec"])

        if state != ActivityState.UNKNOWN:
            prev_in_bed = is_bed_state

    return events


def evaluate_durations(
    predicted_durations: Dict[str, float],
    gt_labels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare predicted vs ground truth durations per state."""
    # Compute GT durations
    gt_durations: Dict[str, float] = {}
    for label in gt_labels:
        action = label["action"].lower().strip()
        state = LABEL_MAP.get(action, ActivityState.UNKNOWN)
        duration = label["end_sec"] - label["start_sec"]
        key = state.value
        gt_durations[key] = gt_durations.get(key, 0.0) + duration

    # Compare
    results = {}
    all_states = set(list(gt_durations.keys()) + list(predicted_durations.keys()))
    total_error = 0.0
    count = 0

    for state in sorted(all_states):
        gt_dur = gt_durations.get(state, 0.0)
        pred_dur = predicted_durations.get(state, 0.0)
        abs_error = abs(pred_dur - gt_dur)
        pct_error = (abs_error / gt_dur * 100) if gt_dur > 0 else None
        results[state] = {
            "ground_truth_sec": round(gt_dur, 3),
            "predicted_sec": round(pred_dur, 3),
            "absolute_error_sec": round(abs_error, 3),
            "percentage_error": round(pct_error, 2) if pct_error is not None else None,
        }
        total_error += abs_error
        count += 1

    mean_error = total_error / count if count > 0 else 0.0

    return {
        "per_state": results,
        "mean_absolute_duration_error_sec": round(mean_error, 3),
    }


def generate_failure_cases(
    timeline: List[TimelineInterval],
    gt_labels: List[Dict[str, Any]],
    max_cases: int = 5,
) -> List[FailureCase]:
    """Identify representative failure cases."""
    failures = []

    for interval in timeline:
        mid_ts = (interval.start_sec + interval.end_sec) / 2.0
        gt_state = _get_gt_state_at_time(gt_labels, mid_ts)

        if gt_state is None:
            continue

        if gt_state != interval.state:
            failures.append(FailureCase(
                timestamp_sec=mid_ts,
                window_sec=interval.duration_sec,
                expected=gt_state.value,
                predicted=interval.state.value,
                reason=f"Predicted {interval.state.value} but ground truth is {gt_state.value}",
                likely_cause=_infer_cause(interval, gt_state),
                suggested_improvement=_suggest_improvement(interval, gt_state),
            ))

    # Sort by duration (larger mismatches first) and limit
    failures.sort(key=lambda f: -f.window_sec)
    return failures[:max_cases]


def _infer_cause(
    interval: TimelineInterval,
    gt_state: ActivityState,
) -> str:
    """Infer likely cause of a failure."""
    if interval.mean_confidence < 0.5:
        return "Low prediction confidence — insufficient visual evidence"
    if interval.state == ActivityState.UNKNOWN:
        return "Poor visibility or occlusion prevented classification"
    if gt_state == ActivityState.LYING_IN_BED and interval.state == ActivityState.SITTING_ON_BED:
        return "Difficulty distinguishing lying from sitting when partially covered"
    if gt_state == ActivityState.STANDING and interval.state == ActivityState.WALKING:
        return "Motion threshold may need calibration for this camera"
    return "State classification rule weights may need camera-specific tuning"


def _suggest_improvement(
    interval: TimelineInterval,
    gt_state: ActivityState,
) -> str:
    """Suggest improvement for a failure case."""
    if interval.mean_confidence < 0.5:
        return "Improve pose estimation under occlusion or add multi-view cameras"
    if interval.state == ActivityState.UNKNOWN:
        return "Add learned temporal model (TCN/LSTM) to better handle ambiguity"
    return "Calibrate thresholds per camera environment or use learned classifier"


def build_evaluation_report(
    timeline: List[TimelineInterval],
    bed_events: List[BedEvent],
    predicted_durations: Dict[str, float],
    gt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a complete evaluation report."""
    report: Dict[str, Any] = {
        "ground_truth_available": gt_path is not None and gt_path.exists(),
    }

    if gt_path is None or not gt_path.exists():
        report["note"] = "No ground truth provided — evaluation skipped"
        report["failure_cases"] = [asdict(f) for f in _default_failure_cases()]
        return report

    # Load ground truth
    if gt_path.suffix == ".csv":
        gt_labels = load_ground_truth_csv(gt_path)
    else:
        gt_labels = load_ground_truth_json(gt_path)

    # Activity recognition
    report["activity_recognition"] = evaluate_activity_recognition(
        timeline, gt_labels
    )

    # Bed events
    report["bed_exit_metrics"] = asdict(evaluate_bed_events(
        bed_events, gt_labels, BedEventType.BED_EXIT
    ))
    report["bed_return_metrics"] = asdict(evaluate_bed_events(
        bed_events, gt_labels, BedEventType.BED_RETURN
    ))

    # Duration estimation
    report["duration_estimation"] = evaluate_durations(
        predicted_durations, gt_labels
    )

    # Failure cases
    failures = generate_failure_cases(timeline, gt_labels)
    if len(failures) < 3:
        failures.extend(_default_failure_cases()[:3 - len(failures)])
    report["failure_cases"] = [asdict(f) for f in failures]

    return report


def _default_failure_cases() -> List[FailureCase]:
    """Return documented failure modes when no GT is available."""
    return [
        FailureCase(
            timestamp_sec=0.0,
            window_sec=0.0,
            expected="N/A",
            predicted="N/A",
            reason="Heavy blanket occlusion",
            likely_cause=(
                "Pose landmarks unavailable when body is covered "
                "by blankets; prediction becomes UNKNOWN"
            ),
            suggested_improvement=(
                "Use thermal imaging or depth sensors; "
                "train occlusion-aware pose estimator"
            ),
        ),
        FailureCase(
            timestamp_sec=0.0,
            window_sec=0.0,
            expected="N/A",
            predicted="N/A",
            reason="Bed region unavailable or miscalibrated",
            likely_cause=(
                "Without accurate bed geometry, SITTING_ON_BED "
                "cannot be distinguished from SITTING_OUTSIDE_BED"
            ),
            suggested_improvement=(
                "Implement automatic bed segmentation using "
                "semantic segmentation (e.g. DeepLabV3)"
            ),
        ),
        FailureCase(
            timestamp_sec=0.0,
            window_sec=0.0,
            expected="N/A",
            predicted="N/A",
            reason="Severe occlusion or caregiver overlap",
            likely_cause=(
                "Tracker may temporarily switch or lose identity "
                "when caregiver occludes the elderly person"
            ),
            suggested_improvement=(
                "Add ReID module for identity recovery; "
                "implement multi-person scene understanding"
            ),
        ),
    ]
