"""Complete analysis pipeline.

Orchestrates the full flow from Phase 3 outputs through temporal
reasoning, bed event detection, alert generation, timeline building,
and evaluation. This is the main entry point for the `analyze` CLI
command.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import PipelineConfig
from src.models import (
    ActivityState,
    AlertLevel,
    BedEvent,
    BedEventType,
    StatePrediction,
    TemporalObservation,
    TimelineInterval,
)
from src.temporal.observation_builder import build_observations
from src.temporal.feature_extraction import ExtractedFeatures, extract_features
from src.temporal.state_classifier import classify_state
from src.temporal.smoother import TemporalSmoother, SmoothedState
from src.temporal.fsm import ActivityFSM
from src.temporal.bed_events import BedEventDetector
from src.temporal.agentic_resolver import AgenticContextResolver
from src.temporal.alert_engine import AlertEngine
from src.temporal.timeline import build_timeline, aggregate_durations
from src.temporal.evaluation import build_evaluation_report

logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> Any:
    """Make objects JSON-serializable."""
    if hasattr(obj, "value"):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    return obj


def run_analysis(
    video_path: str,
    config: dict,
    output_dir: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete analysis pipeline.

    This operates on existing Phase 3 outputs (pose_observations.csv
    and keypoints.jsonl). If those don't exist, it will run Phase 3
    first.
    """
    # Resolve configuration
    phase3_cfg = config.get("phase3", {})
    temporal_cfg = config.get("temporal", {})

    pipeline_config = PipelineConfig.from_dict(temporal_cfg)

    # Override from phase3 config
    if phase3_cfg.get("bed_region_file"):
        pipeline_config.bed_context_enabled = True
    else:
        pipeline_config.bed_context_enabled = False

    # Resolve paths
    phase3_dir = Path(phase3_cfg.get("output_directory", "outputs/phase3"))
    out_dir = Path(output_dir) if output_dir else phase3_dir.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    observations_csv = phase3_dir / "pose_observations.csv"
    keypoints_jsonl = phase3_dir / "keypoints.jsonl"
    summary_json = phase3_dir / "phase3_summary.json"

    # Check if Phase 3 outputs exist; run Phase 3 if not
    if not observations_csv.exists() or not keypoints_jsonl.exists():
        logger.info("Phase 3 outputs not found. Running pose pipeline...")
        from src.perception.pose_pipeline import run_pose_context
        run_pose_context(video_path, config)

    # Load Phase 3 summary for metadata
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    else:
        summary = {}

    fps = summary.get("fps", 30.0)
    frame_count = summary.get("processed_frames", 0)
    # Try to get frame dimensions from config or summary
    # We'll extract from the first observation
    frame_width = 1280  # default
    frame_height = 720  # default

    # Load bed region
    bed_region_pixels = None
    bed_path = phase3_cfg.get("bed_region_file")
    if bed_path and Path(bed_path).exists():
        from src.perception.bed_context import load_bed_region
        bed = load_bed_region(bed_path)
        bed_region_pixels = bed.pixels(frame_width, frame_height)

    # Determine primary track
    # Check phase3 summary for track presence
    track_presence = summary.get("track_presence", {})
    primary_track_id = None
    if track_presence:
        best_track = max(
            track_presence.items(),
            key=lambda x: x[1].get("observations", x[1].get("presence_ratio", 0))
        )
        primary_track_id = int(best_track[0])
        logger.info(f"Primary track ID: {primary_track_id}")

    # ===================================================================
    # STEP 1: Build observations
    # ===================================================================
    logger.info("Building temporal observations...")
    observations = build_observations(
        observations_csv=observations_csv,
        keypoints_jsonl=keypoints_jsonl,
        frame_width=frame_width,
        frame_height=frame_height,
        bed_region_pixels=bed_region_pixels,
        keypoint_threshold=pipeline_config.keypoint_confidence_threshold,
        primary_track_id=primary_track_id,
    )
    logger.info(f"Built {len(observations)} observations")

    if not observations:
        logger.warning("No observations found for the primary track")
        return {"error": "No observations found"}

    # Update frame dimensions from first observation
    if observations[0].frame_width > 0:
        frame_width = observations[0].frame_width
        frame_height = observations[0].frame_height

    # ===================================================================
    # STEP 2: Extract features and classify
    # ===================================================================
    logger.info("Extracting features and classifying states...")
    raw_predictions: List[StatePrediction] = []
    features_list: List[ExtractedFeatures] = []
    movement_history: List[float] = []
    stationary_since: Optional[float] = None

    for obs in observations:
        features = extract_features(
            obs, pipeline_config,
            movement_history=movement_history,
            stationary_since=stationary_since,
        )
        features_list.append(features)

        prediction = classify_state(features, pipeline_config)
        raw_predictions.append(prediction)

        # Track motion history
        movement_history.append(obs.motion_displacement)
        if len(movement_history) > 30:
            movement_history.pop(0)

        if obs.motion_speed < pipeline_config.stationary_speed_threshold:
            if stationary_since is None:
                stationary_since = obs.timestamp_sec
        else:
            stationary_since = None

    # ===================================================================
    # STEP 3: Agentic context resolution
    # ===================================================================
    logger.info("Running agentic context resolution...")
    resolver = AgenticContextResolver(pipeline_config)
    timestamps = [o.timestamp_sec for o in observations]

    resolved_predictions = list(raw_predictions)
    for i, pred in enumerate(raw_predictions):
        if pred.state == ActivityState.UNKNOWN or pred.confidence < pipeline_config.state_confidence_threshold:
            result = resolver.resolve(
                i, pred, timestamps[i], raw_predictions, timestamps
            )
            if result.revised_state is not None and result.resolution == "revised":
                resolved_predictions[i] = StatePrediction(
                    state=result.revised_state,
                    confidence=result.confidence,
                    scores=pred.scores,
                    reasons=pred.reasons + [f"agentic: {result.resolution}"],
                    bed_status=pred.bed_status,
                )

    # ===================================================================
    # STEP 4: Temporal smoothing
    # ===================================================================
    logger.info("Applying temporal smoothing...")
    smoother = TemporalSmoother(pipeline_config)
    smoothed_states: List[SmoothedState] = []

    for i, pred in enumerate(resolved_predictions):
        smoothed = smoother.update(pred, timestamps[i])
        smoothed_states.append(smoothed)

    # ===================================================================
    # STEP 5: FSM transitions
    # ===================================================================
    logger.info("Processing FSM transitions...")
    fsm = ActivityFSM()

    for smoothed in smoothed_states:
        fsm.update(smoothed)

    # ===================================================================
    # STEP 6: Bed event detection
    # ===================================================================
    logger.info("Detecting bed events...")
    bed_detector = BedEventDetector(pipeline_config)

    for i, smoothed in enumerate(smoothed_states):
        transition = None
        if i > 0 and smoothed.smoothed_state != smoothed_states[i - 1].smoothed_state:
            # Find the corresponding FSM transition
            for t in fsm.transitions:
                if abs(t.confirmed_timestamp - smoothed.timestamp_sec) < 0.01:
                    transition = t
                    break
        bed_detector.update(smoothed.smoothed_state, smoothed.timestamp_sec, transition)

    # ===================================================================
    # STEP 7: Alert engine
    # ===================================================================
    logger.info("Generating alerts...")
    alert_engine = AlertEngine(pipeline_config)
    alert_events = []

    bed_event_lookup = {
        e.confirmed_time: e for e in bed_detector.events
    }

    for smoothed in smoothed_states:
        bed_event = bed_event_lookup.get(smoothed.timestamp_sec)
        person_detected = True  # We have an observation
        alert = alert_engine.evaluate(
            smoothed.smoothed_state,
            smoothed.timestamp_sec,
            bed_event=bed_event,
            person_detected=person_detected,
        )
        alert_events.append(alert)

    # ===================================================================
    # STEP 8: Build timeline
    # ===================================================================
    logger.info("Building timeline...")
    final_states = [s.smoothed_state for s in smoothed_states]
    final_confidences = [s.smoothed_confidence for s in smoothed_states]

    timeline = build_timeline(final_states, final_confidences, timestamps)

    # ===================================================================
    # STEP 9: Duration aggregation
    # ===================================================================
    logger.info("Aggregating durations...")
    video_duration = frame_count / fps if fps > 0 else 0.0
    activity_summary = aggregate_durations(
        timeline, bed_detector.events, video_duration
    )

    # ===================================================================
    # STEP 10: Evaluation
    # ===================================================================
    logger.info("Building evaluation report...")
    gt_path = Path(ground_truth_path) if ground_truth_path else None
    if gt_path is None:
        # Check default location
        default_gt = Path("data/ground_truth/sample_labels.csv")
        if default_gt.exists():
            gt_path = default_gt

    evaluation = build_evaluation_report(
        timeline,
        bed_detector.events,
        activity_summary.activity_durations,
        gt_path,
    )

    # ===================================================================
    # WRITE OUTPUTS
    # ===================================================================
    logger.info("Writing output files...")

    # observations.jsonl
    _write_jsonl(
        out_dir / "observations.jsonl",
        [_obs_to_dict(obs) for obs in observations],
    )

    # state_predictions.jsonl
    _write_jsonl(
        out_dir / "state_predictions.jsonl",
        [
            {
                "index": i,
                "timestamp_sec": timestamps[i],
                "raw_state": raw_predictions[i].state.value,
                "raw_confidence": raw_predictions[i].confidence,
                "smoothed_state": smoothed_states[i].smoothed_state.value,
                "smoothed_confidence": smoothed_states[i].smoothed_confidence,
                "reasons": raw_predictions[i].reasons,
                "scores": _serialize(raw_predictions[i].scores),
            }
            for i in range(len(observations))
        ],
    )

    # state_transitions.json
    _write_json(
        out_dir / "state_transitions.json",
        _serialize([asdict(t) for t in fsm.transitions]),
    )

    # activity_timeline.json
    _write_json(
        out_dir / "activity_timeline.json",
        _serialize([asdict(iv) for iv in timeline]),
    )

    # activity_summary.json
    _write_json(
        out_dir / "activity_summary.json",
        _serialize(asdict(activity_summary)),
    )

    # bed_events.json
    _write_json(
        out_dir / "bed_events.json",
        _serialize([asdict(e) for e in bed_detector.events]),
    )

    # alert_events.json
    # Only write unique / significant alerts
    significant_alerts = _deduplicate_alerts(alert_events)
    _write_json(
        out_dir / "alert_events.json",
        _serialize([asdict(a) for a in significant_alerts]),
    )

    # agent_reasoning.json
    _write_json(
        out_dir / "agent_reasoning.json",
        _serialize([asdict(r) for r in resolver.results]),
    )

    # evaluation_report.json
    _write_json(
        out_dir / "evaluation_report.json",
        _serialize(evaluation),
    )

    # failure_cases.json
    failure_cases = evaluation.get("failure_cases", [])
    _write_json(
        out_dir / "failure_cases.json",
        _serialize(failure_cases),
    )

    logger.info(f"Analysis complete. Outputs written to {out_dir}")

    return {
        "output_dir": str(out_dir),
        "observations": len(observations),
        "timeline_intervals": len(timeline),
        "transitions": len(fsm.transitions),
        "bed_exits": bed_detector.exit_count,
        "bed_returns": bed_detector.return_count,
        "alerts_generated": len(significant_alerts),
        "observation_duration_sec": activity_summary.observation_duration_sec,
        "final_state": activity_summary.final_state,
    }


def _obs_to_dict(obs: TemporalObservation) -> dict:
    """Convert observation to a JSON-safe dictionary."""
    d = {}
    for attr in [
        "frame_index", "timestamp_sec", "track_id", "detection_index",
        "person_confidence", "bbox_width", "bbox_height", "bbox_area_ratio",
        "pose_quality", "torso_angle_deg", "body_aspect_ratio",
        "body_orientation", "motion_displacement", "motion_speed",
        "bbox_bed_overlap", "distance_to_bed", "near_bed", "on_bed",
        "bed_relation", "visibility_quality",
    ]:
        val = getattr(obs, attr, None)
        d[attr] = val
    if obs.bbox:
        d["bbox"] = list(obs.bbox)
    if obs.bbox_center:
        d["bbox_center"] = list(obs.bbox_center)
    return d


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def _deduplicate_alerts(alerts: list) -> list:
    """Keep only distinct alert changes and periodic MONITOR/ALERT."""
    if not alerts:
        return []

    result = [alerts[0]]
    last_level = alerts[0].level
    last_rule = alerts[0].rule

    for alert in alerts[1:]:
        if alert.level != last_level or alert.rule != last_rule:
            result.append(alert)
            last_level = alert.level
            last_rule = alert.rule
        elif alert.level in (AlertLevel.MONITOR, AlertLevel.ALERT):
            # Keep periodic monitor/alert updates at lower frequency
            if not result or (alert.timestamp - result[-1].timestamp) >= 10.0:
                result.append(alert)

    return result
