from __future__ import annotations

import csv
import json
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import cv2

from src.video.loader import VideoLoader
from src.perception.bed_context import (
    load_bed_region,
    spatial_features,
)


KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

OBSERVATION_FIELDS = [
    "frame_index",
    "timestamp_sec",
    "detection_index",
    "track_id",
    "detection_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "bed_overlap_fraction",
    "anchor_source",
    "anchor_in_bed",
    "bed_distance_normalized",
    "torso_angle_from_vertical_deg",
    "bed_relation",
    "pose_quality",
]

FRAME_FIELDS = [
    "frame_index",
    "timestamp_sec",
    "person_detection_count",
    "tracked_person_count",
    "candidate_detection_index",
    "candidate_track_id",
    "candidate_status",
    "candidate_bed_relation",
]


def run_pose_context(video_path: str, config: dict) -> dict:
    from ultralytics import YOLO

    cfg = config["phase3"]
    bed_path = cfg.get("bed_region_file")
    bed = load_bed_region(bed_path) if bed_path else None

    output_dir = Path(cfg["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    video_output = output_dir / "pose_context_video.mp4"
    observation_output = output_dir / "pose_observations.csv"
    keypoint_output = output_dir / "keypoints.jsonl"
    frame_output = output_dir / "frame_context.csv"
    summary_output = output_dir / "phase3_summary.json"

    keypoint_threshold = float(
        cfg.get("keypoint_confidence", 0.40)
    )
    overlap_threshold = float(
        cfg.get("bed_overlap_threshold", 0.30)
    )
    near_distance = float(
        cfg.get("near_bed_distance", 0.05)
    )

    if not 0 <= keypoint_threshold <= 1:
        raise ValueError("Invalid keypoint confidence threshold.")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("Invalid bed overlap threshold.")
    if near_distance < 0:
        raise ValueError("Invalid near-bed distance.")

    max_frames = config.get("video", {}).get("max_frames")

    if max_frames is not None:
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer or null.")

    # A separate pose model means IDs are local to this Phase 3 run.
    model = YOLO(cfg.get("pose_model", "yolo11s-pose.pt"))

    processed = 0
    observations_total = 0
    tracked_total = 0
    frames_without_detections = 0
    frames_without_candidate = 0

    track_counts = Counter()
    relation_counts = Counter()

    with VideoLoader(video_path) as loader:
        meta = loader.metadata()

        if not math.isfinite(meta.fps) or meta.fps <= 0:
            raise ValueError("Video must have a valid positive FPS.")

        if meta.width <= 0 or meta.height <= 0:
            raise ValueError("Video must have valid dimensions.")

        bed_box = (
            bed.pixels(meta.width, meta.height)
            if bed is not None else None
        )

        writer = cv2.VideoWriter(
            str(video_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            meta.fps,
            (meta.width, meta.height),
        )

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Cannot create video: {video_output}"
            )

        try:
            with ExitStack() as stack:
                obs_file = stack.enter_context(
                    observation_output.open(
                        "w", newline="", encoding="utf-8"
                    )
                )
                frame_file = stack.enter_context(
                    frame_output.open(
                        "w", newline="", encoding="utf-8"
                    )
                )
                kp_file = stack.enter_context(
                    keypoint_output.open(
                        "w", encoding="utf-8"
                    )
                )

                obs_writer = csv.DictWriter(
                    obs_file, fieldnames=OBSERVATION_FIELDS
                )
                frame_writer = csv.DictWriter(
                    frame_file, fieldnames=FRAME_FIELDS
                )
                obs_writer.writeheader()
                frame_writer.writeheader()

                for frame_index, timestamp, frame in loader.frames():
                    if max_frames is not None and processed >= max_frames:
                        break

                    result = model.track(
                        source=frame,
                        persist=True,
                        tracker=cfg.get(
                            "tracker",
                            "config/botsort_custom.yaml",
                        ),
                        classes=[0],
                        conf=float(cfg.get("confidence", 0.10)),
                        iou=float(cfg.get("iou", 0.50)),
                        device=cfg.get("device"),
                        verbose=False,
                    )[0]

                    boxes = result.boxes
                    keypoints = result.keypoints

                    coordinates = (
                        boxes.xyxy.cpu().tolist()
                        if boxes is not None else []
                    )
                    confidences = (
                        boxes.conf.cpu().tolist()
                        if boxes is not None else []
                    )
                    ids = (
                        boxes.id.cpu().tolist()
                        if boxes is not None and boxes.id is not None
                        else None
                    )

                    pose_xy = (
                        keypoints.xy.cpu().tolist()
                        if keypoints is not None else []
                    )
                    pose_conf = (
                        keypoints.conf.cpu().tolist()
                        if (
                            keypoints is not None
                            and keypoints.conf is not None
                        )
                        else []
                    )

                    rows = []

                    for index, box in enumerate(coordinates):
                        track_id = (
                            int(ids[index])
                            if ids is not None
                            else None
                        )
                        xy = (
                            pose_xy[index]
                            if index < len(pose_xy) else None
                        )
                        kp_conf = (
                            pose_conf[index]
                            if index < len(pose_conf) else None
                        )

                        features = spatial_features(
                            box,
                            xy,
                            kp_conf,
                            bed_box if bed_box is not None else (0, 0, 0, 0),
                            meta.width,
                            meta.height,
                            keypoint_threshold,
                            overlap_threshold,
                            near_distance,
                        )

                        if bed is None:
                            # Retain pose features, discard bed-related calculations.
                            features.update({
                                "bed_overlap_fraction": None,
                                "anchor_in_bed": None,
                                "bed_distance_normalized": None,
                                "bed_relation": "not_configured",
                            })

                        row = {
                            "frame_index": frame_index,
                            "timestamp_sec": round(timestamp, 6),
                            "detection_index": index,
                            "track_id": track_id,
                            "detection_confidence": float(
                                confidences[index]
                            ),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            **features,
                        }

                        obs_writer.writerow(row)
                        rows.append(row)

                        points = []

                        if xy is not None:
                            for k, point in enumerate(xy):
                                confidence = (
                                    float(kp_conf[k])
                                    if kp_conf is not None and k < len(kp_conf)
                                    else None
                                )

                                points.append({
                                    "name": (
                                        KEYPOINT_NAMES[k]
                                        if k < len(KEYPOINT_NAMES)
                                        else f"keypoint_{k}"
                                    ),
                                    "x": float(point[0]),
                                    "y": float(point[1]),
                                    "confidence": confidence,
                                    "valid": (
                                        confidence is not None
                                        and confidence >= keypoint_threshold
                                    ),
                                })

                        kp_file.write(
                            json.dumps({
                                "frame_index": frame_index,
                                "timestamp_sec": round(timestamp, 6),
                                "detection_index": index,
                                "track_id": track_id,
                                "keypoints": points,
                            }, allow_nan=False) + "\n"
                        )

                        observations_total += 1
                        relation_counts[features["bed_relation"]] += 1

                        if track_id is not None:
                            tracked_total += 1
                            track_counts[track_id] += 1

                    # Spatial candidate only. This is not patient identity.
                    eligible = [
                        row for row in rows
                        if row["bed_relation"] in (
                            "overlapping_bed",
                            "near_bed",
                        )
                    ] if bed is not None else []

                    candidate = (
                        max(
                            eligible,
                            key=lambda row: (
                                row["bed_overlap_fraction"],
                                -row["bed_distance_normalized"],
                                row["detection_confidence"],
                            ),
                        )
                        if eligible else None
                    )

                    tracked_in_frame = sum(
                        row["track_id"] is not None for row in rows
                    )

                    if not rows:
                        frames_without_detections += 1

                    if candidate is None:
                        frames_without_candidate += 1

                    if bed is None:
                        status = (
                            "pose_only_person_detected"
                            if rows
                            else "unknown_no_person_detection"
                        )
                    else:
                        status = (
                            "provisional_spatial_candidate"
                            if candidate is not None
                            else (
                                "unknown_no_person_detection"
                                if not rows
                                else "unknown_no_bed_candidate"
                            )
                        )

                    frame_writer.writerow({
                        "frame_index": frame_index,
                        "timestamp_sec": round(timestamp, 6),
                        "person_detection_count": len(rows),
                        "tracked_person_count": tracked_in_frame,
                        "candidate_detection_index": (
                            candidate["detection_index"]
                            if candidate else None
                        ),
                        "candidate_track_id": (
                            candidate["track_id"]
                            if candidate else None
                        ),
                        "candidate_status": status,
                        "candidate_bed_relation": (
                            "not_configured"
                            if bed is None
                            else (
                                candidate["bed_relation"]
                                if candidate else "unknown"
                            )
                        ),
                    })

                    annotated = result.plot(
                        kpt_line=True,
                        kpt_radius=3,
                        conf=False,
                    )

                    if bed_box is not None:
                        bx1, by1, bx2, by2 = map(int, bed_box)
                        cv2.rectangle(
                            annotated,
                            (bx1, by1),
                            (bx2, by2),
                            (255, 180, 0),
                            2,
                        )
                        cv2.putText(
                            annotated,
                            "Calibrated bed region",
                            (bx1, max(20, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 180, 0),
                            2,
                        )

                    for row in rows:
                        tid = (
                            row["track_id"]
                            if row["track_id"] is not None
                            else "unassigned"
                        )
                        label = f"ID {tid}: {row['bed_relation']}"

                        cv2.putText(
                            annotated,
                            label,
                            (
                                int(row["x1"]),
                                min(
                                    meta.height - 5,
                                    max(20, int(row["y2"]) + 18),
                                ),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )

                    cv2.putText(
                        annotated,
                        f"{timestamp:.2f}s | {status}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                    writer.write(annotated)
                    processed += 1

                    if processed % 100 == 0:
                        print(
                            f"Processed {processed} frames | "
                            f"{observations_total} observations"
                        )
        finally:
            writer.release()

    if processed == 0:
        raise RuntimeError("No video frames were processed.")

    summary = {
        "processed_frames": processed,
        "source_reported_frames": meta.frame_count,
        "fps": meta.fps,
        "person_observations": observations_total,
        "tracked_observations": tracked_total,
        "frames_without_person_detections": frames_without_detections,
        "frames_without_bed_candidate": (
            frames_without_candidate if bed is not None else None
        ),
        "bed_context_enabled": bed is not None,
        "track_presence": {
            str(tid): {
                "observations": count,
                "presence_ratio": count / processed,
            }
            for tid, count in sorted(track_counts.items())
        },
        "spatial_relation_observation_counts": dict(relation_counts),
        "bed_region": (
            {
                "normalized_rectangle": [
                    bed.x1, bed.y1, bed.x2, bed.y2
                ],
            }
            if bed is not None else None
        ),
        "limitations": [
            "Track IDs are local to this run and can change after absence.",
            "Spatial candidates are not confirmed patient identities.",
            "No detection means unknown; it does not prove physical absence.",
            "Bed overlap is a 2D image feature, not proof of lying in bed.",
            "Torso angle is image-relative and depends on camera perspective.",
            "No activity events or durations are inferred in Phase 3.",
            "Bed calibration assumes a fixed camera and unchanged scene.",
            "Ground-truth labels are not read during inference.",
        ],
        "configuration": cfg,
    }

    summary_output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return {
        "video": str(video_output),
        "observations": str(observation_output),
        "keypoints": str(keypoint_output),
        "frames": str(frame_output),
        "summary": str(summary_output),
    }