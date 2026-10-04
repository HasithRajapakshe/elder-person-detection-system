"""Observation builder: converts raw Phase 3 CSV/JSONL data into
normalised TemporalObservation instances.

This module bridges the existing perception pipeline output with the
new temporal reasoning pipeline.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models import TemporalObservation

# COCO-17 keypoint indices
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


def _safe_float(value, default: float = 0.0) -> float:
    """Convert to float, returning *default* for None or empty strings."""
    if value is None or value == "" or value == "None":
        return default
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "" or value == "None":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_bool(value, default: bool = False) -> bool:
    if value is None or value == "" or value == "None":
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _midpoint(p1: Optional[Tuple], p2: Optional[Tuple]) -> Optional[Tuple[float, float]]:
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _valid_kp(
    keypoints: List[dict],
    index: int,
    threshold: float,
) -> Optional[Tuple[float, float]]:
    """Return (x, y) if the keypoint at *index* is valid, else None."""
    if index >= len(keypoints):
        return None
    kp = keypoints[index]
    if not kp.get("valid", False):
        conf = kp.get("confidence")
        if conf is None or conf < threshold:
            return None
    x, y = float(kp["x"]), float(kp["y"])
    if x == 0.0 and y == 0.0:
        return None
    return (x, y)


def build_observations(
    observations_csv: Path,
    keypoints_jsonl: Path,
    frame_width: int,
    frame_height: int,
    bed_region_pixels: Optional[Tuple[float, float, float, float]] = None,
    keypoint_threshold: float = 0.40,
    primary_track_id: Optional[int] = None,
) -> List[TemporalObservation]:
    """Build TemporalObservation list from Phase 3 outputs.

    If *primary_track_id* is given, only observations matching that
    track are included. When None, the most frequent track is used.
    """
    # Load keypoints indexed by (frame_index, detection_index)
    kp_lookup: Dict[Tuple[int, int], List[dict]] = {}
    with keypoints_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["frame_index"], rec["detection_index"])
            kp_lookup[key] = rec.get("keypoints", [])

    # Load CSV observations
    raw_rows: List[dict] = []
    with observations_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_rows.append(row)

    # Determine primary track if not specified
    if primary_track_id is None:
        from collections import Counter
        track_counter = Counter()
        for row in raw_rows:
            tid = _safe_int(row.get("track_id"))
            if tid is not None:
                track_counter[tid] += 1
        if track_counter:
            primary_track_id = track_counter.most_common(1)[0][0]

    diagonal = math.hypot(frame_width, frame_height)
    frame_area = frame_width * frame_height

    observations: List[TemporalObservation] = []
    prev_center: Optional[Tuple[float, float]] = None
    prev_ts: Optional[float] = None

    for row in raw_rows:
        tid = _safe_int(row.get("track_id"))

        # Filter to primary track only
        if primary_track_id is not None and tid != primary_track_id:
            continue

        fi = int(row["frame_index"])
        ts = _safe_float(row["timestamp_sec"])
        det_idx = int(row.get("detection_index", 0))
        conf = _safe_float(row.get("detection_confidence"), 0.0)

        x1 = _safe_float(row.get("x1"))
        y1 = _safe_float(row.get("y1"))
        x2 = _safe_float(row.get("x2"))
        y2 = _safe_float(row.get("y2"))
        bbox = (x1, y1, x2, y2)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        area_ratio = (bw * bh) / frame_area if frame_area > 0 else 0.0

        # Keypoints
        kp_key = (fi, det_idx)
        kp_list = kp_lookup.get(kp_key, [])

        sl = _valid_kp(kp_list, KP_LEFT_SHOULDER, keypoint_threshold)
        sr = _valid_kp(kp_list, KP_RIGHT_SHOULDER, keypoint_threshold)
        hl = _valid_kp(kp_list, KP_LEFT_HIP, keypoint_threshold)
        hr = _valid_kp(kp_list, KP_RIGHT_HIP, keypoint_threshold)
        kl = _valid_kp(kp_list, KP_LEFT_KNEE, keypoint_threshold)
        kr = _valid_kp(kp_list, KP_RIGHT_KNEE, keypoint_threshold)
        al = _valid_kp(kp_list, KP_LEFT_ANKLE, keypoint_threshold)
        ar = _valid_kp(kp_list, KP_RIGHT_ANKLE, keypoint_threshold)

        shoulder_mid = _midpoint(sl, sr)
        hip_mid = _midpoint(hl, hr)

        # Torso angle
        torso_angle = _safe_float(
            row.get("torso_angle_from_vertical_deg"), default=None  # type: ignore
        )
        if torso_angle is None and shoulder_mid and hip_mid:
            dx = abs(shoulder_mid[0] - hip_mid[0])
            dy = abs(shoulder_mid[1] - hip_mid[1])
            if math.hypot(dx, dy) > 1e-6:
                torso_angle = math.degrees(math.atan2(dx, dy))

        # Body aspect ratio
        aspect_ratio = bh / bw if bw > 1 else None

        # Orientation
        orientation = None
        if torso_angle is not None:
            if torso_angle < 35:
                orientation = "vertical"
            elif torso_angle > 55:
                orientation = "horizontal"
            else:
                orientation = "diagonal"

        # Motion
        displacement = 0.0
        speed = 0.0
        if prev_center is not None and prev_ts is not None and diagonal > 0:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            displacement = math.hypot(dx, dy) / diagonal  # normalised
            dt = ts - prev_ts
            speed = displacement / dt if dt > 0 else 0.0

        # Bed relation
        bed_overlap = _safe_float(row.get("bed_overlap_fraction"), 0.0)
        bed_rel = row.get("bed_relation", "not_configured")
        anchor_in_bed = _safe_bool(row.get("anchor_in_bed"))
        bed_dist = _safe_float(row.get("bed_distance_normalized"), default=None)  # type: ignore
        near = bed_rel in ("overlapping_bed", "near_bed") or anchor_in_bed
        on = bed_overlap >= 0.20 or (anchor_in_bed and bed_overlap >= 0.10)

        # Count keypoints inside bed
        kp_in_bed = 0
        if bed_region_pixels is not None:
            bx1, by1, bx2, by2 = bed_region_pixels
            for kp_pt in [sl, sr, hl, hr, kl, kr, al, ar]:
                if kp_pt and bx1 <= kp_pt[0] <= bx2 and by1 <= kp_pt[1] <= by2:
                    kp_in_bed += 1

        # Pose quality
        torso_kps_valid = sum(1 for p in [sl, sr, hl, hr] if p is not None)
        if torso_kps_valid >= 3:
            pq = "good"
        elif torso_kps_valid >= 2:
            pq = "partial"
        elif torso_kps_valid >= 1:
            pq = "poor"
        else:
            pq = "missing"

        # Pose keypoints as list-of-lists
        pose_xy = None
        pose_conf = None
        if kp_list:
            pose_xy = [[kp["x"], kp["y"]] for kp in kp_list]
            pose_conf = [kp.get("confidence", 0.0) for kp in kp_list]

        obs = TemporalObservation(
            frame_index=fi,
            timestamp_sec=ts,
            track_id=tid,
            detection_index=det_idx,
            person_confidence=conf,
            bbox=bbox,
            bbox_center=(cx, cy),
            bbox_width=bw,
            bbox_height=bh,
            bbox_area_ratio=area_ratio,
            pose_keypoints=pose_xy,
            pose_confidences=pose_conf,
            pose_quality=pq,
            shoulder_left=sl,
            shoulder_right=sr,
            shoulder_mid=shoulder_mid,
            hip_left=hl,
            hip_right=hr,
            hip_mid=hip_mid,
            knee_left=kl,
            knee_right=kr,
            ankle_left=al,
            ankle_right=ar,
            torso_angle_deg=torso_angle,
            body_aspect_ratio=aspect_ratio,
            body_orientation=orientation,
            motion_displacement=displacement,
            motion_speed=speed,
            bed_region=bed_region_pixels,
            bbox_bed_overlap=bed_overlap,
            keypoints_inside_bed=kp_in_bed,
            distance_to_bed=bed_dist,
            near_bed=near,
            on_bed=on,
            bed_relation=bed_rel,
            visibility_quality=pq,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        observations.append(obs)
        prev_center = (cx, cy)
        prev_ts = ts

    return observations
