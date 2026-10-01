from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BedRegion:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        values = (self.x1, self.y1, self.x2, self.y2)

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Bed coordinates must be finite.")

        if not (
            0 <= self.x1 < self.x2 <= 1
            and 0 <= self.y1 < self.y2 <= 1
        ):
            raise ValueError(
                "Bed coordinates must be normalized to 0–1 "
                "and define a positive rectangle."
            )

    def pixels(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive.")

        return (
            self.x1 * width,
            self.y1 * height,
            self.x2 * width,
            self.y2 * height,
        )


def load_bed_region(path: str | Path) -> BedRegion:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Bed region not found: {path}. "
            "Run --phase calibrate-bed first."
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("coordinate_system") != "normalized":
        raise ValueError(
            "Bed region must use normalized coordinates."
        )

    rectangle = data["rectangle"]

    if len(rectangle) != 4:
        raise ValueError("Bed rectangle requires four values.")

    return BedRegion(*map(float, rectangle))


def bbox_overlap_fraction(box, bed_box) -> float:
    """Intersection area divided by person bounding-box area."""
    x1, y1, x2, y2 = map(float, box)
    bx1, by1, bx2, by2 = map(float, bed_box)

    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    if area <= 0:
        return 0.0

    intersection_width = max(
        0.0, min(x2, bx2) - max(x1, bx1)
    )
    intersection_height = max(
        0.0, min(y2, by2) - max(y1, by1)
    )

    return intersection_width * intersection_height / area


def point_inside(point, rectangle) -> bool:
    x, y = point
    x1, y1, x2, y2 = rectangle

    return x1 <= x <= x2 and y1 <= y <= y2


def point_rectangle_distance(point, rectangle) -> float:
    x, y = point
    x1, y1, x2, y2 = rectangle

    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)

    return math.hypot(dx, dy)


def midpoint(keypoints, confidences, indices, threshold):
    """Require both keypoints to be sufficiently reliable."""
    if keypoints is None or confidences is None:
        return None

    if any(
        i >= len(keypoints) or i >= len(confidences)
        for i in indices
    ):
        return None

    for i in indices:
        x, y = keypoints[i]
        confidence = confidences[i]

        if not all(math.isfinite(float(v)) for v in (x, y, confidence)):
            return None

        if confidence < threshold:
            return None

    return (
        sum(float(keypoints[i][0]) for i in indices) / len(indices),
        sum(float(keypoints[i][1]) for i in indices) / len(indices),
    )


def spatial_features(
    box,
    keypoints,
    confidences,
    bed_box,
    frame_width,
    frame_height,
    keypoint_threshold=0.40,
    overlap_threshold=0.30,
    near_distance=0.05,
):
    """Produce spatial evidence, not an activity classification."""
    if not 0 <= keypoint_threshold <= 1:
        raise ValueError("Keypoint threshold must be between 0 and 1.")

    if not 0 < overlap_threshold <= 1:
        raise ValueError("Overlap threshold must be greater than 0.")

    if near_distance < 0:
        raise ValueError("Near-bed distance cannot be negative.")

    x1, y1, x2, y2 = map(float, box)
    overlap = bbox_overlap_fraction(box, bed_box)

    # COCO keypoint indices: shoulders 5/6; hips 11/12.
    shoulders = midpoint(
        keypoints, confidences, (5, 6), keypoint_threshold
    )
    hips = midpoint(
        keypoints, confidences, (11, 12), keypoint_threshold
    )

    # Prefer hips; explicitly report when bbox center is used.
    anchor = hips or ((x1 + x2) / 2, (y1 + y2) / 2)
    anchor_source = "hips" if hips is not None else "bbox_center"

    diagonal = math.hypot(frame_width, frame_height)

    if diagonal <= 0:
        raise ValueError("Frame dimensions must be positive.")

    distance = (
        point_rectangle_distance(anchor, bed_box) / diagonal
    )
    anchor_in_bed = point_inside(anchor, bed_box)

    torso_angle = None

    if shoulders is not None and hips is not None:
        dx = abs(shoulders[0] - hips[0])
        dy = abs(shoulders[1] - hips[1])

        if math.hypot(dx, dy) > 1e-6:
            # 0 = image-vertical torso; 90 = image-horizontal.
            torso_angle = math.degrees(math.atan2(dx, dy))

    if overlap >= overlap_threshold:
        relation = "overlapping_bed"
    elif anchor_in_bed or distance <= near_distance:
        relation = "near_bed"
    else:
        relation = "away_from_bed"

    return {
        "bed_overlap_fraction": overlap,
        "anchor_source": anchor_source,
        "anchor_in_bed": anchor_in_bed,
        "bed_distance_normalized": distance,
        "torso_angle_from_vertical_deg": torso_angle,
        "bed_relation": relation,
        "pose_quality": (
            "torso_available"
            if torso_angle is not None
            else "insufficient_torso_keypoints"
        ),
    }