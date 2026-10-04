from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import csv
import json

import cv2


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    timestamp_sec: float
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(
            0.0, self.y2 - self.y1
        )


class PersonTracker:
    """
    Pretrained YOLO person detector + BoT-SORT tracker.

    Only COCO class 0 (person) is requested.

    The detector answers:
        "Where is a person?"

    The tracker answers:
        "Is this the same person seen in previous frames?"
    """

    def __init__(
        self,
        model_name: str = "yolo11s.pt",
        confidence: float = 0.35,
        iou: float = 0.50,
        tracker_config: str = "botsort.yaml",
        device: str | None = None,
    ) -> None:

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(model_name)

        self.confidence = confidence
        self.iou = iou
        self.tracker_config = tracker_config
        self.device = device

    def track_frame(
        self,
        frame,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[TrackObservation]:

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[0],          # COCO person class
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        observations: list[TrackObservation] = []

        result = results[0]
        boxes = result.boxes

        if boxes is None or boxes.id is None:
            return observations

        xyxy = boxes.xyxy.cpu().tolist()
        ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for box, track_id, confidence in zip(
            xyxy,
            ids,
            confidences,
        ):
            x1, y1, x2, y2 = box

            observations.append(
                TrackObservation(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    track_id=int(track_id),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(confidence),
                )
            )

        return observations


def choose_primary_track(
    observations: Iterable[TrackObservation],
    frame_width: int,
    frame_height: int,
) -> tuple[int | None, dict[int, dict[str, float]]]:
    """
    Select a provisional monitored person.

    Phase 2 uses:
        - track persistence
        - bounding-box area
        - detection confidence

    Phase 3 will improve this using bed context.
    """

    observations = list(observations)

    if not observations:
        return None, {}

    max_frame = max(
        observation.frame_index
        for observation in observations
    ) + 1

    image_area = max(
        1.0,
        float(frame_width * frame_height),
    )

    grouped: dict[int, list[TrackObservation]] = {}

    for observation in observations:
        grouped.setdefault(
            observation.track_id,
            [],
        ).append(observation)

    stats: dict[int, dict[str, float]] = {}

    for track_id, items in grouped.items():

        unique_frames = len(
            {
                item.frame_index
                for item in items
            }
        )

        presence_ratio = unique_frames / max_frame

        mean_area_ratio = (
            sum(
                item.area / image_area
                for item in items
            )
            / len(items)
        )

        mean_confidence = (
            sum(
                item.confidence
                for item in items
            )
            / len(items)
        )

        score = (
            0.70 * presence_ratio
            + 0.20 * min(
                1.0,
                mean_area_ratio * 4.0,
            )
            + 0.10 * mean_confidence
        )

        stats[track_id] = {
            "presence_ratio": round(
                presence_ratio,
                6,
            ),
            "mean_area_ratio": round(
                mean_area_ratio,
                6,
            ),
            "mean_confidence": round(
                mean_confidence,
                6,
            ),
            "score": round(
                score,
                6,
            ),
        }

    primary_track_id = max(
        stats,
        key=lambda track_id: stats[track_id]["score"],
    )

    return primary_track_id, stats


def write_detections_csv(
    path: str | Path,
    observations: Iterable[TrackObservation],
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "frame_index",
        "timestamp_sec",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for observation in observations:

            row = asdict(observation)

            row["timestamp_sec"] = round(
                row["timestamp_sec"],
                3,
            )

            row["confidence"] = round(
                row["confidence"],
                5,
            )

            writer.writerow(row)


def write_tracking_summary(
    path: str | Path,
    primary_track_id: int | None,
    stats: dict[int, dict[str, float]],
    total_frames: int,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "primary_track_id": primary_track_id,

        "selection_note": (
            "Phase 2 provisional target selected using "
            "track persistence, bounding-box area and "
            "detection confidence. Phase 3 will refine "
            "target identity using bed and scene context."
        ),

        "total_frames": total_frames,

        "tracks": {
            str(track_id): values
            for track_id, values in stats.items()
        },
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def draw_tracks(
    frame,
    observations: Iterable[TrackObservation],
):
    """
    Draw tracked-person bounding boxes for debugging.
    """

    for observation in observations:

        point1 = (
            int(observation.x1),
            int(observation.y1),
        )

        point2 = (
            int(observation.x2),
            int(observation.y2),
        )

        cv2.rectangle(
            frame,
            point1,
            point2,
            (0, 255, 0),
            2,
        )

        label = (
            f"Person ID {observation.track_id} | "
            f"{observation.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (
                point1[0],
                max(
                    20,
                    point1[1] - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame