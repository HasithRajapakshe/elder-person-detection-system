from __future__ import annotations

from pathlib import Path

import cv2

from src.perception.person_tracker import (
    PersonTracker,
    choose_primary_track,
    draw_tracks,
    write_detections_csv,
    write_tracking_summary,
)
from src.video.loader import VideoLoader


def run_tracking(
    video_path: str,
    config: dict,
) -> dict:

    output_dir = Path(
        config["output"]["directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    perception_config = config.get(
        "perception",
        {},
    )

    tracker = PersonTracker(
        model_name=perception_config.get(
            "person_model",
            "yolo11s.pt",
        ),
        confidence=float(
            perception_config.get(
                "confidence",
                0.35,
            )
        ),
        iou=float(
            perception_config.get(
                "iou",
                0.50,
            )
        ),
        tracker_config=perception_config.get(
            "tracker",
            "botsort.yaml",
        ),
        device=perception_config.get(
            "device",
        ),
    )

    all_observations = []

    tracked_video_path = (
        output_dir / "tracked_video.mp4"
    )

    detections_path = (
        output_dir / "detections.csv"
    )

    summary_path = (
        output_dir / "tracking_summary.json"
    )

    with VideoLoader(video_path) as loader:

        metadata = loader.metadata()

        writer = cv2.VideoWriter(
            str(tracked_video_path),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            metadata.fps,
            (
                metadata.width,
                metadata.height,
            ),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not create output video: "
                f"{tracked_video_path}"
            )

        try:

            for (
                frame_index,
                timestamp,
                frame,
            ) in loader.frames():

                observations = (
                    tracker.track_frame(
                        frame,
                        frame_index,
                        timestamp,
                    )
                )

                all_observations.extend(
                    observations
                )

                annotated_frame = draw_tracks(
                    frame.copy(),
                    observations,
                )

                writer.write(
                    annotated_frame
                )

        finally:
            writer.release()

    primary_track_id, stats = (
        choose_primary_track(
            all_observations,
            metadata.width,
            metadata.height,
        )
    )

    write_detections_csv(
        detections_path,
        all_observations,
    )

    write_tracking_summary(
        summary_path,
        primary_track_id,
        stats,
        metadata.frame_count,
    )

    return {
        "video": str(
            tracked_video_path
        ),
        "detections": str(
            detections_path
        ),
        "summary": str(
            summary_path
        ),
        "primary_track_id": (
            primary_track_id
        ),
        "observation_count": len(
            all_observations
        ),
    }