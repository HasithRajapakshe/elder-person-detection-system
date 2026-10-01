import argparse

import yaml

from src.video.loader import VideoLoader
from src.video.sampler import FrameSampler
from src.video.clip_buffer import ClipBuffer
from src.perception.pipeline import run_tracking


def load_config(path: str):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def run_ingestion(
    video_path: str,
    config: dict,
):

    sample_fps = float(
        config["video"]["sample_fps"]
    )

    clip_seconds = float(
        config["video"]["clip_seconds"]
    )

    with VideoLoader(video_path) as loader:

        metadata = loader.metadata()

        sampler = FrameSampler(
            metadata.fps,
            sample_fps,
        )

        buffer = ClipBuffer(
            clip_seconds,
            sampler.sample_fps,
        )

        sampled_frames = 0

        for (
            frame_index,
            timestamp,
            frame,
        ) in loader.frames():

            if not sampler.should_sample(
                frame_index
            ):
                continue

            buffer.add(
                frame_index,
                timestamp,
                frame,
            )

            sampled_frames += 1

        print(
            "Phase 1 - video ingestion complete"
        )

        print(
            f"Resolution: "
            f"{metadata.width}x{metadata.height}"
        )

        print(
            f"FPS: {metadata.fps:.2f}"
        )

        print(
            f"Duration: "
            f"{metadata.duration_sec:.2f}s"
        )

        print(
            f"Sampled frames: "
            f"{sampled_frames}"
        )

        print(
            f"Clip buffer retained: "
            f"{len(buffer.items())}"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Elderly Agentic Vision"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--config",
        default="config/settings.yaml",
    )

    parser.add_argument(
        "--phase",
        choices=[
            "ingest",
            "track",
        ],
        default="track",
        help=(
            "ingest = Phase 1; "
            "track = Phase 2"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    if args.phase == "ingest":

        run_ingestion(
            args.video,
            config,
        )

        return

    print(
        "Phase 2 - Person Detection "
        "and Tracking"
    )

    result = run_tracking(
        args.video,
        config,
    )

    print(
        f"Tracked video: "
        f"{result['video']}"
    )

    print(
        f"Detections CSV: "
        f"{result['detections']}"
    )

    print(
        f"Tracking summary: "
        f"{result['summary']}"
    )

    print(
        f"Provisional primary track ID: "
        f"{result['primary_track_id']}"
    )

    print(
        f"Person observations: "
        f"{result['observation_count']}"
    )


if __name__ == "__main__":
    main()