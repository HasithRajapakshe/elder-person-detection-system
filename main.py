import argparse

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def run_ingestion(video_path, config):
    from src.video.loader import VideoLoader
    from src.video.sampler import FrameSampler
    from src.video.clip_buffer import ClipBuffer

    with VideoLoader(video_path) as loader:
        meta = loader.metadata()

        sampler = FrameSampler(
            meta.fps,
            float(config["video"]["sample_fps"]),
        )
        buffer = ClipBuffer(
            float(config["video"]["clip_seconds"]),
            sampler.sample_fps,
        )

        sampled = 0

        for index, timestamp, frame in loader.frames():
            if sampler.should_sample(index):
                buffer.add(index, timestamp, frame)
                sampled += 1

        print("Phase 1 - video ingestion complete")
        print(f"Sampled frames: {sampled}")
        print(f"Buffer retained: {len(buffer.items())}")


def main():
    parser = argparse.ArgumentParser(
        description="Elderly Agentic Vision"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
    )
    parser.add_argument(
        "--phase",
        choices=["ingest", "track", "calibrate-bed", "pose", "analyze"],
        default="track",
    )
    parser.add_argument(
        "--calibration-second",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.phase == "ingest":
        run_ingestion(args.video, config)

    elif args.phase == "track":
        from src.perception.pipeline import run_tracking

        result = run_tracking(args.video, config)

        for key, value in result.items():
            print(f"{key}: {value}")

    elif args.phase == "calibrate-bed":
        from src.perception.bed_calibration import calibrate_bed

        calibrate_bed(
            args.video,
            config["phase3"]["bed_region_file"],
            args.calibration_second,
        )

    elif args.phase == "pose":
        from src.perception.pose_pipeline import run_pose_context

        result = run_pose_context(args.video, config)

        for key, value in result.items():
            print(f"{key}: {value}")
            
    elif args.phase == "analyze":
        from src.temporal.pipeline import run_analysis
        
        result = run_analysis(
            video_path=args.video,
            config=config,
            ground_truth_path=args.ground_truth,
        )
        
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()