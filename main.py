import argparse
from pathlib import Path
import yaml
from src.video.loader import VideoLoader
from src.video.sampler import FrameSampler
from src.video.clip_buffer import ClipBuffer


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(video_path: str, config_path: str = "config/settings.yaml"):
    config = load_config(config_path)
    sample_fps = float(config["video"]["sample_fps"])
    clip_seconds = float(config["video"]["clip_seconds"])

    with VideoLoader(video_path) as loader:
        meta = loader.metadata()
        sampler = FrameSampler(meta.fps, sample_fps)
        buffer = ClipBuffer(clip_seconds, sampler.sample_fps)

        sampled = 0
        for frame_index, timestamp, frame in loader.frames():
            if not sampler.should_sample(frame_index):
                continue
            buffer.add(frame_index, timestamp, frame)
            sampled += 1

        print("Video ingestion complete")
        print(f"Path: {meta.path}")
        print(f"Resolution: {meta.width}x{meta.height}")
        print(f"FPS: {meta.fps:.2f}")
        print(f"Frames: {meta.frame_count}")
        print(f"Duration: {meta.duration_sec:.2f}s")
        print(f"Sampled frames: {sampled}")
        print(f"Clip buffer frames retained: {len(buffer.items())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elderly Agentic Vision - Phase 1")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    run(args.video, args.config)
