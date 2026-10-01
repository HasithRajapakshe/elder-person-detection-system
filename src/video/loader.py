from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float

class VideoLoader:
    def __init__(self, video_path: str | Path):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video: {self.path}")

    def metadata(self) -> VideoMetadata:
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(str(self.path), fps, frame_count, width, height, duration)

    def frames(self):
        fps = self.metadata().fps
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            timestamp_sec = index / fps if fps > 0 else 0.0
            yield index, timestamp_sec, frame
            index += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
