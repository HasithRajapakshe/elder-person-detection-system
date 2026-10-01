from collections import deque
from dataclasses import dataclass

@dataclass
class BufferedFrame:
    frame_index: int
    timestamp_sec: float
    frame: object

class ClipBuffer:
    def __init__(self, max_seconds: float, sample_fps: float):
        max_items = max(1, int(max_seconds * sample_fps))
        self._frames = deque(maxlen=max_items)

    def add(self, frame_index: int, timestamp_sec: float, frame) -> None:
        self._frames.append(BufferedFrame(frame_index, timestamp_sec, frame))

    def items(self):
        return list(self._frames)
