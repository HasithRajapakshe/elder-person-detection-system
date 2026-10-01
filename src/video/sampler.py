class FrameSampler:
    def __init__(self, source_fps: float, sample_fps: float):
        if source_fps <= 0 or sample_fps <= 0:
            raise ValueError("FPS values must be positive")
        self.source_fps = source_fps
        self.sample_fps = min(sample_fps, source_fps)
        self.interval = max(1, round(source_fps / self.sample_fps))

    def should_sample(self, frame_index: int) -> bool:
        return frame_index % self.interval == 0
