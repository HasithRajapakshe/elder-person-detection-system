from pathlib import Path
import json

import cv2

from src.perception.bed_context import BedRegion


def calibrate_bed(
    video_path: str,
    destination: str,
    timestamp_sec: float = 0.0,
):
    if timestamp_sec < 0:
        raise ValueError("Calibration timestamp cannot be negative.")

    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(
                "Cannot read calibration frame. "
                "Try another --calibration-second value."
            )
    finally:
        cap.release()

    height, width = frame.shape[:2]

    # Resize display to fit a typical laptop screen.
    scale = min(1.0, 1200 / width, 750 / height)
    display = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
    )
    display_height, display_width = display.shape[:2]

    print(
        "Drag a rectangle around the visible mattress/bed surface. "
        "Press ENTER or SPACE to confirm; C to cancel."
    )

    try:
        x, y, w, h = cv2.selectROI(
            "Select bed region",
            display,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyAllWindows()

    if w <= 0 or h <= 0:
        raise RuntimeError("Bed selection cancelled or empty.")

    region = BedRegion(
        x / display_width,
        y / display_height,
        (x + w) / display_width,
        (y + h) / display_height,
    )

    payload = {
        "coordinate_system": "normalized",
        "rectangle": [
            region.x1,
            region.y1,
            region.x2,
            region.y2,
        ],
        "source_video": str(Path(video_path)),
        "calibration_second": timestamp_sec,
        "reference_width": width,
        "reference_height": height,
        "note": (
            "Manual scene calibration. Valid only while the "
            "bed occupies the same image region."
        ),
    }

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Bed region saved: {path}")