# Elderly Agentic Vision

Agentic AI + Vision pipeline for temporal elderly activity monitoring, bed-exit/return detection, duration analysis, and safety decisions.

## Features & Phases Implemented

*   **Phase 1: Video Ingestion** - Metadata extraction, frame sampling, configurable rolling buffer.
*   **Phase 2: Person Tracking** - Pre-trained YOLO detection + BoT-SORT tracker with primary-person selection (based on persistence, area, and confidence).
*   **Phase 3: Pose & Bed Context** - YOLO pose estimation with static bed boundary overlap detection.
*   **Phase 4: Temporal Feature Extraction** - Computes explainable, normalised features from raw pose data (verticality, geometry, motion, etc.).
*   **Phase 5: State Classification** - Rule-based deterministic classifier mapping features to states (`LYING_IN_BED`, `STANDING`, etc.).
*   **Phase 6: Temporal Smoothing** - Rolling-window confidence-weighted smoothing with hysteresis to prevent single-frame noise.
*   **Phase 7: Finite State Machine (FSM)** - Validates plausible state progressions and captures transition metadata.
*   **Phase 8: Bed Event Detection** - Requires temporal persistence to confirm `BED_EXIT` and `BED_RETURN` events.
*   **Phase 9: Agentic Context Resolution** - Resolves low-confidence or `UNKNOWN` states by analysing adjacent temporal windows.
*   **Phase 10: Alert Engine** - Emits NORMAL, MONITOR, or ALERT signals based on duration and context (e.g. prolonged out-of-bed).
*   **Phase 11: Timeline & Evaluation** - Compresses states into intervals, aggregates duration, and runs ground-truth evaluation.

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

The pipeline is split into logical phases to allow caching intermediates.
Run the complete end-to-end analysis using:

```bash
python main.py --phase analyze --video data/videos/sample_video.mp4 --ground-truth data/ground_truth/sample_labels.csv
```

Outputs will be saved in `outputs/analysis/` including timelines, transition metadata, bed events, alerts, and the final evaluation report.

## Testing
Run the complete test suite (Phase 1-11):
```bash
pytest -v
```
