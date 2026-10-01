# Elderly Agentic Vision

Agentic AI + Vision pipeline for temporal elderly activity monitoring, bed-exit/return detection, duration analysis, and safety decisions.

## Phase 1
Video ingestion foundation:
- Open video and extract metadata
- Timestamp every frame
- Configurable frame sampling
- Rolling temporal clip buffer

## Setup
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run
Place a video under `data/videos/`, then:
```bash
python main.py --video data/videos/sample.mp4
```

## Test
```bash
pytest -q
```
