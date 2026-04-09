# RTSP to OpenALPR Watcher

This project pulls an RTSP stream from your home camera, detects a passing vehicle event locally, saves a short event clip, and uploads the sharpest frames from that event to Rekor/OpenALPR for license plate and vehicle recognition.

## How it works

1. The script opens your RTSP stream.
2. It watches a configurable road region of interest (ROI).
3. When motion in that ROI looks vehicle-sized for several consecutive checks, it starts an event.
4. It keeps a short pre-roll and post-roll buffer so the saved clip includes the full pass.
5. It uploads the best event frames to the OpenALPR cloud API.
6. It stores the clip, uploaded frames, API responses, and a summary on disk.

## Files

- `alpr_watcher.py`: main watcher service
- `.env.example`: environment variables to copy into `.env`
- `requirements.txt`: Python dependencies
- `events/`: created at runtime for clips, frames, and JSON results

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

- `RTSP_URL`
- `OPENALPR_SECRET_KEY`
- `OPENALPR_COUNTRY`
- `ROI`

## Run

```bash
source .venv/bin/activate
python3 alpr_watcher.py
```

## Tuning

- `ROI` should cover only the driveway or road where cars pass.
- `MIN_MOTION_AREA` filters out tiny motion like rain, trees, and shadows.
- `MIN_CONSECUTIVE_HITS` helps avoid one-frame false triggers.
- `UPLOAD_TOP_FRAMES` controls how many images go to OpenALPR per event.
- `UPLOAD_MIN_SHARPNESS` skips blurry frames when possible.

## Output

Each event creates a directory under `events/`, for example:

```text
events/home-driveway_20260409T120000Z/
```

Inside you will find:

- `event.mp4`
- `frame_01.jpg`, `frame_02.jpg`, ...
- `frame_01.json`, `frame_02.json`, ...
- `summary.json`

## Important note

This first version uses motion detection plus ROI filtering, not a full vehicle detector. It works best when the camera is stable and the ROI covers only the lane or driveway. If you want, the next step can be upgrading this to use YOLO vehicle detection before sending frames to OpenALPR for much fewer false positives.
