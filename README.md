# RTSP to OpenALPR Watcher

This project pulls an RTSP stream from your home camera, detects a passing vehicle event locally, optionally checks frames with a local `fast-alpr` container, saves a short event clip, and uploads the sharpest frames to Rekor/OpenALPR for cloud recognition.

## How it works

1. The script opens your RTSP stream.
2. It watches a configurable road region of interest (ROI).
3. When motion in that ROI looks vehicle-sized for several consecutive checks, it starts an event.
4. It keeps a short pre-roll and post-roll buffer so the saved clip includes the full pass.
5. If configured, it sends event frames to a local `fast-alpr` container first.
6. It uploads qualifying frames to the OpenALPR cloud API.
7. It stores the clip, uploaded frames, API responses, and a summary on disk.

## Files

- `alpr_watcher.py`: main watcher service
- `.env.example`: environment variables to copy into `.env`
- `requirements.txt`: Python dependencies
- `docker-compose.fast-alpr.yml`: optional local plate-recognition container
- `third_party/fast-alpr-container/`: tracked container app for `fast-alpr`
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

If you want local filtering before OpenALPR, also set:

- `FAST_ALPR_URL`

## Run

```bash
source .venv/bin/activate
python3 alpr_watcher.py
```

To start the local `fast-alpr` container:

```bash
docker compose -f docker-compose.fast-alpr.yml up -d --build
```

## Tuning

- `ROI` should cover only the driveway or road where cars pass.
- `MIN_MOTION_AREA` filters out tiny motion like rain, trees, and shadows.
- `MIN_CONSECUTIVE_HITS` helps avoid one-frame false triggers.
- `PREBUFFER_SECONDS` keeps video from before the trigger.
- `POSTBUFFER_SECONDS` keeps video after the last motion.
- `PREBUFFER_FRAMES` overrides `PREBUFFER_SECONDS` if you prefer an exact frame count.
- `POSTBUFFER_FRAMES` overrides `POSTBUFFER_SECONDS` if you prefer an exact frame count.
- `EVENT_IDLE_SECONDS` is how long motion must stop before the event can close.
- `UPLOAD_TOP_FRAMES` controls how many images go to OpenALPR per event.
- `UPLOAD_MIN_SHARPNESS` skips blurry frames when possible.
- `FAST_ALPR_URL` enables local plate recognition before cloud upload.
- `FAST_ALPR_MIN_CONFIDENCE` is the minimum local OCR confidence required before a frame is sent to OpenALPR.

## Useful environment variables

```dotenv
PROCESS_EVERY_N_FRAMES=2
MIN_MOTION_AREA=6500
MIN_CONSECUTIVE_HITS=3
EVENT_IDLE_SECONDS=1.5
PREBUFFER_SECONDS=2.0
POSTBUFFER_SECONDS=1.5
PREBUFFER_FRAMES=0
POSTBUFFER_FRAMES=0
UPLOAD_TOP_FRAMES=3
UPLOAD_MIN_SHARPNESS=80.0
FAST_ALPR_URL=http://127.0.0.1:8090
FAST_ALPR_MIN_CONFIDENCE=0.75
```

Notes:

- Set `PREBUFFER_FRAMES` or `POSTBUFFER_FRAMES` to `0` to use the time-based values instead.
- If you want exactly 20 frames before the event and 15 frames after it, set `PREBUFFER_FRAMES=20` and `POSTBUFFER_FRAMES=15`.

## Output

Each event creates a directory under `events/`, for example:

```text
events/home-driveway_20260409T120000Z/
```

Inside you will find:

- `event.mp4`
- `frame_01.jpg`, `frame_02.jpg`, ...
- `frame_01.fast_alpr.json`, `frame_02.fast_alpr.json`, ...
- `frame_01.json`, `frame_02.json`, ...
- `summary.json`

## Important note

This first version uses motion detection plus ROI filtering, not a full vehicle detector. It works best when the camera is stable and the ROI covers only the lane or driveway. If you want, the next step can be upgrading this to use YOLO vehicle detection before sending frames to OpenALPR for much fewer false positives.
