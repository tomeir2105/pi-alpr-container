# RTSP to OpenALPR Watcher

This project pulls an RTSP stream from your home camera, detects passing motion locally, samples event frames across the full motion timeline, optionally checks frames with a local `fast-alpr` container, and uploads qualifying frames to Rekor/OpenALPR for cloud recognition.

It also exposes a local dashboard UI:

- Unified dashboard: `http://127.0.0.1:8080/`
- Full-screen live video: `http://127.0.0.1:8080/live`
- Configuration editor: `http://127.0.0.1:8080/stats`
- Captured images: `http://127.0.0.1:8080/images`
- Saved videos: `http://127.0.0.1:8080/videos`
- Detected plates: `http://127.0.0.1:8080/plates`
- Detection test upload: `http://127.0.0.1:8080/test`

Persistent data is stored under:

- `/mnt/localdisk/pi-alpr/events/images`
- `/mnt/localdisk/pi-alpr/events/videos`
- `/mnt/localdisk/pi-alpr/fast-alpr-cache`

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
- `docker-compose.yml`: full watcher + `fast-alpr` stack
- `third_party/fast-alpr-container/`: tracked container app for `fast-alpr`
- `events/`: created at runtime for clips, frames, and JSON results

## Setup

```bash
mkdir -p /mnt/localdisk/pi-alpr/events/images
mkdir -p /mnt/localdisk/pi-alpr/events/videos
mkdir -p /mnt/localdisk/pi-alpr/fast-alpr-cache
mkdir -p /mnt/localdisk/pi-alpr/config
cp .env.example /mnt/localdisk/pi-alpr/config/.env
```

Edit `/mnt/localdisk/pi-alpr/config/.env` and set:

- `RTSP_URL`
- `OPENALPR_SECRET_KEY`
- `OPENALPR_COUNTRY`
- `ROI`

For Hikvision cameras, a typical stream setup is:

```dotenv
RTSP_URL=rtsp://username:password@camera-host:554/Streaming/Channels/102
ALPR_RTSP_URL=rtsp://username:password@camera-host:554/Streaming/Channels/101
```

Use channel `102` for the lighter motion stream and channel `101` for sharper ALPR images.

If you want local filtering before OpenALPR, also set:

- `FAST_ALPR_URL`

For the built-in web UI, you can also set:

- `WEB_PORT`

## Run

```bash
docker compose up -d --build
```

When the watcher is running, open:

- `http://127.0.0.1:8080/`
- `http://127.0.0.1:8080/live`
- `http://127.0.0.1:8080/stats`
- `http://127.0.0.1:8080/images`
- `http://127.0.0.1:8080/videos`
- `http://127.0.0.1:8080/plates`

## Web UI

- The dashboard shows the editable motion zones and motion sensitivity controls.
- The live page shows only the camera feed full-screen.
- The config page edits `.env`; recreate the watcher container after saving Docker env-file changes.
- The images page opens image details in the same tab and supports keyboard navigation: left arrow for newer images, right arrow for older images.
- The videos page opens video details in the same tab with the top menu still visible.
- The top menu includes remove buttons for saved images, saved videos, and detected plate entries.
- The plates page includes the detection test upload link.

## Tuning

- `ROI` should cover only the driveway or road where cars pass.
- Hikvision channel `101` is usually the high-resolution main stream; use `102` when the main stream smears or the host cannot decode it at camera FPS.
- `ALPR_RTSP_URL` should point at the high-resolution `101` stream so live view and saved event images use the sharper camera feed while motion detection can stay on `RTSP_URL`; Hikvision `.../Channels/102` URLs are automatically tried as `.../Channels/101` if `ALPR_RTSP_URL` is blank.
- `PLATE_ROI` can be a tighter sub-zone where plates are expected to appear; this powers the zoom panel and crops frames before ALPR runs.
- `FRAME_WIDTH=960` keeps processing, streaming, and recordings bounded. Use `0` only if the host has enough CPU/RAM for the camera's native resolution.
- `MIN_MOTION_AREA` filters out tiny motion like rain, trees, and shadows.
- `MIN_CONSECUTIVE_HITS` helps avoid one-frame false triggers.
- `PREBUFFER_SECONDS` keeps video from before the trigger.
- `POSTBUFFER_SECONDS` keeps video after the last motion.
- `PREBUFFER_FRAMES` overrides `PREBUFFER_SECONDS` if you prefer an exact frame count.
- `POSTBUFFER_FRAMES` overrides `POSTBUFFER_SECONDS` if you prefer an exact frame count.
- `EVENT_IDLE_SECONDS` is how long motion must stop before the event can close.
- `EVENT_MAX_SECONDS` forces an event to close even if motion detection keeps reporting movement.
- `UPLOAD_TOP_FRAMES` controls how many images are sampled across the event timeline; the watcher always tries to save at least 10 images per kept event.
- `UPLOAD_MIN_SHARPNESS` skips blurry frames when possible.
- `FAST_ALPR_URL` enables local plate recognition before cloud upload.
- `FAST_ALPR_MIN_CONFIDENCE` is the minimum local OCR confidence required before a frame is sent to OpenALPR.
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and `TELEGRAM_ALERT_IMAGES` configure Telegram photo alerts for movement events.
- `TELEGRAM_ALERTS_ENABLED` sets the startup default for per-zone Telegram alerts; the dashboard zone checkboxes can turn alerts on or off at runtime.
- `PLATE_ROI` crops frames to the likely plate area before local/cloud ALPR and powers the zoomed monitoring panel.
- `WEB_PORT` controls the local monitoring UI port.
- `MAX_SAVED_IMAGES` limits how many JPG captures are kept across all events.
- `STREAM_FPS` controls the dashboard MJPEG refresh rate without slowing the RTSP reader.
- `ALPR_CAPTURE_FPS` controls how often the 101 stream is sampled for saved event images; `ALPR_CAPTURE_WARMUP_SECONDS` and `LIVE_STREAM_WARMUP_SECONDS` skip unstable frames right after opening 101.
- `CAPTURE_BUFFER_SIZE` and `RTSP_CAPTURE_OPTIONS` favor stable TCP capture over lowest-latency capture to avoid H.264 smear from damaged reference frames.
- `HOST_DATA_DIR` controls the host folder mounted into both containers as `/data`; keep it on `/mnt/localdisk` to avoid saving outputs on the OS disk.
- `APP_ENV_FILE` controls the `.env` file mounted into the watcher; keep it under `/mnt/localdisk` so config saves from the UI stay off the OS disk.
- `DOCKER_LOG_DRIVER=none` prevents Docker from persisting container stdout/stderr logs under Docker's data directory.
- `EVENT_OUTPUT_DIR`, `IMAGE_OUTPUT_DIR`, and `VIDEO_OUTPUT_DIR` should stay under `/data` in containers so files land under `HOST_DATA_DIR`.

## RTSP authentication

If the dashboard opens but the camera feed never appears, test the RTSP URL from inside the watcher container:

```bash
docker exec -it pi-alpr-watcher ffprobe -rtsp_transport tcp \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 \
  "$RTSP_URL"
```

`401 Unauthorized` means Docker can reach the camera, but the camera rejected RTSP authentication. On Hikvision cameras, check the camera web UI and set RTSP authentication to `Digest/Basic` or `Basic/Digest`, then recreate the watcher container:

```bash
cd /mnt/localdisk/pi-alpr/app
docker compose up -d --force-recreate watcher
```

The active runtime config is `/mnt/localdisk/pi-alpr/config/.env`. Keep real camera passwords and API keys only in `.env`; do not commit them.

## Useful environment variables

```dotenv
PROCESS_EVERY_N_FRAMES=2
FRAME_WIDTH=960
ALPR_RTSP_URL=rtsp://username:password@camera-host:554/Streaming/Channels/101
MIN_MOTION_AREA=2500
MIN_CONSECUTIVE_HITS=3
EVENT_IDLE_SECONDS=1.5
EVENT_MAX_SECONDS=60.0
PREBUFFER_SECONDS=2.0
POSTBUFFER_SECONDS=5.0
PREBUFFER_FRAMES=0
POSTBUFFER_FRAMES=0
UPLOAD_TOP_FRAMES=30
UPLOAD_MIN_SHARPNESS=80.0
FAST_ALPR_URL=http://fast-alpr:8090
FAST_ALPR_MIN_CONFIDENCE=0.75
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_ALERT_IMAGES=3
TELEGRAM_ALERTS_ENABLED=true
WEB_PORT=8080
MAX_SAVED_IMAGES=200
FFMPEG_THREADS=1
STREAM_FPS=3
CAPTURE_BUFFER_SIZE=4
RTSP_CAPTURE_OPTIONS=rtsp_transport;tcp|max_delay;2000000|stimeout;10000000
ALPR_CAPTURE_FPS=0.5
ALPR_CAPTURE_WARMUP_SECONDS=1.5
LIVE_STREAM_WARMUP_SECONDS=1.5
HOST_DATA_DIR=/mnt/localdisk/pi-alpr
APP_ENV_FILE=/mnt/localdisk/pi-alpr/config/.env
DOCKER_LOG_DRIVER=none
EVENT_OUTPUT_DIR=/data/events
IMAGE_OUTPUT_DIR=/data/events/images
VIDEO_OUTPUT_DIR=/data/events/videos
PLATE_ROI=0.30,0.45,0.80,0.75
```

Notes:

- Set `PREBUFFER_FRAMES` or `POSTBUFFER_FRAMES` to `0` to use the time-based values instead.
- If you want exactly 20 frames before the event and 15 frames after it, set `PREBUFFER_FRAMES=20` and `POSTBUFFER_FRAMES=15`.
- With the default `POSTBUFFER_SECONDS=5.0`, the image event keeps collecting frames for 5 seconds after motion stops.
- The images page keeps only the newest `MAX_SAVED_IMAGES` JPG files and removes older frame artifacts automatically.
- Event/test frame policy: crop to `PLATE_ROI` if set, run `fast-alpr`, and only send to OpenALPR if `fast-alpr` found a plate at or above `FAST_ALPR_MIN_CONFIDENCE`.

## Output

Each event creates an image directory under `/mnt/localdisk/pi-alpr/events/images`, for example:

```text
/mnt/localdisk/pi-alpr/events/images/home-driveway_20260409T120000Z/
```

Inside an event directory you will find:

- `frame_01.jpg`, `frame_02.jpg`, ...
- `frame_01.fast_alpr.json`, `frame_02.fast_alpr.json`, ...
- `frame_01.json`, `frame_02.json`, ...
- `summary.json`

Longer motion-triggered MP4 recordings are stored separately under:

```text
/mnt/localdisk/pi-alpr/events/videos/
```

MP4 recordings are written through `ffmpeg` as H.264/yuv420p files and are first stored under `events/videos/recording-tmp/`. They move into the main videos directory only after the recording is closed cleanly.

Docker image layers and container metadata are controlled by the Docker daemon storage location, usually `/var/lib/docker`. Move Docker's `data-root` to `/mnt/localdisk` at the host level if those must also avoid the OS disk.

## Important note

This first version uses motion detection plus ROI filtering, not a full vehicle detector. It works best when the camera is stable and the ROI covers only the lane or driveway. If you want, the next step can be upgrading this to use YOLO vehicle detection before sending frames to OpenALPR for much fewer false positives.
