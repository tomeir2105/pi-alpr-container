# Motion Event and Recording Process

This document describes what the watcher does when motion is detected, what can start a recording, and how the MP4 file is written and finalized.

## Main Loop

The watcher opens the configured `RTSP_URL` and continuously reads frames from the camera. Each frame is resized according to `FRAME_WIDTH`, pushed into an in-memory prebuffer, and used for the dashboard stream.

Motion analysis does not run on every captured frame. Normally it runs every `PROCESS_EVERY_N_FRAMES`. While a video recording is active, the watcher processes fewer motion frames to leave CPU available for ffmpeg:

```text
active recording process interval = PROCESS_EVERY_N_FRAMES * 3
```

Every processed frame goes through `_detect_motion()`.

## Motion Detection

Motion detection uses OpenCV background subtraction:

1. Convert frame to grayscale.
2. Apply Gaussian blur.
3. Apply `cv2.createBackgroundSubtractorMOG2`.
4. Threshold the mask.
5. Clean the mask with morphology open and dilation.
6. Check each enabled motion zone.

For every enabled zone, the watcher counts moving pixels and finds contours. A zone is considered to have raw motion when at least one contour:

- has area at or above `MIN_MOTION_AREA`
- has width at least `12` pixels
- has height at least `12` pixels

The watcher tracks:

- total motion area
- best motion bounding box
- triggered zone IDs
- moving area per zone
- cumulative moving coverage per zone

## What Can Start a Recording

A video recording can start before a full motion event is confirmed.

### 1. Raw Motion Warning

If `_detect_motion()` sees raw motion in an enabled zone, the watcher calls:

```text
_start_video_recording(timestamp, triggered_zone_ids, confirmed=False)
```

This starts a tentative recording. The goal is to capture the beginning of motion without waiting for the full confirmation streak.

If the motion disappears before confirmation and there is no active event, the tentative recording is stopped and discarded as unconfirmed.

### 2. Confirmed Motion Streak

Each processed frame with raw motion increments `motion_streak`, up to `MIN_CONSECUTIVE_HITS`.

When:

```text
motion_streak >= MIN_CONSECUTIVE_HITS
```

the watcher treats the motion as confirmed. If a tentative recording already exists, it is marked `confirmed=True`. If no recording exists, a confirmed recording is started immediately.

Confirmed motion also opens or updates the motion event.

### 3. Cumulative Zone Coverage

The watcher also accumulates motion masks by zone while raw motion is present or while `motion_streak` is still above zero.

If a zone crosses its `coverage_trigger_percent`, that zone can confirm motion and start an event even if the streak path has not done so yet.

This is useful for broader movement that fills a meaningful part of a zone.

### 4. New Motion After Cooldown

A new recording will not start if another recording is already active.

After a recording ends, there is also a cooldown:

```text
cooldown = max(EVENT_IDLE_SECONDS, 5.0)
```

During this cooldown, new motion will not start a new recording.

## What Opens an Event

A motion event is separate from the MP4 writer. The event tracks motion hits, candidate frames, zones, and later image extraction.

An event opens in `_on_motion()` when confirmed motion is present and `self.event` is currently `None`.

When an event opens, it stores:

- `started_at`
- `trigger_count = 1`
- frames from the current prebuffer
- the current candidate frame
- `last_motion_at`
- triggered zone IDs

The event is updated on each confirmed motion frame. Existing events accumulate more candidate frames, update the last motion timestamp, and merge newly triggered zones.

## Event Finalization

When motion is no longer confirmed, the watcher keeps appending postbuffer frames to the active event.

The event finalizes when both conditions are true:

```text
timestamp - event.last_motion_at >= EVENT_IDLE_SECONDS
event.frames_since_motion >= postbuffer frame limit
```

The postbuffer frame limit is:

- `POSTBUFFER_FRAMES`, if set above zero
- otherwise `fps_guess * POSTBUFFER_SECONDS`

When finalized, the event is closed and may be queued for image extraction from the finished video.

Note: `EVENT_MAX_SECONDS` exists in configuration, but the current event loop does not use it to force-close events.

## Recording Setup

Recordings are started by `_start_video_recording()`.

The output filename is:

```text
<VIDEO_OUTPUT_DIR>/<CAMERA_NAME>_<UTC_TIMESTAMP>.mp4
```

The file is first written under:

```text
<VIDEO_OUTPUT_DIR>/recording-tmp/
```

It is moved to the main videos directory only after the recording closes cleanly.

The recording duration comes from the triggered zone policy:

```text
record_seconds = max(record_seconds for triggered zones)
```

If no matching zone is found, it falls back to the primary yellow zone or `VIDEO_RECORDING_SECONDS`.

## Prebuffer

The watcher keeps a rolling prebuffer of recent frames. When recording starts, it copies frames whose timestamps are within:

```text
PREBUFFER_FRAMES / fps_guess
```

if `PREBUFFER_FRAMES > 0`, otherwise:

```text
PREBUFFER_SECONDS
```

To limit CPU and memory, the prebuffer frames written into the video are capped by `MAX_VIDEO_PREBUFFER_FRAMES`.

## Video Writer

The watcher prefers ffmpeg when available. It starts ffmpeg as a subprocess and writes raw BGR frames to stdin:

```text
rawvideo bgr24 -> libx264 -> yuv420p MP4
```

The ffmpeg output settings are:

- codec: `libx264`
- preset: `veryfast`
- pixel format: `yuv420p`
- `+faststart`
- threads from `FFMPEG_THREADS`

If ffmpeg is unavailable, it falls back to OpenCV `VideoWriter` with `mp4v`.

Writes go through `QueuedVideoWriter`, which keeps a bounded queue so the capture loop is not blocked by slow disk or encoding. If the queue fills, older queued frames may be dropped.

## Recording FPS

The recording FPS is chosen from measured processing FPS or camera FPS guess:

```text
fps = max(5.0, min(MAX_RECORDING_FPS, processing_fps or fps_guess))
```

`MAX_RECORDING_FPS` is currently `15.0`.

When appending frames, the watcher may duplicate the latest frame to keep the output timeline close to the target FPS. It caps catch-up writes to at most five seconds of frames at a time.

## Extending an Active Recording

When motion is confirmed against an active recording, the recording end time is guaranteed to be at least the triggered zone's configured minimum duration from the confirmation timestamp:

```text
min_ends_at = confirmation_timestamp + zone_record_seconds
recording.ends_at = max(recording.ends_at, min_ends_at)
recording.record_seconds = max(recording.record_seconds, recording.ends_at - recording.started_at)
```

This ensures every confirmed trigger produces a clip that is at least as long as the zone's configured recording time, even if motion stops immediately after confirmation.

If motion continues or new zones trigger while recording is active, `ends_at` keeps extending the same way on every confirmed motion frame in `_on_motion()`:

```text
recording.ends_at = max(recording.ends_at, timestamp + zone_record_seconds)
```

This means long or repeated motion during the same event stays within the same MP4 file.

## Stopping and Finalizing a Recording

Recording stops when the current timestamp passes `recording.ends_at`, or when the capture loop is shutting down.

On stop:

1. The queued writer is released.
2. ffmpeg stdin is closed.
3. ffmpeg is given a bounded time to finalize the MP4.
4. If ffmpeg reports an error, the watcher checks whether the temp MP4 is still readable.
5. If the temp MP4 is readable, it is kept.
6. If it is not readable, it is deleted.
7. If the recording was never confirmed, it is deleted.
8. If confirmed, the temp MP4 is moved into the main videos directory.
9. A JSON metadata file is written next to the MP4.

The metadata contains:

- camera name
- video path
- source
- start timestamp
- first frame timestamp
- end timestamp
- zone IDs
- queued event count
- prebuffer seconds
- recording seconds

## When a Recording Is Discarded

A temp recording can be discarded when:

- raw motion started a tentative recording, but the motion was never confirmed
- ffmpeg/OpenCV failed and the temp MP4 is missing, empty, or unreadable
- finalizing the recording failed before it could be moved into the main videos directory

A confirmed event can also be ignored for image saving if it has fewer than `MIN_CONSECUTIVE_HITS` motion hits.

## Image Extraction After Recording

If the finalized event uses Fast-ALPR, the event is queued on the associated recording. After the MP4 is finalized, `_save_video_events()` extracts event images from the saved video and runs the detection pipeline.

If Fast-ALPR is disabled for the triggered zone, the motion event still records video, but automatic image creation is skipped.

Manual image extraction from saved videos is still available from the Videos page.

