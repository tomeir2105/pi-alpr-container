import cgi
import html
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote
from typing import Any, Deque, List, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

MIN_ALLOWED_MOTION_AREA = 500
VIDEO_PREBUFFER_SECONDS = 10.0
VIDEO_RECORDING_SECONDS = 180.0
DEFAULT_FRAME_WIDTH = 960
MAX_RECORDING_FPS = 15.0
MAX_VIDEO_PREBUFFER_FRAMES = 30
VIDEO_WRITER_QUEUE_SECONDS = 8.0
DEFAULT_RTSP_CAPTURE_OPTIONS = "rtsp_transport;tcp|max_delay;2000000|stimeout;10000000"
DEFAULT_ALPR_CAPTURE_FPS = 2.0


def parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_normalized_roi(value: str, env_name: str) -> Optional[Tuple[float, float, float, float]]:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{env_name} must contain 4 comma-separated normalized values")
    x1, y1, x2, y2 = parts
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{env_name} values must satisfy 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0")
    return (x1, y1, x2, y2)


@dataclass
class Config:
    rtsp_url: str
    alpr_rtsp_url: str
    secret_key: str
    country: str
    frame_width: int
    process_every_n_frames: int
    min_motion_area: int
    min_consecutive_hits: int
    event_idle_seconds: float
    event_max_seconds: float
    prebuffer_seconds: float
    postbuffer_seconds: float
    prebuffer_frames: int
    postbuffer_frames: int
    upload_top_frames: int
    upload_min_sharpness: float
    event_output_dir: Path
    camera_name: str
    roi: Optional[Tuple[float, float, float, float]]
    plate_roi: Optional[Tuple[float, float, float, float]]
    recognize_vehicle: bool
    debug_windows: bool
    request_timeout_seconds: float
    fast_alpr_url: str
    fast_alpr_min_confidence: float
    web_host: str
    web_port: int
    max_saved_images: int
    ffmpeg_threads: int
    stream_fps: float
    capture_buffer_size: int
    rtsp_capture_options: str
    alpr_capture_fps: float
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_alert_images: int

    @property
    def openalpr_enabled(self) -> bool:
        return bool(self.secret_key)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id and self.telegram_alert_images > 0)

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        rtsp_url = os.getenv("RTSP_URL", "").strip()
        secret_key = os.getenv("OPENALPR_SECRET_KEY", "").strip()
        if not rtsp_url:
            raise ValueError("RTSP_URL is required")

        roi = parse_normalized_roi(os.getenv("ROI", "").strip(), "ROI")
        plate_roi = parse_normalized_roi(os.getenv("PLATE_ROI", "").strip(), "PLATE_ROI")

        return cls(
            rtsp_url=rtsp_url,
            alpr_rtsp_url=os.getenv("ALPR_RTSP_URL", "").strip(),
            secret_key=secret_key,
            country=os.getenv("OPENALPR_COUNTRY", "us").strip(),
            frame_width=int(os.getenv("FRAME_WIDTH", str(DEFAULT_FRAME_WIDTH))),
            process_every_n_frames=max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))),
            min_motion_area=max(MIN_ALLOWED_MOTION_AREA, int(os.getenv("MIN_MOTION_AREA", "6500"))),
            min_consecutive_hits=max(1, int(os.getenv("MIN_CONSECUTIVE_HITS", "3"))),
            event_idle_seconds=float(os.getenv("EVENT_IDLE_SECONDS", "1.5")),
            event_max_seconds=max(1.0, float(os.getenv("EVENT_MAX_SECONDS", "60.0"))),
            prebuffer_seconds=float(os.getenv("PREBUFFER_SECONDS", "2.0")),
            postbuffer_seconds=float(os.getenv("POSTBUFFER_SECONDS", "5.0")),
            prebuffer_frames=max(0, int(os.getenv("PREBUFFER_FRAMES", "0"))),
            postbuffer_frames=max(0, int(os.getenv("POSTBUFFER_FRAMES", "0"))),
            upload_top_frames=max(1, int(os.getenv("UPLOAD_TOP_FRAMES", "30"))),
            upload_min_sharpness=float(os.getenv("UPLOAD_MIN_SHARPNESS", "80.0")),
            event_output_dir=Path(os.getenv("EVENT_OUTPUT_DIR", "./events")).expanduser(),
            camera_name=os.getenv("CAMERA_NAME", "camera").strip(),
            roi=roi,
            plate_roi=plate_roi,
            recognize_vehicle=parse_bool(os.getenv("RECOGNIZE_VEHICLE", "true"), default=True),
            debug_windows=parse_bool(os.getenv("DEBUG_WINDOWS", "false"), default=False),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20")),
            fast_alpr_url=os.getenv("FAST_ALPR_URL", "").strip(),
            fast_alpr_min_confidence=float(os.getenv("FAST_ALPR_MIN_CONFIDENCE", "0.75")),
            web_host=os.getenv("WEB_HOST", "0.0.0.0").strip(),
            web_port=int(os.getenv("WEB_PORT", "8080")),
            max_saved_images=max(1, int(os.getenv("MAX_SAVED_IMAGES", "50"))),
            ffmpeg_threads=max(1, int(os.getenv("FFMPEG_THREADS", "1"))),
            stream_fps=max(1.0, float(os.getenv("STREAM_FPS", "5"))),
            capture_buffer_size=max(1, int(os.getenv("CAPTURE_BUFFER_SIZE", "4"))),
            rtsp_capture_options=os.getenv("RTSP_CAPTURE_OPTIONS", DEFAULT_RTSP_CAPTURE_OPTIONS).strip()
            or DEFAULT_RTSP_CAPTURE_OPTIONS,
            alpr_capture_fps=max(0.1, float(os.getenv("ALPR_CAPTURE_FPS", str(DEFAULT_ALPR_CAPTURE_FPS)))),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            telegram_alert_images=max(0, int(os.getenv("TELEGRAM_ALERT_IMAGES", "3"))),
        )


@dataclass
class CandidateFrame:
    frame: Any
    timestamp: float
    motion_area: int
    sharpness: float
    jpeg_bytes: Optional[bytes] = None
    source: str = "capture"


@dataclass
class MotionZone:
    zone_id: str
    label: str
    roi: Tuple[float, float, float, float]
    enabled: bool
    use_fast_alpr: bool
    color_hex: str
    fill_rgba: str
    overlay_bgr: Tuple[int, int, int]


@dataclass
class Event:
    started_at: float
    trigger_count: int
    frames: List[Tuple[float, Any]]
    candidates: List[CandidateFrame]
    last_motion_at: float
    last_frame_at: float
    frames_since_motion: int
    zones_triggered: set[str]


@dataclass
class PlateDetection:
    plate: str
    confidence: float
    source: str
    image_relative_path: str
    event_name: str
    detected_at_epoch: float


@dataclass
class AlprCaptureSession:
    started_at: float
    stop_event: threading.Event
    thread: threading.Thread
    frames: Deque[CandidateFrame]


@dataclass
class VideoRecording:
    started_at: float
    ends_at: float
    started_from_zone_ids: set[str]
    output_path: Path
    temp_output_path: Path
    writer: Any
    last_written_at: float
    frame_size: Tuple[int, int]
    fps: float


class FfmpegVideoWriter:
    def __init__(self, output_path: Path, fps: float, frame_size: Tuple[int, int], threads: int) -> None:
        width, height = frame_size
        self.output_path = output_path
        self.frame_size = frame_size
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.3f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-threads",
                str(threads),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self.process.poll() is None and self.process.stdin is not None

    def write(self, frame) -> None:
        if not self.isOpened() or self.process.stdin is None:
            raise RuntimeError(f"ffmpeg video writer is not open for {self.output_path}")
        self.process.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
            self.process.stdin = None
        stderr = self.process.stderr.read() if self.process.stderr else b""
        self.process.wait()
        if self.process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to write {self.output_path}: {message}")


class QueuedVideoWriter:
    def __init__(self, writer: Any, output_path: Path, fps: float) -> None:
        self.writer = writer
        self.output_path = output_path
        self.queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=max(30, int(fps * VIDEO_WRITER_QUEUE_SECONDS)))
        self.error: Optional[Exception] = None
        self.dropped_frames = 0
        self.closed = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._write_loop, name=f"video-writer-{output_path.name}", daemon=True)
        self.thread.start()

    def isOpened(self) -> bool:
        return self.writer.isOpened()

    def write(self, frame) -> None:
        with self.lock:
            if self.error:
                raise RuntimeError(f"video writer failed for {self.output_path}: {self.error}")
            if self.closed:
                raise RuntimeError(f"video writer is closed for {self.output_path}")
        frame_copy = frame.copy()
        try:
            self.queue.put_nowait(frame_copy)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass
            with self.lock:
                self.dropped_frames += 1
            self.queue.put_nowait(frame_copy)

    def release(self) -> None:
        with self.lock:
            self.closed = True
        while True:
            try:
                self.queue.put(None, timeout=0.5)
                break
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    with self.lock:
                        self.dropped_frames += 1
                except queue.Empty:
                    pass
                if not self.thread.is_alive():
                    break
        self.thread.join()
        with self.lock:
            error = self.error
            dropped_frames = self.dropped_frames
        if error:
            raise RuntimeError(f"video writer failed for {self.output_path}: {error}") from error
        if dropped_frames:
            logging.warning("Dropped %s frames while writing %s", dropped_frames, self.output_path)

    def _write_loop(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                try:
                    if frame is None:
                        return
                    self.writer.write(frame)
                finally:
                    self.queue.task_done()
        except Exception as exc:
            with self.lock:
                self.error = exc
        finally:
            try:
                self.writer.release()
            except Exception as exc:
                with self.lock:
                    if self.error is None:
                        self.error = exc


class OpenAlprClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()

    def recognize(self, jpeg_bytes: bytes) -> dict:
        params = {
            "secret_key": self.config.secret_key,
            "country": self.config.country,
            "recognize_vehicle": 1 if self.config.recognize_vehicle else 0,
            "topn": 5,
        }
        response = self.session.post(
            "https://api.openalpr.com/v3/recognize",
            params=params,
            files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


class RtspVehicleWatcher:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.runtime_config_path = self.config.event_output_dir.parent / "watcher-config.json"
        self.motion_zones = self._default_motion_zones()
        self._load_runtime_config()
        self.client = OpenAlprClient(config)
        self.prebuffer: Deque[Tuple[float, Any]] = deque()
        self.event: Optional[Event] = None
        self.motion_streak = 0
        self.last_motion_area = 0
        self.last_motion_box: Optional[Tuple[int, int, int, int]] = None
        self.last_triggered_zone_ids: set[str] = set()
        self.last_zone_area_by_id: dict[str, int] = {}
        self.latest_motion_status = "Motion zones: waiting for activity"
        self.video_recording: Optional[VideoRecording] = None
        self.alpr_capture_session: Optional[AlprCaptureSession] = None
        self.alpr_capture_lock = threading.Lock()
        self.frame_index = 0
        self.fps_guess = 12.0
        self.capture = None
        self.background = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=36, detectShadows=False)
        self.config.event_output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_lock = threading.Condition()
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_clean_frame_jpeg: Optional[bytes] = None
        self.latest_plate_zoom_jpeg: Optional[bytes] = None
        self.latest_frame_version = 0
        self.latest_stream_update_at = 0.0
        self.last_stream_encode_enqueued_at = 0.0
        self.stream_encode_queue: queue.Queue[Optional[Tuple[Any, Any, float]]] = queue.Queue(maxsize=1)
        self.stream_encoder_stop = threading.Event()
        self.stream_encoder_thread = threading.Thread(
            target=self._stream_encoder_loop,
            name="stream-jpeg-encoder",
            daemon=True,
        )
        self.stream_encoder_thread.start()
        self.stats_lock = threading.Lock()
        self.capture_started_at = time.time()
        self.capture_frame_times: Deque[float] = deque(maxlen=300)
        self.processing_frame_times: Deque[float] = deque(maxlen=300)
        self.stream_frame_times: Deque[float] = deque(maxlen=300)
        self.capture_gaps: Deque[float] = deque(maxlen=60)
        self.capture_frame_total = 0
        self.processing_frame_total = 0
        self.stream_frame_total = 0
        self.capture_gap_total = 0
        self.estimated_missed_frames = 0
        self.read_failure_total = 0
        self.capture_reconnect_total = 0
        self.max_capture_gap_seconds = 0.0
        self.last_capture_time: Optional[float] = None
        self.http_server: Optional[ThreadingHTTPServer] = None
        self.http_thread: Optional[threading.Thread] = None
        self._cleanup_stale_video_recordings()

    def _default_motion_zones(self) -> List[MotionZone]:
        primary_roi = self.config.roi or (0.05, 0.35, 0.95, 0.95)
        return [
            MotionZone(
                zone_id="yellow",
                label="Yellow zone",
                roi=primary_roi,
                enabled=True,
                use_fast_alpr=True,
                color_hex="#facc15",
                fill_rgba="rgba(250,204,21,0.14)",
                overlay_bgr=(0, 200, 255),
            ),
            MotionZone(
                zone_id="purple",
                label="Purple zone",
                roi=(0.55, 0.18, 0.95, 0.52),
                enabled=False,
                use_fast_alpr=False,
                color_hex="#c084fc",
                fill_rgba="rgba(192,132,252,0.16)",
                overlay_bgr=(250, 120, 170),
            ),
        ]

    def _find_motion_zone(self, zone_id: str) -> Optional[MotionZone]:
        for zone in self.motion_zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def _sync_primary_zone_to_config(self) -> None:
        primary_zone = self._find_motion_zone("yellow")
        if primary_zone:
            self.config.roi = primary_zone.roi

    def _load_runtime_config(self) -> None:
        try:
            should_resave = False
            if not self.runtime_config_path.exists():
                self._sync_primary_zone_to_config()
                return
            payload = json.loads(self.runtime_config_path.read_text())
            zones_payload = payload.get("zones")
            if isinstance(zones_payload, list):
                for zone_payload in zones_payload:
                    if not isinstance(zone_payload, dict):
                        continue
                    zone = self._find_motion_zone(str(zone_payload.get("id") or "").strip())
                    if not zone:
                        continue
                    roi_values = zone_payload.get("roi")
                    if isinstance(roi_values, list) and len(roi_values) == 4:
                        zone.roi = parse_normalized_roi(
                            ",".join(str(float(value)) for value in roi_values),
                            f"{zone.label} ROI",
                        ) or zone.roi
                    zone.enabled = bool(zone_payload.get("enabled", zone.enabled))
                    zone.use_fast_alpr = bool(zone_payload.get("use_fast_alpr", zone.use_fast_alpr))
            roi_value = payload.get("roi")
            if isinstance(roi_value, str):
                primary_zone = self._find_motion_zone("yellow")
                if primary_zone:
                    primary_zone.roi = parse_normalized_roi(roi_value, "ROI") or primary_zone.roi
            min_motion_area = payload.get("min_motion_area")
            if min_motion_area is not None:
                clamped_min_motion_area = max(MIN_ALLOWED_MOTION_AREA, int(min_motion_area))
                should_resave = clamped_min_motion_area != int(min_motion_area)
                self.config.min_motion_area = clamped_min_motion_area
            self._sync_primary_zone_to_config()
            if should_resave:
                self._save_runtime_config()
        except Exception:
            logging.exception("Failed to load runtime watcher config")

    def _save_runtime_config(self) -> None:
        self.runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
        self._sync_primary_zone_to_config()
        roi_value = ""
        if self.config.roi:
            roi_value = ",".join(f"{value:.6f}" for value in self.config.roi)
        self.runtime_config_path.write_text(
            json.dumps(
                {
                    "roi": roi_value,
                    "zones": [
                        {
                            "id": zone.zone_id,
                            "roi": [float(value) for value in zone.roi],
                            "enabled": zone.enabled,
                            "use_fast_alpr": zone.use_fast_alpr,
                        }
                        for zone in self.motion_zones
                    ],
                    "min_motion_area": int(self.config.min_motion_area),
                },
                indent=2,
            )
        )

    def run(self) -> None:
        self._start_http_server()
        while True:
            try:
                self._run_capture_loop()
            except KeyboardInterrupt:
                logging.info("Stopping watcher")
                break
            except Exception:
                with self.stats_lock:
                    self.capture_reconnect_total += 1
                logging.exception("Capture loop failed; reconnecting in 5 seconds")
                time.sleep(5)
        self._stop_http_server()
        self._stop_stream_encoder()

    def _run_capture_loop(self) -> None:
        logging.info("Connecting to RTSP stream")
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            self.config.rtsp_capture_options,
        )
        self.capture = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open RTSP stream")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.capture_buffer_size)

        native_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if native_fps and native_fps > 1:
            self.fps_guess = native_fps
        logging.info(
            "Connected. Camera FPS estimate: %.2f; processing width: %s",
            self.fps_guess,
            "native" if self.config.frame_width <= 0 else self.config.frame_width,
        )

        capture_stop = threading.Event()
        capture_error: list[Exception] = []
        frame_queue: queue.Queue[Tuple[float, Any]] = queue.Queue(maxsize=1)

        def replace_captured_frame(payload: Tuple[float, Any]) -> None:
            try:
                frame_queue.put_nowait(payload)
                return
            except queue.Full:
                pass
            try:
                frame_queue.get_nowait()
                frame_queue.task_done()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(payload)
            except queue.Full:
                pass

        def capture_reader() -> None:
            try:
                while not capture_stop.is_set():
                    ok, captured_frame = self.capture.read()
                    if not ok or captured_frame is None:
                        with self.stats_lock:
                            self.read_failure_total += 1
                        raise RuntimeError("Failed to read frame from RTSP stream")
                    timestamp = time.time()
                    self._record_capture_frame(timestamp)
                    replace_captured_frame((timestamp, self._resize_frame(captured_frame)))
            except Exception as exc:
                capture_error.append(exc)

        capture_thread = threading.Thread(target=capture_reader, name="rtsp-reader", daemon=True)
        capture_thread.start()

        try:
            while True:
                if capture_error:
                    raise RuntimeError("RTSP reader failed") from capture_error[0]
                try:
                    timestamp, frame = frame_queue.get(timeout=2.0)
                    frame_queue.task_done()
                except queue.Empty:
                    if capture_error:
                        raise RuntimeError("RTSP reader failed") from capture_error[0]
                    raise RuntimeError("Timed out waiting for frame from RTSP reader")

                self._push_prebuffer(timestamp, frame)

                process_this_frame = self.frame_index % self.config.process_every_n_frames == 0
                motion_area = self.last_motion_area
                had_motion = False
                triggered_zone_ids: set[str] = set()
                overlay = self._draw_monitor_overlays(frame.copy())

                if process_this_frame:
                    motion_area, raw_motion, best_box, overlay, triggered_zone_ids, zone_area_by_id = self._detect_motion(frame)
                    if raw_motion:
                        self.motion_streak = min(self.config.min_consecutive_hits, self.motion_streak + 1)
                        self.last_triggered_zone_ids = set(triggered_zone_ids)
                    else:
                        self.motion_streak = max(0, self.motion_streak - 1)
                        if self.motion_streak == 0:
                            self.last_triggered_zone_ids = set()
                    had_motion = self.motion_streak >= self.config.min_consecutive_hits
                    self.last_motion_area = motion_area
                    self.last_motion_box = best_box
                    self.last_zone_area_by_id = dict(zone_area_by_id)
                    overlay = self._annotate_motion_overlay(
                        overlay,
                        motion_area,
                        best_box,
                        had_motion,
                        triggered_zone_ids if triggered_zone_ids else self.last_triggered_zone_ids,
                        zone_area_by_id,
                    )
                    if self.config.debug_windows:
                        cv2.imshow("watcher", overlay)
                        cv2.waitKey(1)
                else:
                    had_motion = self.motion_streak >= self.config.min_consecutive_hits
                    overlay = self._annotate_motion_overlay(
                        overlay,
                        motion_area,
                        self.last_motion_box,
                        had_motion,
                        self.last_triggered_zone_ids,
                        self.last_zone_area_by_id,
                    )

                self._update_latest_frames(overlay, frame)

                if had_motion:
                    if not triggered_zone_ids:
                        triggered_zone_ids = set(self.last_triggered_zone_ids)
                    self._start_alpr_capture_session(timestamp)
                    if self.video_recording is None:
                        self._start_video_recording(timestamp, triggered_zone_ids)
                    self._on_motion(timestamp, frame, motion_area, triggered_zone_ids)
                    if self._event_exceeded_max_duration(timestamp):
                        self._finalize_event("max duration reached while motion remained active")
                elif self.event:
                    self._append_event_frame(timestamp, frame, count_as_postbuffer=True)
                    if self._event_ready_to_finalize(timestamp):
                        self._finalize_event("motion idle and postbuffer complete")

                self._append_video_recording_frame(timestamp, frame)

                self.frame_index += 1
                self._record_processing_frame()
        finally:
            capture_stop.set()
            capture_thread.join(timeout=2.0)
            self._stop_alpr_capture_session()
            self._stop_video_recording()
            self.capture.release()

    def _resize_frame(self, frame):
        if self.config.frame_width <= 0:
            return frame
        height, width = frame.shape[:2]
        if width <= self.config.frame_width:
            return frame
        scale = self.config.frame_width / float(width)
        resized = cv2.resize(frame, (self.config.frame_width, int(height * scale)))
        return resized

    def _rolling_fps(self, frame_times: Deque[float]) -> float:
        if len(frame_times) < 2:
            return 0.0
        elapsed = frame_times[-1] - frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(frame_times) - 1) / elapsed

    def _record_capture_frame(self, timestamp: float) -> None:
        with self.stats_lock:
            if self.last_capture_time is not None:
                gap = timestamp - self.last_capture_time
                expected_interval = 1.0 / max(1.0, self.fps_guess)
                if gap > max(1.0, expected_interval * 3.0):
                    self.capture_gap_total += 1
                    self.capture_gaps.append(gap)
                    self.max_capture_gap_seconds = max(self.max_capture_gap_seconds, gap)
                    self.estimated_missed_frames += max(0, int(round(gap / expected_interval)) - 1)
            self.last_capture_time = timestamp
            self.capture_frame_total += 1
            self.capture_frame_times.append(timestamp)

    def _record_processing_frame(self) -> None:
        with self.stats_lock:
            self.processing_frame_total += 1
            self.processing_frame_times.append(time.time())

    def _record_stream_frame(self, timestamp: float) -> None:
        with self.stats_lock:
            self.stream_frame_total += 1
            self.stream_frame_times.append(timestamp)

    def _stats_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self.stats_lock:
            capture_times = deque(self.capture_frame_times)
            processing_times = deque(self.processing_frame_times)
            stream_times = deque(self.stream_frame_times)
            recent_gaps = list(self.capture_gaps)
            last_capture_age = None
            if self.last_capture_time is not None:
                last_capture_age = max(0.0, now - self.last_capture_time)
            recording = self.video_recording
            return {
                "uptime_seconds": max(0.0, now - self.capture_started_at),
                "camera_fps_estimate": float(self.fps_guess),
                "capture_fps": self._rolling_fps(capture_times),
                "processing_fps": self._rolling_fps(processing_times),
                "stream_fps": self._rolling_fps(stream_times),
                "configured_stream_fps": float(self.config.stream_fps),
                "capture_frame_total": self.capture_frame_total,
                "processing_frame_total": self.processing_frame_total,
                "stream_frame_total": self.stream_frame_total,
                "capture_gap_total": self.capture_gap_total,
                "recent_capture_gap_count": len(recent_gaps),
                "recent_max_capture_gap_seconds": max(recent_gaps) if recent_gaps else 0.0,
                "max_capture_gap_seconds": self.max_capture_gap_seconds,
                "estimated_missed_frames": self.estimated_missed_frames,
                "read_failure_total": self.read_failure_total,
                "capture_reconnect_total": self.capture_reconnect_total,
                "last_capture_age_seconds": last_capture_age,
                "frame_width": int(self.config.frame_width),
                "capture_buffer_size": int(self.config.capture_buffer_size),
                "process_every_n_frames": int(self.config.process_every_n_frames),
                "recording_active": recording is not None,
                "recording_fps": float(recording.fps) if recording else 0.0,
                "motion_status": self.latest_motion_status,
            }

    def _push_prebuffer(self, timestamp: float, frame) -> None:
        self.prebuffer.append((timestamp, frame.copy()))
        max_frames = max(1, self._prebuffer_frame_limit())
        while len(self.prebuffer) > max_frames:
            self.prebuffer.popleft()

    def _video_output_dir(self) -> Path:
        path = self.config.event_output_dir / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _video_temp_dir(self) -> Path:
        path = self._video_output_dir() / "recording-tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cleanup_stale_video_recordings(self) -> None:
        temp_dir = self._video_temp_dir()
        for temp_path in temp_dir.glob("*.mp4"):
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                logging.exception("Failed to remove stale temporary video %s", temp_path)

    def _prepare_video_frame(self, frame, frame_size: Tuple[int, int]):
        width, height = frame_size
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        return np.ascontiguousarray(frame)

    def _start_video_recording(self, timestamp: float, triggered_zone_ids: set[str]) -> None:
        if self.video_recording is not None:
            return
        video_dir = self._video_output_dir()
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_path = video_dir / f"{self.config.camera_name}_{stamp}.mp4"
        temp_output_path = self._video_temp_dir() / output_path.name
        prebuffer_frames = [
            (frame_timestamp, frame_copy)
            for frame_timestamp, frame_copy in self.prebuffer
            if frame_timestamp >= timestamp - VIDEO_PREBUFFER_SECONDS
        ]
        if len(prebuffer_frames) > MAX_VIDEO_PREBUFFER_FRAMES:
            last_index = len(prebuffer_frames) - 1
            prebuffer_frames = [
                prebuffer_frames[round(index * last_index / max(1, MAX_VIDEO_PREBUFFER_FRAMES - 1))]
                for index in range(MAX_VIDEO_PREBUFFER_FRAMES)
            ]
        if prebuffer_frames:
            sample_frame = prebuffer_frames[0][1]
        elif self.prebuffer:
            sample_frame = self.prebuffer[-1][1]
        else:
            return
        height, width = sample_frame.shape[:2]
        frame_size = (width, height)
        with self.stats_lock:
            processing_fps = self._rolling_fps(deque(self.processing_frame_times))
        fps = max(5.0, min(MAX_RECORDING_FPS, processing_fps or self.fps_guess))
        if shutil.which("ffmpeg"):
            raw_writer = FfmpegVideoWriter(temp_output_path, fps, frame_size, self.config.ffmpeg_threads)
        else:
            logging.warning("ffmpeg is unavailable; falling back to OpenCV mp4v writer")
            raw_writer = cv2.VideoWriter(str(temp_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
        writer = QueuedVideoWriter(raw_writer, temp_output_path, fps)
        if not writer.isOpened():
            try:
                writer.release()
            except Exception:
                pass
            temp_output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Unable to open video writer for {temp_output_path}")
        last_written_at = 0.0
        min_frame_interval = 1.0 / fps
        try:
            for frame_timestamp, buffered_frame in prebuffer_frames:
                if last_written_at and frame_timestamp - last_written_at < min_frame_interval:
                    continue
                prepared_frame = self._prepare_video_frame(buffered_frame, frame_size)
                writer.write(prepared_frame)
                last_written_at = frame_timestamp
        except Exception:
            writer.release()
            temp_output_path.unlink(missing_ok=True)
            raise
        self.video_recording = VideoRecording(
            started_at=timestamp,
            ends_at=timestamp + VIDEO_RECORDING_SECONDS,
            started_from_zone_ids=set(triggered_zone_ids),
            output_path=output_path,
            temp_output_path=temp_output_path,
            writer=writer,
            last_written_at=last_written_at,
            frame_size=frame_size,
            fps=fps,
        )
        logging.info(
            "Started 3-minute video recording at %s for zones=%s",
            output_path,
            ",".join(sorted(triggered_zone_ids)) or "unknown",
        )

    def _append_video_recording_frame(self, timestamp: float, frame) -> None:
        recording = self.video_recording
        if recording is None:
            return
        if timestamp > recording.ends_at:
            self._stop_video_recording()
            return
        min_frame_interval = 1.0 / recording.fps
        if timestamp <= recording.last_written_at or timestamp - recording.last_written_at < min_frame_interval:
            return
        prepared_frame = self._prepare_video_frame(frame, recording.frame_size)
        recording.writer.write(prepared_frame)
        recording.last_written_at = timestamp

    def _stop_video_recording(self) -> None:
        recording = self.video_recording
        if recording is None:
            return
        release_error: Optional[Exception] = None
        try:
            recording.writer.release()
        except Exception as exc:
            release_error = exc
        finally:
            self.video_recording = None
        if release_error:
            recording.temp_output_path.unlink(missing_ok=True)
            logging.error(
                "Failed to close video recording %s",
                recording.output_path,
                exc_info=(type(release_error), release_error, release_error.__traceback__),
            )
            return
        try:
            recording.temp_output_path.replace(recording.output_path)
            self._write_video_metadata(recording)
        except Exception:
            recording.temp_output_path.unlink(missing_ok=True)
            logging.exception("Failed to finalize video recording %s", recording.output_path)
            return
        logging.info("Saved video recording to %s", recording.output_path)

    def _start_alpr_capture_session(self, timestamp: float) -> None:
        if not self.config.alpr_rtsp_url or self.config.alpr_rtsp_url == self.config.rtsp_url:
            return
        with self.alpr_capture_lock:
            if self.alpr_capture_session is not None:
                return
            stop_event = threading.Event()
            frames: Deque[CandidateFrame] = deque(maxlen=max(self.config.upload_top_frames * 2, 30))
            thread = threading.Thread(
                target=self._run_alpr_capture_session,
                args=(stop_event, frames),
                name="alpr-rtsp-101-sampler",
                daemon=True,
            )
            self.alpr_capture_session = AlprCaptureSession(
                started_at=timestamp,
                stop_event=stop_event,
                thread=thread,
                frames=frames,
            )
            thread.start()
        logging.info("Started high-resolution ALPR sampler from %s", self.config.alpr_rtsp_url)

    def _run_alpr_capture_session(self, stop_event: threading.Event, frames: Deque[CandidateFrame]) -> None:
        try:
            min_interval = 1.0 / self.config.alpr_capture_fps
            last_sample_at = 0.0
            while not stop_event.is_set():
                timestamp = time.time()
                if timestamp - last_sample_at < min_interval:
                    stop_event.wait(min(0.1, min_interval))
                    continue
                frame = self._capture_single_alpr_frame()
                crop = self._plate_crop(frame)
                jpeg_bytes = self._encode_jpeg(crop)
                frames.append(
                    CandidateFrame(
                        frame=None,
                        timestamp=timestamp,
                        motion_area=0,
                        sharpness=self._compute_sharpness(crop),
                        jpeg_bytes=jpeg_bytes,
                        source="alpr-rtsp",
                    )
                )
                last_sample_at = timestamp
        except Exception:
            logging.exception("High-resolution ALPR sampler stopped")

    def _capture_single_alpr_frame(self):
        capture = cv2.VideoCapture(self.config.alpr_rtsp_url, cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                raise RuntimeError("Unable to open ALPR RTSP stream")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read frame from ALPR RTSP stream")
            return frame
        finally:
            capture.release()

    def _stop_alpr_capture_session(self) -> List[CandidateFrame]:
        with self.alpr_capture_lock:
            session = self.alpr_capture_session
            self.alpr_capture_session = None
        if session is None:
            return []
        session.stop_event.set()
        session.thread.join(timeout=3.0)
        frames = list(session.frames)
        logging.info("Stopped high-resolution ALPR sampler with %s frames", len(frames))
        return frames

    def _write_video_metadata(self, recording: VideoRecording) -> None:
        metadata_path = recording.output_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "camera_name": self.config.camera_name,
                    "video_path": str(recording.output_path),
                    "started_at_epoch": recording.started_at,
                    "ends_at_epoch": recording.ends_at,
                    "zone_ids": sorted(recording.started_from_zone_ids),
                    "prebuffer_seconds": VIDEO_PREBUFFER_SECONDS,
                    "recording_seconds": VIDEO_RECORDING_SECONDS,
                },
                indent=2,
            )
        )

    def _prebuffer_frame_limit(self) -> int:
        if self.config.prebuffer_frames > 0:
            return max(self.config.prebuffer_frames, int(self.fps_guess * VIDEO_PREBUFFER_SECONDS))
        return int(self.fps_guess * max(self.config.prebuffer_seconds, VIDEO_PREBUFFER_SECONDS))

    def _postbuffer_frame_limit(self) -> int:
        if self.config.postbuffer_frames > 0:
            return self.config.postbuffer_frames
        return int(self.fps_guess * self.config.postbuffer_seconds)

    def _normalized_roi_bounds(
        self, frame, roi: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        if not roi:
            return 0, 0, width, height
        x1, y1, x2, y2 = roi
        return (
            max(0, min(width - 1, int(x1 * width))),
            max(0, min(height - 1, int(y1 * height))),
            max(1, min(width, int(x2 * width))),
            max(1, min(height, int(y2 * height))),
        )

    def _roi_bounds(self, frame) -> Tuple[int, int, int, int]:
        primary_zone = self._find_motion_zone("yellow")
        roi = primary_zone.roi if primary_zone else self.config.roi
        return self._normalized_roi_bounds(frame, roi)

    def _zone_bounds(self, frame, zone: MotionZone) -> Tuple[int, int, int, int]:
        return self._normalized_roi_bounds(frame, zone.roi)

    def _plate_roi_bounds(self, frame) -> Tuple[int, int, int, int]:
        effective_plate_roi = self.config.plate_roi or self.config.roi
        return self._normalized_roi_bounds(frame, effective_plate_roi)

    def _draw_monitor_overlays(self, frame):
        for zone in self.motion_zones:
            if not zone.enabled:
                continue
            zone_x1, zone_y1, zone_x2, zone_y2 = self._zone_bounds(frame, zone)
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), zone.overlay_bgr, 2)
            cv2.putText(
                frame,
                zone.label.lower(),
                (zone_x1 + 6, max(24, zone_y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                zone.overlay_bgr,
                2,
            )
        if self.config.plate_roi:
            plate_x1, plate_y1, plate_x2, plate_y2 = self._plate_roi_bounds(frame)
            cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 140, 0), 2)
            cv2.putText(
                frame,
                "plate zone",
                (plate_x1 + 6, min(frame.shape[0] - 10, plate_y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 140, 0),
                2,
            )
        return frame

    def _detect_motion(self, frame) -> Tuple[int, bool, Optional[Tuple[int, int, int, int]], Any, set[str], dict[str, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.background.apply(gray)
        _, mask = cv2.threshold(mask, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        total_area = 0
        best_box = None
        triggered_zone_ids: set[str] = set()
        zone_area_by_id: dict[str, int] = {}
        for zone in self.motion_zones:
            if not zone.enabled:
                continue
            x1, y1, x2, y2 = self._zone_bounds(frame, zone)
            zone_mask = mask[y1:y2, x1:x2]
            contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            zone_total_area = 0
            zone_best_box = None
            for contour in contours:
                area = int(cv2.contourArea(contour))
                if area < self.config.min_motion_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 12 or h < 12:
                    continue
                zone_total_area += area
                if zone_best_box is None or area > zone_best_box[0]:
                    zone_best_box = (area, x, y, w, h)
            zone_area_by_id[zone.zone_id] = zone_total_area
            if zone_total_area >= self.config.min_motion_area:
                triggered_zone_ids.add(zone.zone_id)
                total_area += zone_total_area
                if zone_best_box and (best_box is None or zone_best_box[0] > best_box[0]):
                    best_box = (
                        zone_best_box[0],
                        x1 + zone_best_box[1],
                        y1 + zone_best_box[2],
                        zone_best_box[3],
                        zone_best_box[4],
                    )

        overlay = self._draw_monitor_overlays(frame.copy())
        absolute_best_box = None
        if best_box:
            _, x, y, w, h = best_box
            absolute_best_box = (x, y, x + w, y + h)

        had_motion = bool(triggered_zone_ids)
        return total_area, had_motion, absolute_best_box, overlay, triggered_zone_ids, zone_area_by_id

    def _annotate_motion_overlay(
        self,
        overlay,
        motion_area: int,
        best_box: Optional[Tuple[int, int, int, int]],
        confirmed_motion: bool,
        triggered_zone_ids: set[str],
        zone_area_by_id: dict[str, int],
    ):
        if best_box:
            box_color = (0, 255, 0) if confirmed_motion else (0, 200, 255)
            cv2.rectangle(overlay, (best_box[0], best_box[1]), (best_box[2], best_box[3]), box_color, 2)
        status = "confirmed" if confirmed_motion else f"warming {min(self.motion_streak, self.config.min_consecutive_hits)}/{self.config.min_consecutive_hits}"
        zones_text = ",".join(sorted(triggered_zone_ids)) if triggered_zone_ids else "none"
        per_zone = " ".join(
            f"{zone.zone_id}={int(zone_area_by_id.get(zone.zone_id, 0))}"
            for zone in self.motion_zones
            if zone.enabled
        )
        label = f"zones={zones_text} total={motion_area} status={status}"
        detail_label = per_zone or "no-zones"
        self.latest_motion_status = f"{label} | {detail_label}"
        return overlay

    def _on_motion(self, timestamp: float, frame, motion_area: int, triggered_zone_ids: set[str]) -> None:
        sharpness = self._compute_sharpness(frame)
        candidate = CandidateFrame(frame=frame.copy(), timestamp=timestamp, motion_area=motion_area, sharpness=sharpness)

        if self.event is None:
            self.event = Event(
                started_at=timestamp,
                trigger_count=1,
                frames=list(self.prebuffer),
                candidates=[candidate],
                last_motion_at=timestamp,
                last_frame_at=timestamp,
                frames_since_motion=0,
                zones_triggered=set(triggered_zone_ids),
            )
            logging.info("Motion started; opening event")
            self._append_event_frame(timestamp, frame, count_as_postbuffer=False)
            return

        self.event.trigger_count += 1
        self.event.last_motion_at = timestamp
        self.event.frames_since_motion = 0
        self.event.zones_triggered.update(triggered_zone_ids)
        self.event.candidates.append(candidate)
        self._append_event_frame(timestamp, frame, count_as_postbuffer=False)

    def _append_event_frame(self, timestamp: float, frame, count_as_postbuffer: bool) -> None:
        if not self.event:
            return
        self.event.frames.append((timestamp, frame.copy()))
        self.event.last_frame_at = timestamp
        if count_as_postbuffer:
            self.event.frames_since_motion += 1

    def _event_ready_to_finalize(self, timestamp: float) -> bool:
        if not self.event:
            return False
        enough_idle_time = timestamp - self.event.last_motion_at >= self.config.event_idle_seconds
        enough_post_frames = self.event.frames_since_motion >= self._postbuffer_frame_limit()
        return enough_idle_time and enough_post_frames

    def _event_exceeded_max_duration(self, timestamp: float) -> bool:
        if not self.event:
            return False
        return timestamp - self.event.started_at >= self.config.event_max_seconds

    def _run_detection_pipeline(
        self,
        frame,
        event_dir: Path,
        base_name: str,
        event_name: str,
        detected_at_epoch: float,
        enable_alpr: bool = True,
        jpeg_bytes: Optional[bytes] = None,
    ) -> dict[str, Any]:
        frame_path = event_dir / f"{base_name}.jpg"
        json_path = event_dir / f"{base_name}.json"
        if jpeg_bytes is None:
            jpeg_bytes = self._encode_jpeg(self._plate_crop(frame))
        frame_path.write_bytes(jpeg_bytes)
        relative_frame_path = frame_path.relative_to(self.config.event_output_dir).as_posix()

        local_result = None
        fast_alpr_results = []
        openalpr_results = []
        plate_detections: List[PlateDetection] = []
        openalpr_skipped_reason: Optional[str] = None
        fast_alpr_error: Optional[str] = None
        openalpr_error: Optional[str] = None

        if not enable_alpr:
            openalpr_skipped_reason = "zone policy saved images only"
        elif self.config.fast_alpr_url:
            try:
                local_result = self._recognize_fast_alpr(jpeg_bytes)
                (event_dir / f"{base_name}.fast_alpr.json").write_text(json.dumps(local_result, indent=2))
                fast_alpr_results.append(local_result)
                plate_detections.extend(
                    self._extract_fast_alpr_detections(
                        local_result,
                        relative_frame_path,
                        event_name,
                        detected_at_epoch,
                    )
                )
            except Exception as exc:
                fast_alpr_error = str(exc)
                (event_dir / f"{base_name}.fast_alpr.json").write_text(
                    json.dumps({"error": fast_alpr_error}, indent=2)
                )
                logging.exception("Failed to analyze %s with fast-alpr", base_name)
        elif enable_alpr:
            (event_dir / f"{base_name}.fast_alpr.json").write_text(
                json.dumps({"skipped": "FAST_ALPR_URL is not set"}, indent=2)
            )

        if not enable_alpr:
            pass
        elif self.config.fast_alpr_url and not self._fast_alpr_has_confident_plate(local_result):
            openalpr_skipped_reason = "fast-alpr found no confident plate"
        elif not self.config.openalpr_enabled:
            openalpr_skipped_reason = "OPENALPR_SECRET_KEY is not set"
        else:
            try:
                result = self.client.recognize(jpeg_bytes)
                json_path.write_text(json.dumps(result, indent=2))
                openalpr_results.append(result)
                plate_detections.extend(
                    self._extract_openalpr_detections(
                        result,
                        relative_frame_path,
                        event_name,
                        detected_at_epoch,
                    )
                )
            except Exception as exc:
                openalpr_error = str(exc)
                json_path.write_text(json.dumps({"error": openalpr_error}, indent=2))
                logging.exception("Failed to upload %s", base_name)
        if openalpr_skipped_reason:
            json_path.write_text(json.dumps({"skipped": openalpr_skipped_reason}, indent=2))

        return {
            "frame_path": frame_path,
            "relative_frame_path": relative_frame_path,
            "fast_alpr_results": fast_alpr_results,
            "openalpr_results": openalpr_results,
            "plate_detections": plate_detections,
            "openalpr_skipped_reason": openalpr_skipped_reason,
            "fast_alpr_error": fast_alpr_error,
            "openalpr_error": openalpr_error,
        }

    def _finalize_event(self, reason: str) -> None:
        if not self.event:
            return
        event = self.event
        self.event = None
        logging.info("Closing event: %s", reason)
        thread = threading.Thread(
            target=self._save_finalized_event,
            args=(event, reason),
            daemon=True,
        )
        thread.start()

    def _save_finalized_event(self, event: Event, reason: str) -> None:
        try:
            self._save_finalized_event_unchecked(event, reason)
        except Exception:
            logging.exception("Failed to save finalized event: %s", reason)

    def _save_finalized_event_unchecked(self, event: Event, reason: str) -> None:
        if event.trigger_count < self.config.min_consecutive_hits:
            logging.info("Discarded event with only %s motion hits", event.trigger_count)
            self._stop_alpr_capture_session()
            return

        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        event_dir = self.config.event_output_dir / f"{self.config.camera_name}_{stamp}"
        event_dir.mkdir(parents=True, exist_ok=True)
        high_res_candidates = self._stop_alpr_capture_session()
        selected = self._select_best_frames(event, preferred_candidates=high_res_candidates)
        openalpr_results = []
        fast_alpr_results = []
        plate_detections: List[PlateDetection] = []
        saved_frame_paths: List[Path] = []
        alpr_enabled = self._event_uses_fast_alpr(event)
        for index, candidate in enumerate(selected, start=1):
            pipeline = self._run_detection_pipeline(
                candidate.frame,
                event_dir,
                f"frame_{index:02d}",
                event_dir.name,
                candidate.timestamp,
                enable_alpr=alpr_enabled,
                jpeg_bytes=candidate.jpeg_bytes,
            )
            fast_alpr_results.extend(pipeline["fast_alpr_results"])
            openalpr_results.extend(pipeline["openalpr_results"])
            plate_detections.extend(pipeline["plate_detections"])
            saved_frame_paths.append(pipeline["frame_path"])
            if pipeline["openalpr_skipped_reason"]:
                logging.info(
                    "Skipping OpenALPR upload for frame %s because %s",
                    index,
                    pipeline["openalpr_skipped_reason"],
                )
            elif pipeline["openalpr_error"]:
                logging.warning("OpenALPR upload failed for frame %s: %s", index, pipeline["openalpr_error"])
            elif pipeline["openalpr_results"]:
                logging.info("Uploaded frame %s for ALPR analysis", index)

        summary = {
            "camera_name": self.config.camera_name,
            "started_at_epoch": event.started_at,
            "ended_at_epoch": event.last_frame_at,
            "trigger_count": event.trigger_count,
            "saved_frames": len(selected),
            "high_res_alpr_frames": sum(1 for candidate in selected if candidate.source == "alpr-rtsp"),
            "clip_path": None,
            "triggered_zones": self._event_zone_summary(event),
            "event_policy": "fast-alpr" if alpr_enabled else "images-only",
            "fast_alpr_results_count": len(fast_alpr_results),
            "openalpr_results_count": len(openalpr_results),
            "plates": [detection.__dict__ for detection in plate_detections],
        }
        (event_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._send_telegram_alert(event_dir.name, summary, saved_frame_paths)
        self._prune_saved_images()
        logging.info("Event saved to %s", event_dir)

    def _select_best_frames(
        self, event: Event, preferred_candidates: Optional[List[CandidateFrame]] = None
    ) -> List[CandidateFrame]:
        if preferred_candidates:
            return self._select_timeline_candidates(preferred_candidates, self.config.upload_top_frames)
        candidates = [
            candidate for candidate in event.candidates if candidate.sharpness >= self.config.upload_min_sharpness
        ]
        if not candidates:
            candidates = list(event.candidates)
        if not candidates and event.frames:
            fallback_timestamp, fallback_frame = event.frames[-1]
            candidates = [
                CandidateFrame(
                    frame=fallback_frame.copy(),
                    timestamp=fallback_timestamp,
                    motion_area=0,
                    sharpness=self._compute_sharpness(fallback_frame),
                )
            ]

        selected: List[CandidateFrame] = []
        if event.frames:
            selected.extend(self._select_timeline_frames(event.frames, self.config.upload_top_frames))
        selected_timestamps = {candidate.timestamp for candidate in selected}
        ranked = sorted(
            candidates,
            key=lambda item: (item.motion_area, item.sharpness, item.timestamp),
            reverse=True,
        )
        for candidate in ranked:
            if candidate.timestamp in selected_timestamps:
                continue
            selected.append(candidate)
            selected_timestamps.add(candidate.timestamp)
            if len(selected) >= self.config.upload_top_frames:
                break
        if len(selected) < self.config.upload_top_frames:
            for frame_timestamp, frame in event.frames:
                if frame_timestamp in selected_timestamps:
                    continue
                selected.append(
                    CandidateFrame(
                        frame=frame.copy(),
                        timestamp=frame_timestamp,
                        motion_area=0,
                        sharpness=self._compute_sharpness(frame),
                    )
                )
                selected_timestamps.add(frame_timestamp)
                if len(selected) >= self.config.upload_top_frames:
                    break
        return sorted(selected, key=lambda item: item.timestamp)

    def _select_timeline_candidates(self, candidates: List[CandidateFrame], limit: int) -> List[CandidateFrame]:
        if not candidates or limit <= 0:
            return []
        sorted_candidates = sorted(candidates, key=lambda item: item.timestamp)
        if len(sorted_candidates) <= limit:
            return sorted_candidates
        selected = []
        last_index = len(sorted_candidates) - 1
        for index in range(limit):
            frame_index = round(index * last_index / max(1, limit - 1))
            selected.append(sorted_candidates[frame_index])
        return selected

    def _select_timeline_frames(self, frames: List[Tuple[float, Any]], limit: int) -> List[CandidateFrame]:
        if not frames or limit <= 0:
            return []
        if len(frames) <= limit:
            selected_frames = frames
        else:
            selected_frames = []
            last_index = len(frames) - 1
            for index in range(limit):
                frame_index = round(index * last_index / max(1, limit - 1))
                selected_frames.append(frames[frame_index])
        return [
            CandidateFrame(
                frame=frame.copy(),
                timestamp=frame_timestamp,
                motion_area=0,
                sharpness=self._compute_sharpness(frame),
            )
            for frame_timestamp, frame in selected_frames
        ]

    def _telegram_caption(self, event_name: str, summary: dict[str, Any]) -> str:
        plates = summary.get("plates") or []
        if plates:
            plate_text = ", ".join(
                f"{item.get('plate')} ({item.get('source', 'unknown')} {float(item.get('confidence') or 0.0):.2f})"
                for item in plates[:6]
            )
        else:
            plate_text = "none detected yet"
        zone_labels = []
        for item in summary.get("triggered_zones") or []:
            label = item.get("label") or item.get("id")
            if label:
                zone_labels.append(str(label))
        lines = [
            "ALPR motion alert",
            f"Camera: {summary.get('camera_name', self.config.camera_name)}",
            f"Event: {event_name}",
            f"Zones: {', '.join(zone_labels) if zone_labels else 'unknown'}",
            f"Saved frames: {summary.get('saved_frames', 0)}",
            f"High-res ALPR frames: {summary.get('high_res_alpr_frames', 0)}",
            f"Plates: {plate_text}",
        ]
        return "\n".join(lines)[:1000]

    def _send_telegram_alert(self, event_name: str, summary: dict[str, Any], image_paths: List[Path]) -> None:
        if not self.config.telegram_enabled:
            return
        chat_id = self._resolve_telegram_chat_id()
        if not chat_id:
            logging.warning("Telegram alert skipped: TELEGRAM_CHAT_ID is missing and no bot chat was found")
            return
        caption = self._telegram_caption(event_name, summary)
        existing_images = [path for path in image_paths if path.exists() and path.is_file()]
        if not existing_images:
            self._send_telegram_message(caption)
            return
        for index, image_path in enumerate(existing_images[: self.config.telegram_alert_images]):
            data = {"chat_id": chat_id}
            if index == 0:
                data["caption"] = caption
            try:
                with image_path.open("rb") as image_file:
                    response = requests.post(
                        f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendPhoto",
                        data=data,
                        files={"photo": (image_path.name, image_file, "image/jpeg")},
                        timeout=self.config.request_timeout_seconds,
                    )
                if not response.ok:
                    logging.warning(
                        "Telegram photo alert failed for %s: HTTP %s %s",
                        event_name,
                        response.status_code,
                        response.text[:300],
                    )
                    return
            except Exception:
                logging.exception("Telegram photo alert failed for %s", event_name)
                return

    def _resolve_telegram_chat_id(self, allow_discovery: bool = False) -> Optional[str]:
        if self.config.telegram_chat_id:
            return self.config.telegram_chat_id
        if not allow_discovery or not self.config.telegram_bot_token:
            return None
        try:
            response = requests.get(
                f"https://api.telegram.org/bot{self.config.telegram_bot_token}/getUpdates",
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            for item in reversed(payload.get("result", [])):
                message = item.get("message") or item.get("channel_post")
                if not message:
                    continue
                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if chat_id is not None:
                    self.config.telegram_chat_id = str(chat_id)
                    logging.info("Discovered Telegram chat id from bot updates")
                    return self.config.telegram_chat_id
        except Exception:
            logging.exception("Failed to discover Telegram chat id")
        return None

    def _send_telegram_message(self, text: str, allow_discovery: bool = False) -> Tuple[bool, str]:
        if not self.config.telegram_bot_token:
            return False, "TELEGRAM_BOT_TOKEN is not set"
        chat_id = self._resolve_telegram_chat_id(allow_discovery=allow_discovery)
        if not chat_id:
            return False, "TELEGRAM_CHAT_ID is not set and no bot chat was found. Send /start to the bot first."
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage",
                data={"chat_id": chat_id, "text": text[:4000]},
                timeout=self.config.request_timeout_seconds,
            )
            if not response.ok:
                logging.warning(
                    "Telegram text alert failed: HTTP %s %s",
                    response.status_code,
                    response.text[:300],
                )
                return False, f"Telegram returned HTTP {response.status_code}: {response.text[:300]}"
            return True, "Telegram test message sent"
        except Exception:
            logging.exception("Telegram text alert failed")
            return False, "Telegram request failed; check logs"

    def _event_uses_fast_alpr(self, event: Event) -> bool:
        for zone_id in event.zones_triggered:
            zone = self._find_motion_zone(zone_id)
            if zone and zone.use_fast_alpr:
                return True
        return False

    def _event_zone_summary(self, event: Event) -> List[dict[str, Any]]:
        summary: List[dict[str, Any]] = []
        for zone in self.motion_zones:
            if zone.zone_id not in event.zones_triggered:
                continue
            summary.append(
                {
                    "id": zone.zone_id,
                    "label": zone.label,
                    "use_fast_alpr": zone.use_fast_alpr,
                }
            )
        return summary

    def _compute_sharpness(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _encode_jpeg(self, frame) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Could not encode JPEG")
        return encoded.tobytes()

    def _plate_crop(self, frame):
        if not self.config.plate_roi and not self.config.roi:
            return frame
        x1, y1, x2, y2 = self._plate_roi_bounds(frame)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else frame

    def _plate_zoom_frame(self, frame):
        crop = self._plate_crop(frame)
        height, width = crop.shape[:2]
        if width == 0 or height == 0:
            return frame
        target_width = max(width, 640)
        scale = target_width / float(width)
        target_height = max(height, int(height * scale))
        return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    def _update_latest_frames(self, display_frame, source_frame) -> None:
        now = time.time()
        min_interval = 1.0 / self.config.stream_fps
        if self.latest_frame_jpeg is not None and now - self.last_stream_encode_enqueued_at < min_interval:
            return
        self.last_stream_encode_enqueued_at = now
        self._replace_queued_stream_frame((display_frame.copy(), source_frame.copy(), now))

    def _replace_queued_stream_frame(self, payload: Tuple[Any, Any, float]) -> None:
        try:
            self.stream_encode_queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            self.stream_encode_queue.get_nowait()
            self.stream_encode_queue.task_done()
        except queue.Empty:
            pass
        try:
            self.stream_encode_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _stream_encoder_loop(self) -> None:
        while not self.stream_encoder_stop.is_set():
            try:
                payload = self.stream_encode_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if payload is None:
                    return
                display_frame, source_frame, timestamp = payload
                jpeg_bytes = self._encode_jpeg(display_frame)
                clean_jpeg_bytes = self._encode_jpeg(source_frame)
                plate_zoom_bytes = self._encode_jpeg(self._plate_zoom_frame(source_frame))
                with self.frame_lock:
                    self.latest_frame_jpeg = jpeg_bytes
                    self.latest_clean_frame_jpeg = clean_jpeg_bytes
                    self.latest_plate_zoom_jpeg = plate_zoom_bytes
                    self.latest_frame_version += 1
                    self.latest_stream_update_at = timestamp
                    self.frame_lock.notify_all()
                self._record_stream_frame(timestamp)
            except Exception:
                logging.exception("Failed to encode MJPEG stream frame")
            finally:
                self.stream_encode_queue.task_done()

    def _stop_stream_encoder(self) -> None:
        self.stream_encoder_stop.set()
        try:
            self.stream_encode_queue.put_nowait(None)
        except queue.Full:
            try:
                self.stream_encode_queue.get_nowait()
                self.stream_encode_queue.task_done()
            except queue.Empty:
                pass
            try:
                self.stream_encode_queue.put_nowait(None)
            except queue.Full:
                pass
        self.stream_encoder_thread.join(timeout=2.0)

    def _start_http_server(self) -> None:
        watcher = self

        class WatcherHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = unquote(self.path.split("?", 1)[0])
                if path == "/api/roi":
                    self._send_json(
                        {
                            "roi": watcher._roi_value_for_ui(),
                            "zones": watcher._zones_for_ui(),
                            "min_motion_area": int(watcher.config.min_motion_area),
                        }
                    )
                    return
                if path == "/":
                    self._send_html(watcher._render_home_page())
                    return
                if path == "/test":
                    self._send_html(watcher._render_test_page())
                    return
                if path == "/live":
                    self._send_html(watcher._render_live_page())
                    return
                if path == "/stats":
                    self._send_html(watcher._render_stats_page())
                    return
                if path == "/api/stats":
                    self._send_json(watcher._stats_snapshot())
                    return
                if path == "/api/motion-status":
                    self._send_json({"status": watcher.latest_motion_status})
                    return
                if path == "/stream.mjpg":
                    self._send_mjpeg_stream()
                    return
                if path == "/stream-clean.mjpg":
                    self._send_mjpeg_stream(clean=True)
                    return
                if path == "/plate-zone.mjpg":
                    self._send_mjpeg_stream(plate_zoom=True)
                    return
                if path == "/images":
                    self._send_html(watcher._render_images_page())
                    return
                if path == "/videos":
                    self._send_html(watcher._render_videos_page())
                    return
                if path.startswith("/image-view/"):
                    watcher._serve_image_detail_page(self, path[len("/image-view/"):])
                    return
                if path.startswith("/video-view/"):
                    watcher._serve_video_detail_page(self, path[len("/video-view/"):])
                    return
                if path == "/plates":
                    self._send_html(watcher._render_plates_page())
                    return
                if path.startswith("/events/"):
                    watcher._serve_event_file(self, path[len("/events/"):])
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:
                path = unquote(self.path.split("?", 1)[0])
                if path == "/api/roi":
                    watcher._handle_roi_update(self)
                    return
                if path == "/api/motion-settings":
                    watcher._handle_motion_settings_update(self)
                    return
                if path == "/api/images/clear":
                    watcher._handle_clear_images(self)
                    return
                if path == "/api/videos/clear":
                    watcher._handle_clear_videos(self)
                    return
                if path == "/api/plates/clear":
                    watcher._handle_clear_plates(self)
                    return
                if path == "/api/telegram/test":
                    watcher._handle_telegram_test(self)
                    return
                if path == "/test":
                    watcher._handle_test_upload(self)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def log_message(self, format: str, *args: Any) -> None:
                logging.debug("HTTP %s", format % args)

            def _send_html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_mjpeg_stream(self, plate_zoom: bool = False, clean: bool = False) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_frame_version = -1
                try:
                    while True:
                        with watcher.frame_lock:
                            watcher.frame_lock.wait_for(
                                lambda: watcher.latest_frame_version != last_frame_version,
                                timeout=2.0,
                            )
                            if plate_zoom:
                                frame = watcher.latest_plate_zoom_jpeg
                            elif clean:
                                frame = watcher.latest_clean_frame_jpeg
                            else:
                                frame = watcher.latest_frame_jpeg
                            last_frame_version = watcher.latest_frame_version
                        if frame:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.http_server = ThreadingHTTPServer((self.config.web_host, self.config.web_port), WatcherHandler)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()
        logging.info(
            "Web UI available at http://127.0.0.1:%s/ live=http://127.0.0.1:%s/live stats=http://127.0.0.1:%s/stats images=http://127.0.0.1:%s/images",
            self.config.web_port,
            self.config.web_port,
            self.config.web_port,
            self.config.web_port,
        )

    def _stop_http_server(self) -> None:
        if not self.http_server:
            return
        self.http_server.shutdown()
        self.http_server.server_close()
        self.http_server = None

    def _event_file_url(self, relative_path: str) -> str:
        return f"/events/{quote(relative_path.lstrip('/'), safe='/')}"

    def _image_detail_url(self, relative_path: str) -> str:
        return f"/image-view/{quote(relative_path.lstrip('/'), safe='/')}"

    def _video_detail_url(self, relative_path: str) -> str:
        return f"/video-view/{quote(relative_path.lstrip('/'), safe='/')}"

    def _roi_value_for_ui(self) -> list[float]:
        primary_zone = self._find_motion_zone("yellow")
        roi = primary_zone.roi if primary_zone else (0.0, 0.0, 1.0, 1.0)
        return [float(value) for value in roi]

    def _zones_for_ui(self) -> List[dict[str, Any]]:
        return [
            {
                "id": zone.zone_id,
                "label": zone.label,
                "roi": [float(value) for value in zone.roi],
                "enabled": zone.enabled,
                "use_fast_alpr": zone.use_fast_alpr,
                "color": zone.color_hex,
                "fill": zone.fill_rgba,
            }
            for zone in self.motion_zones
        ]

    def _render_roi_editor_markup(self) -> str:
        return f"""<div class="panel">
  <h2>Live video</h2>
  <p style="color:#cbd5e1; margin-top:0;">Edit the yellow and purple motion zones, then choose whether each zone runs fast-alpr or only saves event images.</p>
  <div id="roi-editor" style="position:relative; max-width:100%;">
    <img id="roi-stream" style="width:100%; max-width:100%; border-radius:14px; border:1px solid #334155; display:block;" src="/stream-clean.mjpg" alt="Live stream">
    <div id="roi-overlay" style="position:absolute; inset:0; pointer-events:none;">
      <div id="roi-box-yellow" style="position:absolute; border:3px solid #facc15; background:rgba(250,204,21,0.12); box-sizing:border-box; pointer-events:none;"></div>
      <div id="roi-box-purple" style="position:absolute; border:3px solid #c084fc; background:rgba(192,132,252,0.16); box-sizing:border-box; pointer-events:none;"></div>
      <button class="roi-handle" data-corner="tl" style="position:absolute;"></button>
      <button class="roi-handle" data-corner="tr" style="position:absolute;"></button>
      <button class="roi-handle" data-corner="bl" style="position:absolute;"></button>
      <button class="roi-handle" data-corner="br" style="position:absolute;"></button>
    </div>
  </div>
  <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-top:14px;">
    <div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:12px;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
        <button id="zone-select-yellow" type="button" style="background:#facc15; color:#111827; border:0; border-radius:8px; padding:8px 12px; cursor:pointer; font-weight:700;">Edit Yellow Zone</button>
        <label style="color:#e2e8f0;"><input id="zone-enabled-yellow" type="checkbox" checked> Enabled</label>
      </div>
      <label style="display:block; margin-top:10px; color:#cbd5e1;"><input id="zone-fast-alpr-yellow" type="checkbox" checked> Send to fast-alpr</label>
    </div>
    <div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:12px;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
        <button id="zone-select-purple" type="button" style="background:#c084fc; color:#111827; border:0; border-radius:8px; padding:8px 12px; cursor:pointer; font-weight:700;">Edit Purple Zone</button>
        <label style="color:#e2e8f0;"><input id="zone-enabled-purple" type="checkbox"> Enabled</label>
      </div>
      <label style="display:block; margin-top:10px; color:#cbd5e1;"><input id="zone-fast-alpr-purple" type="checkbox"> Send to fast-alpr</label>
    </div>
  </div>
  <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:14px;">
    <button id="roi-save" type="button" style="background:#eab308; color:#111827; border:0; border-radius:8px; padding:10px 16px; cursor:pointer; font-weight:700;">Save Motion Zones</button>
    <button id="roi-reset" type="button" style="background:#334155; color:#f8fafc; border:0; border-radius:10px; padding:10px 16px; cursor:pointer;">Reset</button>
    <span id="roi-status" style="color:#cbd5e1;">Motion zones: loading...</span>
  </div>
  <div style="margin-top:16px;">
    <label for="motion-sensitivity" style="display:block; margin-bottom:8px; color:#cbd5e1;">Motion sensitivity</label>
    <input id="motion-sensitivity" type="range" min="{MIN_ALLOWED_MOTION_AREA}" max="20000" step="250" style="width:100%;">
    <div style="display:flex; justify-content:space-between; gap:12px; margin-top:6px; color:#cbd5e1; font-size:0.95rem;">
      <span>Smaller motion</span>
      <span id="motion-sensitivity-value">Threshold: loading...</span>
      <span>Larger motion</span>
    </div>
    <button id="motion-sensitivity-save" type="button" style="margin-top:12px; background:#38bdf8; color:#082f49; border:0; border-radius:10px; padding:10px 16px; cursor:pointer; font-weight:700;">Save Sensitivity</button>
  </div>
</div>"""

    def _render_roi_editor_script(self) -> str:
        zones_json = json.dumps(self._zones_for_ui())
        min_motion_area = int(self.config.min_motion_area)
        return f"""<script>
(() => {{
  const initialZones = {zones_json};
  let initialMotionArea = {min_motion_area};
  const editor = document.getElementById('roi-editor');
  const overlay = document.getElementById('roi-overlay');
  const zoneBoxes = {{
    yellow: document.getElementById('roi-box-yellow'),
    purple: document.getElementById('roi-box-purple'),
  }};
  const img = document.getElementById('roi-stream');
  const saveBtn = document.getElementById('roi-save');
  const resetBtn = document.getElementById('roi-reset');
  const status = document.getElementById('roi-status');
  const sensitivity = document.getElementById('motion-sensitivity');
  const sensitivityValue = document.getElementById('motion-sensitivity-value');
  const sensitivitySave = document.getElementById('motion-sensitivity-save');
  const handles = Array.from(document.querySelectorAll('.roi-handle'));
  const zoneButtons = {{
    yellow: document.getElementById('zone-select-yellow'),
    purple: document.getElementById('zone-select-purple'),
  }};
  const zoneEnabled = {{
    yellow: document.getElementById('zone-enabled-yellow'),
    purple: document.getElementById('zone-enabled-purple'),
  }};
  const zoneFastAlpr = {{
    yellow: document.getElementById('zone-fast-alpr-yellow'),
    purple: document.getElementById('zone-fast-alpr-purple'),
  }};
  if (!editor || !overlay || !img || !saveBtn || !resetBtn || !status || !sensitivity || !sensitivityValue || !sensitivitySave || handles.length !== 4 || !zoneBoxes.yellow || !zoneBoxes.purple || !zoneButtons.yellow || !zoneButtons.purple || !zoneEnabled.yellow || !zoneEnabled.purple || !zoneFastAlpr.yellow || !zoneFastAlpr.purple) return;

  let zones = initialZones.map((zone) => ({{ ...zone, roi: zone.roi.slice() }}));
  let activeZoneId = null;
  let dragCorner = null;
  let currentMotionArea = initialMotionArea;
  const minSize = 0.03;
  const handleSize = 18;

  function clamp(value, min, max) {{
    return Math.min(max, Math.max(min, value));
  }}

  function setStatus(text, isError = false) {{
    status.textContent = text;
    status.style.color = isError ? '#fca5a5' : '#cbd5e1';
  }}

  async function refreshMotionStatus() {{
    if (dragCorner || activeZoneId) return;
    try {{
      const response = await fetch('/api/motion-status', {{ cache: 'no-store' }});
      const payload = await response.json();
      if (response.ok && payload.status) {{
        setStatus(payload.status);
      }}
    }} catch (error) {{
    }}
  }}

  function renderSensitivity() {{
    sensitivity.value = String(currentMotionArea);
    sensitivityValue.textContent = `Threshold: ${{currentMotionArea}}`;
  }}

  function findZone(zoneId) {{
    return zones.find((zone) => zone.id === zoneId);
  }}

  function syncZoneControls() {{
    zones.forEach((zone) => {{
      zoneEnabled[zone.id].checked = Boolean(zone.enabled);
      zoneFastAlpr[zone.id].checked = Boolean(zone.use_fast_alpr);
      zoneButtons[zone.id].style.outline = zone.id === activeZoneId ? `2px solid ${{zone.color}}` : 'none';
      zoneButtons[zone.id].textContent = zone.id === activeZoneId ? `Editing ${{zone.label}}` : `Edit ${{zone.label}}`;
    }});
  }}

  function draw() {{
    zones.forEach((zone) => {{
      const [x1, y1, x2, y2] = zone.roi;
      const box = zoneBoxes[zone.id];
      if (!zone.enabled) {{
        box.style.display = 'none';
        return;
      }}
      box.style.display = 'block';
      box.style.left = `${{x1 * 100}}%`;
      box.style.top = `${{y1 * 100}}%`;
      box.style.width = `${{(x2 - x1) * 100}}%`;
      box.style.height = `${{(y2 - y1) * 100}}%`;
      box.style.borderColor = zone.color;
      box.style.background = zone.fill;
      box.style.opacity = '1';
      box.style.borderWidth = zone.id === activeZoneId ? '4px' : '3px';
    }});
    const activeZone = activeZoneId ? findZone(activeZoneId) : null;
    if (!activeZone || !activeZone.enabled) {{
      if (activeZone && !activeZone.enabled) activeZoneId = null;
      handles.forEach((handle) => {{
        handle.style.left = '-9999px';
        handle.style.top = '-9999px';
        handle.style.pointerEvents = 'none';
      }});
      syncZoneControls();
      const zoneSummaries = zones.map((zone) => `${{zone.label}}: ${{zone.enabled ? 'on' : 'off'}}, ${{zone.use_fast_alpr ? 'fast-alpr' : 'images only'}}`);
      setStatus(`Motion zones saved | ${{zoneSummaries.join(' | ')}}`);
      return;
    }}
    const [x1, y1, x2, y2] = activeZone.roi;
    const positions = {{
      tl: [x1, y1],
      tr: [x2, y1],
      bl: [x1, y2],
      br: [x2, y2],
    }};
    handles.forEach((handle) => {{
      const [x, y] = positions[handle.dataset.corner];
      handle.style.left = `calc(${{x * 100}}% - ${{handleSize / 2}}px)`;
      handle.style.top = `calc(${{y * 100}}% - ${{handleSize / 2}}px)`;
      handle.style.width = `${{handleSize}}px`;
      handle.style.height = `${{handleSize}}px`;
      handle.style.border = '2px solid #111827';
      handle.style.borderRadius = '999px';
      handle.style.background = activeZone.color;
      handle.style.cursor = 'grab';
      handle.style.pointerEvents = 'auto';
      handle.style.touchAction = 'none';
      handle.style.padding = '0';
    }});
    syncZoneControls();
    const zoneSummaries = zones.map((zone) => `${{zone.label}}: ${{zone.enabled ? 'on' : 'off'}}, ${{zone.use_fast_alpr ? 'fast-alpr' : 'images only'}}`);
    setStatus(`Editing ${{activeZone.label}} | ROI: ${{activeZone.roi.map((v) => v.toFixed(3)).join(', ')}} | ${{zoneSummaries.join(' | ')}}`);
  }}

  function updateFromPointer(clientX, clientY) {{
    const rect = overlay.getBoundingClientRect();
    if (!rect.width || !rect.height || !dragCorner) return;
    const x = clamp((clientX - rect.left) / rect.width, 0, 1);
    const y = clamp((clientY - rect.top) / rect.height, 0, 1);
    const activeZone = findZone(activeZoneId);
    if (!activeZone) return;
    let [x1, y1, x2, y2] = activeZone.roi;
    if (dragCorner.includes('l')) x1 = clamp(x, 0, x2 - minSize);
    if (dragCorner.includes('r')) x2 = clamp(x, x1 + minSize, 1);
    if (dragCorner.includes('t')) y1 = clamp(y, 0, y2 - minSize);
    if (dragCorner.includes('b')) y2 = clamp(y, y1 + minSize, 1);
    activeZone.roi = [x1, y1, x2, y2];
    draw();
  }}

  handles.forEach((handle) => {{
    handle.addEventListener('pointerdown', (event) => {{
      dragCorner = handle.dataset.corner;
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    }});
    handle.addEventListener('pointermove', (event) => {{
      if (!dragCorner) return;
      updateFromPointer(event.clientX, event.clientY);
    }});
    handle.addEventListener('pointerup', () => {{
      dragCorner = null;
    }});
    handle.addEventListener('pointercancel', () => {{
      dragCorner = null;
    }});
  }});

  resetBtn.addEventListener('click', () => {{
    zones = initialZones.map((zone) => ({{ ...zone, roi: zone.roi.slice() }}));
    activeZoneId = null;
    draw();
  }});

  Object.entries(zoneButtons).forEach(([zoneId, button]) => {{
    button.addEventListener('click', () => {{
      activeZoneId = activeZoneId === zoneId ? null : zoneId;
      draw();
    }});
  }});

  Object.entries(zoneEnabled).forEach(([zoneId, checkbox]) => {{
    checkbox.addEventListener('change', () => {{
      const zone = findZone(zoneId);
      if (!zone) return;
      zone.enabled = checkbox.checked;
      if (!zone.enabled && activeZoneId === zoneId) activeZoneId = null;
      draw();
    }});
  }});

  Object.entries(zoneFastAlpr).forEach(([zoneId, checkbox]) => {{
    checkbox.addEventListener('change', () => {{
      const zone = findZone(zoneId);
      if (!zone) return;
      zone.use_fast_alpr = checkbox.checked;
      draw();
    }});
  }});

  sensitivity.addEventListener('input', () => {{
    currentMotionArea = Number(sensitivity.value);
    renderSensitivity();
  }});

  saveBtn.addEventListener('click', async () => {{
    setStatus('Saving motion zones...');
    try {{
      const response = await fetch('/api/roi', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ zones }}),
      }});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Save failed');
      zones = payload.zones.map((zone) => ({{ ...zone, roi: zone.roi.slice() }}));
      initialZones.splice(0, initialZones.length, ...zones.map((zone) => ({{ ...zone, roi: zone.roi.slice() }})));
      activeZoneId = null;
      dragCorner = null;
      draw();
      setStatus('Saved motion zones.');
    }} catch (error) {{
      setStatus(error.message || 'Save failed', true);
    }}
  }});

  sensitivitySave.addEventListener('click', async () => {{
    setStatus('Saving sensitivity...');
    try {{
      const response = await fetch('/api/motion-settings', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ min_motion_area: currentMotionArea }}),
      }});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Save failed');
      currentMotionArea = Number(payload.min_motion_area);
      initialMotionArea = currentMotionArea;
      renderSensitivity();
      setStatus(`Saved sensitivity threshold: ${{currentMotionArea}}`);
    }} catch (error) {{
      setStatus(error.message || 'Save failed', true);
    }}
  }});

  draw();
  renderSensitivity();
  window.setInterval(refreshMotionStatus, 1000);
  refreshMotionStatus();
}})();
</script>"""

    def _render_shared_styles(self) -> str:
        return """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { min-height: 100vh; background: #07110f; color: #edf7f2; padding: 0; }
a { color: #7dd3fc; }
.page-shell { width: min(1380px, calc(100% - 32px)); margin: 0 auto; padding: 22px 0 34px; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 24px; padding: 12px 14px; background: rgba(9, 18, 17, 0.92); border: 1px solid #24413c; border-radius: 8px; box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28); }
.brand { color: #f8fafc; font-weight: 800; text-decoration: none; letter-spacing: 0; white-space: nowrap; }
.nav-links { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.nav-links a { color: #cde8de; text-decoration: none; border: 1px solid #2f514b; border-radius: 8px; padding: 8px 11px; background: #10211f; line-height: 1; }
.nav-links a:hover { border-color: #5eead4; color: #f8fafc; background: #17302d; }
.nav-links a.active { border-color: #facc15; color: #111827; background: #facc15; font-weight: 700; }
.panel, .card { border: 1px solid #263d39; box-shadow: 0 10px 28px rgba(0, 0, 0, 0.2); }
h1 { margin-top: 0; }
@media (max-width: 760px) {
  .page-shell { width: min(100% - 20px, 1380px); padding-top: 10px; }
  .topbar { align-items: flex-start; flex-direction: column; }
  .nav-links { justify-content: flex-start; }
  .nav-links a { padding: 9px 10px; }
}
"""

    def _render_nav(self, active: str) -> str:
        items = [
            ("dashboard", "/", "Dashboard"),
            ("live", "/live", "Live"),
            ("stats", "/stats", "Stats"),
            ("images", "/images", "Images"),
            ("videos", "/videos", "Videos"),
            ("plates", "/plates", "Plates"),
            ("test", "/test", "Test"),
        ]
        links = "".join(
            f'<a class="{"active" if item_id == active else ""}" href="{href}">{label}</a>'
            for item_id, href, label in items
        )
        return f"""<header class="topbar">
  <a class="brand" href="/">ALPR Watcher</a>
  <nav class="nav-links" aria-label="Main navigation">{links}</nav>
</header>"""

    def _render_live_view_script(self) -> str:
        return """<script>
(() => {
  const stream = document.getElementById('live-clean-stream');
  const readout = document.getElementById('live-zoom-readout');
  const zoomIn = document.getElementById('live-zoom-in');
  const zoomOut = document.getElementById('live-zoom-out');
  const panLeft = document.getElementById('live-pan-left');
  const panRight = document.getElementById('live-pan-right');
  const panUp = document.getElementById('live-pan-up');
  const panDown = document.getElementById('live-pan-down');
  const reset = document.getElementById('live-reset');
  if (!stream || !readout || !zoomIn || !zoomOut || !panLeft || !panRight || !panUp || !panDown || !reset) return;

  let zoom = 1;
  let offsetX = 0;
  let offsetY = 0;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function applyView() {
    if (zoom <= 1) {
      offsetX = 0;
      offsetY = 0;
    }
    const maxOffset = Math.max(0, ((zoom - 1) / zoom) * 50);
    offsetX = clamp(offsetX, -maxOffset, maxOffset);
    offsetY = clamp(offsetY, -maxOffset, maxOffset);
    stream.style.transform = `scale(${zoom}) translate(${offsetX}%, ${offsetY}%)`;
    readout.textContent = `Zoom ${Math.round(zoom * 100)}%`;
  }

  zoomIn.addEventListener('click', () => {
    zoom = clamp(Number((zoom + 0.25).toFixed(2)), 1, 4);
    applyView();
  });
  zoomOut.addEventListener('click', () => {
    zoom = clamp(Number((zoom - 0.25).toFixed(2)), 1, 4);
    applyView();
  });
  panLeft.addEventListener('click', () => {
    offsetX += 6;
    applyView();
  });
  panRight.addEventListener('click', () => {
    offsetX -= 6;
    applyView();
  });
  panUp.addEventListener('click', () => {
    offsetY += 6;
    applyView();
  });
  panDown.addEventListener('click', () => {
    offsetY -= 6;
    applyView();
  });
  reset.addEventListener('click', () => {
    zoom = 1;
    offsetX = 0;
    offsetY = 0;
    applyView();
  });
  applyView();
})();
</script>"""

    def _render_home_page(self) -> str:
        recent = self._render_recent_plate_cards(limit=8)
        gallery = self._render_image_cards(limit=8)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ALPR Watcher</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #f8fafc; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.nav {{ margin-bottom: 18px; color: #cbd5e1; }}
.hero {{ display: flex; justify-content: space-between; gap: 20px; align-items: end; margin-bottom: 20px; flex-wrap: wrap; }}
.hero h1 {{ margin: 0; font-size: 2rem; }}
.hero p {{ margin: 6px 0 0; color: #cbd5e1; }}
.layout {{ display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(320px, 1fr); gap: 20px; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; }}
.stream {{ width: 100%; border-radius: 14px; display: block; border: 1px solid #334155; }}
.section-title {{ margin: 0 0 14px; font-size: 1.2rem; }}
.plates {{ display: grid; gap: 12px; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
.gallery .card {{ background: #0f172a; border-radius: 12px; padding: 12px; }}
.gallery img {{ width: 100%; border-radius: 10px; display: block; }}
.card {{ background: #0f172a; border-radius: 12px; padding: 14px; }}
.plate-code {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.08em; }}
.meta {{ color: #cbd5e1; font-size: 0.95rem; }}
@media (max-width: 960px) {{
  .layout {{ grid-template-columns: 1fr; }}
}}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("dashboard")}
<div class="hero">
<div><h1>ALPR Dashboard</h1><p>Live camera, detected plate numbers, and captured evidence in one place.</p></div>
<div><p>Image retention limit: {self.config.max_saved_images}</p></div>
</div>
<div class="layout">
  {self._render_roi_editor_markup()}
  <div class="panel">
    <h2 class="section-title">Recent plate numbers</h2>
    <div class="plates">{recent}</div>
  </div>
</div>
<div class="panel" style="margin-top:20px;">
  <h2 class="section-title">Latest captured images</h2>
  <div class="gallery">{gallery}</div>
</div>
</div>
{self._render_roi_editor_script()}
</body></html>"""

    def _render_live_page(self) -> str:
        recent = self._render_recent_plate_cards(limit=8)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Live Video</title>
<style>
body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.layout {{ display: grid; grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr); gap: 24px; align-items: start; }}
.panel {{ background: #111827; border-radius: 14px; padding: 16px; }}
.live-viewer {{ background: #050c0b; border: 1px solid #263d39; border-radius: 8px; overflow: hidden; }}
.live-viewport {{ width: 100%; aspect-ratio: 16 / 9; background: #000; overflow: hidden; }}
.live-viewport img {{ width: 100%; height: 100%; object-fit: contain; display: block; transform-origin: center; will-change: transform; }}
.live-controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 12px; background: #10211f; }}
.live-controls button {{ background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 9px 12px; cursor: pointer; font-weight: 700; }}
.live-controls .secondary {{ background: #263d39; color: #edf7f2; }}
.zoom-readout {{ color: #cbd5e1; min-width: 92px; }}
.plate {{ border-bottom: 1px solid #334155; padding: 12px 0; }}
.plate:last-child {{ border-bottom: 0; }}
.plate-code {{ font-size: 1.25rem; font-weight: 700; letter-spacing: 0.08em; }}
.meta {{ color: #cbd5e1; font-size: 0.95rem; margin-top: 4px; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("live")}
<h1>Live video</h1>
<div class="layout">
<div class="live-viewer">
  <div class="live-viewport">
    <img id="live-clean-stream" src="/stream-clean.mjpg" alt="Live camera">
  </div>
  <div class="live-controls" aria-label="Live view controls">
    <button id="live-zoom-in" type="button">Zoom In</button>
    <button id="live-zoom-out" type="button">Zoom Out</button>
    <button id="live-pan-left" type="button" class="secondary">Left</button>
    <button id="live-pan-right" type="button" class="secondary">Right</button>
    <button id="live-pan-up" type="button" class="secondary">Up</button>
    <button id="live-pan-down" type="button" class="secondary">Down</button>
    <button id="live-reset" type="button" class="secondary">Reset</button>
    <span id="live-zoom-readout" class="zoom-readout">Zoom 100%</span>
  </div>
</div>
<div class="panel"><h2>Recent plate numbers</h2>{recent}</div>
</div>
</div>
{self._render_live_view_script()}
</body></html>"""

    def _render_stats_page(self) -> str:
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stream Stats</title>
<style>
body {{ font-family: Arial, sans-serif; background: #07110f; color: #edf7f2; margin: 0; padding: 24px; }}
a {{ color: #7dd3fc; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 18px 0; }}
.metric {{ background: #10211f; border: 1px solid #2f514b; border-radius: 8px; padding: 14px; min-height: 96px; }}
.metric .label {{ color: #a7c4bc; font-size: 0.92rem; margin-bottom: 8px; }}
.metric .value {{ color: #f8fafc; font-size: 1.75rem; font-weight: 800; line-height: 1.1; }}
.metric .hint {{ color: #cde8de; font-size: 0.9rem; margin-top: 8px; }}
.panel {{ background: #0d1a18; border: 1px solid #263d39; border-radius: 8px; padding: 16px; }}
.status-ok {{ color: #86efac; }}
.status-warn {{ color: #facc15; }}
.status-bad {{ color: #fca5a5; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #050c0b; border: 1px solid #263d39; border-radius: 8px; padding: 12px; color: #d9f99d; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("stats")}
<h1>Stream Stats</h1>
<p class="meta">Use this while watching the camera clock. Large capture gaps are the strongest sign that frames are arriving late or the loop is blocked.</p>
<div class="summary">
  <div class="metric"><div class="label">RTSP Capture FPS</div><div id="capture-fps" class="value">...</div><div class="hint">Frames received by the watcher</div></div>
  <div class="metric"><div class="label">Processing FPS</div><div id="processing-fps" class="value">...</div><div class="hint">Frames completed by the motion loop</div></div>
  <div class="metric"><div class="label">Dashboard FPS</div><div id="stream-fps" class="value">...</div><div class="hint">MJPEG frames sent to the UI</div></div>
  <div class="metric"><div class="label">Recent Max Gap</div><div id="recent-gap" class="value">...</div><div id="gap-status" class="hint">Waiting for data</div></div>
  <div class="metric"><div class="label">All-Time Max Gap</div><div id="max-gap" class="value">...</div><div class="hint">Largest measured capture pause</div></div>
  <div class="metric"><div class="label">Estimated Missed Frames</div><div id="missed-frames" class="value">...</div><div class="hint">Estimated from capture gaps</div></div>
  <div class="metric"><div class="label">Read Failures</div><div id="read-failures" class="value">...</div><div class="hint">RTSP read errors</div></div>
  <div class="metric"><div class="label">Reconnects</div><div id="reconnects" class="value">...</div><div class="hint">Capture loop restarts</div></div>
</div>
<div class="panel">
  <h2>Current State</h2>
  <p id="state-line">Loading...</p>
  <p id="motion-line">Motion zones: loading...</p>
  <pre id="raw-stats">{{}}</pre>
</div>
</div>
<script>
(() => {{
  const ids = {{
    captureFps: document.getElementById('capture-fps'),
    processingFps: document.getElementById('processing-fps'),
    streamFps: document.getElementById('stream-fps'),
    recentGap: document.getElementById('recent-gap'),
    gapStatus: document.getElementById('gap-status'),
    maxGap: document.getElementById('max-gap'),
    missedFrames: document.getElementById('missed-frames'),
    readFailures: document.getElementById('read-failures'),
    reconnects: document.getElementById('reconnects'),
    stateLine: document.getElementById('state-line'),
    motionLine: document.getElementById('motion-line'),
    rawStats: document.getElementById('raw-stats'),
  }};

  function number(value, digits = 1) {{
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
    return Number(value).toFixed(digits);
  }}

  function seconds(value) {{
    return `${{number(value, 2)}}s`;
  }}

  function setGapStatus(gap) {{
    ids.gapStatus.className = 'hint status-ok';
    ids.gapStatus.textContent = 'No recent stalls';
    if (gap >= 2) {{
      ids.gapStatus.className = 'hint status-bad';
      ids.gapStatus.textContent = 'Large capture pause';
    }} else if (gap >= 1) {{
      ids.gapStatus.className = 'hint status-warn';
      ids.gapStatus.textContent = 'Capture is uneven';
    }}
  }}

  async function refreshStats() {{
    try {{
      const response = await fetch('/api/stats', {{ cache: 'no-store' }});
      const stats = await response.json();
      if (!response.ok) throw new Error(stats.error || 'Stats failed');
      ids.captureFps.textContent = number(stats.capture_fps);
      ids.processingFps.textContent = number(stats.processing_fps);
      ids.streamFps.textContent = `${{number(stats.stream_fps)}} / ${{number(stats.configured_stream_fps)}}`;
      ids.recentGap.textContent = seconds(stats.recent_max_capture_gap_seconds);
      ids.maxGap.textContent = seconds(stats.max_capture_gap_seconds);
      ids.missedFrames.textContent = String(stats.estimated_missed_frames);
      ids.readFailures.textContent = String(stats.read_failure_total);
      ids.reconnects.textContent = String(stats.capture_reconnect_total);
      setGapStatus(Number(stats.recent_max_capture_gap_seconds || 0));
      ids.stateLine.textContent = `Camera estimate ${{number(stats.camera_fps_estimate)}} FPS, width ${{stats.frame_width}}, processing every ${{stats.process_every_n_frames}} frame(s), recording ${{stats.recording_active ? `active at ${{number(stats.recording_fps)}} FPS` : 'idle'}}, last frame age ${{seconds(stats.last_capture_age_seconds)}}.`;
      ids.motionLine.textContent = stats.motion_status || 'Motion zones: waiting';
      ids.rawStats.textContent = JSON.stringify(stats, null, 2);
    }} catch (error) {{
      ids.stateLine.textContent = error.message || 'Unable to load stats';
      ids.stateLine.className = 'status-bad';
    }}
  }}

  refreshStats();
  window.setInterval(refreshStats, 1000);
}})();
</script>
</body></html>"""

    def _render_images_page(self) -> str:
        gallery = self._render_image_cards(limit=self.config.max_saved_images)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Captured Images</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }}
.card {{ background: #111827; padding: 12px; border-radius: 14px; }}
img {{ width: 100%; border-radius: 10px; display: block; }}
video {{ width: 100%; border-radius: 10px; display: block; margin-top: 10px; background: #000; }}
p {{ margin: 8px 0 0; word-break: break-word; }}
button {{ background: #dc2626; color: white; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("images")}
<h1>Captured images</h1>
<p>Keeping only the newest {self.config.max_saved_images} images.</p>
<form method="post" action="/api/images/clear" onsubmit="return confirm('Remove all saved images?');" style="margin: 16px 0 20px;">
<button type="submit">Remove All Images</button>
</form>
<div class="grid">{gallery}</div>
</div>
</body></html>"""

    def _render_videos_page(self) -> str:
        gallery = self._render_video_cards(limit=100)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Saved Videos</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
.card {{ background: #111827; padding: 12px; border-radius: 14px; }}
video {{ width: 100%; border-radius: 10px; display: block; background: #000; }}
p {{ margin: 8px 0 0; word-break: break-word; }}
button {{ background: #dc2626; color: white; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("videos")}
<h1>Saved videos</h1>
<p>Videos start with at least {int(VIDEO_PREBUFFER_SECONDS)} seconds before the confirmed motion and run for {int(VIDEO_RECORDING_SECONDS // 60)} minutes. New motion does not create another video while one is already recording.</p>
<form method="post" action="/api/videos/clear" onsubmit="return confirm('Remove all saved videos?');" style="margin: 16px 0 20px;">
<button type="submit">Remove All Videos</button>
</form>
<div class="grid">{gallery}</div>
</div>
</body></html>"""

    def _render_plates_page(self) -> str:
        cards = self._render_recent_plate_cards(limit=100)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Detected Plates</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; }}
.card {{ background: #111827; padding: 14px; border-radius: 14px; }}
.card img {{ width: 100%; border-radius: 10px; display: block; margin-top: 10px; }}
.plate-code {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.08em; }}
.meta {{ color: #cbd5e1; margin-top: 6px; }}
.page-action {{ display: inline-flex; align-items: center; margin: 0 0 18px; color: #111827; background: #facc15; border-radius: 8px; padding: 10px 14px; text-decoration: none; font-weight: 700; }}
.page-actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; }}
.page-actions form {{ margin: 0; }}
.telegram-action {{ background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 700; }}
.danger-action {{ background: #dc2626; color: white; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 700; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("plates")}
<h1>Detected plates</h1>
<div class="page-actions">
  <a class="page-action" href="/test">Test Upload</a>
  <form method="post" action="/api/telegram/test">
    <button class="telegram-action" type="submit">Send Telegram Test</button>
  </form>
  <form method="post" action="/api/plates/clear" onsubmit="return confirm('Remove all detected plates? Images and videos will stay saved.');">
    <button class="danger-action" type="submit">Remove All Plates</button>
  </form>
</div>
<div class="grid">{cards}</div>
</div>
</body></html>"""

    def _render_test_page(self) -> str:
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Test Upload</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; max-width: 760px; }}
label {{ display: block; margin-bottom: 10px; color: #cbd5e1; }}
input[type=file] {{ display: block; margin-top: 8px; color: #e2e8f0; }}
button {{ margin-top: 14px; background: #2563eb; color: white; border: 0; border-radius: 10px; padding: 10px 16px; cursor: pointer; }}
button.telegram {{ background: #facc15; color: #111827; }}
code {{ background: #0f172a; padding: 2px 6px; border-radius: 6px; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("test")}
<div class="panel">
<h1>Test Plate Detection</h1>
<p>Upload a JPG or PNG and the server will run the same plate pipeline used for saved event frames: optional <code>PLATE_ROI</code> crop, local <code>fast-alpr</code>, then OpenALPR only if the local confidence gate passes.</p>
<form action="/test" method="post" enctype="multipart/form-data">
<label>Image file
<input type="file" name="image" accept=".jpg,.jpeg,.png,image/jpeg,image/png" required>
</label>
<button type="submit">Run Detection Test</button>
</form>
</div>
<div class="panel" style="margin-top:18px;">
<h2>Telegram Alert Test</h2>
<p>Send a test message to the bot. If the chat ID is not configured yet, send <code>/start</code> to the bot first.</p>
<form action="/api/telegram/test" method="post">
<button class="telegram" type="submit">Send Telegram Test</button>
</form>
</div>
</div>
</body></html>"""

    def _render_test_result_page(self, event_name: str, pipeline: dict[str, Any]) -> str:
        image_path = pipeline["relative_frame_path"]
        fast_alpr_json = f"{event_name}/uploaded.fast_alpr.json"
        openalpr_json = f"{event_name}/uploaded.json"
        image_url = self._event_file_url(image_path)
        fast_alpr_url = self._event_file_url(fast_alpr_json)
        openalpr_url = self._event_file_url(openalpr_json)
        fast_alpr_text = json.dumps(pipeline["fast_alpr_results"][0], indent=2) if pipeline["fast_alpr_results"] else "No fast-alpr result"
        openalpr_text = json.dumps(pipeline["openalpr_results"][0], indent=2) if pipeline["openalpr_results"] else "No OpenALPR result"
        skip_reason = pipeline["openalpr_skipped_reason"] or pipeline["openalpr_error"] or "OpenALPR ran"
        detections = pipeline["plate_detections"]
        detections_html = "".join(
            f'<li><strong>{html.escape(item.plate)}</strong> via {html.escape(item.source)} ({item.confidence:.2f})</li>'
            for item in detections
        ) or "<li>No plates detected.</li>"
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Test Upload Result</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.layout {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 20px; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; }}
img {{ max-width: 100%; border-radius: 14px; display: block; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; padding: 14px; border-radius: 12px; overflow-x: auto; }}
ul {{ padding-left: 20px; }}
@media (max-width: 960px) {{ .layout {{ grid-template-columns: 1fr; }} }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("test")}
<p><a href="/test">Run another upload</a> | <a href="{image_url}">Open cropped test image</a></p>
<h1>Detection Test Result</h1>
<p>Policy applied: crop to <code>PLATE_ROI</code> when configured, run <code>fast-alpr</code>, and only send to OpenALPR when <code>fast-alpr</code> found a plate with confidence at or above <code>{self.config.fast_alpr_min_confidence:.2f}</code>.</p>
<div class="layout">
<div class="panel">
<h2>Cropped image sent into ALPR</h2>
<img src="{image_url}" alt="Uploaded test image">
<h2>Detections</h2>
<ul>{detections_html}</ul>
<p><strong>OpenALPR status:</strong> {html.escape(skip_reason)}</p>
</div>
<div class="panel">
<h2>fast-alpr</h2>
<p><a href="{fast_alpr_url}">Open raw fast-alpr JSON</a></p>
<pre>{html.escape(fast_alpr_text)}</pre>
<h2>OpenALPR</h2>
<p><a href="{openalpr_url}">Open raw OpenALPR JSON</a></p>
<pre>{html.escape(openalpr_text)}</pre>
</div>
</div>
</div>
</body></html>"""

    def _handle_roi_update(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
            zones_payload = payload.get("zones")
            if isinstance(zones_payload, list):
                seen_zone_ids: set[str] = set()
                for zone_payload in zones_payload:
                    if not isinstance(zone_payload, dict):
                        raise ValueError("Each zone must be an object")
                    zone_id = str(zone_payload.get("id") or "").strip()
                    zone = self._find_motion_zone(zone_id)
                    if not zone:
                        raise ValueError(f"Unknown motion zone: {zone_id}")
                    roi_values = zone_payload.get("roi")
                    if not isinstance(roi_values, list) or len(roi_values) != 4:
                        raise ValueError(f"{zone.label} ROI must be a four-number array")
                    zone.roi = parse_normalized_roi(
                        ",".join(str(float(value)) for value in roi_values),
                        f"{zone.label} ROI",
                    ) or zone.roi
                    zone.enabled = bool(zone_payload.get("enabled", zone.enabled))
                    zone.use_fast_alpr = bool(zone_payload.get("use_fast_alpr", zone.use_fast_alpr))
                    seen_zone_ids.add(zone_id)
                if "yellow" not in seen_zone_ids:
                    raise ValueError("Yellow zone is required")
            else:
                roi_values = payload.get("roi")
                if not isinstance(roi_values, list) or len(roi_values) != 4:
                    raise ValueError("ROI must be a four-number array")
                primary_zone = self._find_motion_zone("yellow")
                if not primary_zone:
                    raise ValueError("Yellow zone is unavailable")
                primary_zone.roi = parse_normalized_roi(
                    ",".join(str(float(value)) for value in roi_values),
                    "ROI",
                ) or primary_zone.roi
            self._sync_primary_zone_to_config()
            self._save_runtime_config()
            body = json.dumps({"roi": self._roi_value_for_ui(), "zones": self._zones_for_ui()}).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            handler.send_response(HTTPStatus.BAD_REQUEST)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

    def _handle_motion_settings_update(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
            min_motion_area = int(payload.get("min_motion_area"))
            if min_motion_area < MIN_ALLOWED_MOTION_AREA:
                raise ValueError(f"min_motion_area must be at least {MIN_ALLOWED_MOTION_AREA}")
            self.config.min_motion_area = min_motion_area
            self._save_runtime_config()
            body = json.dumps({"min_motion_area": int(self.config.min_motion_area)}).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            handler.send_response(HTTPStatus.BAD_REQUEST)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

    def _serve_event_file(self, handler: BaseHTTPRequestHandler, relative_path: str) -> None:
        safe_relative = Path(unquote(relative_path))
        target = (self.config.event_output_dir / safe_relative).resolve()
        base = self.config.event_output_dir.resolve()
        if base not in target.parents and target != base:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if target.suffix.lower() not in {".jpg", ".jpeg", ".png", ".mp4", ".json"}:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".mp4": "video/mp4",
            ".json": "application/json",
        }[target.suffix.lower()]
        body = target.read_bytes()
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _handle_test_upload(self, handler: BaseHTTPRequestHandler) -> None:
        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
            },
        )
        file_item = form["image"] if "image" in form else None
        if file_item is None or not getattr(file_item, "file", None):
            handler.send_error(HTTPStatus.BAD_REQUEST, "Missing image upload")
            return

        payload = file_item.file.read()
        if not payload:
            handler.send_error(HTTPStatus.BAD_REQUEST, "Empty image upload")
            return

        frame_array = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame_array is None:
            handler.send_error(HTTPStatus.BAD_REQUEST, "Invalid image upload")
            return

        frame = self._resize_frame(frame_array)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        event_name = f"{self.config.camera_name}_test_{stamp}"
        event_dir = self.config.event_output_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        pipeline = self._run_detection_pipeline(
            frame,
            event_dir,
            "uploaded",
            event_name,
            time.time(),
        )
        summary = {
            "camera_name": self.config.camera_name,
            "started_at_epoch": time.time(),
            "ended_at_epoch": time.time(),
            "trigger_count": 1,
            "saved_frames": 1,
            "clip_path": None,
            "fast_alpr_results_count": len(pipeline["fast_alpr_results"]),
            "openalpr_results_count": len(pipeline["openalpr_results"]),
            "plates": [detection.__dict__ for detection in pipeline["plate_detections"]],
            "test_upload": True,
            "openalpr_skipped_reason": pipeline["openalpr_skipped_reason"],
        }
        (event_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        body = self._render_test_result_page(event_name, pipeline).encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _handle_telegram_test(self, handler: BaseHTTPRequestHandler) -> None:
        ok, message = self._send_telegram_message(
            f"ALPR Watcher Telegram test\nCamera: {self.config.camera_name}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            allow_discovery=True,
        )
        wants_json = handler.headers.get("Content-Type", "").startswith("application/json")
        if wants_json:
            body = json.dumps({"ok": ok, "message": message}).encode("utf-8")
            handler.send_response(HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return
        escaped_message = html.escape(message)
        status_text = "Telegram Test Sent" if ok else "Telegram Test Failed"
        body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{status_text}</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; max-width: 760px; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("test")}
<div class="panel">
<h1>{status_text}</h1>
<p>{escaped_message}</p>
<p><a href="/test">Back to test tools</a></p>
</div>
</div>
</body></html>""".encode("utf-8")
        handler.send_response(HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _list_saved_images(self) -> List[Path]:
        images = sorted(
            self.config.event_output_dir.glob("*/*.jpg"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return images

    def _list_recent_plate_detections(self, limit: int) -> List[PlateDetection]:
        detections: List[PlateDetection] = []
        summaries = sorted(
            self.config.event_output_dir.glob("*/summary.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for summary_path in summaries:
            try:
                payload = json.loads(summary_path.read_text())
            except Exception:
                continue
            for item in payload.get("plates", []):
                plate = str(item.get("plate") or "").strip()
                if not plate:
                    continue
                detections.append(
                    PlateDetection(
                        plate=plate,
                        confidence=float(item.get("confidence") or 0.0),
                        source=str(item.get("source") or "unknown"),
                        image_relative_path=str(item.get("image_relative_path") or ""),
                        event_name=str(item.get("event_name") or summary_path.parent.name),
                        detected_at_epoch=float(item.get("detected_at_epoch") or summary_path.stat().st_mtime),
                    )
                )
                if len(detections) >= limit:
                    return detections
        return detections

    def _render_recent_plate_cards(self, limit: int) -> str:
        detections = self._list_recent_plate_detections(limit)
        if not detections:
            return "<p>No plate detections yet.</p>"
        cards = []
        for detection in detections:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detection.detected_at_epoch))
            image_html = ""
            if detection.image_relative_path:
                image_path = self.config.event_output_dir / Path(detection.image_relative_path)
                if image_path.exists() and image_path.is_file():
                    image_link = self._event_file_url(detection.image_relative_path)
                    image_html = f'<a href="{image_link}"><img src="{image_link}" alt="{html.escape(detection.plate)}"></a>'
                else:
                    image_html = '<div class="meta">Image file no longer retained.</div>'
            cards.append(
                f'<div class="plate card"><div class="plate-code">{html.escape(detection.plate)}</div>'
                f'<div class="meta">{html.escape(detection.source)} | {detection.confidence:.2f}</div>'
                f'<div class="meta">{timestamp}</div>'
                f'<div class="meta">{html.escape(detection.event_name)}</div>{image_html}</div>'
            )
        return "".join(cards)

    def _render_image_cards(self, limit: int) -> str:
        cards = []
        for image_path in self._list_saved_images()[:limit]:
            relative = image_path.relative_to(self.config.event_output_dir).as_posix()
            detail_link = self._image_detail_url(relative)
            image_url = self._event_file_url(relative)
            summary_path = image_path.parent / "summary.json"
            policy_note = ""
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    policy = summary.get("event_policy") or "unknown"
                    policy_note = f"<p>Policy: {html.escape(str(policy))}</p>"
                except Exception:
                    policy_note = ""
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(image_path.stat().st_mtime))
            escaped_relative = html.escape(relative)
            cards.append(
                f'<div class="card"><a href="{detail_link}"><img src="{image_url}" alt="{escaped_relative}"></a>'
                f'<p><a href="{detail_link}">{escaped_relative}</a></p><p>{stamp}</p>{policy_note}</div>'
            )
        return "".join(cards) or "<p>No images saved yet.</p>"

    def _serve_image_detail_page(self, handler: BaseHTTPRequestHandler, relative_path: str) -> None:
        safe_relative = Path(unquote(relative_path))
        image_path = (self.config.event_output_dir / safe_relative).resolve()
        base = self.config.event_output_dir.resolve()
        if base not in image_path.parents:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not image_path.exists() or not image_path.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        relative = image_path.relative_to(self.config.event_output_dir).as_posix()
        images = [path.relative_to(self.config.event_output_dir).as_posix() for path in self._list_saved_images()]
        try:
            current_index = images.index(relative)
        except ValueError:
            current_index = -1
        newer_relative = images[current_index - 1] if current_index > 0 else None
        older_relative = images[current_index + 1] if 0 <= current_index < len(images) - 1 else None
        image_url = self._event_file_url(relative)
        newer_link = self._image_detail_url(newer_relative) if newer_relative else ""
        older_link = self._image_detail_url(older_relative) if older_relative else ""
        newer_html = f'<a href="{newer_link}">Newer image</a>' if newer_link else '<span>Newer image</span>'
        older_html = f'<a href="{older_link}">Older image</a>' if older_link else '<span>Older image</span>'
        body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Captured Image</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; }}
img {{ width: 100%; border-radius: 10px; display: block; max-width: 1100px; }}
.detail-actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 16px; color: #cbd5e1; }}
.detail-actions span {{ color: #64748b; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("images")}
<div class="detail-actions"><a href="/images">Back to captured images</a><a href="{image_url}">Open image file</a>{newer_html}{older_html}</div>
<h1>Captured image</h1>
<div class="panel"><img src="{image_url}" alt="{html.escape(relative)}"><p>{html.escape(relative)}</p></div>
</div>
<script>
(() => {{
  const newer = {json.dumps(newer_link)};
  const older = {json.dumps(older_link)};
  window.addEventListener('keydown', (event) => {{
    if (event.key === 'ArrowLeft' && newer) window.location.href = newer;
    if (event.key === 'ArrowRight' && older) window.location.href = older;
  }});
}})();
</script>
</body></html>""".encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _list_saved_videos(self) -> List[Path]:
        return sorted(
            self._video_output_dir().glob("*.mp4"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

    def _render_video_cards(self, limit: int) -> str:
        cards = []
        for video_path in self._list_saved_videos()[:limit]:
            relative = video_path.relative_to(self.config.event_output_dir).as_posix()
            video_url = self._event_file_url(relative)
            detail_url = self._video_detail_url(relative)
            metadata_path = video_path.with_suffix(".json")
            zone_text = "unknown"
            started_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(video_path.stat().st_mtime))
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    zone_ids = metadata.get("zone_ids") or []
                    zone_text = ",".join(zone_ids) if zone_ids else "unknown"
                    started_at = metadata.get("started_at_epoch")
                    if started_at:
                        started_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(started_at)))
                except Exception:
                    pass
            escaped_relative = html.escape(relative)
            cards.append(
                f'<div class="card"><p><a href="{detail_url}">{escaped_relative}</a></p>'
                f'<p>{started_text}</p><p>Zones: {html.escape(zone_text)}</p>'
                f'<video controls preload="metadata" src="{video_url}"></video></div>'
            )
        return "".join(cards) or "<p>No videos saved yet.</p>"

    def _serve_video_detail_page(self, handler: BaseHTTPRequestHandler, relative_path: str) -> None:
        safe_relative = Path(unquote(relative_path))
        video_path = (self.config.event_output_dir / safe_relative).resolve()
        base = self.config.event_output_dir.resolve()
        if base not in video_path.parents:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not video_path.exists() or not video_path.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if video_path.suffix.lower() != ".mp4":
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        relative = video_path.relative_to(self.config.event_output_dir).as_posix()
        video_url = self._event_file_url(relative)
        body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Saved Video</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; }}
video {{ width: 100%; border-radius: 10px; display: block; background: #000; }}
.detail-actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 16px; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("videos")}
<div class="detail-actions"><a href="/videos">Back to saved videos</a><a href="{video_url}">Open video file</a></div>
<h1>Saved video</h1>
<div class="panel"><video controls autoplay src="{video_url}"></video><p>{html.escape(relative)}</p></div>
</div>
</body></html>""".encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _prune_saved_images(self) -> None:
        images = self._list_saved_images()
        extra = images[self.config.max_saved_images :]
        for image_path in extra:
            logging.info("Pruning old image %s", image_path)
            stem = image_path.stem
            for candidate in image_path.parent.glob(f"{stem}*"):
                if candidate.name == "summary.json":
                    continue
                if candidate.is_file():
                    candidate.unlink(missing_ok=True)

    def _handle_clear_images(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            for event_dir in self.config.event_output_dir.iterdir():
                if event_dir.is_dir():
                    if event_dir.name == "videos":
                        continue
                    shutil.rmtree(event_dir, ignore_errors=True)
            if handler.headers.get("Content-Type", "").startswith("application/json"):
                body = json.dumps({"cleared": True}).encode("utf-8")
                handler.send_response(HTTPStatus.OK)
                handler.send_header("Content-Type", "application/json; charset=utf-8")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)
                return
            handler.send_response(HTTPStatus.SEE_OTHER)
            handler.send_header("Location", "/images")
            handler.end_headers()
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            handler.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

    def _handle_clear_videos(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            video_dir = self._video_output_dir()
            for video_path in video_dir.iterdir():
                if video_path.name == "recording-tmp":
                    if video_path.is_dir():
                        shutil.rmtree(video_path, ignore_errors=True)
                    continue
                if video_path.is_file():
                    video_path.unlink(missing_ok=True)
            self._video_temp_dir()
            if handler.headers.get("Content-Type", "").startswith("application/json"):
                body = json.dumps({"cleared": True}).encode("utf-8")
                handler.send_response(HTTPStatus.OK)
                handler.send_header("Content-Type", "application/json; charset=utf-8")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)
                return
            handler.send_response(HTTPStatus.SEE_OTHER)
            handler.send_header("Location", "/videos")
            handler.end_headers()
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            handler.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

    def _handle_clear_plates(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            cleared = 0
            for summary_path in self.config.event_output_dir.glob("*/summary.json"):
                try:
                    payload = json.loads(summary_path.read_text())
                except Exception:
                    logging.exception("Failed to read summary while clearing plates: %s", summary_path)
                    continue
                if payload.get("plates"):
                    payload["plates"] = []
                    summary_path.write_text(json.dumps(payload, indent=2))
                    cleared += 1
            if handler.headers.get("Content-Type", "").startswith("application/json"):
                body = json.dumps({"cleared": True, "updated_summaries": cleared}).encode("utf-8")
                handler.send_response(HTTPStatus.OK)
                handler.send_header("Content-Type", "application/json; charset=utf-8")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)
                return
            handler.send_response(HTTPStatus.SEE_OTHER)
            handler.send_header("Location", "/plates")
            handler.end_headers()
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            handler.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

    def _recognize_fast_alpr(self, jpeg_bytes: bytes) -> dict:
        response = requests.post(
            f"{self.config.fast_alpr_url.rstrip('/')}/recognize",
            files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _fast_alpr_has_confident_plate(self, result: Optional[dict]) -> bool:
        if not result:
            return False
        for item in result.get("results", []):
            plate = item.get("plate")
            confidence = float(item.get("confidence") or 0.0)
            if plate and confidence >= self.config.fast_alpr_min_confidence:
                return True
        return False

    def _extract_fast_alpr_detections(
        self,
        result: dict,
        image_relative_path: str,
        event_name: str,
        detected_at_epoch: float,
    ) -> List[PlateDetection]:
        detections: List[PlateDetection] = []
        for item in result.get("results", []):
            plate = str(item.get("plate") or "").strip()
            if not plate:
                continue
            detections.append(
                PlateDetection(
                    plate=plate,
                    confidence=float(item.get("confidence") or 0.0),
                    source="fast-alpr",
                    image_relative_path=image_relative_path,
                    event_name=event_name,
                    detected_at_epoch=detected_at_epoch,
                )
            )
        return detections

    def _extract_openalpr_detections(
        self,
        result: dict,
        image_relative_path: str,
        event_name: str,
        detected_at_epoch: float,
    ) -> List[PlateDetection]:
        detections: List[PlateDetection] = []
        for item in result.get("results", []):
            plate = str(item.get("plate") or "").strip()
            if not plate:
                continue
            confidence = float(item.get("confidence") or 0.0)
            if confidence > 1.0:
                confidence /= 100.0
            detections.append(
                PlateDetection(
                    plate=plate,
                    confidence=confidence,
                    source="openalpr",
                    image_relative_path=image_relative_path,
                    event_name=event_name,
                    detected_at_epoch=detected_at_epoch,
                )
            )
        return detections


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = Config.from_env()
    watcher = RtspVehicleWatcher(config)
    watcher.run()


if __name__ == "__main__":
    main()
