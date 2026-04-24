import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, List, Optional, Set, Tuple

from dotenv import load_dotenv

MIN_ALLOWED_MOTION_AREA = 500
VIDEO_PREBUFFER_SECONDS = 10.0
VIDEO_RECORDING_SECONDS = 180.0
DEFAULT_FRAME_WIDTH = 960
MAX_RECORDING_FPS = 15.0
MAX_VIDEO_PREBUFFER_FRAMES = 150
VIDEO_WRITER_QUEUE_SECONDS = 3.0
DEFAULT_RTSP_CAPTURE_OPTIONS = "rtsp_transport;tcp|max_delay;2000000|stimeout;10000000"
DEFAULT_ALPR_CAPTURE_FPS = 1.0
GALLERY_PAGE_SIZE = 24
MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
FILE_STREAM_CHUNK_SIZE = 1024 * 1024


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


def redact_url_credentials(value: str) -> str:
    if not value:
        return value
    return re.sub(r"(?<=://)([^/@:]+)(?::[^/@]*)?@", "***:***@", value)


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
    image_output_dir: Path
    video_output_dir: Path
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
    alpr_capture_warmup_seconds: float
    live_stream_warmup_seconds: float
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_alert_images: int
    telegram_alerts_enabled: bool

    @property
    def openalpr_enabled(self) -> bool:
        return bool(self.secret_key)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id and self.telegram_alert_images > 0)

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, env: Any) -> "Config":
        def getenv(name: str, default: str = "") -> str:
            value = env.get(name, default)
            return default if value is None else str(value)

        rtsp_url = getenv("RTSP_URL").strip()
        secret_key = getenv("OPENALPR_SECRET_KEY").strip()
        if not rtsp_url:
            raise ValueError("RTSP_URL is required")

        try:
            roi = parse_normalized_roi(getenv("ROI").strip(), "ROI")
            plate_roi = parse_normalized_roi(getenv("PLATE_ROI").strip(), "PLATE_ROI")
        except ValueError as exc:
            raise ValueError(f"Invalid ROI configuration: {exc}") from exc

        event_output_dir = Path(getenv("EVENT_OUTPUT_DIR", "./events")).expanduser()
        image_output_dir = Path(getenv("IMAGE_OUTPUT_DIR", str(event_output_dir))).expanduser()
        video_output_dir = Path(getenv("VIDEO_OUTPUT_DIR", str(event_output_dir / "videos"))).expanduser()

        return cls(
            rtsp_url=rtsp_url,
            alpr_rtsp_url=getenv("ALPR_RTSP_URL").strip(),
            secret_key=secret_key,
            country=getenv("OPENALPR_COUNTRY", "us").strip(),
            frame_width=int(getenv("FRAME_WIDTH", str(DEFAULT_FRAME_WIDTH))),
            process_every_n_frames=max(1, int(getenv("PROCESS_EVERY_N_FRAMES", "2"))),
            min_motion_area=max(MIN_ALLOWED_MOTION_AREA, int(getenv("MIN_MOTION_AREA", "6500"))),
            min_consecutive_hits=max(1, int(getenv("MIN_CONSECUTIVE_HITS", "3"))),
            event_idle_seconds=float(getenv("EVENT_IDLE_SECONDS", "1.5")),
            event_max_seconds=max(1.0, float(getenv("EVENT_MAX_SECONDS", "60.0"))),
            prebuffer_seconds=float(getenv("PREBUFFER_SECONDS", "2.0")),
            postbuffer_seconds=float(getenv("POSTBUFFER_SECONDS", "5.0")),
            prebuffer_frames=max(0, int(getenv("PREBUFFER_FRAMES", "0"))),
            postbuffer_frames=max(0, int(getenv("POSTBUFFER_FRAMES", "0"))),
            upload_top_frames=max(1, int(getenv("UPLOAD_TOP_FRAMES", "240"))),
            upload_min_sharpness=float(getenv("UPLOAD_MIN_SHARPNESS", "80.0")),
            event_output_dir=event_output_dir,
            image_output_dir=image_output_dir,
            video_output_dir=video_output_dir,
            camera_name=getenv("CAMERA_NAME", "camera").strip(),
            roi=roi,
            plate_roi=plate_roi,
            recognize_vehicle=parse_bool(getenv("RECOGNIZE_VEHICLE", "true"), default=True),
            debug_windows=parse_bool(getenv("DEBUG_WINDOWS", "false"), default=False),
            request_timeout_seconds=float(getenv("REQUEST_TIMEOUT_SECONDS", "20")),
            fast_alpr_url=getenv("FAST_ALPR_URL").strip(),
            fast_alpr_min_confidence=float(getenv("FAST_ALPR_MIN_CONFIDENCE", "0.75")),
            web_host=getenv("WEB_HOST", "0.0.0.0").strip(),
            web_port=int(getenv("WEB_PORT", "8080")),
            max_saved_images=max(1, int(getenv("MAX_SAVED_IMAGES", "2000"))),
            ffmpeg_threads=max(1, int(getenv("FFMPEG_THREADS", "1"))),
            stream_fps=max(1.0, float(getenv("STREAM_FPS", "5"))),
            capture_buffer_size=max(1, int(getenv("CAPTURE_BUFFER_SIZE", "4"))),
            rtsp_capture_options=getenv("RTSP_CAPTURE_OPTIONS", DEFAULT_RTSP_CAPTURE_OPTIONS).strip()
            or DEFAULT_RTSP_CAPTURE_OPTIONS,
            alpr_capture_fps=max(0.1, float(getenv("ALPR_CAPTURE_FPS", str(DEFAULT_ALPR_CAPTURE_FPS)))),
            alpr_capture_warmup_seconds=max(0.0, float(getenv("ALPR_CAPTURE_WARMUP_SECONDS", "1.5"))),
            live_stream_warmup_seconds=max(0.0, float(getenv("LIVE_STREAM_WARMUP_SECONDS", "1.5"))),
            telegram_bot_token=getenv("TELEGRAM_BOT_TOKEN").strip(),
            telegram_chat_id=getenv("TELEGRAM_CHAT_ID").strip(),
            telegram_alert_images=max(0, int(getenv("TELEGRAM_ALERT_IMAGES", "3"))),
            telegram_alerts_enabled=parse_bool(getenv("TELEGRAM_ALERTS_ENABLED", "true"), default=True),
        )


@dataclass
class CandidateFrame:
    frame: Any
    timestamp: float
    motion_area: int
    sharpness: float
    jpeg_bytes: Optional[bytes] = None
    source: str = "capture"
    zone_ids: Set[str] = field(default_factory=set)


@dataclass
class MotionZone:
    zone_id: str
    label: str
    roi: Tuple[float, float, float, float]
    enabled: bool
    use_fast_alpr: bool
    send_telegram: bool
    record_seconds: float
    image_count: int
    coverage_trigger_percent: float
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
    zones_triggered: Set[str]
    MAX_FRAMES: int = field(default=500, repr=False)
    MAX_CANDIDATES: int = field(default=200, repr=False)

    def append_frame(self, timestamp: float, frame: Any) -> None:
        if len(self.frames) >= self.MAX_FRAMES:
            self.frames.pop(0)
        self.frames.append((timestamp, frame))

    def append_candidate(self, candidate: "CandidateFrame") -> None:
        if len(self.candidates) >= self.MAX_CANDIDATES:
            self.candidates.pop(0)
        self.candidates.append(candidate)


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
    confirmed: bool = False


@dataclass
class VideoRecording:
    started_at: float
    first_frame_at: float
    ends_at: float
    record_seconds: float
    started_from_zone_ids: Set[str]
    output_path: Path
    temp_output_path: Path
    writer: Any
    last_written_at: float
    frame_size: Tuple[int, int]
    fps: float
    last_frame: Any = None
    source_url: str = ""
    confirmed: bool = False
    pending_events: List[Event] = field(default_factory=list)
    MAX_PENDING_EVENTS: int = field(default=20, repr=False)

    def append_pending_event(self, event: "Event") -> None:
        if len(self.pending_events) < self.MAX_PENDING_EVENTS:
            self.pending_events.append(event)
