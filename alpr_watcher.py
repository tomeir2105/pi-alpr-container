import html
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode
from typing import Any, Deque, List, Optional, Set, Tuple
from email.parser import BytesParser
from email.policy import default as email_policy

import cv2
import numpy as np
import requests
from dotenv import dotenv_values

from alpr_models import (
    AlprCaptureSession,
    CandidateFrame,
    Config,
    Event,
    FILE_STREAM_CHUNK_SIZE,
    GALLERY_PAGE_SIZE,
    MAX_IMAGE_UPLOAD_BYTES,
    MAX_VIDEO_PREBUFFER_FRAMES,
    MIN_ALLOWED_MOTION_AREA,
    MotionZone,
    PlateDetection,
    VIDEO_PREBUFFER_SECONDS,
    VIDEO_RECORDING_SECONDS,
    VideoRecording,
    parse_bool,
    parse_normalized_roi,
    redact_url_credentials,
)
from alpr_services import FfmpegVideoWriter, OpenAlprClient, QueuedVideoWriter


class RtspVehicleWatcher:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.runtime_config_path = self.config.event_output_dir.parent / "watcher-config.json"
        self.motion_zones = self._default_motion_zones()
        self._load_runtime_config()
        self.client = OpenAlprClient(config)
        self.prebuffer: Deque[Tuple[float, Any]] = deque(maxlen=max(500, int(MAX_VIDEO_PREBUFFER_FRAMES * 2)))
        self.event: Optional[Event] = None
        self.motion_streak = 0
        self.last_motion_area = 0
        self.last_motion_box: Optional[Tuple[int, int, int, int]] = None
        self.last_triggered_zone_ids: Set[str] = set()
        self.last_zone_area_by_id: dict[str, int] = {}
        self.last_zone_coverage_percent_by_id: dict[str, float] = {}
        self.last_coverage_triggered_zone_ids: Set[str] = set()
        self.cumulative_zone_masks_by_id: dict[str, Any] = {}
        self.latest_motion_status = "Motion zones: waiting for activity"
        self.video_recording: Optional[VideoRecording] = None
        self.last_video_recording: Optional[VideoRecording] = None
        self.last_recording_ended_at: float = 0.0
        self.extraction_status: dict[str, Any] = {}
        self.alpr_capture_session: Optional[AlprCaptureSession] = None
        self.alpr_capture_lock = threading.Lock()
        self.event_log_lock = threading.Lock()
        self.event_log: Deque[dict[str, Any]] = deque(maxlen=100)
        self.frame_index = 0
        self.fps_guess = 12.0
        self.capture = None
        self.background = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=36, detectShadows=False)
        self.config.event_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.video_output_dir.mkdir(parents=True, exist_ok=True)
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
                send_telegram=self.config.telegram_alerts_enabled,
                record_seconds=max(1.0, VIDEO_RECORDING_SECONDS),
                image_count=max(1, self.config.upload_top_frames),
                coverage_trigger_percent=50.0,
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
                send_telegram=self.config.telegram_alerts_enabled,
                record_seconds=max(1.0, VIDEO_RECORDING_SECONDS),
                image_count=max(1, self.config.upload_top_frames),
                coverage_trigger_percent=50.0,
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

    def _zones_for_policy(self, zone_ids: Set[str]) -> List[MotionZone]:
        zones = [zone for zone in self.motion_zones if zone.zone_id in zone_ids]
        if zones:
            return zones
        primary_zone = self._find_motion_zone("yellow")
        return [primary_zone] if primary_zone else []

    def _record_seconds_for_zone_ids(self, zone_ids: Set[str]) -> float:
        return max((zone.record_seconds for zone in self._zones_for_policy(zone_ids)), default=max(1.0, VIDEO_RECORDING_SECONDS))

    def _image_limit_for_zone_ids(self, zone_ids: Set[str]) -> int:
        return max((zone.image_count for zone in self._zones_for_policy(zone_ids)), default=max(1, self.config.upload_top_frames))

    def _load_runtime_config(self) -> None:
        try:
            should_resave = False
            if not self.runtime_config_path.exists():
                self._sync_primary_zone_to_config()
                return
            payload = json.loads(self.runtime_config_path.read_text())
            legacy_telegram_alerts_enabled = payload.get("telegram_alerts_enabled")
            if isinstance(legacy_telegram_alerts_enabled, str):
                legacy_telegram_default = parse_bool(
                    legacy_telegram_alerts_enabled,
                    default=self.config.telegram_alerts_enabled,
                )
            elif legacy_telegram_alerts_enabled is not None:
                legacy_telegram_default = bool(legacy_telegram_alerts_enabled)
            else:
                legacy_telegram_default = self.config.telegram_alerts_enabled
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
                    send_telegram_value = zone_payload.get("send_telegram", legacy_telegram_default)
                    if isinstance(send_telegram_value, str):
                        zone.send_telegram = parse_bool(send_telegram_value, default=zone.send_telegram)
                    else:
                        zone.send_telegram = bool(send_telegram_value)
                    zone.record_seconds = max(1.0, float(zone_payload.get("record_seconds", zone.record_seconds)))
                    zone.image_count = max(1, int(zone_payload.get("image_count", zone.image_count)))
                    zone.coverage_trigger_percent = min(
                        100.0,
                        max(0.0, float(zone_payload.get("coverage_trigger_percent", zone.coverage_trigger_percent))),
                    )
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
            if legacy_telegram_alerts_enabled is not None:
                self.config.telegram_alerts_enabled = legacy_telegram_default
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
                            "send_telegram": zone.send_telegram,
                            "record_seconds": zone.record_seconds,
                            "image_count": zone.image_count,
                            "coverage_trigger_percent": zone.coverage_trigger_percent,
                        }
                        for zone in self.motion_zones
                    ],
                    "min_motion_area": int(self.config.min_motion_area),
                    "telegram_alerts_enabled": bool(self.config.telegram_alerts_enabled),
                },
                indent=2,
            )
        )

    def _prepare_live_hls_workspace(self) -> Path:
        candidate_roots = [
            Path("/dev/shm"),
            Path("/run/shm"),
            Path("/mnt/localdisk"),
        ]
        for root in candidate_roots:
            try:
                if not root.exists() or not root.is_dir():
                    continue
                workspace = root / "pi-alpr-live-hls"
                workspace.mkdir(parents=True, exist_ok=True)
                probe = workspace / ".write-test"
                probe.write_text("ok")
                probe.unlink(missing_ok=True)
                logging.info("Using live HLS workspace at %s", workspace)
                return workspace
            except Exception:
                continue
        fallback = self.config.event_output_dir.parent / "live-hls-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        logging.warning("Falling back to on-disk live HLS workspace at %s", fallback)
        return fallback

    def _live_hls_source_url(self) -> str:
        return self._effective_alpr_rtsp_url() or self.config.rtsp_url

    def _live_hls_playlist_path(self) -> Path:
        return self.live_hls_output_dir / "index.m3u8"

    def _live_hls_url(self) -> str:
        return "/live-hls/index.m3u8"

    def _clear_live_hls_workspace(self) -> None:
        for directory in (self.live_hls_output_dir, self.live_hls_root / "frames"):
            directory.mkdir(parents=True, exist_ok=True)
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)

    def _start_live_hls_writer(self, fps: float, max_width: int, transport: str, source_url: str) -> subprocess.Popen[bytes]:
        self.live_hls_output_dir.mkdir(parents=True, exist_ok=True)
        segment_pattern = str(self.live_hls_output_dir / "segment-%05d.ts")
        return subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-fflags",
                "+discardcorrupt",
                "-rtsp_transport",
                transport,
                "-i",
                source_url,
                "-an",
                "-vf",
                f"fps={fps:.3f},scale={max_width}:-2:force_original_aspect_ratio=decrease",
                "-map",
                "0:v",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(max(2, int(round(fps * 2.0)))),
                "-sc_threshold",
                "0",
                "-force_key_frames",
                "expr:gte(t,n_forced*1)",
                "-f",
                "hls",
                "-hls_time",
                "1",
                "-hls_list_size",
                "20",
                "-hls_delete_threshold",
                "6",
                "-hls_flags",
                "delete_segments+append_list+omit_endlist+independent_segments+program_date_time",
                "-hls_segment_filename",
                segment_pattern,
                str(self._live_hls_playlist_path()),
            ],
            stderr=subprocess.PIPE,
        )

    def _live_hls_loop(self) -> None:
        fps = max(2.0, min(12.0, float(self.config.stream_fps)))
        max_width = max(640, min(1920, int(self.config.frame_width or 1280)))
        transport = self._rtsp_option_value("rtsp_transport", "tcp") or "tcp"
        self._clear_live_hls_workspace()
        while not self.live_hls_stop.is_set():
            source_url = self._live_hls_source_url()
            if not source_url or not shutil.which("ffmpeg"):
                time.sleep(1.0)
                continue
            hls_process: Optional[subprocess.Popen[bytes]] = None
            try:
                hls_process = self._start_live_hls_writer(fps, max_width, transport, source_url)
                last_playlist_mtime = 0.0
                last_progress_at = time.time()
                while not self.live_hls_stop.is_set():
                    if hls_process.poll() is not None:
                        stderr = hls_process.stderr.read().decode("utf-8", errors="replace") if hls_process.stderr else ""
                        raise RuntimeError(stderr.strip() or "ffmpeg live HLS process exited")
                    playlist_path = self._live_hls_playlist_path()
                    if playlist_path.exists():
                        playlist_mtime = playlist_path.stat().st_mtime
                        if playlist_mtime > last_playlist_mtime:
                            last_playlist_mtime = playlist_mtime
                            last_progress_at = time.time()
                            self.live_hls_last_frame_at = last_progress_at
                    if time.time() - last_progress_at > max(30.0, self.config.live_stream_warmup_seconds + 20.0):
                        raise RuntimeError("live HLS playlist stopped updating")
                    time.sleep(1.0)
            except Exception:
                logging.exception("Live HLS pipeline failed; restarting shortly")
                time.sleep(1.0)
            finally:
                if hls_process:
                    try:
                        hls_process.terminate()
                        hls_process.wait(timeout=2)
                    except Exception:
                        try:
                            hls_process.kill()
                            hls_process.wait(timeout=2)
                        except Exception:
                            pass

    def _stop_live_hls_stream(self) -> None:
        self.live_hls_stop.set()
        self.live_hls_thread.join(timeout=3.0)

    def _serve_live_hls_file(self, handler: BaseHTTPRequestHandler, relative_path: str, head_only: bool = False) -> None:
        safe_relative = Path(relative_path.lstrip("/"))
        target = (self.live_hls_output_dir / safe_relative).resolve()
        base = self.live_hls_output_dir.resolve()
        if base not in target.parents and target != base:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        suffix = target.suffix.lower()
        content_type = {
            ".m3u8": "application/vnd.apple.mpegurl",
            ".ts": "video/mp2t",
            ".jpg": "image/jpeg",
        }.get(suffix)
        if not content_type:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        body = target.read_bytes()
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        handler.send_header("Pragma", "no-cache")
        handler.end_headers()
        if not head_only:
            handler.wfile.write(body)

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
        capture_error: queue.Queue[Exception] = queue.Queue(maxsize=1)
        capture_failure: List[Optional[Exception]] = [None]
        frame_queue: queue.Queue[Tuple[float, Any]] = queue.Queue(maxsize=1)

        def current_capture_error() -> Optional[Exception]:
            if capture_failure[0] is not None:
                return capture_failure[0]
            try:
                capture_failure[0] = capture_error.get_nowait()
            except queue.Empty:
                return None
            return capture_failure[0]

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
                    replace_captured_frame((timestamp, captured_frame))
            except Exception as exc:
                if capture_error.empty():
                    capture_error.put_nowait(exc)

        capture_thread = threading.Thread(target=capture_reader, name="rtsp-reader", daemon=True)
        capture_thread.start()

        try:
            while True:
                reader_error = current_capture_error()
                if reader_error is not None:
                    raise RuntimeError("RTSP reader failed") from reader_error
                try:
                    timestamp, original_frame = frame_queue.get(timeout=2.0)
                    frame_queue.task_done()
                except queue.Empty:
                    reader_error = current_capture_error()
                    if reader_error is not None:
                        raise RuntimeError("RTSP reader failed") from reader_error
                    raise RuntimeError("Timed out waiting for frame from RTSP reader")

                frame = self._resize_frame(original_frame)
                self._push_prebuffer(timestamp, original_frame)

                # During active recording, process fewer frames to free CPU for ffmpeg
                process_interval = self.config.process_every_n_frames * 3 if self.video_recording is not None else self.config.process_every_n_frames
                process_this_frame = self.frame_index % process_interval == 0
                motion_area = self.last_motion_area
                had_motion = False
                triggered_zone_ids: Set[str] = set()
                overlay = self._draw_monitor_overlays(frame.copy())

                raw_motion = False
                if process_this_frame:
                    (
                        motion_area,
                        raw_motion,
                        best_box,
                        overlay,
                        triggered_zone_ids,
                        zone_area_by_id,
                        zone_coverage_percent_by_id,
                        zone_motion_masks_by_id,
                    ) = self._detect_motion(frame)
                    if raw_motion:
                        self.motion_streak = min(self.config.min_consecutive_hits, self.motion_streak + 1)
                        self.last_triggered_zone_ids = set(triggered_zone_ids)
                    else:
                        self.motion_streak = max(0, self.motion_streak - 1)
                        if self.motion_streak == 0:
                            self.last_triggered_zone_ids = set()
                    should_accumulate_coverage = raw_motion or self.motion_streak > 0
                    if should_accumulate_coverage:
                        (
                            zone_coverage_percent_by_id,
                            coverage_triggered_zone_ids,
                        ) = self._update_cumulative_zone_coverage(zone_motion_masks_by_id)
                    else:
                        self._reset_coverage_accumulators()
                        coverage_triggered_zone_ids = set()
                    coverage_confirmed = bool(coverage_triggered_zone_ids)
                    if coverage_confirmed:
                        self.last_triggered_zone_ids.update(coverage_triggered_zone_ids)
                    streak_confirmed = self.motion_streak >= self.config.min_consecutive_hits
                    coverage_can_start_event = coverage_confirmed and self.event is None
                    had_motion = streak_confirmed or coverage_can_start_event
                    self.last_motion_area = motion_area
                    self.last_motion_box = best_box
                    self.last_zone_area_by_id = dict(zone_area_by_id)
                    self.last_zone_coverage_percent_by_id = dict(zone_coverage_percent_by_id)
                    self.last_coverage_triggered_zone_ids = set(coverage_triggered_zone_ids)
                    overlay = self._annotate_motion_overlay(
                        overlay,
                        motion_area,
                        best_box,
                        had_motion,
                        triggered_zone_ids if triggered_zone_ids else self.last_triggered_zone_ids,
                        zone_area_by_id,
                        zone_coverage_percent_by_id,
                        coverage_triggered_zone_ids,
                    )
                    if self.config.debug_windows:
                        cv2.imshow("watcher", overlay)
                        cv2.waitKey(1)
                    if raw_motion:
                        self._start_video_recording(timestamp, triggered_zone_ids, confirmed=False)
                    elif not self.event and self.motion_streak == 0:
                        self._discard_unconfirmed_video_recording()
                else:
                    had_motion = self.motion_streak >= self.config.min_consecutive_hits
                    overlay = self._annotate_motion_overlay(
                        overlay,
                        motion_area,
                        self.last_motion_box,
                        had_motion,
                        self.last_triggered_zone_ids,
                        self.last_zone_area_by_id,
                        self.last_zone_coverage_percent_by_id,
                        self.last_coverage_triggered_zone_ids,
                    )

                self._update_latest_frames(overlay, frame)

                if had_motion:
                    if not triggered_zone_ids:
                        triggered_zone_ids = set(self.last_triggered_zone_ids)
                    if self.video_recording is None:
                        self._start_video_recording(timestamp, triggered_zone_ids, confirmed=True)
                    else:
                        self.video_recording.started_from_zone_ids.update(triggered_zone_ids)
                        # Guarantee minimum duration from confirmation time for the triggered zones
                        min_ends_at = timestamp + self._record_seconds_for_zone_ids(triggered_zone_ids)
                        self.video_recording.ends_at = max(self.video_recording.ends_at, min_ends_at)
                        self.video_recording.record_seconds = max(
                            self.video_recording.record_seconds,
                            self.video_recording.ends_at - self.video_recording.started_at,
                        )
                        if not self.video_recording.confirmed:
                            self.video_recording.confirmed = True
                            logging.info("Confirmed video recording for active motion event")
                            self._add_event_log(
                                "motion",
                                f"Confirmed motion for zones={','.join(sorted(triggered_zone_ids)) or 'unknown'}",
                                zone_id=self._primary_zone_id_for(triggered_zone_ids),
                            )
                    self._on_motion(timestamp, original_frame, motion_area, triggered_zone_ids)
                elif self.event:
                    self._append_event_frame(timestamp, original_frame, count_as_postbuffer=True)
                    if self._event_ready_to_finalize(timestamp):
                        self._finalize_event("motion idle and postbuffer complete")

                self._append_video_recording_frame(timestamp, original_frame)

                self.frame_index += 1
                self._record_processing_frame()
        finally:
            capture_stop.set()
            capture_thread.join(timeout=2.0)
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
            snapshot = {
                "uptime_seconds": max(0.0, now - self.capture_started_at),
                "camera_fps_estimate": float(self.fps_guess),
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
        snapshot["capture_fps"] = self._rolling_fps(capture_times)
        snapshot["processing_fps"] = self._rolling_fps(processing_times)
        snapshot["stream_fps"] = self._rolling_fps(stream_times)
        return snapshot

    def _push_prebuffer(self, timestamp: float, frame) -> None:
        self.prebuffer.append((timestamp, frame.copy()))
        max_frames = max(1, self._prebuffer_frame_limit())
        while len(self.prebuffer) > max_frames:
            self.prebuffer.popleft()

    def _video_output_dir(self) -> Path:
        path = self.config.video_output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _video_temp_dir(self) -> Path:
        path = self._video_output_dir() / "recording-tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _relative_image_path(self, path: Path) -> str:
        return path.relative_to(self.config.image_output_dir).as_posix()

    def _relative_video_path(self, path: Path) -> str:
        return path.relative_to(self.config.video_output_dir).as_posix()

    def _image_path_from_relative(self, relative_path: str) -> Path:
        safe_relative = Path(unquote(relative_path))
        target = (self.config.image_output_dir / safe_relative).resolve()
        base = self.config.image_output_dir.resolve()
        if base not in target.parents and target != base:
            raise ValueError("Invalid image path")
        return target

    def _video_path_from_relative(self, relative_path: str) -> Path:
        safe_relative = Path(unquote(relative_path))
        target = (self.config.video_output_dir / safe_relative).resolve()
        base = self.config.video_output_dir.resolve()
        if base not in target.parents and target != base:
            raise ValueError("Invalid video path")
        return target

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

    def _reset_coverage_accumulators(self) -> None:
        self.cumulative_zone_masks_by_id = {}
        self.last_zone_coverage_percent_by_id = {}
        self.last_coverage_triggered_zone_ids = set()

    def _video_prebuffer_seconds(self) -> float:
        if self.config.prebuffer_frames > 0 and self.fps_guess > 0:
            return self.config.prebuffer_frames / max(self.fps_guess, 1.0)
        return max(VIDEO_PREBUFFER_SECONDS, self.config.prebuffer_seconds)

    def _start_video_recording(self, timestamp: float, triggered_zone_ids: Set[str], confirmed: bool = False) -> None:
        if self.video_recording is not None:
            return
        cooldown = max(self.config.event_idle_seconds, 5.0)
        if self.last_recording_ended_at > 0 and timestamp - self.last_recording_ended_at < cooldown:
            return
        vid_zone_id = self._primary_zone_id_for(triggered_zone_ids)
        record_seconds = self._record_seconds_for_zone_ids(triggered_zone_ids)
        video_dir = self._video_output_dir()
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_path = video_dir / f"{self.config.camera_name}_{stamp}.mp4"
        temp_output_path = self._video_temp_dir() / output_path.name
        prebuffer_frames = [
            (frame_timestamp, frame_copy)
            for frame_timestamp, frame_copy in self.prebuffer
            if frame_timestamp >= timestamp - self._video_prebuffer_seconds()
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
        fps = max(5.0, min(60.0, self.fps_guess or processing_fps or 20.0))
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
        first_frame_at = timestamp
        min_frame_interval = 1.0 / fps
        try:
            for frame_timestamp, buffered_frame in prebuffer_frames:
                if last_written_at and frame_timestamp - last_written_at < min_frame_interval:
                    continue
                prepared_frame = self._prepare_video_frame(buffered_frame, frame_size)
                writer.write(prepared_frame)
                if not last_written_at:
                    first_frame_at = frame_timestamp
                last_written_at = frame_timestamp
        except Exception:
            writer.release()
            temp_output_path.unlink(missing_ok=True)
            raise
        self.video_recording = VideoRecording(
            started_at=timestamp,
            first_frame_at=first_frame_at,
            ends_at=timestamp + record_seconds,
            record_seconds=record_seconds,
            started_from_zone_ids=set(triggered_zone_ids),
            output_path=output_path,
            temp_output_path=temp_output_path,
            writer=writer,
            last_written_at=last_written_at,
            frame_size=frame_size,
            fps=fps,
            last_frame=self._prepare_video_frame(sample_frame, frame_size),
            source_url=self.config.rtsp_url,
            confirmed=confirmed,
        )
        status = "confirmed motion" if confirmed else "motion warning"
        recording_mode = "main capture original frames"
        logging.info(
            "Started %.0f-second video recording at %s on %s for zones=%s using %s",
            record_seconds,
            output_path,
            status,
            ",".join(sorted(triggered_zone_ids)) or "unknown",
            recording_mode,
        )
        self._add_event_log(
            "video",
            f"Started local video {output_path.name} for {int(round(record_seconds))}s on {status} using {recording_mode} for zones={','.join(sorted(triggered_zone_ids)) or 'unknown'}",
            zone_id=vid_zone_id,
        )

    def _append_video_recording_frame(self, timestamp: float, frame) -> None:
        recording = self.video_recording
        if recording is None:
            return
        if timestamp > recording.ends_at:
            self._stop_video_recording()
            return
        min_frame_interval = 1.0 / recording.fps
        if timestamp <= recording.last_written_at:
            return
        prepared_frame = self._prepare_video_frame(frame, recording.frame_size)
        elapsed = timestamp - recording.last_written_at
        frames_to_write = max(1, int(round(elapsed / min_frame_interval)))
        frames_to_write = min(frames_to_write, max(1, int(recording.fps * 5)))
        for _ in range(frames_to_write):
            recording.writer.write(prepared_frame)
        recording.last_written_at += frames_to_write * min_frame_interval
        if recording.last_written_at > timestamp:
            recording.last_written_at = timestamp
        recording.last_frame = prepared_frame

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
            self.last_video_recording = recording
            self.last_recording_ended_at = time.time()
        rec_zone_id = self._primary_zone_id_for(recording.started_from_zone_ids)
        if release_error:
            if not self._video_file_is_readable(recording.temp_output_path):
                recording.temp_output_path.unlink(missing_ok=True)
                logging.error(
                    "Failed to close video recording %s",
                    recording.output_path,
                    exc_info=(type(release_error), release_error, release_error.__traceback__),
                )
                self._add_event_log("video", f"Failed to close video {recording.output_path.name}", zone_id=rec_zone_id)
                return
            logging.warning(
                "Video recorder reported close error for %s, but the temp MP4 is readable; keeping it: %s",
                recording.output_path,
                release_error,
            )
            self._add_event_log("video", f"Recovered readable video {recording.output_path.name} after close warning", zone_id=rec_zone_id)
        if not recording.confirmed:
            recording.temp_output_path.unlink(missing_ok=True)
            logging.info("Discarded unconfirmed video recording %s", recording.temp_output_path)
            self._add_event_log("video", f"Discarded unconfirmed video {recording.output_path.name}", zone_id=rec_zone_id)
            return
        try:
            recording.temp_output_path.replace(recording.output_path)
            self._write_video_metadata(recording)
        except Exception:
            recording.temp_output_path.unlink(missing_ok=True)
            logging.exception("Failed to finalize video recording %s", recording.output_path)
            self._add_event_log("video", f"Failed to finalize video {recording.output_path.name}", zone_id=rec_zone_id)
            return
        logging.info("Saved video recording to %s", recording.output_path)
        self._add_event_log("video", f"Saved video {recording.output_path.name}", zone_id=rec_zone_id)
        if recording.pending_events:
            thread = threading.Thread(
                target=self._save_video_events,
                args=(recording,),
                daemon=True,
            )
            thread.start()

    def _video_file_is_readable(self, video_path: Path) -> bool:
        if not video_path.exists() or not video_path.is_file() or video_path.stat().st_size <= 0:
            return False
        if not shutil.which("ffprobe"):
            return True
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return result.returncode == 0 and b"video" in result.stdout

    def _discard_unconfirmed_video_recording(self) -> None:
        recording = self.video_recording
        if recording is None or recording.confirmed:
            return
        self._stop_video_recording()

    def _start_alpr_capture_session(self, timestamp: float, confirmed: bool = False) -> None:
        return

    def _run_alpr_capture_session(self, stop_event: threading.Event, frames: Deque[CandidateFrame]) -> None:
        if shutil.which("ffmpeg") and self._effective_alpr_rtsp_url():
            try:
                self._run_ffmpeg_alpr_capture_session(stop_event, frames)
                return
            except Exception:
                logging.exception("Persistent ffmpeg ALPR sampler failed; falling back to single-frame capture")
        min_interval = 1.0 / self.config.alpr_capture_fps
        last_sample_at = 0.0
        failed_captures = 0
        while not stop_event.is_set():
            timestamp = time.time()
            if timestamp - last_sample_at < min_interval:
                stop_event.wait(min(0.1, min_interval))
                continue
            try:
                frames.append(self._capture_alpr_candidate(timestamp))
                failed_captures = 0
            except Exception as exc:
                failed_captures += 1
                logging.warning("Skipping bad 101 ALPR sample %s: %s", failed_captures, exc)
                stop_event.wait(min(0.5, min_interval))
            finally:
                last_sample_at = timestamp

    def _run_ffmpeg_alpr_capture_session(self, stop_event: threading.Event, frames: Deque[CandidateFrame]) -> None:
        high_res_url = self._effective_alpr_rtsp_url()
        if not high_res_url:
            return
        transport = self._rtsp_option_value("rtsp_transport", "tcp") or "tcp"
        fps = max(0.1, self.config.alpr_capture_fps)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+discardcorrupt",
            "-rtsp_transport",
            transport,
            "-i",
            high_res_url,
            "-an",
            "-vf",
            f"fps={fps:.3f}",
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if process.stdout is None:
            process.kill()
            raise RuntimeError("Unable to open ffmpeg ALPR sampler")
        buffer = b""
        bad_frames = 0
        opened_at = time.time()
        try:
            while not stop_event.is_set():
                chunk = process.stdout.read(65536)
                if not chunk:
                    if process.poll() is not None:
                        break
                    continue
                buffer += chunk
                while not stop_event.is_set():
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        buffer = buffer[-2:]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start > 0:
                            buffer = buffer[start:]
                        break
                    jpeg_bytes = buffer[start : end + 2]
                    buffer = buffer[end + 2 :]
                    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if time.time() - opened_at < self.config.alpr_capture_warmup_seconds:
                        continue
                    if frame is None or self._looks_like_bad_frame(frame):
                        bad_frames += 1
                        if bad_frames in {1, 5, 10} or bad_frames % 25 == 0:
                            logging.warning("Dropped bad 101 ALPR sampler frame count=%s", bad_frames)
                        continue
                    bad_frames = 0
                    frames.append(self._candidate_from_alpr_frame(frame=frame, timestamp=time.time()))
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _capture_alpr_candidate(self, timestamp: Optional[float] = None) -> CandidateFrame:
        frame = self._capture_single_alpr_frame()
        candidate_timestamp = time.time() if timestamp is None else timestamp
        return self._candidate_from_alpr_frame(frame=frame, timestamp=candidate_timestamp)

    def _candidate_from_alpr_frame(self, frame: Any, timestamp: float) -> CandidateFrame:
        zone_ids = set(self.event.zones_triggered) if self.event else None
        crop = self._plate_crop(frame, zone_ids=zone_ids)
        return CandidateFrame(
            frame=frame.copy(),
            timestamp=timestamp,
            motion_area=0,
            sharpness=self._compute_sharpness(crop),
            jpeg_bytes=self._encode_jpeg(frame),
            source="alpr-rtsp",
            zone_ids=set(zone_ids or ()),
        )

    def _capture_single_alpr_frame(self):
        high_res_url = self._effective_alpr_rtsp_url()
        if not high_res_url:
            raise RuntimeError("ALPR 101 stream is not configured")
        if shutil.which("ffmpeg"):
            return self._capture_single_alpr_frame_with_ffmpeg(high_res_url)
        if self.config.rtsp_capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.config.rtsp_capture_options
        capture = cv2.VideoCapture(high_res_url, cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                raise RuntimeError("Unable to open ALPR RTSP stream")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = self._read_first_available_frame(capture, attempts=20)
            if frame is None:
                raise RuntimeError("Failed to read frame from ALPR RTSP stream")
            return frame
        finally:
            capture.release()

    def _capture_single_alpr_frame_with_ffmpeg(self, high_res_url: str):
        transport = self._rtsp_option_value("rtsp_transport", "tcp") or "tcp"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+discardcorrupt",
            "-rtsp_transport",
            transport,
            "-i",
            high_res_url,
            "-an",
            "-frames:v",
            "20",
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(10.0, self.config.request_timeout_seconds),
        )
        best_frame = None
        best_sharpness = -1.0
        for jpeg_bytes in self._iter_jpeg_frames(result.stdout):
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and not self._looks_like_bad_frame(frame):
                zone_ids = set(self.event.zones_triggered) if self.event else None
                sharpness = self._compute_sharpness(self._plate_crop(frame, zone_ids=zone_ids))
                if sharpness > best_sharpness:
                    best_frame = frame
                    best_sharpness = sharpness
        if best_frame is not None:
            return best_frame
        error = result.stderr.decode("utf-8", errors="replace").strip()
        if result.returncode != 0 and error:
            raise RuntimeError(f"ffmpeg ALPR 101 capture failed: {error}")
        raise RuntimeError("ffmpeg ALPR 101 capture produced only corrupted frames")

    def _read_first_available_frame(self, capture, attempts: int = 5):
        for _ in range(max(1, attempts)):
            ok, frame = capture.read()
            if ok and frame is not None and not self._looks_like_bad_frame(frame):
                return frame
        return None

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

    def _discard_unconfirmed_alpr_capture_session(self) -> None:
        with self.alpr_capture_lock:
            session = self.alpr_capture_session
            if session is None or session.confirmed:
                return
            self.alpr_capture_session = None
        session.stop_event.set()
        session.thread.join(timeout=1.0)
        logging.info("Discarded high-resolution ALPR sampler after motion warning cleared")

    def _uses_high_res_image_stream(self) -> bool:
        return False

    def _effective_alpr_rtsp_url(self) -> str:
        return self.config.rtsp_url

    def _derive_hikvision_101_url(self, rtsp_url: str) -> str:
        return rtsp_url

    def _rtsp_option_value(self, name: str, default: str = "") -> str:
        parts = [part.strip() for part in self.config.rtsp_capture_options.split("|") if part.strip()]
        for part in parts:
            key, separator, value = part.partition(";")
            if separator and key.strip().lower() == name.lower():
                return value.strip()
        return default

    def _iter_jpeg_frames(self, payload: bytes):
        buffer = payload
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                return
            end = buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                return
            yield buffer[start : end + 2]
            buffer = buffer[end + 2 :]

    def _looks_like_bad_frame(self, frame) -> bool:
        if frame is None or frame.size == 0:
            return True
        height, width = frame.shape[:2]
        if width < 8 or height < 8:
            return True
        sample_width = min(320, width)
        sample_height = max(8, int(round(sample_width * height / max(1, width))))
        sample = cv2.resize(frame, (sample_width, sample_height))
        means = sample.reshape(-1, 3).mean(axis=0)
        blue, green, red = (float(value) for value in means)
        stddev = float(sample.std())
        channel_spread = max(blue, green, red) - min(blue, green, red)
        green_artifact = green > max(blue, red) + 40.0 and blue < 30.0 and red < 30.0
        flat_gray_artifact = stddev < 10.0 and channel_spread < 8.0
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).astype("int16")
        vertical_grad = float(np.mean(np.abs(np.diff(gray, axis=0)))) if gray.shape[0] > 1 else 0.0
        horizontal_grad = float(np.mean(np.abs(np.diff(gray, axis=1)))) if gray.shape[1] > 1 else 0.0
        column_std = float(gray.std(axis=0).mean())
        stripe_artifact = vertical_grad < 0.08 and horizontal_grad > 0.4 and column_std < 0.5 and stddev > 5.0
        return green_artifact or flat_gray_artifact or stripe_artifact

    def _looks_like_bad_live_jpeg(self, jpeg_bytes: bytes) -> bool:
        if len(jpeg_bytes) < 2048:
            return True
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return self._looks_like_bad_frame(frame)

    def _write_video_metadata(self, recording: VideoRecording) -> None:
        metadata_path = recording.output_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "camera_name": self.config.camera_name,
                    "video_path": str(recording.output_path),
                    "source": "capture-stream",
                    "recording_mode": "main_capture_original_frames",
                    "started_at_epoch": recording.started_at,
                    "first_frame_at_epoch": recording.first_frame_at,
                    "ends_at_epoch": recording.ends_at,
                    "zone_ids": sorted(recording.started_from_zone_ids),
                    "event_count": len(recording.pending_events),
                    "prebuffer_seconds": self._video_prebuffer_seconds(),
                    "recording_seconds": recording.record_seconds,
                },
                indent=2,
            )
        )

    def _prebuffer_frame_limit(self) -> int:
        if self.config.prebuffer_frames > 0:
            return self.config.prebuffer_frames
        return int(self.fps_guess * max(0.1, self._video_prebuffer_seconds()))

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

    def _zone_crop(self, frame, zone: MotionZone):
        x1, y1, x2, y2 = self._zone_bounds(frame, zone)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else frame

    def _update_cumulative_zone_coverage(
        self,
        zone_motion_masks_by_id: dict[str, Any],
    ) -> Tuple[dict[str, float], Set[str]]:
        cumulative_zone_coverage_percent_by_id: dict[str, float] = {}
        coverage_triggered_zone_ids: Set[str] = set()
        for zone in self.motion_zones:
            if not zone.enabled:
                continue
            zone_mask = zone_motion_masks_by_id.get(zone.zone_id)
            if zone_mask is None:
                cumulative_mask = self.cumulative_zone_masks_by_id.get(zone.zone_id)
            else:
                binary_mask = (zone_mask > 0).astype(np.uint8)
                existing_mask = self.cumulative_zone_masks_by_id.get(zone.zone_id)
                if existing_mask is None or getattr(existing_mask, "shape", None) != binary_mask.shape:
                    cumulative_mask = binary_mask
                else:
                    cumulative_mask = np.maximum(existing_mask, binary_mask)
                self.cumulative_zone_masks_by_id[zone.zone_id] = cumulative_mask
            if cumulative_mask is None:
                cumulative_zone_coverage_percent_by_id[zone.zone_id] = 0.0
                continue
            zone_pixel_area = max(1, cumulative_mask.shape[0] * cumulative_mask.shape[1])
            cumulative_coverage_percent = min(100.0, (float(cv2.countNonZero(cumulative_mask)) / float(zone_pixel_area)) * 100.0)
            cumulative_zone_coverage_percent_by_id[zone.zone_id] = cumulative_coverage_percent
            if cumulative_coverage_percent >= zone.coverage_trigger_percent > 0.0:
                coverage_triggered_zone_ids.add(zone.zone_id)
        return cumulative_zone_coverage_percent_by_id, coverage_triggered_zone_ids

    def _alpr_crop_zone(self, zone_ids: Optional[Set[str]] = None) -> Optional[MotionZone]:
        if zone_ids:
            for zone in self.motion_zones:
                if zone.zone_id in zone_ids and zone.enabled and zone.use_fast_alpr:
                    return zone
        return None

    def _plate_roi_bounds(
        self,
        frame,
        zone_ids: Optional[Set[str]] = None,
    ) -> Tuple[int, int, int, int]:
        crop_zone = self._alpr_crop_zone(zone_ids)
        effective_plate_roi = crop_zone.roi if crop_zone else (self.config.plate_roi or self.config.roi)
        return self._normalized_roi_bounds(frame, effective_plate_roi)

    def _draw_monitor_overlays(self, frame):
        for zone in self.motion_zones:
            if not zone.enabled:
                continue
            zone_x1, zone_y1, zone_x2, zone_y2 = self._zone_bounds(frame, zone)
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), zone.overlay_bgr, 2)
        return frame

    def _detect_motion(
        self,
        frame,
    ) -> Tuple[int, bool, Optional[Tuple[int, int, int, int]], Any, Set[str], dict[str, int], dict[str, float], dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.background.apply(gray)
        _, mask = cv2.threshold(mask, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        total_area = 0
        best_box = None
        triggered_zone_ids: Set[str] = set()
        zone_area_by_id: dict[str, int] = {}
        zone_coverage_percent_by_id: dict[str, float] = {}
        zone_motion_masks_by_id: dict[str, Any] = {}
        for zone in self.motion_zones:
            if not zone.enabled:
                continue
            x1, y1, x2, y2 = self._zone_bounds(frame, zone)
            zone_mask = mask[y1:y2, x1:x2]
            zone_motion_masks_by_id[zone.zone_id] = zone_mask.copy()
            zone_pixel_area = max(1, zone_mask.shape[0] * zone_mask.shape[1])
            moving_pixel_area = int(cv2.countNonZero(zone_mask))
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
            zone_coverage_percent = min(100.0, (moving_pixel_area / float(zone_pixel_area)) * 100.0)
            zone_coverage_percent_by_id[zone.zone_id] = zone_coverage_percent
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
        return (
            total_area,
            had_motion,
            absolute_best_box,
            overlay,
            triggered_zone_ids,
            zone_area_by_id,
            zone_coverage_percent_by_id,
            zone_motion_masks_by_id,
        )

    def _annotate_motion_overlay(
        self,
        overlay,
        motion_area: int,
        best_box: Optional[Tuple[int, int, int, int]],
        confirmed_motion: bool,
        triggered_zone_ids: Set[str],
        zone_area_by_id: dict[str, int],
        zone_coverage_percent_by_id: dict[str, float],
        coverage_triggered_zone_ids: Set[str],
    ):
        if best_box:
            box_color = (0, 255, 0) if confirmed_motion else (0, 200, 255)
            cv2.rectangle(overlay, (best_box[0], best_box[1]), (best_box[2], best_box[3]), box_color, 2)
        if confirmed_motion and coverage_triggered_zone_ids:
            status = f"confirmed coverage={','.join(sorted(coverage_triggered_zone_ids))}"
        elif confirmed_motion:
            status = "confirmed"
        else:
            status = f"warming {min(self.motion_streak, self.config.min_consecutive_hits)}/{self.config.min_consecutive_hits}"
        zones_text = ",".join(sorted(triggered_zone_ids)) if triggered_zone_ids else "none"
        per_zone = " ".join(
            f"{zone.zone_id}={int(zone_area_by_id.get(zone.zone_id, 0))} ({zone_coverage_percent_by_id.get(zone.zone_id, 0.0):.0f}%)"
            for zone in self.motion_zones
            if zone.enabled
        )
        label = f"zones={zones_text} total={motion_area} status={status}"
        detail_label = per_zone or "no-zones"
        self.latest_motion_status = f"{label} | {detail_label}"
        return overlay

    def _on_motion(self, timestamp: float, frame, motion_area: int, triggered_zone_ids: Set[str]) -> None:
        sharpness = self._compute_sharpness(frame)
        candidate = CandidateFrame(
            frame=frame.copy(),
            timestamp=timestamp,
            motion_area=motion_area,
            sharpness=sharpness,
            zone_ids=set(triggered_zone_ids),
        )

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
            self._add_event_log(
                "motion",
                f"Opened motion event zones={','.join(sorted(triggered_zone_ids)) or 'unknown'}",
                zone_id=self._primary_zone_id_for(triggered_zone_ids),
            )
            self._append_event_frame(timestamp, frame, count_as_postbuffer=False)
            return

        self.event.trigger_count += 1
        self.event.last_motion_at = timestamp
        self.event.frames_since_motion = 0
        self.event.zones_triggered.update(triggered_zone_ids)
        recording = self.video_recording
        if recording is not None:
            recording.started_from_zone_ids.update(triggered_zone_ids)
            recording.ends_at = max(recording.ends_at, timestamp + self._record_seconds_for_zone_ids(self.event.zones_triggered))
            recording.record_seconds = max(recording.record_seconds, recording.ends_at - recording.started_at)
        self.event.append_candidate(candidate)
        self._append_event_frame(timestamp, frame, count_as_postbuffer=False)

    def _append_event_frame(self, timestamp: float, frame, count_as_postbuffer: bool) -> None:
        if not self.event:
            return
        self.event.append_frame(timestamp, frame.copy())
        self.event.last_frame_at = timestamp
        if count_as_postbuffer:
            self.event.frames_since_motion += 1

    def _event_ready_to_finalize(self, timestamp: float) -> bool:
        if not self.event:
            return False
        enough_idle_time = timestamp - self.event.last_motion_at >= self.config.event_idle_seconds
        enough_post_frames = self.event.frames_since_motion >= self._postbuffer_frame_limit()
        return enough_idle_time and enough_post_frames

    def _run_detection_pipeline(
        self,
        frame,
        event_dir: Path,
        base_name: str,
        event_name: str,
        detected_at_epoch: float,
        enable_alpr: bool = True,
        jpeg_bytes: Optional[bytes] = None,
        source_label: Optional[str] = None,
        zone_ids: Optional[Set[str]] = None,
    ) -> dict[str, Any]:
        frame_path = event_dir / f"{base_name}.jpg"
        json_path = event_dir / f"{base_name}.json"
        frame_to_save = frame
        if frame is not None:
            frame_to_save = self._plate_crop(frame, zone_ids=zone_ids) if enable_alpr else frame
        if jpeg_bytes is None or frame_to_save is not frame:
            jpeg_bytes = self._encode_jpeg(frame_to_save)
        frame_path.write_bytes(jpeg_bytes)
        source = source_label or ("alpr-rtsp" if frame is not None and self._uses_high_res_image_stream() else "capture")
        (event_dir / f"{base_name}.meta.json").write_text(
            json.dumps({"source": source, "detected_at_epoch": detected_at_epoch}, indent=2)
        )
        relative_frame_path = self._relative_image_path(frame_path)

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

    def _save_triggered_zone_images(
        self,
        event: Event,
        event_dir: Path,
        selected: List[CandidateFrame],
    ) -> Tuple[List[Path], List[dict[str, Any]]]:
        image_paths: List[Path] = []
        summary: List[dict[str, Any]] = []
        for zone in self.motion_zones:
            if zone.zone_id == "yellow" or zone.zone_id not in event.zones_triggered:
                continue
            candidates = [
                candidate
                for candidate in selected
                if candidate.frame is not None and candidate.source == "video"
            ]
            if not candidates:
                logging.warning("No video-extracted frame available for %s zone image", zone.zone_id)
                self._add_event_log("image", f"No video-extracted frame available for {zone.zone_id} zone image", zone_id=zone.zone_id)
                continue
            candidates = self._select_timeline_candidates(candidates, zone.image_count)
            for index, candidate in enumerate(candidates, start=1):
                crop = self._zone_crop(candidate.frame, zone)
                image_path = event_dir / f"zone_{zone.zone_id}_{index:02d}.jpg"
                image_path.write_bytes(self._encode_jpeg(crop))
                relative_path = self._relative_image_path(image_path)
                image_paths.append(image_path)
                self._add_event_log("image", f"Saved {zone.zone_id} zone image {image_path.name} from local video", zone_id=zone.zone_id)
                summary.append(
                    {
                        "zone_id": zone.zone_id,
                        "label": zone.label,
                        "image_relative_path": relative_path,
                        "source_frame_timestamp": candidate.timestamp,
                        "source": candidate.source,
                    }
                )
        return image_paths, summary

    def _event_sends_telegram(self, event: Event) -> bool:
        for zone_id in event.zones_triggered:
            zone = self._find_motion_zone(zone_id)
            if zone and zone.send_telegram:
                return True
        return False

    def _telegram_zone_image_paths(self, zone_image_paths: List[Path], zone_image_summary: List[dict[str, Any]]) -> List[Path]:
        image_path_by_relative = {
            self._relative_image_path(path): path
            for path in zone_image_paths
        }
        telegram_paths: List[Path] = []
        for item in zone_image_summary:
            zone = self._find_motion_zone(str(item.get("zone_id") or ""))
            relative_path = str(item.get("image_relative_path") or "")
            image_path = image_path_by_relative.get(relative_path)
            if zone and zone.send_telegram and image_path:
                telegram_paths.append(image_path)
        return telegram_paths

    def _primary_zone_id_for(self, zone_ids: Set[str]) -> str:
        for zone in self.motion_zones:
            if zone.zone_id in zone_ids:
                return zone.zone_id
        return ""

    def _image_log_category(self, zone_ids: Set[str]) -> str:
        return "image"

    def _video_log_category(self, zone_ids: Set[str]) -> str:
        return "video"

    def _finalize_event(self, reason: str) -> None:
        if not self.event:
            return
        event = self.event
        self.event = None
        high_res_candidates = self._stop_alpr_capture_session()
        if high_res_candidates:
            for candidate in high_res_candidates:
                event.append_candidate(candidate)
            logging.info("Attached %s high-resolution ALPR frames to finalized event", len(high_res_candidates))
        logging.info("Closing event: %s", reason)
        zone_id = self._primary_zone_id_for(event.zones_triggered)
        self._add_event_log("motion", f"Closed motion event: {reason}", zone_id=zone_id)
        if not self._event_uses_fast_alpr(event):
            self._add_event_log(
                "image",
                "Skipped automatic image creation because Send to ALPR is disabled for the triggered zone",
                zone_id=zone_id,
            )
            return
        recording = self.video_recording or self.last_video_recording
        if recording is not None and recording.confirmed:
            recording.pending_events.append(event)
            logging.info(
                "Queued event for frame extraction after video %s is finalized",
                recording.output_path,
            )
            self._add_event_log("image", f"Queued image extraction after video {recording.output_path.name} closes", zone_id=zone_id)
            if self.video_recording is None:
                self.last_video_recording = None
                thread = threading.Thread(
                    target=self._save_video_events,
                    args=(recording,),
                    daemon=True,
                )
                thread.start()
            return
        thread = threading.Thread(
            target=self._save_finalized_event,
            args=(event, reason),
            daemon=True,
        )
        thread.start()

    def _save_finalized_event(self, event: Event, reason: str) -> None:
        try:
            while self.video_recording is not None:
                time.sleep(0.5)
            self._save_finalized_event_unchecked(event, reason)
        except Exception:
            logging.exception("Failed to save finalized event: %s", reason)

    def _save_video_events(self, recording: VideoRecording) -> None:
        while self.video_recording is not None:
            time.sleep(0.5)
        for event in list(recording.pending_events):
            event_zone_id = self._primary_zone_id_for(event.zones_triggered)
            try:
                self._add_event_log("image", f"Extracting event images from local video {recording.output_path.name}", zone_id=event_zone_id)
                self._save_finalized_event_unchecked(
                    event,
                    "video finalized",
                    source_video_path=recording.output_path,
                    source_video_first_frame_at=recording.first_frame_at,
                    source_video_ends_at=recording.first_frame_at + self._saved_video_duration_seconds(recording.output_path),
                )
            except Exception:
                logging.exception("Failed to extract event images from %s", recording.output_path)
                self._add_event_log("image", f"Failed extracting images from {recording.output_path.name}", zone_id=event_zone_id)

    def _send_recording_active_extraction_error(self, handler: BaseHTTPRequestHandler) -> None:
        body = json.dumps(
            {
                "error": "A video is still being recorded. Image extraction is available after the recording is saved.",
            }
        ).encode("utf-8")
        handler.send_response(HTTPStatus.CONFLICT)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _save_finalized_event_unchecked(
        self,
        event: Event,
        reason: str,
        source_video_path: Optional[Path] = None,
        source_video_first_frame_at: Optional[float] = None,
        source_video_ends_at: Optional[float] = None,
    ) -> None:
        event_zone_id = self._primary_zone_id_for(event.zones_triggered)
        if event.trigger_count < self.config.min_consecutive_hits:
            logging.info("Discarded event with only %s motion hits", event.trigger_count)
            self._add_event_log("motion", f"Discarded event with only {event.trigger_count} motion hits", zone_id=event_zone_id)
            self._stop_alpr_capture_session()
            return

        if source_video_path is None or source_video_first_frame_at is None:
            logging.warning("No finalized local video is available; using in-memory event frames instead")
            self._add_event_log("image", "No finalized local video is available; using in-memory event frames instead", zone_id=event_zone_id)

        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        event_dir = self.config.image_output_dir / f"{self.config.camera_name}_{stamp}"
        event_dir.mkdir(parents=True, exist_ok=True)
        image_limit = self._image_limit_for_zone_ids(event.zones_triggered)
        event_start_at = event.frames[0][0] if event.frames else event.started_at
        event_end_at = source_video_ends_at if source_video_ends_at else event.last_frame_at
        selected: List[CandidateFrame] = []
        high_res_candidates = [
            candidate
            for candidate in event.candidates
            if candidate.frame is not None and candidate.source == "alpr-rtsp"
        ]
        if high_res_candidates:
            selected.extend(self._select_timeline_candidates(high_res_candidates, image_limit))
        if source_video_path is not None and source_video_first_frame_at is not None:
            remaining = max(0, image_limit - len(selected))
            if remaining > 0:
                selected.extend(
                    self._extract_video_event_candidates(
                        source_video_path,
                        source_video_first_frame_at,
                        event_start_at,
                        event_end_at,
                        event,
                        remaining,
                    )
                )
        if selected:
            selected = self._select_timeline_candidates(selected, image_limit)
        if not selected:
            selected = self._fallback_event_candidates(event, image_limit)
            if selected:
                logging.warning(
                    "Falling back to in-memory event frames for %s because video extraction returned no frames",
                    source_video_path or "current event",
                )
                self._add_event_log("image", "Falling back to in-memory event frames because video extraction returned no images", zone_id=event_zone_id)
            else:
                logging.error("No images extracted from %s. Skipping motion-event image creation.", source_video_path or "current event")
                if source_video_path is not None:
                    self._add_event_log("image", f"Skipped event images: no frames extracted from {source_video_path.name}", zone_id=event_zone_id)
                else:
                    self._add_event_log("image", "Skipped event images: no frames were available", zone_id=event_zone_id)
                self._prune_saved_images()
                return
        openalpr_results = []
        fast_alpr_results = []
        plate_detections: List[PlateDetection] = []
        saved_frame_paths: List[Path] = []
        alpr_enabled = self._event_uses_fast_alpr(event)
        openalpr_uploaded = 0
        openalpr_skipped = 0
        openalpr_failed = 0
        for index, candidate in enumerate(selected, start=1):
            pipeline = self._run_detection_pipeline(
                candidate.frame,
                event_dir,
                f"frame_{index:02d}",
                event_dir.name,
                candidate.timestamp,
                enable_alpr=alpr_enabled,
                jpeg_bytes=candidate.jpeg_bytes,
                source_label=candidate.source,
                zone_ids=candidate.zone_ids,
            )
            fast_alpr_results.extend(pipeline["fast_alpr_results"])
            openalpr_results.extend(pipeline["openalpr_results"])
            plate_detections.extend(pipeline["plate_detections"])
            saved_frame_paths.append(pipeline["frame_path"])
            if pipeline["openalpr_skipped_reason"]:
                openalpr_skipped += 1
            elif pipeline["openalpr_error"]:
                openalpr_failed += 1
            elif pipeline["openalpr_results"]:
                openalpr_uploaded += 1
        if openalpr_uploaded or openalpr_skipped or openalpr_failed:
            parts = []
            if openalpr_uploaded:
                parts.append(f"{openalpr_uploaded} uploaded")
            if openalpr_skipped:
                parts.append(f"{openalpr_skipped} skipped")
            if openalpr_failed:
                parts.append(f"{openalpr_failed} failed")
            logging.info("OpenALPR results for %s: %s", event_dir.name, ", ".join(parts))
            self._add_event_log("lpr", f"OpenALPR: {', '.join(parts)} across {len(selected)} frames", zone_id=event_zone_id)

        zone_image_paths, zone_image_summary = self._save_triggered_zone_images(event, event_dir, selected)
        summary = {
            "camera_name": self.config.camera_name,
            "started_at_epoch": event.started_at,
            "ended_at_epoch": event.last_frame_at,
            "trigger_count": event.trigger_count,
            "saved_frames": len(selected),
            "high_res_alpr_frames": sum(1 for candidate in selected if candidate.source == "alpr-rtsp"),
            "clip_path": self._relative_video_path(source_video_path)
            if source_video_path
            else None,
            "image_source": "video" if selected and all(candidate.source == "video" for candidate in selected) else "event-capture",
            "triggered_zones": self._event_zone_summary(event),
            "event_policy": "fast-alpr" if alpr_enabled else "images-only",
            "zone_images": zone_image_summary,
            "fast_alpr_results_count": len(fast_alpr_results),
            "openalpr_results_count": len(openalpr_results),
            "plates": [detection.__dict__ for detection in plate_detections],
        }
        (event_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._add_event_log("image", f"Saved {len(selected)} images into {event_dir.name}", zone_id=event_zone_id)
        if self._event_sends_telegram(event):
            telegram_zone_images = self._telegram_zone_image_paths(zone_image_paths, zone_image_summary)
            telegram_image_paths = telegram_zone_images or saved_frame_paths
            self._send_telegram_alert(event_dir.name, summary, telegram_image_paths)
        self._prune_saved_images()
        logging.info("Event saved to %s", event_dir)

    def _select_best_frames(
        self,
        event: Event,
        preferred_candidates: Optional[List[CandidateFrame]] = None,
        image_limit: Optional[int] = None,
    ) -> List[CandidateFrame]:
        image_limit = max(1, image_limit or self._image_limit_for_zone_ids(event.zones_triggered))
        if preferred_candidates:
            return self._select_timeline_candidates(preferred_candidates, image_limit)
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
            selected.extend(self._select_timeline_frames(event.frames, image_limit))
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
            if len(selected) >= image_limit:
                break
        if len(selected) < image_limit:
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
                if len(selected) >= image_limit:
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

    def _video_extraction_timestamps(self, start_at: float, end_at: float, limit: int) -> List[float]:
        if limit <= 0:
            return []
        end_at = max(start_at, end_at)
        if limit == 1 or end_at <= start_at:
            return [start_at]
        return [
            start_at + ((end_at - start_at) * index / max(1, limit - 1))
            for index in range(limit)
        ]

    def _fallback_event_candidates(self, event: Event, image_limit: Optional[int] = None) -> List[CandidateFrame]:
        selected = self._select_best_frames(event, image_limit=image_limit)
        fallback_candidates: List[CandidateFrame] = []
        for candidate in selected:
            jpeg_bytes = candidate.jpeg_bytes or self._encode_jpeg(candidate.frame)
            fallback_candidates.append(
                CandidateFrame(
                    frame=candidate.frame.copy(),
                    timestamp=candidate.timestamp,
                    motion_area=candidate.motion_area,
                    sharpness=candidate.sharpness,
                    jpeg_bytes=jpeg_bytes,
                    source=candidate.source,
                    zone_ids=set(candidate.zone_ids),
                )
            )
        return fallback_candidates

    def _extract_local_video_frame_with_ffmpeg(self, video_path: Path, position_seconds: float):
        if not shutil.which("ffmpeg"):
            return None
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{max(0.0, position_seconds):.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            logging.warning(
                "Timed out extracting frame at %.3fs from %s with ffmpeg",
                position_seconds,
                video_path,
            )
            return None
        if result.returncode != 0:
            return None
        for jpeg_bytes in self._iter_jpeg_frames(result.stdout):
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        return None

    def _extract_local_video_frame_with_opencv(self, video_path: Path, position_seconds: float):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, position_seconds) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            return frame
        finally:
            capture.release()

    def _extract_local_video_frame(self, video_path: Path, position_seconds: float):
        frame = self._extract_local_video_frame_with_opencv(video_path, position_seconds)
        if frame is not None:
            return frame
        return self._extract_local_video_frame_with_ffmpeg(video_path, position_seconds)

    def _extract_video_event_candidates(
        self,
        video_path: Path,
        first_frame_at: float,
        start_at: float,
        end_at: float,
        event: Event,
        limit: int,
    ) -> List[CandidateFrame]:
        if limit <= 0 or end_at < start_at:
            return []
        target_timestamps = self._video_extraction_timestamps(
            max(first_frame_at, start_at),
            max(max(first_frame_at, start_at), end_at),
            limit,
        )
        event_zone_id = self._primary_zone_id_for(event.zones_triggered)
        self._add_event_log(
            "image",
            f"Sampling {len(target_timestamps)} image timestamps from local video {video_path.name}",
            zone_id=event_zone_id,
        )
        if shutil.which("ffmpeg"):
            candidates: List[CandidateFrame] = []
            for frame_timestamp in target_timestamps:
                position_seconds = max(0.0, frame_timestamp - first_frame_at)
                frame = self._extract_local_video_frame_with_ffmpeg(video_path, position_seconds)
                if frame is None:
                    logging.warning("Unable to extract video frame at %.3fs from %s using ffmpeg", position_seconds, video_path)
                    self._add_event_log("image", f"Could not extract frame at {position_seconds:.3f}s from {video_path.name}", zone_id=event_zone_id)
                    continue
                candidates.append(
                    CandidateFrame(
                        frame=frame.copy(),
                        timestamp=frame_timestamp,
                        motion_area=0,
                        sharpness=self._compute_sharpness(self._plate_crop(frame, zone_ids=event.zones_triggered)),
                        jpeg_bytes=self._encode_jpeg(frame),
                        source="video",
                        zone_ids=set(event.zones_triggered),
                    )
                )
            self._add_event_log("image", f"Extracted {len(candidates)} frames from local video {video_path.name}", zone_id=event_zone_id)
            return candidates
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logging.warning("Unable to open saved video for frame extraction: %s", video_path)
            self._add_event_log("image", f"Could not open local video {video_path.name} for image extraction", zone_id=event_zone_id)
            return []
        candidates: List[CandidateFrame] = []
        try:
            for frame_timestamp in target_timestamps:
                position_ms = max(0.0, frame_timestamp - first_frame_at) * 1000.0
                capture.set(cv2.CAP_PROP_POS_MSEC, position_ms)
                ok, frame = capture.read()
                if not ok or frame is None:
                    logging.warning("Unable to extract video frame at %.3fs from %s", position_ms / 1000.0, video_path)
                    self._add_event_log("image", f"Could not extract frame at {position_ms / 1000.0:.3f}s from {video_path.name}", zone_id=event_zone_id)
                    continue
                candidates.append(
                    CandidateFrame(
                        frame=frame.copy(),
                        timestamp=frame_timestamp,
                        motion_area=0,
                        sharpness=self._compute_sharpness(self._plate_crop(frame, zone_ids=event.zones_triggered)),
                        jpeg_bytes=self._encode_jpeg(frame),
                        source="video",
                        zone_ids=set(event.zones_triggered),
                    )
                )
        finally:
            capture.release()
        self._add_event_log("image", f"Extracted {len(candidates)} frames from local video {video_path.name}", zone_id=event_zone_id)
        return candidates

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
        if not self.config.telegram_configured:
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
                    "send_telegram": zone.send_telegram,
                    "record_seconds": zone.record_seconds,
                    "image_count": zone.image_count,
                    "coverage_trigger_percent": zone.coverage_trigger_percent,
                }
            )
        return summary

    def _compute_sharpness(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _encode_jpeg(self, frame) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("Could not encode JPEG")
        return encoded.tobytes()

    def _plate_crop(self, frame, zone_ids: Optional[Set[str]] = None):
        if not self.config.plate_roi and not self.config.roi and not self._alpr_crop_zone(zone_ids):
            return frame
        x1, y1, x2, y2 = self._plate_roi_bounds(frame, zone_ids=zone_ids)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else frame

    def _plate_zoom_frame(self, frame):
        zone_ids = set(self.event.zones_triggered) if self.event else None
        crop = self._plate_crop(frame, zone_ids=zone_ids)
        height, width = crop.shape[:2]
        if width == 0 or height == 0:
            return frame
        target_width = max(width, 640)
        scale = target_width / float(width)
        target_height = max(height, int(height * scale))
        return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    def _update_latest_frames(self, display_frame, source_frame) -> None:
        now = time.time()
        # Throttle stream to 1 FPS during active recording to free CPU for ffmpeg
        effective_fps = 1.0 if self.video_recording is not None else self.config.stream_fps
        min_interval = 1.0 / max(1.0, effective_fps)
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
                if path == "/stats":
                    self._send_html(watcher._render_stats_page())
                    return
                if path == "/api/stats":
                    self._send_json(watcher._stats_snapshot())
                    return
                if path == "/api/time":
                    self._send_json(watcher._system_time_snapshot())
                    return
                if path == "/api/motion-status":
                    self._send_json({"status": watcher.latest_motion_status})
                    return
                if path == "/api/event-log":
                    self._send_json({"events": watcher._event_log_snapshot()})
                    return
                if path == "/api/extraction-status":
                    self._send_json(watcher.extraction_status)
                    return
                if path == "/api/telegram/settings":
                    self._send_json(watcher._telegram_settings_for_ui())
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
                    query = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
                    self._send_html(watcher._render_images_page(watcher._page_from_query(query)))
                    return
                if path == "/videos":
                    query = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
                    self._send_html(
                        watcher._render_videos_page(
                            watcher._page_from_query(query),
                            str((query.get("selected") or [""])[0]),
                        )
                    )
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

            def do_HEAD(self) -> None:
                path = unquote(self.path.split("?", 1)[0])
                if path.startswith("/events/"):
                    watcher._serve_event_file(self, path[len("/events/"):], head_only=True)
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
                if path == "/api/videos/delete":
                    watcher._handle_delete_video(self)
                    return
                if path == "/api/videos/extract-images":
                    watcher._handle_video_extract_images(self)
                    return
                if path == "/api/plates/clear":
                    watcher._handle_clear_plates(self)
                    return
                if path == "/api/telegram/test":
                    watcher._handle_telegram_test(self)
                    return
                if path == "/api/telegram/settings":
                    watcher._handle_telegram_settings_update(self)
                    return
                if path == "/api/env":
                    watcher._handle_env_update(self)
                    return
                if path == "/api/time/sync":
                    watcher._handle_system_time_sync(self)
                    return
                if path == "/api/images/alpr":
                    watcher._handle_saved_image_fast_alpr(self)
                    return
                if path == "/api/videos/snapshot-alpr":
                    watcher._handle_video_snapshot_fast_alpr(self)
                    return
                if path == "/api/videos/quick-alpr":
                    watcher._handle_video_quick_alpr(self)
                    return
                if path == "/api/videos/upload":
                    watcher._handle_video_upload(self)
                    return
                if path == "/api/extraction-cancel":
                    watcher._handle_extraction_cancel(self)
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

            def _write_mjpeg_frame(self, frame: bytes) -> None:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            def _send_direct_mjpeg_stream(self, source_url: str) -> None:
                if shutil.which("ffmpeg"):
                    self._send_ffmpeg_mjpeg_stream(source_url)
                    return
                if watcher.config.rtsp_capture_options:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = watcher.config.rtsp_capture_options
                capture = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
                if not capture.isOpened():
                    capture.release()
                    self.send_error(HTTPStatus.BAD_GATEWAY, "Unable to open live stream")
                    return
                capture.set(cv2.CAP_PROP_BUFFERSIZE, watcher.config.capture_buffer_size)
                self.send_response(HTTPStatus.OK)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                min_interval = 1.0 / max(1.0, watcher.config.stream_fps)
                failed_reads = 0
                try:
                    while True:
                        frame = watcher._read_first_available_frame(capture, attempts=3)
                        if frame is None:
                            failed_reads += 1
                            if failed_reads >= 10:
                                break
                            time.sleep(min_interval)
                            continue
                        failed_reads = 0
                        frame = watcher._resize_frame(frame)
                        self._write_mjpeg_frame(watcher._encode_jpeg(frame))
                        time.sleep(min_interval)
                except (BrokenPipeError, ConnectionResetError):
                    return
                finally:
                    capture.release()

            def _send_ffmpeg_mjpeg_stream(self, source_url: str) -> None:
                fps = max(1.0, watcher.config.stream_fps)
                max_width = max(320, min(1920, int(watcher.config.frame_width or 960)))
                transport = watcher._rtsp_option_value("rtsp_transport", "tcp") or "tcp"
                self.send_response(HTTPStatus.OK)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_good_jpeg: Optional[bytes] = None
                while True:
                    opened_at = time.time()
                    command = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-fflags",
                        "+discardcorrupt",
                        "-rtsp_transport",
                        transport,
                        "-i",
                        source_url,
                        "-an",
                        "-vf",
                        f"scale={max_width}:-2:force_original_aspect_ratio=decrease",
                        "-r",
                        f"{fps:.3f}",
                        "-q:v",
                        "4",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "mjpeg",
                        "pipe:1",
                    ]
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    if process.stdout is None:
                        process.kill()
                        return
                    buffer = b""
                    bad_frame_count = 0
                    try:
                        while True:
                            chunk = process.stdout.read(65536)
                            if not chunk:
                                if process.poll() is not None:
                                    break
                                time.sleep(0.05)
                                continue
                            buffer += chunk
                            while True:
                                start = buffer.find(b"\xff\xd8")
                                if start < 0:
                                    buffer = buffer[-2:]
                                    break
                                end = buffer.find(b"\xff\xd9", start + 2)
                                if end < 0:
                                    if start > 0:
                                        buffer = buffer[start:]
                                    break
                                jpeg = buffer[start : end + 2]
                                buffer = buffer[end + 2 :]
                                if time.time() - opened_at < watcher.config.live_stream_warmup_seconds:
                                    continue
                                if watcher._looks_like_bad_live_jpeg(jpeg):
                                    bad_frame_count += 1
                                    if last_good_jpeg:
                                        self._write_mjpeg_frame(last_good_jpeg)
                                    if bad_frame_count >= 8:
                                        raise RuntimeError("too many bad live frames")
                                    continue
                                bad_frame_count = 0
                                last_good_jpeg = jpeg
                                self._write_mjpeg_frame(jpeg)
                    except RuntimeError:
                        time.sleep(0.25)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    time.sleep(0.25)

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
                            self._write_mjpeg_frame(frame)
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.http_server = ThreadingHTTPServer((self.config.web_host, self.config.web_port), WatcherHandler)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()
        logging.info(
            "Web UI available at http://127.0.0.1:%s/ stats=http://127.0.0.1:%s/stats images=http://127.0.0.1:%s/images videos=http://127.0.0.1:%s/videos",
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
                "send_telegram": zone.send_telegram,
                "record_seconds": float(zone.record_seconds),
                "image_count": int(zone.image_count),
                "coverage_trigger_percent": float(zone.coverage_trigger_percent),
                "color": zone.color_hex,
                "fill": zone.fill_rgba,
            }
            for zone in self.motion_zones
        ]

    def _telegram_settings_for_ui(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.telegram_alerts_enabled),
            "configured": bool(self.config.telegram_configured),
            "alert_images": int(self.config.telegram_alert_images),
        }

    def _env_file_path(self) -> Path:
        configured_path = Path(os.getenv("ALPR_ENV_FILE") or os.getenv("APP_ENV_FILE") or ".env").expanduser()
        if configured_path == Path("/app/.env"):
            data_config_path = Path("/data/config/.env")
            if data_config_path.exists() or data_config_path.parent.exists():
                return data_config_path
        return configured_path

    def _read_env_text(self) -> str:
        env_path = self._env_file_path()
        if env_path.exists():
            return env_path.read_text()
        example_path = Path(".env.example")
        if example_path.exists():
            return example_path.read_text()
        return ""

    def _render_roi_editor_markup(self) -> str:
        return f"""<div class="panel">
  <div id="roi-editor" style="position:relative; max-width:100%;">
    <img id="roi-stream" style="width:100%; max-width:100%; border-radius:14px; border:1px solid #334155; display:block; background:#000;" alt="Live camera preview">
    <div id="roi-overlay" style="position:absolute; inset:0; pointer-events:auto; touch-action:none;">
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
      <div style="display:flex; align-items:center; gap:14px; margin-top:10px;">
        <button id="zone-fast-alpr-yellow" type="button" class="toggle-button" data-state="on" aria-label="Toggle fast-alpr" title="Fast-ALPR" style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:999px;border:1px solid #4ade80;background:transparent;color:#4ade80;cursor:pointer;padding:0;"><svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M17 7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h10c2.76 0 5-2.24 5-5s-2.24-5-5-5zm0 8c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/></svg></button>
        <span style="color:#cbd5e1;">Fast-ALPR</span>
        <button id="zone-telegram-yellow" type="button" class="toggle-button" data-state="on" aria-label="Toggle Telegram" title="Telegram" style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:999px;border:1px solid #4ade80;background:transparent;color:#4ade80;cursor:pointer;padding:0;"><svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M17 7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h10c2.76 0 5-2.24 5-5s-2.24-5-5-5zm0 8c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/></svg></button>
        <span style="color:#cbd5e1;">Telegram</span>
      </div>
      <label style="display:block; margin-top:10px; color:#cbd5e1;">Record time
        <select id="zone-record-seconds-yellow" style="display:block; width:100%; margin-top:6px; background:#07110f; color:#edf7f2; border:1px solid #334155; border-radius:8px; padding:8px;"></select>
      </label>
      <label style="display:block; margin-top:10px; color:#cbd5e1;">Saved images
        <select id="zone-image-count-yellow" style="display:block; width:100%; margin-top:6px; background:#07110f; color:#edf7f2; border:1px solid #334155; border-radius:8px; padding:8px;"></select>
      </label>
      <label style="display:block; margin-top:10px; color:#cbd5e1;">Coverage trigger
        <select id="zone-coverage-trigger-yellow" style="display:block; width:100%; margin-top:6px; background:#07110f; color:#edf7f2; border:1px solid #334155; border-radius:8px; padding:8px;"></select>
      </label>
    </div>
    <div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:12px;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
        <button id="zone-select-purple" type="button" style="background:#c084fc; color:#111827; border:0; border-radius:8px; padding:8px 12px; cursor:pointer; font-weight:700;">Edit Purple Zone</button>
        <label style="color:#e2e8f0;"><input id="zone-enabled-purple" type="checkbox"> Enabled</label>
      </div>
      <div style="display:flex; align-items:center; gap:14px; margin-top:10px;">
        <button id="zone-fast-alpr-purple" type="button" class="toggle-button" data-state="off" aria-label="Toggle fast-alpr" title="Fast-ALPR" style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:999px;border:1px solid #334155;background:transparent;color:#64748b;cursor:pointer;padding:0;"><svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M17 7H7C4.24 7 2 9.24 2 12s2.24 5 5 5h10c2.76 0 5-2.24 5-5s-2.24-5-5-5zM7 15c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/></svg></button>
        <span style="color:#cbd5e1;">Fast-ALPR</span>
        <button id="zone-telegram-purple" type="button" class="toggle-button" data-state="off" aria-label="Toggle Telegram" title="Telegram" style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:999px;border:1px solid #334155;background:transparent;color:#64748b;cursor:pointer;padding:0;"><svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M17 7H7C4.24 7 2 9.24 2 12s2.24 5 5 5h10c2.76 0 5-2.24 5-5s-2.24-5-5-5zM7 15c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/></svg></button>
        <span style="color:#cbd5e1;">Telegram</span>
      </div>
      <label style="display:block; margin-top:10px; color:#cbd5e1;">Record time
        <select id="zone-record-seconds-purple" style="display:block; width:100%; margin-top:6px; background:#07110f; color:#edf7f2; border:1px solid #334155; border-radius:8px; padding:8px;"></select>
      </label>
      <label style="display:block; margin-top:10px; color:#cbd5e1;">Saved images
        <select id="zone-image-count-purple" style="display:block; width:100%; margin-top:6px; background:#07110f; color:#edf7f2; border:1px solid #334155; border-radius:8px; padding:8px;"></select>
      </label>
      <label style="display:block; margin-top:10px; color:#cbd5e1;">Coverage trigger
        <select id="zone-coverage-trigger-purple" style="display:block; width:100%; margin-top:6px; background:#07110f; color:#edf7f2; border:1px solid #334155; border-radius:8px; padding:8px;"></select>
      </label>
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
        live_preview_url = json.dumps("/stream.mjpg")
        return f"""<script>
(() => {{
  const initialZones = {zones_json};
  let initialMotionArea = {min_motion_area};
  const livePreviewUrl = {live_preview_url};
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
  const zoneTelegram = {{
    yellow: document.getElementById('zone-telegram-yellow'),
    purple: document.getElementById('zone-telegram-purple'),
  }};
  const zoneRecordSeconds = {{
    yellow: document.getElementById('zone-record-seconds-yellow'),
    purple: document.getElementById('zone-record-seconds-purple'),
  }};
  const zoneImageCount = {{
    yellow: document.getElementById('zone-image-count-yellow'),
    purple: document.getElementById('zone-image-count-purple'),
  }};
  const zoneCoverageTrigger = {{
    yellow: document.getElementById('zone-coverage-trigger-yellow'),
    purple: document.getElementById('zone-coverage-trigger-purple'),
  }};
  if (!editor || !overlay || !img || !saveBtn || !resetBtn || !status || !sensitivity || !sensitivityValue || !sensitivitySave || handles.length !== 4 || !zoneBoxes.yellow || !zoneBoxes.purple || !zoneButtons.yellow || !zoneButtons.purple || !zoneEnabled.yellow || !zoneEnabled.purple || !zoneFastAlpr.yellow || !zoneFastAlpr.purple || !zoneTelegram.yellow || !zoneTelegram.purple || !zoneRecordSeconds.yellow || !zoneRecordSeconds.purple || !zoneImageCount.yellow || !zoneImageCount.purple || !zoneCoverageTrigger.yellow || !zoneCoverageTrigger.purple) return;

  let zones = initialZones.map((zone) => ({{ ...zone, roi: zone.roi.slice() }}));
  let activeZoneId = null;
  let dragCorner = null;
  let dragMove = null;
  let currentMotionArea = initialMotionArea;
  const minSize = 0.03;
  const handleSize = 18;

  const toggleOnSvg = '<svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M17 7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h10c2.76 0 5-2.24 5-5s-2.24-5-5-5zm0 8c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/></svg>';
  const toggleOffSvg = '<svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M17 7H7C4.24 7 2 9.24 2 12s2.24 5 5 5h10c2.76 0 5-2.24 5-5s-2.24-5-5-5zM7 15c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3z"/></svg>';

  function setToggle(btn, on) {{
    btn.dataset.state = on ? 'on' : 'off';
    btn.innerHTML = on ? toggleOnSvg : toggleOffSvg;
    btn.style.color = on ? '#4ade80' : '#64748b';
    btn.style.borderColor = on ? '#4ade80' : '#334155';
  }}

  let livePreviewRetryTimer = null;

  function scheduleLivePreviewRetry(message) {{
    if (message) setStatus(message, true);
    if (livePreviewRetryTimer) return;
    livePreviewRetryTimer = window.setTimeout(() => {{
      livePreviewRetryTimer = null;
      attachLivePreview(img);
    }}, 3000);
  }}

  function attachLivePreview(image) {{
    image.onload = () => setStatus('Motion zones: live stream connected.');
    image.onerror = () => scheduleLivePreviewRetry('Waiting for live stream...');
    image.src = `${{livePreviewUrl}}?v=${{Date.now()}}`;
  }}
  const recordTimeOptions = [
    {{ value: 15, label: '15 seconds' }},
    {{ value: 30, label: '30 seconds' }},
    {{ value: 45, label: '45 seconds' }},
    {{ value: 60, label: '1 minute' }},
    {{ value: 90, label: '1.5 minutes' }},
    {{ value: 120, label: '2 minutes' }},
    {{ value: 180, label: '3 minutes' }},
    {{ value: 240, label: '4 minutes' }},
    {{ value: 300, label: '5 minutes' }},
  ];
  const imageCountOptions = [1, 2, 3, 5, 10, 20, 40, 80, 120, 180, 240];
  const coverageTriggerOptions = [
    {{ value: 0, label: 'Off' }},
    {{ value: 10, label: '10%' }},
    {{ value: 20, label: '20%' }},
    {{ value: 30, label: '30%' }},
    {{ value: 40, label: '40%' }},
    {{ value: 50, label: '50%' }},
    {{ value: 60, label: '60%' }},
    {{ value: 70, label: '70%' }},
    {{ value: 80, label: '80%' }},
    {{ value: 90, label: '90%' }},
  ];

  function populateSelectOptions(select, options, labelForValue) {{
    select.innerHTML = options
      .map((option) => {{
        if (typeof option === 'object') {{
          return `<option value="${{option.value}}">${{option.label}}</option>`;
        }}
        return `<option value="${{option}}">${{labelForValue(option)}}</option>`;
      }})
      .join('');
  }}

  populateSelectOptions(zoneRecordSeconds.yellow, recordTimeOptions, (value) => `${{value}} seconds`);
  populateSelectOptions(zoneRecordSeconds.purple, recordTimeOptions, (value) => `${{value}} seconds`);
  populateSelectOptions(zoneImageCount.yellow, imageCountOptions, (value) => `${{value}} images`);
  populateSelectOptions(zoneImageCount.purple, imageCountOptions, (value) => `${{value}} images`);
  populateSelectOptions(zoneCoverageTrigger.yellow, coverageTriggerOptions, (value) => `${{value}}%`);
  populateSelectOptions(zoneCoverageTrigger.purple, coverageTriggerOptions, (value) => `${{value}}%`);

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

  function getNormalizedPointer(clientX, clientY) {{
    const rect = overlay.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {{
      x: clamp((clientX - rect.left) / rect.width, 0, 1),
      y: clamp((clientY - rect.top) / rect.height, 0, 1),
    }};
  }}

  function findZoneAt(x, y) {{
    const activeZone = activeZoneId ? findZone(activeZoneId) : null;
    const orderedZones = activeZone
      ? [activeZone, ...zones.filter((zone) => zone.id !== activeZone.id)]
      : zones.slice();
    for (const zone of orderedZones) {{
      if (!zone.enabled) continue;
      const [x1, y1, x2, y2] = zone.roi;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return zone;
    }}
    return null;
  }}

  function syncZoneControls() {{
    zones.forEach((zone) => {{
      zoneEnabled[zone.id].checked = Boolean(zone.enabled);
      setToggle(zoneFastAlpr[zone.id], Boolean(zone.use_fast_alpr));
      setToggle(zoneTelegram[zone.id], Boolean(zone.send_telegram));
      zoneRecordSeconds[zone.id].value = String(Math.round(zone.record_seconds || 180));
      zoneImageCount[zone.id].value = String(Math.round(zone.image_count || 1));
      zoneCoverageTrigger[zone.id].value = String(Math.round(zone.coverage_trigger_percent || 0));
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
      box.style.cursor = zone.id === activeZoneId ? 'move' : 'pointer';
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
      const zoneSummaries = zones.map((zone) => `${{zone.label}}: ${{zone.enabled ? 'on' : 'off'}}, ${{zone.use_fast_alpr ? 'fast-alpr' : 'images only'}}, ${{zone.send_telegram ? 'telegram' : 'no telegram'}}, ${{Math.round(zone.record_seconds)}}s, ${{Math.round(zone.image_count)}} images, coverage ${{Math.round(zone.coverage_trigger_percent)}}%`);
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
    const zoneSummaries = zones.map((zone) => `${{zone.label}}: ${{zone.enabled ? 'on' : 'off'}}, ${{zone.use_fast_alpr ? 'fast-alpr' : 'images only'}}, ${{zone.send_telegram ? 'telegram' : 'no telegram'}}, ${{Math.round(zone.record_seconds)}}s, ${{Math.round(zone.image_count)}} images, coverage ${{Math.round(zone.coverage_trigger_percent)}}%`);
    setStatus(`Editing ${{activeZone.label}} | ROI: ${{activeZone.roi.map((v) => v.toFixed(3)).join(', ')}} | ${{zoneSummaries.join(' | ')}}`);
  }}

  function updateFromPointer(clientX, clientY) {{
    const point = getNormalizedPointer(clientX, clientY);
    if (!point) return;
    const x = point.x;
    const y = point.y;
    const activeZone = findZone(activeZoneId);
    if (!activeZone) return;
    if (dragCorner) {{
      let [x1, y1, x2, y2] = activeZone.roi;
      if (dragCorner.includes('l')) x1 = clamp(x, 0, x2 - minSize);
      if (dragCorner.includes('r')) x2 = clamp(x, x1 + minSize, 1);
      if (dragCorner.includes('t')) y1 = clamp(y, 0, y2 - minSize);
      if (dragCorner.includes('b')) y2 = clamp(y, y1 + minSize, 1);
      activeZone.roi = [x1, y1, x2, y2];
    }} else if (dragMove) {{
      const nextX1 = clamp(x - dragMove.offsetX, 0, 1 - dragMove.width);
      const nextY1 = clamp(y - dragMove.offsetY, 0, 1 - dragMove.height);
      activeZone.roi = [nextX1, nextY1, nextX1 + dragMove.width, nextY1 + dragMove.height];
    }} else {{
      return;
    }}
    draw();
  }}

  function stopDragging() {{
    dragCorner = null;
    dragMove = null;
  }}

  handles.forEach((handle) => {{
    handle.addEventListener('pointerdown', (event) => {{
      dragCorner = handle.dataset.corner;
      dragMove = null;
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
      event.stopPropagation();
    }});
    handle.addEventListener('pointermove', (event) => {{
      if (!dragCorner) return;
      updateFromPointer(event.clientX, event.clientY);
    }});
    handle.addEventListener('pointerup', () => {{
      stopDragging();
    }});
    handle.addEventListener('pointercancel', () => {{
      stopDragging();
    }});
  }});

  overlay.addEventListener('pointerdown', (event) => {{
    if (event.target.closest('.roi-handle')) return;
    const point = getNormalizedPointer(event.clientX, event.clientY);
    if (!point) return;
    const hitZone = findZoneAt(point.x, point.y);
    if (!hitZone) return;
    activeZoneId = hitZone.id;
    const [x1, y1, x2, y2] = hitZone.roi;
    dragCorner = null;
    dragMove = {{
      offsetX: point.x - x1,
      offsetY: point.y - y1,
      width: x2 - x1,
      height: y2 - y1,
    }};
    overlay.setPointerCapture(event.pointerId);
    draw();
    event.preventDefault();
  }});

  overlay.addEventListener('pointermove', (event) => {{
    if (!dragMove) return;
    updateFromPointer(event.clientX, event.clientY);
  }});

  overlay.addEventListener('pointerup', () => {{
    stopDragging();
  }});

  overlay.addEventListener('pointercancel', () => {{
    stopDragging();
  }});

  resetBtn.addEventListener('click', () => {{
    zones = initialZones.map((zone) => ({{ ...zone, roi: zone.roi.slice() }}));
    activeZoneId = null;
    stopDragging();
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

  Object.entries(zoneFastAlpr).forEach(([zoneId, btn]) => {{
    btn.addEventListener('click', () => {{
      const zone = findZone(zoneId);
      if (!zone) return;
      zone.use_fast_alpr = !zone.use_fast_alpr;
      setToggle(btn, zone.use_fast_alpr);
      draw();
    }});
  }});

  Object.entries(zoneTelegram).forEach(([zoneId, btn]) => {{
    btn.addEventListener('click', () => {{
      const zone = findZone(zoneId);
      if (!zone) return;
      zone.send_telegram = !zone.send_telegram;
      setToggle(btn, zone.send_telegram);
      draw();
    }});
  }});

  Object.entries(zoneRecordSeconds).forEach(([zoneId, select]) => {{
    select.addEventListener('change', () => {{
      const zone = findZone(zoneId);
      if (!zone) return;
      zone.record_seconds = Number(select.value) || zone.record_seconds;
      draw();
    }});
  }});

  Object.entries(zoneImageCount).forEach(([zoneId, select]) => {{
    select.addEventListener('change', () => {{
      const zone = findZone(zoneId);
      if (!zone) return;
      zone.image_count = Number(select.value) || zone.image_count;
      draw();
    }});
  }});

  Object.entries(zoneCoverageTrigger).forEach(([zoneId, select]) => {{
    select.addEventListener('change', () => {{
      const zone = findZone(zoneId);
      if (!zone) return;
      zone.coverage_trigger_percent = Number(select.value) || 0;
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
      stopDragging();
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
  attachLivePreview(img);
  window.setInterval(refreshMotionStatus, 1000);
  refreshMotionStatus();
}})();
</script>"""

    def _render_shared_styles(self) -> str:
        return """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
body { min-height: 100vh; background: #07110f; color: #edf7f2; padding: 0; overflow-x: hidden; }
a { color: #7dd3fc; }
img, video, canvas, svg { max-width: 100%; }
button, input, select, textarea { font: inherit; max-width: 100%; }
button, a, input, select, textarea { touch-action: manipulation; }
.page-shell { width: min(1380px, calc(100% - 32px)); margin: 0 auto; padding: 22px 0 34px; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 24px; padding: 12px 14px; background: rgba(9, 18, 17, 0.92); border: 1px solid #24413c; border-radius: 8px; box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28); }
.brand { color: #f8fafc; font-weight: 800; text-decoration: none; letter-spacing: 0; white-space: nowrap; }
.nav-links { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.nav-links a { color: #cde8de; text-decoration: none; border: 1px solid #2f514b; border-radius: 8px; padding: 8px 11px; background: #10211f; line-height: 1; }
.nav-links a:hover { border-color: #5eead4; color: #f8fafc; background: #17302d; }
.nav-links a.active { border-color: #facc15; color: #111827; background: #facc15; font-weight: 700; }
.topbar-status { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.system-clock, .extraction-indicator { color: #d9f99d; border: 1px solid #2f514b; border-radius: 8px; background: #07110f; padding: 8px 11px; white-space: nowrap; font-variant-numeric: tabular-nums; line-height: 1; }
.extraction-indicator { display: none; align-items: center; gap: 7px; }
.extraction-indicator.is-visible { display: inline-flex; }
.extraction-indicator svg { width: 14px; height: 14px; fill: currentColor; flex-shrink: 0; }
.extraction-stop { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 8px; border: 1px solid #2f514b; background: #10211f; color: #d9f99d; cursor: pointer; padding: 0; flex-shrink: 0; }
.extraction-stop:hover { border-color: #fca5a5; color: #fee2e2; background: #450a0a; }
.extraction-stop svg { width: 11px; height: 11px; }
.menu-actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.menu-actions form { margin: 0; }
.menu-actions button { color: #fee2e2; text-decoration: none; border: 1px solid #7f1d1d; border-radius: 8px; padding: 8px 11px; background: #450a0a; line-height: 1; cursor: pointer; font: inherit; }
.menu-actions button:hover { border-color: #fca5a5; color: #fff; background: #7f1d1d; }
.menu-actions button.confirm-armed { border-color: #facc15; color: #111827; background: #facc15; }
.menu-icon-button { display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 999px; border: 1px solid #334155; background: transparent; color: #cbd5e1; cursor: pointer; padding: 0; }
.menu-icon-button:hover { border-color: #facc15; color: #facc15; background: #172033; }
.menu-icon-button.danger { border-color: #7f1d1d; color: #fca5a5; background: #450a0a; }
.menu-icon-button.danger:hover { border-color: #fca5a5; color: #fff; background: #7f1d1d; }
.menu-icon-button.confirm-armed { border-color: #facc15; color: #111827; background: #facc15; }
.menu-actions button.view-button { color: #cde8de; border-color: #2f514b; background: #10211f; }
.menu-actions button.view-button:hover { border-color: #5eead4; color: #f8fafc; background: #17302d; }
.menu-actions button.view-button.active { border-color: #facc15; color: #111827; background: #facc15; font-weight: 700; }
.panel, .card { border: 1px solid #263d39; box-shadow: 0 10px 28px rgba(0, 0, 0, 0.2); }
h1 { margin-top: 0; }
@media (max-width: 760px) {
  h1 { font-size: clamp(1.55rem, 8vw, 2rem); line-height: 1.12; overflow-wrap: anywhere; }
  h2 { font-size: clamp(1.15rem, 6vw, 1.45rem); line-height: 1.18; }
  p, li { line-height: 1.45; }
  .page-shell { width: min(100% - 20px, 1380px); padding-top: 10px; padding-bottom: 24px; }
  .topbar { align-items: stretch; flex-direction: column; gap: 12px; margin-bottom: 18px; padding: 10px; }
  .brand { white-space: normal; line-height: 1.2; }
  .nav-links { justify-content: flex-start; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .nav-links a { display: flex; align-items: center; justify-content: center; min-height: 42px; padding: 9px 10px; text-align: center; }
  .topbar-status { justify-content: flex-start; align-items: stretch; }
  .system-clock, .extraction-indicator { padding: 9px 10px; }
  .menu-actions { justify-content: flex-start; align-items: stretch; }
  .menu-actions button { min-height: 42px; padding: 9px 10px; }
  .panel, .card { padding: 14px; border-radius: 8px; }
  .grid, .gallery, .stats-grid { grid-template-columns: 1fr !important; gap: 12px; }
  .pagination { gap: 8px; }
  .pagination a, .pagination span { flex: 1 1 auto; text-align: center; }
  .detail-actions, .page-actions, .config-actions, .time-tools, .video-tools, .card-actions, .alpr-action { align-items: stretch; }
  .detail-actions a:not(.icon-button), .page-action, .telegram-action, .danger-action, .config-actions button, .time-tools button, .video-tools button { width: 100%; justify-content: center; }
  .env-editor { min-height: 52vh; font-size: 13px; }
  pre { max-width: 100%; overflow-x: auto; }
}
@media (max-width: 420px) {
  .page-shell { width: min(100% - 14px, 1380px); }
  .nav-links { grid-template-columns: 1fr; }
}
"""

    def _render_nav(self, active: str) -> str:
        items = [
            ("dashboard", "/", "Dashboard"),
            ("stats", "/stats", "Config"),
            ("images", "/images", "Images"),
            ("videos", "/videos", "Videos"),
            ("plates", "/plates", "Plates"),
            ("test", "/test", "Test"),
        ]
        links = "".join(
            f'<a class="{"active" if item_id == active else ""}" href="{href}">{label}</a>'
            for item_id, href, label in items
        )
        action_by_page = {
            "images": (
                '<form method="post" action="/api/images/clear" data-double-confirm="Remove Images" style="margin:0;">'
                '<button class="menu-icon-button danger" type="submit" aria-label="Remove all images" title="Remove all images">'
                '<svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M9 3h6l1 2h5v2H3V5h5l1-2Zm1 6h2v8h-2V9Zm4 0h2v8h-2V9ZM6 9h2v8H6V9Zm1 12a2 2 0 0 1-2-2V8h14v11a2 2 0 0 1-2 2H7Z"/></svg>'
                '</button></form>'
            ),
            "videos": (
                '<button class="menu-icon-button" id="video-upload-btn" type="button" aria-label="Upload video" title="Upload video">'
                '<svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M11 16V7.85l-2.6 2.6L7 9l5-5 5 5-1.4 1.45-2.6-2.6V16h-2Zm-5 4q-.825 0-1.413-.588T4 18v-3h2v3h12v-3h2v3q0 .825-.588 1.413T18 20H6Z"/></svg>'
                '</button>'
                '<input type="file" id="video-upload-input" accept="video/mp4,video/*,.mp4,.mov,.avi" style="display:none;">'
                '<form id="video-clear-form" method="post" action="/api/videos/clear" style="margin:0;" data-double-confirm="Remove Videos">'
                '<button class="menu-icon-button danger" type="submit" aria-label="Remove all videos" title="Remove all videos">'
                '<svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:currentColor;"><path d="M9 3h6l1 2h5v2H3V5h5l1-2Zm1 6h2v8h-2V9Zm4 0h2v8h-2V9ZM6 9h2v8H6V9Zm1 12a2 2 0 0 1-2-2V8h14v11a2 2 0 0 1-2 2H7Z"/></svg>'
                '</button></form>'
            ),
            "plates": '<form method="post" action="/api/plates/clear" data-double-confirm="Remove Plates"><button type="submit">Remove Plates</button></form>',
        }
        action = action_by_page.get(active, "")
        actions = f'<div class="menu-actions" aria-label="Page actions">{action}</div>' if action else ""
        return f"""<header class="topbar">
  <a class="brand" href="/">ALPR Watcher</a>
  <nav class="nav-links" aria-label="Main navigation">{links}</nav>
  <div class="topbar-status">
    <span class="system-clock" id="system-clock" title="System clock">--:--:--</span>
    <span class="extraction-indicator" id="extraction-indicator" title="Image extraction progress" aria-live="polite">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2ZM5 5h14v9.6l-3.2-3.2a1 1 0 0 0-1.4 0L11 14.8l-1.4-1.4a1 1 0 0 0-1.4 0L5 16.6V5Zm14 14H5v-.6l3.9-3.9 1.4 1.4a1 1 0 0 0 1.4 0l3.4-3.4 3.9 3.9V19ZM8.5 10A1.5 1.5 0 1 0 8.5 7a1.5 1.5 0 0 0 0 3Z"/></svg>
      <span id="extraction-counter">0/0</span>
      <button class="extraction-stop" id="extraction-stop" type="button" aria-label="Stop image extraction" title="Stop image extraction"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6h12v12H6z"/></svg></button>
    </span>
  </div>
  {actions}
</header>
<script>
(() => {{
  const clock = document.getElementById('system-clock');
  if (!clock) return;
  let serverEpochMs = Date.now();
  let syncedAtMs = performance.now();

  function renderClock() {{
    const now = new Date(serverEpochMs + (performance.now() - syncedAtMs));
    clock.textContent = now.toLocaleString();
  }}

  async function syncClockDisplay() {{
    try {{
      const response = await fetch('/api/time', {{ cache: 'no-store' }});
      if (!response.ok) throw new Error('time unavailable');
      const payload = await response.json();
      serverEpochMs = Number(payload.epoch_ms) || Date.now();
      syncedAtMs = performance.now();
      renderClock();
    }} catch (error) {{
      clock.textContent = new Date().toLocaleString();
    }}
  }}

  syncClockDisplay();
  setInterval(renderClock, 1000);
  setInterval(syncClockDisplay, 30000);
}})();

(() => {{
  const indicator = document.getElementById('extraction-indicator');
  const counter = document.getElementById('extraction-counter');
  const stopBtn = document.getElementById('extraction-stop');
  if (!indicator || !counter || !stopBtn) return;

  let activeJobId = null;
  let pollTimer = null;

  function showIndicator(jobId) {{
    activeJobId = jobId;
    indicator.classList.add('is-visible');
    counter.textContent = '0/?';
  }}

  function hideIndicator() {{
    activeJobId = null;
    indicator.classList.remove('is-visible');
    if (pollTimer) {{ clearInterval(pollTimer); pollTimer = null; }}
  }}

  window.__startExtractionIndicator = function(jobId) {{
    showIndicator(jobId);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {{
      try {{
        const r = await fetch('/api/extraction-status', {{ cache: 'no-store' }});
        const status = await r.json();
        const job = status[jobId];
        if (!job) return;
        if (job.done || job.stage === 'cancelled') {{
          counter.textContent = job.error ? 'failed' : (job.stage === 'partial' ? `partial (${{job.saved}}/${{job.total}})` : (job.stage === 'cancelled' ? `stopped (${{job.saved}})` : `done (${{job.saved}})`));
          setTimeout(hideIndicator, 3000);
          clearInterval(pollTimer); pollTimer = null;
          if (window.__onExtractionDone) window.__onExtractionDone(jobId, job);
        }} else {{
          const total = job.total || '?';
          counter.textContent = `${{job.saved}}/${{total}}`;
        }}
      }} catch (e) {{}}
    }}, 500);
  }};

  stopBtn.addEventListener('click', async () => {{
    if (!activeJobId) return;
    stopBtn.disabled = true;
    try {{
      await fetch('/api/extraction-cancel', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ job_id: activeJobId }}),
      }});
      counter.textContent = 'stopping...';
    }} catch (e) {{}}
    finally {{
      window.setTimeout(() => {{ stopBtn.disabled = false; }}, 1000);
    }}
  }});

  (async () => {{
    try {{
      const r = await fetch('/api/extraction-status', {{ cache: 'no-store' }});
      const status = await r.json();
      const active = Object.entries(status).reverse().find(([, job]) => job && !job.done && job.stage !== 'cancelled');
      if (active) window.__startExtractionIndicator(active[0]);
    }} catch (e) {{}}
  }})();
}})();

(() => {{
  const uploadBtn = document.getElementById('video-upload-btn');
  const uploadInput = document.getElementById('video-upload-input');

  function armDoubleConfirm(form) {{
    const button = form && form.querySelector('button[type="submit"], button:not([type])');
    if (!button) return;
    button.classList.add('confirm-armed');
    form.dataset.confirmArmed = 'true';
  }}

  function resetDoubleConfirm(form) {{
    const button = form && form.querySelector('button[type="submit"], button:not([type])');
    if (!button) return;
    button.classList.remove('confirm-armed');
    delete form.dataset.confirmArmed;
  }}

  function resetDeleteConfirmButtons(exceptForm) {{
    Array.from(document.querySelectorAll('form[data-double-confirm]')).forEach((form) => {{
      if (form !== exceptForm) resetDoubleConfirm(form);
    }});
  }}

  window.__resetDeleteConfirmButtons = resetDeleteConfirmButtons;

  document.addEventListener('click', (event) => {{
    const control = event.target.closest('button, a, input[type="button"], input[type="submit"]');
    if (!control) return;
    const armedForm = control.closest('form[data-double-confirm]');
    if (armedForm && armedForm.dataset.confirmArmed === 'true') return;
    resetDeleteConfirmButtons(null);
  }});

  const doubleConfirmForms = Array.from(document.querySelectorAll('form[data-double-confirm]'));
  doubleConfirmForms.forEach((form) => {{
    const button = form.querySelector('button[type="submit"], button:not([type])');
    if (!button) return;
    form.addEventListener('submit', (event) => {{
      if (form.dataset.confirmArmed === 'true') return;
      event.preventDefault();
      resetDeleteConfirmButtons(form);
      armDoubleConfirm(form);
    }});
  }});

  if (uploadBtn && uploadInput) {{
    uploadBtn.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', () => {{
      const file = uploadInput.files && uploadInput.files[0];
      if (!file) return;
      uploadInput.value = '';
      const overlay = document.getElementById('action-overlay');
      const title = document.getElementById('action-title');
      const message = document.getElementById('action-message');
      const spinner = document.getElementById('action-spinner');
      const close = document.getElementById('action-close');
      const progress = document.getElementById('action-progress');
      const progressBar = document.getElementById('action-progress-bar');
      const progressText = document.getElementById('action-progress-text');
      function showOverlay(heading, detail, finished) {{
        if (!overlay || !title || !message || !spinner || !close) return;
        title.textContent = heading;
        message.textContent = detail;
        overlay.classList.add('visible');
        spinner.style.display = finished ? 'none' : 'block';
        close.style.display = finished ? 'inline-flex' : 'none';
      }}
      function showUploadProgress(loaded, total) {{
        if (!progress || !progressBar || !progressText) return;
        progress.style.display = 'block';
        progressText.style.display = 'block';
        if (total > 0) {{
          const percent = Math.max(0, Math.min(100, Math.round((loaded / total) * 100)));
          progressBar.style.width = `${{percent}}%`;
          progressText.textContent = `${{percent}}% • ${{
            (loaded / (1024 * 1024)).toFixed(1)
          }} / ${{(total / (1024 * 1024)).toFixed(1)}} MB`;
        }} else {{
          progressBar.style.width = '100%';
          progressText.textContent = `${{(loaded / (1024 * 1024)).toFixed(1)}} MB uploaded`;
        }}
      }}
      function showProcessingState(detail) {{
        if (!progress || !progressBar || !progressText) return;
        progress.style.display = 'block';
        progressText.style.display = 'block';
        progressBar.style.width = '100%';
        progressText.textContent = 'Upload finished. Waiting for server response...';
        if (message) message.textContent = detail;
      }}
      function hideUploadProgress() {{
        if (!progress || !progressBar || !progressText) return;
        progress.style.display = 'none';
        progressText.style.display = 'none';
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
      }}
      showOverlay('Uploading...', file.name, false);
      showUploadProgress(0, file.size || 0);
      const form = new FormData();
      form.append('video', file, file.name);
      const request = new XMLHttpRequest();
      request.open('POST', '/api/videos/upload');
      request.responseType = 'json';
      request.upload.addEventListener('progress', (event) => {{
        if (event.lengthComputable) showUploadProgress(event.loaded, event.total);
      }});
      request.upload.addEventListener('load', () => {{
        showProcessingState('Upload completed. Saving video on the server...');
      }});
      request.addEventListener('load', () => {{
        const payload = request.response && typeof request.response === 'object'
          ? request.response
          : (() => {{
              try {{
                return JSON.parse(request.responseText || '{{}}');
              }} catch (error) {{
                return {{}};
              }}
            }})();
        if (request.status >= 200 && request.status < 300) {{
          showUploadProgress(file.size || 0, file.size || 0);
          showOverlay('Uploaded', payload.message || 'Video uploaded successfully.', true);
          if (close) close.addEventListener('click', () => window.location.reload(), {{ once: true }});
          return;
        }}
        hideUploadProgress();
        showOverlay('Upload failed', payload.error || 'Upload did not complete.', true);
      }});
      request.addEventListener('error', () => {{
        hideUploadProgress();
        showOverlay('Upload failed', 'Upload did not complete.', true);
      }});
      request.addEventListener('abort', () => {{
        hideUploadProgress();
        showOverlay('Upload canceled', file.name, true);
      }});
      request.send(form);
    }});
  }}

}})();
</script>"""

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
        recent = self._render_recent_plate_cards(limit=8, include_images=False)
        gallery = self._render_image_cards(self._list_saved_images()[:8])
        event_log = self._render_event_log_items()
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ALPR Watcher</title>
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
.gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(220px, 100%), 1fr)); gap: 14px; }}
.gallery .card {{ background: #0f172a; border-radius: 12px; padding: 12px; }}
.gallery img {{ width: 100%; border-radius: 10px; display: block; }}
.card {{ background: #0f172a; border-radius: 12px; padding: 14px; }}
.plate-code {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.08em; }}
.meta {{ color: #cbd5e1; font-size: 0.95rem; }}
.event-log {{ display: grid; gap: 8px; max-height: 420px; overflow: auto; padding-right: 4px; }}
.event-log-item {{ display: grid; grid-template-columns: 132px 88px minmax(0, 1fr); gap: 8px; align-items: start; padding: 9px 10px; background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(9, 14, 26, 0.96)); border-radius: 12px; border: 1px solid #223244; box-shadow: inset 0 1px 0 rgba(255,255,255,0.02); min-width: 0; }}
.event-log-time {{ color: #cbd5e1; font-variant-numeric: tabular-nums; font-size: 0.72rem; line-height: 1.25; overflow-wrap: anywhere; word-break: break-word; }}
.event-log-kind {{ display: inline-flex; align-items: center; justify-content: center; min-height: 22px; padding: 1px 8px; border-radius: 999px; font-size: 0.64rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: #111827; background: #94a3b8; white-space: normal; text-align: center; line-height: 1.15; border: 1px solid transparent; overflow-wrap: anywhere; word-break: break-word; }}
.event-log-kind.zone-yellow {{ background: #fef08a; color: #713f12; border-color: #fde047; box-shadow: 0 0 0 1px rgba(250, 204, 21, 0.12); }}
.event-log-kind.zone-purple {{ background: #e9d5ff; color: #4a1d96; border-color: #d8b4fe; box-shadow: 0 0 0 1px rgba(192, 132, 252, 0.14); }}
.event-log-kind.zone-none {{ background: #94a3b8; color: #0f172a; border-color: #64748b; }}
.event-log-message {{ color: #e2e8f0; font-size: 0.78rem; line-height: 1.28; white-space: normal; overflow-wrap: anywhere; word-break: break-word; min-width: 0; }}
.event-log-empty {{ color: #cbd5e1; }}
@media (max-width: 960px) {{
  .layout {{ grid-template-columns: 1fr; }}
  .event-log-item {{ grid-template-columns: 110px 80px minmax(0, 1fr); }}
}}
@media (max-width: 640px) {{
  .hero {{ align-items: flex-start; }}
  .event-log-item {{ grid-template-columns: 1fr; }}
}}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("dashboard")}
<div class="layout">
  {self._render_roi_editor_markup()}
  <div class="panel">
    <h2 class="section-title">Recent plate numbers</h2>
    <div class="plates">{recent}</div>
    <h2 class="section-title" style="margin-top:20px;">Event log</h2>
    <div id="event-log" class="event-log">{event_log}</div>
  </div>
</div>
<div class="panel" style="margin-top:20px;">
  <h2 class="section-title">Latest captured images</h2>
  <div class="gallery">{gallery}</div>
</div>
</div>
{self._render_roi_editor_script()}
<script>
(() => {{
  const log = document.getElementById('event-log');
  if (!log) return;

  function escapeHtml(value) {{
    return String(value).replace(/[&<>"']/g, (char) => ({{
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }}[char]));
  }}

  function render(events) {{
    if (!events.length) {{
      log.innerHTML = '<div class="event-log-empty">No events logged yet.</div>';
      return;
    }}
    log.innerHTML = events.map((entry) => {{
      const zoneClass = entry.zone_id ? `zone-${{entry.zone_id}}` : 'zone-none';
      const timeText = escapeHtml(entry.time_text || '');
      const categoryText = escapeHtml(entry.category || '');
      const messageText = escapeHtml(entry.message || '');
      return `<div class="event-log-item" title="${{messageText}}"><span class="event-log-time" title="${{timeText}}">${{timeText}}</span><span class="event-log-kind ${{zoneClass}}" title="${{entry.zone_id ? `Zone: ${{entry.zone_id}}` : 'No zone'}}">${{categoryText}}</span><span class="event-log-message" title="${{messageText}}">${{messageText}}</span></div>`;
    }}).join('');
  }}

  async function refreshEventLog() {{
    try {{
      const response = await fetch('/api/event-log', {{ cache: 'no-store' }});
      if (!response.ok) return;
      const payload = await response.json();
      render(Array.isArray(payload.events) ? payload.events : []);
    }} catch (error) {{
      // Keep the last rendered log visible.
    }}
  }}

  setInterval(refreshEventLog, 2000);
  refreshEventLog();
}})();
</script>
</body></html>"""

    def _render_live_page(self) -> str:
        live_preview_url = json.dumps("/stream-clean.mjpg")
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Live Video</title>
<style>
{self._render_shared_styles()}
html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }}
body {{ padding: 0; }}
.page-shell {{ width: 100%; height: 100dvh; margin: 0; padding: 12px; display: flex; flex-direction: column; gap: 12px; }}
.topbar {{ flex-shrink: 0; z-index: 5; margin: 0; }}
.live-frame {{ position: relative; flex: 1; min-height: 0; background: #000; overflow: hidden; }}
.live-frame img {{ position: absolute; left: 50%; top: 50%; width: 100%; height: 100%; object-fit: contain; display: block; background: #000; transform: translate(-50%, -50%); transform-origin: center; }}
.live-frame.vertical img {{ width: 100dvh; height: 100vw; transform: translate(-50%, -50%) rotate(90deg); }}
.live-error {{ position: fixed; left: 50%; bottom: 18px; transform: translateX(-50%); z-index: 6; padding: 10px 14px; border-radius: 999px; background: rgba(15, 23, 42, 0.88); color: #f8fafc; border: 1px solid rgba(148, 163, 184, 0.4); }}
@media (max-width: 760px) {{
  .page-shell {{ padding: 8px; gap: 8px; }}
  .topbar {{ max-height: 44dvh; overflow: auto; }}
}}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("live")}
<main class="live-frame">
  <img id="live-clean-stream" alt="Live camera stream">
</main>
<div id="live-error" class="live-error" style="display:none;">Waiting for live stream...</div>
</div>
<script>
(() => {{
  const livePreviewUrl = {live_preview_url};
  const frame = document.querySelector('.live-frame');
  const horizontal = document.getElementById('live-horizontal');
  const vertical = document.getElementById('live-vertical');
  const image = document.getElementById('live-clean-stream');
  const error = document.getElementById('live-error');
  if (!frame || !horizontal || !vertical || !image || !error) return;

  function showError(text) {{
    error.textContent = text;
    error.style.display = 'block';
  }}

  function hideError() {{
    error.style.display = 'none';
  }}

  let livePreviewRetryTimer = null;

  function scheduleLivePreviewRetry(message) {{
    if (message) showError(message);
    if (livePreviewRetryTimer) return;
    livePreviewRetryTimer = window.setTimeout(() => {{
      livePreviewRetryTimer = null;
      attachLivePreview();
    }}, 3000);
  }}

  function attachLivePreview() {{
    image.onload = hideError;
    image.onerror = () => scheduleLivePreviewRetry('Waiting for live stream...');
    image.src = `${{livePreviewUrl}}?v=${{Date.now()}}`;
  }}

  function setOrientation(orientation) {{
    const isVertical = orientation === 'vertical';
    frame.classList.toggle('vertical', isVertical);
    horizontal.classList.toggle('active', !isVertical);
    vertical.classList.toggle('active', isVertical);
    horizontal.setAttribute('aria-pressed', String(!isVertical));
    vertical.setAttribute('aria-pressed', String(isVertical));
    localStorage.setItem('live-view-orientation', isVertical ? 'vertical' : 'horizontal');
  }}

  horizontal.addEventListener('click', () => setOrientation('horizontal'));
  vertical.addEventListener('click', () => setOrientation('vertical'));
  setOrientation(localStorage.getItem('live-view-orientation') === 'vertical' ? 'vertical' : 'horizontal');
  attachLivePreview();
}})();
</script>
</body></html>"""

    def _render_stats_page(self) -> str:
        env_path = self._env_file_path()
        env_text = self._read_env_text()
        escaped_env_text = html.escape(env_text)
        escaped_env_path = html.escape(str(env_path))
        stats_json = html.escape(json.dumps(self._stats_snapshot(), indent=2))
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Configuration</title>
<style>
body {{ font-family: Arial, sans-serif; background: #07110f; color: #edf7f2; margin: 0; padding: 24px; }}
a {{ color: #7dd3fc; }}
.panel {{ background: #0d1a18; border: 1px solid #263d39; border-radius: 8px; padding: 16px; }}
.config-layout {{ display: grid; gap: 18px; align-items: start; }}
.config-side {{ display: grid; gap: 18px; }}
.config-tools {{ display: grid; grid-template-columns: minmax(0, 0.65fr) minmax(0, 1.35fr); gap: 18px; align-items: start; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
.stat {{ background: #07110f; border: 1px solid #263d39; border-radius: 8px; padding: 10px; }}
.stat-label {{ color: #a7c8bd; font-size: 0.82rem; }}
.stat-value {{ color: #f8fafc; font-size: 1.05rem; font-weight: 700; margin-top: 4px; overflow-wrap: anywhere; }}
.config-actions {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 12px; }}
.config-actions button {{ background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 700; }}
.config-actions button.secondary {{ background: #38bdf8; }}
.config-status {{ color: #cde8de; }}
.config-status.error {{ color: #fca5a5; }}
.time-tools {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 12px; }}
.time-tools button {{ background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 6px 9px; cursor: pointer; font-weight: 700; font-size: 0.86rem; }}
.time-status {{ color: #cde8de; }}
.time-status.error {{ color: #fca5a5; }}
.env-editor {{ width: 100%; min-height: 68vh; resize: vertical; background: #050c0b; color: #d9f99d; border: 1px solid #2f514b; border-radius: 8px; padding: 12px; font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; tab-size: 2; box-sizing: border-box; }}
.meta {{ color: #cde8de; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #050c0b; padding: 12px; border-radius: 8px; overflow-x: auto; }}
@media (max-width: 960px) {{ .config-tools {{ grid-template-columns: 1fr; }} }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("stats")}
<h1>System Configuration</h1>
<div class="config-layout">
<div class="panel">
  <h2>Environment</h2>
  <textarea id="env-content" class="env-editor" spellcheck="false">{escaped_env_text}</textarea>
  <div class="config-actions">
    <button id="env-save-apply" type="button">Save and Apply</button>
    <button id="env-save" class="secondary" type="button">Save Only</button>
    <span id="env-status" class="config-status">Ready.</span>
  </div>
</div>
<div class="config-tools">
<div class="panel">
  <h2>System Clock</h2>
  <p class="meta">Server time: <strong id="settings-system-clock">Loading...</strong></p>
  <p class="meta">Computer time: <strong id="settings-browser-clock">Loading...</strong></p>
  <div class="time-tools">
    <button id="time-sync" type="button">Sync with This Computer</button>
    <span id="time-status" class="time-status">Ready.</span>
  </div>
</div>
<div class="panel">
  <h2>Runtime Stats</h2>
  <div id="stats-grid" class="stats-grid"></div>
  <h3>Raw Stats</h3>
  <pre id="stats-raw">{stats_json}</pre>
</div>
</div>
</div>
<script>
(() => {{
  const editor = document.getElementById('env-content');
  const save = document.getElementById('env-save');
  const saveApply = document.getElementById('env-save-apply');
  const status = document.getElementById('env-status');
  const statsGrid = document.getElementById('stats-grid');
  const statsRaw = document.getElementById('stats-raw');
  const settingsSystemClock = document.getElementById('settings-system-clock');
  const settingsBrowserClock = document.getElementById('settings-browser-clock');
  const timeSync = document.getElementById('time-sync');
  const timeStatus = document.getElementById('time-status');
  if (!editor || !save || !saveApply || !status || !statsGrid || !statsRaw || !settingsSystemClock || !settingsBrowserClock || !timeSync || !timeStatus) return;

  function setStatus(text, isError = false) {{
    status.textContent = text;
    status.className = isError ? 'config-status error' : 'config-status';
  }}

  function setTimeStatus(text, isError = false) {{
    timeStatus.textContent = text;
    timeStatus.className = isError ? 'time-status error' : 'time-status';
  }}

  function formatValue(value) {{
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
    if (value === null || value === undefined) return 'n/a';
    return String(value);
  }}

  function renderStats(stats) {{
    const keys = [
      'motion_status',
      'uptime_seconds',
      'camera_fps_estimate',
      'capture_fps',
      'processing_fps',
      'stream_fps',
      'capture_frame_total',
      'processing_frame_total',
      'stream_frame_total',
      'capture_gap_total',
      'read_failure_total',
      'capture_reconnect_total',
      'last_capture_age_seconds',
      'frame_width',
      'capture_buffer_size',
      'process_every_n_frames',
      'recording_active',
      'recording_fps',
    ];
    statsGrid.innerHTML = keys.map((key) => `
      <div class="stat"><div class="stat-label">${{key.replaceAll('_', ' ')}}</div><div class="stat-value">${{formatValue(stats[key])}}</div></div>
    `).join('');
    statsRaw.textContent = JSON.stringify(stats, null, 2);
  }}

  async function refreshStats() {{
    try {{
      const response = await fetch('/api/stats', {{ cache: 'no-store' }});
      if (!response.ok) return;
      renderStats(await response.json());
    }} catch (error) {{
      // Keep the last stats visible if a refresh fails.
    }}
  }}

  async function refreshSystemTime() {{
    settingsBrowserClock.textContent = new Date().toLocaleString();
    try {{
      const response = await fetch('/api/time', {{ cache: 'no-store' }});
      if (!response.ok) return;
      const payload = await response.json();
      settingsSystemClock.textContent = new Date(Number(payload.epoch_ms)).toLocaleString();
    }} catch (error) {{
      settingsSystemClock.textContent = 'Unavailable';
    }}
  }}

  timeSync.addEventListener('click', async () => {{
    setTimeStatus('Syncing...');
    timeSync.disabled = true;
    try {{
      const response = await fetch('/api/time/sync', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          epoch_ms: Date.now(),
          timezone_offset_minutes: new Date().getTimezoneOffset(),
          browser_time: new Date().toISOString(),
        }}),
      }});
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || payload.message || 'Clock sync failed');
      setTimeStatus(payload.message || 'Clock synced.');
      refreshSystemTime();
    }} catch (error) {{
      setTimeStatus(error.message || 'Clock sync failed', true);
    }} finally {{
      timeSync.disabled = false;
    }}
  }});

  async function saveConfig(apply) {{
    setStatus(apply ? 'Saving and applying...' : 'Saving...');
    save.disabled = true;
    saveApply.disabled = true;
    try {{
      const response = await fetch('/api/env', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ content: editor.value, apply }}),
      }});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Save failed');
      setStatus(payload.message || 'Saved.');
      refreshStats();
    }} catch (error) {{
      setStatus(error.message || 'Save failed', true);
    }} finally {{
      save.disabled = false;
      saveApply.disabled = false;
    }}
  }}

  save.addEventListener('click', () => saveConfig(false));
  saveApply.addEventListener('click', () => saveConfig(true));
  refreshStats();
  refreshSystemTime();
  setInterval(refreshStats, 5000);
  setInterval(refreshSystemTime, 1000);
}})();
</script>
</body></html>"""

    def _page_from_query(self, query: dict[str, List[str]]) -> int:
        try:
            return max(1, int((query.get("page") or ["1"])[0]))
        except (TypeError, ValueError):
            return 1

    def _pagination_html(
        self,
        base_path: str,
        page: int,
        total_items: int,
        page_size: int,
        extra_params: Optional[dict[str, str]] = None,
    ) -> str:
        if total_items <= page_size:
            return ""
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        extra_params = {key: value for key, value in (extra_params or {}).items() if value is not None}

        def page_href(target_page: int) -> str:
            params = {"page": str(target_page)}
            params.update(extra_params)
            return f"{base_path}?{urlencode(params)}"

        previous_html = (
            f'<a href="{page_href(page - 1)}">Previous</a>'
            if page > 1
            else '<span>Previous</span>'
        )
        next_html = (
            f'<a href="{page_href(page + 1)}">Next</a>'
            if page < total_pages
            else '<span>Next</span>'
        )
        return (
            '<nav class="pagination" aria-label="Gallery pages">'
            f'{previous_html}<span>Page {page} of {total_pages} pages</span>{next_html}'
            '</nav>'
        )

    def _gallery_summary_html(self, page: int, total_items: int, page_size: int, label: str) -> str:
        if total_items <= 0:
            return f"<p>No {html.escape(label)} found.</p>"
        start_index = ((page - 1) * page_size) + 1
        end_index = min(total_items, page * page_size)
        return f"<p>Showing {start_index}-{end_index} of {total_items} saved {html.escape(label)}.</p>"

    def _render_images_page(self, page: int = 1) -> str:
        images = self._list_saved_images()
        total_images = len(images)
        total_pages = max(1, (total_images + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
        page = min(max(1, page), total_pages)
        start = (page - 1) * GALLERY_PAGE_SIZE
        gallery = self._render_image_cards(images[start : start + GALLERY_PAGE_SIZE])
        pagination = self._pagination_html("/images", page, total_images, GALLERY_PAGE_SIZE)
        summary = self._gallery_summary_html(page, total_images, GALLERY_PAGE_SIZE, "images")
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Captured Images</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); gap: 18px; }}
.card {{ background: #111827; padding: 12px; border-radius: 14px; }}
.card-actions {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 8px; }}
.card-filename {{ margin: 0; flex: 1; min-width: 0; }}
.card-icon-form {{ margin: 0; flex-shrink: 0; }}
.card-icon-button {{ display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 999px; border: 1px solid #334155; background: transparent; color: #cbd5e1; cursor: pointer; padding: 0; }}
	.card-icon-button svg {{ width: 18px; height: 18px; fill: currentColor; display: block; }}
	.card-icon-button:hover {{ border-color: #facc15; color: #facc15; background: #172033; }}
	.alpr-popup {{ position: fixed; inset: 0; background: rgba(2, 6, 23, 0.72); display: none; align-items: center; justify-content: center; z-index: 1000; padding: 24px; }}
	.alpr-popup.visible {{ display: flex; }}
	.alpr-popup-card {{ width: min(92vw, 420px); background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 22px; text-align: center; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35); }}
	.alpr-popup-card h2 {{ margin: 0; font-size: 1.1rem; }}
	.alpr-popup-card p {{ margin: 10px 0 0; color: #cbd5e1; }}
	.alpr-popup-close {{ margin-top: 14px; background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 9px 14px; cursor: pointer; font-weight: 700; }}
	.pagination {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 18px 0; }}
.pagination a, .pagination span {{ border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; text-decoration: none; }}
.pagination span {{ color: #94a3b8; }}
img {{ width: 100%; border-radius: 10px; display: block; }}
video {{ width: 100%; border-radius: 10px; display: block; margin-top: 10px; background: #000; }}
p {{ margin: 8px 0 0; word-break: break-word; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("images")}
<h1>Captured images</h1>
<p>Keeping only the newest {self.config.max_saved_images} images.</p>
{summary}
	{pagination}
	<div class="grid">{gallery}</div>
	{pagination}
	</div>
	<div class="alpr-popup" id="alpr-popup" role="dialog" aria-modal="true" aria-labelledby="alpr-popup-title">
	  <div class="alpr-popup-card">
	    <h2 id="alpr-popup-title">ALPR</h2>
	    <p id="alpr-popup-message">Sending...</p>
	    <button class="alpr-popup-close" id="alpr-popup-close" type="button">Close</button>
	  </div>
	</div>
	<script>
	(() => {{
	  const popup = document.getElementById('alpr-popup');
	  const title = document.getElementById('alpr-popup-title');
	  const message = document.getElementById('alpr-popup-message');
	  const close = document.getElementById('alpr-popup-close');
	  const forms = Array.from(document.querySelectorAll('form.card-icon-form[action="/api/images/alpr"]'));
	  if (!popup || !title || !message || !close || !forms.length) return;

	  function showPopup(heading, text) {{
	    title.textContent = heading;
	    message.textContent = text;
	    popup.classList.add('visible');
	  }}

	  close.addEventListener('click', () => popup.classList.remove('visible'));
	  popup.addEventListener('click', (event) => {{
	    if (event.target === popup) popup.classList.remove('visible');
	  }});

	  forms.forEach((form) => {{
	    form.addEventListener('submit', async (event) => {{
	      event.preventDefault();
	      const button = form.querySelector('button');
	      if (button) button.disabled = true;
	      showPopup('ALPR', 'Sending image...');
	      try {{
	        const formData = new FormData(form);
	        const payload = {{}};
	        formData.forEach((value, key) => {{ payload[key] = value; }});
	        const response = await fetch(form.action, {{
	          method: 'POST',
	          headers: {{ 'Content-Type': 'application/json' }},
	          body: JSON.stringify(payload),
	        }});
	        const data = await response.json();
	        if (!response.ok) throw new Error(data.error || 'ALPR request failed');
	        showPopup('ALPR Result', data.message || 'ALPR request completed.');
	      }} catch (error) {{
	        showPopup('ALPR Failed', error.message || 'ALPR request failed.');
	      }} finally {{
	        if (button) button.disabled = false;
	      }}
	    }});
	  }});
	}})();
	</script>
	</body></html>"""

    def _render_videos_page(self, page: int = 1, selected_relative: str = "") -> str:
        videos = self._list_saved_videos()
        total_videos = len(videos)
        total_pages = max(1, (total_videos + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
        page = min(max(1, page), total_pages)
        start = (page - 1) * GALLERY_PAGE_SIZE
        page_videos = videos[start : start + GALLERY_PAGE_SIZE]
        selected_video = None
        if page_videos:
            if selected_relative:
                selected_video = next(
                    (video_path for video_path in page_videos if self._relative_video_path(video_path) == selected_relative),
                    None,
                )
            if selected_video is None:
                selected_video = page_videos[0]
        gallery = self._render_video_cards(page_videos, selected_video, page)
        pagination = self._pagination_html(
            "/videos",
            page,
            total_videos,
            GALLERY_PAGE_SIZE,
            extra_params={"selected": selected_relative} if selected_relative else None,
        )
        summary = self._gallery_summary_html(page, total_videos, GALLERY_PAGE_SIZE, "videos")
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Saved Videos</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.stack {{ display: grid; gap: 18px; }}
.hero {{ display: flex; align-items: end; justify-content: space-between; gap: 18px; flex-wrap: wrap; margin-bottom: 8px; }}
.hero h1 {{ margin: 0; font-size: clamp(1.8rem, 3vw, 2.6rem); letter-spacing: -0.03em; }}
.hero p {{ margin: 6px 0 0; color: #94a3b8; max-width: 680px; }}
	.video-layout {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(min(320px, 100%), 0.9fr); gap: 18px; align-items: start; }}
	.player-stack {{ display: grid; gap: 18px; }}
.player-card, .card {{ background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(10, 15, 29, 0.96)); padding: 18px; border-radius: 18px; border: 1px solid rgba(71, 85, 105, 0.42); box-shadow: 0 22px 50px rgba(0, 0, 0, 0.28); }}
.video-list {{ display: grid; gap: 12px; margin-top: 16px; }}
.video-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px; align-items: center; padding: 14px 16px; border: 1px solid #1f2937; border-radius: 14px; background: rgba(15, 23, 42, 0.6); }}
.video-row.active {{ border-color: rgba(250, 204, 21, 0.7); background: linear-gradient(135deg, rgba(250, 204, 21, 0.1), rgba(15, 23, 42, 0.92)); box-shadow: inset 0 0 0 1px rgba(250, 204, 21, 0.16); }}
.video-meta {{ min-width: 0; flex: 1; }}
	.video-name {{ margin: 0; font-size: 1rem; color: #f8fafc; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.video-subtle {{ margin: 6px 0 0; color: #94a3b8; font-size: 0.92rem; }}
.video-badges {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; }}
.video-chip {{ display: inline-flex; align-items: center; min-height: 28px; padding: 0 10px; border-radius: 999px; background: rgba(30, 41, 59, 0.92); color: #cbd5e1; font-size: 0.82rem; border: 1px solid #334155; }}
.video-actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; align-items: flex-start; }}
.video-actions a {{ text-decoration: none; }}
.video-icon-form {{ margin: 0; }}
.video-action-slot {{ display: inline-flex; flex-direction: column; align-items: center; justify-content: flex-start; width: 34px; gap: 6px; }}
.video-icon-button {{ display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 999px; border: 1px solid #334155; background: transparent; color: #cbd5e1; cursor: pointer; padding: 0; }}
.video-icon-button svg {{ width: 18px; height: 18px; fill: currentColor; display: block; }}
.video-icon-button:hover {{ border-color: #facc15; color: #facc15; background: #172033; }}
.video-icon-button.danger:hover {{ border-color: #f87171; color: #f87171; background: #2b1520; }}
.video-icon-button.confirm-armed, .video-icon-button.danger.confirm-armed, .video-icon-button.danger.confirm-armed:hover {{ border-color: #facc15; color: #111827; background: #facc15; }}
.video-icon-button.busy {{ border-color: #38bdf8; color: #38bdf8; background: #0f172a; }}
.video-action-status {{ min-height: 10px; color: #38bdf8; font-size: 0.68rem; font-variant-numeric: tabular-nums; line-height: 1; text-align: center; white-space: nowrap; }}
.player-card {{ position: sticky; top: 18px; }}
.eyebrow {{ display: inline-flex; align-items: center; gap: 8px; color: #facc15; font-size: 0.78rem; letter-spacing: 0.16em; text-transform: uppercase; }}
.eyebrow::before {{ content: ""; width: 26px; height: 1px; background: currentColor; opacity: 0.8; }}
	.player-wrap {{ position: relative; width: 100%; }}
	.player-wrap video {{ width: 100%; max-height: 460px; border-radius: 10px; display: block; background: #000; }}
	.live-wrap {{ position: relative; width: 100%; border-radius: 10px; overflow: hidden; background: #000; }}
	.live-wrap img {{ width: 100%; max-height: 460px; min-height: 240px; object-fit: contain; display: block; background: #000; }}
.player-overlay-button {{ position: absolute; top: 10px; right: 10px; width: 38px; height: 38px; border-radius: 999px; border: 0; background: rgba(2, 6, 23, 0.72); color: #f8fafc; display: inline-flex; align-items: center; justify-content: center; cursor: pointer; }}
.player-overlay-button svg {{ width: 20px; height: 20px; fill: currentColor; display: block; }}
.player-overlay-button:hover {{ background: rgba(15, 23, 42, 0.92); }}
	.player-path {{ margin: 14px 0 0; font-size: 0.98rem; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
	.player-links {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }}
	.player-links a {{ color: #cbd5e1; text-decoration: none; border-bottom: 1px solid transparent; }}
	.player-links a:hover {{ color: #f8fafc; border-color: rgba(248, 250, 252, 0.45); }}
	.player-status {{ margin: 12px 0 0; color: #94a3b8; font-size: 0.92rem; }}
.action-overlay {{ position: fixed; inset: 0; background: rgba(2, 6, 23, 0.72); display: none; align-items: center; justify-content: center; z-index: 1000; padding: 24px; }}
.action-overlay.visible {{ display: flex; }}
.action-overlay-card {{ min-width: min(92vw, 360px); max-width: 420px; background: #111827; border: 1px solid #334155; border-radius: 18px; padding: 22px; text-align: center; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35); }}
.action-overlay-card p {{ margin: 10px 0 0; color: #cbd5e1; white-space: pre-wrap; text-align: left; }}
.action-spinner {{ width: 34px; height: 34px; margin: 0 auto; border-radius: 999px; border: 3px solid #334155; border-top-color: #facc15; animation: spin 0.9s linear infinite; }}
.action-progress {{ width: 100%; height: 12px; margin-top: 14px; border-radius: 999px; background: #1f2937; border: 1px solid #334155; overflow: hidden; display: none; }}
.action-progress-bar {{ width: 0%; height: 100%; background: linear-gradient(90deg, #facc15, #fb7185); transition: width 0.12s ease-out; }}
.action-progress-text {{ margin-top: 10px; color: #facc15; font-variant-numeric: tabular-nums; display: none; }}
.action-overlay-close {{ margin-top: 14px; background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 9px 14px; cursor: pointer; font-weight: 700; display: none; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.pagination {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 18px 0; }}
.pagination a, .pagination span {{ border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; text-decoration: none; }}
.pagination span {{ color: #94a3b8; }}
p {{ margin: 8px 0 0; word-break: break-word; }}
@media (max-width: 960px) {{ .video-layout {{ grid-template-columns: 1fr; }} .player-card {{ position: static; }} }}
@media (max-width: 720px) {{ .video-row {{ grid-template-columns: 1fr; }} .video-actions {{ justify-content: flex-start; }} }}
@media (max-width: 520px) {{
  .hero {{ align-items: flex-start; }}
  .video-actions {{ display: grid; grid-template-columns: repeat(5, minmax(34px, 1fr)); justify-content: stretch; }}
  .video-action-slot {{ width: 100%; }}
  .video-icon-button {{ margin: 0 auto; }}
  .player-wrap video, .live-wrap img {{ max-height: 52vh; }}
}}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("videos")}
<div class="hero"><div><h1>Videos</h1><p>A cleaner browser for reviewing clips and running quick actions.</p></div></div>
{summary}
{pagination}
{gallery}
{pagination}
</div>
<div id="action-overlay" class="action-overlay" aria-live="polite" aria-busy="true">
  <div class="action-overlay-card">
    <div id="action-spinner" class="action-spinner"></div>
    <h2 id="action-title">Working...</h2>
    <p id="action-message">Please wait while the action finishes.</p>
    <div id="action-progress" class="action-progress"><div id="action-progress-bar" class="action-progress-bar"></div></div>
    <div id="action-progress-text" class="action-progress-text">0%</div>
    <button id="action-close" class="action-overlay-close" type="button">Close</button>
  </div>
</div>
</body></html>"""

    def _render_plates_page(self) -> str:
        cards = self._render_recent_plate_cards(limit=100)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Detected Plates</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 18px; }}
.card {{ background: #111827; padding: 14px; border-radius: 14px; }}
.card img {{ width: 100%; border-radius: 10px; display: block; margin-top: 10px; }}
.plate-code {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.08em; }}
.meta {{ color: #cbd5e1; margin-top: 6px; }}
.page-action {{ display: inline-flex; align-items: center; justify-content: center; margin: 0 0 18px; color: #111827; background: #facc15; border-radius: 8px; padding: 10px 14px; text-decoration: none; font-weight: 700; width: 180px; min-height: 40px; }}
.page-actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; }}
.page-actions form {{ margin: 0; }}
.telegram-action {{ background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 700; width: 180px; min-height: 40px; }}
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
</div>
<div class="grid">{cards}</div>
</div>
</body></html>"""

    def _render_test_page(self) -> str:
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Test Upload</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; max-width: 760px; }}
label {{ display: block; margin-bottom: 10px; color: #cbd5e1; }}
input[type=file] {{ display: block; margin-top: 8px; color: #e2e8f0; }}
input[type=file] {{ width: 100%; }}
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
<p>Upload a JPG or PNG and the server will run the same plate pipeline used for saved event frames: zone-based ALPR crop when a zone has fast-alpr enabled, otherwise optional <code>PLATE_ROI</code>, then local <code>fast-alpr</code>, then OpenALPR only if the local confidence gate passes.</p>
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
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Test Upload Result</title>
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
<p>Policy applied: crop to the triggered fast-alpr zone when available, otherwise to <code>PLATE_ROI</code> when configured, run <code>fast-alpr</code>, and only send to OpenALPR when <code>fast-alpr</code> found a plate with confidence at or above <code>{self.config.fast_alpr_min_confidence:.2f}</code>.</p>
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

    def _render_saved_image_fast_alpr_result_page(self, relative_path: str, result: dict[str, Any], result_path: Path) -> str:
        image_url = self._event_file_url(relative_path)
        detail_url = self._image_detail_url(relative_path)
        result_relative = self._relative_image_path(result_path)
        result_url = self._event_file_url(result_relative)
        detections = self._extract_fast_alpr_detections(
            result,
            relative_path,
            Path(relative_path).parent.as_posix(),
            time.time(),
        )
        detections_html = "".join(
            f"<li><strong>{html.escape(item.plate)}</strong> ({item.confidence:.2f})</li>"
            for item in detections
        ) or "<li>No plates detected.</li>"
        result_text = json.dumps(result, indent=2)
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ALPR Image Check</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.layout {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px; align-items: start; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 18px; }}
img {{ width: 100%; border-radius: 8px; display: block; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; padding: 14px; border-radius: 8px; overflow-x: auto; }}
@media (max-width: 960px) {{ .layout {{ grid-template-columns: 1fr; }} }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("images")}
<p><a href="{detail_url}">Back to image</a></p>
<h1>ALPR Image Check</h1>
<div class="layout">
<div class="panel"><img src="{image_url}" alt="{html.escape(relative_path)}"><p>{html.escape(Path(relative_path).name)}</p></div>
<div class="panel">
<h2>Detections</h2>
<ul>{detections_html}</ul>
<p><a href="{result_url}">Open raw fast-alpr JSON</a></p>
<pre>{html.escape(result_text)}</pre>
</div>
</div>
</div>
</body></html>"""

    def _render_saved_image_fast_alpr_error_page(self, message: str) -> str:
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ALPR Image Check Failed</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 18px; max-width: 760px; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("images")}
<div class="panel">
<h1>ALPR Image Check Failed</h1>
<p>{html.escape(message)}</p>
<p><a href="/images">Back to images</a></p>
</div>
</div>
</body></html>"""

    def _handle_roi_update(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
            zones_payload = payload.get("zones")
            if isinstance(zones_payload, list):
                seen_zone_ids: Set[str] = set()
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
                    zone.send_telegram = bool(zone_payload.get("send_telegram", zone.send_telegram))
                    zone.record_seconds = max(1.0, float(zone_payload.get("record_seconds", zone.record_seconds)))
                    zone.image_count = max(1, int(zone_payload.get("image_count", zone.image_count)))
                    zone.coverage_trigger_percent = min(
                        100.0,
                        max(0.0, float(zone_payload.get("coverage_trigger_percent", zone.coverage_trigger_percent))),
                    )
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

    def _handle_telegram_settings_update(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
            if "enabled" not in payload:
                raise ValueError("enabled is required")
            enabled = payload.get("enabled")
            if isinstance(enabled, str):
                self.config.telegram_alerts_enabled = parse_bool(
                    enabled,
                    default=self.config.telegram_alerts_enabled,
                )
            else:
                self.config.telegram_alerts_enabled = bool(enabled)
            self._save_runtime_config()
            body = json.dumps(self._telegram_settings_for_ui()).encode("utf-8")
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

    def _system_time_snapshot(self) -> dict[str, Any]:
        now = datetime.now().astimezone()
        return {
            "epoch_ms": int(time.time() * 1000),
            "iso": now.isoformat(timespec="seconds"),
            "local_text": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "timezone": time.tzname[time.localtime().tm_isdst > 0],
        }

    def _format_system_time(self, epoch_seconds: float) -> str:
        return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    def _default_event_log_zone_id(self, category: str) -> str:
        if category not in {"motion", "video", "image"}:
            return ""
        if self.event and self.event.zones_triggered:
            return self._primary_zone_id_for(self.event.zones_triggered)
        if self.video_recording and self.video_recording.started_from_zone_ids:
            return self._primary_zone_id_for(self.video_recording.started_from_zone_ids)
        if self.last_triggered_zone_ids:
            return self._primary_zone_id_for(self.last_triggered_zone_ids)
        return ""

    def _add_event_log(self, category: str, message: str, zone_id: str = "") -> None:
        effective_zone_id = zone_id or self._default_event_log_zone_id(category)
        entry = {
            "time_epoch": time.time(),
            "category": category,
            "zone_id": effective_zone_id,
            "message": message,
        }
        with self.event_log_lock:
            self.event_log.append(entry)

    def _event_log_snapshot(self) -> List[dict[str, Any]]:
        with self.event_log_lock:
            entries = list(self.event_log)
        return [
            {
                **entry,
                "time_text": self._format_system_time(float(entry["time_epoch"])),
            }
            for entry in reversed(entries)
        ]

    def _render_event_log_items(self) -> str:
        entries = self._event_log_snapshot()
        if not entries:
            return '<div class="event-log-empty">No events logged yet.</div>'
        def _zone_class(entry: dict) -> str:
            zone_id = str(entry.get("zone_id") or "")
            return f"event-log-kind zone-{zone_id}" if zone_id else "event-log-kind zone-none"

        return "".join(
            (
                lambda message_text, time_text, zone_title, category_text:
                f'<div class="event-log-item" title="{message_text}"><span class="event-log-time" title="{time_text}">{time_text}</span>'
                f'<span class="{_zone_class(entry)}" title="{zone_title}">{category_text}</span>'
                f'<span class="event-log-message" title="{message_text}">{message_text}</span></div>'
            )(
                html.escape(str(entry["message"])),
                html.escape(entry["time_text"]),
                html.escape(f"Zone: {entry.get('zone_id')}" if entry.get("zone_id") else "No zone"),
                html.escape(str(entry["category"])),
            )
            for entry in entries
        )

    def _handle_system_time_sync(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
            epoch_ms_value = payload.get("epoch_ms")
            if epoch_ms_value is None:
                raise ValueError("epoch_ms is required")
            epoch_ms = float(epoch_ms_value)
            epoch_seconds = epoch_ms / 1000.0
            target_time = datetime.fromtimestamp(epoch_seconds)
            if target_time.year < 2020 or target_time.year > 2100:
                raise ValueError("Computer time is outside the allowed range")
            result = subprocess.run(
                ["date", "-s", f"@{epoch_seconds:.3f}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            if result.returncode != 0:
                error = (result.stderr or result.stdout or "date command failed").strip()
                raise PermissionError(
                    f"Could not set the system clock: {error}. "
                    "The Docker container may need SYS_TIME permission or host-level time sync."
                )
            body = json.dumps(
                {
                    "ok": True,
                    "message": "System clock synced with this computer.",
                    "time": self._system_time_snapshot(),
                }
            ).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            handler.send_response(HTTPStatus.BAD_REQUEST)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

    def _handle_env_update(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            if content_length > 256 * 1024:
                raise ValueError(".env content is too large")
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
            content = payload.get("content")
            if not isinstance(content, str):
                raise ValueError("content is required")
            if "\x00" in content:
                raise ValueError(".env content cannot contain null bytes")
            apply_runtime = bool(payload.get("apply", False))
            env_path = self._env_file_path()
            env_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = env_path.with_name(f"{env_path.name}.tmp")
            temp_path.write_text(content)
            temp_path.replace(env_path)
            apply_result: dict[str, Any] = {}
            if apply_runtime:
                apply_result = self._apply_env_content_to_runtime(content)
            message = "Saved."
            if apply_runtime:
                message = "Saved and applied to runtime."
                if apply_result.get("restart_needed"):
                    restart_names = ", ".join(apply_result["restart_needed"])
                    message += f" Restart needed for: {restart_names}."
            body = json.dumps(
                {
                    "saved": True,
                    "applied": apply_runtime,
                    "path": str(env_path),
                    "message": message,
                    "changed": apply_result.get("changed", []),
                    "restart_needed": apply_result.get("restart_needed", []),
                }
            ).encode("utf-8")
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

    def _apply_env_content_to_runtime(self, content: str) -> dict[str, Any]:
        parsed_values = dotenv_values(stream=StringIO(content))
        env_values = {
            key: "" if value is None else str(value)
            for key, value in parsed_values.items()
            if key
        }
        new_config = Config.from_mapping(env_values)
        old_config = self.config
        changed = [
            field_name
            for field_name in Config.__dataclass_fields__
            if getattr(old_config, field_name) != getattr(new_config, field_name)
        ]
        restart_needed = [
            name
            for name in changed
            if name in {"web_host", "web_port", "rtsp_url", "rtsp_capture_options", "capture_buffer_size"}
        ]
        self.config = new_config
        self.client = OpenAlprClient(self.config)
        self.config.event_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_config_path = self.config.event_output_dir.parent / "watcher-config.json"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.config.rtsp_capture_options

        primary_zone = self._find_motion_zone("yellow")
        if primary_zone and self.config.roi:
            primary_zone.roi = self.config.roi
        if (
            "TELEGRAM_ALERTS_ENABLED" in env_values
            and old_config.telegram_alerts_enabled != self.config.telegram_alerts_enabled
        ):
            for zone in self.motion_zones:
                zone.send_telegram = self.config.telegram_alerts_enabled
        self._sync_primary_zone_to_config()
        self._save_runtime_config()
        self._prune_saved_images()
        logging.info(
            "Applied runtime environment configuration: changed=%s restart_needed=%s",
            ",".join(changed) or "none",
            ",".join(restart_needed) or "none",
        )
        return {"changed": changed, "restart_needed": restart_needed}

    def _serve_event_file(self, handler: BaseHTTPRequestHandler, relative_path: str, head_only: bool = False) -> None:
        candidates: List[Path] = []
        for resolver in (self._image_path_from_relative, self._video_path_from_relative):
            try:
                candidates.append(resolver(relative_path))
            except ValueError:
                pass
        if not candidates:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        target = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), candidates[0])
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
        file_size = target.stat().st_size
        range_header = handler.headers.get("Range", "")
        start = 0
        end = file_size - 1
        if range_header and range_header.startswith("bytes="):
            try:
                range_spec = range_header[6:].split("-")
                start = int(range_spec[0]) if range_spec[0] else 0
                end = int(range_spec[1]) if len(range_spec) > 1 and range_spec[1] else file_size - 1
                end = min(end, file_size - 1)
                start = max(0, min(start, end))
                length = end - start + 1
                handler.send_response(HTTPStatus.PARTIAL_CONTENT)
                handler.send_header("Content-Type", content_type)
                handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                handler.send_header("Content-Length", str(length))
                handler.send_header("Accept-Ranges", "bytes")
                handler.end_headers()
                if not head_only:
                    with target.open("rb") as f:
                        f.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = f.read(min(FILE_STREAM_CHUNK_SIZE, remaining))
                            if not chunk:
                                break
                            handler.wfile.write(chunk)
                            remaining -= len(chunk)
                return
            except (ValueError, IndexError):
                pass
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(file_size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        if not head_only:
            with target.open("rb") as f:
                while True:
                    chunk = f.read(FILE_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    handler.wfile.write(chunk)

    def _handle_test_upload(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            _, files = self._read_multipart_form(handler)
            image_file = files.get("image")
            if image_file is None:
                raise ValueError("Missing image upload")
            payload = bytes(image_file.get("payload") or b"")
            if not payload:
                raise ValueError("Empty image upload")
        except Exception as exc:
            handler.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        frame_array = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame_array is None:
            handler.send_error(HTTPStatus.BAD_REQUEST, "Invalid image upload")
            return

        frame = self._resize_frame(frame_array)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        event_name = f"{self.config.camera_name}_test_{stamp}"
        event_dir = self.config.image_output_dir / event_name
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
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{status_text}</title>
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

    def _saved_image_path_from_relative(self, relative_path: str) -> Path:
        target = self._image_path_from_relative(relative_path)
        if target.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Invalid image path")
        if not target.exists() or not target.is_file():
            raise ValueError("Image file was not found")
        return target

    def _handle_saved_image_fast_alpr(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            raw_body = handler.rfile.read(content_length)
            if handler.headers.get("Content-Type", "").startswith("application/json"):
                payload = json.loads(raw_body or b"{}")
                relative_path = str(payload.get("path") or "")
            else:
                payload = parse_qs(raw_body.decode("utf-8", errors="replace"))
                relative_path = str((payload.get("path") or [""])[0])
            image_path = self._saved_image_path_from_relative(relative_path)
            if not self.config.fast_alpr_url:
                raise ValueError("FAST_ALPR_URL is not configured")
            image_bytes = image_path.read_bytes()
            result = self._recognize_fast_alpr(image_bytes)
            result_path = image_path.with_name(f"{image_path.stem}.manual_fast_alpr.json")
            result_path.write_text(json.dumps(result, indent=2))
            relative = self._relative_image_path(image_path)
            detections = self._extract_fast_alpr_detections(result, relative, image_path.parent.name, time.time())
            plates_text = ", ".join(f"{d.plate} ({d.confidence:.2f})" for d in detections) or "No plates detected"
            body = json.dumps({"ok": True, "message": f"ALPR: {plates_text}"}).encode("utf-8")
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

    def _saved_video_path_from_relative(self, relative_path: str) -> Path:
        target = self._video_path_from_relative(relative_path)
        if target.suffix.lower() != ".mp4":
            raise ValueError("Invalid video path")
        if not target.exists() or not target.is_file():
            raise ValueError("Video file was not found")
        return target

    def _read_simple_form_value(self, handler: BaseHTTPRequestHandler, key: str) -> str:
        content_length = int(handler.headers.get("Content-Length", "0") or "0")
        raw_body = handler.rfile.read(content_length)
        if handler.headers.get("Content-Type", "").startswith("application/json"):
            payload = json.loads(raw_body or b"{}")
            return str(payload.get(key) or "")
        payload = parse_qs(raw_body.decode("utf-8", errors="replace"))
        return str((payload.get(key) or [""])[0])

    def _read_simple_form(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        content_length = int(handler.headers.get("Content-Length", "0") or "0")
        raw_body = handler.rfile.read(content_length)
        if handler.headers.get("Content-Type", "").startswith("application/json"):
            payload = json.loads(raw_body or b"{}")
            return {str(key): "" if value is None else str(value) for key, value in payload.items()}
        payload = parse_qs(raw_body.decode("utf-8", errors="replace"))
        return {str(key): str(values[0]) for key, values in payload.items() if values}

    def _read_multipart_form(
        self,
        handler: BaseHTTPRequestHandler,
        max_bytes: int = MAX_IMAGE_UPLOAD_BYTES,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        content_type = handler.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("Expected multipart/form-data upload")
        content_length = int(handler.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            raise ValueError("Empty multipart request")
        if content_length > max_bytes:
            raise ValueError(f"Upload is too large (max {max_bytes // (1024 * 1024)} MB)")
        raw_body = handler.rfile.read(content_length)
        message = BytesParser(policy=email_policy).parsebytes(
            (
                f"Content-Type: {content_type}\r\n"
                "MIME-Version: 1.0\r\n"
                "\r\n"
            ).encode("utf-8")
            + raw_body
        )
        fields: dict[str, str] = {}
        files: dict[str, dict[str, Any]] = {}
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename is None:
                fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            else:
                files[name] = {
                    "filename": filename,
                    "content_type": part.get_content_type(),
                    "payload": payload,
                }
        return fields, files

    def _handle_video_snapshot_fast_alpr(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            if self.video_recording is not None:
                self._send_recording_active_extraction_error(handler)
                return
            if not self.config.fast_alpr_url:
                raise ValueError("FAST_ALPR_URL is not configured")
            fields, files = self._read_multipart_form(handler)
            image_item = files.get("image")
            if image_item is None:
                raise ValueError("Missing snapshot image")
            video_path_value = fields.get("video_path", "")
            video_path = self._saved_video_path_from_relative(str(video_path_value))
            image_bytes = bytes(image_item.get("payload") or b"")
            if not image_bytes:
                raise ValueError("Snapshot image is empty")
            frame_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame_array is None:
                raise ValueError("Snapshot image is invalid")

            position_seconds = 0.0
            try:
                position_seconds = float(fields.get("position_seconds", "0") or 0.0)
            except ValueError:
                position_seconds = 0.0

            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            event_name = f"{self.config.camera_name}_video_snapshot_{stamp}"
            event_dir = self.config.image_output_dir / event_name
            event_dir.mkdir(parents=True, exist_ok=True)
            image_path = event_dir / "video_snapshot.jpg"
            image_path.write_bytes(image_bytes)
            (event_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "camera_name": self.config.camera_name,
                        "started_at_epoch": time.time(),
                        "ended_at_epoch": time.time(),
                        "trigger_count": 0,
                        "saved_frames": 1,
                        "clip_path": self._relative_video_path(video_path),
                        "video_snapshot": True,
                        "video_position_seconds": position_seconds,
                    },
                    indent=2,
                )
            )
            result = self._recognize_fast_alpr(image_bytes)
            result_path = image_path.with_name(f"{image_path.stem}.manual_fast_alpr.json")
            result_path.write_text(json.dumps(result, indent=2))
            relative = self._relative_image_path(image_path)
            detections = self._extract_fast_alpr_detections(result, relative, image_path.parent.name, time.time())
            plates_text = ", ".join(f"{d.plate} ({d.confidence:.2f})" for d in detections) or "No plates detected"
            body = json.dumps({"ok": True, "message": f"ALPR: {plates_text}"}).encode("utf-8")
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

    def _video_capture_duration_seconds(self, capture) -> float:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and frame_count > 0:
            return max(0.0, frame_count / fps)
        return 0.0

    def _video_duration_seconds_with_ffprobe(self, video_path: Path) -> float:
        if not shutil.which("ffprobe"):
            return 0.0
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True,
            )
            if result.returncode != 0:
                return 0.0
            return max(0.0, float((result.stdout or "").strip() or 0.0))
        except Exception:
            return 0.0

    def _saved_video_duration_seconds(
        self,
        video_path: Path,
        metadata: Optional[dict] = None,
        allow_probe: bool = True,
    ) -> float:
        if metadata:
            try:
                recording_seconds = float(metadata.get("recording_seconds") or 0.0)
                if recording_seconds > 0:
                    return recording_seconds
            except (TypeError, ValueError):
                pass
        probed_duration = self._video_duration_seconds_with_ffprobe(video_path)
        if probed_duration > 0:
            return probed_duration
        if not allow_probe:
            return 0.0
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return 0.0
        try:
            return self._video_capture_duration_seconds(capture)
        finally:
            capture.release()

    def _load_or_create_video_metadata(self, video_path: Path) -> dict:
        metadata_path = video_path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text())
                if isinstance(loaded, dict):
                    metadata = loaded
            except Exception:
                logging.exception("Failed to read video metadata for %s", video_path)
        recording_seconds = self._saved_video_duration_seconds(video_path, metadata, allow_probe=True)
        changed = not bool(metadata)
        metadata.setdefault("camera_name", self.config.camera_name)
        metadata.setdefault("video_path", str(video_path))
        metadata.setdefault("source", "uploaded")
        metadata.setdefault("started_at_epoch", video_path.stat().st_mtime)
        metadata.setdefault("first_frame_at_epoch", None)
        metadata.setdefault("ends_at_epoch", None)
        metadata.setdefault("zone_ids", [])
        metadata.setdefault("event_count", 0)
        metadata.setdefault("prebuffer_seconds", 0.0)
        if float(metadata.get("recording_seconds") or 0.0) <= 0 and recording_seconds > 0:
            metadata["recording_seconds"] = recording_seconds
            changed = True
        elif "recording_seconds" not in metadata:
            metadata["recording_seconds"] = 0.0
            changed = True
        if changed:
            metadata_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def _format_duration_label(self, seconds: float) -> str:
        if seconds <= 0:
            return "Unknown"
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _extract_saved_video_candidates(
        self,
        video_path: Path,
        limit: int,
    ) -> List[CandidateFrame]:
        if shutil.which("ffmpeg"):
            duration_seconds = self._saved_video_duration_seconds(video_path)
            if duration_seconds <= 0:
                target_seconds = [0.0]
            else:
                target_seconds = self._video_extraction_timestamps(0.0, duration_seconds, limit)
            candidates: List[CandidateFrame] = []
            for position_seconds in target_seconds:
                frame = self._extract_local_video_frame_with_ffmpeg(video_path, position_seconds)
                if frame is None:
                    continue
                candidates.append(
                    CandidateFrame(
                        frame=frame.copy(),
                        timestamp=position_seconds,
                        motion_area=0,
                        sharpness=self._compute_sharpness(self._plate_crop(frame)),
                        jpeg_bytes=self._encode_jpeg(frame),
                        source="video",
                    )
                )
            return candidates
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open {video_path.name}")
        try:
            duration_seconds = self._video_capture_duration_seconds(capture)
            if duration_seconds <= 0:
                target_seconds = [0.0]
            else:
                target_seconds = self._video_extraction_timestamps(0.0, duration_seconds, limit)
            candidates: List[CandidateFrame] = []
            for position_seconds in target_seconds:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, position_seconds) * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                candidates.append(
                    CandidateFrame(
                        frame=frame.copy(),
                        timestamp=position_seconds,
                        motion_area=0,
                        sharpness=self._compute_sharpness(self._plate_crop(frame)),
                        jpeg_bytes=self._encode_jpeg(frame),
                        source="video",
                    )
                )
            return candidates
        finally:
            capture.release()

    def _handle_video_extract_images(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            if self.video_recording is not None:
                self._send_recording_active_extraction_error(handler)
                return
            form = self._read_simple_form(handler)
            relative_path = str(form.get("path") or "")
            video_path = self._saved_video_path_from_relative(relative_path)
            metadata = self._load_or_create_video_metadata(video_path)
            job_id = f"extract_{int(time.time() * 1000)}"
            body = json.dumps({"ok": True, "job_id": job_id, "message": f"Extracting images from {video_path.name}…"}).encode("utf-8")
            handler.send_response(HTTPStatus.ACCEPTED)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            def _do_extract() -> None:
                requested_total = max(1, self.config.manual_extract_image_count)
                expected_total = requested_total
                self.extraction_status[job_id] = {
                    "done": False,
                    "saved": 0,
                    "total": expected_total,
                    "video": video_path.name,
                    "video_path": str(video_path),
                    "error": None,
                    "error_detail": None,
                    "error_type": None,
                    "cancel": False,
                    "stage": "sampling",
                    "requested_total": requested_total,
                }
                last_position_seconds = None
                try:
                    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    event_name = f"{self.config.camera_name}_video_extract_{stamp}"
                    event_dir = self.config.image_output_dir / event_name
                    event_dir.mkdir(parents=True, exist_ok=True)
                    saved_paths: List[Path] = []
                    duration_seconds = self._saved_video_duration_seconds(video_path, metadata, allow_probe=True)
                    attempt_limit = requested_total
                    if duration_seconds <= 0:
                        target_seconds = [0.0]
                    else:
                        attempt_limit = max(requested_total, requested_total * 4)
                        target_seconds = self._video_extraction_timestamps(
                            0.0,
                            duration_seconds,
                            attempt_limit,
                        )
                    self.extraction_status[job_id]["total"] = requested_total
                    self.extraction_status[job_id]["attempt_total"] = len(target_seconds)
                    self.extraction_status[job_id]["duration_seconds"] = duration_seconds
                    self.extraction_status[job_id]["event_dir"] = str(event_dir)
                    self.extraction_status[job_id]["stage"] = "saving"
                    failed_frames = 0
                    for index, position_seconds in enumerate(target_seconds, start=1):
                        if len(saved_paths) >= requested_total:
                            break
                        last_position_seconds = position_seconds
                        self.extraction_status[job_id]["current_index"] = len(saved_paths) + 1
                        self.extraction_status[job_id]["attempt_index"] = index
                        self.extraction_status[job_id]["current_position_seconds"] = position_seconds
                        if self.extraction_status[job_id].get("cancel"):
                            self.extraction_status[job_id]["done"] = True
                            self.extraction_status[job_id]["stage"] = "cancelled"
                            self._add_event_log("image", f"Extraction cancelled after {len(saved_paths)} images from {video_path.name}")
                            return
                        frame = self._extract_local_video_frame(video_path, position_seconds)
                        if frame is None:
                            failed_frames += 1
                            self.extraction_status[job_id]["failed_frames"] = failed_frames
                            continue
                        failed_frames = 0
                        candidate = CandidateFrame(
                            frame=frame.copy(),
                            timestamp=position_seconds,
                            motion_area=0,
                            sharpness=self._compute_sharpness(self._plate_crop(frame)),
                            jpeg_bytes=self._encode_jpeg(frame),
                            source="video",
                        )
                        pipeline = self._run_detection_pipeline(
                            candidate.frame,
                            event_dir,
                            f"frame_{len(saved_paths) + 1:02d}",
                            event_name,
                            position_seconds,
                            enable_alpr=False,
                            jpeg_bytes=candidate.jpeg_bytes,
                        )
                        saved_paths.append(pipeline["frame_path"])
                        self.extraction_status[job_id]["saved"] = len(saved_paths)
                    if not saved_paths:
                        error_detail = (
                            f"No frames could be extracted from {video_path.name}. "
                            f"Duration={duration_seconds:.3f}s, requested={requested_total}, sampled={len(target_seconds)}, failed={failed_frames}, "
                            f"event_dir={event_dir}."
                        )
                        self.extraction_status[job_id].update({
                            "done": True,
                            "saved": 0,
                            "error": "No frames could be extracted",
                            "error_detail": error_detail,
                            "error_type": "ValueError",
                            "stage": "failed",
                        })
                        self._add_event_log("image", error_detail)
                        logging.error(error_detail)
                        return
                    if len(saved_paths) < requested_total:
                        warning_detail = (
                            f"Only {len(saved_paths)} of {requested_total} requested images could be extracted from {video_path.name}. "
                            f"Tried {len(target_seconds)} positions; duration={duration_seconds:.3f}s."
                        )
                        self.extraction_status[job_id]["warning"] = warning_detail
                        logging.warning(warning_detail)
                    summary = {
                        "camera_name": self.config.camera_name,
                        "started_at_epoch": time.time(),
                        "ended_at_epoch": time.time(),
                        "trigger_count": 0,
                        "saved_frames": len(saved_paths),
                        "clip_path": self._relative_video_path(video_path),
                        "video_extract": True,
                        "image_source": "video",
                        "event_policy": "images-only",
                        "zone_images": [],
                        "fast_alpr_results_count": 0,
                        "openalpr_results_count": 0,
                        "plates": [],
                    }
                    (event_dir / "summary.json").write_text(json.dumps(summary, indent=2))
                    self._prune_saved_images()
                    self.extraction_status[job_id]["done"] = True
                    self.extraction_status[job_id]["stage"] = "partial" if len(saved_paths) < requested_total else "done"
                    self._add_event_log("image", f"Extracted {len(saved_paths)} images from {video_path.name}")
                    logging.info("Extracted %s images from %s into %s", len(saved_paths), video_path.name, event_dir.name)
                except Exception as exc:
                    status = self.extraction_status.get(job_id, {})
                    error_detail = (
                        f"{type(exc).__name__}: {exc}\n"
                        f"video={video_path}\n"
                        f"stage={status.get('stage', 'unknown')}, saved={status.get('saved', 0)}/{status.get('total', expected_total)}, "
                        f"requested={requested_total}, duration={status.get('duration_seconds', 'unknown')}\n"
                        f"index={status.get('current_index', 'unknown')}, position={status.get('current_position_seconds', last_position_seconds)}"
                    )
                    status.update({
                        "done": True,
                        "saved": status.get("saved", 0),
                        "total": status.get("total", expected_total),
                        "video": video_path.name,
                        "video_path": str(video_path),
                        "error": str(exc),
                        "error_detail": error_detail,
                        "error_type": type(exc).__name__,
                        "stage": "failed",
                    })
                    self.extraction_status[job_id] = status
                    self._add_event_log("image", f"Extraction failed for {video_path.name}: {type(exc).__name__}: {exc}")
                    logging.exception("Background image extraction failed for %s: %s", video_path, error_detail)
            threading.Thread(target=_do_extract, daemon=True).start()
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            handler.send_response(HTTPStatus.BAD_REQUEST)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
    def _handle_extraction_cancel(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0") or "0")
            payload = json.loads(handler.rfile.read(content_length) or b"{}")
            job_id = str(payload.get("job_id") or "")
            if job_id and job_id in self.extraction_status:
                self.extraction_status[job_id]["cancel"] = True
            body = json.dumps({"ok": True}).encode("utf-8")
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

    def _handle_video_upload(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            _, files = self._read_multipart_form(handler, max_bytes=500 * 1024 * 1024)
            video_file = files.get("video")
            video_bytes = bytes((video_file or {}).get("payload") or (video_file or {}).get("data") or b"")
            if not video_file or not video_bytes:
                raise ValueError("No video file provided")
            filename = video_file.get("filename") or "upload.mp4"
            safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(filename).name)
            if not safe_name.lower().endswith(".mp4"):
                safe_name = Path(safe_name).stem + ".mp4"
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            output_path = self._video_output_dir() / f"{self.config.camera_name}_upload_{stamp}_{safe_name}"
            output_path.write_bytes(video_bytes)
            output_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "camera_name": self.config.camera_name,
                        "video_path": str(output_path),
                        "source": "uploaded",
                        "started_at_epoch": time.time(),
                        "first_frame_at_epoch": None,
                        "ends_at_epoch": None,
                        "zone_ids": [],
                        "event_count": 0,
                        "prebuffer_seconds": 0.0,
                        "recording_seconds": 0.0,
                    },
                    indent=2,
                )
            )
            body = json.dumps({"ok": True, "message": f"Uploaded {output_path.name}"}).encode("utf-8")
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

    def _handle_video_quick_alpr(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            if self.video_recording is not None:
                self._send_recording_active_extraction_error(handler)
                return
            if not self.config.fast_alpr_url:
                raise ValueError("FAST_ALPR_URL is not configured")
            relative_path = self._read_simple_form_value(handler, "path")
            video_path = self._saved_video_path_from_relative(relative_path)
            candidates = self._extract_saved_video_candidates(video_path, 1)
            if not candidates:
                raise ValueError("No video frame could be extracted for ALPR")
            candidate = candidates[0]
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            event_name = f"{self.config.camera_name}_video_snapshot_{stamp}"
            event_dir = self.config.image_output_dir / event_name
            event_dir.mkdir(parents=True, exist_ok=True)
            image_path = event_dir / "video_snapshot.jpg"
            image_path.write_bytes(candidate.jpeg_bytes or self._encode_jpeg(candidate.frame))
            (event_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "camera_name": self.config.camera_name,
                        "started_at_epoch": time.time(),
                        "ended_at_epoch": time.time(),
                        "trigger_count": 0,
                        "saved_frames": 1,
                        "clip_path": self._relative_video_path(video_path),
                        "video_snapshot": True,
                        "video_position_seconds": candidate.timestamp,
                    },
                    indent=2,
                )
            )
            image_bytes = image_path.read_bytes()
            result = self._recognize_fast_alpr(image_bytes)
            result_path = image_path.with_name(f"{image_path.stem}.manual_fast_alpr.json")
            result_path.write_text(json.dumps(result, indent=2))
            relative = self._relative_image_path(image_path)
            detections = self._extract_fast_alpr_detections(result, relative, event_dir.name, time.time())
            plates_text = ", ".join(f"{d.plate} ({d.confidence:.2f})" for d in detections) or "No plates detected"
            body = json.dumps({"ok": True, "message": f"ALPR: {plates_text}"}).encode("utf-8")
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

    def _list_saved_images(self) -> List[Path]:
        images = sorted(
            self.config.image_output_dir.glob("*/*.jpg"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return images

    def _saved_image_epoch(self, image_path: Path) -> float:
        meta_path = image_path.with_name(f"{image_path.stem}.meta.json")
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text())
                detected_at_epoch = payload.get("detected_at_epoch")
                if detected_at_epoch is not None:
                    return float(detected_at_epoch)
            except Exception:
                pass

        summary_path = image_path.parent / "summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text())
                relative = self._relative_image_path(image_path)
                for item in payload.get("zone_images") or []:
                    if item.get("image_relative_path") == relative and item.get("source_frame_timestamp") is not None:
                        return float(item["source_frame_timestamp"])
                for key in ("started_at_epoch", "ended_at_epoch"):
                    if payload.get(key) is not None:
                        return float(payload[key])
            except Exception:
                pass

        return image_path.stat().st_mtime

    def _saved_image_time_text(self, image_path: Path) -> str:
        return self._format_system_time(self._saved_image_epoch(image_path))

    def _list_recent_plate_detections(self, limit: int) -> List[PlateDetection]:
        detections: List[PlateDetection] = []
        summaries = sorted(
            self.config.image_output_dir.glob("*/summary.json"),
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

    def _render_recent_plate_cards(self, limit: int, include_images: bool = True) -> str:
        detections = self._list_recent_plate_detections(limit)
        if not detections:
            return "<p>No plate detections yet.</p>"
        cards = []
        for detection in detections:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detection.detected_at_epoch))
            image_html = ""
            if include_images and detection.image_relative_path:
                image_path = self.config.image_output_dir / Path(detection.image_relative_path)
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

    def _render_image_cards(self, images: List[Path]) -> str:
        cards = []
        car_icon = (
            '<svg viewBox="0 0 24 24" aria-hidden="true">'
            '<path d="M5.5 16a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm13 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z"/>'
            '<path d="M5 14h14l-1.2-4.1a2 2 0 0 0-1.92-1.43H8.12A2 2 0 0 0 6.2 9.9L5 14Zm15 1a1 1 0 0 1 1 1v3h-2v-1H5v1H3v-3a1 1 0 0 1 1-1h16ZM7.12 9.34A3 3 0 0 1 10 7h5a3 3 0 0 1 2.88 2.34L19.78 16H4.22l2.9-6.66Z"/>'
            '</svg>'
        )
        zone_dot_colors = {"yellow": "#facc15", "purple": "#c084fc"}
        for image_path in images:
            relative = self._relative_image_path(image_path)
            detail_link = self._image_detail_url(relative)
            image_url = self._event_file_url(relative)
            summary_path = image_path.parent / "summary.json"
            policy_note = ""
            zone_badges_html = ""
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    policy = summary.get("event_policy") or "unknown"
                    policy_note = f"<p>Policy: {html.escape(str(policy))}</p>"
                    zone_ids = summary.get("zone_ids") or []
                    zone_badges_html = "".join(
                        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{zone_dot_colors.get(str(zone_id), "#94a3b8")};flex-shrink:0;" title="{html.escape(str(zone_id))}"></span>'
                        for zone_id in zone_ids
                    )
                except Exception:
                    policy_note = ""
                    zone_badges_html = ""
            stamp = self._saved_image_time_text(image_path)
            escaped_relative = html.escape(relative)
            escaped_name = html.escape(image_path.name)
            escaped_relative_attr = html.escape(relative, quote=True)
            zone_badges_row_html = zone_badges_html or '<span class="meta">No zone</span>'
            cards.append(
                f'<div class="card"><a href="{detail_link}"><img src="{image_url}" alt="{escaped_relative}"></a>'
                f'<div class="card-actions"><p class="card-filename"><a href="{detail_link}">{escaped_name}</a></p>'
                f'<form method="post" action="/api/images/alpr" class="card-icon-form">'
                f'<input type="hidden" name="path" value="{escaped_relative_attr}">'
                f'<button type="submit" class="card-icon-button" aria-label="Send image to ALPR" title="Send image to ALPR">{car_icon}</button></form></div>'
                f'<p>{stamp}</p>'
                f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:8px;">{zone_badges_row_html}</div>'
                f'{policy_note}</div>'
            )
        return "".join(cards) or "<p>No images saved yet.</p>"

    def _serve_image_detail_page(self, handler: BaseHTTPRequestHandler, relative_path: str) -> None:
        try:
            image_path = self._image_path_from_relative(relative_path)
        except ValueError:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not image_path.exists() or not image_path.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        relative = self._relative_image_path(image_path)
        images = [self._relative_image_path(path) for path in self._list_saved_images()]
        try:
            current_index = images.index(relative)
        except ValueError:
            current_index = -1
        newer_relative = images[current_index - 1] if current_index > 0 else None
        older_relative = images[current_index + 1] if 0 <= current_index < len(images) - 1 else None
        image_url = self._event_file_url(relative)
        stamp = self._saved_image_time_text(image_path)
        newer_link = self._image_detail_url(newer_relative) if newer_relative else ""
        older_link = self._image_detail_url(older_relative) if older_relative else ""
        back_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 19 4 12l7-7v4h9v6h-9v4Z"/></svg>'
        open_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3Z"/><path d="M5 5h6v2H7v10h10v-4h2v6H5V5Z"/></svg>'
        newer_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.5 5 8.5 12l7 7V5Z"/></svg>'
        older_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8.5 5 7 7-7 7V5Z"/></svg>'
        newer_html = (
            f'<a class="icon-button" href="{newer_link}" aria-label="Newer image" title="Newer image">{newer_icon}</a>'
            if newer_link
            else f'<span class="icon-button disabled" aria-label="No newer image" title="No newer image">{newer_icon}</span>'
        )
        older_html = (
            f'<a class="icon-button" href="{older_link}" aria-label="Older image" title="Older image">{older_icon}</a>'
            if older_link
            else f'<span class="icon-button disabled" aria-label="No older image" title="No older image">{older_icon}</span>'
        )
        escaped_relative_attr = html.escape(relative, quote=True)
        body = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Captured Image</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; }}
img {{ width: 100%; border-radius: 10px; display: block; max-width: 1100px; }}
.detail-actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 16px; color: #cbd5e1; }}
.detail-actions span {{ color: #64748b; }}
.icon-button {{ display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border: 1px solid #334155; border-radius: 8px; background: #111827; color: #93c5fd; text-decoration: none; }}
.icon-button svg {{ width: 22px; height: 22px; fill: currentColor; display: block; }}
.icon-button:hover {{ background: #1f2937; color: #bfdbfe; }}
.icon-button.disabled {{ opacity: 0.35; color: #94a3b8; pointer-events: none; }}
.alpr-action {{ margin-top: 12px; display: flex; align-items: center; gap: 10px; }}
.alpr-icon-button {{ display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 999px; border: 1px solid #334155; background: transparent; color: #cbd5e1; cursor: pointer; padding: 0; }}
	.alpr-icon-button svg {{ width: 22px; height: 22px; fill: currentColor; display: block; }}
	.alpr-icon-button:hover {{ border-color: #facc15; color: #facc15; background: #172033; }}
	.alpr-popup {{ position: fixed; inset: 0; background: rgba(2, 6, 23, 0.72); display: none; align-items: center; justify-content: center; z-index: 1000; padding: 24px; }}
	.alpr-popup.visible {{ display: flex; }}
	.alpr-popup-card {{ width: min(92vw, 420px); background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 22px; text-align: center; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35); }}
	.alpr-popup-card h2 {{ margin: 0; font-size: 1.1rem; }}
	.alpr-popup-card p {{ margin: 10px 0 0; color: #cbd5e1; }}
	.alpr-popup-close {{ margin-top: 14px; background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 9px 14px; cursor: pointer; font-weight: 700; }}
@media (max-width: 520px) {{
  .detail-actions {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); }}
  .detail-actions .icon-button {{ width: 100%; }}
  .alpr-action {{ align-items: center; }}
}}
	{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("images")}
<div class="detail-actions"><a class="icon-button" href="/images" aria-label="Back to captured images" title="Back to captured images">{back_icon}</a><a class="icon-button" href="{image_url}" aria-label="Open image file" title="Open image file">{open_icon}</a>{newer_html}{older_html}</div>
<h1>Captured image</h1>
<div class="panel"><p class="meta">System time: {html.escape(stamp)}</p><img src="{image_url}" alt="{html.escape(relative)}"><p>{html.escape(relative)}</p>
<form class="alpr-action" method="post" action="/api/images/alpr">
<input type="hidden" name="path" value="{escaped_relative_attr}">
<button type="submit" class="alpr-icon-button" aria-label="Send to ALPR" title="Send to ALPR"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5.5 16a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm13 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z"/><path d="M5 14h14l-1.2-4.1a2 2 0 0 0-1.92-1.43H8.12A2 2 0 0 0 6.2 9.9L5 14Zm15 1a1 1 0 0 1 1 1v3h-2v-1H5v1H3v-3a1 1 0 0 1 1-1h16ZM7.12 9.34A3 3 0 0 1 10 7h5a3 3 0 0 1 2.88 2.34L19.78 16H4.22l2.9-6.66Z"/></svg></button>
<span style="color:#cbd5e1;">Send to ALPR</span>
	</form></div>
	</div>
	<div class="alpr-popup" id="alpr-popup" role="dialog" aria-modal="true" aria-labelledby="alpr-popup-title">
	  <div class="alpr-popup-card">
	    <h2 id="alpr-popup-title">ALPR</h2>
	    <p id="alpr-popup-message">Sending...</p>
	    <button class="alpr-popup-close" id="alpr-popup-close" type="button">Close</button>
	  </div>
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
	(() => {{
	  const popup = document.getElementById('alpr-popup');
	  const title = document.getElementById('alpr-popup-title');
	  const message = document.getElementById('alpr-popup-message');
	  const close = document.getElementById('alpr-popup-close');
	  const form = document.querySelector('form.alpr-action[action="/api/images/alpr"]');
	  if (!popup || !title || !message || !close || !form) return;

	  function showPopup(heading, text) {{
	    title.textContent = heading;
	    message.textContent = text;
	    popup.classList.add('visible');
	  }}

	  close.addEventListener('click', () => popup.classList.remove('visible'));
	  popup.addEventListener('click', (event) => {{
	    if (event.target === popup) popup.classList.remove('visible');
	  }});

	  form.addEventListener('submit', async (event) => {{
	    event.preventDefault();
	    const button = form.querySelector('button');
	    if (button) button.disabled = true;
	    showPopup('ALPR', 'Sending image...');
	    try {{
	      const formData = new FormData(form);
	      const payload = {{}};
	      formData.forEach((value, key) => {{ payload[key] = value; }});
	      const response = await fetch(form.action, {{
	        method: 'POST',
	        headers: {{ 'Content-Type': 'application/json' }},
	        body: JSON.stringify(payload),
	      }});
	      const data = await response.json();
	      if (!response.ok) throw new Error(data.error || 'ALPR request failed');
	      showPopup('ALPR Result', data.message || 'ALPR request completed.');
	    }} catch (error) {{
	      showPopup('ALPR Failed', error.message || 'ALPR request failed.');
	    }} finally {{
	      if (button) button.disabled = false;
	    }}
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

    def _render_video_cards(
        self,
        videos: List[Path],
        selected_video: Optional[Path] = None,
        page: int = 1,
    ) -> str:
        if not videos:
            return "<p>No videos saved yet.</p>"
        play_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>'
        image_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2ZM8.5 8.5A1.5 1.5 0 1 1 7 7a1.5 1.5 0 0 1 1.5 1.5ZM5 19l4.5-6 3.5 4.5 2.5-3 3.5 4.5H5Z"/></svg>'
        car_icon = (
            '<svg viewBox="0 0 24 24" aria-hidden="true">'
            '<path d="M5.5 16a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm13 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z"/>'
            '<path d="M5 14h14l-1.2-4.1a2 2 0 0 0-1.92-1.43H8.12A2 2 0 0 0 6.2 9.9L5 14Zm15 1a1 1 0 0 1 1 1v3h-2v-1H5v1H3v-3a1 1 0 0 1 1-1h16ZM7.12 9.34A3 3 0 0 1 10 7h5a3 3 0 0 1 2.88 2.34L19.78 16H4.22l2.9-6.66Z"/>'
            '</svg>'
        )
        download_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a1 1 0 0 1 1 1v8.59l2.3-2.29 1.4 1.4-4.7 4.7-4.7-4.7 1.4-1.4L11 12.59V4a1 1 0 0 1 1-1Z"/><path d="M5 19h14v2H5z"/></svg>'
        remove_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l1 2h5v2H3V5h5l1-2Zm1 6h2v8h-2V9Zm4 0h2v8h-2V9ZM6 9h2v8H6V9Zm1 12a2 2 0 0 1-2-2V8h14v11a2 2 0 0 1-2 2H7Z"/></svg>'
        fullscreen_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h5v2H7v3H5V5Zm12 0h2v5h-2V7h-3V5h3ZM5 14h2v3h3v2H5v-5Zm12 3v-3h2v5h-5v-2h3Z"/></svg>'
        selected_relative = self._relative_video_path(selected_video) if selected_video else ""
        selected_player_html = "<p>Select a video from the list below.</p>"
        if selected_video is not None:
            selected_video_url = self._event_file_url(selected_relative)
            selected_player_html = (
                f'<div class="player-card"><div class="player-wrap"><video id="gallery-video-player" controls preload="none" src="{selected_video_url}"></video>'
                f'<button id="gallery-video-fullscreen" class="player-overlay-button" type="button" aria-label="Fullscreen" title="Fullscreen">{fullscreen_icon}</button></div>'
                f'<p class="player-path">{html.escape(selected_relative)}</p>'
                f'<div class="player-links"><a href="{self._video_detail_url(selected_relative)}">Detail</a>'
                f'<a href="{selected_video_url}">File</a></div></div>'
            )
        zone_dot_colors = {"yellow": "#facc15", "purple": "#c084fc"}
        rows = []
        for video_path in videos:
            relative = self._relative_video_path(video_path)
            metadata_path = video_path.with_suffix(".json")
            zone_ids: List[str] = []
            metadata = None
            started_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(video_path.stat().st_mtime))
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    zone_ids = metadata.get("zone_ids") or []
                    started_at = metadata.get("started_at_epoch")
                    if started_at:
                        started_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(started_at)))
                except Exception:
                    pass
            duration_text = self._format_duration_label(
                self._saved_video_duration_seconds(video_path, metadata, allow_probe=False)
            )
            zone_dots = "".join(
                f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{zone_dot_colors.get(z, "#94a3b8")};flex-shrink:0;" title="{html.escape(z)}"></span>'
                for z in zone_ids
            )
            zone_badges_html = zone_dots or '<span class="video-chip">No zone</span>'
            escaped_relative = html.escape(relative)
            item_class = "video-row active" if relative == selected_relative else "video-row"
            select_link = f'/videos?{urlencode({"page": str(page), "selected": relative})}'
            download_url = self._event_file_url(relative)
            rows.append(
                f'<div class="{item_class}"><div class="video-meta"><p class="video-name">{escaped_relative}</p>'
                f'<p class="video-subtle">{started_text}</p>'
                f'<div class="video-badges"><span class="video-chip">{html.escape(duration_text)}</span>'
                f'{zone_badges_html}'
                f'</div></div>'
                f'<div class="video-actions" style="align-items:center;">'
                f'<div class="video-action-slot"><button class="video-icon-button" data-action="play" data-video-url="{self._event_file_url(relative)}" data-video-name="{escaped_relative}" aria-label="Play video" title="Play video">{play_icon}</button><span class="video-action-status"></span></div>'
                f'<div class="video-action-slot"><a class="video-icon-button" href="{download_url}" download aria-label="Download video" title="Download video">{download_icon}</a><span class="video-action-status"></span></div>'
                f'<div class="video-action-slot"><form method="post" action="/api/videos/extract-images" class="video-icon-form" data-action-label="Extracting images" data-action-detail="{escaped_relative}">'
                f'<input type="hidden" name="path" value="{html.escape(relative, quote=True)}">'
                f'<button type="submit" class="video-icon-button" aria-label="Extract images" title="Extract images">{image_icon}</button></form>'
                f'<span class="video-action-status" aria-live="polite"></span></div>'
                f'<div class="video-action-slot"><form method="post" action="/api/videos/quick-alpr" class="video-icon-form" data-action-label="Sending frame to ALPR" data-action-detail="{escaped_relative}">'
                f'<input type="hidden" name="path" value="{html.escape(relative, quote=True)}">'
                f'<button type="submit" class="video-icon-button" aria-label="Send video frame to ALPR" title="Send video frame to ALPR">{car_icon}</button></form><span class="video-action-status"></span></div>'
                f'<div class="video-action-slot"><form method="post" action="/api/videos/delete" class="video-icon-form" data-action-label="Removing video" data-action-detail="{escaped_relative}" data-double-confirm="Remove video">'
                f'<input type="hidden" name="path" value="{html.escape(relative, quote=True)}">'
                f'<input type="hidden" name="page" value="{page}">'
                f'<button type="submit" class="video-icon-button danger" aria-label="Remove video" title="Remove video">{remove_icon}</button></form><span class="video-action-status"></span></div>'
                f'</div></div>'
            )
        player_script = """<script>
(() => {
  const button = document.getElementById('gallery-video-fullscreen');
  const video = document.getElementById('gallery-video-player');
  if (button && video) {
    button.addEventListener('click', async () => {
      const target = video.parentElement || video;
      try {
        if (document.fullscreenElement) {
          await document.exitFullscreen();
        } else if (target.requestFullscreen) {
          await target.requestFullscreen();
        }
      } catch (error) {
        console.warn('Fullscreen failed', error);
      }
    });
  }

  const overlay = document.getElementById('action-overlay');
  const title = document.getElementById('action-title');
  const message = document.getElementById('action-message');
  const spinner = document.getElementById('action-spinner');
  const close = document.getElementById('action-close');
  const forms = Array.from(document.querySelectorAll('.video-icon-form'));
  const playButtons = Array.from(document.querySelectorAll('[data-action="play"]'));

  function armDoubleConfirm(form, button) {
    if (!form || !button) return;
    button.classList.add('confirm-armed');
    form.dataset.confirmArmed = 'true';
  }

  function resetDoubleConfirm(form) {
    const button = form && form.querySelector('button[type="submit"], button:not([type])');
    if (!button) return;
    button.classList.remove('confirm-armed');
    delete form.dataset.confirmArmed;
  }

  function setOverlayState(heading, detail, finished = false) {
    title.textContent = heading;
    message.textContent = detail;
    overlay.classList.add('visible');
    spinner.style.display = finished ? 'none' : 'block';
    close.style.display = finished ? 'inline-flex' : 'none';
  }

  function hideOverlay() {
    overlay.classList.remove('visible');
    spinner.style.display = 'block';
    close.style.display = 'none';
  }

  function extractionErrorText(job) {
    const parts = [
      job.error_detail || job.error || 'Extraction failed.',
      `Video: ${job.video || 'unknown'}`,
      `Stage: ${job.stage || 'unknown'}`,
      `Saved: ${job.saved || 0}/${job.total || '?'}`,
    ];
    if (job.current_position_seconds !== undefined) parts.push(`Last position: ${Number(job.current_position_seconds).toFixed(3)}s`);
    if (job.duration_seconds !== undefined) parts.push(`Duration: ${Number(job.duration_seconds).toFixed(3)}s`);
    return parts.filter(Boolean).join('\\n');
  }

  function extractionProgressText(job) {
    const total = job.total || '?';
    const attempt = job.attempt_index && job.attempt_total ? ` (attempt ${job.attempt_index}/${job.attempt_total})` : '';
    return `Saved ${job.saved || 0} of ${total} images from ${job.video}${attempt}`;
  }

  if (close) close.addEventListener('click', hideOverlay);

  playButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const videoUrl = btn.dataset.videoUrl;
      const videoName = btn.dataset.videoName;
      const playerStack = document.querySelector('.player-stack');
      if (!playerStack || !videoUrl) return;
      const fullscreenIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h5v2H7v3H5V5Zm12 0h2v5h-2V7h-3V5h3ZM5 14h2v3h3v2H5v-5Zm12 3v-3h2v5h-5v-2h3Z"/></svg>';
      playerStack.innerHTML = `<div class="player-card"><div class="player-wrap"><video id="gallery-video-player" controls preload="metadata" src="${videoUrl}"></video><button id="gallery-video-fullscreen" class="player-overlay-button" type="button" aria-label="Fullscreen" title="Fullscreen">${fullscreenIcon}</button></div><p class="player-path">${videoName}</p></div>`;
      const newVideo = document.getElementById('gallery-video-player');
      if (newVideo) newVideo.play().catch(() => {});
      const fsBtn = document.getElementById('gallery-video-fullscreen');
      if (fsBtn && newVideo) {
        fsBtn.addEventListener('click', async () => {
          const target = newVideo.parentElement || newVideo;
          try {
            if (document.fullscreenElement) await document.exitFullscreen();
            else if (target.requestFullscreen) await target.requestFullscreen();
          } catch (e) {}
        });
      }
      document.querySelectorAll('.video-row').forEach((row) => row.classList.remove('active'));
      btn.closest('.video-row')?.classList.add('active');
    });
  });

  forms.forEach((form) => {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const submitter = form.querySelector('button');
      const statusEl = form.action.includes('/api/videos/extract-images') && form.nextElementSibling && form.nextElementSibling.classList.contains('video-action-status')
        ? form.nextElementSibling
        : null;
      if (form.dataset.doubleConfirm && form.dataset.confirmArmed !== 'true') {
        if (window.__resetDeleteConfirmButtons) window.__resetDeleteConfirmButtons(form);
        else forms.forEach((otherForm) => {
          if (otherForm !== form) resetDoubleConfirm(otherForm);
        });
        armDoubleConfirm(form, submitter);
        return;
      }
      const label = form.dataset.actionLabel || 'Working';
      const detail = form.dataset.actionDetail || 'Please wait while the request completes.';
      if (submitter) submitter.disabled = true;
      if (submitter && form.action.includes('/api/videos/extract-images')) submitter.classList.add('busy');
      if (statusEl) statusEl.textContent = '';
      if (overlay) setOverlayState(label + '...', detail);
      try {
        const formData = new FormData(form);
        const payload = {};
        formData.forEach((value, key) => { payload[key] = value; });
        const response = await fetch(form.action, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Action failed');
        if (data.deleted) {
          form.closest('.video-row')?.remove();
          hideOverlay();
        } else if (data.job_id) {
          if (overlay) setOverlayState('Extracting images…', 'Starting…', false);
          const jobId = data.job_id;
          if (window.__startExtractionIndicator) window.__startExtractionIndicator(jobId);
          const poll = setInterval(async () => {
            try {
              const r = await fetch('/api/extraction-status', { cache: 'no-store' });
              const status = await r.json();
              const job = status[jobId];
              if (!job) return;
              if (job.error) {
                clearInterval(poll);
                if (submitter) submitter.classList.remove('busy');
                if (submitter) submitter.disabled = false;
                const errorText = extractionErrorText(job);
                if (statusEl) {
                  statusEl.textContent = 'Failed';
                  statusEl.title = errorText;
                }
                if (overlay) setOverlayState('Extraction failed', errorText, true);
              } else if (job.done) {
                clearInterval(poll);
                if (submitter) submitter.classList.remove('busy');
                if (submitter) submitter.disabled = false;
                if (statusEl) statusEl.textContent = job.stage === 'cancelled' ? 'Stopped' : (job.stage === 'partial' ? `${job.saved}/${job.total}` : `${job.saved}`);
	                if (overlay) {
	                  if (job.stage === 'cancelled') {
	                    setOverlayState('Stopped', `Saved ${job.saved} images from ${job.video} before stopping.`, true);
	                  } else if (job.stage === 'partial') {
	                    setOverlayState('Partial extraction', job.warning || `Saved ${job.saved} of ${job.total} requested images from ${job.video}.`, true);
	                  } else {
	                    setOverlayState('Done', `Saved ${job.saved} images from ${job.video}. Open the Images page to review them.`, true);
	                  }
	                }
	              } else {
	                if (overlay) setOverlayState('Extracting images…', extractionProgressText(job), false);
	              }
            } catch (e) {}
          }, 500);
        } else {
          if (overlay) setOverlayState('Done', data.message || 'Action completed.', true);
        }
      } catch (error) {
        if (statusEl) statusEl.textContent = 'Err';
        if (overlay) setOverlayState('Action failed', error.message || 'The request did not finish successfully.', true);
      } finally {
        resetDoubleConfirm(form);
        if (submitter && !form.action.includes('/api/videos/extract-images')) submitter.disabled = false;
        if (submitter && !form.action.includes('/api/videos/extract-images')) submitter.classList.remove('busy');
      }
    });
  });
})();
</script>"""
        return f'<div class="video-layout"><div class="player-stack">{selected_player_html}</div><div class="card"><span class="eyebrow">Clips</span><div class="video-list">{"".join(rows)}</div></div></div>' + player_script

    def _serve_video_detail_page(self, handler: BaseHTTPRequestHandler, relative_path: str) -> None:
        try:
            video_path = self._video_path_from_relative(relative_path)
        except ValueError:
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not video_path.exists() or not video_path.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if video_path.suffix.lower() != ".mp4":
            handler.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        relative = self._relative_video_path(video_path)
        video_url = self._event_file_url(relative)
        body = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Saved Video</title>
<style>
body {{ font-family: Arial, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
a {{ color: #93c5fd; }}
.panel {{ background: #111827; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; }}
video {{ width: 100%; border-radius: 10px; display: block; background: #000; }}
.detail-actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 16px; }}
.video-tools {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 12px; }}
.video-tools button {{ background: #facc15; color: #111827; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 700; }}
.video-tools span {{ color: #cbd5e1; }}
.video-tools span.error {{ color: #fca5a5; }}
{self._render_shared_styles()}
</style></head>
<body>
<div class="page-shell">
{self._render_nav("videos")}
<div class="detail-actions"><a href="/videos">Back to saved videos</a><a href="{video_url}">Open video file</a></div>
<h1>Saved video</h1>
<div class="panel">
<video id="saved-video" controls autoplay src="{video_url}"></video>
<div class="video-tools">
  <button id="snapshot-alpr" type="button">Snapshot and Send to ALPR</button>
  <span id="snapshot-status">Pause or seek, then send the visible frame.</span>
</div>
<p>{html.escape(relative)}</p>
</div>
</div>
<script>
(() => {{
  const video = document.getElementById('saved-video');
  const button = document.getElementById('snapshot-alpr');
  const status = document.getElementById('snapshot-status');
  const videoPath = {json.dumps(relative)};
  if (!video || !button || !status) return;

  function setStatus(text, isError = false) {{
    status.textContent = text;
    status.className = isError ? 'error' : '';
  }}

  button.addEventListener('click', async () => {{
    if (!video.videoWidth || !video.videoHeight) {{
      setStatus('Video frame is not ready yet.', true);
      return;
    }}
    button.disabled = true;
    setStatus('Creating snapshot...');
    try {{
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.95));
      if (!blob) throw new Error('Could not create snapshot');
      const form = new FormData();
      form.append('image', blob, 'video_snapshot.jpg');
      form.append('video_path', videoPath);
      form.append('position_seconds', String(video.currentTime || 0));
      setStatus('Sending snapshot to ALPR...');
      const response = await fetch('/api/videos/snapshot-alpr', {{ method: 'POST', body: form }});
      const text = await response.text();
      if (!response.ok) throw new Error(text || 'Snapshot ALPR failed');
      document.open();
      document.write(text);
      document.close();
    }} catch (error) {{
      setStatus(error.message || 'Snapshot ALPR failed', true);
    }} finally {{
      button.disabled = false;
    }}
  }});
}})();
</script>
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
            for event_dir in self.config.image_output_dir.iterdir():
                if event_dir.is_dir():
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

    def _handle_delete_video(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            form = self._read_simple_form(handler)
            relative_path = str(form.get("path") or "")
            page = max(1, int(form.get("page") or "1"))
            video_path = self._saved_video_path_from_relative(relative_path)
            metadata_path = video_path.with_suffix(".json")
            video_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            body = json.dumps({"ok": True, "message": "Video deleted", "deleted": relative_path}).encode("utf-8")
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

    def _handle_clear_plates(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            cleared = 0
            for summary_path in self.config.image_output_dir.glob("*/summary.json"):
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
    config = Config.from_env()
    log_dir = Path(os.getenv("LOG_OUTPUT_DIR", str(config.event_output_dir.parent / "logs"))).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.Formatter.converter = time.localtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z %z",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "watcher.log"),
        ],
        force=True,
    )
    watcher = RtspVehicleWatcher(config)
    watcher.run()


if __name__ == "__main__":
    main()
