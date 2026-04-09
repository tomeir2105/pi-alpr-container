import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple

import cv2
import requests
from dotenv import load_dotenv


def parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    rtsp_url: str
    secret_key: str
    country: str
    frame_width: int
    process_every_n_frames: int
    min_motion_area: int
    min_consecutive_hits: int
    event_idle_seconds: float
    prebuffer_seconds: float
    postbuffer_seconds: float
    upload_top_frames: int
    upload_min_sharpness: float
    event_output_dir: Path
    camera_name: str
    roi: Optional[Tuple[float, float, float, float]]
    recognize_vehicle: bool
    debug_windows: bool
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        rtsp_url = os.getenv("RTSP_URL", "").strip()
        secret_key = os.getenv("OPENALPR_SECRET_KEY", "").strip()
        if not rtsp_url:
            raise ValueError("RTSP_URL is required")
        if not secret_key:
            raise ValueError("OPENALPR_SECRET_KEY is required")

        roi_value = os.getenv("ROI", "").strip()
        roi = None
        if roi_value:
            parts = [float(part.strip()) for part in roi_value.split(",")]
            if len(parts) != 4:
                raise ValueError("ROI must contain 4 comma-separated normalized values")
            roi = tuple(parts)  # type: ignore[assignment]

        return cls(
            rtsp_url=rtsp_url,
            secret_key=secret_key,
            country=os.getenv("OPENALPR_COUNTRY", "us").strip(),
            frame_width=int(os.getenv("FRAME_WIDTH", "960")),
            process_every_n_frames=max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))),
            min_motion_area=int(os.getenv("MIN_MOTION_AREA", "6500")),
            min_consecutive_hits=max(1, int(os.getenv("MIN_CONSECUTIVE_HITS", "3"))),
            event_idle_seconds=float(os.getenv("EVENT_IDLE_SECONDS", "1.5")),
            prebuffer_seconds=float(os.getenv("PREBUFFER_SECONDS", "2.0")),
            postbuffer_seconds=float(os.getenv("POSTBUFFER_SECONDS", "1.5")),
            upload_top_frames=max(1, int(os.getenv("UPLOAD_TOP_FRAMES", "3"))),
            upload_min_sharpness=float(os.getenv("UPLOAD_MIN_SHARPNESS", "80.0")),
            event_output_dir=Path(os.getenv("EVENT_OUTPUT_DIR", "./events")).expanduser(),
            camera_name=os.getenv("CAMERA_NAME", "camera").strip(),
            roi=roi,
            recognize_vehicle=parse_bool(os.getenv("RECOGNIZE_VEHICLE", "true"), default=True),
            debug_windows=parse_bool(os.getenv("DEBUG_WINDOWS", "false"), default=False),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20")),
        )


@dataclass
class CandidateFrame:
    frame: Any
    timestamp: float
    motion_area: int
    sharpness: float


@dataclass
class Event:
    started_at: float
    trigger_count: int
    frames: List[Tuple[float, Any]]
    candidates: List[CandidateFrame]
    last_motion_at: float
    last_frame_at: float


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
        self.client = OpenAlprClient(config)
        self.prebuffer: Deque[Tuple[float, Any]] = deque()
        self.event: Optional[Event] = None
        self.frame_index = 0
        self.fps_guess = 12.0
        self.capture = None
        self.background = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=36, detectShadows=False)
        self.config.event_output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        while True:
            try:
                self._run_capture_loop()
            except KeyboardInterrupt:
                logging.info("Stopping watcher")
                break
            except Exception:
                logging.exception("Capture loop failed; reconnecting in 5 seconds")
                time.sleep(5)

    def _run_capture_loop(self) -> None:
        logging.info("Connecting to RTSP stream")
        self.capture = cv2.VideoCapture(self.config.rtsp_url)
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open RTSP stream")

        native_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if native_fps and native_fps > 1:
            self.fps_guess = native_fps
        logging.info("Connected. Camera FPS estimate: %.2f", self.fps_guess)

        try:
            while True:
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Failed to read frame from RTSP stream")

                timestamp = time.time()
                frame = self._resize_frame(frame)
                self._push_prebuffer(timestamp, frame)

                process_this_frame = self.frame_index % self.config.process_every_n_frames == 0
                motion_area = 0
                had_motion = False

                if process_this_frame:
                    motion_area, had_motion, overlay = self._detect_motion(frame)
                    if self.config.debug_windows:
                        cv2.imshow("watcher", overlay)
                        cv2.waitKey(1)

                if had_motion:
                    self._on_motion(timestamp, frame, motion_area)
                elif self.event:
                    self._append_event_frame(timestamp, frame)
                    if timestamp - self.event.last_motion_at >= self.config.event_idle_seconds + self.config.postbuffer_seconds:
                        self._finalize_event()

                self.frame_index += 1
        finally:
            self.capture.release()

    def _resize_frame(self, frame):
        height, width = frame.shape[:2]
        if width <= self.config.frame_width:
            return frame
        scale = self.config.frame_width / float(width)
        resized = cv2.resize(frame, (self.config.frame_width, int(height * scale)))
        return resized

    def _push_prebuffer(self, timestamp: float, frame) -> None:
        self.prebuffer.append((timestamp, frame.copy()))
        max_frames = max(1, int(self.fps_guess * self.config.prebuffer_seconds))
        while len(self.prebuffer) > max_frames:
            self.prebuffer.popleft()

    def _roi_bounds(self, frame) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        if not self.config.roi:
            return 0, 0, width, height
        x1, y1, x2, y2 = self.config.roi
        return (
            max(0, min(width - 1, int(x1 * width))),
            max(0, min(height - 1, int(y1 * height))),
            max(1, min(width, int(x2 * width))),
            max(1, min(height, int(y2 * height))),
        )

    def _detect_motion(self, frame) -> Tuple[int, bool, Any]:
        x1, y1, x2, y2 = self._roi_bounds(frame)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.background.apply(gray)
        _, mask = cv2.threshold(mask, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0
        best_box = None
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.config.min_motion_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < h * 0.9:
                continue
            total_area += area
            if best_box is None or area > best_box[0]:
                best_box = (area, x, y, w, h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
        if best_box:
            _, x, y, w, h = best_box
            cv2.rectangle(overlay, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)

        had_motion = total_area >= self.config.min_motion_area
        label = f"motion_area={total_area}"
        cv2.putText(overlay, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
        return total_area, had_motion, overlay

    def _on_motion(self, timestamp: float, frame, motion_area: int) -> None:
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
            )
            logging.info("Motion started; opening event")
            self._append_event_frame(timestamp, frame)
            return

        self.event.trigger_count += 1
        self.event.last_motion_at = timestamp
        self.event.candidates.append(candidate)
        self._append_event_frame(timestamp, frame)

    def _append_event_frame(self, timestamp: float, frame) -> None:
        if not self.event:
            return
        self.event.frames.append((timestamp, frame.copy()))
        self.event.last_frame_at = timestamp

    def _finalize_event(self) -> None:
        if not self.event:
            return
        event = self.event
        self.event = None

        if event.trigger_count < self.config.min_consecutive_hits:
            logging.info("Discarded event with only %s motion hits", event.trigger_count)
            return

        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        event_dir = self.config.event_output_dir / f"{self.config.camera_name}_{stamp}"
        event_dir.mkdir(parents=True, exist_ok=True)
        clip_path = event_dir / "event.mp4"
        self._write_clip(event.frames, clip_path)

        selected = self._select_best_frames(event)
        results = []
        for index, candidate in enumerate(selected, start=1):
            frame_path = event_dir / f"frame_{index:02d}.jpg"
            json_path = event_dir / f"frame_{index:02d}.json"
            jpeg_bytes = self._encode_jpeg(candidate.frame)
            frame_path.write_bytes(jpeg_bytes)
            try:
                result = self.client.recognize(jpeg_bytes)
                json_path.write_text(json.dumps(result, indent=2))
                results.append(result)
                logging.info("Uploaded frame %s for ALPR analysis", index)
            except Exception:
                logging.exception("Failed to upload frame %s", index)

        summary = {
            "camera_name": self.config.camera_name,
            "started_at_epoch": event.started_at,
            "ended_at_epoch": event.last_frame_at,
            "trigger_count": event.trigger_count,
            "saved_frames": len(selected),
            "clip_path": str(clip_path),
            "alpr_results_count": len(results),
        }
        (event_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logging.info("Event saved to %s", event_dir)

    def _select_best_frames(self, event: Event) -> List[CandidateFrame]:
        candidates = [
            candidate for candidate in event.candidates if candidate.sharpness >= self.config.upload_min_sharpness
        ]
        if not candidates:
            candidates = list(event.candidates)

        ranked = sorted(
            candidates,
            key=lambda item: (item.motion_area, item.sharpness, item.timestamp),
            reverse=True,
        )

        selected: List[CandidateFrame] = []
        min_spacing = 0.25
        for candidate in ranked:
            if any(abs(candidate.timestamp - chosen.timestamp) < min_spacing for chosen in selected):
                continue
            selected.append(candidate)
            if len(selected) >= self.config.upload_top_frames:
                break
        return selected

    def _write_clip(self, frames: List[Tuple[float, Any]], path: Path) -> None:
        if not frames:
            return
        sample_frame = frames[0][1]
        height, width = sample_frame.shape[:2]
        fps = max(5.0, min(20.0, self.fps_guess))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            for _, frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def _compute_sharpness(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _encode_jpeg(self, frame) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Could not encode JPEG")
        return encoded.tobytes()


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
