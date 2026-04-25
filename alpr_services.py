import logging
import os
import queue
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import requests

from alpr_models import Config, VIDEO_WRITER_QUEUE_SECONDS


HIKVISION_ALERT_STREAM_PATH = "/ISAPI/Event/notification/alertStream"


def _strip_xml_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_child_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if _strip_xml_namespace(element.tag) == name and element.text:
            return element.text.strip()
    return ""


def _xml_first_element(root: ET.Element, name: str) -> Optional[ET.Element]:
    for element in root.iter():
        if _strip_xml_namespace(element.tag) == name:
            return element
    return None


def _xml_child_element(root: ET.Element, name: str) -> Optional[ET.Element]:
    for element in list(root):
        if _strip_xml_namespace(element.tag) == name:
            return element
    return None


def _xml_namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def _xml_tag_like(parent: ET.Element, local_name: str) -> str:
    namespace = _xml_namespace(parent.tag)
    return f"{{{namespace}}}{local_name}" if namespace else local_name


def _xml_ensure_child(parent: ET.Element, local_name: str) -> ET.Element:
    existing = _xml_child_element(parent, local_name)
    if existing is not None:
        return existing
    return ET.SubElement(parent, _xml_tag_like(parent, local_name))


def _hikvision_xml_payloads(chunks: Iterator[bytes]) -> Iterator[bytes]:
    buffer = b""
    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        while True:
            start_match = re.search(rb"<[A-Za-z0-9_:.-]*EventNotificationAlert\b", buffer)
            if not start_match:
                buffer = buffer[-2048:]
                break
            start = start_match.start()
            end_match = re.search(rb"</[A-Za-z0-9_:.-]*EventNotificationAlert>", buffer[start:])
            if not end_match:
                if start > 0:
                    buffer = buffer[start:]
                break
            end = start + end_match.end()
            yield buffer[start:end]
            buffer = buffer[end:]


class HikvisionMotionEventStream:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.active = False
        self.last_event_at = 0.0
        self.last_summary = "Hikvision motion API: not connected"
        self.error_count = 0

    def configured(self) -> bool:
        return bool(self.config.hikvision_host and self.config.hikvision_user and self.config.hikvision_password)

    def start(self) -> None:
        if self.thread is not None or not self.config.use_camera_motion_api or not self.configured():
            return
        self.session = requests.Session()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="hikvision-motion-events", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.thread = None
        self.session.close()

    def snapshot(self) -> Tuple[bool, float, str]:
        with self.lock:
            return self.active, self.last_event_at, self.last_summary

    def _base_url(self) -> str:
        host = self.config.hikvision_host
        if self.config.hikvision_port and ":" not in host:
            host = f"{host}:{self.config.hikvision_port}"
        return f"{self.config.hikvision_scheme}://{host}"

    def _request_stream(self) -> requests.Response:
        url = f"{self._base_url()}{HIKVISION_ALERT_STREAM_PATH}"
        response = self.session.get(
            url,
            auth=requests.auth.HTTPDigestAuth(self.config.hikvision_user, self.config.hikvision_password),
            stream=True,
            timeout=(10, self.config.hikvision_event_timeout_seconds),
            headers={"Accept": "multipart/x-mixed-replace, application/xml"},
        )
        if response.status_code == 401:
            response.close()
            response = self.session.get(
                url,
                auth=requests.auth.HTTPBasicAuth(self.config.hikvision_user, self.config.hikvision_password),
                stream=True,
                timeout=(10, self.config.hikvision_event_timeout_seconds),
                headers={"Accept": "multipart/x-mixed-replace, application/xml"},
            )
        response.raise_for_status()
        return response

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self._base_url()}{path}"
        session = requests.Session()
        response = session.request(
            method,
            url,
            auth=requests.auth.HTTPDigestAuth(self.config.hikvision_user, self.config.hikvision_password),
            **kwargs,
        )
        if response.status_code == 401:
            response.close()
            response = session.request(
                method,
                url,
                auth=requests.auth.HTTPBasicAuth(self.config.hikvision_user, self.config.hikvision_password),
                **kwargs,
            )
        response.raise_for_status()
        return response

    def apply_motion_settings(self, zones: List[Any], sensitivity: int) -> None:
        if not self.configured():
            raise RuntimeError("Hikvision API is not configured")
        enabled_zones = [zone for zone in zones if getattr(zone, "enabled", False)]
        if not enabled_zones:
            raise RuntimeError("At least one motion zone must be enabled before applying Hikvision motion settings")
        path = f"/ISAPI/System/Video/inputs/channels/{self.config.hikvision_channel}/motionDetection"
        response = self._request("GET", path, timeout=10)
        root = ET.fromstring(response.content)
        if not (self._configure_grid_motion(root, enabled_zones, sensitivity) or self._configure_region_motion(root, enabled_zones, sensitivity)):
            raise RuntimeError("Unsupported Hikvision motion XML: no grid or region layout found")
        payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        self._request(
            "PUT",
            path,
            data=payload,
            timeout=10,
            headers={"Content-Type": "application/xml"},
        )

    def _configure_grid_motion(self, root: ET.Element, zones: List[Any], sensitivity: int) -> bool:
        grid = _xml_first_element(root, "Grid")
        layout = _xml_first_element(root, "MotionDetectionLayout")
        if grid is None and layout is None:
            return False
        if grid is None:
            grid = ET.SubElement(root, _xml_tag_like(root, "Grid"))
        rows = int(_xml_child_text(grid, "rowGranularity") or os.getenv("HIKVISION_GRID_ROWS", "18"))
        columns = int(_xml_child_text(grid, "columnGranularity") or os.getenv("HIKVISION_GRID_COLUMNS", "22"))
        _xml_ensure_child(grid, "rowGranularity").text = str(rows)
        _xml_ensure_child(grid, "columnGranularity").text = str(columns)
        if layout is None:
            layout = ET.SubElement(root, _xml_tag_like(root, "MotionDetectionLayout"))
        _xml_ensure_child(layout, "sensitivityLevel").text = str(sensitivity)
        layout_container = _xml_child_element(layout, "layout") or layout
        if layout_container is layout and _xml_child_element(layout, "layout") is None:
            layout_container = ET.SubElement(layout, _xml_tag_like(layout, "layout"))
        _xml_ensure_child(layout_container, "gridMap").text = self._zone_grid_map(zones, rows, columns)
        _xml_ensure_child(root, "enabled").text = "true"
        _xml_ensure_child(root, "regionType").text = "grid"
        return True

    def _configure_region_motion(self, root: ET.Element, zones: List[Any], sensitivity: int) -> bool:
        region_list = _xml_first_element(root, "MotionDetectionRegionList")
        if region_list is None:
            return False
        for child in list(region_list):
            region_list.remove(child)
        for index, zone in enumerate(zones, start=1):
            x1, y1, x2, y2 = zone.roi
            region = ET.SubElement(region_list, _xml_tag_like(region_list, "MotionDetectionRegion"))
            ET.SubElement(region, _xml_tag_like(region, "id")).text = str(index)
            ET.SubElement(region, _xml_tag_like(region, "enabled")).text = "true"
            ET.SubElement(region, _xml_tag_like(region, "sensitivityLevel")).text = str(sensitivity)
            coords = ET.SubElement(region, _xml_tag_like(region, "RegionCoordinatesList"))
            for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                coord = ET.SubElement(coords, _xml_tag_like(coords, "RegionCoordinates"))
                ET.SubElement(coord, _xml_tag_like(coord, "positionX")).text = str(int(round(x * 1000)))
                ET.SubElement(coord, _xml_tag_like(coord, "positionY")).text = str(int(round(y * 1000)))
        _xml_ensure_child(root, "enabled").text = "true"
        return True

    def _zone_grid_map(self, zones: List[Any], rows: int, columns: int) -> str:
        encoded_rows: List[str] = []
        for row in range(rows):
            cell_y1 = row / rows
            cell_y2 = (row + 1) / rows
            bits: List[str] = []
            for column in range(columns):
                cell_x1 = column / columns
                cell_x2 = (column + 1) / columns
                enabled = any(
                    cell_x2 > zone.roi[0] and cell_x1 < zone.roi[2] and cell_y2 > zone.roi[1] and cell_y1 < zone.roi[3]
                    for zone in zones
                )
                bits.append("1" if enabled else "0")
            while len(bits) % 4:
                bits.append("0")
            encoded_rows.append(f"{int(''.join(bits), 2):0{len(bits) // 4}x}")
        return "".join(encoded_rows)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            response: Optional[requests.Response] = None
            try:
                with self.lock:
                    self.last_summary = f"Hikvision motion API: connecting to {self._base_url()}"
                response = self._request_stream()
                with self.lock:
                    self.last_summary = "Hikvision motion API: connected; waiting for VMD events"
                for payload in _hikvision_xml_payloads(response.iter_content(chunk_size=4096)):
                    if self.stop_event.is_set():
                        return
                    self._handle_payload(payload)
            except Exception as exc:
                with self.lock:
                    self.error_count += 1
                    self.active = False
                    self.last_summary = f"Hikvision motion API error: {exc}"
                logging.warning("Hikvision motion event stream failed: %s", exc)
                self.stop_event.wait(5.0)
            finally:
                if response is not None:
                    response.close()

    def _handle_payload(self, payload: bytes) -> None:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return
        event_type = (_xml_child_text(root, "eventType") or "").lower()
        event_state = (_xml_child_text(root, "eventState") or "").lower()
        channel = _xml_child_text(root, "channelID") or _xml_child_text(root, "dynChannelID")
        if channel and channel.isdigit() and int(channel) != self.config.hikvision_channel:
            return
        if "vmd" not in event_type and "motion" not in event_type:
            return
        active = event_state in {"active", "start", "true"} or not event_state
        now = time.time()
        summary = f"Hikvision motion API: type={event_type or 'motion'} state={event_state or 'active'} channel={channel or self.config.hikvision_channel}"
        with self.lock:
            self.active = active
            self.last_event_at = now
            self.last_summary = summary


class SingleFfmpegRtspCapture:
    STOP_TIMEOUT_SECONDS = 5
    CONCAT_TIMEOUT_SECONDS = 45

    def __init__(self, config: Config, segment_dir: Path, transport: str) -> None:
        self.config = config
        self.segment_dir = segment_dir
        self.transport = transport
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.segment_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self._cleanup_old_segments(force=True)
        segment_pattern = str(self.segment_dir / "seg-%Y%m%dT%H%M%S.mp4")
        fps = max(1.0, self.config.stream_fps)
        max_width = max(320, int(self.config.frame_width or 960))
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-rtsp_transport",
            self.transport,
            "-analyzeduration",
            self.config.ffmpeg_analyze_duration,
            "-probesize",
            self.config.ffmpeg_probe_size,
            "-fflags",
            "+discardcorrupt",
            "-i",
            self.config.rtsp_url,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            f"{self.config.record_segment_seconds:.3f}",
            "-segment_format",
            "mp4",
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            segment_pattern,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"fps={fps:.3f},scale={max_width}:-2:force_original_aspect_ratio=decrease",
            "-q:v",
            "5",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "AV_LOG_FORCE_NOCOLOR": "1"},
        )
        if self.process.stdout is None:
            self.stop()
            raise RuntimeError("Unable to open ffmpeg RTSP capture")

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.process.stdout is not None

    def read_frame(self, timeout_seconds: float = 5.0):
        if not self.is_running() or self.process is None or self.process.stdout is None:
            raise RuntimeError("ffmpeg RTSP capture is not running")
        deadline = time.time() + timeout_seconds
        buffer = b""
        while time.time() < deadline:
            chunk = self.process.stdout.read(65536)
            if not chunk:
                if self.process.poll() is not None:
                    raise RuntimeError("ffmpeg RTSP capture exited")
                time.sleep(0.05)
                continue
            buffer += chunk
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                buffer = buffer[-2:]
                continue
            end = buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start > 0:
                    buffer = buffer[start:]
                continue
            jpeg_bytes = buffer[start : end + 2]
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                self._cleanup_old_segments()
                return frame
        raise RuntimeError("Timed out waiting for ffmpeg frame")

    def copy_clip(self, start_at: float, end_at: float, output_path: Path) -> List[Path]:
        time.sleep(min(5.0, max(1.0, self.config.record_segment_seconds + 0.5)))
        segments = self._segments_for_range(start_at, end_at)
        if not segments:
            raise RuntimeError("No stream-copy segments are available for event clip")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        concat_path = output_path.with_suffix(".concat.txt")
        concat_path.write_text(
            "".join(f"file '{segment.as_posix()}'\n" for segment in segments),
            encoding="utf-8",
        )
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_path),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.CONCAT_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                message = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg concat failed for {output_path}: {message}")
            return segments
        finally:
            concat_path.unlink(missing_ok=True)

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=self.STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self.STOP_TIMEOUT_SECONDS)

    def _segments_for_range(self, start_at: float, end_at: float) -> List[Path]:
        segment_slack = max(1.0, self.config.record_segment_seconds * 2.0)
        selected: List[Path] = []
        for path in sorted(self.segment_dir.glob("seg-*.mp4")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= start_at - segment_slack and mtime <= end_at + segment_slack:
                selected.append(path)
        return selected

    def _cleanup_old_segments(self, force: bool = False) -> None:
        cutoff = 0.0 if force else time.time() - self.config.record_segment_retention_seconds
        for path in self.segment_dir.glob("seg-*.mp4"):
            try:
                if force or path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                logging.exception("Failed to remove old recording segment %s", path)


class FfmpegVideoWriter:
    FINALIZE_TIMEOUT_SECONDS = 30
    KILL_TIMEOUT_SECONDS = 10

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
        try:
            _, stderr = self.process.communicate(timeout=self.FINALIZE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            _, stderr = self.process.communicate(timeout=self.KILL_TIMEOUT_SECONDS)
            message = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            detail = f": {message}" if message else ""
            raise RuntimeError(f"ffmpeg timed out finalizing {self.output_path}{detail}") from exc
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
        while True:
            try:
                self.queue.put_nowait(frame_copy)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    pass
                with self.lock:
                    self.dropped_frames += 1

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
