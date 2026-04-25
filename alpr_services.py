import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import requests

from alpr_models import Config, VIDEO_WRITER_QUEUE_SECONDS


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
