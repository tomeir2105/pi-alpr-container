import logging
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import requests

from alpr_models import Config, VIDEO_WRITER_QUEUE_SECONDS


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


class DirectRtspVideoRecorder:
    def __init__(self, source_url: str, output_path: Path, transport: str, duration_seconds: float) -> None:
        normalized_source_url = source_url.strip()
        lower_source_url = normalized_source_url.lower()
        if not lower_source_url.startswith("rtsp://"):
            raise ValueError(f"Direct RTSP recorder requires an rtsp:// source URL, got: {source_url}")
        if ".m3u8" in lower_source_url:
            raise ValueError(f"Direct RTSP recorder cannot record from an HLS playlist source: {source_url}")
        self.source_url = normalized_source_url
        self.output_path = output_path
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-rtsp_transport",
                transport,
                "-i",
                self.source_url,
                "-an",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self.process.poll() is None

    def release(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin:
                try:
                    self.process.stdin.write(b"q\n")
                    self.process.stdin.flush()
                    self.process.stdin.close()
                except Exception:
                    self.process.terminate()
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        stderr = self.process.stderr.read() if self.process.stderr else b""
        if self.process.returncode not in {0, 255, -15}:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to record {self.output_path}: {message}")


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
