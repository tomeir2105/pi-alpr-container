#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

try:
    from dotenv import load_dotenv as dotenv_load
except ModuleNotFoundError:
    dotenv_load = None


ALERT_STREAM_PATH = "/ISAPI/Event/notification/alertStream"
MOTION_CONFIG_PATH = "/ISAPI/System/Video/inputs/channels/{channel}/motionDetection"
EVENT_TRIGGERS_PATH = "/ISAPI/Event/triggers"
DEVICE_INFO_PATH = "/ISAPI/System/deviceInfo"


@dataclass
class CameraConfig:
    host: str
    username: str
    password: str
    scheme: str = "http"
    port: Optional[int] = None

    @property
    def base_url(self) -> str:
        host = self.host
        if self.port and ":" not in host:
            host = f"{host}:{self.port}"
        return f"{self.scheme}://{host}"


@dataclass
class AlertZone:
    name: str
    roi: Tuple[float, float, float, float]

    @property
    def x1(self) -> float:
        return self.roi[0]

    @property
    def y1(self) -> float:
        return self.roi[1]

    @property
    def x2(self) -> float:
        return self.roi[2]

    @property
    def y2(self) -> float:
        return self.roi[3]


def parse_rtsp_url(rtsp_url: str) -> tuple[str, str, str]:
    parsed = urlparse(rtsp_url)
    if not parsed.hostname:
        raise ValueError("RTSP_URL does not contain a hostname")
    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    if not username or not password:
        raise ValueError("RTSP_URL does not contain username and password")
    return parsed.hostname, username, password


def parse_normalized_roi(value: str, name: str) -> Tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"{name} must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{name} must satisfy 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0")
    return x1, y1, x2, y2


def load_alert_zones() -> List[AlertZone]:
    zones: List[AlertZone] = []
    for index in range(1, 3):
        value = os.getenv(f"HIKVISION_ALERT_ZONE_{index}", "").strip()
        if not value:
            continue
        name = os.getenv(f"HIKVISION_ALERT_ZONE_{index}_NAME", f"zone-{index}").strip() or f"zone-{index}"
        zones.append(AlertZone(name=name, roi=parse_normalized_roi(value, f"HIKVISION_ALERT_ZONE_{index}")))
    if not zones:
        fallback_roi = os.getenv("ROI", "").strip()
        if fallback_roi:
            zones.append(AlertZone(name="roi", roi=parse_normalized_roi(fallback_roi, "ROI")))
    return zones


def load_local_env_file(path: str = ".env") -> None:
    if dotenv_load is not None:
        dotenv_load(path)
        return
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_camera_config(args: argparse.Namespace) -> CameraConfig:
    load_local_env_file()
    env_host = os.getenv("HIKVISION_HOST", "").strip()
    env_user = (os.getenv("HIKVISION_USER", "") or os.getenv("HIKVISION_USERNAME", "")).strip()
    env_password = os.getenv("HIKVISION_PASSWORD", "").strip()
    rtsp_url = os.getenv("RTSP_URL", "").strip()

    host = args.host or env_host
    username = args.username or env_user
    password = args.password or env_password
    if not (host and username and password) and rtsp_url:
        rtsp_host, rtsp_user, rtsp_password = parse_rtsp_url(rtsp_url)
        host = host or rtsp_host
        username = username or rtsp_user
        password = password or rtsp_password
    if not host or not username or not password:
        raise ValueError(
            "Camera credentials are required. Set HIKVISION_HOST/HIKVISION_USER/HIKVISION_PASSWORD, "
            "or keep credentials in RTSP_URL, or pass --host --username --password."
        )
    return CameraConfig(
        host=host,
        username=username,
        password=password,
        scheme="https" if args.https else "http",
        port=args.port,
    )


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def child_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if strip_namespace(element.tag) == name and element.text:
            return element.text.strip()
    return ""


def first_element(root: ET.Element, name: str) -> Optional[ET.Element]:
    for element in root.iter():
        if strip_namespace(element.tag) == name:
            return element
    return None


def child_element(root: ET.Element, name: str) -> Optional[ET.Element]:
    for element in list(root):
        if strip_namespace(element.tag) == name:
            return element
    return None


def namespace_from_tag(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def tag_like(parent: ET.Element, local_name: str) -> str:
    namespace = namespace_from_tag(parent.tag)
    return f"{{{namespace}}}{local_name}" if namespace else local_name


def ensure_child(parent: ET.Element, local_name: str) -> ET.Element:
    existing = child_element(parent, local_name)
    if existing is not None:
        return existing
    return ET.SubElement(parent, tag_like(parent, local_name))


def summarize_event(xml_payload: bytes) -> str:
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError:
        text = xml_payload.decode("utf-8", errors="replace").strip()
        return f"non-xml event: {text[:300]}"

    event_type = child_text(root, "eventType") or "unknown"
    event_state = child_text(root, "eventState") or child_text(root, "activePostCount") or "unknown"
    channel = child_text(root, "channelID") or child_text(root, "dynChannelID") or "unknown"
    description = child_text(root, "eventDescription")
    date_time = child_text(root, "dateTime") or child_text(root, "time")
    active_count = child_text(root, "activePostCount")
    pieces = [
        f"type={event_type}",
        f"state={event_state}",
        f"channel={channel}",
    ]
    if active_count:
        pieces.append(f"count={active_count}")
    if date_time:
        pieces.append(f"time={date_time}")
    if description:
        pieces.append(f"description={description}")
    return " | ".join(pieces)


def is_motion_event(summary: str) -> bool:
    normalized = summary.lower()
    return "vmd" in normalized or "motion" in normalized or "fielddetection" in normalized or "linedetection" in normalized


def xml_payloads_from_stream(chunks: Iterator[bytes]) -> Iterator[bytes]:
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


def request_with_auth(session: requests.Session, method: str, url: str, config: CameraConfig, **kwargs) -> requests.Response:
    response = session.request(method, url, auth=HTTPDigestAuth(config.username, config.password), **kwargs)
    if response.status_code == 401:
        response.close()
        response = session.request(method, url, auth=HTTPBasicAuth(config.username, config.password), **kwargs)
    return response


def fetch_motion_config(session: requests.Session, config: CameraConfig, channel: int, verify_tls: bool) -> requests.Response:
    url = f"{config.base_url}{MOTION_CONFIG_PATH.format(channel=channel)}"
    response = request_with_auth(session, "GET", url, config, timeout=10, verify=verify_tls)
    response.raise_for_status()
    return response


def show_motion_config(session: requests.Session, config: CameraConfig, channel: int, verify_tls: bool) -> None:
    url = f"{config.base_url}{MOTION_CONFIG_PATH.format(channel=channel)}"
    response = request_with_auth(session, "GET", url, config, timeout=10, verify=verify_tls)
    if response.status_code == 404:
        print(f"Motion config endpoint returned 404 for channel {channel}")
        return
    response.raise_for_status()
    print(f"Motion config for channel {channel}:")
    print(response.text.strip())


def zone_to_grid_map(zones: List[AlertZone], rows: int, columns: int) -> str:
    cells: List[str] = []
    for row in range(rows):
        cell_y1 = row / rows
        cell_y2 = (row + 1) / rows
        for column in range(columns):
            cell_x1 = column / columns
            cell_x2 = (column + 1) / columns
            enabled = any(
                cell_x2 > zone.x1 and cell_x1 < zone.x2 and cell_y2 > zone.y1 and cell_y1 < zone.y2
                for zone in zones
            )
            cells.append("1" if enabled else "0")
    return "".join(cells)


def configure_grid_motion(root: ET.Element, zones: List[AlertZone]) -> bool:
    grid = first_element(root, "Grid")
    layout = first_element(root, "MotionDetectionLayout")
    if grid is None and layout is None:
        return False
    if grid is None:
        grid = ET.SubElement(root, tag_like(root, "Grid"))
    rows = int(child_text(grid, "rowGranularity") or os.getenv("HIKVISION_GRID_ROWS", "18"))
    columns = int(child_text(grid, "columnGranularity") or os.getenv("HIKVISION_GRID_COLUMNS", "22"))
    ensure_child(grid, "rowGranularity").text = str(rows)
    ensure_child(grid, "columnGranularity").text = str(columns)
    if layout is None:
        layout = ET.SubElement(root, tag_like(root, "MotionDetectionLayout"))
    sensitivity = os.getenv("HIKVISION_MOTION_SENSITIVITY", "").strip()
    if sensitivity:
        ensure_child(layout, "sensitivityLevel").text = sensitivity
    layout_container = child_element(layout, "layout") or layout
    if layout_container is layout and child_element(layout, "layout") is None:
        layout_container = ET.SubElement(layout, tag_like(layout, "layout"))
    ensure_child(layout_container, "gridMap").text = zone_to_grid_map(zones, rows, columns)
    ensure_child(root, "enabled").text = "true"
    ensure_child(root, "regionType").text = "grid"
    return True


def configure_region_motion(root: ET.Element, zones: List[AlertZone]) -> bool:
    region_list = first_element(root, "MotionDetectionRegionList")
    if region_list is None:
        return False
    for child in list(region_list):
        region_list.remove(child)
    for index, zone in enumerate(zones, start=1):
        region = ET.SubElement(region_list, tag_like(region_list, "MotionDetectionRegion"))
        ET.SubElement(region, tag_like(region, "id")).text = str(index)
        ET.SubElement(region, tag_like(region, "enabled")).text = "true"
        ET.SubElement(region, tag_like(region, "sensitivityLevel")).text = os.getenv("HIKVISION_MOTION_SENSITIVITY", "60")
        coords = ET.SubElement(region, tag_like(region, "RegionCoordinatesList"))
        for x, y in (
            (zone.x1, zone.y1),
            (zone.x2, zone.y1),
            (zone.x2, zone.y2),
            (zone.x1, zone.y2),
        ):
            coord = ET.SubElement(coords, tag_like(coords, "RegionCoordinates"))
            ET.SubElement(coord, tag_like(coord, "positionX")).text = str(int(round(x * 1000)))
            ET.SubElement(coord, tag_like(coord, "positionY")).text = str(int(round(y * 1000)))
    ensure_child(root, "enabled").text = "true"
    return True


def apply_alert_zones(
    session: requests.Session,
    config: CameraConfig,
    channel: int,
    verify_tls: bool,
    zones: List[AlertZone],
) -> None:
    if not zones:
        raise ValueError("No alert zones configured. Set HIKVISION_ALERT_ZONE_1 and HIKVISION_ALERT_ZONE_2 in .env.")
    response = fetch_motion_config(session, config, channel, verify_tls)
    root = ET.fromstring(response.content)
    updated = configure_grid_motion(root, zones) or configure_region_motion(root, zones)
    if not updated:
        raise RuntimeError("Unsupported Hikvision motion XML: no Grid/MotionDetectionLayout or RegionList found")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    url = f"{config.base_url}{MOTION_CONFIG_PATH.format(channel=channel)}"
    put_response = request_with_auth(
        session,
        "PUT",
        url,
        config,
        data=payload,
        timeout=10,
        verify=verify_tls,
        headers={"Content-Type": "application/xml"},
    )
    put_response.raise_for_status()
    print(f"Applied {len(zones)} alert zone(s) to Hikvision motion detection channel {channel}.")


def print_xml_endpoint(
    session: requests.Session,
    config: CameraConfig,
    title: str,
    path: str,
    verify_tls: bool,
    required: bool = False,
) -> None:
    url = f"{config.base_url}{path}"
    print(f"\n--- {title} ---")
    response = request_with_auth(session, "GET", url, config, timeout=10, verify=verify_tls)
    if response.status_code == 404 and not required:
        print(f"Not available on this camera/firmware: {url}")
        return
    response.raise_for_status()
    print(response.text.strip())


def print_alert_zones(zones: List[AlertZone]) -> None:
    print("\n--- Alert zones from .env ---")
    if not zones:
        print("No zones configured. Set HIKVISION_ALERT_ZONE_1 and HIKVISION_ALERT_ZONE_2.")
        return
    for index, zone in enumerate(zones, start=1):
        x1, y1, x2, y2 = zone.roi
        print(f"{index}. {zone.name}: {x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}")


def print_current_config(
    session: requests.Session,
    config: CameraConfig,
    channel: int,
    verify_tls: bool,
    zones: List[AlertZone],
) -> None:
    print("--- Tester config ---")
    print(f"camera={config.base_url}")
    print(f"user={config.username}")
    print(f"channel={channel}")
    print(f"alert_stream={config.base_url}{ALERT_STREAM_PATH}")
    print_alert_zones(zones)
    print_xml_endpoint(session, config, "Device info", DEVICE_INFO_PATH, verify_tls)
    print_xml_endpoint(
        session,
        config,
        f"Motion detection config for channel {channel}",
        MOTION_CONFIG_PATH.format(channel=channel),
        verify_tls,
        required=True,
    )
    print_xml_endpoint(session, config, "Event triggers config", EVENT_TRIGGERS_PATH, verify_tls)
    print()


def follow_alert_stream(
    session: requests.Session,
    config: CameraConfig,
    timeout: float,
    verify_tls: bool,
    motion_only: bool,
    once: bool,
) -> None:
    url = f"{config.base_url}{ALERT_STREAM_PATH}"
    print(f"Connecting to Hikvision event stream: {url}")
    print("Waiting for events. Trigger motion in front of the camera to test.")
    response = request_with_auth(
        session,
        "GET",
        url,
        config,
        stream=True,
        timeout=(10, timeout),
        verify=verify_tls,
        headers={"Accept": "multipart/x-mixed-replace, application/xml"},
    )
    response.raise_for_status()
    event_count = 0
    started_at = time.time()
    try:
        for payload in xml_payloads_from_stream(response.iter_content(chunk_size=4096)):
            summary = summarize_event(payload)
            if motion_only and not is_motion_event(summary):
                continue
            event_count += 1
            print(f"[{event_count:04d}] {summary}", flush=True)
            if once:
                return
            if timeout > 0 and time.time() - started_at > timeout:
                return
    finally:
        response.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Hikvision ISAPI motion/events from a DS-2CD2085G1-I or similar camera."
    )
    parser.add_argument("--host", help="Camera host/IP. Defaults to HIKVISION_HOST or RTSP_URL host.")
    parser.add_argument("--username", help="Camera username. Defaults to HIKVISION_USER or RTSP_URL username.")
    parser.add_argument("--password", help="Camera password. Defaults to HIKVISION_PASSWORD or RTSP_URL password.")
    parser.add_argument("--https", action="store_true", help="Use HTTPS instead of HTTP.")
    parser.add_argument("--port", type=int, help="HTTP/HTTPS port if not the default.")
    parser.add_argument("--channel", type=int, default=1, help="Video channel for motion config checks.")
    parser.add_argument(
        "--skip-current-config",
        action="store_true",
        help="Do not print camera config before listening for events.",
    )
    parser.add_argument(
        "--show-motion-config",
        action="store_true",
        help="Compatibility alias; current config is printed by default.",
    )
    parser.add_argument(
        "--apply-alert-zones",
        action="store_true",
        help="Write HIKVISION_ALERT_ZONE_1/2 from .env into the camera motion-detection region config before listening.",
    )
    parser.add_argument("--all-events", action="store_true", help="Print every ISAPI event instead of motion-like events only.")
    parser.add_argument("--once", action="store_true", help="Exit after the first matching event.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Seconds to wait. Use 0 to wait forever.")
    parser.add_argument("--verify-tls", action="store_true", help="Verify HTTPS certificate.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_camera_config(args)
        zones = load_alert_zones()
        session = requests.Session()
        if args.apply_alert_zones:
            apply_alert_zones(session, config, args.channel, args.verify_tls, zones)
        if not args.skip_current_config:
            print_current_config(session, config, args.channel, args.verify_tls, zones)
        follow_alert_stream(
            session=session,
            config=config,
            timeout=args.timeout,
            verify_tls=args.verify_tls,
            motion_only=not args.all_events,
            once=args.once,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
