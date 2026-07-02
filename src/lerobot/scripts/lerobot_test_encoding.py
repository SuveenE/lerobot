#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test video encoding pipeline by recording RealSense cameras to disk.

For every connected RealSense camera, this script:

1. Captures frames as PNGs into ``outputs/test_encoding/<run>/<camera>/`` for the
   requested duration (default 3 minutes).
2. After capture finishes, encodes each camera's PNG sequence into a video file
   (``outputs/test_encoding/<run>/<camera>.mp4`` by default) using the same
   ``encode_video_frames`` pipeline that the dataset code uses.
3. Optionally deletes the temporary PNGs after successful encoding.

Examples:

```shell
# Default: 3 minutes per camera, all detected RealSenses.
lerobot-test-encoding

# 30 seconds for a quick smoke test.
lerobot-test-encoding --time 30

# Pick an output directory and keep raw PNGs after encoding.
lerobot-test-encoding --time 60 --output-dir outputs/my_run --keep-frames
```
"""

import argparse
import logging
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.video_utils import encode_video_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DURATION_S = 3 * 60  # 3 minutes
DEFAULT_FPS = 30
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_VCODEC = "libsvtav1"

_running = True


@dataclass
class CameraRecorder:
    """Per-camera state for capture + encode."""

    instance: RealSenseCamera
    meta: dict[str, Any]
    frames_dir: Path
    video_path: Path
    frame_count: int = 0
    capture_errors: int = 0
    thread: threading.Thread | None = None


def _safe_name(value: str) -> str:
    """Make a string safe to use as a directory/file name."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value).strip("_")


def find_realsense_cameras() -> list[dict[str, Any]]:
    """Find all available RealSense cameras."""
    logger.info("Searching for RealSense cameras...")
    try:
        found = RealSenseCamera.find_cameras()
        logger.info(f"Found {len(found)} RealSense camera(s).")
        return found
    except ImportError:
        logger.error("pyrealsense2 is not installed; cannot enumerate RealSense cameras.")
        return []
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error finding RealSense cameras: {e}")
        return []


def connect_camera(
    cam_meta: dict[str, Any], fps: int, width: int, height: int
) -> RealSenseCamera | None:
    """Create and connect to a RealSense camera based on its metadata."""
    cam_id = cam_meta.get("id")
    cam_name = cam_meta.get("name", "")
    logger.info(f"Connecting to RealSense camera: {cam_id} ({cam_name})")

    try:
        config = RealSenseCameraConfig(
            serial_number_or_name=cam_id,
            fps=fps,
            width=width,
            height=height,
            color_mode=ColorMode.RGB,  # PNG/video expect RGB
        )
        camera = RealSenseCamera(config)
        camera.connect(warmup=True)
        logger.info(f"Connected to RealSense camera: {cam_id}")
        return camera
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to connect to RealSense camera {cam_id}: {e}")
        return None


def capture_loop(rec: CameraRecorder, deadline: float):
    """Continuously read frames from `rec.instance` and dump them as PNGs.

    Runs until `deadline` (monotonic time) is reached or the global `_running`
    flag is cleared (e.g. on Ctrl+C).
    """
    cam = rec.instance
    cam_id = rec.meta.get("id")

    # The dataset video pipeline expects frame-NNNNNN.png filenames.
    while _running and time.monotonic() < deadline:
        try:
            frame_rgb = cam.read()
            # OpenCV writes BGR to disk by default, but the camera is already
            # configured for RGB; convert before writing so the encoded video
            # uses true colors.
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out_path = rec.frames_dir / f"frame-{rec.frame_count:06d}.png"
            ok = cv2.imwrite(str(out_path), frame_bgr)
            if not ok:
                rec.capture_errors += 1
                logger.warning(f"[{cam_id}] cv2.imwrite returned False for {out_path}")
                continue
            rec.frame_count += 1
        except Exception as e:  # noqa: BLE001
            rec.capture_errors += 1
            logger.warning(f"[{cam_id}] capture error: {e}")
            # Brief backoff so we don't spin on persistent errors.
            time.sleep(0.05)


def encode_recorder(rec: CameraRecorder, fps: int, vcodec: str) -> bool:
    """Encode the captured PNG sequence into a video file. Returns success."""
    cam_id = rec.meta.get("id")
    if rec.frame_count == 0:
        logger.error(f"[{cam_id}] no frames captured; skipping encoding.")
        return False

    logger.info(
        f"[{cam_id}] encoding {rec.frame_count} frames -> {rec.video_path} "
        f"(codec={vcodec}, fps={fps})"
    )
    start = time.monotonic()
    try:
        encode_video_frames(
            imgs_dir=rec.frames_dir,
            video_path=rec.video_path,
            fps=fps,
            vcodec=vcodec,
            overwrite=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{cam_id}] encoding failed: {e}")
        return False

    elapsed = time.monotonic() - start
    size_mb = rec.video_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"[{cam_id}] encoded in {elapsed:.1f}s -> {rec.video_path} ({size_mb:.2f} MB)"
    )
    return True


def run_test_encoding(
    duration_s: float = DEFAULT_DURATION_S,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    vcodec: str = DEFAULT_VCODEC,
    output_dir: Path | None = None,
    keep_frames: bool = False,
) -> int:
    """Run the full record + encode pipeline. Returns process exit code."""
    global _running

    metas = find_realsense_cameras()
    if not metas:
        logger.error("No RealSense cameras detected. Exiting.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("outputs") / "test_encoding" / timestamp
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recorders: list[CameraRecorder] = []
    for meta in metas:
        cam = connect_camera(meta, fps=fps, width=width, height=height)
        if cam is None:
            continue
        cam_label = _safe_name(str(meta.get("name") or meta.get("id") or f"cam{len(recorders)}"))
        frames_dir = output_dir / "frames" / cam_label
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{cam_label}.mp4"
        recorders.append(
            CameraRecorder(
                instance=cam,
                meta=meta,
                frames_dir=frames_dir,
                video_path=video_path,
            )
        )

    if not recorders:
        logger.error("No RealSense cameras could be connected. Exiting.")
        return 1

    print("\n=== Test Encoding Run ===")
    print(f"  Output dir : {output_dir.resolve()}")
    print(f"  Duration   : {duration_s:.0f}s")
    print(f"  Resolution : {width}x{height} @ {fps}fps")
    print(f"  Codec      : {vcodec}")
    print(f"  Cameras    : {len(recorders)}")
    for r in recorders:
        print(f"    - {r.meta.get('id')} ({r.meta.get('name', 'N/A')}) -> {r.video_path}")
    print("=" * 27)

    def _signal_handler(sig, frame):  # noqa: ARG001
        global _running
        print("\nCtrl+C received; stopping capture early...")
        _running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --- Capture phase ---
    capture_start = time.monotonic()
    deadline = capture_start + duration_s
    for rec in recorders:
        rec.thread = threading.Thread(
            target=capture_loop, args=(rec, deadline), daemon=True
        )
        rec.thread.start()

    # Print a progress line every few seconds so the user knows it's alive.
    try:
        while _running and time.monotonic() < deadline:
            time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))
            elapsed = time.monotonic() - capture_start
            total_frames = sum(r.frame_count for r in recorders)
            logger.info(
                f"Capturing... {elapsed:6.1f}s / {duration_s:.0f}s "
                f"({total_frames} frames across {len(recorders)} cameras)"
            )
    finally:
        _running = False  # tell capture threads to stop ASAP
        for rec in recorders:
            if rec.thread is not None:
                rec.thread.join(timeout=5.0)

    capture_elapsed = time.monotonic() - capture_start
    total_frames = sum(r.frame_count for r in recorders)
    logger.info(
        f"Capture done in {capture_elapsed:.1f}s. Total frames: {total_frames}."
    )

    # Disconnect cameras before the (potentially slow) encoding step.
    for rec in recorders:
        try:
            if rec.instance.is_connected:
                rec.instance.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error disconnecting camera {rec.meta.get('id')}: {e}")

    # --- Encoding phase ---
    success_count = 0
    for rec in recorders:
        if encode_recorder(rec, fps=fps, vcodec=vcodec):
            success_count += 1
            if not keep_frames:
                try:
                    shutil.rmtree(rec.frames_dir)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Could not remove frames dir {rec.frames_dir}: {e}")

    if not keep_frames and success_count == len(recorders):
        # Best-effort cleanup of the now-empty `frames/` parent.
        frames_parent = output_dir / "frames"
        try:
            frames_parent.rmdir()
        except OSError:
            pass

    print("\n=== Summary ===")
    for rec in recorders:
        status = "OK " if rec.video_path.exists() else "FAIL"
        print(
            f"  [{status}] {rec.meta.get('id')}: {rec.frame_count} frames, "
            f"{rec.capture_errors} errors -> {rec.video_path}"
        )
    print(f"Output: {output_dir.resolve()}")
    print("=" * 15)

    return 0 if success_count == len(recorders) else 2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Record from every connected RealSense camera for a fixed duration, "
            "then encode the captured frames to video."
        )
    )
    parser.add_argument(
        "--time",
        "-t",
        dest="duration_s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION_S}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Camera/encoder FPS (default: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Capture width in pixels (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Capture height in pixels (default: {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default=DEFAULT_VCODEC,
        choices=["h264", "hevc", "libsvtav1"],
        help=f"Video codec (default: {DEFAULT_VCODEC}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write frames and final videos to "
            "(default: outputs/test_encoding/<timestamp>)."
        ),
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the temporary PNG frames after encoding (default: delete).",
    )

    args = parser.parse_args()

    exit_code = run_test_encoding(
        duration_s=args.duration_s,
        fps=args.fps,
        width=args.width,
        height=args.height,
        vcodec=args.vcodec,
        output_dir=args.output_dir,
        keep_frames=args.keep_frames,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
