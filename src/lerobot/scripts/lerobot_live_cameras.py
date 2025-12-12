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
Live camera preview tool for adjusting camera positions.

Shows live video feeds from connected cameras in real-time windows.
Press 'q' to quit.

Example:

```shell
# Show all cameras
lerobot-live-cameras

# Show only RealSense cameras
lerobot-live-cameras realsense

# Show only OpenCV cameras
lerobot-live-cameras opencv
```
"""

import argparse
import logging
import signal
import sys
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """Find all available OpenCV cameras."""
    logger.info("Searching for OpenCV cameras...")
    try:
        cameras = OpenCVCamera.find_cameras()
        logger.info(f"Found {len(cameras)} OpenCV cameras.")
        return cameras
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")
        return []


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """Find all available RealSense cameras."""
    logger.info("Searching for RealSense cameras...")
    try:
        cameras = RealSenseCamera.find_cameras()
        logger.info(f"Found {len(cameras)} RealSense cameras.")
        return cameras
    except ImportError:
        logger.warning("Skipping RealSense: pyrealsense2 not installed.")
        return []
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")
        return []


def find_cameras(camera_type: str | None = None) -> list[dict[str, Any]]:
    """Find cameras, optionally filtered by type."""
    cameras: list[dict[str, Any]] = []
    
    if camera_type:
        camera_type = camera_type.lower()
    
    if camera_type is None or camera_type == "opencv":
        cameras.extend(find_all_opencv_cameras())
    if camera_type is None or camera_type == "realsense":
        cameras.extend(find_all_realsense_cameras())
    
    return cameras


def create_camera_instance(cam_meta: dict[str, Any]) -> Any | None:
    """Create and connect to a camera based on its metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    
    logger.info(f"Connecting to {cam_type} camera: {cam_id}")
    
    try:
        if cam_type == "OpenCV":
            config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.BGR,  # BGR for OpenCV display
            )
            camera = OpenCVCamera(config)
        elif cam_type == "RealSense":
            config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.BGR,  # BGR for OpenCV display
            )
            camera = RealSenseCamera(config)
        else:
            logger.warning(f"Unknown camera type: {cam_type}")
            return None
        
        camera.connect(warmup=True)
        logger.info(f"Connected to {cam_type} camera: {cam_id}")
        return {"instance": camera, "meta": cam_meta}
    
    except Exception as e:
        logger.error(f"Failed to connect to {cam_type} camera {cam_id}: {e}")
        return None


def create_window_name(cam_meta: dict[str, Any]) -> str:
    """Create a descriptive window name for a camera."""
    cam_type = cam_meta.get("type", "Unknown")
    cam_id = cam_meta.get("id", "unknown")
    cam_name = cam_meta.get("name", "")
    return f"{cam_type}: {cam_id} - {cam_name}"


def add_info_overlay(frame: np.ndarray, cam_meta: dict[str, Any], fps: float) -> np.ndarray:
    """Add information overlay to the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Camera info
    cam_type = cam_meta.get("type", "Unknown")
    cam_id = str(cam_meta.get("id", "unknown"))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Type: {cam_type}", (20, 35), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"ID: {cam_id}", (20, 55), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f} | Res: {w}x{h}", (20, 75), font, 0.6, (0, 255, 0), 1)
    
    # Quit instruction at bottom
    cv2.putText(frame, "Press 'q' to quit", (w - 150, h - 15), font, 0.5, (200, 200, 200), 1)
    
    return frame


def run_live_preview(camera_type: str | None = None):
    """
    Run live camera preview.
    
    Args:
        camera_type: Optional filter for camera type ("realsense" or "opencv").
    """
    # Find cameras
    camera_metas = find_cameras(camera_type)
    
    if not camera_metas:
        if camera_type:
            logger.error(f"No {camera_type} cameras detected.")
        else:
            logger.error("No cameras detected.")
        return
    
    print("\n--- Detected Cameras ---")
    for i, meta in enumerate(camera_metas):
        print(f"  {i}: {meta.get('type')} - {meta.get('id')} ({meta.get('name', 'N/A')})")
    print("-" * 25)
    
    # Connect to cameras
    cameras: list[dict[str, Any]] = []
    for meta in camera_metas:
        cam = create_camera_instance(meta)
        if cam:
            cameras.append(cam)
    
    if not cameras:
        logger.error("No cameras could be connected.")
        return
    
    # Setup signal handler for clean exit
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        print("\nInterrupted. Shutting down...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create windows
    window_names = []
    for cam in cameras:
        name = create_window_name(cam["meta"])
        window_names.append(name)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 600)
    
    print(f"\n🎥 Live preview running for {len(cameras)} camera(s)")
    print("Press 'q' in any window to quit.\n")
    
    # FPS tracking
    fps_counters = [{"frames": 0, "last_time": cv2.getTickCount(), "fps": 0.0} for _ in cameras]
    
    try:
        while running:
            for i, cam in enumerate(cameras):
                try:
                    # Read frame
                    frame = cam["instance"].read()
                    
                    # Update FPS counter
                    fps_counters[i]["frames"] += 1
                    current_time = cv2.getTickCount()
                    elapsed = (current_time - fps_counters[i]["last_time"]) / cv2.getTickFrequency()
                    if elapsed >= 1.0:
                        fps_counters[i]["fps"] = fps_counters[i]["frames"] / elapsed
                        fps_counters[i]["frames"] = 0
                        fps_counters[i]["last_time"] = current_time
                    
                    # Add overlay and display
                    frame_with_overlay = add_info_overlay(frame, cam["meta"], fps_counters[i]["fps"])
                    cv2.imshow(window_names[i], frame_with_overlay)
                    
                except Exception as e:
                    logger.warning(f"Error reading from camera {cam['meta'].get('id')}: {e}")
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed.")
                break
            
            # Check if any window was closed
            for name in window_names:
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed.")
                    running = False
                    break
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        
        for cam in cameras:
            try:
                if cam["instance"].is_connected:
                    cam["instance"].disconnect()
                    logger.info(f"Disconnected camera: {cam['meta'].get('id')}")
            except Exception as e:
                logger.error(f"Error disconnecting camera {cam['meta'].get('id')}: {e}")
        
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Live camera preview for adjusting camera positions. Press 'q' to quit."
    )
    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Filter by camera type (e.g., 'realsense', 'opencv'). Shows all if omitted.",
    )
    
    args = parser.parse_args()
    run_live_preview(camera_type=args.camera_type)


if __name__ == "__main__":
    main()

