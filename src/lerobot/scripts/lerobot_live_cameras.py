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

Streams live video feeds from connected cameras to your web browser.
Open http://localhost:8000 to view the feeds.
Press Ctrl+C to stop.

Example:

```shell
# Show all cameras
lerobot-live-cameras

# Show only RealSense cameras
lerobot-live-cameras realsense

# Show only OpenCV cameras
lerobot-live-cameras opencv

# Use a different port
lerobot-live-cameras --port 9000
```
"""

import argparse
import logging
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
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

# Global state for cameras
cameras: list[dict[str, Any]] = []
running = True


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """Find all available OpenCV cameras."""
    logger.info("Searching for OpenCV cameras...")
    try:
        found = OpenCVCamera.find_cameras()
        logger.info(f"Found {len(found)} OpenCV cameras.")
        return found
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")
        return []


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """Find all available RealSense cameras."""
    logger.info("Searching for RealSense cameras...")
    try:
        found = RealSenseCamera.find_cameras()
        logger.info(f"Found {len(found)} RealSense cameras.")
        return found
    except ImportError:
        logger.warning("Skipping RealSense: pyrealsense2 not installed.")
        return []
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")
        return []


def find_cameras(camera_type: str | None = None) -> list[dict[str, Any]]:
    """Find cameras, optionally filtered by type."""
    found: list[dict[str, Any]] = []
    
    if camera_type:
        camera_type = camera_type.lower()
    
    if camera_type is None or camera_type == "opencv":
        found.extend(find_all_opencv_cameras())
    if camera_type is None or camera_type == "realsense":
        found.extend(find_all_realsense_cameras())
    
    return found


def create_camera_instance(cam_meta: dict[str, Any]) -> Any | None:
    """Create and connect to a camera based on its metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    
    logger.info(f"Connecting to {cam_type} camera: {cam_id}")
    
    try:
        if cam_type == "OpenCV":
            config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.BGR,
            )
            camera = OpenCVCamera(config)
        elif cam_type == "RealSense":
            config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.BGR,
            )
            camera = RealSenseCamera(config)
        else:
            logger.warning(f"Unknown camera type: {cam_type}")
            return None
        
        camera.connect(warmup=True)
        logger.info(f"Connected to {cam_type} camera: {cam_id}")
        return {"instance": camera, "meta": cam_meta, "frame": None, "fps": 0.0, "lock": threading.Lock()}
    
    except Exception as e:
        logger.error(f"Failed to connect to {cam_type} camera {cam_id}: {e}")
        return None


def add_info_overlay(frame: np.ndarray, cam_meta: dict[str, Any], fps: float) -> np.ndarray:
    """Add information overlay to the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Camera info
    cam_type = cam_meta.get("type", "Unknown")
    cam_id = str(cam_meta.get("id", "unknown"))
    cam_name = cam_meta.get("name", "")
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{cam_name}", (20, 35), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"ID: {cam_id}", (20, 55), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Type: {cam_type}", (20, 75), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f} | {w}x{h}", (20, 95), font, 0.5, (0, 255, 0), 1)
    
    return frame


def camera_capture_thread(cam_dict: dict[str, Any]):
    """Thread to continuously capture frames from a camera."""
    global running
    
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    
    frame_count = 0
    last_fps_time = time.time()
    
    while running:
        try:
            frame = cam.read()
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - last_fps_time
            if elapsed >= 1.0:
                cam_dict["fps"] = frame_count / elapsed
                frame_count = 0
                last_fps_time = current_time
            
            # Add overlay
            frame_with_overlay = add_info_overlay(frame, meta, cam_dict["fps"])
            
            # Store frame
            with cam_dict["lock"]:
                cam_dict["frame"] = frame_with_overlay
                
        except Exception as e:
            logger.warning(f"Error reading from camera {meta.get('id')}: {e}")
            time.sleep(0.1)


def generate_html_page() -> str:
    """Generate HTML page with all camera feeds."""
    num_cameras = len(cameras)
    
    # Calculate grid layout
    if num_cameras == 1:
        cols = 1
    elif num_cameras == 2:
        cols = 2
    elif num_cameras <= 4:
        cols = 2
    else:
        cols = 3
    
    camera_divs = []
    for i, cam in enumerate(cameras):
        meta = cam["meta"]
        cam_name = meta.get("name", f"Camera {i}")
        cam_id = meta.get("id", "unknown")
        camera_divs.append(f'''
            <div class="camera-container">
                <img src="/stream/{i}" alt="{cam_name}">
                <div class="camera-label">{cam_name}<br><small>{cam_id}</small></div>
            </div>
        ''')
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>LeRobot Live Cameras</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }}
        header {{
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid #333;
        }}
        h1 {{
            font-size: 1.8em;
            font-weight: 300;
            letter-spacing: 3px;
            color: #00ff88;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }}
        .subtitle {{
            color: #888;
            font-size: 0.85em;
            margin-top: 8px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat({cols}, 1fr);
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .camera-container {{
            background: rgba(20, 20, 30, 0.8);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #333;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .camera-container:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 255, 136, 0.15);
            border-color: #00ff88;
        }}
        .camera-container img {{
            width: 100%;
            display: block;
            background: #111;
        }}
        .camera-label {{
            padding: 12px 15px;
            background: rgba(0, 0, 0, 0.5);
            font-size: 0.9em;
            color: #ccc;
        }}
        .camera-label small {{
            color: #666;
            font-size: 0.8em;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #555;
            font-size: 0.8em;
        }}
        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>🤖 LEROBOT LIVE</h1>
        <p class="subtitle">{num_cameras} camera{'s' if num_cameras != 1 else ''} connected • Press Ctrl+C in terminal to stop</p>
    </header>
    <div class="grid">
        {''.join(camera_divs)}
    </div>
    <footer>LeRobot Camera Preview • Refresh page if streams freeze</footer>
</body>
</html>'''
    return html


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MJPEG streaming."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        global running
        
        if self.path == "/" or self.path == "/index.html":
            # Serve HTML page
            content = generate_html_page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
            
        elif self.path.startswith("/stream/"):
            # Stream MJPEG for specific camera
            try:
                cam_index = int(self.path.split("/")[-1])
                if cam_index >= len(cameras):
                    self.send_error(404, "Camera not found")
                    return
                
                cam_dict = cameras[cam_index]
                
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                
                while running:
                    with cam_dict["lock"]:
                        frame = cam_dict["frame"]
                    
                    if frame is not None:
                        # Encode frame as JPEG
                        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        jpeg_bytes = jpeg.tobytes()
                        
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode())
                            self.wfile.write(jpeg_bytes)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    
                    time.sleep(0.033)  # ~30 FPS
                    
            except (ValueError, IndexError):
                self.send_error(400, "Invalid camera index")
        else:
            self.send_error(404, "Not found")


def run_live_preview(camera_type: str | None = None, port: int = 8000):
    """
    Run live camera preview with web streaming.
    
    Args:
        camera_type: Optional filter for camera type ("realsense" or "opencv").
        port: HTTP server port (default: 8000).
    """
    global cameras, running
    
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
    for meta in camera_metas:
        cam = create_camera_instance(meta)
        if cam:
            cameras.append(cam)
    
    if not cameras:
        logger.error("No cameras could be connected.")
        return
    
    # Start capture threads
    threads = []
    for cam in cameras:
        t = threading.Thread(target=camera_capture_thread, args=(cam,), daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for first frames
    time.sleep(0.5)
    
    # Setup signal handler
    def signal_handler(sig, frame):
        global running
        print("\n\nShutting down...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start HTTP server
    server = HTTPServer(("0.0.0.0", port), StreamHandler)
    server.timeout = 1
    
    print(f"\n{'='*50}")
    print(f"🎥 Live camera preview running!")
    print(f"   Open in browser: http://localhost:{port}")
    print(f"   {len(cameras)} camera(s) streaming")
    print(f"{'='*50}")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while running:
            server.handle_request()
    finally:
        print("\nCleaning up...")
        running = False
        server.server_close()
        
        for cam in cameras:
            try:
                if cam["instance"].is_connected:
                    cam["instance"].disconnect()
                    logger.info(f"Disconnected camera: {cam['meta'].get('id')}")
            except Exception as e:
                logger.error(f"Error disconnecting camera {cam['meta'].get('id')}: {e}")
        
        cameras.clear()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Live camera preview via web browser. Open http://localhost:8000 to view feeds."
    )
    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Filter by camera type (e.g., 'realsense', 'opencv'). Shows all if omitted.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP server port (default: 8000).",
    )
    
    args = parser.parse_args()
    run_live_preview(camera_type=args.camera_type, port=args.port)


if __name__ == "__main__":
    main()
