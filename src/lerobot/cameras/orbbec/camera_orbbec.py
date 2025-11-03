#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Orbbec Gemini camera with optional depth capture.

Mirrors the RealSense camera interface, including async paired RGB/depth reading
for synchronized frames.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

import pyorbbecsdk as ob

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig
from ...errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

# Shared Orbbec SDK Context for all camera instances
# The Orbbec SDK requires a single Context to be shared across all devices
# Creating multiple Context objects causes segmentation faults
_SHARED_CONTEXT: ob.Context | None = None
_SHARED_CONTEXT_LOCK = Lock()

def _get_shared_context() -> ob.Context:
    """Get or create the shared Orbbec SDK Context.
    
    The Orbbec SDK has stability issues when multiple Context objects are created.
    This function ensures all OrbbecCamera instances share a single Context.
    """
    global _SHARED_CONTEXT
    with _SHARED_CONTEXT_LOCK:
        if _SHARED_CONTEXT is None:
            _SHARED_CONTEXT = ob.Context()
            logger.info("Created shared Orbbec SDK Context")
        return _SHARED_CONTEXT

class OrbbecCamera(Camera):
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config

        self.index_or_path = config.index_or_path
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        # SDK handles
        self._ctx: ob.Context | None = None
        self._device: ob.Device | None = None
        self._pipeline: ob.Pipeline | None = None
        self._config: ob.Config | None = None

        # Async thread management
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()
        self.latest_depth_frame: np.ndarray | None = None
        self.new_depth_frame_event: Event = Event()

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None and self._device is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        context = _get_shared_context()
        device_list = context.query_devices()
        found: list[dict[str, Any]] = []
        for i in range(device_list.get_count()):
            dev = device_list[i]
            try:
                info = dev.get_device_info()
                name = info.name() if hasattr(info, "name") else "Orbbec"
                serial = info.serial_number() if hasattr(info, "serial_number") else ""
            except Exception:
                name, serial = "Orbbec", ""

            cam = {"name": name, "type": "Orbbec", "id": serial}
            found.append(cam)
        return found

    def _match_device(self, context: "ob.Context") -> "ob.Device":
        device_list = context.query_devices()
        count = device_list.get_count()
        
        if count == 0:
            raise ConnectionError("No Orbbec devices detected.")
        
        # Convert index_or_path to integer index
        if isinstance(self.index_or_path, int):
            device_index = self.index_or_path
        else:
            try:
                device_index = int(self.index_or_path)
            except (ValueError, TypeError):
                raise ValueError(
                    f"index_or_path must be an integer or string integer, got: '{self.index_or_path}'"
                )
        
        # Validate index range
        if device_index < 0 or device_index >= count:
            raise ValueError(
                f"Device index {device_index} is out of range. "
                f"Found {count} device(s), valid indices are 0-{count-1}."
            )
        
        # Get device by index
        dev = device_list.get_device_by_index(device_index)
        info = dev.get_device_info()
        name = info.name() if hasattr(info, "name") else "Orbbec"
        serial = info.serial_number() if hasattr(info, "serial_number") else ""
        logger.info(f"Using Orbbec device at index {device_index}: name={name}, serial={serial}")
        
        return dev

    def _find_stream_profile(self, sensor_type: ob.OBSensorType, stream_type: ob.OBStreamType, 
                             width: int = None, height: int = None, 
                             fmt: ob.OBFormat = None, fps: int = None) -> ob.StreamProfile:
        """Find a stream profile matching the desired parameters using pipeline."""
        try:
            profile_list = self._pipeline.get_stream_profile_list(sensor_type)
        except Exception as e:
            raise RuntimeError(f"Failed to get stream profile list for {sensor_type}: {e}")
        
        if profile_list is None:
            raise RuntimeError(f"No stream profiles available for {sensor_type}")
        
        # If no specific requirements, return the default profile
        if width is None and height is None and fmt is None and fps is None:
            return profile_list.get_default_video_stream_profile()
        
        # Try to find exact match
        for i in range(profile_list.get_count()):
            profile = profile_list.get_stream_profile_by_index(i)
            if profile.get_type() != stream_type:
                continue
                
            video_profile = profile.as_video_stream_profile()
            matches = True
            
            if width is not None and video_profile.get_width() != width:
                matches = False
            if height is not None and video_profile.get_height() != height:
                matches = False
            if fmt is not None and video_profile.get_format() != fmt:
                matches = False
            if fps is not None and video_profile.get_fps() != fps:
                matches = False
                
            if matches:
                return profile
        
        # If no exact match, use default profile
        logger.warning(f"Exact match not found for {sensor_type}, using default profile")
        return profile_list.get_default_video_stream_profile()

    def _build_config(self) -> "ob.Config":
        cfg = ob.Config()
        
        # Color stream configuration
        # Follow Orbbec SDK examples: get profile list, then get specific or default profile
        try:
            profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            if profile_list:
                if self.width and self.height and self.fps:
                    # Try to get specific profile matching requested parameters
                    try:
                        # Try MJPG format first for better bandwidth
                        color_profile = profile_list.get_video_stream_profile(
                            self.capture_width, self.capture_height, ob.OBFormat.MJPG, self.fps
                        )
                        logger.info(f"Using MJPG color profile: {self.capture_width}x{self.capture_height}@{self.fps}fps")
                    except Exception:
                        # Fallback to any format with matching resolution/fps
                        try:
                            color_profile = profile_list.get_video_stream_profile(
                                self.capture_width, 0, ob.OBFormat.ANY, self.fps
                            )
                            logger.info(f"Using ANY format color profile: {self.capture_width}x*@{self.fps}fps")
                        except Exception:
                            color_profile = profile_list.get_default_video_stream_profile()
                            logger.warning("Using default color profile (requested profile not available)")
                else:
                    # No specific requirements, use default
                    color_profile = profile_list.get_default_video_stream_profile()
                    logger.info("Using default color profile")
                
                cfg.enable_stream(color_profile)
        except Exception as e:
            logger.error(f"Failed to enable color stream: {e}")
            raise

        # Depth stream configuration
        if self.use_depth:
            try:
                profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                logger.info(f"Depth profile list: {profile_list}")

                if profile_list:
                    depth_profile = None
                    if self.width and self.height and self.fps:
                        # Try to get specific profile matching requested parameters
                        try:
                            # Use Y10  format for depth (uint8 millimeters)
                            depth_profile = profile_list.get_video_stream_profile(
                                self.capture_width, self.capture_height, ob.OBFormat.Y11, self.fps
                            )
                            logger.info(f"Using Y10 depth profile: {self.capture_width}x{self.capture_height}@{self.fps}fps")
                        except Exception:
                            depth_profile = profile_list.get_default_video_stream_profile()
                            logger.info("Using default depth profile")

                    if depth_profile is None:
                        depth_profile = profile_list.get_default_video_stream_profile()
                        logger.info("Using default depth profile")

                    cfg.enable_stream(depth_profile)
            except Exception as e:
                logger.error(f"Failed to enable depth stream: {e}")
                raise

        return cfg

    def _configure_capture_settings(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot validate settings for {self} as it is not connected.")

        # Query stream profiles to resolve actual width/height/fps for color
        # Orbbec SDK typically aligns to requested values; no-op if provided.
        if self.width is None or self.height is None:
            # Fallback to a single frameset read to infer dimensions
            ret = self._pipeline.wait_for_frames(1000)
            if ret is None:
                raise RuntimeError(f"Failed to fetch initial frameset for {self}.")
            color_frame = ret.get_color_frame()
            if color_frame is None:
                raise RuntimeError(f"{self} failed to get color frame for shape inference.")
            # Get frame as video stream profile to get dimensions
            video_profile = color_frame.get_stream_profile().as_video_stream_profile()
            w = video_profile.get_width()
            h = video_profile.get_height()
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = h, w
                self.capture_width, self.capture_height = w, h
            else:
                self.width, self.height = w, h
                self.capture_width, self.capture_height = w, h

    def _i420_to_rgb(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        y = frame[0:height, :]
        u = frame[height:height + height // 4].reshape(height // 2, width // 2)
        v = frame[height + height // 4:].reshape(height // 2, width // 2)
        yuv_image = cv2.merge([y, u, v])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def _nv12_to_rgb(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        y = frame[0:height, :]
        uv = frame[height:height + height // 2].reshape(height // 2, width)
        yuv_image = cv2.merge([y, uv])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def _nv21_to_rgb(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        y = frame[0:height, :]
        uv = frame[height:height + height // 2].reshape(height // 2, width)
        yuv_image = cv2.merge([y, uv])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def _frame_to_rgb_image(self, frame: ob.VideoFrame) -> np.ndarray:
        """Convert Orbbec frame to RGB format."""
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if color_format == ob.OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            # Already RGB
        elif color_format == ob.OBFormat.BGR:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_format == ob.OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUYV)
        elif color_format == ob.OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            # imdecode returns BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_format == ob.OBFormat.I420:
            return self._i420_to_rgb(data, width, height)
        elif color_format == ob.OBFormat.NV12:
            return self._nv12_to_rgb(data, width, height)
        elif color_format == ob.OBFormat.NV21:
            return self._nv21_to_rgb(data, width, height)
        elif color_format == ob.OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
        else:
            logger.error(f"Unsupported color format: {color_format}")
            return None
        return image

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Use shared context to avoid SDK segmentation faults
        self._ctx = _get_shared_context()
        self._device = self._match_device(self._ctx)
        self._pipeline = ob.Pipeline(self._device)
        self._config = self._build_config()

        try:
            self._pipeline.start(self._config)
        except Exception as e:
            self._pipeline = None
            self._device = None
            self._ctx = None
            raise ConnectionError(f"Failed to open {self}.") from e

        self._configure_capture_settings()

        if warmup:
            time.sleep(1)
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    _ = self.read()
                except Exception:
                    pass
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _postprocess_image(self, image: np.ndarray, is_depth: bool = False) -> np.ndarray:
        if is_depth:
            h, w = image.shape
        else:
            h, w, c = image.shape
            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        # Handle resolution mismatch - update configured dimensions if actual stream differs
        # This happens when SDK falls back to default profile (e.g., Gemini E uses 640x360 depth)
        if h != self.capture_height or w != self.capture_width:
            logger.warning(
                f"{self} received frame {w}x{h}, expected {self.capture_width}x{self.capture_height}. "
                f"Updating configuration to match actual stream."
            )
            self.capture_width, self.capture_height = w, h
            # Update output dimensions based on rotation
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = h, w
            else:
                self.width, self.height = w, h

        processed = image
        if not is_depth and self.color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)

        return processed

    def _process_depth_frame(self, depth_frame: ob.DepthFrame) -> np.ndarray:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        
        # Process depth data following Orbbec SDK example
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0).astype(np.uint16)
        
        logger.debug(f"Orbbec depth scale: {scale}, depth range: {depth_data[depth_data > 0].min() if depth_data.any() else 0}-{depth_data.max()}")
        
        depth_data = self._postprocess_image(depth_data, is_depth=True)
        return depth_data
    
    def _process_color_frame(self, color_frame: ob.ColorFrame) -> np.ndarray:
        color_image = self._frame_to_rgb_image(color_frame)
        color_image = self._postprocess_image(color_image, is_depth=False)
        return color_image

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        frameset = self._pipeline.wait_for_frames(200)
        if frameset is None:
            raise RuntimeError(f"{self} read failed (no frameset).")

        color_frame = frameset.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} color frame is None.")
        
        # Get format and decode/convert as needed
        color_image = self._process_color_frame(color_frame)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")
        return color_image

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        start_time = time.perf_counter()
        frameset = self._pipeline.wait_for_frames(timeout_ms)
        if frameset is None:
            raise RuntimeError(f"{self} read_depth failed (no frameset).")
        depth_frame = frameset.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError(f"{self} depth frame is None.")
        
        
        # Process depth data
        depth_image = self._process_depth_frame(depth_frame)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read_depth took: {read_duration_ms:.1f}ms")
        return depth_image

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                frameset = self._pipeline.wait_for_frames(500)
                if frameset is None:
                    continue

                color_image = None
                depth_image = None

                c = frameset.get_color_frame()
                if c is not None:
                    color_image = self._process_color_frame(c)

                if self.use_depth:
                    d = frameset.get_depth_frame()
                    if d is not None:
                        depth_image = self._process_depth_frame(d)

                set_color_event = False
                set_depth_event = False
                with self.frame_lock:
                    if color_image is not None:
                        self.latest_frame = color_image
                        set_color_event = True
                    if depth_image is not None:
                        self.latest_depth_frame = depth_image
                        set_depth_event = True

                if set_color_event:
                    self.new_frame_event.set()
                if set_depth_event:
                    self.new_depth_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
        return frame

    def async_read_depth(self, timeout_ms: float = 200) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame with 'async_read_depth()'. Depth stream is not enabled for {self}."
            )

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_depth_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for depth frame from camera {self} after {timeout_ms} ms. Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
            depth_frame = self.latest_depth_frame
            self.new_depth_frame_event.clear()

        if depth_frame is None:
            raise RuntimeError(f"Internal error: Event set but no depth frame available for {self}.")
        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame, depth_frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._config = None
            self._device = None
            # Don't set _ctx = None since it's a shared resource
            # self._ctx remains pointing to the shared context

        logger.info(f"{self} disconnected.")