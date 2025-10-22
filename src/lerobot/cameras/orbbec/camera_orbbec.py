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

try:
    import pyorbbecsdk as ob
except Exception as e:
    logging.info(f"Could not import pyorbbecsdk: {e}")

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig
from ...errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config

        self.serial_or_name = config.serial_number_or_name
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
        return f"{self.__class__.__name__}({self.serial_or_name})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None and self._device is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        context = ob.Context()
        device_list = context.query_devices()
        found: list[dict[str, Any]] = []
        for i in range(device_list.get_count()):
            dev = device_list.get_device(i)
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
        if device_list.get_count() == 0:
            raise ConnectionError("No Orbbec devices detected.")

        # If digits only, match serial number; else match name
        for i in range(device_list.get_count()):
            dev = device_list.get_device(i)
            info = dev.get_device_info()
            name = info.name() if hasattr(info, "name") else "Orbbec"
            serial = info.serial_number() if hasattr(info, "serial_number") else ""
            if self.serial_or_name == serial or self.serial_or_name == name:
                return dev

        raise ValueError(
            f"No Orbbec device found for identifier '{self.serial_or_name}'.")

    def _build_config(self) -> "ob.Config":
        cfg = ob.Config()

        # Color stream
        if self.width and self.height and self.fps:
            cfg.enable_stream(ob.StreamType.COLOR, self.capture_width, self.capture_height, ob.Format.RGB, self.fps)
        else:
            cfg.enable_stream(ob.StreamType.COLOR)

        # Depth stream
        if self.use_depth:
            if self.width and self.height and self.fps:
                cfg.enable_stream(ob.StreamType.DEPTH, self.capture_width, self.capture_height, ob.Format.Y16, self.fps)
            else:
                cfg.enable_stream(ob.StreamType.DEPTH)

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
            color_frame = ret.color_frame()
            if color_frame is None:
                raise RuntimeError(f"{self} failed to get color frame for shape inference.")
            h, w, _ = np.asanyarray(color_frame.get_data()).shape
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = h, w
                self.capture_width, self.capture_height = w, h
            else:
                self.width, self.height = w, h
                self.capture_width, self.capture_height = w, h

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self._ctx = ob.Context()
        self._device = self._match_device(self._ctx)
        self._pipeline = ob.Pipeline(self._device)
        self._config = self._build_config()

        try:
            self._pipeline.start(self._config)
        except Exception as e:
            self._pipeline = None
            self._device = None
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

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed = image
        if not is_depth and self.color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)

        return processed

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        frameset = self._pipeline.wait_for_frames(200)
        if frameset is None:
            raise RuntimeError(f"{self} read failed (no frameset).")

        color_frame = frameset.color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} color frame is None.")
        color_image_raw = np.asanyarray(color_frame.get_data())
        color_image = self._postprocess_image(color_image_raw, is_depth=False)

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
        depth_frame = frameset.depth_frame()
        if depth_frame is None:
            raise RuntimeError(f"{self} depth frame is None.")
        depth_raw = np.asanyarray(depth_frame.get_data())
        if depth_raw.dtype != np.uint16:
            depth_raw = depth_raw.astype(np.uint16, copy=False)

        depth_image = self._postprocess_image(depth_raw, is_depth=True)
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

                c = frameset.color_frame()
                if c is not None:
                    color_image_raw = np.asanyarray(c.get_data())
                    color_image = self._postprocess_image(color_image_raw, is_depth=False)

                if self.use_depth:
                    d = frameset.depth_frame()
                    if d is not None:
                        depth_raw = np.asanyarray(d.get_data())
                        if depth_raw.dtype != np.uint16:
                            depth_raw = depth_raw.astype(np.uint16, copy=False)
                        depth_image = self._postprocess_image(depth_raw, is_depth=True)

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
            self._ctx = None

        logger.info(f"{self} disconnected.")


