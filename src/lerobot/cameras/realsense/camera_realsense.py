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
Provides the RealSenseCamera class for capturing frames from Intel RealSense cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    logging.info(f"Could not import realsense: {e}")

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


class RealSenseCamera(Camera):
    """
    Manages interactions with Intel RealSense cameras for frame and depth recording.

    This class provides an interface similar to `OpenCVCamera` but tailored for
    RealSense devices, leveraging the `pyrealsense2` library. It uses the camera's
    unique serial number for identification, offering more stability than device
    indices, especially on Linux. It also supports capturing depth maps alongside
    color frames.

    Use the provided utility script to find available camera indices and default profiles:
    ```bash
    lerobot-find-cameras realsense
    ```

    A `RealSenseCamera` instance requires a configuration object specifying the
    camera's serial number or a unique device name. If using the name, ensure only
    one camera with that name is connected.

    The camera's default settings (FPS, resolution, color mode) from the stream
    profile are used unless overridden in the configuration.

    Example:
        ```python
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Basic usage with serial number
        config = RealSenseCameraConfig(serial_number_or_name="0123456789") # Replace with actual SN
        camera = RealSenseCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera using
        camera.disconnect()

        # Example with depth capture and custom settings
        custom_config = RealSenseCameraConfig(
            serial_number_or_name="0123456789", # Replace with actual SN
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = RealSenseCamera(custom_config)
        depth_camera.connect()

        # Read 1 depth frame
        depth_map = depth_camera.read_depth()

        # Example using a unique camera name
        name_config = RealSenseCameraConfig(serial_number_or_name="Intel RealSense D435") # If unique
        name_camera = RealSenseCamera(name_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: RealSenseCameraConfig):
        """
        Initializes the RealSenseCamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        # Ensure serial_number_or_name is a string (handles case where JSON parsing returns int)
        serial_str = str(config.serial_number_or_name) if not isinstance(config.serial_number_or_name, str) else config.serial_number_or_name
        
        if serial_str.isdigit():
            self.serial_number = serial_str
        else:
            self.serial_number = self._find_serial_number_from_name(serial_str)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.rs_pipeline: rs.pipeline | None = None
        self.rs_profile: rs.pipeline_profile | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        # Depth frame management for async reading
        self.latest_depth_frame: np.ndarray | None = None
        self.new_depth_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streams are active."""
        return self.rs_pipeline is not None and self.rs_profile is not None

    def connect(self, warmup: bool = True):
        """
        Connects to the RealSense camera specified in the configuration.

        Initializes the RealSense pipeline, configures the required streams (color
        and optionally depth), starts the pipeline, and validates the actual stream settings.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration is invalid (e.g., missing serial/name, name not unique).
            ConnectionError: If the camera is found but fails to start the pipeline or no RealSense devices are detected at all.
            RuntimeError: If the pipeline starts but fails to apply requested settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        self._configure_rs_pipeline_config(rs_config)

        try:
            self.rs_profile = self.rs_pipeline.start(rs_config)
        except RuntimeError as e:
            self.rs_profile = None
            self.rs_pipeline = None
            
            # Try to get available stream profiles to suggest alternatives
            suggested_resolutions = []
            try:
                context = rs.context()
                devices = context.query_devices()
                for device in devices:
                    if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                        sensors = device.query_sensors()
                        for sensor in sensors:
                            profiles = sensor.get_stream_profiles()
                            for profile in profiles:
                                if profile.is_video_stream_profile():
                                    vprofile = profile.as_video_stream_profile()
                                    stream_type = vprofile.stream_type()
                                    if stream_type == rs.stream.color or (self.use_depth and stream_type == rs.stream.depth):
                                        suggested_resolutions.append(
                                            f"{vprofile.width()}x{vprofile.height()}@{vprofile.fps()}fps"
                                        )
                        break
            except Exception:
                pass  # If we can't query profiles, just skip the suggestion
            
            config_str = f"serial={self.serial_number}"
            if self.width and self.height and self.fps:
                config_str += f", resolution={self.capture_width}x{self.capture_height}@{self.fps}fps"
                if self.use_depth:
                    config_str += " (with depth)"
            
            error_msg = (
                f"Failed to open {self} with configuration: {config_str}. "
                f"Error: {str(e)}."
            )
            
            if suggested_resolutions:
                # Remove duplicates and sort
                unique_resolutions = sorted(set(suggested_resolutions))
                error_msg += f"\nSupported resolutions for this camera include: {', '.join(unique_resolutions[:10])}"
            
            error_msg += (
                f"\nRun `lerobot-find-cameras realsense` to find all available cameras and supported resolutions."
            )
            
            raise ConnectionError(error_msg) from e

        self._configure_capture_settings()

        if warmup:
            time.sleep(
                5
                # NOTE(Steven): RS cameras need a bit of time to warm up before the first read. If we don't wait, the first read from the warmup will raise.
            )
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Intel RealSense cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (serial number), 'name',
            firmware version, USB type, and other available specs, and the default profile properties (width, height, fps, format).

        Raises:
            OSError: If pyrealsense2 is not installed.
            ImportError: If pyrealsense2 is not installed.
        """
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            camera_info = {
                "name": device.get_info(rs.camera_info.name),
                "type": "RealSense",
                "id": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }

            # Get stream profiles for each sensor
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()

                for profile in profiles:
                    if profile.is_video_stream_profile() and profile.is_default():
                        vprofile = profile.as_video_stream_profile()
                        stream_info = {
                            "stream_type": vprofile.stream_name(),
                            "format": vprofile.format().name,
                            "width": vprofile.width(),
                            "height": vprofile.height(),
                            "fps": vprofile.fps(),
                        }
                        camera_info["default_stream_profile"] = stream_info

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds the serial number for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [
            cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No RealSense camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["serial_number"] for dev in found_devices]
            raise ValueError(
                f"Multiple RealSense cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        serial_number = str(found_devices[0]["serial_number"])
        return serial_number

    def _find_compatible_stream_profiles(self, requested_width: int, requested_height: int, requested_fps: int):
        """Find compatible stream profiles for color and depth streams.
        
        RealSense cameras may have constraints on which stream profiles can be enabled
        simultaneously. This method queries the device to find compatible profiles that
        match the requested resolution and FPS.
        
        Returns:
            tuple: (color_profile, depth_profile) or (color_profile, None) if depth not needed.
                   Returns (None, None) if device not found or profiles can't be queried.
        """
        try:
            context = rs.context()
            devices = context.query_devices()
            
            device = None
            for d in devices:
                if d.get_info(rs.camera_info.serial_number) == self.serial_number:
                    device = d
                    break
            
            if device is None:
                logger.warning(f"Device with serial {self.serial_number} not found when querying stream profiles")
                return None, None
            
            sensors = device.query_sensors()
            color_sensor = None
            depth_sensor = None
            
            # Find the sensors
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()
                for profile in profiles:
                    if profile.is_video_stream_profile():
                        vprofile = profile.as_video_stream_profile()
                        if vprofile.stream_type() == rs.stream.color and color_sensor is None:
                            color_sensor = sensor
                        elif vprofile.stream_type() == rs.stream.depth and depth_sensor is None:
                            depth_sensor = sensor
            
            color_profile = None
            depth_profile = None
            
            # Find matching color profile - REQUIRED: RGB8 format only
            if color_sensor:
                profiles = color_sensor.get_stream_profiles()
                for profile in profiles:
                    if profile.is_video_stream_profile():
                        vprofile = profile.as_video_stream_profile()
                        if (vprofile.stream_type() == rs.stream.color and
                            vprofile.width() == requested_width and
                            vprofile.height() == requested_height and
                            vprofile.fps() == requested_fps and
                            vprofile.format() == rs.format.rgb8):
                            color_profile = profile
                            break
            
            # Find matching depth profile (if needed) - REQUIRED: Z16 format only
            if self.use_depth and depth_sensor:
                profiles = depth_sensor.get_stream_profiles()
                for profile in profiles:
                    if profile.is_video_stream_profile():
                        vprofile = profile.as_video_stream_profile()
                        if (vprofile.stream_type() == rs.stream.depth and
                            vprofile.width() == requested_width and
                            vprofile.height() == requested_height and
                            vprofile.fps() == requested_fps and
                            vprofile.format() == rs.format.z16):
                            depth_profile = profile
                            break
            
            # print both profiles verbosely
            # Log selected color profile details
            if color_profile is not None:
                logger.info(
                    f"Color profile: {color_profile} | "
                    f"Format: {color_profile.format()} | "
                    f"Resolution: {color_profile.width()}x{color_profile.height()} | "
                    f"FPS: {color_profile.fps()}"
                )

            # Log selected depth profile details
            if depth_profile is not None:
                logger.info(
                    f"Depth profile: {depth_profile} | "
                    f"Format: {depth_profile.format()} | "
                    f"Resolution: {depth_profile.width()}x{depth_profile.height()} | "
                    f"FPS: {depth_profile.fps()}"
                )
            return color_profile, depth_profile
        except Exception as e:
            logger.debug(f"Error querying stream profiles: {e}")
            return None, None

    def _configure_rs_pipeline_config(self, rs_config):
        """Creates and configures the RealSense pipeline configuration object."""
        rs.config.enable_device(rs_config, self.serial_number)

        if self.width and self.height and self.fps:
            # REQUIRED: RGB8 format for color - try to enable directly like minimal example
            try:
                rs_config.enable_stream(
                    rs.stream.color,
                    self.capture_width,
                    self.capture_height,
                    rs.format.rgb8,
                    self.fps
                )
            except RuntimeError as e:
                raise ValueError(
                    f"{self} does not support RGB8 format at {self.capture_width}x{self.capture_height}@{self.fps}fps. "
                    f"RGB8 format is required for color stream. Error: {e}. "
                    f"Run `lerobot-find-cameras realsense` to see available formats."
                )
            
            # REQUIRED: Z16 format for depth (if depth is enabled)
            if self.use_depth:
                try:
                    rs_config.enable_stream(
                        rs.stream.depth,
                        self.capture_width,
                        self.capture_height,
                        rs.format.z16,
                        self.fps
                    )
                except RuntimeError as e:
                    raise ValueError(
                        f"{self} does not support Z16 format at {self.capture_width}x{self.capture_height}@{self.fps}fps. "
                        f"Z16 format is required for depth stream. Error: {e}. "
                        f"Run `lerobot-find-cameras realsense` to see available formats."
                    )
        else:
            # When width/height/fps not specified, use default but still require RGB8/Z16
            # Try to enable with explicit formats - this will fail if formats not supported
            try:
                rs_config.enable_stream(rs.stream.color, rs.format.rgb8)
            except RuntimeError as e:
                raise ValueError(
                    f"{self} does not support RGB8 format for color stream. "
                    f"RGB8 format is required. Error: {e}"
                )
            if self.use_depth:
                try:
                    rs_config.enable_stream(rs.stream.depth, rs.format.z16)
                except RuntimeError as e:
                    raise ValueError(
                        f"{self} does not support Z16 format for depth stream. "
                        f"Z16 format is required. Error: {e}"
                    )

    def _configure_capture_settings(self) -> None:
        """Sets fps, width, and height from device stream if not already configured.

        Uses the color stream profile to update unset attributes. Handles rotation by
        swapping width/height when needed. Original capture dimensions are always stored.

        Raises:
            DeviceNotConnectedError: If device is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"Cannot validate settings for {self} as it is not connected.")

        stream = self.rs_profile.get_stream(
            rs.stream.color).as_video_stream_profile()

        if self.fps is None:
            self.fps = stream.fps()

        if self.width is None or self.height is None:
            actual_width = int(round(stream.width()))
            actual_height = int(round(stream.height()))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    def _process_depth_frame(self, depth_frame: Any) -> np.ndarray:
        """Process RealSense depth frame to convert get_data() units to distance values.
        
        Similar to Orbbec's _process_depth_frame, converts depth data from get_data()
        to distance values in millimeters using get_units() scaling factor.
        get_units() provides the scaling factor to convert from get_data() units to meters.
        
        Args:
            depth_frame: RealSense depth frame
            
        Returns:
            np.ndarray: Depth map as uint16 array with distance values in millimeters
        """
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # Get depth units: scaling factor to convert from get_data() units to meters
        # get_units() provides the scaling factor to use when converting from get_data() units to meters
        depth_units = depth_frame.get_units()
        
        # Extract raw depth data from get_data() (units depend on camera format)
        raw_depth_data = np.asanyarray(depth_frame.get_data())
        raw_depth_image = np.resize(raw_depth_data, (height, width))
        
        # Convert get_data() units to distance matrix (in meters)
        # Similar to Orbbec: depth_data * depth_scale = distance in millimeters
        # For RealSense: get_data() * depth_units = distance in meters
        distance_matrix_m = raw_depth_image.astype(np.float32) * depth_units
        
        # Convert to millimeters for storage (like Orbbec output format)
        distance_matrix_mm = distance_matrix_m * 1000.0
        
        # Convert back to uint16 for storage (preserve precision, values in millimeters)
        depth_data = distance_matrix_mm.astype(np.uint16)
        
        depth_data = self._postprocess_image(depth_data, depth_frame=True)
        return depth_data

    def read_depth(self, timeout_ms: int = 600) -> np.ndarray:
        """
        Reads a single frame (depth) synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames (depth)
        from the camera hardware via the RealSense pipeline.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            np.ndarray: The depth map as a NumPy array (height, width)
                  of type `np.uint16` (distance values in millimeters) and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        start_time = time.perf_counter()

        # Use wait_for_frames() like the minimal example - blocking call that returns frameset
        try:
            frameset = self.rs_pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as e:
            raise RuntimeError(f"{self} read_depth failed: {e}")

        if frameset is None:
            raise RuntimeError(f"{self} read_depth failed (no frameset).")

        depth_frame = frameset.get_depth_frame()
        if not depth_frame:
            raise RuntimeError(f"{self} read_depth failed: depth frame is None.")
        
        # Process depth frame to convert raw pixels to distance values (millimeters)
        depth_map_processed = self._process_depth_frame(depth_frame)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read_depth took: {read_duration_ms:.1f}ms")

        return depth_map_processed

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 600) -> np.ndarray:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames (color)
        from the camera hardware via the RealSense pipeline.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
            ValueError: If an invalid `color_mode` is requested.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        # Use wait_for_frames() like the minimal example - blocking call that returns frameset
        try:
            frameset = self.rs_pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as e:
            raise RuntimeError(f"{self} read failed: {e}")

        if frameset is None:
            raise RuntimeError(f"{self} read failed (no frameset).")

        color_frame = frameset.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self} read failed: color frame is None.")
        
        # RGB8 format: 3 bytes per pixel (24 bits)
        # get_data() returns uint8 array: shape (height * width * 3,)
        # Reshape to (height, width, 3) - already RGB format
        height = color_frame.get_height()
        width = color_frame.get_width()
        frame_data = np.asanyarray(color_frame.get_data())
        color_image_raw = np.resize(frame_data, (height, width, 3))

        color_image_processed = self._postprocess_image(
            color_image_raw, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return color_image_processed

    def _postprocess_image(
        self, image: np.ndarray, color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> np.ndarray:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.

        Args:
            image (np.ndarray): The raw image frame (expected RGB format from RealSense).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                             uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame according to `self.color_mode` and `self.rotation`.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """

        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(
                    f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image
        # Only apply color conversion for color frames, not depth frames
        if not depth_frame and self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads frameset from pipeline with 500ms timeout
        2. Extracts color frame and optionally depth frame from the SAME frameset
        3. Stores results in latest_frame and latest_depth_frame (thread-safe)
        4. Sets new_frame_event and new_depth_frame_event to notify listeners

        This ensures color and depth are synchronized from the same capture instant.

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        while not self.stop_event.is_set():
            try:
                # Use wait_for_frames() like the minimal example - blocking call that returns frameset
                try:
                    frameset = self.rs_pipeline.wait_for_frames(500)
                except RuntimeError:
                    logger.debug(
                        f"{self} failed to get frameset in background thread (timeout or error)")
                    continue

                if frameset is None:
                    logger.debug(
                        f"{self} failed to get frameset in background thread")
                    continue

                # Extract and process color and depth frames (same frameset)
                color_image = None
                depth_image = None

                color_frame = frameset.get_color_frame()
                if color_frame:
                    # RGB8 format: 3 bytes per pixel (24 bits)
                    height = color_frame.get_height()
                    width = color_frame.get_width()
                    frame_data = np.asanyarray(color_frame.get_data())
                    color_image_raw = np.resize(frame_data, (height, width, 3))
                    
                    color_image = self._postprocess_image(color_image_raw)

                if self.use_depth:
                    depth_frame = frameset.get_depth_frame()
                    if depth_frame:
                        # Process depth frame to convert raw pixels to distance values (millimeters)
                        depth_image = self._process_depth_frame(depth_frame)
                        # Log center pixel depth in mm
                        # h, w = depth_image.shape
                        # center_depth_mm = depth_image[h // 2, w // 2]
                        # logger.info(f"{self} center depth: {center_depth_mm} mm")

                # Atomically publish both results under the same lock
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
                logger.warning(
                    f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop,
                             args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 600) -> np.ndarray:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray:
            The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(
                f"Internal error: Event set but no frame available for {self}.")

        return frame

    def async_read_depth(self, timeout_ms: float = 600) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads the latest available depth frame asynchronously.

        This method retrieves the most recent depth frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a depth frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The latest captured color frame (height, width, channels)
                - The latest captured depth frame as a NumPy array (height, width)
                  of type `np.uint16` (raw depth values in millimeters), processed according
                  to configuration (rotation applied).

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If depth stream is not enabled or if the background thread 
                died unexpectedly.
            TimeoutError: If no depth frame becomes available within the specified timeout.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame with 'async_read_depth()'. "
                f"Depth stream is not enabled for {self}."
            )

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_depth_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for depth frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

            depth_frame = self.latest_depth_frame
            self.new_depth_frame_event.clear()

        if depth_frame is None:
            raise RuntimeError(
                f"Internal error: Event set but no depth frame available for {self}.")

        if frame is None:
            raise RuntimeError(
                f"Internal error: Event set but no frame available for {self}.")

        return frame, depth_frame

    def disconnect(self):
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Stops the background read thread (if running) and stops the RealSense pipeline.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (pipeline not running).
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_profile = None

        logger.info(f"{self} disconnected.")
