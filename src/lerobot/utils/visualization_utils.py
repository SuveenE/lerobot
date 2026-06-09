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

import logging
import numbers
import os
import queue
import threading
import time
from typing import Any

import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR

# Tracks the last time images were logged, used to rate-limit the live stream
# independently of the (higher) control/recording fps.
_last_image_log_t = 0.0


def _should_skip_images() -> bool:
    """Rate-limit image logging to LEROBOT_RERUN_MAX_FPS (if set).

    Scalars are always logged; only image streaming is throttled, since images
    dominate bandwidth when streaming many cameras over the network.
    """
    max_fps = os.getenv("LEROBOT_RERUN_MAX_FPS")
    if not max_fps:
        return False

    global _last_image_log_t
    min_dt = 1.0 / float(max_fps)
    now = time.perf_counter()
    if now - _last_image_log_t < min_dt:
        return True
    _last_image_log_t = now
    return False


def _is_camera_streamed(key: str) -> bool:
    """Whitelist which camera views are streamed via LEROBOT_RERUN_CAMERAS (if set).

    Set e.g. LEROBOT_RERUN_CAMERAS="front,back,top" to only stream those views and
    skip the rest. Matching is case-insensitive against the last segment of the key
    (e.g. "observation.images.front" -> "front"). Recording is unaffected.
    """
    whitelist = os.getenv("LEROBOT_RERUN_CAMERAS")
    if not whitelist:
        return True

    allowed = {name.strip().lower() for name in whitelist.split(",") if name.strip()}
    cam_name = str(key).split(".")[-1].lower()
    return cam_name in allowed


def _log_image(key: str, arr: np.ndarray) -> None:
    """Log an image to Rerun, optionally downscaling and/or JPEG-compressing.

    Controlled via env vars (display-only, does not affect the recorded dataset):
    - LEROBOT_RERUN_IMAGE_MAX_WIDTH: downscale frames wider than this (keeps aspect).
    - LEROBOT_RERUN_JPEG_QUALITY: 1-100, send JPEG instead of raw RGB (huge bandwidth win).
    """
    # Convert CHW -> HWC when needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    max_w = os.getenv("LEROBOT_RERUN_IMAGE_MAX_WIDTH")
    quality = os.getenv("LEROBOT_RERUN_JPEG_QUALITY")

    can_process = arr.ndim == 3 and arr.shape[2] in (3, 4) and arr.dtype == np.uint8
    if (max_w or quality) and can_process:
        import cv2

        img = arr[:, :, :3] if arr.shape[2] == 4 else arr

        if max_w:
            h, w = img.shape[:2]
            target_w = int(max_w)
            if w > target_w:
                target_h = int(round(h * target_w / w))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if quality:
            # cv2 treats input as BGR, so convert from RGB to keep colors correct.
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if ok:
                rr.log(key, rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"), static=True)
                return

        rr.log(key, rr.Image(img), static=True)
        return

    rr.log(key, rr.Image(arr), static=True)


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)

    # Stream to a Rerun viewer running on another machine in the network.
    # Start the viewer there with `rerun --port 9876` and set e.g.
    # LEROBOT_RERUN_CONNECT="rerun+http://192.168.1.50:9876/proxy".
    remote = os.getenv("LEROBOT_RERUN_CONNECT")
    if remote:
        rr.connect_grpc(url=remote)
    else:
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
    """
    skip_images = _should_skip_images()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                elif not skip_images and _is_camera_streamed(key):
                    _log_image(key, v)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


class AsyncRerunLogger:
    """Logs observation/action data to Rerun on a background thread.

    Logging many raw camera images synchronously can take tens of milliseconds per
    frame, which blows the control/record loop budget and drops frames. Visualization
    only needs the latest frame, so this uses a drop-latest queue (maxsize=1): if the
    logger thread is still busy when a newer frame arrives, the stale frame is discarded
    and the loop never blocks or accumulates lag.

    The caller must not mutate the passed observation/action arrays after calling
    ``log`` (the record loop builds fresh dicts each iteration, so this holds there).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._loop,
            name="rerun_logger",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            observation, action = item
            try:
                log_rerun_data(observation=observation, action=action)
            except Exception as e:
                logging.warning(f"Rerun logging failed: {e}")
            finally:
                self._queue.task_done()

    def log(self, observation: dict[str, Any] | None = None, action: dict[str, Any] | None = None) -> None:
        """Enqueue the latest frame for logging, dropping any unprocessed older frame."""
        try:
            self._queue.put_nowait((observation, action))
        except queue.Full:
            # Drop the stale frame still waiting in the queue and replace it.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((observation, action))
            except queue.Full:
                pass

    def close(self) -> None:
        # Ensure the sentinel gets in even if a frame is queued.
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logging.warning("Rerun logger did not stop within 5s")
