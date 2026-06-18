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

import atexit
import logging
import numbers
import os
import threading
import time
from typing import Any

import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR
from .udp_video import UDPVideoSender, get_udp_video_sender_from_env

# Tracks the last time images were logged, used to rate-limit the live stream
# independently of the (higher) control/recording fps.
_last_image_log_t = 0.0

# Optional low-latency UDP video side-channel (see udp_video.py). Created lazily
# by init_rerun() when LEROBOT_VIDEO_UDP is set; None means "don't stream video".
_udp_sender: "UDPVideoSender | None" = None

# Guards against installing our bounded shutdown handler more than once.
_bounded_shutdown_installed = False

# Background logger that decouples rerun logging from the caller (e.g. the record
# loop) so image serialization / socket flushes never block the control loop.
# Created lazily by init_rerun(); None means "log synchronously".
_async_logger: "_AsyncRerunLogger | None" = None


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


def _rerun_images_enabled() -> bool:
    """Whether camera images should be streamed to Rerun.

    Acts as the switch between the Rerun (gRPC) and UDP video backends:
    - LEROBOT_RERUN_IMAGES=1 forces images to Rerun (use with UDP to get both).
    - LEROBOT_RERUN_IMAGES=0 forces images off for Rerun.
    - Unset (default): images go to Rerun unless the UDP video backend is active
      (LEROBOT_VIDEO_UDP set), in which case UDP takes over the video stream.

    Scalars (joint positions, torques, ...) always go to Rerun regardless.
    """
    force = os.getenv("LEROBOT_RERUN_IMAGES")
    if force is not None:
        return force != "0"
    return _udp_sender is None


def _should_skip_actions() -> bool:
    """Skip logging action data unless LEROBOT_RERUN_ACTIONS=1.

    Each action vector is logged element-by-element (one ``rr.log`` per scalar),
    so a multi-DoF bimanual setup floods the stream with many small messages every
    frame. When streaming to a remote viewer (e.g. the teleop PC) this adds load
    without much value, so actions are skipped by default. Set
    LEROBOT_RERUN_ACTIONS=1 to log them. Recording is unaffected.
    """
    return os.getenv("LEROBOT_RERUN_ACTIONS", "0") == "0"


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


def _bounded_rerun_shutdown() -> None:
    """Flush and tear down the Rerun stream without blocking process exit forever.

    Rerun installs an ``atexit`` handler (``rerun.rerun_shutdown``) that flushes the
    recording stream at interpreter shutdown with *no timeout*. When streaming to a
    remote/slow gRPC viewer, that flush can block indefinitely — and since it runs in
    native (Rust) code during shutdown, SIGINT (Ctrl-C) won't interrupt it, so the
    whole process appears to hang after recording (e.g. right after `push_to_hub`).

    We run the disconnect/flush in a daemon thread and wait at most
    ``LEROBOT_RERUN_SHUTDOWN_TIMEOUT_SEC`` (default 10s). If it doesn't finish, we
    abandon the flush and let the process exit anyway. The leftover daemon thread and
    native Rerun threads don't block interpreter teardown.
    """
    timeout_s = float(os.getenv("LEROBOT_RERUN_SHUTDOWN_TIMEOUT_SEC", "10"))

    def _work() -> None:
        global _async_logger, _udp_sender
        if _async_logger is not None:
            # Stop the background logger first so its last rr.log calls land before
            # we tear down the stream (and so nothing logs concurrently with disconnect).
            _async_logger.stop()
            _async_logger = None
        if _udp_sender is not None:
            _udp_sender.close()
            _udp_sender = None
        try:
            rr.disconnect()
        except Exception as e:  # nosec B110 - best-effort cleanup at exit
            logging.debug(f"Rerun disconnect during shutdown failed: {e}")
        try:
            rr.rerun_shutdown()
        except Exception as e:  # nosec B110 - best-effort cleanup at exit
            logging.debug(f"Rerun shutdown failed: {e}")

    thread = threading.Thread(target=_work, name="rerun_teardown", daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        logging.warning(
            f"Rerun did not flush/disconnect within {timeout_s:.0f}s "
            "(viewer slow or gone); abandoning flush to avoid hanging on exit."
        )


def _install_bounded_rerun_shutdown() -> None:
    """Swap Rerun's unbounded atexit flush for our bounded one (idempotent)."""
    global _bounded_shutdown_installed
    if _bounded_shutdown_installed:
        return
    try:
        rr.unregister_shutdown()
    except Exception as e:  # nosec B110 - non-fatal, we still register our handler
        logging.debug(f"Could not unregister Rerun's default shutdown handler: {e}")
    atexit.register(_bounded_rerun_shutdown)
    _bounded_shutdown_installed = True


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    # Batch more bytes before flushing. 8 KB is tiny for image streaming and causes
    # frequent (potentially blocking) flushes; a larger batch reduces flush pressure,
    # which matters most when streaming to a remote viewer over gRPC.
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "1048576")
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

    # Prevent Rerun's unbounded shutdown flush from hanging the process on exit.
    _install_bounded_rerun_shutdown()

    # Decouple logging from the caller's hot loop unless explicitly disabled.
    global _async_logger
    if _async_logger is None and os.getenv("LEROBOT_RERUN_ASYNC", "1") != "0":
        _async_logger = _AsyncRerunLogger()
        _async_logger.start()

    # Optionally stream a low-latency video preview over UDP alongside Rerun.
    global _udp_sender
    if _udp_sender is None:
        _udp_sender = get_udp_video_sender_from_env()


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def _log_rerun_data_sync(
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
                elif _is_camera_streamed(key):
                    # UDP video has its own throttle and is independent of the
                    # Rerun image rate-limit, so send it before the skip check.
                    if _udp_sender is not None:
                        _udp_sender.maybe_send(key, v)
                    if not skip_images and _rerun_images_enabled():
                        _log_image(key, v)

    if action and not _should_skip_actions():
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


def _snapshot(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a shallow copy of ``data`` with numpy arrays copied.

    Used before handing data to the background logger: the caller (e.g. a camera
    driver) may reuse/overwrite its buffers on the next frame, so we copy arrays to
    avoid logging torn or stale frames from another thread.
    """
    if not data:
        return data
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}


class _AsyncRerunLogger:
    """Logs observation/action to Rerun on a background thread.

    The caller submits the latest snapshot and returns immediately; the worker
    serializes and logs it. Only the most recent snapshot is kept — if the worker
    falls behind (slow serialization or a blocking remote flush), older snapshots
    are dropped. This keeps the live view best-effort while guaranteeing the
    control loop (and any sibling threads, like the UDP receiver) are never
    starved by display work.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: tuple[dict | None, dict | None] | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rerun-logger", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, observation: dict | None, action: dict | None) -> None:
        """Hand the latest data to the worker without blocking. Drops any prior
        un-logged snapshot (the live view is allowed to skip frames)."""
        snap = (_snapshot(observation), _snapshot(action))
        with self._lock:
            self._pending = snap
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                continue
            observation, action = item
            try:
                _log_rerun_data_sync(observation, action)
            except Exception as e:  # nosec B110 - display is best-effort
                logging.debug(f"Async rerun logging failed: {e}")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """Log observation/action data to Rerun for real-time visualization.

    When a background logger is active (started by ``init_rerun``; the default for
    the record/teleop loops), the data is snapshotted and handed off to a worker
    thread so this call returns immediately and never blocks the control loop on
    image serialization or a remote viewer flush. Set ``LEROBOT_RERUN_ASYNC=0`` to
    force synchronous logging. See ``_log_rerun_data_sync`` for the logging details.
    """
    if _async_logger is not None:
        _async_logger.submit(observation, action)
    else:
        _log_rerun_data_sync(observation, action)
