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

"""Low-latency UDP video side-channel for live previews.

Rerun streams over gRPC/TCP, which is reliable and ordered: when the link can't
keep up it buffers and retransmits rather than dropping, so a live camera preview
falls further and further behind. This module provides an alternative video path
that sends JPEG-encoded frames over UDP. UDP drops stale/late packets instead of
queuing them, so the viewer always converges on the most recent frame.

It is display-only and completely independent of the recorded dataset.

Wire format (network byte order), one datagram per chunk so frames larger than the
MTU are split at the application layer (instead of relying on IP fragmentation,
where a single lost fragment drops the whole datagram):

    magic        4s   b"LRV1"
    frame_id     I    per-camera monotonic counter (wraps); identifies a frame
    total_chunks H    number of chunks the frame was split into
    chunk_idx    H    index of this chunk within the frame
    chunk_len    H    number of payload bytes in this datagram
    name_len     B    length of the camera name
    name         {name_len}s   short camera name (e.g. "front")
    payload      {chunk_len}s  slice of the JPEG byte stream

The receiver reassembles chunks per (camera, frame_id); if a newer frame_id
arrives before the current one completes, the incomplete frame is dropped
(newest wins).
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time

import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"LRV1"
_HEADER_FMT = "!4sIHHHB"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Keep each datagram under a typical Ethernet MTU (1500) to avoid IP fragmentation.
MAX_DATAGRAM = 1400


def _parse_endpoint(value: str) -> tuple[str, int]:
    """Parse "host:port" into (host, port). Raises ValueError on bad input."""
    host, sep, port = value.rpartition(":")
    if not sep or not host:
        raise ValueError(f"Expected 'host:port', got {value!r}")
    return host, int(port)


def _to_hwc_uint8_rgb(arr: np.ndarray) -> np.ndarray | None:
    """Normalize an image array to a contiguous HWC uint8 RGB(A->RGB) array.

    Returns None if the array doesn't look like a displayable image.
    """
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or arr.dtype != np.uint8:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


class UDPVideoSender:
    """Encodes camera frames to JPEG and streams them to a remote viewer over UDP.

    One sender targets a single host:port. Call :meth:`maybe_send` with the camera
    key and frame; encoding and transmission happen inline (cheap, non-blocking
    UDP sends), so call it from a background thread (e.g. the async rerun logger).
    """

    def __init__(
        self,
        host: str,
        port: int,
        quality: int = 50,
        max_fps: float | None = None,
        max_width: int | None = None,
    ) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # A larger send buffer smooths out bursts of chunks for a multi-camera frame.
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except OSError as e:  # nosec B110 - best effort, default buffer still works
            logger.debug(f"Could not enlarge UDP send buffer: {e}")
        self._quality = int(quality)
        self._min_dt = (1.0 / max_fps) if max_fps else 0.0
        self._max_width = int(max_width) if max_width else None
        self._frame_ids: dict[str, int] = {}
        self._last_sent: dict[str, float] = {}

    @staticmethod
    def short_name(key: str) -> str:
        """Map a full key like "observation.images.front" to "front"."""
        return str(key).split(".")[-1]

    def maybe_send(self, key: str, arr: np.ndarray) -> None:
        """Encode and send a frame, honoring the optional per-camera FPS throttle."""
        name = self.short_name(key)

        if self._min_dt:
            now = time.perf_counter()
            if now - self._last_sent.get(name, 0.0) < self._min_dt:
                return
            self._last_sent[name] = now

        try:
            self._send(name, arr)
        except Exception as e:  # nosec B110 - preview is best-effort, never crash the loop
            logger.debug(f"UDP video send failed for {name}: {e}")

    def _send(self, name: str, arr: np.ndarray) -> None:
        import cv2

        img = _to_hwc_uint8_rgb(arr)
        if img is None:
            return

        if self._max_width is not None:
            h, w = img.shape[:2]
            if w > self._max_width:
                target_h = int(round(h * self._max_width / w))
                img = cv2.resize(img, (self._max_width, target_h), interpolation=cv2.INTER_AREA)

        # cv2 encodes BGR; our frames are RGB, so convert to keep colors correct.
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return

        payload = buf.tobytes()
        name_bytes = name.encode("utf-8")[:255]
        chunk_size = MAX_DATAGRAM - HEADER_SIZE - len(name_bytes)
        if chunk_size <= 0:
            return

        total_chunks = max(1, (len(payload) + chunk_size - 1) // chunk_size)
        if total_chunks > 0xFFFF:  # too big to address with a uint16 chunk index
            return

        frame_id = (self._frame_ids.get(name, 0) + 1) & 0xFFFFFFFF
        self._frame_ids[name] = frame_id

        for idx in range(total_chunks):
            chunk = payload[idx * chunk_size : (idx + 1) * chunk_size]
            header = struct.pack(
                _HEADER_FMT, MAGIC, frame_id, total_chunks, idx, len(chunk), len(name_bytes)
            )
            self._sock.sendto(header + name_bytes + chunk, self._addr)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def get_udp_video_sender_from_env() -> UDPVideoSender | None:
    """Build a sender from environment variables, or None if disabled.

    Env vars:
    - LEROBOT_VIDEO_UDP: "host:port" to enable streaming (required to enable).
    - LEROBOT_VIDEO_UDP_JPEG_QUALITY: JPEG quality 1-100 (default 50).
    - LEROBOT_VIDEO_UDP_MAX_FPS: per-camera send rate cap (default: unthrottled).
    - LEROBOT_VIDEO_UDP_MAX_WIDTH: downscale frames wider than this (keeps aspect).
    """
    endpoint = os.getenv("LEROBOT_VIDEO_UDP")
    if not endpoint:
        return None

    try:
        host, port = _parse_endpoint(endpoint)
    except ValueError as e:
        logger.warning(f"Invalid LEROBOT_VIDEO_UDP={endpoint!r} ({e}); UDP video disabled.")
        return None

    quality = int(os.getenv("LEROBOT_VIDEO_UDP_JPEG_QUALITY", "50"))
    max_fps_env = os.getenv("LEROBOT_VIDEO_UDP_MAX_FPS")
    max_fps = float(max_fps_env) if max_fps_env else None
    max_width_env = os.getenv("LEROBOT_VIDEO_UDP_MAX_WIDTH")
    max_width = int(max_width_env) if max_width_env else None

    logger.info(
        f"UDP video preview enabled -> {host}:{port} "
        f"(quality={quality}, max_fps={max_fps or 'uncapped'}, max_width={max_width or 'native'})"
    )
    return UDPVideoSender(host, port, quality=quality, max_fps=max_fps, max_width=max_width)
