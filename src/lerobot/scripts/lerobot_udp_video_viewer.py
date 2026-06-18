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

"""Low-latency UDP video viewer for the teleop PC.

Receives JPEG frames streamed by :mod:`lerobot.utils.udp_video` (enabled on the
robot PC via ``LEROBOT_VIDEO_UDP=<this_host>:<port>``) and displays them in one
OpenCV window per camera. Because the transport is UDP, late/stale packets are
dropped instead of queued, so the view always converges on the most recent frame
instead of lagging behind like the Rerun gRPC stream.

Run this on the machine whose address you passed to ``LEROBOT_VIDEO_UDP``:

```shell
lerobot-udp-video-viewer --port 5005
```

Then start recording on the robot PC with, e.g.:

```shell
LEROBOT_VIDEO_UDP=172.16.0.89:5005 LEROBOT_RERUN_CAMERAS="top,right,left" lerobot-record ...
```

Press ``q`` (or ``ESC``) in any window to quit.
"""

import argparse
import logging
import socket
import struct
import threading
import time

import numpy as np

from lerobot.utils.udp_video import HEADER_SIZE, MAGIC, _HEADER_FMT

logger = logging.getLogger(__name__)


class _FrameAssembler:
    """Reassembles per-camera frames from UDP chunks, keeping only the newest.

    For each camera we track the in-progress ``frame_id`` and its received chunks.
    A datagram for a newer frame discards any incomplete older frame (newest wins);
    datagrams for an older frame are ignored. Completed JPEG payloads are stored as
    the camera's latest frame for the display loop to decode.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._partial: dict[str, dict] = {}
        self._latest: dict[str, bytes] = {}
        self._stats: dict[str, int] = {}

    def add_packet(self, packet: bytes) -> None:
        if len(packet) < HEADER_SIZE:
            return
        magic, frame_id, total_chunks, chunk_idx, chunk_len, name_len = struct.unpack(
            _HEADER_FMT, packet[:HEADER_SIZE]
        )
        if magic != MAGIC:
            return

        name_start = HEADER_SIZE
        name_end = name_start + name_len
        payload_end = name_end + chunk_len
        if payload_end > len(packet):
            return
        name = packet[name_start:name_end].decode("utf-8", errors="replace")
        chunk = packet[name_end:payload_end]

        with self._lock:
            entry = self._partial.get(name)
            if entry is None or entry["frame_id"] != frame_id:
                # Chunk belongs to a different frame than the one in progress.
                if entry is not None and frame_id_is_older(frame_id, entry["frame_id"]):
                    return  # stale chunk from an older frame; ignore (newest wins)
                # Start a fresh frame, dropping any incomplete previous one.
                entry = {"frame_id": frame_id, "total": total_chunks, "chunks": {}}
                self._partial[name] = entry

            entry["chunks"][chunk_idx] = chunk
            if len(entry["chunks"]) >= entry["total"]:
                payload = b"".join(entry["chunks"][i] for i in range(entry["total"]))
                self._latest[name] = payload
                self._stats[name] = self._stats.get(name, 0) + 1
                del self._partial[name]

    def pop_latest(self) -> dict[str, bytes]:
        """Return and clear the latest completed frame for each camera."""
        with self._lock:
            latest = self._latest
            self._latest = {}
            return latest

    def fps_snapshot(self) -> dict[str, int]:
        with self._lock:
            stats = self._stats
            self._stats = {}
            return stats


def frame_id_is_older(candidate: int, current: int) -> bool:
    """True if ``candidate`` is an older frame than ``current`` under uint32 wrap.

    Uses signed wrap-around distance so the comparison stays correct across the
    32-bit counter rollover.
    """
    return ((current - candidate) & 0xFFFFFFFF) < 0x80000000 and candidate != current


def _receiver_loop(sock: socket.socket, assembler: _FrameAssembler, stop: threading.Event) -> None:
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            packet, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        assembler.add_packet(packet)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Low-latency UDP video viewer for the teleop PC.")
    parser.add_argument("--port", type=int, default=5005, help="UDP port to listen on. Default: 5005.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # nosec B104 - intentionally listen on all interfaces by default
        help="Local interface to bind. Default: 0.0.0.0 (all interfaces).",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=30.0,
        help="Max display refresh rate. Default: 30.",
    )
    args = parser.parse_args()

    import cv2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
    except OSError as e:
        logger.debug(f"Could not enlarge UDP receive buffer: {e}")
    sock.bind((args.host, args.port))
    logger.info(f"Listening for UDP video on {args.host}:{args.port}. Press 'q' or ESC to quit.")

    assembler = _FrameAssembler()
    stop = threading.Event()
    recv_thread = threading.Thread(
        target=_receiver_loop, args=(sock, assembler, stop), name="udp-video-recv", daemon=True
    )
    recv_thread.start()

    windows: set[str] = set()
    min_dt = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    last_fps_log = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            for name, payload in assembler.pop_latest().items():
                buf = np.frombuffer(payload, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if name not in windows:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    windows.add(name)
                cv2.imshow(name, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break

            now = time.perf_counter()
            if now - last_fps_log >= 5.0:
                stats = assembler.fps_snapshot()
                if stats:
                    summary = ", ".join(f"{n}: {c / 5.0:.1f} fps" for n, c in sorted(stats.items()))
                    logger.info(f"Receiving — {summary}")
                last_fps_log = now

            if min_dt:
                elapsed = time.perf_counter() - loop_start
                if elapsed < min_dt:
                    time.sleep(min_dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        recv_thread.join(timeout=1.0)
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
