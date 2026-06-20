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

"""UDP subscriber client for a single YAM leader arm.

Drop-in replacement for ``YamLeaderClient`` (the portal/TCP client) that reads
leader state from a UDP push stream instead. The server continuously sends
state datagrams; this client keeps only the freshest one in memory, so
``get_observations`` returns immediately with no per-frame network round-trip.
"""

import logging
import select
import socket
import threading
import time

import numpy as np

from .yam_udp_protocol import decode_state, encode_heartbeat

logger = logging.getLogger(__name__)

# How often the client re-sends a heartbeat so the server keeps streaming to us.
_HEARTBEAT_INTERVAL_S = 0.5
# How long connect() waits for the first packet before giving up.
_CONNECT_TIMEOUT_S = 5.0
# Minimum spacing between repeated stale-stream warnings.
_STALE_WARN_INTERVAL_S = 1.0
# A consumed sample older than this is counted as "stale" for stats. 33.3 ms is one
# control-loop frame at 30 fps, i.e. the leader data was already more than a frame old
# when the record loop used it.
_STALE_CONSUME_AGE_S = 1.0 / 30.0


class _ImmediateFuture:
    """Minimal future-like wrapper so callers can use the same ``.result()`` API
    as the portal client. The value is already available locally (latest packet),
    so ``result()`` simply returns it."""

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def result(self, timeout: float | None = None):
        return self._value


class YamLeaderUDPClient:
    """Client interface for a single YAM leader arm over a UDP push stream.

    Exposes the same surface as ``YamLeaderClient`` (portal) so ``BiYamLeader``
    is transport-agnostic: ``connect``, ``disconnect``, ``is_connected``,
    ``num_dofs``, ``get_observations``, ``request_observations`` and
    ``get_gripper_from_encoder``.
    """

    def __init__(
        self,
        port: int,
        host: str = "localhost",
        max_obs_age_s: float = 0.1,
        watchdog_timeout_s: float = 0.5,
    ):
        """
        Args:
            port: UDP port the leader server is publishing on.
            host: Leader server host address.
            max_obs_age_s: Age (seconds) of the freshest packet beyond which the
                stream is considered stale. Past this the last-known sample is still
                served, with a throttled warning.
            watchdog_timeout_s: Age (seconds) of the freshest packet beyond which the
                link is treated as dead and ``get_observations`` raises. Clamped to be
                at least ``max_obs_age_s``.
        """
        self.port = port
        self.host = host
        self.max_obs_age_s = max_obs_age_s
        self.watchdog_timeout_s = max(watchdog_timeout_s, max_obs_age_s)

        self._sock: socket.socket | None = None
        self._server_addr: tuple[str, int] | None = None

        self._lock = threading.Lock()
        self._latest_obs: dict | None = None
        self._latest_seq: int = -1
        # All monotonic timestamps are stored as integer nanoseconds (time.monotonic_ns)
        # so timestamp subtractions are exact; we only convert to float seconds for the
        # small resulting durations (where double precision is ample).
        self._latest_recv_t: int = 0
        self._num_dofs: int | None = None
        self._last_stale_warn_t: int = 0

        # Stream statistics (all guarded by _lock).
        self._connect_t: int = 0
        self._pkts_received: int = 0  # packets accepted as newer than the previous
        self._pkts_lost: int = 0  # gap-inferred packets that never arrived
        self._pkts_reordered: int = 0  # late/duplicate packets discarded on arrival
        self._consume_count: int = 0  # get_observations() calls that returned a sample
        self._consume_age_sum: float = 0.0  # sum of consumed-packet ages (seconds)
        self._consume_age_max: float = 0.0  # worst consumed-packet age (seconds)
        self._consume_stale_count: int = 0  # consumes whose age exceeded _STALE_CONSUME_AGE_S

        # --- Diagnostics for attributing high "consumed age" spikes ---
        # Monotonic ns of the last consume call, to measure how often the loop reads us.
        self._last_consume_t: int = 0
        # Monotonic ns the recv thread last ran a loop iteration; lets us tell whether
        # the thread was starved (didn't run) vs simply blocked waiting on no data.
        self._last_recv_iter_t: int = 0
        # Worst inter-arrival gap between two accepted packets, and the seq jump across
        # it (a large jump => loss during the gap; ~1 => server paused/stalled).
        self._recv_gap_max_s: float = 0.0
        self._recv_gap_seq_jump: int = 0

        self._stop = threading.Event()
        self._first_packet = threading.Event()
        self._recv_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None

    def connect(self):
        """Open the UDP socket, start the background receiver, and block until the
        first state packet arrives (so ``num_dofs`` can be derived)."""
        logger.info(f"Connecting to YAM leader UDP stream at {self.host}:{self.port}")
        self._server_addr = (socket.gethostbyname(self.host), self.port)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.settimeout(0.2)

        self._stop.clear()
        self._first_packet.clear()

        now = time.monotonic_ns()
        with self._lock:
            self._connect_t = now
            self._pkts_received = 0
            self._pkts_lost = 0
            self._pkts_reordered = 0
            self._consume_count = 0
            self._consume_age_sum = 0.0
            self._consume_age_max = 0.0
            self._consume_stale_count = 0
            self._last_consume_t = 0
            self._last_recv_iter_t = now
            self._recv_gap_max_s = 0.0
            self._recv_gap_seq_jump = 0

        self._recv_thread = threading.Thread(target=self._recv_loop, name=f"yam-udp-recv-{self.port}", daemon=True)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name=f"yam-udp-hb-{self.port}", daemon=True
        )
        self._recv_thread.start()
        self._heartbeat_thread.start()

        # Send an initial heartbeat right away so the server starts streaming.
        self._send_heartbeat()

        if not self._first_packet.wait(timeout=_CONNECT_TIMEOUT_S):
            self.disconnect()
            raise RuntimeError(
                f"Timed out after {_CONNECT_TIMEOUT_S}s waiting for first UDP packet from "
                f"YAM leader server at {self.host}:{self.port}. Is the server running with --transport udp?"
            )

        logger.info(f"Successfully connected to YAM leader UDP stream at {self.host}:{self.port}")

    def disconnect(self):
        """Stop the background threads and close the socket."""
        logger.info(f"Disconnecting from YAM leader UDP stream at {self.host}:{self.port}")
        if self._pkts_received > 0:
            s = self.get_stats()
            logger.info(
                f"YAM leader UDP {self.host}:{self.port} final stats: "
                f"recv {s['received']} @ {s['recv_rate_hz']:.0f} Hz, "
                f"lost {s['lost']} ({s['loss_pct']:.1f}%), reordered {s['reordered']}, "
                f"consumed {s['consume_count']}, avg consumed age {s['avg_consumed_age_ms']:.1f} ms "
                f"(max {s['max_consumed_age_ms']:.1f} ms), "
                f"stale(>{s['stale_consume_threshold_ms']:.0f}ms) {s['stale_consume_count']} "
                f"({s['stale_consume_pct']:.1f}%), "
                f"max recv gap {s['max_recv_gap_ms']:.1f} ms (seq_jump={s['max_recv_gap_seq_jump']})"
            )
        self._stop.set()
        for thread in (self._recv_thread, self._heartbeat_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._recv_thread = None
        self._heartbeat_thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception as e:
                logger.warning(f"Error closing YAM leader UDP socket at {self.host}:{self.port}: {e}")
            self._sock = None

    @property
    def is_connected(self) -> bool:
        """Check if the client socket is open."""
        return self._sock is not None

    def num_dofs(self) -> int:
        """Number of arm degrees of freedom, derived from the latest packet."""
        if self._num_dofs is None:
            raise RuntimeError("Client not connected or no packet received yet")
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        """Latest joint positions from the leader arm."""
        return self.get_observations()["joint_pos"]

    def get_observations(self) -> dict[str, np.ndarray]:
        """Return the freshest received observation dict.

        Staleness policy (mirrors LeKiwi's tolerate-then-watchdog approach, since
        UDP has no delivery guarantee and brief gaps are normal):
        - fresh (age <= ``max_obs_age_s``): return the latest sample.
        - stale but recoverable (``max_obs_age_s`` < age <= ``watchdog_timeout_s``):
          keep serving the last-known sample and log a throttled warning.
        - dead link (age > ``watchdog_timeout_s``): raise ``RuntimeError``.

        Raises immediately if no packet has ever been received.
        """
        with self._lock:
            obs = self._latest_obs
            recv_t = self._latest_recv_t
        if obs is None:
            raise RuntimeError(f"No UDP packet received yet from YAM leader server at {self.host}:{self.port}")

        now = time.monotonic_ns()
        age = (now - recv_t) / 1e9
        if age > self.watchdog_timeout_s:
            raise RuntimeError(
                f"Lost YAM leader UDP stream at {self.host}:{self.port}: latest packet is {age * 1e3:.0f} ms old "
                f"(> watchdog {self.watchdog_timeout_s * 1e3:.0f} ms). Link down or server stopped publishing."
            )
        if age > self.max_obs_age_s and (now - self._last_stale_warn_t) / 1e9 > _STALE_WARN_INTERVAL_S:
            self._last_stale_warn_t = now
            logger.warning(
                f"Stale YAM leader UDP stream at {self.host}:{self.port}: latest packet is {age * 1e3:.0f} ms "
                f"old (> {self.max_obs_age_s * 1e3:.0f} ms); serving last-known sample."
            )

        with self._lock:
            self._consume_count += 1
            self._consume_age_sum += age
            if age > _STALE_CONSUME_AGE_S:
                self._consume_stale_count += 1
            is_new_max = age > self._consume_age_max
            if is_new_max:
                self._consume_age_max = age
            since_last_consume = (now - self._last_consume_t) / 1e9 if self._last_consume_t else 0.0
            self._last_consume_t = now
            recv_idle = (now - self._last_recv_iter_t) / 1e9
            latest_seq = self._latest_seq
            lost_total = self._pkts_lost
            recv_gap_max = self._recv_gap_max_s
            recv_gap_jump = self._recv_gap_seq_jump

        # On a new worst-case age (above the freshness budget), log a one-off breakdown
        # that attributes the spike. Logging only on new maxima is naturally self-throttling.
        if is_new_max and age > self.max_obs_age_s:
            socket_unread = self._socket_has_unread_data()
            logger.warning(
                f"YAM leader UDP {self.host}:{self.port} new max consumed age {age * 1e3:.1f} ms | "
                f"since_last_consume={since_last_consume * 1e3:.1f} ms, "
                f"recv_thread_idle={recv_idle * 1e3:.1f} ms, socket_unread={socket_unread}, "
                f"latest_seq={latest_seq}, lost_total={lost_total}, "
                f"max_recv_gap={recv_gap_max * 1e3:.1f} ms (seq_jump={recv_gap_jump}). "
                f"Interpretation: socket_unread=True => receiver thread was starved "
                f"(consumer held the GIL, e.g. display/encoding on the main loop); "
                f"socket_unread=False with large since_last_consume => the consumer loop "
                f"itself was slow; otherwise a real gap on the wire (seq_jump>>1 => burst "
                f"loss, seq_jump~1 => server stalled/stopped publishing)."
            )

        return obs

    def _socket_has_unread_data(self) -> bool:
        """Return True if a datagram is already waiting unread in the socket buffer.

        Probed from the consumer thread at the moment of a stale read: if data is
        waiting here but our latest packet is old, the receiver thread didn't get a
        chance to run (starvation), as opposed to no packet having arrived (loss/stall).
        """
        sock = self._sock
        if sock is None:
            return False
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            return bool(readable)
        except (OSError, ValueError):
            return False

    def get_stats(self) -> dict:
        """Return cumulative stream statistics since the last ``connect()``.

        Keys: ``received``, ``lost``, ``reordered``, ``loss_pct``, ``recv_rate_hz``,
        ``consume_count``, ``avg_consumed_age_ms``, ``max_consumed_age_ms``.

        Note: ``avg_consumed_age_ms`` is measured on the client clock (time between
        receiving a packet and consuming it), so it is independent of any clock skew
        between the leader server and this host.
        """
        with self._lock:
            elapsed = max((time.monotonic_ns() - self._connect_t) / 1e9, 1e-9)
            received = self._pkts_received
            lost = self._pkts_lost
            reordered = self._pkts_reordered
            consume_count = self._consume_count
            age_sum = self._consume_age_sum
            age_max = self._consume_age_max
            stale_count = self._consume_stale_count
            recv_gap_max = self._recv_gap_max_s
            recv_gap_jump = self._recv_gap_seq_jump
        published_est = received + lost
        return {
            "received": received,
            "lost": lost,
            "reordered": reordered,
            "loss_pct": (100.0 * lost / published_est) if published_est else 0.0,
            "recv_rate_hz": received / elapsed,
            "consume_count": consume_count,
            "avg_consumed_age_ms": (1e3 * age_sum / consume_count) if consume_count else 0.0,
            "max_consumed_age_ms": 1e3 * age_max,
            "stale_consume_count": stale_count,
            "stale_consume_threshold_ms": 1e3 * _STALE_CONSUME_AGE_S,
            "stale_consume_pct": (100.0 * stale_count / consume_count) if consume_count else 0.0,
            "max_recv_gap_ms": 1e3 * recv_gap_max,
            "max_recv_gap_seq_jump": recv_gap_jump,
        }

    def request_observations(self):
        """Return a future-like wrapper around the freshest observation.

        Mirrors the portal client's non-blocking API so ``BiYamLeader.get_action``
        can fire both arms and then collect; here the data is already local so the
        wrapper resolves immediately.
        """
        return _ImmediateFuture(self.get_observations())

    @staticmethod
    def gripper_from_encoder_obs(obs: dict) -> float:
        """Derive gripper state (0=closed, 1=open) from the teaching-handle encoder
        button in an observation dict. Falls back to open if unavailable."""
        try:
            if "io_inputs" in obs:
                return 0.0 if obs["io_inputs"][0] > 0.5 else 1.0
            return 1.0
        except Exception:
            return 1.0

    def get_gripper_from_encoder(self, obs: dict | None = None) -> float:
        """Gripper state from the teaching-handle encoder button.

        Args:
            obs: Optional pre-fetched observations. When provided, avoids a read.
        """
        if obs is not None:
            return self.gripper_from_encoder_obs(obs)
        try:
            return self.gripper_from_encoder_obs(self.get_observations())
        except Exception:
            return 1.0

    def _send_heartbeat(self):
        if self._sock is None or self._server_addr is None:
            return
        try:
            self._sock.sendto(encode_heartbeat(), self._server_addr)
        except OSError as e:
            logger.debug(f"Heartbeat send failed for {self.host}:{self.port}: {e}")

    def _heartbeat_loop(self):
        while not self._stop.is_set():
            self._send_heartbeat()
            self._stop.wait(_HEARTBEAT_INTERVAL_S)

    def _recv_loop(self):
        while not self._stop.is_set():
            # Mark that the thread is actively running an iteration. If this stops
            # advancing while packets are waiting, the thread is being starved.
            self._last_recv_iter_t = time.monotonic_ns()
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue

            obs = decode_state(data)
            if obs is None:
                continue

            seq = obs["seq"]
            recv_t = time.monotonic_ns()
            with self._lock:
                # Keep only the newest packet; UDP can deliver out of order.
                # Tolerate sequence wraparound by accepting a large backward jump.
                if seq <= self._latest_seq and (self._latest_seq - seq) < (1 << 31):
                    self._pkts_reordered += 1
                    continue
                # Infer dropped packets from gaps in the sequence number.
                if self._latest_seq >= 0 and seq > self._latest_seq + 1:
                    self._pkts_lost += seq - self._latest_seq - 1
                # Track the worst inter-arrival gap (and the seq jump across it) so we
                # can tell a long silent gap (loss/stall) from a busy-but-late stream.
                if self._latest_recv_t > 0:
                    gap = (recv_t - self._latest_recv_t) / 1e9
                    if gap > self._recv_gap_max_s:
                        self._recv_gap_max_s = gap
                        self._recv_gap_seq_jump = (seq - self._latest_seq) if self._latest_seq >= 0 else 1
                self._pkts_received += 1
                self._latest_obs = obs
                self._latest_seq = seq
                self._latest_recv_t = recv_t
                if self._num_dofs is None:
                    self._num_dofs = int(obs["joint_pos"].shape[0])

            if not self._first_packet.is_set():
                self._first_packet.set()
