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

"""Wire format for the YAM leader-arm UDP streaming transport.

The leader server pushes one datagram per observation; the client reads the
freshest sample locally instead of paying a per-frame TCP round-trip. The
format is a compact, versioned little-endian struct so a single observation
fits in one UDP frame and is cheap to (de)serialize on the control loop.

Two packet kinds share this module, distinguished by their leading magic:
- STATE  (``YAMS``): server -> client, carries one observation.
- HEARTBEAT (``YAMH``): client -> server, registers/refreshes a subscriber.
"""

from __future__ import annotations

import struct

import numpy as np

PROTOCOL_VERSION = 1

STATE_MAGIC = b"YAMS"
HEARTBEAT_MAGIC = b"YAMH"

# magic(4s) version(B) seq(I) t_send(d) num_joints(B)
_STATE_HEADER = struct.Struct("<4sBIdB")
# has_gripper(B) gripper_pos(f) io_input(B)
_STATE_TRAILER = struct.Struct("<BfB")

# magic(4s) version(B)
_HEARTBEAT = struct.Struct("<4sB")


def encode_heartbeat() -> bytes:
    """Build a heartbeat datagram a client sends to (re)subscribe to a server."""
    return _HEARTBEAT.pack(HEARTBEAT_MAGIC, PROTOCOL_VERSION)


def is_heartbeat(data: bytes) -> bool:
    """Return True if ``data`` is a well-formed heartbeat datagram."""
    if len(data) < _HEARTBEAT.size:
        return False
    magic, version = _HEARTBEAT.unpack_from(data, 0)
    return magic == HEARTBEAT_MAGIC and version == PROTOCOL_VERSION


def encode_state(
    seq: int,
    t_send: float,
    joint_pos: np.ndarray,
    gripper_pos: float | None,
    io_input: float,
) -> bytes:
    """Encode a single leader observation into a state datagram.

    Args:
        seq: Monotonic sequence number; lets the receiver drop stale/reordered packets.
        t_send: Sender monotonic timestamp (seconds) for staleness detection.
        joint_pos: 1D array of arm joint positions (radians), excluding gripper.
        gripper_pos: Gripper position in [0, 1], or None if the arm has no gripper.
        io_input: Teaching-handle button state (0/1); preserved so the client can
            derive a gripper command from the encoder button when needed.
    """
    joints = np.ascontiguousarray(joint_pos, dtype=np.float32).reshape(-1)
    num_joints = int(joints.shape[0])
    has_gripper = 1 if gripper_pos is not None else 0
    return (
        _STATE_HEADER.pack(STATE_MAGIC, PROTOCOL_VERSION, seq & 0xFFFFFFFF, t_send, num_joints)
        + joints.tobytes()
        + _STATE_TRAILER.pack(has_gripper, float(gripper_pos or 0.0), int(io_input > 0.5))
    )


def decode_state(data: bytes) -> dict | None:
    """Decode a state datagram into a dict, or return None if it is not valid.

    The returned dict mirrors the keys produced by the portal-based leader
    observations so downstream code is transport-agnostic:
    ``seq``, ``t_send``, ``joint_pos`` and (optionally) ``gripper_pos`` /
    ``io_inputs``.
    """
    if len(data) < _STATE_HEADER.size:
        return None
    magic, version, seq, t_send, num_joints = _STATE_HEADER.unpack_from(data, 0)
    if magic != STATE_MAGIC or version != PROTOCOL_VERSION:
        return None

    offset = _STATE_HEADER.size
    joints_nbytes = num_joints * 4
    if len(data) < offset + joints_nbytes + _STATE_TRAILER.size:
        return None

    joint_pos = np.frombuffer(data, dtype=np.float32, count=num_joints, offset=offset).astype(np.float64)
    offset += joints_nbytes
    has_gripper, gripper_pos, io_input = _STATE_TRAILER.unpack_from(data, offset)

    obs: dict = {
        "seq": seq,
        "t_send": t_send,
        "joint_pos": joint_pos,
    }
    if has_gripper:
        obs["gripper_pos"] = np.array([gripper_pos], dtype=np.float64)
    obs["io_inputs"] = np.array([float(io_input)], dtype=np.float64)
    return obs
