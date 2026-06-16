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

from dataclasses import dataclass

from ..config import TeleoperatorConfig

# Supported leader-state transports. Kept as a module constant (not a typing.Literal
# on the field) because the draccus CLI parser used by lerobot has no decoder for
# Literal types.
TRANSPORTS = ("portal", "udp")


@TeleoperatorConfig.register_subclass("bi_yam_leader")
@dataclass
class BiYamLeaderConfig(TeleoperatorConfig):
    # Server ports for left and right arm leaders
    # These should be different from the follower ports
    # Note: You'll need to run separate server processes for the leader arms
    # that expose their state for reading (see i2rt minimum_gello.py)
    left_arm_port: int = 5002
    right_arm_port: int = 5001

    # Server host (usually localhost for local setup)
    server_host: str = "localhost"

    # Network transport for reading leader arm state. One of TRANSPORTS:
    # - "portal": TCP request/response RPC (default, blocks on a round-trip per frame)
    # - "udp": leader servers push state datagrams; client reads the freshest sample
    #   locally with no per-frame round-trip. Lower latency, no delivery guarantee.
    transport: str = "portal"

    # UDP transport only: age (s) of the freshest packet beyond which the stream is
    # considered "stale". Past this we keep serving the last-known sample but log a
    # throttled warning. Ignored when transport == "portal".
    max_obs_age_s: float = 0.1

    # UDP transport only: age (s) of the freshest packet beyond which the link is
    # treated as dead and get_observations() raises (watchdog hard-fail). Must be
    # >= max_obs_age_s. Ignored when transport == "portal".
    watchdog_timeout_s: float = 0.5

    # UDP transport only: how often (s) to log a stream-health summary (receive rate,
    # packet loss, average age of consumed packets). 0 disables periodic logging.
    stats_log_interval_s: float = 10.0

    def __post_init__(self):
        if self.transport not in TRANSPORTS:
            raise ValueError(f"transport must be one of {TRANSPORTS}, got {self.transport!r}")

