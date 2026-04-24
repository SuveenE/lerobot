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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_yam_linear_bot")
@dataclass
class BiYamLinearBotConfig(RobotConfig):
    # Yam follower arm server ports
    left_arm_port: int = 1235
    right_arm_port: int = 1234

    # Server host for the Yam arm servers
    arm_server_host: str = "localhost"

    # FlowBase server host (may differ from arm host in a split setup)
    flow_base_host: str = "localhost"

    # Whether the FlowBase has a linear rail lift module
    with_linear_rail: bool = True

    # If True, only record `base.x` in observations and `base.x.vel` in actions
    # (plus the usual arm / camera fields). Used for experiments where storage
    # must be minimised and the task only needs X-axis base motion. The
    # FlowBase controller still receives a 3D velocity command internally, with
    # Y and theta forced to zero when a policy drives the robot.
    base_x_only: bool = False

    # FlowBase controller velocity limits – must match the values used by the
    # running flow_base_controller so that policy outputs (physical velocities)
    # are correctly normalised to the [-1, 1] range the controller expects.
    base_max_vel: tuple[float, float, float] = (0.4, 0.4, 1.5)
    rail_max_vel: float = 7.0

    # Optional: Maximum relative target for arm safety
    left_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_max_relative_target: float | dict[str, float] | None = None

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
