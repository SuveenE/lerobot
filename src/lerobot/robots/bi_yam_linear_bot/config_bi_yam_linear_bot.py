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

import math
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

    # FlowBase controller velocity limits – must match the values used by the
    # running flow_base_controller so that policy outputs (physical velocities)
    # are correctly normalised to the [-1, 1] range the controller expects.
    # Base limits align with FlowBaseClient DEFAULT_MAX_VEL_{X,Y,THETA}.
    base_max_vel: tuple[float, float, float] = (0.5, 0.5, math.pi / 2)
    # Deprecated / unused: the rail is now commanded end-to-end in physical m/s
    # (the controller converts m/s -> motor rad/s via its meters_per_rad
    # calibration), so this rad/s normalisation scale is no longer applied by the
    # robot. Kept only for backward compatibility with existing configs/CLIs.
    rail_max_vel: float = 7.0
    # Linear rail speed cap in m/s (mirrors FlowBaseClient DEFAULT_MAX_VEL_Z).
    # Bounds both the recorded rail command and policy rail output so the rail
    # action space stays within ±this, like base_max_vel does for the base.
    rail_max_vel_mps: float = 0.5

    # Optional: Maximum relative target for arm safety
    left_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_max_relative_target: float | dict[str, float] | None = None

    # When True, also record per-arm motor torques (joint_eff / gripper_eff) as
    # dedicated `observation.left_torques` and `observation.right_torques`
    # columns. Defaults to False so existing recordings / pretrained
    # positions-only policies remain unaffected. Enable via
    # `--robot.record_torques=true` on the lerobot-record CLI.
    record_torques: bool = False

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
